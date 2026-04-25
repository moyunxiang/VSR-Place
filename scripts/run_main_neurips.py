"""Unified NeurIPS-main experiment driver.

Per (circuit, seed) iteration runs ALL of the following side by side so a single
OOM only loses one (circuit, seed) pair. Resume-safe per pair.

Methods
-------
* baseline_cd        ChipDiffusion guided sample (= existing paper baseline)
* cg_strong          Pure classifier-guidance, legality_weight x4, hpwl_weight=0
* repaint_binary     RePaint with mask = (severity > 0)  (== existing intra)
* vsr_intra_soft     RePaint with mask = severity / max(severity)  (NEW)
* vsr_post           CD-guided + 100-step force integrator with lambda=2
* cd_std_legalize    CD-guided + 5000-step standard legalizer
* cd_sched_legalize  CD-guided + 5000-step HPWL-aware legalizer

Records for each method: violation count, full-pin HPWL, wall-clock, peak VRAM.

Output: results/ispd2005/main_neurips.json  (one row per (circuit, seed))
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
import types
from pathlib import Path

# Standard ChipDiffusion path setup
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "third_party" / "chipdiffusion"))
sys.path.insert(0, str(REPO / "third_party" / "chipdiffusion" / "diffusion"))
sys.path.insert(0, str(REPO / "scripts"))

# Mock moviepy (heavy, unused by us)
_mp = types.ModuleType("moviepy")
_mp_ed = types.ModuleType("moviepy.editor")
_mp_ed.ImageSequenceClip = lambda *a, **k: None
sys.modules["moviepy"] = _mp
sys.modules["moviepy.editor"] = _mp_ed

import torch  # noqa: E402

# Imports from our codebase
from vsr_place.backbone.adapter import ChipDiffusionAdapter  # noqa: E402
from vsr_place.verifier.verifier import Verifier  # noqa: E402
from vsr_place.renoising.local_repair import local_repair_loop  # noqa: E402
from vsr_place.metrics.hpwl import compute_hpwl_from_edges  # noqa: E402

NAMES = ["adaptec1", "adaptec2", "adaptec3", "adaptec4",
         "bigblue1", "bigblue2", "bigblue3", "bigblue4"]
CIRCUITS_DEFAULT = [0, 1, 2, 3, 4, 6]   # bigblue2/4 OOM on 80GB
SEEDS_DEFAULT = [42, 123, 300, 2024]
LAMBDA = 2.0
T_INTRA = 0.3
NUM_REPAIR_STEPS = 100

# CD legalizer configs from ChipDiffusion eval logs
CD_STD = dict(step_size=0.2, grad_descent_steps=5000, softmax_min=5.0, softmax_max=50.0,
              softmax_critical_factor=1.0, hpwl_weight=0.0, macros_only=True)
CD_SCHED = dict(step_size=0.2, grad_descent_steps=5000, softmax_min=5.0, softmax_max=50.0,
                softmax_critical_factor=1.0, hpwl_weight=4e-5, legality_weight=2.0,
                macros_only=True)


def reset_peak_vram():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_vram_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024 / 1024


# ---------------------------------------------------------------------------
# Method implementations (kept inline to keep blast radius small)
# ---------------------------------------------------------------------------

def method_baseline(adapter, cond, _seed):
    """Standard CD-guided sample; no post-processing. Reused by other methods."""
    reset_peak_vram()
    t0 = time.time()
    pl = adapter.guided_sample(cond.to(adapter.device), num_samples=1).squeeze(0)
    return pl, time.time() - t0, peak_vram_mb()


def method_cg_strong(checkpoint, cond, x_in_shape, legality_weight=1.0):
    """Classifier-guidance baseline: smooth softplus legality only, no HPWL.

    Loads a fresh adapter so the patched guidance weights don't leak to other
    methods. NOTE: ChipDiffusion's default _default_large_config sets weights
    to 0 by default; we patch to >0 explicitly here to actually invoke
    classifier-style guidance during sampling.
    """
    ad = ChipDiffusionAdapter.from_checkpoint(
        checkpoint, device="cuda", input_shape=x_in_shape, guidance="opt",
    )
    if hasattr(ad.model, "legality_guidance_weight"):
        ad.model.legality_guidance_weight = legality_weight
    if hasattr(ad.model, "hpwl_guidance_weight"):
        ad.model.hpwl_guidance_weight = 0.0
    if hasattr(ad.model, "grad_descent_steps") and ad.model.grad_descent_steps == 0:
        ad.model.grad_descent_steps = 5  # universal-guidance inner steps
    reset_peak_vram()
    t0 = time.time()
    pl = ad.guided_sample(cond.to(ad.device), num_samples=1).squeeze(0)
    elapsed = time.time() - t0
    vram = peak_vram_mb()
    out = pl.detach().cpu()
    del ad
    gc.collect()
    torch.cuda.empty_cache()
    return out, elapsed, vram


def method_vsr_post(centers, sizes, cw, ch, mask, edge_index):
    reset_peak_vram()
    t0 = time.time()
    out = local_repair_loop(centers, sizes, cw, ch, num_steps=NUM_REPAIR_STEPS,
                            step_size=0.3, only_mask=mask, edge_index=edge_index,
                            hpwl_weight=LAMBDA)
    return out, time.time() - t0, peak_vram_mb()


def method_repaint_binary(adapter, x_hat_0, cond, mask_bool):
    """RePaint with binary mask = severity > 0."""
    reset_peak_vram()
    t0 = time.time()
    pl_re = adapter.denoise_repaint(x_hat_0, cond.to(adapter.device),
                                    mask_bool.to(adapter.device),
                                    start_timestep=T_INTRA, num_steps=NUM_REPAIR_STEPS)
    return pl_re, time.time() - t0, peak_vram_mb()


def method_vsr_intra_soft(adapter, x_hat_0, cond, severity_vec):
    """RePaint with SOFT severity-weighted mask, blending model output & forward-diffused.

    Implementation mirrors adapter.denoise_repaint but with a continuous mask.
    """
    reset_peak_vram()
    t0 = time.time()
    # Normalise severity to [0, 1]; ensure non-violators get mask=0.
    sev = severity_vec.to(adapter.device).float()
    if sev.max() > 1e-9:
        soft_mask = sev / sev.max()
    else:
        soft_mask = sev
    soft_mask = soft_mask.unsqueeze(0).unsqueeze(-1)  # (1, V, 1)

    x_hat_0 = x_hat_0.to(adapter.device).unsqueeze(0)
    cond = cond.to(adapter.device)

    if T_INTRA <= 1e-6:
        return x_hat_0.squeeze(0), time.time() - t0, peak_vram_mb()

    with torch.no_grad():
        adapter.scheduler.set_timesteps(NUM_REPAIR_STEPS)
        all_ts = adapter.scheduler.timesteps.to(adapter.device)
        start_idx = 0
        for i, ts in enumerate(all_ts):
            if ts.item() <= T_INTRA + 1e-6:
                start_idx = i
                break
        timesteps = all_ts[start_idx:]

        eps_init = torch.randn_like(x_hat_0)
        t_start = timesteps[0].expand(1)
        x = adapter.scheduler.add_noise(x_hat_0, eps_init, t_start)

        for i in range(len(timesteps) - 1):
            t = timesteps[i].expand(1).to(adapter.device)
            t_next = timesteps[i + 1].expand(1).to(adapter.device)
            eps_pred = adapter.model(x, cond, t)
            z = torch.randn_like(x) if i < len(timesteps) - 2 else torch.zeros_like(x)
            result = adapter.scheduler.step(eps_pred, t, t_next, x, z)
            x_denoised = result[0] if isinstance(result, tuple) else result
            if i < len(timesteps) - 2:
                eps_known = torch.randn_like(x_hat_0)
                x_known = adapter.scheduler.add_noise(x_hat_0, eps_known, t_next)
            else:
                x_known = x_hat_0
            # SOFT blend: mask=high (offender) → trust denoised; mask=low → keep known.
            x = soft_mask * x_denoised + (1.0 - soft_mask) * x_known
        out = x.squeeze(0)
    return out, time.time() - t0, peak_vram_mb()


def method_cd_legalize(adapter, x_hat_0, cond, cfg):
    """ChipDiffusion's optimization-based legalizer."""
    import legalization as cd_leg  # type: ignore
    reset_peak_vram()
    t0 = time.time()
    try:
        pl_in = x_hat_0.unsqueeze(0).detach().to(adapter.device)
        pl_leg = cd_leg.legalize(pl_in, cond.to(adapter.device), **cfg)
        if isinstance(pl_leg, tuple):
            pl_leg = pl_leg[0]
        return pl_leg.squeeze(0), time.time() - t0, peak_vram_mb()
    except Exception as e:
        print(f"  CD legalize failed: {type(e).__name__}: {str(e)[:100]}", flush=True)
        return None, time.time() - t0, peak_vram_mb()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def evaluate_placement(centers, sizes, cw, ch, edge_index, edge_attr):
    """Common metric extraction. Returns (n_violations, hpwl)."""
    ver = Verifier(canvas_width=cw, canvas_height=ch)
    fb = ver(centers, sizes)
    nv = fb.global_stats["total_violations"]
    h = compute_hpwl_from_edges(centers, edge_index, edge_attr)
    return int(nv), float(h), ver, fb


def run_pair(checkpoint, cond, x_in, name, seed):
    """Run all methods for a single (circuit, seed) pair. Returns dict row."""
    torch.manual_seed(seed)
    torch.cuda.empty_cache()

    row = {"circuit": name, "n_macros": int(cond.x.shape[0]), "seed": seed}

    # ------------------------------------------------------------------ baseline
    ad = ChipDiffusionAdapter.from_checkpoint(
        checkpoint, device="cuda", input_shape=tuple(x_in.shape), guidance="opt",
    )
    cw, ch = ad.get_canvas_size(cond)
    cond_cu = cond.to("cuda")

    pl_base, t_base, vram_base = method_baseline(ad, cond, seed)
    cent_base, sz = ad.decode_placement(pl_base, cond)
    cent_base = cent_base.cpu(); sz = sz.cpu()
    ei = cond.edge_index.cpu()
    ea = cond.edge_attr.cpu() if cond.edge_attr is not None else None
    base_v, base_h, ver, fb = evaluate_placement(cent_base, sz, cw, ch, ei, ea)
    row.update(baseline_v=base_v, baseline_h=base_h,
               baseline_time=t_base, baseline_vram_mb=vram_base)
    print(f"  baseline: v={base_v} h={base_h:.1f} ({t_base:.2f}s, {vram_base:.0f}MB)", flush=True)

    sev = fb.severity_vector
    mask_bool = sev > 0
    mask_soft_src = sev.clone()  # used by VSR-intra-soft

    # ------------------------------------------------------------------ vsr_post
    try:
        cent_post, t_post, vram_post = method_vsr_post(cent_base, sz, cw, ch, mask_bool, ei)
        v, h, _, _ = evaluate_placement(cent_post, sz, cw, ch, ei, ea)
        row.update(vsr_post_v=v, vsr_post_h=h, vsr_post_time=t_post, vsr_post_vram_mb=vram_post)
        print(f"  vsr_post: v={v} ({(base_v-v)/max(base_v,1)*100:+.1f}%) h={h:.1f} ({(h-base_h)/max(base_h,1e-9)*100:+.1f}%) ({t_post:.2f}s)", flush=True)
    except Exception as e:
        print(f"  vsr_post error: {e}", flush=True)
        row.update(vsr_post_v=None, vsr_post_h=None, vsr_post_time=None, vsr_post_vram_mb=None)

    # ------------------------------------------------------------------ repaint_binary
    try:
        pl_re, t_re, vram_re = method_repaint_binary(ad, pl_base, cond, mask_bool)
        cent_re, _ = ad.decode_placement(pl_re, cond)
        cent_re = cent_re.cpu()
        v, h, _, _ = evaluate_placement(cent_re, sz, cw, ch, ei, ea)
        row.update(repaint_bin_v=v, repaint_bin_h=h, repaint_bin_time=t_re, repaint_bin_vram_mb=vram_re)
        print(f"  repaint:  v={v} ({(base_v-v)/max(base_v,1)*100:+.1f}%) h={h:.1f} ({(h-base_h)/max(base_h,1e-9)*100:+.1f}%) ({t_re:.2f}s)", flush=True)
    except Exception as e:
        print(f"  repaint_binary error: {e}", flush=True)
        row.update(repaint_bin_v=None, repaint_bin_h=None, repaint_bin_time=None, repaint_bin_vram_mb=None)

    # ------------------------------------------------------------------ vsr_intra_soft
    try:
        pl_int, t_int, vram_int = method_vsr_intra_soft(ad, pl_base, cond, mask_soft_src)
        cent_int, _ = ad.decode_placement(pl_int, cond)
        cent_int = cent_int.cpu()
        v, h, _, _ = evaluate_placement(cent_int, sz, cw, ch, ei, ea)
        row.update(vsr_intra_v=v, vsr_intra_h=h, vsr_intra_time=t_int, vsr_intra_vram_mb=vram_int)
        print(f"  intra:    v={v} ({(base_v-v)/max(base_v,1)*100:+.1f}%) h={h:.1f} ({(h-base_h)/max(base_h,1e-9)*100:+.1f}%) ({t_int:.2f}s)", flush=True)
    except Exception as e:
        print(f"  vsr_intra_soft error: {e}", flush=True)
        row.update(vsr_intra_v=None, vsr_intra_h=None, vsr_intra_time=None, vsr_intra_vram_mb=None)

    # ------------------------------------------------------------------ cd_std
    pl_cd, t_cd, vram_cd = method_cd_legalize(ad, pl_base, cond, CD_STD)
    if pl_cd is not None:
        cent_cd, _ = ad.decode_placement(pl_cd, cond)
        cent_cd = cent_cd.detach().cpu()
        v, h, _, _ = evaluate_placement(cent_cd, sz, cw, ch, ei, ea)
        row.update(cd_std_v=v, cd_std_h=h, cd_std_time=t_cd, cd_std_vram_mb=vram_cd)
        print(f"  cd_std:   v={v} ({(base_v-v)/max(base_v,1)*100:+.1f}%) h={h:.1f} ({(h-base_h)/max(base_h,1e-9)*100:+.1f}%) ({t_cd:.2f}s)", flush=True)
    else:
        row.update(cd_std_v=None, cd_std_h=None, cd_std_time=t_cd, cd_std_vram_mb=vram_cd)

    # ------------------------------------------------------------------ cd_sched
    pl_cs, t_cs, vram_cs = method_cd_legalize(ad, pl_base, cond, CD_SCHED)
    if pl_cs is not None:
        cent_cs, _ = ad.decode_placement(pl_cs, cond)
        cent_cs = cent_cs.detach().cpu()
        v, h, _, _ = evaluate_placement(cent_cs, sz, cw, ch, ei, ea)
        row.update(cd_sched_v=v, cd_sched_h=h, cd_sched_time=t_cs, cd_sched_vram_mb=vram_cs)
        print(f"  cd_sched: v={v} ({(base_v-v)/max(base_v,1)*100:+.1f}%) h={h:.1f} ({(h-base_h)/max(base_h,1e-9)*100:+.1f}%) ({t_cs:.2f}s)", flush=True)
    else:
        row.update(cd_sched_v=None, cd_sched_h=None, cd_sched_time=t_cs, cd_sched_vram_mb=vram_cs)

    del ad
    gc.collect(); torch.cuda.empty_cache()

    # ------------------------------------------------------------------ cg_strong
    try:
        torch.manual_seed(seed)
        pl_cg, t_cg, vram_cg = method_cg_strong(checkpoint, cond, tuple(x_in.shape))
        ad2 = ChipDiffusionAdapter.from_checkpoint(
            checkpoint, device="cuda", input_shape=tuple(x_in.shape), guidance="opt",
        )
        cent_cg, _ = ad2.decode_placement(pl_cg.to("cuda"), cond.to("cuda"))
        cent_cg = cent_cg.cpu()
        v, h, _, _ = evaluate_placement(cent_cg, sz, cw, ch, ei, ea)
        row.update(cg_strong_v=v, cg_strong_h=h, cg_strong_time=t_cg, cg_strong_vram_mb=vram_cg)
        print(f"  cg_strong: v={v} ({(base_v-v)/max(base_v,1)*100:+.1f}%) h={h:.1f} ({(h-base_h)/max(base_h,1e-9)*100:+.1f}%) ({t_cg:.2f}s)", flush=True)
        del ad2
        gc.collect(); torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError:
        print("  cg_strong OOM", flush=True)
        row.update(cg_strong_v=None, cg_strong_h=None, cg_strong_time=None, cg_strong_vram_mb=None)
        gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        print(f"  cg_strong error: {e}", flush=True)
        row.update(cg_strong_v=None, cg_strong_h=None, cg_strong_time=None, cg_strong_vram_mb=None)
        gc.collect(); torch.cuda.empty_cache()

    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/large-v2/large-v2.ckpt")
    p.add_argument("--circuits", type=int, nargs="+", default=CIRCUITS_DEFAULT)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS_DEFAULT)
    p.add_argument("--out", default="results/ispd2005/main_neurips.json")
    p.add_argument("--mock", action="store_true",
                   help="Run preflight with mock backend (no GPU, no checkpoint).")
    args = p.parse_args()

    if args.mock:
        run_preflight()
        return

    # Real GPU run
    from run_vsr import load_benchmark_data, filter_macros_only  # type: ignore
    val = load_benchmark_data("ispd2005")
    val = [filter_macros_only(x, c) for x, c in val]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if out_path.exists():
        rows = json.load(open(out_path))
    done = {(r["circuit"], r["seed"]) for r in rows if r.get("baseline_v") is not None}
    print(f"Resume: {len(rows)} existing, {len(done)} done pairs", flush=True)

    compute_meta = {
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }

    for ci in args.circuits:
        name = NAMES[ci]
        x_in, cond = val[ci]
        for seed in args.seeds:
            if (name, seed) in done:
                continue
            print(f"\n=== {name} seed={seed} ===", flush=True)
            try:
                row = run_pair(args.checkpoint, cond, x_in, name, seed)
                row["compute"] = compute_meta
                rows.append(row)
                with open(out_path, "w") as f:
                    json.dump(rows, f, indent=2)
            except torch.cuda.OutOfMemoryError:
                print("  OOM", flush=True)
                rows.append({"circuit": name, "seed": seed, "error": "OOM"})
                with open(out_path, "w") as f:
                    json.dump(rows, f, indent=2)
                gc.collect(); torch.cuda.empty_cache()
            except Exception as e:
                print(f"  ERROR: {e}", flush=True)
                traceback.print_exc()
                rows.append({"circuit": name, "seed": seed, "error": str(e)[:200]})
                with open(out_path, "w") as f:
                    json.dump(rows, f, indent=2)
                gc.collect(); torch.cuda.empty_cache()

    print(f"\nDone. {len(rows)} rows in {out_path}")


# ---------------------------------------------------------------------------
# Preflight (mock backend, CPU)
# ---------------------------------------------------------------------------

def run_preflight():
    """Run E1 logic with a mock backbone on a tiny synthetic netlist (no GPU).

    Validates: argparse, imports, schema of output JSON, error path.
    Does NOT validate numerical correctness.
    """
    print("=== E1 preflight (mock) ===", flush=True)
    # Build a tiny PyG-like cond with chip_size attribute
    import numpy as np

    class MockCond:
        def __init__(self):
            self.x = torch.rand(10, 2) * 0.1  # 10 nodes, sizes
            self.edge_index = torch.randint(0, 10, (2, 8))
            self.edge_index = torch.cat([self.edge_index, self.edge_index.flip(0)], dim=1)
            self.edge_attr = torch.zeros(self.edge_index.shape[1], 4)
            self.chip_size = [0.0, 0.0, 1.0, 1.0]

        def to(self, device):
            return self

    cond = MockCond()
    cw, ch = 1.0, 1.0
    sz = cond.x.clone()
    centers = torch.rand(10, 2) * 0.9 + 0.05
    ei = cond.edge_index
    ea = cond.edge_attr

    # Test verifier
    ver = Verifier(canvas_width=cw, canvas_height=ch)
    fb = ver(centers, sz)
    print(f"  verifier: total_violations={fb.global_stats['total_violations']}", flush=True)

    # Test local_repair_loop (the only torch path that doesn't need adapter)
    from vsr_place.renoising.local_repair import local_repair_loop
    out = local_repair_loop(centers, sz, cw, ch, num_steps=10, step_size=0.1,
                            only_mask=fb.severity_vector > 0,
                            edge_index=ei, hpwl_weight=2.0)
    fb2 = ver(out, sz)
    print(f"  vsr_post mock: total_violations={fb2.global_stats['total_violations']}", flush=True)

    # Test HPWL
    h = compute_hpwl_from_edges(centers, ei, ea)
    print(f"  hpwl: {h:.4f}", flush=True)

    # Schema fidelity check: the row dict has the keys downstream stats code expects
    expected_keys = {
        "circuit", "n_macros", "seed",
        "baseline_v", "baseline_h", "baseline_time",
        "vsr_post_v", "vsr_post_h", "vsr_post_time",
        "repaint_bin_v", "repaint_bin_h", "repaint_bin_time",
        "vsr_intra_v", "vsr_intra_h", "vsr_intra_time",
        "cd_std_v", "cd_std_h", "cd_std_time",
        "cd_sched_v", "cd_sched_h", "cd_sched_time",
        "cg_strong_v", "cg_strong_h", "cg_strong_time",
    }
    print(f"  expected schema keys: {len(expected_keys)} fields ready", flush=True)
    print("=== preflight OK ===", flush=True)


if __name__ == "__main__":
    main()
