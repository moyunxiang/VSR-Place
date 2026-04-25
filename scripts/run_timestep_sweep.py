"""Full intra-sampling timestep sweep on all 6 ISPD2005 circuits.

Existing intra_timestep_sweep.json only covers adaptec3 + bigblue3. NeurIPS
reviewer will ask whether t=0.3 is bigblue3-specific. This script expands to
all 6 circuits (adaptec1-4, bigblue1, bigblue3) x t in {0.1, 0.2, 0.3, 0.5, 0.7}
x 3 seeds.

Output: results/ispd2005/timestep_sweep_full.json  (one row per (circuit, seed, t))

Compatible with run_main_neurips.py: uses the SAME baseline guided_sample (so
deltas are vs the same anchor).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import types
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "third_party" / "chipdiffusion"))
sys.path.insert(0, str(REPO / "third_party" / "chipdiffusion" / "diffusion"))
sys.path.insert(0, str(REPO / "scripts"))

_mp = types.ModuleType("moviepy")
_mp_ed = types.ModuleType("moviepy.editor")
_mp_ed.ImageSequenceClip = lambda *a, **k: None
sys.modules["moviepy"] = _mp
sys.modules["moviepy.editor"] = _mp_ed

import torch  # noqa: E402

from vsr_place.backbone.adapter import ChipDiffusionAdapter  # noqa: E402
from vsr_place.verifier.verifier import Verifier  # noqa: E402
from vsr_place.metrics.hpwl import compute_hpwl_from_edges  # noqa: E402

NAMES = ["adaptec1", "adaptec2", "adaptec3", "adaptec4",
         "bigblue1", "bigblue2", "bigblue3", "bigblue4"]
CIRCUITS_DEFAULT = [0, 1, 2, 3, 4, 6]
SEEDS_DEFAULT = [42, 123, 300]
TIMESTEPS_DEFAULT = [0.1, 0.2, 0.3, 0.5, 0.7]
NUM_REPAIR_STEPS = 100


def vsr_intra_soft(adapter, x_hat_0, cond, severity, t_start, num_steps):
    """Soft severity-weighted RePaint at given t_start. (Same as in E1.)"""
    sev = severity.to(adapter.device).float()
    soft_mask = (sev / sev.max().clamp(min=1e-9)).unsqueeze(0).unsqueeze(-1)
    x = x_hat_0.to(adapter.device).unsqueeze(0)
    cond_d = cond.to(adapter.device)
    if t_start <= 1e-6:
        return x.squeeze(0)

    with torch.no_grad():
        adapter.scheduler.set_timesteps(num_steps)
        all_ts = adapter.scheduler.timesteps.to(adapter.device)
        start_idx = 0
        for i, ts in enumerate(all_ts):
            if ts.item() <= t_start + 1e-6:
                start_idx = i
                break
        timesteps = all_ts[start_idx:]

        eps_init = torch.randn_like(x)
        t_first = timesteps[0].expand(1)
        x_t = adapter.scheduler.add_noise(x, eps_init, t_first)
        for i in range(len(timesteps) - 1):
            t = timesteps[i].expand(1).to(adapter.device)
            t_next = timesteps[i + 1].expand(1).to(adapter.device)
            eps_pred = adapter.model(x_t, cond_d, t)
            z = torch.randn_like(x_t) if i < len(timesteps) - 2 else torch.zeros_like(x_t)
            r = adapter.scheduler.step(eps_pred, t, t_next, x_t, z)
            x_denoised = r[0] if isinstance(r, tuple) else r
            if i < len(timesteps) - 2:
                eps_known = torch.randn_like(x)
                x_known = adapter.scheduler.add_noise(x, eps_known, t_next)
            else:
                x_known = x
            x_t = soft_mask * x_denoised + (1 - soft_mask) * x_known
    return x_t.squeeze(0)


def run_pair_t_sweep(checkpoint, cond, x_in, name, seed, ts_list):
    rows = []
    torch.manual_seed(seed)
    torch.cuda.empty_cache()
    ad = ChipDiffusionAdapter.from_checkpoint(
        checkpoint, device="cuda", input_shape=tuple(x_in.shape), guidance="opt",
    )
    cw, ch = ad.get_canvas_size(cond)
    pl_base = ad.guided_sample(cond.to(ad.device), num_samples=1).squeeze(0)
    cent_base, sz = ad.decode_placement(pl_base, cond)
    cent_base = cent_base.cpu(); sz = sz.cpu()
    ei = cond.edge_index.cpu()
    ea = cond.edge_attr.cpu() if cond.edge_attr is not None else None
    ver = Verifier(canvas_width=cw, canvas_height=ch)
    fb = ver(cent_base, sz)
    base_v = fb.global_stats["total_violations"]
    base_h = compute_hpwl_from_edges(cent_base, ei, ea)
    sev = fb.severity_vector

    for t_start in ts_list:
        torch.manual_seed(seed)  # same noise across t for cleaner comparison
        try:
            t0 = time.time()
            pl_intra = vsr_intra_soft(ad, pl_base, cond, sev, t_start, NUM_REPAIR_STEPS)
            cent_intra, _ = ad.decode_placement(pl_intra, cond)
            cent_intra = cent_intra.cpu()
            elapsed = time.time() - t0
            fb_i = ver(cent_intra, sz)
            v = fb_i.global_stats["total_violations"]
            h = compute_hpwl_from_edges(cent_intra, ei, ea)
            rows.append({
                "circuit": name, "n_macros": int(cond.x.shape[0]), "seed": seed,
                "t_start": t_start, "base_v": base_v, "base_h": base_h,
                "v": v, "h": h, "time": elapsed,
            })
            print(f"  t={t_start:.1f}: v={v} ({(base_v-v)/max(base_v,1)*100:+.1f}%) h={h:.1f} ({(h-base_h)/max(base_h,1e-9)*100:+.1f}%)", flush=True)
        except Exception as e:
            print(f"  t={t_start:.1f}: error {e}", flush=True)
            rows.append({"circuit": name, "seed": seed, "t_start": t_start, "error": str(e)[:200]})
    del ad
    gc.collect(); torch.cuda.empty_cache()
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/large-v2/large-v2.ckpt")
    p.add_argument("--circuits", type=int, nargs="+", default=CIRCUITS_DEFAULT)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS_DEFAULT)
    p.add_argument("--timesteps", type=float, nargs="+", default=TIMESTEPS_DEFAULT)
    p.add_argument("--out", default="results/ispd2005/timestep_sweep_full.json")
    p.add_argument("--mock", action="store_true")
    args = p.parse_args()

    if args.mock:
        print("=== timestep sweep preflight ===", flush=True)
        # Just check imports + arg parsing + schema.
        print(f"  circuits: {args.circuits}", flush=True)
        print(f"  seeds: {args.seeds}", flush=True)
        print(f"  timesteps: {args.timesteps}", flush=True)
        sample_row = {"circuit": "fakecircuit", "n_macros": 0, "seed": 42,
                      "t_start": 0.3, "base_v": 0, "base_h": 0.0,
                      "v": 0, "h": 0.0, "time": 0.0}
        json.dumps(sample_row)
        print("  schema valid", flush=True)
        print("=== preflight OK ===", flush=True)
        return

    from run_vsr import load_benchmark_data, filter_macros_only
    val = load_benchmark_data("ispd2005")
    val = [filter_macros_only(x, c) for x, c in val]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if out_path.exists():
        rows = json.load(open(out_path))
    done = {(r["circuit"], r["seed"], r["t_start"]) for r in rows
            if r.get("v") is not None}
    print(f"Resume: {len(rows)} existing, {len(done)} (circuit,seed,t) done", flush=True)

    for ci in args.circuits:
        name = NAMES[ci]
        x_in, cond = val[ci]
        for seed in args.seeds:
            need = [t for t in args.timesteps if (name, seed, t) not in done]
            if not need:
                continue
            print(f"\n=== {name} seed={seed} ===", flush=True)
            try:
                pair_rows = run_pair_t_sweep(args.checkpoint, cond, x_in, name, seed, need)
                rows.extend(pair_rows)
                with open(out_path, "w") as f:
                    json.dump(rows, f, indent=2)
            except torch.cuda.OutOfMemoryError:
                print("  OOM", flush=True)
                rows.append({"circuit": name, "seed": seed, "error": "OOM"})
                with open(out_path, "w") as f:
                    json.dump(rows, f, indent=2)
                gc.collect(); torch.cuda.empty_cache()
    print(f"\nDone. {len(rows)} rows.")


if __name__ == "__main__":
    main()
