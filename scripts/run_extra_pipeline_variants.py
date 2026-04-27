"""Phase 5 round 8: extra pipeline variants we haven't tested.

For each (circuit, seed) ∈ 24 trials, generate baseline + VSR variants,
then feed each through DREAMPlace AND cd-sched AND cd-std pipelines.

Variants we add over what's been done:
  - VSR-post λ=12  (haven't tested at >8)
  - VSR-post λ=16  (extreme HPWL bias)
  - VSR-intra-soft (different operator family) at t_start=0.3
  - FD-pure  → cd-sched / cd-std (already done for DREAMPlace, missing cd)
  - FD+spring → cd-sched / cd-std

To minimise time, pipeline only what's missing: write each
treatment's centers, then run all 3 legalizers (DREAMPlace, cd-sched,
cd-std) per treatment.

Output: results/vsr_extra/extra_pipeline_variants.json
"""
from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
import sys
import time
import types
import traceback
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "third_party" / "chipdiffusion"))
sys.path.insert(0, str(REPO / "third_party" / "chipdiffusion" / "diffusion"))
sys.path.insert(0, str(REPO / "scripts"))

_mp = types.ModuleType("moviepy"); _mp_ed = types.ModuleType("moviepy.editor")
_mp_ed.ImageSequenceClip = lambda *a, **k: None
sys.modules["moviepy"] = _mp; sys.modules["moviepy.editor"] = _mp_ed

import torch  # noqa: E402

NAMES = ["adaptec1", "adaptec2", "adaptec3", "adaptec4",
         "bigblue1", "bigblue2", "bigblue3", "bigblue4"]
CIRCUITS = [0, 1, 2, 3, 4, 6]
SEEDS = [42, 123, 300, 2024]
NUM_REPAIR = 100
T_INTRA = 0.3

CHECKPOINT = "checkpoints/large-v2/large-v2.ckpt"
DREAMPLACE_DIR = "/root/autodl-tmp/DREAMPlace/install"
BENCH_DIR = "/root/autodl-tmp/VSR-Place/third_party/chipdiffusion/benchmarks/ispd2005/ispd2005"
OUT = REPO / "results" / "vsr_extra" / "extra_pipeline_variants.json"

DP_CONFIG = {
    "gpu": 1,
    "num_bins_x": 512, "num_bins_y": 512,
    "global_place_stages": [
        {"num_bins_x": 512, "num_bins_y": 512, "iteration": 0, "learning_rate": 0.01,
         "wirelength": "weighted_average", "optimizer": "nesterov",
         "Llambda_density_weight_iteration": 1, "Lsub_iteration": 1}
    ],
    "target_density": 1.0, "density_weight": 8e-5, "gamma": 4.0,
    "scale_factor": 1.0, "ignore_net_degree": 100, "enable_fillers": 0,
    "gp_noise_ratio": 0.025,
    "global_place_flag": 0, "legalize_flag": 1, "detailed_place_flag": 1,
    "detailed_place_engine": "", "detailed_place_command": "",
    "stop_overflow": 0.07, "dtype": "float32", "plot_flag": 0,
    "random_center_init_flag": 0, "gift_init_flag": 0,
    "sort_nets_by_degree": 0, "num_threads": 8, "deterministic_flag": 1,
}

CD_SCHED = dict(step_size=0.2, grad_descent_steps=5000, softmax_min=5.0, softmax_max=50.0,
                softmax_critical_factor=1.0, hpwl_weight=4e-5, legality_weight=2.0,
                macros_only=True)
CD_STD = dict(step_size=0.2, grad_descent_steps=5000, softmax_min=5.0, softmax_max=50.0,
              softmax_critical_factor=1.0, hpwl_weight=0.0, macros_only=True)


def vsr_intra_soft(adapter, x_hat_0, cond, severity, t_start=T_INTRA, num_steps=NUM_REPAIR):
    """Intra-sampling soft-mask repair (matches run_full_metrics)."""
    sev = severity.to(adapter.device).float()
    soft_mask = (sev / sev.max().clamp(min=1e-9)).unsqueeze(0).unsqueeze(-1)
    x_hat_0 = x_hat_0.to(adapter.device).unsqueeze(0)
    cond = cond.to(adapter.device)
    if t_start <= 1e-6:
        return x_hat_0.squeeze(0)
    with torch.no_grad():
        adapter.scheduler.set_timesteps(num_steps)
        all_ts = adapter.scheduler.timesteps.to(adapter.device)
        start_idx = 0
        for i, ts in enumerate(all_ts):
            if ts.item() <= t_start + 1e-6:
                start_idx = i; break
        timesteps = all_ts[start_idx:]
        eps_init = torch.randn_like(x_hat_0)
        t_first = timesteps[0].expand(1)
        x = adapter.scheduler.add_noise(x_hat_0, eps_init, t_first)
        for i in range(len(timesteps) - 1):
            t = timesteps[i].expand(1).to(adapter.device)
            t_next = timesteps[i + 1].expand(1).to(adapter.device)
            eps_pred = adapter.model(x, cond, t)
            z = torch.randn_like(x) if i < len(timesteps) - 2 else torch.zeros_like(x)
            r = adapter.scheduler.step(eps_pred, t, t_next, x, z)
            x_d = r[0] if isinstance(r, tuple) else r
            if i < len(timesteps) - 2:
                eps_known = torch.randn_like(x_hat_0)
                x_known = adapter.scheduler.add_noise(x_hat_0, eps_known, t_next)
            else:
                x_known = x_hat_0
            x = soft_mask * x_d + (1.0 - soft_mask) * x_known
    return x.squeeze(0)


def read_pl(pl_path):
    out = {}
    for line in open(pl_path):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("UCLA"):
            continue
        parts = line.split()
        if len(parts) < 3: continue
        try:
            x = float(parts[1]); y = float(parts[2])
        except ValueError:
            continue
        out[parts[0]] = (x, y)
    return out


def write_pl(pl_path, src_pl_lines, overrides):
    with open(pl_path, "w") as f:
        for line in src_pl_lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("UCLA"):
                f.write(line); continue
            parts = stripped.split()
            if len(parts) < 3:
                f.write(line); continue
            name = parts[0]
            if name in overrides:
                x, y = overrides[name]
                rest = " ".join(parts[3:]) if len(parts) > 3 else ": N"
                f.write(f"{name}\t{int(round(x))}\t{int(round(y))}\t{rest}\n")
            else:
                f.write(line)


def get_macro_terms(nodes_path):
    out = []
    for line in open(nodes_path):
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("UCLA"):
            continue
        parts = s.split()
        if parts[0] in ("NumNodes", "NumTerminals"):
            continue
        if len(parts) >= 3 and any(t in s for t in ["terminal_NI", "terminal"]):
            try:
                w = int(parts[1]); h = int(parts[2])
                out.append((parts[0], w, h))
            except ValueError:
                pass
    return out


def run_dreamplace(work_dir, aux_path, seed):
    cfg = dict(DP_CONFIG)
    cfg["aux_input"] = aux_path
    cfg["random_seed"] = seed
    cfg_path = work_dir / "config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{DREAMPLACE_DIR}:{env.get('PYTHONPATH', '')}"
    env["LD_LIBRARY_PATH"] = (
        "/root/miniconda3/lib/python3.12/site-packages/torch/lib:"
        "/usr/local/cuda-12/lib64:" + env.get("LD_LIBRARY_PATH", "")
    )
    env["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
    t0 = time.time()
    try:
        out = subprocess.run(
            ["/root/miniconda3/bin/python",
             f"{DREAMPLACE_DIR}/dreamplace/Placer.py", str(cfg_path)],
            capture_output=True, text=True, timeout=600, env=env, cwd=str(work_dir),
        )
    except subprocess.TimeoutExpired:
        return None, time.time() - t0, "TIMEOUT"
    elapsed = time.time() - t0
    if out.returncode != 0:
        return None, elapsed, (out.stderr[-500:] if out.stderr else "nonzero rc")
    base = Path(aux_path).stem
    candidates = []
    for sub in ("results/" + base, "."):
        for ext in ("dp.pl", "lg.pl", "gp.pl"):
            candidates.extend((work_dir / sub).glob(f"{base}.{ext}"))
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0], elapsed, None
    return None, elapsed, "no output .pl found"


def main():
    from run_vsr import load_benchmark_data, filter_macros_only
    from vsr_place.backbone.adapter import ChipDiffusionAdapter
    from vsr_place.verifier.verifier import Verifier
    from vsr_place.renoising.local_repair import local_repair_loop
    from vsr_place.metrics.hpwl import compute_hpwl_from_edges
    import legalization as cd_leg

    val_full = load_benchmark_data("ispd2005")
    val = [filter_macros_only(x, c) for x, c in val_full]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = json.load(open(OUT)) if OUT.exists() else []
    done = {(r["circuit"], r["seed"]) for r in rows if r.get("baseline_v") is not None}
    print(f"Resume: {len(done)} pairs done", flush=True)

    for ci in CIRCUITS:
        name = NAMES[ci]
        full_x, full_cond = val_full[ci]
        x_in, cond = val[ci]
        bench_circuit = Path(BENCH_DIR) / name
        macro_terms = get_macro_terms(bench_circuit / f"{name}.nodes")

        for seed in SEEDS:
            if (name, seed) in done: continue
            print(f"\n=== {name} seed={seed} ===", flush=True)
            row = {"circuit": name, "seed": seed, "n_macros": int(cond.x.shape[0])}
            try:
                torch.manual_seed(seed); torch.cuda.empty_cache()
                ad = ChipDiffusionAdapter.from_checkpoint(
                    CHECKPOINT, device="cuda", input_shape=tuple(x_in.shape), guidance="opt",
                )
                cw, ch = ad.get_canvas_size(cond)
                ver = Verifier(canvas_width=cw, canvas_height=ch)
                pl_base = ad.guided_sample(cond.to("cuda"), num_samples=1).squeeze(0)
                cent_b, sz = ad.decode_placement(pl_base, cond)
                cent_b = cent_b.cpu(); sz = sz.cpu()
                ei = cond.edge_index.cpu()
                ea = cond.edge_attr.cpu() if cond.edge_attr is not None else None
                fb_b = ver(cent_b, sz); sev = fb_b.severity_vector
                bv = int(fb_b.global_stats["total_violations"])
                bh = float(compute_hpwl_from_edges(cent_b, ei, ea))
                row.update(baseline_v=bv, baseline_h=bh)
                print(f"  baseline v={bv} h={bh:.0f}", flush=True)

                # Generate VSR variants
                cent_v12 = local_repair_loop(cent_b, sz, cw, ch, num_steps=NUM_REPAIR,
                                             step_size=0.3, only_mask=(sev > 0),
                                             edge_index=ei, hpwl_weight=12.0)
                cent_v16 = local_repair_loop(cent_b, sz, cw, ch, num_steps=NUM_REPAIR,
                                             step_size=0.3, only_mask=(sev > 0),
                                             edge_index=ei, hpwl_weight=16.0)
                # VSR-intra-soft (in placement coord domain)
                pl_intra = vsr_intra_soft(ad, pl_base, cond, sev)
                cent_intra, _ = ad.decode_placement(pl_intra, cond)
                cent_intra = cent_intra.cpu()

                # Macros-only metrics for each variant
                for vname, cent in [("vsr12", cent_v12), ("vsr16", cent_v16),
                                    ("intra", cent_intra)]:
                    fb = ver(cent, sz)
                    v = int(fb.global_stats["total_violations"])
                    h = float(compute_hpwl_from_edges(cent, ei, ea))
                    row[f"{vname}_v"] = v; row[f"{vname}_h"] = h

                # Source .pl
                src_pl = bench_circuit / f"{name}.lg.pl"
                if not src_pl.exists():
                    src_pl = bench_circuit / f"{name}.pl"
                src_pl_lines = open(src_pl).readlines()

                def make_overrides(centers):
                    half = sz / 2.0
                    return {macro_terms[i][0]: (
                        float(centers[i, 0]) - float(half[i, 0]),
                        float(centers[i, 1]) - float(half[i, 1])
                    ) for i in range(min(cent_b.shape[0], len(macro_terms)))}

                # ---- DREAMPlace pipeline for each variant ----
                for vname, cent in [("vsr12", cent_v12), ("vsr16", cent_v16),
                                    ("intra", cent_intra)]:
                    work = REPO / "results" / "vsr_extra" / f"extra_dp_{name}_seed{seed}_{vname}"
                    if work.exists(): shutil.rmtree(work)
                    work.mkdir(parents=True)
                    for ext in (".aux", ".nodes", ".nets", ".scl", ".wts"):
                        src = bench_circuit / f"{name}{ext}"
                        if src.exists():
                            (work / f"{name}{ext}").symlink_to(src)
                    write_pl(work / f"{name}.pl", src_pl_lines, make_overrides(cent))
                    out_pl, t, err = run_dreamplace(work, str(work / f"{name}.aux"), seed)
                    if out_pl is None:
                        row[f"{vname}_dp_error"] = (err or "no_output")[:200]
                        row[f"{vname}_dp_time"] = t
                        print(f"  {vname}+DP FAIL: {(err or '')[:80]}", flush=True)
                        continue
                    out_d = read_pl(out_pl)
                    final = []
                    for i, (mn, mw, mh) in enumerate(macro_terms[:cent_b.shape[0]]):
                        if mn in out_d:
                            x, y = out_d[mn]
                            final.append([x + mw / 2.0, y + mh / 2.0])
                        else:
                            final.append([float(cent[i, 0]), float(cent[i, 1])])
                    final_t = torch.tensor(final, dtype=torch.float32)
                    fb = ver(final_t, sz)
                    fv = int(fb.global_stats["total_violations"])
                    fh = float(compute_hpwl_from_edges(final_t, ei, ea))
                    row[f"{vname}_dp_v"] = fv; row[f"{vname}_dp_h"] = fh
                    row[f"{vname}_dp_time"] = t
                    print(f"  {vname}+DP: v={fv} ({(fv-bv)/max(bv,1)*100:+.1f}%) ({t:.0f}s)", flush=True)

                # ---- cd-sched / cd-std with FD baselines (parallel completeness) ----
                # FD-pure / FD+spring on macros only
                cent_fd = local_repair_loop(cent_b, sz, cw, ch, num_steps=NUM_REPAIR,
                                            step_size=0.3, only_mask=None,
                                            edge_index=None, hpwl_weight=0.0)
                cent_fs = local_repair_loop(cent_b, sz, cw, ch, num_steps=NUM_REPAIR,
                                            step_size=0.3, only_mask=None,
                                            edge_index=ei, hpwl_weight=1.0)

                for vname, cent in [("fdpure", cent_fd), ("fdspring", cent_fs)]:
                    pl_norm = ad.encode_placement(cent, cond)
                    pl_d = pl_norm.to(ad.device).unsqueeze(0)
                    # cd-sched
                    try:
                        t0 = time.time()
                        out_c = cd_leg.legalize(pl_d, cond.to(ad.device), **CD_SCHED)
                        if isinstance(out_c, tuple): out_c = out_c[0]
                        t = time.time() - t0
                        cent_c, _ = ad.decode_placement(out_c.squeeze(0), cond)
                        cent_c = cent_c.cpu()
                        v = int(ver(cent_c, sz).global_stats["total_violations"])
                        h = float(compute_hpwl_from_edges(cent_c, ei, ea))
                        row[f"{vname}_cdsched_v"] = v; row[f"{vname}_cdsched_h"] = h
                        row[f"{vname}_cdsched_time"] = t
                        print(f"  {vname}+cd-sched: v={v} ({(v-bv)/max(bv,1)*100:+.1f}%) ({t:.0f}s)", flush=True)
                    except Exception as e:
                        row[f"{vname}_cdsched_error"] = str(e)[:200]
                    # cd-std
                    try:
                        t0 = time.time()
                        out_c = cd_leg.legalize(pl_d, cond.to(ad.device), **CD_STD)
                        if isinstance(out_c, tuple): out_c = out_c[0]
                        t = time.time() - t0
                        cent_c, _ = ad.decode_placement(out_c.squeeze(0), cond)
                        cent_c = cent_c.cpu()
                        v = int(ver(cent_c, sz).global_stats["total_violations"])
                        h = float(compute_hpwl_from_edges(cent_c, ei, ea))
                        row[f"{vname}_cdstd_v"] = v; row[f"{vname}_cdstd_h"] = h
                        row[f"{vname}_cdstd_time"] = t
                        print(f"  {vname}+cd-std:   v={v} ({(v-bv)/max(bv,1)*100:+.1f}%) ({t:.0f}s)", flush=True)
                    except Exception as e:
                        row[f"{vname}_cdstd_error"] = str(e)[:200]

                rows.append(row)
                with open(OUT, "w") as f: json.dump(rows, f, indent=2)
                del ad; gc.collect(); torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                print("  OOM", flush=True)
                row["error"] = "OOM"; rows.append(row)
                with open(OUT, "w") as f: json.dump(rows, f, indent=2)
                gc.collect(); torch.cuda.empty_cache()
            except Exception as e:
                traceback.print_exc()
                row["error"] = str(e)[:300]; rows.append(row)
                with open(OUT, "w") as f: json.dump(rows, f, indent=2)

    print(f"\nDone. {len(rows)} rows.", flush=True)


if __name__ == "__main__":
    main()
