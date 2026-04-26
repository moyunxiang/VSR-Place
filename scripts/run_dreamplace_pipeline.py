"""External legalizer pipeline using DREAMPlace as second-stage.

Per (circuit, seed):
  1. Generate ChipDiffusion baseline draft (paired with run_round2_review)
  2. Compute VSR-post(lambda=8) repair
  3. For each treatment (raw_diffusion / vsr_post), build a
     Bookshelf input dir with:
       - original .nodes/.nets/.scl/.wts unchanged
       - .pl with macros at treatment positions, cells at industrial .lg.pl
  4. Run DREAMPlace with global_place_flag=0, legalize_flag=1,
     detailed_place_flag=1  (use as a legalizer, not global placer)
  5. Read output .pl, compute residual violations + full-design HPWL

Output: results/vsr_extra/dreamplace_pipeline.json

Run on AutoDL only (DREAMPlace requires CUDA + the built install dir).
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
# Full coverage: 4 seeds = 24 trials × 4 treatments (raw, vsr8, fd_pure, fd_spring) = 96 DP runs
SEEDS = [42, 123, 300, 2024]
LAMBDA8 = 8.0
NUM_REPAIR = 100

CHECKPOINT = "checkpoints/large-v2/large-v2.ckpt"
DREAMPLACE_DIR = "/root/autodl-tmp/DREAMPlace/install"
BENCH_DIR = "/root/autodl-tmp/VSR-Place/third_party/chipdiffusion/benchmarks/ispd2005/ispd2005"
OUT = REPO / "results" / "vsr_extra" / "dreamplace_pipeline.json"

# DREAMPlace config: legalize-only mode (no global placement)
DP_LEGALIZE_CONFIG_TEMPLATE = {
    "gpu": 1,
    "num_bins_x": 512, "num_bins_y": 512,
    # Provide a stages entry so Placer.py can read learning_rate, but we
    # disable GP via global_place_flag=0 below so this is never executed.
    "global_place_stages": [
        {"num_bins_x": 512, "num_bins_y": 512, "iteration": 0, "learning_rate": 0.01,
         "wirelength": "weighted_average", "optimizer": "nesterov",
         "Llambda_density_weight_iteration": 1, "Lsub_iteration": 1}
    ],
    "target_density": 1.0,
    "density_weight": 8e-5,
    "gamma": 4.0,
    "random_seed": 1000,
    "scale_factor": 1.0,
    "ignore_net_degree": 100,
    "enable_fillers": 0,
    "gp_noise_ratio": 0.025,
    "global_place_flag": 0,
    "legalize_flag": 1,
    "detailed_place_flag": 1,
    "detailed_place_engine": "",
    "detailed_place_command": "",
    "stop_overflow": 0.07,
    "dtype": "float32",
    "plot_flag": 0,
    "random_center_init_flag": 0,
    "gift_init_flag": 0,
    "sort_nets_by_degree": 0,
    "num_threads": 8,
    "deterministic_flag": 1,
}


def read_pl(pl_path):
    """Read Bookshelf .pl, return dict name -> (x, y, orient, fixed)."""
    out = {}
    for line in open(pl_path):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("UCLA"):
            continue
        parts = line.split()
        if len(parts) < 3: continue
        name = parts[0]
        try:
            x = float(parts[1]); y = float(parts[2])
        except ValueError:
            continue
        orient = parts[4] if len(parts) > 4 and parts[3] == ":" else "N"
        fixed = "/FIXED" in line
        out[name] = (x, y, orient, fixed)
    return out


def write_pl(pl_path, src_pl_lines, overrides):
    """Write a new .pl: keep header + line structure from src, override
    positions for names in `overrides` (dict name -> (x, y))."""
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


def get_macro_names(nodes_path, sizes, indices_in_cond):
    """Bookshelf .nodes lists nodes in order with their (w, h) and a
    'terminal' or 'terminal_NI' flag for fixed-position macros.  We need
    to identify which names correspond to the macros in cond.x.

    ISPD2005 uses 'terminal' / 'terminal_NI' to mark macros (immovable).
    ChipDiffusion's data loader treats both as macros (movable in our
    sense).  We assume the order in cond.x matches the order in .nodes
    where (terminal | terminal_NI) is True, which is how ChipDiffusion's
    benchmark loader builds the macro list.
    """
    terms = []
    in_nodes = False
    for line in open(nodes_path):
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("UCLA"):
            continue
        parts = s.split()
        if parts[0] in ("NumNodes", "NumTerminals"):
            continue
        if len(parts) >= 3:
            name = parts[0]
            try:
                w = int(parts[1]); h = int(parts[2])
            except ValueError:
                continue
            is_term = any(t in s for t in ["terminal_NI", "terminal"])
            if is_term:
                terms.append((name, w, h))
    # Heuristic: cond.x macros == terms in same order
    return terms


def run_dreamplace(work_dir, aux_path, seed):
    cfg = dict(DP_LEGALIZE_CONFIG_TEMPLATE)
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
    # DREAMPlace writes to <cwd>/results/<base>/<base>.{gp,lg,dp}.pl
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

    val_full = load_benchmark_data("ispd2005")
    val = [filter_macros_only(x, c) for x, c in val_full]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = json.load(open(OUT)) if OUT.exists() else []
    done = {(r["circuit"], r["seed"]) for r in rows if r.get("baseline_v") is not None}

    for ci in CIRCUITS:
        name = NAMES[ci]
        full_x, full_cond = val_full[ci]
        x_in, cond = val[ci]
        macro_mask = full_cond.is_macros.bool() if hasattr(full_cond, "is_macros") and full_cond.is_macros is not None else None
        ei_full = full_cond.edge_index.cpu()
        ea_full = full_cond.edge_attr.cpu() if full_cond.edge_attr is not None else None

        # Bookshelf paths
        bench_circuit = Path(BENCH_DIR) / name
        nodes_path = bench_circuit / f"{name}.nodes"
        macro_terms = get_macro_names(nodes_path, None, None)
        n_macros_bookshelf = len(macro_terms)
        print(f"\n=== {name}: {n_macros_bookshelf} terminals/macros in bookshelf ===", flush=True)

        for seed in SEEDS:
            if (name, seed) in done:
                continue
            print(f"\n--- {name} seed={seed} ---", flush=True)
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
                row["baseline_v"] = bv; row["baseline_h"] = bh
                print(f"  baseline v={bv} h={bh:.0f}", flush=True)

                # VSR-post
                cent_v = local_repair_loop(cent_b, sz, cw, ch, num_steps=NUM_REPAIR,
                                           step_size=0.3, only_mask=(sev > 0),
                                           edge_index=ei, hpwl_weight=LAMBDA8)
                # FD-pure (no mask, no spring) — classical force-directed
                cent_fd = local_repair_loop(cent_b, sz, cw, ch, num_steps=NUM_REPAIR,
                                            step_size=0.3, only_mask=None,
                                            edge_index=None, hpwl_weight=0.0)
                # FD+spring (no mask, net-spring weight 1.0) — NTUplace3-style
                cent_fs = local_repair_loop(cent_b, sz, cw, ch, num_steps=NUM_REPAIR,
                                            step_size=0.3, only_mask=None,
                                            edge_index=ei, hpwl_weight=1.0)
                # Sanity check we have macro count match
                if cent_b.shape[0] != n_macros_bookshelf:
                    print(f"  WARNING: cond macros {cent_b.shape[0]} != bookshelf macros {n_macros_bookshelf}", flush=True)

                # Build override dicts: convert centers to bottom-left bookshelf coords
                # cent is already in absolute (canvas) coords as float; bookshelf is integer
                def make_overrides(centers):
                    half = sz / 2.0
                    overrides = {}
                    for i, (mn, mw, mh) in enumerate(macro_terms[:cent_b.shape[0]]):
                        x_bl = float(centers[i, 0]) - float(half[i, 0])
                        y_bl = float(centers[i, 1]) - float(half[i, 1])
                        overrides[mn] = (x_bl, y_bl)
                    return overrides

                # Reference .pl source: prefer .lg.pl (legalized industrial) for cell positions
                src_pl = bench_circuit / f"{name}.lg.pl"
                if not src_pl.exists():
                    src_pl = bench_circuit / f"{name}.pl"
                src_pl_lines = open(src_pl).readlines()

                for treat, centers in [("raw", cent_b), ("vsr8", cent_v),
                                       ("fdpure", cent_fd), ("fdspring", cent_fs)]:
                    work = REPO / "results" / "vsr_extra" / f"dp_{name}_seed{seed}_{treat}"
                    if work.exists(): shutil.rmtree(work)
                    work.mkdir(parents=True)
                    # Symlink read-only files
                    for ext in (".aux", ".nodes", ".nets", ".scl", ".wts"):
                        src = bench_circuit / f"{name}{ext}"
                        if src.exists():
                            (work / f"{name}{ext}").symlink_to(src)
                    # Write modified .pl
                    new_pl = work / f"{name}.pl"
                    overrides = make_overrides(centers)
                    write_pl(new_pl, src_pl_lines, overrides)
                    aux = work / f"{name}.aux"

                    out_pl, t, err = run_dreamplace(work, str(aux), seed)
                    if out_pl is None:
                        print(f"  {treat} DREAMPlace FAIL: {err[:200]}", flush=True)
                        row[f"{treat}_dp_error"] = (err or "no_output")[:300]
                        row[f"{treat}_dp_time"] = t
                        continue
                    # Read output .pl, get final macro positions, compute metrics
                    out_d = read_pl(out_pl)
                    final_centers = []
                    for i, (mn, mw, mh) in enumerate(macro_terms[:cent_b.shape[0]]):
                        if mn in out_d:
                            x, y, _, _ = out_d[mn]
                            final_centers.append([x + mw / 2.0, y + mh / 2.0])
                        else:
                            final_centers.append([float(centers[i, 0]), float(centers[i, 1])])
                    final_t = torch.tensor(final_centers, dtype=torch.float32)
                    fb = ver(final_t, sz)
                    fv = int(fb.global_stats["total_violations"])
                    fh = float(compute_hpwl_from_edges(final_t, ei, ea))
                    row[f"{treat}_dp_v"] = fv
                    row[f"{treat}_dp_h"] = fh
                    row[f"{treat}_dp_time"] = t
                    print(f"  {treat}+DREAMPlace: v={fv} ({(fv-bv)/max(bv,1)*100:+.1f}%) "
                          f"h={fh:.0f} ({(fh-bh)/max(abs(bh),1e-9)*100:+.1f}%) ({t:.0f}s)", flush=True)

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
