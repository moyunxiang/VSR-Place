"""Hyperparameter sweeps for the two ML baselines:
  - classifier-guidance (cg): legality_weight ∈ {0.5, 1, 2, 4, 8}, hpwl=0
  - RePaint-binary       : t_start ∈ {0.1, 0.2, 0.3, 0.5, 0.7}

Same 6 circuits, 1 seed each (seed=42) for cost.  Records per-(circuit, hp).
Output: results/vsr_extra/baseline_sweeps.json
"""
from __future__ import annotations

import gc
import json
import os
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

_mp = types.ModuleType("moviepy")
_mp_ed = types.ModuleType("moviepy.editor")
_mp_ed.ImageSequenceClip = lambda *a, **k: None
sys.modules["moviepy"] = _mp
sys.modules["moviepy.editor"] = _mp_ed

import torch  # noqa: E402

NAMES = ["adaptec1", "adaptec2", "adaptec3", "adaptec4",
         "bigblue1", "bigblue2", "bigblue3", "bigblue4"]
CIRCUITS = [0, 1, 2, 3, 4, 6]
SEED = 42
NUM_STEPS = 100
CHECKPOINT = "checkpoints/large-v2/large-v2.ckpt"
OUT = REPO / "results" / "vsr_extra" / "baseline_sweeps.json"

CG_LEGALITY_WEIGHTS = [0.5, 1.0, 2.0, 4.0, 8.0]
REPAINT_TSTARTS = [0.1, 0.2, 0.3, 0.5, 0.7]


def main():
    from run_vsr import load_benchmark_data, filter_macros_only
    from vsr_place.backbone.adapter import ChipDiffusionAdapter
    from vsr_place.verifier.verifier import Verifier
    from vsr_place.metrics.hpwl import compute_hpwl_from_edges

    val = load_benchmark_data("ispd2005")
    val = [filter_macros_only(x, c) for x, c in val]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if OUT.exists():
        rows = json.load(open(OUT))
    done = {(r["family"], r["circuit"], r["hp"]) for r in rows
            if r.get("v") is not None}

    for ci in CIRCUITS:
        name = NAMES[ci]
        x_in, cond = val[ci]
        torch.manual_seed(SEED); torch.cuda.empty_cache()

        # ----- Baseline reference (one guided sample per circuit) -----
        try:
            ad_ref = ChipDiffusionAdapter.from_checkpoint(
                CHECKPOINT, device="cuda", input_shape=tuple(x_in.shape), guidance="opt",
            )
            cw, ch = ad_ref.get_canvas_size(cond)
            pl_ref = ad_ref.guided_sample(cond.to("cuda"), num_samples=1).squeeze(0)
            cent_ref, sz = ad_ref.decode_placement(pl_ref, cond)
            cent_ref = cent_ref.cpu(); sz = sz.cpu()
            ei = cond.edge_index.cpu()
            ea = cond.edge_attr.cpu() if cond.edge_attr is not None else None
            ver = Verifier(canvas_width=cw, canvas_height=ch)
            fb_ref = ver(cent_ref, sz)
            base_v = int(fb_ref.global_stats["total_violations"])
            base_h = float(compute_hpwl_from_edges(cent_ref, ei, ea))
            sev = fb_ref.severity_vector
            mask_bool = sev > 0
            print(f"\n=== {name} seed={SEED}  baseline v={base_v} h={base_h:.0f} ===", flush=True)
            del ad_ref; gc.collect(); torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            print(f"{name} baseline OOM", flush=True)
            continue

        # ----- (1) Classifier-guidance sweep -----
        for w in CG_LEGALITY_WEIGHTS:
            if ("cg", name, w) in done: continue
            try:
                torch.manual_seed(SEED); torch.cuda.empty_cache()
                ad = ChipDiffusionAdapter.from_checkpoint(
                    CHECKPOINT, device="cuda", input_shape=tuple(x_in.shape), guidance="opt",
                )
                if hasattr(ad.model, "legality_guidance_weight"):
                    ad.model.legality_guidance_weight = w
                if hasattr(ad.model, "hpwl_guidance_weight"):
                    ad.model.hpwl_guidance_weight = 0.0
                if hasattr(ad.model, "grad_descent_steps") and ad.model.grad_descent_steps == 0:
                    ad.model.grad_descent_steps = 5
                t0 = time.time()
                pl = ad.guided_sample(cond.to(ad.device), num_samples=1).squeeze(0)
                t = time.time() - t0
                cent, _ = ad.decode_placement(pl, cond)
                cent = cent.detach().cpu()
                fb = ver(cent, sz)
                v = int(fb.global_stats["total_violations"])
                h = float(compute_hpwl_from_edges(cent, ei, ea))
                row = {"family": "cg", "circuit": name, "seed": SEED, "hp": w,
                       "baseline_v": base_v, "baseline_h": base_h,
                       "v": v, "h": h, "time": t}
                rows.append(row)
                print(f"  cg w={w}: v={v} ({(base_v-v)/max(base_v,1)*100:+.1f}%)"
                      f" h={h:.0f} ({(h-base_h)/max(abs(base_h),1e-9)*100:+.1f}%) t={t:.1f}s",
                      flush=True)
                del ad; gc.collect(); torch.cuda.empty_cache()
                with open(OUT, "w") as f: json.dump(rows, f, indent=2)
            except torch.cuda.OutOfMemoryError:
                print(f"  cg w={w} OOM", flush=True)
                rows.append({"family": "cg", "circuit": name, "seed": SEED, "hp": w,
                             "error": "OOM"})
                with open(OUT, "w") as f: json.dump(rows, f, indent=2)
                gc.collect(); torch.cuda.empty_cache()
            except Exception as e:
                print(f"  cg w={w} error: {e}", flush=True)
                rows.append({"family": "cg", "circuit": name, "seed": SEED, "hp": w,
                             "error": str(e)[:200]})
                with open(OUT, "w") as f: json.dump(rows, f, indent=2)

        # ----- (2) RePaint t_start sweep (binary mask) -----
        try:
            ad = ChipDiffusionAdapter.from_checkpoint(
                CHECKPOINT, device="cuda", input_shape=tuple(x_in.shape), guidance="opt",
            )
            for ts in REPAINT_TSTARTS:
                if ("repaint", name, ts) in done: continue
                torch.manual_seed(SEED); torch.cuda.empty_cache()
                try:
                    t0 = time.time()
                    pl_re = ad.denoise_repaint(pl_ref, cond.to(ad.device),
                                               mask_bool.to(ad.device),
                                               start_timestep=ts, num_steps=NUM_STEPS)
                    t = time.time() - t0
                    cent, _ = ad.decode_placement(pl_re, cond)
                    cent = cent.cpu()
                    fb = ver(cent, sz)
                    v = int(fb.global_stats["total_violations"])
                    h = float(compute_hpwl_from_edges(cent, ei, ea))
                    rows.append({"family": "repaint", "circuit": name, "seed": SEED, "hp": ts,
                                 "baseline_v": base_v, "baseline_h": base_h,
                                 "v": v, "h": h, "time": t})
                    print(f"  repaint t={ts}: v={v} ({(base_v-v)/max(base_v,1)*100:+.1f}%)"
                          f" h={h:.0f} ({(h-base_h)/max(abs(base_h),1e-9)*100:+.1f}%) t={t:.1f}s",
                          flush=True)
                    with open(OUT, "w") as f: json.dump(rows, f, indent=2)
                except Exception as e:
                    print(f"  repaint t={ts} error: {e}", flush=True)
                    rows.append({"family": "repaint", "circuit": name, "seed": SEED, "hp": ts,
                                 "error": str(e)[:200]})
                    with open(OUT, "w") as f: json.dump(rows, f, indent=2)
            del ad; gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            print(f"{name} repaint setup error: {e}", flush=True)
            gc.collect(); torch.cuda.empty_cache()

    print(f"\nDone. {len(rows)} rows.")


if __name__ == "__main__":
    main()
