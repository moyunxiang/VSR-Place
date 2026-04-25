"""Extend Pareto sweep with additional lambda values {3, 6, 8}.

Reads results/ispd2005/pareto_3seed_6w.json (existing 6 lambdas) and
appends rows for the new values, reusing the SAME baseline guided sample
per (circuit, seed) for fairness with existing rows.
"""
from __future__ import annotations

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

NAMES = ["adaptec1", "adaptec2", "adaptec3", "adaptec4",
         "bigblue1", "bigblue2", "bigblue3", "bigblue4"]
CIRCUITS = [0, 1, 2, 3, 4, 6]
SEEDS = [42, 123, 300]
NEW_WEIGHTS = [3.0, 6.0, 8.0]
NUM_STEPS = 100


def main():
    from run_vsr import load_benchmark_data, filter_macros_only
    from vsr_place.backbone.adapter import ChipDiffusionAdapter
    from vsr_place.verifier.verifier import Verifier
    from vsr_place.renoising.local_repair import local_repair_loop
    from vsr_place.metrics.hpwl import compute_hpwl_from_edges

    val = load_benchmark_data("ispd2005")
    val = [filter_macros_only(x, c) for x, c in val]

    out_path = REPO / "results" / "ispd2005" / "pareto_3seed_6w.json"
    rows = json.load(open(out_path))
    done = {(r["circuit"], r["seed"], r["hpwl_weight"]) for r in rows}
    print(f"existing rows: {len(rows)}", flush=True)

    for ci in CIRCUITS:
        name = NAMES[ci]
        x_in, cond = val[ci]
        for seed in SEEDS:
            need = [w for w in NEW_WEIGHTS if (name, seed, w) not in done]
            if not need:
                continue
            print(f"\n=== {name} seed={seed} (need lambdas: {need}) ===", flush=True)
            torch.manual_seed(seed); torch.cuda.empty_cache()
            try:
                ad = ChipDiffusionAdapter.from_checkpoint(
                    "checkpoints/large-v2/large-v2.ckpt", device="cuda",
                    input_shape=tuple(x_in.shape), guidance="opt",
                )
                cw, ch = ad.get_canvas_size(cond)
                pl = ad.guided_sample(cond.to("cuda"), num_samples=1).squeeze(0)
                cent, sz = ad.decode_placement(pl, cond)
                cent, sz = cent.cpu(), sz.cpu()
                ei = cond.edge_index.cpu()
                ea = cond.edge_attr.cpu() if cond.edge_attr is not None else None
                ver = Verifier(canvas_width=cw, canvas_height=ch)
                fb = ver(cent, sz)
                base_v = fb.global_stats["total_violations"]
                base_h = compute_hpwl_from_edges(cent, ei, ea)
                mask = fb.severity_vector > 0
                n_off = int(mask.sum().item())

                for w in need:
                    t0 = time.time()
                    c2 = local_repair_loop(cent, sz, cw, ch, num_steps=NUM_STEPS,
                                           step_size=0.3, only_mask=mask,
                                           edge_index=ei, hpwl_weight=w)
                    t_rep = time.time() - t0
                    fb2 = ver(c2, sz)
                    rv = fb2.global_stats["total_violations"]
                    rh = compute_hpwl_from_edges(c2, ei, ea)
                    row = {
                        "circuit": name, "n_macros": int(cond.x.shape[0]),
                        "seed": seed, "hpwl_weight": w,
                        "baseline_violations": int(base_v),
                        "repaired_violations": int(rv),
                        "viol_reduction_pct": (base_v - rv) / max(base_v, 1) * 100,
                        "baseline_hpwl": float(base_h),
                        "repaired_hpwl": float(rh),
                        "hpwl_delta_pct": (rh - base_h) / max(abs(base_h), 1e-9) * 100,
                        "n_offending_macros": n_off,
                        "t_sample": None, "t_repair": t_rep,
                    }
                    rows.append(row)
                    print(f"  λ={w:.1f}: v={rv} ({(base_v-rv)/max(base_v,1)*100:+.1f}%)"
                          f" h={rh:.1f} ({(rh-base_h)/max(abs(base_h),1e-9)*100:+.1f}%)"
                          f" t={t_rep:.2f}s", flush=True)
                    with open(out_path, "w") as f:
                        json.dump(rows, f, indent=2)
                del ad
                gc.collect(); torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                print("  OOM", flush=True); gc.collect(); torch.cuda.empty_cache()
            except Exception as e:
                print(f"  error: {e}", flush=True)

    print(f"\nDone. Total rows: {len(rows)}")


if __name__ == "__main__":
    main()
