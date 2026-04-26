"""Round-3 review Q7 sensitivity check: cd-std as second-stage legalizer.

For each (circuit, seed) pair, runs:
  - baseline (raw guided sample)
  - VSR-post(lambda=8) — to pair with cd-std as preconditioner
  - cd_std(raw)              ← raw→cd-std
  - cd_std(VSR-post output)  ← raw→VSR(λ=8)→cd-std

Records both macros-only HPWL and full-design HPWL.

Output: results/vsr_extra/cdstd_pipeline.json
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
import traceback
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
SEEDS = [42, 123, 300, 2024]
LAMBDA8 = 8.0
NUM_REPAIR = 100

CD_STD = dict(step_size=0.2, grad_descent_steps=5000, softmax_min=5.0, softmax_max=50.0,
              softmax_critical_factor=1.0, hpwl_weight=0.0, macros_only=True)

CHECKPOINT = "checkpoints/large-v2/large-v2.ckpt"
OUT = REPO / "results" / "vsr_extra" / "cdstd_pipeline.json"


def _overlap_metrics(centers, sizes):
    half = sizes / 2.0
    mins = centers - half; maxs = centers + half
    inter_min = torch.maximum(mins.unsqueeze(1), mins.unsqueeze(0))
    inter_max = torch.minimum(maxs.unsqueeze(1), maxs.unsqueeze(0))
    d = torch.clamp(inter_max - inter_min, min=0.0)
    A = d[..., 0] * d[..., 1]
    eye = torch.eye(centers.shape[0], dtype=torch.bool, device=centers.device)
    A = A.masked_fill(eye, 0.0)
    return float(A.sum().item() / 2.0), float(A.max().item())


def _full_hpwl(centers_macro, full_centers, macro_mask, edge_index_full, edge_attr_full):
    from vsr_place.metrics.hpwl import compute_hpwl_from_edges
    full = full_centers.clone()
    full[macro_mask] = centers_macro
    return float(compute_hpwl_from_edges(full, edge_index_full, edge_attr_full))


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
        macro_mask = full_cond.is_macros.bool() if hasattr(full_cond, "is_macros") and full_cond.is_macros is not None else None
        ei_full = full_cond.edge_index.cpu()
        ea_full = full_cond.edge_attr.cpu() if full_cond.edge_attr is not None else None

        for seed in SEEDS:
            if (name, seed) in done:
                continue
            print(f"\n=== {name} seed={seed} ===", flush=True)
            row = {"circuit": name, "seed": seed, "n_macros": int(cond.x.shape[0])}
            try:
                torch.manual_seed(seed); torch.cuda.empty_cache()

                ad = ChipDiffusionAdapter.from_checkpoint(
                    CHECKPOINT, device="cuda", input_shape=tuple(x_in.shape), guidance="opt",
                )
                cw, ch = ad.get_canvas_size(cond)
                ver = Verifier(canvas_width=cw, canvas_height=ch)

                t0 = time.time()
                pl_base = ad.guided_sample(cond.to("cuda"), num_samples=1).squeeze(0)
                t_base = time.time() - t0
                cent_b, sz = ad.decode_placement(pl_base, cond)
                cent_b = cent_b.cpu(); sz = sz.cpu()
                ei = cond.edge_index.cpu()
                ea = cond.edge_attr.cpu() if cond.edge_attr is not None else None
                fb_b = ver(cent_b, sz); sev = fb_b.severity_vector
                bv = int(fb_b.global_stats["total_violations"])
                bh = float(compute_hpwl_from_edges(cent_b, ei, ea))
                bhf = _full_hpwl(cent_b, full_x.cpu(), macro_mask, ei_full, ea_full) if macro_mask is not None else None
                row.update(baseline_v=bv, baseline_h=bh, baseline_full_h=bhf, baseline_time=t_base)
                print(f"  baseline      v={bv} h={bh:.0f} full_h={bhf}", flush=True)

                # VSR-post(lambda=8)
                t0 = time.time()
                cent_v8 = local_repair_loop(cent_b, sz, cw, ch, num_steps=NUM_REPAIR,
                                            step_size=0.3, only_mask=(sev > 0),
                                            edge_index=ei, hpwl_weight=LAMBDA8)
                t_v8 = time.time() - t0

                # cd_std(raw)
                try:
                    t0 = time.time()
                    pl_in = pl_base.unsqueeze(0).detach().to(ad.device)
                    pl_cdr = cd_leg.legalize(pl_in, cond.to(ad.device), **CD_STD)
                    if isinstance(pl_cdr, tuple): pl_cdr = pl_cdr[0]
                    t_cdr = time.time() - t0
                    cent_cdr, _ = ad.decode_placement(pl_cdr.squeeze(0), cond)
                    cent_cdr = cent_cdr.cpu()
                    fb_cdr = ver(cent_cdr, sz)
                    v = int(fb_cdr.global_stats["total_violations"])
                    h = float(compute_hpwl_from_edges(cent_cdr, ei, ea))
                    hf = _full_hpwl(cent_cdr, full_x.cpu(), macro_mask, ei_full, ea_full) if macro_mask is not None else None
                    row.update(cdstd_raw_v=v, cdstd_raw_h=h, cdstd_raw_full_h=hf, cdstd_raw_time=t_cdr)
                    print(f"  cd_std(raw)   v={v} ({(v-bv)/max(bv,1)*100:+.1f}%) h={h:.0f} full_h={hf} ({t_cdr:.0f}s)", flush=True)
                except Exception as e:
                    print(f"  cd_std(raw) error: {e}", flush=True)
                    row.update(cdstd_raw_error=str(e)[:200])

                # cd_std(VSR output)
                try:
                    pl_v8_norm = ad.encode_placement(cent_v8, cond)
                    pl_v8 = pl_v8_norm.to(ad.device).unsqueeze(0)
                    t0 = time.time()
                    pl_v8c = cd_leg.legalize(pl_v8, cond.to(ad.device), **CD_STD)
                    if isinstance(pl_v8c, tuple): pl_v8c = pl_v8c[0]
                    t_pipe = time.time() - t0
                    cent_pipe, _ = ad.decode_placement(pl_v8c.squeeze(0), cond)
                    cent_pipe = cent_pipe.cpu()
                    fb_p = ver(cent_pipe, sz)
                    v = int(fb_p.global_stats["total_violations"])
                    h = float(compute_hpwl_from_edges(cent_pipe, ei, ea))
                    hf = _full_hpwl(cent_pipe, full_x.cpu(), macro_mask, ei_full, ea_full) if macro_mask is not None else None
                    row.update(pipe_vsr8_cdstd_v=v, pipe_vsr8_cdstd_h=h,
                               pipe_vsr8_cdstd_full_h=hf,
                               pipe_vsr8_cdstd_time=t_pipe + t_v8)
                    print(f"  vsr8+cd_std   v={v} ({(v-bv)/max(bv,1)*100:+.1f}%) h={h:.0f} full_h={hf} ({t_pipe:.0f}s)", flush=True)
                except Exception as e:
                    traceback.print_exc()
                    row.update(pipe_vsr8_cdstd_error=str(e)[:200])

                rows.append(row)
                with open(OUT, "w") as f: json.dump(rows, f, indent=2)
                del ad; gc.collect(); torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                print("  OOM at trial level", flush=True)
                row["error"] = "OOM"; rows.append(row)
                with open(OUT, "w") as f: json.dump(rows, f, indent=2)
                gc.collect(); torch.cuda.empty_cache()
            except Exception as e:
                traceback.print_exc()
                row["error"] = str(e)[:200]; rows.append(row)
                with open(OUT, "w") as f: json.dump(rows, f, indent=2)

    print(f"\nDone. {len(rows)} rows.", flush=True)


if __name__ == "__main__":
    main()
