"""Re-run main_neurips trials saving full per-trial metrics:
   - macros-only HPWL (matches main paper)
   - FULL-DESIGN HPWL (with cells fixed at .pl positions)
   - total overlap area (sum_ij A_ij / 2)
   - max single-pair overlap
   - fully-legal indicator (v == 0)
   Plus baseline / vsr_post / vsr_intra centers for downstream figure regen.

Output: results/vsr_extra/full_metrics.json + per-trial pickle saves.
"""
from __future__ import annotations

import gc
import json
import os
import pickle
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
SEEDS = [42, 123, 300, 2024]
LAMBDA = 2.0
T_INTRA = 0.3
NUM_REPAIR = 100
CHECKPOINT = "checkpoints/large-v2/large-v2.ckpt"
OUT = REPO / "results" / "vsr_extra" / "full_metrics.json"


def _overlap_metrics(centers: "torch.Tensor", sizes: "torch.Tensor"):
    """Total overlap area + max pair overlap."""
    half = sizes / 2.0
    mins = centers - half
    maxs = centers + half
    inter_min = torch.maximum(mins.unsqueeze(1), mins.unsqueeze(0))
    inter_max = torch.minimum(maxs.unsqueeze(1), maxs.unsqueeze(0))
    inter_dims = torch.clamp(inter_max - inter_min, min=0.0)
    A = inter_dims[..., 0] * inter_dims[..., 1]
    eye = torch.eye(centers.shape[0], dtype=torch.bool, device=centers.device)
    A = A.masked_fill(eye, 0.0)
    total = float(A.sum().item() / 2.0)  # symmetric
    mx = float(A.max().item())
    return total, mx


def _full_hpwl(centers_macro, full_centers, macro_mask,
               edge_index_full, edge_attr_full):
    """Full-design HPWL = HPWL on full netlist with macros at new positions
    and cells held at their .pl baselines."""
    import torch
    from vsr_place.metrics.hpwl import compute_hpwl_from_edges
    full = full_centers.clone()
    full[macro_mask] = centers_macro
    return compute_hpwl_from_edges(full, edge_index_full, edge_attr_full)


def vsr_intra_soft(adapter, x_hat_0, cond, severity, t_start=T_INTRA, num_steps=NUM_REPAIR):
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


def main():
    from run_vsr import load_benchmark_data, filter_macros_only
    from vsr_place.backbone.adapter import ChipDiffusionAdapter
    from vsr_place.verifier.verifier import Verifier
    from vsr_place.renoising.local_repair import local_repair_loop
    from vsr_place.metrics.hpwl import compute_hpwl_from_edges

    val_full = load_benchmark_data("ispd2005")  # full graphs
    val = [filter_macros_only(x, c) for x, c in val_full]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        rows = json.load(open(OUT))
    else:
        rows = []
    done = {(r["circuit"], r["seed"]) for r in rows if r.get("baseline_v") is not None}
    print(f"Resume: {len(done)} pairs done", flush=True)

    for ci in CIRCUITS:
        name = NAMES[ci]
        full_x, full_cond = val_full[ci]      # full graph (cells + macros)
        x_in, cond = val[ci]                  # macro-only
        # Macro mask: which entries in full_cond.x are macros?
        if hasattr(full_cond, "is_macros") and full_cond.is_macros is not None:
            macro_mask = full_cond.is_macros.bool()
        else:
            macro_mask = None

        for seed in SEEDS:
            if (name, seed) in done:
                continue
            print(f"\n=== {name} seed={seed} ===", flush=True)
            try:
                torch.manual_seed(seed); torch.cuda.empty_cache()
                ad = ChipDiffusionAdapter.from_checkpoint(
                    CHECKPOINT, device="cuda", input_shape=tuple(x_in.shape), guidance="opt",
                )
                cw, ch = ad.get_canvas_size(cond)
                pl_base = ad.guided_sample(cond.to("cuda"), num_samples=1).squeeze(0)
                cent_b, sz = ad.decode_placement(pl_base, cond)
                cent_b = cent_b.cpu(); sz = sz.cpu()
                ei = cond.edge_index.cpu()
                ea = cond.edge_attr.cpu() if cond.edge_attr is not None else None
                ver = Verifier(canvas_width=cw, canvas_height=ch)
                fb_b = ver(cent_b, sz)
                base_v = int(fb_b.global_stats["total_violations"])
                base_h = float(compute_hpwl_from_edges(cent_b, ei, ea))
                base_oa, base_mx = _overlap_metrics(cent_b, sz)
                base_full_h = None
                if macro_mask is not None:
                    base_full_h = float(_full_hpwl(cent_b, full_x.cpu(), macro_mask,
                                                    full_cond.edge_index.cpu(),
                                                    full_cond.edge_attr.cpu() if full_cond.edge_attr is not None else None))

                sev = fb_b.severity_vector
                # vsr_post lambda=2
                cent_p = local_repair_loop(cent_b, sz, cw, ch, num_steps=NUM_REPAIR,
                                           step_size=0.3, only_mask=(sev > 0),
                                           edge_index=ei, hpwl_weight=LAMBDA)
                fb_p = ver(cent_p, sz)
                p_v = int(fb_p.global_stats["total_violations"])
                p_h = float(compute_hpwl_from_edges(cent_p, ei, ea))
                p_oa, p_mx = _overlap_metrics(cent_p, sz)
                p_full_h = None
                if macro_mask is not None:
                    p_full_h = float(_full_hpwl(cent_p, full_x.cpu(), macro_mask,
                                                full_cond.edge_index.cpu(),
                                                full_cond.edge_attr.cpu() if full_cond.edge_attr is not None else None))

                # vsr_intra_soft
                pl_intra = vsr_intra_soft(ad, pl_base, cond, sev)
                cent_i, _ = ad.decode_placement(pl_intra, cond)
                cent_i = cent_i.cpu()
                fb_i = ver(cent_i, sz)
                i_v = int(fb_i.global_stats["total_violations"])
                i_h = float(compute_hpwl_from_edges(cent_i, ei, ea))
                i_oa, i_mx = _overlap_metrics(cent_i, sz)
                i_full_h = None
                if macro_mask is not None:
                    i_full_h = float(_full_hpwl(cent_i, full_x.cpu(), macro_mask,
                                                full_cond.edge_index.cpu(),
                                                full_cond.edge_attr.cpu() if full_cond.edge_attr is not None else None))

                row = {
                    "circuit": name, "seed": seed, "n_macros": int(cond.x.shape[0]),
                    "baseline_v": base_v, "baseline_h": base_h,
                    "baseline_overlap_area": base_oa, "baseline_max_overlap": base_mx,
                    "baseline_full_hpwl": base_full_h,
                    "vsr_post_v": p_v, "vsr_post_h": p_h,
                    "vsr_post_overlap_area": p_oa, "vsr_post_max_overlap": p_mx,
                    "vsr_post_full_hpwl": p_full_h,
                    "vsr_intra_v": i_v, "vsr_intra_h": i_h,
                    "vsr_intra_overlap_area": i_oa, "vsr_intra_max_overlap": i_mx,
                    "vsr_intra_full_hpwl": i_full_h,
                }
                rows.append(row)
                with open(OUT, "w") as f:
                    json.dump(rows, f, indent=2)
                print(f"  base v={base_v} h={base_h:.0f} oa={base_oa:.2f} mx={base_mx:.2f} "
                      f"full_h={base_full_h}", flush=True)
                print(f"  post v={p_v} h={p_h:.0f} oa={p_oa:.2f} mx={p_mx:.2f} "
                      f"full_h={p_full_h}", flush=True)
                print(f"  intra v={i_v} h={i_h:.0f} oa={i_oa:.2f} mx={i_mx:.2f} "
                      f"full_h={i_full_h}", flush=True)
                del ad
                gc.collect(); torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                print("  OOM", flush=True)
                rows.append({"circuit": name, "seed": seed, "error": "OOM"})
                with open(OUT, "w") as f: json.dump(rows, f, indent=2)
                gc.collect(); torch.cuda.empty_cache()
            except Exception as e:
                traceback.print_exc()
                rows.append({"circuit": name, "seed": seed, "error": str(e)[:200]})
                with open(OUT, "w") as f: json.dump(rows, f, indent=2)

    print(f"\nDone. {len(rows)} rows.")


if __name__ == "__main__":
    main()
