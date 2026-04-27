"""Q4: does iterative VSR + cd-sched alternation reduce residual?

For each (circuit, seed) trial, run:
  state_0 = raw
  for k in 1..K:
    state_k = cd_sched(VSR(state_{k-1}))
  Track residual_v + macros_h + overlap_area + max_overlap at each k.

Also report violation-area decomposition (Q3 / S2 — error analysis).

Output: results/vsr_extra/iterative_vsr_legalizer.json
"""
from __future__ import annotations

import gc, json, os, sys, time, types, traceback
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
LAMBDA = 8.0
K_ALT = 3                  # number of (VSR, cd_sched) iterations
NUM_REPAIR = 100
CHECKPOINT = "checkpoints/large-v2/large-v2.ckpt"
OUT = REPO / "results" / "vsr_extra" / "iterative_vsr_legalizer.json"

CD_SCHED = dict(step_size=0.2, grad_descent_steps=5000, softmax_min=5.0, softmax_max=50.0,
                softmax_critical_factor=1.0, hpwl_weight=4e-5, legality_weight=2.0,
                macros_only=True)


def _overlap_metrics(centers, sizes):
    """Returns (total_overlap_area, max_pair_overlap, n_overlap_pairs)."""
    half = sizes / 2.0
    mins = centers - half; maxs = centers + half
    inter_min = torch.maximum(mins.unsqueeze(1), mins.unsqueeze(0))
    inter_max = torch.minimum(maxs.unsqueeze(1), maxs.unsqueeze(0))
    d = torch.clamp(inter_max - inter_min, min=0.0)
    A = d[..., 0] * d[..., 1]
    eye = torch.eye(centers.shape[0], dtype=torch.bool, device=centers.device)
    A = A.masked_fill(eye, 0.0)
    n_pairs = int((A > 0).sum().item() // 2)
    return float(A.sum().item() / 2.0), float(A.max().item()), n_pairs


def _spatial_clustering(centers, sizes, severity_threshold=0):
    """Q3: are violations spatially clustered or scattered?
    Returns bounding-box-area / canvas-area of violator macros.
    """
    return None  # placeholder; set in main where verifier exposes severity


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

    for ci in CIRCUITS:
        name = NAMES[ci]
        full_x, full_cond = val_full[ci]
        x_in, cond = val[ci]

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
                cent = ad.decode_placement(pl_base, cond)[0].cpu()
                sz = ad.decode_placement(pl_base, cond)[1].cpu()
                ei = cond.edge_index.cpu()
                ea = cond.edge_attr.cpu() if cond.edge_attr is not None else None

                def measure(c, label):
                    fb = ver(c, sz)
                    v = int(fb.global_stats["total_violations"])
                    h = float(compute_hpwl_from_edges(c, ei, ea))
                    oa, mx, npairs = _overlap_metrics(c, sz)
                    sev = fb.severity_vector
                    n_violators = int((sev > 0).sum().item())
                    # Spatial clustering: bbox of violators / canvas
                    bbox = 0.0
                    if n_violators > 0:
                        viol = c[sev > 0]
                        bbox_w = float(viol[:, 0].max() - viol[:, 0].min())
                        bbox_h = float(viol[:, 1].max() - viol[:, 1].min())
                        bbox = (bbox_w * bbox_h) / (cw * ch)
                    print(f"  {label}: v={v} h={h:.0f} oa={oa:.1f} mx={mx:.1f} "
                          f"npairs={npairs} viols={n_violators} bbox={bbox:.3f}", flush=True)
                    return dict(v=v, h=h, oa=oa, mx=mx, npairs=npairs,
                                n_violators=n_violators, bbox=bbox)

                row["baseline"] = measure(cent, "baseline")
                row["baseline_v"] = row["baseline"]["v"]

                # Iteration 0: cd_sched on raw (reference)
                pl_in = pl_base.unsqueeze(0).detach().to(ad.device)
                t0 = time.time()
                pl_cd = cd_leg.legalize(pl_in, cond.to(ad.device), **CD_SCHED)
                if isinstance(pl_cd, tuple): pl_cd = pl_cd[0]
                cent_cd = ad.decode_placement(pl_cd.squeeze(0), cond)[0].cpu()
                t = time.time() - t0
                row["raw_cd"] = measure(cent_cd, "raw->cd_sched")
                row["raw_cd"]["time"] = t

                # Iterative VSR + cd_sched
                cur = cent.clone()
                for k in range(1, K_ALT + 1):
                    sev = ver(cur, sz).severity_vector
                    cur = local_repair_loop(cur, sz, cw, ch, num_steps=NUM_REPAIR,
                                            step_size=0.3, only_mask=(sev > 0),
                                            edge_index=ei, hpwl_weight=LAMBDA)
                    row[f"after_vsr_{k}"] = measure(cur, f"after_vsr_{k}")
                    # encode -> cd
                    pl_norm = ad.encode_placement(cur, cond).to(ad.device).unsqueeze(0)
                    t0 = time.time()
                    pl_cd = cd_leg.legalize(pl_norm, cond.to(ad.device), **CD_SCHED)
                    if isinstance(pl_cd, tuple): pl_cd = pl_cd[0]
                    cur = ad.decode_placement(pl_cd.squeeze(0), cond)[0].cpu()
                    t = time.time() - t0
                    row[f"after_cd_{k}"] = measure(cur, f"after_cd_{k}")
                    row[f"after_cd_{k}"]["time"] = t

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
