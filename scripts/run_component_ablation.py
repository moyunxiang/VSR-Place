"""Component ablation for the verifier-guided repair operator.

Reviewer question: "is the structured feedback necessary?"  This script
isolates each component of the verifier signal, repair operator, and
selector and measures the marginal effect.

Variants tested per (circuit, seed):
  full           : VSR-post (lambda=2, severity-weighted force, full verifier)
  binary_mask    : same operator, but Boolean mask instead of severity-soft.
                   (force magnitude becomes uniform across offenders)
  no_overlap     : verifier returns boundary only (drop pairwise A_ij)
  no_boundary    : verifier returns overlap only (drop b_i)
  no_attract     : skip wirelength attractive force (lambda=0 effectively)
  no_repulsive   : skip pairwise repulsive force (boundary-only)
  random_select  : random K macros (where K = |offenders| from full mask)
  uniform_select : all macros forced uniformly (no mask, full graph)

Reuse the same diffusion-guided draft per (circuit, seed) for fairness.
Output: results/vsr_extra/component_ablation.json

Pre-flight (--mock) tests verifier shapes + repair signatures locally.
"""
from __future__ import annotations

import argparse
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
CIRCUITS_DEFAULT = [0, 1, 2, 3, 4, 6]
SEEDS_DEFAULT = [42, 123, 300]
LAMBDA_DEFAULT = 2.0
NUM_STEPS = 100
STEP_SIZE = 0.3


def _force_step(
    centers: "torch.Tensor",
    sizes: "torch.Tensor",
    cw: float,
    ch: float,
    edge_index,
    *,
    use_repulsive: bool,
    use_attract: bool,
    use_boundary: bool,
    weight_mode: str,           # "severity" | "binary" | "uniform"
    selector_mode: str,         # "violators" | "random" | "all"
    severity: "torch.Tensor",
    rng_seed: int,
    n_iters: int = NUM_STEPS,
    step: float = STEP_SIZE,
    lam: float = LAMBDA_DEFAULT,
):
    """One-shot ablation-aware repair loop.

    All variants share the explicit-Euler structure of local_repair_loop;
    only the forces and the per-macro multiplier differ.
    """
    import torch

    rng = torch.Generator(device=centers.device)
    rng.manual_seed(rng_seed)

    N = centers.shape[0]
    sev = severity.to(centers.device)
    if selector_mode == "violators":
        mask_pos = sev > 0
    elif selector_mode == "random":
        k = int((sev > 0).sum().item())
        idx = torch.randperm(N, generator=rng, device=centers.device)[:k]
        mask_pos = torch.zeros(N, dtype=torch.bool, device=centers.device)
        mask_pos[idx] = True
    elif selector_mode == "all":
        mask_pos = torch.ones(N, dtype=torch.bool, device=centers.device)
    else:
        raise ValueError(selector_mode)

    if weight_mode == "severity":
        w = sev / sev.max().clamp(min=1e-9)
    elif weight_mode == "binary":
        w = mask_pos.float()
    elif weight_mode == "uniform":
        w = torch.ones(N, device=centers.device)
    else:
        raise ValueError(weight_mode)
    w = w * mask_pos.float()  # gate by selector
    w = w.unsqueeze(-1)        # (N, 1) for broadcasting

    src = edge_index[0].to(centers.device)
    dst = edge_index[1].to(centers.device)
    sw = sizes[:, 0]
    sh = sizes[:, 1]
    s_bar = float(sizes.mean().item())
    canvas = torch.tensor([cw, ch], device=centers.device, dtype=centers.dtype)

    x = centers.clone()
    for _ in range(n_iters):
        diff = x.unsqueeze(0) - x.unsqueeze(1)        # (N, N, 2)
        d = diff.norm(dim=-1).clamp(min=1e-6)
        sum_w = sw.unsqueeze(0) + sw.unsqueeze(1)
        sum_h = sh.unsqueeze(0) + sh.unsqueeze(1)
        depth_x = torch.clamp(sum_w / 2 - diff[..., 0].abs(), min=0.0)
        depth_y = torch.clamp(sum_h / 2 - diff[..., 1].abs(), min=0.0)
        depth = torch.minimum(depth_x, depth_y)
        eye = torch.eye(N, dtype=torch.bool, device=centers.device)
        depth = depth.masked_fill(eye, 0.0)
        unit = diff / d.unsqueeze(-1)
        repulsive = (depth.unsqueeze(-1) * unit).sum(dim=1)  # push i away from j
        repulsive = torch.clamp(repulsive, min=-0.5 * s_bar, max=0.5 * s_bar)

        # Boundary
        bd = torch.zeros_like(x)
        bd[:, 0] = (torch.clamp(sw / 2 - x[:, 0], min=0.0) -
                    torch.clamp(x[:, 0] + sw / 2 - cw, min=0.0))
        bd[:, 1] = (torch.clamp(sh / 2 - x[:, 1], min=0.0) -
                    torch.clamp(x[:, 1] + sh / 2 - ch, min=0.0))

        # Attract along edges
        attr = torch.zeros_like(x)
        if use_attract:
            src_pos = x.index_select(0, src)
            dst_pos = x.index_select(0, dst)
            direction = dst_pos - src_pos
            attr.index_add_(0, src, direction)
            attr.index_add_(0, dst, -direction)
            deg = torch.zeros(N, device=x.device)
            deg.index_add_(0, src, torch.ones_like(src, dtype=torch.float32))
            deg.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
            attr = attr / deg.clamp(min=1.0).unsqueeze(-1)

        force = torch.zeros_like(x)
        if use_repulsive:
            force = force + repulsive
        if use_boundary:
            force = force + bd
        if use_attract:
            force = force + lam * attr * (s_bar / 4)
        x = x + step * w * force

    return x


def _run_variant(name, *, ad, cond, x_in, sz, cw, ch, ei, ea, fb, pl_base, seed):
    """Run a named variant and return (cents, time)."""
    from vsr_place.metrics.hpwl import compute_hpwl_from_edges
    from vsr_place.verifier.verifier import Verifier

    variants = {
        "full":           dict(use_repulsive=True, use_attract=True, use_boundary=True,
                               weight_mode="severity", selector_mode="violators"),
        "binary_mask":    dict(use_repulsive=True, use_attract=True, use_boundary=True,
                               weight_mode="binary", selector_mode="violators"),
        "no_overlap":     dict(use_repulsive=False, use_attract=True, use_boundary=True,
                               weight_mode="severity", selector_mode="violators"),
        "no_boundary":    dict(use_repulsive=True, use_attract=True, use_boundary=False,
                               weight_mode="severity", selector_mode="violators"),
        "no_attract":     dict(use_repulsive=True, use_attract=False, use_boundary=True,
                               weight_mode="severity", selector_mode="violators"),
        "no_repulsive":   dict(use_repulsive=False, use_attract=True, use_boundary=True,
                               weight_mode="severity", selector_mode="violators"),
        "random_select":  dict(use_repulsive=True, use_attract=True, use_boundary=True,
                               weight_mode="severity", selector_mode="random"),
        "uniform_select": dict(use_repulsive=True, use_attract=True, use_boundary=True,
                               weight_mode="uniform", selector_mode="all"),
    }
    cfg = variants[name]
    cent_in, _ = ad.decode_placement(pl_base, cond)
    cent_in = cent_in.cpu()
    sev = fb.severity_vector

    t0 = time.time()
    out_cent = _force_step(cent_in, sz, cw, ch, ei,
                           severity=sev, rng_seed=seed, **cfg)
    elapsed = time.time() - t0

    ver = Verifier(canvas_width=cw, canvas_height=ch)
    fb_v = ver(out_cent, sz)
    v = fb_v.global_stats["total_violations"]
    h = compute_hpwl_from_edges(out_cent, ei, ea)
    return {"v": int(v), "h": float(h), "time": elapsed}


def run_pair(checkpoint, cond, x_in, name, seed, variants):
    from vsr_place.backbone.adapter import ChipDiffusionAdapter
    from vsr_place.verifier.verifier import Verifier
    from vsr_place.metrics.hpwl import compute_hpwl_from_edges

    torch.manual_seed(seed); torch.cuda.empty_cache()
    ad = ChipDiffusionAdapter.from_checkpoint(
        checkpoint, device="cuda", input_shape=tuple(x_in.shape), guidance="opt",
    )
    cw, ch = ad.get_canvas_size(cond)
    pl_base = ad.guided_sample(cond.to(ad.device), num_samples=1).squeeze(0)
    cent_b, sz = ad.decode_placement(pl_base, cond)
    cent_b = cent_b.cpu(); sz = sz.cpu()
    ei = cond.edge_index.cpu()
    ea = cond.edge_attr.cpu() if cond.edge_attr is not None else None
    ver = Verifier(canvas_width=cw, canvas_height=ch)
    fb = ver(cent_b, sz)
    base_v = fb.global_stats["total_violations"]
    base_h = compute_hpwl_from_edges(cent_b, ei, ea)

    row = {"circuit": name, "seed": seed,
           "n_macros": int(cond.x.shape[0]),
           "baseline_v": int(base_v), "baseline_h": float(base_h),
           "variants": {}}
    for v_name in variants:
        try:
            res = _run_variant(v_name, ad=ad, cond=cond, x_in=x_in,
                               sz=sz, cw=cw, ch=ch, ei=ei, ea=ea,
                               fb=fb, pl_base=pl_base, seed=seed)
            print(f"  {v_name:<16} v={res['v']:>7} ({(base_v-res['v'])/max(base_v,1)*100:+.1f}%)"
                  f"  h={res['h']:.1f} ({(res['h']-base_h)/max(abs(base_h),1e-9)*100:+.1f}%)"
                  f"  t={res['time']:.2f}s", flush=True)
            row["variants"][v_name] = res
        except Exception as e:
            print(f"  {v_name} error: {e}", flush=True)
            row["variants"][v_name] = {"error": str(e)[:200]}
    del ad
    gc.collect(); torch.cuda.empty_cache()
    return row


def run_preflight():
    """CPU-only mock to validate _force_step + variant signatures."""
    import torch
    print("=== component-ablation preflight (CPU mock) ===")
    N = 12
    centers = torch.rand(N, 2)
    sizes = torch.full((N, 2), 0.05)
    cw = ch = 1.0
    ei = torch.stack([
        torch.randint(0, N, (12,)),
        torch.randint(0, N, (12,)),
    ])
    severity = torch.rand(N)
    severity[N // 2:] = 0.0  # half are non-violators
    for vname in ["full", "binary_mask", "no_overlap", "no_boundary",
                  "no_attract", "no_repulsive",
                  "random_select", "uniform_select"]:
        out = _force_step(
            centers, sizes, cw, ch, ei,
            severity=severity, rng_seed=42, n_iters=5,
            **{
                "full":           dict(use_repulsive=True, use_attract=True, use_boundary=True,
                                       weight_mode="severity", selector_mode="violators"),
                "binary_mask":    dict(use_repulsive=True, use_attract=True, use_boundary=True,
                                       weight_mode="binary", selector_mode="violators"),
                "no_overlap":     dict(use_repulsive=False, use_attract=True, use_boundary=True,
                                       weight_mode="severity", selector_mode="violators"),
                "no_boundary":    dict(use_repulsive=True, use_attract=True, use_boundary=False,
                                       weight_mode="severity", selector_mode="violators"),
                "no_attract":     dict(use_repulsive=True, use_attract=False, use_boundary=True,
                                       weight_mode="severity", selector_mode="violators"),
                "no_repulsive":   dict(use_repulsive=False, use_attract=True, use_boundary=True,
                                       weight_mode="severity", selector_mode="violators"),
                "random_select":  dict(use_repulsive=True, use_attract=True, use_boundary=True,
                                       weight_mode="severity", selector_mode="random"),
                "uniform_select": dict(use_repulsive=True, use_attract=True, use_boundary=True,
                                       weight_mode="uniform", selector_mode="all"),
            }[vname]
        )
        assert out.shape == (N, 2)
        print(f"  variant={vname:<16} OK shape={tuple(out.shape)}")
    print("=== preflight OK ===")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/large-v2/large-v2.ckpt")
    p.add_argument("--circuits", type=int, nargs="+", default=CIRCUITS_DEFAULT)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS_DEFAULT)
    p.add_argument("--variants", nargs="+",
                   default=["full", "binary_mask", "no_overlap", "no_boundary",
                            "no_attract", "no_repulsive",
                            "random_select", "uniform_select"])
    p.add_argument("--out", default="results/vsr_extra/component_ablation.json")
    p.add_argument("--mock", action="store_true")
    args = p.parse_args()

    if args.mock:
        run_preflight()
        return

    from run_vsr import load_benchmark_data, filter_macros_only
    val = load_benchmark_data("ispd2005")
    val = [filter_macros_only(x, c) for x, c in val]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if out_path.exists():
        rows = json.load(open(out_path))
    done = {(r["circuit"], r["seed"]) for r in rows
            if r.get("baseline_v") is not None}

    for ci in args.circuits:
        name = NAMES[ci]
        x_in, cond = val[ci]
        for seed in args.seeds:
            if (name, seed) in done:
                continue
            print(f"\n=== {name} seed={seed} ===", flush=True)
            try:
                row = run_pair(args.checkpoint, cond, x_in, name, seed, args.variants)
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
                traceback.print_exc()
                rows.append({"circuit": name, "seed": seed, "error": str(e)[:200]})
                with open(out_path, "w") as f:
                    json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
