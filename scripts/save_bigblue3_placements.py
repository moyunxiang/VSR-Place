"""Save bigblue3 placements (baseline + vsr_post + vsr_intra_soft) for fig9.

Replicates the EXACT protocol of run_main_neurips.py:
  - guided sample (CD opt mode, num_steps=100)
  - vsr_post: local_repair_loop with lambda=2, num_steps=100, step_size=0.3
  - vsr_intra_soft: structured RePaint with severity-weighted soft mask,
    t_start=0.3, num_steps=100

Output: results/placements/bigblue3_neurips.pkl with keys:
  name, seed, canvas_w, canvas_h, sizes, edge_index,
  baseline_centers, baseline_severity,
  vsr_post_centers, vsr_post_severity,
  vsr_intra_centers, vsr_intra_severity,
  stats (per method)
"""
from __future__ import annotations

import os
import pickle
import sys
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

CIRCUIT = 6  # bigblue3
SEED = 42
LAMBDA = 2.0
T_INTRA = 0.3
NUM_STEPS = 100
CHECKPOINT = "checkpoints/large-v2/large-v2.ckpt"
OUT = REPO / "results" / "placements" / "bigblue3_neurips.pkl"


def vsr_intra_soft(adapter, x_hat_0, cond, severity, t_start, num_steps):
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
                start_idx = i
                break
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
            x_denoised = r[0] if isinstance(r, tuple) else r
            if i < len(timesteps) - 2:
                eps_known = torch.randn_like(x_hat_0)
                x_known = adapter.scheduler.add_noise(x_hat_0, eps_known, t_next)
            else:
                x_known = x_hat_0
            x = soft_mask * x_denoised + (1.0 - soft_mask) * x_known
    return x.squeeze(0)


def main():
    from run_vsr import load_benchmark_data, filter_macros_only
    from vsr_place.backbone.adapter import ChipDiffusionAdapter
    from vsr_place.verifier.verifier import Verifier
    from vsr_place.renoising.local_repair import local_repair_loop
    from vsr_place.metrics.hpwl import compute_hpwl_from_edges

    val = load_benchmark_data("ispd2005")
    val = [filter_macros_only(x, c) for x, c in val]
    x_in, cond = val[CIRCUIT]
    name = "bigblue3"

    torch.manual_seed(SEED)
    torch.cuda.empty_cache()
    ad = ChipDiffusionAdapter.from_checkpoint(
        CHECKPOINT, device="cuda", input_shape=tuple(x_in.shape), guidance="opt",
    )
    cw, ch = ad.get_canvas_size(cond)
    cond_cu = cond.to("cuda")

    # baseline
    pl_base = ad.guided_sample(cond_cu, num_samples=1).squeeze(0)
    cent_b, sz = ad.decode_placement(pl_base, cond)
    cent_b = cent_b.cpu(); sz = sz.cpu()
    ei = cond.edge_index.cpu()
    ea = cond.edge_attr.cpu() if cond.edge_attr is not None else None
    ver = Verifier(canvas_width=cw, canvas_height=ch)
    fb_b = ver(cent_b, sz)
    base_v = fb_b.global_stats["total_violations"]
    base_h = compute_hpwl_from_edges(cent_b, ei, ea)
    print(f"baseline: v={base_v}  h={base_h:.1f}", flush=True)

    # vsr_post
    sev = fb_b.severity_vector
    cent_post = local_repair_loop(cent_b, sz, cw, ch, num_steps=NUM_STEPS,
                                  step_size=0.3, only_mask=(sev > 0),
                                  edge_index=ei, hpwl_weight=LAMBDA)
    fb_p = ver(cent_post, sz)
    post_v = fb_p.global_stats["total_violations"]
    post_h = compute_hpwl_from_edges(cent_post, ei, ea)
    print(f"vsr_post: v={post_v}  h={post_h:.1f}", flush=True)

    # vsr_intra_soft
    pl_intra = vsr_intra_soft(ad, pl_base, cond, sev, T_INTRA, NUM_STEPS)
    cent_intra, _ = ad.decode_placement(pl_intra, cond)
    cent_intra = cent_intra.cpu()
    fb_i = ver(cent_intra, sz)
    intra_v = fb_i.global_stats["total_violations"]
    intra_h = compute_hpwl_from_edges(cent_intra, ei, ea)
    print(f"vsr_intra: v={intra_v}  h={intra_h:.1f}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "name": name,
        "seed": SEED,
        "canvas_w": cw,
        "canvas_h": ch,
        "sizes": sz.detach().numpy(),
        "edge_index": ei.detach().numpy(),
        "baseline_centers": cent_b.detach().numpy(),
        "baseline_severity": fb_b.severity_vector.detach().numpy(),
        "vsr_post_centers": cent_post.detach().numpy(),
        "vsr_post_severity": fb_p.severity_vector.detach().numpy(),
        "vsr_intra_centers": cent_intra.detach().numpy(),
        "vsr_intra_severity": fb_i.severity_vector.detach().numpy(),
        "stats": {
            "base_v": int(base_v), "base_h": float(base_h),
            "vsr_post_v": int(post_v), "vsr_post_h": float(post_h),
            "vsr_intra_v": int(intra_v), "vsr_intra_h": float(intra_h),
        },
    }
    with open(OUT, "wb") as f:
        pickle.dump(blob, f)
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
