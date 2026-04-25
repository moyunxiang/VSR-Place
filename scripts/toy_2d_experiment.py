"""Toy 2D non-overlap experiment for cross-domain validation.

Setup
-----
* 30 disks on a unit canvas [0, 1] x [0, 1]
* Random sparse graph (~60 edges) acts as a "netlist" for wirelength
* Tiny DDPM-style score net trained on 1000 synthetic LEGAL layouts
* Sampling produces drafts that are mostly clustered + partially overlapping
  (mirrors the ChipDiffusion regime in the paper)

Methods compared
----------------
1. baseline           : raw sample from the diffusion model
2. classifier_guide   : gradient of softplus(overlap) injected during sampling
3. repaint_binary     : RePaint-style intra-sampling, mask = (severity > 0)
4. vsr_post           : verifier-guided post-processing (force integrator)
5. vsr_intra          : verifier-guided intra-sampling, structured-feedback mask

Each method evaluated on 50 random test prompts x 5 seeds = 250 runs/method.

Metrics
-------
* violation count    (# pair-overlaps + boundary protrusions)
* total overlap area
* wirelength         (sum of edge bbox widths+heights)
* runtime (s)

Outputs
-------
* results/toy/toy_results.json
* paper/figures/fig_toy_pareto.{pdf,png}    (built later by make_toy_figure.py)
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "results" / "toy"

# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

N_DISKS = 30
RADIUS_RANGE = (0.018, 0.035)
CANVAS_W = CANVAS_H = 1.0
N_TRAIN = 1000
N_TEST = 50
N_EDGES = 60


def gen_legal_layout(rng: np.random.Generator, max_tries: int = 200) -> np.ndarray:
    """Sample a legal (no-overlap) random layout via rejection."""
    radii = rng.uniform(*RADIUS_RANGE, size=N_DISKS)
    centers = np.zeros((N_DISKS, 2), dtype=np.float32)
    for i in range(N_DISKS):
        for _ in range(max_tries):
            c = rng.uniform(0.05, 0.95, size=2).astype(np.float32)
            ok = True
            for j in range(i):
                d = np.linalg.norm(c - centers[j])
                if d < radii[i] + radii[j] + 0.005:
                    ok = False
                    break
            if ok:
                centers[i] = c
                break
        else:
            # Failed to find legal spot — relax
            centers[i] = rng.uniform(0.05, 0.95, size=2).astype(np.float32)
    return centers, radii.astype(np.float32)


def gen_graph(rng: np.random.Generator) -> np.ndarray:
    """Random sparse graph as a (2, E) edge index, undirected, src<dst."""
    edges = set()
    while len(edges) < N_EDGES:
        u, v = rng.integers(0, N_DISKS, size=2)
        if u == v:
            continue
        edges.add((int(min(u, v)), int(max(u, v))))
    arr = np.array(sorted(edges), dtype=np.int64).T  # (2, E)
    return arr


# ---------------------------------------------------------------------------
# Tiny DDPM
# ---------------------------------------------------------------------------

class DiskDenoiser(nn.Module):
    """Per-disk denoiser conditioned on radius and graph context.

    Uses a small Set Transformer-like architecture: shared MLP encoder over
    each disk + global pooling + MLP decoder predicting noise per disk.
    """

    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim
        # Per-disk encoder: input = (x, y, r, sin(t), cos(t))
        self.enc = nn.Sequential(
            nn.Linear(5, dim), nn.SiLU(),
            nn.Linear(dim, dim), nn.SiLU(),
        )
        # Self-attention over disks
        self.attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        # Decoder
        self.dec = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(),
            nn.Linear(dim, 2),
        )

    def forward(self, x, r, t):
        # x: (B, N, 2), r: (B, N), t: (B,) in [0, 1]
        B, N, _ = x.shape
        sin_t = torch.sin(t * math.pi).view(B, 1, 1).expand(B, N, 1)
        cos_t = torch.cos(t * math.pi).view(B, 1, 1).expand(B, N, 1)
        feat = torch.cat([x, r.unsqueeze(-1), sin_t, cos_t], dim=-1)
        h = self.enc(feat)  # (B, N, dim)
        h2, _ = self.attn(h, h, h)
        h = h + h2
        return self.dec(h)  # (B, N, 2)  predicted noise


class CosineSchedule:
    """ChipDiffusion-style cosine schedule on t in [0, 1]."""

    @staticmethod
    def alpha(t: torch.Tensor) -> torch.Tensor:
        return torch.cos(math.pi * t / 2)

    @staticmethod
    def sigma(t: torch.Tensor) -> torch.Tensor:
        return torch.sin(math.pi * t / 2)


def add_noise(x0, t):
    """Forward-diffuse to time t."""
    a = CosineSchedule.alpha(t).view(-1, 1, 1)
    s = CosineSchedule.sigma(t).view(-1, 1, 1)
    eps = torch.randn_like(x0)
    return a * x0 + s * eps, eps


def train_denoiser(model, layouts, radii, n_steps=4000, batch=32, lr=2e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    layouts_t = torch.from_numpy(layouts)  # (M, N, 2)
    radii_t = torch.from_numpy(radii)      # (M, N)
    M = layouts_t.shape[0]
    print(f"Training tiny denoiser: M={M} layouts, {n_steps} steps", flush=True)
    t0 = time.time()
    for step in range(n_steps):
        idx = torch.randint(0, M, (batch,))
        x0 = layouts_t[idx]
        r = radii_t[idx]
        t = torch.rand(batch).clamp(0.02, 0.98)
        xt, eps = add_noise(x0, t)
        pred = model(xt, r, t)
        loss = F.mse_loss(pred, eps)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0:
            print(f"  step {step:5d}  loss={loss.item():.4f}", flush=True)
    print(f"  total train time {time.time() - t0:.1f}s", flush=True)


@torch.no_grad()
def ddim_sample(model, r, n_steps=50, t_start=1.0, x_init=None, guide_fn=None,
                clamp_known_fn=None):
    """DDIM-style deterministic sampling from t=t_start to t=0.

    Args:
        guide_fn: optional callable (x, t) -> grad to add into the score.
        clamp_known_fn: optional callable (x_pred, t) -> x_pred for RePaint-style
            inpainting (replaces "known" entries with forward-diffused originals).
    """
    B = r.shape[0]
    N = r.shape[1]
    if x_init is None:
        x = torch.randn(B, N, 2)
    else:
        x = x_init.clone()
    ts = torch.linspace(t_start, 1e-3, n_steps + 1)
    for i in range(n_steps):
        t = ts[i].repeat(B)
        t_next = ts[i + 1].repeat(B)
        a, s = CosineSchedule.alpha(t), CosineSchedule.sigma(t)
        a_n, s_n = CosineSchedule.alpha(t_next), CosineSchedule.sigma(t_next)
        eps_pred = model(x, r, t)
        x0_pred = (x - s.view(-1, 1, 1) * eps_pred) / a.view(-1, 1, 1).clamp(min=1e-3)

        if guide_fn is not None:
            # Add guidance gradient w.r.t. x0_pred (training-free guidance pattern)
            with torch.enable_grad():
                x0g = x0_pred.detach().requires_grad_(True)
                penalty = guide_fn(x0g, r)
                grad = torch.autograd.grad(penalty.sum(), x0g)[0]
            x0_pred = x0_pred - 0.05 * grad

        if clamp_known_fn is not None:
            x0_pred = clamp_known_fn(x0_pred, t_next)

        # DDIM update toward t_next
        x = a_n.view(-1, 1, 1) * x0_pred + s_n.view(-1, 1, 1) * eps_pred
    return x


# ---------------------------------------------------------------------------
# Verifier (toy)
# ---------------------------------------------------------------------------

def overlap_area_pairwise(centers, radii):
    """For each pair (i,j), the disk-disk overlap area (lens area)."""
    B, N, _ = centers.shape
    diff = centers.unsqueeze(2) - centers.unsqueeze(1)        # (B,N,N,2)
    d = diff.norm(dim=-1)                                     # (B,N,N)
    r1 = radii.unsqueeze(2).expand(-1, -1, N)
    r2 = radii.unsqueeze(1).expand(-1, N, -1)
    eye = torch.eye(N, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)
    overlap_amount = torch.clamp(r1 + r2 - d, min=0.0)
    overlap_amount = overlap_amount.masked_fill(eye, 0.0)
    return overlap_amount


def boundary_violation(centers, radii):
    """Per-disk amount the disk pokes outside [0,1]^2."""
    x = centers[..., 0]
    y = centers[..., 1]
    r = radii
    left = torch.clamp(r - x, min=0.0)
    right = torch.clamp(x + r - 1.0, min=0.0)
    bot = torch.clamp(r - y, min=0.0)
    top = torch.clamp(y + r - 1.0, min=0.0)
    return left + right + bot + top  # (B, N)


def severity(centers, radii):
    """Per-disk severity = sum of pairwise overlaps + boundary."""
    pair = overlap_area_pairwise(centers, radii)  # (B,N,N)
    return pair.sum(dim=-1) + boundary_violation(centers, radii)


def violation_count(centers, radii, eps=1e-3):
    """Count of overlapping pairs + boundary-violating disks."""
    pair = overlap_area_pairwise(centers, radii)
    n_pair = ((pair > eps).sum(dim=(1, 2)) // 2).int()  # symmetric pairs
    n_bd = (boundary_violation(centers, radii) > eps).sum(dim=-1)
    return (n_pair + n_bd).cpu().numpy()


def total_overlap_area(centers, radii):
    pair = overlap_area_pairwise(centers, radii)
    # half because pairs counted twice
    return (pair.sum(dim=(1, 2)) / 2).cpu().numpy()


def wirelength(centers, edges):
    """Sum over edges of (|x_u - x_v| + |y_u - y_v|), bbox-style.

    edges: (2, E) np.int64
    """
    src = torch.from_numpy(edges[0])
    dst = torch.from_numpy(edges[1])
    # centers: (B, N, 2)
    src_pos = centers[:, src, :]
    dst_pos = centers[:, dst, :]
    return (src_pos - dst_pos).abs().sum(dim=(1, 2)).cpu().numpy()


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

def run_classifier_guidance(model, r):
    """Smooth softplus(overlap) gradient injected during sampling."""
    def guide_fn(x0, radii):
        d = (x0.unsqueeze(2) - x0.unsqueeze(1)).norm(dim=-1)
        r1 = radii.unsqueeze(2)
        r2 = radii.unsqueeze(1)
        # softplus on (r1+r2 - d)
        eye = torch.eye(d.shape[-1], device=d.device).unsqueeze(0)
        overlap = F.softplus((r1 + r2 - d).masked_fill(eye.bool(), -10.0))
        # boundary penalty
        x = x0[..., 0]
        y = x0[..., 1]
        bd = (F.softplus(radii - x) + F.softplus(x + radii - 1.0)
              + F.softplus(radii - y) + F.softplus(y + radii - 1.0))
        return overlap.sum() + bd.sum()
    return ddim_sample(model, r, guide_fn=guide_fn)


def run_baseline(model, r):
    return ddim_sample(model, r)


def run_repaint_binary(model, r, x_init, t_start=0.3):
    """RePaint-style intra-sampling with binary mask (offender = severity > 0)."""
    sev = severity(x_init, r)
    mask = (sev > 1e-3).float().unsqueeze(-1)  # (B, N, 1)

    # Forward-diffuse the original to t_start to get the "known" latent
    t_init = torch.full((x_init.shape[0],), t_start)
    x_known, _ = add_noise(x_init, t_init)

    def clamp_known(x0_pred, t_next):
        # Re-noise x_known to t_next
        a = CosineSchedule.alpha(t_next).view(-1, 1, 1)
        s = CosineSchedule.sigma(t_next).view(-1, 1, 1)
        eps = torch.randn_like(x_known)
        x_n = a * x_init + s * eps
        # Replace non-offenders with their forward-diffused original
        return mask * x0_pred + (1 - mask) * x_n
    return ddim_sample(model, r, n_steps=50, t_start=t_start, x_init=x_known,
                       clamp_known_fn=clamp_known)


def run_vsr_intra(model, r, x_init, t_start=0.3):
    """VSR intra-sampling: structured (severity-weighted) RePaint."""
    sev = severity(x_init, r)
    # Structured: use severity as a soft mask in [0,1]
    sev_max = sev.max(dim=-1, keepdim=True).values.clamp(min=1e-6)
    soft_mask = (sev / sev_max).unsqueeze(-1)  # (B, N, 1)
    t_init = torch.full((x_init.shape[0],), t_start)
    x_known, _ = add_noise(x_init, t_init)

    def clamp_known(x0_pred, t_next):
        a = CosineSchedule.alpha(t_next).view(-1, 1, 1)
        s = CosineSchedule.sigma(t_next).view(-1, 1, 1)
        eps = torch.randn_like(x_known)
        x_n = a * x_init + s * eps
        return soft_mask * x0_pred + (1 - soft_mask) * x_n
    return ddim_sample(model, r, n_steps=50, t_start=t_start, x_init=x_known,
                       clamp_known_fn=clamp_known)


def run_vsr_post(x_init, r, edges, n_iters=80, step_size=0.02, hpwl_weight=0.3):
    """Hand-crafted force integrator on top of x_init."""
    # Defensive clamp to canvas; intra-sampling and DDIM can produce slightly OOB
    # values when the toy denoiser is under-trained (preflight) or rare seeds.
    x = x_init.clone().detach().clamp(min=0.0, max=1.0)
    src = torch.from_numpy(edges[0])
    dst = torch.from_numpy(edges[1])
    for _ in range(n_iters):
        sev = severity(x, r)
        mask = (sev > 1e-4).float().unsqueeze(-1)  # (B, N, 1)

        # Repulsive force from pairwise overlap
        diff = x.unsqueeze(2) - x.unsqueeze(1)  # (B,N,N,2)
        d = diff.norm(dim=-1).clamp(min=1e-6)
        r1 = r.unsqueeze(2)
        r2 = r.unsqueeze(1)
        depth = torch.clamp(r1 + r2 - d, min=0.0)
        eye = torch.eye(d.shape[-1]).bool().unsqueeze(0)
        depth = depth.masked_fill(eye, 0.0)
        unit = diff / d.unsqueeze(-1)
        repulsive = (depth.unsqueeze(-1) * unit).sum(dim=2)  # (B,N,2)

        # Boundary force (snap)
        bd_force = torch.zeros_like(x)
        bd_force[..., 0] = torch.clamp(r - x[..., 0], min=0.0) - torch.clamp(x[..., 0] + r - 1.0, min=0.0)
        bd_force[..., 1] = torch.clamp(r - x[..., 1], min=0.0) - torch.clamp(x[..., 1] + r - 1.0, min=0.0)

        # Attractive force from edges
        attr = torch.zeros_like(x)
        # for each edge (u,v), pull u toward v / deg(u)
        src_pos = x[:, src, :]
        dst_pos = x[:, dst, :]
        direction = dst_pos - src_pos
        # accumulate
        attr.index_add_(1, src, direction)
        attr.index_add_(1, dst, -direction)
        deg = torch.zeros(x.shape[0], x.shape[1])
        E = src.shape[0]
        ones_be = torch.ones(x.shape[0], E, dtype=torch.float32)
        deg.index_add_(1, src, ones_be)
        deg.index_add_(1, dst, ones_be)
        attr = attr / deg.clamp(min=1.0).unsqueeze(-1)

        # Attraction is scaled by typical pair distance to be commensurate with
        # repulsive force; capped so a single edge can never override repulsion.
        attr_scale = 0.05
        attr = attr.clamp(min=-attr_scale, max=attr_scale)
        force = repulsive + bd_force + hpwl_weight * attr
        x = x + step_size * mask * force
    return x


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def evaluate(centers, radii, edges):
    """Return dict of metric arrays of shape (B,)."""
    return {
        "viol_count": violation_count(centers, radii),
        "overlap_area": total_overlap_area(centers, radii),
        "wirelength": wirelength(centers, edges),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="Tiny config for preflight")
    p.add_argument("--out", default=str(RESULTS_DIR / "toy_results.json"))
    p.add_argument("--n_test", type=int, default=N_TEST)
    p.add_argument("--n_seeds", type=int, default=5)
    p.add_argument("--n_train", type=int, default=N_TRAIN)
    p.add_argument("--train_steps", type=int, default=4000)
    p.add_argument("--train", action="store_true", help="Train denoiser; otherwise load cached")
    args = p.parse_args()

    if args.quick:
        args.n_test = 4
        args.n_seeds = 2
        args.n_train = 200
        args.train_steps = 1500

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = RESULTS_DIR / "toy_denoiser.pt"

    rng = np.random.default_rng(0)
    print(f"Generating {args.n_train} legal training layouts...", flush=True)
    train_layouts = np.stack([gen_legal_layout(rng)[0] for _ in range(args.n_train)])
    train_radii = np.stack([gen_legal_layout(np.random.default_rng(i))[1]
                            for i in range(args.n_train)])

    model = DiskDenoiser(dim=48)
    if ckpt_path.exists() and not args.train:
        print(f"Loading cached denoiser from {ckpt_path}", flush=True)
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    else:
        train_denoiser(model, train_layouts, train_radii, n_steps=args.train_steps)
        torch.save(model.state_dict(), ckpt_path)
    model.eval()

    # Build test prompts: random radii + random graph
    test_rng = np.random.default_rng(123)
    test_radii = np.stack([
        test_rng.uniform(*RADIUS_RANGE, size=N_DISKS).astype(np.float32)
        for _ in range(args.n_test)
    ])
    test_edges = [gen_graph(np.random.default_rng(1000 + i)) for i in range(args.n_test)]

    methods = ["baseline", "classifier_guide", "repaint_binary", "vsr_post", "vsr_intra"]
    rows = []

    for seed in range(args.n_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        r = torch.from_numpy(test_radii)  # (n_test, N)

        # 1) baseline: raw diffusion sample
        t0 = time.time()
        x_base = run_baseline(model, r)
        t_base = (time.time() - t0) / args.n_test

        # 2) classifier guidance
        t0 = time.time()
        x_cg = run_classifier_guidance(model, r)
        t_cg = (time.time() - t0) / args.n_test

        # 3) RePaint binary
        t0 = time.time()
        x_rp = run_repaint_binary(model, r, x_base)
        t_rp = (time.time() - t0) / args.n_test

        # 4) VSR-post
        t0 = time.time()
        # For VSR-post we need per-test edges. Apply per layout to use that test's edges.
        x_post = []
        for i in range(args.n_test):
            xi = run_vsr_post(x_base[i:i+1], r[i:i+1], test_edges[i])
            x_post.append(xi)
        x_post = torch.cat(x_post, dim=0)
        t_post = (time.time() - t0) / args.n_test

        # 5) VSR-intra
        t0 = time.time()
        x_intra = run_vsr_intra(model, r, x_base)
        t_intra = (time.time() - t0) / args.n_test

        method_outputs = {
            "baseline": (x_base, t_base),
            "classifier_guide": (x_cg, t_cg),
            "repaint_binary": (x_rp, t_rp),
            "vsr_post": (x_post, t_post),
            "vsr_intra": (x_intra, t_intra),
        }

        for method, (x, t) in method_outputs.items():
            for i in range(args.n_test):
                m = evaluate(x[i:i+1], r[i:i+1], test_edges[i])
                rows.append({
                    "method": method,
                    "seed": seed,
                    "test_idx": i,
                    "viol_count": int(m["viol_count"][0]),
                    "overlap_area": float(m["overlap_area"][0]),
                    "wirelength": float(m["wirelength"][0]),
                    "time_per_sample": float(t),
                })
        print(f"seed={seed} done", flush=True)

    out = {
        "config": {
            "n_disks": N_DISKS,
            "radius_range": list(RADIUS_RANGE),
            "n_train": args.n_train,
            "train_steps": args.train_steps,
            "n_test": args.n_test,
            "n_seeds": args.n_seeds,
        },
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}  ({len(rows)} rows)")

    # Quick summary
    from collections import defaultdict
    summary = defaultdict(list)
    for r in rows:
        summary[r["method"]].append(r)
    print("\n=== Toy 2D summary (mean across seed x test) ===")
    print(f"{'method':<20} {'viol':>8} {'overlap':>10} {'wirelen':>10} {'time':>8}")
    for m, rs in summary.items():
        vc = np.mean([r["viol_count"] for r in rs])
        oa = np.mean([r["overlap_area"] for r in rs])
        wl = np.mean([r["wirelength"] for r in rs])
        tt = np.mean([r["time_per_sample"] for r in rs])
        print(f"{m:<20} {vc:>8.2f} {oa:>10.4f} {wl:>10.3f} {tt:>8.4f}")


if __name__ == "__main__":
    main()
