"""Synthetic dataset for training NeuralVSR.

Training signal construction:
1. Generate random macro sizes + a random netlist.
2. Generate a LEGAL placement (grid-based, no overlap).
3. Perturb to create violations: x_bad = x_legal + noise.
4. Run verifier on x_bad → violation features.
5. Train target: delta = x_legal - x_bad (the displacement that "undoes" the perturbation).

This gives the GNN the signal: given bad placement + violations, predict
the displacement that moves toward legal.

Advantages:
- No need for an oracle legalizer (we know x_legal by construction)
- Distribution of violations is controlled by noise magnitude
- Training is fast (all operations are cheap)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from vsr_place.verifier.verifier import Verifier


@dataclass
class SyntheticSample:
    """One training sample: (bad placement, graph, target displacement)."""
    centers_bad: torch.Tensor     # (N, 2)
    target_delta: torch.Tensor    # (N, 2)  = centers_legal - centers_bad
    sizes: torch.Tensor           # (N, 2)
    edge_index: torch.Tensor      # (2, E)
    edge_attr: torch.Tensor       # (E, 4)
    canvas_w: float
    canvas_h: float


def _generate_macro_sizes(n: int, canvas_side: float, size_scale: float = 0.08,
                           aspect_range: tuple = (0.5, 2.0), rng: torch.Generator = None) -> torch.Tensor:
    """Sample macro sizes from a clipped log-normal distribution.

    Matches ChipDiffusion's v1 data generator statistics.
    """
    # Areas: clipped log-normal (mean in log space = log(size_scale))
    areas = torch.distributions.LogNormal(
        loc=math.log(size_scale * canvas_side ** 2 / n),
        scale=0.5,
    ).sample((n,))
    areas = torch.clamp(areas, min=(0.02 * canvas_side) ** 2, max=(0.3 * canvas_side) ** 2)

    # Aspect ratios
    aspect_low = math.log(aspect_range[0])
    aspect_high = math.log(aspect_range[1])
    aspects_log = aspect_low + (aspect_high - aspect_low) * torch.rand(n)
    aspects = torch.exp(aspects_log)

    widths = torch.sqrt(areas * aspects)
    heights = torch.sqrt(areas / aspects)
    return torch.stack([widths, heights], dim=-1)  # (N, 2)


def _generate_legal_placement(sizes: torch.Tensor, canvas_w: float, canvas_h: float,
                               max_attempts: int = 200) -> torch.Tensor:
    """Generate a legal placement using greedy grid packing."""
    n = sizes.shape[0]
    centers = torch.zeros(n, 2)

    # Try grid-based assignment
    grid_rows = max(1, int(math.sqrt(n)))
    grid_cols = max(1, (n + grid_rows - 1) // grid_rows)

    cell_w = canvas_w / grid_cols
    cell_h = canvas_h / grid_rows

    # Sort by decreasing size, assign to cells
    size_order = torch.argsort(-(sizes[:, 0] * sizes[:, 1]))
    for rank, idx in enumerate(size_order):
        row = rank // grid_cols
        col = rank % grid_cols
        if row >= grid_rows:
            row = grid_rows - 1
        # Random position within cell (clamped to fit)
        half_w = sizes[idx, 0].item() / 2
        half_h = sizes[idx, 1].item() / 2
        x_min = col * cell_w + half_w
        x_max = min((col + 1) * cell_w - half_w, canvas_w - half_w)
        y_min = row * cell_h + half_h
        y_max = min((row + 1) * cell_h - half_h, canvas_h - half_h)

        if x_max < x_min:
            x = (x_min + x_max) / 2
        else:
            x = x_min + (x_max - x_min) * torch.rand(1).item()
        if y_max < y_min:
            y = (y_min + y_max) / 2
        else:
            y = y_min + (y_max - y_min) * torch.rand(1).item()

        centers[idx, 0] = x
        centers[idx, 1] = y

    return centers


def _generate_random_netlist(n: int, centers: torch.Tensor, edge_density: float = 0.05,
                              max_degree: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a random netlist with distance-based edge probability.

    Returns:
        edge_index: (2, E) bidirectional edges.
        edge_attr: (E, 4) pin offsets (random within macros).
    """
    # Pairwise distances
    dists = torch.cdist(centers, centers)
    # Avoid self-loops
    dists.fill_diagonal_(float("inf"))

    # Probability decreases with distance
    mean_dist = dists[dists < float("inf")].mean().clamp(min=1e-3)
    probs = edge_density * torch.exp(-dists / mean_dist)

    # Sample edges (upper triangular)
    mask = torch.rand_like(probs) < probs
    mask = torch.triu(mask, diagonal=1)

    # Limit degree
    src, dst = mask.nonzero(as_tuple=True)
    if len(src) > max_degree * n:
        keep = torch.randperm(len(src))[:max_degree * n]
        src, dst = src[keep], dst[keep]

    # Random pin offsets (will be rescaled later)
    e_uni = len(src)
    edge_attr_uni = (torch.rand(e_uni, 4) - 0.5) * 0.01  # small offsets

    # Make bidirectional
    edge_index = torch.stack([
        torch.cat([src, dst]),
        torch.cat([dst, src]),
    ], dim=0)
    edge_attr = torch.cat([
        edge_attr_uni,
        edge_attr_uni[:, [2, 3, 0, 1]],  # swap src/dst offsets for reverse edges
    ], dim=0)

    return edge_index, edge_attr


def generate_synthetic_sample(
    n_range: tuple = (30, 300),
    canvas_side: float = 10.0,
    perturb_scale: float = 0.5,
    edge_density: float = 0.05,
    seed: int | None = None,
    target_mode: str = "teacher",
) -> SyntheticSample:
    """Generate one synthetic training sample.

    Args:
        n_range: (min, max) number of macros.
        canvas_side: canvas size.
        perturb_scale: perturbation magnitude (fraction of canvas).
        edge_density: netlist edge probability multiplier.
        seed: RNG seed.
        target_mode:
            'legal' - target = displacement back to original legal placement
                      (unstable because multiple legal placements exist)
            'teacher' - target = displacement predicted by running hand-crafted
                        local_repair to convergence. Distill repair policy.

    Returns:
        SyntheticSample with (x_bad, target_delta, sizes, edge_index, edge_attr).
    """
    if seed is not None:
        torch.manual_seed(seed)

    n = int(torch.randint(n_range[0], n_range[1] + 1, (1,)).item())

    # Sizes
    sizes = _generate_macro_sizes(n, canvas_side)
    max_w = sizes[:, 0].max().item()
    max_h = sizes[:, 1].max().item()
    if max_w > canvas_side / 2 or max_h > canvas_side / 2:
        scale = min(canvas_side / (2 * max_w), canvas_side / (2 * max_h))
        sizes = sizes * scale

    centers_legal = _generate_legal_placement(sizes, canvas_side, canvas_side)
    edge_index, edge_attr = _generate_random_netlist(n, centers_legal, edge_density)

    # Perturb to create violations
    perturb = torch.randn_like(centers_legal) * perturb_scale
    centers_bad = centers_legal + perturb
    centers_bad[:, 0] = centers_bad[:, 0].clamp(0, canvas_side)
    centers_bad[:, 1] = centers_bad[:, 1].clamp(0, canvas_side)

    if target_mode == "legal":
        target_delta = centers_legal - centers_bad
    elif target_mode == "teacher":
        # Target = final displacement after running hand-crafted repair to convergence.
        # NeuralVSR learns to predict this full trajectory in a single forward pass.
        from vsr_place.renoising.local_repair import local_repair_loop
        centers_repaired = local_repair_loop(
            centers_bad, sizes, canvas_side, canvas_side,
            num_steps=100, step_size=0.3,
        )
        target_delta = centers_repaired - centers_bad
    else:
        raise ValueError(f"Unknown target_mode: {target_mode}")

    return SyntheticSample(
        centers_bad=centers_bad,
        target_delta=target_delta,
        sizes=sizes,
        edge_index=edge_index,
        edge_attr=edge_attr,
        canvas_w=canvas_side,
        canvas_h=canvas_side,
    )


def compute_violation_features(
    centers: torch.Tensor,
    sizes: torch.Tensor,
    canvas_w: float,
    canvas_h: float,
) -> torch.Tensor:
    """Compute per-macro violation features for GNN input.

    Returns:
        (N, 5) tensor: [severity, boundary, overlap_count, width, height]
    """
    verifier = Verifier(canvas_width=canvas_w, canvas_height=canvas_h)
    fb = verifier(centers, sizes)

    severity = fb.severity_vector  # (N,)
    boundary = fb.boundary_violations  # (N,)
    # Count of macros each one overlaps with
    overlap_count = (fb.pairwise_overlap > 0).sum(dim=1).float()  # (N,)

    return torch.stack([
        severity,
        boundary,
        overlap_count,
        sizes[:, 0],
        sizes[:, 1],
    ], dim=-1)


class SyntheticVSRDataset(Dataset):
    """Dataset of synthetic (placement, violations, target_displacement) triples.

    Args:
        precompute: If True, generate all samples up front (faster training
                    but higher memory). If False, regenerate on __getitem__
                    with deterministic seed.
    """

    def __init__(
        self,
        num_samples: int = 10000,
        n_range: tuple = (30, 300),
        canvas_side: float = 10.0,
        perturb_scale: float = 0.5,
        edge_density: float = 0.05,
        seed: int = 0,
        precompute: bool = True,
        verbose: bool = False,
    ):
        self.num_samples = num_samples
        self.n_range = n_range
        self.canvas_side = canvas_side
        self.perturb_scale = perturb_scale
        self.edge_density = edge_density
        self.base_seed = seed
        self.precompute = precompute

        if precompute:
            if verbose:
                print(f"Precomputing {num_samples} synthetic samples...", flush=True)
            self._cache = []
            for i in range(num_samples):
                if verbose and i % 500 == 0 and i > 0:
                    print(f"  {i}/{num_samples}", flush=True)
                self._cache.append(self._make_item(i))
            if verbose:
                print(f"  done", flush=True)
        else:
            self._cache = None

    def __len__(self):
        return self.num_samples

    def _make_item(self, idx: int) -> dict:
        sample = generate_synthetic_sample(
            n_range=self.n_range,
            canvas_side=self.canvas_side,
            perturb_scale=self.perturb_scale,
            edge_density=self.edge_density,
            seed=self.base_seed + idx,
        )
        node_features = compute_violation_features(
            sample.centers_bad, sample.sizes,
            sample.canvas_w, sample.canvas_h,
        )
        return {
            "centers": sample.centers_bad,
            "node_features": node_features,
            "edge_index": sample.edge_index,
            "edge_attr": sample.edge_attr,
            "target_delta": sample.target_delta,
            "sizes": sample.sizes,
            "canvas_w": sample.canvas_w,
            "canvas_h": sample.canvas_h,
        }

    def __getitem__(self, idx: int) -> dict:
        if self._cache is not None:
            return self._cache[idx]
        return self._make_item(idx)
