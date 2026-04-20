"""Local position repair for VSR-Place post-processing.

Instead of re-noising + re-denoising (which wipes out guided sampling
quality), directly nudge offending macros away from overlaps and
boundary violations. Used as a post-processing step after guided
sampling.

This is similar in spirit to ChipDiffusion's legalization but much
cheaper (single forward pass, no gradient descent through model).
"""

import torch
from torch import Tensor


def local_repair_step(
    centers: Tensor,
    sizes: Tensor,
    canvas_w: float,
    canvas_h: float,
    step_size: float = 0.1,
    only_mask: Tensor | None = None,
) -> Tensor:
    """Nudge overlapping/boundary-violating macros to reduce violations.

    Uses a repulsive force model:
    - For overlapping pairs: push apart along their center-to-center direction
    - For boundary violations: pull back into canvas

    Args:
        centers: (N, 2) current macro centers.
        sizes: (N, 2) macro (width, height).
        canvas_w, canvas_h: canvas dimensions.
        step_size: fraction of violation magnitude to move per step.
        only_mask: (N,) bool, if provided only move these macros.

    Returns:
        (N, 2) updated centers.
    """
    n = centers.shape[0]
    new_centers = centers.clone()

    half = sizes / 2.0

    # --- Pairwise overlap repulsion ---
    mins = centers - half
    maxs = centers + half
    inter_min = torch.max(mins.unsqueeze(1), mins.unsqueeze(0))
    inter_max = torch.min(maxs.unsqueeze(1), maxs.unsqueeze(0))
    inter_dims = torch.clamp(inter_max - inter_min, min=0.0)  # (N,N,2)
    overlap_area = inter_dims[:, :, 0] * inter_dims[:, :, 1]
    overlap_area[range(n), range(n)] = 0.0

    has_overlap = overlap_area > 0  # (N, N)

    # Direction: center i minus center j (push i away from j)
    delta = centers.unsqueeze(0) - centers.unsqueeze(1)  # (N, N, 2), delta[i,j] = c[j]-c[i]
    # Want i to move away from j (in direction c[i] - c[j])
    push_dir = -delta  # (N, N, 2)
    norm = torch.norm(push_dir, dim=-1, keepdim=True).clamp(min=1e-6)
    push_unit = push_dir / norm

    # Magnitude proportional to overlap in smaller dimension
    overlap_mag = torch.min(inter_dims[:, :, 0], inter_dims[:, :, 1])  # (N, N)

    # Sum forces from all overlapping neighbors
    forces = (push_unit * (has_overlap * overlap_mag).unsqueeze(-1)).sum(dim=1)  # (N, 2)

    # --- Boundary repulsion ---
    x_min = centers[:, 0] - half[:, 0]
    x_max = centers[:, 0] + half[:, 0]
    y_min = centers[:, 1] - half[:, 1]
    y_max = centers[:, 1] + half[:, 1]

    boundary_force = torch.zeros_like(centers)
    boundary_force[:, 0] += torch.clamp(-x_min, min=0.0)   # push right
    boundary_force[:, 0] -= torch.clamp(x_max - canvas_w, min=0.0)  # push left
    boundary_force[:, 1] += torch.clamp(-y_min, min=0.0)   # push up
    boundary_force[:, 1] -= torch.clamp(y_max - canvas_h, min=0.0)  # push down

    total_force = forces + boundary_force

    if only_mask is not None:
        only_mask_expanded = only_mask.unsqueeze(-1).to(total_force.device)
        total_force = total_force * only_mask_expanded

    new_centers = centers + step_size * total_force
    return new_centers


def local_repair_loop(
    centers: Tensor,
    sizes: Tensor,
    canvas_w: float,
    canvas_h: float,
    num_steps: int = 50,
    step_size: float = 0.2,
    only_mask: Tensor | None = None,
) -> Tensor:
    """Iteratively apply local repair until convergence or budget exhausted.

    Returns:
        (N, 2) repaired centers.
    """
    x = centers.clone()
    for _ in range(num_steps):
        x = local_repair_step(x, sizes, canvas_w, canvas_h, step_size, only_mask)
    return x
