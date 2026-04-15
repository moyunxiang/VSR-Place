"""Vectorized geometry primitives for rectangle overlap and boundary checking.

All operations use PyTorch tensors for GPU acceleration.
Rectangles are represented as (center_x, center_y, width, height).
"""

import torch
from torch import Tensor


def compute_overlap_area_pairwise(
    centers: Tensor, sizes: Tensor
) -> Tensor:
    """Compute pairwise overlap areas between all rectangle pairs.

    Args:
        centers: (N, 2) tensor of rectangle center coordinates.
        sizes: (N, 2) tensor of rectangle (width, height).

    Returns:
        (N, N) tensor where entry (i, j) is the overlap area between rectangles i and j.
        Diagonal entries are zero (no self-overlap).
    """
    # Compute half-sizes
    half = sizes / 2.0  # (N, 2)

    # Compute min/max corners: (N, 2)
    mins = centers - half  # (x_min, y_min)
    maxs = centers + half  # (x_max, y_max)

    # Pairwise intersection: broadcast (N, 1, 2) vs (1, N, 2)
    inter_min = torch.max(mins.unsqueeze(1), mins.unsqueeze(0))  # (N, N, 2)
    inter_max = torch.min(maxs.unsqueeze(1), maxs.unsqueeze(0))  # (N, N, 2)

    # Intersection dimensions, clamped to >= 0
    inter_dims = torch.clamp(inter_max - inter_min, min=0.0)  # (N, N, 2)
    overlap = inter_dims[:, :, 0] * inter_dims[:, :, 1]  # (N, N)

    # Zero out diagonal (self-overlap is not a violation)
    # Zero out diagonal (self-overlap is not a violation)
    # Use index assignment instead of fill_diagonal_ for compatibility
    n = overlap.shape[0]
    overlap[range(n), range(n)] = 0.0

    return overlap


def compute_boundary_violation(
    centers: Tensor,
    sizes: Tensor,
    canvas_width: float,
    canvas_height: float,
) -> Tensor:
    """Compute per-macro boundary violation area (area outside canvas).

    Args:
        centers: (N, 2) tensor of rectangle center coordinates.
        sizes: (N, 2) tensor of rectangle (width, height).
        canvas_width: Width of the chip canvas.
        canvas_height: Height of the chip canvas.

    Returns:
        (N,) tensor of boundary violation magnitude per macro.
        Value is the total protrusion length across all edges (not area, for simplicity
        and gradient-free use). Zero means fully inside.
    """
    half = sizes / 2.0  # (N, 2)

    mins = centers - half  # (N, 2): (x_min, y_min)
    maxs = centers + half  # (N, 2): (x_max, y_max)

    # Protrusion on each side, clamped to >= 0
    left = torch.clamp(-mins[:, 0], min=0.0)
    bottom = torch.clamp(-mins[:, 1], min=0.0)
    right = torch.clamp(maxs[:, 0] - canvas_width, min=0.0)
    top = torch.clamp(maxs[:, 1] - canvas_height, min=0.0)

    return left + bottom + right + top


def compute_spacing_violation_pairwise(
    centers: Tensor,
    sizes: Tensor,
    min_spacing: float,
) -> Tensor:
    """Compute pairwise minimum-spacing violations.

    Args:
        centers: (N, 2) tensor of rectangle center coordinates.
        sizes: (N, 2) tensor of rectangle (width, height).
        min_spacing: Required minimum gap between rectangles.

    Returns:
        (N, N) tensor where entry (i, j) is the spacing shortfall between
        rectangles i and j. Positive means they are too close. Zero means OK.
        Diagonal entries are zero.
    """
    half = sizes / 2.0

    mins = centers - half
    maxs = centers + half

    # Signed gap in each dimension: positive = separated, negative = overlapping
    # gap_x(i,j) = max(mins_j_x - maxs_i_x, mins_i_x - maxs_j_x) but simplified:
    gap_x = torch.max(mins.unsqueeze(1)[:, :, 0] - maxs.unsqueeze(0)[:, :, 0],
                       mins.unsqueeze(0)[:, :, 0] - maxs.unsqueeze(1)[:, :, 0])
    gap_y = torch.max(mins.unsqueeze(1)[:, :, 1] - maxs.unsqueeze(0)[:, :, 1],
                       mins.unsqueeze(0)[:, :, 1] - maxs.unsqueeze(1)[:, :, 1])

    # The actual gap is the max of (gap_x, gap_y) for axis-aligned rects,
    # but only if at least one is positive (they don't overlap in that axis).
    # For overlapping rects, gap is negative. For separated rects, gap = max(gap_x, gap_y).
    # Actually for AABB: gap = max(gap_x, gap_y). If both negative -> overlapping.
    # Spacing violation = max(0, min_spacing - actual_gap)
    actual_gap = torch.max(gap_x, gap_y)  # (N, N)
    violation = torch.clamp(min_spacing - actual_gap, min=0.0)

    n = violation.shape[0]
    violation[range(n), range(n)] = 0.0
    return violation
