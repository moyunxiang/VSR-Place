"""Half-Perimeter Wirelength (HPWL) computation.

HPWL is the standard wirelength proxy for placement quality.
For a net connecting a set of pins, HPWL = (max_x - min_x) + (max_y - min_y).
"""

import torch
from torch import Tensor


def compute_hpwl(
    centers: Tensor,
    net_to_macros: list[list[int]],
    pin_offsets: Tensor | None = None,
) -> float:
    """Compute total HPWL for a placement.

    Args:
        centers: (N, 2) tensor of macro center coordinates.
        net_to_macros: List of nets, each a list of macro indices connected by the net.
        pin_offsets: Optional (N, 2) tensor of pin offsets from macro centers.
            If None, pins are assumed at macro centers.

    Returns:
        Total HPWL (scalar).
    """
    if pin_offsets is not None:
        pin_positions = centers + pin_offsets
    else:
        pin_positions = centers

    total_hpwl = 0.0
    for macro_indices in net_to_macros:
        if len(macro_indices) < 2:
            continue
        pins = pin_positions[macro_indices]  # (K, 2)
        bbox_max = pins.max(dim=0).values  # (2,)
        bbox_min = pins.min(dim=0).values  # (2,)
        net_hpwl = (bbox_max - bbox_min).sum()
        total_hpwl += net_hpwl.item()

    return total_hpwl


def compute_hpwl_vectorized(
    centers: Tensor,
    net_mask: Tensor,
) -> float:
    """Compute total HPWL using a vectorized net mask.

    Args:
        centers: (N, 2) tensor of macro center coordinates.
        net_mask: (M, N) boolean tensor where net_mask[m, i] is True if
            macro i belongs to net m.

    Returns:
        Total HPWL (scalar).
    """
    # Expand centers for broadcasting: (1, N, 2) * (M, N, 1)
    mask_expanded = net_mask.unsqueeze(-1)  # (M, N, 1)
    centers_expanded = centers.unsqueeze(0)  # (1, N, 2)

    # Set non-member positions to +inf / -inf for min/max
    big_val = 1e12
    masked_for_max = torch.where(mask_expanded, centers_expanded, torch.tensor(-big_val))
    masked_for_min = torch.where(mask_expanded, centers_expanded, torch.tensor(big_val))

    net_max = masked_for_max.max(dim=1).values  # (M, 2)
    net_min = masked_for_min.min(dim=1).values  # (M, 2)

    # Filter out nets with < 2 members
    net_sizes = net_mask.sum(dim=1)  # (M,)
    valid = net_sizes >= 2

    hpwl_per_net = (net_max - net_min).sum(dim=1)  # (M,)
    total_hpwl = hpwl_per_net[valid].sum().item()

    return total_hpwl
