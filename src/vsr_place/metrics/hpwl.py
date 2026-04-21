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


def compute_hpwl_from_edges(
    centers: Tensor,
    edge_index: Tensor,
    edge_attr: Tensor | None = None,
) -> float:
    """Compute HPWL using edge list (pairwise 2-pin net approximation).

    For each edge (u, v), wirelength = |x_u + off_u_x - (x_v + off_v_x)|
                                     + |y_u + off_u_y - (y_v + off_v_y)|
    Works directly on PyG-style bidirectional edge_index (only uses each undirected
    edge once by filtering u < v).

    Args:
        centers: (N, 2) macro centers.
        edge_index: (2, E) edges.
        edge_attr: (E, 4) edge features where columns [0,1] are src pin offset,
            [2,3] are dst pin offset.

    Returns:
        Total HPWL (sum over unique edges).
    """
    src, dst = edge_index[0], edge_index[1]
    unique_mask = src < dst
    src_u = src[unique_mask]
    dst_u = dst[unique_mask]

    src_pos = centers[src_u]
    dst_pos = centers[dst_u]
    if edge_attr is not None:
        attr_u = edge_attr[unique_mask]
        src_pos = src_pos + attr_u[:, :2]
        dst_pos = dst_pos + attr_u[:, 2:4]

    delta = (src_pos - dst_pos).abs()
    return delta.sum().item()


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
