"""Inference utilities for NeuralVSR.

`neural_repair_loop` is a drop-in replacement for `local_repair_loop`
(hand-crafted) that uses a trained NeuralVSR model instead.
"""

from __future__ import annotations

import torch

from vsr_place.neural.dataset import compute_violation_features
from vsr_place.neural.model import NeuralVSR


@torch.no_grad()
def neural_residual_repair(
    centers: torch.Tensor,
    sizes: torch.Tensor,
    canvas_w: float,
    canvas_h: float,
    model: NeuralVSR,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor | None = None,
    hpwl_weight: float = 2.0,
    hand_steps: int = 100,
    hand_step_size: float = 0.3,
    only_mask: torch.Tensor | None = None,
    device: str = "cpu",
) -> torch.Tensor:
    """Residual repair: hand-crafted first, then neural correction.

    This uses the trained model to refine the output of hand-crafted repair.
    The model was trained to predict (x_gt_legal - x_hand_crafted), so its
    output at inference is directly added to the hand-crafted result.
    """
    from vsr_place.renoising.local_repair import local_repair_loop
    # Stage 1: hand-crafted repair
    c_hand = local_repair_loop(
        centers, sizes, canvas_w, canvas_h,
        num_steps=hand_steps, step_size=hand_step_size,
        only_mask=only_mask, edge_index=edge_index, hpwl_weight=hpwl_weight,
    )

    # Stage 2: neural residual
    model = model.to(device).eval()
    x = c_hand.to(device)
    sizes_dev = sizes.to(device)
    edge_index_dev = edge_index.to(device)
    edge_attr_dev = edge_attr.to(device) if edge_attr is not None else None
    canvas_scale = max(canvas_w, canvas_h)

    feats = compute_violation_features(x.cpu(), sizes_dev.cpu(), canvas_w, canvas_h).to(device)
    delta = model(x, feats, edge_index_dev, edge_attr_dev, canvas_scale=canvas_scale)
    x = x + delta

    # Clamp to canvas bounds
    half_w = sizes_dev[:, 0] / 2
    half_h = sizes_dev[:, 1] / 2
    x[:, 0] = x[:, 0].clamp(half_w, canvas_w - half_w)
    x[:, 1] = x[:, 1].clamp(half_h, canvas_h - half_h)
    return x.cpu()


@torch.no_grad()
def neural_repair_loop(
    centers: torch.Tensor,
    sizes: torch.Tensor,
    canvas_w: float,
    canvas_h: float,
    model: NeuralVSR,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor | None = None,
    num_steps: int = 10,
    step_size: float = 1.0,
    only_mask: torch.Tensor | None = None,
    device: str = "cpu",
) -> torch.Tensor:
    """Iteratively apply the trained NeuralVSR to repair violations.

    Args:
        centers: (N, 2) current macro centers.
        sizes: (N, 2) macro (width, height).
        canvas_w, canvas_h: canvas dimensions.
        model: trained NeuralVSR.
        edge_index: (2, E) netlist connectivity.
        edge_attr: (E, F_e) edge features.
        num_steps: number of repair iterations (typically 10-20).
        step_size: scale factor for displacement (1.0 = apply as predicted).
        only_mask: (N,) bool, if provided only update these macros.
        device: torch device.

    Returns:
        (N, 2) repaired centers.
    """
    model = model.to(device).eval()
    x = centers.to(device).clone()
    sizes_dev = sizes.to(device)
    edge_index_dev = edge_index.to(device)
    edge_attr_dev = edge_attr.to(device) if edge_attr is not None else None

    if only_mask is not None:
        only_mask = only_mask.to(device)

    # Use max canvas dimension as scale (assumes roughly square-ish)
    canvas_scale = max(canvas_w, canvas_h)

    for _ in range(num_steps):
        feats = compute_violation_features(
            x.cpu(), sizes_dev.cpu(), canvas_w, canvas_h,
        ).to(device)

        delta = model(x, feats, edge_index_dev, edge_attr_dev, canvas_scale=canvas_scale)

        if only_mask is not None:
            delta = delta * only_mask.unsqueeze(-1)

        x = x + step_size * delta

        # Clamp to canvas bounds (with sizes accounted)
        half_w = sizes_dev[:, 0] / 2
        half_h = sizes_dev[:, 1] / 2
        x[:, 0] = x[:, 0].clamp(half_w, canvas_w - half_w)
        x[:, 1] = x[:, 1].clamp(half_h, canvas_h - half_h)

    return x.cpu()
