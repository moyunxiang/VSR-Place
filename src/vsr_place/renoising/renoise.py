"""Selective re-noising core implementation.

Applies noise only to selected macros while preserving others.
Core formula for selected macro i:
    x_i' = sqrt(1 - alpha_i) * x_hat_0_i + sqrt(alpha_i) * epsilon_i
For non-selected macros:
    x_i' = x_hat_0_i  (unchanged)
"""

import torch
from torch import Tensor


def selective_renoise(
    x_hat_0: Tensor,
    mask: Tensor,
    alpha: Tensor,
    noise: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Apply selective re-noising to a placement.

    Args:
        x_hat_0: (N, 2) tensor of current clean placement estimates (macro centers).
        mask: (N,) boolean tensor indicating which macros to re-noise.
        alpha: (N,) tensor of re-noising strengths in [0, 1].
        noise: Optional (N, 2) tensor of noise. If None, sampled from N(0, I).
        generator: Optional torch Generator for reproducible noise.

    Returns:
        (N, 2) tensor of selectively re-noised placement.
    """
    if noise is None:
        noise = torch.randn(x_hat_0.shape, device=x_hat_0.device,
                            dtype=x_hat_0.dtype, generator=generator)

    # Expand alpha to (N, 1) for broadcasting with (N, 2)
    alpha_expanded = alpha.unsqueeze(-1)  # (N, 1)

    # Compute re-noised positions for all macros
    signal_coeff = torch.sqrt(1.0 - alpha_expanded)
    noise_coeff = torch.sqrt(alpha_expanded)
    x_noised = signal_coeff * x_hat_0 + noise_coeff * noise

    # Apply mask: only selected macros are re-noised
    mask_expanded = mask.unsqueeze(-1)  # (N, 1)
    x_out = torch.where(mask_expanded, x_noised, x_hat_0)

    return x_out


def selective_renoise_batch(
    x_hat_0: Tensor,
    mask: Tensor,
    alpha: Tensor,
    noise: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Apply selective re-noising to a batch of placements.

    Args:
        x_hat_0: (B, N, 2) tensor of current clean placement estimates.
        mask: (B, N) boolean tensor indicating which macros to re-noise.
        alpha: (B, N) tensor of re-noising strengths in [0, 1].
        noise: Optional (B, N, 2) tensor of noise. If None, sampled from N(0, I).
        generator: Optional torch Generator for reproducible noise.

    Returns:
        (B, N, 2) tensor of selectively re-noised placements.
    """
    if noise is None:
        noise = torch.randn(x_hat_0.shape, device=x_hat_0.device,
                            dtype=x_hat_0.dtype, generator=generator)

    alpha_expanded = alpha.unsqueeze(-1)  # (B, N, 1)
    signal_coeff = torch.sqrt(1.0 - alpha_expanded)
    noise_coeff = torch.sqrt(alpha_expanded)
    x_noised = signal_coeff * x_hat_0 + noise_coeff * noise

    mask_expanded = mask.unsqueeze(-1)  # (B, N, 1)
    x_out = torch.where(mask_expanded, x_noised, x_hat_0)

    return x_out
