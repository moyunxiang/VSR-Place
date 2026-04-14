"""Structured violation feedback dataclass.

Encapsulates the multi-level output of the verifier:
- Pairwise violation matrix (N x N)
- Object-wise severity vector (N,)
- Global summary statistics
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor


@dataclass
class ViolationFeedback:
    """Structured feedback from the legality verifier.

    Attributes:
        pairwise_overlap: (N, N) overlap area between macro pairs.
        pairwise_spacing: (N, N) spacing shortfall between macro pairs. None if not checked.
        boundary_violations: (N,) boundary violation magnitude per macro.
        severity_vector: (N,) aggregated severity per macro.
        global_stats: Summary statistics dict.
    """

    pairwise_overlap: Tensor
    boundary_violations: Tensor
    severity_vector: Tensor
    global_stats: dict = field(default_factory=dict)
    pairwise_spacing: Optional[Tensor] = None

    @property
    def num_macros(self) -> int:
        return self.severity_vector.shape[0]

    def is_legal(self) -> bool:
        """Check if the placement passes all constraints."""
        return self.global_stats.get("is_legal", False)

    def offending_macros(self, threshold: float = 0.0) -> Tensor:
        """Return indices of macros with severity above threshold.

        Args:
            threshold: Minimum severity to be considered offending.

        Returns:
            1D tensor of macro indices.
        """
        return torch.where(self.severity_vector > threshold)[0]

    def top_k_offending(self, k: int) -> Tensor:
        """Return indices of the k most severely offending macros.

        Args:
            k: Number of top offenders to return.

        Returns:
            1D tensor of macro indices, sorted by decreasing severity.
        """
        k = min(k, self.num_macros)
        _, indices = torch.topk(self.severity_vector, k)
        return indices

    def offending_mask(self, threshold: float = 0.0) -> Tensor:
        """Return a boolean mask of offending macros.

        Args:
            threshold: Minimum severity to be considered offending.

        Returns:
            (N,) boolean tensor.
        """
        return self.severity_vector > threshold

    def top_k_mask(self, k: int) -> Tensor:
        """Return a boolean mask selecting the top-k offending macros.

        Args:
            k: Number of top offenders.

        Returns:
            (N,) boolean tensor.
        """
        mask = torch.zeros(self.num_macros, dtype=torch.bool,
                           device=self.severity_vector.device)
        indices = self.top_k_offending(k)
        mask[indices] = True
        return mask
