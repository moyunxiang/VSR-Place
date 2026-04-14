"""Macro selection strategies for selective re-noising.

Given a ViolationFeedback, determine which macros should be re-noised.
All selectors return a boolean mask of shape (N,).
"""

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from vsr_place.verifier.feedback import ViolationFeedback


class MacroSelector(ABC):
    """Base class for macro selection strategies."""

    @abstractmethod
    def select(self, feedback: ViolationFeedback) -> Tensor:
        """Select macros to re-noise based on violation feedback.

        Args:
            feedback: Structured violation feedback from the verifier.

        Returns:
            (N,) boolean tensor indicating which macros to re-noise.
        """


class GlobalSelector(MacroSelector):
    """Select all macros (baseline: global re-noising)."""

    def select(self, feedback: ViolationFeedback) -> Tensor:
        return torch.ones(feedback.num_macros, dtype=torch.bool,
                          device=feedback.severity_vector.device)


class ThresholdSelector(MacroSelector):
    """Select macros with severity above a fixed threshold.

    Args:
        threshold: Minimum severity to be selected for re-noising.
    """

    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold

    def select(self, feedback: ViolationFeedback) -> Tensor:
        return feedback.offending_mask(self.threshold)


class TopKSelector(MacroSelector):
    """Select the top-k most severely offending macros.

    Args:
        k: Number of macros to select.
    """

    def __init__(self, k: int):
        self.k = k

    def select(self, feedback: ViolationFeedback) -> Tensor:
        return feedback.top_k_mask(self.k)


class AdaptiveThresholdSelector(MacroSelector):
    """Select macros with severity above a percentile-based threshold.

    The threshold is computed as a percentile of the severity distribution,
    considering only macros with non-zero severity.

    Args:
        percentile: Percentile (0-100) of severity distribution to use as threshold.
    """

    def __init__(self, percentile: float = 50.0):
        self.percentile = percentile

    def select(self, feedback: ViolationFeedback) -> Tensor:
        severity = feedback.severity_vector
        nonzero_mask = severity > 0

        if not nonzero_mask.any():
            return torch.zeros(feedback.num_macros, dtype=torch.bool,
                               device=severity.device)

        nonzero_values = severity[nonzero_mask]
        # Compute percentile threshold among non-zero severities
        k = max(1, int((1.0 - self.percentile / 100.0) * nonzero_values.numel()))
        if k >= nonzero_values.numel():
            threshold = nonzero_values.min()
        else:
            threshold = torch.topk(nonzero_values, k).values[-1]

        return severity >= threshold
