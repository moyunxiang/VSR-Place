"""Re-noising strength schedules.

Determine the noise level (alpha) to apply to each selected macro
during selective re-noising.
"""

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from vsr_place.verifier.feedback import ViolationFeedback


class StrengthSchedule(ABC):
    """Base class for re-noising strength strategies."""

    @abstractmethod
    def compute_alpha(
        self,
        feedback: ViolationFeedback,
        mask: Tensor,
        loop_iter: int = 0,
    ) -> Tensor:
        """Compute per-macro re-noising strength.

        Args:
            feedback: Structured violation feedback from the verifier.
            mask: (N,) boolean tensor of selected macros.
            loop_iter: Current repair loop iteration (0-indexed).

        Returns:
            (N,) tensor of alpha values in [0, 1].
            Alpha = 0 means no noise added, alpha = 1 means full noise.
            Non-selected macros should have alpha = 0.
        """


class FixedStrength(StrengthSchedule):
    """Apply the same re-noising strength to all selected macros.

    Args:
        alpha: Fixed noise level in (0, 1].
    """

    def __init__(self, alpha: float = 0.3):
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha

    def compute_alpha(
        self,
        feedback: ViolationFeedback,
        mask: Tensor,
        loop_iter: int = 0,
    ) -> Tensor:
        alpha = torch.zeros(feedback.num_macros, device=mask.device)
        alpha[mask] = self.alpha
        return alpha


class SeverityAdaptiveStrength(StrengthSchedule):
    """Scale re-noising strength proportionally to violation severity.

    More severely offending macros get higher noise levels.

    Args:
        alpha_min: Minimum noise level for mildly offending macros.
        alpha_max: Maximum noise level for the worst offender.
    """

    def __init__(self, alpha_min: float = 0.1, alpha_max: float = 0.5):
        if not 0.0 < alpha_min <= alpha_max <= 1.0:
            raise ValueError(
                f"Need 0 < alpha_min <= alpha_max <= 1, got ({alpha_min}, {alpha_max})"
            )
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

    def compute_alpha(
        self,
        feedback: ViolationFeedback,
        mask: Tensor,
        loop_iter: int = 0,
    ) -> Tensor:
        alpha = torch.zeros(feedback.num_macros, device=mask.device)

        if not mask.any():
            return alpha

        severity = feedback.severity_vector[mask]
        s_min = severity.min()
        s_max = severity.max()

        if s_max > s_min:
            # Linear mapping from [s_min, s_max] to [alpha_min, alpha_max]
            normalized = (severity - s_min) / (s_max - s_min)
            alpha[mask] = self.alpha_min + normalized * (self.alpha_max - self.alpha_min)
        else:
            # All selected macros have the same severity
            alpha[mask] = (self.alpha_min + self.alpha_max) / 2.0

        return alpha


class ScheduledStrength(StrengthSchedule):
    """Decrease re-noising strength over successive repair loops.

    Args:
        alpha_schedule: List of alpha values for each loop iteration.
            If loop_iter exceeds the list length, the last value is used.
    """

    def __init__(self, alpha_schedule: list[float]):
        if not alpha_schedule:
            raise ValueError("alpha_schedule must not be empty")
        for a in alpha_schedule:
            if not 0.0 < a <= 1.0:
                raise ValueError(f"All alpha values must be in (0, 1], got {a}")
        self.alpha_schedule = alpha_schedule

    def compute_alpha(
        self,
        feedback: ViolationFeedback,
        mask: Tensor,
        loop_iter: int = 0,
    ) -> Tensor:
        idx = min(loop_iter, len(self.alpha_schedule) - 1)
        current_alpha = self.alpha_schedule[idx]

        alpha = torch.zeros(feedback.num_macros, device=mask.device)
        alpha[mask] = current_alpha
        return alpha
