"""Variant A: Mask-only repair.

The simplest conditioning variant. Uses violation severity only to decide
which macros are re-noised. Does not modify the model's conditioning features.
"""

from torch import Tensor

from vsr_place.renoising.renoise import selective_renoise
from vsr_place.renoising.selector import MacroSelector
from vsr_place.renoising.strength import StrengthSchedule
from vsr_place.verifier.feedback import ViolationFeedback


class MaskOnlyRepair:
    """Variant A repair: select offending macros and re-noise them.

    This is the core VSR-Place method. It:
    1. Uses the selector to identify offending macros from verifier feedback
    2. Computes re-noising strength based on the strength schedule
    3. Applies selective re-noising to the current placement

    No model conditioning features are modified.

    Args:
        selector: Strategy for selecting which macros to re-noise.
        strength: Strategy for computing re-noising strength per macro.
    """

    def __init__(self, selector: MacroSelector, strength: StrengthSchedule):
        self.selector = selector
        self.strength = strength

    def repair(
        self,
        x_hat_0: Tensor,
        feedback: ViolationFeedback,
        loop_iter: int = 0,
    ) -> tuple[Tensor, Tensor, float]:
        """Apply mask-only repair to a placement.

        Args:
            x_hat_0: (N, 2) current clean placement estimate.
            feedback: Structured violation feedback.
            loop_iter: Current repair loop iteration.

        Returns:
            Tuple of:
                - (N, 2) re-noised placement
                - (N,) boolean mask of re-noised macros
                - float: max alpha among selected macros (for timestep mapping)
        """
        mask = self.selector.select(feedback)
        alpha = self.strength.compute_alpha(feedback, mask, loop_iter=loop_iter)

        x_renoised = selective_renoise(x_hat_0, mask, alpha)

        max_alpha = alpha[mask].max().item() if mask.any() else 0.0

        return x_renoised, mask, max_alpha
