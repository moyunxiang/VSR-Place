"""Repair budget and termination criteria for the VSR loop."""

from dataclasses import dataclass

from vsr_place.verifier.feedback import ViolationFeedback


@dataclass
class RepairBudget:
    """Controls when the VSR repair loop should terminate.

    Args:
        max_loops: Maximum number of repair iterations.
        early_stop_on_legal: Stop immediately when placement becomes legal.
        min_improvement: Minimum violation reduction (fraction) to continue.
            If improvement falls below this, stop early.
    """

    max_loops: int = 4
    early_stop_on_legal: bool = True
    min_improvement: float = 0.0

    def is_exhausted(
        self,
        loop_count: int,
        feedback_history: list[ViolationFeedback],
    ) -> bool:
        """Check if the repair budget is exhausted.

        Args:
            loop_count: Number of repair iterations completed.
            feedback_history: List of ViolationFeedback from each iteration.

        Returns:
            True if the loop should terminate.
        """
        # Hard budget limit
        if loop_count >= self.max_loops:
            return True

        if not feedback_history:
            return False

        # Early stop on legal placement
        if self.early_stop_on_legal and feedback_history[-1].is_legal():
            return True

        # Check improvement rate
        if self.min_improvement > 0 and len(feedback_history) >= 2:
            prev_violations = feedback_history[-2].global_stats["total_violations"]
            curr_violations = feedback_history[-1].global_stats["total_violations"]

            if prev_violations == 0:
                return True

            improvement = (prev_violations - curr_violations) / prev_violations
            if improvement < self.min_improvement:
                return True

        return False
