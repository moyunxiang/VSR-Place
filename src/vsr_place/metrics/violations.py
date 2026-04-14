"""Violation tracking metrics for experiment analysis."""

from vsr_place.verifier.feedback import ViolationFeedback


def violation_reduction_rate(
    initial: ViolationFeedback, final: ViolationFeedback
) -> float:
    """Compute the fraction of violations eliminated.

    Args:
        initial: ViolationFeedback before repair.
        final: ViolationFeedback after repair.

    Returns:
        Reduction rate in [0, 1]. 1.0 means all violations eliminated.
    """
    init_count = initial.global_stats["total_violations"]
    if init_count == 0:
        return 1.0
    final_count = final.global_stats["total_violations"]
    return (init_count - final_count) / init_count


def violation_trajectory(feedbacks: list[ViolationFeedback]) -> list[int]:
    """Extract violation count trajectory across repair iterations.

    Args:
        feedbacks: List of ViolationFeedback from successive iterations.

    Returns:
        List of total violation counts.
    """
    return [fb.global_stats["total_violations"] for fb in feedbacks]
