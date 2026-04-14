"""Legality metrics computation."""

from vsr_place.verifier.feedback import ViolationFeedback


def compute_legality_metrics(feedback: ViolationFeedback) -> dict:
    """Extract legality-related metrics from verification feedback.

    Args:
        feedback: ViolationFeedback from the verifier.

    Returns:
        Dict with legality metrics.
    """
    return {
        "is_legal": feedback.is_legal(),
        "total_violations": feedback.global_stats["total_violations"],
        "num_overlap_violations": feedback.global_stats["num_overlap_violations"],
        "total_overlap_area": feedback.global_stats["total_overlap_area"],
        "num_boundary_violations": feedback.global_stats["num_boundary_violations"],
        "total_boundary_protrusion": feedback.global_stats["total_boundary_protrusion"],
        "num_spacing_violations": feedback.global_stats.get("num_spacing_violations", 0),
        "num_offending_macros": feedback.global_stats["num_offending_macros"],
    }


def compute_pass_rate(feedbacks: list[ViolationFeedback]) -> float:
    """Compute the fraction of legal placements in a batch.

    Args:
        feedbacks: List of ViolationFeedback objects.

    Returns:
        Pass rate in [0, 1].
    """
    if not feedbacks:
        return 0.0
    return sum(1 for fb in feedbacks if fb.is_legal()) / len(feedbacks)
