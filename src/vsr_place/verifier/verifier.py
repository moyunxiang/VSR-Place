"""Non-differentiable legality verifier for macro placements.

Checks boundary violations, pairwise overlaps, and optional spacing constraints.
Returns structured ViolationFeedback with multi-level attribution.
"""

import torch
from torch import Tensor

from .feedback import ViolationFeedback
from .geometry import (
    compute_boundary_violation,
    compute_overlap_area_pairwise,
    compute_spacing_violation_pairwise,
)


class Verifier:
    """Legality verifier for macro placements.

    Args:
        canvas_width: Width of the chip canvas.
        canvas_height: Height of the chip canvas.
        min_spacing: Minimum required spacing between macros. 0 means overlap-only check.
        boundary_weight: Weight for boundary violations in severity aggregation.
        overlap_weight: Weight for overlap violations in severity aggregation.
        spacing_weight: Weight for spacing violations in severity aggregation.
        check_spacing: Whether to check minimum spacing constraints.
    """

    def __init__(
        self,
        canvas_width: float,
        canvas_height: float,
        min_spacing: float = 0.0,
        boundary_weight: float = 1.0,
        overlap_weight: float = 1.0,
        spacing_weight: float = 1.0,
        check_spacing: bool = False,
    ):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.min_spacing = min_spacing
        self.boundary_weight = boundary_weight
        self.overlap_weight = overlap_weight
        self.spacing_weight = spacing_weight
        self.check_spacing = check_spacing or (min_spacing > 0.0)

    @torch.no_grad()
    def __call__(self, centers: Tensor, sizes: Tensor) -> ViolationFeedback:
        """Run legality verification on a placement.

        Args:
            centers: (N, 2) tensor of macro center coordinates (absolute, not normalized).
            sizes: (N, 2) tensor of macro (width, height).

        Returns:
            ViolationFeedback with structured violation information.
        """
        n = centers.shape[0]

        # Pairwise overlap
        overlap_matrix = compute_overlap_area_pairwise(centers, sizes)

        # Boundary violations
        boundary_vec = compute_boundary_violation(
            centers, sizes, self.canvas_width, self.canvas_height
        )

        # Optional spacing violations
        spacing_matrix = None
        if self.check_spacing:
            spacing_matrix = compute_spacing_violation_pairwise(
                centers, sizes, self.min_spacing
            )

        # Aggregate severity per macro
        overlap_severity = overlap_matrix.sum(dim=1)  # (N,)
        severity = (
            self.overlap_weight * overlap_severity
            + self.boundary_weight * boundary_vec
        )
        if spacing_matrix is not None:
            spacing_severity = spacing_matrix.sum(dim=1)
            severity = severity + self.spacing_weight * spacing_severity

        # Global statistics
        num_overlap_violations = (overlap_matrix > 0).sum().item() // 2  # symmetric
        total_overlap_area = overlap_matrix.sum().item() / 2.0  # symmetric
        num_boundary_violations = (boundary_vec > 0).sum().item()
        total_boundary_protrusion = boundary_vec.sum().item()
        num_spacing_violations = 0
        if spacing_matrix is not None:
            # Exclude pairs that already overlap (spacing is subsumed by overlap)
            spacing_only = (spacing_matrix > 0) & (overlap_matrix == 0)
            num_spacing_violations = spacing_only.sum().item() // 2

        total_violations = num_overlap_violations + num_boundary_violations + num_spacing_violations
        is_legal = total_violations == 0

        global_stats = {
            "is_legal": is_legal,
            "total_violations": total_violations,
            "num_overlap_violations": num_overlap_violations,
            "total_overlap_area": total_overlap_area,
            "num_boundary_violations": num_boundary_violations,
            "total_boundary_protrusion": total_boundary_protrusion,
            "num_spacing_violations": num_spacing_violations,
            "num_offending_macros": (severity > 0).sum().item(),
        }

        return ViolationFeedback(
            pairwise_overlap=overlap_matrix,
            boundary_violations=boundary_vec,
            severity_vector=severity,
            global_stats=global_stats,
            pairwise_spacing=spacing_matrix,
        )

    def verify_batch(
        self, centers_batch: Tensor, sizes_batch: Tensor
    ) -> list[ViolationFeedback]:
        """Run verification on a batch of placements.

        Args:
            centers_batch: (B, N, 2) tensor of macro center coordinates.
            sizes_batch: (B, N, 2) or (N, 2) tensor of macro sizes.
                If (N, 2), the same sizes are used for all placements in the batch.

        Returns:
            List of B ViolationFeedback objects.
        """
        b = centers_batch.shape[0]
        if sizes_batch.dim() == 2:
            sizes_batch = sizes_batch.unsqueeze(0).expand(b, -1, -1)

        results = []
        for i in range(b):
            results.append(self(centers_batch[i], sizes_batch[i]))
        return results
