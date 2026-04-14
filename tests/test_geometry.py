"""Tests for geometry primitives."""

import torch
import pytest

from vsr_place.verifier.geometry import (
    compute_overlap_area_pairwise,
    compute_boundary_violation,
    compute_spacing_violation_pairwise,
)


class TestOverlapAreaPairwise:
    def test_no_overlap(self, device):
        """Two rectangles far apart should have zero overlap."""
        centers = torch.tensor([[1.0, 1.0], [8.0, 8.0]], device=device)
        sizes = torch.tensor([[2.0, 2.0], [2.0, 2.0]], device=device)
        overlap = compute_overlap_area_pairwise(centers, sizes)
        assert overlap.shape == (2, 2)
        assert torch.allclose(overlap, torch.zeros(2, 2, device=device))

    def test_partial_overlap(self, device):
        """Two rectangles with partial overlap."""
        # Rect 0: center (4, 5), size (3, 3) -> x: [2.5, 5.5], y: [3.5, 6.5]
        # Rect 1: center (5, 5), size (3, 3) -> x: [3.5, 6.5], y: [3.5, 6.5]
        # Overlap: x: [3.5, 5.5] = 2.0, y: [3.5, 6.5] = 3.0 -> area = 6.0
        centers = torch.tensor([[4.0, 5.0], [5.0, 5.0]], device=device)
        sizes = torch.tensor([[3.0, 3.0], [3.0, 3.0]], device=device)
        overlap = compute_overlap_area_pairwise(centers, sizes)
        assert pytest.approx(overlap[0, 1].item(), abs=1e-6) == 6.0
        assert pytest.approx(overlap[1, 0].item(), abs=1e-6) == 6.0

    def test_diagonal_zero(self, device):
        """Diagonal entries should always be zero."""
        centers = torch.tensor([[3.0, 3.0], [3.0, 3.0]], device=device)
        sizes = torch.tensor([[4.0, 4.0], [4.0, 4.0]], device=device)
        overlap = compute_overlap_area_pairwise(centers, sizes)
        assert overlap[0, 0].item() == 0.0
        assert overlap[1, 1].item() == 0.0

    def test_symmetric(self, device):
        """Overlap matrix should be symmetric."""
        centers = torch.tensor([[2.0, 3.0], [4.0, 5.0], [3.0, 3.0]], device=device)
        sizes = torch.tensor([[3.0, 3.0], [2.0, 2.0], [4.0, 4.0]], device=device)
        overlap = compute_overlap_area_pairwise(centers, sizes)
        assert torch.allclose(overlap, overlap.T)

    def test_touching_no_overlap(self, device):
        """Rectangles that touch edges but don't overlap should have zero overlap."""
        # Rect 0: center (1, 1), size (2, 2) -> x: [0, 2], y: [0, 2]
        # Rect 1: center (3, 1), size (2, 2) -> x: [2, 4], y: [0, 2]
        # They share edge x=2 but overlap area = 0
        centers = torch.tensor([[1.0, 1.0], [3.0, 1.0]], device=device)
        sizes = torch.tensor([[2.0, 2.0], [2.0, 2.0]], device=device)
        overlap = compute_overlap_area_pairwise(centers, sizes)
        assert overlap[0, 1].item() == 0.0

    def test_fully_contained(self, device):
        """One rectangle fully inside another."""
        # Outer: center (5, 5), size (10, 10) -> x: [0, 10], y: [0, 10]
        # Inner: center (5, 5), size (2, 2) -> x: [4, 6], y: [4, 6]
        # Overlap area = 2*2 = 4
        centers = torch.tensor([[5.0, 5.0], [5.0, 5.0]], device=device)
        sizes = torch.tensor([[10.0, 10.0], [2.0, 2.0]], device=device)
        overlap = compute_overlap_area_pairwise(centers, sizes)
        assert pytest.approx(overlap[0, 1].item(), abs=1e-6) == 4.0


class TestBoundaryViolation:
    def test_inside_canvas(self, device):
        """Macro fully inside canvas should have zero violation."""
        centers = torch.tensor([[5.0, 5.0]], device=device)
        sizes = torch.tensor([[2.0, 2.0]], device=device)
        violation = compute_boundary_violation(centers, sizes, 10.0, 10.0)
        assert violation[0].item() == 0.0

    def test_left_boundary(self, device):
        """Macro protruding from left boundary."""
        # Center (0.5, 5), size (2, 2) -> x_min = -0.5
        centers = torch.tensor([[0.5, 5.0]], device=device)
        sizes = torch.tensor([[2.0, 2.0]], device=device)
        violation = compute_boundary_violation(centers, sizes, 10.0, 10.0)
        assert pytest.approx(violation[0].item(), abs=1e-6) == 0.5

    def test_right_boundary(self, device):
        """Macro protruding from right boundary."""
        # Center (9.5, 5), size (2, 2) -> x_max = 10.5
        centers = torch.tensor([[9.5, 5.0]], device=device)
        sizes = torch.tensor([[2.0, 2.0]], device=device)
        violation = compute_boundary_violation(centers, sizes, 10.0, 10.0)
        assert pytest.approx(violation[0].item(), abs=1e-6) == 0.5

    def test_corner_violation(self, device):
        """Macro protruding from corner (two edges)."""
        # Center (0.5, 0.5), size (2, 2) -> x_min=-0.5, y_min=-0.5
        centers = torch.tensor([[0.5, 0.5]], device=device)
        sizes = torch.tensor([[2.0, 2.0]], device=device)
        violation = compute_boundary_violation(centers, sizes, 10.0, 10.0)
        assert pytest.approx(violation[0].item(), abs=1e-6) == 1.0  # 0.5 + 0.5

    def test_multiple_macros(self, device):
        """Mix of inside and outside macros."""
        centers = torch.tensor([[5.0, 5.0], [0.5, 5.0]], device=device)
        sizes = torch.tensor([[2.0, 2.0], [2.0, 2.0]], device=device)
        violation = compute_boundary_violation(centers, sizes, 10.0, 10.0)
        assert violation[0].item() == 0.0
        assert pytest.approx(violation[1].item(), abs=1e-6) == 0.5


class TestSpacingViolation:
    def test_well_separated(self, device):
        """Macros with plenty of spacing should have zero violation."""
        centers = torch.tensor([[2.0, 5.0], [8.0, 5.0]], device=device)
        sizes = torch.tensor([[2.0, 2.0], [2.0, 2.0]], device=device)
        violation = compute_spacing_violation_pairwise(centers, sizes, min_spacing=1.0)
        assert violation[0, 1].item() == 0.0

    def test_too_close(self, device):
        """Macros closer than min_spacing."""
        # Rect 0: center (3, 5), size (2, 2) -> x: [2, 4]
        # Rect 1: center (4.5, 5), size (2, 2) -> x: [3.5, 5.5]
        # Gap in x: 3.5 - 4 = -0.5 (overlapping), gap in y: same
        # Actually these overlap, so spacing violation = min_spacing - (-overlap) = min_spacing + overlap
        centers = torch.tensor([[3.0, 5.0], [5.5, 5.0]], device=device)
        sizes = torch.tensor([[2.0, 2.0], [2.0, 2.0]], device=device)
        # Gap: x_min_1 - x_max_0 = 4.5 - 4.0 = 0.5
        violation = compute_spacing_violation_pairwise(centers, sizes, min_spacing=1.0)
        assert pytest.approx(violation[0, 1].item(), abs=1e-6) == 0.5  # need 1.0, have 0.5

    def test_symmetric(self, device):
        """Spacing violation matrix should be symmetric."""
        centers = torch.tensor([[2.0, 3.0], [4.0, 3.0], [3.0, 6.0]], device=device)
        sizes = torch.tensor([[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]], device=device)
        violation = compute_spacing_violation_pairwise(centers, sizes, min_spacing=0.5)
        assert torch.allclose(violation, violation.T)
