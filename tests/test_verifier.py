"""Tests for the Verifier class and ViolationFeedback."""

import torch
import pytest

from vsr_place.verifier.verifier import Verifier
from vsr_place.verifier.feedback import ViolationFeedback


class TestVerifier:
    def test_legal_placement(self, simple_placement, canvas_10x10):
        """Legal placement should pass all checks."""
        centers, sizes = simple_placement
        w, h = canvas_10x10
        verifier = Verifier(canvas_width=w, canvas_height=h)
        feedback = verifier(centers, sizes)

        assert feedback.is_legal()
        assert feedback.global_stats["total_violations"] == 0
        assert feedback.global_stats["num_overlap_violations"] == 0
        assert feedback.global_stats["num_boundary_violations"] == 0

    def test_overlap_detected(self, overlapping_placement, canvas_10x10):
        """Overlapping macros should be detected."""
        centers, sizes = overlapping_placement
        w, h = canvas_10x10
        verifier = Verifier(canvas_width=w, canvas_height=h)
        feedback = verifier(centers, sizes)

        assert not feedback.is_legal()
        assert feedback.global_stats["num_overlap_violations"] == 1
        assert feedback.global_stats["total_overlap_area"] > 0

    def test_boundary_detected(self, boundary_violating_placement, canvas_10x10):
        """Boundary violations should be detected."""
        centers, sizes = boundary_violating_placement
        w, h = canvas_10x10
        verifier = Verifier(canvas_width=w, canvas_height=h)
        feedback = verifier(centers, sizes)

        assert not feedback.is_legal()
        assert feedback.global_stats["num_boundary_violations"] == 2

    def test_mixed_violations(self, mixed_violation_placement, canvas_10x10):
        """Both overlap and boundary violations should be detected."""
        centers, sizes = mixed_violation_placement
        w, h = canvas_10x10
        verifier = Verifier(canvas_width=w, canvas_height=h)
        feedback = verifier(centers, sizes)

        assert not feedback.is_legal()
        assert feedback.global_stats["num_overlap_violations"] > 0
        assert feedback.global_stats["num_boundary_violations"] > 0

    def test_severity_vector_shape(self, mixed_violation_placement, canvas_10x10):
        """Severity vector should have correct shape."""
        centers, sizes = mixed_violation_placement
        w, h = canvas_10x10
        verifier = Verifier(canvas_width=w, canvas_height=h)
        feedback = verifier(centers, sizes)

        assert feedback.severity_vector.shape == (4,)
        assert feedback.pairwise_overlap.shape == (4, 4)
        assert feedback.boundary_violations.shape == (4,)

    def test_offending_macros(self, mixed_violation_placement, canvas_10x10):
        """Offending macros should be identified correctly."""
        centers, sizes = mixed_violation_placement
        w, h = canvas_10x10
        verifier = Verifier(canvas_width=w, canvas_height=h)
        feedback = verifier(centers, sizes)

        offending = feedback.offending_macros(threshold=0.0)
        assert len(offending) > 0
        # Macro 3 (5, 5, 1x1) should not be offending
        assert 3 not in offending.tolist()

    def test_spacing_check(self, device):
        """Spacing violations should be detected when enabled."""
        # Two macros just barely not overlapping but too close
        centers = torch.tensor([[3.0, 5.0], [5.5, 5.0]], device=device)
        sizes = torch.tensor([[2.0, 2.0], [2.0, 2.0]], device=device)
        # Gap = 0.5, min_spacing = 1.0
        verifier = Verifier(canvas_width=10.0, canvas_height=10.0,
                            min_spacing=1.0, check_spacing=True)
        feedback = verifier(centers, sizes)

        assert not feedback.is_legal()
        assert feedback.global_stats["num_spacing_violations"] > 0
        assert feedback.pairwise_spacing is not None


class TestViolationFeedback:
    def test_top_k_offending(self, device):
        """Top-k selection should return the most severe macros."""
        feedback = ViolationFeedback(
            pairwise_overlap=torch.zeros(5, 5, device=device),
            boundary_violations=torch.zeros(5, device=device),
            severity_vector=torch.tensor([0.0, 3.0, 1.0, 5.0, 2.0], device=device),
        )
        top2 = feedback.top_k_offending(2)
        assert set(top2.tolist()) == {1, 3}

    def test_offending_mask(self, device):
        """Offending mask should correctly threshold."""
        feedback = ViolationFeedback(
            pairwise_overlap=torch.zeros(4, 4, device=device),
            boundary_violations=torch.zeros(4, device=device),
            severity_vector=torch.tensor([0.0, 0.5, 0.0, 1.5], device=device),
        )
        mask = feedback.offending_mask(threshold=0.3)
        assert mask.tolist() == [False, True, False, True]

    def test_is_legal_when_no_violations(self, device):
        """is_legal should return True when global_stats says so."""
        feedback = ViolationFeedback(
            pairwise_overlap=torch.zeros(3, 3, device=device),
            boundary_violations=torch.zeros(3, device=device),
            severity_vector=torch.zeros(3, device=device),
            global_stats={"is_legal": True},
        )
        assert feedback.is_legal()
