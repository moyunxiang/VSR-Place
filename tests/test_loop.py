"""Integration tests for VSR loop with mock diffusion backend."""

import torch
import pytest

from vsr_place.loop.vsr_loop import VSRLoop, VSRResult
from vsr_place.loop.budget import RepairBudget
from vsr_place.verifier.verifier import Verifier
from vsr_place.renoising.selector import ThresholdSelector
from vsr_place.renoising.strength import FixedStrength


class MockBackend:
    """Mock diffusion backend that moves macros toward valid positions."""

    def __init__(self, canvas_w: float = 10.0, canvas_h: float = 10.0, num_macros: int = 4):
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h
        self.num_macros = num_macros
        self.sample_count = 0
        self.denoise_count = 0

    def sample(self, cond, **kwargs):
        """Generate a placement with some violations."""
        self.sample_count += 1
        # Place macros with some overlaps and boundary violations
        centers = torch.tensor([
            [2.0, 2.0],   # OK
            [2.5, 2.0],   # Overlaps with macro 0
            [9.5, 5.0],   # Boundary violation (right edge)
            [5.0, 5.0],   # OK
        ])
        return centers

    def denoise_from(self, x_start, cond, start_timestep, num_steps, **kwargs):
        """Simulate re-denoising by nudging macros toward valid positions."""
        self.denoise_count += 1
        x = x_start.clone()
        # Simple heuristic: move overlapping macros apart, pull boundary violations in
        for i in range(x.shape[0]):
            # Pull toward center if outside boundary
            if x[i, 0] > self.canvas_w - 1:
                x[i, 0] -= 0.5
            if x[i, 0] < 1:
                x[i, 0] += 0.5
            if x[i, 1] > self.canvas_h - 1:
                x[i, 1] -= 0.5
            if x[i, 1] < 1:
                x[i, 1] += 0.5
        # Spread apart macros 0 and 1
        if x.shape[0] >= 2 and abs(x[0, 0] - x[1, 0]) < 2.0:
            x[1, 0] = x[0, 0] + 3.0
        return x

    def decode_placement(self, x, cond):
        """Identity decode (already in absolute coords for mock)."""
        sizes = torch.tensor([
            [2.0, 2.0],
            [2.0, 2.0],
            [2.0, 2.0],
            [1.0, 1.0],
        ])
        return x, sizes

    def encode_placement(self, centers, cond):
        return centers


class TestVSRLoop:
    def test_loop_reduces_violations(self):
        """VSR loop should reduce violations over iterations."""
        backend = MockBackend(canvas_w=10.0, canvas_h=10.0)
        verifier = Verifier(canvas_width=10.0, canvas_height=10.0)
        selector = ThresholdSelector(threshold=0.0)
        strength = FixedStrength(alpha=0.3)
        budget = RepairBudget(max_loops=4, early_stop_on_legal=True)

        loop = VSRLoop(
            backend=backend,
            verifier=verifier,
            selector=selector,
            strength=strength,
            budget=budget,
            save_intermediates=True,
        )

        result = loop.run(cond=None)

        assert isinstance(result, VSRResult)
        assert result.placement is not None
        assert len(result.feedback_history) >= 1
        assert result.metrics["num_repair_loops"] >= 0

        # Violations should decrease or reach zero
        trajectory = result.metrics["violation_history"]
        assert trajectory[-1] <= trajectory[0]

    def test_loop_terminates_on_legal(self):
        """Loop should stop early when placement becomes legal."""
        backend = MockBackend()
        verifier = Verifier(canvas_width=10.0, canvas_height=10.0)
        selector = ThresholdSelector(threshold=0.0)
        strength = FixedStrength(alpha=0.3)
        budget = RepairBudget(max_loops=10, early_stop_on_legal=True)

        loop = VSRLoop(
            backend=backend,
            verifier=verifier,
            selector=selector,
            strength=strength,
            budget=budget,
        )

        result = loop.run(cond=None)

        # Should not use all 10 loops if placement becomes legal
        assert result.metrics["num_repair_loops"] < 10

    def test_loop_respects_budget(self):
        """Loop should terminate when budget is exhausted."""

        class NeverFixBackend(MockBackend):
            def denoise_from(self, x_start, cond, start_timestep, num_steps, **kwargs):
                return x_start  # Never fixes anything

        backend = NeverFixBackend()
        verifier = Verifier(canvas_width=10.0, canvas_height=10.0)
        selector = ThresholdSelector(threshold=0.0)
        strength = FixedStrength(alpha=0.3)
        budget = RepairBudget(max_loops=3, early_stop_on_legal=False)

        loop = VSRLoop(
            backend=backend,
            verifier=verifier,
            selector=selector,
            strength=strength,
            budget=budget,
        )

        result = loop.run(cond=None)
        assert result.metrics["num_repair_loops"] == 3

    def test_loop_records_metrics(self):
        """Loop should record proper metrics."""
        backend = MockBackend()
        verifier = Verifier(canvas_width=10.0, canvas_height=10.0)
        selector = ThresholdSelector(threshold=0.0)
        strength = FixedStrength(alpha=0.3)
        budget = RepairBudget(max_loops=2)

        loop = VSRLoop(
            backend=backend,
            verifier=verifier,
            selector=selector,
            strength=strength,
            budget=budget,
        )

        result = loop.run(cond=None)

        assert "elapsed_time" in result.metrics
        assert "num_repair_loops" in result.metrics
        assert "num_verifier_calls" in result.metrics
        assert "total_violations" in result.metrics
        assert "violation_history" in result.metrics


class TestRepairBudget:
    def test_max_loops(self):
        budget = RepairBudget(max_loops=3)
        assert not budget.is_exhausted(2, [])
        assert budget.is_exhausted(3, [])

    def test_early_stop_on_legal(self):
        from vsr_place.verifier.feedback import ViolationFeedback

        budget = RepairBudget(max_loops=10, early_stop_on_legal=True)
        legal_fb = ViolationFeedback(
            pairwise_overlap=torch.zeros(2, 2),
            boundary_violations=torch.zeros(2),
            severity_vector=torch.zeros(2),
            global_stats={"is_legal": True, "total_violations": 0},
        )
        assert budget.is_exhausted(1, [legal_fb])

    def test_min_improvement(self):
        from vsr_place.verifier.feedback import ViolationFeedback

        budget = RepairBudget(max_loops=10, min_improvement=0.1, early_stop_on_legal=False)

        fb1 = ViolationFeedback(
            pairwise_overlap=torch.zeros(2, 2),
            boundary_violations=torch.zeros(2),
            severity_vector=torch.zeros(2),
            global_stats={"is_legal": False, "total_violations": 10},
        )
        fb2 = ViolationFeedback(
            pairwise_overlap=torch.zeros(2, 2),
            boundary_violations=torch.zeros(2),
            severity_vector=torch.zeros(2),
            global_stats={"is_legal": False, "total_violations": 10},  # No improvement
        )
        assert budget.is_exhausted(2, [fb1, fb2])
