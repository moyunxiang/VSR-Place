"""Tests for macro selection strategies."""

import torch
import pytest

from vsr_place.verifier.feedback import ViolationFeedback
from vsr_place.renoising.selector import (
    GlobalSelector,
    ThresholdSelector,
    TopKSelector,
    AdaptiveThresholdSelector,
)


def _make_feedback(severity: list[float], device=None) -> ViolationFeedback:
    """Helper to create a ViolationFeedback with given severity vector."""
    n = len(severity)
    dev = device or torch.device("cpu")
    return ViolationFeedback(
        pairwise_overlap=torch.zeros(n, n, device=dev),
        boundary_violations=torch.zeros(n, device=dev),
        severity_vector=torch.tensor(severity, device=dev),
    )


class TestGlobalSelector:
    def test_selects_all(self):
        feedback = _make_feedback([0.0, 1.0, 0.5])
        selector = GlobalSelector()
        mask = selector.select(feedback)
        assert mask.all()
        assert mask.shape == (3,)


class TestThresholdSelector:
    def test_selects_above_threshold(self):
        feedback = _make_feedback([0.0, 1.0, 0.3, 0.5])
        selector = ThresholdSelector(threshold=0.4)
        mask = selector.select(feedback)
        assert mask.tolist() == [False, True, False, True]

    def test_zero_threshold_selects_nonzero(self):
        feedback = _make_feedback([0.0, 0.1, 0.0])
        selector = ThresholdSelector(threshold=0.0)
        mask = selector.select(feedback)
        assert mask.tolist() == [False, True, False]

    def test_high_threshold_selects_none(self):
        feedback = _make_feedback([0.1, 0.2, 0.3])
        selector = ThresholdSelector(threshold=10.0)
        mask = selector.select(feedback)
        assert not mask.any()


class TestTopKSelector:
    def test_selects_top_k(self):
        feedback = _make_feedback([0.1, 0.5, 0.3, 0.9, 0.2])
        selector = TopKSelector(k=2)
        mask = selector.select(feedback)
        assert mask.sum().item() == 2
        assert mask[1].item()  # severity 0.5
        assert mask[3].item()  # severity 0.9

    def test_k_larger_than_n(self):
        feedback = _make_feedback([0.1, 0.5])
        selector = TopKSelector(k=10)
        mask = selector.select(feedback)
        assert mask.all()  # Should select all when k > n

    def test_k_one(self):
        feedback = _make_feedback([0.1, 0.5, 0.3])
        selector = TopKSelector(k=1)
        mask = selector.select(feedback)
        assert mask.sum().item() == 1
        assert mask[1].item()


class TestAdaptiveThresholdSelector:
    def test_selects_top_percentile(self):
        feedback = _make_feedback([0.0, 0.1, 0.0, 0.5, 0.3])
        selector = AdaptiveThresholdSelector(percentile=50.0)
        mask = selector.select(feedback)
        # Non-zero values: [0.1, 0.5, 0.3]. Top 50% = top 1-2 values
        assert mask[3].item()  # highest severity

    def test_all_zero_selects_none(self):
        feedback = _make_feedback([0.0, 0.0, 0.0])
        selector = AdaptiveThresholdSelector(percentile=50.0)
        mask = selector.select(feedback)
        assert not mask.any()
