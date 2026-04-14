"""Shared test fixtures for VSR-Place tests."""

import pytest
import torch


@pytest.fixture
def device():
    """Return the test device (CPU for CI, GPU if available)."""
    return torch.device("cpu")


@pytest.fixture
def simple_placement(device):
    """Two non-overlapping macros inside a 10x10 canvas."""
    centers = torch.tensor([[2.0, 2.0], [7.0, 7.0]], device=device)
    sizes = torch.tensor([[2.0, 2.0], [2.0, 2.0]], device=device)
    return centers, sizes


@pytest.fixture
def overlapping_placement(device):
    """Two overlapping macros inside a 10x10 canvas."""
    centers = torch.tensor([[4.0, 5.0], [5.0, 5.0]], device=device)
    sizes = torch.tensor([[3.0, 3.0], [3.0, 3.0]], device=device)
    return centers, sizes


@pytest.fixture
def boundary_violating_placement(device):
    """One macro partially outside a 10x10 canvas."""
    centers = torch.tensor([[1.0, 5.0], [9.5, 5.0]], device=device)
    sizes = torch.tensor([[3.0, 2.0], [2.0, 2.0]], device=device)
    # First macro: x_min = -0.5 (violates left boundary)
    # Second macro: x_max = 10.5 (violates right boundary)
    return centers, sizes


@pytest.fixture
def mixed_violation_placement(device):
    """Multiple macros with both overlap and boundary violations."""
    centers = torch.tensor([
        [2.0, 2.0],   # OK
        [2.5, 2.0],   # Overlaps with macro 0
        [9.5, 5.0],   # Boundary violation (right)
        [5.0, 5.0],   # OK
    ], device=device)
    sizes = torch.tensor([
        [2.0, 2.0],
        [2.0, 2.0],
        [2.0, 2.0],
        [1.0, 1.0],
    ], device=device)
    return centers, sizes


@pytest.fixture
def canvas_10x10():
    """10x10 canvas dimensions."""
    return 10.0, 10.0
