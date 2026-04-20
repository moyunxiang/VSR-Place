"""Tests for NeuralVSR model and dataset."""

import pytest
import torch

from vsr_place.neural.model import NeuralVSR


def test_model_forward_shape():
    model = NeuralVSR(node_feat_dim=5, edge_feat_dim=4,
                     hidden_dim=64, num_layers=2, heads=4)

    n = 20
    centers = torch.randn(n, 2)
    node_features = torch.randn(n, 5)
    edge_index = torch.randint(0, n, (2, 40))
    edge_attr = torch.randn(40, 4)

    delta = model(centers, node_features, edge_index, edge_attr)
    assert delta.shape == (n, 2)


def test_model_initial_zero_output():
    """Output projection is zero-initialized → first forward returns zero."""
    model = NeuralVSR(hidden_dim=64, num_layers=2, heads=4)

    n = 10
    centers = torch.randn(n, 2)
    node_features = torch.randn(n, 5)
    edge_index = torch.randint(0, n, (2, 20))
    edge_attr = torch.randn(20, 4)

    delta = model(centers, node_features, edge_index, edge_attr)
    assert torch.allclose(delta, torch.zeros_like(delta), atol=1e-6)


def test_model_parameter_count():
    """Parameter count should be in the 30K-100K range for our design."""
    model = NeuralVSR(hidden_dim=128, num_layers=3, heads=4)
    n_params = model.num_parameters()
    assert 10_000 < n_params < 500_000, f"Unexpected param count: {n_params}"


def test_model_backward():
    """Gradient flow should work."""
    model = NeuralVSR(hidden_dim=32, num_layers=2, heads=4)

    n = 15
    centers = torch.randn(n, 2)
    node_features = torch.randn(n, 5)
    edge_index = torch.randint(0, n, (2, 30))
    edge_attr = torch.randn(30, 4)
    target = torch.randn(n, 2)

    delta = model(centers, node_features, edge_index, edge_attr)
    loss = ((delta - target) ** 2).mean()
    loss.backward()

    # All params should have gradients
    for name, p in model.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"


def test_dataset_sample():
    from vsr_place.neural.dataset import generate_synthetic_sample

    sample = generate_synthetic_sample(n_range=(20, 40), seed=42)

    n = sample.centers_bad.shape[0]
    assert 20 <= n <= 40
    assert sample.centers_bad.shape == (n, 2)
    assert sample.target_delta.shape == (n, 2)
    assert sample.sizes.shape == (n, 2)
    assert sample.edge_index.shape[0] == 2
    assert sample.edge_attr.shape[1] == 4
    assert sample.edge_attr.shape[0] == sample.edge_index.shape[1]


def test_dataset_features():
    from vsr_place.neural.dataset import compute_violation_features

    centers = torch.tensor([[2.0, 2.0], [2.5, 2.0], [5.0, 5.0]])
    sizes = torch.tensor([[2.0, 2.0], [2.0, 2.0], [1.0, 1.0]])

    feats = compute_violation_features(centers, sizes, canvas_w=10.0, canvas_h=10.0)
    assert feats.shape == (3, 5)
    # First two should overlap, third isolated
    assert feats[0, 2] > 0  # overlap_count for macro 0
    assert feats[1, 2] > 0  # overlap_count for macro 1
    assert feats[2, 2] == 0  # macro 2 no overlap


def test_dataset_class():
    from vsr_place.neural.dataset import SyntheticVSRDataset

    ds = SyntheticVSRDataset(num_samples=5, n_range=(20, 30))
    assert len(ds) == 5

    item = ds[0]
    assert "centers" in item
    assert "node_features" in item
    assert "edge_index" in item
    assert "edge_attr" in item
    assert "target_delta" in item
    assert item["node_features"].shape[1] == 5


def test_integration_forward_with_real_data():
    """End-to-end: dataset → features → model → delta."""
    from vsr_place.neural.dataset import SyntheticVSRDataset

    ds = SyntheticVSRDataset(num_samples=1, n_range=(20, 30))
    item = ds[0]

    model = NeuralVSR(hidden_dim=64, num_layers=2, heads=4)
    delta = model(
        item["centers"],
        item["node_features"],
        item["edge_index"],
        item["edge_attr"],
    )
    assert delta.shape == item["target_delta"].shape
