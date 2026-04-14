"""Tests for selective re-noising."""

import torch
import pytest

from vsr_place.renoising.renoise import selective_renoise, selective_renoise_batch


class TestSelectiveRenoise:
    def test_unselected_macros_unchanged(self):
        """Non-selected macros should be bit-identical to input."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        mask = torch.tensor([True, False, True])
        alpha = torch.tensor([0.5, 0.0, 0.3])

        x_out = selective_renoise(x, mask, alpha)

        # Macro 1 (not selected) should be unchanged
        assert torch.equal(x_out[1], x[1])

    def test_selected_macros_modified(self):
        """Selected macros should be different from input (with high probability)."""
        torch.manual_seed(42)
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        mask = torch.tensor([True, True])
        alpha = torch.tensor([0.5, 0.5])

        x_out = selective_renoise(x, mask, alpha)

        # Very unlikely to be identical with random noise
        assert not torch.equal(x_out, x)

    def test_zero_alpha_no_change(self):
        """Alpha = 0 should produce no noise (identity)."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        mask = torch.tensor([True, True])
        alpha = torch.tensor([0.0, 0.0])

        x_out = selective_renoise(x, mask, alpha)
        assert torch.allclose(x_out, x)

    def test_full_alpha_is_pure_noise(self):
        """Alpha = 1 should produce pure noise (no signal)."""
        torch.manual_seed(123)
        noise = torch.randn(2, 2)
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        mask = torch.tensor([True, True])
        alpha = torch.tensor([1.0, 1.0])

        x_out = selective_renoise(x, mask, alpha, noise=noise)

        # With alpha=1: x' = sqrt(0)*x + sqrt(1)*noise = noise
        assert torch.allclose(x_out, noise)

    def test_deterministic_with_fixed_noise(self):
        """Same noise should produce same output."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        mask = torch.tensor([True, False])
        alpha = torch.tensor([0.5, 0.0])
        noise = torch.tensor([[0.1, 0.2], [0.3, 0.4]])

        x_out1 = selective_renoise(x, mask, alpha, noise=noise)
        x_out2 = selective_renoise(x, mask, alpha, noise=noise)

        assert torch.equal(x_out1, x_out2)

    def test_output_shape(self):
        """Output should have same shape as input."""
        x = torch.randn(10, 2)
        mask = torch.ones(10, dtype=torch.bool)
        alpha = torch.full((10,), 0.3)

        x_out = selective_renoise(x, mask, alpha)
        assert x_out.shape == x.shape

    def test_noise_variance(self):
        """Re-noised positions should have expected variance."""
        torch.manual_seed(0)
        n = 1000
        x = torch.zeros(n, 2)  # All at origin
        mask = torch.ones(n, dtype=torch.bool)
        alpha_val = 0.4
        alpha = torch.full((n,), alpha_val)

        x_out = selective_renoise(x, mask, alpha)

        # Expected: x' = sqrt(1-alpha)*0 + sqrt(alpha)*eps = sqrt(alpha)*eps
        # Variance should be approximately alpha
        empirical_var = x_out.var().item()
        assert pytest.approx(empirical_var, abs=0.05) == alpha_val


class TestSelectiveRenoiseBatch:
    def test_batch_shape(self):
        """Batch output should have correct shape."""
        b, n = 4, 5
        x = torch.randn(b, n, 2)
        mask = torch.ones(b, n, dtype=torch.bool)
        alpha = torch.full((b, n), 0.3)

        x_out = selective_renoise_batch(x, mask, alpha)
        assert x_out.shape == (b, n, 2)

    def test_batch_unselected_unchanged(self):
        """Non-selected macros in batch should be unchanged."""
        x = torch.randn(2, 3, 2)
        mask = torch.tensor([[True, False, True], [False, True, False]])
        alpha = torch.tensor([[0.5, 0.0, 0.3], [0.0, 0.4, 0.0]])

        x_out = selective_renoise_batch(x, mask, alpha)

        assert torch.equal(x_out[0, 1], x[0, 1])
        assert torch.equal(x_out[1, 0], x[1, 0])
        assert torch.equal(x_out[1, 2], x[1, 2])
