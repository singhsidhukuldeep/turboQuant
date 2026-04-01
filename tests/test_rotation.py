"""Tests for rotation matrices."""

import pytest
import torch

from turboquant.rotation import (
    hadamard_rotate,
    hadamard_rotate_inverse,
    qr_rotate,
    qr_rotate_inverse,
    rotate,
    rotate_inverse,
)


class TestHadamardTransform:
    def test_invertible(self):
        x = torch.randn(4, 128)
        y = hadamard_rotate(x, seed=42)
        x_recovered = hadamard_rotate_inverse(y, seed=42)
        assert torch.allclose(x, x_recovered, atol=1e-5)

    def test_preserves_norm(self):
        """Orthogonal transforms preserve vector norms."""
        x = torch.randn(8, 64)
        y = hadamard_rotate(x, seed=42)
        norms_x = torch.norm(x, dim=-1)
        norms_y = torch.norm(y, dim=-1)
        assert torch.allclose(norms_x, norms_y, atol=1e-5)

    def test_different_seeds_give_different_results(self):
        x = torch.randn(4, 128)
        y1 = hadamard_rotate(x, seed=42)
        y2 = hadamard_rotate(x, seed=99)
        assert not torch.allclose(y1, y2)

    def test_deterministic(self):
        x = torch.randn(4, 128)
        y1 = hadamard_rotate(x, seed=42)
        y2 = hadamard_rotate(x, seed=42)
        assert torch.equal(y1, y2)

    def test_batch_dimensions(self):
        x = torch.randn(2, 3, 4, 64)
        y = hadamard_rotate(x, seed=42)
        assert y.shape == x.shape


class TestQRRotation:
    def test_invertible(self):
        x = torch.randn(4, 64)
        y = qr_rotate(x, seed=42)
        x_recovered = qr_rotate_inverse(y, seed=42)
        assert torch.allclose(x, x_recovered, atol=1e-5)

    def test_preserves_norm(self):
        x = torch.randn(8, 64)
        y = qr_rotate(x, seed=42)
        norms_x = torch.norm(x, dim=-1)
        norms_y = torch.norm(y, dim=-1)
        assert torch.allclose(norms_x, norms_y, atol=1e-5)


class TestUnifiedAPI:
    @pytest.mark.parametrize("method", ["hadamard", "qr"])
    def test_rotate_inverse_roundtrip(self, method):
        dim = 64
        x = torch.randn(10, dim)
        y = rotate(x, method=method, seed=42)
        x_rec = rotate_inverse(y, method=method, seed=42)
        assert torch.allclose(x, x_rec, atol=1e-4)

    def test_invalid_method_raises(self):
        x = torch.randn(4, 64)
        with pytest.raises(ValueError):
            rotate(x, method="invalid")
