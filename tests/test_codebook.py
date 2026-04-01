"""Tests for the Lloyd-Max codebook generation."""

import torch

from turboquant.codebook import (
    build_codebook,
    dequantize_from_indices,
    get_codebook,
    quantize_to_indices,
)


class TestBuildCodebook:
    def test_correct_number_of_centroids(self):
        for n_bits in [2, 3, 4]:
            centroids, boundaries = build_codebook(dim=128, n_bits=n_bits)
            assert centroids.shape == (1 << n_bits,)
            assert boundaries.shape == (((1 << n_bits) - 1),)

    def test_centroids_are_sorted(self):
        centroids, _ = build_codebook(dim=128, n_bits=4)
        assert torch.all(centroids[1:] > centroids[:-1])

    def test_boundaries_between_centroids(self):
        centroids, boundaries = build_codebook(dim=128, n_bits=4)
        for i, b in enumerate(boundaries):
            assert centroids[i] < b < centroids[i + 1]

    def test_centroids_symmetric(self):
        """For symmetric Beta distribution, centroids should be roughly symmetric."""
        centroids, _ = build_codebook(dim=128, n_bits=4)
        # Check that centroids are approximately symmetric around 0
        assert abs(centroids.mean().item()) < 0.01

    def test_higher_dim_narrower_centroids(self):
        """Higher dimensions => more concentrated distribution => narrower centroid range."""
        c_low, _ = build_codebook(dim=32, n_bits=4)
        c_high, _ = build_codebook(dim=256, n_bits=4)
        assert c_low.max() > c_high.max()

    def test_caching(self):
        c1, b1 = get_codebook(128, 4)
        c2, b2 = get_codebook(128, 4)
        assert torch.equal(c1, c2)
        assert torch.equal(b1, b2)


class TestQuantizeDequantize:
    def test_round_trip_indices_in_range(self):
        centroids, boundaries = build_codebook(dim=128, n_bits=4)
        x = torch.randn(100) * 0.1  # values in typical range
        indices = quantize_to_indices(x, boundaries)
        assert indices.min() >= 0
        assert indices.max() < 16

    def test_dequantize_returns_centroids(self):
        centroids, boundaries = build_codebook(dim=128, n_bits=4)
        indices = torch.tensor([0, 3, 7, 15])
        values = dequantize_from_indices(indices, centroids)
        assert torch.equal(values, centroids[indices])

    def test_quantize_preserves_ordering(self):
        """Nearby values should map to same or adjacent centroids."""
        centroids, boundaries = build_codebook(dim=128, n_bits=4)
        x = torch.linspace(-0.2, 0.2, 50)
        indices = quantize_to_indices(x, boundaries)
        # Indices should be non-decreasing
        assert torch.all(indices[1:] >= indices[:-1])
