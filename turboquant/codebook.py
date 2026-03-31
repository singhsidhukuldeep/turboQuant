"""Lloyd-Max optimal scalar quantizer for the Beta distribution arising from TurboQuant rotation."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor


def _beta_log_pdf(x: Tensor, dim: int) -> Tensor:
    """Unnormalized log-PDF of the symmetric Beta distribution on [-1, 1].

    After rotating a unit vector with a random orthogonal matrix in R^dim,
    each coordinate follows Beta((dim-1)/2, (dim-1)/2) on [-1, 1].
    """
    alpha = (dim - 1) / 2.0
    return (alpha - 1.0) * torch.log(torch.clamp(1.0 - x * x, min=1e-30))


def build_codebook(
    dim: int,
    n_bits: int,
    n_grid: int = 50_000,
    max_iter: int = 500,
    device: Optional[torch.device] = None,
) -> tuple[Tensor, Tensor]:
    """Build an optimal Lloyd-Max codebook for the TurboQuant distribution.

    Args:
        dim: Dimensionality of vectors (= group_size).
        n_bits: Number of quantization bits (2, 3, or 4).
        n_grid: Number of grid points for numerical integration.
        max_iter: Maximum Lloyd-Max iterations.
        device: Device for computation.

    Returns:
        centroids: Tensor of shape (2^n_bits,) with optimal centroid values.
        boundaries: Tensor of shape (2^n_bits - 1,) with decision boundaries.
    """
    n_levels = 1 << n_bits
    std = 1.0 / math.sqrt(dim)

    # Focus grid on the high-probability region: +-6 sigma
    lo, hi = -min(6.0 * std, 0.999), min(6.0 * std, 0.999)
    grid = torch.linspace(lo, hi, n_grid, device=device, dtype=torch.float64)
    dx = grid[1] - grid[0]

    # Compute PDF weights on the grid
    log_pdf = _beta_log_pdf(grid, dim)
    pdf = torch.exp(log_pdf - log_pdf.max())
    pdf = pdf / (pdf.sum() * dx)  # normalize

    # Initialize centroids evenly across the distribution support
    centroids = torch.linspace(lo, hi, n_levels + 2, device=device, dtype=torch.float64)[1:-1]

    for _ in range(max_iter):
        # Decision boundaries = midpoints between centroids
        boundaries = 0.5 * (centroids[:-1] + centroids[1:])

        # Assign grid points to nearest centroid using boundaries
        bins = torch.bucketize(grid, boundaries)

        # Update centroids as PDF-weighted means within each bin
        new_centroids = torch.zeros_like(centroids)
        for i in range(n_levels):
            mask = bins == i
            if mask.any():
                w = pdf[mask]
                new_centroids[i] = (grid[mask] * w).sum() / w.sum()
            else:
                new_centroids[i] = centroids[i]

        if torch.allclose(centroids, new_centroids, atol=1e-12):
            break
        centroids = new_centroids

    # Final boundaries
    boundaries = 0.5 * (centroids[:-1] + centroids[1:])

    return centroids.float(), boundaries.float()


# In-memory cache keyed by (dim, n_bits)
_codebook_cache: dict[tuple[int, int], tuple[Tensor, Tensor]] = {}


def get_codebook(
    dim: int, n_bits: int, device: Optional[torch.device] = None
) -> tuple[Tensor, Tensor]:
    """Get or build a Lloyd-Max codebook, with caching.

    Returns:
        centroids: (2^n_bits,)
        boundaries: (2^n_bits - 1,)
    """
    key = (dim, n_bits)
    if key not in _codebook_cache:
        centroids, boundaries = build_codebook(dim, n_bits, device=device)
        _codebook_cache[key] = (centroids.cpu(), boundaries.cpu())

    centroids, boundaries = _codebook_cache[key]
    c = centroids.to(device) if device is not None else centroids.clone()
    b = boundaries.to(device) if device is not None else boundaries.clone()
    # Always return copies to prevent callers from mutating the cache
    if c.data_ptr() == centroids.data_ptr():
        c = c.clone()
    if b.data_ptr() == boundaries.data_ptr():
        b = b.clone()
    return c, b


def quantize_to_indices(x: Tensor, boundaries: Tensor) -> Tensor:
    """Quantize values to codebook indices using decision boundaries.

    Args:
        x: Values to quantize, any shape.
        boundaries: Sorted decision boundaries of shape (n_levels - 1,).

    Returns:
        Integer indices of same shape as x, values in [0, n_levels).
    """
    return torch.bucketize(x, boundaries)


def dequantize_from_indices(indices: Tensor, centroids: Tensor) -> Tensor:
    """Look up centroid values from indices.

    Args:
        indices: Integer indices, any shape.
        centroids: Centroid values of shape (n_levels,).

    Returns:
        Reconstructed values, same shape as indices.
    """
    return centroids[indices.long()]
