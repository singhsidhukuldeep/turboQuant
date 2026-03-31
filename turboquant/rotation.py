"""Random orthogonal rotation matrices for TurboQuant.

Two methods are supported:
- Hadamard: Fast O(d log d) transform using Walsh-Hadamard + random signs.
- QR: Exact Haar-distributed orthogonal matrix via QR decomposition, O(d^2).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Fast Walsh-Hadamard Transform
# ---------------------------------------------------------------------------

def _fast_hadamard_transform(x: Tensor) -> Tensor:
    """Apply the normalized Walsh-Hadamard transform along the last dimension.

    The last dimension must be a power of 2.
    """
    d = x.shape[-1]
    h = 1
    while h < d:
        # Split the last dimension into blocks of size 2h
        x = x.unflatten(-1, (-1, 2 * h))  # (..., d/(2h), 2h)
        lo = x[..., :h]  # (..., d/(2h), h)
        hi = x[..., h:]  # (..., d/(2h), h)
        x = torch.cat([lo + hi, lo - hi], dim=-1)  # (..., d/(2h), 2h)
        x = x.flatten(-2)  # (..., d)
        h *= 2
    return x / math.sqrt(d)


def _get_random_signs(dim: int, seed: int, device: torch.device) -> Tensor:
    """Generate a vector of random +1/-1 signs."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    signs = torch.randint(0, 2, (dim,), generator=gen, dtype=torch.float32) * 2 - 1
    return signs.to(device)


def hadamard_rotate(x: Tensor, seed: int = 42) -> Tensor:
    """Apply randomized Hadamard rotation: H @ diag(signs) @ x.

    Args:
        x: Input tensor with last dim being a power of 2.
        seed: Random seed for sign generation.

    Returns:
        Rotated tensor, same shape as x.
    """
    signs = _get_random_signs(x.shape[-1], seed, x.device).to(x.dtype)
    return _fast_hadamard_transform(x * signs)


def hadamard_rotate_inverse(x: Tensor, seed: int = 42) -> Tensor:
    """Inverse of hadamard_rotate: diag(signs) @ H @ x.

    Since H is self-inverse (unitary symmetric) and D is self-inverse,
    the inverse of H @ D is D @ H.
    """
    signs = _get_random_signs(x.shape[-1], seed, x.device).to(x.dtype)
    return _fast_hadamard_transform(x) * signs


# ---------------------------------------------------------------------------
# QR-based Haar-distributed random orthogonal matrix
# ---------------------------------------------------------------------------

def _make_haar_matrix(dim: int, seed: int) -> Tensor:
    """Generate a Haar-distributed random orthogonal matrix via QR."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    G = torch.randn(dim, dim, generator=gen, dtype=torch.float32)
    Q, R = torch.linalg.qr(G)
    # Fix signs to ensure Haar distribution
    Q = Q @ torch.diag(torch.sign(torch.diag(R)))
    return Q


# Cache for QR matrices (they're deterministic given dim+seed)
_qr_cache: dict[tuple[int, int], Tensor] = {}


def qr_rotate(x: Tensor, seed: int = 42) -> Tensor:
    """Apply Haar-distributed random orthogonal rotation via cached QR matrix."""
    dim = x.shape[-1]
    key = (dim, seed)
    if key not in _qr_cache:
        _qr_cache[key] = _make_haar_matrix(dim, seed)
    Q = _qr_cache[key].to(device=x.device, dtype=x.dtype)
    return x @ Q.T


def qr_rotate_inverse(x: Tensor, seed: int = 42) -> Tensor:
    """Inverse of qr_rotate: x @ Q (since Q^{-1} = Q^T for orthogonal Q)."""
    dim = x.shape[-1]
    key = (dim, seed)
    if key not in _qr_cache:
        _qr_cache[key] = _make_haar_matrix(dim, seed)
    Q = _qr_cache[key].to(device=x.device, dtype=x.dtype)
    return x @ Q


# ---------------------------------------------------------------------------
# Unified API
# ---------------------------------------------------------------------------

def rotate(x: Tensor, method: str = "hadamard", seed: int = 42) -> Tensor:
    """Apply random rotation to x along the last dimension."""
    if method == "hadamard":
        return hadamard_rotate(x, seed=seed)
    elif method == "qr":
        return qr_rotate(x, seed=seed)
    raise ValueError(f"Unknown rotation method: {method}")


def rotate_inverse(x: Tensor, method: str = "hadamard", seed: int = 42) -> Tensor:
    """Apply inverse random rotation to x along the last dimension."""
    if method == "hadamard":
        return hadamard_rotate_inverse(x, seed=seed)
    elif method == "qr":
        return qr_rotate_inverse(x, seed=seed)
    raise ValueError(f"Unknown rotation method: {method}")
