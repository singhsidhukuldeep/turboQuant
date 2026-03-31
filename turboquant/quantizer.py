"""Core TurboQuant quantization engine.

Implements the full pipeline: normalize -> rotate -> scalar quantize -> pack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from .codebook import dequantize_from_indices, get_codebook, quantize_to_indices
from .config import TurboQuantConfig
from .packing import pack_indices, unpack_indices
from .rotation import rotate, rotate_inverse


@dataclass
class QuantizedTensor:
    """Container for a quantized 2D weight tensor.

    Attributes:
        packed_indices: Bit-packed quantization indices for full groups (uint8).
        norms: Per-group norms, shape (n_rows, n_groups), float32.
        shape: Original tensor shape (out_features, in_features).
        bit_width: Bits per element.
        group_size: Quantization group size.
        n_full_groups: Number of complete groups per row.
        remainder_cols: Number of leftover columns stored unquantized (0 if divisible).
        remainder_weights: Unquantized remainder columns, float16. Empty if no remainder.
    """

    packed_indices: Tensor
    norms: Tensor
    shape: tuple[int, int]
    bit_width: int
    group_size: int
    n_full_groups: int
    remainder_cols: int
    remainder_weights: Tensor


class TurboQuantizer:
    """TurboQuant quantization engine for weight tensors.

    Usage:
        quantizer = TurboQuantizer(config)
        qtensor = quantizer.quantize(weight)
        weight_approx = quantizer.dequantize(qtensor)
    """

    def __init__(self, config: TurboQuantConfig):
        self.config = config
        self._codebook_cache: dict[str, tuple[Tensor, Tensor]] = {}

    def _get_codebook(
        self, dim: int, n_bits: int, device: torch.device
    ) -> tuple[Tensor, Tensor]:
        """Get codebook, using instance cache. Always returns safe copies."""
        key = f"{dim}_{n_bits}"
        if key not in self._codebook_cache:
            centroids, boundaries = get_codebook(dim, n_bits, device=device)
            self._codebook_cache[key] = (centroids.cpu(), boundaries.cpu())
        cached_c, cached_b = self._codebook_cache[key]
        c = cached_c.to(device)
        b = cached_b.to(device)
        # Ensure we never return a reference to the cached tensor
        if c.data_ptr() == cached_c.data_ptr():
            c = c.clone()
        if b.data_ptr() == cached_b.data_ptr():
            b = b.clone()
        return c, b

    def _quantize_single_pass(
        self, w: Tensor, bit_width: int, group_size: int, seed: int
    ) -> tuple[QuantizedTensor, Tensor]:
        """Single-pass quantization. Returns quantized tensor and reconstruction."""
        original_shape = w.shape
        compute_dtype = getattr(torch, self.config.compute_dtype)
        n_rows, n_cols = original_shape

        w = w.to(compute_dtype)
        n_full_groups = n_cols // group_size
        remainder_cols = n_cols % group_size

        # Split into quantizable full groups and unquantized remainder
        quantizable_cols = n_full_groups * group_size

        if n_full_groups > 0:
            w_groups = w[:, :quantizable_cols]
            groups = w_groups.view(n_rows, n_full_groups, group_size).reshape(
                -1, group_size
            )

            # Step 1: Extract and store norms (float32 for precision)
            norms = torch.norm(groups, dim=-1, keepdim=True)
            norms = torch.clamp(norms, min=1e-10)
            normalized = groups / norms

            # Step 2: Rotate
            rotated = rotate(
                normalized, method=self.config.rotation_method, seed=seed
            )

            # Step 3: Scalar quantize using Lloyd-Max codebook
            centroids, boundaries = self._get_codebook(
                group_size, bit_width, w.device
            )
            indices = quantize_to_indices(rotated, boundaries)

            # Step 4: Pack indices
            packed = pack_indices(indices.flatten(), bit_width).to("cpu")

            # Compute reconstruction for residual computation
            reconstructed_rotated = dequantize_from_indices(
                indices, centroids.to(rotated.dtype)
            )
            reconstructed_normalized = rotate_inverse(
                reconstructed_rotated, method=self.config.rotation_method, seed=seed
            )
            reconstructed_groups = reconstructed_normalized * norms
            reconstructed_groups = reconstructed_groups.view(
                n_rows, n_full_groups * group_size
            )

            # Store norms as float32 for precision
            norms_stored = norms.view(n_rows, n_full_groups).to(torch.float32).cpu()
        else:
            packed = torch.empty(0, dtype=torch.uint8)
            norms_stored = torch.empty(n_rows, 0, dtype=torch.float32)
            reconstructed_groups = torch.empty(
                n_rows, 0, dtype=compute_dtype, device=w.device
            )

        # Handle remainder columns: store as-is in float16 (no quantization)
        if remainder_cols > 0:
            remainder = w[:, quantizable_cols:].to(torch.float16).cpu()
            reconstructed_remainder = w[:, quantizable_cols:]
        else:
            remainder = torch.empty(n_rows, 0, dtype=torch.float16)
            reconstructed_remainder = torch.empty(
                n_rows, 0, dtype=compute_dtype, device=w.device
            )

        # Full reconstruction for residual computation
        reconstruction = torch.cat(
            [reconstructed_groups, reconstructed_remainder], dim=1
        )

        qtensor = QuantizedTensor(
            packed_indices=packed,
            norms=norms_stored,
            shape=original_shape,
            bit_width=bit_width,
            group_size=group_size,
            n_full_groups=n_full_groups,
            remainder_cols=remainder_cols,
            remainder_weights=remainder,
        )

        return qtensor, reconstruction

    def quantize(
        self, weight: Tensor
    ) -> QuantizedTensor | tuple[QuantizedTensor, QuantizedTensor]:
        """Quantize a 2D weight tensor.

        Args:
            weight: Weight tensor of shape (out_features, in_features).

        Returns:
            Single QuantizedTensor if no residual, or tuple of
            (primary QuantizedTensor, residual QuantizedTensor) if residual_bit_width > 0.
        """
        if weight.ndim != 2:
            raise ValueError(f"Expected 2D tensor, got {weight.ndim}D")

        primary, reconstruction = self._quantize_single_pass(
            weight,
            self.config.bit_width,
            self.config.group_size,
            self.config.rotation_seed,
        )

        if self.config.residual_bit_width <= 0:
            return primary

        # Second pass: quantize the residual
        residual = weight.to(reconstruction.dtype) - reconstruction
        residual_seed = self.config.rotation_seed + 7919  # different seed
        residual_gs = self.config.residual_group_size or self.config.group_size

        residual_qt, _ = self._quantize_single_pass(
            residual, self.config.residual_bit_width, residual_gs, residual_seed
        )

        return primary, residual_qt

    def dequantize(
        self,
        qtensor: QuantizedTensor | tuple[QuantizedTensor, QuantizedTensor],
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """Dequantize back to a float tensor.

        Args:
            qtensor: QuantizedTensor or (primary, residual) tuple.
            device: Target device. Defaults to CPU.

        Returns:
            Reconstructed weight tensor of original shape, float32.
        """
        if isinstance(qtensor, tuple):
            primary, residual = qtensor
            w1 = self._dequantize_single(primary, device, self.config.rotation_seed)
            residual_seed = self.config.rotation_seed + 7919
            w2 = self._dequantize_single(residual, device, residual_seed)
            return w1 + w2
        return self._dequantize_single(qtensor, device, self.config.rotation_seed)

    def _dequantize_single(
        self, qt: QuantizedTensor, device: Optional[torch.device], seed: int
    ) -> Tensor:
        """Dequantize a single QuantizedTensor."""
        device = device or torch.device("cpu")
        compute_dtype = getattr(torch, self.config.compute_dtype)

        n_rows = qt.shape[0]

        if qt.n_full_groups > 0:
            total_quantized = n_rows * qt.n_full_groups * qt.group_size

            # Unpack indices
            indices = unpack_indices(
                qt.packed_indices.to(device), qt.bit_width, total_quantized
            )
            indices = indices.view(n_rows * qt.n_full_groups, qt.group_size)

            # Lookup centroids
            centroids, _ = self._get_codebook(qt.group_size, qt.bit_width, device)
            reconstructed = dequantize_from_indices(
                indices, centroids.to(compute_dtype)
            )

            # Inverse rotate
            reconstructed = rotate_inverse(
                reconstructed, method=self.config.rotation_method, seed=seed
            )

            # Scale by norms
            norms = qt.norms.to(device=device, dtype=compute_dtype).view(-1, 1)
            reconstructed = reconstructed * norms

            # Reshape: (n_rows * n_full_groups, group_size) -> (n_rows, quantized_cols)
            reconstructed = reconstructed.view(
                n_rows, qt.n_full_groups * qt.group_size
            )
        else:
            reconstructed = torch.empty(
                n_rows, 0, dtype=compute_dtype, device=device
            )

        # Append remainder columns
        if qt.remainder_cols > 0:
            remainder = qt.remainder_weights.to(device=device, dtype=compute_dtype)
            reconstructed = torch.cat([reconstructed, remainder], dim=1)

        return reconstructed
