"""TurboQuant KV cache compression for inference.

Compresses key-value cache tensors during generation, reducing memory
usage for long-context inference. Uses proper bit-packing for real
compression savings.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .codebook import dequantize_from_indices, get_codebook, quantize_to_indices
from .packing import pack_indices, unpack_indices
from .rotation import rotate, rotate_inverse


@dataclass
class _CompressedChunk:
    """A batch of compressed KV vectors."""

    packed_indices: Tensor  # uint8, bit-packed
    norms: Tensor  # float32, shape (batch, heads, seq_len, 1)
    seq_len: int
    head_dim: int
    n_bits: int


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


class TurboQuantKVCache:
    """Compressed KV cache using TurboQuant.

    Quantizes key and value tensors independently with configurable
    bit-widths and proper bit-packing. Recent tokens are kept in full
    precision for stability.

    Args:
        key_bits: Bit-width for key quantization (2, 3, or 4).
        value_bits: Bit-width for value quantization (2, 3, or 4).
        residual_window: Number of recent tokens kept in full precision.
        rotation_seed: Seed for rotation matrix.
        rotation_method: "hadamard" or "qr". If "hadamard", head_dim must
            be a power of 2.
    """

    def __init__(
        self,
        key_bits: int = 4,
        value_bits: int = 4,
        residual_window: int = 128,
        rotation_seed: int = 42,
        rotation_method: str = "hadamard",
    ):
        if key_bits not in (2, 3, 4):
            raise ValueError(f"key_bits must be 2, 3, or 4, got {key_bits}")
        if value_bits not in (2, 3, 4):
            raise ValueError(f"value_bits must be 2, 3, or 4, got {value_bits}")
        if residual_window < 1:
            raise ValueError(
                f"residual_window must be >= 1, got {residual_window}"
            )

        self.key_bits = key_bits
        self.value_bits = value_bits
        self.residual_window = residual_window
        self.rotation_seed = rotation_seed
        self.rotation_method = rotation_method

        # Per-layer compressed storage
        self._key_chunks: dict[int, list[_CompressedChunk]] = {}
        self._value_chunks: dict[int, list[_CompressedChunk]] = {}

        # Full-precision window for recent tokens
        self._key_window: dict[int, Tensor] = {}
        self._value_window: dict[int, Tensor] = {}

        self._seq_len: dict[int, int] = {}
        self._validated_head_dims: set[int] = set()

    def _validate_head_dim(self, head_dim: int):
        """Validate head_dim is compatible with the rotation method."""
        if head_dim in self._validated_head_dims:
            return
        if self.rotation_method == "hadamard" and not _is_power_of_two(head_dim):
            raise ValueError(
                f"head_dim={head_dim} is not a power of 2, which is required "
                f"for rotation_method='hadamard'. Use rotation_method='qr' instead."
            )
        self._validated_head_dims.add(head_dim)

    def _compress_batch(
        self, x: Tensor, n_bits: int
    ) -> _CompressedChunk:
        """Compress a batch of KV vectors with proper bit-packing.

        Args:
            x: Shape (batch, heads, seq_len, head_dim).
            n_bits: Quantization bits.

        Returns:
            CompressedChunk with bit-packed indices and float32 norms.
        """
        head_dim = x.shape[-1]
        batch, heads, seq_len, _ = x.shape

        # Reshape to (batch * heads * seq_len, head_dim)
        flat = x.reshape(-1, head_dim).float()

        # Normalize
        norms = torch.norm(flat, dim=-1, keepdim=True)
        norms = torch.clamp(norms, min=1e-10)
        normalized = flat / norms

        # Rotate
        rotated = rotate(
            normalized, method=self.rotation_method, seed=self.rotation_seed
        )

        # Quantize
        centroids, boundaries = get_codebook(head_dim, n_bits, device=x.device)
        indices = quantize_to_indices(rotated, boundaries)

        # Bit-pack the indices
        packed = pack_indices(indices.flatten(), n_bits).cpu()

        # Store norms efficiently as float32
        norms_stored = norms.view(batch, heads, seq_len, 1).to(torch.float32).cpu()

        return _CompressedChunk(
            packed_indices=packed,
            norms=norms_stored,
            seq_len=seq_len,
            head_dim=head_dim,
            n_bits=n_bits,
        )

    def _decompress_batch(
        self, chunk: _CompressedChunk, device: torch.device, target_shape: tuple
    ) -> Tensor:
        """Decompress a chunk back to float tensors.

        Args:
            chunk: CompressedChunk to decompress.
            device: Target device.
            target_shape: (batch, heads, seq_len, head_dim).

        Returns:
            Decompressed tensor of target_shape, float16.
        """
        batch, heads, seq_len, head_dim = target_shape
        total_vectors = batch * heads * seq_len
        total_elements = total_vectors * head_dim

        # Unpack indices
        indices = unpack_indices(
            chunk.packed_indices.to(device), chunk.n_bits, total_elements
        )
        indices = indices.view(total_vectors, head_dim)

        # Lookup centroids
        centroids, _ = get_codebook(head_dim, chunk.n_bits, device=device)
        reconstructed = dequantize_from_indices(indices.long(), centroids)

        # Inverse rotate
        reconstructed = rotate_inverse(
            reconstructed, method=self.rotation_method, seed=self.rotation_seed
        )

        # Scale by norms
        norms = chunk.norms.to(device=device, dtype=torch.float32).view(-1, 1)
        reconstructed = reconstructed * norms

        return reconstructed.view(target_shape).half()

    def update(
        self,
        layer_idx: int,
        key_states: Tensor,
        value_states: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Add new KV states to the cache.

        Args:
            layer_idx: Transformer layer index.
            key_states: Shape (batch, heads, new_seq_len, head_dim).
            value_states: Same shape as key_states.

        Returns:
            Full key and value tensors for attention computation.
        """
        head_dim = key_states.shape[-1]
        self._validate_head_dim(head_dim)
        self._seq_len.setdefault(layer_idx, 0)

        if layer_idx not in self._key_window:
            self._key_chunks[layer_idx] = []
            self._value_chunks[layer_idx] = []
            self._key_window[layer_idx] = key_states
            self._value_window[layer_idx] = value_states
        else:
            self._key_window[layer_idx] = torch.cat(
                [self._key_window[layer_idx], key_states], dim=2
            )
            self._value_window[layer_idx] = torch.cat(
                [self._value_window[layer_idx], value_states], dim=2
            )

        self._seq_len[layer_idx] += key_states.shape[2]

        # If window exceeds limit, compress all overflow tokens at once
        window_len = self._key_window[layer_idx].shape[2]
        if window_len > self.residual_window:
            overflow = window_len - self.residual_window

            # Batch compress all overflow tokens at once
            to_compress_k = self._key_window[layer_idx][:, :, :overflow, :]
            to_compress_v = self._value_window[layer_idx][:, :, :overflow, :]

            k_chunk = self._compress_batch(to_compress_k, self.key_bits)
            v_chunk = self._compress_batch(to_compress_v, self.value_bits)

            self._key_chunks[layer_idx].append(k_chunk)
            self._value_chunks[layer_idx].append(v_chunk)

            # Trim window to just the recent tokens
            self._key_window[layer_idx] = (
                self._key_window[layer_idx][:, :, overflow:, :]
            )
            self._value_window[layer_idx] = (
                self._value_window[layer_idx][:, :, overflow:, :]
            )

        return self.get(layer_idx)

    def get(self, layer_idx: int) -> tuple[Tensor, Tensor]:
        """Get the full key and value tensors for a layer.

        Returns:
            keys: Shape (batch, heads, total_seq_len, head_dim).
            values: Same shape.
        """
        window_k = self._key_window[layer_idx]
        window_v = self._value_window[layer_idx]
        device = window_k.device
        batch, heads, _, head_dim = window_k.shape

        parts_k: list[Tensor] = []
        parts_v: list[Tensor] = []

        for chunk in self._key_chunks[layer_idx]:
            shape = (batch, heads, chunk.seq_len, head_dim)
            parts_k.append(self._decompress_batch(chunk, device, shape))

        for chunk in self._value_chunks[layer_idx]:
            shape = (batch, heads, chunk.seq_len, head_dim)
            parts_v.append(self._decompress_batch(chunk, device, shape))

        parts_k.append(window_k)
        parts_v.append(window_v)

        keys = torch.cat(parts_k, dim=2) if len(parts_k) > 1 else parts_k[0]
        values = torch.cat(parts_v, dim=2) if len(parts_v) > 1 else parts_v[0]

        return keys, values

    @property
    def seq_length(self) -> int:
        """Total sequence length across all layers (from layer 0)."""
        return self._seq_len.get(0, 0)

    def clear(self):
        """Clear all cached data."""
        self._key_chunks.clear()
        self._value_chunks.clear()
        self._key_window.clear()
        self._value_window.clear()
        self._seq_len.clear()
        self._validated_head_dims.clear()
