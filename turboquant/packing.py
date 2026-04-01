"""Bit-packing utilities for storing quantized indices compactly."""

from __future__ import annotations

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# 4-bit packing (2 values per byte)
# ---------------------------------------------------------------------------

def _pack_4bit(indices: Tensor) -> Tensor:
    indices = indices.to(torch.uint8).flatten()
    # Pad to even length
    if indices.numel() % 2 != 0:
        indices = torch.cat([indices, torch.zeros(1, dtype=torch.uint8, device=indices.device)])
    lo = indices[0::2] & 0x0F
    hi = indices[1::2] & 0x0F
    return lo | (hi << 4)


def _unpack_4bit(packed: Tensor, count: int) -> Tensor:
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    interleaved = torch.stack([lo, hi], dim=-1).flatten()
    return interleaved[:count].to(torch.int32)


# ---------------------------------------------------------------------------
# 2-bit packing (4 values per byte)
# ---------------------------------------------------------------------------

def _pack_2bit(indices: Tensor) -> Tensor:
    indices = indices.to(torch.uint8).flatten()
    # Pad to multiple of 4
    pad = (4 - indices.numel() % 4) % 4
    if pad > 0:
        indices = torch.cat([indices, torch.zeros(pad, dtype=torch.uint8, device=indices.device)])
    v0 = indices[0::4] & 0x03
    v1 = indices[1::4] & 0x03
    v2 = indices[2::4] & 0x03
    v3 = indices[3::4] & 0x03
    return v0 | (v1 << 2) | (v2 << 4) | (v3 << 6)


def _unpack_2bit(packed: Tensor, count: int) -> Tensor:
    v0 = packed & 0x03
    v1 = (packed >> 2) & 0x03
    v2 = (packed >> 4) & 0x03
    v3 = (packed >> 6) & 0x03
    interleaved = torch.stack([v0, v1, v2, v3], dim=-1).flatten()
    return interleaved[:count].to(torch.int32)


# ---------------------------------------------------------------------------
# 3-bit packing (8 values per 3 bytes)
# ---------------------------------------------------------------------------

def _pack_3bit(indices: Tensor) -> Tensor:
    indices = indices.to(torch.int32).flatten()
    n = indices.numel()
    # Pad to multiple of 8
    pad = (8 - n % 8) % 8
    if pad > 0:
        indices = torch.cat([indices, torch.zeros(pad, dtype=torch.int32, device=indices.device)])

    indices = indices.view(-1, 8)  # (groups, 8)

    # Pack 8 x 3-bit values into 3 bytes (24 bits)
    # byte0: val0[2:0] | val1[2:0] << 3 | val2[1:0] << 6
    # byte1: val2[2] | val3[2:0] << 1 | val4[2:0] << 4 | val5[0] << 7
    # byte2: val5[2:1] | val6[2:0] << 2 | val7[2:0] << 5
    v = indices & 0x07  # ensure 3-bit

    byte0 = v[:, 0] | (v[:, 1] << 3) | ((v[:, 2] & 0x03) << 6)
    byte1 = ((v[:, 2] >> 2) & 0x01) | (v[:, 3] << 1) | (v[:, 4] << 4) | ((v[:, 5] & 0x01) << 7)
    byte2 = ((v[:, 5] >> 1) & 0x03) | (v[:, 6] << 2) | (v[:, 7] << 5)

    packed = torch.stack([byte0, byte1, byte2], dim=-1).to(torch.uint8).flatten()
    return packed


def _unpack_3bit(packed: Tensor, count: int) -> Tensor:
    packed = packed.to(torch.int32).view(-1, 3)  # (groups, 3 bytes)

    b0, b1, b2 = packed[:, 0], packed[:, 1], packed[:, 2]

    v0 = b0 & 0x07
    v1 = (b0 >> 3) & 0x07
    v2 = ((b0 >> 6) & 0x03) | ((b1 & 0x01) << 2)
    v3 = (b1 >> 1) & 0x07
    v4 = (b1 >> 4) & 0x07
    v5 = ((b1 >> 7) & 0x01) | ((b2 & 0x03) << 1)
    v6 = (b2 >> 2) & 0x07
    v7 = (b2 >> 5) & 0x07

    indices = torch.stack([v0, v1, v2, v3, v4, v5, v6, v7], dim=-1).flatten()
    return indices[:count].to(torch.int32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def pack_indices(indices: Tensor, bit_width: int) -> Tensor:
    """Pack quantization indices into a compact uint8 tensor.

    Args:
        indices: Integer tensor of quantization indices in [0, 2^bit_width).
        bit_width: Bits per index (2, 3, or 4).

    Returns:
        Packed uint8 tensor.
    """
    if bit_width == 4:
        return _pack_4bit(indices)
    elif bit_width == 2:
        return _pack_2bit(indices)
    elif bit_width == 3:
        return _pack_3bit(indices)
    raise ValueError(f"Unsupported bit_width: {bit_width}")


def unpack_indices(packed: Tensor, bit_width: int, count: int) -> Tensor:
    """Unpack quantization indices from a packed uint8 tensor.

    Args:
        packed: Packed uint8 tensor from pack_indices().
        bit_width: Bits per index (2, 3, or 4).
        count: Number of indices to unpack.

    Returns:
        Integer tensor of shape (count,).
    """
    if bit_width == 4:
        return _unpack_4bit(packed, count)
    elif bit_width == 2:
        return _unpack_2bit(packed, count)
    elif bit_width == 3:
        return _unpack_3bit(packed, count)
    raise ValueError(f"Unsupported bit_width: {bit_width}")
