"""TurboQuant configuration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TurboQuantConfig:
    """Configuration for TurboQuant weight quantization.

    Args:
        bit_width: Bits per weight element (2, 3, or 4).
        group_size: Number of weight elements quantized together. Must be power of 2.
        residual_bit_width: If > 0, applies a second quantization pass on the residual
            for higher quality. Total bits = bit_width + residual_bit_width.
        rotation_seed: Seed for generating the random orthogonal rotation matrix.
        rotation_method: "hadamard" (fast, O(d log d)) or "qr" (exact Haar, O(d^2)).
        modules_to_not_convert: List of module name patterns to skip during quantization.
        compute_dtype: Dtype used during quantization computation.
    """

    bit_width: int = 4
    group_size: int = 128
    residual_bit_width: int = 0
    residual_group_size: Optional[int] = None
    rotation_seed: int = 42
    rotation_method: str = "hadamard"
    modules_to_not_convert: list[str] = field(
        default_factory=lambda: ["lm_head"]
    )
    compute_dtype: str = "float32"

    def __post_init__(self):
        if self.bit_width not in (2, 3, 4):
            raise ValueError(f"bit_width must be 2, 3, or 4, got {self.bit_width}")
        if self.residual_bit_width not in (0, 2, 3, 4):
            raise ValueError(
                f"residual_bit_width must be 0, 2, 3, or 4, got {self.residual_bit_width}"
            )
        if self.group_size < 2 or (self.group_size & (self.group_size - 1)) != 0:
            raise ValueError(f"group_size must be a power of 2 >= 2, got {self.group_size}")
        if self.rotation_method not in ("hadamard", "qr"):
            raise ValueError(
                f"rotation_method must be 'hadamard' or 'qr', got {self.rotation_method}"
            )
        if self.residual_group_size is None:
            self.residual_group_size = self.group_size
        elif self.residual_group_size < 2 or (
            self.residual_group_size & (self.residual_group_size - 1)
        ) != 0:
            raise ValueError(
                f"residual_group_size must be a power of 2 >= 2, got {self.residual_group_size}"
            )
        valid_dtypes = ("float16", "bfloat16", "float32", "float64")
        if self.compute_dtype not in valid_dtypes:
            raise ValueError(
                f"compute_dtype must be one of {valid_dtypes}, got {self.compute_dtype}"
            )

    @property
    def total_bits(self) -> int:
        return self.bit_width + self.residual_bit_width

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> TurboQuantConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def load(cls, path: str | Path) -> TurboQuantConfig:
        with open(path) as f:
            return cls.from_dict(json.load(f))
