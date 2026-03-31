"""TurboQuantLinear: drop-in replacement for nn.Linear with quantized weights."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import TurboQuantConfig
from .quantizer import QuantizedTensor, TurboQuantizer


class TurboQuantLinear(nn.Module):
    """Linear layer with TurboQuant-compressed weights.

    This is a drop-in replacement for nn.Linear. Weights are stored in
    bit-packed format and dequantized on-the-fly during every forward pass
    to preserve memory savings.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        config: Optional[TurboQuantConfig] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config or TurboQuantConfig()
        self._quantizer = TurboQuantizer(self.config)

        # Quantized state (set by from_linear or load_state_dict)
        self._primary_qt: Optional[QuantizedTensor] = None
        self._residual_qt: Optional[QuantizedTensor] = None

        # Register buffers for serialization
        self.register_buffer("packed_indices", torch.empty(0, dtype=torch.uint8))
        self.register_buffer("norms", torch.empty(0, dtype=torch.float32))
        self.register_buffer(
            "packed_indices_residual", torch.empty(0, dtype=torch.uint8)
        )
        self.register_buffer("norms_residual", torch.empty(0, dtype=torch.float32))
        self.register_buffer("_n_full_groups", torch.tensor(0, dtype=torch.int64))
        self.register_buffer("_remainder_cols", torch.tensor(0, dtype=torch.int64))
        self.register_buffer(
            "remainder_weights", torch.empty(0, dtype=torch.float16)
        )
        self.register_buffer(
            "_n_full_groups_residual", torch.tensor(0, dtype=torch.int64)
        )
        self.register_buffer(
            "_remainder_cols_residual", torch.tensor(0, dtype=torch.int64)
        )
        self.register_buffer(
            "remainder_weights_residual", torch.empty(0, dtype=torch.float16)
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        config: TurboQuantConfig,
    ) -> TurboQuantLinear:
        """Create a TurboQuantLinear from an existing nn.Linear layer."""
        module = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            config=config,
        )

        quantizer = TurboQuantizer(config)
        result = quantizer.quantize(linear.weight.data)

        if isinstance(result, tuple):
            primary, residual = result
            module._set_primary_buffers(primary)
            module._set_residual_buffers(residual)
        else:
            module._set_primary_buffers(result)

        if linear.bias is not None:
            module.bias = nn.Parameter(linear.bias.data.to(torch.float16))

        return module

    def _set_primary_buffers(self, qt: QuantizedTensor):
        """Copy QuantizedTensor data into registered buffers."""
        self._primary_qt = qt
        self.packed_indices = qt.packed_indices
        self.norms = qt.norms
        self._n_full_groups = torch.tensor(qt.n_full_groups, dtype=torch.int64)
        self._remainder_cols = torch.tensor(qt.remainder_cols, dtype=torch.int64)
        self.remainder_weights = qt.remainder_weights

    def _set_residual_buffers(self, qt: QuantizedTensor):
        """Copy residual QuantizedTensor data into registered buffers."""
        self._residual_qt = qt
        self.packed_indices_residual = qt.packed_indices
        self.norms_residual = qt.norms
        self._n_full_groups_residual = torch.tensor(
            qt.n_full_groups, dtype=torch.int64
        )
        self._remainder_cols_residual = torch.tensor(
            qt.remainder_cols, dtype=torch.int64
        )
        self.remainder_weights_residual = qt.remainder_weights

    def _rebuild_qt_from_buffers(self):
        """Rebuild QuantizedTensor objects from serialized buffers."""
        if self._primary_qt is not None:
            return

        # A layer is considered loaded if it has EITHER packed indices (full groups)
        # OR remainder weights (partial groups when in_features < group_size).
        has_full_groups = (
            self.packed_indices.numel() > 0 or self._n_full_groups.item() > 0
        )
        has_remainder = self.remainder_weights.numel() > 0
        if not has_full_groups and not has_remainder:
            return

        self._primary_qt = QuantizedTensor(
            packed_indices=self.packed_indices,
            norms=self.norms,
            shape=(self.out_features, self.in_features),
            bit_width=self.config.bit_width,
            group_size=self.config.group_size,
            n_full_groups=self._n_full_groups.item(),
            remainder_cols=self._remainder_cols.item(),
            remainder_weights=self.remainder_weights,
        )

        if self.packed_indices_residual.numel() > 0 or (
            self._n_full_groups_residual.item() > 0
        ):
            rgs = self.config.residual_group_size or self.config.group_size
            self._residual_qt = QuantizedTensor(
                packed_indices=self.packed_indices_residual,
                norms=self.norms_residual,
                shape=(self.out_features, self.in_features),
                bit_width=self.config.residual_bit_width,
                group_size=rgs,
                n_full_groups=self._n_full_groups_residual.item(),
                remainder_cols=self._remainder_cols_residual.item(),
                remainder_weights=self.remainder_weights_residual,
            )

    def dequantize_weight(self, device: Optional[torch.device] = None) -> Tensor:
        """Get the dequantized weight tensor."""
        self._rebuild_qt_from_buffers()

        if self._primary_qt is None:
            raise RuntimeError("No quantized weights loaded")

        if self._residual_qt is not None:
            return self._quantizer.dequantize(
                (self._primary_qt, self._residual_qt), device=device
            )
        return self._quantizer.dequantize(self._primary_qt, device=device)

    def forward(self, x: Tensor) -> Tensor:
        # Dequantize weights on every forward pass to preserve memory savings.
        # No caching: the whole point of quantization is reduced memory.
        w = self.dequantize_weight(device=x.device).to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)

    def extra_repr(self) -> str:
        total = self.config.total_bits
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, bits={total}, "
            f"group_size={self.config.group_size}"
        )
