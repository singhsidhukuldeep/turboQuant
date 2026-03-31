"""Tests for TurboQuantLinear module."""

import torch
import torch.nn as nn
import pytest

from turboquant.config import TurboQuantConfig
from turboquant.module import TurboQuantLinear


class TestTurboQuantLinear:
    def test_from_linear(self):
        linear = nn.Linear(128, 64)
        config = TurboQuantConfig(bit_width=4, group_size=128)
        tql = TurboQuantLinear.from_linear(linear, config)

        assert tql.in_features == 128
        assert tql.out_features == 64
        assert tql.packed_indices.numel() > 0

    def test_forward_shape(self):
        linear = nn.Linear(128, 64)
        config = TurboQuantConfig(bit_width=4, group_size=128)
        tql = TurboQuantLinear.from_linear(linear, config)

        x = torch.randn(2, 10, 128)
        y = tql(x)
        assert y.shape == (2, 10, 64)

    def test_forward_with_bias(self):
        linear = nn.Linear(128, 64, bias=True)
        config = TurboQuantConfig(bit_width=4, group_size=128)
        tql = TurboQuantLinear.from_linear(linear, config)

        x = torch.randn(4, 128)
        y = tql(x)
        assert y.shape == (4, 64)
        assert tql.bias is not None

    def test_forward_without_bias(self):
        linear = nn.Linear(128, 64, bias=False)
        config = TurboQuantConfig(bit_width=4, group_size=128)
        tql = TurboQuantLinear.from_linear(linear, config)

        x = torch.randn(4, 128)
        y = tql(x)
        assert y.shape == (4, 64)
        assert tql.bias is None

    def test_output_quality(self):
        """Quantized output should be somewhat close to original."""
        linear = nn.Linear(128, 64)
        config = TurboQuantConfig(bit_width=4, group_size=128)
        tql = TurboQuantLinear.from_linear(linear, config)

        x = torch.randn(8, 128)
        y_original = linear(x)
        y_quantized = tql(x.float())

        cos_sim = torch.nn.functional.cosine_similarity(
            y_original.flatten().unsqueeze(0).float(),
            y_quantized.flatten().unsqueeze(0).float(),
        )
        assert cos_sim.item() > 0.85

    def test_no_permanent_weight_cache(self):
        """Verify that forward does not permanently cache dequantized weights."""
        linear = nn.Linear(128, 64)
        config = TurboQuantConfig(bit_width=4, group_size=128)
        tql = TurboQuantLinear.from_linear(linear, config)

        x = torch.randn(4, 128)
        _ = tql(x)

        # No _weight_cache attribute should exist
        assert not hasattr(tql, "_weight_cache")

    def test_state_dict_serialization(self, tmp_path):
        linear = nn.Linear(128, 64)
        config = TurboQuantConfig(bit_width=4, group_size=128)
        tql = TurboQuantLinear.from_linear(linear, config)

        state = tql.state_dict()
        torch.save(state, tmp_path / "tql.pt")

        tql2 = TurboQuantLinear(128, 64, bias=True, config=config)
        state_loaded = torch.load(tmp_path / "tql.pt", weights_only=True)
        tql2.load_state_dict(state_loaded)

        x = torch.randn(4, 128)
        y1 = tql(x)
        y2 = tql2(x)
        assert torch.allclose(y1, y2, atol=1e-5)

    def test_state_dict_serialization_remainder_only(self, tmp_path):
        """Layers where in_features < group_size must survive save/load."""
        linear = nn.Linear(50, 32)  # 50 < 128, so 0 full groups
        config = TurboQuantConfig(bit_width=4, group_size=128)
        tql = TurboQuantLinear.from_linear(linear, config)

        state = tql.state_dict()
        torch.save(state, tmp_path / "tql_rem.pt")

        tql2 = TurboQuantLinear(50, 32, bias=True, config=config)
        state_loaded = torch.load(tmp_path / "tql_rem.pt", weights_only=True)
        tql2.load_state_dict(state_loaded)

        x = torch.randn(4, 50)
        y1 = tql(x)
        y2 = tql2(x)
        assert torch.allclose(y1, y2, atol=1e-5)

    def test_residual_quantization(self):
        linear = nn.Linear(128, 64)
        config = TurboQuantConfig(bit_width=4, residual_bit_width=4, group_size=128)
        tql = TurboQuantLinear.from_linear(linear, config)

        x = torch.randn(4, 128)
        y = tql(x)
        assert y.shape == (4, 64)
        assert tql.packed_indices_residual.numel() > 0

    def test_norms_are_float32(self):
        """Norms should be stored as float32 for precision."""
        linear = nn.Linear(128, 64)
        config = TurboQuantConfig(bit_width=4, group_size=128)
        tql = TurboQuantLinear.from_linear(linear, config)
        assert tql.norms.dtype == torch.float32

    def test_remainder_columns(self):
        """Linear layers with non-divisible dimensions should work."""
        linear = nn.Linear(100, 64)  # 100 not divisible by 128
        config = TurboQuantConfig(bit_width=4, group_size=128)
        tql = TurboQuantLinear.from_linear(linear, config)

        x = torch.randn(4, 100)
        y = tql(x)
        assert y.shape == (4, 64)

    def test_extra_repr(self):
        linear = nn.Linear(128, 64)
        config = TurboQuantConfig(bit_width=4, group_size=128)
        tql = TurboQuantLinear.from_linear(linear, config)
        repr_str = tql.extra_repr()
        assert "128" in repr_str
        assert "64" in repr_str
        assert "bits=4" in repr_str
