"""Tests for the core TurboQuant quantizer."""

import torch
import pytest

from turboquant.config import TurboQuantConfig
from turboquant.quantizer import TurboQuantizer


class TestTurboQuantizer:
    def test_quantize_dequantize_shape(self):
        config = TurboQuantConfig(bit_width=4, group_size=64)
        quantizer = TurboQuantizer(config)

        w = torch.randn(128, 256)
        qt = quantizer.quantize(w)
        w_rec = quantizer.dequantize(qt)

        assert w_rec.shape == w.shape

    def test_quantize_dequantize_quality(self):
        """4-bit quantization should have reasonable reconstruction quality."""
        config = TurboQuantConfig(bit_width=4, group_size=128)
        quantizer = TurboQuantizer(config)

        w = torch.randn(64, 128) * 0.1
        qt = quantizer.quantize(w)
        w_rec = quantizer.dequantize(qt).float()

        cos_sim = torch.nn.functional.cosine_similarity(
            w.flatten().unsqueeze(0), w_rec.flatten().unsqueeze(0)
        )
        assert cos_sim.item() > 0.9

    def test_higher_bits_better_quality(self):
        w = torch.randn(64, 128) * 0.1

        mse_by_bits = {}
        for bits in [2, 3, 4]:
            config = TurboQuantConfig(bit_width=bits, group_size=128)
            quantizer = TurboQuantizer(config)
            qt = quantizer.quantize(w)
            w_rec = quantizer.dequantize(qt).float()
            mse = ((w - w_rec) ** 2).mean().item()
            mse_by_bits[bits] = mse

        assert mse_by_bits[4] < mse_by_bits[3] < mse_by_bits[2]

    def test_residual_quantization(self):
        config = TurboQuantConfig(bit_width=4, residual_bit_width=4, group_size=64)
        quantizer = TurboQuantizer(config)

        w = torch.randn(64, 128) * 0.1
        result = quantizer.quantize(w)
        assert isinstance(result, tuple)
        assert len(result) == 2

        w_rec = quantizer.dequantize(result).float()
        assert w_rec.shape == w.shape

    def test_residual_improves_quality(self):
        w = torch.randn(64, 128) * 0.1

        config1 = TurboQuantConfig(bit_width=4, group_size=128)
        q1 = TurboQuantizer(config1)
        qt1 = q1.quantize(w)
        mse1 = ((w - q1.dequantize(qt1).float()) ** 2).mean().item()

        config2 = TurboQuantConfig(bit_width=4, residual_bit_width=4, group_size=128)
        q2 = TurboQuantizer(config2)
        qt2 = q2.quantize(w)
        mse2 = ((w - q2.dequantize(qt2).float()) ** 2).mean().item()

        assert mse2 < mse1

    def test_remainder_columns_preserved(self):
        """Columns not divisible by group_size are stored unquantized."""
        config = TurboQuantConfig(bit_width=4, group_size=128)
        quantizer = TurboQuantizer(config)

        w = torch.randn(32, 100)  # 100 is not a multiple of 128
        qt = quantizer.quantize(w)
        w_rec = quantizer.dequantize(qt)
        assert w_rec.shape == (32, 100)

        # Remainder columns (last 100 cols since 100 < 128) should be exact
        # Since 100 < 128, n_full_groups=0 and everything is stored as remainder
        assert qt.n_full_groups == 0
        assert qt.remainder_cols == 100

    def test_partial_group_handled_correctly(self):
        """When in_features has a partial group, it should not be zero-padded."""
        config = TurboQuantConfig(bit_width=4, group_size=64)
        quantizer = TurboQuantizer(config)

        # 200 = 3 full groups of 64 + 8 remainder
        w = torch.randn(16, 200)
        qt = quantizer.quantize(w)
        assert qt.n_full_groups == 3
        assert qt.remainder_cols == 8
        assert qt.remainder_weights.shape == (16, 8)

        w_rec = quantizer.dequantize(qt)
        assert w_rec.shape == (16, 200)

        # The remainder columns should be losslessly preserved (stored as float16)
        remainder_original = w[:, 192:].to(torch.float16).float()
        remainder_recovered = w_rec[:, 192:].float()
        assert torch.allclose(remainder_original, remainder_recovered, atol=1e-3)

    def test_norms_stored_as_float32(self):
        """Norms should be float32 for precision."""
        config = TurboQuantConfig(bit_width=4, group_size=128)
        quantizer = TurboQuantizer(config)

        w = torch.randn(8, 128) * 0.001  # very small weights
        qt = quantizer.quantize(w)
        assert qt.norms.dtype == torch.float32


class TestTurboQuantConfig:
    def test_valid_configs(self):
        TurboQuantConfig(bit_width=2, group_size=64)
        TurboQuantConfig(bit_width=3, group_size=128)
        TurboQuantConfig(bit_width=4, group_size=256)

    def test_invalid_bit_width(self):
        with pytest.raises(ValueError):
            TurboQuantConfig(bit_width=5)

    def test_invalid_group_size(self):
        with pytest.raises(ValueError):
            TurboQuantConfig(group_size=100)

    def test_invalid_residual_group_size(self):
        with pytest.raises(ValueError):
            TurboQuantConfig(bit_width=4, residual_group_size=7)

    def test_invalid_compute_dtype(self):
        with pytest.raises(ValueError):
            TurboQuantConfig(compute_dtype="float16")

    def test_serialization(self, tmp_path):
        config = TurboQuantConfig(bit_width=3, group_size=64, rotation_seed=123)
        path = tmp_path / "config.json"
        config.save(path)
        loaded = TurboQuantConfig.load(path)
        assert loaded.bit_width == 3
        assert loaded.group_size == 64
        assert loaded.rotation_seed == 123

    def test_total_bits(self):
        config = TurboQuantConfig(bit_width=4, residual_bit_width=2)
        assert config.total_bits == 6
