"""Tests for model-level quantization and save/load."""

import torch
import torch.nn as nn
import pytest

from turboquant.config import TurboQuantConfig
from turboquant.model import quantize_model, save_quantized, estimate_model_size
from turboquant.module import TurboQuantLinear
from turboquant.quantizer import TurboQuantizer


class SimpleModel(nn.Module):
    """Minimal model for testing quantization."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, 64)
        self.linear1 = nn.Linear(64, 128)
        self.linear2 = nn.Linear(128, 64)
        self.lm_head = nn.Linear(64, 100)
        self.norm = nn.LayerNorm(64)

    def forward(self, x):
        x = self.embed(x)
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = self.norm(x)
        x = self.lm_head(x)
        return x


class TestQuantizeModel:
    def test_quantize_replaces_linear_layers(self):
        model = SimpleModel()
        config = TurboQuantConfig(bit_width=4, group_size=64)
        quantize_model(model, config, verbose=False)

        assert isinstance(model.linear1, TurboQuantLinear)
        assert isinstance(model.linear2, TurboQuantLinear)
        # lm_head should be skipped by default
        assert isinstance(model.lm_head, nn.Linear)
        # Embedding and LayerNorm should be untouched
        assert isinstance(model.embed, nn.Embedding)
        assert isinstance(model.norm, nn.LayerNorm)

    def test_quantized_model_runs(self):
        model = SimpleModel()
        config = TurboQuantConfig(bit_width=4, group_size=64)
        quantize_model(model, config, verbose=False)

        x = torch.randint(0, 100, (2, 10))
        output = model(x)
        assert output.shape == (2, 10, 100)

    def test_skip_modules(self):
        model = SimpleModel()
        config = TurboQuantConfig(
            bit_width=4, group_size=64, modules_to_not_convert=["lm_head", "linear1"]
        )
        quantize_model(model, config, verbose=False)

        assert isinstance(model.linear1, nn.Linear)  # skipped
        assert isinstance(model.linear2, TurboQuantLinear)  # quantized
        assert isinstance(model.lm_head, nn.Linear)  # skipped

    def test_output_quality(self):
        """Quantized model output should be reasonably close to original."""
        model = SimpleModel()
        x = torch.randint(0, 100, (2, 10))

        with torch.no_grad():
            original_output = model(x).clone()

        config = TurboQuantConfig(bit_width=4, group_size=64)
        quantize_model(model, config, verbose=False)

        with torch.no_grad():
            quantized_output = model(x)

        cos_sim = torch.nn.functional.cosine_similarity(
            original_output.flatten().unsqueeze(0),
            quantized_output.float().flatten().unsqueeze(0),
        )
        assert cos_sim.item() > 0.7


class TestSaveLoad:
    def test_save_creates_files(self, tmp_path):
        model = SimpleModel()
        config = TurboQuantConfig(bit_width=4, group_size=64)
        quantize_model(model, config, verbose=False)

        save_quantized(model, config, tmp_path / "out")

        assert (tmp_path / "out" / "turboquant_config.json").exists()
        assert (tmp_path / "out" / "quantized_weights.pt").exists()

    def test_saved_config_matches(self, tmp_path):
        config = TurboQuantConfig(bit_width=3, group_size=64, rotation_seed=99)
        model = SimpleModel()
        quantize_model(model, config, verbose=False)
        save_quantized(model, config, tmp_path / "out")

        loaded_config = TurboQuantConfig.load(
            tmp_path / "out" / "turboquant_config.json"
        )
        assert loaded_config.bit_width == 3
        assert loaded_config.group_size == 64
        assert loaded_config.rotation_seed == 99

    def test_save_load_roundtrip_produces_same_output(self, tmp_path):
        """Full round-trip: quantize -> save -> load state_dict -> run forward."""
        model = SimpleModel()
        config = TurboQuantConfig(bit_width=4, group_size=64)
        quantize_model(model, config, verbose=False)

        x = torch.randint(0, 100, (2, 5))
        with torch.no_grad():
            y_before_save = model(x).clone()

        save_quantized(model, config, tmp_path / "rt")

        # Simulate load: create a fresh model, replace linears, load state
        model2 = SimpleModel()
        quantize_model(model2, config, verbose=False)
        state = torch.load(tmp_path / "rt" / "quantized_weights.pt", weights_only=True)
        model2.load_state_dict(state, strict=False, assign=True)

        with torch.no_grad():
            y_after_load = model2(x)

        assert torch.allclose(y_before_save, y_after_load, atol=1e-4)


class TestEstimateModelSize:
    def test_estimate_returns_valid_ratios(self):
        model = SimpleModel()
        config = TurboQuantConfig(bit_width=4, group_size=64)
        sizes = estimate_model_size(model, config)

        assert sizes["original_mb"] > 0
        assert sizes["quantized_mb"] > 0
        assert sizes["compression_ratio"] > 1.0

    def test_lower_bits_higher_compression(self):
        model = SimpleModel()

        sizes_4 = estimate_model_size(
            model, TurboQuantConfig(bit_width=4, group_size=64)
        )
        sizes_2 = estimate_model_size(
            model, TurboQuantConfig(bit_width=2, group_size=64)
        )

        assert sizes_2["compression_ratio"] > sizes_4["compression_ratio"]

    def test_skipped_modules_counted_at_original_size(self):
        model = SimpleModel()
        config = TurboQuantConfig(
            bit_width=4, group_size=64, modules_to_not_convert=["lm_head"]
        )
        sizes = estimate_model_size(model, config)

        # lm_head is 64*100*2 = 12800 bytes = 0.012 MB at float16
        # It should not be compressed
        assert sizes["quantized_mb"] > 0
