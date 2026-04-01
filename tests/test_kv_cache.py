"""Tests for TurboQuant KV cache compression."""

import pytest
import torch

from turboquant.kv_cache import TurboQuantKVCache


class TestTurboQuantKVCache:
    def _make_kv(self, batch=1, heads=4, seq_len=10, head_dim=64):
        k = torch.randn(batch, heads, seq_len, head_dim)
        v = torch.randn(batch, heads, seq_len, head_dim)
        return k, v

    def test_basic_update_and_get(self):
        cache = TurboQuantKVCache(key_bits=4, value_bits=4, residual_window=32)
        k, v = self._make_kv(seq_len=10)
        keys, values = cache.update(0, k, v)
        assert keys.shape == (1, 4, 10, 64)
        assert values.shape == (1, 4, 10, 64)

    def test_compression_triggers_on_overflow(self):
        cache = TurboQuantKVCache(
            key_bits=4, value_bits=4, residual_window=16
        )
        # Add 20 tokens, exceeding window of 16
        k, v = self._make_kv(seq_len=20)
        keys, values = cache.update(0, k, v)

        assert keys.shape[2] == 20
        assert values.shape[2] == 20

        # Should have compressed chunks and a trimmed window
        assert len(cache._key_chunks[0]) == 1
        assert cache._key_window[0].shape[2] == 16

    def test_incremental_updates(self):
        cache = TurboQuantKVCache(
            key_bits=4, value_bits=4, residual_window=8
        )

        # Add tokens one at a time
        for i in range(20):
            k, v = self._make_kv(seq_len=1)
            keys, values = cache.update(0, k, v)

        assert keys.shape[2] == 20
        assert cache.seq_length == 20
        assert cache._key_window[0].shape[2] <= 8

    def test_batch_compression_correctness(self):
        """Compressed+decompressed KV should be close to original."""
        cache = TurboQuantKVCache(
            key_bits=4, value_bits=4, residual_window=4
        )
        k, v = self._make_kv(seq_len=20, head_dim=64)
        keys, values = cache.update(0, k, v)

        # Recent window should be exact
        assert torch.equal(keys[:, :, -4:, :], k[:, :, -4:, :].half())

        # Compressed portion should be approximately correct
        cos_sim = torch.nn.functional.cosine_similarity(
            k[:, :, :16, :].flatten().unsqueeze(0),
            keys[:, :, :16, :].float().flatten().unsqueeze(0),
        )
        assert cos_sim.item() > 0.9

    def test_multiple_layers(self):
        cache = TurboQuantKVCache(key_bits=4, value_bits=4, residual_window=8)

        for layer in range(4):
            k, v = self._make_kv(seq_len=10)
            cache.update(layer, k, v)

        for layer in range(4):
            keys, values = cache.get(layer)
            assert keys.shape[2] == 10

    def test_clear(self):
        cache = TurboQuantKVCache(key_bits=4, value_bits=4, residual_window=8)
        k, v = self._make_kv(seq_len=10)
        cache.update(0, k, v)
        cache.clear()
        assert cache.seq_length == 0
        assert len(cache._key_chunks) == 0

    def test_head_dim_validation_hadamard(self):
        """Non-power-of-2 head_dim should raise with hadamard rotation."""
        cache = TurboQuantKVCache(
            key_bits=4, value_bits=4, rotation_method="hadamard"
        )
        k, v = self._make_kv(head_dim=65)  # not power of 2
        with pytest.raises(ValueError, match="power of 2"):
            cache.update(0, k, v)

    def test_head_dim_qr_fallback(self):
        """Non-power-of-2 head_dim should work with QR rotation."""
        cache = TurboQuantKVCache(
            key_bits=4, value_bits=4, rotation_method="qr", residual_window=4
        )
        k, v = self._make_kv(head_dim=65, seq_len=10)
        keys, values = cache.update(0, k, v)
        assert keys.shape == (1, 4, 10, 65)

    def test_per_head_dim_validation(self):
        """Each distinct head_dim must be validated, not just the first."""
        cache = TurboQuantKVCache(
            key_bits=4, value_bits=4, rotation_method="hadamard", residual_window=32
        )
        # First layer: head_dim=64 (power of 2) -- should pass
        k64, v64 = self._make_kv(head_dim=64, seq_len=5)
        cache.update(0, k64, v64)

        # Second layer: head_dim=65 (NOT power of 2) -- must still be caught
        k65, v65 = self._make_kv(head_dim=65, seq_len=5)
        with pytest.raises(ValueError, match="power of 2"):
            cache.update(1, k65, v65)

    def test_invalid_bits(self):
        with pytest.raises(ValueError):
            TurboQuantKVCache(key_bits=5)
        with pytest.raises(ValueError):
            TurboQuantKVCache(value_bits=1)

    def test_bit_packing_saves_memory(self):
        """Compressed storage should be smaller than raw int16 indices."""
        cache = TurboQuantKVCache(
            key_bits=4, value_bits=4, residual_window=4
        )
        k, v = self._make_kv(seq_len=100, head_dim=128)
        cache.update(0, k, v)

        # Check that indices are bit-packed (uint8), not int16
        chunk = cache._key_chunks[0][0]
        assert chunk.packed_indices.dtype == torch.uint8

        # 4-bit: 96 tokens * 4 heads * 128 dims = 49152 values
        # Packed: 49152 * 4 / 8 = 24576 bytes
        # Int16 would be: 49152 * 2 = 98304 bytes
        expected_packed = (96 * 4 * 128 * 4 + 7) // 8
        assert chunk.packed_indices.numel() == expected_packed
