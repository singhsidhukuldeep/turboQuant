"""Tests for bit-packing utilities."""

import pytest
import torch

from turboquant.packing import pack_indices, unpack_indices


class TestPacking:
    @pytest.mark.parametrize("bit_width", [2, 3, 4])
    def test_round_trip(self, bit_width):
        n_levels = 1 << bit_width
        indices = torch.randint(0, n_levels, (256,), dtype=torch.int32)
        packed = pack_indices(indices, bit_width)
        unpacked = unpack_indices(packed, bit_width, 256)
        assert torch.equal(indices, unpacked)

    @pytest.mark.parametrize("bit_width", [2, 3, 4])
    def test_odd_count(self, bit_width):
        n_levels = 1 << bit_width
        count = 137  # prime number, won't divide evenly
        indices = torch.randint(0, n_levels, (count,), dtype=torch.int32)
        packed = pack_indices(indices, bit_width)
        unpacked = unpack_indices(packed, bit_width, count)
        assert torch.equal(indices, unpacked)

    def test_4bit_packing_size(self):
        indices = torch.zeros(100, dtype=torch.int32)
        packed = pack_indices(indices, 4)
        assert packed.shape[0] == 50  # 2 values per byte

    def test_2bit_packing_size(self):
        indices = torch.zeros(100, dtype=torch.int32)
        packed = pack_indices(indices, 2)
        assert packed.shape[0] == 25  # 4 values per byte

    def test_3bit_packing_size(self):
        indices = torch.zeros(128, dtype=torch.int32)
        packed = pack_indices(indices, 3)
        # 128 values * 3 bits = 384 bits = 48 bytes
        assert packed.shape[0] == 48

    def test_all_max_values(self):
        for bit_width in [2, 3, 4]:
            max_val = (1 << bit_width) - 1
            indices = torch.full((64,), max_val, dtype=torch.int32)
            packed = pack_indices(indices, bit_width)
            unpacked = unpack_indices(packed, bit_width, 64)
            assert torch.equal(indices, unpacked)

    def test_invalid_bit_width(self):
        indices = torch.zeros(10, dtype=torch.int32)
        with pytest.raises(ValueError):
            pack_indices(indices, 5)
