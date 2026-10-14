from __future__ import annotations

import pytest
import torch

from engine.perf.cuda_architecture_b_child_policy import (
    CROSSOVER_MASK_ENCODING_V1,
    pack_crossover_mask_lsb0_v1,
    packed_crossover_row_bytes_v1,
    unpack_crossover_mask_lsb0_v1,
)


def test_policy_crossover_mask_encoding_id_is_canonical() -> None:
    assert CROSSOVER_MASK_ENCODING_V1 == "bitpack_lsb0_v1"


@pytest.mark.parametrize(
    "width",
    [1, 7, 8, 9, 31, 32, 33, 63, 64, 65, 9415],
)
def test_policy_crossover_mask_bitpack_round_trips_width_and_padding(
    width: int,
) -> None:
    mask = (torch.arange(width, dtype=torch.int64) % 3) == 0
    packed = pack_crossover_mask_lsb0_v1(mask)

    assert packed.dtype == torch.uint8
    assert tuple(packed.shape) == (packed_crossover_row_bytes_v1(width),)
    assert torch.equal(unpack_crossover_mask_lsb0_v1(packed, logical_width=width), mask)

    used_bits = width % 8
    if used_bits:
        padding_mask = (~((1 << used_bits) - 1)) & 0xFF
        assert int(packed[-1].item()) & padding_mask == 0


def test_policy_crossover_mask_bitpack_uses_lsb0_order() -> None:
    packed = pack_crossover_mask_lsb0_v1(
        torch.tensor(
            [True, False, True, True, False, False, True, False, True],
            dtype=torch.bool,
        )
    )

    assert packed.tolist() == [0b01001101, 0b00000001]


def test_policy_crossover_mask_bitpack_rejects_wrong_schema() -> None:
    with pytest.raises(ValueError, match="bool tensor"):
        pack_crossover_mask_lsb0_v1(torch.zeros((4,), dtype=torch.uint8))
    with pytest.raises(ValueError, match="rank 1 or rank 2"):
        pack_crossover_mask_lsb0_v1(torch.zeros((1, 1, 1), dtype=torch.bool))
    with pytest.raises(ValueError, match="uint8 tensor"):
        unpack_crossover_mask_lsb0_v1(torch.zeros((1,), dtype=torch.bool), logical_width=1)
    with pytest.raises(ValueError, match="row width"):
        unpack_crossover_mask_lsb0_v1(torch.zeros((2,), dtype=torch.uint8), logical_width=1)
