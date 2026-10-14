from __future__ import annotations

from pathlib import Path

import pytest
import torch

from engine.perf.cuda_architecture_b_hapb_v1 import (
    CELL_COUNT_V1,
    HAPBContractError,
    POLICY_WIDTH_V1,
    SLOT_CAPACITY_V1,
    _content_row_map_from_canonical_v1,
    canonical_host_policy_bank_from_cell_rows_v1,
    hapb_canonical_bank_shape_v1,
    hapb_physical_cell_count_v1,
    restore_hapb_policy_bank_checkpoint_v1,
    stable_overflow_row_map_v1,
)
from engine.perf.cuda_architecture_b_hapb_v1 import (
    export_hapb_policy_bank_checkpoint_v1,
)


def _cell_row(seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(
        (SLOT_CAPACITY_V1, POLICY_WIDTH_V1),
        dtype=torch.float32,
        generator=generator,
    ).contiguous()


def test_shared_canonical_bank_adopts_b60_row_views() -> None:
    from engine.perf.cuda_architecture_b_direct_resident_source_allocator_v1 import (
        _shared_hapb_canonical_bank_from_rows_v1,
    )

    bank = torch.zeros((60, SLOT_CAPACITY_V1, POLICY_WIDTH_V1), dtype=torch.float32)
    adopted = _shared_hapb_canonical_bank_from_rows_v1(tuple(bank[cell] for cell in range(60)))
    assert adopted is not None
    assert tuple(adopted.shape) == (60, SLOT_CAPACITY_V1, POLICY_WIDTH_V1)
    assert adopted.data_ptr() == bank.data_ptr()


def test_h4_single_generation_profiler_path_uses_official_g3_execute() -> None:
    text = (
        Path(__file__).resolve().parents[1]
        / "tools/scripts/eiel_b48_wdl_jungle_eed_hapb_h4_continuation_v1.py"
    ).read_text()
    assert "--single-diagnostic-end-generation" in text
    assert "single_g0_to_g{diagnostic_end}_started" in text
    assert "HAPB single generation diagnostic must be 1 (residual-wall) " in text
    assert "or the qualified G3 minimum" in text
    assert "emit_checkpoint=False" in text


def test_hapb_prepare_source_admits_physical_b96_batch_cells() -> None:
    text = (
        Path(__file__).resolve().parents[1]
        / "engine/perf/cuda_architecture_b_direct_resident_preparation_v1.py"
    ).read_text()
    assert 'raise _fail("batch_cells must be in [1,256]")' in text
    assert "batch_cells > 256" in text
    assert "energy_shape[0] <= 256" in text


def test_hapb_source_allocator_admits_physical_b96_batch_cells() -> None:
    text = (
        Path(__file__).resolve().parents[1]
        / "engine/perf/cuda_architecture_b_direct_resident_source_allocator_v1.py"
    ).read_text()
    assert 'raise _fail("batch_cells must be an exact integer in [1,256]")' in text
    assert text.count("batch_cells > 256") >= 1


def test_hapb_wdl_prepare_admits_physical_b96_request_count() -> None:
    text = (
        Path(__file__).resolve().parents[1]
        / "engine/perf/cuda_architecture_b_direct_resident_wdl_v2.py"
    ).read_text()
    assert "admitted_max = 256" in text
    assert "admitted_max = 256 if hapb_host_policy_enable else 72" not in text
    assert "B1..B{admitted_max}" in text


def test_hapb_native_binding_admits_physical_b_to_256() -> None:
    from engine.campaigns.production_engine_native_bindings import (
        wdl_jungle_eed_hapb_fixed_mlp_native_binding_v1,
    )

    binding = wdl_jungle_eed_hapb_fixed_mlp_native_binding_v1()
    assert binding.batch_max == 256
    assert 60 <= binding.batch_max


def test_hapb_physical_cell_count_admits_non_multiple_of_eight() -> None:
    assert hapb_physical_cell_count_v1(60) == 60
    assert 60 % 8 != 0
    with pytest.raises(HAPBContractError, match=r"\[1, 256\]"):
        hapb_physical_cell_count_v1(0)
    with pytest.raises(HAPBContractError, match=r"\[1, 256\]"):
        hapb_physical_cell_count_v1(257)


def test_hapb_canonical_factory_b48_matches_oracle_extent() -> None:
    rows = tuple(_cell_row(1000 + cell) for cell in range(CELL_COUNT_V1))
    bank = canonical_host_policy_bank_from_cell_rows_v1(rows)
    assert tuple(bank.shape) == hapb_canonical_bank_shape_v1(CELL_COUNT_V1)
    assert tuple(bank.shape[1:]) == (SLOT_CAPACITY_V1, POLICY_WIDTH_V1)
    for cell, row in enumerate(rows):
        assert torch.equal(bank[cell], row)
    alias = _content_row_map_from_canonical_v1(bank)
    assert tuple(alias.shape) == (CELL_COUNT_V1, SLOT_CAPACITY_V1)
    assert int(alias[0, 0].item()) == 0
    assert int(alias[1, 0].item()) == 256


def test_hapb_canonical_factory_accepts_b60_and_checkpoint_round_trips() -> None:
    cells = 60
    rows = tuple(_cell_row(2000 + cell) for cell in range(cells))
    bank = canonical_host_policy_bank_from_cell_rows_v1(rows)
    assert tuple(bank.shape) == (60, SLOT_CAPACITY_V1, POLICY_WIDTH_V1)
    assert hapb_physical_cell_count_v1(bank) == 60
    alias = _content_row_map_from_canonical_v1(bank)
    assert tuple(alias.shape) == (60, SLOT_CAPACITY_V1)
    assert int(alias[59, 0].item()) == 59 * 256
    slots = torch.full((60, 3), -1, dtype=torch.int64)
    slots[0, 0] = 300
    slots[59, 0] = 400
    overflow = stable_overflow_row_map_v1(slots)
    assert overflow.tolist() == [300, 59 * SLOT_CAPACITY_V1 + 400]


def test_hapb_checkpoint_restore_preserves_b60_bytes() -> None:
    from engine.perf.cuda_architecture_b_hapb_v1 import (
        HostAuthoritativePolicyBankV1,
        _ACTIVE_POLICY_CONTEXT_V1,
    )

    rows = tuple(_cell_row(3000 + cell) for cell in range(60))
    bank = canonical_host_policy_bank_from_cell_rows_v1(rows)
    owner = object.__new__(HostAuthoritativePolicyBankV1)
    object.__setattr__(owner, "canonical_host_bank", bank)
    object.__setattr__(owner, "content_alias_enabled", False)
    object.__setattr__(owner, "pool_epoch", 1)
    object.__setattr__(owner, "content_row_map", _content_row_map_from_canonical_v1(bank))
    object.__setattr__(
        owner,
        "content_row_versions",
        torch.ones((60, SLOT_CAPACITY_V1), dtype=torch.int64),
    )
    object.__setattr__(
        owner,
        "slot_generations",
        torch.zeros((60, SLOT_CAPACITY_V1), dtype=torch.int64),
    )
    token = _ACTIVE_POLICY_CONTEXT_V1.set(None)
    try:
        checkpoint = export_hapb_policy_bank_checkpoint_v1(owner)
    finally:
        _ACTIVE_POLICY_CONTEXT_V1.reset(token)
    assert checkpoint["shape"] == [60, SLOT_CAPACITY_V1, POLICY_WIDTH_V1]
    restored = restore_hapb_policy_bank_checkpoint_v1(checkpoint)
    assert tuple(restored.shape) == (60, SLOT_CAPACITY_V1, POLICY_WIDTH_V1)
    assert torch.equal(restored, bank)
