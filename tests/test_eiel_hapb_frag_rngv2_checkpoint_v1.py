from __future__ import annotations

import torch

from engine.hapb_rngv2.checkpoint import (
    capture_hapb_rngv2_checkpoint,
    identity_from_checkpoint,
    restore_hapb_rngv2_checkpoint,
)
from engine.hapb_rngv2.identity import HapbIdentityMap
from engine.hapb_rngv2.oracle import HapbEvent, HapbOracle
from engine.hapb_rngv2.route import HapbRngv2ContractError, reject_v1_v2_mix
from engine.perf.cuda_architecture_b_hapb_v1 import (
    CONTENT_POOL_ROWS_PER_CELL_V1,
    HAPB_TRAFFIC_COUNTER_COUNT_V1,
    HostAuthoritativePolicyBankV1,
    POLICY_WIDTH_V1,
    SLOT_CAPACITY_V1,
    bind_hapb_rngv2_frontier_v1,
    export_hapb_policy_bank_checkpoint_v1,
    initialize_selected_hapb_owner_v1,
    restore_hapb_policy_bank_checkpoint_v1,
    restore_hapb_policy_owner_transient_state_v1,
    restore_hapb_rngv2_frontier_v1,
)
from engine.tw_rngv2.status import Status
import pytest


def _forbidden(payload: dict) -> None:
    assert payload["host_tape_bytes"] == 0
    assert payload["host_gaussian_cache_bytes"] == 0
    assert "python_random_gauss_primary" not in payload
    assert "python_random_gauss_secondary" not in payload
    assert "mt19937" not in payload
    assert "future_tape" not in payload
    identity_blob = str(payload["identity"]).lower()
    assert "gauss" not in identity_blob
    assert "tape" not in identity_blob


def test_checkpoint_restore_after_physical_remap_matches_future_draws() -> None:
    mapper = HapbIdentityMap(cells=2, slots=4)
    uid_a = mapper.allocate(0, 0)
    uid_b = mapper.allocate(1, 2)
    oracle = HapbOracle()
    before = oracle.generate(HapbEvent(0, 0x0331, uid_a, 20, 0), n=7)
    checkpoint = capture_hapb_rngv2_checkpoint(
        identity=mapper,
        world_uids=(0, 1),
        absolute_tick=20,
    )
    payload = checkpoint.to_dict()
    _forbidden(payload)
    mapper.remap_physical(0, 0, 3)
    mapper.move_host_page(0, 3, 4)
    restored = restore_hapb_rngv2_checkpoint(payload)
    restored_map = identity_from_checkpoint(restored)
    assert restored_map.uid_at(0, 0) == uid_a
    assert restored_map.uid_at(1, 2) == uid_b
    after = oracle.generate(
        HapbEvent(0, 0x0331, restored_map.uid_at(0, 0), 20, 0), n=7
    )
    assert after.status is Status.RNG_OK
    assert after.words == before.words
    assert after.z0_bits == before.z0_bits
    later = oracle.generate(
        HapbEvent(0, 0x0331, restored_map.uid_at(0, 0), 21, 0), n=7
    )
    assert later.words != before.words


def test_checkpoint_rejects_tapes_and_legacy_mix() -> None:
    mapper = HapbIdentityMap(cells=1, slots=2)
    mapper.allocate(0, 0)
    payload = capture_hapb_rngv2_checkpoint(
        identity=mapper,
        world_uids=(0,),
        absolute_tick=1,
    ).to_dict()
    payload["host_tape_bytes"] = 16
    with pytest.raises(HapbRngv2ContractError, match="tape"):
        restore_hapb_rngv2_checkpoint(payload)
    reject_v1_v2_mix(hapb_rng_semantics_version=2, tape_capacity=0)
    with pytest.raises(HapbRngv2ContractError, match="tapes"):
        reject_v1_v2_mix(hapb_rng_semantics_version=2, tape_capacity=2048)


def _cpu_policy_owner(*, cells: int = 1) -> HostAuthoritativePolicyBankV1:
    bank = torch.zeros((cells, SLOT_CAPACITY_V1, POLICY_WIDTH_V1), dtype=torch.float32)
    content_row_map = torch.full((cells, SLOT_CAPACITY_V1), -1, dtype=torch.int64)
    for cell in range(cells):
        content_row_map[cell, :CONTENT_POOL_ROWS_PER_CELL_V1] = (
            torch.arange(CONTENT_POOL_ROWS_PER_CELL_V1, dtype=torch.int64)
            + cell * CONTENT_POOL_ROWS_PER_CELL_V1
        )
    return HostAuthoritativePolicyBankV1(
        canonical_host_bank=bank,
        pinned_staging_buffers=(
            torch.empty((512, POLICY_WIDTH_V1), dtype=torch.float32),
        ),
        hot_prefix_cache=None,
        cuda_tile_arena=None,
        tile_rows=512,
        hot_prefix_valid=False,
        content_row_map=content_row_map,
        content_row_versions=torch.ones((cells, SLOT_CAPACITY_V1), dtype=torch.int64),
        slot_generations=torch.zeros((cells, SLOT_CAPACITY_V1), dtype=torch.int64),
        pool_epoch=1,
        content_alias_enabled=True,
        policy_traffic_counters=torch.zeros(
            (HAPB_TRAFFIC_COUNTER_COUNT_V1,), dtype=torch.int64
        ),
    )


def test_hapb_policy_bank_checkpoint_roundtrips_v2_frontier() -> None:
    mapper = HapbIdentityMap(cells=1, slots=SLOT_CAPACITY_V1)
    uid = mapper.allocate(0, 7)
    oracle = HapbOracle()
    before = oracle.generate(HapbEvent(0, 0x0331, uid, 9, 0), n=7)
    owner = bind_hapb_rngv2_frontier_v1(
        _cpu_policy_owner(),
        identity=mapper,
        world_uids=(0,),
        absolute_tick=9,
    )
    payload = export_hapb_policy_bank_checkpoint_v1(owner)
    assert payload["hapb_rng_semantics_version"] == 2
    assert payload["hapb_rngv2_frontier"]["host_tape_bytes"] == 0
    assert "python_random_gauss_primary" not in payload
    bank = restore_hapb_policy_bank_checkpoint_v1(payload)
    assert tuple(bank.shape) == (1, SLOT_CAPACITY_V1, POLICY_WIDTH_V1)
    frontier = restore_hapb_rngv2_frontier_v1(payload)
    assert frontier is not None
    restored = restore_hapb_rngv2_checkpoint(frontier)
    restored_map = identity_from_checkpoint(restored)
    assert restored_map.uid_at(0, 7) == uid
    after = oracle.generate(
        HapbEvent(0, 0x0331, restored_map.uid_at(0, 7), 9, 0), n=7
    )
    assert after.words == before.words
    assert after.z0_bits == before.z0_bits


def test_legacy_hapb_policy_bank_checkpoint_omits_v2_frontier() -> None:
    payload = export_hapb_policy_bank_checkpoint_v1(_cpu_policy_owner())
    assert "hapb_rngv2_frontier" not in payload
    assert restore_hapb_rngv2_frontier_v1(payload) is None


def test_transient_restore_keeps_v2_frontier_on_live_owner_path() -> None:
    mapper = HapbIdentityMap(cells=1, slots=SLOT_CAPACITY_V1)
    uid = mapper.allocate(0, 3)
    owner = bind_hapb_rngv2_frontier_v1(
        _cpu_policy_owner(),
        identity=mapper,
        world_uids=(0,),
        absolute_tick=11,
    )
    payload = export_hapb_policy_bank_checkpoint_v1(owner)
    blank = _cpu_policy_owner()
    assert blank.hapb_rng_semantics_version == 1
    assert blank.hapb_rngv2_frontier is None
    restored = restore_hapb_policy_owner_transient_state_v1(blank, payload)
    assert restored.hapb_rng_semantics_version == 2
    assert restored.hapb_rngv2_frontier is not None
    restored_map = identity_from_checkpoint(
        restore_hapb_rngv2_checkpoint(restored.hapb_rngv2_frontier)
    )
    assert restored_map.uid_at(0, 3) == uid
    oracle = HapbOracle()
    first = oracle.generate(HapbEvent(0, 0x0331, uid, 11, 0), n=7)
    again = oracle.generate(
        HapbEvent(0, 0x0331, restored_map.uid_at(0, 3), 11, 0), n=7
    )
    assert again.words == first.words


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for HAPB owner init")
def test_initialize_selected_hapb_owner_binds_v2_when_route_selected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EIEL_HAPB_RNG_SEMANTICS_VERSION", "2")
    bank = torch.zeros((1, SLOT_CAPACITY_V1, POLICY_WIDTH_V1), dtype=torch.float32)
    owner = initialize_selected_hapb_owner_v1(
        bank,
        device="cuda",
        populate_hot_prefix=False,
    )
    assert owner.hapb_rng_semantics_version == 2
    assert owner.hapb_rngv2_frontier is not None
    assert owner.hapb_rngv2_frontier["host_tape_bytes"] == 0
    payload = export_hapb_policy_bank_checkpoint_v1(owner)
    assert payload["hapb_rng_semantics_version"] == 2
    restored = restore_hapb_policy_owner_transient_state_v1(owner, payload)
    assert restored.hapb_rng_semantics_version == 2
    assert restored.hapb_rngv2_frontier["contract_hash"] == (
        owner.hapb_rngv2_frontier["contract_hash"]
    )
