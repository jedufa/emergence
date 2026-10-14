"""CPU-only admitted maximum-load and overflow rollback tests for C4 R8."""

from __future__ import annotations

import pytest

from engine.tensor_world.c4_population import (
    apply_c4_death_settlement_oracle_v1,
    commit_c4_ordinary_births_oracle_v1,
    plan_c4_ordinary_births_oracle_v1,
)
from engine.tensor_world.c4_resident_maxload_r8 import (
    C4_RESIDENT_MAXLOAD_R8_DECISION_V1,
    C4_RESIDENT_MAXLOAD_R8_IDENTITY_LIMITATION_V1,
    c4_resident_maxload_r8_contract_v1,
    make_c4_resident_maxload_r8_fixture_v1,
)
from engine.tensor_world.c4_resident_mixed_r5 import (
    capture_c4_resident_mixed_r5_authoritative_v1,
    compare_c4_resident_mixed_r5_authoritative_v1,
)
from engine.tensor_world.c4_transaction import (
    C4PopulationTransactionError,
    execute_c4_population_transaction_oracle_v1,
)


def test_c4_r8_contract_freezes_admitted_maxima_and_staging_footprint() -> None:
    contract = c4_resident_maxload_r8_contract_v1()
    assert contract["route"] == "CPU_ORACLE_PREPARATION_ONLY"
    assert contract["batch"] == 4
    assert contract["capacity"] == 32
    assert contract["initial_active_per_world"] == 16
    assert contract["initial_inactive_per_world"] == 16
    assert contract["admitted_max_deaths_per_world"] == 8
    assert contract["admitted_max_births_per_world"] == 8
    assert contract["admitted_event_rows_per_world"] == 16
    assert contract["event_capacity_per_world"] == 64
    assert contract["identity_next_before"] == 17
    assert contract["identity_next_after"] == 25
    assert contract["identity_map_headroom_after"] == 7
    assert contract["staging_slot_rows"] == 128
    assert contract["staging_footprint_bytes"] == 10384
    assert contract["staging_footprint_classification"] == (
        "logical_minimum_preparation_inventory_not_native_vram_measurement"
    )
    assert contract["identity_map_limitation"] == (
        C4_RESIDENT_MAXLOAD_R8_IDENTITY_LIMITATION_V1
    )
    assert contract["decision"] == C4_RESIDENT_MAXLOAD_R8_DECISION_V1
    assert contract["gate_claim"] is False


def test_c4_r8_maximum_admitted_combined_turnover_matches_full_oracle() -> None:
    expected = make_c4_resident_maxload_r8_fixture_v1(seed=59)
    actual = make_c4_resident_maxload_r8_fixture_v1(seed=59)
    apply_c4_death_settlement_oracle_v1(
        expected.bundle, event_sequence=expected.replay.event_sequence
    )
    plan_c4_ordinary_births_oracle_v1(
        expected.bundle,
        expected.birth_config,
        spawn_uniforms=expected.spawn_uniforms,
    )
    commit_c4_ordinary_births_oracle_v1(
        expected.bundle,
        expected.birth_config,
    )
    receipt = execute_c4_population_transaction_oracle_v1(
        actual.bundle,
        birth_config=actual.birth_config,
        spawn_uniforms=actual.spawn_uniforms,
    )
    expected.replay.event_sequence.copy_(expected.bundle.events.active_count)
    actual.replay.event_sequence.copy_(actual.bundle.events.active_count)

    assert receipt.status == "PREPARATION_PASS"
    assert receipt.counts_after["death_count"] == (8, 8, 8, 8)
    assert receipt.counts_after["birth_count"] == (8, 8, 8, 8)
    assert receipt.counts_after["active_count"] == (16, 16, 16, 16)
    assert receipt.counts_after["event_active_count"] == (16, 16, 16, 16)
    population = actual.bundle.population
    assert population.death_slots[:, :8].tolist() == [list(range(8, 16))] * 4
    assert population.child_slots[:, :8].tolist() == [list(range(8, 16))] * 4
    assert population.birth_child_ids[:, :8].tolist() == [list(range(17, 25))] * 4
    assert population.next_agent_id.tolist() == [25, 25, 25, 25]
    assert population.free_queue_count.tolist() == [0, 0, 0, 0]
    assert population.slot_tokens[:, 8:16].tolist() == [list(range(9, 17))] * 4
    events = actual.bundle.events
    assert events.typed_event_type[:, :8].tolist() == [[4] * 8] * 4
    assert events.typed_event_type[:, 8:16].tolist() == [[1] * 8] * 4
    comparison = compare_c4_resident_mixed_r5_authoritative_v1(
        capture_c4_resident_mixed_r5_authoritative_v1(
            expected.bundle, expected.replay
        ),
        capture_c4_resident_mixed_r5_authoritative_v1(
            actual.bundle, actual.replay
        ),
    )
    assert comparison["pass"] is True
    assert comparison["mismatches"] == []
    assert comparison["expected_digest"] == comparison["actual_digest"]


@pytest.mark.parametrize("overflow", ("event", "queue", "identity"))
def test_c4_r8_overflow_restores_complete_b4_authority(overflow: str) -> None:
    fixture = make_c4_resident_maxload_r8_fixture_v1(seed=61)
    if overflow == "event":
        fixture.bundle.events.active_count[2] = fixture.bundle.events.capacity - 15
        fixture.replay.event_sequence.copy_(fixture.bundle.events.active_count)
    elif overflow == "queue":
        population = fixture.bundle.population
        population.free_queue_count[3] = 25
        population.free_queue[3, :16] = fixture.bundle.base.base.core.identity.slot_ids[
            3, 16:32
        ]
    else:
        # R11's monotonic biological-ID authority is independent of slot
        # capacity.  Exercise the real failure boundary: signed int64
        # exhaustion, not the retired dense-map capacity assumption.
        fixture.bundle.population.next_agent_id[1] = (1 << 63) - 5
    before = capture_c4_resident_mixed_r5_authoritative_v1(
        fixture.bundle, fixture.replay
    )

    with pytest.raises(C4PopulationTransactionError) as raised:
        execute_c4_population_transaction_oracle_v1(
            fixture.bundle,
            birth_config=fixture.birth_config,
            spawn_uniforms=fixture.spawn_uniforms,
        )

    receipt = raised.value.receipt
    assert receipt.rollback_applied is True
    assert receipt.counts_before == receipt.counts_after
    after = capture_c4_resident_mixed_r5_authoritative_v1(
        fixture.bundle, fixture.replay
    )
    comparison = compare_c4_resident_mixed_r5_authoritative_v1(before, after)
    assert comparison["pass"] is True
    assert comparison["mismatches"] == []
    assert comparison["expected_digest"] == comparison["actual_digest"]
