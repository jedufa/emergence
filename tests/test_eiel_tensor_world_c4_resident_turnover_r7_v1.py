"""CPU-only repeated-turnover preparation tests for C4 R7."""

from __future__ import annotations

from dataclasses import fields

import pytest
import torch

from engine.tensor_world.c4_resident_mixed_r5 import (
    capture_c4_resident_mixed_r5_authoritative_v1,
    compare_c4_resident_mixed_r5_authoritative_v1,
)
from engine.tensor_world.c4_resident_turnover_r7 import (
    C4_RESIDENT_TURNOVER_R7_DECISION_V1,
    C4_RESIDENT_TURNOVER_R7_IDENTITY_LIMITATION_V1,
    make_c4_resident_turnover_r7_fixture_v1,
    run_c4_resident_turnover_r7_oracle_v1,
)
from engine.tensor_world.contracts import TensorWorldContractError


def _strictly_increasing(values: list[int]) -> bool:
    return all(left < right for left, right in zip(values, values[1:]))


def _scalar_rng_bytes(fixture) -> dict[str, torch.Tensor]:
    state = fixture.bundle.scalar_rng
    assert state is not None
    return {
        field.name: getattr(state, field.name)
        .detach()
        .clone()
        .contiguous()
        .view(torch.uint8)
        for field in fields(state)
        if isinstance(getattr(state, field.name), torch.Tensor)
    }


def test_c4_r7_requires_dense_safe_b4_capacity_without_gate_claim() -> None:
    with pytest.raises(TensorWorldContractError, match="capacity>=32"):
        make_c4_resident_turnover_r7_fixture_v1(capacity=31)
    fixture = make_c4_resident_turnover_r7_fixture_v1()
    fixture.validate()
    assert fixture.bundle.batch == 4
    assert fixture.bundle.capacity == 32
    assert fixture.reused_slot == 1
    assert fixture.decision == C4_RESIDENT_TURNOVER_R7_DECISION_V1
    assert fixture.gate_claim is False
    assert fixture.identity_map_limitation == (
        C4_RESIDENT_TURNOVER_R7_IDENTITY_LIMITATION_V1
    )
    assert "repeat_across_worlds" in fixture.identity_map_limitation


def test_c4_r7_five_ticks_seal_slot_identity_lineage_events_and_rng() -> None:
    fixture = make_c4_resident_turnover_r7_fixture_v1(seed=43)
    scalar_before = _scalar_rng_bytes(fixture)
    receipt = run_c4_resident_turnover_r7_oracle_v1(fixture, ticks=5)

    assert receipt.status == "PREPARATION_PASS"
    assert receipt.completed_ticks == receipt.requested_ticks == 5
    assert receipt.decision == C4_RESIDENT_TURNOVER_R7_DECISION_V1
    assert receipt.gate_claim is False
    assert len(receipt.ticks) == 5
    for world in range(4):
        dead_ids = [row.dead_agent_ids[world] for row in receipt.ticks]
        child_ids = [row.child_agent_ids[world] for row in receipt.ticks]
        tokens = [row.slot_tokens[world] for row in receipt.ticks]
        event_counts = [row.event_active_counts[world] for row in receipt.ticks]
        cursors = [row.rng_cursors[world] for row in receipt.ticks]
        semantic = [row.rng_semantic_draws[world] for row in receipt.ticks]
        assert dead_ids[0] == 2
        assert child_ids == [5, 6, 7, 8, 9]
        assert dead_ids[1:] == child_ids[:-1]
        assert _strictly_increasing(child_ids)
        assert tokens == [2, 3, 4, 5, 6]
        assert event_counts == [2, 4, 6, 8, 10]
        assert _strictly_increasing(cursors)
        assert _strictly_increasing(semantic)
        assert [row.reused_slots[world] for row in receipt.ticks] == [1] * 5
        assert [row.death_founders[world] for row in receipt.ticks] == [2, 1, 1, 1, 1]
        assert [row.death_depths[world] for row in receipt.ticks] == [0, 1, 1, 1, 1]
        assert [row.child_founders[world] for row in receipt.ticks] == [1] * 5
        assert [row.child_depths[world] for row in receipt.ticks] == [1] * 5

    population = fixture.bundle.population
    events = fixture.bundle.events
    assert population.next_agent_id.tolist() == [10, 10, 10, 10]
    assert population.active_count.tolist() == [4, 4, 4, 4]
    for world in range(4):
        assert events.typed_sequence[world, :10].tolist() == list(range(10))
        assert events.typed_event_type[world, :10:2].tolist() == [4] * 5
        assert events.typed_event_type[world, 1:10:2].tolist() == [1] * 5
        assert int(population.agent_id_to_slot[world, 9]) == 1
        for retired_id in (2, 5, 6, 7, 8):
            assert int(population.agent_id_to_slot[world, retired_id]) == -1
    scalar_after = _scalar_rng_bytes(fixture)
    for name in scalar_before:
        assert torch.equal(scalar_after[name], scalar_before[name]), name


def test_c4_r7_injected_late_failure_restores_full_multitick_boundary() -> None:
    control = make_c4_resident_turnover_r7_fixture_v1(seed=47)
    failed = make_c4_resident_turnover_r7_fixture_v1(seed=47)
    control_receipt = run_c4_resident_turnover_r7_oracle_v1(control, ticks=3)
    failed_receipt = run_c4_resident_turnover_r7_oracle_v1(
        failed, ticks=5, inject_failure_tick=3, inject_failure_world=2
    )

    assert control_receipt.status == "PREPARATION_PASS"
    assert failed_receipt.status == "INJECTED_FAILURE_ROLLED_BACK"
    assert failed_receipt.completed_ticks == 3
    assert failed_receipt.failure_tick == 3
    assert failed_receipt.rollback_applied is True
    assert failed_receipt.failure_receipt is not None
    assert failed_receipt.failure_receipt["rollback_applied"] is True
    comparison = compare_c4_resident_mixed_r5_authoritative_v1(
        capture_c4_resident_mixed_r5_authoritative_v1(
            control.bundle, control.replay
        ),
        capture_c4_resident_mixed_r5_authoritative_v1(
            failed.bundle, failed.replay
        ),
    )
    assert comparison["pass"] is True
    assert comparison["mismatches"] == []
    assert comparison["expected_digest"] == comparison["actual_digest"]
