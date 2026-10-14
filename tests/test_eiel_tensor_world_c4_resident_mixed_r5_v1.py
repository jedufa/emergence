"""CPU-only B4 mixed-world contract preparation for C4 R5."""

from __future__ import annotations

from dataclasses import fields

import pytest
import torch

from engine.tensor_world.c4_population import apply_c4_death_settlement_oracle_v1
from engine.tensor_world.c4_contract import C4_PHASE_ORDER
from engine.tensor_world.c4_resident_mixed_r5 import (
    C4_RESIDENT_MIXED_R5_COUNTS_V1,
    C4_RESIDENT_MIXED_R5_DECISION_V1,
    C4_RESIDENT_MIXED_R5_IDENTITY_LIMITATION_V1,
    capture_c4_resident_mixed_r5_authoritative_v1,
    compare_c4_resident_mixed_r5_authoritative_v1,
    make_c4_resident_mixed_r5_fixture_v1,
)
from engine.tensor_world.c4_transaction import (
    C4_TRANSACTION_PREPARATION_STATUS_V1,
    C4_TRANSACTION_ROLLED_BACK_STATUS_V1,
    C4PopulationTransactionError,
    execute_c4_population_transaction_oracle_v1,
)


def _tensor_bytes(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone().contiguous().view(torch.uint8)


def _rng_owners(fixture) -> dict[str, torch.Tensor]:
    scalar = fixture.bundle.scalar_rng
    assert scalar is not None
    replay = fixture.replay
    owners = {
        f"scalar.{field.name}": _tensor_bytes(getattr(scalar, field.name))
        for field in fields(scalar)
        if isinstance(getattr(scalar, field.name), torch.Tensor)
    }
    for name in (
        "rng_state_words",
        "rng_buffer_words",
        "rng_cursors",
        "semantic_draw_counters",
        "event_order_keys",
    ):
        tensor = getattr(replay, name)
        assert tensor is not None
        owners[f"replay.{name}"] = _tensor_bytes(tensor)
    return owners


def test_c4_r5_fixture_is_true_b4_and_declares_identity_limit() -> None:
    fixture = make_c4_resident_mixed_r5_fixture_v1()
    fixture.validate()

    assert fixture.bundle.batch == 4
    assert fixture.expected_transaction_counts == (0, 1, 2, 1)
    assert fixture.expected_transaction_counts == C4_RESIDENT_MIXED_R5_COUNTS_V1
    assert fixture.decision == C4_RESIDENT_MIXED_R5_DECISION_V1
    assert fixture.gate_claim is False
    assert fixture.identity_map_limitation == (
        C4_RESIDENT_MIXED_R5_IDENTITY_LIMITATION_V1
    )
    assert "repeat_across_worlds" in fixture.identity_map_limitation
    assert fixture.bundle.base.lifecycle.death_candidate.sum(dim=1).tolist() == [0, 1, 2, 1]
    # The fixture limitation is observable: biological IDs are world-local,
    # while every world's capacity-sized identity map is internally valid.
    ids = fixture.bundle.base.base.core.agents.agent_id[:, :3]
    assert torch.equal(ids[0], ids[1])
    assert torch.equal(ids[1], ids[2])


def test_c4_r5_full_authoritative_comparator_and_event_rng_ownership() -> None:
    expected = make_c4_resident_mixed_r5_fixture_v1(seed=31)
    actual = make_c4_resident_mixed_r5_fixture_v1(seed=31)
    rng_before = _rng_owners(actual)

    apply_c4_death_settlement_oracle_v1(
        expected.bundle, event_sequence=expected.replay.event_sequence
    )
    receipt = execute_c4_population_transaction_oracle_v1(
        actual.bundle, rng_context=actual.replay
    )

    assert receipt.status == C4_TRANSACTION_PREPARATION_STATUS_V1
    assert receipt.decision == C4_RESIDENT_MIXED_R5_DECISION_V1
    assert receipt.gate_claim is False
    assert receipt.counts_after["death_count"] == C4_RESIDENT_MIXED_R5_COUNTS_V1
    assert receipt.counts_after["event_active_count"] == C4_RESIDENT_MIXED_R5_COUNTS_V1
    assert actual.bundle.events.active_count.tolist() == [0, 1, 2, 1]
    assert actual.replay.event_sequence.tolist() == [0, 1, 2, 1]
    assert actual.bundle.events.typed_event_type[1, :1].tolist() == [4]
    assert actual.bundle.events.typed_event_type[2, :2].tolist() == [4, 4]
    assert actual.bundle.events.typed_event_type[3, :1].tolist() == [4]

    comparison = compare_c4_resident_mixed_r5_authoritative_v1(
        capture_c4_resident_mixed_r5_authoritative_v1(
            expected.bundle, expected.replay
        ),
        capture_c4_resident_mixed_r5_authoritative_v1(actual.bundle, actual.replay),
    )
    assert comparison["pass"] is True
    assert comparison["mismatches"] == []
    assert comparison["tensor_count"] > 200
    assert comparison["expected_digest"] == comparison["actual_digest"]

    # Death/event settlement owns only the event cursor. Neither the PCG64
    # continuation nor CPython mutation continuation consumes a draw.
    rng_after = _rng_owners(actual)
    assert tuple(rng_before) == tuple(rng_after)
    for path in rng_before:
        assert torch.equal(rng_after[path], rng_before[path]), path

    divergent = make_c4_resident_mixed_r5_fixture_v1(seed=31)
    divergent.bundle.base.base.core.agents.energy[0, 0] += 1.0
    baseline = make_c4_resident_mixed_r5_fixture_v1(seed=31)
    mismatch = compare_c4_resident_mixed_r5_authoritative_v1(
        capture_c4_resident_mixed_r5_authoritative_v1(
            baseline.bundle, baseline.replay
        ),
        capture_c4_resident_mixed_r5_authoritative_v1(
            divergent.bundle, divergent.replay
        ),
    )
    assert mismatch["pass"] is False
    assert mismatch["mismatches"] == [
        {"path": "bundle.base.base.core.agents.energy", "reason": "bytes"}
    ]


def test_c4_r5_one_world_event_failure_rolls_back_all_b4_authority() -> None:
    fixture = make_c4_resident_mixed_r5_fixture_v1(seed=37)
    events = fixture.bundle.events
    events.active_count[2] = events.capacity - 1
    fixture.replay.event_sequence.copy_(events.active_count)
    before = capture_c4_resident_mixed_r5_authoritative_v1(
        fixture.bundle, fixture.replay
    )

    with pytest.raises(C4PopulationTransactionError) as raised:
        execute_c4_population_transaction_oracle_v1(
            fixture.bundle, rng_context=fixture.replay
        )

    receipt = raised.value.receipt
    assert receipt.status == C4_TRANSACTION_ROLLED_BACK_STATUS_V1
    assert receipt.rollback_applied is True
    assert receipt.failure_phase == C4_PHASE_ORDER[1]
    assert "capacity" in (receipt.failure_message or "")
    assert receipt.counts_before == receipt.counts_after
    after = capture_c4_resident_mixed_r5_authoritative_v1(
        fixture.bundle, fixture.replay
    )
    comparison = compare_c4_resident_mixed_r5_authoritative_v1(before, after)
    assert comparison["pass"] is True
    assert comparison["expected_digest"] == comparison["actual_digest"]
