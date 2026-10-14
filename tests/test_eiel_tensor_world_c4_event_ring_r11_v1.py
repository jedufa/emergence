"""Long-lifetime and atomicity tests for the R11 event double buffer."""

import copy

import pytest

from engine.tensor_world.c4_event_ring_r11 import (
    C4_EVENT_RING_R11_DECISION_V1,
    C4_EVENT_RING_R11_SCHEMA_VERSION_V1,
    TensorWorldC4EventRingRowR11V1,
    c4_event_ring_r11_source_receipt_v1,
    make_c4_event_ring_r11_v1,
)
from engine.tensor_world.contracts import TensorWorldContractError


def _rows(count: int, *, tick: int = 0):
    return tuple(
        TensorWorldC4EventRingRowR11V1(
            event_type=1,
            generation=3,
            tick=tick,
            actor_id=100 + index,
            target_id=-1,
            value=float(index),
            actor_slot=index % 4,
            founder_id=100,
        )
        for index in range(count)
    )


def test_r11_event_ring_recycles_storage_without_resetting_sequence() -> None:
    ring = make_c4_event_ring_r11_v1(batch=1, capacity=4, event_capacity=4)
    assert ring.append(world=0, rows=_rows(4, tick=1)) == (0, 1, 2, 3)
    ring.acknowledge(world=0, upto_sequence=4)
    assert ring.append(world=0, rows=_rows(4, tick=2)) == (4, 5, 6, 7)
    ring.acknowledge(world=0, upto_sequence=8)

    # Repeated turnover continues past the physical 4-row capacity.  The
    # sequence is absolute and never wraps or resets when a bank is cleared.
    for tick in range(3, 30):
        start = int(ring.producer_cursor[0])
        assert ring.append(world=0, rows=_rows(2, tick=tick)) == (start, start + 1)
        ring.acknowledge(world=0, upto_sequence=int(ring.producer_cursor[0]))
    assert int(ring.producer_cursor[0]) == 62
    assert int(ring.consumer_cursor[0]) == 62
    assert ring.rows(world=0) == ()


def test_r11_event_ring_capacity_failure_is_atomic() -> None:
    ring = make_c4_event_ring_r11_v1(batch=1, capacity=4, event_capacity=2)
    ring.append(world=0, rows=_rows(4, tick=1))
    before = copy.deepcopy(ring)
    with pytest.raises(TensorWorldContractError, match="capacity"):
        ring.append(world=0, rows=_rows(1, tick=2))
    assert int(ring.producer_cursor[0]) == int(before.producer_cursor[0])
    assert int(ring.consumer_cursor[0]) == int(before.consumer_cursor[0])
    assert ring.bank_counts.equal(before.bank_counts)
    assert ring.bank_start_sequences.equal(before.bank_start_sequences)
    for actual, saved in zip(ring.banks, before.banks, strict=True):
        assert actual.typed_sequence.equal(saved.typed_sequence)


def test_r11_event_ring_receipt_is_not_a_gate_claim() -> None:
    receipt = c4_event_ring_r11_source_receipt_v1()
    assert receipt["schema_version"] == C4_EVENT_RING_R11_SCHEMA_VERSION_V1
    assert receipt["decision"] == C4_EVENT_RING_R11_DECISION_V1
    assert receipt["sequence_reset_on_reuse"] is False
    assert receipt["gate_claim"] is False

