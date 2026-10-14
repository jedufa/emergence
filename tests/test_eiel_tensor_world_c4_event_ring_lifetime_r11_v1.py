"""Source-current lifetime evidence for R11 event-ring acknowledgement."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from engine.tensor_world.c4_event_ring_lifetime_r11 import (
    C4_EVENT_RING_LIFETIME_R11_BOUNDARY_V1,
    C4_EVENT_RING_LIFETIME_R11_DECISION_V1,
    C4_EVENT_RING_LIFETIME_R11_SCHEMA_VERSION_V1,
    TensorWorldC4EventRingWindowBoundaryR11V1,
    c4_event_ring_lifetime_r11_source_receipt_v1,
    execute_c4_event_ring_window_boundary_r11_v1,
    run_c4_event_ring_lifetime_r11_v1,
)
from engine.tensor_world.c4_event_ring_r11 import (
    TensorWorldC4EventRingRowR11V1,
    _event_ring_tensors,
    make_c4_event_ring_r11_v1,
)
from engine.tensor_world.c4_resident_multigen_r10 import (
    C4ResidentMultigenR10FixtureV1,
    make_c4_resident_multigen_r10_fixture_v1,
)
from engine.tensor_world.c5_determinism import make_c5_pcg64_context_v1
from engine.tensor_world.c5_graph_checkpoint_r11 import (
    TensorWorldC5GraphRunControlR11V1,
    capture_c5_graph_checkpoint_r11_v1,
    make_c5_graph_run_control_r11_v1,
    restore_c5_graph_checkpoint_r11_v1,
)
from engine.tensor_world.contracts import TensorWorldContractError


ROOT = Path(__file__).resolve().parents[1]
BACKEND = "tensor_world_cuda_graph_r11"
SOURCE = "event-ring-lifetime-source-current"
PROVIDER = "R11_EVENT_RING_LIFETIME"


def _row(
    *,
    generation: int,
    tick: int,
    world: int,
    ordinal: int,
) -> TensorWorldC4EventRingRowR11V1:
    identity = generation * 100_000 + tick * 100 + world * 10 + ordinal
    return TensorWorldC4EventRingRowR11V1(
        event_type=1 + ordinal,
        generation=generation,
        tick=tick,
        actor_id=identity,
        target_id=-1,
        value=float(identity),
        actor_slot=world,
        parent_id=identity - 1,
        child_id=identity,
        founder_id=generation * 100_000,
        lineage_depth=generation,
        subtype=ordinal,
        reason=world,
        success=1,
    )


def _schedule(*, batch: int, windows: int = 12, tick_start: int = 40):
    boundaries = []
    for window in range(windows):
        generation = 7 + window // 4
        start = tick_start + 5 * window
        rows = []
        acknowledgements = []
        for world in range(batch):
            if world == 0:
                rows.append(
                    (
                        _row(
                            generation=generation,
                            tick=start,
                            world=world,
                            ordinal=0,
                        ),
                        _row(
                            generation=generation,
                            tick=start + 4,
                            world=world,
                            ordinal=1,
                        ),
                    )
                )
                acknowledgements.append((window + 1) % 3 == 0)
            elif world == 1:
                rows.append(
                    (
                        _row(
                            generation=generation,
                            tick=start + 2,
                            world=world,
                            ordinal=0,
                        ),
                    )
                )
                acknowledgements.append((window + 1) % 5 == 0)
            else:
                rows.append(())
                acknowledgements.append(False)
        boundaries.append(
            TensorWorldC4EventRingWindowBoundaryR11V1(
                generation=generation,
                tick_start=start,
                rows_by_world=tuple(rows),
                acknowledge_worlds=tuple(acknowledgements),
            )
        )
    return tuple(boundaries)


def _fixture(seed: int) -> C4ResidentMultigenR10FixtureV1:
    fixture = make_c4_resident_multigen_r10_fixture_v1(seed=seed)
    fixture.replay = make_c5_pcg64_context_v1(
        fixture.bundle, streams=2, seed=seed, typed=True
    )
    fixture.bundle.event_ring = make_c4_event_ring_r11_v1(
        batch=48,
        capacity=32,
        event_capacity=4,
    )
    fixture.validate()
    fixture.replay.validate_typed_continuation_for_bundle(fixture.bundle)
    return fixture


def _control(*, generation: int, next_tick: int):
    control = make_c5_graph_run_control_r11_v1(batch=48)
    _set_control(control, generation=generation, next_tick=next_tick)
    return control


def _set_control(
    control: TensorWorldC5GraphRunControlR11V1,
    *,
    generation: int,
    next_tick: int,
) -> None:
    control.generation.fill_(generation)
    control.tick.fill_(next_tick)
    control.requested_ticks.fill_(5)
    control.completed_ticks.fill_(5)
    control.current_tick.fill_(next_tick - 1)
    control.global_status.zero_()
    control.first_error_stage.fill_(-1)
    control.first_error_world.fill_(-1)
    control.first_error_code.zero_()
    control.first_error_tick.fill_(-1)
    control.stop_requested.zero_()
    control.receipt_write_index.fill_(5)
    control.validate(semantic=True)


def _capture(
    fixture: C4ResidentMultigenR10FixtureV1,
    control: TensorWorldC5GraphRunControlR11V1,
    *,
    label: str,
):
    return capture_c5_graph_checkpoint_r11_v1(
        fixture.bundle,
        fixture.replay,
        control,
        backend_id=BACKEND,
        source_commit=SOURCE,
        provider_contract_id=PROVIDER,
        metadata={"label": label},
    )


def test_r11_event_lifetime_is_boundary_only_and_source_bound() -> None:
    receipt = c4_event_ring_lifetime_r11_source_receipt_v1()
    assert receipt["schema_version"] == C4_EVENT_RING_LIFETIME_R11_SCHEMA_VERSION_V1
    assert receipt["decision"] == C4_EVENT_RING_LIFETIME_R11_DECISION_V1
    assert receipt["graph_window_ticks"] == 5
    assert receipt["acknowledgement_boundary"] == (
        C4_EVENT_RING_LIFETIME_R11_BOUNDARY_V1
    )
    assert receipt["normal_tick_cpu_drain"] is False
    assert receipt["between_tick_cpu_drain"] is False
    assert receipt["native_event_production"] == "unchanged_resident_owner"
    assert receipt["boundary_atomicity"] == "full_ring_shadow_then_commit"
    assert receipt["gate_claim"] is False
    assert all(
        len(source["sha256"]) == 64 and (ROOT / source["path"]).is_file()
        for source in receipt["sources"].values()
    )

    driver_source = (
        ROOT / "engine/tensor_world/c4_event_ring_lifetime_r11.py"
    ).read_text(encoding="utf-8")
    graph_owner_source = (
        ROOT / "engine/tensor_world/c4_resident_composed_r11.py"
    ).read_text(encoding="utf-8")
    assert "for tick" not in driver_source
    assert driver_source.count(".acknowledge(") == 1
    assert ".rows(" not in driver_source
    assert "c4_event_ring_lifetime_r11" not in graph_owner_source


def test_r11_event_lifetime_wraps_both_banks_with_monotonic_mixed_time() -> None:
    schedule = _schedule(batch=2)
    first = make_c4_event_ring_r11_v1(batch=2, capacity=32, event_capacity=4)
    second = make_c4_event_ring_r11_v1(batch=2, capacity=32, event_capacity=4)
    first_trace = run_c4_event_ring_lifetime_r11_v1(first, schedule)
    second_trace = run_c4_event_ring_lifetime_r11_v1(second, schedule)
    assert first_trace == second_trace

    world_zero_sequences = tuple(
        sequence
        for trace in first_trace
        for sequence in trace.emitted_sequences_by_world[0]
    )
    world_one_sequences = tuple(
        sequence
        for trace in first_trace
        for sequence in trace.emitted_sequences_by_world[1]
    )
    assert world_zero_sequences == tuple(range(24))
    assert world_one_sequences == tuple(range(12))
    assert int(first.producer_cursor[0]) == 24
    assert int(first.producer_cursor[1]) == 12
    assert {trace.active_banks[0] for trace in first_trace} == {0, 1}
    assert {trace.active_banks[1] for trace in first_trace} == {0, 1}
    assert [trace.generation for trace in first_trace] == [
        7,
        7,
        7,
        7,
        8,
        8,
        8,
        8,
        9,
        9,
        9,
        9,
    ]
    assert [trace.tick_start for trace in first_trace] == list(range(40, 100, 5))
    for boundary_index in (2, 5, 8, 11):
        trace = first_trace[boundary_index]
        assert trace.consumer_cursors[0] == trace.producer_cursors[0]
        assert trace.bank_counts[0] == (0, 0)
    for boundary_index in (4, 9):
        trace = first_trace[boundary_index]
        assert trace.consumer_cursors[1] == trace.producer_cursors[1]
        assert trace.bank_counts[1] == (0, 0)
    assert all(
        earlier.consumer_cursors[world] <= later.consumer_cursors[world]
        for earlier, later in zip(first_trace, first_trace[1:])
        for world in range(2)
    )


def test_r11_event_lifetime_checkpoint_restart_continues_exactly() -> None:
    schedule = _schedule(batch=48)
    source = _fixture(307)
    first_trace = run_c4_event_ring_lifetime_r11_v1(
        source.bundle.event_ring,
        schedule[:6],
    )
    source.replay.event_sequence.copy_(
        source.bundle.event_ring.producer_cursor
    )
    assert first_trace[-1].producer_cursors[:2] == (12, 6)
    source_control = _control(generation=8, next_tick=70)
    checkpoint = _capture(source, source_control, label="after-six-windows")

    restored = _fixture(311)
    restored_control = make_c5_graph_run_control_r11_v1(batch=48)
    restored_digest = restore_c5_graph_checkpoint_r11_v1(
        checkpoint,
        restored.bundle,
        restored.replay,
        restored_control,
        expected_backend_id=BACKEND,
        expected_source_commit=SOURCE,
        expected_provider_contract_id=PROVIDER,
    )
    assert restored_digest == checkpoint.digest

    source_tail = run_c4_event_ring_lifetime_r11_v1(
        source.bundle.event_ring,
        schedule[6:],
    )
    restored_tail = run_c4_event_ring_lifetime_r11_v1(
        restored.bundle.event_ring,
        schedule[6:],
    )
    source.replay.event_sequence.copy_(
        source.bundle.event_ring.producer_cursor
    )
    restored.replay.event_sequence.copy_(
        restored.bundle.event_ring.producer_cursor
    )
    assert restored_tail == source_tail
    _set_control(source_control, generation=9, next_tick=100)
    _set_control(restored_control, generation=9, next_tick=100)
    source_terminal = _capture(source, source_control, label="after-twelve-windows")
    restored_terminal = _capture(
        restored,
        restored_control,
        label="after-twelve-windows",
    )
    assert restored_terminal.digest == source_terminal.digest
    assert int(restored.bundle.event_ring.producer_cursor[0]) == 24
    assert int(restored.bundle.event_ring.producer_cursor[1]) == 12


def test_r11_event_lifetime_near_capacity_failure_rolls_back_all_worlds() -> None:
    ring = make_c4_event_ring_r11_v1(batch=2, capacity=32, event_capacity=4)
    initial = TensorWorldC4EventRingWindowBoundaryR11V1(
        generation=12,
        tick_start=200,
        rows_by_world=(
            tuple(
                _row(generation=12, tick=200 + index % 5, world=0, ordinal=index)
                for index in range(7)
            ),
            tuple(
                _row(generation=12, tick=200 + index % 5, world=1, ordinal=index)
                for index in range(8)
            ),
        ),
        acknowledge_worlds=(False, False),
    )
    execute_c4_event_ring_window_boundary_r11_v1(ring, initial)
    before = copy.deepcopy(ring)
    failing = TensorWorldC4EventRingWindowBoundaryR11V1(
        generation=13,
        tick_start=205,
        rows_by_world=(
            (_row(generation=13, tick=205, world=0, ordinal=0),),
            (_row(generation=13, tick=205, world=1, ordinal=0),),
        ),
        acknowledge_worlds=(True, True),
    )
    with pytest.raises(TensorWorldContractError, match="capacity"):
        execute_c4_event_ring_window_boundary_r11_v1(ring, failing)
    assert all(
        actual.equal(saved)
        for actual, saved in zip(
            _event_ring_tensors(ring),
            _event_ring_tensors(before),
            strict=True,
        )
    )
