"""CPU/static and opt-in CUDA contracts for R11 ring acknowledgement."""

from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass
import os
from pathlib import Path

import pytest
import torch

from engine.tensor_world.c4_event_ring_cuda_boundary_r11 import (
    C4_EVENT_RING_CUDA_BOUNDARY_R11_CHECKPOINT_V1,
    C4_EVENT_RING_CUDA_BOUNDARY_R11_DECISION_V1,
    C4_EVENT_RING_CUDA_BOUNDARY_R11_REPORT_V1,
    C4_EVENT_RING_CUDA_BOUNDARY_R11_SCHEMA_VERSION_V1,
    _guarded_acknowledge_ring_v1,
    accept_c4_event_ring_boundary_r11_v1,
    c4_event_ring_cuda_boundary_r11_source_receipt_v1,
    make_c4_event_ring_acknowledgement_candidate_r11_v1,
    materialize_c4_event_ring_report_boundary_r11_v1,
)
from engine.tensor_world.c4_event_ring_lifetime_r11 import (
    TensorWorldC4EventRingWindowBoundaryR11V1,
    execute_c4_event_ring_window_boundary_r11_v1,
)
from engine.tensor_world.c4_event_ring_r11 import (
    TensorWorldC4EventRingRowR11V1,
    _event_ring_tensors,
    make_c4_event_ring_r11_v1,
)
from engine.tensor_world.c4_mating import TensorWorldC4MateBirthConfigV1
from engine.tensor_world.c4_resident_composed_r11 import (
    bind_c4_resident_composed_r11_v1,
    capture_c4_resident_composed_r11_window_v1,
    execute_c4_resident_composed_r11_window_cuda_v1,
    make_c4_resident_composed_r11_staging_v1,
    prepare_c4_resident_composed_r11_window_control_v1,
)
from engine.tensor_world.c4_resident_multitick_r10 import (
    make_c4_resident_multitick_r10_fixture_v1,
)
from engine.tensor_world.c5_determinism import make_c5_pcg64_context_v1
from engine.tensor_world.c5_graph_checkpoint_r11 import (
    make_c5_graph_run_control_r11_v1,
)
from engine.tensor_world.contracts import TensorWorldContractError


ROOT = Path(__file__).resolve().parents[1]
OWNER_BINDING = "a" * 64


def _row(*, generation: int, tick: int, ordinal: int):
    return TensorWorldC4EventRingRowR11V1(
        event_type=1,
        generation=generation,
        tick=tick,
        actor_id=100 + ordinal,
        target_id=-1,
        value=float(ordinal),
        actor_slot=ordinal,
        founder_id=100,
    )


def _control(*, batch: int, generation: int, next_tick: int):
    control = make_c5_graph_run_control_r11_v1(batch=batch)
    control.generation.fill_(generation)
    control.tick.fill_(next_tick)
    control.requested_ticks.fill_(5)
    control.completed_ticks.fill_(5)
    control.current_tick.fill_(next_tick - 1)
    control.receipt_write_index.fill_(5)
    control.validate(semantic=True)
    return control


def _candidate(ring, control):
    return make_c4_event_ring_acknowledgement_candidate_r11_v1(
        ring,
        control,
        owner_binding=OWNER_BINDING,
        acceptance_kind=C4_EVENT_RING_CUDA_BOUNDARY_R11_REPORT_V1,
    )


def _clone_tensors(ring):
    return tuple(tensor.clone() for tensor in _event_ring_tensors(ring))


def test_r11_cuda_acknowledgement_contract_is_post_window_only() -> None:
    receipt = c4_event_ring_cuda_boundary_r11_source_receipt_v1()
    assert (
        receipt["schema_version"] == C4_EVENT_RING_CUDA_BOUNDARY_R11_SCHEMA_VERSION_V1
    )
    assert receipt["decision"] == C4_EVENT_RING_CUDA_BOUNDARY_R11_DECISION_V1
    assert receipt["boundary"] == "after_complete_five_tick_graph_window_only"
    assert receipt["normal_tick_cpu_semantic_work"] is False
    assert receipt["between_tick_cpu_semantic_work"] is False
    assert receipt["report_materialization"] == "one_grouped_boundary_D2H"
    assert receipt["checkpoint_materialization"] == ("reuse_C5_snapshot_zero_extra_D2H")
    assert receipt["reclamation"] == "next_native_window_only"
    assert receipt["bank_payload_host_mutation"] is False
    assert receipt["native_source_changed"] is False
    assert receipt["gate_claim"] is False
    assert all(
        len(source["sha256"]) == 64 and (ROOT / source["path"]).is_file()
        for source in receipt["sources"].values()
    )

    adapter_source = (
        ROOT / "engine/tensor_world/c4_event_ring_cuda_boundary_r11.py"
    ).read_text(encoding="utf-8")
    graph_owner_source = (
        ROOT / "engine/tensor_world/c4_resident_composed_r11.py"
    ).read_text(encoding="utf-8")
    assert "execute_c4_resident_composed_r11_window_cuda_v1" not in adapter_source
    assert "c4_event_ring_cuda_boundary_r11" not in graph_owner_source
    assert ".acknowledge(" not in adapter_source


def test_r11_guarded_acceptance_defers_reclamation_to_next_window() -> None:
    ring = make_c4_event_ring_r11_v1(batch=1, capacity=32, event_capacity=4)
    ring.append(
        world=0,
        rows=tuple(_row(generation=7, tick=40, ordinal=index) for index in range(8)),
    )
    control = _control(batch=1, generation=8, next_tick=45)
    candidate = _candidate(ring, control)
    assert (
        candidate.event_ring.producer_cursor.data_ptr()
        != ring.producer_cursor.data_ptr()
    )
    replay_sequence = ring.producer_cursor.clone()
    status = _guarded_acknowledge_ring_v1(
        ring,
        control,
        replay_sequence,
        candidate,
    )
    assert status.tolist() == [1]
    assert ring.consumer_cursor.tolist() == [8]
    assert ring.bank_counts.tolist() == [[4, 4]]

    boundary = TensorWorldC4EventRingWindowBoundaryR11V1(
        generation=8,
        tick_start=45,
        rows_by_world=((_row(generation=8, tick=45, ordinal=8),),),
        acknowledge_worlds=(False,),
    )
    trace = execute_c4_event_ring_window_boundary_r11_v1(ring, boundary)
    assert trace.emitted_sequences_by_world == ((8,),)
    assert trace.producer_cursors == (9,)
    assert trace.consumer_cursors == (8,)
    assert trace.active_banks == (0,)
    assert trace.bank_counts == ((1, 4),)
    assert trace.bank_start_sequences[0][0] == 8
    assert int(ring.banks[0].typed_sequence[0, 0]) == 8

    fill_second_cycle = TensorWorldC4EventRingWindowBoundaryR11V1(
        generation=8,
        tick_start=45,
        rows_by_world=(
            tuple(
                _row(generation=8, tick=45 + index % 5, ordinal=index)
                for index in range(9, 16)
            ),
        ),
        acknowledge_worlds=(False,),
    )
    second_trace = execute_c4_event_ring_window_boundary_r11_v1(
        ring,
        fill_second_cycle,
    )
    assert second_trace.emitted_sequences_by_world == (tuple(range(9, 16)),)
    assert second_trace.bank_start_sequences == ((8, 12),)
    assert second_trace.bank_counts == ((4, 4),)

    second_control = _control(batch=1, generation=9, next_tick=50)
    second_candidate = _candidate(ring, second_control)
    second_status = _guarded_acknowledge_ring_v1(
        ring,
        second_control,
        ring.producer_cursor.clone(),
        second_candidate,
    )
    assert second_status.tolist() == [1]
    reclaim_again = TensorWorldC4EventRingWindowBoundaryR11V1(
        generation=9,
        tick_start=50,
        rows_by_world=((_row(generation=9, tick=50, ordinal=16),),),
        acknowledge_worlds=(False,),
    )
    third_trace = execute_c4_event_ring_window_boundary_r11_v1(
        ring,
        reclaim_again,
    )
    assert third_trace.emitted_sequences_by_world == ((16,),)
    assert third_trace.producer_cursors == (17,)
    assert third_trace.consumer_cursors == (16,)
    assert third_trace.active_banks == (0,)
    assert third_trace.bank_start_sequences == ((16, 12),)
    assert third_trace.bank_counts == ((1, 4),)
    assert int(ring.banks[0].typed_sequence[0, 0]) == 16


def test_r11_failed_or_stale_acknowledgement_never_mutates_owner_ring() -> None:
    ring = make_c4_event_ring_r11_v1(batch=1, capacity=32, event_capacity=4)
    ring.append(
        world=0,
        rows=(_row(generation=11, tick=90, ordinal=0),),
    )
    control = _control(batch=1, generation=12, next_tick=95)
    candidate = _candidate(ring, control)
    before_rejection = _clone_tensors(ring)
    with pytest.raises(TensorWorldContractError, match="not accepted"):
        accept_c4_event_ring_boundary_r11_v1(
            object(),  # Rejection precedes CUDA-owner inspection.
            candidate,
            accepted_artifact_digest="b" * 64,
        )
    assert all(
        actual.equal(saved)
        for actual, saved in zip(
            _event_ring_tensors(ring), before_rejection, strict=True
        )
    )

    ring.append(
        world=0,
        rows=(_row(generation=12, tick=95, ordinal=1),),
    )
    replay_sequence = ring.producer_cursor.clone()
    before_stale = _clone_tensors(ring)
    status = _guarded_acknowledge_ring_v1(
        ring,
        control,
        replay_sequence,
        candidate,
    )
    assert status.tolist() == [0]
    assert all(
        actual.equal(saved)
        for actual, saved in zip(_event_ring_tensors(ring), before_stale, strict=True)
    )

    payload_candidate = _candidate(ring, control)
    ring.banks[0].typed_value[0, 0].add_(0.5)
    before_payload_drift = _clone_tensors(ring)
    payload_status = _guarded_acknowledge_ring_v1(
        ring,
        control,
        replay_sequence,
        payload_candidate,
    )
    assert payload_status.tolist() == [0]
    assert all(
        actual.equal(saved)
        for actual, saved in zip(
            _event_ring_tensors(ring), before_payload_drift, strict=True
        )
    )

    corrupted = copy.deepcopy(candidate)
    corrupted.event_ring.producer_cursor.add_(1)
    with pytest.raises(TensorWorldContractError, match="corrupted"):
        _guarded_acknowledge_ring_v1(
            ring,
            control,
            replay_sequence,
            corrupted,
        )

    checkpoint_candidate = make_c4_event_ring_acknowledgement_candidate_r11_v1(
        ring,
        control,
        owner_binding=OWNER_BINDING,
        acceptance_kind=C4_EVENT_RING_CUDA_BOUNDARY_R11_CHECKPOINT_V1,
        artifact_digest="d" * 64,
    )
    before_checkpoint_rejection = _clone_tensors(ring)
    with pytest.raises(TensorWorldContractError, match="not accepted"):
        accept_c4_event_ring_boundary_r11_v1(
            object(),
            checkpoint_candidate,
            accepted_artifact_digest="e" * 64,
        )
    assert all(
        actual.equal(saved)
        for actual, saved in zip(
            _event_ring_tensors(ring), before_checkpoint_rejection, strict=True
        )
    )


def test_r11_incomplete_window_cannot_create_acknowledgement_candidate() -> None:
    ring = make_c4_event_ring_r11_v1(batch=1, capacity=32, event_capacity=4)
    control = _control(batch=1, generation=4, next_tick=25)
    control.completed_ticks.fill_(4)
    before = _clone_tensors(ring)
    with pytest.raises(TensorWorldContractError, match="completion|complete five-tick"):
        _candidate(ring, control)
    assert all(
        actual.equal(saved)
        for actual, saved in zip(_event_ring_tensors(ring), before, strict=True)
    )


def _move_dataclass_tensors(value: object, device: torch.device) -> object:
    clone = copy.deepcopy(value)

    def move(owner: object) -> None:
        if isinstance(owner, (tuple, list)):
            for item in owner:
                move(item)
            return
        if not is_dataclass(owner):
            return
        for field in fields(owner):
            item = getattr(owner, field.name)
            if isinstance(item, torch.Tensor):
                setattr(owner, field.name, item.to(device=device))
            else:
                move(item)

    move(clone)
    return clone


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.skipif(
    os.environ.get("EIEL_C4_R11_CUDA_ACK_PROBE") != "1",
    reason="R11 CUDA acknowledgement probe is explicitly opt-in",
)
def test_r11_real_cuda_owner_acknowledges_then_reclaims_next_window(
    tmp_path: Path,
) -> None:
    fixture = make_c4_resident_multitick_r10_fixture_v1(seed=337)
    ring = make_c4_event_ring_r11_v1(
        batch=48,
        capacity=32,
        event_capacity=16,
    )
    for world in range(48):
        ring.append(
            world=world,
            rows=tuple(
                _row(generation=20, tick=200, ordinal=index) for index in range(32)
            ),
        )
    fixture.bundle.event_ring = ring
    fixture.bundle.events.active_count.fill_(32)
    replay = make_c5_pcg64_context_v1(
        fixture.bundle,
        streams=2,
        seed=337,
        typed=True,
    )
    replay.event_sequence.fill_(32)
    fixture.bundle.base.agent.repro_threshold.fill_(float("inf"))
    bundle = _move_dataclass_tensors(fixture.bundle, torch.device("cuda"))
    cuda_replay = _move_dataclass_tensors(replay, torch.device("cuda"))
    config = TensorWorldC4MateBirthConfigV1(
        max_births=0,
        allow_mutation=True,
        max_energy=1000.0,
    )
    staging = make_c4_resident_composed_r11_staging_v1(
        batch=48,
        capacity=32,
        grid_size=8,
        streams=int(cuda_replay.rng_state_words.shape[1]),
        device="cuda",
    )
    c2_path = os.environ.get(
        "TENSOR_WORLD_C2_PREBUILT",
        "/tmp/emergence-test/runtime/evidence_raw/tensor_world/production_c0_c7/"
        "20260815_c2_430489534_build_01/tensor_world_c2_ext.so",
    )
    build_dir = Path(os.environ.get("EIEL_C4_R11_BUILD_DIR", tmp_path / "build"))
    owner = bind_c4_resident_composed_r11_v1(
        bundle,
        cuda_replay,
        staging,
        enable_native=True,
        population_config=config,
        c2_extension_path=c2_path,
        native_build_directory=build_dir,
    )
    capture_c4_resident_composed_r11_window_v1(owner, config)
    prepare_c4_resident_composed_r11_window_control_v1(
        owner,
        generation=20,
        tick=200,
    )
    result = execute_c4_resident_composed_r11_window_cuda_v1(owner)
    torch.cuda.synchronize()
    assert result.completed_ticks.cpu().tolist() == [5] * 48
    candidate = materialize_c4_event_ring_report_boundary_r11_v1(owner)
    assert candidate.event_ring.producer_cursor.tolist() == [32] * 48
    consumer_before = bundle.event_ring.consumer_cursor.clone()
    with pytest.raises(TensorWorldContractError, match="not accepted"):
        accept_c4_event_ring_boundary_r11_v1(
            owner,
            candidate,
            accepted_artifact_digest="c" * 64,
        )
    assert bundle.event_ring.consumer_cursor.equal(consumer_before)

    status = accept_c4_event_ring_boundary_r11_v1(
        owner,
        candidate,
        accepted_artifact_digest=candidate.digest,
    )
    torch.cuda.synchronize()
    assert status.cpu().tolist() == [1]
    assert bundle.event_ring.consumer_cursor.cpu().tolist() == [32] * 48
    bundle.base.lifecycle.death_candidate[:, 1] = True
    bundle.base.base.core.agents.health[:, 1] = 0.0
    prepare_c4_resident_composed_r11_window_control_v1(
        owner,
        generation=25,
        tick=205,
    )
    execute_c4_resident_composed_r11_window_cuda_v1(owner)
    torch.cuda.synchronize()
    producer = bundle.event_ring.producer_cursor.cpu()
    assert bool((producer > 32).all())
    assert bundle.event_ring.active_bank.cpu().tolist() == [0] * 48
    assert bool((bundle.event_ring.bank_start_sequences[:, 0].cpu() == 32).all())
    assert bool((bundle.event_ring.bank_counts[:, 0].cpu() > 0).all())
    assert bundle.event_ring.banks[0].typed_sequence[:, 0].cpu().tolist() == [32] * 48
