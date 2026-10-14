"""CPU/static integration tests for C5 checkpoint and R11 fatal owners."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import pytest

from engine.tensor_world.c4_event_ring_r11 import (
    TensorWorldC4EventRingRowR11V1,
    make_c4_event_ring_r11_v1,
)
from engine.tensor_world.c4_resident_multigen_r10 import (
    C4ResidentMultigenR10FixtureV1,
    make_c4_resident_multigen_r10_fixture_v1,
)
from engine.tensor_world.c4_runtime_fault_r11 import (
    C4_RUNTIME_OWNER_ACTIVE_V1,
    C4_RUNTIME_OWNER_FATAL_INVALID_V1,
)
from engine.tensor_world.c4_transaction import (
    execute_c4_population_transaction_oracle_v1,
)
from engine.tensor_world.c5_determinism import make_c5_pcg64_context_v1
from engine.tensor_world.c5_graph_checkpoint_r11 import (
    TensorWorldC5GraphRunControlR11V1,
    capture_c5_graph_checkpoint_r11_v1,
    make_c5_graph_run_control_r11_v1,
)
from engine.tensor_world.c5_graph_fault_boundary_r11 import (
    C5_GRAPH_FAULT_BOUNDARY_R11_CONTROL_BOUNDARY_V1,
    C5_GRAPH_FAULT_BOUNDARY_R11_DECISION_V1,
    C5_GRAPH_FAULT_BOUNDARY_R11_SCHEMA_VERSION_V1,
    accept_c5_graph_checkpoint_at_control_boundary_r11_v1,
    c5_graph_fault_boundary_r11_source_receipt_v1,
    capture_c5_graph_checkpoint_at_control_boundary_r11_v1,
    make_c5_graph_fault_boundary_owner_r11_v1,
    restore_c5_graph_checkpoint_into_fresh_owner_at_control_boundary_r11_v1,
)
from engine.tensor_world.contracts import TensorWorldContractError


ROOT = Path(__file__).resolve().parents[1]
BACKEND = "tensor_world_cuda_graph_r11"
SOURCE = "88975a5e89fa440891b4934a6c3737e1800dfe0a"
PROVIDER = "R11_B48_GRAPH_OWNER"
GRAPH_A = "a" * 64
GRAPH_B = "b" * 64


def _fixture(seed: int) -> C4ResidentMultigenR10FixtureV1:
    fixture = make_c4_resident_multigen_r10_fixture_v1(seed=seed)
    fixture.replay = make_c5_pcg64_context_v1(
        fixture.bundle, streams=2, seed=seed, typed=True
    )
    fixture.bundle.event_ring = make_c4_event_ring_r11_v1(
        batch=48, capacity=32, event_capacity=64
    )
    fixture.validate()
    fixture.replay.validate_typed_continuation_for_bundle(fixture.bundle)
    return fixture


def _advance_window(
    fixture: C4ResidentMultigenR10FixtureV1,
    *,
    generation: int,
    tick: int,
) -> None:
    bundle = fixture.bundle
    events = bundle.events
    ring = bundle.event_ring
    assert events is not None and ring is not None
    for offset in range(5):
        bundle.base.base.core.tick = tick + offset
        bundle.base.lifecycle.death_candidate.zero_()
        bundle.base.lifecycle.death_candidate[:, fixture.reused_slot] = True
        bundle.base.base.core.agents.health[:, fixture.reused_slot] = 0.0
        bundle.event_ring = None
        events.active_count.zero_()
        events.status.zero_()
        fixture.replay.event_sequence.zero_()
        try:
            execute_c4_population_transaction_oracle_v1(
                bundle,
                mate_config=replace(
                    fixture.config, generation=generation + offset
                ),
                rng_context=fixture.replay,
            )
        finally:
            bundle.event_ring = ring
        for world in range(bundle.batch):
            count = int(events.active_count[world])
            rows = tuple(
                TensorWorldC4EventRingRowR11V1(
                    event_type=int(events.typed_event_type[world, row]),
                    generation=int(events.typed_generation[world, row]),
                    tick=int(events.typed_tick[world, row]),
                    actor_id=int(events.typed_actor_id[world, row]),
                    target_id=int(events.typed_target_id[world, row]),
                    value=float(events.typed_value[world, row]),
                    actor_slot=int(events.typed_actor_slot[world, row]),
                    target_slot=int(events.typed_target_slot[world, row]),
                    parent_id=int(events.typed_parent_id[world, row]),
                    child_id=int(events.typed_child_id[world, row]),
                    founder_id=int(events.typed_founder_id[world, row]),
                    lineage_depth=int(events.typed_lineage_depth[world, row]),
                    subtype=int(events.typed_subtype[world, row]),
                    reason=int(events.typed_reason[world, row]),
                    success=int(events.typed_success[world, row]),
                )
                for row in range(count)
            )
            ring.append(world=world, rows=rows)
        fixture.replay.event_sequence.copy_(ring.producer_cursor)
        events.active_count.zero_()
        events.status.zero_()


def _completed_control(*, generation: int, next_tick: int):
    control = make_c5_graph_run_control_r11_v1(batch=48)
    control.generation.fill_(generation)
    control.tick.fill_(next_tick)
    control.requested_ticks.fill_(5)
    control.completed_ticks.fill_(5)
    control.current_tick.fill_(next_tick - 1)
    control.receipt_write_index.fill_(5)
    control.validate(semantic=True)
    return control


def _direct_capture(
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


def _accepted_checkpoint(seed: int = 211):
    fixture = _fixture(seed)
    _advance_window(fixture, generation=10, tick=1)
    control = _completed_control(generation=15, next_tick=6)
    owner = make_c5_graph_fault_boundary_owner_r11_v1(owner_id="owner-before-fault")
    owner.record_graph_capture_at_control_boundary(
        graph_capture_fingerprint=GRAPH_A
    )
    checkpoint = capture_c5_graph_checkpoint_at_control_boundary_r11_v1(
        owner,
        fixture.bundle,
        fixture.replay,
        control,
        backend_id=BACKEND,
        source_commit=SOURCE,
        provider_contract_id=PROVIDER,
        metadata={"label": "accepted-window"},
    )
    assert owner.runtime_fault.last_accepted_checkpoint_digest is None
    assert owner.pending_checkpoint_digest == checkpoint.digest
    accept_c5_graph_checkpoint_at_control_boundary_r11_v1(owner, checkpoint)
    assert owner.pending_checkpoint_digest is None
    return fixture, control, owner, checkpoint


def test_r11_fault_checkpoint_adapter_is_control_boundary_only() -> None:
    receipt = c5_graph_fault_boundary_r11_source_receipt_v1()
    assert receipt["schema_version"] == C5_GRAPH_FAULT_BOUNDARY_R11_SCHEMA_VERSION_V1
    assert receipt["decision"] == C5_GRAPH_FAULT_BOUNDARY_R11_DECISION_V1
    assert receipt["control_boundary"] == (
        C5_GRAPH_FAULT_BOUNDARY_R11_CONTROL_BOUNDARY_V1
    )
    assert receipt["graph_window_ticks"] == 5
    assert receipt["normal_tick_cpu_semantic_work"] is False
    assert receipt["between_tick_cpu_semantic_work"] is False
    assert receipt["checkpoint_cpu_work"] == "declared_control_boundaries_only"
    assert receipt["checkpoint_acceptance"] == (
        "caller_explicit_after_durable_publication"
    )
    assert receipt["graph_executable_serialized"] is False
    assert receipt["graph_recapture_execution"] == (
        "caller_owned_not_performed_by_adapter"
    )
    assert receipt["in_memory_runtime_fault_rollback_claim"] is False
    assert receipt["gate_claim"] is False
    assert all(
        len(source["sha256"]) == 64 and (ROOT / source["path"]).is_file()
        for source in receipt["sources"].values()
    )

    adapter_source = (
        ROOT / "engine/tensor_world/c5_graph_fault_boundary_r11.py"
    ).read_text(encoding="utf-8")
    graph_owner_source = (
        ROOT / "engine/tensor_world/c4_resident_composed_r11.py"
    ).read_text(encoding="utf-8")
    assert "execute_c4_resident_composed_r11" not in adapter_source
    assert "c5_graph_fault_boundary_r11" not in graph_owner_source

    owner = make_c5_graph_fault_boundary_owner_r11_v1(owner_id="initial-owner")
    with pytest.raises(TensorWorldContractError, match="fresh graph capture"):
        owner.assert_graph_window_replay_allowed_at_control_boundary()
    owner.record_graph_capture_at_control_boundary(graph_capture_fingerprint=GRAPH_A)
    owner.assert_graph_window_replay_allowed_at_control_boundary()


def test_r11_fatal_owner_recovers_only_via_fresh_owner_restore_and_recapture() -> None:
    source, _source_control, failed, checkpoint = _accepted_checkpoint()
    failed.mark_runtime_fault_at_control_boundary(
        fault_class="CUDA_GRAPH_LAUNCH_FAULT",
        detail="launch fault detected while synchronizing the graph-window boundary",
    )
    assert failed.runtime_fault.owner_state == C4_RUNTIME_OWNER_FATAL_INVALID_V1
    assert failed.runtime_fault.last_accepted_checkpoint_digest == checkpoint.digest
    assert failed.pending_checkpoint_digest is None
    assert failed.graph_capture_fingerprint is None
    with pytest.raises(TensorWorldContractError, match="FATAL/INVALID"):
        failed.assert_graph_window_replay_allowed_at_control_boundary()
    with pytest.raises(TensorWorldContractError, match="FATAL/INVALID"):
        capture_c5_graph_checkpoint_at_control_boundary_r11_v1(
            failed,
            source.bundle,
            source.replay,
            _source_control,
            backend_id=BACKEND,
            source_commit=SOURCE,
            provider_contract_id=PROVIDER,
        )

    restored = _fixture(223)
    restored.bundle.population.next_agent_id.add_(17)
    restored.bundle.population.slot_tokens[:, 1].add_(5)
    restored.replay.rng_state_words[:, :, 0].add_(1)
    restored_control = make_c5_graph_run_control_r11_v1(batch=48)
    fresh = make_c5_graph_fault_boundary_owner_r11_v1(owner_id="fresh-owner")
    restored_digest = (
        restore_c5_graph_checkpoint_into_fresh_owner_at_control_boundary_r11_v1(
            failed,
            fresh,
            checkpoint,
            restored.bundle,
            restored.replay,
            restored_control,
            expected_backend_id=BACKEND,
            expected_source_commit=SOURCE,
            expected_provider_contract_id=PROVIDER,
        )
    )
    assert restored_digest == checkpoint.digest
    assert fresh.runtime_fault.owner_state == C4_RUNTIME_OWNER_ACTIVE_V1
    assert fresh.runtime_fault.last_accepted_checkpoint_digest == checkpoint.digest
    assert fresh.pending_checkpoint_digest is None
    assert fresh.recovery_checkpoint_digest == checkpoint.digest
    assert fresh.graph_capture_fingerprint is None
    with pytest.raises(TensorWorldContractError, match="fresh graph capture"):
        fresh.assert_graph_window_replay_allowed_at_control_boundary()
    with pytest.raises(TensorWorldContractError, match="not bound"):
        fresh.record_graph_capture_at_control_boundary(
            graph_capture_fingerprint=GRAPH_B,
            recovery_checkpoint_digest="c" * 64,
        )

    fresh.record_graph_capture_at_control_boundary(
        graph_capture_fingerprint=GRAPH_B,
        recovery_checkpoint_digest=checkpoint.digest,
    )
    assert fresh.recovery_checkpoint_digest is None
    fresh.assert_graph_window_replay_allowed_at_control_boundary()
    with pytest.raises(TensorWorldContractError, match="owner's candidate"):
        accept_c5_graph_checkpoint_at_control_boundary_r11_v1(fresh, checkpoint)
    recaptured = capture_c5_graph_checkpoint_at_control_boundary_r11_v1(
        fresh,
        restored.bundle,
        restored.replay,
        restored_control,
        backend_id=BACKEND,
        source_commit=SOURCE,
        provider_contract_id=PROVIDER,
        metadata={"label": "accepted-window"},
    )
    assert recaptured.digest == checkpoint.digest
    accept_c5_graph_checkpoint_at_control_boundary_r11_v1(fresh, recaptured)
    assert failed.runtime_fault.owner_state == C4_RUNTIME_OWNER_FATAL_INVALID_V1


def test_r11_failed_recovery_is_atomic_and_requires_pristine_distinct_owner() -> None:
    _source, _control, failed, checkpoint = _accepted_checkpoint(seed=227)
    failed.mark_runtime_fault_at_control_boundary(
        fault_class="CUDA_DEVICE_FAULT",
        detail="device fault at explicit checkpoint boundary",
    )
    destination = _fixture(229)
    destination_control = make_c5_graph_run_control_r11_v1(batch=48)
    before = _direct_capture(destination, destination_control, label="unchanged")
    fresh = make_c5_graph_fault_boundary_owner_r11_v1(owner_id="fresh-atomic")
    corrupted = copy.deepcopy(checkpoint)
    corrupted.base_checkpoint.state.population.slot_tokens[0, 0] = -1

    with pytest.raises(TensorWorldContractError):
        restore_c5_graph_checkpoint_into_fresh_owner_at_control_boundary_r11_v1(
            failed,
            fresh,
            corrupted,
            destination.bundle,
            destination.replay,
            destination_control,
            expected_backend_id=BACKEND,
            expected_source_commit=SOURCE,
            expected_provider_contract_id=PROVIDER,
        )
    after = _direct_capture(destination, destination_control, label="unchanged")
    assert after.digest == before.digest
    assert fresh.runtime_fault.owner_state == C4_RUNTIME_OWNER_ACTIVE_V1
    assert fresh.runtime_fault.last_accepted_checkpoint_digest is None
    assert fresh.pending_checkpoint_digest is None
    assert fresh.recovery_checkpoint_digest is None
    assert fresh.graph_capture_fingerprint is None

    same_identity = make_c5_graph_fault_boundary_owner_r11_v1(
        owner_id=failed.owner_id
    )
    with pytest.raises(TensorWorldContractError, match="fresh owner"):
        restore_c5_graph_checkpoint_into_fresh_owner_at_control_boundary_r11_v1(
            failed,
            same_identity,
            checkpoint,
            destination.bundle,
            destination.replay,
            destination_control,
            expected_backend_id=BACKEND,
            expected_source_commit=SOURCE,
            expected_provider_contract_id=PROVIDER,
        )
