"""CPU/static checkpoint continuation contracts for the C5/R11 graph owner."""

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
from engine.tensor_world.c4_transaction import (
    execute_c4_population_transaction_oracle_v1,
)
from engine.tensor_world.c5_determinism import make_c5_pcg64_context_v1
from engine.tensor_world.c5_graph_checkpoint_r11 import (
    C5_GRAPH_CHECKPOINT_R11_DECISION_V1,
    C5_GRAPH_CHECKPOINT_R11_SCHEMA_VERSION_V1,
    TensorWorldC5GraphRunControlR11V1,
    bind_c5_graph_run_control_r11_v1,
    c5_graph_checkpoint_r11_source_receipt_v1,
    capture_c5_graph_checkpoint_r11_v1,
    load_c5_graph_checkpoint_r11_v1,
    make_c5_graph_run_control_r11_v1,
    restore_c5_graph_checkpoint_r11_v1,
    write_c5_graph_checkpoint_r11_v1,
)
from engine.tensor_world.contracts import TensorWorldContractError


ROOT = Path(__file__).resolve().parents[1]
BACKEND = "tensor_world_cuda_graph_r11"
SOURCE = "def1f0dcd"
PROVIDER = "R11_B48_C32_GRAPH_OWNER"


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
        # The CPU oracle writes the legacy tape.  R11 treats that tape as
        # transaction-local scratch, then seals every world into the durable
        # ring and restores the absolute replay cursor at the tick boundary.
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


def _set_completed_window(
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


def _capture(fixture, control, *, label: str):
    return capture_c5_graph_checkpoint_r11_v1(
        fixture.bundle,
        fixture.replay,
        control,
        backend_id=BACKEND,
        source_commit=SOURCE,
        provider_contract_id=PROVIDER,
        metadata={"label": label},
    )


def test_c5_r11_graph_checkpoint_receipt_and_control_alias_contract() -> None:
    receipt = c5_graph_checkpoint_r11_source_receipt_v1()
    assert receipt["schema_version"] == C5_GRAPH_CHECKPOINT_R11_SCHEMA_VERSION_V1
    assert receipt["decision"] == C5_GRAPH_CHECKPOINT_R11_DECISION_V1
    assert receipt["boundary"] == "between_captured_graph_windows_only"
    assert receipt["admitted_capacities"] == [32, 512]
    assert receipt["production_geometry"] == "B48_W32_P256_C512"
    assert receipt["normal_tick_cpu_work"] is False
    assert receipt["restore_requirement"] == (
        "fresh_owner_bind_and_graph_recapture_before_replay"
    )
    assert "double_buffer_event_ring" in receipt["included"]
    assert "monotonic_identity_and_slot_tokens" in receipt["included"]
    assert "CUDA_graph_executables" in receipt["excluded_ephemeral"]
    assert receipt["gate_claim"] is False
    assert all(
        len(source["sha256"]) == 64 and (ROOT / source["path"]).is_file()
        for source in receipt["sources"].values()
    )

    control = make_c5_graph_run_control_r11_v1(batch=48)
    aliases = bind_c5_graph_run_control_r11_v1(control)
    assert aliases.generation.data_ptr() == control.generation.data_ptr()
    aliases.tick.fill_(7)
    assert int(control.tick[0]) == 7
    aliases.requested_ticks.fill_(1)
    aliases.completed_ticks.fill_(1)
    # The historical one-tick graph does not advance the window receipt index.
    aliases.receipt_write_index.zero_()
    aliases.validate(semantic=True)
    composed_source = (
        ROOT / "engine/tensor_world/c4_resident_composed_r11.py"
    ).read_text(encoding="utf-8")
    assert "c5_graph_checkpoint_r11" not in composed_source
    execute = composed_source.split(
        "def execute_c4_resident_composed_r11_cuda_v1(", 1
    )[1].split("def execute_c4_resident_composed_r11_window_cuda_v1(", 1)[0]
    assert "checkpoint" not in execute
    assert "restore" not in execute


def test_c5_r11_exact_restore_digest_and_continued_window_contract() -> None:
    source = _fixture(seed=97)
    source_control = make_c5_graph_run_control_r11_v1(batch=48)
    _advance_window(source, generation=10, tick=1)
    _set_completed_window(source_control, generation=15, next_tick=6)
    checkpoint = _capture(source, source_control, label="first_window")
    checkpoint.validate()
    changed_control = copy.deepcopy(checkpoint)
    changed_control.run_control.generation.add_(1)
    with pytest.raises(TensorWorldContractError, match="digest"):
        changed_control.validate()

    restored = _fixture(seed=101)
    restored_control = make_c5_graph_run_control_r11_v1(batch=48)
    restored.bundle.population.next_agent_id.add_(7)
    restored.bundle.population.slot_tokens[:, 1].add_(3)
    restored.replay.rng_state_words[:, :, 0].add_(1)
    restored_control.generation.fill_(99)
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
    restored_boundary = _capture(restored, restored_control, label="first_window")
    assert restored_boundary.digest == checkpoint.digest
    assert restored.bundle.population.next_agent_id.tolist() == [10] * 48
    assert restored.bundle.population.slot_tokens[:, 1].tolist() == [6] * 48
    assert int(restored.bundle.event_ring.producer_cursor[0]) == 15
    assert restored.replay.typed_role_indices == (0, 1, 2, 3, 4)
    assert restored_control.generation.tolist() == [15] * 48
    assert restored_control.current_tick.tolist() == [5] * 48

    _advance_window(source, generation=15, tick=6)
    _advance_window(restored, generation=15, tick=6)
    _set_completed_window(source_control, generation=20, next_tick=11)
    _set_completed_window(restored_control, generation=20, next_tick=11)
    source_terminal = _capture(source, source_control, label="second_window")
    restored_terminal = _capture(restored, restored_control, label="second_window")
    assert source_terminal.digest == restored_terminal.digest
    assert int(restored.bundle.event_ring.producer_cursor[0]) == 30
    assert restored.bundle.population.next_agent_id.tolist() == [15] * 48
    assert restored.bundle.population.slot_tokens[:, 1].tolist() == [11] * 48


def test_c5_r11_corrupt_checkpoint_fails_before_destination_mutation() -> None:
    source = _fixture(seed=103)
    control = make_c5_graph_run_control_r11_v1(batch=48)
    _advance_window(source, generation=10, tick=1)
    _set_completed_window(control, generation=15, next_tick=6)
    checkpoint = _capture(source, control, label="corruption_source")

    destination = _fixture(seed=107)
    destination_control = make_c5_graph_run_control_r11_v1(batch=48)
    before = _capture(destination, destination_control, label="destination_before")
    corrupted = copy.deepcopy(checkpoint)
    corrupted.base_checkpoint.state.population.slot_tokens[0, 0] = -1
    with pytest.raises(TensorWorldContractError):
        restore_c5_graph_checkpoint_r11_v1(
            corrupted,
            destination.bundle,
            destination.replay,
            destination_control,
            expected_backend_id=BACKEND,
            expected_source_commit=SOURCE,
            expected_provider_contract_id=PROVIDER,
        )
    after = _capture(destination, destination_control, label="destination_before")
    assert after.digest == before.digest


def test_c5_r11_atomic_checkpoint_file_round_trip_and_digest_guard(
    tmp_path: Path,
) -> None:
    source = _fixture(seed=109)
    control = make_c5_graph_run_control_r11_v1(batch=48)
    _advance_window(source, generation=10, tick=1)
    _set_completed_window(control, generation=15, next_tick=6)
    checkpoint = _capture(source, control, label="atomic_round_trip")
    path = tmp_path / "checkpoint.pt"

    receipt = write_c5_graph_checkpoint_r11_v1(checkpoint, path)
    assert receipt["path"] == str(path.resolve())
    assert receipt["bytes"] == path.stat().st_size
    assert receipt["checkpoint_digest"] == checkpoint.digest
    assert len(receipt["sha256"]) == 64

    loaded = load_c5_graph_checkpoint_r11_v1(
        path, expected_sha256=str(receipt["sha256"])
    )
    assert loaded.digest == checkpoint.digest
    assert loaded.metadata == {"label": "atomic_round_trip"}

    with pytest.raises(TensorWorldContractError, match="overwrite"):
        write_c5_graph_checkpoint_r11_v1(checkpoint, path)
    with pytest.raises(TensorWorldContractError, match="file digest differs"):
        load_c5_graph_checkpoint_r11_v1(path, expected_sha256="0" * 64)
