"""Opt-in sequential CUDA continuation evidence for the R11 graph owner.

The executable test keeps every host operation at an explicit graph-window
boundary.  Normal execution remains one prepared five-tick graph replay.  The
default geometry is B48/C32; setting ``EIEL_C4_R11_SEQUENTIAL_CAPACITIES=512``
selects the admitted C512 geometry without changing the harness.
"""

from __future__ import annotations

import ast
from dataclasses import fields, is_dataclass
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess

import pytest
import torch

from engine.tensor_world.c4_event_ring_cuda_boundary_r11 import (
    C4_EVENT_RING_CUDA_BOUNDARY_R11_CHECKPOINT_V1,
    accept_c4_event_ring_boundary_r11_v1,
    prepare_c4_event_ring_checkpoint_boundary_r11_v1,
)
from engine.tensor_world.c4_event_ring_r11 import (
    TensorWorldC4EventRingRowR11V1,
    make_c4_event_ring_r11_v1,
)
from engine.tensor_world.c4_fixture import make_c4_fixture_v1
from engine.tensor_world.c4_mating import TensorWorldC4MateBirthConfigV1
from engine.tensor_world.c4_resident_composed_r11 import (
    bind_c4_resident_composed_r11_v1,
    capture_c4_resident_composed_r11_window_v1,
    execute_c4_resident_composed_r11_window_cuda_v1,
    make_c4_resident_composed_r11_staging_v1,
    prepare_c4_resident_composed_r11_window_control_v1,
)
from engine.tensor_world.c5_determinism import make_c5_pcg64_context_v1
from engine.tensor_world.c5_graph_checkpoint_r11 import (
    bind_c5_graph_run_control_r11_v1,
    capture_c5_graph_checkpoint_r11_v1,
    restore_c5_graph_checkpoint_r11_v1,
)
from engine.tensor_world.c5_graph_fault_boundary_r11 import (
    accept_c5_graph_checkpoint_at_control_boundary_r11_v1,
    capture_c5_graph_checkpoint_at_control_boundary_r11_v1,
    make_c5_graph_fault_boundary_owner_r11_v1,
)


ROOT = Path(__file__).resolve().parents[1]
BACKEND = "tensor-world-r11-sequential-cuda"
PROVIDER = "tensor-world-r11-sequential-continuation-v1"
WINDOW_TICKS = 5
EVENT_BANK_CAPACITY = 16
PREHISTORY_ROWS = EVENT_BANK_CAPACITY * 2
DISABLED_REPRO_THRESHOLD = 1.0e30


def _parse_probe_capacities(value: str) -> tuple[int, ...]:
    try:
        capacities = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise ValueError("R11 sequential capacities must be integers") from exc
    if not capacities or any(capacity not in (32, 512) for capacity in capacities):
        raise ValueError("R11 sequential capacities must be a nonempty subset of 32,512")
    if len(set(capacities)) != len(capacities):
        raise ValueError("R11 sequential capacities must not contain duplicates")
    return capacities


PROBE_CAPACITIES = _parse_probe_capacities(
    os.environ.get("EIEL_C4_R11_SEQUENTIAL_CAPACITIES", "32")
)


def _move_in_place(value: object, device: torch.device) -> object:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, tuple):
        return tuple(_move_in_place(item, device) for item in value)
    if isinstance(value, list):
        return [_move_in_place(item, device) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            setattr(
                value,
                field.name,
                _move_in_place(getattr(value, field.name), device),
            )
    return value


def _event_row(*, ordinal: int) -> TensorWorldC4EventRingRowR11V1:
    return TensorWorldC4EventRingRowR11V1(
        event_type=1,
        generation=19,
        tick=190 + ordinal // 8,
        actor_id=10_000 + ordinal,
        target_id=-1,
        value=float(ordinal),
        actor_slot=ordinal % 4,
        founder_id=10_000,
    )


def _make_cuda_state(*, capacity: int, seed: int) -> tuple[object, object]:
    bundle, _ = make_c4_fixture_v1(
        batch=48,
        capacity=capacity,
        active=4,
        width=8,
        seed=seed,
        world_local_identity=True,
    )
    bundle.base.base.core.agents.health[:, :4] = 1_000.0
    bundle.base.base.core.agents.energy[:, :4] = 1_000.0
    bundle.base.agent.policy_energy_observation[:, :4] = 1_000.0
    bundle.base.agent.repro_threshold.fill_(DISABLED_REPRO_THRESHOLD)
    bundle.base.genome.reproduction_threshold.fill_(DISABLED_REPRO_THRESHOLD)
    replay = make_c5_pcg64_context_v1(
        bundle,
        streams=2,
        seed=seed,
        typed=True,
    )
    bundle.event_ring = make_c4_event_ring_r11_v1(
        batch=48,
        capacity=capacity,
        event_capacity=EVENT_BANK_CAPACITY,
    )
    rows = tuple(_event_row(ordinal=ordinal) for ordinal in range(PREHISTORY_ROWS))
    for world in range(bundle.batch):
        bundle.event_ring.append(world=world, rows=rows)
    replay.event_sequence.copy_(bundle.event_ring.producer_cursor)
    bundle.validate()
    replay.validate_typed_continuation_for_bundle(bundle)
    return (
        _move_in_place(bundle, torch.device("cuda")),
        _move_in_place(replay, torch.device("cuda")),
    )


def _bind_owner(
    *,
    capacity: int,
    seed: int,
    config: TensorWorldC4MateBirthConfigV1,
    c2_path: str,
    build_dir: Path,
) -> tuple[object, object, object, object]:
    bundle, replay = _make_cuda_state(capacity=capacity, seed=seed)
    staging = make_c4_resident_composed_r11_staging_v1(
        batch=48,
        capacity=capacity,
        grid_size=8,
        streams=int(replay.rng_state_words.shape[1]),
        device="cuda",
    )
    owner = bind_c4_resident_composed_r11_v1(
        bundle,
        replay,
        staging,
        enable_native=True,
        population_config=config,
        c2_extension_path=c2_path,
        native_build_directory=build_dir,
    )
    return owner, bundle, replay, staging


def _source_commit() -> str:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert len(commit) == 40
    return commit


def _graph_fingerprint(owner: object, *, capacity: int, source_commit: str) -> str:
    assert owner.window_graph is not None and owner.r11_extension is not None
    nodes = int(
        owner.r11_extension.captured_graph_node_count(
            owner.window_graph.raw_cuda_graph()
        )
    )
    assert nodes > 0
    payload = json.dumps(
        {
            "capacity": capacity,
            "graph_nodes": nodes,
            "source_commit": source_commit,
            "ticks": WINDOW_TICKS,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _inject_one_death_per_world(bundle: object) -> None:
    bundle.base.lifecycle.death_candidate.zero_()
    bundle.base.lifecycle.death_candidate[:, 1] = True
    bundle.base.base.core.agents.health[:, 1] = 0.0


def _assert_monotonic_materialized_ring(ring: object) -> None:
    ring.validate(semantic=True)
    for world in range(ring.batch):
        consumer = int(ring.consumer_cursor[world])
        producer = int(ring.producer_cursor[world])
        sequences: list[int] = []
        for bank_index, bank in enumerate(ring.banks):
            count = int(ring.bank_counts[world, bank_index])
            start = int(ring.bank_start_sequences[world, bank_index])
            if count == 0 or start + count <= consumer:
                continue
            sequences.extend(
                int(value)
                for value in bank.typed_sequence[world, :count].tolist()
                if consumer <= int(value) < producer
            )
        assert sorted(sequences) == list(range(consumer, producer))


def test_r11_sequential_harness_is_parameterizable_and_boundary_only() -> None:
    assert _parse_probe_capacities("32") == (32,)
    assert _parse_probe_capacities("512") == (512,)
    assert _parse_probe_capacities("32,512") == (32, 512)
    with pytest.raises(ValueError, match="subset"):
        _parse_probe_capacities("64")
    with pytest.raises(ValueError, match="duplicates"):
        _parse_probe_capacities("32,32")

    source = Path(__file__).read_text(encoding="utf-8")
    module = ast.parse(source)
    function = next(
        node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "test_r11_sequential_cuda_continuation"
    )
    executable = ast.get_source_segment(source, function)
    assert executable is not None
    assert executable.count(
        "prepare_c4_resident_composed_r11_window_control_v1("
    ) == 3
    assert executable.count("execute_c4_resident_composed_r11_window_cuda_v1(") == 3
    assert executable.count("capture_c4_resident_composed_r11_window_v1(") == 2
    assert "execute_c4_resident_composed_r11_window_compat_cuda_v1" not in executable
    assert "execute_c4_resident_composed_r11_cuda_v1" not in executable
    cursor = 0
    for _ in range(3):
        prepare_at = executable.index(
            "prepare_c4_resident_composed_r11_window_control_v1(", cursor
        )
        execute_at = executable.index(
            "execute_c4_resident_composed_r11_window_cuda_v1(", prepare_at
        )
        assert "torch.cuda.synchronize()" in executable[prepare_at:execute_at]
        cursor = execute_at + 1

    fixture_source = inspect.getsource(_make_cuda_state)
    assert 'float("inf")' not in fixture_source
    assert fixture_source.index("make_c5_pcg64_context_v1(") < fixture_source.index(
        "bundle.event_ring = make_c4_event_ring_r11_v1("
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.skipif(
    os.environ.get("EIEL_C4_R11_SEQUENTIAL_CONTINUATION_PROBE") != "1",
    reason="R11 sequential continuation probe is explicitly opt-in",
)
@pytest.mark.parametrize("capacity", PROBE_CAPACITIES)
def test_r11_sequential_cuda_continuation(
    capacity: int,
    tmp_path: Path,
) -> None:
    """Prove accepted-checkpoint continuation through physical owner release."""

    source_commit = _source_commit()
    config = TensorWorldC4MateBirthConfigV1(
        max_births=0,
        allow_mutation=True,
        max_energy=1_000.0,
    )
    c2_path = os.environ.get(
        "TENSOR_WORLD_C2_PREBUILT",
        "/tmp/emergence-test/runtime/evidence_raw/tensor_world/production_c0_c7/"
        "20260815_c2_430489534_build_01/tensor_world_c2_ext.so",
    )
    build_dir = Path(os.environ.get("EIEL_C4_R11_BUILD_DIR", tmp_path / "build"))

    source_owner, source_bundle, source_replay, source_staging = _bind_owner(
        capacity=capacity,
        seed=401,
        config=config,
        c2_path=c2_path,
        build_dir=build_dir,
    )
    capture_c4_resident_composed_r11_window_v1(source_owner, config)
    source_lifecycle = make_c5_graph_fault_boundary_owner_r11_v1(
        owner_id=f"source-B48-C{capacity}"
    )
    source_lifecycle.record_graph_capture_at_control_boundary(
        graph_capture_fingerprint=_graph_fingerprint(
            source_owner,
            capacity=capacity,
            source_commit=source_commit,
        )
    )
    assert source_owner.c2_config_ints is not None
    assert source_owner.c2_config_floats is not None
    c2_config_ptrs = (
        source_owner.c2_config_ints.data_ptr(),
        source_owner.c2_config_floats.data_ptr(),
    )
    prepare_c4_resident_composed_r11_window_control_v1(
        source_owner,
        generation=20,
        tick=200,
    )
    torch.cuda.synchronize()
    source_result = execute_c4_resident_composed_r11_window_cuda_v1(source_owner)
    torch.cuda.synchronize()
    assert source_result.global_status.cpu().tolist() == [0] * 48
    assert source_result.completed_ticks.cpu().tolist() == [WINDOW_TICKS] * 48

    source_control = bind_c5_graph_run_control_r11_v1(source_owner.shadow_staging)
    checkpoint = capture_c5_graph_checkpoint_at_control_boundary_r11_v1(
        source_lifecycle,
        source_bundle,
        source_replay,
        source_control,
        backend_id=BACKEND,
        source_commit=source_commit,
        provider_contract_id=PROVIDER,
        metadata={"phase": "accepted_first_window", "capacity": capacity},
    )
    assert (
        source_owner.c2_config_ints.data_ptr(),
        source_owner.c2_config_floats.data_ptr(),
    ) == c2_config_ptrs
    acceptance_path = tmp_path / f"accepted_B48_C{capacity}.json"
    acceptance_path.write_text(
        json.dumps(
            {
                "checkpoint_digest": checkpoint.digest,
                "source_commit": source_commit,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    assert acceptance_path.read_text(encoding="utf-8")
    accept_c5_graph_checkpoint_at_control_boundary_r11_v1(
        source_lifecycle,
        checkpoint,
    )
    checkpoint_ring = checkpoint.base_checkpoint.state.event_ring
    assert checkpoint_ring is not None
    assert checkpoint_ring.producer_cursor.tolist() == [PREHISTORY_ROWS] * 48
    _assert_monotonic_materialized_ring(checkpoint_ring)

    source_candidate = prepare_c4_event_ring_checkpoint_boundary_r11_v1(
        source_owner,
        checkpoint,
    )
    assert (
        source_candidate.acceptance_kind
        == C4_EVENT_RING_CUDA_BOUNDARY_R11_CHECKPOINT_V1
    )
    source_ack = accept_c4_event_ring_boundary_r11_v1(
        source_owner,
        source_candidate,
        accepted_artifact_digest=checkpoint.digest,
    )
    torch.cuda.synchronize()
    assert source_ack.cpu().tolist() == [1]
    assert source_bundle.event_ring.consumer_cursor.cpu().tolist() == [
        PREHISTORY_ROWS
    ] * 48

    _inject_one_death_per_world(source_bundle)
    prepare_c4_resident_composed_r11_window_control_v1(
        source_owner,
        generation=25,
        tick=205,
    )
    torch.cuda.synchronize()
    source_result = execute_c4_resident_composed_r11_window_cuda_v1(source_owner)
    torch.cuda.synchronize()
    assert source_result.global_status.cpu().tolist() == [0] * 48
    reference_terminal = capture_c5_graph_checkpoint_r11_v1(
        source_bundle,
        source_replay,
        bind_c5_graph_run_control_r11_v1(source_owner.shadow_staging),
        backend_id=BACKEND,
        source_commit=source_commit,
        provider_contract_id=PROVIDER,
        metadata={"phase": "terminal_second_window", "capacity": capacity},
    )
    reference_ring = reference_terminal.base_checkpoint.state.event_ring
    assert reference_ring is not None
    assert reference_ring.consumer_cursor.tolist() == [PREHISTORY_ROWS] * 48
    assert bool((reference_ring.producer_cursor > PREHISTORY_ROWS).all())
    assert reference_ring.active_bank.tolist() == [0] * 48
    assert reference_ring.bank_start_sequences[:, 0].tolist() == [
        PREHISTORY_ROWS
    ] * 48
    assert reference_ring.banks[0].typed_sequence[:, 0].tolist() == [
        PREHISTORY_ROWS
    ] * 48
    _assert_monotonic_materialized_ring(reference_ring)

    # The canonical graph owner and every CUDA alias are retired before the
    # continuation owner is allocated.  The accepted checkpoint is CPU-owned.
    del source_result, source_ack, source_control, source_candidate
    del source_owner, source_bundle, source_replay, source_staging
    gc.collect()
    torch.cuda.empty_cache()

    fresh_owner, fresh_bundle, fresh_replay, fresh_staging = _bind_owner(
        capacity=capacity,
        seed=409,
        config=config,
        c2_path=c2_path,
        build_dir=build_dir,
    )
    fresh_control = bind_c5_graph_run_control_r11_v1(fresh_owner.shadow_staging)
    restored_digest = restore_c5_graph_checkpoint_r11_v1(
        checkpoint,
        fresh_bundle,
        fresh_replay,
        fresh_control,
        expected_backend_id=BACKEND,
        expected_source_commit=source_commit,
        expected_provider_contract_id=PROVIDER,
    )
    assert restored_digest == checkpoint.digest
    capture_c4_resident_composed_r11_window_v1(fresh_owner, config)
    # Graph capture initializes the fresh owner's private run-control arena.
    # Reapply the accepted checkpoint at this same declared control boundary
    # before the first normal replay; no tick path performs this restore.
    restored_digest = restore_c5_graph_checkpoint_r11_v1(
        checkpoint,
        fresh_bundle,
        fresh_replay,
        fresh_control,
        expected_backend_id=BACKEND,
        expected_source_commit=source_commit,
        expected_provider_contract_id=PROVIDER,
    )
    assert restored_digest == checkpoint.digest
    recaptured = capture_c5_graph_checkpoint_r11_v1(
        fresh_bundle,
        fresh_replay,
        fresh_control,
        backend_id=BACKEND,
        source_commit=source_commit,
        provider_contract_id=PROVIDER,
        metadata={"phase": "accepted_first_window", "capacity": capacity},
    )
    assert recaptured.digest == checkpoint.digest

    fresh_candidate = prepare_c4_event_ring_checkpoint_boundary_r11_v1(
        fresh_owner,
        checkpoint,
    )
    fresh_ack = accept_c4_event_ring_boundary_r11_v1(
        fresh_owner,
        fresh_candidate,
        accepted_artifact_digest=checkpoint.digest,
    )
    torch.cuda.synchronize()
    assert fresh_ack.cpu().tolist() == [1]

    _inject_one_death_per_world(fresh_bundle)
    prepare_c4_resident_composed_r11_window_control_v1(
        fresh_owner,
        generation=25,
        tick=205,
    )
    torch.cuda.synchronize()
    fresh_result = execute_c4_resident_composed_r11_window_cuda_v1(fresh_owner)
    torch.cuda.synchronize()
    assert fresh_result.global_status.cpu().tolist() == [0] * 48
    fresh_terminal = capture_c5_graph_checkpoint_r11_v1(
        fresh_bundle,
        fresh_replay,
        bind_c5_graph_run_control_r11_v1(fresh_owner.shadow_staging),
        backend_id=BACKEND,
        source_commit=source_commit,
        provider_contract_id=PROVIDER,
        metadata={"phase": "terminal_second_window", "capacity": capacity},
    )
    assert fresh_terminal.digest == reference_terminal.digest
    assert fresh_terminal.base_checkpoint.digest == (
        reference_terminal.base_checkpoint.digest
    )
    fresh_ring = fresh_terminal.base_checkpoint.state.event_ring
    assert fresh_ring is not None
    assert fresh_ring.producer_cursor.tolist() == reference_ring.producer_cursor.tolist()
    _assert_monotonic_materialized_ring(fresh_ring)
