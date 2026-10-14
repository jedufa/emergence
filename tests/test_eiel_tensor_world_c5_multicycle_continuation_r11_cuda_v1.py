"""Opt-in C5 multi-cycle exact checkpoint/replay/resume CUDA evidence.

Normal execution is only a prepared five-tick graph replay.  Host work in this
harness is confined to declared initialization, control, checkpoint, report,
or restore boundaries and is timed separately from the steady replay window.
"""

from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass
import gc
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import time
from typing import Mapping

import pytest
import torch

from engine.tensor_world.c4_event_ring_cuda_boundary_r11 import (
    accept_c4_event_ring_boundary_r11_v1,
    prepare_c4_event_ring_checkpoint_boundary_r11_v1,
)
from engine.tensor_world.c4_event_ring_r11 import make_c4_event_ring_r11_v1
from engine.tensor_world.c4_events import (
    C4_EVENT_BIRTH_COMMITTED_V1,
    C4_EVENT_DEATH_COMMITTED_V1,
    C4_EVENT_MUTATION_V1,
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
from engine.tensor_world.c4_state import make_c4_cpython_random_state_v1
from engine.tensor_world.c5_determinism import make_c5_pcg64_context_v1
from engine.tensor_world.c5_determinism import (
    C5_GAUSSIAN_WINDOW_EXHAUSTED_R11_V1,
    C5_GAUSSIAN_WINDOW_REFILL_IN_PROGRESS_R11_V1,
)
from engine.tensor_world.c5_gaussian_window_r11 import (
    admit_c5_gaussian_window_r11_v1,
)
from engine.tensor_world.c5_graph_checkpoint_r11 import (
    bind_c5_graph_run_control_r11_v1,
    capture_c5_graph_checkpoint_r11_v1,
    load_c5_graph_checkpoint_r11_v1,
    restore_c5_graph_checkpoint_r11_v1,
    write_c5_graph_checkpoint_r11_v1,
)


ROOT = Path(__file__).resolve().parents[1]
BACKEND = "tensor-world-c5-r11-multicycle-cuda"
PROVIDER = "tensor-world-c5-r11-multicycle-continuation-v1"
WINDOW_TICKS = 5
CAPACITY = 32
TAPE_CAPACITY = 512
EVENT_CAPACITY = 16
C512_EXACT = ROOT / (
    "tests/test_eiel_tensor_world_c4_resident_composed_r11_c512_qualification_v1.py"
)


def _source_commit() -> str:
    value = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert len(value) == 40
    return value


def _move_in_place(value: object, device: torch.device) -> object:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, tuple):
        return tuple(_move_in_place(item, device) for item in value)
    if isinstance(value, list):
        return [_move_in_place(item, device) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            setattr(value, field.name, _move_in_place(getattr(value, field.name), device))
    return value


def _make_cpu_fixture(
    *,
    seed: int,
    source_commit: str,
) -> tuple[object, object, object, float]:
    bundle, _ = make_c4_fixture_v1(
        batch=48,
        capacity=CAPACITY,
        active=16,
        width=8,
        seed=seed,
        world_local_identity=True,
    )
    core = bundle.base.base.core
    core.occupancy.agent_id_grid.fill_(-1)
    core.occupancy.slot_grid.fill_(-1)
    bundle.base.base.patch.agent_grid_i32.fill_(-1)
    for slot in range(16):
        pair = slot // 2
        y = (pair // 2) * 2
        x = (pair % 2) * 4 + (slot & 1)
        core.agents.pos_y[:, slot] = y
        core.agents.pos_x[:, slot] = x
        core.occupancy.agent_id_grid[:, y, x] = core.agents.agent_id[:, slot]
        core.occupancy.slot_grid[:, y, x] = slot
        bundle.base.base.patch.agent_grid_i32[:, y, x] = core.agents.agent_id[
            :, slot
        ].to(torch.int32)
    core.agents.energy[:, :16] = 160.0
    core.agents.health[:, :16] = 12.0
    bundle.base.agent.policy_energy_observation[:, :16] = 160.0
    bundle.base.agent.repro_threshold[:, :16] = 0.0
    bundle.base.genome.reproduction_threshold[:, :16] = 0.0
    bundle.base.genome.mutation_rate[:, :16] = 1.0
    for packed in (
        core.policy.packed_weights,
        bundle.base.base.policy.packed_weights,
    ):
        packed[..., -7:] = -20.0
        packed[..., -2] = 20.0
    config = TensorWorldC4MateBirthConfigV1(
        max_births=1,
        parent_energy_share=0.1,
        cooldown_steps=0,
        max_energy=160.0,
        allow_mutation=True,
    )
    bundle.scalar_rng = make_c4_cpython_random_state_v1(
        batch=48, tape_capacity=TAPE_CAPACITY, seed=seed
    )
    bundle.event_ring = make_c4_event_ring_r11_v1(
        batch=48, capacity=CAPACITY, event_capacity=EVENT_CAPACITY
    )
    replay = make_c5_pcg64_context_v1(
        bundle,
        streams=2,
        seed=seed,
        typed=True,
        source_commit=source_commit,
    )
    initial_eed_sum = float(bundle.base.base.eed.resource_a_amount.sum())
    bundle.validate()
    replay.validate_typed_continuation_for_bundle(bundle)
    return bundle, replay, config, initial_eed_sum


def _make_owner(
    *,
    seed: int,
    source_commit: str,
    c2_path: str,
    build_dir: Path,
) -> tuple[object, object, object, object, object, float, float]:
    bundle, replay, config, initial_eed_sum = _make_cpu_fixture(
        seed=seed, source_commit=source_commit
    )
    bundle = _move_in_place(bundle, torch.device("cuda"))
    replay = _move_in_place(replay, torch.device("cuda"))
    staging = make_c4_resident_composed_r11_staging_v1(
        batch=48,
        capacity=CAPACITY,
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
        source_commit=source_commit,
    )
    started = time.perf_counter()
    capture_c4_resident_composed_r11_window_v1(owner, config)
    torch.cuda.synchronize()
    capture_ms = (time.perf_counter() - started) * 1_000.0
    return owner, bundle, replay, staging, config, initial_eed_sum, capture_ms


def _inject_one_death_per_world(bundle: object, *, slot: int) -> None:
    bundle.base.lifecycle.death_candidate.zero_()
    bundle.base.lifecycle.death_candidate[:, slot] = True
    bundle.base.base.core.agents.health[:, slot] = 0.0


def _checkpoint(owner: object, *, phase: str, source_commit: str) -> object:
    return capture_c5_graph_checkpoint_r11_v1(
        owner.bundle,
        owner.replay,
        bind_c5_graph_run_control_r11_v1(owner.shadow_staging),
        backend_id=BACKEND,
        source_commit=source_commit,
        provider_contract_id=PROVIDER,
        metadata={"capacity": CAPACITY, "phase": phase},
    )


def _run_window(
    owner: object,
    *,
    generation: int,
    tick: int,
    death_slot: int,
    phase: str,
    source_commit: str,
) -> tuple[object, dict[str, object]]:
    _inject_one_death_per_world(owner.bundle, slot=death_slot)
    started = time.perf_counter()
    prepare_c4_resident_composed_r11_window_control_v1(
        owner, generation=generation, tick=tick
    )
    torch.cuda.synchronize()
    prepare_ms = (time.perf_counter() - started) * 1_000.0
    admission = owner.last_gaussian_window_admission
    assert admission is not None

    started = time.perf_counter()
    result = execute_c4_resident_composed_r11_window_cuda_v1(owner)
    torch.cuda.synchronize()
    replay_ms = (time.perf_counter() - started) * 1_000.0
    assert result.global_status.cpu().tolist() == [0] * 48
    assert result.completed_ticks.cpu().tolist() == [WINDOW_TICKS] * 48

    started = time.perf_counter()
    checkpoint = _checkpoint(owner, phase=phase, source_commit=source_commit)
    checkpoint_ms = (time.perf_counter() - started) * 1_000.0
    return checkpoint, {
        "phase": phase,
        "prepare_ms": prepare_ms,
        "steady_window_replay_ms": replay_ms,
        "steady_ms_per_tick": replay_ms / WINDOW_TICKS,
        "checkpoint_capture_ms": checkpoint_ms,
        "checkpoint_capture_amortized_ms_per_tick": checkpoint_ms / WINDOW_TICKS,
        "gaussian_admission": admission.as_dict(),
    }


def _ack(owner: object, checkpoint: object) -> None:
    candidate = prepare_c4_event_ring_checkpoint_boundary_r11_v1(owner, checkpoint)
    status = accept_c4_event_ring_boundary_r11_v1(
        owner, candidate, accepted_artifact_digest=checkpoint.digest
    )
    torch.cuda.synchronize()
    assert status.cpu().tolist() == [1]


def _assert_lifecycle(checkpoint: object, *, initial_eed_sum: float) -> None:
    state = checkpoint.base_checkpoint.state
    replay = checkpoint.base_checkpoint.replay_context
    ring = state.event_ring
    scalar = state.scalar_rng
    assert ring is not None and scalar is not None and replay.gaussian_window is not None
    rows = ring.rows(world=0)
    event_types = {row.event_type for row in rows}
    assert {
        C4_EVENT_DEATH_COMMITTED_V1,
        C4_EVENT_MUTATION_V1,
        C4_EVENT_BIRTH_COMMITTED_V1,
    }.issubset(event_types)
    births = [row for row in rows if row.event_type == C4_EVENT_BIRTH_COMMITTED_V1]
    assert births
    assert len({row.child_id for row in births}) == len(births)
    assert all(
        row.parent_id >= 0
        and row.child_id > row.parent_id
        and row.founder_id >= 0
        and row.lineage_depth >= 1
        for row in births
    )
    assert torch.equal(replay.event_sequence, ring.producer_cursor)
    assert int(state.population.next_agent_id[0]) > 17
    assert not torch.equal(
        state.population.slot_tokens[0], torch.arange(CAPACITY, dtype=torch.int64)
    )
    assert int(scalar.python_random_raw_draws[0]) > 0
    assert int(scalar.mutation_py_semantic_draws[0]) > 0
    assert int(state.base.base.wdl.next_absolute_step[0]) >= WINDOW_TICKS
    assert float(state.base.base.eed.resource_a_amount.sum()) != initial_eed_sum


def _restore_owner(
    checkpoint: object,
    *,
    source_commit: str,
    c2_path: str,
    build_dir: Path,
) -> tuple[object, object, dict[str, float]]:
    owner, bundle, replay, _staging, config, _eed, capture_ms = _make_owner(
        seed=991,
        source_commit=source_commit,
        c2_path=c2_path,
        build_dir=build_dir,
    )
    control = bind_c5_graph_run_control_r11_v1(owner.shadow_staging)
    started = time.perf_counter()
    restore_c5_graph_checkpoint_r11_v1(
        checkpoint,
        bundle,
        replay,
        control,
        expected_backend_id=BACKEND,
        expected_source_commit=source_commit,
        expected_provider_contract_id=PROVIDER,
    )
    restore_ms = (time.perf_counter() - started) * 1_000.0
    # The fresh graph was captured before this explicit restore boundary; the
    # restored canonical tensors are therefore authoritative for first replay.
    owner.window_control_initialized = True
    return owner, config, {"restore_ms": restore_ms, "graph_recapture_ms": capture_ms}


def _release_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _canonical_byte_digest(bundle: object, replay: object) -> str:
    entries: list[tuple[str, torch.Tensor]] = []

    def collect(value: object, path: str) -> None:
        if isinstance(value, torch.Tensor):
            entries.append((path, value))
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                collect(item, f"{path}[{index}]")
        elif is_dataclass(value) and not isinstance(value, type):
            for field in fields(value):
                collect(getattr(value, field.name), f"{path}.{field.name}")

    collect(bundle, "bundle")
    collect(replay, "replay")
    digest = hashlib.sha256()
    for path, tensor in sorted(entries):
        host = tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu()
        digest.update(path.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(host.numpy().tobytes())
    return digest.hexdigest()


def _assert_maps_byte_exact_with_first_difference(
    expected: Mapping[str, torch.Tensor], actual: Mapping[str, torch.Tensor]
) -> None:
    assert expected.keys() == actual.keys()
    for path in expected:
        left = expected[path].detach().contiguous().cpu()
        right = actual[path].detach().contiguous().cpu()
        assert left.dtype == right.dtype, path
        assert left.shape == right.shape, path
        left_bytes = left.reshape(-1).view(torch.uint8)
        right_bytes = right.reshape(-1).view(torch.uint8)
        if torch.equal(left_bytes, right_bytes):
            continue
        differing_byte = int((left_bytes != right_bytes).nonzero()[0])
        element_size = left.element_size()
        flat_element = differing_byte // element_size
        index: list[int] = []
        remainder = flat_element
        for extent in reversed(left.shape):
            index.append(remainder % int(extent))
            remainder //= int(extent)
        tensor_index = tuple(reversed(index))
        left_value = left[tensor_index].item()
        right_value = right[tensor_index].item()
        left_bits = bytes(
            left.reshape(-1).view(torch.uint8)[
                flat_element * element_size : (flat_element + 1) * element_size
            ].tolist()
        ).hex()
        right_bits = bytes(
            right.reshape(-1).view(torch.uint8)[
                flat_element * element_size : (flat_element + 1) * element_size
            ].tolist()
        ).hex()
        raise AssertionError(
            f"{path} differs at {tensor_index}: expected={left_value!r} "
            f"actual={right_value!r} expected_le={left_bits} actual_le={right_bits}"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.skipif(
    os.environ.get("EIEL_C5_R11_MULTICYCLE_QUALIFICATION") != "1",
    reason="C5 multi-cycle qualification is explicitly opt-in",
)
def test_c5_r11_post_refill_windows_are_exact_against_cpu_oracle(
    tmp_path: Path,
) -> None:
    """Compare every canonical byte after two nonzero-base exact refills."""

    source_commit = _source_commit()
    c2_path = os.environ.get(
        "TENSOR_WORLD_C2_PREBUILT",
        "/tmp/emergence-test/runtime/evidence_raw/tensor_world/production_c0_c7/"
        "20260815_c2_430489534_build_01/tensor_world_c2_ext.so",
    )
    build_dir = Path(os.environ.get("EIEL_C4_R11_BUILD_DIR", tmp_path / "build"))
    reference_bundle, reference_replay, config, _eed = _make_cpu_fixture(
        seed=433, source_commit=source_commit
    )
    helpers = runpy.run_path(str(C512_EXACT), run_name="_c5_refill_exact_helpers")
    run_reference = helpers["_run_host_orchestrated_reference_window"]
    canonical_leaves = helpers["_canonical_leaves"]
    run_control_leaves = helpers["_run_control_leaves"]
    refilled: list[bool] = []
    window_bases: list[int] = []
    reference_snapshots: list[tuple[object, object, object, object]] = []

    for index, (generation, tick, death_slot) in enumerate(
        ((20, 200, 1), (25, 205, 2), (30, 210, 3))
    ):
        _inject_one_death_per_world(reference_bundle, slot=death_slot)
        reference_admission = admit_c5_gaussian_window_r11_v1(
            reference_bundle,
            reference_replay,
            requested_ticks=WINDOW_TICKS,
            max_births_per_tick=1,
            boundary_kind="initialization" if index == 0 else "control",
            source_commit=source_commit,
            boundary_label=f"generation_{generation}_tick_{tick}",
        )
        if reference_admission.refill_performed:
            assert (
                reference_admission.base_raw_draws_by_world
                == reference_admission.absolute_raw_draws_by_world
            )
        window_bases.append(reference_admission.base_raw_draws_by_world[0])
        reference_bundle, reference_replay, reference_control = run_reference(
            reference_bundle,
            reference_replay,
            config=config,
            generation=generation,
            tick=tick,
            c2_extension_path=c2_path,
            move_in_place=_move_in_place,
        )
        reference_snapshots.append(
            (
                copy.deepcopy(reference_bundle),
                copy.deepcopy(reference_replay),
                copy.deepcopy(reference_control),
                reference_admission,
            )
        )
        refilled.append(reference_admission.refill_performed)
        if index < 2:
            ring = reference_bundle.event_ring
            assert ring is not None
            ring.consumer_cursor.copy_(ring.producer_cursor)

    assert refilled == [False, True, True]
    assert window_bases[0] == 0
    assert 0 < window_bases[1] < window_bases[2]
    assert reference_replay.gaussian_window is not None
    assert int(reference_replay.gaussian_window.base_raw_draws[0]) == window_bases[2]
    del reference_bundle, reference_replay
    _release_cuda()

    owner, bundle, replay, staging, _config, _eed, _capture_ms = _make_owner(
        seed=433,
        source_commit=source_commit,
        c2_path=c2_path,
        build_dir=build_dir,
    )
    for index, (generation, tick, death_slot) in enumerate(
        ((20, 200, 1), (25, 205, 2), (30, 210, 3))
    ):
        (
            expected_bundle,
            expected_replay,
            expected_control,
            expected_admission,
        ) = reference_snapshots[index]
        _inject_one_death_per_world(bundle, slot=death_slot)
        prepare_c4_resident_composed_r11_window_control_v1(
            owner, generation=generation, tick=tick
        )
        candidate_admission = owner.last_gaussian_window_admission
        assert candidate_admission is not None
        assert candidate_admission.as_dict() == expected_admission.as_dict()
        result = execute_c4_resident_composed_r11_window_cuda_v1(owner)
        torch.cuda.synchronize()
        _assert_maps_byte_exact_with_first_difference(
            canonical_leaves(expected_bundle, expected_replay),
            canonical_leaves(bundle, replay),
        )
        _assert_maps_byte_exact_with_first_difference(
            expected_control, run_control_leaves(result)
        )

        if index < 2:
            checkpoint = _checkpoint(
                owner, phase=f"exact_{index}", source_commit=source_commit
            )
            _ack(owner, checkpoint)
    del owner, bundle, replay, staging
    _release_cuda()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.skipif(
    os.environ.get("EIEL_C5_R11_MULTICYCLE_QUALIFICATION") != "1",
    reason="C5 multi-cycle qualification is explicitly opt-in",
)
@pytest.mark.parametrize(
    "fault",
    ("refill_in_progress", "exhausted", "cursor_past_end"),
)
def test_c5_r11_non_authoritative_window_fails_before_canonical_mutation(
    fault: str,
    tmp_path: Path,
) -> None:
    source_commit = _source_commit()
    c2_path = os.environ.get(
        "TENSOR_WORLD_C2_PREBUILT",
        "/tmp/emergence-test/runtime/evidence_raw/tensor_world/production_c0_c7/"
        "20260815_c2_430489534_build_01/tensor_world_c2_ext.so",
    )
    owner, bundle, replay, staging, _config, _eed, _capture_ms = _make_owner(
        seed=457,
        source_commit=source_commit,
        c2_path=c2_path,
        build_dir=Path(os.environ.get("EIEL_C4_R11_BUILD_DIR", tmp_path / "build")),
    )
    prepare_c4_resident_composed_r11_window_control_v1(
        owner, generation=20, tick=200
    )
    window = replay.gaussian_window
    scalar = bundle.scalar_rng
    assert window is not None and scalar is not None
    if fault == "refill_in_progress":
        window.status.fill_(C5_GAUSSIAN_WINDOW_REFILL_IN_PROGRESS_R11_V1)
    elif fault == "exhausted":
        window.status.fill_(C5_GAUSSIAN_WINDOW_EXHAUSTED_R11_V1)
        window.exhaustion_raw_draws.copy_(scalar.python_random_raw_draws)
    else:
        scalar.python_random_raw_draws.copy_(window.valid_raw_draw_ends + 1)
    before = _canonical_byte_digest(bundle, replay)
    result = execute_c4_resident_composed_r11_window_cuda_v1(owner)
    torch.cuda.synchronize()
    assert result.global_status.cpu().tolist() == [1] * 48
    assert result.completed_ticks.cpu().tolist() == [0] * 48
    assert _canonical_byte_digest(bundle, replay) == before
    del owner, bundle, replay, staging
    _release_cuda()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.skipif(
    os.environ.get("EIEL_C5_R11_MULTICYCLE_QUALIFICATION") != "1",
    reason="C5 multi-cycle qualification is explicitly opt-in",
)
def test_c5_r11_multicycle_exact_replay_checkpoint_and_two_resumes(
    tmp_path: Path,
) -> None:
    source_commit = _source_commit()
    c2_path = os.environ.get(
        "TENSOR_WORLD_C2_PREBUILT",
        "/tmp/emergence-test/runtime/evidence_raw/tensor_world/production_c0_c7/"
        "20260815_c2_430489534_build_01/tensor_world_c2_ext.so",
    )
    build_dir = Path(os.environ.get("EIEL_C4_R11_BUILD_DIR", tmp_path / "build"))
    timings: dict[str, object] = {"source_commit": source_commit, "windows": []}

    owner, _bundle, _replay, _staging, _config, eed_sum, capture_ms = _make_owner(
        seed=401,
        source_commit=source_commit,
        c2_path=c2_path,
        build_dir=build_dir,
    )
    timings["source_graph_capture_ms"] = capture_ms
    checkpoint_a, timing_a = _run_window(
        owner,
        generation=20,
        tick=200,
        death_slot=1,
        phase="A",
        source_commit=source_commit,
    )
    _assert_lifecycle(checkpoint_a, initial_eed_sum=eed_sum)

    checkpoint_path = tmp_path / "checkpoint_A.pt"
    started = time.perf_counter()
    file_receipt = write_c5_graph_checkpoint_r11_v1(checkpoint_a, checkpoint_path)
    timings["checkpoint_write_ms"] = (time.perf_counter() - started) * 1_000.0
    started = time.perf_counter()
    loaded_a = load_c5_graph_checkpoint_r11_v1(
        checkpoint_path, expected_sha256=str(file_receipt["sha256"])
    )
    timings["checkpoint_read_ms"] = (time.perf_counter() - started) * 1_000.0
    timings["checkpoint_persistence_amortized_ms_per_tick"] = (
        float(timings["checkpoint_write_ms"]) / WINDOW_TICKS
    )
    assert loaded_a.digest == checkpoint_a.digest

    _ack(owner, checkpoint_a)
    checkpoint_b, timing_b = _run_window(
        owner,
        generation=25,
        tick=205,
        death_slot=2,
        phase="B",
        source_commit=source_commit,
    )
    _assert_lifecycle(checkpoint_b, initial_eed_sum=eed_sum)
    _ack(owner, checkpoint_b)
    checkpoint_c, timing_c = _run_window(
        owner,
        generation=30,
        tick=210,
        death_slot=3,
        phase="C",
        source_commit=source_commit,
    )
    _assert_lifecycle(checkpoint_c, initial_eed_sum=eed_sum)
    admissions = [
        timing_a["gaussian_admission"],
        timing_b["gaussian_admission"],
        timing_c["gaussian_admission"],
    ]
    assert admissions[0]["refill_performed"] is False
    assert any(item["refill_performed"] for item in admissions[1:])
    assert checkpoint_a.digest != checkpoint_b.digest != checkpoint_c.digest
    del owner, _bundle, _replay, _staging, _config
    _release_cuda()

    repeat, *_repeat_values, repeat_eed, repeat_capture_ms = _make_owner(
        seed=401,
        source_commit=source_commit,
        c2_path=c2_path,
        build_dir=build_dir,
    )
    repeated_a, repeated_timing = _run_window(
        repeat,
        generation=20,
        tick=200,
        death_slot=1,
        phase="A",
        source_commit=source_commit,
    )
    assert repeated_a.digest == checkpoint_a.digest
    _assert_lifecycle(repeated_a, initial_eed_sum=repeat_eed)
    timings["repeat_graph_capture_ms"] = repeat_capture_ms
    del repeat, _repeat_values
    _release_cuda()

    resumed_b_owner, _config_b, resume_b_timing = _restore_owner(
        loaded_a,
        source_commit=source_commit,
        c2_path=c2_path,
        build_dir=build_dir,
    )
    _ack(resumed_b_owner, loaded_a)
    resumed_b, resumed_b_window = _run_window(
        resumed_b_owner,
        generation=25,
        tick=205,
        death_slot=2,
        phase="B",
        source_commit=source_commit,
    )
    assert resumed_b.digest == checkpoint_b.digest
    del resumed_b_owner, _config_b
    _release_cuda()

    resumed_c_owner, _config_c, resume_c_timing = _restore_owner(
        checkpoint_b,
        source_commit=source_commit,
        c2_path=c2_path,
        build_dir=build_dir,
    )
    _ack(resumed_c_owner, checkpoint_b)
    resumed_c, resumed_c_window = _run_window(
        resumed_c_owner,
        generation=30,
        tick=210,
        death_slot=3,
        phase="C",
        source_commit=source_commit,
    )
    assert resumed_c.digest == checkpoint_c.digest
    del resumed_c_owner, _config_c
    _release_cuda()

    timings["windows"] = [
        timing_a,
        timing_b,
        timing_c,
        repeated_timing,
        resumed_b_window,
        resumed_c_window,
    ]
    timings["resume_from_A"] = resume_b_timing
    timings["resume_from_B"] = resume_c_timing
    timings["checkpoint_file"] = file_receipt
    timings["digests"] = {
        "A": checkpoint_a.digest,
        "B": checkpoint_b.digest,
        "C": checkpoint_c.digest,
    }
    timings["result"] = "PASS"
    receipt_path = os.environ.get("EIEL_C5_MULTICYCLE_RECEIPT")
    if receipt_path:
        Path(receipt_path).write_text(
            json.dumps(timings, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
