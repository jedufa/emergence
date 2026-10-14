"""Opt-in target-geometry resource/performance probe for the R11 graph owner."""

from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

import pytest
import torch

from engine.tensor_world.c4_event_ring_r11 import make_c4_event_ring_r11_v1
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


ROOT = Path(__file__).resolve().parents[1]


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


def _percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _cuda_graph_node_count(owner: object) -> int:
    """Query the retained raw graph through the source-bound native ABI."""

    assert owner.window_graph is not None and owner.r11_extension is not None
    count = int(
        owner.r11_extension.captured_graph_node_count(
            owner.window_graph.raw_cuda_graph()
        )
    )
    assert count > 0
    return count


def _tensor_leaves(value: object, path: str = "") -> dict[str, torch.Tensor]:
    leaves: dict[str, torch.Tensor] = {}
    if isinstance(value, torch.Tensor):
        leaves[path] = value
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            leaves.update(_tensor_leaves(item, f"{path}[{index}]"))
    elif is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            leaves.update(
                _tensor_leaves(getattr(value, field.name), f"{path}.{field.name}")
            )
    return leaves


def _prepare_m64_fixture(
    *, source_commit: str = "UNSPECIFIED_TEST_SOURCE"
) -> tuple[object, object]:
    """Build 128 isolated mate pairs for the admitted five-tick M64 window.

    This is qualification preparation at an explicit control boundary.  The
    fixed graph receives no synthetic host stimulus between its five ticks.
    """

    bundle, _ = make_c4_fixture_v1(
        batch=48,
        capacity=512,
        active=256,
        width=32,
        seed=97,
        world_local_identity=True,
    )
    core = bundle.base.base.core
    core.occupancy.agent_id_grid.fill_(-1)
    core.occupancy.slot_grid.fill_(-1)
    bundle.base.base.patch.agent_grid_i32.fill_(-1)
    for slot in range(256):
        pair = slot // 2
        y = (pair // 8) * 2
        x = (pair % 8) * 4 + (slot & 1)
        core.agents.pos_y[:, slot] = y
        core.agents.pos_x[:, slot] = x
        core.occupancy.agent_id_grid[:, y, x] = core.agents.agent_id[:, slot]
        core.occupancy.slot_grid[:, y, x] = slot
        bundle.base.base.patch.agent_grid_i32[:, y, x] = core.agents.agent_id[
            :, slot
        ].to(torch.int32)
    core.agents.energy[:, :256] = 160.0
    core.agents.health[:, :256] = 12.0
    bundle.base.agent.policy_energy_observation[:, :256] = 160.0
    bundle.base.agent.repro_threshold[:, :256] = 0.0
    bundle.base.genome.reproduction_threshold[:, :256] = 0.0
    bundle.base.genome.mutation_rate[:, :256] = 1.0
    for packed in (
        core.policy.packed_weights,
        bundle.base.base.policy.packed_weights,
    ):
        packed[..., -7:] = -20.0
        packed[..., -2] = 20.0  # action 5: rest, preserving mate topology
    bundle.scalar_rng = make_c4_cpython_random_state_v1(
        # Frozen five-tick M320 cap-64 bound needs 33,211 Gaussian-tape
        # positions (33,280 total MT raw32 draws). This allocation admits one
        # graph window; longer continuation needs remaining resident tape or a
        # separately qualified device-local refill.
        batch=48,
        tape_capacity=65_536,
        seed=97,
    )
    bundle.scalar_rng.python_random_gauss_primary_tape.fill_(0.01)
    bundle.scalar_rng.python_random_gauss_secondary_tape.zero_()
    bundle.event_ring = make_c4_event_ring_r11_v1(
        batch=48,
        capacity=512,
        event_capacity=bundle.events.capacity,
        device="cpu",
    )
    replay = make_c5_pcg64_context_v1(
        bundle,
        streams=5,
        seed=97,
        typed=True,
        source_commit=source_commit,
    )
    bundle.validate()
    replay.validate_typed_continuation_for_bundle(bundle)
    return bundle, replay


def _restore_qualification_sample(
    *, cpu_bundle: object, cpu_replay: object, bundle: object, replay: object
) -> None:
    """Restore one independent timed sample at a qualification boundary."""

    source = _tensor_leaves(cpu_bundle, "bundle") | _tensor_leaves(
        cpu_replay, "replay"
    )
    destination = _tensor_leaves(bundle, "bundle") | _tensor_leaves(
        replay, "replay"
    )
    assert source.keys() == destination.keys()
    guarded_policy = {
        "bundle.base.base.core.policy.packed_weights",
        "bundle.base.base.policy.packed_weights",
    }
    for path, target in destination.items():
        # Only inactive/newborn rows of these two large payloads can change in
        # this fixture, and every such row is overwritten before first use.
        # Avoiding their roughly two-GiB H2D reset keeps the qualification
        # boundary within the target device's physical memory envelope.
        if path in guarded_policy:
            continue
        target.copy_(source[path])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.skipif(
    os.environ.get("EIEL_C4_R11_PRODUCTION_PROBE") != "1",
    reason="R11 production probe is explicitly opt-in",
)
def test_c4_r11_b48_w32_p256_c512_graph_resource_and_timing(
    tmp_path: Path,
) -> None:
    """Measure one canonical owner at the admitted production geometry.

    All CPU inspection happens after the measured graph-replay sequence at an
    explicit qualification boundary.  No host semantic decision, status read,
    synchronization, or allocation occurs inside a five-tick graph window.
    """

    device = torch.device("cuda")
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert len(source_commit) == 40
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    baseline_allocated = torch.cuda.memory_allocated(device)
    baseline_reserved = torch.cuda.memory_reserved(device)

    cpu_bundle, cpu_replay = _prepare_m64_fixture(source_commit=source_commit)
    bundle = _move_in_place(copy.deepcopy(cpu_bundle), device)
    replay = _move_in_place(copy.deepcopy(cpu_replay), device)
    after_state_allocated = torch.cuda.memory_allocated(device)
    after_state_reserved = torch.cuda.memory_reserved(device)

    staging = make_c4_resident_composed_r11_staging_v1(
        batch=48,
        capacity=512,
        grid_size=32,
        streams=int(replay.rng_state_words.shape[1]),
        device=device,
    )
    after_staging_allocated = torch.cuda.memory_allocated(device)
    config = TensorWorldC4MateBirthConfigV1(
        max_births=64,
        allow_mutation=True,
        max_energy=160.0,
    )
    c2_path = os.environ.get(
        "TENSOR_WORLD_C2_PREBUILT",
        "/tmp/emergence-test/runtime/evidence_raw/tensor_world/production_c0_c7/"
        "20260815_c2_430489534_build_01/tensor_world_c2_ext.so",
    )
    build_dir = Path(os.environ.get("EIEL_C4_R11_BUILD_DIR", tmp_path / "build"))
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
    after_owner_allocated = torch.cuda.memory_allocated(device)
    after_owner_reserved = torch.cuda.memory_reserved(device)
    assert owner.shadow_bundle is not None
    assert owner.shadow_bundle.base.base.core.policy.packed_weights is (
        bundle.base.base.core.policy.packed_weights
    )
    assert owner.shadow_bundle.base.base.policy.packed_weights is (
        bundle.base.base.policy.packed_weights
    )

    capture_started = time.perf_counter()
    capture_c4_resident_composed_r11_window_v1(owner, config)
    capture_build_seconds = time.perf_counter() - capture_started
    assert owner.window_graph is not None
    graph_node_count = _cuda_graph_node_count(owner)
    after_capture_allocated = torch.cuda.memory_allocated(device)
    after_capture_reserved = torch.cuda.memory_reserved(device)

    sample_count = int(os.environ.get("EIEL_C4_R11_TIMING_SAMPLES", "20"))
    assert sample_count >= 5
    # Materialize any lazy CUDA allocator reservations in one explicit,
    # untimed qualification window before recording the steady replay
    # allocator baseline.  Boundary preparation and warm-up are not steady
    # tick work and are reported separately from the sampled graph replays.
    _restore_qualification_sample(
        cpu_bundle=cpu_bundle,
        cpu_replay=cpu_replay,
        bundle=bundle,
        replay=replay,
    )
    owner.window_control_initialized = False
    warmup_started = time.perf_counter()
    prepare_c4_resident_composed_r11_window_control_v1(
        owner, generation=100, tick=1_000
    )
    torch.cuda.synchronize(device)
    warmup_preparation_seconds = time.perf_counter() - warmup_started
    execute_c4_resident_composed_r11_window_cuda_v1(owner)
    torch.cuda.synchronize(device)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(sample_count)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(sample_count)]
    allocated_before_replay = torch.cuda.memory_allocated(device)
    reserved_before_replay = torch.cuda.memory_reserved(device)
    result = None
    for sample in range(sample_count):
        _restore_qualification_sample(
            cpu_bundle=cpu_bundle,
            cpu_replay=cpu_replay,
            bundle=bundle,
            replay=replay,
        )
        # Each timing sample is an explicitly independent qualification
        # boundary restored to the same canonical state.  Re-admit it as an
        # initialization window rather than pretending it follows the prior
        # sample's completed trajectory.
        owner.window_control_initialized = False
        prepare_c4_resident_composed_r11_window_control_v1(
            owner, generation=100, tick=1_000
        )
        torch.cuda.synchronize(device)
        starts[sample].record()
        result = execute_c4_resident_composed_r11_window_cuda_v1(owner)
        ends[sample].record()
    torch.cuda.synchronize(device)
    assert result is not None
    allocated_after_replay = torch.cuda.memory_allocated(device)
    reserved_after_replay = torch.cuda.memory_reserved(device)
    window_ms = [
        float(start.elapsed_time(end))
        for start, end in zip(starts, ends, strict=True)
    ]
    tick_ms = [value / 5.0 for value in window_ms]

    assert result.global_status.cpu().tolist() == [0] * 48
    assert result.completed_ticks.cpu().tolist() == [5] * 48
    assert result.current_tick.cpu().tolist() == [1_004] * 48
    # With the current frozen C4 energy/debit semantics, the prepared pairs
    # accept 64 births on each of the first four ticks (the fifth tick has no
    # additional eligible pair), so the monotonic ID advances 257 -> 513.
    # The former 385 assertion described an obsolete two-tick expectation and
    # made the resource harness fail after otherwise successful exact parity.
    assert bundle.population.next_agent_id.cpu().tolist() == [513] * 48
    assert bundle.event_ring is not None
    producer_cursor = bundle.event_ring.producer_cursor.cpu()
    birth_count = bundle.population.next_agent_id.cpu() - 257
    # The ring contains one birth row plus optional mutation rows per child;
    # mutation decisions are data-dependent, so the cursor is not equal to
    # the birth count.  Exact row/type parity is covered by the C512 harness.
    assert torch.all(producer_cursor >= birth_count)
    assert torch.all(producer_cursor <= bundle.event_ring.capacity)
    assert torch.equal(
        bundle.event_ring.producer_cursor,
        replay.event_sequence,
    )
    assert allocated_after_replay == allocated_before_replay
    assert reserved_after_replay == reserved_before_replay

    receipt = {
        "schema_version": "tensor_world_c4_r11_target_resource_timing_v1",
        "source_commit": source_commit,
        "route": "R11_B48_W32_P256_C512_FIVE_TICK_GRAPH_WINDOW",
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "batch": 48,
        "width": 32,
        "active_population": 256,
        "capacity": 512,
        "samples": sample_count,
        "ticks_per_window": 5,
        "graph_node_count": graph_node_count,
        "graph_capture_and_instantiate_seconds": capture_build_seconds,
        "untimed_warmup_preparation_seconds": warmup_preparation_seconds,
        "workload": "128_adjacent_parent_pairs_M64_first_four_ticks",
        "sample_reset_boundary": "qualification_only_before_each_graph_window",
        "median_tick_ms": statistics.median(tick_ms),
        "p95_tick_ms": _percentile(tick_ms, 0.95),
        "p99_tick_ms": _percentile(tick_ms, 0.99),
        "min_tick_ms": min(tick_ms),
        "max_tick_ms": max(tick_ms),
        "baseline_allocated_bytes": baseline_allocated,
        "baseline_reserved_bytes": baseline_reserved,
        "state_allocated_bytes": after_state_allocated - baseline_allocated,
        "state_reserved_bytes": after_state_reserved - baseline_reserved,
        "staging_increment_bytes": after_staging_allocated - after_state_allocated,
        "owner_increment_bytes": after_owner_allocated - after_staging_allocated,
        "owner_reserved_increment_bytes": after_owner_reserved - after_state_reserved,
        "capture_increment_bytes": after_capture_allocated - after_owner_allocated,
        "capture_reserved_increment_bytes": after_capture_reserved - after_owner_reserved,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "steady_state_allocator_delta_bytes": (
            allocated_after_replay - allocated_before_replay
        ),
        "steady_state_reserved_delta_bytes": (
            reserved_after_replay - reserved_before_replay
        ),
        "completed_ticks": int(result.completed_ticks[0].cpu()),
        "event_producer_cursor_min": int(producer_cursor.min()),
        "event_producer_cursor_max": int(producer_cursor.max()),
        "gate_claim": False,
    }
    receipt_path = Path(
        os.environ.get(
            "EIEL_C4_R11_PRODUCTION_RECEIPT",
            tmp_path / "c4_r11_target_resource_timing.json",
        )
    )
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
