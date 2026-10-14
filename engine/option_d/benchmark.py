"""D6 standalone performance and memory characterization for Option D.

The benchmark compares the exact current-source-shaped schedule against the
current C5 host-prepared resident Gaussian tape over complete refill cycles.
It deliberately does not import or activate a Tensor World route.  GPU
execution requires prior exclusive Resource Manager admission; this command
records but does not manufacture that authority.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import platform
import random
import statistics
import subprocess
import time
from typing import Any, Final

import torch

from engine.perf.cuda_architecture_b_lifecycle import (
    pack_cpython_random_state_cuda,
)
from engine.perf.cuda_architecture_b_resident_window_runtime_v1 import (
    refill_cpython_gaussian_window_,
)
from engine.tensor_world.c4_resident_rng_budget_r11 import (
    MAX_BIRTHS_V1,
    R11B_GAUSSIAN_TAPE_CAPACITY_V1,
    R11B_WINDOW_TICKS_V1,
)
from engine.tensor_world.c5_gaussian_window_r11 import _mt_tape_headroom

from .contracts import (
    DrawOperation,
    DrawPredicate,
    DrawSchedule,
    GenerationResult,
    Operator,
    RNGStateBundle,
    canonical_json_bytes,
)
from .cpu import execute_cpu, state_from_seed
from .cuda import (
    CudaResidentEngine,
    _check_counter_headroom,
    _decode_cuda_outputs,
    _gaussian_strategy_id,
    _schedule_host_tensors,
    cuda_source_digest,
    load_cuda_extension,
)
from .workloads import c4_scalar_schedule


ROI_THRESHOLD_PERCENT: Final[float] = 5.0
D6_INITIAL_COMPLETE_CYCLES: Final[int] = 31
D6_EXTENSION_CYCLE_CAP: Final[int] = 63
CURRENT_C5_TAPE_CAPACITY: Final[int] = R11B_GAUSSIAN_TAPE_CAPACITY_V1
SOURCE_BINDING_DEFAULT: Final[str] = "option-d-v1-cpython-b7e1856-glibc2.43"
ARTIFACT_BINDING: Final[str] = "option-d-d6-standalone-characterization-v2"


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("benchmark sample set must be nonempty")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _sample_summary(values: list[float]) -> dict[str, float | int]:
    return {
        "median": statistics.median(values),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "samples": len(values),
    }


def _numeric_summary(records: list[dict[str, float | int]]) -> dict[str, Any]:
    if not records:
        raise ValueError("benchmark records must be nonempty")
    return {
        key: _sample_summary([float(record[key]) for record in records])
        for key in records[0]
    }


def _tensor_bytes(tensors: tuple[torch.Tensor, ...]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def _rss_bytes() -> int:
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("VmRSS is unavailable")


def _bundle(worlds: int, source_binding: str, *, seed_base: int) -> RNGStateBundle:
    return RNGStateBundle(
        tuple(state_from_seed(seed_base + world) for world in range(worlds)),
        source_binding=source_binding,
    )


def _current_schedule(worlds: int, births: int) -> DrawSchedule:
    rates = tuple(
        tuple(0.01 + 0.19 * ((world + birth) % 11) / 10.0 for birth in range(births))
        for world in range(worlds)
    )
    memory = tuple(
        tuple(0.002 + 0.048 * ((world * 3 + birth) % 7) / 6.0 for birth in range(births))
        for world in range(worlds)
    )
    return c4_scalar_schedule(rates, memory)


def _raw_control_schedule(schedule: DrawSchedule) -> DrawSchedule:
    """Remove Gaussian slots while retaining and rebasing source decisions."""

    worlds: list[tuple[DrawOperation, ...]] = []
    for operations in schedule.worlds:
        index_map: dict[int, int] = {}
        retained: list[DrawOperation] = []
        for old_index, operation in enumerate(operations):
            if operation.operator is Operator.GAUSS:
                continue
            predicate = operation.predicate
            if predicate is not None:
                predicate = DrawPredicate(
                    predicate.kind,
                    index_map[predicate.source_operation],
                    predicate.threshold_bits,
                )
            index_map[old_index] = len(retained)
            retained.append(replace(operation, predicate=predicate))
        worlds.append(tuple(retained))
    result = DrawSchedule(tuple(worlds))
    result.validate(expected_worlds=len(schedule.worlds))
    return result


def _gaussian_component_schedule(schedule: DrawSchedule) -> DrawSchedule:
    """Return the dense source-shaped Gaussian upper envelope."""

    result = DrawSchedule(
        tuple(
            tuple(
                replace(operation, predicate=None)
                for operation in operations
                if operation.operator is Operator.GAUSS
            )
            for operations in schedule.worlds
        )
    )
    result.validate(expected_worlds=len(schedule.worlds))
    return result


def _material_host_tensors(result: GenerationResult) -> tuple[torch.Tensor, ...]:
    worlds = len(result.material.worlds)
    capacity = max(len(world) for world in result.material.worlds)
    bits = torch.zeros((worlds, capacity), dtype=torch.uint64)
    executed = torch.zeros((worlds, capacity), dtype=torch.bool)
    for world_index, values in enumerate(result.material.worlds):
        for operation_index, value in enumerate(values):
            bits[world_index, operation_index] = value.value_bits
            executed[world_index, operation_index] = value.executed
    return bits, executed


def _prepare_current_host_tape(
    owner: CudaResidentEngine,
    *,
    capacity: int,
    validate_current_helper: bool,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    dict[str, float | int],
    dict[str, Any] | None,
]:
    """Profile the exact current C5 host-prepared resident tape boundary."""

    if capacity < 1:
        raise ValueError("current host-tape capacity must be positive")
    device = owner.device
    bundle = owner.bundle
    state_tensors = owner._state_tensors  # noqa: SLF001 - D6 boundary probe
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    allocated_before = torch.cuda.memory_allocated(device)
    rss_before = _rss_bytes()
    total_start = time.perf_counter_ns()
    export_start_event, export_end_event = _event_pair()
    export_start = time.perf_counter_ns()
    export_start_event.record()
    host_state = tuple(tensor.cpu() for tensor in state_tensors)
    export_end_event.record()
    export_end_event.synchronize()
    export_end = time.perf_counter_ns()
    if host_state[0].tolist() != [list(world.words) for world in bundle.worlds]:
        raise RuntimeError("current host-tape state export does not match owner mirror")

    prepared_primary: list[torch.Tensor] = []
    prepared_secondary: list[torch.Tensor] = []
    for world in bundle.worlds:
        prepared = pack_cpython_random_state_cuda(
            world.to_python_state(),
            device="cpu",
            semantic_draws=world.counters.semantic_draws,
            gauss_tape_raw_capacity=capacity,
            raw_draws=world.counters.raw_words,
            raw_draw_base=world.counters.raw_words,
        )
        prepared_primary.append(
            prepared["python_random_gauss_primary_tape"].contiguous()
        )
        prepared_secondary.append(
            prepared["python_random_gauss_secondary_tape"].contiguous()
        )
    primary_host = torch.stack(prepared_primary).contiguous()
    secondary_host = torch.stack(prepared_secondary).contiguous()
    bases_host = torch.tensor(
        [world.counters.raw_words for world in bundle.worlds],
        dtype=torch.int64,
    )
    preparation_end = time.perf_counter_ns()

    h2d_start_event, h2d_end_event = _event_pair()
    h2d_start = time.perf_counter_ns()
    h2d_start_event.record()
    primary = primary_host.to(device=device, non_blocking=False)
    secondary = secondary_host.to(device=device, non_blocking=False)
    bases = bases_host.to(device=device, non_blocking=False)
    h2d_end_event.record()
    h2d_end_event.synchronize()
    h2d_end = time.perf_counter_ns()
    total_end = time.perf_counter_ns()
    profile_rss_delta = _rss_bytes() - rss_before
    profile_vram_peak_delta = max(
        0,
        torch.cuda.max_memory_allocated(device) - allocated_before,
    )

    helper_receipt: dict[str, Any] | None = None
    if validate_current_helper:
        helper_primary = torch.empty_like(primary)
        helper_secondary = torch.empty_like(secondary)
        helper_raw = state_tensors[4][:, 0].clone()
        helper_semantic = state_tensors[4][:, 1].clone()
        helper_base = helper_raw.clone()
        helper_receipt = refill_cpython_gaussian_window_(
            {
                "python_random_words": state_tensors[0],
                "python_random_index": state_tensors[1],
                "python_random_has_gauss": state_tensors[2],
                "python_random_gauss_next": state_tensors[3],
                "python_random_gauss_primary_tape": helper_primary,
                "python_random_gauss_secondary_tape": helper_secondary,
                "python_random_raw_draws": helper_raw,
                "mutation_py_semantic_draws": helper_semantic,
            },
            python_random_window_base=helper_base,
        )
        if not (
            torch.equal(primary, helper_primary)
            and torch.equal(secondary, helper_secondary)
            and torch.equal(bases, helper_base)
        ):
            raise RuntimeError("profiled host tape differs from current C5 helper")

    gpu_export_ms = export_start_event.elapsed_time(export_end_event)
    export_wall_ms = (export_end - export_start) / 1_000_000.0
    gpu_h2d_ms = h2d_start_event.elapsed_time(h2d_end_event)
    h2d_wall_ms = (h2d_end - h2d_start) / 1_000_000.0
    host_tensors = (primary_host, secondary_host, bases_host)
    record: dict[str, float | int] = {
        "d2h_state_bytes": _tensor_bytes(host_state),
        "d2h_state_gpu_ms": gpu_export_ms,
        "d2h_state_sync_overhead_ms": max(
            0.0,
            export_wall_ms - gpu_export_ms,
        ),
        "d2h_state_wall_ms": export_wall_ms,
        "end_to_end_ms": (total_end - total_start) / 1_000_000.0,
        "h2d_tape_bytes": _tensor_bytes(host_tensors),
        "h2d_tape_gpu_ms": gpu_h2d_ms,
        "h2d_tape_sync_overhead_ms": max(0.0, h2d_wall_ms - gpu_h2d_ms),
        "h2d_tape_wall_ms": h2d_wall_ms,
        "host_preparation_ms": (
            preparation_end - export_end
        ) / 1_000_000.0,
        "host_rss_delta_bytes": profile_rss_delta,
        "vram_peak_delta_bytes": profile_vram_peak_delta,
    }
    return (primary, secondary, bases), record, helper_receipt


def _allocate_current_helper_tape(
    owner: CudaResidentEngine,
    *,
    capacity: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Preallocate the exact current helper's in-place resident tape surfaces."""

    if capacity < 1:
        raise ValueError("current helper tape capacity must be positive")
    worlds = len(owner.bundle.worlds)
    return (
        torch.empty(
            (worlds, capacity),
            dtype=torch.float64,
            device=owner.device,
        ),
        torch.empty(
            (worlds, capacity),
            dtype=torch.float64,
            device=owner.device,
        ),
        torch.zeros(worlds, dtype=torch.int64, device=owner.device),
    )


def _refill_current_helper_tape(
    owner: CudaResidentEngine,
    tape_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[dict[str, float | int], dict[str, Any]]:
    """Time the exact current in-place host/tape helper as one mixed boundary."""

    owner._require_owner()  # noqa: SLF001 - D6 boundary probe
    primary, secondary, base = tape_tensors
    state_tensors = owner._state_tensors  # noqa: SLF001 - D6 boundary probe
    device = owner.device
    # The standalone counter ABI is [B,K], while the current helper owns two
    # contiguous [B] counters.  Materialize this benchmark-only adapter before
    # the gate timer and report it separately; it is not a current-C5 cost.
    adapter_start = time.perf_counter_ns()
    raw = state_tensors[4][:, 0].contiguous()
    semantic = state_tensors[4][:, 1].contiguous()
    torch.cuda.synchronize(device)
    adapter_end = time.perf_counter_ns()
    raw_before = raw.clone()
    semantic_before = semantic.clone()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    allocated_before = torch.cuda.memory_allocated(device)
    rss_before = _rss_bytes()
    start = time.perf_counter_ns()
    receipt = refill_cpython_gaussian_window_(
        {
            "python_random_words": state_tensors[0],
            "python_random_index": state_tensors[1],
            "python_random_has_gauss": state_tensors[2],
            "python_random_gauss_next": state_tensors[3],
            "python_random_gauss_primary_tape": primary,
            "python_random_gauss_secondary_tape": secondary,
            "python_random_raw_draws": raw,
            "mutation_py_semantic_draws": semantic,
        },
        python_random_window_base=base,
    )
    torch.cuda.synchronize(device)
    end = time.perf_counter_ns()
    if not torch.equal(raw, raw_before) or not torch.equal(semantic, semantic_before):
        raise RuntimeError("current helper refill changed absolute counters")
    if receipt["pointer_identity"]["all_exact"] is not True:
        raise RuntimeError("current helper refill changed resident tape identity")
    if receipt["absolute_raw_draws_preserved"] is not True:
        raise RuntimeError("current helper refill changed raw-draw authority")
    if receipt["semantic_draws_preserved"] is not True:
        raise RuntimeError("current helper refill changed semantic authority")
    return {
        "adapter_bytes": _tensor_bytes((raw, semantic)),
        "adapter_setup_ns": adapter_end - adapter_start,
        "end_to_end_ms": (end - start) / 1_000_000.0,
        "end_to_end_ns": end - start,
        "host_rss_delta_bytes": _rss_bytes() - rss_before,
        "vram_peak_delta_bytes": max(
            0,
            torch.cuda.max_memory_allocated(device) - allocated_before,
        ),
    }, receipt


def _current_c5_admission(
    bundle: RNGStateBundle,
    *,
    base_raw_words: tuple[int, ...],
    capacity: int,
    requested_ticks: int = R11B_WINDOW_TICKS_V1,
    max_births_per_tick: int = MAX_BIRTHS_V1,
) -> dict[str, Any]:
    """Apply the exact current five-tick worst-case tape-headroom predicate."""

    if len(base_raw_words) != len(bundle.worlds):
        raise ValueError("current C5 admission bases must match the world count")
    if capacity < 1:
        raise ValueError("current C5 admission capacity must be positive")
    headroom = tuple(
        _mt_tape_headroom(
            births=requested_ticks * max_births_per_tick,
            initial_has_gauss=world.gauss_next_bits is not None,
        )
        for world in bundle.worlds
    )
    raw = tuple(world.counters.raw_words for world in bundle.worlds)
    valid_ends = tuple(base + capacity for base in base_raw_words)
    usable = all(
        base <= cursor <= end and required <= end - cursor
        for base, cursor, end, required in zip(
            base_raw_words,
            raw,
            valid_ends,
            headroom,
            strict=True,
        )
    )
    return {
        "base_raw_words": base_raw_words,
        "headroom_by_world": headroom,
        "raw_words": raw,
        "refill_required": not usable,
        "requested_ticks": requested_ticks,
        "valid_raw_draw_ends": valid_ends,
    }


def _paired_opportunity_percent(*, baseline_ns: int, candidate_ns: int) -> float:
    if baseline_ns <= 0 or candidate_ns <= 0:
        raise ValueError("paired cycle timings must be positive")
    return 100.0 * (baseline_ns - candidate_ns) / baseline_ns


def _bootstrap_median_interval(
    values: list[float],
    *,
    resamples: int = 4096,
) -> dict[str, float | int]:
    if len(values) < 3 or resamples < 1:
        raise ValueError("bootstrap interval requires at least three observations")
    generator = random.Random(0xD6B00757)
    medians = [
        statistics.median(generator.choices(values, k=len(values)))
        for _ in range(resamples)
    ]
    return {
        "confidence": 0.95,
        "high": _percentile(medians, 0.975),
        "low": _percentile(medians, 0.025),
        "resamples": resamples,
    }


def _d6_roi_decision(
    paired_opportunities: list[float],
) -> tuple[str, dict[str, float | int], dict[str, float | int]]:
    """Apply the frozen threshold to paired complete-cycle observations."""

    summary = _sample_summary(paired_opportunities)
    interval = _bootstrap_median_interval(paired_opportunities)
    median_opportunity = float(summary["median"])
    if (
        median_opportunity >= ROI_THRESHOLD_PERCENT
        and float(interval["low"]) >= ROI_THRESHOLD_PERCENT
    ):
        decision = "D6-ROI-GO"
    elif (
        median_opportunity < ROI_THRESHOLD_PERCENT
        and float(interval["high"]) < ROI_THRESHOLD_PERCENT
    ):
        decision = "D6-ROI-NO-GO"
    else:
        decision = "D6-ROI-INCONCLUSIVE"
    return decision, summary, interval


def _event_pair() -> tuple[torch.cuda.Event, torch.cuda.Event]:
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def _baseline_sample(
    bundle: RNGStateBundle,
    schedule: DrawSchedule,
    *,
    device: torch.device,
) -> tuple[GenerationResult, dict[str, float | int]]:
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    allocated_before = torch.cuda.memory_allocated(device)
    rss_before = _rss_bytes()
    total_start = time.perf_counter_ns()
    result = execute_cpu(bundle, schedule)
    generation_end = time.perf_counter_ns()
    host_tensors = _material_host_tensors(result)
    preparation_end = time.perf_counter_ns()
    transfer_start_event, transfer_end_event = _event_pair()
    transfer_start = time.perf_counter_ns()
    transfer_start_event.record()
    device_tensors = tuple(
        tensor.to(device=device, non_blocking=False) for tensor in host_tensors
    )
    transfer_end_event.record()
    transfer_end_event.synchronize()
    transfer_end = time.perf_counter_ns()
    total_end = time.perf_counter_ns()
    gpu_h2d_ms = transfer_start_event.elapsed_time(transfer_end_event)
    h2d_wall_ms = (transfer_end - transfer_start) / 1_000_000.0
    peak_delta = max(
        0,
        torch.cuda.max_memory_allocated(device) - allocated_before,
    )
    record: dict[str, float | int] = {
        "end_to_end_ms": (total_end - total_start) / 1_000_000.0,
        "gpu_h2d_ms": gpu_h2d_ms,
        "h2d_sync_overhead_ms": max(0.0, h2d_wall_ms - gpu_h2d_ms),
        "h2d_wall_ms": h2d_wall_ms,
        "host_generation_ms": (generation_end - total_start) / 1_000_000.0,
        "host_material_preparation_ms": (
            preparation_end - generation_end
        ) / 1_000_000.0,
        "host_rss_delta_bytes": _rss_bytes() - rss_before,
        "h2d_bytes": _tensor_bytes(host_tensors),
        "vram_peak_delta_bytes": peak_delta,
    }
    del device_tensors
    return result, record


def _candidate_sample(
    owner: CudaResidentEngine,
    schedule: DrawSchedule,
    *,
    gaussian_strategy: str,
    tape_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    prepared_host_schedule: tuple[torch.Tensor, ...] | None = None,
) -> tuple[GenerationResult, dict[str, float | int]]:
    owner._require_owner()  # noqa: SLF001 - bounded D6 instrumentation
    device = owner.device
    initial = owner.bundle
    schedule.validate(expected_worlds=len(initial.worlds))
    if tape_tensors is None:
        strategy_id = _gaussian_strategy_id(schedule, gaussian_strategy)
    elif gaussian_strategy != "host_tape_baseline":
        raise ValueError("resident tape tensors require the host-tape baseline")
    _check_counter_headroom(initial, schedule)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    allocated_before = torch.cuda.memory_allocated(device)
    rss_before = _rss_bytes()
    total_start = time.perf_counter_ns()
    host_schedule = (
        _schedule_host_tensors(schedule)
        if prepared_host_schedule is None
        else prepared_host_schedule
    )
    preparation_end = time.perf_counter_ns()

    h2d_start_event, h2d_end_event = _event_pair()
    h2d_start = time.perf_counter_ns()
    h2d_start_event.record()
    device_schedule = tuple(
        tensor.to(device=device, non_blocking=False) for tensor in host_schedule
    )
    h2d_end_event.record()
    h2d_end_event.synchronize()
    h2d_end = time.perf_counter_ns()

    kernel_start_event, kernel_end_event = _event_pair()
    kernel_start = time.perf_counter_ns()
    kernel_start_event.record()
    if tape_tensors is None:
        outputs = owner._module.execute(  # noqa: SLF001
            *owner._state_tensors,  # noqa: SLF001
            *device_schedule,
            strategy_id,
        )
    else:
        outputs = owner._module.execute_tape_baseline(  # noqa: SLF001
            *owner._state_tensors,  # noqa: SLF001
            *device_schedule,
            *tape_tensors,
        )
    kernel_end_event.record()
    kernel_end_event.synchronize()
    kernel_end = time.perf_counter_ns()

    d2h_start_event, d2h_end_event = _event_pair()
    d2h_start = time.perf_counter_ns()
    d2h_start_event.record()
    host_outputs = tuple(tensor.cpu() for tensor in outputs)
    d2h_end_event.record()
    d2h_end_event.synchronize()
    d2h_end = time.perf_counter_ns()

    gpu_h2d_ms = h2d_start_event.elapsed_time(h2d_end_event)
    gpu_kernel_ms = kernel_start_event.elapsed_time(kernel_end_event)
    gpu_d2h_ms = d2h_start_event.elapsed_time(d2h_end_event)
    h2d_wall_ms = (h2d_end - h2d_start) / 1_000_000.0
    kernel_wall_ms = (kernel_end - kernel_start) / 1_000_000.0
    d2h_wall_ms = (d2h_end - d2h_start) / 1_000_000.0

    result = _decode_cuda_outputs(
        initial,
        schedule,
        host_outputs,
        gaussian_strategy=gaussian_strategy,
        resident=True,
    )
    commit_end = time.perf_counter_ns()
    owner._state_tensors = tuple(outputs[:5])  # noqa: SLF001
    owner._bundle = result.successor  # noqa: SLF001
    owner._window_count += 1  # noqa: SLF001
    total_end = time.perf_counter_ns()
    peak_delta = max(
        0,
        torch.cuda.max_memory_allocated(device) - allocated_before,
    )
    return result, {
        "d2h_bytes": _tensor_bytes(tuple(outputs)),
        "d2h_sync_overhead_ms": max(0.0, d2h_wall_ms - gpu_d2h_ms),
        "d2h_wall_ms": d2h_wall_ms,
        "end_to_end_ms": (total_end - total_start) / 1_000_000.0,
        "end_to_end_ns": total_end - total_start,
        "gpu_d2h_ms": gpu_d2h_ms,
        "gpu_h2d_ms": gpu_h2d_ms,
        "gpu_kernel_ms": gpu_kernel_ms,
        "h2d_bytes": _tensor_bytes(host_schedule),
        "h2d_sync_overhead_ms": max(0.0, h2d_wall_ms - gpu_h2d_ms),
        "h2d_wall_ms": h2d_wall_ms,
        "host_commit_ms": (commit_end - d2h_end) / 1_000_000.0,
        "host_commit_ns": commit_end - d2h_end,
        "host_rss_delta_bytes": _rss_bytes() - rss_before,
        "host_schedule_preparation_ms": (
            preparation_end - total_start
        ) / 1_000_000.0,
        "host_schedule_preparation_ns": preparation_end - total_start,
        "kernel_sync_overhead_ms": max(
            0.0,
            kernel_wall_ms - gpu_kernel_ms,
        ),
        "kernel_wall_ms": kernel_wall_ms,
        "kernel_wall_ns": kernel_end - kernel_start,
        "synchronization_overhead_ms": (
            max(0.0, h2d_wall_ms - gpu_h2d_ms)
            + max(0.0, kernel_wall_ms - gpu_kernel_ms)
            + max(0.0, d2h_wall_ms - gpu_d2h_ms)
        ),
        "vram_peak_delta_bytes": peak_delta,
    }


def _operation_count(schedule: DrawSchedule) -> int:
    return sum(len(world) for world in schedule.worlds)


def _executed_count(result: GenerationResult) -> int:
    return sum(
        value.executed
        for world in result.material.worlds
        for value in world
    )


def _capture_tape_checkpoint(
    owner: CudaResidentEngine,
    tape_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    schedule_sha256: str,
) -> tuple[str, dict[str, float | int]]:
    device = owner.device
    torch.cuda.synchronize(device)
    start = time.perf_counter_ns()
    snapshot = owner.capture_checkpoint(
        artifact_binding=ARTIFACT_BINDING,
        schedule_sha256=schedule_sha256,
    )
    d2h_start_event, d2h_end_event = _event_pair()
    d2h_start = time.perf_counter_ns()
    d2h_start_event.record()
    host_tape = tuple(tensor.cpu() for tensor in tape_tensors)
    d2h_end_event.record()
    d2h_end_event.synchronize()
    d2h_end = time.perf_counter_ns()
    hasher = sha256(snapshot.sha256.encode("ascii"))
    for tensor in host_tape:
        hasher.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
    end = time.perf_counter_ns()
    gpu_d2h_ms = d2h_start_event.elapsed_time(d2h_end_event)
    d2h_wall_ms = (d2h_end - d2h_start) / 1_000_000.0
    return hasher.hexdigest(), {
        "bytes": len(snapshot.data) + _tensor_bytes(host_tape),
        "d2h_gpu_ms": gpu_d2h_ms,
        "d2h_sync_overhead_ms": max(0.0, d2h_wall_ms - gpu_d2h_ms),
        "d2h_wall_ms": d2h_wall_ms,
        "end_to_end_ms": (end - start) / 1_000_000.0,
    }


def _capture_candidate_checkpoint(
    owner: CudaResidentEngine,
    *,
    schedule_sha256: str,
) -> tuple[str, dict[str, float | int]]:
    """Time checkpoint capture without recomputing the frozen schedule digest."""

    start = time.perf_counter_ns()
    snapshot = owner.capture_checkpoint(
        artifact_binding=ARTIFACT_BINDING,
        schedule_sha256=schedule_sha256,
    )
    end = time.perf_counter_ns()
    return snapshot.sha256, {
        "bytes": len(snapshot.data),
        "end_to_end_ms": (end - start) / 1_000_000.0,
    }


def _component_profile(
    schedule: DrawSchedule,
    *,
    gaussian_strategy: str,
    source_binding: str,
    build_root: Path,
    warmups: int,
    samples: int,
    seed_base: int,
    tape_capacity: int | None = None,
) -> dict[str, Any]:
    owner = CudaResidentEngine(
        _bundle(len(schedule.worlds), source_binding, seed_base=seed_base),
        build_root=build_root,
    )
    tape_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    tape_refill: dict[str, Any] | None = None
    if gaussian_strategy == "host_tape_baseline":
        if tape_capacity is None:
            raise ValueError("host-tape component profile requires a capacity")
        tape_tensors, tape_refill_record, _ = _prepare_current_host_tape(
            owner,
            capacity=tape_capacity,
            validate_current_helper=False,
        )
        tape_refill = {
            "capacity": tape_capacity,
            "profile": tape_refill_record,
        }
    for _ in range(warmups):
        _candidate_sample(
            owner,
            schedule,
            gaussian_strategy=gaussian_strategy,
            tape_tensors=tape_tensors,
        )
    records: list[dict[str, float | int]] = []
    executed: list[int] = []
    for _ in range(samples):
        result, record = _candidate_sample(
            owner,
            schedule,
            gaussian_strategy=gaussian_strategy,
            tape_tensors=tape_tensors,
        )
        records.append(record)
        executed.append(_executed_count(result))
    summary = _numeric_summary(records)
    operations = _operation_count(schedule)
    median_seconds = float(summary["gpu_kernel_ms"]["median"]) / 1000.0
    return {
        "executed_operations": {
            "max": max(executed),
            "median": statistics.median(executed),
            "min": min(executed),
        },
        "gaussian_strategy": gaussian_strategy,
        "scheduled_operations": operations,
        "tape_refill": tape_refill,
        "timings": summary,
        "throughput_scheduled_operations_per_second": operations / median_seconds,
    }


def _git_metadata(root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        return subprocess.run(
            ("git", *arguments),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "branch": run("branch", "--show-current"),
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain", "--untracked-files=no")),
    }


def _run_d6_benchmark_v1_unqualified(
    *,
    output: Path,
    build_root: Path,
    source_binding: str,
    resource_admission: str,
    worlds: int = 48,
    births: int = 64,
    warmups: int = 5,
    samples: int = D6_INITIAL_COMPLETE_CYCLES,
    refill_samples: int = 3,
) -> dict[str, Any]:
    if worlds != 48 or births != 64:
        raise ValueError("the D6 gate geometry is frozen at B48/M64")
    if warmups < 1 or samples < 3 or refill_samples < 1:
        raise ValueError("D6 requires at least one warmup and three samples")
    if not resource_admission.strip():
        raise ValueError("exclusive GPU Resource Manager admission must be recorded")
    if not torch.cuda.is_available():
        raise RuntimeError("D6 requires an available CUDA device")
    device = torch.device("cuda")
    root = Path(__file__).resolve().parents[2]

    setup_start = time.perf_counter_ns()
    schedule = _current_schedule(worlds, births)
    schedule_sha256 = schedule.digest()
    raw_schedule = _raw_control_schedule(schedule)
    gaussian_schedule = _gaussian_component_schedule(schedule)
    initial = _bundle(worlds, source_binding, seed_base=0xD6000000)
    expected_results: list[GenerationResult] = []
    expected_state = initial
    for _ in range(warmups + samples):
        expected = execute_cpu(expected_state, schedule)
        expected_results.append(expected)
        expected_state = expected.successor
    total_raw_words_by_world = tuple(
        after.counters.raw_words - before.counters.raw_words
        for before, after in zip(
            initial.worlds,
            expected_state.worlds,
            strict=True,
        )
    )
    if max(total_raw_words_by_world) > CURRENT_C5_TAPE_CAPACITY:
        raise RuntimeError(
            "the frozen D6 sample count exceeds the current C5 tape capacity"
        )
    setup_end = time.perf_counter_ns()
    torch.cuda.synchronize(device)
    module_start = time.perf_counter_ns()
    load_cuda_extension(build_root=build_root)
    torch.cuda.synchronize(device)
    module_end = time.perf_counter_ns()
    tape_import_start = time.perf_counter_ns()
    tape_owner = CudaResidentEngine(initial, build_root=build_root)
    torch.cuda.synchronize(device)
    tape_import_end = time.perf_counter_ns()
    candidate_import_start = time.perf_counter_ns()
    candidate_owner = CudaResidentEngine(initial, build_root=build_root)
    torch.cuda.synchronize(device)
    candidate_import_end = time.perf_counter_ns()

    refill_records: list[dict[str, float | int]] = []
    helper_receipt: dict[str, Any] | None = None
    tape_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    for refill_index in range(refill_samples):
        prepared_tape, refill_record, current_helper = _prepare_current_host_tape(
            tape_owner,
            capacity=CURRENT_C5_TAPE_CAPACITY,
            validate_current_helper=refill_index == 0,
        )
        refill_records.append(refill_record)
        if current_helper is not None:
            helper_receipt = current_helper
        if refill_index == refill_samples - 1:
            tape_tensors = prepared_tape
        else:
            del prepared_tape
    assert tape_tensors is not None and helper_receipt is not None

    for index in range(warmups):
        expected = expected_results[index]
        tape_result, _ = _candidate_sample(
            tape_owner,
            schedule,
            gaussian_strategy="host_tape_baseline",
            tape_tensors=tape_tensors,
        )
        candidate_result, _ = _candidate_sample(
            candidate_owner,
            schedule,
            gaussian_strategy="software",
        )
        if (
            tape_result.material != expected.material
            or tape_result.successor != expected.successor
            or candidate_result.material != expected.material
            or candidate_result.successor != expected.successor
        ):
            raise RuntimeError("D6 warmup exactness comparison failed")

    tape_records: list[dict[str, float | int]] = []
    candidate_records: list[dict[str, float | int]] = []
    executed_counts: list[int] = []
    raw_word_deltas: list[int] = []
    for sample_index in range(samples):
        expected = expected_results[warmups + sample_index]
        tape_result, tape_record = _candidate_sample(
            tape_owner,
            schedule,
            gaussian_strategy="host_tape_baseline",
            tape_tensors=tape_tensors,
        )
        candidate_result, candidate_record = _candidate_sample(
            candidate_owner,
            schedule,
            gaussian_strategy="software",
        )
        if (
            tape_result.material != expected.material
            or tape_result.successor != expected.successor
            or candidate_result.material != expected.material
            or candidate_result.successor != expected.successor
        ):
            raise RuntimeError("D6 measured exactness comparison failed")
        tape_records.append(tape_record)
        candidate_records.append(candidate_record)
        executed_counts.append(_executed_count(candidate_result))
        raw_word_deltas.extend(
            receipt.counters_delta.raw_words
            for receipt in candidate_result.receipt.worlds
        )

    tape_summary = _numeric_summary(tape_records)
    candidate_summary = _numeric_summary(candidate_records)
    refill_summary = _numeric_summary(refill_records)
    maximum_raw_words_per_tick = max(raw_word_deltas)
    reusable_ticks = CURRENT_C5_TAPE_CAPACITY // maximum_raw_words_per_tick
    amortization_ticks = (reusable_ticks // 5) * 5
    if amortization_ticks < 5:
        raise RuntimeError("current C5 tape cannot admit one five-tick window")
    tape_steady_median = float(tape_summary["end_to_end_ms"]["median"])
    candidate_median = float(candidate_summary["end_to_end_ms"]["median"])
    refill_median = float(refill_summary["end_to_end_ms"]["median"])
    tape_effective_median = tape_steady_median + refill_median / amortization_ticks
    tape_effective_summary: dict[str, float | int] = {
        quantile: (
            float(tape_summary["end_to_end_ms"][quantile])
            + float(refill_summary["end_to_end_ms"][quantile])
            / amortization_ticks
        )
        for quantile in ("median", "p95", "p99")
    }
    tape_effective_summary["samples"] = samples
    opportunity = (
        100.0
        * (tape_effective_median - candidate_median)
        / tape_effective_median
    )
    steady_opportunity = (
        100.0
        * (tape_steady_median - candidate_median)
        / tape_steady_median
    )

    candidate_checkpoint_records: list[dict[str, float | int]] = []
    candidate_checkpoint_digests: list[str] = []
    tape_checkpoint_records: list[dict[str, float | int]] = []
    tape_checkpoint_digests: list[str] = []
    checkpoint_samples = max(5, min(samples, 9))
    tape_state_snapshot = tape_owner.capture_checkpoint(
        artifact_binding=ARTIFACT_BINDING,
        schedule_sha256=schedule_sha256,
    )
    tape_state_checkpoint_sha256 = sha256(tape_state_snapshot.data).hexdigest()
    for _ in range(samples):
        digest, record = _capture_candidate_checkpoint(
            candidate_owner,
            schedule_sha256=schedule_sha256,
        )
        candidate_checkpoint_digests.append(digest)
        candidate_checkpoint_records.append(record)
        if digest != tape_state_checkpoint_sha256:
            raise RuntimeError("D6 candidate and tape state checkpoints differ")
    for _ in range(checkpoint_samples):
        digest, record = _capture_tape_checkpoint(
            tape_owner,
            tape_tensors,
            schedule_sha256=schedule_sha256,
        )
        tape_checkpoint_digests.append(digest)
        tape_checkpoint_records.append(record)
    if len(set(tape_checkpoint_digests)) != 1:
        raise RuntimeError("current host-tape checkpoint materialization is unstable")
    if len(set(candidate_checkpoint_digests)) != 1:
        raise RuntimeError("Option D checkpoint materialization is unstable")
    if tape_owner.bundle != candidate_owner.bundle:
        raise RuntimeError("D6 final continuation states differ")

    operations = _operation_count(schedule)
    tape_host = (
        float(tape_summary["host_schedule_preparation_ms"]["median"])
        + float(tape_summary["host_commit_ms"]["median"])
        + (
            float(refill_summary["d2h_state_wall_ms"]["median"])
            + float(refill_summary["host_preparation_ms"]["median"])
            + float(refill_summary["h2d_tape_wall_ms"]["median"])
        )
        / amortization_ticks
    )
    candidate_host = (
        float(candidate_summary["host_schedule_preparation_ms"]["median"])
        + float(candidate_summary["host_commit_ms"]["median"])
    )
    tape_gpu = sum(
        float(tape_summary[name]["median"])
        for name in ("gpu_h2d_ms", "gpu_kernel_ms", "gpu_d2h_ms")
    ) + (
        float(refill_summary["d2h_state_gpu_ms"]["median"])
        + float(refill_summary["h2d_tape_gpu_ms"]["median"])
    ) / amortization_ticks
    candidate_gpu = sum(
        float(candidate_summary[name]["median"])
        for name in ("gpu_h2d_ms", "gpu_kernel_ms", "gpu_d2h_ms")
    )
    checkpoint_candidate_summary = _numeric_summary(candidate_checkpoint_records)
    checkpoint_tape_summary = _numeric_summary(tape_checkpoint_records)
    component_samples = max(5, min(samples, 15))
    component_warmups = max(1, min(warmups, 3))
    raw_profile = _component_profile(
        raw_schedule,
        gaussian_strategy="disabled",
        source_binding=source_binding,
        build_root=build_root,
        warmups=component_warmups,
        samples=component_samples,
        seed_base=0xD6100000,
    )
    gaussian_profile = _component_profile(
        gaussian_schedule,
        gaussian_strategy="software",
        source_binding=source_binding,
        build_root=build_root,
        warmups=component_warmups,
        samples=component_samples,
        seed_base=0xD6200000,
    )
    gaussian_tape_capacity = 32_768
    gaussian_tape_profile = _component_profile(
        gaussian_schedule,
        gaussian_strategy="host_tape_baseline",
        source_binding=source_binding,
        build_root=build_root,
        warmups=component_warmups,
        samples=component_samples,
        seed_base=0xD6200000,
        tape_capacity=gaussian_tape_capacity,
    )

    decision = (
        "D6-ROI-GO"
        if opportunity >= ROI_THRESHOLD_PERCENT
        else "D6-ROI-NO-GO"
    )
    payload: dict[str, Any] = {
        "benchmark": {
            "births_per_world": births,
            "current_source_geometry": "B48/C512/M64 one-tick scalar RNG envelope",
            "samples": samples,
            "schedule_sha256": schedule_sha256,
            "scheduled_operations_per_sample": operations,
            "warmups": warmups,
            "worlds": worlds,
        },
        "candidate": {
            "backend": "standalone Option D resident CUDA software-math V1",
            "checkpoint_capture": checkpoint_candidate_summary,
            "resident_state_bytes": candidate_owner.resident_device_bytes,
            "state_import_ms": (
                candidate_import_end - candidate_import_start
            ) / 1_000_000.0,
            "timings": candidate_summary,
            "throughput_scheduled_operations_per_second": (
                operations / (candidate_median / 1000.0)
            ),
        },
        "claim_boundary": (
            "Standalone source-shaped characterization only; no Tensor World "
            "integration, activation, or improvement claim."
        ),
        "component_profiles": {
            "gaussian_transform_dense_source_upper_envelope": {
                "current_host_tape": gaussian_tape_profile,
                "option_d_software_math": gaussian_profile,
            },
            "raw_rng_current_control_shape": raw_profile,
        },
        "decision": decision,
        "derived": {
            "checkpoint_delta_ms": (
                float(checkpoint_candidate_summary["end_to_end_ms"]["median"])
                - float(checkpoint_tape_summary["end_to_end_ms"]["median"])
            ),
            "end_to_end_opportunity_percent": opportunity,
            "gpu_wall_added_ms": candidate_gpu - tape_gpu,
            "host_wall_removed_ms": tape_host - candidate_host,
            "roi_threshold_percent": ROI_THRESHOLD_PERCENT,
            "steady_suffix_reuse_opportunity_percent": steady_opportunity,
        },
        "exactness": {
            "checkpoint_samples_equal": samples,
            "executed_operations": {
                "max": max(executed_counts),
                "median": statistics.median(executed_counts),
                "min": min(executed_counts),
            },
            "measured_samples_equal": samples,
            "warmup_samples_equal": warmups,
        },
        "current_host_tape_baseline": {
            "amortization": {
                "current_c5_five_tick_boundary": True,
                "effective_end_to_end_ms": tape_effective_summary,
                "maximum_observed_raw_words_per_world_tick": (
                    maximum_raw_words_per_tick
                ),
                "reusable_ticks_conservative_observed_envelope": amortization_ticks,
            },
            "backend": (
                "current C5 host-prepared float64 Gaussian tape with resident "
                "suffix reuse and device MT consumption"
            ),
            "checkpoint_materialization": checkpoint_tape_summary,
            "current_helper_receipt": helper_receipt,
            "refill_boundary": {
                "capacity_per_world": CURRENT_C5_TAPE_CAPACITY,
                "profile": refill_summary,
                "samples": refill_samples,
            },
            "resident_state_and_tape_bytes": (
                tape_owner.resident_device_bytes + _tensor_bytes(tape_tensors)
            ),
            "state_import_ms": (
                tape_import_end - tape_import_start
            ) / 1_000_000.0,
            "steady_suffix_reuse_timings": tape_summary,
            "throughput_scheduled_operations_per_second": (
                operations / (tape_effective_median / 1000.0)
            ),
        },
        "cpu_oracle_role": (
            "Exactness oracle only; deliberately excluded from the ROI denominator "
            "because current C5 uses resident tape suffix reuse."
        ),
        "memory_method": {
            "host": "observed process VmRSS end-minus-start per sample",
            "vram": "PyTorch max_memory_allocated minus pre-sample allocated bytes",
            "resident_vram_delta_candidate_minus_tape_bytes": (
                candidate_owner.resident_device_bytes
                - tape_owner.resident_device_bytes
                - _tensor_bytes(tape_tensors)
            ),
        },
        "provenance": {
            "current_c5_baseline_sources": {
                str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
                for path in (
                    root / "engine/perf/cuda_architecture_b_lifecycle.py",
                    root / "engine/perf/cuda_architecture_b_resident_window_runtime_v1.py",
                    root / "engine/tensor_world/c5_gaussian_window_r11.py",
                    root / "engine/tensor_world/c4_resident_rng_budget_r11.py",
                )
            },
            "cuda_device": torch.cuda.get_device_name(device),
            "cuda_source_sha256": cuda_source_digest(),
            "git": _git_metadata(root),
            "platform": platform.platform(),
            "resource_manager_exclusive_admission": resource_admission,
            "source_binding": source_binding,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "schema_version": "eiel_option_d_d6_characterization_v1",
        "setup_ms": {
            "contract_and_schedule": (setup_end - setup_start) / 1_000_000.0,
            "native_module_cached_load": (module_end - module_start) / 1_000_000.0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(payload) + b"\n")
    return payload


def run_d6_benchmark(
    *,
    output: Path,
    build_root: Path,
    source_binding: str,
    resource_admission: str,
    worlds: int = 48,
    births: int = 64,
    warmups: int = 1,
    samples: int = 31,
    refill_samples: int = 10,
) -> dict[str, Any]:
    """Run decision-grade paired complete current-C5 tape cycles."""

    if worlds != 48 or births != MAX_BIRTHS_V1:
        raise ValueError("the D6 gate geometry is frozen at B48/M64")
    if (
        warmups < 1
        or samples not in {D6_INITIAL_COMPLETE_CYCLES, D6_EXTENSION_CYCLE_CAP}
        or refill_samples < 4
    ):
        raise ValueError(
            "D6 requires a warmup, 31 or 63 paired samples, and four checkpoints"
        )
    if not resource_admission.strip():
        raise ValueError("exclusive GPU Resource Manager admission must be recorded")
    if not torch.cuda.is_available():
        raise RuntimeError("D6 requires an available CUDA device")
    device = torch.device("cuda")
    root = Path(__file__).resolve().parents[2]
    git_metadata = _git_metadata(root)
    if git_metadata["dirty"]:
        raise RuntimeError("D6 decision runs require a clean committed source tree")
    benchmark_source_sha256 = sha256(Path(__file__).read_bytes()).hexdigest()

    setup_start = time.perf_counter_ns()
    schedule = _current_schedule(worlds, births)
    schedule_sha256 = schedule.digest()
    raw_schedule = _raw_control_schedule(schedule)
    gaussian_schedule = _gaussian_component_schedule(schedule)
    initial = _bundle(worlds, source_binding, seed_base=0xD6000000)
    expected_state = initial
    setup_end = time.perf_counter_ns()

    torch.cuda.synchronize(device)
    module_start = time.perf_counter_ns()
    load_cuda_extension(build_root=build_root)
    torch.cuda.synchronize(device)
    module_end = time.perf_counter_ns()
    tape_import_start = time.perf_counter_ns()
    tape_owner = CudaResidentEngine(initial, build_root=build_root)
    tape_tensors = _allocate_current_helper_tape(
        tape_owner,
        capacity=CURRENT_C5_TAPE_CAPACITY,
    )
    initial_refill_record, initial_refill_receipt = _refill_current_helper_tape(
        tape_owner,
        tape_tensors,
    )
    torch.cuda.synchronize(device)
    tape_import_end = time.perf_counter_ns()
    candidate_import_start = time.perf_counter_ns()
    candidate_owner = CudaResidentEngine(initial, build_root=build_root)
    torch.cuda.synchronize(device)
    candidate_import_end = time.perf_counter_ns()

    helper_receipts: list[dict[str, Any]] = []

    def retain_helper_receipt(
        receipt: dict[str, Any],
        *,
        cycle: int,
        phase: str,
    ) -> None:
        helper_receipts.append(
            {
                "absolute_raw_draws_before": receipt["absolute_raw_draws_before"],
                "cycle": cycle,
                "device_to_host_bytes": receipt["device_to_host_bytes"],
                "host_prepared_bytes": receipt["host_prepared_bytes"],
                "host_to_device_bytes": receipt["host_to_device_bytes"],
                "phase": phase,
                "pointer_identity_exact": receipt["pointer_identity"]["all_exact"],
                "receipt_sha256": sha256(canonical_json_bytes(receipt)).hexdigest(),
                "window_base_after": receipt["window_base_after"],
            }
        )

    retain_helper_receipt(initial_refill_receipt, cycle=-1, phase="initialization")
    pending_admission: tuple[dict[str, Any], int] | None = None
    global_tick = 0

    def timed_admission() -> tuple[dict[str, Any], int]:
        start = time.perf_counter_ns()
        bases = tuple(int(value) for value in tape_tensors[2].cpu().tolist())
        admission = _current_c5_admission(
            tape_owner.bundle,
            base_raw_words=bases,
            capacity=CURRENT_C5_TAPE_CAPACITY,
        )
        end = time.perf_counter_ns()
        return admission, end - start

    def execute_cycle(
        *,
        cycle: int,
        measured: bool,
    ) -> dict[str, Any]:
        nonlocal expected_state, global_tick, pending_admission
        if pending_admission is None:
            admission, admission_ns = timed_admission()
        else:
            admission, admission_ns = pending_admission
            pending_admission = None
        baseline_total_ns = admission_ns
        candidate_total_ns = 0
        baseline_gpu_ms = 0.0
        candidate_gpu_ms = 0.0
        baseline_host_commit_ns = 0
        candidate_host_commit_ns = 0
        refill_records: list[dict[str, float | int]] = []
        admission_records: list[dict[str, Any]] = [
            {**admission, "check_ns": admission_ns}
        ]
        if admission["refill_required"]:
            refill_record, receipt = _refill_current_helper_tape(
                tape_owner,
                tape_tensors,
            )
            retain_helper_receipt(
                receipt,
                cycle=cycle,
                phase="measured" if measured else "warmup",
            )
            refill_records.append(refill_record)
            baseline_total_ns += int(refill_record["end_to_end_ns"])
            admission, check_ns = timed_admission()
            admission_records.append({**admission, "check_ns": check_ns})
            baseline_total_ns += check_ns
            if admission["refill_required"]:
                raise RuntimeError("fresh current helper tape cannot admit five ticks")
        elif measured or cycle > 0:
            raise RuntimeError("a complete measured cycle must begin at a refill")

        order = "tape_first" if cycle % 2 == 0 else "candidate_first"
        tick_records: list[dict[str, Any]] = []
        windows = 0
        while True:
            for _ in range(R11B_WINDOW_TICKS_V1):
                expected = execute_cpu(expected_state, schedule)
                schedule_prepare_start = time.perf_counter_ns()
                prepared_host_schedule = _schedule_host_tensors(schedule)
                schedule_prepare_end = time.perf_counter_ns()
                shared_schedule_ns = schedule_prepare_end - schedule_prepare_start
                baseline_total_ns += shared_schedule_ns
                candidate_total_ns += shared_schedule_ns
                if order == "tape_first":
                    tape_result, tape_record = _candidate_sample(
                        tape_owner,
                        schedule,
                        gaussian_strategy="host_tape_baseline",
                        tape_tensors=tape_tensors,
                        prepared_host_schedule=prepared_host_schedule,
                    )
                    candidate_result, candidate_record = _candidate_sample(
                        candidate_owner,
                        schedule,
                        gaussian_strategy="software",
                        prepared_host_schedule=prepared_host_schedule,
                    )
                else:
                    candidate_result, candidate_record = _candidate_sample(
                        candidate_owner,
                        schedule,
                        gaussian_strategy="software",
                        prepared_host_schedule=prepared_host_schedule,
                    )
                    tape_result, tape_record = _candidate_sample(
                        tape_owner,
                        schedule,
                        gaussian_strategy="host_tape_baseline",
                        tape_tensors=tape_tensors,
                        prepared_host_schedule=prepared_host_schedule,
                    )
                if (
                    tape_result.material != expected.material
                    or tape_result.successor != expected.successor
                    or candidate_result.material != expected.material
                    or candidate_result.successor != expected.successor
                ):
                    raise RuntimeError("D6 paired-cycle exactness comparison failed")
                expected_state = expected.successor
                baseline_total_ns += int(tape_record["end_to_end_ns"])
                candidate_total_ns += int(candidate_record["end_to_end_ns"])
                baseline_gpu_ms += sum(
                    float(tape_record[name])
                    for name in ("gpu_h2d_ms", "gpu_kernel_ms", "gpu_d2h_ms")
                )
                candidate_gpu_ms += sum(
                    float(candidate_record[name])
                    for name in ("gpu_h2d_ms", "gpu_kernel_ms", "gpu_d2h_ms")
                )
                baseline_host_commit_ns += int(tape_record["host_commit_ns"])
                candidate_host_commit_ns += int(candidate_record["host_commit_ns"])
                if measured:
                    tick_records.append(
                        {
                            "candidate": candidate_record,
                            "executed_operations": _executed_count(candidate_result),
                            "global_tick": global_tick,
                            "order": order,
                            "shared_schedule_preparation_ms": (
                                shared_schedule_ns / 1_000_000.0
                            ),
                            "shared_schedule_preparation_ns": shared_schedule_ns,
                            "successor_sha256": candidate_result.successor.digest(),
                            "tape": tape_record,
                        }
                    )
                global_tick += 1
            windows += 1
            if windows > 64:
                raise RuntimeError("current C5 tape did not reach a bounded refill")
            admission, check_ns = timed_admission()
            if admission["refill_required"]:
                pending_admission = (admission, check_ns)
                break
            admission_records.append({**admission, "check_ns": check_ns})
            baseline_total_ns += check_ns

        if tape_owner.bundle != candidate_owner.bundle:
            raise RuntimeError("D6 paired-cycle final continuation states differ")
        return {
            "admissions": admission_records,
            "baseline_cycle_ns": baseline_total_ns,
            "baseline_gpu_ms": baseline_gpu_ms,
            "baseline_host_commit_ns": baseline_host_commit_ns,
            "candidate_cycle_ns": candidate_total_ns,
            "candidate_gpu_ms": candidate_gpu_ms,
            "candidate_host_commit_ns": candidate_host_commit_ns,
            "cycle": cycle,
            "exact_successor_sha256": candidate_owner.bundle.digest(),
            "opportunity_percent": _paired_opportunity_percent(
                baseline_ns=baseline_total_ns,
                candidate_ns=candidate_total_ns,
            ),
            "order": order,
            "paired_delta_ns": baseline_total_ns - candidate_total_ns,
            "refills": refill_records,
            "shared_schedule_preparation_ns": sum(
                int(record["shared_schedule_preparation_ns"])
                for record in tick_records
            ),
            "ticks": windows * R11B_WINDOW_TICKS_V1,
            "windows": windows,
            "tick_records": tick_records,
        }

    warmup_records = [
        execute_cycle(cycle=cycle, measured=False)
        for cycle in range(warmups)
    ]
    if warmup_records[0]["ticks"] != 25:
        raise RuntimeError("frozen current C5 trajectory must first refill at tick 25")
    cycle_records = [
        execute_cycle(cycle=cycle, measured=True)
        for cycle in range(samples)
    ]
    baseline_cycle_ms = [
        int(record["baseline_cycle_ns"]) / 1_000_000.0
        for record in cycle_records
    ]
    candidate_cycle_ms = [
        int(record["candidate_cycle_ns"]) / 1_000_000.0
        for record in cycle_records
    ]
    paired_opportunities = [
        float(record["opportunity_percent"])
        for record in cycle_records
    ]
    decision, opportunity_summary, opportunity_interval = _d6_roi_decision(
        paired_opportunities
    )
    median_opportunity = float(opportunity_summary["median"])
    if decision == "D6-ROI-INCONCLUSIVE":
        evidence_sufficiency = (
            "EXTEND_UNCHANGED_TO_63_COMPLETE_CYCLES"
            if samples < D6_EXTENSION_CYCLE_CAP
            else "CAP_REACHED_RETURN_INCONCLUSIVE"
        )
    else:
        evidence_sufficiency = "DECISION_SUPPORTED"

    order_strata = {
        order: _sample_summary(
            [
                float(record["opportunity_percent"])
                for record in cycle_records
                if record["order"] == order
            ]
        )
        for order in ("tape_first", "candidate_first")
        if any(record["order"] == order for record in cycle_records)
    }
    operations_per_tick = _operation_count(schedule)
    measured_ticks = sum(int(record["ticks"]) for record in cycle_records)
    total_operations = operations_per_tick * measured_ticks
    total_baseline_ns = sum(
        int(record["baseline_cycle_ns"]) for record in cycle_records
    )
    total_candidate_ns = sum(
        int(record["candidate_cycle_ns"]) for record in cycle_records
    )
    total_baseline_seconds = total_baseline_ns / 1_000_000_000.0
    total_candidate_seconds = total_candidate_ns / 1_000_000_000.0

    schedule_checkpoint_sha256 = schedule_sha256
    if tape_owner.bundle != candidate_owner.bundle:
        raise RuntimeError("D6 checkpoint owners do not share one exact state")
    checkpoint_samples = max(4, min(refill_samples, 10))
    checkpoint_samples -= checkpoint_samples % 2
    candidate_checkpoint_records: list[dict[str, float | int]] = []
    candidate_checkpoint_digests: list[str] = []
    checkpoint_pair_records: list[dict[str, Any]] = []
    tape_checkpoint_records: list[dict[str, float | int]] = []
    tape_checkpoint_digests: list[str] = []
    _capture_candidate_checkpoint(
        candidate_owner,
        schedule_sha256=schedule_checkpoint_sha256,
    )
    _capture_tape_checkpoint(
        tape_owner,
        tape_tensors,
        schedule_sha256=schedule_checkpoint_sha256,
    )
    for index in range(checkpoint_samples):
        if index % 2 == 0:
            order = "candidate_first"
            candidate_digest, candidate_record = _capture_candidate_checkpoint(
                candidate_owner,
                schedule_sha256=schedule_checkpoint_sha256,
            )
            tape_digest, tape_record = _capture_tape_checkpoint(
                tape_owner,
                tape_tensors,
                schedule_sha256=schedule_checkpoint_sha256,
            )
        else:
            order = "tape_first"
            tape_digest, tape_record = _capture_tape_checkpoint(
                tape_owner,
                tape_tensors,
                schedule_sha256=schedule_checkpoint_sha256,
            )
            candidate_digest, candidate_record = _capture_candidate_checkpoint(
                candidate_owner,
                schedule_sha256=schedule_checkpoint_sha256,
            )
        candidate_checkpoint_digests.append(candidate_digest)
        candidate_checkpoint_records.append(candidate_record)
        tape_checkpoint_digests.append(tape_digest)
        tape_checkpoint_records.append(tape_record)
        checkpoint_pair_records.append(
            {
                "candidate": candidate_record,
                "order": order,
                "pair": index,
                "tape_copy_hash_proxy": tape_record,
            }
        )
    if len(set(candidate_checkpoint_digests)) != 1:
        raise RuntimeError("Option D checkpoint materialization is unstable")
    if len(set(tape_checkpoint_digests)) != 1:
        raise RuntimeError("current host-tape checkpoint materialization is unstable")
    checkpoint_candidate_summary = _numeric_summary(candidate_checkpoint_records)
    checkpoint_tape_summary = _numeric_summary(tape_checkpoint_records)

    component_samples = max(5, min(samples, 15))
    component_warmups = max(1, min(warmups, 3))
    raw_profile = _component_profile(
        raw_schedule,
        gaussian_strategy="disabled",
        source_binding=source_binding,
        build_root=build_root,
        warmups=component_warmups,
        samples=component_samples,
        seed_base=0xD6100000,
    )
    gaussian_profile = _component_profile(
        gaussian_schedule,
        gaussian_strategy="software",
        source_binding=source_binding,
        build_root=build_root,
        warmups=component_warmups,
        samples=component_samples,
        seed_base=0xD6200000,
    )
    gaussian_tape_profile = _component_profile(
        gaussian_schedule,
        gaussian_strategy="host_tape_baseline",
        source_binding=source_binding,
        build_root=build_root,
        warmups=component_warmups,
        samples=component_samples,
        seed_base=0xD6200000,
        tape_capacity=32_768,
    )

    gpu_added = [
        float(record["candidate_gpu_ms"]) - float(record["baseline_gpu_ms"])
        for record in cycle_records
    ]
    host_removed = [
        (
            int(record["baseline_host_commit_ns"])
            - int(record["candidate_host_commit_ns"])
        )
        / 1_000_000.0
        for record in cycle_records
    ]
    refill_wall = [
        sum(float(item["end_to_end_ms"]) for item in record["refills"])
        for record in cycle_records
    ]
    payload: dict[str, Any] = {
        "benchmark": {
            "births_per_world": births,
            "complete_paired_cycles": samples,
            "current_source_geometry": "B48/C512/M64 five-tick scalar RNG windows",
            "measured_ticks": measured_ticks,
            "operations_per_tick": operations_per_tick,
            "schedule_sha256": schedule_sha256,
            "warmup_cycles": warmups,
            "worlds": worlds,
        },
        "candidate": {
            "backend": "standalone Option D resident CUDA software-math V1",
            "checkpoint_capture": checkpoint_candidate_summary,
            "cycle_end_to_end_ms": _sample_summary(candidate_cycle_ms),
            "resident_state_bytes": candidate_owner.resident_device_bytes,
            "state_import_ms": (
                candidate_import_end - candidate_import_start
            ) / 1_000_000.0,
            "throughput_scheduled_operations_per_second": (
                total_operations / total_candidate_seconds
            ),
        },
        "claim_boundary": (
            "Standalone B48/M64 paired-cycle characterization against the exact "
            "current generic C5 host/tape helper and five-tick admission predicate; "
            "not integrated Tensor World graph throughput or an activation claim."
        ),
        "component_profiles": {
            "gaussian_transform_dense_source_upper_envelope": {
                "current_host_tape": gaussian_tape_profile,
                "option_d_software_math": gaussian_profile,
            },
            "raw_rng_current_control_shape": raw_profile,
        },
        "current_host_tape_baseline": {
            "admission": {
                "capacity_per_world": CURRENT_C5_TAPE_CAPACITY,
                "max_births_per_tick": MAX_BIRTHS_V1,
                "requested_ticks": R11B_WINDOW_TICKS_V1,
                "warmup_cycle_ticks": [record["ticks"] for record in warmup_records],
            },
            "backend": (
                "exact current refill_cpython_gaussian_window_ helper with "
                "preallocated resident tape and current five-tick headroom"
            ),
            "checkpoint_copy_hash_proxy": checkpoint_tape_summary,
            "cycle_end_to_end_ms": _sample_summary(baseline_cycle_ms),
            "helper_receipts": helper_receipts,
            "initial_refill": initial_refill_record,
            "resident_state_and_tape_bytes": (
                tape_owner.resident_device_bytes + _tensor_bytes(tape_tensors)
            ),
            "state_import_and_initial_refill_ms": (
                tape_import_end - tape_import_start
            ) / 1_000_000.0,
            "throughput_scheduled_operations_per_second": (
                total_operations / total_baseline_seconds
            ),
        },
        "decision": decision,
        "derived": {
            "checkpoint_candidate_minus_tape_copy_hash_proxy_ms": (
                float(checkpoint_candidate_summary["end_to_end_ms"]["median"])
                - float(checkpoint_tape_summary["end_to_end_ms"]["median"])
            ),
            "end_to_end_opportunity_percent": median_opportunity,
            "aggregate_end_to_end_opportunity_percent": (
                _paired_opportunity_percent(
                    baseline_ns=total_baseline_ns,
                    candidate_ns=total_candidate_ns,
                )
            ),
            "aggregate_total_baseline_cycle_ns": total_baseline_ns,
            "aggregate_total_candidate_cycle_ns": total_candidate_ns,
            "tick_path_gpu_wall_added_ms": _sample_summary(gpu_added),
            "host_decode_wall_removed_ms": _sample_summary(host_removed),
            "paired_median_bootstrap_interval": opportunity_interval,
            "paired_opportunity_percent": opportunity_summary,
            "refill_boundary_wall_removed_ms": _sample_summary(refill_wall),
            "roi_threshold_percent": ROI_THRESHOLD_PERCENT,
            "shared_schedule_preparation_ms": _sample_summary(
                [
                    float(tick["shared_schedule_preparation_ms"])
                    for record in cycle_records
                    for tick in record["tick_records"]
                ]
            ),
        },
        "exactness": {
            "checkpoint_samples": checkpoint_samples,
            "measured_cycles_equal": samples,
            "measured_ticks_equal": measured_ticks,
            "successor_sha256": candidate_owner.bundle.digest(),
        },
        "checkpoint_raw_pair_records": checkpoint_pair_records,
        "checkpoint_scope": (
            "Candidate canonical standalone snapshot versus current RNG-state "
            "snapshot plus resident tape D2H/copy/hash proxy; not a complete "
            "Tensor World C5 replay-context checkpoint comparison."
        ),
        "memory_method": {
            "host": "observed process VmRSS end-minus-start per operation",
            "resident_vram_delta_candidate_minus_tape_bytes": (
                candidate_owner.resident_device_bytes
                - tape_owner.resident_device_bytes
                - _tensor_bytes(tape_tensors)
            ),
            "vram": "PyTorch max_memory_allocated minus pre-operation allocation",
        },
        "paired_cycle_raw_records": cycle_records,
        "provenance": {
            "current_c5_baseline_sources": {
                str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
                for path in (
                    root / "engine/perf/cuda_architecture_b_lifecycle.py",
                    root / "engine/perf/cuda_architecture_b_resident_window_runtime_v1.py",
                    root / "engine/tensor_world/c5_gaussian_window_r11.py",
                    root / "engine/tensor_world/c4_resident_rng_budget_r11.py",
                )
            },
            "cuda_device": torch.cuda.get_device_name(device),
            "cuda_source_sha256": cuda_source_digest(),
            "git": git_metadata,
            "benchmark_source_sha256": benchmark_source_sha256,
            "platform": platform.platform(),
            "resource_manager_exclusive_admission": resource_admission,
            "source_binding": source_binding,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "schema_version": "eiel_option_d_d6_characterization_v2",
        "setup": {
            "contract_and_schedule_ms": (setup_end - setup_start) / 1_000_000.0,
            "native_module_cached_load_ms": (module_end - module_start) / 1_000_000.0,
            "shared_schedule_preparation_treatment": (
                "measured once per paired tick and charged identically to both arms"
            ),
        },
        "statistical_method": {
            "decision_statistic": "median of paired complete-cycle opportunity percentages",
            "paired_median_bootstrap_interval_role": (
                "evidence-sufficiency sentinel; does not replace the prescribed median gate"
            ),
            "paired_median_bootstrap_interval_clears_threshold": (
                float(opportunity_interval["low"]) >= ROI_THRESHOLD_PERCENT
                or float(opportunity_interval["high"]) < ROI_THRESHOLD_PERCENT
            ),
            "evidence_sufficiency": evidence_sufficiency,
            "initial_complete_cycle_count": D6_INITIAL_COMPLETE_CYCLES,
            "predeclared_extension_cycle_cap": D6_EXTENSION_CYCLE_CAP,
            "checkpoint_order_counts": {
                order: sum(
                    record["order"] == order for record in checkpoint_pair_records
                )
                for order in ("candidate_first", "tape_first")
            },
            "order_strata": order_strata,
            "quantiles": "directly computed from retained raw cycle observations",
            "sum_of_marginal_quantiles": False,
        },
        "transfer_bytes": {
            "candidate_h2d_plus_d2h": sum(
                int(tick["candidate"]["h2d_bytes"])
                + int(tick["candidate"]["d2h_bytes"])
                for record in cycle_records
                for tick in record["tick_records"]
            ),
            "current_helper_refill_d2h_plus_h2d": sum(
                int(receipt["device_to_host_bytes"])
                + int(receipt["host_to_device_bytes"])
                for receipt in helper_receipts
                if receipt["phase"] == "measured"
            ),
            "tape_h2d_plus_d2h": sum(
                int(tick["tape"]["h2d_bytes"])
                + int(tick["tape"]["d2h_bytes"])
                for record in cycle_records
                for tick in record["tick_records"]
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(payload) + b"\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--resource-admission", required=True)
    parser.add_argument("--source-binding", default=SOURCE_BINDING_DEFAULT)
    parser.add_argument("--worlds", type=int, default=48)
    parser.add_argument("--births", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument(
        "--samples",
        type=int,
        default=D6_INITIAL_COMPLETE_CYCLES,
    )
    parser.add_argument("--refill-samples", type=int, default=10)
    arguments = parser.parse_args(argv)
    payload = run_d6_benchmark(
        output=arguments.output,
        build_root=arguments.build_root,
        source_binding=arguments.source_binding,
        resource_admission=arguments.resource_admission,
        worlds=arguments.worlds,
        births=arguments.births,
        warmups=arguments.warmups,
        samples=arguments.samples,
        refill_samples=arguments.refill_samples,
    )
    print(payload["decision"])
    print(payload["derived"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
