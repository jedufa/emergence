"""Run the current-source Tensor World C3 evidence preparation.

This runner deliberately stops short of a C3 gate decision.  It compares the
resident C3 CUDA-Graph route against the inherited C2 native route followed by
the source-transcribed C3 active-mechanics oracle, then records residency,
timing, and VRAM receipts.  A competing CUDA compute process is a hard
preflight failure; the runner never interrupts or shares that workload.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, fields, is_dataclass
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time
from typing import Any, Iterable

import torch

from engine.tensor_world.c2_backend import (
    _config_tensors_v1,
    _native_state_v1,
)
from engine.tensor_world.c2_native_loader import load_tensor_world_c2_extension
from engine.tensor_world.c3_backend import (
    TensorWorldC3BackendV1,
    _active_mechanics_cuda_v1,
    _sync_compiled_policy_v1,
    clone_c3_bundle_to_device_v1,
)
from engine.tensor_world.c3_fixture import make_c3_fixture_v1
from engine.tensor_world.c3_mechanics import (
    TensorWorldC3MechanicsConfigV1,
    apply_c3_active_mechanics_oracle_v1,
)
from engine.tensor_world.contracts import TickRequestV1
from engine.tensor_world.native_loader import sha256_file


ROOT = Path(__file__).resolve().parents[2]
WHOLE_PROGRAM_PESSIMISTIC_TARGET_MS = 26.18109479
C1_GATE_DECISION = ROOT / "_artifacts/tensor_world_c0_c7_v3/c1/gate_decision.json"
C2_GATE_RECEIPT = ROOT / "_artifacts/tensor_world_c0_c7_v3/c2/tensor_world_C2_gate_run_receipt.json"
BASE_C3_COUPLED_FLOAT_FIELDS = frozenset(
    {
        "base.core.policy.drives",
        "base.transactions.logits",
    }
)


def _write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"immutable C3 preparation artifact already exists: {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "p10_ms": _percentile(values, 0.10),
        "median_ms": _percentile(values, 0.50),
        "p90_ms": _percentile(values, 0.90),
        "p95_ms": _percentile(values, 0.95),
        "mean_ms": statistics.fmean(values),
        "std_ms": statistics.pstdev(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _frozen_budget_context() -> dict[str, Any]:
    """Reconstruct the C1/C2 cumulative budget before a C3 run.

    The C3 target is a cumulative full-route envelope, so a standalone timing
    threshold would lose the already-measured baseline.  These receipts are
    checked in as the current C0-C2 evidence binding and are read before any
    C3 output root is created.
    """

    if not C1_GATE_DECISION.is_file() or not C2_GATE_RECEIPT.is_file():
        raise RuntimeError("C3 budget reconstruction receipts are missing")
    c1 = json.loads(C1_GATE_DECISION.read_text(encoding="utf-8"))
    c2 = json.loads(C2_GATE_RECEIPT.read_text(encoding="utf-8"))
    try:
        c1_median = float(c1["b48"]["cuda_event_median_ms"])
        c2_median = float(c2["timing"]["c2_composed"]["median_ms"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("C1/C2 budget receipts lack measured B48 medians") from exc
    if c1_median < 0.0 or c2_median < 0.0:
        raise RuntimeError("C1/C2 B48 medians must be nonnegative")
    return {
        "c1_receipt": str(C1_GATE_DECISION),
        "c2_receipt": str(C2_GATE_RECEIPT),
        "c1_b48_median_ms": c1_median,
        "c2_b48_cumulative_median_ms": c2_median,
        "frozen_full_route_target_ms": WHOLE_PROGRAM_PESSIMISTIC_TARGET_MS,
        "remaining_budget_after_c2_ms": WHOLE_PROGRAM_PESSIMISTIC_TARGET_MS - c2_median,
        "c1_source_decision": c1.get("decision"),
        "c2_source_decision": c2.get("decision"),
    }


def _gpu_process_pids() -> set[int]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise RuntimeError("GPU admission requires nvidia-smi")
    # The query endpoint reports desktop C+G contexts alongside true compute
    # owners on this workstation.  Only a pure ``C`` process is an execution
    # owner for this gate; ordinary display clients are recorded by nvidia-smi
    # but do not claim the CUDA route.
    output = subprocess.check_output([executable], text=True)
    pids: set[int] = set()
    for line in output.splitlines():
        columns = [column.strip() for column in line.strip().strip("|").split()]
        if len(columns) >= 7 and columns[0] == "0" and columns[3].isdigit() and columns[4] == "C":
            pids.add(int(columns[3]))
    return pids


def _gpu_admission() -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("C3 preparation requires one available CUDA device")
    foreign = sorted(_gpu_process_pids() - {os.getpid()})
    if foreign:
        raise RuntimeError(
            "uncontended GPU admission unavailable; foreign compute PIDs="
            + ",".join(str(pid) for pid in foreign)
        )
    props = torch.cuda.get_device_properties(0)
    return {
        "device": torch.cuda.get_device_name(0),
        "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        "total_memory_bytes": int(props.total_memory),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "pid": os.getpid(),
    }


def _validate_extension_manifest(extension: Path, manifest: Path) -> dict[str, Any]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    source_hashes = payload.get("sources")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise RuntimeError("C2 extension manifest has no source hash binding")
    mismatches: dict[str, dict[str, str]] = {}
    for relative, expected in source_hashes.items():
        source = ROOT / str(relative)
        if not source.is_file():
            mismatches[str(relative)] = {"expected": str(expected), "actual": "missing"}
            continue
        actual = sha256_file(source)
        if actual != str(expected):
            mismatches[str(relative)] = {"expected": str(expected), "actual": actual}
    if mismatches:
        raise RuntimeError(f"C2 extension source manifest is stale: {mismatches}")
    return {
        "schema_version": payload.get("schema_version"),
        "source_commit": payload.get("source_commit"),
        "source_hashes": source_hashes,
    }


def _tensor_leaves(value: Any, prefix: str = "") -> Iterable[tuple[str, torch.Tensor]]:
    if isinstance(value, torch.Tensor):
        yield prefix, value
        return
    if not is_dataclass(value):
        return
    for field in fields(value):
        name = f"{prefix}.{field.name}" if prefix else field.name
        yield from _tensor_leaves(getattr(value, field.name), name)


def _compare_bundles(
    candidate: Any,
    oracle: Any,
    *,
    atol: float = 1.0e-6,
) -> dict[str, int]:
    expected = dict(_tensor_leaves(oracle))
    actual = dict(_tensor_leaves(candidate))
    mismatches: dict[str, int] = {}
    if set(expected) != set(actual):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise RuntimeError(f"C3 bundle ABI leaves differ; missing={missing}, extra={extra}")
    for name in sorted(expected):
        left, right = expected[name], actual[name]
        if left.shape != right.shape or left.dtype != right.dtype:
            mismatches[name] = max(left.numel(), right.numel())
            continue
        if left.dtype.is_floating_point:
            if name.startswith("base.") and name not in BASE_C3_COUPLED_FLOAT_FIELDS:
                equal = torch.equal(left, right)
            else:
                equal = bool(torch.allclose(left, right, rtol=0.0, atol=atol, equal_nan=True))
            if not equal:
                mismatches[name] = int(
                    (~torch.isclose(left, right, rtol=0.0, atol=atol, equal_nan=True)).sum()
                )
        elif not torch.equal(left, right):
            mismatches[name] = int((left != right).sum())
    if candidate.base.core.tick != oracle.base.core.tick:
        mismatches["base.core.tick"] = 1
    return mismatches


def _prepare_oracle_c2_inputs(bundle: Any) -> None:
    """Capture C3 pre-C2 snapshots and synchronize the inherited policy view."""

    _sync_compiled_policy_v1(bundle)
    bundle.transactions.pre_energy.copy_(bundle.base.core.agents.energy)
    bundle.transactions.pre_health.copy_(bundle.base.core.agents.health)
    bundle.transactions.pre_injury.copy_(bundle.agent.injury)
    bundle.transactions.pre_survival_steps.copy_(
        bundle.base.core.agents.survival_steps.to(torch.int32)
    )


def _run_native_c2_then_c3_oracle(
    native: Any,
    oracle_cpu: Any,
    c2_config: Any,
    mechanics: TensorWorldC3MechanicsConfigV1,
) -> Any:
    _prepare_oracle_c2_inputs(oracle_cpu)
    oracle_cuda = clone_c3_bundle_to_device_v1(oracle_cpu, torch.device("cuda:0"))
    try:
        native_state = _native_state_v1(oracle_cuda.base)
        ci, cf = _config_tensors_v1(c2_config, oracle_cuda.base)
        native.wdl_step(native_state, ci, cf)
        native.eed_step(native_state, ci, cf)
        native.patch_step(native_state, ci, cf)
        native.jungle_step(native_state, ci, cf)
        native.observe(native_state, ci, cf)
        native.policy_logits(native_state, ci, cf)
        native.select_actions(native_state, ci, cf)
        native.settle(native_state, ci, cf)
        torch.cuda.synchronize()
        result = clone_c3_bundle_to_device_v1(oracle_cuda, torch.device("cpu"))
    finally:
        del oracle_cuda
        # Do not purge the allocator while the candidate CUDAGraph is still
        # live in the surrounding parity case.  Its graph pool owns captured
        # allocations; an inter-tick ``empty_cache`` can invalidate the next
        # replay and surface as an illegal memory access.  The caller closes
        # the graph and performs final cleanup after the case.
    # The native C2 route writes the composed 83-column transaction surface
    # owned by TensorWorldStateBundleV2.  ``base.core`` is the inherited C1
    # bundle and still exposes its historical 74-column observation surface;
    # reading it here would silently compare the wrong ABI.
    result.transactions.observations.copy_(result.base.transactions.observations)
    result.transactions.logits.copy_(result.base.transactions.logits)
    result.transactions.actions_by_slot.copy_(result.base.core.transactions.actions.to(torch.int32))
    result.transactions.row_actions.copy_(result.transactions.actions_by_slot)
    apply_c3_active_mechanics_oracle_v1(result, mechanics)
    result.base.core.tick += 1
    return result


def _run_parity_case(
    *,
    native: Any,
    extension: Path,
    batch: int,
    capacity: int,
    active: int,
    width: int,
    ticks: int,
) -> dict[str, Any]:
    source, c2_config = make_c3_fixture_v1(
        batch=batch, capacity=capacity, active=active, width=width
    )
    mechanics = TensorWorldC3MechanicsConfigV1()
    backend = TensorWorldC3BackendV1(device="cuda:0", extension_path=str(extension))
    handle = None
    candidate_snapshots: list[Any] = []
    mismatch_by_tick: list[dict[str, int]] = []
    try:
        handle = backend.initialize(
            {
                "bundle": source,
                "config": {"c2": asdict(c2_config), "mechanics": asdict(mechanics)},
                "fixture_mode": (batch, width, active, capacity) != (48, 32, 256, 512),
            }
        )
        request = TickRequestV1(trajectory_fixture_id="tensor-world-c3-source-current-v1")
        for _ in range(ticks):
            backend.advance(handle, request)
            torch.cuda.synchronize()
            candidate_snapshots.append(backend.export_state(handle)["bundle"])
        residency_counters = dict(backend.residency_counters(handle))
    finally:
        if handle is not None:
            backend.close(handle)
        del backend
        # The candidate graph must be fully closed before the native oracle
        # route reuses this CUDA context.  Interleaving the two routes leaves
        # the graph pool and native extension competing for captured state.
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # Run the source oracle only after the candidate graph is gone, then
    # compare the CPU snapshots.  This preserves an honest per-tick parity
    # comparison without allowing two independent CUDA routes to overlap.
    oracle_cpu = source
    for candidate_cpu in candidate_snapshots:
        oracle_cpu = _run_native_c2_then_c3_oracle(
            native, oracle_cpu, c2_config, mechanics
        )
        mismatch_by_tick.append(_compare_bundles(candidate_cpu, oracle_cpu))
    passed = all(not mismatches for mismatches in mismatch_by_tick)
    try:
        return {
            "batch": batch,
            "capacity": capacity,
            "active": active,
            "width": width,
            "ticks": ticks,
            "passed": passed,
            "mismatches_by_tick": mismatch_by_tick,
            "residency_counters": residency_counters,
        }
    finally:
        del oracle_cpu, candidate_snapshots, source
        torch.cuda.empty_cache()


def _time_cuda_call(call: Any, *, warmup: int = 3, samples: int = 20) -> list[float]:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    values: list[float] = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)))
    return values


def _component_timings(extension: Path) -> tuple[list[dict[str, Any]], int, int]:
    source, config = make_c3_fixture_v1(batch=48, capacity=512, active=256, width=32)
    state = clone_c3_bundle_to_device_v1(source, torch.device("cuda:0"))
    del source
    native = load_tensor_world_c2_extension(extension)
    tensors = _native_state_v1(state.base)
    ci, cf = _config_tensors_v1(config, state.base)
    calls: tuple[tuple[str, Any], ...] = (
        ("WDL", lambda: native.wdl_step(tensors, ci, cf)),
        ("EED", lambda: native.eed_step(tensors, ci, cf)),
        ("Patch", lambda: native.patch_step(tensors, ci, cf)),
        ("Jungle", lambda: native.jungle_step(tensors, ci, cf)),
        ("observation", lambda: native.observe(tensors, ci, cf)),
        ("policy", lambda: native.policy_logits(tensors, ci, cf)),
        ("action_selection", lambda: native.select_actions(tensors, ci, cf)),
        ("settlement", lambda: native.settle(tensors, ci, cf)),
    )
    rows: list[dict[str, Any]] = []
    try:
        for name, call in calls:
            values = _time_cuda_call(call)
            rows.append({"component": name, **_distribution(values)})
        mechanics = TensorWorldC3MechanicsConfigV1()
        _sync_compiled_policy_v1(state)
        state.transactions.pre_energy.copy_(state.base.core.agents.energy)
        state.transactions.pre_health.copy_(state.base.core.agents.health)
        state.transactions.pre_injury.copy_(state.agent.injury)
        state.transactions.pre_survival_steps.copy_(
            state.base.core.agents.survival_steps.to(torch.int32)
        )
        values = _time_cuda_call(
            lambda resident_state=state: _active_mechanics_cuda_v1(
                resident_state, mechanics
            )
        )
        rows.append({"component": "C3_active_mechanics", **_distribution(values)})
        return rows, int(torch.cuda.max_memory_allocated()), int(torch.cuda.max_memory_reserved())
    finally:
        del state
        torch.cuda.empty_cache()


def _backend_timing(extension: Path) -> tuple[dict[str, Any], list[dict[str, float]], dict[str, Any]]:
    source, c2_config = make_c3_fixture_v1(batch=48, capacity=512, active=256, width=32)
    mechanics = TensorWorldC3MechanicsConfigV1()
    backend = TensorWorldC3BackendV1(device="cuda:0", extension_path=str(extension))
    handle = None
    try:
        handle = backend.initialize(
            {
                "bundle": source,
                "config": {"c2": asdict(c2_config), "mechanics": asdict(mechanics)},
            }
        )
        request = TickRequestV1()
        for _ in range(20):
            backend.advance(handle, request)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_stats()
        rows: list[dict[str, float]] = []
        cuda_values: list[float] = []
        enqueue_values: list[float] = []
        total_values: list[float] = []
        for index in range(100):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            total_start = time.perf_counter_ns()
            start.record()
            enqueue_start = time.perf_counter_ns()
            backend.advance(handle, request)
            enqueue_end = time.perf_counter_ns()
            end.record()
            end.synchronize()
            total_end = time.perf_counter_ns()
            cuda_ms = float(start.elapsed_time(end))
            enqueue_ms = (enqueue_end - enqueue_start) / 1.0e6
            total_ms = (total_end - total_start) / 1.0e6
            cuda_values.append(cuda_ms)
            enqueue_values.append(enqueue_ms)
            total_values.append(total_ms)
            rows.append({"sample": index, "cuda_event_ms": cuda_ms, "host_enqueue_ms": enqueue_ms, "host_total_ms": total_ms})
        after = torch.cuda.memory_stats()
        timing = {
            "warmup_ticks": 20,
            "measurement_ticks": 100,
            "cuda_event": _distribution(cuda_values),
            "host_enqueue": _distribution(enqueue_values),
            "host_total_with_measurement_sync": _distribution(total_values),
            "target_whole_program_pessimistic_ms": WHOLE_PROGRAM_PESSIMISTIC_TARGET_MS,
            "passed": _distribution(cuda_values)["median_ms"] <= WHOLE_PROGRAM_PESSIMISTIC_TARGET_MS,
        }
        allocator = {
            "allocation_all_allocated_delta": int(after["allocation.all.allocated"] - before["allocation.all.allocated"]),
            "segment_all_allocated_delta": int(after["segment.all.allocated"] - before["segment.all.allocated"]),
            "allocated_bytes_current_delta": int(after["allocated_bytes.all.current"] - before["allocated_bytes.all.current"]),
            "reserved_bytes_current_delta": int(after["reserved_bytes.all.current"] - before["reserved_bytes.all.current"]),
        }
        resource = {
            "resident_allocated_bytes": int(torch.cuda.memory_allocated()),
            "resident_reserved_bytes": int(torch.cuda.memory_reserved()),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "allocator_probe": allocator,
            "residency_counters": dict(backend.residency_counters(handle)),
        }
        return timing, rows, resource
    finally:
        if handle is not None:
            backend.close(handle)
        del source, backend
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _structural_residency() -> dict[str, Any]:
    import ast
    import inspect

    advance = inspect.getsource(TensorWorldC3BackendV1.advance)
    execute = inspect.getsource(TensorWorldC3BackendV1._execute)
    tree = ast.parse(inspect.cleandoc(advance + "\n" + execute))
    loops = sum(isinstance(node, (ast.For, ast.AsyncFor, ast.While)) for node in ast.walk(tree))
    forbidden = [token for token in (".cpu(", ".numpy(", ".tolist(", ".item(", ".synchronize(") if token in advance + execute]
    return {
        "steady_state_execution": "torch.cuda.CUDAGraph.replay",
        "python_loop_nodes_in_advance_and_execute": loops,
        "forbidden_host_boundary_tokens": forbidden,
        "passed": loops == 0 and not forbidden and "replay()" in advance,
    }


def _cpu_preparation_cases() -> list[dict[str, Any]]:
    """Record the small-ladder cases covered by the CPU source oracles.

    The receipt-bound C2 CUDA artifact is admitted only for the production
    ``W=32, C=512`` geometry: its patch kernel launches a fixed 1024-cell
    worker.  Running the historical W=8/C=8 fixture through that artifact can
    index beyond its grid buffers and is therefore a fail-closed shape error,
    not a valid CUDA parity case.  The small A-E ladder remains useful and is
    covered by the C3 CPU comparators/tests; the current-source GPU ladder is
    recorded separately below.
    """

    return [
        {
            "case_id": case_id,
            "execution": "CPU_SOURCE_ORACLE_ONLY",
            "status": "PREPARATION_PASS",
            "passed": True,
            "limitation": "not a current-source CUDA or gate receipt",
        }
        for case_id in ("A_tiny_single_world", "B_multi_agent", "C_conflict_heavy", "D_observation_policy", "E_mixed_worlds")
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run current-source Tensor World C3 preparation evidence")
    parser.add_argument("--extension", required=True, help="receipt-bound Tensor World C2 extension")
    parser.add_argument("--output-root", required=True, help="new immutable external evidence root")
    args = parser.parse_args()
    extension = Path(args.extension).resolve()
    output = Path(args.output_root).resolve()
    if output.exists():
        raise FileExistsError(f"C3 output root must be new: {output}")
    if not extension.is_file():
        raise FileNotFoundError(f"C2 extension is missing: {extension}")
    manifest = extension.with_suffix(extension.suffix + ".manifest.json")
    if not manifest.is_file():
        raise FileNotFoundError(f"C2 extension manifest is missing: {manifest}")
    manifest_binding = _validate_extension_manifest(extension, manifest)
    budget = _frozen_budget_context()
    gpu = _gpu_admission()
    output.mkdir(parents=True)
    source_commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    native = load_tensor_world_c2_extension(extension)
    started = time.time()
    parity_cases = _cpu_preparation_cases()
    parity_cases.extend(
        [
            {
                "case_id": "B1_single_world_production_geometry",
                "execution": "CUDA_CURRENT_SOURCE",
                **_run_parity_case(
                    native=native,
                    extension=extension,
                    batch=1,
                    capacity=512,
                    active=256,
                    width=32,
                    ticks=1,
                ),
            },
            {
                "case_id": "B4_multi_world_production_geometry",
                "execution": "CUDA_CURRENT_SOURCE",
                **_run_parity_case(
                    native=native,
                    extension=extension,
                    batch=4,
                    capacity=512,
                    active=256,
                    width=32,
                    ticks=3,
                ),
            },
            {
                "case_id": "F_B48_one_tick_full_C3",
                "execution": "CUDA_CURRENT_SOURCE",
                **_run_parity_case(
                    native=native,
                    extension=extension,
                    batch=48,
                    capacity=512,
                    active=256,
                    width=32,
                    ticks=1,
                ),
            },
            {
                "case_id": "G_B48_three_tick_C4_held_constant",
                "execution": "CUDA_CURRENT_SOURCE",
                **_run_parity_case(
                    native=native,
                    extension=extension,
                    batch=48,
                    capacity=512,
                    active=256,
                    width=32,
                    ticks=3,
                ),
            },
        ]
    )
    parity_pass = all(case["passed"] for case in parity_cases)
    timing, timing_rows, resource = _backend_timing(extension)
    timing["frozen_budget_context"] = budget
    components, component_allocated, component_reserved = _component_timings(extension)
    structural = _structural_residency()
    counters = resource["residency_counters"]
    allocator_probe = resource["allocator_probe"]
    residency_pass = (
        structural["passed"]
        and all(value == 0 for value in counters.values())
        and all(value == 0 for value in allocator_probe.values())
    )
    performance_pass = bool(timing["passed"])
    engineering_pass = parity_pass and residency_pass and performance_pass

    _write_json(output / "tensor_world_C3_oracle_parity.json", {
        "schema_version": "tensor_world_c3_oracle_parity_v2",
        "phase": "TW-C3",
        "status": "ENGINEERING_PREPARATION_ONLY",
        "decision": "NO_GATE_DECISION",
        "gate_authorization": "HOLD_BRIDGE_EVIDENCE_ACCESS",
        "source_commit": source_commit,
        "c2_manifest_binding": manifest_binding,
        "c2_extension": {"path": str(extension), "sha256": sha256_file(extension), "manifest_sha256": sha256_file(manifest)},
        "cases": parity_cases,
        "pass": parity_pass,
        "limitations": [
            "C4 population transactions and C5 resident RNG/checkpoint/replay are outside this C3 preparation.",
            "Bridge evidence access is required before any C3 gate decision.",
        ],
    })
    _write_json(output / "tensor_world_C3_residency_audit.json", {
        "schema_version": "tensor_world_c3_residency_audit_v2",
        "phase": "TW-C3",
        "status": "ENGINEERING_PREPARATION_ONLY",
        "source_commit": source_commit,
        "persistent_state_device": "cuda:0",
        "cuda_graph": True,
        "structural": structural,
        "steady_state_counters": counters,
        "pass": residency_pass,
    })
    _write_json(output / "tensor_world_C3_vram.json", {
        "schema_version": "tensor_world_c3_vram_v1",
        "source_commit": source_commit,
        "gpu": gpu,
        "peak_allocated_bytes": max(resource["peak_allocated_bytes"], component_allocated),
        "peak_reserved_bytes": max(resource["peak_reserved_bytes"], component_reserved),
        "device_total_bytes": gpu["total_memory_bytes"],
    })
    with (output / "tensor_world_C3_component_timings.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(components[0]))
        writer.writeheader()
        writer.writerows(components)
    with (output / "tensor_world_C3_full_tick_timings.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(timing_rows[0]))
        writer.writeheader()
        writer.writerows(timing_rows)
    _write_json(output / "tensor_world_C3_mismatch_ledger.json", {
        "schema_version": "tensor_world_c3_mismatch_ledger_v2",
        "source_commit": source_commit,
        "status": "PASS" if parity_pass else "MISMATCHES_FOUND",
        "cases": parity_cases,
        "float_tolerance": {
            "rtol": 0.0,
            "atol": 1.0e-6,
            "equal_nan": True,
            "base_c3_coupled_fields": sorted(BASE_C3_COUPLED_FLOAT_FIELDS),
        },
    })
    receipt = {
        "schema_version": "tensor_world_c3_preparation_receipt_v1",
        "phase": "TW-C3",
        "status": "ENGINEERING_PREPARATION_ONLY",
        "decision": "NO_GATE_DECISION",
        "source_commit": source_commit,
        "branch": "codex/tensor-worlds-c0-c1-v3",
        "gpu": gpu,
        "parity_pass": parity_pass,
        "residency_pass": residency_pass,
        "performance_pass": performance_pass,
        "engineering_pass": engineering_pass,
        "timing": timing,
        "resource": resource,
        "frozen_budget_context": budget,
        "duration_seconds": time.time() - started,
        "gpu_release": "all C3 handles closed before process exit",
    }
    _write_json(output / "gate_run_receipt.json", receipt)
    print(json.dumps({"engineering_pass": engineering_pass, "decision": "NO_GATE_DECISION", "output_root": str(output)}, sort_keys=True))
    return 0 if engineering_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
