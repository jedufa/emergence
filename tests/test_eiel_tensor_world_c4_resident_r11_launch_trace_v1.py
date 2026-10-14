"""B48/C32 Nsight launch-boundary harness for the qualified R11 window API."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import copy
from dataclasses import fields, is_dataclass
import inspect
import json
import os
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest
import torch
from torch.profiler import ProfilerActivity, profile, record_function

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


NVTX_RANGE = "eiel_c4_r11_qualified_window_replay_only_v1"
TORCH_PROFILER_RANGE = "eiel_c4_r11_torch_profiler_replay_only_v1"


class R11LaunchTraceError(RuntimeError):
    """Raised when an exported Nsight trace violates the replay-only ABI."""


def _move_dataclass_tensors(value: object, device: torch.device) -> object:
    clone = copy.deepcopy(value)

    def move(owner: object) -> None:
        if not is_dataclass(owner):
            return
        for field in fields(owner):
            item = getattr(owner, field.name)
            if isinstance(item, torch.Tensor):
                setattr(owner, field.name, item.to(device=device))
            elif is_dataclass(item):
                move(item)

    move(clone)
    return clone


def _measured_graph_replay_only_v1(owner: object) -> object:
    torch.cuda.nvtx.range_push(NVTX_RANGE)
    try:
        result = execute_c4_resident_composed_r11_window_cuda_v1(owner)
    finally:
        torch.cuda.nvtx.range_pop()
    return result


def _table_exists(connection: sqlite3.Connection, name: str) -> bool:
    return connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def _range_bounds(connection: sqlite3.Connection) -> tuple[int, int]:
    if not _table_exists(connection, "NVTX_EVENTS"):
        raise R11LaunchTraceError("Nsight SQLite export lacks NVTX_EVENTS")
    rows = connection.execute(
        """
        SELECT event.start, event.end
        FROM NVTX_EVENTS AS event
        LEFT JOIN StringIds AS strings ON strings.id = event.textId
        WHERE COALESCE(event.text, strings.value) = ? AND event.end IS NOT NULL
        """,
        (NVTX_RANGE,),
    ).fetchall()
    if len(rows) != 1:
        raise R11LaunchTraceError(
            f"expected one {NVTX_RANGE!r} range, observed {len(rows)}"
        )
    return int(rows[0][0]), int(rows[0][1])


def _api_names(
    connection: sqlite3.Connection, table: str, start: int, end: int
) -> list[str]:
    if not _table_exists(connection, table):
        return []
    return [
        str(row[0])
        for row in connection.execute(
            f"""
            SELECT strings.value
            FROM {table} AS activity
            JOIN StringIds AS strings ON strings.id = activity.nameId
            WHERE activity.start >= ? AND activity.end <= ?
            """,
            (start, end),
        )
    ]


def validate_r11_launch_trace_sqlite_v1(sqlite_path: Path) -> dict[str, object]:
    """Fail closed unless the NVTX span contains only one graph replay launch."""

    with sqlite3.connect(sqlite_path) as connection:
        start, end = _range_bounds(connection)
        runtime = _api_names(
            connection, "CUPTI_ACTIVITY_KIND_RUNTIME", start, end
        )
        driver = _api_names(connection, "CUPTI_ACTIVITY_KIND_DRIVER", start, end)
        api_names = runtime + driver
        graph_launches = [name for name in runtime if "cudaGraphLaunch" in name]
        escaped_kernel_launches = [
            name
            for name in api_names
            if "LaunchKernel" in name and "Graph" not in name
        ]
        synchronizations = [name for name in api_names if "Synchronize" in name]
        allocators = [
            name
            for name in api_names
            if any(token in name for token in ("Malloc", "Alloc", "Free"))
        ]
        callbacks = [
            name
            for name in api_names
            if any(token in name for token in ("LaunchHostFunc", "AddCallback"))
        ]
        h2d_d2h = 0
        if _table_exists(connection, "CUPTI_ACTIVITY_KIND_MEMCPY"):
            h2d_d2h = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_MEMCPY
                    WHERE start >= ? AND end <= ? AND copyKind IN (1, 2)
                    """,
                    (start, end),
                ).fetchone()[0]
            )
    receipt = {
        "nvtx_range": NVTX_RANGE,
        "cuda_graph_launches": len(graph_launches),
        "escaped_kernel_launches": len(escaped_kernel_launches),
        "h2d_d2h_operations": h2d_d2h,
        "synchronization_calls": len(synchronizations),
        "allocator_calls": len(allocators),
        "callback_calls": len(callbacks),
        "runtime_api_names": runtime,
        "driver_api_names": driver,
    }
    expected = {
        "cuda_graph_launches": 1,
        "escaped_kernel_launches": 0,
        "h2d_d2h_operations": 0,
        "synchronization_calls": 0,
        "allocator_calls": 0,
        "callback_calls": 0,
    }
    failures = {
        key: (receipt[key], value)
        for key, value in expected.items()
        if receipt[key] != value
    }
    if failures:
        raise R11LaunchTraceError(f"R11 replay-only trace failed: {failures}")
    return receipt


def validate_r11_torch_profiler_names_v1(names: list[str]) -> dict[str, object]:
    """Fail closed on raw CUPTI event names below the marked replay scope."""

    counts = Counter(names)
    lowered = [(name, name.lower()) for name in names]
    graph_launches = [
        name for name, lower in lowered if lower.startswith("cudagraphlaunch")
    ]
    escaped_kernel_launches = [
        name
        for name, lower in lowered
        if lower.startswith(("cudalaunchkernel", "culaunchkernel"))
    ]
    memcpy_api_calls = [
        name
        for name, lower in lowered
        if lower.startswith(("cudamemcpy", "cumemcpy"))
    ]
    h2d_d2h = [
        name
        for name, lower in lowered
        if any(
            token in lower
            for token in ("htod", "dtoh", "h2d", "d2h", "host to device", "device to host")
        )
    ]
    d2d = [
        name
        for name, lower in lowered
        if any(token in lower for token in ("dtod", "d2d", "device to device"))
    ]
    synchronizations = [
        name
        for name, lower in lowered
        if lower.startswith(("cuda", "cu"))
        and any(token in lower for token in ("synchronize", "streamwait", "eventwait"))
    ]
    allocators = [
        name
        for name, lower in lowered
        if lower.startswith(("cuda", "cu"))
        and any(token in lower for token in ("malloc", "alloc", "free"))
    ]
    callbacks = [
        name
        for name, lower in lowered
        if lower.startswith(("cuda", "cu"))
        and any(token in lower for token in ("callback", "launchhostfunc"))
    ]
    receipt = {
        "profiler_range": TORCH_PROFILER_RANGE,
        "cuda_graph_launches": len(graph_launches),
        "escaped_kernel_launches": len(escaped_kernel_launches),
        "memcpy_api_calls": len(memcpy_api_calls),
        "h2d_d2h_events": len(h2d_d2h),
        "d2d_memcpy_events": len(d2d),
        "synchronization_calls": len(synchronizations),
        "allocator_free_calls": len(allocators),
        "callback_calls": len(callbacks),
        "raw_event_name_counts": dict(sorted(counts.items())),
    }
    expected = {
        "cuda_graph_launches": 1,
        "escaped_kernel_launches": 0,
        "memcpy_api_calls": 0,
        "h2d_d2h_events": 0,
        "synchronization_calls": 0,
        "allocator_free_calls": 0,
        "callback_calls": 0,
    }
    failures = {
        key: (receipt[key], value)
        for key, value in expected.items()
        if receipt[key] != value
    }
    if failures:
        raise R11LaunchTraceError(
            f"R11 torch.profiler replay-only trace failed: {failures}; "
            f"raw={receipt['raw_event_name_counts']}"
        )
    return receipt


def _events_below_profiler_scope_v1(profiler: object) -> list[str]:
    events = list(profiler.events())
    scopes = [
        event
        for event in events
        if event.name == TORCH_PROFILER_RANGE
        and event.device_type == torch.autograd.DeviceType.CPU
    ]
    if len(scopes) != 1:
        raise R11LaunchTraceError(
            "expected one CPU torch profiler range, "
            f"observed {len(scopes)}"
        )
    scope = scopes[0]
    names: list[str] = []
    for event in events:
        parent = event.cpu_parent
        while parent is not None:
            if parent is scope:
                names.append(str(event.name))
                break
            parent = parent.cpu_parent
    if not names:
        raise R11LaunchTraceError("torch profiler range has no raw child events")
    return names


def test_r11_measured_nvtx_body_has_exactly_one_qualified_api_call() -> None:
    source = inspect.getsource(_measured_graph_replay_only_v1)
    function = ast.parse(source).body[0]
    try_node = next(node for node in ast.walk(function) if isinstance(node, ast.Try))
    measured_calls = [
        ast.unparse(node.func)
        for statement in try_node.body
        for node in ast.walk(statement)
        if isinstance(node, ast.Call)
    ]
    assert measured_calls == ["execute_c4_resident_composed_r11_window_cuda_v1"]
    assert "prepare_c4_resident_composed_r11_window_control_v1" not in ast.unparse(
        try_node
    )
    assert "synchronize" not in ast.unparse(try_node)

    harness = inspect.getsource(_make_prepared_b48_c32_owner_v1)
    bind_at = harness.index("bind_c4_resident_composed_r11_v1(")
    capture_at = harness.index("capture_c4_resident_composed_r11_window_v1(")
    prepare_at = harness.index(
        "prepare_c4_resident_composed_r11_window_control_v1("
    )
    synchronize_at = harness.index("torch.cuda.synchronize()")
    assert bind_at < capture_at < prepare_at < synchronize_at
    launch_harness = inspect.getsource(
        test_r11_b48_c32_qualified_window_launch_trace
    )
    assert launch_harness.index("_make_prepared_b48_c32_owner_v1(") < (
        launch_harness.index("_measured_graph_replay_only_v1(owner)")
    )

    fallback = inspect.getsource(test_r11_b48_c32_torch_profiler_launch_trace)
    fallback_function = ast.parse(fallback).body[0]
    record_scope = next(
        node
        for node in ast.walk(fallback_function)
        if isinstance(node, ast.With)
        and "record_function" in ast.unparse(node.items[0].context_expr)
    )
    scoped_calls = [
        ast.unparse(node.func)
        for statement in record_scope.body
        for node in ast.walk(statement)
        if isinstance(node, ast.Call)
    ]
    assert scoped_calls == ["_measured_graph_replay_only_v1"]


def test_r11_launch_trace_sqlite_contract(tmp_path: Path) -> None:
    path = tmp_path / "trace.sqlite"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE NVTX_EVENTS(
                start INTEGER NOT NULL, end INTEGER, text TEXT, textId INTEGER
            );
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(
                start INTEGER NOT NULL, end INTEGER NOT NULL, nameId INTEGER NOT NULL
            );
            CREATE TABLE CUPTI_ACTIVITY_KIND_DRIVER(
                start INTEGER NOT NULL, end INTEGER NOT NULL, nameId INTEGER NOT NULL
            );
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(
                start INTEGER NOT NULL, end INTEGER NOT NULL, copyKind INTEGER
            );
            INSERT INTO StringIds VALUES (1, 'cudaGraphLaunch');
            INSERT INTO NVTX_EVENTS VALUES (
                100, 200, 'eiel_c4_r11_qualified_window_replay_only_v1', NULL
            );
            INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (120, 180, 1);
            """
        )
    receipt = validate_r11_launch_trace_sqlite_v1(path)
    assert receipt["cuda_graph_launches"] == 1
    assert all(
        receipt[key] == 0
        for key in (
            "escaped_kernel_launches",
            "h2d_d2h_operations",
            "synchronization_calls",
            "allocator_calls",
            "callback_calls",
        )
    )


def test_r11_torch_profiler_raw_name_contract_is_fail_closed() -> None:
    receipt = validate_r11_torch_profiler_names_v1(["cudaGraphLaunch"])
    assert receipt["cuda_graph_launches"] == 1
    assert receipt["raw_event_name_counts"] == {"cudaGraphLaunch": 1}
    with pytest.raises(R11LaunchTraceError, match="escaped_kernel_launches"):
        validate_r11_torch_profiler_names_v1(
            ["cudaGraphLaunch", "cudaLaunchKernel"]
        )


def test_r11_torch_profiler_selects_cpu_scope_over_cuda_mirror() -> None:
    cpu_scope = SimpleNamespace(
        name=TORCH_PROFILER_RANGE,
        device_type=torch.autograd.DeviceType.CPU,
        cpu_parent=None,
    )
    cuda_scope = SimpleNamespace(
        name=TORCH_PROFILER_RANGE,
        device_type=torch.autograd.DeviceType.CUDA,
        cpu_parent=None,
    )
    graph_launch = SimpleNamespace(
        name="cudaGraphLaunch",
        device_type=torch.autograd.DeviceType.CPU,
        cpu_parent=cpu_scope,
    )
    profiler = SimpleNamespace(events=lambda: [cpu_scope, cuda_scope, graph_launch])

    assert _events_below_profiler_scope_v1(profiler) == ["cudaGraphLaunch"]


def _make_prepared_b48_c32_owner_v1(tmp_path: Path) -> object:
    fixture = make_c4_resident_multitick_r10_fixture_v1(seed=181)
    bundle = _move_dataclass_tensors(fixture.bundle, torch.device("cuda"))
    replay = _move_dataclass_tensors(
        make_c5_pcg64_context_v1(
            fixture.bundle, streams=2, seed=181, typed=True
        ),
        torch.device("cuda"),
    )
    staging = make_c4_resident_composed_r11_staging_v1(
        batch=48,
        capacity=32,
        grid_size=8,
        streams=int(replay.rng_state_words.shape[1]),
        device="cuda",
    )
    build_dir = Path(os.environ.get("EIEL_C4_R11_BUILD_DIR", tmp_path / "build"))
    c2_path = os.environ.get(
        "TENSOR_WORLD_C2_PREBUILT",
        "/tmp/emergence-test/runtime/evidence_raw/tensor_world/production_c0_c7/"
        "20260815_c2_430489534_build_01/tensor_world_c2_ext.so",
    )
    owner = bind_c4_resident_composed_r11_v1(
        bundle,
        replay,
        staging,
        enable_native=True,
        population_config=fixture.config,
        c2_extension_path=c2_path,
        native_build_directory=build_dir,
    )
    capture_c4_resident_composed_r11_window_v1(owner, fixture.config)
    prepare_c4_resident_composed_r11_window_control_v1(
        owner, generation=fixture.generation_start, tick=fixture.tick_start
    )
    torch.cuda.synchronize()
    return owner


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.skipif(
    os.environ.get("EIEL_C4_R11_LAUNCH_TRACE") != "1",
    reason="R11 launch-trace harness is explicitly opt-in",
)
def test_r11_b48_c32_qualified_window_launch_trace(tmp_path: Path) -> None:
    owner = _make_prepared_b48_c32_owner_v1(tmp_path)
    result = _measured_graph_replay_only_v1(owner)
    torch.cuda.synchronize()
    assert result.completed_ticks.cpu().tolist() == [5] * 48
    assert result.global_status.cpu().tolist() == [0] * 48


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.skipif(
    os.environ.get("EIEL_C4_R11_TORCH_PROFILER_TRACE") != "1",
    reason="R11 torch.profiler launch trace is explicitly opt-in",
)
def test_r11_b48_c32_torch_profiler_launch_trace(tmp_path: Path) -> None:
    owner = _make_prepared_b48_c32_owner_v1(tmp_path)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as profiler:
        with record_function(TORCH_PROFILER_RANGE):
            result = _measured_graph_replay_only_v1(owner)
        torch.cuda.synchronize()
    names = _events_below_profiler_scope_v1(profiler)
    trace_path = Path(
        os.environ.get(
            "EIEL_C4_R11_TORCH_TRACE_PATH",
            tmp_path / "c4_r11_replay_only_torch_trace.json",
        )
    )
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(str(trace_path))
    raw_names_path = trace_path.with_suffix(".raw_event_names.json")
    raw_names_path.write_text(
        json.dumps(dict(sorted(Counter(names).items())), indent=2) + "\n"
    )
    receipt = validate_r11_torch_profiler_names_v1(names)
    receipt_path = trace_path.with_suffix(".receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    assert result.completed_ticks.cpu().tolist() == [5] * 48
    assert result.global_status.cpu().tolist() == [0] * 48


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate_r11_launch_trace_sqlite_v1(args.sqlite), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
