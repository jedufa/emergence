#!/usr/bin/env python3
"""Reduce one bounded Nsight Systems B48 tick trace to a JSON anatomy."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "eiel_gpu_tick_anatomy_analysis_v1"
CORE = "flattened_controller_tick_core_and_birth_plan_kernel_v1"
PREPARE = "flattened_controller_birth_prepare_kernel_v1"
COMMIT = "flattened_controller_birth_transaction_and_settlement_kernel_v1"
STAGES = (CORE, PREPARE, COMMIT)
MEMCPY_KINDS = {
    1: "host_to_device",
    2: "device_to_host",
    8: "device_to_device",
}


class TickAnalysisError(RuntimeError):
    pass


def _stats(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise TickAnalysisError("cannot summarize an empty sample")
    ordered = sorted(float(value) for value in values)
    position = 0.95 * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    p95 = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": p95,
        "min": ordered[0],
        "max": ordered[-1],
    }


def _tables(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }


def _strings(connection: sqlite3.Connection) -> dict[int, str]:
    if "StringIds" not in _tables(connection):
        raise TickAnalysisError("Nsight export lacks StringIds")
    return {
        int(row[0]): str(row[1])
        for row in connection.execute("SELECT id, value FROM StringIds")
    }


def _resolved_name(row: Mapping[str, Any], strings: Mapping[int, str]) -> str:
    for column in ("shortName", "demangledName", "mangledName", "nameId"):
        value = row.get(column)
        if isinstance(value, int) and value in strings:
            return strings[value]
        if isinstance(value, str) and value:
            return value
    return "<unresolved>"


def _rows(connection: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    connection.row_factory = sqlite3.Row
    return [dict(row) for row in connection.execute(f'SELECT * FROM "{table}"')]


def _kernel_rows(
    connection: sqlite3.Connection, strings: Mapping[int, str]
) -> list[dict[str, Any]]:
    table = "CUPTI_ACTIVITY_KIND_KERNEL"
    if table not in _tables(connection):
        raise TickAnalysisError("Nsight export lacks CUDA kernel activity")
    result: list[dict[str, Any]] = []
    for row in _rows(connection, table):
        row["name"] = _resolved_name(row, strings)
        row["duration_ns"] = int(row["end"]) - int(row["start"])
        result.append(row)
    return sorted(result, key=lambda row: (int(row["start"]), int(row["end"])))


def _semantic_name(name: str) -> str | None:
    for stage in STAGES:
        if stage in name:
            return stage
    return None


def _tick_groups(kernels: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    relevant = [row for row in kernels if _semantic_name(str(row["name"]))]
    groups: list[list[Mapping[str, Any]]] = []
    current: list[Mapping[str, Any]] = []
    for row in relevant:
        if _semantic_name(str(row["name"])) == CORE:
            if current:
                groups.append(current)
            current = [row]
        elif current:
            current.append(row)
    if current:
        groups.append(current)
    result: list[dict[str, Any]] = []
    for index, group in enumerate(groups):
        by_stage: dict[str, Mapping[str, Any]] = {}
        for row in group:
            stage = _semantic_name(str(row["name"]))
            if stage is None or stage in by_stage:
                raise TickAnalysisError(f"tick {index} has duplicate or unknown stage")
            by_stage[stage] = row
        if set(by_stage) != set(STAGES):
            raise TickAnalysisError(f"tick {index} does not have the three-stage ABI")
        result.append({"index": index, "stages": by_stage})
    return result


def _runtime_summary(
    connection: sqlite3.Connection,
    strings: Mapping[int, str],
    *,
    start_ns: int,
    end_ns: int,
) -> dict[str, Any]:
    table = "CUPTI_ACTIVITY_KIND_RUNTIME"
    if table not in _tables(connection):
        return {"available": False}
    rows = _rows(connection, table)
    named = [
        {
            **row,
            "name": _resolved_name(row, strings),
            "duration_ns": int(row["end"]) - int(row["start"]),
        }
        for row in rows
        if int(row["start"]) >= start_ns and int(row["end"]) <= end_ns
    ]
    sync = [row for row in named if "Synchronize" in row["name"]]
    allocation = [
        row
        for row in named
        if any(token in row["name"] for token in ("Malloc", "Free", "Alloc"))
    ]
    call_summary = {
        name: {
            "count": len(group),
            "total_ms": sum(row["duration_ns"] for row in group) / 1e6,
        }
        for name in sorted({str(row["name"]) for row in named})
        if (
            group := [row for row in named if str(row["name"]) == name]
        )
    }
    return {
        "available": True,
        "api_call_count": len(named),
        "synchronization_call_count": len(sync),
        "synchronization_total_ms": sum(row["duration_ns"] for row in sync) / 1e6,
        "synchronization_calls": dict(Counter(row["name"] for row in sync)),
        "allocation_call_count": len(allocation),
        "allocation_calls": dict(Counter(row["name"] for row in allocation)),
        "call_summary": call_summary,
    }


def _memop_summary(
    connection: sqlite3.Connection, *, start_ns: int, end_ns: int
) -> dict[str, Any]:
    tables = _tables(connection)
    candidates = (
        "CUPTI_ACTIVITY_KIND_MEMCPY",
        "CUPTI_ACTIVITY_KIND_MEMSET",
        "CUPTI_ACTIVITY_KIND_MEMORY",
    )
    rows: list[dict[str, Any]] = []
    used: list[str] = []
    for table in candidates:
        if table not in tables:
            continue
        used.append(table)
        for row in _rows(connection, table):
            if int(row["start"]) < start_ns or int(row["end"]) > end_ns:
                continue
            row["table"] = table
            row["duration_ns"] = int(row["end"]) - int(row["start"])
            rows.append(row)
    by_copy_kind: dict[str, dict[str, int | float]] = {}
    for kind in sorted(
        {int(row["copyKind"]) for row in rows if row.get("copyKind") is not None}
    ):
        selected = [row for row in rows if int(row.get("copyKind", -1)) == kind]
        by_copy_kind[MEMCPY_KINDS.get(kind, f"copy_kind_{kind}")] = {
            "copy_kind": kind,
            "operation_count": len(selected),
            "total_ms": sum(int(row["duration_ns"]) for row in selected) / 1e6,
            "total_bytes": sum(int(row.get("bytes", 0)) for row in selected),
        }
    return {
        "tables": used,
        "operation_count": len(rows),
        "total_ms": sum(int(row["duration_ns"]) for row in rows) / 1e6,
        "total_bytes_when_reported": sum(
            int(row.get("bytes", 0)) for row in rows if row.get("bytes") is not None
        ),
        "by_copy_kind": by_copy_kind,
    }


def _utilization(
    path: Path | None, *, started_ns: int, completed_ns: int
) -> dict[str, Any]:
    if path is None:
        return {"available": False}
    samples: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) < 8:
                continue
            try:
                timestamp_ns = int(datetime.strptime(
                    row[0].strip(), "%Y/%m/%d %H:%M:%S.%f"
                ).timestamp() * 1e9)
                values = [float(value.strip()) for value in row[2:8]]
            except ValueError:
                continue
            if started_ns <= timestamp_ns <= completed_ns:
                samples.append(
                    {
                        "gpu_percent": values[0],
                        "memory_percent": values[1],
                        "memory_mib": values[2],
                        "power_w": values[3],
                        "sm_clock_mhz": values[4],
                        "memory_clock_mhz": values[5],
                    }
                )
    if not samples:
        return {"available": False, "reason": "no samples in native window"}
    return {
        "available": True,
        "sample_count": len(samples),
        **{
            name: _stats([row[name] for row in samples])
            for name in samples[0]
        },
    }


def analyze(
    sqlite_path: Path,
    receipt_path: Path,
    *,
    warmup_ticks: int,
    baseline_receipt_path: Path | None,
    utilization_path: Path | None,
    utilization_receipt_path: Path | None,
) -> dict[str, Any]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    with sqlite3.connect(sqlite_path) as connection:
        strings = _strings(connection)
        kernels = _kernel_rows(connection, strings)
        ticks = _tick_groups(kernels)
        if len(ticks) != int(receipt["ticks"]):
            raise TickAnalysisError("Nsight tick count differs from execution receipt")
        if not 0 <= warmup_ticks < len(ticks) - 1:
            raise TickAnalysisError("warm-up exclusion leaves no complete intervals")
        window_start_ns = int(ticks[0]["stages"][CORE]["start"])
        window_end_ns = int(ticks[-1]["stages"][COMMIT]["end"])
        window_kernels = [
            row
            for row in kernels
            if int(row["start"]) >= window_start_ns
            and int(row["end"]) <= window_end_ns
        ]
        steady = ticks[warmup_ticks:-1]
        interval_ms = [
            (
                int(ticks[index + 1]["stages"][CORE]["start"])
                - int(ticks[index]["stages"][CORE]["start"])
            )
            / 1e6
            for index in range(warmup_ticks, len(ticks) - 1)
        ]
        stage_samples = {
            stage: [float(tick["stages"][stage]["duration_ns"]) / 1e6 for tick in steady]
            for stage in STAGES
        }
        residual_ms = [
            interval_ms[index]
            - sum(stage_samples[stage][index] for stage in STAGES)
            for index in range(len(interval_ms))
        ]
        total_mean = statistics.fmean(interval_ms)
        stage_rows = []
        for stage in (*STAGES, "residual"):
            samples = residual_ms if stage == "residual" else stage_samples[stage]
            summary = _stats(samples)
            row = {
                "stage": stage,
                "stats_ms_per_tick": summary,
                "percent_of_steady_tick": 100.0 * summary["mean"] / total_mean,
            }
            if stage != "residual":
                representative = steady[0]["stages"][stage]
                row["launch"] = {
                    name: representative.get(name)
                    for name in (
                        "gridX",
                        "gridY",
                        "gridZ",
                        "blockX",
                        "blockY",
                        "blockZ",
                        "registersPerThread",
                        "staticSharedMemory",
                        "dynamicSharedMemory",
                        "localMemoryPerThread",
                        "localMemoryTotal",
                    )
                }
            stage_rows.append(row)
        streams = sorted(
            {int(row["streamId"]) for row in window_kernels if "streamId" in row}
        )
        overlaps = sum(
            int(right["start"]) < int(left["end"])
            for left, right in zip(window_kernels, window_kernels[1:])
        )
        runtime = _runtime_summary(
            connection, strings, start_ns=window_start_ns, end_ns=window_end_ns
        )
        memops = _memop_summary(
            connection, start_ns=window_start_ns, end_ns=window_end_ns
        )

    baseline: dict[str, Any] | None = None
    if baseline_receipt_path is not None:
        baseline_receipt = json.loads(
            baseline_receipt_path.read_text(encoding="utf-8")
        )
        baseline_ms = float(baseline_receipt["wall_ms_per_tick"])
        profiled_ms = float(receipt["wall_ms_per_tick"])
        baseline = {
            "unprofiled_wall_ms_per_tick": baseline_ms,
            "profiled_wall_ms_per_tick": profiled_ms,
            "overhead_ms_per_tick": profiled_ms - baseline_ms,
            "overhead_percent": 100.0 * (profiled_ms - baseline_ms) / baseline_ms,
        }
    utilization_receipt = receipt
    if utilization_receipt_path is not None:
        utilization_receipt = json.loads(
            utilization_receipt_path.read_text(encoding="utf-8")
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "sqlite_path": str(sqlite_path.resolve()),
        "execution_receipt": str(receipt_path.resolve()),
        "ticks_captured": len(ticks),
        "warmup_ticks_excluded": warmup_ticks,
        "steady_tick_intervals": len(interval_ms),
        "total_tick_stats_ms": _stats(interval_ms),
        "stage_rows": stage_rows,
        "kernel_count_total": len(window_kernels),
        "kernel_count_per_tick": len(window_kernels) / len(ticks),
        "semantic_kernel_counts": {
            stage: sum(_semantic_name(str(row["name"])) == stage for row in kernels)
            for stage in STAGES
        },
        "streams": streams,
        "overlapping_adjacent_kernel_pairs": overlaps,
        "runtime_api": runtime,
        "memory_operations": memops,
        "utilization": _utilization(
            utilization_path,
            started_ns=int(utilization_receipt["native_window_started_at_unix_ns"]),
            completed_ns=int(utilization_receipt["native_window_completed_at_unix_ns"]),
        ),
        "profiling_overhead": baseline,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--baseline-receipt", type=Path)
    parser.add_argument("--utilization-csv", type=Path)
    parser.add_argument("--utilization-receipt", type=Path)
    parser.add_argument("--warmup-ticks", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = analyze(
        args.sqlite,
        args.receipt,
        warmup_ticks=args.warmup_ticks,
        baseline_receipt_path=args.baseline_receipt,
        utilization_path=args.utilization_csv,
        utilization_receipt_path=args.utilization_receipt,
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
