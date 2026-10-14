#!/usr/bin/env python3
from __future__ import annotations

"""Summarize bounded Stage 3 CUPTI JSONL traces without retaining raw rows."""

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any, Final


SCHEMA_VERSION: Final[str] = "eiel_cuda_stage3_cupti_trace_summary_v1"
ACTIVITY_TYPES: Final[frozenset[str]] = frozenset({"kernel", "memcpy", "memset"})
CLUSTER_GAP_NS: Final[int] = 50_000_000


def _case(value: str) -> tuple[str, Path, Path]:
    parts = value.split("=", 1)
    if len(parts) != 2 or not parts[0] or "," not in parts[1]:
        raise argparse.ArgumentTypeError("case must be label=trace.ndjson,benchmark.json")
    trace, benchmark = parts[1].split(",", 1)
    return parts[0], Path(trace), Path(benchmark)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", type=_case, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _overlaps(record: dict[str, Any], start: int, end: int) -> bool:
    record_start = int(record.get("start_ns", record.get("timestamp_ns", 0)) or 0)
    record_end = int(record.get("end_ns", record_start) or record_start)
    return record_start <= end and record_end >= start


def _union_duration(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    total = 0
    current_start, current_end = sorted(intervals)[0]
    for start, end in sorted(intervals)[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def _activity_pass(trace: Path) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, Any]]:
    activities: list[dict[str, Any]] = []
    counts: collections.Counter[str] = collections.Counter()
    final_metadata: dict[str, Any] = {}
    with trace.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{trace}:{line_number}: invalid JSON: {exc}") from exc
            record_type = str(row.get("record_type") or "missing")
            counts[record_type] += 1
            if record_type in ACTIVITY_TYPES:
                activities.append(row)
            elif record_type == "metadata" and row.get("event") == "final":
                final_metadata = row
    if not activities or not final_metadata:
        raise ValueError(f"{trace} is missing activity records or final metadata")
    return activities, dict(counts), final_metadata


def _dense_generation_cluster(activities: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(activities, key=lambda row: int(row["start_ns"]))
    clusters: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_end = 0
    for row in ordered:
        start = int(row["start_ns"])
        end = int(row["end_ns"])
        if current and start - current_end > CLUSTER_GAP_NS:
            clusters.append(current)
            current = []
        current.append(row)
        current_end = max(current_end, end)
    if current:
        clusters.append(current)
    selected = max(clusters, key=lambda rows: (len(rows), sum(int(r["end_ns"]) - int(r["start_ns"]) for r in rows)))
    start = min(int(row["start_ns"]) for row in selected)
    end = max(int(row["end_ns"]) for row in selected)
    return {
        "start_ns": start,
        "end_ns": end,
        "span_ns": end - start,
        "activity_record_count": len(selected),
        "cluster_count": len(clusters),
        "selection_rule": "largest kernel/memcpy/memset cluster separated by gaps over 50 ms",
    }


def _summarize_window(trace: Path, window: dict[str, Any]) -> dict[str, Any]:
    start = int(window["start_ns"])
    end = int(window["end_ns"])
    record_counts: collections.Counter[str] = collections.Counter()
    api_counts: collections.Counter[tuple[str, str]] = collections.Counter()
    api_duration: collections.Counter[tuple[str, str]] = collections.Counter()
    kernel_summary: dict[str, dict[str, Any]] = {}
    copy_counts: collections.Counter[str] = collections.Counter()
    copy_bytes: collections.Counter[str] = collections.Counter()
    copy_duration: collections.Counter[str] = collections.Counter()
    memory_counts: collections.Counter[str] = collections.Counter()
    memory_bytes: collections.Counter[str] = collections.Counter()
    intervals: list[tuple[int, int]] = []
    kernel_correlations: set[int] = set()
    memcpy_correlations: set[int] = set()
    api_correlations: set[int] = set()
    launch_runtime_count = 0
    launch_driver_count = 0
    launch_runtime_cpu_ns = 0
    launch_driver_cpu_ns = 0
    queue_delay_ns: list[int] = []
    submit_to_start_ns: list[int] = []
    with trace.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            record_type = str(row.get("record_type") or "missing")
            if record_type in ACTIVITY_TYPES and _overlaps(row, start, end):
                record_counts[record_type] += 1
                row_start = int(row["start_ns"])
                row_end = int(row["end_ns"])
                intervals.append((max(start, row_start), min(end, row_end)))
                if record_type == "kernel":
                    name = str(row.get("name") or "unknown")
                    summary = kernel_summary.setdefault(
                        name,
                        {
                            "name": name,
                            "count": 0,
                            "duration_ns": 0,
                            "grid_block_resource_variants": set(),
                        },
                    )
                    summary["count"] += 1
                    summary["duration_ns"] += row_end - row_start
                    summary["grid_block_resource_variants"].add(
                        (
                            int(row.get("grid_x", 0)),
                            int(row.get("grid_y", 0)),
                            int(row.get("grid_z", 0)),
                            int(row.get("block_x", 0)),
                            int(row.get("block_y", 0)),
                            int(row.get("block_z", 0)),
                            int(row.get("registers_per_thread", 0)),
                            int(row.get("static_shared_memory_bytes", 0)),
                            int(row.get("dynamic_shared_memory_bytes", 0)),
                        )
                    )
                    kernel_correlations.add(int(row.get("correlation_id", 0)))
                    queued = int(row.get("queued_ns", 0) or 0)
                    submitted = int(row.get("submitted_ns", 0) or 0)
                    if queued and submitted and submitted >= queued:
                        queue_delay_ns.append(submitted - queued)
                    if submitted and row_start >= submitted:
                        submit_to_start_ns.append(row_start - submitted)
                elif record_type == "memcpy":
                    direction = str(row.get("direction") or "unknown")
                    copy_counts[direction] += 1
                    copy_bytes[direction] += int(row.get("bytes", 0))
                    copy_duration[direction] += row_end - row_start
                    memcpy_correlations.update(
                        {
                            int(row.get("correlation_id", 0)),
                            int(row.get("runtime_correlation_id", 0)),
                        }
                    )
                continue
            if record_type == "api" and _overlaps(row, start, end):
                record_counts["api"] += 1
                domain = str(row.get("api_domain") or "unknown")
                name = str(row.get("api_name") or "unknown")
                duration = int(row.get("end_ns", 0)) - int(row.get("start_ns", 0))
                api_counts[(domain, name)] += 1
                api_duration[(domain, name)] += max(0, duration)
                api_correlations.add(int(row.get("correlation_id", 0)))
                if domain == "runtime" and "cudaLaunchKernel" in name:
                    launch_runtime_count += 1
                    launch_runtime_cpu_ns += max(0, duration)
                if domain == "driver" and "cuLaunchKernel" in name:
                    launch_driver_count += 1
                    launch_driver_cpu_ns += max(0, duration)
                continue
            if record_type == "memory":
                timestamp = int(row.get("timestamp_ns", 0) or 0)
                if start <= timestamp <= end:
                    record_counts["memory"] += 1
                    operation = str(row.get("operation") or "unknown")
                    memory_counts[operation] += 1
                    memory_bytes[operation] += int(row.get("bytes", 0))
            elif record_type in {"synchronization", "cuda_event"} and _overlaps(
                row, start, end
            ):
                record_counts[record_type] += 1

    top_kernels = sorted(
        kernel_summary.values(), key=lambda row: int(row["duration_ns"]), reverse=True
    )[:30]
    for row in top_kernels:
        row["grid_block_resource_variants"] = [
            {
                "grid": list(values[:3]),
                "block": list(values[3:6]),
                "registers_per_thread": values[6],
                "static_shared_memory_bytes": values[7],
                "dynamic_shared_memory_bytes": values[8],
            }
            for values in sorted(row["grid_block_resource_variants"])
        ]
    active_ns = _union_duration(intervals)
    span_ns = end - start
    api_rows = sorted(
        (
            {
                "domain": key[0],
                "name": key[1],
                "count": count,
                "cpu_duration_ns": api_duration[key],
            }
            for key, count in api_counts.items()
        ),
        key=lambda row: int(row["cpu_duration_ns"]),
        reverse=True,
    )[:50]
    return {
        "window": window,
        "record_counts": dict(record_counts),
        "device_activity": {
            "union_active_ns": active_ns,
            "span_ns": span_ns,
            "idle_or_submission_gap_ns": max(0, span_ns - active_ns),
            "union_active_fraction": active_ns / span_ns if span_ns else None,
        },
        "launches": {
            "runtime_cudaLaunchKernel_count": launch_runtime_count,
            "runtime_cudaLaunchKernel_cpu_ns": launch_runtime_cpu_ns,
            "driver_cuLaunchKernel_count": launch_driver_count,
            "driver_cuLaunchKernel_cpu_ns": launch_driver_cpu_ns,
            "kernel_activity_count": record_counts["kernel"],
            "kernel_correlation_join_fraction": (
                len(kernel_correlations & api_correlations) / len(kernel_correlations)
                if kernel_correlations
                else None
            ),
        },
        "copies": {
            direction: {
                "count": copy_counts[direction],
                "bytes": copy_bytes[direction],
                "duration_ns": copy_duration[direction],
            }
            for direction in sorted(copy_counts)
        },
        "copy_correlation_join_fraction": (
            len(memcpy_correlations & api_correlations) / len(memcpy_correlations)
            if memcpy_correlations
            else None
        ),
        "memory": {
            operation: {
                "count": memory_counts[operation],
                "bytes": memory_bytes[operation],
            }
            for operation in sorted(memory_counts)
        },
        "kernel_queue_delay_ns": _distribution(queue_delay_ns),
        "kernel_submit_to_start_ns": _distribution(submit_to_start_ns),
        "top_kernels": top_kernels,
        "top_api_calls": api_rows,
        "occupancy_and_bandwidth": {
            "available": False,
            "reason": "CUPTI Activity records expose launch resources and transfer bytes, not hardware occupancy or achieved memory bandwidth counters.",
        },
    }


def _distribution(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "minimum": None, "median": None, "maximum": None}
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2
    )
    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "median": median,
        "maximum": ordered[-1],
    }


def _benchmark_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary") or {}
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "status": [row.get("status") for row in payload.get("repetitions") or []],
        "parity": summary.get("parity"),
        "timing": (summary.get("timing") or {}),
        "diagnostic_trace_timing_is_not_performance_evidence": True,
    }


def summarize_case(label: str, trace: Path, benchmark: Path) -> dict[str, Any]:
    activities, counts, metadata = _activity_pass(trace)
    if any(
        int(metadata.get(name, 0)) != 0
        for name in (
            "dropped_records_total",
            "buffer_requests_denied",
            "invalid_buffer_returns",
            "unknown_records",
            "parse_errors",
            "output_errors_before_final",
        )
    ):
        raise ValueError(f"{trace} failed collector integrity")
    window = _dense_generation_cluster(activities)
    return {
        "label": label,
        "trace": {
            "path": str(trace),
            "bytes": trace.stat().st_size,
            "sha256": _sha256(trace),
            "record_counts": counts,
            "final_metadata": metadata,
        },
        "benchmark": _benchmark_summary(benchmark),
        "generation_activity": _summarize_window(trace, window),
    }


def main() -> int:
    namespace = build_parser().parse_args()
    labels = [label for label, _, _ in namespace.case]
    if len(labels) != len(set(labels)):
        raise ValueError("case labels must be unique")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "cluster_gap_ns": CLUSTER_GAP_NS,
        "cases": {
            label: summarize_case(label, trace, benchmark)
            for label, trace, benchmark in namespace.case
        },
        "claims": {
            "trace_is_diagnostic_not_performance_evidence": True,
            "occupancy_measured": False,
            "memory_bandwidth_measured": False,
            "promotion_qualified": False,
        },
    }
    namespace.output.parent.mkdir(parents=True, exist_ok=True)
    namespace.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(namespace.output), "cases": labels}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
