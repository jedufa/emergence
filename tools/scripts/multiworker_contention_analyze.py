#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import statistics
from pathlib import Path
from typing import Any


BUCKET_FIELDS = [
    "workers",
    "children",
    "mean_sec_per_gen",
    "tick_loop_total_s",
    "policy_s",
    "env_action_s",
    "drive_s",
    "reproduction_s",
    "checkpoint_s",
    "metrics_s",
    "boundary_other_s",
    "policy_pct",
    "env_action_pct",
    "drive_pct",
    "reproduction_pct",
    "checkpoint_pct",
    "metrics_pct",
    "boundary_other_pct",
]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _parse_time(value: str) -> dt.datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return dt.datetime.fromisoformat(text)
    except Exception:
        return None


def _num(value: Any) -> float:
    try:
        if value is None or value == "":
            return 0.0
        return float(str(value).strip().replace("%", ""))
    except Exception:
        return 0.0


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _family(label: str) -> str:
    text = str(label or "").upper()
    if text.startswith("B"):
        return "B"
    return "A"


def _is_checkpoint_heavy(label: str) -> bool:
    return _family(label) == "B"


def _bucket_category(name: str) -> str | None:
    low = str(name or "").lower()
    if low == "tick_loop_total":
        return "tick_loop_total"
    # Use non-overlapping top-level rows only. perf_buckets.csv also includes
    # nested detail buckets, and summing those makes category percentages exceed
    # the tick-loop denominator.
    if low in ("policy_prep_ms", "policy_fwd_ms"):
        return "policy"
    if low == "action_apply_ms":
        return "env_action"
    if low == "drive_update_ms":
        return "drive"
    if low == "reproduction_ms":
        return "reproduction"
    if low in ("checkpoint_total", "checkpoint_write_total"):
        return "checkpoint"
    if low == "metrics_total":
        return "metrics"
    return None


def _latest_bucket_totals(run_dir: Path) -> tuple[float, dict[str, float]]:
    rows = _read_csv(run_dir / "perf_buckets.csv")
    totals: dict[str, float] = {
        "policy": 0.0,
        "env_action": 0.0,
        "drive": 0.0,
        "reproduction": 0.0,
        "checkpoint": 0.0,
        "metrics": 0.0,
        "boundary_other": 0.0,
    }
    tick_total = 0.0
    for row in rows:
        category = _bucket_category(row.get("bucket", ""))
        seconds = _num(row.get("seconds"))
        if category == "tick_loop_total":
            tick_total += seconds
        elif category in totals:
            totals[category] += seconds
    accounted = sum(totals.values())
    if tick_total > accounted:
        totals["boundary_other"] += tick_total - accounted
    return tick_total, totals


def _wave_rows(root: Path, workers: int) -> tuple[list[dict[str, Any]], dict[str, float]]:
    starts: dict[int, dict[str, Any]] = {}
    ends: dict[int, dict[str, Any]] = {}
    for line in (root / "parallel_command_log.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if row.get("kind") == "child_start":
            starts[int(row.get("index", 0))] = row
        elif row.get("kind") == "child_end":
            ends[int(row.get("index", 0))] = row
    start_times = [_parse_time(row.get("timestamp_utc", "")) for row in starts.values()]
    start_times = [item for item in start_times if item is not None]
    base = min(start_times) if start_times else None
    out: list[dict[str, Any]] = []
    used = 0.0
    latest_end = 0.0
    for rank, index in enumerate(sorted(starts)):
        s = starts.get(index, {})
        e = ends.get(index, {})
        st = _parse_time(s.get("timestamp_utc", ""))
        en = _parse_time(e.get("timestamp_utc", ""))
        start_offset = (st - base).total_seconds() if base and st else 0.0
        end_offset = (en - base).total_seconds() if base and en else start_offset + _num(e.get("wall_s"))
        duration = _num(e.get("wall_s")) or max(0.0, end_offset - start_offset)
        used += duration
        latest_end = max(latest_end, end_offset)
        label = str(s.get("label") or e.get("label") or "")
        out.append(
            {
                "index": index,
                "label": label,
                "worker_slot": s.get("worker_slot", e.get("worker_slot", "")),
                "wave_rank": rank // max(1, workers),
                "start_offset_s": f"{start_offset:.3f}",
                "end_offset_s": f"{end_offset:.3f}",
                "duration_s": f"{duration:.3f}",
                "scenario_family": _family(label),
                "checkpoint_heavy": "yes" if _is_checkpoint_heavy(label) else "no",
                "returncode": e.get("returncode", ""),
            }
        )
    worker_seconds = max(latest_end * max(1, workers), used)
    idle = max(0.0, worker_seconds - used)
    return out, {
        "worker_seconds_available": worker_seconds,
        "used_child_seconds": used,
        "idle_tail_seconds": idle,
        "idle_tail_share_of_worker_capacity": idle / worker_seconds if worker_seconds else 0.0,
        "campaign_span_s_from_child_timestamps": latest_end,
    }


def _monitor_summary(root: Path, label: str) -> dict[str, Any]:
    proc_rows = _read_csv(root / f"process_monitor_{label}.csv")
    gpu_rows = _read_csv(root / f"gpu_monitor_{label}.csv")
    system_cpu = [_num(row.get("cpu_total_percent")) for row in proc_rows if row.get("kind") == "system" and row.get("cpu_total_percent")]
    process_cpu = [_num(row.get("cpu_percent")) for row in proc_rows if row.get("kind") == "process" and row.get("role") != "monitor"]
    runnable = [_num(row.get("runnable_processes")) for row in proc_rows if row.get("kind") == "system"]
    rss = [_num(row.get("rss_kib")) for row in proc_rows if row.get("kind") == "process"]
    gpu_util = [_num(row.get("utilization_gpu_percent")) for row in gpu_rows if row.get("utilization_gpu_percent")]
    gpu_mem = [_num(row.get("memory_used_mib")) for row in gpu_rows if row.get("memory_used_mib")]
    gpu_power = [_num(row.get("power_draw_w")) for row in gpu_rows if row.get("power_draw_w")]
    app_counts = [_num(row.get("compute_app_count")) for row in gpu_rows if row.get("compute_app_count") != ""]
    return {
        "process_samples": len(proc_rows),
        "gpu_samples": len(gpu_rows),
        "max_aggregate_cpu_percent": max(system_cpu) if system_cpu else 0.0,
        "mean_aggregate_cpu_percent": _mean(system_cpu),
        "max_process_cpu_percent": max(process_cpu) if process_cpu else 0.0,
        "max_runnable_processes": max(runnable) if runnable else 0.0,
        "max_process_rss_mib": (max(rss) / 1024.0) if rss else 0.0,
        "max_gpu_util_percent": max(gpu_util) if gpu_util else 0.0,
        "mean_gpu_util_percent": _mean(gpu_util),
        "max_gpu_mem_mib": max(gpu_mem) if gpu_mem else 0.0,
        "max_gpu_power_w": max(gpu_power) if gpu_power else 0.0,
        "max_compute_app_count": max(app_counts) if app_counts else 0.0,
    }


def _variant(root: Path, label: str, workers: int) -> dict[str, Any]:
    run_root = root / label
    summary = _load_json(run_root / "parallel_campaign_summary.json")
    results = list(summary.get("results") or [])
    walls = [_num(row.get("wall_s")) for row in results]
    tick_total = 0.0
    bucket_totals = {
        "policy": 0.0,
        "env_action": 0.0,
        "drive": 0.0,
        "reproduction": 0.0,
        "checkpoint": 0.0,
        "metrics": 0.0,
        "boundary_other": 0.0,
    }
    metric_rows = 0
    checkpoint_files = 0
    checkpoint_bytes = 0
    for row in results:
        run_dir = Path(str(row.get("run_dir") or ""))
        if run_dir.exists():
            metric_rows += int(row.get("metrics_row_count") or 0)
            tick, buckets = _latest_bucket_totals(run_dir)
            tick_total += tick
            for key, value in buckets.items():
                bucket_totals[key] += value
            for ckpt in run_dir.rglob("*"):
                if ckpt.is_file() and ("checkpoint" in ckpt.name.lower() or "checkpoint" in str(ckpt.parent).lower()):
                    checkpoint_files += 1
                    try:
                        checkpoint_bytes += ckpt.stat().st_size
                    except Exception:
                        pass
    denom = tick_total or sum(bucket_totals.values()) or 1.0
    wave_rows, wave_summary = _wave_rows(run_root, workers)
    monitor = _monitor_summary(root, label)
    return {
        "label": label,
        "workers": workers,
        "summary_path": str(run_root / "parallel_campaign_summary.json"),
        "runs_root": str(run_root),
        "cells": int(summary.get("planned_runs") or len(results)),
        "completed": len([row for row in results if row.get("status") == "ok" and int(row.get("returncode", 1)) == 0]),
        "failures": len(summary.get("failures") or []),
        "total_wall_s": _num(summary.get("total_wall_s")),
        "mean_child_wall_s": _num(summary.get("mean_child_wall_s")) or _mean(walls),
        "max_child_wall_s": max(walls) if walls else 0.0,
        "metric_rows": metric_rows,
        "checkpoint_files": checkpoint_files,
        "checkpoint_bytes": checkpoint_bytes,
        "bucket_totals": bucket_totals,
        "tick_loop_total_s": tick_total,
        "bucket_shares": {key: value / denom for key, value in bucket_totals.items()},
        "mean_sec_per_gen": (tick_total / metric_rows) if metric_rows else 0.0,
        "wave_rows": wave_rows,
        "wave_summary": wave_summary,
        "monitor": monitor,
    }


def _write_wave_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "index",
        "label",
        "worker_slot",
        "wave_rank",
        "start_offset_s",
        "end_offset_s",
        "duration_s",
        "scenario_family",
        "checkpoint_heavy",
        "returncode",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_bucket_csv(path: Path, variants: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BUCKET_FIELDS)
        writer.writeheader()
        for item in variants:
            totals = item["bucket_totals"]
            shares = item["bucket_shares"]
            row = {
                "workers": item["workers"],
                "children": item["cells"],
                "mean_sec_per_gen": f"{item['mean_sec_per_gen']:.6f}",
                "tick_loop_total_s": f"{item['tick_loop_total_s']:.6f}",
                "policy_s": f"{totals['policy']:.6f}",
                "env_action_s": f"{totals['env_action']:.6f}",
                "drive_s": f"{totals['drive']:.6f}",
                "reproduction_s": f"{totals['reproduction']:.6f}",
                "checkpoint_s": f"{totals['checkpoint']:.6f}",
                "metrics_s": f"{totals['metrics']:.6f}",
                "boundary_other_s": f"{totals['boundary_other']:.6f}",
                "policy_pct": f"{100.0 * shares['policy']:.3f}",
                "env_action_pct": f"{100.0 * shares['env_action']:.3f}",
                "drive_pct": f"{100.0 * shares['drive']:.3f}",
                "reproduction_pct": f"{100.0 * shares['reproduction']:.3f}",
                "checkpoint_pct": f"{100.0 * shares['checkpoint']:.3f}",
                "metrics_pct": f"{100.0 * shares['metrics']:.3f}",
                "boundary_other_pct": f"{100.0 * shares['boundary_other']:.3f}",
            }
            writer.writerow(row)


def _diagnosis(variants: list[dict[str, Any]]) -> tuple[list[str], str]:
    if len(variants) < 3 or any(item["failures"] for item in variants):
        return ["CONT-INCONCLUSIVE"], "One or more mandatory profiles failed or were missing."
    w4, w6, w8 = variants
    tail = max(item["wave_summary"].get("idle_tail_share_of_worker_capacity", 0.0) for item in variants)
    gpu8 = w8["monitor"].get("mean_gpu_util_percent", 0.0)
    cpu8 = w8["monitor"].get("max_aggregate_cpu_percent", 0.0)
    latency_growth = (w8["mean_child_wall_s"] / w4["mean_child_wall_s"]) if w4["mean_child_wall_s"] else 1.0
    labels: list[str] = []
    if latency_growth > 1.20 and gpu8 < 80.0:
        labels.append("CONT-CUDA-MPROC")
    if gpu8 < 70.0 and cpu8 < 90.0:
        labels.append("CONT-GPU-LAUNCH")
    if cpu8 >= 90.0 or w8["monitor"].get("max_runnable_processes", 0.0) > w8["workers"]:
        labels.append("CONT-CPU")
    if tail > 0.20:
        labels.append("CONT-WAVE")
    if not labels:
        labels.append("CONT-MIXED")
    if len(labels) > 2:
        labels = ["CONT-MIXED", *labels[:2]]
    explanation = (
        f"W8 child latency is {latency_growth:.2f}x W4, mean sampled GPU util is {gpu8:.1f}%, "
        f"max aggregate CPU is {cpu8:.1f}%, and max idle/tail worker-capacity share is {tail:.1%}."
    )
    return labels, explanation


def _markdown(summary: dict[str, Any]) -> str:
    variants = summary["variants"]
    labels = ", ".join(summary["diagnostic_labels"])
    lines = [
        "# EIEL Multi-Worker Contention Profile Summary",
        "",
        f"- generated_at_utc: `{summary['generated_at_utc']}`",
        f"- runtime_root: `{summary['runtime_root']}`",
        f"- diagnostic_label: `{labels}`",
        f"- diagnosis: {summary['diagnosis']}",
        "",
        "## Campaign Results",
        "",
        "| workers | cells | failures | total wall sec | mean child wall sec | max child wall sec | max VRAM MiB | max GPU util % | max aggregate CPU % |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in variants:
        mon = item["monitor"]
        lines.append(
            f"| {item['workers']} | {item['cells']} | {item['failures']} | {item['total_wall_s']:.3f} | "
            f"{item['mean_child_wall_s']:.3f} | {item['max_child_wall_s']:.3f} | "
            f"{mon['max_gpu_mem_mib']:.0f} | {mon['max_gpu_util_percent']:.1f} | {mon['max_aggregate_cpu_percent']:.1f} |"
        )
    lines.extend(["", "## Perf Bucket Shares", ""])
    lines.append("| workers | policy % | env/action % | drive % | reproduction % | checkpoint % | metrics % | boundary/other % |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for item in variants:
        s = item["bucket_shares"]
        lines.append(
            f"| {item['workers']} | {100*s['policy']:.2f} | {100*s['env_action']:.2f} | {100*s['drive']:.2f} | "
            f"{100*s['reproduction']:.2f} | {100*s['checkpoint']:.2f} | {100*s['metrics']:.2f} | {100*s['boundary_other']:.2f} |"
        )
    lines.extend(["", "## Wave/Tail", ""])
    lines.append("| workers | campaign span sec | used child sec | idle/tail sec | idle/tail capacity share |")
    lines.append("|---:|---:|---:|---:|---:|")
    for item in variants:
        ws = item["wave_summary"]
        lines.append(
            f"| {item['workers']} | {ws['campaign_span_s_from_child_timestamps']:.3f} | {ws['used_child_seconds']:.3f} | "
            f"{ws['idle_tail_seconds']:.3f} | {100*ws['idle_tail_share_of_worker_capacity']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            summary["recommended_engineering_move"],
            "",
            "## Limitations",
            "",
            "- GPU sampling uses `nvidia-smi` under WSL and may miss short CUDA bursts.",
            "- Process sampling is external `/proc` polling and does not replace a profiler.",
            "- Bucket percentages are based on existing `perf_buckets.csv` timing rows; counter-like rows are excluded.",
        ]
    )
    return "\n".join(lines) + "\n"


def _recommend(labels: list[str]) -> str:
    if "CONT-CUDA-MPROC" in labels or "CONT-GPU-LAUNCH" in labels:
        return (
            "Recommended next engineering move: design a batched multi-cell execution branch with fewer Python "
            "processes and fewer CUDA contexts, where one worker handles multiple same-shape seeds in larger "
            "policy/env batches while the current process-wrapper remains the fallback."
        )
    if "CONT-WAVE" in labels:
        return (
            "Recommended next engineering move: add duration-aware scheduling, such as longest-processing-time-first "
            "dispatch from prior scenario-family timings, without changing output identity."
        )
    if "CONT-CPU" in labels:
        return (
            "Recommended next engineering move: reduce Python/control-plane overhead with persistent workers and "
            "less repeated registry subprocess startup."
        )
    return (
        "Recommended next engineering move: make one small scheduler improvement first, then evaluate a larger "
        "batched execution design if GPU duty cycle remains capped."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze EIEL W4/W6/W8 contention-profile outputs")
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    variants = [_variant(root, "w4", 4), _variant(root, "w6", 6), _variant(root, "w8", 8)]
    root.mkdir(parents=True, exist_ok=True)
    for item in variants:
        _write_wave_csv(root / f"wave_schedule_{item['label']}.csv", item["wave_rows"])
    _write_bucket_csv(root / "bucket_compare_w4_w6_w8.csv", variants)
    labels, diagnosis = _diagnosis(variants)
    summary = {
        "schema_version": "eiel_multiworker_contention_profile_v1",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "runtime_root": str(root),
        "variants": [
            {key: value for key, value in item.items() if key != "wave_rows"}
            for item in variants
        ],
        "diagnostic_labels": labels,
        "diagnosis": diagnosis,
        "recommended_engineering_move": _recommend(labels),
        "analysis_outputs": {
            "bucket_compare": str(root / "bucket_compare_w4_w6_w8.csv"),
            "wave_schedule_w4": str(root / "wave_schedule_w4.csv"),
            "wave_schedule_w6": str(root / "wave_schedule_w6.csv"),
            "wave_schedule_w8": str(root / "wave_schedule_w8.csv"),
        },
    }
    (root / "contention_profile_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "contention_profile_summary.md").write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps({"summary": str(root / "contention_profile_summary.json"), "labels": labels}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
