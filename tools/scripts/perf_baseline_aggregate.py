#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


COUNT_SUFFIXES = ("_count", "_requests")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _benchmark_for(run_id: str) -> str:
    if run_id.startswith("phaseIV_crb_x1_prediction_readiness_bridge"):
        return "A_prediction_heavy"
    if run_id.startswith("lbc1_lineage_bottleneck_closure_source"):
        return "B_lineage_funnel"
    if run_id.startswith("phaseII_b1_founder_balance_validation"):
        return "C_founder_fast_lineage"
    return "unknown"


def _config_map(run_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    path = run_dir / "run_config.csv"
    if not path.exists():
        return out
    for row in _read_csv(path):
        if row.get("section") in {"sim", "pop", "env"}:
            out[f"{row.get('section')}.{row.get('key')}"] = row.get("value", "")
    return out


def _load_runs(root: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for summary_path in sorted(root.rglob("summary.json")):
        summary = _read_json(summary_path)
        if summary.get("status") != "complete":
            continue
        for row in summary.get("runs", []) or []:
            if not isinstance(row, dict) or row.get("status") != "ok":
                continue
            run_dir = Path(str(row.get("run_dir", "")))
            perf_path = run_dir / "perf_buckets.csv"
            if not perf_path.exists():
                continue
            run_id = str(row.get("run_id", ""))
            cfg = _config_map(run_dir)
            perf_rows = _read_csv(perf_path)
            gen_total: dict[int, float] = {}
            bucket_seconds: dict[str, float] = defaultdict(float)
            bucket_counts: dict[str, float] = defaultdict(float)
            for perf in perf_rows:
                gen = int(float(perf.get("gen", "0") or 0))
                gen_total[gen] = _float(perf.get("total_s"))
                bucket = str(perf.get("bucket", ""))
                seconds = _float(perf.get("seconds"))
                if bucket.endswith(COUNT_SUFFIXES):
                    bucket_counts[bucket] += seconds
                else:
                    bucket_seconds[bucket] += seconds
            totals = list(gen_total.values())
            total_s = sum(totals)
            runs.append(
                {
                    "benchmark": _benchmark_for(run_id),
                    "run_id": run_id,
                    "runtime_run_id": str(row.get("runtime_run_id", "")),
                    "seed": int(row.get("seed", 0) or 0),
                    "axes": row.get("axes", {}),
                    "run_dir": str(run_dir),
                    "generations": int(float(cfg.get("sim.generations", len(totals)) or len(totals))),
                    "steps_per_generation": int(float(cfg.get("sim.steps_per_generation", 0) or 0)),
                    "device": cfg.get("sim.device", ""),
                    "bank_device": cfg.get("sim.bank_device", ""),
                    "behavior_log_enable": cfg.get("sim.behavior_log_enable", ""),
                    "checkpoint_period": cfg.get("sim.checkpoint_bundle_v2_period_gen", ""),
                    "total_wall_s": total_s,
                    "mean_sec_per_gen": statistics.mean(totals) if totals else 0.0,
                    "median_sec_per_gen": statistics.median(totals) if totals else 0.0,
                    "bucket_seconds": dict(bucket_seconds),
                    "bucket_counts": dict(bucket_counts),
                }
            )
    return runs


def _bucket_sum(bucket_seconds: dict[str, float], names: set[str]) -> float:
    return sum(float(v) for k, v in bucket_seconds.items() if k in names)


def _category_totals(bucket_seconds: dict[str, float]) -> dict[str, float]:
    """Legacy overlapping category totals retained for old reports."""
    policy = sum(
        v
        for k, v in bucket_seconds.items()
        if k.startswith("policy") or k in {"obs_build", "tensor_stage", "h2d_copy_ms", "gru_bank_sync", "bank_sync"}
    )
    env_step = sum(
        v
        for k, v in bucket_seconds.items()
        if k
        in {
            "env_apply",
            "drive_update_ms",
            "chemistry_update_ms",
            "maintenance_update_ms",
            "jungle_predator_pressure",
            "env_spawn_food",
            "alive_list_build",
        }
    )
    checkpoint = sum(v for k, v in bucket_seconds.items() if "checkpoint" in k or "bundle" in k)
    logging = sum(v for k, v in bucket_seconds.items() if "log" in k or "write" in k)
    metrics = sum(v for k, v in bucket_seconds.items() if "metric" in k or "csv" in k)
    return {
        "policy": policy,
        "env_step": env_step,
        "checkpoint": checkpoint,
        "logging": logging,
        "metrics_csv": metrics,
    }


def _non_overlap_category_totals(bucket_seconds: dict[str, float], total_wall: float) -> dict[str, float]:
    """Best-effort non-overlapping timing ledger from generation wrapper buckets.

    Newer runs emit wrapper buckets such as `tick_loop_total`,
    `metrics_total`, and `checkpoint_total`. Those are preferred because they
    partition the generation boundary better than the older nested hot-path
    buckets. For legacy runs without these wrappers, this intentionally leaves
    most time in `generation_boundary_other` instead of pretending nested
    buckets are additive.
    """

    checkpoint_total = float(bucket_seconds.get("checkpoint_total", 0.0) or 0.0)
    control_plane_total = max(
        0.0,
        float(bucket_seconds.get("control_plane_total", 0.0) or 0.0)
        - float(bucket_seconds.get("checkpoint_control_plane_total", 0.0) or 0.0),
    )
    metrics_total = _bucket_sum(
        bucket_seconds,
        {
            "metrics_total",
            "post_boundary_metrics_total",
            "phase_e_metrics_total",
            "phase_e_taxonomy_total",
        },
    )
    logging_total = _bucket_sum(
        bucket_seconds,
        {
            "behavior_logging_total",
            "progress_logging_total",
            "heartbeat_write_total",
            "viz_write_total",
        },
    )
    if logging_total <= 0.0:
        logging_total = sum(v for k, v in bucket_seconds.items() if "log" in k or "write" in k)

    categories = {
        "gen_start": float(bucket_seconds.get("gen_start_total", 0.0) or 0.0),
        "tick_loop": float(bucket_seconds.get("tick_loop_total", 0.0) or 0.0),
        "selection": float(bucket_seconds.get("selection_total", 0.0) or 0.0),
        "metrics": metrics_total,
        "generation_reproduce_cull": float(bucket_seconds.get("generation_reproduce_cull_total", 0.0) or 0.0),
        "env_population_sync": float(bucket_seconds.get("env_population_sync_total", 0.0) or 0.0),
        "control_plane": control_plane_total,
        "pressure_autopilot": float(bucket_seconds.get("pressure_autopilot_total", 0.0) or 0.0),
        "boundary_hash_replay": float(bucket_seconds.get("boundary_hash_replay_total", 0.0) or 0.0),
        "checkpoint": checkpoint_total,
        "logging_observability": logging_total,
    }
    accounted = sum(max(0.0, float(v)) for v in categories.values())
    categories["generation_boundary_other"] = max(0.0, float(total_wall) - accounted)
    return categories


def _nested_detail_totals(bucket_seconds: dict[str, float]) -> dict[str, float]:
    """Overlapping hot-path detail groups for bottleneck inspection."""

    return {
        "policy_total": sum(
            v
            for k, v in bucket_seconds.items()
            if k.startswith("policy") or k in {"obs_build", "tensor_stage", "h2d_copy_ms", "gru_bank_sync", "bank_sync"}
        ),
        "policy_prep": float(bucket_seconds.get("policy_prep_ms", 0.0) or 0.0),
        "policy_forward_total": float(bucket_seconds.get("policy_fwd_ms", 0.0) or 0.0),
        "policy_forward_gru": float(bucket_seconds.get("policy_forward_gru", 0.0) or 0.0),
        "policy_forward_mlp": float(bucket_seconds.get("policy_forward_mlp", 0.0) or 0.0),
        "policy_output_transfer": float(bucket_seconds.get("policy_output_transfer", 0.0) or 0.0),
        "policy_bank_sync": _bucket_sum(bucket_seconds, {"bank_sync", "gru_bank_sync"}),
        "obs_build_total": float(bucket_seconds.get("obs_build", 0.0) or 0.0),
        "env_action_apply_total": max(
            float(bucket_seconds.get("env_apply", 0.0) or 0.0),
            float(bucket_seconds.get("action_apply_ms", 0.0) or 0.0),
        ),
        "drive_update_total": float(bucket_seconds.get("drive_update_ms", 0.0) or 0.0),
        "drive_p1_gather": float(bucket_seconds.get("drive_p1_gather", 0.0) or 0.0),
        "reproduction_total": max(
            float(bucket_seconds.get("reproduction_ms", 0.0) or 0.0),
            float(bucket_seconds.get("generation_reproduce_cull_total", 0.0) or 0.0),
        ),
        "checkpoint_total": float(bucket_seconds.get("checkpoint_total", 0.0) or 0.0),
        "metrics_total": float(bucket_seconds.get("metrics_total", 0.0) or 0.0),
        "csv_write_total": float(bucket_seconds.get("csv_write_total", 0.0) or 0.0),
        "behavior_logging_total": _bucket_sum(bucket_seconds, {"behavior_log_tick", "behavior_logging_total"}),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def _fmt(value: float) -> str:
    return f"{value:.2f}"


def aggregate(root: Path, out_dir: Path, markdown_path: Path) -> None:
    runs = _load_runs(root)
    if not runs:
        raise SystemExit(f"no complete runs with perf_buckets.csv under {root}")

    run_rows: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []
    bench: dict[str, dict[str, Any]] = {}
    for run in runs:
        total = float(run["total_wall_s"])
        top_bucket, top_seconds = "", 0.0
        for bucket, seconds in sorted(run["bucket_seconds"].items(), key=lambda kv: kv[1], reverse=True):
            if seconds > top_seconds:
                top_bucket, top_seconds = bucket, float(seconds)
            bucket_rows.append(
                {
                    "benchmark": run["benchmark"],
                    "runtime_run_id": run["runtime_run_id"],
                    "seed": run["seed"],
                    "bucket": bucket,
                    "seconds": f"{seconds:.6f}",
                    "percent_of_generation_wall": f"{(seconds / total * 100.0) if total else 0.0:.3f}",
                }
            )
        run_rows.append(
            {
                "benchmark": run["benchmark"],
                "runtime_run_id": run["runtime_run_id"],
                "seed": run["seed"],
                "generations": run["generations"],
                "steps_per_generation": run["steps_per_generation"],
                "device": run["device"],
                "bank_device": run["bank_device"],
                "behavior_log_enable": run["behavior_log_enable"],
                "checkpoint_period": run["checkpoint_period"],
                "total_wall_s": f"{total:.6f}",
                "mean_sec_per_gen": f"{run['mean_sec_per_gen']:.6f}",
                "median_sec_per_gen": f"{run['median_sec_per_gen']:.6f}",
                "top_bucket": top_bucket,
                "top_bucket_percent": f"{(top_seconds / total * 100.0) if total else 0.0:.3f}",
                "run_dir": run["run_dir"],
                "run_id": run["run_id"],
            }
        )
        b = bench.setdefault(
            run["benchmark"],
            {
                "runs": [],
                "bucket_seconds": defaultdict(float),
                "category_seconds": defaultdict(float),
                "non_overlap_seconds": defaultdict(float),
                "nested_detail_seconds": defaultdict(float),
            },
        )
        b["runs"].append(run)
        for k, v in run["bucket_seconds"].items():
            b["bucket_seconds"][k] += float(v)
        for k, v in _category_totals(run["bucket_seconds"]).items():
            b["category_seconds"][k] += float(v)
        for k, v in _non_overlap_category_totals(run["bucket_seconds"], total).items():
            b["non_overlap_seconds"][k] += float(v)
        for k, v in _nested_detail_totals(run["bucket_seconds"]).items():
            b["nested_detail_seconds"][k] += float(v)

    bench_rows: list[dict[str, Any]] = []
    bottleneck_rows: list[dict[str, Any]] = []
    non_overlap_rows: list[dict[str, Any]] = []
    nested_detail_rows: list[dict[str, Any]] = []
    for name, data in sorted(bench.items()):
        bruns = data["runs"]
        total_wall = sum(float(r["total_wall_s"]) for r in bruns)
        total_gens = sum(int(r["generations"]) for r in bruns)
        bucket_seconds = dict(data["bucket_seconds"])
        top = sorted(bucket_seconds.items(), key=lambda kv: kv[1], reverse=True)[:5]
        top_bottleneck = top[0][0] if top else ""
        notes = []
        if any(str(r.get("checkpoint_period")) not in {"", "0"} for r in bruns):
            notes.append("checkpoint bundles every generation")
        if any(str(r.get("behavior_log_enable")).lower() == "true" for r in bruns):
            notes.append("behavior logging enabled")
        bench_rows.append(
            {
                "benchmark": name,
                "runs": len(bruns),
                "generations": total_gens,
                "steps_per_generation": bruns[0]["steps_per_generation"] if bruns else "",
                "total_wall_sec": f"{total_wall:.6f}",
                "sec_gen": f"{(total_wall / total_gens) if total_gens else 0.0:.6f}",
                "top_bottleneck": top_bottleneck,
                "notes": "; ".join(notes),
            }
        )
        categories = dict(data["category_seconds"])
        cat_sum = sum(categories.values())
        other = max(0.0, total_wall - cat_sum)
        bottleneck_rows.append(
            {
                "benchmark": name,
                "policy_pct": f"{(categories.get('policy', 0.0) / total_wall * 100.0) if total_wall else 0.0:.3f}",
                "env_step_pct": f"{(categories.get('env_step', 0.0) / total_wall * 100.0) if total_wall else 0.0:.3f}",
                "checkpoint_pct": f"{(categories.get('checkpoint', 0.0) / total_wall * 100.0) if total_wall else 0.0:.3f}",
                "logging_pct": f"{(categories.get('logging', 0.0) / total_wall * 100.0) if total_wall else 0.0:.3f}",
                "metrics_csv_pct": f"{(categories.get('metrics_csv', 0.0) / total_wall * 100.0) if total_wall else 0.0:.3f}",
                "other_pct": f"{(other / total_wall * 100.0) if total_wall else 0.0:.3f}",
            }
        )
        for category, seconds in sorted(dict(data["non_overlap_seconds"]).items()):
            non_overlap_rows.append(
                {
                    "benchmark": name,
                    "category": category,
                    "seconds": f"{float(seconds):.6f}",
                    "percent_of_generation_wall": f"{(float(seconds) / total_wall * 100.0) if total_wall else 0.0:.3f}",
                    "view": "non_overlapping_wrapper",
                }
            )
        for category, seconds in sorted(dict(data["nested_detail_seconds"]).items()):
            nested_detail_rows.append(
                {
                    "benchmark": name,
                    "category": category,
                    "seconds": f"{float(seconds):.6f}",
                    "percent_of_generation_wall": f"{(float(seconds) / total_wall * 100.0) if total_wall else 0.0:.3f}",
                    "view": "nested_detail",
                }
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        out_dir / "perf_run_summary.csv",
        run_rows,
        [
            "benchmark",
            "runtime_run_id",
            "seed",
            "generations",
            "steps_per_generation",
            "device",
            "bank_device",
            "behavior_log_enable",
            "checkpoint_period",
            "total_wall_s",
            "mean_sec_per_gen",
            "median_sec_per_gen",
            "top_bucket",
            "top_bucket_percent",
            "run_dir",
            "run_id",
        ],
    )
    _write_csv(
        out_dir / "perf_bucket_summary.csv",
        bucket_rows,
        ["benchmark", "runtime_run_id", "seed", "bucket", "seconds", "percent_of_generation_wall"],
    )
    _write_csv(
        out_dir / "perf_benchmark_summary.csv",
        bench_rows,
        ["benchmark", "runs", "generations", "steps_per_generation", "total_wall_sec", "sec_gen", "top_bottleneck", "notes"],
    )
    _write_csv(
        out_dir / "perf_bottleneck_categories.csv",
        bottleneck_rows,
        ["benchmark", "policy_pct", "env_step_pct", "checkpoint_pct", "logging_pct", "metrics_csv_pct", "other_pct"],
    )
    _write_csv(
        out_dir / "perf_category_nonoverlap.csv",
        non_overlap_rows,
        ["benchmark", "category", "seconds", "percent_of_generation_wall", "view"],
    )
    _write_csv(
        out_dir / "perf_nested_detail_summary.csv",
        nested_detail_rows,
        ["benchmark", "category", "seconds", "percent_of_generation_wall", "view"],
    )

    lines: list[str] = [
        "# EIEL Perf Baseline Aggregation",
        "",
        "Counter-like perf fields ending in `_count` or `_requests` were excluded from timing percentages.",
        "Bucket timings can overlap by instrumentation region, so detailed bucket percentages are bottleneck signals, not an additive accounting ledger.",
        "The non-overlapping view uses explicit wrapper buckets when available and leaves uninstrumented time in `generation_boundary_other`.",
        "",
        "## Benchmark Summary",
        "",
        "| benchmark | runs | generations | steps/gen | total wall sec | sec/gen | top bottleneck | notes |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in bench_rows:
        lines.append(
            f"| {row['benchmark']} | {row['runs']} | {row['generations']} | {row['steps_per_generation']} | "
            f"{_fmt(float(row['total_wall_sec']))} | {_fmt(float(row['sec_gen']))} | {row['top_bottleneck']} | {row['notes']} |"
        )
    lines.extend(
        [
            "",
            "## Non-Overlapping Category View",
            "",
            "| benchmark | category | seconds | % generation wall |",
            "|---|---|---:|---:|",
        ]
    )
    for row in non_overlap_rows:
        lines.append(
            f"| {row['benchmark']} | {row['category']} | {_fmt(float(row['seconds']))} | "
            f"{_fmt(float(row['percent_of_generation_wall']))} |"
        )
    lines.extend(
        [
            "",
            "## Legacy Overlapping Bottleneck Categories",
            "",
            "| benchmark | policy % | env step % | checkpoint % | logging % | metrics/CSV % | other % |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in bottleneck_rows:
        lines.append(
            f"| {row['benchmark']} | {_fmt(float(row['policy_pct']))} | {_fmt(float(row['env_step_pct']))} | "
            f"{_fmt(float(row['checkpoint_pct']))} | {_fmt(float(row['logging_pct']))} | "
            f"{_fmt(float(row['metrics_csv_pct']))} | {_fmt(float(row['other_pct']))} |"
        )
    lines.extend(
        [
            "",
            "## Nested Detail View",
            "",
            "| benchmark | detail | seconds | % generation wall |",
            "|---|---|---:|---:|",
        ]
    )
    for row in nested_detail_rows:
        if float(row["seconds"]) <= 0.0:
            continue
        lines.append(
            f"| {row['benchmark']} | {row['category']} | {_fmt(float(row['seconds']))} | "
            f"{_fmt(float(row['percent_of_generation_wall']))} |"
        )
    lines.extend(["", "## Top Timing Buckets", ""])
    for name, data in sorted(bench.items()):
        total_wall = sum(float(r["total_wall_s"]) for r in data["runs"])
        lines.extend([f"### {name}", "", "| bucket | seconds | % generation wall |", "|---|---:|---:|"])
        for bucket, seconds in sorted(dict(data["bucket_seconds"]).items(), key=lambda kv: kv[1], reverse=True)[:10]:
            pct = (float(seconds) / total_wall * 100.0) if total_wall else 0.0
            lines.append(f"| {bucket} | {_fmt(float(seconds))} | {_fmt(pct)} |")
        lines.append("")
    lines.extend(["## Run Inventory", ""])
    for row in run_rows:
        lines.append(f"- `{row['runtime_run_id']}` {row['benchmark']} seed {row['seed']}: `{row['run_dir']}`")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate EIEL perf_buckets.csv files across benchmark batches")
    parser.add_argument("root", help="Runtime root or runs root to scan")
    parser.add_argument("--out-dir", required=True, help="Directory for CSV summaries")
    parser.add_argument("--markdown", required=True, help="Markdown summary output path")
    args = parser.parse_args()
    aggregate(Path(args.root), Path(args.out_dir), Path(args.markdown))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
