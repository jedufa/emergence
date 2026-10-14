#!/usr/bin/env python3
from __future__ import annotations

"""Generate MEM-EVO-3 tiny memory selection-shift summaries from batch outputs."""

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "docs/900_lifecycle/940_incoming/mem_evo_3_tiny_memory_selection_shift"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _f(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value in ("", None):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _json(row: Mapping[str, Any], key: str) -> dict[str, Any]:
    try:
        value = json.loads(str(row.get(key, "{}") or "{}"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _hist_mean(row: Mapping[str, Any], key: str) -> float:
    return float(_json(row, key).get("mean", 0.0) or 0.0)


def _hist_bucket_fraction(row: Mapping[str, Any], key: str, bucket: str) -> float:
    payload = _json(row, key)
    fractions = payload.get("fractions", {})
    if isinstance(fractions, dict):
        return float(fractions.get(str(bucket), 0.0) or 0.0)
    return 0.0


def _mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _stdev(values: list[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def load_runs(batch_dir: Path) -> list[dict[str, Any]]:
    matrix_path = batch_dir / "matrix.csv"
    rows: list[dict[str, Any]] = []
    for matrix in read_csv(matrix_path):
        metrics_path = Path(matrix["run_dir"]) / "metrics.csv"
        metrics = read_csv(metrics_path)
        if not metrics:
            raise ValueError(f"empty metrics: {metrics_path}")
        first = metrics[0]
        last = metrics[-1]
        selected_values = [_f(row, "mem_evo_memory_dim_selected_minus_population_mean") for row in metrics]
        population_values = [_hist_mean(row, "mem_evo_population_memory_dim_histogram") for row in metrics]
        parent_values = [_hist_mean(row, "mem_evo_selected_parent_memory_dim_histogram") for row in metrics]
        final_lineage = _json(last, "mem_evo_lineage_memory_dim_summary_json")
        lineages = final_lineage.get("lineages", []) if isinstance(final_lineage.get("lineages", []), list) else []
        lineage_memory_means = [float(item.get("memory_dim_mean", 0.0) or 0.0) for item in lineages if isinstance(item, dict)]
        lineage_counts = [float(item.get("count", 0.0) or 0.0) for item in lineages if isinstance(item, dict)]
        max_lineage_count = max(lineage_counts) if lineage_counts else 0.0
        lineage_total_count = sum(lineage_counts)
        rows.append(
            {
                "run_id": matrix["run_id"],
                "runtime_run_id": matrix["runtime_run_id"],
                "run_dir": matrix["run_dir"],
                "metrics_path": str(metrics_path),
                "metrics_sha256": sha256_file(metrics_path),
                "memory_arm": matrix["memory_arm"],
                "world_context": matrix["world_context"],
                "seed": int(matrix["seed"]),
                "generation_count": len(metrics),
                "initial_population_memory_dim_mean": _hist_mean(first, "mem_evo_population_memory_dim_histogram"),
                "final_population_memory_dim_mean": _hist_mean(last, "mem_evo_population_memory_dim_histogram"),
                "final_effective_memory_dim_mean": _hist_mean(last, "mem_evo_effective_memory_dim_histogram"),
                "final_memory_dim_bucket_0_fraction": _hist_bucket_fraction(last, "mem_evo_population_memory_dim_histogram", "0"),
                "final_memory_dim_bucket_8_fraction": _hist_bucket_fraction(last, "mem_evo_population_memory_dim_histogram", "8"),
                "final_memory_dim_bucket_16_fraction": _hist_bucket_fraction(last, "mem_evo_population_memory_dim_histogram", "16"),
                "final_memory_dim_bucket_32_fraction": _hist_bucket_fraction(last, "mem_evo_population_memory_dim_histogram", "32"),
                "final_memory_dim_bucket_64_fraction": _hist_bucket_fraction(last, "mem_evo_population_memory_dim_histogram", "64"),
                "descendant_minus_initial_mean": _hist_mean(last, "mem_evo_population_memory_dim_histogram")
                - _hist_mean(first, "mem_evo_population_memory_dim_histogram"),
                "selected_minus_population_final": _f(last, "mem_evo_memory_dim_selected_minus_population_mean"),
                "selected_minus_population_mean": _mean(selected_values),
                "selected_minus_population_stdev": _stdev(selected_values),
                "population_memory_dim_mean_over_generations": _mean(population_values),
                "selected_parent_memory_dim_mean_over_generations": _mean(parent_values),
                "final_use_gru_share": _f(last, "mem_evo_use_gru_share"),
                "final_memory_cost_mean": _f(last, "mem_evo_memory_cost_mean"),
                "final_memory_cost_total": _f(last, "mem_evo_memory_cost_total"),
                "final_memory_ablation_active_flag": _f(last, "memory_ablation_active_flag"),
                "final_memory_force_disabled_active_flag": _f(last, "memory_force_disabled_active_flag"),
                "lineage_count": int(final_lineage.get("lineage_count", 0) or 0),
                "top_lineage_share": float(max_lineage_count / lineage_total_count) if lineage_total_count > 0.0 else 0.0,
                "lineage_memory_dim_mean_mean": _mean(lineage_memory_means),
                "lineage_memory_dim_mean_stdev": _stdev(lineage_memory_means),
                "final_lineage_memory_dim_summary_json": json.dumps(final_lineage, sort_keys=True),
            }
        )
    return rows


def group_summary(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    keys = sorted({(str(row["memory_arm"]), str(row["world_context"])) for row in rows})
    for memory_arm, world_context in keys:
        group = [row for row in rows if row["memory_arm"] == memory_arm and row["world_context"] == world_context]
        out.append(
            {
                "memory_arm": memory_arm,
                "world_context": world_context,
                "run_count": len(group),
                "final_population_memory_dim_mean": _mean([float(row["final_population_memory_dim_mean"]) for row in group]),
                "descendant_minus_initial_mean": _mean([float(row["descendant_minus_initial_mean"]) for row in group]),
                "selected_minus_population_final_mean": _mean([float(row["selected_minus_population_final"]) for row in group]),
                "selected_minus_population_generation_mean": _mean([float(row["selected_minus_population_mean"]) for row in group]),
                "effective_memory_dim_final_mean": _mean([float(row["final_effective_memory_dim_mean"]) for row in group]),
                "memory_cost_final_mean": _mean([float(row["final_memory_cost_mean"]) for row in group]),
                "top_lineage_share_mean": _mean([float(row["top_lineage_share"]) for row in group]),
            }
        )
    return out


def _group_value(summary: list[Mapping[str, Any]], arm: str, world: str, key: str) -> float:
    for row in summary:
        if row["memory_arm"] == arm and row["world_context"] == world:
            return float(row[key])
    return 0.0


def wdl_minus_flat(summary: list[Mapping[str, Any]], arm: str, key: str) -> float:
    return _group_value(summary, arm, "wdl_delayed_hazard", key) - _group_value(summary, arm, "flat_control_clean", key)


def parity_summary(full_batch: Path, parity_batch: Path | None) -> dict[str, Any]:
    if parity_batch is None:
        return {"status": "not_run"}
    full_matrix = {row["runtime_run_id"]: row for row in read_csv(full_batch / "matrix.csv")}
    out: list[dict[str, Any]] = []
    for row in read_csv(parity_batch / "matrix.csv"):
        rid = row["runtime_run_id"]
        full_metrics = Path(full_matrix[rid]["run_dir"]) / "metrics.csv"
        repeat_metrics = Path(row["run_dir"]) / "metrics.csv"
        with full_metrics.open(newline="", encoding="utf-8") as handle:
            full_rows = list(csv.DictReader(handle))
        with repeat_metrics.open(newline="", encoding="utf-8") as handle:
            repeat_rows = list(csv.DictReader(handle))
        full_last = full_rows[-1] if full_rows else {}
        repeat_last = repeat_rows[-1] if repeat_rows else {}
        final_mismatch_count = sum(1 for key in set(full_last) | set(repeat_last) if full_last.get(key) != repeat_last.get(key))
        full_sha = sha256_file(full_metrics)
        repeat_sha = sha256_file(repeat_metrics)
        out.append(
            {
                "run_id": row["run_id"],
                "runtime_run_id": rid,
                "metrics_file_match": full_sha == repeat_sha,
                "final_row_match": final_mismatch_count == 0,
                "final_row_mismatch_count": final_mismatch_count,
                "full_metrics_sha256": full_sha,
                "repeat_metrics_sha256": repeat_sha,
            }
        )
    return {
        "status": "passed" if out and all(row["metrics_file_match"] and row["final_row_match"] for row in out) else "failed",
        "case_count": len(out),
        "cases": out,
    }


def decide(rows: list[Mapping[str, Any]], summary: list[Mapping[str, Any]], parity: Mapping[str, Any]) -> dict[str, Any]:
    normal_final_delta = wdl_minus_flat(summary, "memory_normal", "final_population_memory_dim_mean")
    normal_desc_delta = wdl_minus_flat(summary, "memory_normal", "descendant_minus_initial_mean")
    normal_selected_final_delta = wdl_minus_flat(summary, "memory_normal", "selected_minus_population_final_mean")
    normal_selected_gen_delta = wdl_minus_flat(summary, "memory_normal", "selected_minus_population_generation_mean")
    ablated_selected_gen_delta = wdl_minus_flat(summary, "memory_ablated", "selected_minus_population_generation_mean")
    memory_normal_cost = _group_value(summary, "memory_normal", "wdl_delayed_hazard", "memory_cost_final_mean")
    effective_normal = _group_value(summary, "memory_normal", "wdl_delayed_hazard", "effective_memory_dim_final_mean")
    effective_ablated = _group_value(summary, "memory_ablated", "wdl_delayed_hazard", "effective_memory_dim_final_mean")
    repeat_ok = str(parity.get("status")) == "passed"
    integrity_ok = len(rows) == 8 and repeat_ok and memory_normal_cost > 0.0 and effective_normal > 0.0 and effective_ablated == 0.0
    wdl_specific_clean = (
        normal_final_delta > 0.0
        and normal_desc_delta > 0.0
        and normal_selected_final_delta > 0.0
        and normal_selected_gen_delta > 0.0
        and abs(ablated_selected_gen_delta) < abs(normal_selected_gen_delta)
    )
    if not integrity_ok:
        decision = "MEM-EVO-3C"
        recommendation = "Repair reporting, cost/effective-memory context, or repeat parity before interpreting MEM-EVO-3."
    elif wdl_specific_clean:
        decision = "MEM-EVO-3A"
        recommendation = "Treat as promising tiny selection-shift evidence and design a bounded confirmation, not a broad scale-up."
    else:
        decision = "MEM-EVO-3B"
        recommendation = "Keep MEM-EVO-B and do not scale until design changes or a narrower follow-up are justified."
    return {
        "decision": decision,
        "integrity_ok": bool(integrity_ok),
        "repeat_parity_ok": bool(repeat_ok),
        "memory_normal_wdl_minus_flat_final_population_memory_dim_mean": normal_final_delta,
        "memory_normal_wdl_minus_flat_descendant_minus_initial_mean": normal_desc_delta,
        "memory_normal_wdl_minus_flat_selected_minus_population_final_mean": normal_selected_final_delta,
        "memory_normal_wdl_minus_flat_selected_minus_population_generation_mean": normal_selected_gen_delta,
        "memory_ablated_wdl_minus_flat_selected_minus_population_generation_mean": ablated_selected_gen_delta,
        "memory_normal_wdl_final_memory_cost_mean": memory_normal_cost,
        "memory_normal_wdl_final_effective_memory_dim_mean": effective_normal,
        "memory_ablated_wdl_final_effective_memory_dim_mean": effective_ablated,
        "wdl_specific_clean_shift": bool(wdl_specific_clean),
        "recommendation": recommendation,
    }


def generate(batch_dir: Path, parity_batch_dir: Path | None, out_dir: Path) -> dict[str, Any]:
    rows = load_runs(batch_dir)
    summary = group_summary(rows)
    parity = parity_summary(batch_dir, parity_batch_dir)
    decision = decide(rows, summary, parity)
    matrix_rows = read_csv(batch_dir / "matrix.csv")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "run_matrix.csv", matrix_rows, list(matrix_rows[0].keys()))
    run_fields = [
        "run_id",
        "runtime_run_id",
        "memory_arm",
        "world_context",
        "seed",
        "initial_population_memory_dim_mean",
        "final_population_memory_dim_mean",
        "descendant_minus_initial_mean",
        "selected_minus_population_final",
        "selected_minus_population_mean",
        "final_effective_memory_dim_mean",
        "final_memory_cost_mean",
        "final_use_gru_share",
        "top_lineage_share",
        "metrics_sha256",
        "metrics_path",
    ]
    write_csv(out_dir / "memory_dim_shift_by_world_arm_seed.csv", rows, run_fields)
    write_json(out_dir / "memory_dim_shift_by_world_arm_seed.json", rows)
    selected_fields = [
        "memory_arm",
        "world_context",
        "run_count",
        "selected_minus_population_final_mean",
        "selected_minus_population_generation_mean",
    ]
    write_csv(out_dir / "selected_parent_shift_summary.csv", summary, selected_fields)
    write_json(out_dir / "selected_parent_shift_summary.json", summary)
    descendant_fields = [
        "memory_arm",
        "world_context",
        "run_count",
        "final_population_memory_dim_mean",
        "descendant_minus_initial_mean",
    ]
    write_csv(out_dir / "descendant_shift_summary.csv", summary, descendant_fields)
    write_json(out_dir / "descendant_shift_summary.json", summary)
    lineage_fields = [
        "run_id",
        "memory_arm",
        "world_context",
        "seed",
        "lineage_count",
        "top_lineage_share",
        "lineage_memory_dim_mean_mean",
        "lineage_memory_dim_mean_stdev",
        "final_lineage_memory_dim_summary_json",
    ]
    write_csv(out_dir / "lineage_memory_dim_shift_summary.csv", rows, lineage_fields)
    write_json(out_dir / "lineage_memory_dim_shift_summary.json", rows)
    cost_fields = [
        "run_id",
        "memory_arm",
        "world_context",
        "seed",
        "final_population_memory_dim_mean",
        "final_effective_memory_dim_mean",
        "final_memory_cost_mean",
        "final_memory_cost_total",
        "final_memory_ablation_active_flag",
        "final_memory_force_disabled_active_flag",
        "final_use_gru_share",
    ]
    write_csv(out_dir / "cost_effective_memory_context.csv", rows, cost_fields)
    write_json(out_dir / "cost_effective_memory_context.json", rows)
    write_json(out_dir / "repeat_parity_summary.json", parity)
    write_json(out_dir / "mem_evo_3_decision.json", decision)
    return {"decision": decision, "parity": parity, "summary": summary, "rows": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--parity-batch-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    result = generate(args.batch_dir, args.parity_batch_dir, args.out_dir)
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
