#!/usr/bin/env python3
"""Summarize the MEM-MAX-2 tiny technical preflight.

The script reads ignored runtime outputs and writes compact report-facing
CSV/JSON summaries. It does not modify simulation behavior.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


DEMAND_ORDER = [
    "flat_control_clean",
    "cue_observation_off_control",
    "hazard_off_control",
    "memmax_short_gap",
    "memmax_useful_gap",
    "memmax_too_long_gap",
]
MEMORY_ORDER = ["memory_normal", "memory_ablated"]
REQUIRED_COLUMNS = [
    "mem_evo_population_memory_dim_histogram",
    "mem_evo_selected_parent_memory_dim_histogram",
    "mem_evo_descendant_memory_dim_histogram",
    "mem_evo_lineage_memory_dim_summary_json",
    "mem_evo_effective_memory_dim_histogram",
    "mem_evo_use_gru_share",
    "mem_evo_memory_cost_mean",
    "mem_evo_memory_cost_total",
    "mem_evo_memory_dim_selected_minus_population_mean",
    "mem_evo_memory_dim_descendant_minus_initial_mean",
    "mem_evo_memory_dim_wdl_minus_flat_delta",
    "mean_memory_dim_parent",
    "memory_dim_mean",
    "effective_memory_mean",
    "memory_cost_mean",
    "memory_cost_total",
    "memory_ablation_active_flag",
    "memory_reset_each_tick_active_flag",
    "wdl_delayed_hazard_enable",
    "wdl_delayed_hazard_agent_cue_exposure_count",
    "wdl_delayed_hazard_agent_active_exposure_count",
    "wdl_delayed_hazard_damage_total",
    "wdl_delayed_hazard_cue_to_active_delay_mean",
    "wdl_delayed_hazard_active_without_prior_cue_count",
    "lineage_top1_alive_share",
    "lineage_effective_alive_lineages",
    "top_lineage_memory_dim",
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(row: dict[str, str], key: str) -> float:
    try:
        value = row.get(key, "")
        if value == "":
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _mean(rows: list[dict[str, str]], key: str) -> float:
    vals = [_f(row, key) for row in rows]
    return sum(vals) / len(vals) if vals else 0.0


def _max(rows: list[dict[str, str]], key: str) -> float:
    vals = [_f(row, key) for row in rows]
    return max(vals) if vals else 0.0


def _runtime_parts(run_id: str) -> tuple[str, str, int]:
    marker = "__memory_"
    if marker not in run_id:
        return "", "", 0
    memory = run_id.split("__memory_", 1)[1].split("__demand_", 1)[0]
    demand = run_id.split("__demand_", 1)[1].split("__seed", 1)[0]
    seed = int(run_id.rsplit("__seed", 1)[1])
    return memory, demand, seed


def _load_full(full_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary = _read_json(full_root / "parallel_campaign_summary.json")
    results = summary.get("results", [])
    rows: list[dict[str, Any]] = []
    for result in results:
        run_dir = Path(str(result.get("run_dir", "")))
        metrics = run_dir / "metrics.csv"
        if not metrics.exists():
            continue
        data = _rows(metrics)
        final = data[-1] if data else {}
        run_id = str(result.get("run_id", ""))
        memory_arm, demand_cell, seed = _runtime_parts(run_id)
        row = {
            "run_id": run_id,
            "runtime_run_id": run_dir.name,
            "memory_arm": memory_arm,
            "demand_cell": demand_cell,
            "seed": seed,
            "run_dir": str(run_dir),
            "metrics_path": str(metrics),
            "metrics_rows": len(data),
            "metrics_sha256": _sha256(metrics),
            "config_sha256": _sha256(run_dir / "config.json") if (run_dir / "config.json").exists() else "",
            "run_config_sha256": _sha256(run_dir / "run_config.csv") if (run_dir / "run_config.csv").exists() else "",
            "status": str(result.get("status", "")),
            "returncode": int(result.get("returncode", -1)),
            "wall_s": float(result.get("wall_s", 0.0) or 0.0),
            "nonfinite_cells": 0,
            "missing_required_columns": [c for c in REQUIRED_COLUMNS if c not in final],
            "generation_final": int(float(final.get("generation", "-1") or -1)),
        }
        for key in REQUIRED_COLUMNS:
            row[f"{key}_final"] = _f(final, key)
            row[f"{key}_mean"] = _mean(data, key)
            row[f"{key}_max"] = _max(data, key)
        for record in data:
            for value in record.values():
                try:
                    x = float(value)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(x):
                    row["nonfinite_cells"] += 1
        rows.append(row)
    rows.sort(key=lambda r: (MEMORY_ORDER.index(r["memory_arm"]), DEMAND_ORDER.index(r["demand_cell"]), r["seed"]))
    return rows, summary


def _latest_batch(root: Path) -> Path:
    batches = sorted(root.glob("e/*/b*"), key=lambda p: p.stat().st_mtime)
    if not batches:
        raise FileNotFoundError(f"no batch under {root}")
    return batches[-1]


def _load_repeat(root: Path) -> dict[str, dict[str, Any]]:
    batch = _latest_batch(root)
    out: dict[str, dict[str, Any]] = {}
    for metrics in sorted((batch / "r").glob("*/metrics.csv")):
        run_dir = metrics.parent
        data = _rows(metrics)
        final_payload = json.dumps(data[-1] if data else {}, sort_keys=True, separators=(",", ":"))
        out[run_dir.name] = {
            "batch": str(batch),
            "run_dir": str(run_dir),
            "metrics_sha256": _sha256(metrics),
            "final_row_sha256": hashlib.sha256(final_payload.encode("utf-8")).hexdigest(),
            "config_sha256": _sha256(run_dir / "config.json") if (run_dir / "config.json").exists() else "",
            "run_config_sha256": _sha256(run_dir / "run_config.csv") if (run_dir / "run_config.csv").exists() else "",
            "metrics_rows": len(data),
        }
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _by(rows: list[dict[str, Any]], memory: str, demand: str) -> dict[str, Any]:
    for row in rows:
        if row["memory_arm"] == memory and row["demand_cell"] == demand:
            return row
    return {}


def _decision(rows: list[dict[str, Any]], repeat_pass: bool, completion_ok: bool) -> tuple[str, list[str], str]:
    reasons: list[str] = []
    useful = _by(rows, "memory_normal", "memmax_useful_gap")
    ablated_useful = _by(rows, "memory_ablated", "memmax_useful_gap")
    if not completion_ok or not repeat_pass:
        return "MEM-MAX-2C", ["completion or repeat parity did not pass"], "repair reporting/runtime gate"
    if not useful:
        return "MEM-MAX-2C", ["missing useful-gap normal-memory cell"], "repair reporting"
    useful_sel = float(useful.get("mem_evo_memory_dim_selected_minus_population_mean_final", 0.0))
    useful_desc = float(useful.get("mem_evo_memory_dim_descendant_minus_initial_mean_final", 0.0))
    if ablated_useful:
        ablated_sel = float(ablated_useful.get("mem_evo_memory_dim_selected_minus_population_mean_final", 0.0))
        if ablated_sel >= useful_sel:
            reasons.append(
                f"useful-gap ablated selected-minus-population ({ablated_sel:.6f}) >= useful-gap normal ({useful_sel:.6f})"
            )
    for demand in ["flat_control_clean", "cue_observation_off_control", "hazard_off_control", "memmax_short_gap", "memmax_too_long_gap"]:
        control = _by(rows, "memory_normal", demand)
        if not control:
            continue
        control_sel = float(control.get("mem_evo_memory_dim_selected_minus_population_mean_final", 0.0))
        if control_sel >= useful_sel:
            reasons.append(f"normal {demand} selected-minus-population ({control_sel:.6f}) >= useful-gap normal ({useful_sel:.6f})")
    if useful_desc <= 0.0:
        reasons.append("useful-gap normal descendant-minus-initial is not positive")
    top1 = float(useful.get("lineage_top1_alive_share_final", 0.0))
    if top1 >= 0.90:
        reasons.append(f"useful-gap normal final top1 lineage share is high ({top1:.6f})")
    if reasons:
        return "MEM-MAX-2D", reasons, "revise or repair ladder controls before broader MEM-MAX"
    return "MEM-MAX-2A", ["useful-gap normal dominates controls without ablated/control invalidation"], "run a slightly broader MEM-MAX calibration"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-root", required=True)
    ap.add_argument("--repeat-a-root", required=True)
    ap.add_argument("--repeat-b-root", required=True)
    ap.add_argument("--dry-run-batch", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, full_summary = _load_full(Path(args.full_root))
    repeat_a = _load_repeat(Path(args.repeat_a_root))
    repeat_b = _load_repeat(Path(args.repeat_b_root))
    comparisons: list[dict[str, Any]] = []
    for runtime_id in sorted(set(repeat_a) | set(repeat_b)):
        a = repeat_a.get(runtime_id, {})
        b = repeat_b.get(runtime_id, {})
        comparisons.append(
            {
                "runtime_run_id": runtime_id,
                "metrics_sha_match": a.get("metrics_sha256") == b.get("metrics_sha256"),
                "final_row_sha_match": a.get("final_row_sha256") == b.get("final_row_sha256"),
                "config_sha_match": a.get("config_sha256") == b.get("config_sha256"),
                "run_config_sha_match": a.get("run_config_sha256") == b.get("run_config_sha256"),
                "metrics_rows_a": a.get("metrics_rows", 0),
                "metrics_rows_b": b.get("metrics_rows", 0),
                "repeat_a_metrics_sha256": a.get("metrics_sha256", ""),
                "repeat_b_metrics_sha256": b.get("metrics_sha256", ""),
            }
        )
    repeat_pass = all(
        c["metrics_sha_match"] and c["final_row_sha_match"] and c["config_sha_match"] and c["run_config_sha_match"]
        for c in comparisons
    )
    completion_ok = (
        int(full_summary.get("planned_runs", 0) or 0) == 12
        and len(rows) == 12
        and len(full_summary.get("failures", [])) == 0
        and all(r["status"] == "ok" and r["returncode"] == 0 and r["metrics_rows"] == 24 for r in rows)
    )
    decision, reasons, recommendation = _decision(rows, repeat_pass, completion_ok)

    base_fields = [
        "run_id",
        "runtime_run_id",
        "memory_arm",
        "demand_cell",
        "seed",
        "status",
        "returncode",
        "metrics_rows",
        "wall_s",
        "run_dir",
        "metrics_sha256",
        "config_sha256",
        "run_config_sha256",
    ]
    run_matrix_fields = base_fields
    _write_csv(out_dir / "run_matrix.csv", rows, run_matrix_fields)

    metric_fields = [
        "memory_arm",
        "demand_cell",
        "seed",
        "mem_evo_memory_dim_selected_minus_population_mean_final",
        "mem_evo_memory_dim_descendant_minus_initial_mean_final",
        "mean_memory_dim_parent_final",
        "memory_dim_mean_final",
        "effective_memory_mean_final",
        "mem_evo_use_gru_share_final",
        "mem_evo_memory_dim_wdl_minus_flat_delta_final",
    ]
    _write_csv(out_dir / "memory_demand_ladder_summary.csv", rows, metric_fields)
    _write_json(out_dir / "memory_demand_ladder_summary.json", rows)

    shift_fields = metric_fields + [
        "mem_evo_population_memory_dim_histogram_final",
        "mem_evo_selected_parent_memory_dim_histogram_final",
        "mem_evo_descendant_memory_dim_histogram_final",
        "mem_evo_effective_memory_dim_histogram_final",
    ]
    _write_csv(out_dir / "mem_evo_shift_summary.csv", rows, shift_fields)
    _write_json(out_dir / "mem_evo_shift_summary.json", rows)

    ablated_rows = [r for r in rows if r["memory_arm"] == "memory_ablated"]
    _write_csv(out_dir / "ablated_control_summary.csv", ablated_rows, metric_fields)
    _write_json(out_dir / "ablated_control_summary.json", ablated_rows)

    wdl_fields = [
        "memory_arm",
        "demand_cell",
        "seed",
        "wdl_delayed_hazard_enable_final",
        "wdl_delayed_hazard_agent_cue_exposure_count_final",
        "wdl_delayed_hazard_agent_active_exposure_count_final",
        "wdl_delayed_hazard_damage_total_final",
        "wdl_delayed_hazard_damage_mean_final",
        "wdl_delayed_hazard_cue_to_active_delay_mean_final",
        "wdl_delayed_hazard_active_without_prior_cue_count_final",
    ]
    _write_csv(out_dir / "wdl_timing_damage_summary.csv", rows, wdl_fields)
    _write_json(out_dir / "wdl_timing_damage_summary.json", rows)

    lineage_fields = [
        "memory_arm",
        "demand_cell",
        "seed",
        "lineage_top1_alive_share_final",
        "lineage_top1_alive_share_max",
        "lineage_effective_alive_lineages_final",
        "top_lineage_memory_dim_final",
    ]
    _write_csv(out_dir / "lineage_confidence_summary.csv", rows, lineage_fields)
    _write_json(out_dir / "lineage_confidence_summary.json", rows)

    cost_fields = [
        "memory_arm",
        "demand_cell",
        "seed",
        "effective_memory_mean_final",
        "memory_cost_mean_final",
        "memory_cost_total_final",
        "mem_evo_memory_cost_mean_final",
        "mem_evo_memory_cost_total_final",
        "memory_ablation_active_flag_final",
        "memory_reset_each_tick_active_flag_final",
    ]
    _write_csv(out_dir / "cost_effective_memory_context.csv", rows, cost_fields)
    _write_json(out_dir / "cost_effective_memory_context.json", rows)

    completion = {
        "planned_runs": full_summary.get("planned_runs"),
        "completed_runs": len(rows),
        "failures": len(full_summary.get("failures", [])),
        "missing_runs": max(0, int(full_summary.get("planned_runs", 0) or 0) - len(rows)),
        "completion_ok": completion_ok,
        "nonfinite_total": sum(int(r["nonfinite_cells"]) for r in rows),
        "rows_with_missing_required_columns": [
            {"run_id": r["run_id"], "missing_required_columns": r["missing_required_columns"]}
            for r in rows
            if r["missing_required_columns"]
        ],
        "full_root": str(Path(args.full_root)),
        "dry_run_batch": str(Path(args.dry_run_batch)),
        "parallel_summary": str(Path(args.full_root) / "parallel_campaign_summary.json"),
        "total_wall_s": full_summary.get("total_wall_s"),
        "workers": full_summary.get("workers"),
    }
    _write_json(out_dir / "completion_summary.json", completion)

    repeat = {
        "repeat_pass": repeat_pass,
        "repeat_a_root": str(Path(args.repeat_a_root)),
        "repeat_b_root": str(Path(args.repeat_b_root)),
        "comparisons": comparisons,
    }
    _write_json(out_dir / "repeat_parity_summary.json", repeat)

    decision_payload = {
        "decision": decision,
        "reasons": reasons,
        "recommended_next_step": recommendation,
        "completion_ok": completion_ok,
        "repeat_pass": repeat_pass,
        "behavior_neutrality": {
            "mechanics_changes": False,
            "direct_memory_reward": False,
            "direct_cognition_reward": False,
            "reward_shaping": False,
            "lineage_rescue": False,
            "protected_reproduction": False,
            "archive_reseeding": False,
            "survival_subsidy": False,
        },
        "no_claims_statement": (
            "MEM-MAX-2 does not show intelligence, emergence, consciousness, planning, culture, "
            "prediction selection, general cognition, broad adaptive memory, learned refuge use, "
            "evolved refuge strategy, superintelligence, or MEM-EVO-A. It does not add direct "
            "cognition reward, memory reward, diversity reward, lineage rescue, protected reproduction, "
            "archive reseeding, or survival subsidy. It is a tiny technical preflight of a memory-demand "
            "calibration ladder whose purpose is to test whether deliberately delayed-consequence "
            "conditions can produce stronger memory pressure than matched controls."
        ),
    }
    _write_json(out_dir / "mem_max_2_decision.json", decision_payload)

    print(json.dumps({"decision": decision, "completion_ok": completion_ok, "repeat_pass": repeat_pass, "reasons": reasons}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
