#!/usr/bin/env python3
from __future__ import annotations

"""MEM-MAX-6 reporting repair, registry gate, and result summarizer.

The helper is reporting-only. It validates that the MEM-MAX-6 registry wires
existing default-off instrumentation into the MEM-MAX-4 gate contract, then
summarizes dry-run or run outputs when present.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = PROJECT_ROOT / "_spec/registry/v1/mem_max_6_repaired_memory_demand_tiny_preflight.yaml"
OUT_DIR = PROJECT_ROOT / "docs/900_lifecycle/940_incoming/mem_max_6_reporting_repair_and_preflight"
MEM4_HELPER = PROJECT_ROOT / "tools/bin/mem_memory_demand_interpretation_gate.py"

MEMORY_ARMS = ["memory_normal", "memory_ablated"]
DEMAND_CELLS = [
    "flat_control_clean",
    "cue_observation_off_control",
    "hazard_off_control",
    "memmax_useful_gap",
    "memmax_short_gap",
]
NO_CLAIMS = (
    "MEM-MAX-6 does not show intelligence, emergence, consciousness, planning, culture, prediction selection, "
    "general cognition, broad adaptive memory, learned refuge use, evolved refuge strategy, superintelligence, "
    "or MEM-EVO-A. It does not add direct cognition reward, memory reward, diversity reward, lineage rescue, "
    "protected reproduction, archive reseeding, or survival subsidy. It repairs/report-wires the memory-demand "
    "interpretation gates and, only if allowed by those gates, runs a tiny technical preflight of a repaired "
    "memory-demand calibration ladder."
)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[Mapping[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _float(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def _bool(row: Mapping[str, Any], key: str, default: bool = False) -> bool:
    value = row.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_registry(path: Path = REGISTRY) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _knobs(pairs: Iterable[Any]) -> dict[str, str]:
    items: list[str] = []
    for pair in pairs:
        if isinstance(pair, list):
            items.extend(str(x) for x in pair)
    out: dict[str, str] = {}
    for idx, item in enumerate(items):
        if item == "--knob" and idx + 1 < len(items):
            key, _, value = items[idx + 1].partition("=")
            out[key] = value
    return out


def _flags_knobs(flags: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for idx, item in enumerate(flags):
        if item == "--knob" and idx + 1 < len(flags):
            key, _, value = str(flags[idx + 1]).partition("=")
            out[key] = value
    return out


def _mem4():
    spec = importlib.util.spec_from_file_location("mem_memory_demand_interpretation_gate", MEM4_HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {MEM4_HELPER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reporting_repair_field_map(registry: Mapping[str, Any] | None = None) -> dict[str, Any]:
    reg = dict(registry or load_registry())
    defaults = _knobs(reg.get("cli_defaults", []))
    return {
        "schema_version": "mem_max_6_reporting_repair_field_map_v2",
        "status": "reporting_repair_wired",
        "registry": str(REGISTRY),
        "behavior_neutrality": {
            "reporting_only": True,
            "prediction_demand_action_context_default_off_in_engine": True,
            "reward_changes": False,
            "selection_changes": False,
            "reproduction_changes": False,
            "mutation_changes": False,
            "memory_architecture_changes": False,
            "wdl_mechanics_changes": False,
            "policy_behavior_changes": False,
        },
        "actual_field_map": {
            "mem_max_hidden_window_behavior_metric_available": [
                "hidden_window_behavior_coupling_summary.json",
                "wdl_timing_damage_summary.json",
                "metrics.csv: wdl_cue_seen_agents, wdl_pre_hazard_exit_count, wdl_hazard_exposure_after_cue_count, wdl_delayed_hazard_damage_total",
            ],
            "mem_max_hidden_window_behavior_coupling_supported": [
                "hidden_window_behavior_coupling_summary.json",
                "behavior_coupling_supported",
            ],
            "mem_max_hidden_window_action_shift_summary_json": [
                "hidden_window_behavior_coupling_summary.json:hidden_window_behavior_delta",
                "metrics.csv: wdl_cue_exposed_next_move_away_count, wdl_pre_hazard_exit_count",
                "prediction_demand_action_context compact summaries are emitted but rows_emitted_total is 0 for WDL-only cells",
            ],
            "mem_max_hidden_window_exposure_shift_summary_json": [
                "wdl_timing_damage_summary.json",
                "metrics.csv: wdl_hazard_exposure_after_cue_count, wdl_delayed_hazard_damage_total, wdl_pre_hazard_exit_count",
            ],
            "mem_max_pre_consequence_behavior_delta": [
                "hidden_window_behavior_coupling_summary.json:hidden_window_behavior_delta",
                "computed as wdl_pre_hazard_exit_count + wdl_cue_exposed_next_move_away_count - wdl_hazard_exposure_after_cue_count",
            ],
            "mem_max_initial_memory_dim_histogram": [
                "mem_evo_population_memory_dim_histogram initial row",
                "initial_memory_founder_distribution_summary.json",
            ],
            "mem_max_initial_memory_dim_matched_across_cells": [
                "sim_cfg.lineage_founder_balance_enable",
                "founder_initial_top1_share",
                "initial_distribution_matched",
            ],
            "mem_max_founder_memory_dim_summary_json": [
                "mem_evo_lineage_memory_dim_summary_json",
                "lineage_founder_summary.json",
            ],
            "mem_max_founder_distribution_matched_or_audited": [
                "lineage founder balance config",
                "founder_initial_effective_lineages",
                "founder_initial_top1_share",
            ],
            "mem_max_selected_parent_to_descendant_replacement_supported": [
                "mem_evo_memory_dim_selected_minus_population_mean",
                "mem_evo_memory_dim_descendant_minus_initial_mean",
            ],
            "mem_max_timing_replacement_metric_available": [
                "selected parent snapshot metrics",
                "descendant/final population metrics",
            ],
            "mem_max_gate_input_schema_version": ["mem_max_6_gate_input_schema_v1"],
        },
        "explicit_unavailable_or_fail_closed_fields": {
            "prediction_demand_action_context_event_rows": (
                "0 for final WDL delayed-hazard cells; not used as positive behavior-coupling evidence"
            ),
            "full_hidden_window_event_jsonl": (
                "not emitted in compact profile for this WDL ladder; gate uses WDL metric summaries and fails closed when unsupported"
            ),
        },
        "required_registry_knobs": {
            "prediction_demand_action_context_enable": defaults.get("sim_cfg.prediction_demand_action_context_enable"),
            "prediction_demand_action_context_write_events": defaults.get("sim_cfg.prediction_demand_action_context_write_events"),
            "prediction_demand_action_context_include_lineage": defaults.get("sim_cfg.prediction_demand_action_context_include_lineage"),
            "prediction_demand_action_context_sign_summary_enable": defaults.get(
                "sim_cfg.prediction_demand_action_context_sign_summary_enable"
            ),
            "lineage_founder_balance_enable": defaults.get("sim_cfg.lineage_founder_balance_enable"),
            "enable_experimental_memory_cost_charging": defaults.get("sim_cfg.enable_experimental_memory_cost_charging"),
        },
    }


def validate_registry(registry: Mapping[str, Any] | None = None) -> dict[str, Any]:
    reg = dict(registry or load_registry())
    errors: list[str] = []
    axes = reg.get("axes", {})
    if reg.get("matrix_order") != ["memory_arm", "demand_cell", "seed"]:
        errors.append("matrix_order mismatch")
    if axes.get("memory_arm", {}).get("values") != MEMORY_ARMS:
        errors.append("memory_arm values mismatch")
    if axes.get("demand_cell", {}).get("values") != DEMAND_CELLS:
        errors.append("demand_cell values mismatch")
    if reg.get("seeds") != [101]:
        errors.append("seed policy must stay at one tiny seed for this technical preflight")
    defaults = _knobs(reg.get("cli_defaults", []))
    required_default_knobs = {
        "sim_cfg.prediction_demand_action_context_enable": "true",
        "sim_cfg.prediction_demand_action_context_write_events": "true",
        "sim_cfg.prediction_demand_action_context_include_lineage": "true",
        "sim_cfg.prediction_demand_action_context_sign_summary_enable": "true",
        "sim_cfg.enable_experimental_memory_cost_charging": "true",
        "sim_cfg.lineage_founder_balance_enable": "true",
        "sim_cfg.disable_prediction_error_selection_pressure": "true",
        "sim_cfg.episode0_fitness_weight": "0.0",
        "sim_cfg.episode0_pred_err_fitness_weight": "0.0",
        "sim_cfg.lineage_qd_archive_enable": "false",
        "sim_cfg.checkpoint_bundle_v2_period_gen": "0",
    }
    for key, expected in required_default_knobs.items():
        if defaults.get(key) != expected:
            errors.append(f"{key} expected {expected}, got {defaults.get(key)}")
    memory_flags = axes.get("memory_arm", {}).get("flags", {})
    normal = _flags_knobs(memory_flags.get("memory_normal", []))
    ablated = _flags_knobs(memory_flags.get("memory_ablated", []))
    if normal.get("sim_cfg.force_memory_disabled") != "false" or normal.get("sim_cfg.reset_memory_each_tick") != "false":
        errors.append("memory_normal ablation flags invalid")
    if ablated.get("sim_cfg.force_memory_disabled") != "true" or ablated.get("sim_cfg.reset_memory_each_tick") != "true":
        errors.append("memory_ablated ablation flags invalid")
    planned = len(MEMORY_ARMS) * len(DEMAND_CELLS) * len(reg.get("seeds", []))
    if int((reg.get("intent") or {}).get("planned_runs", -1)) != planned:
        errors.append("planned_runs does not match matrix")
    return {
        "schema_version": "mem_max_6_registry_validation_v1",
        "registry": str(REGISTRY),
        "valid": not errors,
        "errors": errors,
        "planned_runs": planned,
        "memory_arms": MEMORY_ARMS,
        "demand_cells": DEMAND_CELLS,
        "seeds": list(reg.get("seeds", [])),
        "scientific_breadth": "tiny",
    }


def _positive_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    controls = {
        "flat_control_clean": 0.10,
        "cue_observation_off_control": 0.12,
        "hazard_off_control": 0.11,
        "memmax_short_gap": 0.20,
    }
    for demand, selected in controls.items():
        rows.append(
            {
                "memory_arm": "memory_normal",
                "demand_cell": demand,
                "seed": 101,
                "selected_minus_population_final": selected,
                "descendant_minus_initial_final": 0.05,
                "effective_memory_final": 4.0,
                "memory_cost_mean_final": 0.01,
                "lineage_top1_alive_share_final": 0.35,
                "initial_distribution_audited": True,
                "initial_distribution_matched": True,
                "behavior_coupling_metric_available": True,
                "behavior_coupling_supported": True,
                "hidden_window_behavior_delta": 0.05,
            }
        )
    rows.append(
        {
            "memory_arm": "memory_normal",
            "demand_cell": "memmax_useful_gap",
            "seed": 101,
            "selected_minus_population_final": 2.0,
            "descendant_minus_initial_final": 1.0,
            "effective_memory_final": 8.0,
            "memory_cost_mean_final": 0.02,
            "lineage_top1_alive_share_final": 0.35,
            "initial_distribution_audited": True,
            "initial_distribution_matched": True,
            "behavior_coupling_metric_available": True,
            "behavior_coupling_supported": True,
            "hidden_window_behavior_delta": 0.50,
        }
    )
    for demand in DEMAND_CELLS:
        rows.append(
            {
                "memory_arm": "memory_ablated",
                "demand_cell": demand,
                "seed": 101,
                "selected_minus_population_final": 0.0,
                "descendant_minus_initial_final": 0.0,
                "effective_memory_final": 0.0,
                "memory_cost_mean_final": 0.0,
                "lineage_top1_alive_share_final": 0.35,
                "initial_distribution_audited": True,
                "initial_distribution_matched": True,
                "behavior_coupling_metric_available": True,
                "behavior_coupling_supported": True,
                "hidden_window_behavior_delta": 0.0,
            }
        )
    return rows


def _mutated_rows(kind: str) -> list[dict[str, Any]]:
    rows = _positive_rows()
    for row in rows:
        if kind == "missing_behavior":
            row["behavior_coupling_metric_available"] = False
            row["behavior_coupling_supported"] = False
        if kind == "unmatched_initial":
            row["initial_distribution_matched"] = False
        if kind == "selected_parent_only" and row["memory_arm"] == "memory_normal" and row["demand_cell"] == "memmax_useful_gap":
            row["descendant_minus_initial_final"] = 0.0
        if kind == "ablated_stronger" and row["memory_arm"] == "memory_ablated" and row["demand_cell"] == "memmax_useful_gap":
            row["selected_minus_population_final"] = 3.0
        if kind == "lineage_top1" and row["memory_arm"] == "memory_normal" and row["demand_cell"] == "memmax_useful_gap":
            row["lineage_top1_alive_share_final"] = 0.95
    return rows


def gate_evaluator_fixture_results() -> dict[str, Any]:
    helper = _mem4()
    cases = {
        "all_gates_pass": _positive_rows(),
        "missing_behavior_coupling": _mutated_rows("missing_behavior"),
        "unmatched_initial_distribution": _mutated_rows("unmatched_initial"),
        "selected_parent_only": _mutated_rows("selected_parent_only"),
        "ablated_useful_gap_stronger": _mutated_rows("ablated_stronger"),
        "top_lineage_dominance": _mutated_rows("lineage_top1"),
        "mem_max_2_recheck": helper.normalize_mem_max_2_rows(),
        "mem_max_3_recheck": helper.normalize_mem_max_3_rows(),
    }
    full: dict[str, Any] = {}
    summary: dict[str, Any] = {}
    for name, rows in cases.items():
        evaluation = helper.evaluate_rows(rows, dataset_id=f"mem_max_6_{name}")
        full[name] = evaluation
        summary[name] = {
            "overall_status": evaluation["overall_status"],
            "blocking_gate_count": evaluation["blocking_gate_count"],
            "positive_memory_demand_interpretation_supported": evaluation[
                "positive_memory_demand_interpretation_supported"
            ],
        }
    return {
        "schema_version": "mem_max_6_gate_evaluator_fixture_results_v2",
        "status": "passed" if summary["all_gates_pass"]["overall_status"] == "pass" and all(
            summary[name]["overall_status"] == "fail_closed" for name in summary if name != "all_gates_pass"
        ) else "failed",
        "summary": summary,
        "full_evaluations": full,
    }


def dry_run_summary_from_root(runs_root: Path | None) -> dict[str, Any]:
    if runs_root is None or not runs_root.exists():
        return {
            "schema_version": "mem_max_6_repaired_ladder_dry_run_summary_v2",
            "status": "not_run",
            "dry_run_valid": False,
        }
    summaries = sorted(runs_root.rglob("summary.json"), key=lambda p: p.stat().st_mtime)
    summary_path = summaries[-1] if summaries else None
    if summary_path is None:
        return {
            "schema_version": "mem_max_6_repaired_ladder_dry_run_summary_v2",
            "status": "missing_summary",
            "dry_run_valid": False,
            "runs_root": str(runs_root),
        }
    payload = read_json(summary_path)
    matrix_path = summary_path.parent / "matrix.csv"
    rows: list[dict[str, str]] = []
    if matrix_path.exists():
        with matrix_path.open(newline="", encoding="utf-8") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
    required = {
        ("memory_normal", "memmax_useful_gap"),
        ("memory_ablated", "memmax_useful_gap"),
        ("memory_normal", "flat_control_clean"),
        ("memory_ablated", "flat_control_clean"),
        ("memory_normal", "hazard_off_control"),
        ("memory_ablated", "hazard_off_control"),
    }
    observed = {(row.get("memory_arm", ""), row.get("demand_cell", "")) for row in rows}
    return {
        "schema_version": "mem_max_6_repaired_ladder_dry_run_summary_v2",
        "status": str(payload.get("status", payload.get("dry_run_status", "")) or ""),
        "dry_run_valid": len(rows) == 10 and required.issubset(observed),
        "summary_path": str(summary_path),
        "matrix_path": str(matrix_path),
        "planned_runs": len(rows),
        "required_representative_cells_present": required.issubset(observed),
        "storage_preflight": payload.get("storage_preflight", {}),
        "registry": payload.get("registry", {}),
        "runs_root": str(runs_root),
    }


RUN_ID_RE = re.compile(r"^mem_max_6__memory_(?P<memory_arm>.+)__demand_(?P<demand_cell>.+)__seed(?P<seed>\d+)$")
REPORTING_ARTIFACTS = [
    "prediction_demand_action_context_generation_summary.jsonl",
    "prediction_demand_action_context_run_summary.json",
    "prediction_demand_action_context_sign_summary.json",
]


def _parallel_summary(root: Path | None) -> dict[str, Any]:
    if root is None or not root.exists():
        return {"status": "not_run", "root": str(root) if root else ""}
    path = root / "parallel_campaign_summary.json"
    if not path.exists():
        return {"status": "missing_summary", "root": str(root)}
    payload = read_json(path)
    results = list(payload.get("results", []))
    ok = [
        row
        for row in results
        if row.get("status") == "ok" and int(row.get("returncode") if row.get("returncode") is not None else 1) == 0
    ]
    return {
        "status": "ok" if len(ok) == int(payload.get("planned_runs", 0) or 0) and not payload.get("failures") else "failed",
        "summary_path": str(path),
        "root": str(root),
        "planned_runs": int(payload.get("planned_runs", 0) or 0),
        "ok_runs": len(ok),
        "failure_count": len(payload.get("failures", [])),
        "total_wall_s": payload.get("total_wall_s"),
        "workers": payload.get("workers"),
        "results": results,
    }


def repeat_parity_summary(repeat_a_root: Path | None, repeat_b_root: Path | None) -> dict[str, Any]:
    a = _parallel_summary(repeat_a_root)
    b = _parallel_summary(repeat_b_root)
    if a["status"] != "ok" or b["status"] != "ok":
        return {
            "schema_version": "mem_max_6_repeat_parity_summary_v1",
            "status": "not_passed",
            "repeat_a": {k: v for k, v in a.items() if k != "results"},
            "repeat_b": {k: v for k, v in b.items() if k != "results"},
            "parity_pass": False,
            "mismatches": ["repeat summary missing or failed"],
        }
    a_by_id = {str(row["run_id"]): row for row in a["results"]}
    b_by_id = {str(row["run_id"]): row for row in b["results"]}
    mismatches: list[str] = []
    rows: list[dict[str, Any]] = []
    if set(a_by_id) != set(b_by_id):
        mismatches.append("run_id sets differ")
    for run_id in sorted(set(a_by_id) & set(b_by_id)):
        a_dir = Path(a_by_id[run_id]["run_dir"])
        b_dir = Path(b_by_id[run_id]["run_dir"])
        a_metrics = _sha256(a_dir / "metrics.csv")
        b_metrics = _sha256(b_dir / "metrics.csv")
        artifact_hashes_a: dict[str, str] = {}
        artifact_hashes_b: dict[str, str] = {}
        for name in REPORTING_ARTIFACTS:
            artifact_hashes_a[name] = _sha256(a_dir / "prediction_demand_action_context" / name)
            artifact_hashes_b[name] = _sha256(b_dir / "prediction_demand_action_context" / name)
        if a_metrics != b_metrics:
            mismatches.append(f"{run_id}: metrics.csv sha256 mismatch")
        if artifact_hashes_a != artifact_hashes_b:
            mismatches.append(f"{run_id}: reporting artifact sha256 mismatch")
        rows.append(
            {
                "run_id": run_id,
                "metrics_sha256": a_metrics,
                "metrics_sha256_repeat_b": b_metrics,
                "reporting_artifact_sha256_parity": artifact_hashes_a == artifact_hashes_b,
                "final_metrics": a_by_id[run_id].get("final_metrics", {}),
            }
        )
    return {
        "schema_version": "mem_max_6_repeat_parity_summary_v1",
        "status": "passed" if not mismatches else "failed",
        "parity_pass": not mismatches,
        "repeat_a": {k: v for k, v in a.items() if k != "results"},
        "repeat_b": {k: v for k, v in b.items() if k != "results"},
        "run_count": len(rows),
        "mismatches": mismatches,
        "rows": rows,
    }


def _parse_run_id(run_id: str) -> dict[str, Any]:
    match = RUN_ID_RE.match(run_id)
    if not match:
        return {"memory_arm": "", "demand_cell": "", "seed": 0}
    return {
        "memory_arm": match.group("memory_arm"),
        "demand_cell": match.group("demand_cell"),
        "seed": int(match.group("seed")),
    }


def _final_metrics(run_dir: Path) -> dict[str, str]:
    with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    return rows[-1] if rows else {}


def _pdac_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "prediction_demand_action_context" / "prediction_demand_action_context_sign_summary.json"
    return read_json(path) if path.exists() else {}


def preflight_rows_from_root(preflight_root: Path | None) -> list[dict[str, Any]]:
    summary = _parallel_summary(preflight_root)
    if summary["status"] != "ok":
        return []
    rows: list[dict[str, Any]] = []
    for result in summary["results"]:
        run_id = str(result.get("run_id", ""))
        axes = _parse_run_id(run_id)
        run_dir = Path(result["run_dir"])
        final = _final_metrics(run_dir)
        pdac = _pdac_summary(run_dir)
        founder_groups = _float(final, "founder_group_count", 0.0)
        expected_top1 = (1.0 / founder_groups) if founder_groups > 0 else 0.0
        founder_top1 = _float(final, "founder_initial_top1_share", 0.0)
        founder_violations = _float(final, "founder_balance_violation_count", 1.0)
        wdl_events = _float(final, "wdl_delayed_hazard_event_count", 0.0)
        cue_seen = _float(final, "wdl_cue_seen_agents", 0.0)
        pre_exit = _float(final, "wdl_pre_hazard_exit_count", 0.0)
        move_away = _float(final, "wdl_cue_exposed_next_move_away_count", 0.0)
        hazard_after = _float(final, "wdl_hazard_exposure_after_cue_count", 0.0)
        behavior_available = wdl_events > 0 and cue_seen > 0
        behavior_delta = pre_exit + move_away - hazard_after
        rows.append(
            {
                **axes,
                "run_id": run_id,
                "status": result.get("status"),
                "selected_minus_population_final": _float(final, "mem_evo_memory_dim_selected_minus_population_mean"),
                "descendant_minus_initial_final": _float(final, "mem_evo_memory_dim_descendant_minus_initial_mean"),
                "effective_memory_final": _float(final, "effective_memory_mean"),
                "memory_cost_mean_final": _float(final, "memory_cost_mean"),
                "memory_ablation_active_flag_final": _float(final, "memory_ablation_active_flag"),
                "memory_reset_each_tick_active_flag_final": _float(final, "memory_reset_each_tick_active_flag"),
                "lineage_top1_alive_share_final": _float(final, "lineage_top1_alive_share"),
                "initial_distribution_audited": _bool(final, "founder_balance_enable"),
                "initial_distribution_matched": founder_violations == 0 and abs(founder_top1 - expected_top1) < 1.0e-9,
                "behavior_coupling_metric_available": behavior_available,
                "behavior_coupling_supported": behavior_available and behavior_delta > 0,
                "hidden_window_behavior_delta": behavior_delta,
                "wdl_delayed_hazard_event_count": wdl_events,
                "wdl_cue_seen_agents": cue_seen,
                "wdl_pre_hazard_exit_count": pre_exit,
                "wdl_hazard_exposure_after_cue_count": hazard_after,
                "wdl_delayed_hazard_damage_total": _float(final, "wdl_delayed_hazard_damage_total"),
                "pdac_rows_emitted_total": int(pdac.get("rows_emitted_total", 0) or 0),
                "read_only_reporting": bool(pdac.get("read_only", False)),
                "reward_no_intervention_audit": pdac.get("reward_no_intervention_audit", {}),
            }
        )
    return sorted(rows, key=lambda row: (MEMORY_ARMS.index(row["memory_arm"]), DEMAND_CELLS.index(row["demand_cell"])))


def gate_evaluation_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "schema_version": "mem_max_6_gate_evaluation_summary_v1",
            "status": "not_run",
            "overall_status": "not_run",
            "blocking_gate_count": None,
        }
    helper = _mem4()
    evaluation = helper.evaluate_rows(rows, dataset_id="mem_max_6_tiny_preflight")
    return {
        "schema_version": "mem_max_6_gate_evaluation_summary_v1",
        "status": "evaluated",
        **evaluation,
    }


def _csv_scalar_rows(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    return [{key: row.get(key, "") for key in keys} for row in rows]


def write_preflight_outputs(out_dir: Path, rows: list[dict[str, Any]], gate: Mapping[str, Any]) -> None:
    scalar_keys = [
        "memory_arm",
        "demand_cell",
        "seed",
        "run_id",
        "selected_minus_population_final",
        "descendant_minus_initial_final",
        "effective_memory_final",
        "memory_cost_mean_final",
        "lineage_top1_alive_share_final",
        "initial_distribution_audited",
        "initial_distribution_matched",
        "behavior_coupling_metric_available",
        "behavior_coupling_supported",
        "hidden_window_behavior_delta",
    ]
    write_json(out_dir / "memory_demand_ladder_summary.json", rows)
    write_csv(out_dir / "memory_demand_ladder_summary.csv", _csv_scalar_rows(rows, scalar_keys), scalar_keys)
    write_json(out_dir / "mem_evo_shift_summary.json", rows)
    write_csv(out_dir / "mem_evo_shift_summary.csv", _csv_scalar_rows(rows, scalar_keys[:8]), scalar_keys[:8])
    write_json(out_dir / "cost_effective_memory_context.json", rows)
    cost_keys = [
        "memory_arm",
        "demand_cell",
        "seed",
        "effective_memory_final",
        "memory_cost_mean_final",
        "memory_ablation_active_flag_final",
        "memory_reset_each_tick_active_flag_final",
    ]
    write_csv(out_dir / "cost_effective_memory_context.csv", _csv_scalar_rows(rows, cost_keys), cost_keys)
    lineage_keys = [
        "memory_arm",
        "demand_cell",
        "seed",
        "lineage_top1_alive_share_final",
        "initial_distribution_audited",
        "initial_distribution_matched",
    ]
    write_json(out_dir / "lineage_founder_summary.json", rows)
    write_csv(out_dir / "lineage_founder_summary.csv", _csv_scalar_rows(rows, lineage_keys), lineage_keys)
    write_json(out_dir / "initial_memory_founder_distribution_summary.json", rows)
    write_csv(out_dir / "initial_memory_founder_distribution_summary.csv", _csv_scalar_rows(rows, lineage_keys), lineage_keys)
    behavior_keys = [
        "memory_arm",
        "demand_cell",
        "seed",
        "behavior_coupling_metric_available",
        "behavior_coupling_supported",
        "hidden_window_behavior_delta",
        "pdac_rows_emitted_total",
        "read_only_reporting",
    ]
    write_json(out_dir / "hidden_window_behavior_coupling_summary.json", rows)
    write_csv(out_dir / "hidden_window_behavior_coupling_summary.csv", _csv_scalar_rows(rows, behavior_keys), behavior_keys)
    wdl_keys = [
        "memory_arm",
        "demand_cell",
        "seed",
        "wdl_delayed_hazard_event_count",
        "wdl_cue_seen_agents",
        "wdl_pre_hazard_exit_count",
        "wdl_hazard_exposure_after_cue_count",
        "wdl_delayed_hazard_damage_total",
    ]
    write_json(out_dir / "wdl_timing_damage_summary.json", rows)
    write_csv(out_dir / "wdl_timing_damage_summary.csv", _csv_scalar_rows(rows, wdl_keys), wdl_keys)
    write_json(out_dir / "gate_evaluation_summary.json", gate)


def write_default_outputs(
    out_dir: Path = OUT_DIR,
    *,
    dry_run_root: Path | None = None,
    repeat_a_root: Path | None = None,
    repeat_b_root: Path | None = None,
    preflight_root: Path | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    registry_validation = validate_registry()
    field_map = reporting_repair_field_map()
    fixtures = gate_evaluator_fixture_results()
    dry = dry_run_summary_from_root(dry_run_root)
    repeat = repeat_parity_summary(repeat_a_root, repeat_b_root)
    preflight = _parallel_summary(preflight_root)
    rows = preflight_rows_from_root(preflight_root)
    gate = gate_evaluation_from_rows(rows)
    if rows:
        write_preflight_outputs(out_dir, rows, gate)
    else:
        empty_fields = ["memory_arm", "demand_cell", "seed", "run_id"]
        for name in (
            "memory_demand_ladder_summary",
            "mem_evo_shift_summary",
            "cost_effective_memory_context",
            "lineage_founder_summary",
            "initial_memory_founder_distribution_summary",
            "hidden_window_behavior_coupling_summary",
            "wdl_timing_damage_summary",
        ):
            write_json(out_dir / f"{name}.json", [])
            write_csv(out_dir / f"{name}.csv", [], empty_fields)
        write_json(out_dir / "gate_evaluation_summary.json", gate)
    matrix_rows = rows if rows else []
    write_csv(
        out_dir / "run_matrix.csv",
        _csv_scalar_rows(matrix_rows, ["memory_arm", "demand_cell", "seed", "run_id", "status"]),
        ["memory_arm", "demand_cell", "seed", "run_id", "status"],
    )
    reporting_ok = registry_validation["valid"] and fixtures["status"] == "passed"
    dry_ok = bool(dry.get("dry_run_valid"))
    repeat_ok = bool(repeat.get("parity_pass"))
    preflight_ok = preflight.get("status") == "ok"
    gate_pass = gate.get("overall_status") == "pass"
    if not reporting_ok or not dry_ok:
        final_decision = "MEM-MAX-6C"
        status = "reporting_or_dry_run_insufficient"
        recommendation = "Repair reporting or dry-run matrix before another preflight."
        win_loss = "Loss for interpretation readiness because reporting or dry-run coverage is incomplete."
    elif not repeat_ok:
        final_decision = "MEM-MAX-6C"
        status = "repeat_parity_failed_or_missing"
        recommendation = "Repair repeat parity before another preflight."
        win_loss = "Win for reporting repair; loss for preflight readiness because repeat parity did not pass."
    elif not preflight_ok:
        final_decision = "MEM-MAX-6C"
        status = "preflight_missing_or_failed"
        recommendation = "Repair tiny preflight execution before interpreting the ladder."
        win_loss = "Win for reporting repair and repeat parity; loss because the preflight did not complete cleanly."
    elif gate_pass:
        final_decision = "MEM-MAX-6A"
        status = "preflight_clean_and_gate_passed"
        recommendation = "Proceed to the next narrow memory-demand design step."
        win_loss = "Win for reporting repair and tiny ladder gate support."
    else:
        final_decision = "MEM-MAX-6B"
        status = "preflight_clean_but_ladder_failed_interpretation_gates"
        recommendation = "Run a narrow MEM-MAX follow-up focused on demand-loop calibration and behavior-coupling reporting."
        win_loss = "Win for executable reporting repair; loss for the repaired ladder result because the tiny preflight failed interpretation gates."
    decision = {
        "schema_version": "mem_max_6_decision_v2",
        "decision": final_decision,
        "stage_reached": "stage_6_tiny_preflight_gate_evaluation" if preflight_ok else "stage_5_repeat_parity",
        "status": status,
        "reporting_repair_completed": reporting_ok,
        "dry_run_registry_created": True,
        "dry_run_valid": dry_ok,
        "repeat_parity_run": repeat_a_root is not None or repeat_b_root is not None,
        "repeat_parity_pass": repeat_ok,
        "preflight_run": preflight_root is not None,
        "preflight_clean": preflight_ok,
        "gate_overall_status": gate.get("overall_status"),
        "gate_blocking_count": gate.get("blocking_gate_count"),
        "scientific_campaign_run": False,
        "recommended_next_step": recommendation,
        "win_loss": win_loss,
        "no_claims_statement": NO_CLAIMS,
    }
    completion = {
        "schema_version": "mem_max_6_completion_summary_v1",
        "decision": final_decision,
        "reporting_repair_completed": reporting_ok,
        "dry_run_valid": dry_ok,
        "repeat_parity_pass": repeat_ok,
        "preflight_clean": preflight_ok,
        "gate_overall_status": gate.get("overall_status"),
        "planned_preflight_runs": preflight.get("planned_runs", 0),
        "ok_preflight_runs": preflight.get("ok_runs", 0),
        "raw_runtime_roots": {
            "dry_run": str(dry_run_root) if dry_run_root else "",
            "repeat_a": str(repeat_a_root) if repeat_a_root else "",
            "repeat_b": str(repeat_b_root) if repeat_b_root else "",
            "tiny_preflight": str(preflight_root) if preflight_root else "",
        },
    }
    write_json(out_dir / "reporting_repair_field_map.json", field_map)
    write_json(out_dir / "gate_evaluator_fixture_results.json", fixtures)
    write_json(out_dir / "repaired_ladder_dry_run_summary.json", dry)
    write_json(out_dir / "repeat_parity_summary.json", repeat)
    write_json(out_dir / "completion_summary.json", completion)
    write_json(out_dir / "mem_max_6_decision.json", decision)
    return {
        "decision": decision,
        "reporting_repair_field_map": field_map,
        "gate_evaluator_fixture_results": fixtures,
        "repaired_ladder_dry_run_summary": dry,
        "repeat_parity_summary": repeat,
        "completion_summary": completion,
        "gate_evaluation_summary": gate,
        "registry_validation": registry_validation,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--dry-run-root", default="")
    parser.add_argument("--repeat-a-root", default="")
    parser.add_argument("--repeat-b-root", default="")
    parser.add_argument("--preflight-root", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    dry_root = Path(args.dry_run_root) if str(args.dry_run_root).strip() else None
    repeat_a_root = Path(args.repeat_a_root) if str(args.repeat_a_root).strip() else None
    repeat_b_root = Path(args.repeat_b_root) if str(args.repeat_b_root).strip() else None
    preflight_root = Path(args.preflight_root) if str(args.preflight_root).strip() else None
    payload = write_default_outputs(
        Path(args.out_dir),
        dry_run_root=dry_root,
        repeat_a_root=repeat_a_root,
        repeat_b_root=repeat_b_root,
        preflight_root=preflight_root,
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            json.dumps(
                {
                    "decision": payload["decision"]["decision"],
                    "reporting_repair_completed": payload["decision"]["reporting_repair_completed"],
                    "fixture_status": payload["gate_evaluator_fixture_results"]["status"],
                    "dry_run_valid": payload["decision"]["dry_run_valid"],
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
