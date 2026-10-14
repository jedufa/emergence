#!/usr/bin/env python3
from __future__ import annotations

"""Fail-closed interpretation gates for memory-demand evidence.

This helper is reporting-only. It consumes summary rows from MEM-MAX/MEM-EVO
style reports or synthetic fixtures and emits per-gate pass/fail/unknown
records. It does not run simulations or change mechanics.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "docs/900_lifecycle/940_incoming/mem_max_4_memory_demand_interpretation_gate"
MEM_MAX_2_DIR = PROJECT_ROOT / "docs/900_lifecycle/940_incoming/mem_max_2_tiny_memory_demand_preflight"
MEM_MAX_3_DIR = PROJECT_ROOT / "docs/900_lifecycle/940_incoming/mem_max_3_ladder_control_root_cause"

DEMAND_ORDER = [
    "flat_control_clean",
    "cue_observation_off_control",
    "hazard_off_control",
    "memmax_short_gap",
    "memmax_useful_gap",
    "memmax_too_long_gap",
]
CONTROL_DEMANDS = [
    "flat_control_clean",
    "cue_observation_off_control",
    "hazard_off_control",
    "memmax_short_gap",
    "memmax_too_long_gap",
]
MEMORY_ORDER = ["memory_normal", "memory_ablated"]
EPS = 1.0e-9
LINEAGE_TOP1_FAIL_THRESHOLD = 0.90


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _f(row: Mapping[str, Any], key: str, default: float | None = 0.0) -> float | None:
    try:
        value = row.get(key, default)
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def _b(row: Mapping[str, Any], key: str, default: bool = False) -> bool:
    value = row.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def gate_contract() -> dict[str, Any]:
    return {
        "schema_version": "mem_memory_demand_interpretation_gate_contract_v1",
        "positive_interpretation_rule": (
            "Selected-parent genome.memory_dim pressure is diagnostic only unless all blocking gates pass: "
            "effective memory, descendant movement, control specificity, behavior coupling, founder/lineage, "
            "cost, and timing/replacement."
        ),
        "lineage_top1_fail_threshold": LINEAGE_TOP1_FAIL_THRESHOLD,
        "gates": [
            {
                "gate_id": "effective_memory_gate",
                "required": True,
                "fail_closed_rule": "Normal cells need behavior-facing effective memory, ablated cells need effective memory suppressed, and ablated genome memory_dim pressure cannot support a positive interpretation.",
            },
            {
                "gate_id": "descendant_movement_gate",
                "required": True,
                "fail_closed_rule": "Selected-parent pressure alone is insufficient; descendant or final population memory_dim must move in the same direction.",
            },
            {
                "gate_id": "control_specificity_gate",
                "required": True,
                "fail_closed_rule": "Useful-gap normal must exceed flat, cue-off, hazard-off, short-gap, too-long-gap, and useful-gap ablated controls; difference-in-differences must be positive.",
            },
            {
                "gate_id": "behavior_coupling_gate",
                "required": True,
                "fail_closed_rule": "Hidden-window action/exposure behavior metrics must be available and move in the expected direction.",
            },
            {
                "gate_id": "founder_lineage_gate",
                "required": True,
                "fail_closed_rule": "Initial memory distributions must be matched or audited, and top-lineage dominance or one-founder concentration must fail or downgrade interpretation.",
            },
            {
                "gate_id": "cost_gate",
                "required": True,
                "fail_closed_rule": "Costly-memory claims require nonzero reported memory cost in normal cells.",
            },
            {
                "gate_id": "timing_replacement_gate",
                "required": True,
                "fail_closed_rule": "If selected-parent pressure does not propagate to descendants or final population, treat the result as timing/horizon/metric unresolved.",
            },
        ],
    }


def interpretation_gate_schema() -> dict[str, Any]:
    return {
        "schema_version": "mem_memory_demand_interpretation_gate_schema_v1",
        "required_row_fields": [
            "memory_arm",
            "demand_cell",
            "selected_minus_population_final",
            "descendant_minus_initial_final",
            "effective_memory_final",
            "memory_cost_mean_final",
            "lineage_top1_alive_share_final",
        ],
        "optional_row_fields": [
            "initial_distribution_audited",
            "initial_distribution_matched",
            "behavior_coupling_metric_available",
            "behavior_coupling_supported",
            "hidden_window_behavior_delta",
            "population_memory_dim_final",
            "parent_memory_dim_final",
            "memory_ablation_active_flag_final",
            "memory_reset_each_tick_active_flag_final",
        ],
        "gate_result_fields": [
            "gate_id",
            "status",
            "severity",
            "blocking",
            "reason",
            "recommended_next_action",
            "observed",
        ],
    }


def _row_key(row: Mapping[str, Any]) -> tuple[int, int, int]:
    memory = str(row.get("memory_arm", "memory_normal"))
    demand = str(row.get("demand_cell", "flat_control_clean"))
    seed = int(float(row.get("seed", 0) or 0))
    return (
        MEMORY_ORDER.index(memory) if memory in MEMORY_ORDER else 99,
        DEMAND_ORDER.index(demand) if demand in DEMAND_ORDER else 99,
        seed,
    )


def _index(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    out: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        out[(str(row.get("memory_arm", "")), str(row.get("demand_cell", "")))] = row
    return out


def _gate(
    gate_id: str,
    status: str,
    reason: str,
    recommended_next_action: str,
    observed: Mapping[str, Any],
    *,
    severity: str | None = None,
    blocking: bool | None = None,
) -> dict[str, Any]:
    if severity is None:
        severity = "info" if status == "pass" else ("blocking" if status == "fail" else "unknown")
    if blocking is None:
        blocking = status != "pass"
    return {
        "gate_id": gate_id,
        "status": status,
        "severity": severity,
        "blocking": bool(blocking),
        "reason": reason,
        "recommended_next_action": recommended_next_action,
        "observed": dict(observed),
    }


def normalize_mem_max_2_rows(root: Path = MEM_MAX_2_DIR) -> list[dict[str, Any]]:
    ladder = read_json(root / "memory_demand_ladder_summary.json")
    cost = _index(read_json(root / "cost_effective_memory_context.json"))
    lineage = _index(read_json(root / "lineage_confidence_summary.json"))
    out: list[dict[str, Any]] = []
    for row in ladder:
        key = (str(row["memory_arm"]), str(row["demand_cell"]))
        cost_row = cost.get(key, {})
        lineage_row = lineage.get(key, {})
        out.append(
            {
                "memory_arm": row["memory_arm"],
                "demand_cell": row["demand_cell"],
                "seed": int(row.get("seed", 0) or 0),
                "selected_minus_population_final": _f(row, "mem_evo_memory_dim_selected_minus_population_mean_final"),
                "descendant_minus_initial_final": _f(row, "mem_evo_memory_dim_descendant_minus_initial_mean_final"),
                "effective_memory_final": _f(row, "effective_memory_mean_final"),
                "memory_cost_mean_final": _f(cost_row, "memory_cost_mean_final"),
                "memory_ablation_active_flag_final": _f(cost_row, "memory_ablation_active_flag_final"),
                "memory_reset_each_tick_active_flag_final": _f(cost_row, "memory_reset_each_tick_active_flag_final"),
                "lineage_top1_alive_share_final": _f(lineage_row, "lineage_top1_alive_share_final"),
                "initial_distribution_audited": False,
                "initial_distribution_matched": False,
                "behavior_coupling_metric_available": False,
                "behavior_coupling_supported": False,
            }
        )
    return sorted(out, key=_row_key)


def normalize_mem_max_3_rows(root: Path = MEM_MAX_3_DIR) -> list[dict[str, Any]]:
    timing = _index(read_json(root / "selected_parent_vs_descendant_timing.json"))
    cost = _index(read_json(root / "cost_effective_memory_control_check.json"))
    lineage = _index(read_json(root / "founder_lineage_decomposition.json"))
    behavior = _index(read_json(root / "wdl_behavior_coupling_check.json"))
    out: list[dict[str, Any]] = []
    for key in sorted(set(timing) | set(cost) | set(lineage) | set(behavior)):
        timing_row = timing.get(key, {})
        cost_row = cost.get(key, {})
        lineage_row = lineage.get(key, {})
        behavior_row = behavior.get(key, {})
        out.append(
            {
                "memory_arm": key[0],
                "demand_cell": key[1],
                "seed": int(timing_row.get("seed", cost_row.get("seed", 0)) or 0),
                "selected_minus_population_final": _f(timing_row, "selected_minus_population_final"),
                "descendant_minus_initial_final": _f(timing_row, "descendant_minus_initial_final"),
                "effective_memory_final": _f(cost_row, "effective_memory_final"),
                "memory_cost_mean_final": _f(cost_row, "memory_cost_mean_final"),
                "memory_ablation_active_flag_final": _f(cost_row, "memory_ablation_active_flag_final"),
                "memory_reset_each_tick_active_flag_final": _f(cost_row, "memory_reset_each_tick_active_flag_final"),
                "lineage_top1_alive_share_final": _f(lineage_row, "lineage_top1_alive_share_final"),
                "initial_distribution_audited": True,
                "initial_distribution_matched": False,
                "behavior_coupling_metric_available": _b(behavior_row, "behavior_coupling_metric_available"),
                "behavior_coupling_supported": _b(behavior_row, "behavior_coupling_supported"),
                "population_memory_dim_final": _f(timing_row, "population_memory_dim_final"),
                "parent_memory_dim_final": _f(timing_row, "parent_memory_dim_final"),
            }
        )
    return sorted(out, key=_row_key)


def evaluate_rows(rows: list[Mapping[str, Any]], *, dataset_id: str) -> dict[str, Any]:
    rows = sorted([dict(row) for row in rows], key=_row_key)
    idx = _index(rows)
    useful_normal = idx.get(("memory_normal", "memmax_useful_gap"), {})
    useful_ablated = idx.get(("memory_ablated", "memmax_useful_gap"), {})
    flat_normal = idx.get(("memory_normal", "flat_control_clean"), {})
    flat_ablated = idx.get(("memory_ablated", "flat_control_clean"), {})
    useful_sel = float(_f(useful_normal, "selected_minus_population_final", 0.0) or 0.0)
    useful_desc = float(_f(useful_normal, "descendant_minus_initial_final", 0.0) or 0.0)
    useful_eff = float(_f(useful_normal, "effective_memory_final", 0.0) or 0.0)

    gates: list[dict[str, Any]] = []

    normal_effectives = [
        float(_f(row, "effective_memory_final", 0.0) or 0.0)
        for row in rows
        if row.get("memory_arm") == "memory_normal"
    ]
    ablated_effectives = [
        float(_f(row, "effective_memory_final", 0.0) or 0.0)
        for row in rows
        if row.get("memory_arm") == "memory_ablated"
    ]
    ablated_selected_positive = [
        row
        for row in rows
        if row.get("memory_arm") == "memory_ablated"
        and float(_f(row, "selected_minus_population_final", 0.0) or 0.0) > EPS
    ]
    if not normal_effectives or max(normal_effectives) <= EPS:
        gates.append(
            _gate(
                "effective_memory_gate",
                "fail",
                "memory_normal rows do not show nonzero behavior-facing effective memory.",
                "Repair effective-memory reporting before interpreting memory demand.",
                {"normal_effective_memory_max": max(normal_effectives) if normal_effectives else None},
            )
        )
    elif ablated_effectives and max(ablated_effectives) > EPS:
        gates.append(
            _gate(
                "effective_memory_gate",
                "fail",
                "memory_ablated rows did not suppress effective memory.",
                "Repair ablation controls before interpreting memory demand.",
                {"ablated_effective_memory_max": max(ablated_effectives)},
            )
        )
    elif ablated_selected_positive:
        gates.append(
            _gate(
                "effective_memory_gate",
                "fail",
                "Ablated cells show genome memory_dim pressure while effective memory is zero; this cannot count as behavior-facing memory pressure.",
                "Demote ablated genome memory_dim pressure to invalid-control evidence.",
                {
                    "ablated_effective_memory_max": max(ablated_effectives) if ablated_effectives else None,
                    "ablated_positive_selected_pressure_cells": [
                        str(row.get("demand_cell")) for row in ablated_selected_positive
                    ],
                },
            )
        )
    else:
        gates.append(
            _gate(
                "effective_memory_gate",
                "pass",
                "Normal effective memory is nonzero and ablated effective memory is suppressed without ablated pressure.",
                "Keep effective-memory checks in future reports.",
                {
                    "normal_effective_memory_max": max(normal_effectives),
                    "ablated_effective_memory_max": max(ablated_effectives) if ablated_effectives else None,
                },
                severity="info",
                blocking=False,
            )
        )

    if useful_sel > EPS and useful_desc <= EPS:
        gates.append(
            _gate(
                "descendant_movement_gate",
                "fail",
                "Useful-gap selected-parent memory_dim pressure did not propagate to descendant/final population movement.",
                "Treat selected-parent pressure as weak/uninterpretable until descendant movement is present.",
                {"useful_selected_minus_population": useful_sel, "useful_descendant_minus_initial": useful_desc},
            )
        )
    elif useful_sel > EPS and useful_desc > EPS:
        gates.append(
            _gate(
                "descendant_movement_gate",
                "pass",
                "Useful-gap descendant/final movement is positive in the same direction as selected-parent pressure.",
                "Retain descendant movement as a mandatory positive-evidence gate.",
                {"useful_selected_minus_population": useful_sel, "useful_descendant_minus_initial": useful_desc},
                severity="info",
                blocking=False,
            )
        )
    else:
        gates.append(
            _gate(
                "descendant_movement_gate",
                "unknown",
                "Useful-gap selected-parent pressure is absent or missing, so descendant movement cannot support a positive interpretation.",
                "Collect complete selected-parent and descendant movement summaries.",
                {"useful_selected_minus_population": useful_sel, "useful_descendant_minus_initial": useful_desc},
            )
        )

    normal_control_pressures = {
        demand: float(_f(idx.get(("memory_normal", demand), {}), "selected_minus_population_final", 0.0) or 0.0)
        for demand in CONTROL_DEMANDS
    }
    ablated_useful_sel = float(_f(useful_ablated, "selected_minus_population_final", 0.0) or 0.0)
    normal_useful_minus_flat = useful_sel - float(_f(flat_normal, "selected_minus_population_final", 0.0) or 0.0)
    ablated_useful_minus_flat = ablated_useful_sel - float(
        _f(flat_ablated, "selected_minus_population_final", 0.0) or 0.0
    )
    did = normal_useful_minus_flat - ablated_useful_minus_flat
    bad_controls = {k: v for k, v in normal_control_pressures.items() if v >= useful_sel - EPS}
    if ablated_useful_sel >= useful_sel - EPS or bad_controls or did <= EPS:
        gates.append(
            _gate(
                "control_specificity_gate",
                "fail",
                "Useful-gap normal memory_dim pressure is not specific to useful behavior-facing memory demand.",
                "Fail closed if ablated useful-gap or any normal control matches/exceeds useful-gap normal, or if difference-in-differences is not positive.",
                {
                    "useful_normal_selected_minus_population": useful_sel,
                    "useful_ablated_selected_minus_population": ablated_useful_sel,
                    "normal_controls_matching_or_exceeding_useful": bad_controls,
                    "normal_useful_minus_flat": normal_useful_minus_flat,
                    "ablated_useful_minus_flat": ablated_useful_minus_flat,
                    "difference_in_differences": did,
                },
            )
        )
    else:
        gates.append(
            _gate(
                "control_specificity_gate",
                "pass",
                "Useful-gap normal exceeds ablated and normal controls with positive difference-in-differences.",
                "Keep difference-in-differences and control dominance as hard gates.",
                {
                    "useful_normal_selected_minus_population": useful_sel,
                    "useful_ablated_selected_minus_population": ablated_useful_sel,
                    "normal_controls": normal_control_pressures,
                    "difference_in_differences": did,
                },
                severity="info",
                blocking=False,
            )
        )

    behavior_available = bool(_b(useful_normal, "behavior_coupling_metric_available"))
    behavior_supported = bool(_b(useful_normal, "behavior_coupling_supported"))
    if not behavior_available:
        gates.append(
            _gate(
                "behavior_coupling_gate",
                "unknown",
                "Hidden-window behavior-coupling metrics are missing, so memory_dim pressure cannot count as memory demand.",
                "Add hidden-window action/exposure movement reporting before another MEM-MAX preflight.",
                {"behavior_coupling_metric_available": False, "behavior_coupling_supported": False},
            )
        )
    elif not behavior_supported:
        gates.append(
            _gate(
                "behavior_coupling_gate",
                "fail",
                "Behavior-coupling metrics are present but do not move in the expected direction.",
                "Repair demand loop or policy/action interface before interpreting memory_dim pressure.",
                {
                    "behavior_coupling_metric_available": True,
                    "behavior_coupling_supported": False,
                    "hidden_window_behavior_delta": _f(useful_normal, "hidden_window_behavior_delta", None),
                },
            )
        )
    else:
        gates.append(
            _gate(
                "behavior_coupling_gate",
                "pass",
                "Hidden-window action/exposure metrics support behavior-coupled memory demand.",
                "Keep behavior-coupling as a mandatory positive-evidence gate.",
                {
                    "behavior_coupling_metric_available": True,
                    "behavior_coupling_supported": True,
                    "hidden_window_behavior_delta": _f(useful_normal, "hidden_window_behavior_delta", None),
                },
                severity="info",
                blocking=False,
            )
        )

    useful_top1 = float(_f(useful_normal, "lineage_top1_alive_share_final", 0.0) or 0.0)
    initial_audited = bool(_b(useful_normal, "initial_distribution_audited"))
    initial_matched = bool(_b(useful_normal, "initial_distribution_matched"))
    if useful_top1 >= LINEAGE_TOP1_FAIL_THRESHOLD:
        gates.append(
            _gate(
                "founder_lineage_gate",
                "fail",
                "Useful-gap normal is top-lineage dominated.",
                "Repair founder/memory_dim matching and lineage contribution reporting.",
                {
                    "useful_top1_lineage_share": useful_top1,
                    "initial_distribution_audited": initial_audited,
                    "initial_distribution_matched": initial_matched,
                },
            )
        )
    elif not initial_audited:
        gates.append(
            _gate(
                "founder_lineage_gate",
                "unknown",
                "Initial memory_dim/founder distributions were not audited.",
                "Audit or enforce matched founder memory_dim spectra across cells before interpretation.",
                {
                    "useful_top1_lineage_share": useful_top1,
                    "initial_distribution_audited": initial_audited,
                    "initial_distribution_matched": initial_matched,
                },
            )
        )
    elif not initial_matched:
        gates.append(
            _gate(
                "founder_lineage_gate",
                "fail",
                "Initial distributions were audited but not matched enough to support a positive interpretation.",
                "Use matched founder memory_dim spectra or downgrade the result.",
                {
                    "useful_top1_lineage_share": useful_top1,
                    "initial_distribution_audited": initial_audited,
                    "initial_distribution_matched": initial_matched,
                },
            )
        )
    else:
        gates.append(
            _gate(
                "founder_lineage_gate",
                "pass",
                "Lineage dominance is below threshold and initial memory_dim/founder distributions are matched.",
                "Keep founder and lineage audits as hard interpretation gates.",
                {
                    "useful_top1_lineage_share": useful_top1,
                    "initial_distribution_audited": initial_audited,
                    "initial_distribution_matched": initial_matched,
                },
                severity="info",
                blocking=False,
            )
        )

    normal_costs = [
        float(_f(row, "memory_cost_mean_final", 0.0) or 0.0)
        for row in rows
        if row.get("memory_arm") == "memory_normal"
    ]
    if not normal_costs:
        gates.append(
            _gate(
                "cost_gate",
                "unknown",
                "Memory cost fields are missing for normal cells.",
                "Report memory cost before interpreting costly-memory evolution.",
                {"normal_memory_cost_count": 0},
            )
        )
    elif max(normal_costs) <= EPS:
        gates.append(
            _gate(
                "cost_gate",
                "fail",
                "Normal cells report zero memory cost, so costly-memory claims are unsupported.",
                "Enable or report nonzero memory cost before MEM-EVO-A-style interpretation.",
                {"normal_memory_cost_max": max(normal_costs)},
            )
        )
    else:
        gates.append(
            _gate(
                "cost_gate",
                "pass",
                "Normal cells report nonzero memory cost.",
                "Keep cost reporting as a mandatory claim boundary.",
                {"normal_memory_cost_max": max(normal_costs), "normal_memory_cost_min": min(normal_costs)},
                severity="info",
                blocking=False,
            )
        )

    if useful_sel > EPS and useful_desc <= EPS:
        gates.append(
            _gate(
                "timing_replacement_gate",
                "fail",
                "Selected-parent pressure did not propagate to descendants/final population.",
                "Audit metric timing or horizon before interpreting selection pressure.",
                {"useful_selected_minus_population": useful_sel, "useful_descendant_minus_initial": useful_desc},
            )
        )
    elif useful_sel > EPS and useful_desc > EPS:
        gates.append(
            _gate(
                "timing_replacement_gate",
                "pass",
                "Selected-parent pressure has descendant/final replacement support.",
                "Keep replacement timing as a hard gate.",
                {"useful_selected_minus_population": useful_sel, "useful_descendant_minus_initial": useful_desc},
                severity="info",
                blocking=False,
            )
        )
    else:
        gates.append(
            _gate(
                "timing_replacement_gate",
                "unknown",
                "Selected-parent or descendant fields are missing or nonpositive.",
                "Collect selected-parent and descendant/final movement fields.",
                {"useful_selected_minus_population": useful_sel, "useful_descendant_minus_initial": useful_desc},
            )
        )

    blocking_gates = [gate for gate in gates if gate["blocking"]]
    positive_supported = not blocking_gates
    return {
        "schema_version": "mem_memory_demand_interpretation_gate_evaluation_v1",
        "dataset_id": dataset_id,
        "row_count": len(rows),
        "gate_count": len(gates),
        "blocking_gate_count": len(blocking_gates),
        "overall_status": "pass" if positive_supported else "fail_closed",
        "positive_memory_demand_interpretation_supported": positive_supported,
        "gates": gates,
        "summary": {
            "useful_normal_selected_minus_population": useful_sel,
            "useful_normal_effective_memory": useful_eff,
            "useful_normal_descendant_minus_initial": useful_desc,
            "useful_ablated_selected_minus_population": ablated_useful_sel,
            "normal_useful_minus_flat": normal_useful_minus_flat,
            "ablated_useful_minus_flat": ablated_useful_minus_flat,
            "difference_in_differences": did,
        },
        "row_excerpt": rows,
    }


def next_ladder_acceptance_criteria() -> dict[str, Any]:
    return {
        "schema_version": "mem_max_4_next_ladder_acceptance_criteria_v1",
        "status": "dry_run_only_future_gate_specification",
        "scientific_registry_created": False,
        "criteria": [
            "Gate evaluator must run on planned summary fixtures and fail closed on missing behavior-coupling fields.",
            "Ablated useful-gap selected-parent genome memory_dim pressure must be lower than useful-gap normal and treated as invalid-control evidence if positive with effective memory zero.",
            "Useful-gap normal must exceed flat, cue-off, hazard-off, short-gap, too-long-gap, and useful-gap ablated controls.",
            "Difference-in-differences must be positive: (normal useful-minus-flat) - (ablated useful-minus-flat) > 0.",
            "Descendant/final population memory_dim movement must be positive when selected-parent pressure is positive.",
            "Hidden-window behavior/action/exposure metrics must be present and move in the expected direction.",
            "Founder/memory_dim distributions must be matched or explicitly audited before interpretation.",
            "Top-lineage dominance at or above 0.90 must fail or downgrade interpretation.",
            "Normal memory cost must be nonzero and reported for costly-memory claims.",
        ],
        "not_allowed": [
            "No scientific preflight or campaign is authorized by MEM-MAX-4.",
            "No direct memory/cognition reward.",
            "No lineage rescue, protected reproduction, archive reseeding, or survival subsidy.",
        ],
    }


def decision_payload(mem2_eval: Mapping[str, Any], mem3_eval: Mapping[str, Any]) -> dict[str, Any]:
    implemented = {
        "effective_memory_gate": True,
        "descendant_movement_gate": True,
        "control_specificity_gate": True,
        "behavior_coupling_gate": True,
        "founder_lineage_gate": True,
        "cost_gate": True,
        "timing_replacement_gate": True,
    }
    return {
        "schema_version": "mem_max_4_decision_v1",
        "decision": "MEM-MAX-4A",
        "win_loss": "Win for interpretation-gate repair; no memory-demand calibration win claimed.",
        "implemented_gates": implemented,
        "mem_max_2_overall_status": mem2_eval.get("overall_status"),
        "mem_max_3_overall_status": mem3_eval.get("overall_status"),
        "revised_scientific_ladder_created": False,
        "next_ladder_acceptance_criteria_created": True,
        "recommended_next_step": "Design MEM-MAX-5 repaired ladder only after the new gate contract is wired into future dry-run/reporting plans.",
        "no_claims_statement": (
            "MEM-MAX-4 does not show intelligence, emergence, consciousness, planning, culture, "
            "prediction selection, general cognition, broad adaptive memory, learned refuge use, "
            "evolved refuge strategy, superintelligence, MEM-EVO-A, or valid memory-demand calibration. "
            "It does not add direct cognition reward, memory reward, diversity reward, lineage rescue, "
            "protected reproduction, archive reseeding, or survival subsidy. It repairs interpretation "
            "gates so future memory-demand calibration results cannot be misread from selected-parent "
            "genome memory_dim pressure alone."
        ),
    }


def write_default_outputs(out_dir: Path = OUT_DIR) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    contract = gate_contract()
    schema = interpretation_gate_schema()
    mem2_eval = evaluate_rows(normalize_mem_max_2_rows(MEM_MAX_2_DIR), dataset_id="mem_max_2")
    mem3_eval = evaluate_rows(normalize_mem_max_3_rows(MEM_MAX_3_DIR), dataset_id="mem_max_3")
    decision = decision_payload(mem2_eval, mem3_eval)
    criteria = next_ladder_acceptance_criteria()
    write_json(out_dir / "mem_max_4_gate_contract.json", contract)
    write_json(out_dir / "interpretation_gate_schema.json", schema)
    write_json(out_dir / "mem_max_4_gate_evaluation_mem_max_2.json", mem2_eval)
    write_json(out_dir / "mem_max_4_gate_evaluation_mem_max_3.json", mem3_eval)
    write_json(out_dir / "next_ladder_acceptance_criteria.json", criteria)
    write_json(out_dir / "mem_max_4_decision.json", decision)
    return {
        "decision": decision,
        "mem_max_2": mem2_eval,
        "mem_max_3": mem3_eval,
        "contract": contract,
        "schema": schema,
        "criteria": criteria,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--json", action="store_true", help="print full payload")
    args = parser.parse_args()
    payload = write_default_outputs(Path(args.out_dir))
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            json.dumps(
                {
                    "decision": payload["decision"]["decision"],
                    "mem_max_2_status": payload["mem_max_2"]["overall_status"],
                    "mem_max_3_status": payload["mem_max_3"]["overall_status"],
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
