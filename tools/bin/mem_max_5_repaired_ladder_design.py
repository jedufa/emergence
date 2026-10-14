#!/usr/bin/env python3
from __future__ import annotations

"""Generate the MEM-MAX-5 repaired ladder design package.

This is reporting-only design work. It does not launch simulations, create a
scientific registry, or change reward/selection/reproduction/memory mechanics.
"""

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "docs/900_lifecycle/940_incoming/mem_max_5_repaired_ladder_design"
MEM4_HELPER = PROJECT_ROOT / "tools/bin/mem_memory_demand_interpretation_gate.py"
MEM4_DIR = PROJECT_ROOT / "docs/900_lifecycle/940_incoming/mem_max_4_memory_demand_interpretation_gate"

DEMAND_CELLS = [
    "flat_control_clean",
    "cue_observation_off_control",
    "hazard_off_control",
    "memmax_short_gap",
    "memmax_useful_gap",
    "memmax_too_long_gap",
]
MEMORY_ARMS = ["memory_normal", "memory_ablated"]
SEED = 101


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_mem4_helper():
    spec = importlib.util.spec_from_file_location("mem_memory_demand_interpretation_gate", MEM4_HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import MEM-MAX-4 helper: {MEM4_HELPER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gate_requirement_mapping() -> dict[str, Any]:
    return {
        "schema_version": "mem_max_5_gate_requirement_mapping_v1",
        "source_contract": str(MEM4_DIR / "mem_max_4_gate_contract.json"),
        "positive_interpretation_rule": (
            "Selected-parent genome.memory_dim pressure is diagnostic only. A positive memory-demand "
            "interpretation requires every MEM-MAX-4 blocking gate to pass."
        ),
        "gates": [
            {
                "gate_id": "effective_memory_gate",
                "planned_outputs": [
                    "effective_memory_final",
                    "memory_ablation_active_flag_final",
                    "memory_reset_each_tick_active_flag_final",
                    "population_memory_dim_final",
                ],
                "healthy_ladder_expectation": (
                    "memory_normal rows report nonzero behavior-facing effective memory; memory_ablated rows "
                    "report effective memory suppressed to zero."
                ),
                "fail_closed_if": [
                    "normal effective memory is missing or zero",
                    "ablated effective memory is nonzero",
                    "ablated selected-parent memory_dim pressure is positive while effective memory is zero",
                ],
            },
            {
                "gate_id": "descendant_movement_gate",
                "planned_outputs": [
                    "selected_minus_population_final",
                    "descendant_minus_initial_final",
                    "population_memory_dim_final",
                    "descendant_memory_dim_final",
                ],
                "healthy_ladder_expectation": (
                    "useful-gap selected-parent pressure propagates into descendants or final population memory_dim."
                ),
                "fail_closed_if": [
                    "selected-parent pressure is positive but descendant/final movement is missing or nonpositive",
                    "only selected-parent snapshots move",
                ],
            },
            {
                "gate_id": "control_specificity_gate",
                "planned_outputs": [
                    "selected_minus_population_final by arm and demand_cell",
                    "difference_in_differences",
                    "normal useful-minus-flat",
                    "ablated useful-minus-flat",
                ],
                "healthy_ladder_expectation": (
                    "useful-gap normal exceeds flat, cue-off, hazard-off, short-gap, too-long-gap, and useful-gap "
                    "ablated controls; difference-in-differences is positive."
                ),
                "fail_closed_if": [
                    "any normal control matches or exceeds useful-gap normal",
                    "useful-gap ablated matches or exceeds useful-gap normal",
                    "difference-in-differences is not positive",
                ],
            },
            {
                "gate_id": "behavior_coupling_gate",
                "planned_outputs": [
                    "behavior_coupling_metric_available",
                    "behavior_coupling_supported",
                    "hidden_window_behavior_delta",
                    "post_cue_pre_hazard_action_delta",
                    "hazard_exposure_delta",
                    "hazard_damage_delta",
                    "cue_exposure_count",
                    "active_without_prior_cue_count",
                ],
                "healthy_ladder_expectation": (
                    "hidden-window action/exposure/damage metrics are present and show useful-gap normal behavior "
                    "moving in the expected direction relative to controls."
                ),
                "fail_closed_if": [
                    "hidden-window behavior metrics are missing",
                    "metrics are present but do not move in the expected direction",
                    "damage/exposure changes can be explained by cue-off or hazard-off controls",
                ],
            },
            {
                "gate_id": "founder_lineage_gate",
                "planned_outputs": [
                    "initial_memory_dim_histogram_by_cell",
                    "initial_founder_memory_dim_by_cell",
                    "initial_distribution_matched",
                    "initial_distribution_audited",
                    "lineage_top1_alive_share_final",
                ],
                "healthy_ladder_expectation": (
                    "initial memory_dim spectra are matched or audited across cells, and top-lineage share stays "
                    "below the MEM-MAX-4 fail threshold."
                ),
                "fail_closed_if": [
                    "initial memory_dim distributions are not audited",
                    "audited distributions are not matched",
                    "top lineage alive share is >= 0.90",
                ],
            },
            {
                "gate_id": "cost_gate",
                "planned_outputs": [
                    "memory_cost_mean_final",
                    "memory_cost_total_final",
                    "effective_memory_final",
                ],
                "healthy_ladder_expectation": (
                    "normal rows report nonzero memory cost when making costly-memory claims; ablated rows report "
                    "zero effective-memory cost."
                ),
                "fail_closed_if": [
                    "normal memory cost is missing",
                    "normal memory cost is zero while claiming costly memory",
                ],
            },
            {
                "gate_id": "timing_replacement_gate",
                "planned_outputs": [
                    "selected_minus_population_final",
                    "descendant_minus_initial_final",
                    "generation_of_selected_parent_snapshot",
                    "generation_of_descendant_snapshot",
                    "population_memory_dim_final",
                ],
                "healthy_ladder_expectation": (
                    "selected-parent pressure is measured early enough to test whether it propagates into later "
                    "descendant/final population movement."
                ),
                "fail_closed_if": [
                    "selected-parent pressure is not followed by descendant/final movement",
                    "snapshot timing makes replacement impossible to interpret",
                ],
            },
        ],
    }


def _cell(demand_cell: str, memory_arm: str) -> dict[str, Any]:
    expected_by_demand = {
        "flat_control_clean": "no WDL-specific memory demand should appear",
        "cue_observation_off_control": "cue-retention memory should not help because the cue is not observable",
        "hazard_off_control": "cue-only memory should not pay rent because the hazard has no consequence",
        "memmax_short_gap": "short timing may be handled by current-state or near-term policy behavior",
        "memmax_useful_gap": "only this cell may support positive memory-demand interpretation if every gate passes",
        "memmax_too_long_gap": "too-long timing should not dominate useful-gap unless the ladder is miscalibrated",
    }
    invalidates_by_demand = {
        "flat_control_clean": "flat control matches or exceeds useful-gap normal pressure",
        "cue_observation_off_control": "observation-off control matches or exceeds useful-gap normal pressure",
        "hazard_off_control": "hazard-off control matches or exceeds useful-gap normal pressure",
        "memmax_short_gap": "short-gap control dominates useful-gap normal pressure",
        "memmax_useful_gap": "useful-gap normal fails any MEM-MAX-4 gate or ablated useful-gap matches/exceeds it",
        "memmax_too_long_gap": "too-long control dominates useful-gap normal without behavior-coupled support",
    }
    contributes = {
        "flat_control_clean": ["control_specificity_gate"],
        "cue_observation_off_control": ["control_specificity_gate", "behavior_coupling_gate"],
        "hazard_off_control": ["control_specificity_gate", "behavior_coupling_gate"],
        "memmax_short_gap": ["control_specificity_gate", "timing_replacement_gate"],
        "memmax_useful_gap": [
            "effective_memory_gate",
            "descendant_movement_gate",
            "control_specificity_gate",
            "behavior_coupling_gate",
            "founder_lineage_gate",
            "cost_gate",
            "timing_replacement_gate",
        ],
        "memmax_too_long_gap": ["control_specificity_gate", "timing_replacement_gate"],
    }
    return {
        "cell_id": f"{memory_arm}__{demand_cell}__seed{SEED}",
        "memory_arm": memory_arm,
        "demand_cell": demand_cell,
        "seed": SEED,
        "expected_memory_demand_interpretation": expected_by_demand[demand_cell],
        "gate_inputs_contributed": contributes[demand_cell],
        "healthy_ladder_expected_result": (
            "normal useful-gap may be positive only with matching descendant, behavior, lineage, cost, and control "
            "support; all other cells should stay below useful-gap normal"
        ),
        "invalidates_ladder_if": invalidates_by_demand[demand_cell],
        "mechanics_change_required": False,
        "scientific_run_authorized": False,
    }


def repaired_ladder_design() -> dict[str, Any]:
    cells = [_cell(demand_cell, memory_arm) for memory_arm in MEMORY_ARMS for demand_cell in DEMAND_CELLS]
    return {
        "schema_version": "mem_max_5_repaired_ladder_design_v1",
        "status": "designed_reporting_repair_required",
        "decision_implication": "MEM-MAX-5B",
        "scientific_registry_created": False,
        "scientific_campaign_or_preflight_authorized": False,
        "planned_runs_if_repaired_reporting_exists": len(cells),
        "axes": {
            "memory_arm": MEMORY_ARMS,
            "demand_cell": DEMAND_CELLS,
            "seed": [SEED],
        },
        "constraints": {
            "direct_memory_reward": False,
            "direct_cognition_reward": False,
            "diversity_reward": False,
            "lineage_rescue": False,
            "protected_reproduction": False,
            "archive_reseeding": False,
            "survival_subsidy": False,
            "mechanics_changes_in_mem_max_5": False,
        },
        "design_changes_vs_mem_max_1": [
            "selected-parent memory_dim is downgraded to diagnostic-only",
            "all MEM-MAX-4 gate fields are mandatory for future interpretation",
            "founder/memory_dim initial distributions must be matched or audited before any run",
            "hidden-window action/exposure behavior reporting is a preflight prerequisite",
            "descendant/final replacement movement is a hard gate",
            "difference-in-differences against ablated useful-gap and normal controls is a hard gate",
        ],
        "cells": cells,
    }


def behavior_coupling_requirement_table() -> dict[str, Any]:
    return {
        "schema_version": "mem_max_5_behavior_coupling_requirement_table_v1",
        "current_reporting_status": "blocked_for_positive_interpretation",
        "reason": (
            "MEM-MAX-3 reported timing/damage visibility but behavior_coupling_metric_available=false for all cells."
        ),
        "required_before_mem_max_6_preflight": True,
        "metrics": [
            {
                "metric_id": "post_cue_pre_hazard_action_delta",
                "description": "movement/action change after cue disappearance and before hazard activation",
                "positive_requirement": "useful-gap normal changes in the expected direction versus controls",
                "missing_field_result": "fail_closed",
            },
            {
                "metric_id": "refuge_or_safe_zone_arrival_delta",
                "description": "arrival, shelter, cover, or avoidance behavior if available in the substrate",
                "positive_requirement": "useful-gap normal improves arrival/avoidance without cue-off or hazard-off matching it",
                "missing_field_result": "fail_closed_or_not_applicable_with_explicit_reason",
            },
            {
                "metric_id": "hazard_exposure_and_damage_delta",
                "description": "active hazard exposure, damage, and prevented-damage context",
                "positive_requirement": "memory-normal useful-gap reduces harmful exposure/damage relative to controls",
                "missing_field_result": "fail_closed",
            },
            {
                "metric_id": "cue_exposure_and_hidden_gap_timing",
                "description": "cue exposure count, active exposure count, event timing, and hidden gap",
                "positive_requirement": "behavior comparison is restricted to agents/cells with meaningful cue-to-active timing",
                "missing_field_result": "fail_closed",
            },
            {
                "metric_id": "active_without_prior_cue",
                "description": "active hazard exposure when no prior cue was observed, if observable",
                "positive_requirement": "separates memory demand from contemporaneous hazard response",
                "missing_field_result": "downgrade_or_fail_closed",
            },
            {
                "metric_id": "memory_activation_context",
                "description": "effective memory, ablation/reset flags, and memory cost at the same timing boundary",
                "positive_requirement": "behavior-coupling rows align with effective-memory rows",
                "missing_field_result": "fail_closed",
            },
        ],
        "implementation_plan": [
            "Add or expose per-cell hidden-window behavior summaries in the MEM-MAX extraction/reporting layer.",
            "Use these summaries to populate behavior_coupling_metric_available and behavior_coupling_supported.",
            "Keep behavior metrics observational only; do not add reward shaping or policy mechanics.",
            "Run gate-evaluator fixtures before any scientific MEM-MAX-6 preflight.",
        ],
    }


def founder_initial_distribution_control_plan() -> dict[str, Any]:
    return {
        "schema_version": "mem_max_5_founder_initial_distribution_control_plan_v1",
        "purpose": "prevent founder memory_dim skew from faking selection for memory",
        "allowed_controls": [
            "matched initial memory_dim histogram across all planned cells",
            "explicit per-cell initial distribution audit with fail-closed threshold",
            "founder-stratified reporting if already available",
        ],
        "not_allowed": [
            "lineage rescue",
            "founder survival protection",
            "archive reseeding",
            "protected reproduction",
        ],
        "planned_fields": [
            "initial_distribution_audited",
            "initial_distribution_matched",
            "initial_memory_dim_histogram_by_cell",
            "initial_founder_memory_dim_by_cell",
            "lineage_top1_alive_share_final",
            "top_lineage_memory_dim_initial",
            "top_lineage_memory_dim_final",
        ],
        "fail_closed_thresholds": {
            "lineage_top1_alive_share_final": ">= 0.90 fails or downgrades",
            "initial_distribution_audited": "false fails",
            "initial_distribution_matched": "false fails unless a documented stratified analysis replaces it",
        },
    }


def next_preflight_acceptance_criteria() -> dict[str, Any]:
    return {
        "schema_version": "mem_max_5_next_preflight_acceptance_criteria_v1",
        "status": "preflight_not_authorized_until_reporting_repair",
        "next_allowed_task": "MEM-MAX-6 tiny technical preflight only after reporting repair is implemented and fixture-gated",
        "criteria": [
            "MEM-MAX-4 evaluator must accept a synthetic all-gates-pass fixture.",
            "MEM-MAX-4 evaluator must fail closed on missing behavior-coupling fields.",
            "MEM-MAX-2 and MEM-MAX-3 style failures must remain fail_closed.",
            "Future run outputs must include every field required by the seven MEM-MAX-4 gates.",
            "Initial memory_dim distributions must be matched or explicitly audited before interpretation.",
            "Hidden-window action/exposure behavior metrics must be present before any positive interpretation.",
            "Useful-gap normal must beat all normal controls and useful-gap ablated with positive DID.",
            "Selected-parent pressure must propagate to descendant/final movement.",
            "Normal memory cost and effective memory must be nonzero for costly-memory claims.",
        ],
        "not_allowed": [
            "scientific campaign from MEM-MAX-5",
            "direct memory reward",
            "direct cognition reward",
            "diversity reward",
            "lineage rescue",
            "protected reproduction",
            "archive reseeding",
            "survival subsidy",
        ],
    }


def _fixture_rows(*, behavior_available: bool = True) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    control_selected = {
        "flat_control_clean": 0.10,
        "cue_observation_off_control": 0.12,
        "hazard_off_control": 0.11,
        "memmax_short_gap": 0.20,
        "memmax_too_long_gap": 0.15,
    }
    for demand, selected in control_selected.items():
        rows.append(
            {
                "memory_arm": "memory_normal",
                "demand_cell": demand,
                "seed": SEED,
                "selected_minus_population_final": selected,
                "descendant_minus_initial_final": 0.05,
                "effective_memory_final": 4.0,
                "memory_cost_mean_final": 0.01,
                "lineage_top1_alive_share_final": 0.35,
                "initial_distribution_audited": True,
                "initial_distribution_matched": True,
                "behavior_coupling_metric_available": behavior_available,
                "behavior_coupling_supported": behavior_available,
                "hidden_window_behavior_delta": 0.05 if behavior_available else None,
            }
        )
    rows.append(
        {
            "memory_arm": "memory_normal",
            "demand_cell": "memmax_useful_gap",
            "seed": SEED,
            "selected_minus_population_final": 2.0,
            "descendant_minus_initial_final": 1.0,
            "effective_memory_final": 8.0,
            "memory_cost_mean_final": 0.02,
            "lineage_top1_alive_share_final": 0.34,
            "initial_distribution_audited": True,
            "initial_distribution_matched": True,
            "behavior_coupling_metric_available": behavior_available,
            "behavior_coupling_supported": behavior_available,
            "hidden_window_behavior_delta": 0.5 if behavior_available else None,
        }
    )
    for demand in DEMAND_CELLS:
        rows.append(
            {
                "memory_arm": "memory_ablated",
                "demand_cell": demand,
                "seed": SEED,
                "selected_minus_population_final": 0.0,
                "descendant_minus_initial_final": 0.0,
                "effective_memory_final": 0.0,
                "memory_cost_mean_final": 0.0,
                "lineage_top1_alive_share_final": 0.34,
                "initial_distribution_audited": True,
                "initial_distribution_matched": True,
                "behavior_coupling_metric_available": behavior_available,
                "behavior_coupling_supported": behavior_available,
                "hidden_window_behavior_delta": 0.0 if behavior_available else None,
            }
        )
    return rows


def gate_evaluator_fixture_results() -> dict[str, Any]:
    helper = load_mem4_helper()
    synthetic_pass = helper.evaluate_rows(_fixture_rows(behavior_available=True), dataset_id="mem_max_5_synthetic_all_gates_pass")
    missing_behavior = helper.evaluate_rows(
        _fixture_rows(behavior_available=False),
        dataset_id="mem_max_5_missing_behavior_coupling_fixture",
    )
    mem2_eval = helper.evaluate_rows(helper.normalize_mem_max_2_rows(), dataset_id="mem_max_2")
    mem3_eval = helper.evaluate_rows(helper.normalize_mem_max_3_rows(), dataset_id="mem_max_3")
    return {
        "schema_version": "mem_max_5_gate_evaluator_fixture_results_v1",
        "evaluator": str(MEM4_HELPER),
        "fixtures": {
            "synthetic_all_gates_pass": {
                "overall_status": synthetic_pass["overall_status"],
                "blocking_gate_count": synthetic_pass["blocking_gate_count"],
                "positive_memory_demand_interpretation_supported": synthetic_pass[
                    "positive_memory_demand_interpretation_supported"
                ],
            },
            "missing_behavior_coupling": {
                "overall_status": missing_behavior["overall_status"],
                "blocking_gate_count": missing_behavior["blocking_gate_count"],
                "behavior_gate_status": _gate_status(missing_behavior, "behavior_coupling_gate"),
            },
            "mem_max_2_recheck": {
                "overall_status": mem2_eval["overall_status"],
                "blocking_gate_count": mem2_eval["blocking_gate_count"],
            },
            "mem_max_3_recheck": {
                "overall_status": mem3_eval["overall_status"],
                "blocking_gate_count": mem3_eval["blocking_gate_count"],
            },
        },
        "full_evaluations": {
            "synthetic_all_gates_pass": synthetic_pass,
            "missing_behavior_coupling": missing_behavior,
            "mem_max_2_recheck": mem2_eval,
            "mem_max_3_recheck": mem3_eval,
        },
    }


def _gate_status(evaluation: Mapping[str, Any], gate_id: str) -> str:
    for gate in evaluation["gates"]:
        if gate["gate_id"] == gate_id:
            return str(gate["status"])
    return "missing"


def decision_payload(fixture_results: Mapping[str, Any]) -> dict[str, Any]:
    fixtures = fixture_results["fixtures"]
    reporting_blockers = [
        "hidden-window behavior/action/exposure coupling fields are not present in MEM-MAX-2/3 outputs",
        "future extraction must populate behavior_coupling_metric_available and behavior_coupling_supported",
        "future extraction must report matched or audited initial memory_dim distributions before interpretation",
    ]
    return {
        "schema_version": "mem_max_5_decision_v1",
        "decision": "MEM-MAX-5B",
        "win_loss": "Win for repaired ladder design and gate harness; no scientific memory-demand preflight authorized.",
        "repaired_ladder_designed": True,
        "scientific_registry_created": False,
        "dry_run_registry_created": False,
        "scientific_campaign_or_preflight_run": False,
        "gate_inputs_planned": True,
        "gate_inputs_currently_available_for_future_run": False,
        "reporting_blockers_before_mem_max_6": reporting_blockers,
        "fixture_summary": fixtures,
        "recommended_next_step": (
            "Implement a bounded MEM-MAX reporting repair for hidden-window behavior coupling and initial "
            "memory_dim/founder distribution outputs, then run one tiny MEM-MAX-6 technical preflight."
        ),
        "no_claims_statement": no_claims_statement(),
    }


def no_claims_statement() -> str:
    return (
        "MEM-MAX-5 does not show intelligence, emergence, consciousness, planning, culture, prediction "
        "selection, general cognition, broad adaptive memory, learned refuge use, evolved refuge strategy, "
        "superintelligence, MEM-EVO-A, or valid memory-demand calibration. It does not add direct cognition "
        "reward, memory reward, diversity reward, lineage rescue, protected reproduction, archive reseeding, "
        "or survival subsidy. It designs a repaired memory-demand ladder wired to the MEM-MAX-4 fail-closed "
        "gate contract before any further MEM-MAX preflight."
    )


def write_default_outputs(out_dir: Path = OUT_DIR) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping = gate_requirement_mapping()
    ladder = repaired_ladder_design()
    behavior = behavior_coupling_requirement_table()
    founder = founder_initial_distribution_control_plan()
    fixtures = gate_evaluator_fixture_results()
    criteria = next_preflight_acceptance_criteria()
    decision = decision_payload(fixtures)

    write_json(out_dir / "gate_requirement_mapping.json", mapping)
    write_json(out_dir / "repaired_ladder_design.json", ladder)
    write_json(out_dir / "behavior_coupling_requirement_table.json", behavior)
    write_json(out_dir / "founder_initial_distribution_control_plan.json", founder)
    write_json(out_dir / "gate_evaluator_fixture_results.json", fixtures)
    write_json(out_dir / "next_preflight_acceptance_criteria.json", criteria)
    write_json(out_dir / "mem_max_5_decision.json", decision)
    return {
        "decision": decision,
        "gate_requirement_mapping": mapping,
        "repaired_ladder_design": ladder,
        "behavior_coupling_requirement_table": behavior,
        "founder_initial_distribution_control_plan": founder,
        "gate_evaluator_fixture_results": fixtures,
        "next_preflight_acceptance_criteria": criteria,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--json", action="store_true", help="print full generated payload")
    args = parser.parse_args()
    payload = write_default_outputs(Path(args.out_dir))
    decision = payload["decision"]
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            json.dumps(
                {
                    "decision": decision["decision"],
                    "dry_run_registry_created": decision["dry_run_registry_created"],
                    "gate_inputs_planned": decision["gate_inputs_planned"],
                    "gate_inputs_currently_available_for_future_run": decision[
                        "gate_inputs_currently_available_for_future_run"
                    ],
                    "synthetic_fixture": decision["fixture_summary"]["synthetic_all_gates_pass"]["overall_status"],
                    "missing_behavior_fixture": decision["fixture_summary"]["missing_behavior_coupling"][
                        "overall_status"
                    ],
                    "mem_max_2_recheck": decision["fixture_summary"]["mem_max_2_recheck"]["overall_status"],
                    "mem_max_3_recheck": decision["fixture_summary"]["mem_max_3_recheck"]["overall_status"],
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
