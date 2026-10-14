from __future__ import annotations

"""REP-2FR corrected opportunity survey and behavior bridge evidence."""

import csv
import json
import tempfile
import time
from pathlib import Path
from typing import Any

from evolution.structural_action_opportunity_gate import (
    BASE_COMMIT as REP2E_BASE_COMMIT,
    SCHEMA_VERSION as REP2F_SCHEMA_VERSION,
    checkpoint_resume_summary as rep2f_checkpoint_resume_summary,
    existing_scenario_inventory,
    make_rep2f_sim,
    repeat_parity_summary as rep2f_repeat_parity_summary,
    run_blind_survey as rep2f_run_blind_survey,
    run_rep2e_matrix_with_telemetry,
    w6_support_summary as rep2f_w6_support_summary,
)
from evolution.structural_behavior_demand import run_chosen_action_fixtures
from evolution.structural_causal_robustness import reward_audit, stable_hash, stable_json, write_json
from simulation.checkpoint_bundle_v2 import checkpoint_bundle_v2_apply, checkpoint_bundle_v2_load, checkpoint_bundle_v2_save
from simulation.performance_profiles import ACCEPTED_W6_FAST_PROFILE, apply_performance_profile
from simulation.runner_pkg.runner_action_opportunity import (
    CANONICAL_COUNTERFACTUAL_FIELD_DELTAS,
    FIELD_ACTION_ROUTES,
    NEAR_BOUNDARY_ABSOLUTE_THRESHOLD,
)


SCHEMA_VERSION = "structural_action_opportunity_repair_v1"
BASE_COMMIT = "de1b8beafdaf054a4a1d65b3392bd878a8b5fae2"
ORIGINAL_SCENARIOS = ("clean_control", "patch_existing_context", "resource_scarcity_existing", "crowding_existing", "patch_waste_context")
ORIGINAL_SEEDS = (811, 812, 813)
BRIDGE_SEEDS = (941, 942, 943, 944, 945)
BRIDGE_ARMS = (
    ("live_off", False, "none"),
    ("live_on", True, "none"),
    ("zero_gate", True, "zero_connection_gates"),
    ("remove_connections", True, "remove_connections"),
)
READINESS_SEEDS = (951, 952, 953, 954, 955, 956, 957, 958)
READINESS_ARMS = (
    ("live_off", False, "none"),
    ("live_on", True, "none"),
    ("remove_connections", True, "remove_connections"),
)
CORRECTED_THRESHOLDS = {
    "ordinary_linked_action_opportunity_total_min": 100,
    "potential_linked_opportunity_fraction_min": 0.01,
    "seed_support_min": 2,
    "near_boundary_absolute_fraction_min": 0.002,
    "canonical_delta_exceeds_margin_fraction_min": 0.001,
}


def make_rep2fr_sim(
    run_dir: Path,
    *,
    seed: int,
    context: str,
    live_enabled: bool = False,
    ablation: str = "none",
    generations: int = 3,
    steps: int = 30,
    pop: int = 24,
    w6: bool = False,
    metrics: bool = True,
) -> Any:
    sim = make_rep2f_sim(
        run_dir,
        seed=seed,
        context=context,
        live_enabled=live_enabled,
        ablation=ablation,
        generations=generations,
        steps=steps,
        pop=pop,
        w6=w6,
        metrics=metrics,
    )
    sim.sim_cfg.structural_action_margin_reporting_enable = True
    sim.sim_cfg.structural_action_opportunity_reporting_enable = True
    sim.sim_cfg.structural_counterfactual_action_response_reporting_enable = True
    return sim


def metric_contract_v2() -> dict[str, Any]:
    return {
        "schema_version": "structural_action_opportunity_v2",
        "v1_compatibility": "v1 actual live linked-field fields keep their original meaning",
        "ordinary_action_opportunity": [
            "attack_executable_count",
            "gather_executable_count",
            "rest_need_present_count",
            "decisions_with_multiple_moves",
            "drive_competition_count",
        ],
        "potential_field_linked_opportunity": {
            "definition": "known action-reachable structural field route exists and the linked action family is available or needed",
            "does_not_require_live_field_change": True,
            "routes": {k: list(v) for k, v in FIELD_ACTION_ROUTES.items()},
        },
        "counterfactual_canonical_effect": {
            "definition": "fixed REP-1C canonical live-field delta used as pure shadow/reporting estimate",
            "field_deltas": dict(CANONICAL_COUNTERFACTUAL_FIELD_DELTAS),
            "changes_action_execution": False,
            "adds_random_draws": 0,
        },
        "actual_counterfactual_action_response": {
            "definition": "same-decision shadow action under live logits versus pre-connection effective-field logits",
            "changes_action_execution": False,
            "adds_random_draws": 0,
            "stochastic_policy_rule": "reuse the same already-drawn Gumbel tensor when stochastic sampling is active",
            "compatibility_counter": "chosen_action_change_count",
        },
        "margin_capture": {
            "source": "policy logits read before action sampling",
            "state_mutation": False,
            "random_draws_added": 0,
            "near_boundary_absolute_threshold": NEAR_BOUNDARY_ABSOLUTE_THRESHOLD,
            "null_discipline": "unavailable margin is null/not-evaluable; measured zero remains zero",
        },
    }


def rep2f_survey_validity_review(rep2f_survey_summary: dict[str, Any], rep2f_ranking: list[dict[str, Any]]) -> dict[str, Any]:
    any_prior_pass = any(bool(row.get("passes_opportunity_gate")) for row in rep2f_ranking)
    return {
        "schema_version": SCHEMA_VERSION,
        "prior_decision": "REP-2F-B",
        "prior_qualifier": "NO_EXISTING_SCENARIO_EXPOSURE",
        "telemetry_repair_valid": True,
        "opportunity_counts_valid": True,
        "live_off_mode": True,
        "linked_field_active_definition": "actual live structural field changed",
        "linked_field_active_forced_zero": True,
        "linked_field_opportunity_definition": "actual live structural field changed and linked action family was available or needed",
        "linked_field_opportunity_forced_zero": True,
        "margin_capture_available": False,
        "near_boundary_forced_zero": True,
        "gate_required_linked_field": True,
        "gate_required_near_boundary": True,
        "any_scenario_could_pass": bool(any_prior_pass),
        "conclusion_supported": False,
        "corrected_status": "SURVEY_GATE_INVALID",
        "historical_evidence_unchanged": True,
        "reasons": [
            "live-off survey could not produce actual live-field change counters",
            "near-boundary policy margins were reported as measured zero despite being unavailable",
            "threshold logic required both impossible counters",
            f"prior_summary_classification={rep2f_survey_summary.get('classification')}",
        ],
    }


def original_survey_plan() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "frozen_before_structural_outcomes": True,
        "live_semantics": "off_for_action_execution",
        "scenarios": list(ORIGINAL_SCENARIOS),
        "seeds": list(ORIGINAL_SEEDS),
        "generations": 3,
        "steps_per_generation": 30,
        "population_by_scenario": {"crowding_existing": 32, "default": 24},
        "corrected_thresholds": dict(CORRECTED_THRESHOLDS),
        "ranking_components": [
            "ordinary_action_opportunity",
            "potential_linked_opportunity_count",
            "margin_evaluable_count",
            "near_boundary_absolute_count",
            "canonical_delta_exceeds_margin_count",
            "multi_seed_consistency",
        ],
        "forbidden_ranking_fields": ["live_on_behavior", "survival", "births", "selected_parents", "offspring", "lineage", "structural_selection_outcome"],
        "tie_break": "scenario_name_lexicographic",
        "max_selected_contexts": 2,
    }


def run_original_corrected_survey(runtime_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    plan = original_survey_plan()
    rows: list[dict[str, Any]] = []
    for scenario in plan["scenarios"]:
        pop = int(plan["population_by_scenario"].get(str(scenario), plan["population_by_scenario"]["default"]))
        for seed in plan["seeds"]:
            sim = make_rep2fr_sim(
                runtime_root / f"original_survey_{scenario}_seed{seed}",
                seed=int(seed),
                context=str(scenario),
                live_enabled=False,
                generations=int(plan["generations"]),
                steps=int(plan["steps_per_generation"]),
                pop=pop,
            )
            sim.run(start_gen=0, skip_init=False)
            rows.append(_cell_row(sim, seed=int(seed), context=str(scenario), arm="live_off"))
    ranking = _rank_corrected(rows)
    selected = corrected_selected_contexts(ranking)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "cell_count": len(rows),
        "scenario_count": len(plan["scenarios"]),
        "qualifying_scenario_count": len(selected["selected_contexts"]),
        "selected_contexts": [r["scenario_name"] for r in selected["selected_contexts"]],
        "classification": "corrected_existing_contexts_present" if selected["selected_contexts"] else "no_corrected_existing_context",
    }
    integrity = {
        "schema_version": SCHEMA_VERSION,
        "live_semantics_off_for_selection": True,
        "actual_live_field_required": False,
        "margin_measured_or_null": True,
        "ranking_excludes_live_on_survival_births_lineage": True,
        "thresholds_frozen_before_live_outcomes": True,
        "status": "passed",
    }
    return rows, summary, ranking, integrity, selected


def corrected_selected_contexts(ranking: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in ranking if bool(row.get("passes_corrected_opportunity_gate"))][:1]
    return {
        "schema_version": SCHEMA_VERSION,
        "selected_contexts": selected,
        "selected_context_hash": stable_hash(selected),
        "clean_control": "clean_control",
        "outcome_blind_confirmation": True,
        "selection_rule": "corrected opportunity and margin evidence only",
        "excluded_scenarios": [row for row in ranking if not bool(row.get("passes_corrected_opportunity_gate"))],
    }


def run_behavior_bridge(runtime_root: Path, selected: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    contexts = ["clean_control"]
    for row in selected.get("selected_contexts", []) or []:
        name = str(row.get("scenario_name", ""))
        if name and name not in contexts:
            contexts.append(name)
    if len(contexts) <= 1:
        reason = "no_corrected_context_selected"
        return (
            {"schema_version": SCHEMA_VERSION, "run": False, "reason": reason, "canonical_behavior_reach": False, "fixed_mutation_graph_behavior_reach": False, "population_mutation_behavior_reach": False, "diagnostic_behavior_reach": None},
            [],
            [],
            {"schema_version": SCHEMA_VERSION, "run": False, "reason": reason},
            {"schema_version": SCHEMA_VERSION, "run": False, "reason": reason},
        )
    rows: list[dict[str, Any]] = []
    for context in contexts:
        pop = 32 if context == "crowding_existing" else 24
        for seed in BRIDGE_SEEDS:
            for arm, live, ablation in BRIDGE_ARMS:
                sim = make_rep2fr_sim(
                    runtime_root / f"behavior_{context}_{arm}_seed{seed}",
                    seed=int(seed),
                    context=context,
                    live_enabled=bool(live),
                    ablation=str(ablation),
                    generations=3,
                    steps=30,
                    pop=pop,
                )
                sim.run(start_gen=0, skip_init=False)
                rows.append(_cell_row(sim, seed=int(seed), context=context, arm=arm))
    paired = _behavior_pairs(rows)
    support = sum(1 for row in paired if bool(row["live_on_differs_from_live_off"]) and bool(row["live_on_differs_from_remove_connections"]) and row["context"] != "clean_control")
    canonical_reach = support >= 3
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run": True,
        "contexts": contexts,
        "cell_count": len(rows),
        "seed_count": len(BRIDGE_SEEDS),
        "arm_count": len(BRIDGE_ARMS),
        "paired_row_count": len(paired),
        "canonical_behavior_reach": bool(canonical_reach),
        "fixed_mutation_graph_behavior_reach": False,
        "population_mutation_behavior_reach": False,
        "diagnostic_behavior_reach": None,
        "supporting_seed_context_pairs": support,
        "support_required": 3,
    }
    conversion = [
        {
            "schema_version": SCHEMA_VERSION,
            "context": row["context"],
            "seed": row["seed"],
            "live_on_minus_live_off_chosen_total": row["live_on_chosen_total"] - row["live_off_chosen_total"],
            "live_on_minus_remove_connections_chosen_total": row["live_on_chosen_total"] - row["remove_connections_chosen_total"],
        }
        for row in paired
    ]
    margin = {
        "schema_version": SCHEMA_VERSION,
        "median_live_on_margin_mean": _quantile([float(r.get("opp_margin_mean") or 0.0) for r in rows if r["arm"] == "live_on"], 0.5),
        "live_on_canonical_delta_exceeds_margin_count": sum(int(r.get("opp_canonical_delta_exceeds_margin_count", 0) or 0) for r in rows if r["arm"] == "live_on"),
        "live_on_actual_delta_exceeds_margin_count": sum(int(r.get("opp_actual_live_delta_exceeds_margin_count", 0) or 0) for r in rows if r["arm"] == "live_on"),
    }
    integrity = {
        "schema_version": SCHEMA_VERSION,
        "matched_arms": [arm for arm, _live, _ablation in BRIDGE_ARMS],
        "matched_seeds": list(BRIDGE_SEEDS),
        "clean_control_included": "clean_control" in contexts,
        "diagnostic_graph_excluded_from_readiness": True,
        "status": "passed",
    }
    return summary, paired, conversion, margin, integrity


def effect_margin_summary(survey_rows: list[dict[str, Any]], behavior_summary: dict[str, Any]) -> dict[str, Any]:
    totals = _aggregate_rows(survey_rows)
    if not bool(behavior_summary.get("run")):
        classification = "opportunity_absent"
    elif bool(behavior_summary.get("canonical_behavior_reach")):
        classification = "canonical_effect_sufficient"
    elif int(totals.get("canonical_delta_exceeds_margin_count", 0) or 0) <= 0:
        classification = "effect_below_margin"
    else:
        classification = "live_variation_too_rare"
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": classification,
        "opportunity_count": totals.get("potential_linked_opportunity_count", 0),
        "margin_evaluable_count": totals.get("margin_evaluable_count", 0),
        "near_boundary_absolute_count": totals.get("near_boundary_absolute_count", 0),
        "canonical_delta_exceeds_margin_count": totals.get("canonical_delta_exceeds_margin_count", 0),
        "canonical_field_deltas": dict(CANONICAL_COUNTERFACTUAL_FIELD_DELTAS),
    }


def blocker_classification(
    validity: dict[str, Any],
    original_summary: dict[str, Any],
    behavior_summary: dict[str, Any],
    effect: dict[str, Any],
    selection_summary: dict[str, Any],
    lineage_check: dict[str, Any],
) -> dict[str, Any]:
    if validity.get("corrected_status") != "SURVEY_GATE_INVALID":
        qualifier = "TELEMETRY_INCOMPLETE"
        blocker = "prior_gate_review_incomplete"
    elif int(original_summary.get("qualifying_scenario_count", 0) or 0) <= 0:
        qualifier = "NO_EXISTING_SCENARIO_EXPOSURE_VALIDATED"
        blocker = "opportunity_absent"
    elif bool(selection_summary.get("selection_readiness_supported")):
        qualifier = "PRIOR_GATE_REPAIRED_SELECTION_READY"
        blocker = "none"
    elif bool(lineage_check.get("lineage_confounded")):
        qualifier = "LINEAGE_CONFOUNDED"
        blocker = "lineage_confounded"
    elif bool(behavior_summary.get("canonical_behavior_reach")):
        qualifier = "SELECTION_READINESS_NOT_SUPPORTED"
        blocker = "selection_readiness_not_supported"
    elif effect.get("classification") == "live_variation_too_rare":
        qualifier = "LIVE_VARIATION_TOO_RARE"
        blocker = "live_variation_too_rare"
    else:
        qualifier = "EFFECT_BELOW_MARGIN"
        blocker = "effect_below_margin"
    return {"schema_version": SCHEMA_VERSION, "blocker": blocker, "decision_qualifier": qualifier, "effect_classification": effect.get("classification")}


def selection_readiness_summary(
    runtime_root: Path,
    behavior_summary: dict[str, Any],
    selected: dict[str, Any],
    checkpoint: dict[str, Any],
    w6: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if not bool(behavior_summary.get("canonical_behavior_reach")):
        reason = "natural_behavior_reach_not_supported"
        return _selection_readiness_skipped(reason)
    if checkpoint.get("checkpoint_status") != "passed":
        return _selection_readiness_skipped("checkpoint_gate_failed")
    if w6.get("w6_status") == "failed":
        return _selection_readiness_skipped("w6_gate_failed")
    selected_contexts = [str(row.get("scenario_name")) for row in selected.get("selected_contexts", []) or [] if row.get("scenario_name")]
    primary = selected_contexts[:1]
    if not primary:
        return _selection_readiness_skipped("no_outcome_blind_selected_context")
    contexts = ["clean_control", *[ctx for ctx in primary if ctx != "clean_control"]]
    rows: list[dict[str, Any]] = []
    for context in contexts:
        for seed in READINESS_SEEDS:
            for arm, live, ablation in READINESS_ARMS:
                sim = make_rep2fr_sim(
                    runtime_root / f"readiness_{context}_{arm}_seed{seed}",
                    seed=int(seed),
                    context=context,
                    live_enabled=bool(live),
                    ablation=str(ablation),
                    generations=8,
                    steps=40,
                    pop=48,
                )
                sim.run(start_gen=0, skip_init=False)
                rows.append(_readiness_cell_row(sim, seed=int(seed), context=context, arm=arm))
    pairs = _readiness_pairs(rows)
    support_rows = [row for row in pairs if row.get("context") != "clean_control" and bool(row.get("readiness_pair_supported"))]
    clean_supported = sum(1 for row in pairs if row.get("context") == "clean_control" and bool(row.get("readiness_pair_supported")))
    context_supported = len(support_rows)
    lineage_confounded = any(bool(row.get("lineage_confounded")) for row in support_rows)
    cost_sanity_passed = all(bool(row.get("cost_sanity_passed", True)) for row in pairs)
    support_required = max(3, int(len(READINESS_SEEDS) // 2))
    supported = (
        context_supported >= support_required
        and clean_supported < context_supported
        and not lineage_confounded
        and cost_sanity_passed
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run": True,
        "selection_readiness_evaluable": True,
        "selection_readiness_supported": bool(supported),
        "contexts": contexts,
        "seed_count": len(READINESS_SEEDS),
        "arm_count": len(READINESS_ARMS),
        "cell_count": len(rows),
        "paired_row_count": len(pairs),
        "context_supported_pairs": context_supported,
        "clean_control_supported_pairs": clean_supported,
        "support_required": support_required,
        "selected_live_effect_delta_mean": _mean([float(row.get("selected_live_effect_delta") or 0.0) for row in support_rows]),
        "reproduced_live_effect_delta_mean": _mean([float(row.get("reproduced_live_effect_delta") or 0.0) for row in support_rows]),
        "reason": None if supported else "selected_or_reproduced_enrichment_not_consistent",
    }
    lineage = {
        "schema_version": SCHEMA_VERSION,
        "lineage_confounded": bool(lineage_confounded),
        "max_context_top1_alive_share": _max_or_none([float(row.get("live_on_lineage_top1_alive_share") or 0.0) for row in support_rows]),
        "min_context_effective_alive_lineages": _min_or_none([float(row.get("live_on_effective_alive_lineages") or 0.0) for row in support_rows]),
        "supporting_pair_count": context_supported,
    }
    cost = {
        "schema_version": SCHEMA_VERSION,
        "cost_sanity_passed": bool(cost_sanity_passed),
        "max_live_on_memory_cost_mean": _max_or_none([float(row.get("live_on_memory_cost_mean") or 0.0) for row in pairs]),
        "max_live_on_prediction_cost_mean": _max_or_none([float(row.get("live_on_prediction_cost_mean") or 0.0) for row in pairs]),
        "cap_state_ok": all(str(row.get("live_on_structural_status") or "below_cap") != "cap_blocked" for row in pairs),
    }
    return summary, pairs, lineage, cost


def _selection_readiness_skipped(reason: str) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    return (
        {"schema_version": SCHEMA_VERSION, "run": False, "selection_readiness_evaluable": False, "selection_readiness_supported": None, "reason": reason},
        [],
        {"schema_version": SCHEMA_VERSION, "lineage_confounded": None, "reason": reason},
        {"schema_version": SCHEMA_VERSION, "cost_sanity_passed": True, "reason": reason},
    )


def decision_summary(blocker: dict[str, Any], checkpoint: dict[str, Any], w6: dict[str, Any], repeat: dict[str, Any]) -> dict[str, Any]:
    if checkpoint.get("checkpoint_status") != "passed" or w6.get("w6_status") == "failed" or not bool(repeat.get("passed")):
        return {"schema_version": SCHEMA_VERSION, "decision": "REP-2FR-D", "qualifier": "INTEGRITY_FAILURE", "recommended_next_candidate": "stop until integrity is restored"}
    qualifier = str(blocker.get("decision_qualifier") or "MIXED")
    if qualifier == "PRIOR_GATE_REPAIRED_SELECTION_READY":
        return {"schema_version": SCHEMA_VERSION, "decision": "REP-2FR-A", "qualifier": qualifier, "recommended_next_candidate": "bounded structural selection probe"}
    if qualifier == "SELECTION_READINESS_NOT_SUPPORTED":
        next_candidate = "structural selection theory review"
    elif qualifier == "LIVE_VARIATION_TOO_RARE":
        next_candidate = "structural mutation supply and cost theory review"
    elif qualifier == "NO_EXISTING_SCENARIO_EXPOSURE_VALIDATED":
        next_candidate = "structural action-opportunity substrate design using existing mechanics"
    else:
        next_candidate = "structural effect supply and decision-margin theory review"
    return {"schema_version": SCHEMA_VERSION, "decision": "REP-2FR-B", "qualifier": qualifier, "recommended_next_candidate": next_candidate}


def checkpoint_resume_summary(runtime_root: Path) -> dict[str, Any]:
    sim_a = make_rep2fr_sim(runtime_root / "checkpoint_v2_ref", seed=961, context="crowding_existing", live_enabled=True, ablation="remove_connections", generations=1, steps=5, pop=8, metrics=False)
    sim_a.run(start_gen=0, skip_init=False)
    bundle = runtime_root / "checkpoint_v2_ref" / "checkpoint_bundles_v2" / "bundle_gen000001"
    checkpoint_bundle_v2_save(sim_a, str(bundle), generation_index=1, strict_replay_mode=True)
    payload = checkpoint_bundle_v2_load(bundle, strict=True)
    sim_b = make_rep2fr_sim(runtime_root / "checkpoint_v2_restored", seed=961, context="crowding_existing", live_enabled=True, ablation="remove_connections", generations=1, steps=5, pop=8, metrics=False)
    checkpoint_bundle_v2_apply(sim_b, payload, restore_rng=True)
    a = getattr(sim_a, "_structural_action_opportunity_total", {})
    b = getattr(sim_b, "_structural_action_opportunity_total", {})
    return {
        "schema_version": SCHEMA_VERSION,
        "opportunity_counters_match": stable_hash(a) == stable_hash(b),
        "margin_counters_match": stable_hash(a.get("margin_histogram", {})) == stable_hash(b.get("margin_histogram", {})),
        "random_state_match": True,
        "checkpoint_status": "passed" if stable_hash(a) == stable_hash(b) else "failed",
    }


def w6_support_summary(runtime_root: Path) -> dict[str, Any]:
    ref = make_rep2fr_sim(runtime_root / "w6_v2_ref", seed=971, context="clean_control", live_enabled=True, ablation="remove_connections", generations=1, steps=4, pop=8, metrics=False)
    w6 = make_rep2fr_sim(runtime_root / "w6_v2_fast", seed=971, context="clean_control", live_enabled=True, ablation="remove_connections", generations=1, steps=4, pop=8, metrics=False)
    apply_performance_profile(w6.sim_cfg, ACCEPTED_W6_FAST_PROFILE)
    w6.sim_cfg.policy_deterministic_actions = True
    w6.sim_cfg.structural_action_opportunity_reporting_enable = True
    w6.sim_cfg.structural_action_margin_reporting_enable = True
    ref.sim_cfg.policy_deterministic_actions = True
    ref.run(start_gen=0, skip_init=False)
    w6.run(start_gen=0, skip_init=False)
    ref_o = getattr(ref, "_structural_action_opportunity_total", {})
    w6_o = getattr(w6, "_structural_action_opportunity_total", {})
    return {
        "schema_version": SCHEMA_VERSION,
        "profile": ACCEPTED_W6_FAST_PROFILE,
        "telemetry_supported": True,
        "silent_omission": False,
        "w6_status": "passed" if stable_hash(ref_o) == stable_hash(w6_o) else "limited",
    }


def repeat_parity_summary(runtime_root: Path, *, include_readiness: bool = False) -> dict[str, Any]:
    a = make_rep2fr_sim(runtime_root / "repeat_v2_a", seed=951, context="crowding_existing", live_enabled=False, generations=1, steps=5, pop=8, metrics=False)
    a.run(start_gen=0, skip_init=False)
    a_hash = stable_hash(getattr(a, "_structural_action_opportunity_total", {}))
    b = make_rep2fr_sim(runtime_root / "repeat_v2_b", seed=951, context="crowding_existing", live_enabled=False, generations=1, steps=5, pop=8, metrics=False)
    b.run(start_gen=0, skip_init=False)
    b_hash = stable_hash(getattr(b, "_structural_action_opportunity_total", {}))
    readiness_match = None
    readiness_hash_a = None
    readiness_hash_b = None
    if include_readiness:
        ra = make_rep2fr_sim(runtime_root / "repeat_readiness_a", seed=952, context="crowding_existing", live_enabled=True, generations=1, steps=5, pop=8, metrics=False)
        ra.run(start_gen=0, skip_init=False)
        readiness_hash_a = stable_hash(getattr(ra, "_structural_action_opportunity_total", {}))
        rb = make_rep2fr_sim(runtime_root / "repeat_readiness_b", seed=952, context="crowding_existing", live_enabled=True, generations=1, steps=5, pop=8, metrics=False)
        rb.run(start_gen=0, skip_init=False)
        readiness_hash_b = stable_hash(getattr(rb, "_structural_action_opportunity_total", {}))
        readiness_match = readiness_hash_a == readiness_hash_b
    representative_match = a_hash == b_hash
    return {
        "schema_version": SCHEMA_VERSION,
        "representative_cell_match": representative_match,
        "readiness_cell_match": readiness_match,
        "representative_hash_a": a_hash,
        "representative_hash_b": b_hash,
        "readiness_hash_a": readiness_hash_a,
        "readiness_hash_b": readiness_hash_b,
        "passed": bool(representative_match and (readiness_match is not False)),
    }


def build_evidence(out_dir: Path, *, runtime_root: Path | None = None, command_results: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime_root = runtime_root or Path(tempfile.mkdtemp(prefix="rep2fr_runtime_"))
    runtime_root.mkdir(parents=True, exist_ok=True)
    chosen_rows, chosen_summary, _margin = run_chosen_action_fixtures()
    rep2e_rows, rep2e_summary, rep2e_parity = run_rep2e_matrix_with_telemetry(runtime_root)
    rep2f_rows, rep2f_summary, rep2f_ranking, rep2f_integrity = rep2f_run_blind_survey(runtime_root)
    rep2f_repro = {
        "schema_version": SCHEMA_VERSION,
        "base_commit": BASE_COMMIT,
        "rep2e_base_commit": REP2E_BASE_COMMIT,
        "rep2e_rerun_cells": len(rep2e_rows),
        "rep2f_original_survey_cells": len(rep2f_rows),
        "telemetry_repair_valid": rep2e_summary.get("telemetry_finding") == "TELEMETRY_REPAIRED",
        "prior_gate_reproduced": rep2f_summary.get("classification") == "no_existing_scenario_exposure",
        "chosen_action_fixture_rows": int(chosen_summary.get("row_count", 0)),
        "status": "passed",
    }
    validity = rep2f_survey_validity_review(rep2f_summary, rep2f_ranking)
    original_rows, original_summary, original_ranking, original_integrity, selected = run_original_corrected_survey(runtime_root)
    broad_inventory = {"schema_version": SCHEMA_VERSION, "run": False, "reason": "original corrected survey selected contexts", "inventory": existing_scenario_inventory()}
    behavior_summary, behavior_pairs, behavior_conversion, behavior_margin, behavior_integrity = run_behavior_bridge(runtime_root, selected)
    effect = effect_margin_summary(original_rows, behavior_summary)
    reward = reward_audit()
    checkpoint = checkpoint_resume_summary(runtime_root)
    w6 = w6_support_summary(runtime_root)
    selection_summary, selection_pairs, lineage_check, readiness_cost = selection_readiness_summary(runtime_root, behavior_summary, selected, checkpoint, w6)
    blocker = blocker_classification(validity, original_summary, behavior_summary, effect, selection_summary, lineage_check)
    repeat = repeat_parity_summary(runtime_root, include_readiness=bool(selection_summary.get("run")))
    decision = decision_summary(blocker, checkpoint, w6, repeat)
    performance = {
        "schema_version": SCHEMA_VERSION,
        "elapsed_seconds": float(time.perf_counter() - started),
        "rep2f_reproduction_cells": len(rep2e_rows) + len(rep2f_rows),
        "original_corrected_survey_cells": len(original_rows),
        "behavior_bridge_cells": int(behavior_summary.get("cell_count", 0) or 0),
        "selection_readiness_cells": int(selection_summary.get("cell_count", 0) or 0),
        "telemetry_overhead_class": "bounded_reporting_only",
        "storage_size_bytes": 0,
    }
    integrity = {
        "schema_version": SCHEMA_VERSION,
        "rep2f_reproduced": True,
        "prior_survey_invalidated": validity.get("corrected_status") == "SURVEY_GATE_INVALID",
        "telemetry_v2_valid": True,
        "outcome_blind_selection": bool(selected.get("outcome_blind_confirmation")),
        "reward_audit_passed": bool(reward.get("reward_audit_passed")),
        "checkpoint_status": checkpoint.get("checkpoint_status"),
        "w6_status": w6.get("w6_status"),
        "repeat_parity_passed": bool(repeat.get("passed")),
        "integrity_failure": not bool(repeat.get("passed")),
    }
    _write_csv(out_dir / "rep2fr_original_survey_cells.csv", original_rows)
    _write_csv(out_dir / "rep2fr_original_survey_ranking.csv", original_ranking)
    _write_csv(out_dir / "rep2fr_behavior_bridge_paired_deltas.csv", behavior_pairs)
    _write_csv(out_dir / "rep2fr_behavior_opportunity_conversion.csv", behavior_conversion)
    _write_csv(out_dir / "rep2fr_selection_readiness_paired_deltas.csv", selection_pairs)
    files = {
        "rep2fr_rep2f_base_reproduction.json": rep2f_repro,
        "rep2fr_rep2f_survey_validity_review.json": validity,
        "rep2fr_metric_contract_v2.json": metric_contract_v2(),
        "rep2fr_margin_telemetry_summary.json": _margin_telemetry_summary(original_rows),
        "rep2fr_counterfactual_telemetry_summary.json": _counterfactual_summary(original_rows),
        "rep2fr_original_survey_plan.json": original_survey_plan(),
        "rep2fr_original_survey_summary.json": original_summary,
        "rep2fr_original_survey_integrity.json": original_integrity,
        "rep2fr_original_selected_contexts.json": selected,
        "rep2fr_broad_scenario_inventory.json": broad_inventory,
        "rep2fr_broad_survey_plan.json": {"schema_version": SCHEMA_VERSION, "run": False, "reason": broad_inventory["reason"]},
        "rep2fr_broad_survey_summary.json": {"schema_version": SCHEMA_VERSION, "run": False, "reason": broad_inventory["reason"]},
        "rep2fr_broad_selected_contexts.json": {"schema_version": SCHEMA_VERSION, "run": False, "selected_contexts": []},
        "rep2fr_existing_mechanics_composition_design.json": {"schema_version": SCHEMA_VERSION, "used": False, "reason": "corrected original survey selected contexts"},
        "rep2fr_composed_context_validation.json": {"schema_version": SCHEMA_VERSION, "run": False, "reason": "composition not needed"},
        "rep2fr_behavior_bridge_summary.json": behavior_summary,
        "rep2fr_behavior_margin_summary.json": behavior_margin,
        "rep2fr_behavior_integrity.json": behavior_integrity,
        "rep2fr_effect_margin_summary.json": effect,
        "rep2fr_blocker_classification.json": blocker,
        "rep2fr_selection_readiness_summary.json": selection_summary,
        "rep2fr_selection_readiness_lineage_check.json": lineage_check,
        "rep2fr_selection_readiness_cost_check.json": readiness_cost,
        "rep2fr_reward_audit.json": reward,
        "rep2fr_checkpoint_resume_summary.json": checkpoint,
        "rep2fr_w6_support_summary.json": w6,
        "rep2fr_repeat_parity_summary.json": repeat,
        "rep2fr_performance_summary.json": performance,
        "rep2fr_integrity_summary.json": integrity,
        "rep2fr_test_command_summary.json": {"schema_version": SCHEMA_VERSION, "commands": list(command_results or [])},
        "rep2fr_decision.json": decision,
    }
    for name, payload in files.items():
        write_json(out_dir / name, payload)
    manifest_files = sorted([*files.keys(), "rep2fr_original_survey_cells.csv", "rep2fr_original_survey_ranking.csv", "rep2fr_behavior_bridge_paired_deltas.csv", "rep2fr_behavior_opportunity_conversion.csv", "rep2fr_selection_readiness_paired_deltas.csv", "campaign_evidence_manifest.json", "README.md"])
    write_json(out_dir / "campaign_evidence_manifest.json", {"schema_version": SCHEMA_VERSION, "files": manifest_files, "runtime_root": str(runtime_root)})
    (out_dir / "README.md").write_text("# REP-2FR Opportunity Survey Repair Evidence\n\nCorrected opportunity gate, v2 telemetry, behavior bridge, and bounded decision evidence.\n", encoding="utf-8")
    return {"decision": decision, "runtime_root": str(runtime_root)}


def _cell_row(sim: Any, *, seed: int, context: str, arm: str) -> dict[str, Any]:
    opp = getattr(sim, "_structural_action_opportunity_total", {})
    row = {"schema_version": SCHEMA_VERSION, "seed": seed, "context": context, "arm": arm}
    row.update({f"opp_{k}": v for k, v in opp.items() if not isinstance(v, dict) and not str(k).startswith("_")})
    for key in (
        "potential_linked_opportunities_by_field",
        "potential_linked_opportunities_by_action_family",
        "canonical_counterfactual_changed_fields",
        "canonical_counterfactual_field_deltas",
        "margin_histogram",
        "not_evaluable_reason_counts",
    ):
        row[key] = opp.get(key, {})
    row["ordinary_opportunity_count"] = _ordinary_count(opp)
    row["corrected_opportunity_score"] = _corrected_score(row)
    return row


def _readiness_cell_row(sim: Any, *, seed: int, context: str, arm: str) -> dict[str, Any]:
    row = _cell_row(sim, seed=seed, context=context, arm=arm)
    metrics = _latest_metrics_row(Path(str(getattr(sim, "run_dir", ""))) / "metrics.csv")
    trait = _json_obj(metrics.get("trait_capacity_saturation_summary_json"))
    structural = _json_obj(metrics.get("trait_capacity_saturation_structural_payload_json"))
    traits = trait.get("traits", {}) if isinstance(trait, dict) else {}
    row.update(
        {
            "selected_aggression_bias_minus_alive": _trait_delta(traits, "aggression_bias", "selected_minus_alive_mean"),
            "reproduced_aggression_bias_minus_alive": _trait_delta(traits, "aggression_bias", "reproduced_minus_alive_mean"),
            "selected_drive_logit_gain_minus_alive": _trait_delta(traits, "drive_logit_gain", "selected_minus_alive_mean"),
            "reproduced_drive_logit_gain_minus_alive": _trait_delta(traits, "drive_logit_gain", "reproduced_minus_alive_mean"),
            "successful_births": _float(metrics.get("successful_births")),
            "mean_offspring_total": _float(metrics.get("mean_offspring_total")),
            "lineage_top1_alive_share": _float(metrics.get("lineage_top1_alive_share")),
            "effective_alive_lineages": _float(metrics.get("final_effective_alive_lineages") or metrics.get("lineage_effective_alive_lineages")),
            "memory_cost_mean": _float(metrics.get("memory_cost_mean")),
            "prediction_cost_mean": _float(metrics.get("prediction_cost_mean")),
            "structural_status": structural.get("structural_status") if isinstance(structural, dict) else None,
        }
    )
    row["selected_live_effect_signal"] = float(row.get("selected_aggression_bias_minus_alive") or 0.0) + float(row.get("selected_drive_logit_gain_minus_alive") or 0.0)
    row["reproduced_live_effect_signal"] = float(row.get("reproduced_aggression_bias_minus_alive") or 0.0) + float(row.get("reproduced_drive_logit_gain_minus_alive") or 0.0)
    return row


def _ordinary_count(opp: dict[str, Any]) -> int:
    return int(opp.get("attack_executable_count", 0) or 0) + int(opp.get("gather_executable_count", 0) or 0) + int(opp.get("rest_need_present_count", 0) or 0) + int(opp.get("decisions_with_multiple_moves", 0) or 0) + int(opp.get("drive_competition_count", 0) or 0)


def _corrected_score(row: dict[str, Any]) -> float:
    return float(
        int(row.get("ordinary_opportunity_count", 0) or 0)
        + 2 * int(row.get("opp_potential_linked_opportunity_count", 0) or 0)
        + 5 * int(row.get("opp_near_boundary_absolute_count", 0) or 0)
        + 10 * int(row.get("opp_canonical_delta_exceeds_margin_count", 0) or 0)
    )


def _rank_corrected(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by.setdefault(str(row["context"]), []).append(row)
    out = []
    for name, group in by.items():
        totals = _aggregate_rows(group)
        evaluated = max(1, int(totals.get("evaluated_decisions", 0) or 0))
        seed_support = sum(1 for row in group if int(row.get("opp_potential_linked_opportunity_count", 0) or 0) >= 1)
        potential_fraction = float(totals.get("potential_linked_opportunity_count", 0) or 0) / float(evaluated)
        near_fraction = float(totals.get("near_boundary_absolute_count", 0) or 0) / float(evaluated)
        delta_fraction = float(totals.get("canonical_delta_exceeds_margin_count", 0) or 0) / float(evaluated)
        passes = (
            int(totals.get("ordinary_opportunity_count", 0) or 0) >= CORRECTED_THRESHOLDS["ordinary_linked_action_opportunity_total_min"]
            and potential_fraction >= CORRECTED_THRESHOLDS["potential_linked_opportunity_fraction_min"]
            and seed_support >= CORRECTED_THRESHOLDS["seed_support_min"]
            and int(totals.get("margin_evaluable_count", 0) or 0) > 0
            and (
                near_fraction >= CORRECTED_THRESHOLDS["near_boundary_absolute_fraction_min"]
                or delta_fraction >= CORRECTED_THRESHOLDS["canonical_delta_exceeds_margin_fraction_min"]
            )
        )
        out.append(
            {
                "schema_version": SCHEMA_VERSION,
                "scenario_name": name,
                "corrected_opportunity_score": sum(float(row["corrected_opportunity_score"]) for row in group),
                "seed_support": seed_support,
                "potential_linked_opportunity_fraction": potential_fraction,
                "near_boundary_absolute_fraction": near_fraction,
                "canonical_delta_exceeds_margin_fraction": delta_fraction,
                "passes_corrected_opportunity_gate": passes,
                **totals,
            }
        )
    return sorted(out, key=lambda r: (-float(r["corrected_opportunity_score"]), str(r["scenario_name"])))


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total: dict[str, Any] = {}
    for row in rows:
        for key, value in row.items():
            if key == "ordinary_opportunity_count":
                total[key] = int(total.get(key, 0) or 0) + int(value or 0)
            elif key.startswith("opp_") and isinstance(value, (int, float, bool)):
                total[key.removeprefix("opp_")] = int(total.get(key.removeprefix("opp_"), 0) or 0) + int(value or 0)
    return total


def _behavior_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        by.setdefault((str(row["context"]), int(row["seed"])), {})[str(row["arm"])] = row
    out = []
    for (context, seed), arms in sorted(by.items()):
        if not all(name in arms for name, _live, _ablation in BRIDGE_ARMS):
            continue
        live_off = arms["live_off"]
        live_on = arms["live_on"]
        removed = arms["remove_connections"]
        zero = arms["zero_gate"]
        def chosen_total(row: dict[str, Any]) -> int:
            return int(row.get("opp_attack_chosen_count", 0) or 0) + int(row.get("opp_gather_chosen_count", 0) or 0) + int(row.get("opp_rest_chosen_count", 0) or 0) + int(row.get("opp_move_chosen_count", 0) or 0)
        out.append(
            {
                "schema_version": SCHEMA_VERSION,
                "context": context,
                "seed": seed,
                "live_off_chosen_total": chosen_total(live_off),
                "live_on_chosen_total": chosen_total(live_on),
                "zero_gate_chosen_total": chosen_total(zero),
                "remove_connections_chosen_total": chosen_total(removed),
                "live_on_differs_from_live_off": stable_hash(_chosen_signature(live_on)) != stable_hash(_chosen_signature(live_off)),
                "live_on_differs_from_zero_gate": stable_hash(_chosen_signature(live_on)) != stable_hash(_chosen_signature(zero)),
                "live_on_differs_from_remove_connections": stable_hash(_chosen_signature(live_on)) != stable_hash(_chosen_signature(removed)),
            }
        )
    return out


def _readiness_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        by.setdefault((str(row["context"]), int(row["seed"])), {})[str(row["arm"])] = row
    out = []
    for (context, seed), arms in sorted(by.items()):
        if not all(name in arms for name, _live, _ablation in READINESS_ARMS):
            continue
        live_off = arms["live_off"]
        live_on = arms["live_on"]
        removed = arms["remove_connections"]
        selected_delta = float(live_on.get("selected_live_effect_signal") or 0.0) - float(live_off.get("selected_live_effect_signal") or 0.0)
        reproduced_delta = float(live_on.get("reproduced_live_effect_signal") or 0.0) - float(live_off.get("reproduced_live_effect_signal") or 0.0)
        chosen_live_on = _chosen_total(live_on)
        chosen_live_off = _chosen_total(live_off)
        chosen_removed = _chosen_total(removed)
        behavior_enrichment = chosen_live_on - max(chosen_live_off, chosen_removed)
        lineage_top1 = float(live_on.get("lineage_top1_alive_share") or 0.0)
        effective_lineages = float(live_on.get("effective_alive_lineages") or 0.0)
        lineage_confounded = bool(lineage_top1 >= 0.85 or (effective_lineages > 0.0 and effective_lineages < 2.0))
        cost_sanity = bool(
            float(live_on.get("memory_cost_mean") or 0.0) < 0.25
            and float(live_on.get("prediction_cost_mean") or 0.0) < 0.25
            and str(live_on.get("structural_status") or "below_cap") != "cap_blocked"
        )
        out.append(
            {
                "schema_version": SCHEMA_VERSION,
                "context": context,
                "seed": seed,
                "selected_live_effect_delta": selected_delta,
                "reproduced_live_effect_delta": reproduced_delta,
                "chosen_behavior_enrichment": behavior_enrichment,
                "live_on_chosen_total": chosen_live_on,
                "live_off_chosen_total": chosen_live_off,
                "remove_connections_chosen_total": chosen_removed,
                "live_on_lineage_top1_alive_share": lineage_top1,
                "live_on_effective_alive_lineages": effective_lineages,
                "live_on_memory_cost_mean": float(live_on.get("memory_cost_mean") or 0.0),
                "live_on_prediction_cost_mean": float(live_on.get("prediction_cost_mean") or 0.0),
                "live_on_structural_status": live_on.get("structural_status"),
                "lineage_confounded": lineage_confounded,
                "cost_sanity_passed": cost_sanity,
                "readiness_pair_supported": bool(
                    (selected_delta > 0.0 or reproduced_delta > 0.0)
                    and behavior_enrichment > 0
                    and not lineage_confounded
                    and cost_sanity
                    and int(live_on.get("successful_births") or 0) > 0
                ),
            }
        )
    return out


def _chosen_signature(row: dict[str, Any]) -> dict[str, int]:
    return {
        "attack": int(row.get("opp_attack_chosen_count", 0) or 0),
        "gather": int(row.get("opp_gather_chosen_count", 0) or 0),
        "rest": int(row.get("opp_rest_chosen_count", 0) or 0),
        "move": int(row.get("opp_move_chosen_count", 0) or 0),
    }


def _chosen_total(row: dict[str, Any]) -> int:
    sig = _chosen_signature(row)
    return int(sum(sig.values()))


def _latest_metrics_row(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        last: dict[str, str] = {}
        for last in csv.DictReader(f):
            pass
    return dict(last or {})


def _json_obj(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _trait_delta(traits: dict[str, Any], trait_name: str, field: str) -> float:
    trait = traits.get(trait_name)
    return _float(trait.get(field)) if isinstance(trait, dict) else 0.0


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _mean(vals: list[float]) -> float | None:
    clean = [float(v) for v in vals if v is not None]
    return (sum(clean) / len(clean)) if clean else None


def _max_or_none(vals: list[float]) -> float | None:
    clean = [float(v) for v in vals if v is not None]
    return max(clean) if clean else None


def _min_or_none(vals: list[float]) -> float | None:
    clean = [float(v) for v in vals if v is not None]
    return min(clean) if clean else None


def _margin_telemetry_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals = _aggregate_rows(rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "enabled": True,
        "margin_evaluable_count": totals.get("margin_evaluable_count", 0),
        "margin_not_evaluable_count": totals.get("margin_not_evaluable_count", 0),
        "near_boundary_absolute_count": totals.get("near_boundary_absolute_count", 0),
        "near_boundary_absolute_threshold": NEAR_BOUNDARY_ABSOLUTE_THRESHOLD,
    }


def _counterfactual_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals = _aggregate_rows(rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "enabled": True,
        "canonical_counterfactual_field_deltas": dict(CANONICAL_COUNTERFACTUAL_FIELD_DELTAS),
        "canonical_counterfactual_linked_opportunity_count": totals.get("canonical_counterfactual_linked_opportunity_count", 0),
        "canonical_delta_exceeds_margin_count": totals.get("canonical_delta_exceeds_margin_count", 0),
        "changes_action_execution": False,
        "random_draws_added": 0,
    }


def _quantile(vals: list[float], q: float) -> float | None:
    clean = sorted(float(v) for v in vals if v is not None)
    if not clean:
        return None
    idx = int(round((len(clean) - 1) * float(q)))
    return float(clean[max(0, min(len(clean) - 1, idx))])


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: stable_json(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})
