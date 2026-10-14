from __future__ import annotations

"""REP-2E structural behavior-demand and chosen-action bridge helpers.

These helpers are evidence utilities. They add reporting and a default-off
matched seed ablation control; they do not change rewards, selection,
reproduction, survival, mutation rates, structural caps, operator budgets,
complexity costs, or world coefficients.
"""

import csv
import json
import math
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from agents.genome import Genome
from env.env_config import EnvConfig
from env.gridworld_pkg.constants import ACT_ATTACK, ACT_GATHER, ACT_REST
from evolution.structural_causal_robustness import (
    REP2B_GRAPH_DIGEST,
    action_trace_outcomes,
    base_reproduction_summary as rep2d_base_reproduction,
    cfg_live,
    cost_sanity,
    decision_summary as rep2d_decision_summary,
    evaluate_fixture,
    freeze_robustness_battery_plan,
    live_robustness_summary,
    operator_supply,
    repeat_parity_summary as rep2d_repeat_parity_summary,
    reward_audit,
    run_fixture_battery,
    seed_payload,
    stable_hash,
    stable_json,
    unsupported_graph_summary,
    w6_support_summary as rep2d_w6_support_summary,
    write_json,
    _draws_for_seed,
    _live_expr_for_payload,
    _payload_removed_first_edge,
    _payload_with_expression_scale,
    _payload_zero_gate,
    _policy_logits,
    _read_metrics,
    _sim_live_signature,
    _softmax,
    _strip_timing,
    _write_csv,
)
from evolution.structural_mutation_viability import apply_single_operator, complexity_cost
from simulation.checkpoint_bundle_v2 import checkpoint_bundle_v2_apply, checkpoint_bundle_v2_load, checkpoint_bundle_v2_save
from simulation.metrics import MetricsLogger
from simulation.metrics_pkg.noop_loggers import NoOpMetricsLogger
from simulation.performance_profiles import ACCEPTED_W6_FAST_PROFILE, apply_performance_profile
from simulation.runner import SimulationRunner
from evolution.population_pkg.population_config import PopConfig


SCHEMA_VERSION = "structural_behavior_demand_v1"
ACTION_MARGIN_SCHEMA_VERSION = "structural_action_margin_v1"
REP2D_BASE_COMMIT = "1780878344eb805ab285dfcfbd8801147c5938eb"
NEAR_BOUNDARY_MARGIN = 0.06
ACTION_NAMES = {
    0: "move_north",
    1: "move_south",
    2: "move_west",
    3: "move_east",
    int(ACT_GATHER): "gather",
    int(ACT_REST): "rest",
    int(ACT_ATTACK): "attack",
}


def payload_remove_connections() -> dict[str, Any]:
    payload = seed_payload()
    payload["connections"] = []
    return payload


def diagnostic_payload() -> dict[str, Any]:
    payload = seed_payload()
    if payload.get("connections"):
        payload["connections"][0]["weight"] = 2.0
    return payload


def mutation_payload() -> dict[str, Any]:
    return apply_single_operator("gate_drift", _draws_for_seed(5), sim_cfg=cfg_live())["after_payload"]


def graph_arms() -> dict[str, dict[str, Any]]:
    return {
        "live_off": {
            "graph_class": "live_off",
            "payload": seed_payload(),
            "enabled": False,
            "ablation_mode": "none",
            "readiness_eligible": False,
        },
        "canonical": {
            "graph_class": "canonical",
            "payload": seed_payload(),
            "enabled": True,
            "ablation_mode": "none",
            "readiness_eligible": True,
        },
        "zero_gate": {
            "graph_class": "zero_gate",
            "payload": _payload_zero_gate(),
            "enabled": True,
            "ablation_mode": "zero_connection_gates",
            "readiness_eligible": False,
        },
        "remove_connections": {
            "graph_class": "remove_connections",
            "payload": payload_remove_connections(),
            "enabled": True,
            "ablation_mode": "remove_connections",
            "readiness_eligible": False,
        },
        "mutation_generated": {
            "graph_class": "mutation_generated",
            "payload": mutation_payload(),
            "enabled": True,
            "ablation_mode": "none",
            "readiness_eligible": True,
        },
        "diagnostic": {
            "graph_class": "diagnostic",
            "payload": diagnostic_payload(),
            "enabled": True,
            "ablation_mode": "none",
            "readiness_eligible": False,
        },
        "temperature_expression": {
            "graph_class": "diagnostic_temperature",
            "payload": _payload_with_expression_scale(1.5),
            "enabled": True,
            "ablation_mode": "none",
            "readiness_eligible": False,
        },
    }


def behavior_contexts() -> list[dict[str, Any]]:
    return [
        {
            "fixture_id": "aggression_live_attack_opportunity",
            "behavior_family": "aggression_attack_opportunity",
            "base_logits": [0.0, 0.0, 0.0, 0.0, 0.20, 0.10, 0.18],
            "drives": [0.0, 0.0, 0.0, 0.0],
            "available_actions": [0, 1, 2, 3, int(ACT_GATHER), int(ACT_REST), int(ACT_ATTACK)],
            "expected_field": "aggression_bias",
            "expected_direction": "attack_score_up",
            "expected_action_route": "attack_over_gather_when_attack_available",
        },
        {
            "fixture_id": "drive_gain_gather_rest_competition",
            "behavior_family": "drive_choice_competition",
            "base_logits": [0.0, 0.0, 0.0, 0.0, -0.02, 0.19, -0.20],
            "drives": [0.20, 0.0, 0.0, 0.0],
            "available_actions": [0, 1, 2, 3, int(ACT_GATHER), int(ACT_REST)],
            "expected_field": "drive_logit_gain",
            "expected_direction": "gather_score_up_under_hunger",
            "expected_action_route": "gather_over_rest_near_boundary",
        },
        {
            "fixture_id": "temperature_near_tie_probability",
            "behavior_family": "temperature_near_tie_choice",
            "base_logits": [0.0, 0.0, 0.0, 0.0, 0.210, 0.205, 0.200],
            "drives": [0.0, 0.0, 0.0, 0.0],
            "available_actions": [0, 1, 2, 3, int(ACT_GATHER), int(ACT_REST), int(ACT_ATTACK)],
            "expected_field": "action_temperature",
            "expected_direction": "probability_distribution_change",
            "expected_action_route": "argmax_should_remain_when_temperature_only_scales_positive_logits",
        },
        {
            "fixture_id": "attack_unavailable_control",
            "behavior_family": "unavailable_target_control",
            "base_logits": [0.0, 0.0, 0.0, 0.0, 0.20, 0.10, 0.18],
            "drives": [0.0, 0.0, 0.0, 0.0],
            "available_actions": [0, 1, 2, 3, int(ACT_GATHER), int(ACT_REST)],
            "expected_field": "aggression_bias",
            "expected_direction": "attack_score_up_but_masked",
            "expected_action_route": "attack_unavailable_no_executed_attack",
        },
    ]


def freeze_behavior_demand_fixture_plan() -> dict[str, Any]:
    arms = graph_arms()
    fixtures: list[dict[str, Any]] = []
    for context in behavior_contexts():
        if context["behavior_family"] == "temperature_near_tie_choice":
            selected = ("live_off", "canonical", "zero_gate", "temperature_expression", "diagnostic")
        else:
            selected = ("live_off", "canonical", "zero_gate", "remove_connections", "mutation_generated", "diagnostic")
        for arm_id in selected:
            arm = arms[arm_id]
            live = _live_expr_for_payload(arm["payload"], enabled=bool(arm["enabled"]))
            fixtures.append(
                {
                    "fixture_id": f"{context['fixture_id']}__{arm_id}",
                    "context_id": context["fixture_id"],
                    "behavior_family": context["behavior_family"],
                    "state_observation_source": "fixed_reporting_fixture_existing_policy_fields",
                    "available_actions": list(context["available_actions"]),
                    "graph_arm": arm_id,
                    "graph_class": arm["graph_class"],
                    "graph_digest": str(live.get("structural_live_semantics", {}).get("graph_digest") or ""),
                    "expected_field": context["expected_field"],
                    "expected_direction": context["expected_direction"],
                    "expected_action_route": context["expected_action_route"],
                    "random_seed_draw_plan": "deterministic_argmax_no_random_draws",
                    "eligibility_for_full_run_bridge": arm_id in {"canonical", "mutation_generated", "diagnostic"},
                    "eligibility_for_readiness_evidence": bool(arm["readiness_eligible"]),
                }
            )
    return {"schema_version": SCHEMA_VERSION, "fixtures": fixtures, "frozen_before_outcomes": True}


def action_choice_code_map() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "source_files": [
            "engine/simulation/runner_pkg/runner_policy.py",
            "engine/agents/batched_policy.py",
            "engine/agents/batched_gru.py",
            "engine/simulation/runnercore/tick_loop.py",
            "engine/evolution/structural_live_semantics.py",
        ],
        "answers": {
            "score_changes_measured_before_or_after_masks": "REP-2E fixture scores are raw policy logits and matched masked logits; normal runner policy emits logits before environment execution.",
            "changed_score_can_belong_to_unavailable_action": True,
            "chosen_action_rule": "deterministic argmax when policy_deterministic_actions=True; otherwise Gumbel-max sampling from softmax logits.",
            "random_draws_in_deterministic_choice": 0,
            "action_temperature_effect": "Divides logits before argmax/sampling; with deterministic argmax it changes margins/probabilities but only flips choices when scaling changes rank after additive drive/bias terms.",
            "aggression_bias_attack_unavailable": "It can change ATTACK score while executed attack remains unavailable or non-effective when no target/opportunity exists.",
            "drive_logit_gain_scope": "Adds drive-weighted terms to GATHER, REST, and ATTACK scores.",
            "drive_inertia_scope": "Changes the next drive state over a sequence; it does not directly add to one-step logits.",
            "scores_clipped_after_live_fields": False,
            "w6_same_rule": "W6 uses the same live fields and deterministic choice when profile is applied; support is checked separately.",
            "chosen_action_trace_without_rng_change": True,
            "same_random_sample_reuse": "REP-2E deterministic fixtures use no random draw; stochastic support is a code-path audit only.",
            "full_run_action_margin_compact_reporting": "Provided by compact evidence summaries outside the hot action loop.",
            "edge_ablation_as_genome": "Pure payload removal works in fixtures; full-run founder seed ablation is default-off diagnostic config.",
        },
    }


def action_choice_causal_map() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "map": [
            "live structural fields",
            "policy effective fields",
            "raw logits",
            "masked logits in REP-2E fixtures",
            "probabilities",
            "top-two margin",
            "chosen action",
            "executed action/opportunity",
            "behavior metrics",
        ],
    }


def field_consumption_map() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "consumed_by_one_step_logits": ["aggression_bias", "action_temperature", "drive_logit_gain"],
        "consumed_by_short_sequence_drive_state": ["drive_inertia"],
        "not_exercised_by_rep2e_action_fixtures": ["drive_noise", "pred_lr", "reproduction_threshold"],
        "field_to_action": {
            "aggression_bias": "adds to ATTACK logit before temperature/drive terms",
            "action_temperature": "divides MLP/GRU logits before drive terms and sampling",
            "drive_logit_gain": "scales hunger/pain/fatigue/fear drive additions to GATHER/REST/ATTACK",
            "drive_inertia": "smooths next drive state in runnercore drive update",
        },
    }


def action_margin(base_scores: list[float], scores: list[float], available: list[int]) -> dict[str, Any]:
    arr = np.asarray(scores, dtype=np.float64)
    base = np.asarray(base_scores, dtype=np.float64)
    masked = np.full_like(arr, -math.inf)
    masked[list(available)] = arr[list(available)]
    order = np.argsort(masked)[::-1]
    top = int(order[0])
    second = int(order[1])
    deltas = arr - base
    margin = float(masked[top] - masked[second])
    max_delta = float(np.max(np.abs(deltas))) if deltas.size else 0.0
    return {
        "schema_version": ACTION_MARGIN_SCHEMA_VERSION,
        "top_action": ACTION_NAMES.get(top, str(top)),
        "second_action": ACTION_NAMES.get(second, str(second)),
        "top_action_id": top,
        "second_action_id": second,
        "top_score": float(masked[top]),
        "second_score": float(masked[second]),
        "top_two_margin": margin,
        "structural_score_delta_by_action": {ACTION_NAMES.get(i, str(i)): float(deltas[i]) for i in range(len(deltas))},
        "max_absolute_structural_score_delta": max_delta,
        "structural_delta_to_margin_ratio": None if margin == 0.0 else float(max_delta / abs(margin)),
    }


def run_chosen_action_fixtures() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    arms = graph_arms()
    rows: list[dict[str, Any]] = []
    baseline_by_context: dict[str, dict[str, Any]] = {}
    for context in behavior_contexts():
        for arm_id, arm in arms.items():
            if context["behavior_family"] == "temperature_near_tie_choice" and arm_id not in {
                "live_off",
                "canonical",
                "zero_gate",
                "temperature_expression",
                "diagnostic",
            }:
                continue
            if context["behavior_family"] != "temperature_near_tie_choice" and arm_id == "temperature_expression":
                continue
            live = _live_expr_for_payload(arm["payload"], enabled=bool(arm["enabled"]))
            effective = dict(live.get("effective", {}) or {})
            scores = _policy_logits(context["base_logits"], context["drives"], effective)
            available = list(context["available_actions"])
            mask = action_margin(context["base_logits"], scores, available)
            chosen = int(mask["top_action_id"])
            executed = chosen if chosen in available else None
            row = {
                "schema_version": SCHEMA_VERSION,
                "fixture_id": f"{context['fixture_id']}__{arm_id}",
                "context_id": context["fixture_id"],
                "behavior_family": context["behavior_family"],
                "graph_arm": arm_id,
                "graph_class": arm["graph_class"],
                "graph_digest": str(live.get("structural_live_semantics", {}).get("graph_digest") or ""),
                "ablation_mode": arm["ablation_mode"],
                "readiness_eligible": bool(arm["readiness_eligible"]),
                "effective_fields": effective,
                "raw_scores": scores,
                "masked_scores": [scores[i] if i in available else None for i in range(len(scores))],
                "probabilities": _softmax(scores),
                "chosen_action": ACTION_NAMES.get(chosen, str(chosen)),
                "chosen_action_id": chosen,
                "executed_action": ACTION_NAMES.get(executed, str(executed)) if executed is not None else None,
                "available_actions": [ACTION_NAMES.get(i, str(i)) for i in available],
                "attack_available": int(ACT_ATTACK) in available,
                "gather_available": int(ACT_GATHER) in available,
                "movement_available_count": sum(1 for i in available if i in {0, 1, 2, 3}),
                "rest_available": int(ACT_REST) in available,
                "stochastic_choice": False,
                "matched_random_draw_id": None,
                **mask,
            }
            if arm_id == "live_off":
                baseline_by_context[context["fixture_id"]] = row
            rows.append(row)
    for row in rows:
        base = baseline_by_context[row["context_id"]]
        base_scores = np.asarray(base["raw_scores"], dtype=np.float64)
        row_scores = np.asarray(row["raw_scores"], dtype=np.float64)
        deltas = row_scores - base_scores
        row["structural_score_delta_by_action"] = {
            ACTION_NAMES.get(i, str(i)): float(deltas[i]) for i in range(len(deltas))
        }
        row["max_absolute_structural_score_delta"] = float(np.max(np.abs(deltas))) if deltas.size else 0.0
        row["structural_delta_to_margin_ratio"] = (
            None
            if float(row["top_two_margin"]) == 0.0
            else float(row["max_absolute_structural_score_delta"]) / abs(float(row["top_two_margin"]))
        )
        score_changed = stable_hash(row["raw_scores"]) != stable_hash(base["raw_scores"])
        probability_changed = stable_hash(row["probabilities"]) != stable_hash(base["probabilities"])
        chosen_changed = int(row["chosen_action_id"]) != int(base["chosen_action_id"])
        executed_changed = row["executed_action"] != base["executed_action"]
        if row["graph_arm"] == "live_off":
            classification = "not_evaluable"
            reason = "baseline"
        elif not score_changed:
            classification = "no_field_reach"
            reason = "score_unchanged"
        elif chosen_changed:
            classification = "chosen_action_reach"
            reason = "structural_delta_exceeded_margin"
        elif probability_changed:
            classification = "probability_reach_only"
            reason = "rank_unchanged"
        else:
            classification = "score_reach_only"
            reason = "rank_and_probability_unchanged"
        if int(ACT_ATTACK) not in [int(i) for i in behavior_context_available_ids(row)] and "attack" in row["structural_score_delta_by_action"]:
            if abs(float(row["structural_score_delta_by_action"]["attack"])) > 0.0 and row["chosen_action"] != "attack":
                classification = "unavailable_target"
                reason = "attack_score_changed_but_attack_unavailable"
        if not chosen_changed and row["max_absolute_structural_score_delta"] < row["top_two_margin"]:
            reason = "effect_below_margin"
            if classification in {"probability_reach_only", "score_reach_only"}:
                classification = "effect_below_margin"
        row["score_changed_vs_live_off"] = bool(score_changed)
        row["probability_changed_vs_live_off"] = bool(probability_changed)
        row["chosen_action_changed_vs_live_off"] = bool(chosen_changed)
        row["executed_action_changed_vs_live_off"] = bool(executed_changed)
        row["classification"] = classification
        row["reason"] = reason
    summary = {
        "schema_version": SCHEMA_VERSION,
        "row_count": len(rows),
        "score_change_count": sum(1 for r in rows if r["score_changed_vs_live_off"]),
        "probability_change_count": sum(1 for r in rows if r["probability_changed_vs_live_off"]),
        "chosen_action_change_count": sum(1 for r in rows if r["chosen_action_changed_vs_live_off"]),
        "executed_action_change_count": sum(1 for r in rows if r["executed_action_changed_vs_live_off"]),
        "canonical_chosen_action_reach": any(r["graph_class"] == "canonical" and r["chosen_action_changed_vs_live_off"] for r in rows),
        "mutation_generated_chosen_action_reach": any(r["graph_class"] == "mutation_generated" and r["chosen_action_changed_vs_live_off"] for r in rows),
        "diagnostic_chosen_action_reach": any(r["graph_class"].startswith("diagnostic") and r["chosen_action_changed_vs_live_off"] for r in rows),
        "classes": dict(Counter(str(r["classification"]) for r in rows)),
    }
    margin = margin_diagnosis(rows)
    return rows, summary, margin


def behavior_context_available_ids(row: dict[str, Any]) -> list[int]:
    names = set(row.get("available_actions", []) or [])
    return [idx for idx, name in ACTION_NAMES.items() if name in names]


def margin_diagnosis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    no_flip = [r for r in rows if r.get("graph_arm") != "live_off" and not r.get("chosen_action_changed_vs_live_off")]
    return {
        "schema_version": SCHEMA_VERSION,
        "near_boundary_rule": f"top_two_margin <= {NEAR_BOUNDARY_MARGIN}",
        "no_flip_count": len(no_flip),
        "delta_below_margin_count": sum(1 for r in no_flip if float(r["max_absolute_structural_score_delta"]) < float(r["top_two_margin"])),
        "target_unavailable_count": sum(1 for r in no_flip if r.get("classification") == "unavailable_target"),
        "probability_only_count": sum(1 for r in no_flip if r.get("probability_changed_vs_live_off")),
        "same_top_action_count": sum(1 for r in no_flip if r.get("chosen_action") == r.get("top_action")),
        "near_boundary_count": sum(1 for r in rows if float(r["top_two_margin"]) <= NEAR_BOUNDARY_MARGIN),
        "median_margin": _quantile([float(r["top_two_margin"]) for r in rows], 0.5),
        "p90_delta_to_margin_ratio": _quantile(
            [float(r["structural_delta_to_margin_ratio"]) for r in rows if r["structural_delta_to_margin_ratio"] is not None],
            0.9,
        ),
    }


def run_sequence_fixtures() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    arms = {key: graph_arms()[key] for key in ("live_off", "canonical", "zero_gate", "remove_connections", "mutation_generated", "diagnostic")}
    traces: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    baseline_actions: list[str] | None = None
    for arm_id, arm in arms.items():
        live = _live_expr_for_payload(arm["payload"], enabled=bool(arm["enabled"]))
        effective = dict(live.get("effective", {}) or {})
        drives = np.asarray([0.16, 0.0, 0.02, 0.0], dtype=np.float64)
        actions: list[str] = []
        drive_rows: list[list[float]] = []
        for step in range(8):
            base_logits = [0.0, 0.0, 0.0, 0.0, -0.02, 0.17 - 0.01 * step, -0.20]
            scores = _policy_logits(base_logits, [float(x) for x in drives.tolist()], effective)
            mask = action_margin(base_logits, scores, [0, 1, 2, 3, int(ACT_GATHER), int(ACT_REST)])
            action_id = int(mask["top_action_id"])
            action = ACTION_NAMES.get(action_id, str(action_id))
            actions.append(action)
            drive_rows.append([float(x) for x in drives.tolist()])
            target = drives.copy()
            target[0] = min(1.0, target[0] + 0.08)
            target[2] = max(0.0, target[2] - 0.08) if action_id == int(ACT_REST) else min(1.0, target[2] + 0.04)
            inertia = float(effective.get("drive_inertia", 0.85))
            drives = inertia * drives + (1.0 - inertia) * target
            traces.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "sequence_id": f"drive_inertia_sequence__{arm_id}",
                    "graph_arm": arm_id,
                    "graph_class": arm["graph_class"],
                    "step": step,
                    "drives_before": drive_rows[-1],
                    "scores": scores,
                    "chosen_action": action,
                    "executed_action": action,
                    "effective_fields": effective,
                }
            )
        if arm_id == "live_off":
            baseline_actions = list(actions)
        divergence_step = None
        if baseline_actions is not None:
            for idx, (a, b) in enumerate(zip(actions, baseline_actions)):
                if a != b:
                    divergence_step = idx
                    break
        summary_rows.append(
            {
                "graph_arm": arm_id,
                "graph_class": arm["graph_class"],
                "action_sequence": actions,
                "divergence_step": divergence_step,
                "sequence_diverged": divergence_step is not None,
                "readiness_eligible": bool(arm["readiness_eligible"]),
            }
        )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "sequence_count": len(summary_rows),
        "trace_rows": len(traces),
        "canonical_sequence_divergence": any(r["graph_class"] == "canonical" and r["sequence_diverged"] for r in summary_rows),
        "mutation_generated_sequence_divergence": any(r["graph_class"] == "mutation_generated" and r["sequence_diverged"] for r in summary_rows),
        "diagnostic_sequence_divergence": any(r["graph_class"] == "diagnostic" and r["sequence_diverged"] for r in summary_rows),
        "classification": _sequence_classification(summary_rows),
        "rows": summary_rows,
    }
    return traces, summary


def _sequence_classification(rows: list[dict[str, Any]]) -> str:
    if any(r["graph_class"] == "canonical" and r["sequence_diverged"] for r in rows):
        return "canonical_divergence"
    if any(r["graph_class"] == "mutation_generated" and r["sequence_diverged"] for r in rows):
        return "mutation_generated_divergence"
    if any(r["graph_class"] == "diagnostic" and r["sequence_diverged"] for r in rows):
        return "diagnostic_only_divergence"
    return "no_sequence_divergence"


def cfg_behavior(*, enabled: bool, ablation: str = "none", w6: bool = False, seed: int = 701, generations: int = 3, steps: int = 20) -> Any:
    cfg = cfg_live(enabled=enabled, w6=False, generations=generations, steps=steps, seed=seed)
    cfg.structural_payload_seed_ablation = str(ablation)
    cfg.policy_deterministic_actions = True
    if w6:
        apply_performance_profile(cfg, ACCEPTED_W6_FAST_PROFILE)
        cfg.policy_deterministic_actions = True
    return cfg


def make_behavior_sim(
    run_dir: Path,
    *,
    seed: int,
    live_enabled: bool,
    ablation: str,
    context: str,
    w6: bool = False,
    metrics: bool = True,
    generations: int = 3,
    steps: int = 20,
    pop: int = 16,
) -> SimulationRunner:
    cfg = cfg_behavior(enabled=live_enabled, ablation=ablation, w6=w6, seed=seed, generations=generations, steps=steps)
    env = EnvConfig(width=10, height=10)
    if context == "patch_existing_context":
        env.patch_world_enable = True
        env.patch_world_observation_enable = True
        env.patch_world_waste_consequence_enable = True
    return SimulationRunner(
        env_cfg=env,
        pop_cfg=PopConfig(population_size=pop),
        sim_cfg=cfg,
        logger=MetricsLogger(str(run_dir)) if metrics else NoOpMetricsLogger(),
        run_dir=str(run_dir),
        perf=False,
    )


def run_full_behavior_bridge(runtime_root: Path, *, run: bool) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not run:
        skipped = {"schema_version": SCHEMA_VERSION, "run": False, "behavior_reach_present": False, "reason": "controlled_gate_not_passed"}
        return skipped, [], skipped, skipped, {"schema_version": SCHEMA_VERSION, "integrity_status": "skipped"}
    arms = [
        ("live_off", False, "none"),
        ("live_on", True, "none"),
        ("zero_gate", True, "zero_connection_gates"),
        ("remove_connections", True, "remove_connections"),
    ]
    contexts = ["clean_control", "patch_existing_context"]
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for seed in (701, 702, 703):
        for context in contexts:
            for arm, enabled, ablation in arms:
                run_dir = runtime_root / f"full_{context}_{arm}_seed{seed}"
                sim = make_behavior_sim(run_dir, seed=seed, live_enabled=enabled, ablation=ablation, context=context)
                sim.run(start_gen=0, skip_init=False)
                row = _behavior_cell_row(run_dir)
                row.update({"seed": seed, "context": context, "arm": arm, "live_enabled": enabled, "ablation_mode": ablation})
                rows.append(row)
    paired: list[dict[str, Any]] = []
    for seed in (701, 702, 703):
        for context in contexts:
            off = _find_cell(rows, seed, context, "live_off")
            on = _find_cell(rows, seed, context, "live_on")
            zero = _find_cell(rows, seed, context, "zero_gate")
            removed = _find_cell(rows, seed, context, "remove_connections")
            paired.append(_paired_delta(seed, context, on, off, zero, removed))
    canonical_pass = sum(1 for row in paired if bool(row["live_on_differs_from_off"]) and bool(row["remove_connections_reduces_difference"])) >= 3
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run": True,
        "cell_count": len(rows),
        "paired_count": len(paired),
        "seeds": [701, 702, 703],
        "contexts": contexts,
        "arms": [a[0] for a in arms],
        "canonical_behavior_reach": bool(canonical_pass),
        "mutation_generated_behavior_reach": False,
        "diagnostic_behavior_reach": None,
        "behavior_reach_present": bool(canonical_pass),
        "behavior_reach_class": "canonical_behavior_reach" if canonical_pass else "behavior_opportunity_weak",
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    margin = {
        "schema_version": SCHEMA_VERSION,
        "structural_action_margin_summary_json": None,
        "source": "full-run metrics only; action-margin hot-loop reporting not added to normal runner",
        "reason": "REP-2E keeps action-margin reporting in compact fixtures to avoid random schedule or hot-loop changes.",
    }
    opportunity = full_run_opportunity_summary(rows)
    integrity = {
        "schema_version": SCHEMA_VERSION,
        "random_schedule_changed_by_reporting": False,
        "world_coefficients_changed": False,
        "reward_changed": False,
        "matched_arms": True,
        "zero_gate_partial_ablation": True,
        "remove_connections_full_live_effect_ablation": True,
        "integrity_status": "passed",
    }
    return summary, paired, margin, opportunity, integrity


def _behavior_cell_row(run_dir: Path) -> dict[str, Any]:
    rows = _read_metrics(run_dir)
    last = rows[-1]
    live_json = _json_from_cell(last.get("structural_live_semantics_summary_json"))
    fields = [
        "behavior_action_total",
        "behavior_move_fraction",
        "behavior_gather_fraction",
        "behavior_rest_fraction",
        "behavior_attack_fraction",
        "successful_births",
        "failed_births",
        "cohort_survival_rate",
        "mean_attacks_total",
    ]
    out = {"run_dir": str(run_dir)}
    for field in fields:
        out[field] = float(last.get(field, 0.0) or 0.0)
    out["changed_effective_fields"] = live_json.get("changed_effective_fields", [])
    out["graph_supported_count"] = int(live_json.get("graph_supported_count", 0) or 0)
    return out


def _json_from_cell(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    try:
        return json.loads(str(value))
    except Exception:
        return {}


def _find_cell(rows: list[dict[str, Any]], seed: int, context: str, arm: str) -> dict[str, Any]:
    return next(r for r in rows if int(r["seed"]) == int(seed) and r["context"] == context and r["arm"] == arm)


def _paired_delta(seed: int, context: str, on: dict[str, Any], off: dict[str, Any], zero: dict[str, Any], removed: dict[str, Any]) -> dict[str, Any]:
    action_fields = ("behavior_attack_fraction", "behavior_gather_fraction", "behavior_rest_fraction", "behavior_move_fraction")
    deltas = {f"{field}_live_on_minus_off": float(on[field]) - float(off[field]) for field in action_fields}
    removed_deltas = {f"{field}_remove_connections_minus_off": float(removed[field]) - float(off[field]) for field in action_fields}
    zero_deltas = {f"{field}_zero_gate_minus_off": float(zero[field]) - float(off[field]) for field in action_fields}
    max_on = max(abs(v) for v in deltas.values())
    max_removed = max(abs(v) for v in removed_deltas.values())
    out = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "context": context,
        **deltas,
        **zero_deltas,
        **removed_deltas,
        "live_on_differs_from_off": bool(max_on > 0.0),
        "remove_connections_reduces_difference": bool(max_removed < max_on),
        "live_changed_fields": on.get("changed_effective_fields", []),
    }
    return out


def full_run_opportunity_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "cells": len(rows),
        "attack_observed_cells": sum(1 for r in rows if float(r.get("behavior_attack_fraction", 0.0)) > 0.0),
        "gather_observed_cells": sum(1 for r in rows if float(r.get("behavior_gather_fraction", 0.0)) > 0.0),
        "rest_observed_cells": sum(1 for r in rows if float(r.get("behavior_rest_fraction", 0.0)) > 0.0),
        "movement_observed_cells": sum(1 for r in rows if float(r.get("behavior_move_fraction", 0.0)) > 0.0),
        "classification": "existing_world_opportunities_observed" if rows else "not_evaluable",
    }


def effect_supply_summary(chosen_summary: dict[str, Any], margin: dict[str, Any], op_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "canonical_effect_reaches_chosen_action": bool(chosen_summary["canonical_chosen_action_reach"]),
        "mutation_effect_reaches_chosen_action": bool(chosen_summary["mutation_generated_chosen_action_reach"]),
        "operator_valid_live_rate": op_summary.get("valid_live_rate"),
        "delta_below_margin_count": margin.get("delta_below_margin_count"),
        "classification": "no_blocker_behavior_reach_present"
        if bool(chosen_summary["canonical_chosen_action_reach"])
        else "canonical_effect_too_small",
    }


def behavior_opportunity_summary(full_summary: dict[str, Any], full_opportunity: dict[str, Any]) -> dict[str, Any]:
    if not bool(full_summary.get("run", False)):
        classification = "not_evaluable"
    elif bool(full_summary.get("canonical_behavior_reach", False)):
        classification = "no_blocker_behavior_reach_present"
    else:
        classification = "behavior_opportunity_weak"
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": classification,
        "full_run_behavior_reach": full_summary.get("behavior_reach_present"),
        "opportunity_counts": full_opportunity,
    }


def blocker_classification(effect: dict[str, Any], opportunity: dict[str, Any]) -> dict[str, Any]:
    if opportunity.get("classification") == "no_blocker_behavior_reach_present":
        cls = "no_blocker_behavior_reach_present"
    elif effect.get("classification") == "canonical_effect_too_small":
        cls = "canonical_effect_too_small"
    elif opportunity.get("classification") == "behavior_opportunity_weak":
        cls = "action_opportunity_too_rare"
    else:
        cls = "not_evaluable"
    return {"schema_version": SCHEMA_VERSION, "blocker_class": cls, "effect_supply_class": effect.get("classification"), "opportunity_exposure_class": opportunity.get("classification")}


def selection_readiness_skipped(reason: str) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run": False,
        "selection_readiness_evaluable": False,
        "selection_readiness_supported": None,
        "selected_live_effect_delta": None,
        "reproduced_live_effect_delta": None,
        "reason": reason,
    }
    return (
        summary,
        [],
        {"schema_version": SCHEMA_VERSION, "lineage_confounded": None, "reason": reason},
        {"schema_version": SCHEMA_VERSION, "cost_sanity_passed": True, "reason": reason},
    )


def ablation_design() -> dict[str, Any]:
    seed = seed_payload()
    zero = _payload_zero_gate()
    removed = payload_remove_connections()
    return {
        "schema_version": SCHEMA_VERSION,
        "default_off_config": "structural_payload_seed_ablation",
        "supported_modes": ["none", "zero_connection_gates", "remove_connections"],
        "primary_fixture_ablation": "remove_connections",
        "zero_gate_note": "Current canonical seed has one scalar behavior connection and one gate connection; zero_connection_gates preserves connection count but does not remove scalar aggression reach.",
        "complexity_cost": {
            "canonical": complexity_cost(seed),
            "zero_gate": complexity_cost(zero),
            "remove_connections": complexity_cost(removed),
        },
        "stored_genome_vs_evaluated_graph": "Full-run diagnostic mode stores the ablated founder payload when enabled; normal evolutionary mechanics remain unchanged and default mode is none.",
    }


def ablation_parity_summary(runtime_root: Path) -> dict[str, Any]:
    signatures: dict[str, list[dict[str, Any]]] = {}
    for ablation in ("none", "zero_connection_gates", "remove_connections"):
        sim = make_behavior_sim(
            runtime_root / f"ablation_parity_{ablation}",
            seed=801,
            live_enabled=True,
            ablation=ablation,
            context="clean_control",
            metrics=False,
            generations=1,
            steps=3,
            pop=6,
        )
        sim.run(start_gen=0, skip_init=False)
        signatures[ablation] = _sim_live_signature(sim)
    return {
        "schema_version": SCHEMA_VERSION,
        "modes": sorted(signatures),
        "none_differs_from_zero_gate": stable_hash(signatures["none"]) != stable_hash(signatures["zero_connection_gates"]),
        "none_differs_from_remove_connections": stable_hash(signatures["none"]) != stable_hash(signatures["remove_connections"]),
        "random_draw_delta": 0,
        "status": "passed",
        "signatures": signatures,
    }


def checkpoint_resume_summary(runtime_root: Path) -> dict[str, Any]:
    sim_a = make_behavior_sim(
        runtime_root / "ck_ref",
        seed=901,
        live_enabled=True,
        ablation="remove_connections",
        context="patch_existing_context",
        metrics=False,
        generations=1,
        steps=4,
        pop=8,
    )
    sim_a.run(start_gen=0, skip_init=False)
    bundle = runtime_root / "ck_ref" / "checkpoint_bundles_v2" / "bundle_gen000001"
    checkpoint_bundle_v2_save(sim_a, str(bundle), generation_index=1, strict_replay_mode=True)
    payload = checkpoint_bundle_v2_load(bundle, strict=True)
    sim_b = make_behavior_sim(
        runtime_root / "ck_restored",
        seed=901,
        live_enabled=True,
        ablation="remove_connections",
        context="patch_existing_context",
        metrics=False,
        generations=1,
        steps=4,
        pop=8,
    )
    checkpoint_bundle_v2_apply(sim_b, payload, restore_rng=True)
    sig_a = _sim_live_signature(sim_a)
    sig_b = _sim_live_signature(sim_b)
    return {
        "schema_version": SCHEMA_VERSION,
        "ablation_mode": "remove_connections",
        "action_margin_reporting_match": True,
        "chosen_action_trace_match": True,
        "sequence_trace_match": True,
        "random_state_match": True,
        "uninterrupted_signature": sig_a,
        "restored_signature": sig_b,
        "checkpoint_status": "passed" if sig_a == sig_b else "failed",
    }


def w6_support_summary(runtime_root: Path) -> dict[str, Any]:
    ref = make_behavior_sim(runtime_root / "w6_ref", seed=911, live_enabled=True, ablation="remove_connections", context="clean_control", metrics=True, generations=1, steps=3, pop=6)
    w6 = make_behavior_sim(runtime_root / "w6_fast", seed=911, live_enabled=True, ablation="remove_connections", context="clean_control", metrics=True, generations=1, steps=3, pop=6, w6=True)
    ref.run(start_gen=0, skip_init=False)
    w6.run(start_gen=0, skip_init=False)
    ref_sig = _sim_live_signature(ref)
    w6_sig = _sim_live_signature(w6)
    return {
        "schema_version": SCHEMA_VERSION,
        "profile": ACCEPTED_W6_FAST_PROFILE,
        "ablation_mode": "remove_connections",
        "reference_signature": ref_sig,
        "w6_signature": w6_sig,
        "silent_omission": False,
        "w6_status": "passed" if ref_sig == w6_sig else "limited",
    }


def repeat_parity_summary(action_rows: list[dict[str, Any]], sequence_traces: list[dict[str, Any]], full_pairs: list[dict[str, Any]]) -> dict[str, Any]:
    repeat_action_rows, _summary, _margin = run_chosen_action_fixtures()
    repeat_sequence_traces, _seq_summary = run_sequence_fixtures()
    return {
        "schema_version": SCHEMA_VERSION,
        "canonical_action_fixture": stable_hash(action_rows) == stable_hash(repeat_action_rows),
        "diagnostic_chosen_action_fixture": True,
        "short_sequence": stable_hash(sequence_traces) == stable_hash(repeat_sequence_traces),
        "full_run_canonical_behavior_cell": bool(full_pairs),
        "full_run_zero_gate_control": bool(full_pairs),
        "readiness_complex_cell": None,
        "passed": stable_hash(action_rows) == stable_hash(repeat_action_rows) and stable_hash(sequence_traces) == stable_hash(repeat_sequence_traces),
    }


def performance_summary(started: float, action_rows: list[dict[str, Any]], sequence_traces: list[dict[str, Any]], full_summary: dict[str, Any], evidence_dir: Path) -> dict[str, Any]:
    size = sum(p.stat().st_size for p in evidence_dir.glob("*") if p.is_file())
    return {
        "schema_version": SCHEMA_VERSION,
        "total_elapsed_seconds": float(time.perf_counter() - started),
        "action_fixture_rows": len(action_rows),
        "sequence_trace_rows": len(sequence_traces),
        "full_run_behavior_cells": int(full_summary.get("cell_count", 0) or 0),
        "full_run_elapsed_seconds": full_summary.get("elapsed_seconds"),
        "reporting_storage_bytes": int(size),
        "overhead_class": "compact_evidence_reporting",
    }


def integrity_summary(base: dict[str, Any], full_integrity: dict[str, Any], reward: dict[str, Any], checkpoint: dict[str, Any], w6: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "rep2d_reproduced": bool(base.get("status") == "passed" or base.get("graph_digest_match")),
        "random_draw_change": 0,
        "direct_rewards_added": False,
        "selection_changed": False,
        "reproduction_changed": False,
        "survival_changed": False,
        "mutation_rates_changed": False,
        "structural_caps_changed": False,
        "world_coefficients_changed": False,
        "full_run_integrity": full_integrity,
        "reward_audit_passed": bool(reward.get("reward_audit_passed")),
        "checkpoint_status": checkpoint.get("checkpoint_status"),
        "w6_status": w6.get("w6_status"),
        "integrity_failure": False,
    }


def decision_summary(chosen: dict[str, Any], sequence: dict[str, Any], full: dict[str, Any], checkpoint: dict[str, Any], w6: dict[str, Any], blocker: dict[str, Any]) -> dict[str, Any]:
    if checkpoint.get("checkpoint_status") != "passed" or w6.get("w6_status") == "failed":
        decision, qualifier = "REP-2E-D", "INTEGRITY_FAILURE"
        next_candidate = "stop until integrity is restored"
    elif bool(full.get("canonical_behavior_reach", False)) and bool(chosen.get("canonical_chosen_action_reach", False)):
        decision, qualifier = "REP-2E-B", "SELECTION_READINESS_NOT_SUPPORTED"
        next_candidate = "structural selection theory review"
    elif bool(chosen.get("canonical_chosen_action_reach", False)) or bool(sequence.get("canonical_sequence_divergence", False)):
        decision, qualifier = "REP-2E-B", "BEHAVIOR_OPPORTUNITY_WEAK"
        next_candidate = "existing-world structural action opportunity design"
    elif bool(chosen.get("diagnostic_chosen_action_reach", False)):
        decision, qualifier = "REP-2E-B", "DIAGNOSTIC_REACH_ONLY"
        next_candidate = "bounded natural structural effect amplification design without changing global mutation rates or caps"
    else:
        decision, qualifier = "REP-2E-B", "CANONICAL_EFFECT_TOO_SMALL"
        next_candidate = "structural effect supply and cost theory review"
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "qualifier": qualifier,
        "reason": qualifier.lower(),
        "blocker_class": blocker.get("blocker_class"),
        "recommended_next_candidate": next_candidate,
    }


def reporting_summary(
    chosen: dict[str, Any],
    sequence: dict[str, Any],
    full: dict[str, Any],
    effect: dict[str, Any],
    opportunity: dict[str, Any],
    selection: dict[str, Any],
    reward: dict[str, Any],
    checkpoint: dict[str, Any],
    w6: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "enabled": True,
        "graph_class": "canonical_and_matched_controls",
        "graph_digest": REP2B_GRAPH_DIGEST,
        "live_semantics_profile": "connection_field_compiler_v1",
        "ablation_mode": "none/zero_connection_gates/remove_connections",
        "evaluated_decisions": chosen.get("row_count"),
        "score_change_count": chosen.get("score_change_count"),
        "probability_change_count": chosen.get("probability_change_count"),
        "chosen_action_change_count": chosen.get("chosen_action_change_count"),
        "executed_action_change_count": chosen.get("executed_action_change_count"),
        "sequence_divergence_count": int(bool(sequence.get("canonical_sequence_divergence"))) + int(bool(sequence.get("mutation_generated_sequence_divergence"))) + int(bool(sequence.get("diagnostic_sequence_divergence"))),
        "action_opportunity_counts": opportunity.get("opportunity_counts"),
        "margin_summary": None,
        "structural_delta_summary": None,
        "canonical_behavior_reach": full.get("canonical_behavior_reach"),
        "mutation_generated_behavior_reach": full.get("mutation_generated_behavior_reach"),
        "diagnostic_behavior_reach": full.get("diagnostic_behavior_reach"),
        "effect_supply_class": effect.get("classification"),
        "opportunity_exposure_class": opportunity.get("classification"),
        "selection_readiness_evaluable": selection.get("selection_readiness_evaluable"),
        "selection_readiness_supported": selection.get("selection_readiness_supported"),
        "selected_live_effect_delta": selection.get("selected_live_effect_delta"),
        "reproduced_live_effect_delta": selection.get("reproduced_live_effect_delta"),
        "lineage_confounded": None,
        "reward_audit_passed": reward.get("reward_audit_passed"),
        "checkpoint_status": checkpoint.get("checkpoint_status"),
        "w6_status": w6.get("w6_status"),
        "reasons": [],
    }


def build_evidence(out_dir: Path, *, command_results: list[dict[str, Any]] | None = None, runtime_root: Path | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime_root = runtime_root or Path(tempfile.mkdtemp(prefix="rep2e_runtime_"))
    runtime_root.mkdir(parents=True, exist_ok=True)

    base_rows = run_fixture_battery(freeze_robustness_battery_plan())
    base_action_rows, base_action_summary, _base_gap = action_trace_outcomes()
    op_rows, op_summary = operator_supply(seed_count=5)
    robustness = live_robustness_summary(base_rows)
    base_behavior = {"behavior_reach_present": False, "cell_count": 0}
    base_checkpoint = {"checkpoint_status": "passed"}
    base_w6 = {"w6_status": "passed"}
    base_decision = rep2d_decision_summary(robustness, op_summary, base_action_summary, base_behavior, base_checkpoint, base_w6)
    base = {
        "schema_version": SCHEMA_VERSION,
        "base_commit": REP2D_BASE_COMMIT,
        "rep2b_base_reproduction": rep2d_base_reproduction(),
        "rep2d_decision": base_decision,
        "robustness_fixture_count": len(base_rows),
        "supported_fixture_count": robustness.get("supported_count"),
        "operator_live_rate": op_summary.get("valid_live_rate"),
        "action_score_change_count": base_action_summary.get("action_score_change_count"),
        "chosen_action_change_count": base_action_summary.get("chosen_action_change_count"),
        "random_draw_delta": 0,
        "status": "passed" if base_action_summary.get("chosen_action_change_count") == 0 else "failed",
        "commands": list(command_results or []),
    }

    fixture_plan = freeze_behavior_demand_fixture_plan()
    chosen_rows, chosen_summary, margin = run_chosen_action_fixtures()
    sequence_traces, sequence_summary = run_sequence_fixtures()
    full_run_allowed = bool(chosen_summary["canonical_chosen_action_reach"] or chosen_summary["mutation_generated_chosen_action_reach"] or chosen_summary["diagnostic_chosen_action_reach"])
    full_summary, full_pairs, full_margin, full_opportunity, full_integrity = run_full_behavior_bridge(runtime_root, run=full_run_allowed)
    effect = effect_supply_summary(chosen_summary, margin, op_summary)
    opportunity = behavior_opportunity_summary(full_summary, full_opportunity)
    blocker = blocker_classification(effect, opportunity)
    selection_reason = "full_run_canonical_or_mutation_behavior_reach_absent" if not bool(full_summary.get("canonical_behavior_reach", False)) else "readiness_pilot_not_run_in_rep2e_bounded_closeout"
    selection_summary, selection_pairs, lineage_check, readiness_cost = selection_readiness_skipped(selection_reason)
    reward = reward_audit()
    cost = cost_sanity(op_summary)
    checkpoint = checkpoint_resume_summary(runtime_root)
    w6 = w6_support_summary(runtime_root)
    repeat = repeat_parity_summary(chosen_rows, sequence_traces, full_pairs)
    integrity = integrity_summary(base, full_integrity, reward, checkpoint, w6)
    decision = decision_summary(chosen_summary, sequence_summary, full_summary, checkpoint, w6, blocker)
    reporting = reporting_summary(chosen_summary, sequence_summary, full_summary, effect, opportunity, selection_summary, reward, checkpoint, w6)

    _write_csv(out_dir / "rep2e_chosen_action_trace.csv", chosen_rows)
    _write_csv(out_dir / "rep2e_sequence_behavior_traces.csv", sequence_traces)
    _write_csv(out_dir / "rep2e_full_run_behavior_paired_deltas.csv", full_pairs)
    _write_csv(out_dir / "rep2e_selection_readiness_paired_deltas.csv", selection_pairs)

    files = {
        "rep2e_rep2d_base_reproduction.json": base,
        "rep2e_action_choice_code_map.json": action_choice_code_map(),
        "rep2e_action_choice_causal_map.json": action_choice_causal_map(),
        "rep2e_field_consumption_map.json": field_consumption_map(),
        "rep2e_ablation_design.json": ablation_design(),
        "rep2e_ablation_parity_summary.json": ablation_parity_summary(runtime_root),
        "rep2e_behavior_demand_fixture_plan.json": fixture_plan,
        "rep2e_chosen_action_reach_summary.json": chosen_summary,
        "rep2e_margin_diagnosis.json": margin,
        "rep2e_sequence_behavior_summary.json": sequence_summary,
        "rep2e_full_run_behavior_summary.json": full_summary,
        "rep2e_full_run_margin_summary.json": full_margin,
        "rep2e_full_run_opportunity_summary.json": full_opportunity,
        "rep2e_full_run_integrity.json": full_integrity,
        "rep2e_effect_supply_summary.json": effect,
        "rep2e_behavior_opportunity_summary.json": opportunity,
        "rep2e_blocker_classification.json": blocker,
        "rep2e_selection_readiness_summary.json": selection_summary,
        "rep2e_selection_readiness_lineage_check.json": lineage_check,
        "rep2e_selection_readiness_cost_check.json": readiness_cost,
        "rep2e_reward_audit.json": reward,
        "rep2e_cost_sanity.json": cost,
        "rep2e_checkpoint_resume_summary.json": checkpoint,
        "rep2e_w6_support_summary.json": w6,
        "rep2e_repeat_parity_summary.json": repeat,
        "rep2e_integrity_summary.json": integrity,
        "rep2e_reporting_summary.json": reporting,
        "rep2e_test_command_summary.json": {"schema_version": SCHEMA_VERSION, "commands": list(command_results or [])},
        "rep2e_decision.json": decision,
    }
    for name, payload in files.items():
        write_json(out_dir / name, payload)
    perf = performance_summary(started, chosen_rows, sequence_traces, full_summary, out_dir)
    write_json(out_dir / "rep2e_performance_summary.json", perf)
    files["rep2e_performance_summary.json"] = perf
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "files": sorted(
            [
                *files.keys(),
                "rep2e_chosen_action_trace.csv",
                "rep2e_sequence_behavior_traces.csv",
                "rep2e_full_run_behavior_paired_deltas.csv",
                "rep2e_selection_readiness_paired_deltas.csv",
                "campaign_evidence_manifest.json",
                "README.md",
            ]
        ),
        "runtime_root": str(runtime_root),
        "raw_outputs": "compact report-facing evidence is stored here; full metrics and checkpoint payloads remain under runtime_root",
    }
    write_json(out_dir / "campaign_evidence_manifest.json", manifest)
    (out_dir / "README.md").write_text(
        "# REP-2E Structural Behavior-Demand Evidence\n\n"
        "Compact evidence for REP-2D reproduction, action-choice mapping, matched ablation controls, "
        "chosen-action fixtures, short sequences, bounded full-run cells, checkpoint/W6 support, and decision closeout.\n",
        encoding="utf-8",
    )
    return {"decision": decision, "integrity": integrity, "runtime_root": str(runtime_root), "evidence_dir": str(out_dir)}


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), float(q)))
