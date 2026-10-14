from __future__ import annotations

"""Default-off structural action opportunity telemetry."""

import hashlib
import json
from collections import Counter
from typing import Any

import numpy as np

from env.gridworld_pkg.constants import ACT_ATTACK, ACT_GATHER, ACT_MOVE_E, ACT_MOVE_N, ACT_MOVE_S, ACT_MOVE_W, ACT_REST, FOOD, N_ACTIONS
from evolution.policy_action_family import (
    ACTION_ID_TO_NAME,
    FACTORIZED_SHADOW_SCHEMA_VERSION,
    FAMILY_ORDER,
    FAMILY_TO_ACTION_IDS,
    action_family_or_unknown,
    factorized_family_shadow_payload,
    family_probability_payload,
)
from agents.family_head_residuals import FAMILY_HEAD_RESIDUAL_FIELD, family_head_residual_effective_map
from evolution.structural_live_semantics import ACTION_REACHABLE_FIELDS, FIELD_BOUNDS


SCHEMA_VERSION = "structural_action_opportunity_v2"
SCHEMA_VERSION_V1 = "structural_action_opportunity_v1"
DECISION_MARGIN_SCHEMA_VERSION = "structural_decision_margin_v1"
POLICY_ACTION_FAMILY_DECISION_SCHEMA_VERSION = "structural_policy_action_family_decision_event_v1"
POLICY_ACTION_EXECUTION_SCHEMA_VERSION = "policy_action_execution_event_v1"
POLICY_FAMILY_HEAD_SHADOW_SUMMARY_SCHEMA_VERSION = "factorized_family_shadow_summary_v1"
FAMILY_HEAD_AUTHORITY_CHOICE_MARGIN_SCHEMA_VERSION = "family_head_authority_choice_margin_v1"
FAMILY_HEAD_AUTHORITY_MAGNITUDE_SURFACE_SCHEMA_VERSION = "family_head_residual_magnitude_surface_v1"
FAMILY_HEAD_AUTHORITY_MAGNITUDE_SURFACE_ALPHAS = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0)
OPPORTUNITY_CONDITIONED_AUTHORITY_CANDIDATE_SCHEMA_VERSION = "opportunity_conditioned_family_authority_candidate_v1"
OPPORTUNITY_CONDITIONED_AUTHORITY_CANDIDATES = (
    "current_unconditioned",
    "binary_executable_gate",
    "bounded_opportunity_intensity_gate",
    "existing_drive_gate",
    "no_policy_gate",
)
OPPORTUNITY_CONDITIONED_AUTHORITY_L1_VALUES = (0.2, 0.6, 1.0, 2.0)
NEAR_BOUNDARY_ABSOLUTE_THRESHOLD = 0.25
CANONICAL_COUNTERFACTUAL_FIELD_DELTAS = {
    "aggression_bias": 0.05,
    "drive_logit_gain": 0.0640000016,
}
FIELD_ACTION_ROUTES = {
    "aggression_bias": ("attack",),
    "action_temperature": ("attack", "gather", "rest", "move"),
    "drive_logit_gain": ("attack", "gather", "rest", "move"),
}


COUNT_KEYS = (
    "evaluated_decisions",
    "attack_action_unmasked_count",
    "attack_target_available_count",
    "attack_executable_count",
    "attack_chosen_count",
    "attack_executed_count",
    "gather_action_unmasked_count",
    "gather_resource_available_count",
    "gather_executable_count",
    "gather_chosen_count",
    "gather_executed_count",
    "rest_action_unmasked_count",
    "rest_need_present_count",
    "rest_competitive_count",
    "rest_chosen_count",
    "rest_executed_count",
    "legal_move_count_total",
    "decisions_with_multiple_moves",
    "movement_competitive_count",
    "move_chosen_count",
    "move_executed_count",
    "drive_competition_count",
    "near_boundary_count",
    "linked_field_active_count",
    "linked_field_opportunity_count",
    "structural_delta_exceeds_margin_count",
    "chosen_action_change_count",
    "potential_linked_field_count",
    "potential_linked_action_family_count",
    "potential_linked_opportunity_count",
    "canonical_counterfactual_linked_opportunity_count",
    "margin_evaluable_count",
    "margin_not_evaluable_count",
    "near_boundary_absolute_count",
    "near_boundary_canonical_relative_count",
    "canonical_delta_exceeds_margin_count",
    "actual_live_delta_exceeds_margin_count",
    "actual_counterfactual_evaluated_count",
    "actual_counterfactual_same_action_count",
    "actual_counterfactual_action_change_count",
    "actual_counterfactual_not_evaluable_count",
)

DICT_KEYS = (
    "developmental_stage_counts",
    "unavailable_reason_counts",
    "not_evaluable_reason_counts",
    "potential_linked_opportunities_by_field",
    "potential_linked_opportunities_by_action_family",
    "canonical_counterfactual_changed_fields",
    "canonical_counterfactual_field_deltas",
    "margin_histogram",
    "reasons",
    "actual_counterfactual_changed_action_pairs",
    "actual_counterfactual_not_evaluable_reasons",
    "actual_counterfactual_evaluated_by_variant_class",
    "actual_counterfactual_changed_by_variant_class",
    "actual_counterfactual_evaluated_by_mutation_origin",
    "actual_counterfactual_changed_by_mutation_origin",
)


def empty_opportunity_summary(*, enabled: bool) -> dict[str, Any]:
    out: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "enabled": bool(enabled)}
    for key in COUNT_KEYS:
        out[key] = 0 if enabled else None
    out["developmental_stage_counts"] = {}
    out["unavailable_reason_counts"] = {}
    out["not_evaluable_reason_counts"] = {} if enabled else {"reporting_disabled": 1}
    out["potential_linked_opportunities_by_field"] = {}
    out["potential_linked_opportunities_by_action_family"] = {}
    out["canonical_counterfactual_enabled"] = bool(enabled)
    out["canonical_counterfactual_changed_fields"] = {}
    out["canonical_counterfactual_field_deltas"] = dict(CANONICAL_COUNTERFACTUAL_FIELD_DELTAS) if enabled else {}
    out["margin_histogram"] = {}
    out["margin_mean"] = None
    out["margin_min"] = None
    out["margin_max"] = None
    out["near_boundary_absolute_threshold"] = NEAR_BOUNDARY_ABSOLUTE_THRESHOLD
    out["reasons"] = {}
    out["actual_counterfactual_mode"] = "pre_connection_effective_fields" if enabled else None
    out["actual_counterfactual_changed_action_pairs"] = {}
    out["actual_counterfactual_not_evaluable_reasons"] = {}
    out["actual_counterfactual_evaluated_by_variant_class"] = {}
    out["actual_counterfactual_changed_by_variant_class"] = {}
    out["actual_counterfactual_evaluated_by_mutation_origin"] = {}
    out["actual_counterfactual_changed_by_mutation_origin"] = {}
    return out


def merge_opportunity_summary(target: dict[str, Any], delta: dict[str, Any]) -> dict[str, Any]:
    if not target:
        target.update(empty_opportunity_summary(enabled=True))
    target["enabled"] = True
    for key in COUNT_KEYS:
        target[key] = int(target.get(key, 0) or 0) + int(delta.get(key, 0) or 0)
    for field in DICT_KEYS:
        if field == "canonical_counterfactual_field_deltas":
            merged = dict(target.get(field, {}) or {})
            merged.update(delta.get(field, {}) or {})
            target[field] = dict(sorted(merged.items()))
            continue
        counter = Counter(target.get(field, {}) or {})
        counter.update(delta.get(field, {}) or {})
        target[field] = dict(sorted(counter.items()))
    margin_count = int(target.get("margin_evaluable_count", 0) or 0)
    prev_count = margin_count - int(delta.get("margin_evaluable_count", 0) or 0)
    if int(delta.get("margin_evaluable_count", 0) or 0) > 0:
        prev_sum = float(target.get("_margin_sum", 0.0) or 0.0)
        delta_sum = float(delta.get("_margin_sum", 0.0) or 0.0)
        target["_margin_sum"] = prev_sum + delta_sum
        target["margin_mean"] = float(target["_margin_sum"]) / float(max(1, margin_count))
        vals = [v for v in (target.get("margin_min"), delta.get("margin_min")) if v is not None]
        target["margin_min"] = min(vals) if vals else None
        vals = [v for v in (target.get("margin_max"), delta.get("margin_max")) if v is not None]
        target["margin_max"] = max(vals) if vals else None
    elif prev_count == 0 and margin_count == 0:
        target["margin_mean"] = None
        target["margin_min"] = None
        target["margin_max"] = None
    return target


def record_policy_margins(runner: Any, *, alive: list[Any], row_indices: list[int], logits: Any) -> None:
    if not bool(getattr(getattr(runner, "sim_cfg", None), "structural_action_margin_reporting_enable", False)):
        return
    try:
        if hasattr(logits, "detach"):
            arr = logits.detach().to("cpu").numpy().astype(np.float64, copy=False)
        else:
            arr = np.asarray(logits, dtype=np.float64)
    except Exception:
        return
    pending = getattr(runner, "_structural_action_margin_pending", None)
    if not isinstance(pending, dict):
        pending = {}
    for local_row, alive_index in enumerate(row_indices):
        if local_row >= arr.shape[0] or alive_index >= len(alive):
            continue
        row = np.asarray(arr[local_row], dtype=np.float64)
        if row.size < 2 or not np.all(np.isfinite(row)):
            continue
        order = np.argsort(row)
        top = int(order[-1])
        second = int(order[-2])
        margin = float(row[top] - row[second])
        agent = alive[alive_index]
        pending[int(getattr(agent, "agent_id"))] = {
            "top_action_id": top,
            "second_action_id": second,
            "top_score": float(row[top]),
            "second_score": float(row[second]),
            "top_two_margin": margin,
        }
    runner._structural_action_margin_pending = pending


def empty_policy_action_family_summary(*, enabled: bool) -> dict[str, Any]:
    out: dict[str, Any] = {
        "schema_version": "structural_policy_action_family_mapping_telemetry_v1",
        "enabled": bool(enabled),
        "decision_event_schema_version": POLICY_ACTION_FAMILY_DECISION_SCHEMA_VERSION if enabled else None,
        "execution_event_schema_version": POLICY_ACTION_EXECUTION_SCHEMA_VERSION if enabled else None,
        "decision_count": 0 if enabled else None,
        "execution_join_count": 0 if enabled else None,
        "execution_missing_decision_count": 0 if enabled else None,
        "action_substitution_count": 0 if enabled else None,
        "true_action_substitution_count": 0 if enabled else None,
        "chosen_applied_same_count": 0 if enabled else None,
        "prefetched_not_applied_count": 0 if enabled else None,
        "stale_prefetch_decision_count": 0 if enabled else None,
        "movement_dilution_case_count": 0 if enabled else None,
        "raw_action_observation_count": 0 if enabled else None,
        "unique_action_application_count": 0 if enabled else None,
        "duplicate_action_application_count": 0 if enabled else None,
        "unknown_source_count": 0 if enabled else None,
        "primitive_entropy_sum": 0.0 if enabled else None,
        "family_entropy_sum": 0.0 if enabled else None,
        "family_margin_sum": 0.0 if enabled else None,
        "families": list(FAMILY_ORDER),
        "source_class_counts": {},
        "decision_source_class_counts": {},
        "application_source_class_counts": {},
        "probability_status_counts": {},
        "join_status_counts": {},
        "decision_family_counts": {},
        "dominant_family_counts": {},
        "executed_family_counts": {},
        "execution_success_family_counts": {},
        "execution_failure_family_counts": {},
        "consequence_not_evaluable_family_counts": {},
        "decision_not_evaluable_reasons": {} if enabled else {"reporting_disabled": 1},
        "execution_not_evaluable_reasons": {} if enabled else {"reporting_disabled": 1},
        "family_probability_mass_sum": {family: 0.0 for family in FAMILY_ORDER} if enabled else {},
        "decision_event_sample": [],
        "execution_event_sample": [],
    }
    return out


def empty_policy_family_head_shadow_summary(*, enabled: bool) -> dict[str, Any]:
    return {
        "schema_version": POLICY_FAMILY_HEAD_SHADOW_SUMMARY_SCHEMA_VERSION,
        "shadow_schema_version": FACTORIZED_SHADOW_SCHEMA_VERSION if enabled else None,
        "enabled": bool(enabled),
        "shadow_authority": False,
        "decision_count": 0 if enabled else None,
        "evaluable_count": 0 if enabled else None,
        "not_evaluable_count": 0 if enabled else None,
        "random_draws": 0 if enabled else None,
        "action_authority_changes": 0 if enabled else None,
        "max_reconstruction_error": 0.0 if enabled else None,
        "mean_reconstruction_error_sum": 0.0 if enabled else None,
        "primitive_probability_sum_residual_abs_max": 0.0 if enabled else None,
        "family_probability_sum_residual_abs_max": 0.0 if enabled else None,
        "entropy_decomposition_residual_abs_max": 0.0 if enabled else None,
        "movement_conditional_entropy_sum": 0.0 if enabled else None,
        "movement_family_mass_sum": 0.0 if enabled else None,
        "cache_application_count": 0 if enabled else None,
        "stride_refill_count": 0 if enabled else None,
        "stride_cached_count": 0 if enabled else None,
        "prefetched_not_applied_count": 0 if enabled else None,
        "families": list(FAMILY_ORDER),
        "policy_kind_counts": {},
        "source_class_counts": {},
        "stride_mode_counts": {},
        "family_rank_counts": {},
        "chosen_family_counts": {},
        "valid_family_pattern_counts": {},
        "movement_valid_direction_count_histogram": {},
        "not_evaluable_reasons": {} if enabled else {"reporting_disabled": 1},
        "family_probability_sum": {family: 0.0 for family in FAMILY_ORDER} if enabled else {},
        "event_sample": [],
    }


def merge_policy_family_head_shadow_summary(target: dict[str, Any], delta: dict[str, Any]) -> dict[str, Any]:
    if not target:
        target.update(empty_policy_family_head_shadow_summary(enabled=True))
    target["enabled"] = True
    for key in (
        "decision_count",
        "evaluable_count",
        "not_evaluable_count",
        "random_draws",
        "action_authority_changes",
        "cache_application_count",
        "stride_refill_count",
        "stride_cached_count",
        "prefetched_not_applied_count",
    ):
        target[key] = int(target.get(key, 0) or 0) + int(delta.get(key, 0) or 0)
    for key in (
        "mean_reconstruction_error_sum",
        "movement_conditional_entropy_sum",
        "movement_family_mass_sum",
    ):
        target[key] = float(target.get(key, 0.0) or 0.0) + float(delta.get(key, 0.0) or 0.0)
    for key in (
        "max_reconstruction_error",
        "primitive_probability_sum_residual_abs_max",
        "family_probability_sum_residual_abs_max",
        "entropy_decomposition_residual_abs_max",
    ):
        target[key] = max(float(target.get(key, 0.0) or 0.0), float(delta.get(key, 0.0) or 0.0))
    for field in (
        "policy_kind_counts",
        "source_class_counts",
        "stride_mode_counts",
        "family_rank_counts",
        "chosen_family_counts",
        "valid_family_pattern_counts",
        "movement_valid_direction_count_histogram",
        "not_evaluable_reasons",
    ):
        counter = Counter(target.get(field, {}) or {})
        counter.update(delta.get(field, {}) or {})
        target[field] = dict(sorted(counter.items()))
    for family, value in (delta.get("family_probability_sum", {}) or {}).items():
        target["family_probability_sum"][str(family)] = float(target["family_probability_sum"].get(str(family), 0.0) or 0.0) + float(value or 0.0)
    limit = int(delta.get("_event_sample_limit", 0) or 0)
    if limit > 0:
        existing = list(target.get("event_sample", []) or [])
        for event in list(delta.get("event_sample", []) or []):
            if len(existing) >= limit:
                break
            existing.append(event)
        target["event_sample"] = existing
    return target


def merge_policy_action_family_summary(target: dict[str, Any], delta: dict[str, Any]) -> dict[str, Any]:
    if not target:
        target.update(empty_policy_action_family_summary(enabled=True))
    target["enabled"] = True
    for key in (
        "decision_count",
        "execution_join_count",
        "execution_missing_decision_count",
        "action_substitution_count",
        "true_action_substitution_count",
        "chosen_applied_same_count",
        "prefetched_not_applied_count",
        "stale_prefetch_decision_count",
        "movement_dilution_case_count",
        "raw_action_observation_count",
        "unique_action_application_count",
        "duplicate_action_application_count",
        "unknown_source_count",
    ):
        target[key] = int(target.get(key, 0) or 0) + int(delta.get(key, 0) or 0)
    for key in ("primitive_entropy_sum", "family_entropy_sum", "family_margin_sum"):
        target[key] = float(target.get(key, 0.0) or 0.0) + float(delta.get(key, 0.0) or 0.0)
    for key in (
        "decision_family_counts",
        "source_class_counts",
        "decision_source_class_counts",
        "application_source_class_counts",
        "probability_status_counts",
        "join_status_counts",
        "dominant_family_counts",
        "executed_family_counts",
        "execution_success_family_counts",
        "execution_failure_family_counts",
        "consequence_not_evaluable_family_counts",
        "decision_not_evaluable_reasons",
        "execution_not_evaluable_reasons",
    ):
        counter = Counter(target.get(key, {}) or {})
        counter.update(delta.get(key, {}) or {})
        target[key] = dict(sorted(counter.items()))
    mass = dict(target.get("family_probability_mass_sum", {}) or {})
    for family, value in (delta.get("family_probability_mass_sum", {}) or {}).items():
        mass[str(family)] = float(mass.get(str(family), 0.0) or 0.0) + float(value or 0.0)
    target["family_probability_mass_sum"] = {family: float(mass.get(family, 0.0) or 0.0) for family in FAMILY_ORDER}
    limit = int(delta.get("_event_sample_limit", target.get("_event_sample_limit", 512)) or 512)
    for key in ("decision_event_sample", "execution_event_sample"):
        combined = list(target.get(key, []) or [])
        for event in list(delta.get(key, []) or []):
            if len(combined) < limit:
                combined.append(event)
        target[key] = combined
    target["_event_sample_limit"] = int(limit)
    return target


def record_policy_action_family_decisions(
    runner: Any,
    *,
    alive: list[Any],
    row_indices: list[int],
    logits: Any,
    actions: Any,
    sample_noise: Any = None,
    policy_kind: str = "",
    drives: Any = None,
    action_temperature: Any = None,
    action_source_class: str = "neural_policy_direct",
) -> None:
    if not _action_family_reporting_enabled(runner):
        return
    try:
        logits_arr = np.asarray(_as_numpy(logits), dtype=np.float64)
        actions_arr = np.asarray(_as_numpy(actions), dtype=np.int32)
        if actions_arr.ndim == 1:
            actions_arr = actions_arr.reshape((-1, 1))
        elif actions_arr.ndim != 2:
            actions_arr = actions_arr.reshape((actions_arr.shape[0], -1))
    except Exception:
        _merge_policy_action_family_delta(runner, _policy_family_not_evaluable("decision_input_unavailable", len(row_indices)))
        return
    noise_arr = None
    if sample_noise is not None:
        try:
            noise_arr = np.asarray(_as_numpy(sample_noise), dtype=np.float64)
        except Exception:
            noise_arr = None
    drives_arr = None
    if drives is not None:
        try:
            drives_arr = np.asarray(_as_numpy(drives), dtype=np.float64)
        except Exception:
            drives_arr = None
    temps_arr = None
    if action_temperature is not None:
        try:
            temps_arr = np.asarray(_as_numpy(action_temperature), dtype=np.float64).reshape((-1,))
        except Exception:
            temps_arr = None

    pending = getattr(runner, "_structural_policy_action_family_pending", None)
    if not isinstance(pending, dict):
        pending = {}
    delta = empty_policy_action_family_summary(enabled=True)
    shadow_enabled = _policy_family_head_shadow_enabled(runner)
    shadow_delta = empty_policy_family_head_shadow_summary(enabled=True) if shadow_enabled else None
    limit = _policy_event_sample_limit(runner)
    delta["_event_sample_limit"] = int(limit)
    if shadow_delta is not None:
        shadow_delta["_event_sample_limit"] = int(limit) if _policy_family_head_shadow_record_events(runner) else 0
    for local_row, alive_index in enumerate(row_indices):
        if local_row >= logits_arr.shape[0] or local_row >= actions_arr.shape[0] or alive_index >= len(alive):
            _dict_increment(delta, "decision_not_evaluable_reasons", "row_out_of_range", 1)
            if shadow_delta is not None:
                shadow_delta["not_evaluable_count"] = int(shadow_delta.get("not_evaluable_count", 0) or 0) + 1
                _dict_increment(shadow_delta, "not_evaluable_reasons", "row_out_of_range", 1)
            continue
        agent = alive[alive_index]
        agent_id = int(getattr(agent, "agent_id", -1))
        valid_actions, availability = _valid_actions_for_agent(runner, alive, int(alive_index))
        try:
            payload = family_probability_payload(logits_arr[local_row], valid_actions=valid_actions)
        except Exception:
            _dict_increment(delta, "decision_not_evaluable_reasons", "family_probability_unavailable", 1)
            if shadow_delta is not None:
                shadow_delta["not_evaluable_count"] = int(shadow_delta.get("not_evaluable_count", 0) or 0) + 1
                _dict_increment(shadow_delta, "not_evaluable_reasons", "family_probability_unavailable", 1)
            continue
        prior_queue = pending.get(int(agent_id))
        if isinstance(prior_queue, list) and prior_queue:
            stale_count = len(prior_queue)
            delta["prefetched_not_applied_count"] = int(delta.get("prefetched_not_applied_count", 0) or 0) + int(stale_count)
            delta["stale_prefetch_decision_count"] = int(delta.get("stale_prefetch_decision_count", 0) or 0) + int(stale_count)
            _dict_increment(delta, "join_status_counts", "rejected_before_apply", int(stale_count))
            if shadow_delta is not None:
                shadow_delta["prefetched_not_applied_count"] = int(shadow_delta.get("prefetched_not_applied_count", 0) or 0) + int(stale_count)
        queue = []
        refill_source_class = _source_class_for_slot(
            base=str(action_source_class or "neural_policy_direct"),
            policy_kind=str(policy_kind or ""),
            action_slot=0,
            slot_count=int(actions_arr.shape[1]),
        )
        originating_refill_event_id = _decision_event_id(
            runner,
            agent_id,
            action_slot=0,
            source_class=refill_source_class,
        )
        for action_slot in range(int(actions_arr.shape[1])):
            chosen_action = int(actions_arr[local_row, action_slot])
            chosen_family = action_family_or_unknown(chosen_action)
            source_class = _source_class_for_slot(
                base=str(action_source_class or "neural_policy_direct"),
                policy_kind=str(policy_kind or ""),
                action_slot=int(action_slot),
                slot_count=int(actions_arr.shape[1]),
            )
            decision_event_id = _decision_event_id(runner, agent_id, action_slot=action_slot, source_class=source_class)
            shadow = None
            shadow_status = "disabled"
            if shadow_delta is not None:
                try:
                    shadow = factorized_family_shadow_payload(
                        logits_arr[local_row],
                        valid_actions=valid_actions,
                        chosen_action_id=chosen_action,
                    )
                    shadow_status = "evaluable" if bool(shadow.get("factorization_evaluable", False)) else "not_evaluable"
                except Exception as exc:
                    shadow_status = "not_evaluable"
                    shadow = {
                        "schema_version": FACTORIZED_SHADOW_SCHEMA_VERSION,
                        "factorization_evaluable": False,
                        "not_evaluable_reason": f"shadow_factorization_error:{type(exc).__name__}",
                    }
            noise_digest = ""
            noise_present = False
            if noise_arr is not None:
                if noise_arr.ndim == 3 and local_row < noise_arr.shape[0] and action_slot < noise_arr.shape[1]:
                    noise_present = True
                    noise_digest = _row_digest(noise_arr[local_row, action_slot])
                elif noise_arr.ndim == 2 and local_row < noise_arr.shape[0]:
                    noise_present = True
                    noise_digest = _row_digest(noise_arr[local_row])
            event = {
                "schema_version": POLICY_ACTION_FAMILY_DECISION_SCHEMA_VERSION,
                "decision_event_id": decision_event_id,
                "generation": _runner_generation(runner),
                "tick": _runner_tick(runner),
                "agent_id": int(agent_id),
                "slot_token": int(getattr(agent, "slot_token", 0) or 0),
                "policy_kind": str(policy_kind or ""),
                "action_source_class": source_class,
                "probability_status": "probability_bearing",
                "action_slot": int(action_slot),
                "slot_count": int(actions_arr.shape[1]),
                "chosen_action_id": int(chosen_action),
                "chosen_action_name": str(ACTION_ID_TO_NAME.get(int(chosen_action), "UNKNOWN")),
                "chosen_family": str(chosen_family),
                "valid_actions": [int(v) for v in valid_actions],
                "availability": availability,
                "raw_logits_json": _json_list(logits_arr[local_row]),
                "masked_logits_json": _json_list(_masked_logits(logits_arr[local_row], valid_actions)),
                "primitive_probability_json": _json_list(np.asarray(payload.get("primitive_probabilities", []), dtype=np.float64)),
                "family_probability_json": _json_dict(
                    {str(k): float(v) for k, v in (payload.get("family_probability_mass", {}) or {}).items()}
                ),
                "primitive_entropy": float(payload.get("primitive_entropy", 0.0) or 0.0),
                "family_entropy": float(payload.get("family_entropy", 0.0) or 0.0),
                "dominant_family": str(payload.get("dominant_family", "") or ""),
                "family_top_two_margin": float(payload.get("family_top_two_margin", 0.0) or 0.0),
                "movement_dilution_case": bool(payload.get("movement_dilution_case", False)),
                "sample_noise_present": bool(noise_present),
                "sample_noise_digest": noise_digest,
                "drive_vector_json": _json_list(drives_arr[local_row]) if drives_arr is not None and local_row < drives_arr.shape[0] else "[]",
                "action_temperature": (
                    float(temps_arr[local_row]) if temps_arr is not None and local_row < temps_arr.size else None
                ),
            }
            if shadow is not None:
                stride_mode = "direct"
                if str(source_class).startswith("neural_policy_stride_refill"):
                    stride_mode = "stride_refill"
                elif str(source_class).startswith("neural_policy_stride_cached"):
                    stride_mode = "stride_cached"
                shadow_distribution_id = str(shadow.get("shadow_distribution_id", "") or "")
                event.update(
                    {
                        "family_shadow_schema_version": FACTORIZED_SHADOW_SCHEMA_VERSION,
                        "family_shadow_status": shadow_status,
                        "family_shadow_distribution_id": shadow_distribution_id,
                        "family_shadow_authority": False,
                        "family_shadow_random_draws": 0,
                        "family_shadow_stride_mode": stride_mode,
                        "family_shadow_originating_refill_event_id": originating_refill_event_id,
                        "family_shadow_max_reconstruction_error": float(shadow.get("max_abs_reconstruction_error", 0.0) or 0.0),
                        "family_shadow_entropy_decomposition_residual": float(shadow.get("entropy_decomposition_residual", 0.0) or 0.0),
                    }
                )
                if _policy_family_head_shadow_record_events(runner):
                    event["family_shadow_payload"] = shadow
                shadow_delta["decision_count"] = int(shadow_delta.get("decision_count", 0) or 0) + 1
                if bool(shadow.get("factorization_evaluable", False)):
                    shadow_delta["evaluable_count"] = int(shadow_delta.get("evaluable_count", 0) or 0) + 1
                else:
                    shadow_delta["not_evaluable_count"] = int(shadow_delta.get("not_evaluable_count", 0) or 0) + 1
                    _dict_increment(shadow_delta, "not_evaluable_reasons", str(shadow.get("not_evaluable_reason", "not_evaluable") or "not_evaluable"), 1)
                shadow_delta["max_reconstruction_error"] = max(
                    float(shadow_delta.get("max_reconstruction_error", 0.0) or 0.0),
                    float(shadow.get("max_abs_reconstruction_error", 0.0) or 0.0),
                )
                shadow_delta["mean_reconstruction_error_sum"] = float(shadow_delta.get("mean_reconstruction_error_sum", 0.0) or 0.0) + float(shadow.get("mean_abs_reconstruction_error", 0.0) or 0.0)
                shadow_delta["primitive_probability_sum_residual_abs_max"] = max(
                    float(shadow_delta.get("primitive_probability_sum_residual_abs_max", 0.0) or 0.0),
                    abs(float(shadow.get("primitive_probability_sum_residual", 0.0) or 0.0)),
                )
                shadow_delta["family_probability_sum_residual_abs_max"] = max(
                    float(shadow_delta.get("family_probability_sum_residual_abs_max", 0.0) or 0.0),
                    abs(float(shadow.get("family_probability_sum_residual", 0.0) or 0.0)),
                )
                shadow_delta["entropy_decomposition_residual_abs_max"] = max(
                    float(shadow_delta.get("entropy_decomposition_residual_abs_max", 0.0) or 0.0),
                    abs(float(shadow.get("entropy_decomposition_residual", 0.0) or 0.0)),
                )
                family_probs = shadow.get("family_probabilities", {}) if isinstance(shadow.get("family_probabilities", {}), dict) else {}
                for family, value in family_probs.items():
                    shadow_delta["family_probability_sum"][str(family)] = float(shadow_delta["family_probability_sum"].get(str(family), 0.0) or 0.0) + float(value or 0.0)
                shadow_delta["movement_family_mass_sum"] = float(shadow_delta.get("movement_family_mass_sum", 0.0) or 0.0) + float(family_probs.get("movement", 0.0) or 0.0)
                shadow_delta["movement_conditional_entropy_sum"] = float(shadow_delta.get("movement_conditional_entropy_sum", 0.0) or 0.0) + float(shadow.get("movement_conditional_entropy", 0.0) or 0.0)
                _dict_increment(shadow_delta, "policy_kind_counts", str(policy_kind or "unknown"), 1)
                _dict_increment(shadow_delta, "source_class_counts", source_class, 1)
                _dict_increment(shadow_delta, "stride_mode_counts", stride_mode, 1)
                _dict_increment(shadow_delta, "chosen_family_counts", chosen_family, 1)
                _dict_increment(shadow_delta, "family_rank_counts", str(shadow.get("dominant_family", "unknown") or "unknown"), 1)
                valid_families = [
                    family for family, is_valid in (shadow.get("family_validity", {}) or {}).items()
                    if bool(is_valid)
                ]
                _dict_increment(shadow_delta, "valid_family_pattern_counts", "+".join(sorted(valid_families)), 1)
                movement_mask = shadow.get("movement_valid_direction_mask", {}) if isinstance(shadow.get("movement_valid_direction_mask", {}), dict) else {}
                _dict_increment(
                    shadow_delta,
                    "movement_valid_direction_count_histogram",
                    str(sum(1 for value in movement_mask.values() if bool(value))),
                    1,
                )
                if stride_mode == "stride_refill":
                    shadow_delta["stride_refill_count"] = int(shadow_delta.get("stride_refill_count", 0) or 0) + 1
                elif stride_mode == "stride_cached":
                    shadow_delta["stride_cached_count"] = int(shadow_delta.get("stride_cached_count", 0) or 0) + 1
                    shadow_delta["cache_application_count"] = int(shadow_delta.get("cache_application_count", 0) or 0) + 1
                if len(shadow_delta["event_sample"]) < int(shadow_delta.get("_event_sample_limit", 0) or 0):
                    shadow_delta["event_sample"].append(
                        {
                            "decision_event_id": decision_event_id,
                            "originating_refill_event_id": originating_refill_event_id,
                            "agent_id": int(agent_id),
                            "policy_kind": str(policy_kind or ""),
                            "source_class": source_class,
                            "stride_mode": stride_mode,
                            "chosen_action_id": int(chosen_action),
                            "chosen_family": str(chosen_family),
                            "shadow_distribution_id": shadow_distribution_id,
                            "max_reconstruction_error": float(shadow.get("max_abs_reconstruction_error", 0.0) or 0.0),
                        }
                    )
            variant = _variant_identity_for_agent(runner, agent)
            event["variant_class"] = str(variant.get("variant_class", "not_evaluable") or "not_evaluable")
            queue.append(event)
            delta["decision_count"] = int(delta.get("decision_count", 0) or 0) + 1
            _dict_increment(delta, "decision_family_counts", chosen_family, 1)
            _dict_increment(delta, "dominant_family_counts", str(event["dominant_family"] or "unknown"), 1)
            _dict_increment(delta, "source_class_counts", source_class, 1)
            _dict_increment(delta, "decision_source_class_counts", source_class, 1)
            _dict_increment(delta, "probability_status_counts", "probability_bearing", 1)
            if bool(event["movement_dilution_case"]):
                delta["movement_dilution_case_count"] = int(delta.get("movement_dilution_case_count", 0) or 0) + 1
            delta["primitive_entropy_sum"] = float(delta.get("primitive_entropy_sum", 0.0) or 0.0) + float(event["primitive_entropy"])
            delta["family_entropy_sum"] = float(delta.get("family_entropy_sum", 0.0) or 0.0) + float(event["family_entropy"])
            delta["family_margin_sum"] = float(delta.get("family_margin_sum", 0.0) or 0.0) + float(event["family_top_two_margin"])
            for family, value in (payload.get("family_probability_mass", {}) or {}).items():
                delta["family_probability_mass_sum"][str(family)] = float(delta["family_probability_mass_sum"].get(str(family), 0.0) or 0.0) + float(value or 0.0)
            if len(delta["decision_event_sample"]) < limit:
                delta["decision_event_sample"].append(event)
        pending[int(agent_id)] = queue
    runner._structural_policy_action_family_pending = pending
    _merge_policy_action_family_delta(runner, delta)
    if shadow_delta is not None:
        _merge_policy_family_head_shadow_delta(runner, shadow_delta)


def record_policy_action_execution_consequences(
    runner: Any,
    *,
    alive: list[Any],
    agent_ids_np: Any,
    actions_np: Any,
    pre_energy: Any,
    pre_health: Any,
    pre_injury: Any,
    pre_pos_y: Any,
    pre_pos_x: Any,
    post_pos_y: Any,
    post_pos_x: Any,
    new_energy: Any,
    new_health: Any,
    food_gained: Any,
    new_injury: Any,
) -> None:
    if not _action_family_reporting_enabled(runner):
        return
    pending = getattr(runner, "_structural_policy_action_family_pending", None)
    if not isinstance(pending, dict):
        pending = {}
    delta = empty_policy_action_family_summary(enabled=True)
    limit = _policy_event_sample_limit(runner)
    delta["_event_sample_limit"] = int(limit)
    try:
        agent_ids = np.asarray(agent_ids_np, dtype=np.int64).reshape((-1,))
        actions = np.asarray(actions_np, dtype=np.int32).reshape((-1,))
        pre_e = np.asarray(pre_energy, dtype=np.float64).reshape((-1,))
        pre_h = np.asarray(pre_health, dtype=np.float64).reshape((-1,))
        pre_i = np.asarray(pre_injury, dtype=np.float64).reshape((-1,))
        post_e = np.asarray(new_energy, dtype=np.float64).reshape((-1,))
        post_h = np.asarray(new_health, dtype=np.float64).reshape((-1,))
        post_i = np.asarray(new_injury, dtype=np.float64).reshape((-1,))
        food = np.asarray(food_gained, dtype=np.float64).reshape((-1,))
        y0 = np.asarray(pre_pos_y, dtype=np.int64).reshape((-1,))
        x0 = np.asarray(pre_pos_x, dtype=np.int64).reshape((-1,))
        y1 = np.asarray(post_pos_y, dtype=np.int64).reshape((-1,))
        x1 = np.asarray(post_pos_x, dtype=np.int64).reshape((-1,))
    except Exception:
        _merge_policy_action_family_delta(runner, _policy_family_execution_not_evaluable("execution_input_unavailable", len(alive)))
        return
    n = min(len(alive), agent_ids.size, actions.size, pre_e.size, pre_h.size, pre_i.size, post_e.size, post_h.size, post_i.size, food.size, y0.size, x0.size, y1.size, x1.size)
    for i in range(int(n)):
        agent_id = int(agent_ids[i])
        executed_action = int(actions[i])
        executed_family = action_family_or_unknown(executed_action)
        delta["raw_action_observation_count"] = int(delta.get("raw_action_observation_count", 0) or 0) + 1
        queue = pending.get(agent_id)
        if isinstance(queue, list) and queue:
            decision = queue.pop(0)
            if queue:
                pending[agent_id] = queue
            else:
                pending.pop(agent_id, None)
        else:
            decision = None
        if not isinstance(decision, dict):
            delta["execution_missing_decision_count"] = int(delta.get("execution_missing_decision_count", 0) or 0) + 1
            _dict_increment(delta, "execution_not_evaluable_reasons", "decision_event_missing", 1)
            decision_event_id = ""
            chosen_action = None
            chosen_family = ""
            availability = {}
            source_class = "unknown"
            probability_status = "unknown"
        else:
            delta["execution_join_count"] = int(delta.get("execution_join_count", 0) or 0) + 1
            decision_event_id = str(decision.get("decision_event_id", "") or "")
            chosen_action = int(decision.get("chosen_action_id", -999))
            chosen_family = str(decision.get("chosen_family", "") or "")
            availability = decision.get("availability", {}) if isinstance(decision.get("availability", {}), dict) else {}
            source_class = str(decision.get("action_source_class", "") or "unknown")
            probability_status = str(decision.get("probability_status", "") or "unknown")
            if int(chosen_action) != int(executed_action):
                delta["action_substitution_count"] = int(delta.get("action_substitution_count", 0) or 0) + 1
                delta["true_action_substitution_count"] = int(delta.get("true_action_substitution_count", 0) or 0) + 1
            else:
                delta["chosen_applied_same_count"] = int(delta.get("chosen_applied_same_count", 0) or 0) + 1
        action_application_id = _action_application_id(runner, agent_id, executed_action, source_class)
        execution_event_id = _execution_event_id(runner, action_application_id)
        seen = getattr(runner, "_structural_policy_action_application_seen", None)
        if not isinstance(seen, set):
            seen = set()
        if action_application_id in seen:
            delta["duplicate_action_application_count"] = int(delta.get("duplicate_action_application_count", 0) or 0) + 1
            _dict_increment(delta, "join_status_counts", "duplicate_observation", 1)
        else:
            seen.add(action_application_id)
            delta["unique_action_application_count"] = int(delta.get("unique_action_application_count", 0) or 0) + 1
            _dict_increment(delta, "join_status_counts", "joined" if isinstance(decision, dict) else "unknown", 1)
        runner._structural_policy_action_application_seen = seen
        _dict_increment(delta, "source_class_counts", source_class, 1)
        _dict_increment(delta, "application_source_class_counts", source_class, 1)
        _dict_increment(delta, "probability_status_counts", probability_status, 1)
        if source_class == "unknown":
            delta["unknown_source_count"] = int(delta.get("unknown_source_count", 0) or 0) + 1
        consequence = _action_consequence(
            runner=runner,
            agent_id=agent_id,
            action_application_id=action_application_id,
            decision_event_id=decision_event_id,
            execution_event_id=execution_event_id,
            source_class=source_class,
            executed_action=int(executed_action),
            availability=availability,
            pre_y=int(y0[i]),
            pre_x=int(x0[i]),
            post_y=int(y1[i]),
            post_x=int(x1[i]),
            pre_energy=float(pre_e[i]),
            post_energy=float(post_e[i]),
            pre_health=float(pre_h[i]),
            post_health=float(post_h[i]),
            pre_injury=float(pre_i[i]),
            post_injury=float(post_i[i]),
            food_gained=float(food[i]),
        )
        event = {
            "schema_version": POLICY_ACTION_EXECUTION_SCHEMA_VERSION,
            "decision_event_id": decision_event_id,
            "action_application_id": action_application_id,
            "execution_event_id": execution_event_id,
            "generation": _runner_generation(runner),
            "tick": _runner_tick(runner),
            "agent_id": int(agent_id),
            "action_source_class": source_class,
            "probability_status": probability_status,
            "chosen_action_id": chosen_action,
            "chosen_family": chosen_family,
            "executed_action_id": int(executed_action),
            "executed_action_name": str(ACTION_ID_TO_NAME.get(int(executed_action), "UNKNOWN")),
            "executed_family": str(executed_family),
            "action_substituted": bool(chosen_action is not None and int(chosen_action) != int(executed_action)),
            "execution_accepted": bool(consequence.get("execution_accepted", False)),
            "consequence_success": consequence.get("consequence_success"),
            "consequence_status": str(consequence.get("consequence_status", "") or ""),
            "consequence_reason": str(consequence.get("consequence_reason", "") or ""),
            "energy_delta": float(consequence.get("energy_delta", 0.0) or 0.0),
            "health_delta": float(consequence.get("health_delta", 0.0) or 0.0),
            "injury_delta": float(consequence.get("injury_delta", 0.0) or 0.0),
            "food_gained": float(food[i]),
            "position_changed": bool(int(y0[i]) != int(y1[i]) or int(x0[i]) != int(x1[i])),
        }
        _dict_increment(delta, "executed_family_counts", executed_family, 1)
        if event["consequence_status"] == "not_evaluable":
            _dict_increment(delta, "consequence_not_evaluable_family_counts", executed_family, 1)
            _dict_increment(delta, "execution_not_evaluable_reasons", str(event["consequence_reason"] or "not_evaluable"), 1)
        elif bool(event["consequence_success"]):
            _dict_increment(delta, "execution_success_family_counts", executed_family, 1)
        else:
            _dict_increment(delta, "execution_failure_family_counts", executed_family, 1)
        if len(delta["execution_event_sample"]) < limit:
            event["decision_event"] = decision if isinstance(decision, dict) else None
            delta["execution_event_sample"].append(event)
    runner._structural_policy_action_family_pending = pending
    _merge_policy_action_family_delta(runner, delta)


def counterfactual_policy_scalars_for_agent(
    runner: Any,
    agent: Any,
    *,
    actual_aggression_bias: float,
    actual_action_temperature: float,
    actual_drive_logit_gain: float,
) -> tuple[float, float, float, str]:
    if agent is None or not bool(
        getattr(
            getattr(runner, "sim_cfg", None),
            "structural_counterfactual_action_response_reporting_enable",
            False,
        )
    ):
        return (
            float(actual_aggression_bias),
            float(actual_action_temperature),
            float(actual_drive_logit_gain),
            "reporting_disabled",
        )
    if not bool(getattr(getattr(runner, "sim_cfg", None), "structural_live_semantics_enable", False)):
        return (
            float(actual_aggression_bias),
            float(actual_action_temperature),
            float(actual_drive_logit_gain),
            "live_semantics_disabled",
        )
    try:
        expr = runner.resolve_developmental_expression_for_agent(agent)
    except Exception:
        return (
            float(actual_aggression_bias),
            float(actual_action_temperature),
            float(actual_drive_logit_gain),
            "expression_unavailable",
        )
    summary = expr.get("structural_live_semantics", {}) if isinstance(expr, dict) else {}
    before = summary.get("effective_fields_before", {}) if isinstance(summary, dict) else {}
    if not isinstance(before, dict):
        return (
            float(actual_aggression_bias),
            float(actual_action_temperature),
            float(actual_drive_logit_gain),
            "pre_connection_fields_missing",
        )
    if not bool(summary.get("graph_supported", False)):
        return (
            float(actual_aggression_bias),
            float(actual_action_temperature),
            float(actual_drive_logit_gain),
            "graph_not_supported",
        )
    if int(summary.get("connection_count", 0) or 0) <= 0:
        return (
            float(actual_aggression_bias),
            float(actual_action_temperature),
            float(actual_drive_logit_gain),
            "no_connections_to_ablate",
        )
    try:
        return (
            float(before.get("aggression_bias", actual_aggression_bias)),
            float(before.get("action_temperature", actual_action_temperature)),
            float(before.get("drive_logit_gain", actual_drive_logit_gain)),
            "ok",
        )
    except Exception:
        return (
            float(actual_aggression_bias),
            float(actual_action_temperature),
            float(actual_drive_logit_gain),
            "nonfinite_or_invalid_pre_connection_fields",
        )


def record_counterfactual_action_response(
    runner: Any,
    *,
    alive: list[Any],
    row_indices: list[int],
    actual_logits: Any,
    counterfactual_logits: Any,
    actual_actions: Any,
    sample_noise: Any = None,
    status_by_row: list[str] | None = None,
    policy_kind: str = "",
    action_source_class: str = "neural_policy_direct",
) -> None:
    if not bool(
        getattr(
            getattr(runner, "sim_cfg", None),
            "structural_counterfactual_action_response_reporting_enable",
            False,
        )
    ):
        return
    try:
        actual = _as_numpy(actual_logits)
        counterfactual = _as_numpy(counterfactual_logits)
        actions = np.asarray(_as_numpy(actual_actions), dtype=np.int64).reshape((-1,))
        noise = None if sample_noise is None else _as_numpy(sample_noise)
    except Exception:
        _merge_counterfactual_delta(runner, _counterfactual_not_evaluable("tensor_conversion_failed", len(row_indices)))
        return
    if actual.shape != counterfactual.shape or actual.ndim != 2:
        _merge_counterfactual_delta(runner, _counterfactual_not_evaluable("shape_mismatch", len(row_indices)))
        return
    deterministic = bool(getattr(getattr(runner, "sim_cfg", None), "policy_deterministic_actions", False))
    if deterministic:
        cf_actions = np.argmax(counterfactual, axis=1).astype(np.int64, copy=False)
    else:
        if noise is None or np.asarray(noise).shape != counterfactual.shape:
            _merge_counterfactual_delta(runner, _counterfactual_not_evaluable("same_sample_noise_missing", len(row_indices)))
            return
        cf_actions = np.argmax(counterfactual + np.asarray(noise, dtype=np.float64), axis=1).astype(np.int64, copy=False)
    _record_decision_margin_events(
        runner,
        alive=alive,
        row_indices=row_indices,
        actual=actual,
        counterfactual=counterfactual,
        actions=actions,
        cf_actions=cf_actions,
        noise=noise,
        status_by_row=status_by_row,
        policy_kind=str(policy_kind or ""),
        action_source_class=str(action_source_class or "neural_policy_direct"),
    )
    delta = empty_opportunity_summary(enabled=True)
    statuses = list(status_by_row or [])
    for local_row, alive_index in enumerate(row_indices):
        if local_row >= counterfactual.shape[0] or local_row >= actions.shape[0] or alive_index >= len(alive):
            delta["actual_counterfactual_not_evaluable_count"] += 1
            _dict_increment(delta, "actual_counterfactual_not_evaluable_reasons", "row_out_of_range", 1)
            continue
        status = statuses[local_row] if local_row < len(statuses) else "ok"
        if status not in {"ok", "no_connections_to_ablate"}:
            delta["actual_counterfactual_not_evaluable_count"] += 1
            _dict_increment(delta, "actual_counterfactual_not_evaluable_reasons", status, 1)
            continue
        row_actual = np.asarray(actual[local_row], dtype=np.float64)
        row_cf = np.asarray(counterfactual[local_row], dtype=np.float64)
        if not (np.all(np.isfinite(row_actual)) and np.all(np.isfinite(row_cf))):
            delta["actual_counterfactual_not_evaluable_count"] += 1
            _dict_increment(delta, "actual_counterfactual_not_evaluable_reasons", "nonfinite_logits", 1)
            continue
        action = int(actions[local_row])
        cf_action = int(cf_actions[local_row])
        variant_class = ""
        mutation_origins: list[str] = []
        if bool(getattr(getattr(runner, "sim_cfg", None), "structural_variant_identity_reporting_enable", False)):
            try:
                from evolution.structural_variant_identity import agent_identity

                ident = agent_identity(
                    alive[alive_index],
                    canonical_semantic_digest=str(
                        getattr(runner, "_structural_variant_identity_canonical_semantic_digest", "") or ""
                    ),
                )
                variant_class = str(ident.get("variant_class", "not_evaluable") or "not_evaluable")
                mutation_origins = [str(item) for item in list(ident.get("mutation_operator_origin_set", []) or [])]
            except Exception:
                variant_class = "not_evaluable"
                mutation_origins = []
        delta["actual_counterfactual_evaluated_count"] += 1
        if variant_class:
            _dict_increment(delta, "actual_counterfactual_evaluated_by_variant_class", variant_class, 1)
            for origin in mutation_origins:
                _dict_increment(delta, "actual_counterfactual_evaluated_by_mutation_origin", origin, 1)
        if action == cf_action:
            delta["actual_counterfactual_same_action_count"] += 1
        else:
            delta["actual_counterfactual_action_change_count"] += 1
            delta["chosen_action_change_count"] += 1
            _dict_increment(delta, "actual_counterfactual_changed_action_pairs", f"{action}->{cf_action}", 1)
            if variant_class:
                _dict_increment(delta, "actual_counterfactual_changed_by_variant_class", variant_class, 1)
                for origin in mutation_origins:
                    _dict_increment(delta, "actual_counterfactual_changed_by_mutation_origin", origin, 1)
    _merge_counterfactual_delta(runner, delta)


def record_family_head_authority_choice_margin(
    runner: Any,
    *,
    alive: list[Any],
    row_indices: list[int],
    neutral_logits: Any,
    directional_logits: Any,
    actions: Any,
    sample_noise: Any = None,
    policy_kind: str = "",
    action_source_class: str = "neural_policy_direct",
) -> None:
    cfg = getattr(runner, "sim_cfg", None)
    choice_margin_enabled = bool(getattr(cfg, "policy_family_head_authority_choice_margin_reporting_enable", False))
    magnitude_surface_enabled = bool(getattr(cfg, "policy_family_head_authority_magnitude_surface_reporting_enable", False))
    candidate_enabled = bool(getattr(cfg, "policy_family_head_opportunity_conditioned_authority_candidate_reporting_enable", False))
    if not (choice_margin_enabled or magnitude_surface_enabled or candidate_enabled):
        return
    try:
        neutral = np.asarray(_as_numpy(neutral_logits), dtype=np.float64)
        directional = np.asarray(_as_numpy(directional_logits), dtype=np.float64)
        chosen = np.asarray(_as_numpy(actions), dtype=np.int64)
        if chosen.ndim == 1:
            chosen = chosen.reshape((-1, 1))
        elif chosen.ndim != 2:
            chosen = chosen.reshape((chosen.shape[0], -1))
    except Exception:
        _merge_family_head_choice_margin_delta(
            runner,
            _family_head_choice_margin_not_evaluable("tensor_conversion_failed", len(row_indices)),
        )
        return
    if neutral.shape != directional.shape or neutral.ndim != 2:
        _merge_family_head_choice_margin_delta(
            runner,
            _family_head_choice_margin_not_evaluable("shape_mismatch", len(row_indices)),
        )
        return
    noise = None
    if sample_noise is not None:
        try:
            noise = np.asarray(_as_numpy(sample_noise), dtype=np.float64)
        except Exception:
            noise = None
    deterministic = bool(getattr(cfg, "policy_deterministic_actions", False))
    if not deterministic and noise is None:
        if choice_margin_enabled:
            _merge_family_head_choice_margin_delta(
                runner,
                _family_head_choice_margin_not_evaluable("same_sample_noise_missing", len(row_indices)),
            )
        if magnitude_surface_enabled:
            not_evaluable = _empty_family_head_magnitude_surface_summary(enabled=True)
            _dict_increment(not_evaluable, "not_evaluable_reasons", "same_sample_noise_missing", len(row_indices))
            _merge_family_head_magnitude_surface_delta(runner, not_evaluable)
        if candidate_enabled:
            not_evaluable = _empty_opportunity_conditioned_candidate_summary(enabled=True)
            _dict_increment(not_evaluable, "not_evaluable_reasons", "same_sample_noise_missing", len(row_indices))
            _merge_opportunity_conditioned_candidate_delta(runner, not_evaluable)
        return
    delta = _empty_family_head_choice_margin_summary(enabled=True) if choice_margin_enabled else None
    surface_delta = _empty_family_head_magnitude_surface_summary(enabled=True) if magnitude_surface_enabled else None
    candidate_delta = _empty_opportunity_conditioned_candidate_summary(enabled=True) if candidate_enabled else None
    limit = _family_head_choice_margin_sample_limit(runner)
    if delta is not None:
        delta["_event_sample_limit"] = int(limit)
    if surface_delta is not None:
        surface_delta["_event_sample_limit"] = int(_family_head_magnitude_surface_sample_limit(runner))
    if candidate_delta is not None:
        candidate_delta["_event_sample_limit"] = int(_family_head_opportunity_conditioned_candidate_sample_limit(runner))
    def _note_not_evaluable(reason: str) -> None:
        if delta is not None:
            _dict_increment(delta, "not_evaluable_reasons", reason, 1)
        if surface_delta is not None:
            _dict_increment(surface_delta, "not_evaluable_reasons", reason, 1)
        if candidate_delta is not None:
            _dict_increment(candidate_delta, "not_evaluable_reasons", reason, 1)

    for local_row, alive_index in enumerate(row_indices):
        if local_row >= neutral.shape[0] or local_row >= directional.shape[0] or local_row >= chosen.shape[0] or alive_index >= len(alive):
            _note_not_evaluable("row_out_of_range")
            continue
        agent = alive[alive_index]
        opportunity_actions, availability = _valid_actions_for_agent(runner, alive, int(alive_index))
        choice_actions = [
            int(action)
            for action in range(int(N_ACTIONS))
            if np.isfinite(neutral[local_row, int(action)]) and np.isfinite(directional[local_row, int(action)])
        ]
        if len(choice_actions) < 1:
            _note_not_evaluable("no_valid_actions")
            continue
        neutral_row = _masked_logits(neutral[local_row], choice_actions)
        directional_row = _masked_logits(directional[local_row], choice_actions)
        if not (np.any(np.isfinite(neutral_row)) and np.any(np.isfinite(directional_row))):
            _note_not_evaluable("nonfinite_masked_logits")
            continue
        residual = _agent_family_head_residual(agent)
        alpha_cap = _residual_alpha_cap(residual)
        slot_count = int(chosen.shape[1])
        for action_slot in range(slot_count):
            source_class = _source_class_for_slot(
                base=str(action_source_class or "neural_policy_direct"),
                policy_kind=str(policy_kind or ""),
                action_slot=int(action_slot),
                slot_count=int(slot_count),
            )
            noise_row = None
            if not deterministic:
                if noise is None:
                    _note_not_evaluable("same_sample_noise_missing")
                    continue
                if noise.ndim == 3 and local_row < noise.shape[0] and action_slot < noise.shape[1]:
                    noise_row = np.asarray(noise[local_row, action_slot], dtype=np.float64)
                elif noise.ndim == 2 and local_row < noise.shape[0]:
                    noise_row = np.asarray(noise[local_row], dtype=np.float64)
                else:
                    _note_not_evaluable("same_sample_noise_shape_mismatch")
                    continue
            neutral_score = neutral_row if deterministic else neutral_row + np.asarray(noise_row, dtype=np.float64)
            directional_score = directional_row if deterministic else directional_row + np.asarray(noise_row, dtype=np.float64)
            neutral_choice = _argmax_stable(neutral_score)
            directional_choice = _argmax_stable(directional_score)
            actual_choice = int(chosen[local_row, action_slot])
            neutral_family = action_family_or_unknown(neutral_choice)
            directional_family = action_family_or_unknown(directional_choice)
            actual_family = action_family_or_unknown(actual_choice)
            primitive_changed = bool(neutral_choice != directional_choice)
            family_changed = bool(neutral_family != directional_family)
            within_family_changed = bool(primitive_changed and not family_changed)
            neutral_margin = decision_margin_payload(neutral_score, choice_actions)
            directional_margin = decision_margin_payload(directional_score, choice_actions)
            neutral_family_margin = _family_score_margin(neutral_row, choice_actions, residual=None, alpha=0.0)
            directional_family_margin = _family_score_margin(neutral_row, choice_actions, residual=residual, alpha=1.0)
            primitive_alpha = _find_crossing_alpha(
                neutral_row,
                choice_actions,
                residual=residual,
                alpha_cap=alpha_cap,
                noise=noise_row,
                mode="primitive",
            )
            family_alpha = _find_crossing_alpha(
                neutral_row,
                choice_actions,
                residual=residual,
                alpha_cap=alpha_cap,
                noise=None,
                mode="family",
            )
            event = {
                "schema_version": FAMILY_HEAD_AUTHORITY_CHOICE_MARGIN_SCHEMA_VERSION,
                "run_id": str(getattr(runner, "run_id", "") or getattr(runner, "_run_id", "") or ""),
                "generation": _runner_generation(runner),
                "tick": _runner_tick(runner),
                "agent_id": int(getattr(agent, "agent_id", -1)),
                "row_index": int(local_row),
                "alive_index": int(alive_index),
                "policy_kind": str(policy_kind or ""),
                "action_source_class": source_class,
                "action_slot": int(action_slot),
                "slot_count": int(slot_count),
                "deterministic_choice": bool(deterministic),
                "same_sample_noise_present": bool(noise_row is not None),
                "sample_noise_digest": _row_digest(noise_row) if noise_row is not None else "",
                "neutral_chosen_action": int(neutral_choice),
                "neutral_chosen_family": str(neutral_family),
                "directional_chosen_action": int(directional_choice),
                "directional_chosen_family": str(directional_family),
                "actual_sampled_action": int(actual_choice),
                "actual_sampled_family": str(actual_family),
                "reconstruction_parity": bool(actual_choice == directional_choice),
                "primitive_choice_changed": bool(primitive_changed),
                "family_choice_changed": bool(family_changed),
                "within_family_primitive_changed": bool(within_family_changed),
                "neutral_primitive_margin": neutral_margin.get("top_two_margin", ""),
                "directional_primitive_margin": directional_margin.get("top_two_margin", ""),
                "neutral_top_action": neutral_margin.get("top_action", -1),
                "directional_top_action": directional_margin.get("top_action", -1),
                "neutral_family_margin": neutral_family_margin.get("top_two_margin", ""),
                "directional_family_margin": directional_family_margin.get("top_two_margin", ""),
                "neutral_top_family": neutral_family_margin.get("top_family", ""),
                "directional_top_family": directional_family_margin.get("top_family", ""),
                "primitive_crossing_alpha": primitive_alpha,
                "family_crossing_alpha": family_alpha,
                "alpha_cap": float(alpha_cap),
                "primitive_crossing_within_cap": bool(primitive_alpha is not None and primitive_alpha <= alpha_cap + 1e-9),
                "family_crossing_within_cap": bool(family_alpha is not None and family_alpha <= alpha_cap + 1e-9),
                "choice_action_mask": "finite_choice_logits",
                "valid_actions": ",".join(str(int(v)) for v in choice_actions),
                "opportunity_actions": ",".join(str(int(v)) for v in opportunity_actions),
                "residual_json": _json_dict({family: float(residual.get(family, 0.0) or 0.0) for family in FAMILY_ORDER}),
                "neutral_score_json": _json_list(neutral_row),
                "directional_score_json": _json_list(directional_row),
                "neutral_probability_json": _json_list(_softmax_masked_finite(neutral_row)),
                "directional_probability_json": _json_list(_softmax_masked_finite(directional_row)),
                "attack_target_available": int(availability.get("attack_target_available", False)),
                "attack_executable": int(availability.get("attack_executable", False)),
                "gather_resource_available": int(availability.get("gather_resource_available", False)),
                "rest_need_present": int(availability.get("rest_need_present", False)),
                "legal_move_count": int(availability.get("legal_move_count", 0)),
            }
            if delta is not None:
                _update_family_head_choice_margin_summary(delta, event)
                if len(delta["event_sample"]) < limit:
                    delta["event_sample"].append(event)
            if surface_delta is not None:
                _update_family_head_magnitude_surface_summary(surface_delta, event, neutral_row, choice_actions, residual, noise_row)
            if candidate_delta is not None:
                _update_opportunity_conditioned_candidate_summary(candidate_delta, event, neutral_row, choice_actions, residual, noise_row)
    if delta is not None:
        _merge_family_head_choice_margin_delta(runner, delta)
    if surface_delta is not None:
        _merge_family_head_magnitude_surface_delta(runner, surface_delta)
    if candidate_delta is not None:
        _merge_opportunity_conditioned_candidate_delta(runner, candidate_delta)


def _record_decision_margin_events(
    runner: Any,
    *,
    alive: list[Any],
    row_indices: list[int],
    actual: np.ndarray,
    counterfactual: np.ndarray,
    actions: np.ndarray,
    cf_actions: np.ndarray,
    noise: np.ndarray | None,
    status_by_row: list[str] | None,
    policy_kind: str,
    action_source_class: str,
) -> None:
    if not bool(getattr(getattr(runner, "sim_cfg", None), "structural_decision_margin_reporting_enable", False)):
        return
    limit = int(getattr(getattr(runner, "sim_cfg", None), "structural_decision_margin_event_sample_limit", 4096) or 0)
    if limit <= 0:
        return
    events = getattr(runner, "_structural_decision_margin_events", None)
    if not isinstance(events, list):
        events = []
    summary = getattr(runner, "_structural_decision_margin_summary", None)
    if not isinstance(summary, dict):
        summary = _empty_decision_margin_summary()
    statuses = list(status_by_row or [])
    deterministic = bool(getattr(getattr(runner, "sim_cfg", None), "policy_deterministic_actions", False))
    for local_row, alive_index in enumerate(row_indices):
        if len(events) >= limit:
            summary["sample_limit_reached"] = True
            break
        if local_row >= actual.shape[0] or local_row >= actions.shape[0] or alive_index >= len(alive):
            _bump_summary_reason(summary, "row_out_of_range")
            continue
        status = statuses[local_row] if local_row < len(statuses) else "ok"
        if status not in {"ok", "no_connections_to_ablate"}:
            _bump_summary_reason(summary, status)
            continue
        agent = alive[alive_index]
        actual_row = np.asarray(actual[local_row], dtype=np.float64)
        cf_row = np.asarray(counterfactual[local_row], dtype=np.float64)
        if not (np.all(np.isfinite(actual_row)) and np.all(np.isfinite(cf_row))):
            _bump_summary_reason(summary, "nonfinite_logits")
            continue
        valid_actions, availability = _valid_actions_for_agent(runner, alive, alive_index)
        margin_actual = decision_margin_payload(actual_row, valid_actions)
        margin_cf = decision_margin_payload(cf_row, valid_actions)
        score_delta = actual_row - cf_row
        chosen = int(actions[local_row])
        cf_chosen = int(cf_actions[local_row]) if local_row < cf_actions.shape[0] else -1
        expr_summary = _live_expression_summary(runner, agent)
        identity = _variant_identity_for_agent(runner, agent)
        event = {
            "schema_version": DECISION_MARGIN_SCHEMA_VERSION,
            "run_id": str(getattr(runner, "run_id", "") or getattr(runner, "run_dir", "") or ""),
            "generation": _runner_generation(runner),
            "tick": _runner_tick(runner),
            "agent_id": int(getattr(agent, "agent_id", -1)),
            "row_index": int(local_row),
            "alive_index": int(alive_index),
            "policy_kind": str(policy_kind or ""),
            "action_source_class": _source_class_for_slot(
                base=str(action_source_class or "neural_policy_direct"),
                policy_kind=str(policy_kind or ""),
                action_slot=0,
                slot_count=1,
            ),
            "deterministic_choice": bool(deterministic),
            "same_sample_noise_present": bool(noise is not None),
            "actual_chosen_action": chosen,
            "ablated_chosen_action": cf_chosen,
            "chosen_action_changed": int(chosen != cf_chosen),
            "valid_action_count": int(len(valid_actions)),
            "valid_actions": ",".join(str(int(v)) for v in valid_actions),
            "actual_top_action": margin_actual.get("top_action", -1),
            "actual_second_action": margin_actual.get("second_action", -1),
            "actual_margin_evaluable": bool(margin_actual.get("evaluable", False)),
            "actual_top_score": margin_actual.get("top_score", ""),
            "actual_second_score": margin_actual.get("second_score", ""),
            "actual_top_two_margin": margin_actual.get("top_two_margin", ""),
            "ablated_top_action": margin_cf.get("top_action", -1),
            "ablated_second_action": margin_cf.get("second_action", -1),
            "ablated_margin_evaluable": bool(margin_cf.get("evaluable", False)),
            "ablated_top_score": margin_cf.get("top_score", ""),
            "ablated_second_score": margin_cf.get("second_score", ""),
            "ablated_top_two_margin": margin_cf.get("top_two_margin", ""),
            "max_abs_score_delta": float(np.max(np.abs(score_delta))) if score_delta.size else 0.0,
            "score_delta_norm_l2": float(np.linalg.norm(score_delta)) if score_delta.size else 0.0,
            "chosen_action_score_delta": float(score_delta[chosen]) if 0 <= chosen < score_delta.size else "",
            "rank_changed": int(margin_actual.get("top_action", -1) != margin_cf.get("top_action", -1)),
            "margin_crossed": int(chosen != cf_chosen),
            "score_delta_vector_json": _json_list(score_delta),
            "actual_score_vector_json": _json_list(actual_row),
            "ablated_score_vector_json": _json_list(cf_row),
            "actual_probability_vector_json": _json_list(_softmax(actual_row)),
            "ablated_probability_vector_json": _json_list(_softmax(cf_row)),
            "probability_l1_distance": float(np.sum(np.abs(_softmax(actual_row) - _softmax(cf_row)))),
            "attack_target_available": int(availability.get("attack_target_available", False)),
            "attack_executable": int(availability.get("attack_executable", False)),
            "gather_resource_available": int(availability.get("gather_resource_available", False)),
            "rest_need_present": int(availability.get("rest_need_present", False)),
            "legal_move_count": int(availability.get("legal_move_count", 0)),
            "action_mask_blocked": int(chosen not in valid_actions),
            "target_unavailable": int(chosen == int(ACT_ATTACK) and not availability.get("attack_target_available", False)),
            "changed_effective_fields": ",".join(expr_summary["changed_effective_fields"]),
            "effective_field_delta_json": expr_summary["effective_field_delta_json"],
            "clipped_fields": ",".join(expr_summary["clipped_fields"]),
            "saturated_fields": ",".join(expr_summary["saturated_fields"]),
            "developmental_stage": expr_summary["developmental_stage"],
            "graph_identity_digest": identity.get("graph_identity_digest", ""),
            "graph_semantic_digest": identity.get("graph_semantic_digest", ""),
            "variant_class": identity.get("variant_class", "not_evaluable"),
            "canonical_seed_exact": int(bool(identity.get("canonical_seed_exact", False))),
            "mutation_operator_origin_set": ",".join(identity.get("mutation_operator_origin_set", []) or []),
            "operator_origin_set": ",".join(identity.get("operator_origin_set", []) or []),
        }
        events.append(event)
        _update_decision_margin_summary(summary, event)
    runner._structural_decision_margin_events = events
    runner._structural_decision_margin_summary = summary


def decision_margin_payload(scores: np.ndarray, valid_actions: list[int]) -> dict[str, Any]:
    valid = [int(v) for v in valid_actions if 0 <= int(v) < int(scores.shape[0])]
    if len(valid) < 2:
        top_action = int(valid[0]) if valid else -1
        top_score = float(scores[top_action]) if top_action >= 0 and np.isfinite(scores[top_action]) else ""
        return {"evaluable": False, "top_action": top_action, "second_action": -1, "top_score": top_score, "second_score": "", "top_two_margin": 0.0}
    vals = [(idx, float(scores[idx])) for idx in valid]
    vals.sort(key=lambda item: (-item[1], item[0]))
    top, second = vals[0], vals[1]
    return {
        "evaluable": True,
        "top_action": int(top[0]),
        "second_action": int(second[0]),
        "top_score": float(top[1]),
        "second_score": float(second[1]),
        "top_two_margin": float(top[1] - second[1]),
    }


def shadow_margin_rows(event: dict[str, Any], multipliers: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)) -> list[dict[str, Any]]:
    actual = np.asarray(_json_load_list(event.get("actual_score_vector_json", "[]")), dtype=np.float64)
    ablated = np.asarray(_json_load_list(event.get("ablated_score_vector_json", "[]")), dtype=np.float64)
    valid = [int(v) for v in str(event.get("valid_actions", "") or "").split(",") if str(v).strip()]
    if actual.shape != ablated.shape or actual.ndim != 1:
        return []
    base = ablated
    delta = actual - ablated
    out = []
    for mult in multipliers:
        scaled = base + float(mult) * delta
        margin = decision_margin_payload(scaled, valid)
        chosen = int(margin.get("top_action", -1)) if margin.get("evaluable") else -1
        out.append(
            {
                "schema_version": DECISION_MARGIN_SCHEMA_VERSION,
                "run_id": event.get("run_id", ""),
                "generation": event.get("generation", ""),
                "tick": event.get("tick", ""),
                "agent_id": event.get("agent_id", ""),
                "multiplier": float(mult),
                "shadow_top_action": chosen,
                "shadow_top_two_margin": margin.get("top_two_margin", ""),
                "shadow_rank_changed_vs_ablated": int(chosen != int(event.get("ablated_top_action", -1))),
                "shadow_chosen_changed_vs_actual": int(chosen != int(event.get("actual_chosen_action", -1))),
                "context": event.get("context", ""),
                "seed": event.get("seed", ""),
                "variant_class": event.get("variant_class", ""),
                "canonical_seed_exact": event.get("canonical_seed_exact", ""),
            }
        )
    return out


def _empty_family_head_choice_margin_summary(*, enabled: bool) -> dict[str, Any]:
    return {
        "schema_version": FAMILY_HEAD_AUTHORITY_CHOICE_MARGIN_SCHEMA_VERSION,
        "enabled": bool(enabled),
        "event_count": 0 if enabled else None,
        "deterministic_count": 0 if enabled else None,
        "stochastic_count": 0 if enabled else None,
        "same_sample_noise_count": 0 if enabled else None,
        "choice_reconstruction_mismatch_count": 0 if enabled else None,
        "primitive_choice_change_count": 0 if enabled else None,
        "family_choice_change_count": 0 if enabled else None,
        "within_family_primitive_change_count": 0 if enabled else None,
        "no_choice_change_count": 0 if enabled else None,
        "primitive_crossing_within_cap_count": 0 if enabled else None,
        "family_crossing_within_cap_count": 0 if enabled else None,
        "primitive_no_crossing_within_cap_count": 0 if enabled else None,
        "family_no_crossing_within_cap_count": 0 if enabled else None,
        "neutral_primitive_margin_sum": 0.0 if enabled else None,
        "directional_primitive_margin_sum": 0.0 if enabled else None,
        "neutral_family_margin_sum": 0.0 if enabled else None,
        "directional_family_margin_sum": 0.0 if enabled else None,
        "primitive_crossing_alpha_sum": 0.0 if enabled else None,
        "family_crossing_alpha_sum": 0.0 if enabled else None,
        "primitive_crossing_alpha_count": 0 if enabled else None,
        "family_crossing_alpha_count": 0 if enabled else None,
        "source_class_counts": {},
        "policy_kind_counts": {},
        "neutral_chosen_family_counts": {},
        "directional_chosen_family_counts": {},
        "not_evaluable_reasons": {} if enabled else {"reporting_disabled": 1},
        "event_sample": [],
        "_event_sample_limit": 0,
    }


def _family_head_choice_margin_not_evaluable(reason: str, count: int) -> dict[str, Any]:
    delta = _empty_family_head_choice_margin_summary(enabled=True)
    if count:
        _dict_increment(delta, "not_evaluable_reasons", reason, int(count))
    return delta


def _update_family_head_choice_margin_summary(summary: dict[str, Any], event: dict[str, Any]) -> None:
    summary["event_count"] = int(summary.get("event_count", 0) or 0) + 1
    if bool(event.get("deterministic_choice", False)):
        summary["deterministic_count"] = int(summary.get("deterministic_count", 0) or 0) + 1
    else:
        summary["stochastic_count"] = int(summary.get("stochastic_count", 0) or 0) + 1
    if bool(event.get("same_sample_noise_present", False)):
        summary["same_sample_noise_count"] = int(summary.get("same_sample_noise_count", 0) or 0) + 1
    if not bool(event.get("reconstruction_parity", False)):
        summary["choice_reconstruction_mismatch_count"] = int(summary.get("choice_reconstruction_mismatch_count", 0) or 0) + 1
    primitive_changed = bool(event.get("primitive_choice_changed", False))
    family_changed = bool(event.get("family_choice_changed", False))
    within_changed = bool(event.get("within_family_primitive_changed", False))
    summary["primitive_choice_change_count"] = int(summary.get("primitive_choice_change_count", 0) or 0) + int(primitive_changed)
    summary["family_choice_change_count"] = int(summary.get("family_choice_change_count", 0) or 0) + int(family_changed)
    summary["within_family_primitive_change_count"] = int(summary.get("within_family_primitive_change_count", 0) or 0) + int(within_changed)
    if not primitive_changed:
        summary["no_choice_change_count"] = int(summary.get("no_choice_change_count", 0) or 0) + 1
    if bool(event.get("primitive_crossing_within_cap", False)):
        summary["primitive_crossing_within_cap_count"] = int(summary.get("primitive_crossing_within_cap_count", 0) or 0) + 1
    else:
        summary["primitive_no_crossing_within_cap_count"] = int(summary.get("primitive_no_crossing_within_cap_count", 0) or 0) + 1
    if bool(event.get("family_crossing_within_cap", False)):
        summary["family_crossing_within_cap_count"] = int(summary.get("family_crossing_within_cap_count", 0) or 0) + 1
    else:
        summary["family_no_crossing_within_cap_count"] = int(summary.get("family_no_crossing_within_cap_count", 0) or 0) + 1
    for src, dst in (
        ("neutral_primitive_margin", "neutral_primitive_margin_sum"),
        ("directional_primitive_margin", "directional_primitive_margin_sum"),
        ("neutral_family_margin", "neutral_family_margin_sum"),
        ("directional_family_margin", "directional_family_margin_sum"),
    ):
        value = _finite_summary_float(event.get(src, 0.0))
        if value is None:
            _dict_increment(summary, "not_evaluable_reasons", f"nonfinite_{src}", 1)
            continue
        current = _finite_summary_float(summary.get(dst, 0.0))
        summary[dst] = (current or 0.0) + value
    for src, sum_key, count_key in (
        ("primitive_crossing_alpha", "primitive_crossing_alpha_sum", "primitive_crossing_alpha_count"),
        ("family_crossing_alpha", "family_crossing_alpha_sum", "family_crossing_alpha_count"),
    ):
        value = _finite_summary_float(event.get(src))
        if value is not None:
            current = _finite_summary_float(summary.get(sum_key, 0.0))
            summary[sum_key] = (current or 0.0) + value
            summary[count_key] = int(summary.get(count_key, 0) or 0) + 1
    _dict_increment(summary, "source_class_counts", str(event.get("action_source_class", "unknown") or "unknown"), 1)
    _dict_increment(summary, "policy_kind_counts", str(event.get("policy_kind", "unknown") or "unknown"), 1)
    _dict_increment(summary, "neutral_chosen_family_counts", str(event.get("neutral_chosen_family", "unknown") or "unknown"), 1)
    _dict_increment(summary, "directional_chosen_family_counts", str(event.get("directional_chosen_family", "unknown") or "unknown"), 1)


def _merge_family_head_choice_margin_delta(runner: Any, delta: dict[str, Any]) -> None:
    total = getattr(runner, "_policy_family_head_authority_choice_margin_total", None)
    if not isinstance(total, dict):
        total = _empty_family_head_choice_margin_summary(enabled=True)
    for key in (
        "event_count",
        "deterministic_count",
        "stochastic_count",
        "same_sample_noise_count",
        "choice_reconstruction_mismatch_count",
        "primitive_choice_change_count",
        "family_choice_change_count",
        "within_family_primitive_change_count",
        "no_choice_change_count",
        "primitive_crossing_within_cap_count",
        "family_crossing_within_cap_count",
        "primitive_no_crossing_within_cap_count",
        "family_no_crossing_within_cap_count",
        "primitive_crossing_alpha_count",
        "family_crossing_alpha_count",
    ):
        total[key] = int(total.get(key, 0) or 0) + int(delta.get(key, 0) or 0)
    for key in (
        "neutral_primitive_margin_sum",
        "directional_primitive_margin_sum",
        "neutral_family_margin_sum",
        "directional_family_margin_sum",
        "primitive_crossing_alpha_sum",
        "family_crossing_alpha_sum",
    ):
        total_value = _finite_summary_float(total.get(key, 0.0)) or 0.0
        delta_value = _finite_summary_float(delta.get(key, 0.0))
        if delta_value is None:
            _dict_increment(total, "not_evaluable_reasons", f"nonfinite_{key}", 1)
            delta_value = 0.0
        total[key] = total_value + delta_value
    for field in (
        "source_class_counts",
        "policy_kind_counts",
        "neutral_chosen_family_counts",
        "directional_chosen_family_counts",
        "not_evaluable_reasons",
    ):
        counter = Counter(total.get(field, {}) or {})
        counter.update(delta.get(field, {}) or {})
        total[field] = dict(sorted(counter.items()))
    limit = int(delta.get("_event_sample_limit", total.get("_event_sample_limit", 0)) or 0)
    if limit > 0:
        existing = list(total.get("event_sample", []) or [])
        for event in list(delta.get("event_sample", []) or []):
            if len(existing) >= limit:
                break
            existing.append(event)
        total["event_sample"] = existing
    total["_event_sample_limit"] = int(limit)
    _finalize_family_head_choice_margin_summary(total)
    runner._policy_family_head_authority_choice_margin_total = total


def _finalize_family_head_choice_margin_summary(summary: dict[str, Any]) -> None:
    count = max(1, int(summary.get("event_count", 0) or 0))
    for sum_key, mean_key in (
        ("neutral_primitive_margin_sum", "neutral_primitive_margin_mean"),
        ("directional_primitive_margin_sum", "directional_primitive_margin_mean"),
        ("neutral_family_margin_sum", "neutral_family_margin_mean"),
        ("directional_family_margin_sum", "directional_family_margin_mean"),
    ):
        value = _finite_summary_float(summary.get(sum_key, 0.0)) or 0.0
        summary[sum_key] = value
        summary[mean_key] = value / float(count)
    for sum_key, count_key, mean_key in (
        ("primitive_crossing_alpha_sum", "primitive_crossing_alpha_count", "primitive_crossing_alpha_mean"),
        ("family_crossing_alpha_sum", "family_crossing_alpha_count", "family_crossing_alpha_mean"),
    ):
        denom = int(summary.get(count_key, 0) or 0)
        value = _finite_summary_float(summary.get(sum_key, 0.0)) or 0.0
        summary[sum_key] = value
        summary[mean_key] = (value / float(denom)) if denom else None


def _finite_summary_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def _empty_family_head_magnitude_surface_summary(*, enabled: bool) -> dict[str, Any]:
    return {
        "schema_version": FAMILY_HEAD_AUTHORITY_MAGNITUDE_SURFACE_SCHEMA_VERSION,
        "enabled": bool(enabled),
        "alpha_values": [float(alpha) for alpha in FAMILY_HEAD_AUTHORITY_MAGNITUDE_SURFACE_ALPHAS],
        "event_count": 0 if enabled else None,
        "same_sample_noise_count": 0 if enabled else None,
        "choice_reconstruction_mismatch_count_at_alpha_1": 0 if enabled else None,
        "not_evaluable_reasons": {} if enabled else {"reporting_disabled": 1},
        "alpha_surface": {},
        "source_class_counts": {},
        "policy_kind_counts": {},
        "surface_sample": [],
        "_event_sample_limit": 0,
    }


def _empty_alpha_surface_record(alpha: float) -> dict[str, Any]:
    return {
        "alpha": float(alpha),
        "event_count": 0,
        "primitive_choice_change_count": 0,
        "family_choice_change_count": 0,
        "promoted_family_choice_count": 0,
        "demoted_family_choice_count": 0,
        "promoted_family_probability_delta_sum": 0.0,
        "family_probability_l1_delta_sum": 0.0,
        "primitive_probability_l1_delta_sum": 0.0,
        "entropy_sum": 0.0,
        "neutral_entropy_sum": 0.0,
        "top_two_margin_sum": 0.0,
        "family_margin_sum": 0.0,
        "meaningful_promoted_opportunity_count": 0,
        "choice_change_with_promoted_opportunity_count": 0,
        "choice_change_without_promoted_opportunity_count": 0,
        "source_class_counts": {},
        "policy_kind_counts": {},
        "candidate_family_counts": {},
    }


def _update_family_head_magnitude_surface_summary(
    summary: dict[str, Any],
    base_event: dict[str, Any],
    neutral_row: np.ndarray,
    choice_actions: list[int],
    residual: dict[str, float],
    noise_row: np.ndarray | None,
) -> None:
    neutral_score = np.asarray(neutral_row, dtype=np.float64)
    if noise_row is not None:
        neutral_choice_score = neutral_score + np.asarray(noise_row, dtype=np.float64)
        summary["same_sample_noise_count"] = int(summary.get("same_sample_noise_count", 0) or 0) + 1
    else:
        neutral_choice_score = neutral_score
    neutral_choice = _argmax_stable(neutral_choice_score)
    neutral_family = action_family_or_unknown(neutral_choice)
    neutral_probs = _softmax_masked_finite(neutral_score)
    neutral_family_probs = _family_probability_vector(neutral_probs)
    neutral_entropy = _entropy(neutral_probs)
    promoted = _promoted_family(residual)
    demoted = _demoted_family(residual)
    promoted_opportunity = _meaningful_family_opportunity(promoted, base_event)
    summary["event_count"] = int(summary.get("event_count", 0) or 0) + 1
    _dict_increment(summary, "source_class_counts", str(base_event.get("action_source_class", "unknown") or "unknown"), 1)
    _dict_increment(summary, "policy_kind_counts", str(base_event.get("policy_kind", "unknown") or "unknown"), 1)
    for alpha in FAMILY_HEAD_AUTHORITY_MAGNITUDE_SURFACE_ALPHAS:
        alpha_key = _alpha_key(alpha)
        surface = summary.setdefault("alpha_surface", {})
        record = surface.setdefault(alpha_key, _empty_alpha_surface_record(float(alpha)))
        candidate_score = _authority_scores_for_alpha(neutral_score, choice_actions, residual, float(alpha))
        candidate_choice_score = candidate_score + np.asarray(noise_row, dtype=np.float64) if noise_row is not None else candidate_score
        candidate_choice = _argmax_stable(candidate_choice_score)
        candidate_family = action_family_or_unknown(candidate_choice)
        primitive_changed = bool(candidate_choice != neutral_choice)
        family_changed = bool(candidate_family != neutral_family)
        candidate_probs = _softmax_masked_finite(candidate_score)
        candidate_family_probs = _family_probability_vector(candidate_probs)
        primitive_l1 = float(np.sum(np.abs(candidate_probs - neutral_probs)))
        family_l1 = float(sum(abs(float(candidate_family_probs.get(family, 0.0)) - float(neutral_family_probs.get(family, 0.0))) for family in FAMILY_ORDER))
        promoted_delta = _family_probability_delta(neutral_family_probs, candidate_family_probs, promoted)
        margin = decision_margin_payload(candidate_choice_score, choice_actions)
        family_margin = _family_score_margin(neutral_score, choice_actions, residual=residual, alpha=float(alpha))
        record["event_count"] = int(record.get("event_count", 0) or 0) + 1
        record["primitive_choice_change_count"] = int(record.get("primitive_choice_change_count", 0) or 0) + int(primitive_changed)
        record["family_choice_change_count"] = int(record.get("family_choice_change_count", 0) or 0) + int(family_changed)
        record["promoted_family_choice_count"] = int(record.get("promoted_family_choice_count", 0) or 0) + int(candidate_family == promoted)
        record["demoted_family_choice_count"] = int(record.get("demoted_family_choice_count", 0) or 0) + int(candidate_family == demoted)
        record["promoted_family_probability_delta_sum"] = float(record.get("promoted_family_probability_delta_sum", 0.0) or 0.0) + promoted_delta
        record["family_probability_l1_delta_sum"] = float(record.get("family_probability_l1_delta_sum", 0.0) or 0.0) + family_l1
        record["primitive_probability_l1_delta_sum"] = float(record.get("primitive_probability_l1_delta_sum", 0.0) or 0.0) + primitive_l1
        record["entropy_sum"] = float(record.get("entropy_sum", 0.0) or 0.0) + _entropy(candidate_probs)
        record["neutral_entropy_sum"] = float(record.get("neutral_entropy_sum", 0.0) or 0.0) + neutral_entropy
        record["top_two_margin_sum"] = float(record.get("top_two_margin_sum", 0.0) or 0.0) + float(margin.get("top_two_margin", 0.0) or 0.0)
        record["family_margin_sum"] = float(record.get("family_margin_sum", 0.0) or 0.0) + float(family_margin.get("top_two_margin", 0.0) or 0.0)
        record["meaningful_promoted_opportunity_count"] = int(record.get("meaningful_promoted_opportunity_count", 0) or 0) + int(promoted_opportunity)
        if family_changed and promoted_opportunity:
            record["choice_change_with_promoted_opportunity_count"] = int(record.get("choice_change_with_promoted_opportunity_count", 0) or 0) + 1
        if family_changed and not promoted_opportunity:
            record["choice_change_without_promoted_opportunity_count"] = int(record.get("choice_change_without_promoted_opportunity_count", 0) or 0) + 1
        _dict_increment(record, "source_class_counts", str(base_event.get("action_source_class", "unknown") or "unknown"), 1)
        _dict_increment(record, "policy_kind_counts", str(base_event.get("policy_kind", "unknown") or "unknown"), 1)
        _dict_increment(record, "candidate_family_counts", str(candidate_family or "unknown"), 1)
        if abs(float(alpha) - 1.0) <= 1e-12 and int(base_event.get("actual_sampled_action", -1)) != int(candidate_choice):
            summary["choice_reconstruction_mismatch_count_at_alpha_1"] = int(summary.get("choice_reconstruction_mismatch_count_at_alpha_1", 0) or 0) + 1
        if len(summary.get("surface_sample", []) or []) < int(summary.get("_event_sample_limit", 0) or 0) and (primitive_changed or float(alpha) in {0.0, 1.0, 5.0, 10.0}):
            summary.setdefault("surface_sample", []).append(
                {
                    "schema_version": FAMILY_HEAD_AUTHORITY_MAGNITUDE_SURFACE_SCHEMA_VERSION,
                    "run_id": base_event.get("run_id", ""),
                    "generation": base_event.get("generation", ""),
                    "tick": base_event.get("tick", ""),
                    "agent_id": base_event.get("agent_id", ""),
                    "policy_kind": base_event.get("policy_kind", ""),
                    "action_source_class": base_event.get("action_source_class", ""),
                    "alpha": float(alpha),
                    "neutral_action": int(neutral_choice),
                    "neutral_family": str(neutral_family),
                    "candidate_action": int(candidate_choice),
                    "candidate_family": str(candidate_family),
                    "primitive_choice_changed": int(primitive_changed),
                    "family_choice_changed": int(family_changed),
                    "promoted_family": str(promoted),
                    "demoted_family": str(demoted),
                    "meaningful_promoted_opportunity": int(promoted_opportunity),
                    "promoted_family_probability_delta": float(promoted_delta),
                    "primitive_probability_l1_delta": float(primitive_l1),
                    "family_probability_l1_delta": float(family_l1),
                    "candidate_entropy": float(_entropy(candidate_probs)),
                    "neutral_entropy": float(neutral_entropy),
                    "candidate_top_two_margin": margin.get("top_two_margin", ""),
                    "candidate_family_margin": family_margin.get("top_two_margin", ""),
                    "opportunity_actions": base_event.get("opportunity_actions", ""),
                    "attack_target_available": base_event.get("attack_target_available", 0),
                    "gather_resource_available": base_event.get("gather_resource_available", 0),
                    "rest_need_present": base_event.get("rest_need_present", 0),
                    "legal_move_count": base_event.get("legal_move_count", 0),
                }
            )


def _merge_family_head_magnitude_surface_delta(runner: Any, delta: dict[str, Any]) -> None:
    total = getattr(runner, "_policy_family_head_authority_magnitude_surface_total", None)
    if not isinstance(total, dict):
        total = _empty_family_head_magnitude_surface_summary(enabled=True)
    for key in ("event_count", "same_sample_noise_count", "choice_reconstruction_mismatch_count_at_alpha_1"):
        total[key] = int(total.get(key, 0) or 0) + int(delta.get(key, 0) or 0)
    for field in ("source_class_counts", "policy_kind_counts", "not_evaluable_reasons"):
        counter = Counter(total.get(field, {}) or {})
        counter.update(delta.get(field, {}) or {})
        total[field] = dict(sorted(counter.items()))
    surface = total.setdefault("alpha_surface", {})
    for alpha_key, record in (delta.get("alpha_surface") or {}).items():
        target = surface.setdefault(alpha_key, _empty_alpha_surface_record(float(record.get("alpha", alpha_key))))
        for key in (
            "event_count",
            "primitive_choice_change_count",
            "family_choice_change_count",
            "promoted_family_choice_count",
            "demoted_family_choice_count",
            "meaningful_promoted_opportunity_count",
            "choice_change_with_promoted_opportunity_count",
            "choice_change_without_promoted_opportunity_count",
        ):
            target[key] = int(target.get(key, 0) or 0) + int(record.get(key, 0) or 0)
        for key in (
            "promoted_family_probability_delta_sum",
            "family_probability_l1_delta_sum",
            "primitive_probability_l1_delta_sum",
            "entropy_sum",
            "neutral_entropy_sum",
            "top_two_margin_sum",
            "family_margin_sum",
        ):
            target[key] = float(target.get(key, 0.0) or 0.0) + float(record.get(key, 0.0) or 0.0)
        for field in ("source_class_counts", "policy_kind_counts", "candidate_family_counts"):
            counter = Counter(target.get(field, {}) or {})
            counter.update(record.get(field, {}) or {})
            target[field] = dict(sorted(counter.items()))
    limit = int(delta.get("_event_sample_limit", total.get("_event_sample_limit", 0)) or 0)
    if limit > 0:
        existing = list(total.get("surface_sample", []) or [])
        for event in list(delta.get("surface_sample", []) or []):
            if len(existing) >= limit:
                break
            existing.append(event)
        total["surface_sample"] = existing
    total["_event_sample_limit"] = int(limit)
    _finalize_family_head_magnitude_surface_summary(total)
    runner._policy_family_head_authority_magnitude_surface_total = total


def _finalize_family_head_magnitude_surface_summary(summary: dict[str, Any]) -> None:
    for record in (summary.get("alpha_surface") or {}).values():
        count = max(1, int(record.get("event_count", 0) or 0))
        for sum_key, mean_key in (
            ("promoted_family_probability_delta_sum", "promoted_family_probability_delta_mean"),
            ("family_probability_l1_delta_sum", "family_probability_l1_delta_mean"),
            ("primitive_probability_l1_delta_sum", "primitive_probability_l1_delta_mean"),
            ("entropy_sum", "entropy_mean"),
            ("neutral_entropy_sum", "neutral_entropy_mean"),
            ("top_two_margin_sum", "top_two_margin_mean"),
            ("family_margin_sum", "family_margin_mean"),
        ):
            record[mean_key] = float(record.get(sum_key, 0.0) or 0.0) / float(count)
        record["primitive_choice_change_rate"] = float(record.get("primitive_choice_change_count", 0) or 0) / float(count)
        record["family_choice_change_rate"] = float(record.get("family_choice_change_count", 0) or 0) / float(count)
        opp = max(1, int(record.get("meaningful_promoted_opportunity_count", 0) or 0))
        record["meaningful_promoted_opportunity_rate"] = float(record.get("meaningful_promoted_opportunity_count", 0) or 0) / float(count)
        record["choice_change_with_promoted_opportunity_rate"] = float(record.get("choice_change_with_promoted_opportunity_count", 0) or 0) / float(opp)


def _empty_opportunity_conditioned_candidate_summary(*, enabled: bool) -> dict[str, Any]:
    return {
        "schema_version": OPPORTUNITY_CONDITIONED_AUTHORITY_CANDIDATE_SCHEMA_VERSION,
        "enabled": bool(enabled),
        "candidate_ids": list(OPPORTUNITY_CONDITIONED_AUTHORITY_CANDIDATES),
        "l1_values": [float(value) for value in OPPORTUNITY_CONDITIONED_AUTHORITY_L1_VALUES],
        "event_count": 0 if enabled else None,
        "same_sample_noise_count": 0 if enabled else None,
        "not_evaluable_reasons": {} if enabled else {"reporting_disabled": 1},
        "candidate_surface": {},
        "source_class_counts": {},
        "policy_kind_counts": {},
        "event_sample": [],
        "_event_sample_limit": 0,
    }


def _empty_opportunity_conditioned_candidate_record(candidate_id: str, l1_value: float) -> dict[str, Any]:
    return {
        "candidate_id": str(candidate_id),
        "stored_l1": float(l1_value),
        "event_count": 0,
        "not_evaluable_count": 0,
        "primitive_choice_change_count": 0,
        "family_choice_change_count": 0,
        "choice_change_with_promoted_opportunity_count": 0,
        "choice_change_without_promoted_opportunity_count": 0,
        "no_opportunity_suppression_count": 0,
        "promoted_family_probability_delta_sum": 0.0,
        "family_probability_l1_delta_sum": 0.0,
        "primitive_probability_l1_delta_sum": 0.0,
        "expressed_l1_sum": 0.0,
        "expressed_l1_max": 0.0,
        "entropy_sum": 0.0,
        "neutral_entropy_sum": 0.0,
        "family_margin_sum": 0.0,
        "top_two_margin_sum": 0.0,
        "source_class_counts": {},
        "policy_kind_counts": {},
        "promoted_family_counts": {},
        "not_evaluable_reasons": {},
    }


def _update_opportunity_conditioned_candidate_summary(
    summary: dict[str, Any],
    base_event: dict[str, Any],
    neutral_row: np.ndarray,
    choice_actions: list[int],
    residual: dict[str, float],
    noise_row: np.ndarray | None,
) -> None:
    neutral_score = np.asarray(neutral_row, dtype=np.float64)
    if noise_row is not None:
        neutral_choice_score = neutral_score + np.asarray(noise_row, dtype=np.float64)
        summary["same_sample_noise_count"] = int(summary.get("same_sample_noise_count", 0) or 0) + 1
    else:
        neutral_choice_score = neutral_score
    neutral_choice = _argmax_stable(neutral_choice_score)
    neutral_family = action_family_or_unknown(neutral_choice)
    neutral_probs = _softmax_masked_finite(neutral_score)
    neutral_family_probs = _family_probability_vector(neutral_probs)
    neutral_entropy = _entropy(neutral_probs)
    opportunity = _family_opportunity_gate_values(base_event)
    valid_families = _valid_families_for_actions(choice_actions)
    direction = _residual_direction_for_l1(residual, 1.0)
    promoted = _promoted_family(direction)
    promoted_opportunity = _meaningful_family_opportunity(promoted, base_event)
    summary["event_count"] = int(summary.get("event_count", 0) or 0) + 1
    _dict_increment(summary, "source_class_counts", str(base_event.get("action_source_class", "unknown") or "unknown"), 1)
    _dict_increment(summary, "policy_kind_counts", str(base_event.get("policy_kind", "unknown") or "unknown"), 1)
    for l1_value in OPPORTUNITY_CONDITIONED_AUTHORITY_L1_VALUES:
        scaled = _residual_direction_for_l1(residual, float(l1_value))
        unconditioned_scores = _authority_scores_for_alpha(neutral_score, choice_actions, scaled, 1.0)
        unconditioned_choice_score = unconditioned_scores + np.asarray(noise_row, dtype=np.float64) if noise_row is not None else unconditioned_scores
        unconditioned_family_changed = action_family_or_unknown(_argmax_stable(unconditioned_choice_score)) != neutral_family
        for candidate_id in OPPORTUNITY_CONDITIONED_AUTHORITY_CANDIDATES:
            candidate_key = f"{candidate_id}|{float(l1_value):g}"
            record = summary.setdefault("candidate_surface", {}).setdefault(
                candidate_key,
                _empty_opportunity_conditioned_candidate_record(candidate_id, float(l1_value)),
            )
            record["event_count"] = int(record.get("event_count", 0) or 0) + 1
            _dict_increment(record, "source_class_counts", str(base_event.get("action_source_class", "unknown") or "unknown"), 1)
            _dict_increment(record, "policy_kind_counts", str(base_event.get("policy_kind", "unknown") or "unknown"), 1)
            _dict_increment(record, "promoted_family_counts", promoted or "unknown", 1)
            expressed, not_evaluable_reason = _candidate_expressed_residual(candidate_id, scaled, opportunity, valid_families)
            if not_evaluable_reason:
                record["not_evaluable_count"] = int(record.get("not_evaluable_count", 0) or 0) + 1
                _dict_increment(record, "not_evaluable_reasons", not_evaluable_reason, 1)
                continue
            candidate_score = _authority_scores_for_alpha(neutral_score, choice_actions, expressed, 1.0)
            candidate_choice_score = candidate_score + np.asarray(noise_row, dtype=np.float64) if noise_row is not None else candidate_score
            candidate_choice = _argmax_stable(candidate_choice_score)
            candidate_family = action_family_or_unknown(candidate_choice)
            primitive_changed = bool(candidate_choice != neutral_choice)
            family_changed = bool(candidate_family != neutral_family)
            candidate_probs = _softmax_masked_finite(candidate_score)
            candidate_family_probs = _family_probability_vector(candidate_probs)
            primitive_l1 = float(np.sum(np.abs(candidate_probs - neutral_probs)))
            family_l1 = float(sum(abs(float(candidate_family_probs.get(family, 0.0)) - float(neutral_family_probs.get(family, 0.0))) for family in FAMILY_ORDER))
            promoted_delta = _family_probability_delta(neutral_family_probs, candidate_family_probs, promoted)
            expressed_l1 = float(sum(abs(float(expressed.get(family, 0.0) or 0.0)) for family in FAMILY_ORDER))
            margin = decision_margin_payload(candidate_choice_score, choice_actions)
            family_margin = _family_score_margin(neutral_score, choice_actions, residual=expressed, alpha=1.0)
            record["primitive_choice_change_count"] = int(record.get("primitive_choice_change_count", 0) or 0) + int(primitive_changed)
            record["family_choice_change_count"] = int(record.get("family_choice_change_count", 0) or 0) + int(family_changed)
            if family_changed and promoted_opportunity:
                record["choice_change_with_promoted_opportunity_count"] = int(record.get("choice_change_with_promoted_opportunity_count", 0) or 0) + 1
            if family_changed and not promoted_opportunity:
                record["choice_change_without_promoted_opportunity_count"] = int(record.get("choice_change_without_promoted_opportunity_count", 0) or 0) + 1
            if unconditioned_family_changed and not family_changed and not promoted_opportunity:
                record["no_opportunity_suppression_count"] = int(record.get("no_opportunity_suppression_count", 0) or 0) + 1
            record["promoted_family_probability_delta_sum"] = float(record.get("promoted_family_probability_delta_sum", 0.0) or 0.0) + promoted_delta
            record["family_probability_l1_delta_sum"] = float(record.get("family_probability_l1_delta_sum", 0.0) or 0.0) + family_l1
            record["primitive_probability_l1_delta_sum"] = float(record.get("primitive_probability_l1_delta_sum", 0.0) or 0.0) + primitive_l1
            record["expressed_l1_sum"] = float(record.get("expressed_l1_sum", 0.0) or 0.0) + expressed_l1
            record["expressed_l1_max"] = max(float(record.get("expressed_l1_max", 0.0) or 0.0), expressed_l1)
            record["entropy_sum"] = float(record.get("entropy_sum", 0.0) or 0.0) + _entropy(candidate_probs)
            record["neutral_entropy_sum"] = float(record.get("neutral_entropy_sum", 0.0) or 0.0) + neutral_entropy
            record["top_two_margin_sum"] = float(record.get("top_two_margin_sum", 0.0) or 0.0) + float(margin.get("top_two_margin", 0.0) or 0.0)
            record["family_margin_sum"] = float(record.get("family_margin_sum", 0.0) or 0.0) + float(family_margin.get("top_two_margin", 0.0) or 0.0)
            if len(summary.get("event_sample", []) or []) < int(summary.get("_event_sample_limit", 0) or 0) and (family_changed or candidate_id != "current_unconditioned"):
                summary.setdefault("event_sample", []).append(
                    {
                        "schema_version": OPPORTUNITY_CONDITIONED_AUTHORITY_CANDIDATE_SCHEMA_VERSION,
                        "candidate_id": candidate_id,
                        "stored_l1": float(l1_value),
                        "generation": base_event.get("generation", ""),
                        "tick": base_event.get("tick", ""),
                        "agent_id": base_event.get("agent_id", ""),
                        "policy_kind": base_event.get("policy_kind", ""),
                        "action_source_class": base_event.get("action_source_class", ""),
                        "neutral_action": int(neutral_choice),
                        "neutral_family": str(neutral_family),
                        "candidate_action": int(candidate_choice),
                        "candidate_family": str(candidate_family),
                        "family_choice_changed": int(family_changed),
                        "primitive_choice_changed": int(primitive_changed),
                        "promoted_family": str(promoted),
                        "promoted_opportunity": int(promoted_opportunity),
                        "opportunity_values": json.dumps(opportunity, sort_keys=True, separators=(",", ":")),
                        "expressed_residual": json.dumps({family: round(float(expressed.get(family, 0.0) or 0.0), 12) for family in FAMILY_ORDER}, sort_keys=True, separators=(",", ":")),
                        "expressed_l1": float(expressed_l1),
                        "family_probability_l1_delta": float(family_l1),
                        "primitive_probability_l1_delta": float(primitive_l1),
                        "promoted_family_probability_delta": float(promoted_delta),
                    }
                )


def _merge_opportunity_conditioned_candidate_delta(runner: Any, delta: dict[str, Any]) -> None:
    total = getattr(runner, "_policy_family_head_opportunity_conditioned_candidate_total", None)
    if not isinstance(total, dict):
        total = _empty_opportunity_conditioned_candidate_summary(enabled=True)
    for key in ("event_count", "same_sample_noise_count"):
        total[key] = int(total.get(key, 0) or 0) + int(delta.get(key, 0) or 0)
    for field in ("source_class_counts", "policy_kind_counts", "not_evaluable_reasons"):
        counter = Counter(total.get(field, {}) or {})
        counter.update(delta.get(field, {}) or {})
        total[field] = dict(sorted(counter.items()))
    surface = total.setdefault("candidate_surface", {})
    for candidate_key, record in (delta.get("candidate_surface") or {}).items():
        target = surface.setdefault(
            candidate_key,
            _empty_opportunity_conditioned_candidate_record(str(record.get("candidate_id", "")), float(record.get("stored_l1", 0.0) or 0.0)),
        )
        for key in (
            "event_count",
            "not_evaluable_count",
            "primitive_choice_change_count",
            "family_choice_change_count",
            "choice_change_with_promoted_opportunity_count",
            "choice_change_without_promoted_opportunity_count",
            "no_opportunity_suppression_count",
        ):
            target[key] = int(target.get(key, 0) or 0) + int(record.get(key, 0) or 0)
        for key in (
            "promoted_family_probability_delta_sum",
            "family_probability_l1_delta_sum",
            "primitive_probability_l1_delta_sum",
            "expressed_l1_sum",
            "entropy_sum",
            "neutral_entropy_sum",
            "family_margin_sum",
            "top_two_margin_sum",
        ):
            target[key] = float(target.get(key, 0.0) or 0.0) + float(record.get(key, 0.0) or 0.0)
        target["expressed_l1_max"] = max(float(target.get("expressed_l1_max", 0.0) or 0.0), float(record.get("expressed_l1_max", 0.0) or 0.0))
        for field in ("source_class_counts", "policy_kind_counts", "promoted_family_counts", "not_evaluable_reasons"):
            counter = Counter(target.get(field, {}) or {})
            counter.update(record.get(field, {}) or {})
            target[field] = dict(sorted(counter.items()))
    limit = int(delta.get("_event_sample_limit", total.get("_event_sample_limit", 0)) or 0)
    if limit > 0:
        existing = list(total.get("event_sample", []) or [])
        for event in list(delta.get("event_sample", []) or []):
            if len(existing) >= limit:
                break
            existing.append(event)
        total["event_sample"] = existing
    total["_event_sample_limit"] = int(limit)
    _finalize_opportunity_conditioned_candidate_summary(total)
    runner._policy_family_head_opportunity_conditioned_candidate_total = total


def _finalize_opportunity_conditioned_candidate_summary(summary: dict[str, Any]) -> None:
    for record in (summary.get("candidate_surface") or {}).values():
        count = max(1, int(record.get("event_count", 0) or 0))
        evaluable = max(1, int(record.get("event_count", 0) or 0) - int(record.get("not_evaluable_count", 0) or 0))
        for sum_key, mean_key in (
            ("promoted_family_probability_delta_sum", "promoted_family_probability_delta_mean"),
            ("family_probability_l1_delta_sum", "family_probability_l1_delta_mean"),
            ("primitive_probability_l1_delta_sum", "primitive_probability_l1_delta_mean"),
            ("expressed_l1_sum", "expressed_l1_mean"),
            ("entropy_sum", "entropy_mean"),
            ("neutral_entropy_sum", "neutral_entropy_mean"),
            ("family_margin_sum", "family_margin_mean"),
            ("top_two_margin_sum", "top_two_margin_mean"),
        ):
            record[mean_key] = float(record.get(sum_key, 0.0) or 0.0) / float(evaluable)
        record["not_evaluable_rate"] = float(record.get("not_evaluable_count", 0) or 0) / float(count)
        record["primitive_choice_change_rate"] = float(record.get("primitive_choice_change_count", 0) or 0) / float(evaluable)
        record["family_choice_change_rate"] = float(record.get("family_choice_change_count", 0) or 0) / float(evaluable)
        record["choice_change_with_promoted_opportunity_rate"] = float(record.get("choice_change_with_promoted_opportunity_count", 0) or 0) / float(evaluable)
        record["choice_change_without_promoted_opportunity_rate"] = float(record.get("choice_change_without_promoted_opportunity_count", 0) or 0) / float(evaluable)
        record["no_opportunity_suppression_rate"] = float(record.get("no_opportunity_suppression_count", 0) or 0) / float(evaluable)


def _alpha_key(alpha: float) -> str:
    return f"{float(alpha):g}"


def _promoted_family(residual: dict[str, float]) -> str:
    vals = [(family, float(residual.get(family, 0.0) or 0.0)) for family in FAMILY_ORDER]
    vals.sort(key=lambda item: (-item[1], item[0]))
    return vals[0][0] if vals and vals[0][1] > 1e-12 else ""


def _demoted_family(residual: dict[str, float]) -> str:
    vals = [(family, float(residual.get(family, 0.0) or 0.0)) for family in FAMILY_ORDER]
    vals.sort(key=lambda item: (item[1], item[0]))
    return vals[0][0] if vals and vals[0][1] < -1e-12 else ""


def _meaningful_family_opportunity(family: str, event: dict[str, Any]) -> bool:
    if family == "movement":
        return int(event.get("legal_move_count", 0) or 0) > 0
    if family == "gather":
        return bool(int(event.get("gather_resource_available", 0) or 0))
    if family == "rest":
        return bool(int(event.get("rest_need_present", 0) or 0))
    if family == "attack":
        return bool(int(event.get("attack_target_available", 0) or 0) or int(event.get("attack_executable", 0) or 0))
    return False


def _family_probability_vector(primitive_probs: np.ndarray) -> dict[str, float]:
    arr = np.asarray(primitive_probs, dtype=np.float64).reshape((-1,))
    out: dict[str, float] = {}
    for family in FAMILY_ORDER:
        out[family] = float(sum(float(arr[int(action)]) for action in FAMILY_TO_ACTION_IDS[family] if 0 <= int(action) < arr.size))
    return out


def _family_probability_delta(before: dict[str, float], after: dict[str, float], family: str) -> float:
    if not family:
        return 0.0
    return float(after.get(family, 0.0) - before.get(family, 0.0))


def _entropy(probs: np.ndarray) -> float:
    arr = np.asarray(probs, dtype=np.float64).reshape((-1,))
    finite = arr[np.isfinite(arr) & (arr > 0.0)]
    if finite.size == 0:
        return 0.0
    return float(-np.sum(finite * np.log(finite)))


def _family_opportunity_gate_values(event: dict[str, Any]) -> dict[str, float]:
    legal_move_count = max(0, int(event.get("legal_move_count", 0) or 0))
    attack_available = int(event.get("attack_target_available", 0) or 0) or int(event.get("attack_executable", 0) or 0)
    return {
        "movement_binary": 1.0 if legal_move_count > 0 else 0.0,
        "movement_intensity": max(0.0, min(1.0, float(legal_move_count) / 4.0)),
        "gather_binary": 1.0 if int(event.get("gather_resource_available", 0) or 0) else 0.0,
        "gather_intensity": 1.0 if int(event.get("gather_resource_available", 0) or 0) else 0.0,
        "rest_binary": 1.0 if int(event.get("rest_need_present", 0) or 0) else 0.0,
        "rest_intensity": 1.0 if int(event.get("rest_need_present", 0) or 0) else 0.0,
        "attack_binary": 1.0 if attack_available else 0.0,
        "attack_intensity": 1.0 if attack_available else 0.0,
    }


def _valid_families_for_actions(valid_actions: list[int]) -> list[str]:
    valid = {int(action) for action in valid_actions}
    out: list[str] = []
    for family in FAMILY_ORDER:
        if any(int(action) in valid for action in FAMILY_TO_ACTION_IDS[family]):
            out.append(family)
    return out or list(FAMILY_ORDER)


def _residual_direction_for_l1(residual: dict[str, float], target_l1: float) -> dict[str, float]:
    values = {family: float(residual.get(family, 0.0) or 0.0) for family in FAMILY_ORDER}
    current_l1 = sum(abs(value) for value in values.values())
    if current_l1 <= 1e-12 or target_l1 <= 0.0:
        return {family: 0.0 for family in FAMILY_ORDER}
    scale = float(target_l1) / float(current_l1)
    return {family: float(value) * scale for family, value in values.items()}


def _center_over_valid_families(values: dict[str, float], valid_families: list[str]) -> dict[str, float]:
    valid = [family for family in valid_families if family in FAMILY_ORDER]
    if not valid:
        return {family: 0.0 for family in FAMILY_ORDER}
    mean_value = sum(float(values.get(family, 0.0) or 0.0) for family in valid) / float(len(valid))
    return {
        family: (float(values.get(family, 0.0) or 0.0) - mean_value) if family in valid else 0.0
        for family in FAMILY_ORDER
    }


def _candidate_expressed_residual(
    candidate_id: str,
    residual: dict[str, float],
    opportunity: dict[str, float],
    valid_families: list[str],
) -> tuple[dict[str, float], str]:
    if candidate_id == "current_unconditioned":
        return dict(residual), ""
    if candidate_id == "no_policy_gate":
        return ({family: 0.0 for family in FAMILY_ORDER}, "")
    if candidate_id == "existing_drive_gate":
        return ({family: 0.0 for family in FAMILY_ORDER}, "existing_drive_source_not_available_in_choice_margin_event")
    gated: dict[str, float] = {}
    for family in FAMILY_ORDER:
        if candidate_id == "binary_executable_gate":
            gate = float(opportunity.get(f"{family}_binary", 0.0) or 0.0)
        elif candidate_id == "bounded_opportunity_intensity_gate":
            gate = float(opportunity.get(f"{family}_intensity", 0.0) or 0.0)
        else:
            return ({family: 0.0 for family in FAMILY_ORDER}, "unknown_candidate")
        gated[family] = float(residual.get(family, 0.0) or 0.0) * max(0.0, min(1.0, gate))
    return _center_over_valid_families(gated, valid_families), ""


def _agent_family_head_residual(agent: Any) -> dict[str, float]:
    genome = getattr(agent, "genome", None)
    block = getattr(genome, FAMILY_HEAD_RESIDUAL_FIELD, None)
    try:
        return {family: float(value) for family, value in family_head_residual_effective_map(block).items()}
    except Exception:
        return {family: 0.0 for family in FAMILY_ORDER}


def _argmax_stable(scores: np.ndarray) -> int:
    arr = np.asarray(scores, dtype=np.float64).reshape((-1,))
    finite = np.isfinite(arr)
    if not bool(np.any(finite)):
        return -1
    best_score = float(np.max(arr[finite]))
    candidates = [idx for idx, value in enumerate(arr.tolist()) if np.isfinite(value) and float(value) == best_score]
    return int(min(candidates)) if candidates else -1


def _residual_alpha_cap(residual: dict[str, float], *, component_bound: float = 1.0, l1_cap: float = 2.0) -> float:
    values = [abs(float(residual.get(family, 0.0) or 0.0)) for family in FAMILY_ORDER]
    nonzero = [value for value in values if value > 1e-12]
    if not nonzero:
        return 0.0
    component_alpha = float(component_bound) / max(nonzero)
    l1 = sum(values)
    l1_alpha = float(l1_cap) / l1 if l1 > 1e-12 else component_alpha
    return float(max(0.0, min(component_alpha, l1_alpha)))


def _authority_scores_for_alpha(base_logits: np.ndarray, valid_actions: list[int], residual: dict[str, float], alpha: float) -> np.ndarray:
    masked = _masked_logits(base_logits, valid_actions)
    family_scores: dict[str, float] = {}
    for family in FAMILY_ORDER:
        vals = [float(masked[int(action)]) for action in FAMILY_TO_ACTION_IDS[family] if 0 <= int(action) < masked.size and np.isfinite(masked[int(action)])]
        family_scores[family] = _logsumexp_np(vals) + float(alpha) * float(residual.get(family, 0.0) or 0.0) if vals else -np.inf
    family_norm = _logsumexp_np([value for value in family_scores.values() if np.isfinite(value)])
    out = np.full_like(masked, -np.inf, dtype=np.float64)
    for family in FAMILY_ORDER:
        actions = [int(action) for action in FAMILY_TO_ACTION_IDS[family] if 0 <= int(action) < masked.size and np.isfinite(masked[int(action)])]
        if not actions or not np.isfinite(family_scores[family]) or not np.isfinite(family_norm):
            continue
        within_norm = _logsumexp_np([float(masked[action]) for action in actions])
        for action in actions:
            out[action] = (family_scores[family] - family_norm) + (float(masked[action]) - within_norm)
    return out


def _family_score_margin(base_logits: np.ndarray, valid_actions: list[int], *, residual: dict[str, float] | None, alpha: float) -> dict[str, Any]:
    masked = _masked_logits(base_logits, valid_actions)
    values: list[tuple[str, float]] = []
    for family in FAMILY_ORDER:
        vals = [float(masked[int(action)]) for action in FAMILY_TO_ACTION_IDS[family] if 0 <= int(action) < masked.size and np.isfinite(masked[int(action)])]
        if vals:
            values.append((family, _logsumexp_np(vals) + float(alpha) * float((residual or {}).get(family, 0.0) or 0.0)))
    values.sort(key=lambda item: (-item[1], item[0]))
    if len(values) < 2:
        return {"top_family": values[0][0] if values else "", "second_family": "", "top_two_margin": 0.0}
    return {"top_family": values[0][0], "second_family": values[1][0], "top_two_margin": float(values[0][1] - values[1][1])}


def _find_crossing_alpha(
    base_logits: np.ndarray,
    valid_actions: list[int],
    *,
    residual: dict[str, float],
    alpha_cap: float,
    noise: np.ndarray | None,
    mode: str,
) -> float | None:
    if alpha_cap <= 0.0:
        return None
    def chosen_at(alpha: float) -> Any:
        scores = _authority_scores_for_alpha(base_logits, valid_actions, residual, float(alpha))
        if mode == "family":
            margin = _family_score_margin(base_logits, valid_actions, residual=residual, alpha=float(alpha))
            return str(margin.get("top_family", "") or "")
        if noise is not None:
            scores = scores + np.asarray(noise, dtype=np.float64)
        return _argmax_stable(scores)

    base_choice = chosen_at(0.0)
    if chosen_at(alpha_cap) == base_choice:
        return None
    lo = 0.0
    hi = float(alpha_cap)
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if chosen_at(mid) == base_choice:
            lo = mid
        else:
            hi = mid
    return float(hi)


def _logsumexp_np(values: list[float]) -> float:
    vals = np.asarray([float(v) for v in values if np.isfinite(v)], dtype=np.float64)
    if vals.size == 0:
        return -np.inf
    m = float(np.max(vals))
    return float(m + np.log(float(np.sum(np.exp(vals - m)))))


def _softmax_masked_finite(scores: np.ndarray) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float64).reshape((-1,))
    finite = np.isfinite(arr)
    out = np.zeros_like(arr, dtype=np.float64)
    if not bool(np.any(finite)):
        return out
    m = float(np.max(arr[finite]))
    out[finite] = np.exp(arr[finite] - m)
    denom = float(np.sum(out))
    return out / denom if denom > 0.0 else np.zeros_like(out)


def _family_head_choice_margin_sample_limit(runner: Any) -> int:
    try:
        limit = int(getattr(getattr(runner, "sim_cfg", None), "policy_family_head_authority_choice_margin_event_sample_limit", 2048) or 0)
    except Exception:
        limit = 2048
    return max(0, min(50000, int(limit)))


def _family_head_magnitude_surface_sample_limit(runner: Any) -> int:
    try:
        limit = int(getattr(getattr(runner, "sim_cfg", None), "policy_family_head_authority_magnitude_surface_event_sample_limit", 2048) or 0)
    except Exception:
        limit = 2048
    return max(0, min(50000, int(limit)))


def _family_head_opportunity_conditioned_candidate_sample_limit(runner: Any) -> int:
    try:
        limit = int(getattr(getattr(runner, "sim_cfg", None), "policy_family_head_opportunity_conditioned_authority_candidate_event_sample_limit", 2048) or 0)
    except Exception:
        limit = 2048
    return max(0, min(50000, int(limit)))


def _empty_decision_margin_summary() -> dict[str, Any]:
    return {
        "schema_version": DECISION_MARGIN_SCHEMA_VERSION,
        "evaluated_decision_count": 0,
        "chosen_action_change_count": 0,
        "rank_change_count": 0,
        "margin_evaluable_count": 0,
        "one_valid_action_count": 0,
        "score_delta_sum": 0.0,
        "margin_sum": 0.0,
        "max_abs_score_delta_max": 0.0,
        "action_mask_blocked_count": 0,
        "target_unavailable_count": 0,
        "clipping_count": 0,
        "saturation_count": 0,
        "variant_class_counts": {},
        "changed_effective_field_counts": {},
        "not_evaluable_reasons": {},
        "sample_limit_reached": False,
    }


def _update_decision_margin_summary(summary: dict[str, Any], event: dict[str, Any]) -> None:
    summary["evaluated_decision_count"] = int(summary.get("evaluated_decision_count", 0) or 0) + 1
    summary["chosen_action_change_count"] = int(summary.get("chosen_action_change_count", 0) or 0) + int(event.get("chosen_action_changed", 0) or 0)
    summary["rank_change_count"] = int(summary.get("rank_change_count", 0) or 0) + int(event.get("rank_changed", 0) or 0)
    margin_evaluable = bool(event.get("actual_margin_evaluable", False))
    if "actual_margin_evaluable" not in event:
        margin_evaluable = str(event.get("actual_top_two_margin", "")) != "" and int(event.get("valid_action_count", 2) or 0) >= 2
    if margin_evaluable:
        summary["margin_evaluable_count"] = int(summary.get("margin_evaluable_count", 0) or 0) + 1
        summary["margin_sum"] = float(summary.get("margin_sum", 0.0) or 0.0) + float(event.get("actual_top_two_margin", 0.0) or 0.0)
    else:
        summary["one_valid_action_count"] = int(summary.get("one_valid_action_count", 0) or 0) + 1
    summary["score_delta_sum"] = float(summary.get("score_delta_sum", 0.0) or 0.0) + float(event.get("max_abs_score_delta", 0.0) or 0.0)
    summary["max_abs_score_delta_max"] = max(float(summary.get("max_abs_score_delta_max", 0.0) or 0.0), float(event.get("max_abs_score_delta", 0.0) or 0.0))
    summary["action_mask_blocked_count"] = int(summary.get("action_mask_blocked_count", 0) or 0) + int(event.get("action_mask_blocked", 0) or 0)
    summary["target_unavailable_count"] = int(summary.get("target_unavailable_count", 0) or 0) + int(event.get("target_unavailable", 0) or 0)
    clipped = [v for v in str(event.get("clipped_fields", "") or "").split(",") if v]
    saturated = [v for v in str(event.get("saturated_fields", "") or "").split(",") if v]
    summary["clipping_count"] = int(summary.get("clipping_count", 0) or 0) + len(clipped)
    summary["saturation_count"] = int(summary.get("saturation_count", 0) or 0) + len(saturated)
    _dict_increment(summary, "variant_class_counts", str(event.get("variant_class", "not_evaluable") or "not_evaluable"), 1)
    for field in str(event.get("changed_effective_fields", "") or "").split(","):
        if field:
            _dict_increment(summary, "changed_effective_field_counts", field, 1)


def _bump_summary_reason(summary: dict[str, Any], reason: str) -> None:
    _dict_increment(summary, "not_evaluable_reasons", str(reason or "unknown"), 1)


def _valid_actions_for_agent(runner: Any, alive: list[Any], alive_index: int) -> tuple[list[int], dict[str, Any]]:
    agent = alive[alive_index] if 0 <= alive_index < len(alive) else None
    env = getattr(runner, "env", None)
    if env is None or agent is None:
        return [], {"reason": "missing_env_or_agent"}
    try:
        agent_grid = np.asarray(getattr(env, "agent_grid"), dtype=np.int32)
        grid = np.asarray(getattr(env, "grid"), dtype=np.int8)
        pos_y = np.asarray(getattr(env, "agent_pos_y"), dtype=np.int32)
        pos_x = np.asarray(getattr(env, "agent_pos_x"), dtype=np.int32)
        aid = int(getattr(agent, "agent_id", -1))
        y, x = int(pos_y[aid]), int(pos_x[aid])
        h, w = agent_grid.shape
    except Exception:
        return [], {"reason": "position_unavailable"}
    if y < 0 or x < 0 or y >= h or x >= w:
        return [], {"reason": "agent_out_of_bounds"}
    valid: list[int] = []
    for action, (ny, nx) in enumerate(((y - 1, x), (y + 1, x), (y, x + 1), (y, x - 1))):
        if 0 <= ny < h and 0 <= nx < w and int(agent_grid[ny, nx]) == -1:
            valid.append(action)
    id_to_idx = {int(getattr(a, "agent_id", -1)): i for i, a in enumerate(alive)}
    healths = np.asarray([float(getattr(a, "health", 0.0)) for a in alive], dtype=np.float64)
    attack_target = _attack_target_available(agent_grid, id_to_idx, healths, y, x, h, w, alive_index)
    energy = float(getattr(agent, "energy", 0.0) or 0.0)
    attack_executable = bool(attack_target and energy > 0.0 and float(getattr(runner.env_cfg, "attack_damage", 0.0)) > 0.0)
    if attack_executable:
        valid.append(int(ACT_ATTACK))
    threshold = _patch_threshold(env)
    patch_amount = getattr(env, "patch_food_amount", None)
    if patch_amount is not None:
        gather_available = bool(float(np.asarray(patch_amount, dtype=np.float32)[y, x]) > threshold)
    else:
        gather_available = bool(grid[y, x] == FOOD)
    if gather_available:
        valid.append(int(ACT_GATHER))
    rest_need = _rest_need_present(runner, agent, float(getattr(agent, "health", 0.0) or 0.0), float(getattr(agent, "injury", 0.0) or 0.0))
    valid.append(int(ACT_REST))
    return sorted(set(valid)), {
        "attack_target_available": attack_target,
        "attack_executable": attack_executable,
        "gather_resource_available": gather_available,
        "rest_need_present": rest_need,
        "legal_move_count": sum(1 for a in valid if 0 <= int(a) <= 3),
    }


def _live_expression_summary(runner: Any, agent: Any) -> dict[str, Any]:
    out = {
        "changed_effective_fields": [],
        "effective_field_delta_json": "{}",
        "clipped_fields": [],
        "saturated_fields": [],
        "developmental_stage": "",
    }
    try:
        expr = runner.resolve_developmental_expression_for_agent(agent)
    except Exception:
        return out
    if not isinstance(expr, dict):
        return out
    summary = expr.get("structural_live_semantics", {}) if isinstance(expr.get("structural_live_semantics", {}), dict) else {}
    before = summary.get("effective_fields_before", {}) if isinstance(summary.get("effective_fields_before", {}), dict) else {}
    after = summary.get("effective_fields_after", {}) if isinstance(summary.get("effective_fields_after", {}), dict) else {}
    deltas: dict[str, float] = {}
    clipped: list[str] = []
    saturated: list[str] = []
    for field in ACTION_REACHABLE_FIELDS:
        if field not in before or field not in after:
            continue
        try:
            b = float(before.get(field, 0.0))
            a = float(after.get(field, 0.0))
        except Exception:
            continue
        deltas[field] = float(a - b)
        lo, hi = FIELD_BOUNDS[field]
        if a <= float(lo) or a >= float(hi):
            clipped.append(field)
        # Saturation is stricter than clipping: values inside 2 percent of either bound.
        span = max(1e-12, float(hi) - float(lo))
        if a <= float(lo) + 0.02 * span or a >= float(hi) - 0.02 * span:
            saturated.append(field)
    out["changed_effective_fields"] = sorted(str(v) for v in list(summary.get("changed_effective_fields", []) or []))
    out["effective_field_delta_json"] = _json_dict(deltas)
    out["clipped_fields"] = sorted(set(clipped))
    out["saturated_fields"] = sorted(set(saturated))
    out["developmental_stage"] = str(expr.get("stage_id", "") or "")
    return out


def _variant_identity_for_agent(runner: Any, agent: Any) -> dict[str, Any]:
    if not bool(getattr(getattr(runner, "sim_cfg", None), "structural_variant_identity_reporting_enable", False)):
        return {"variant_class": "not_evaluable"}
    try:
        from evolution.structural_variant_identity import agent_identity

        return agent_identity(
            agent,
            canonical_semantic_digest=str(getattr(runner, "_structural_variant_identity_canonical_semantic_digest", "") or ""),
        )
    except Exception:
        return {"variant_class": "not_evaluable"}


def _softmax(row: np.ndarray) -> np.ndarray:
    arr = np.asarray(row, dtype=np.float64)
    if arr.size == 0:
        return arr
    shifted = arr - float(np.max(arr))
    exp = np.exp(shifted)
    denom = float(np.sum(exp))
    return exp / denom if denom > 0.0 else np.zeros_like(arr)


def _json_list(values: np.ndarray) -> str:
    parts = []
    for v in np.asarray(values, dtype=np.float64).reshape((-1,)).tolist():
        fv = float(v)
        parts.append(f"{fv:.9g}" if np.isfinite(fv) else "null")
    return "[" + ",".join(parts) + "]"


def _json_load_list(value: Any) -> list[float]:
    import json

    try:
        vals = json.loads(str(value or "[]"))
    except Exception:
        return []
    if not isinstance(vals, list):
        return []
    out: list[float] = []
    for val in vals:
        try:
            out.append(float(val))
        except Exception:
            pass
    return out


def _json_dict(payload: dict[str, float]) -> str:
    import json

    return json.dumps({str(k): float(v) for k, v in sorted(payload.items())}, sort_keys=True, separators=(",", ":"))


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().to("cpu").numpy()
    return np.asarray(value)


def _runner_generation(runner: Any) -> int:
    try:
        if hasattr(runner, "_dev_generation_index"):
            return int(runner._dev_generation_index())
    except Exception:
        pass
    return int(getattr(runner, "_eiel_fdp_cur_gen", getattr(runner, "_current_gen", -1)) or -1)


def _runner_tick(runner: Any) -> int:
    try:
        if hasattr(runner, "_dev_tick_index"):
            return int(runner._dev_tick_index())
    except Exception:
        pass
    return int(getattr(runner, "_eiel_fdp_cur_tick", getattr(runner, "tick", -1)) or -1)


def _action_family_reporting_enabled(runner: Any) -> bool:
    cfg = getattr(runner, "sim_cfg", None)
    return bool(
        getattr(cfg, "structural_policy_action_family_mapping_reporting_enable", False)
        or getattr(cfg, "structural_action_consequence_telemetry_reporting_enable", False)
        or getattr(cfg, "policy_family_head_shadow_enable", False)
        or getattr(cfg, "policy_family_head_shadow_record_events", False)
    )


def _policy_family_head_shadow_enabled(runner: Any) -> bool:
    cfg = getattr(runner, "sim_cfg", None)
    return bool(
        getattr(cfg, "policy_family_head_shadow_enable", False)
        or getattr(cfg, "policy_family_head_shadow_record_events", False)
    )


def _policy_family_head_shadow_record_events(runner: Any) -> bool:
    return bool(getattr(getattr(runner, "sim_cfg", None), "policy_family_head_shadow_record_events", False))


def _source_class_for_slot(*, base: str, policy_kind: str, action_slot: int, slot_count: int) -> str:
    base = str(base or "neural_policy_direct")
    policy_kind = str(policy_kind or "policy")
    if int(slot_count) <= 1:
        if base == "neural_policy_stride":
            return f"neural_policy_stride_refill_{policy_kind}"
        if base.endswith(f"_{policy_kind}") or base == "neural_policy_direct":
            return f"neural_policy_direct_{policy_kind}"
        return base
    if int(action_slot) <= 0:
        return f"neural_policy_stride_refill_{policy_kind}"
    return f"neural_policy_stride_cached_{policy_kind}"


def _decision_event_id(runner: Any, agent_id: int, *, action_slot: int = 0, source_class: str = "") -> str:
    run_id = str(
        getattr(runner, "run_id", "")
        or getattr(runner, "_run_id", "")
        or getattr(getattr(runner, "sim_cfg", None), "run_id", "")
        or ""
    )
    raw = f"{run_id}|{_runner_generation(runner)}|{_runner_tick(runner)}|{int(agent_id)}|{int(action_slot)}|{source_class}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _action_application_id(runner: Any, agent_id: int, action_id: int, source_class: str) -> str:
    run_id = str(
        getattr(runner, "run_id", "")
        or getattr(runner, "_run_id", "")
        or getattr(getattr(runner, "sim_cfg", None), "run_id", "")
        or ""
    )
    raw = f"{run_id}|{_runner_generation(runner)}|{_runner_tick(runner)}|{int(agent_id)}|{int(action_id)}|{source_class}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _execution_event_id(runner: Any, action_application_id: str) -> str:
    run_id = str(
        getattr(runner, "run_id", "")
        or getattr(runner, "_run_id", "")
        or getattr(getattr(runner, "sim_cfg", None), "run_id", "")
        or ""
    )
    raw = f"{run_id}|execution|{action_application_id}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _row_digest(row: Any) -> str:
    arr = np.asarray(row, dtype=np.float64).reshape((-1,))
    payload = ",".join("nan" if not np.isfinite(v) else f"{float(v):.17g}" for v in arr.tolist()).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _masked_logits(row: Any, valid_actions: list[int]) -> np.ndarray:
    arr = np.asarray(row, dtype=np.float64).reshape((-1,))
    if arr.size != int(N_ACTIONS):
        return arr
    masked = np.full((int(N_ACTIONS),), -np.inf, dtype=np.float64)
    for action in valid_actions:
        if 0 <= int(action) < int(N_ACTIONS):
            masked[int(action)] = arr[int(action)]
    return masked


def _policy_event_sample_limit(runner: Any) -> int:
    try:
        limit = int(getattr(getattr(runner, "sim_cfg", None), "structural_decision_margin_event_sample_limit", 512) or 512)
    except Exception:
        limit = 512
    return max(0, min(10000, int(limit)))


def _merge_policy_action_family_delta(runner: Any, delta: dict[str, Any]) -> None:
    gen = getattr(runner, "_structural_policy_action_family_gen", None)
    if not isinstance(gen, dict):
        gen = empty_policy_action_family_summary(enabled=True)
    total = getattr(runner, "_structural_policy_action_family_total", None)
    if not isinstance(total, dict):
        total = empty_policy_action_family_summary(enabled=True)
    runner._structural_policy_action_family_gen = merge_policy_action_family_summary(gen, delta)
    runner._structural_policy_action_family_total = merge_policy_action_family_summary(total, delta)


def _merge_policy_family_head_shadow_delta(runner: Any, delta: dict[str, Any]) -> None:
    gen = getattr(runner, "_policy_family_head_shadow_gen", None)
    if not isinstance(gen, dict):
        gen = empty_policy_family_head_shadow_summary(enabled=True)
    total = getattr(runner, "_policy_family_head_shadow_total", None)
    if not isinstance(total, dict):
        total = empty_policy_family_head_shadow_summary(enabled=True)
    runner._policy_family_head_shadow_gen = merge_policy_family_head_shadow_summary(gen, delta)
    runner._policy_family_head_shadow_total = merge_policy_family_head_shadow_summary(total, delta)


def _policy_family_not_evaluable(reason: str, count: int) -> dict[str, Any]:
    delta = empty_policy_action_family_summary(enabled=True)
    delta["decision_count"] = 0
    if count:
        _dict_increment(delta, "decision_not_evaluable_reasons", reason, int(count))
    return delta


def _policy_family_execution_not_evaluable(reason: str, count: int) -> dict[str, Any]:
    delta = empty_policy_action_family_summary(enabled=True)
    delta["execution_missing_decision_count"] = int(max(0, count))
    if count:
        _dict_increment(delta, "execution_not_evaluable_reasons", reason, int(count))
    return delta


def _action_consequence(
    *,
    runner: Any,
    agent_id: int,
    action_application_id: str,
    decision_event_id: str,
    execution_event_id: str,
    source_class: str,
    executed_action: int,
    availability: dict[str, Any],
    pre_y: int,
    pre_x: int,
    post_y: int,
    post_x: int,
    pre_energy: float,
    post_energy: float,
    pre_health: float,
    post_health: float,
    pre_injury: float,
    post_injury: float,
    food_gained: float,
) -> dict[str, Any]:
    energy_delta = float(post_energy - pre_energy)
    health_delta = float(post_health - pre_health)
    injury_delta = float(post_injury - pre_injury)
    moved = bool(pre_y != post_y or pre_x != post_x)
    if int(executed_action) in {int(ACT_MOVE_N), int(ACT_MOVE_S), int(ACT_MOVE_E), int(ACT_MOVE_W)}:
        return {
            "execution_accepted": bool(moved),
            "consequence_success": bool(moved),
            "consequence_status": "observed",
            "consequence_reason": "",
            "energy_delta": energy_delta,
            "health_delta": health_delta,
            "injury_delta": injury_delta,
        }
    if int(executed_action) == int(ACT_GATHER):
        success = bool(float(food_gained) > 0.0)
        return {
            "execution_accepted": bool(success),
            "consequence_success": bool(success),
            "consequence_status": "observed",
            "consequence_reason": "" if success else "no_food_gained",
            "energy_delta": energy_delta,
            "health_delta": health_delta,
            "injury_delta": injury_delta,
        }
    if int(executed_action) == int(ACT_REST):
        changed = bool(abs(energy_delta) > 1e-12 or abs(health_delta) > 1e-12 or abs(injury_delta) > 1e-12)
        return {
            "execution_accepted": True,
            "consequence_success": bool(changed),
            "consequence_status": "observed",
            "consequence_reason": "" if changed else "no_scalar_consequence_observed",
            "energy_delta": energy_delta,
            "health_delta": health_delta,
            "injury_delta": injury_delta,
        }
    if int(executed_action) == int(ACT_ATTACK):
        attack = _attack_consequence_for_agent(
            runner,
            agent_id=int(agent_id),
            action_application_id=action_application_id,
            decision_event_id=decision_event_id,
            execution_event_id=execution_event_id,
            source_class=source_class,
            energy_delta=energy_delta,
            health_delta=health_delta,
            injury_delta=injury_delta,
        )
        if attack is not None:
            return attack
        return {
            "execution_accepted": bool(availability.get("attack_executable", False)),
            "consequence_success": None,
            "consequence_status": "not_evaluable",
            "consequence_reason": "per_attacker_combat_effect_not_reported",
            "energy_delta": energy_delta,
            "health_delta": health_delta,
            "injury_delta": injury_delta,
        }
    return {
        "execution_accepted": False,
        "consequence_success": None,
        "consequence_status": "not_evaluable",
        "consequence_reason": "unknown_action",
        "energy_delta": energy_delta,
        "health_delta": health_delta,
        "injury_delta": injury_delta,
    }


def _attack_consequence_for_agent(
    runner: Any,
    *,
    agent_id: int,
    action_application_id: str,
    decision_event_id: str,
    execution_event_id: str,
    source_class: str,
    energy_delta: float,
    health_delta: float,
    injury_delta: float,
) -> dict[str, Any] | None:
    env = getattr(runner, "env", None)
    rows = getattr(env, "last_attack_consequence_events", None)
    if not isinstance(rows, list):
        return None
    match = None
    for row in rows:
        if isinstance(row, dict) and int(row.get("attacker_id", -1)) == int(agent_id):
            match = dict(row)
            break
    if match is None:
        return None
    match.update(
        {
            "action_application_id": action_application_id,
            "decision_event_id": decision_event_id,
            "execution_event_id": execution_event_id,
            "action_source_class": source_class,
        }
    )
    tape = getattr(runner, "_structural_attack_consequence_events", None)
    if not isinstance(tape, list):
        tape = []
    if len(tape) < _policy_event_sample_limit(runner):
        tape.append(match)
    runner._structural_attack_consequence_events = tape
    accepted = bool(match.get("attack_accepted", False))
    reason = str(match.get("rejection_reason", "") or "")
    return {
        "execution_accepted": accepted,
        "consequence_success": accepted and float(match.get("applied_damage", 0.0) or 0.0) > 0.0,
        "consequence_status": "observed" if bool(match.get("consequence_evaluable", False)) else "not_evaluable",
        "consequence_reason": "" if bool(match.get("consequence_evaluable", False)) else str(match.get("not_evaluable_reason", reason or "attack_not_evaluable")),
        "energy_delta": energy_delta,
        "health_delta": health_delta,
        "injury_delta": injury_delta,
        "attack_target_id": int(match.get("target_id", -1)),
        "attack_attempted_damage": float(match.get("attempted_damage", 0.0) or 0.0),
        "attack_applied_damage": float(match.get("applied_damage", 0.0) or 0.0),
        "attack_target_killed_by_event": bool(match.get("target_killed_by_event", False)),
        "attack_rejection_reason": reason,
    }


def _counterfactual_not_evaluable(reason: str, count: int) -> dict[str, Any]:
    delta = empty_opportunity_summary(enabled=True)
    delta["actual_counterfactual_not_evaluable_count"] = int(max(0, count))
    if count:
        _dict_increment(delta, "actual_counterfactual_not_evaluable_reasons", reason, int(count))
    return delta


def _merge_counterfactual_delta(runner: Any, delta: dict[str, Any]) -> None:
    gen_summary = getattr(runner, "_structural_action_opportunity_gen", None)
    if not isinstance(gen_summary, dict):
        gen_summary = {}
    merge_opportunity_summary(gen_summary, delta)
    runner._structural_action_opportunity_gen = gen_summary
    total_summary = getattr(runner, "_structural_action_opportunity_total", None)
    if not isinstance(total_summary, dict):
        total_summary = {}
    merge_opportunity_summary(total_summary, delta)
    runner._structural_action_opportunity_total = total_summary


def record_action_opportunities(
    runner: Any,
    *,
    alive: list[Any],
    agent_ids_np: np.ndarray,
    actions_np: np.ndarray,
    energies: np.ndarray,
    healths: np.ndarray,
    injuries: np.ndarray,
) -> None:
    if not bool(getattr(getattr(runner, "sim_cfg", None), "structural_action_opportunity_reporting_enable", False)):
        return
    delta = _compute_action_opportunities(
        runner,
        alive=alive,
        agent_ids_np=agent_ids_np,
        actions_np=actions_np,
        energies=energies,
        healths=healths,
        injuries=injuries,
    )
    gen_summary = getattr(runner, "_structural_action_opportunity_gen", None)
    if not isinstance(gen_summary, dict):
        gen_summary = {}
    merge_opportunity_summary(gen_summary, delta)
    runner._structural_action_opportunity_gen = gen_summary
    total_summary = getattr(runner, "_structural_action_opportunity_total", None)
    if not isinstance(total_summary, dict):
        total_summary = {}
    merge_opportunity_summary(total_summary, delta)
    runner._structural_action_opportunity_total = total_summary


def _compute_action_opportunities(
    runner: Any,
    *,
    alive: list[Any],
    agent_ids_np: np.ndarray,
    actions_np: np.ndarray,
    energies: np.ndarray,
    healths: np.ndarray,
    injuries: np.ndarray,
) -> dict[str, Any]:
    env = runner.env
    grid = np.asarray(getattr(env, "grid"), dtype=np.int8)
    agent_grid = np.asarray(getattr(env, "agent_grid"), dtype=np.int32)
    pos_y = np.asarray(getattr(env, "agent_pos_y"), dtype=np.int32)
    pos_x = np.asarray(getattr(env, "agent_pos_x"), dtype=np.int32)
    h, w = agent_grid.shape
    ids = np.asarray(agent_ids_np, dtype=np.int32)
    actions = np.asarray(actions_np, dtype=np.int32)
    id_to_idx = {int(aid): idx for idx, aid in enumerate(ids.tolist())}
    out = empty_opportunity_summary(enabled=True)
    unavailable = Counter()
    not_eval = Counter()
    stages = Counter()
    threshold = _patch_threshold(env)
    patch_amount = getattr(env, "patch_food_amount", None)
    patch_amount_arr = None if patch_amount is None else np.asarray(patch_amount, dtype=np.float32)
    margin_reporting = bool(getattr(getattr(runner, "sim_cfg", None), "structural_action_margin_reporting_enable", False))
    for i, aid_raw in enumerate(ids.tolist()):
        aid = int(aid_raw)
        if aid < 0 or aid >= pos_y.shape[0]:
            not_eval["missing_agent_position"] += 1
            continue
        y = int(pos_y[aid])
        x = int(pos_x[aid])
        if y < 0 or y >= h or x < 0 or x >= w:
            not_eval["agent_out_of_bounds"] += 1
            continue
        action = int(actions[i])
        out["evaluated_decisions"] += 1
        out["attack_action_unmasked_count"] += 1
        out["gather_action_unmasked_count"] += 1
        out["rest_action_unmasked_count"] += 1
        legal_moves = _legal_move_count(agent_grid, y, x)
        out["legal_move_count_total"] += legal_moves
        if legal_moves >= 2:
            out["decisions_with_multiple_moves"] += 1
        attack_target = _attack_target_available(agent_grid, id_to_idx, healths, y, x, h, w, i)
        if attack_target:
            out["attack_target_available_count"] += 1
        attack_executable = bool(attack_target and float(energies[i]) > 0.0 and float(getattr(runner.env_cfg, "attack_damage", 0.0)) > 0.0)
        if attack_executable:
            out["attack_executable_count"] += 1
        else:
            unavailable["attack_no_target_or_energy"] += 1
        gather_available = bool(grid[y, x] == FOOD)
        if patch_amount_arr is not None:
            gather_available = bool(float(patch_amount_arr[y, x]) > threshold)
        if gather_available:
            out["gather_resource_available_count"] += 1
            out["gather_executable_count"] += 1
        else:
            unavailable["gather_no_resource_at_cell"] += 1
        rest_need = _rest_need_present(runner, alive[i] if i < len(alive) else None, healths[i], injuries[i])
        if rest_need:
            out["rest_need_present_count"] += 1
        competing = sum(1 for flag in (attack_executable, gather_available, rest_need, legal_moves > 0) if flag)
        if rest_need and competing >= 2:
            out["rest_competitive_count"] += 1
        if legal_moves > 0 and competing >= 2:
            out["movement_competitive_count"] += 1
        if competing >= 2:
            out["drive_competition_count"] += 1
        if action == int(ACT_ATTACK):
            out["attack_chosen_count"] += 1
            if attack_executable:
                out["attack_executed_count"] += 1
        elif action == int(ACT_GATHER):
            out["gather_chosen_count"] += 1
            if gather_available:
                out["gather_executed_count"] += 1
        elif action == int(ACT_REST):
            out["rest_chosen_count"] += 1
            out["rest_executed_count"] += 1
        elif 0 <= action <= 3:
            out["move_chosen_count"] += 1
            if _chosen_move_legal(agent_grid, y, x, h, w, action):
                out["move_executed_count"] += 1
        field_active, linked, stage = _field_exposure(runner, alive[i] if i < len(alive) else None, attack_executable, gather_available, rest_need, legal_moves)
        margin = _margin_for_agent(runner, aid) if margin_reporting else None
        if margin_reporting and margin is None:
            out["margin_not_evaluable_count"] += 1
            not_eval["policy_margin_missing"] += 1
        elif margin is not None:
            m = float(margin["top_two_margin"])
            out["margin_evaluable_count"] += 1
            out["_margin_sum"] = float(out.get("_margin_sum", 0.0) or 0.0) + m
            out["margin_min"] = m if out["margin_min"] is None else min(float(out["margin_min"]), m)
            out["margin_max"] = m if out["margin_max"] is None else max(float(out["margin_max"]), m)
            _bump_margin_histogram(out, m)
        potential = _potential_field_opportunities(
            attack=attack_executable,
            gather=gather_available,
            rest=rest_need,
            move=legal_moves > 0,
        )
        if potential:
            out["potential_linked_opportunity_count"] += 1
            out["potential_linked_field_count"] += len(potential)
            families = sorted({family for vals in potential.values() for family in vals})
            out["potential_linked_action_family_count"] += len(families)
            for field, families_for_field in potential.items():
                _dict_increment(out, "potential_linked_opportunities_by_field", field, 1)
                for family in families_for_field:
                    _dict_increment(out, "potential_linked_opportunities_by_action_family", family, 1)
            canonical_fields = [field for field in potential if field in CANONICAL_COUNTERFACTUAL_FIELD_DELTAS and CANONICAL_COUNTERFACTUAL_FIELD_DELTAS[field] != 0.0]
            if canonical_fields:
                out["canonical_counterfactual_linked_opportunity_count"] += 1
                for field in canonical_fields:
                    _dict_increment(out, "canonical_counterfactual_changed_fields", field, 1)
                if margin is not None:
                    effect = max(abs(float(CANONICAL_COUNTERFACTUAL_FIELD_DELTAS[field])) for field in canonical_fields)
                    if float(margin["top_two_margin"]) <= NEAR_BOUNDARY_ABSOLUTE_THRESHOLD:
                        out["near_boundary_absolute_count"] += 1
                    if float(margin["top_two_margin"]) <= effect:
                        out["near_boundary_canonical_relative_count"] += 1
                        out["canonical_delta_exceeds_margin_count"] += 1
        if stage:
            stages[stage] += 1
        if field_active:
            out["linked_field_active_count"] += 1
        if linked:
            out["linked_field_opportunity_count"] += 1
            if margin is not None and float(margin["top_two_margin"]) <= max(abs(v) for v in CANONICAL_COUNTERFACTUAL_FIELD_DELTAS.values()):
                out["actual_live_delta_exceeds_margin_count"] += 1
    if not margin_reporting:
        not_eval["policy_margin_not_captured"] += int(out["evaluated_decisions"])
    if int(out["margin_evaluable_count"] or 0) > 0:
        out["margin_mean"] = float(out.get("_margin_sum", 0.0) or 0.0) / float(out["margin_evaluable_count"])
    else:
        out["margin_mean"] = None
        out["margin_min"] = None
        out["margin_max"] = None
    out["developmental_stage_counts"] = dict(sorted(stages.items()))
    out["unavailable_reason_counts"] = dict(sorted(unavailable.items()))
    out["not_evaluable_reason_counts"] = dict(sorted(not_eval.items()))
    return out


def _margin_for_agent(runner: Any, agent_id: int) -> dict[str, Any] | None:
    if not bool(getattr(getattr(runner, "sim_cfg", None), "structural_action_margin_reporting_enable", False)):
        return None
    pending = getattr(runner, "_structural_action_margin_pending", None)
    if not isinstance(pending, dict):
        return None
    payload = pending.get(int(agent_id))
    return payload if isinstance(payload, dict) else None


def _dict_increment(out: dict[str, Any], field: str, key: str, amount: int) -> None:
    target = out.get(field)
    if not isinstance(target, dict):
        target = {}
        out[field] = target
    target[str(key)] = int(target.get(str(key), 0) or 0) + int(amount)


def _bump_margin_histogram(out: dict[str, Any], margin: float) -> None:
    bins = ((0.0, 0.05, "le_0.05"), (0.05, 0.10, "le_0.10"), (0.10, 0.25, "le_0.25"), (0.25, 0.50, "le_0.50"))
    label = "gt_0.50"
    for lo, hi, name in bins:
        if margin <= hi:
            label = name
            break
    _dict_increment(out, "margin_histogram", label, 1)


def _potential_field_opportunities(*, attack: bool, gather: bool, rest: bool, move: bool) -> dict[str, tuple[str, ...]]:
    available = {
        "attack": bool(attack),
        "gather": bool(gather),
        "rest": bool(rest),
        "move": bool(move),
    }
    out: dict[str, tuple[str, ...]] = {}
    for field, families in FIELD_ACTION_ROUTES.items():
        linked = tuple(family for family in families if available.get(family, False))
        if linked:
            out[field] = linked
    return out


def _legal_move_count(agent_grid: np.ndarray, y: int, x: int) -> int:
    h, w = agent_grid.shape
    count = 0
    for ny, nx in ((y - 1, x), (y + 1, x), (y, x + 1), (y, x - 1)):
        if 0 <= ny < h and 0 <= nx < w and int(agent_grid[ny, nx]) == -1:
            count += 1
    return count


def _chosen_move_legal(agent_grid: np.ndarray, y: int, x: int, h: int, w: int, action: int) -> bool:
    ny, nx = y, x
    if action == 0:
        ny -= 1
    elif action == 1:
        ny += 1
    elif action == 2:
        nx += 1
    else:
        nx -= 1
    return bool(0 <= ny < h and 0 <= nx < w and int(agent_grid[ny, nx]) == -1)


def _attack_target_available(agent_grid: np.ndarray, id_to_idx: dict[int, int], healths: np.ndarray, y: int, x: int, h: int, w: int, self_idx: int) -> bool:
    for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
        if 0 <= ny < h and 0 <= nx < w:
            tid = int(agent_grid[ny, nx])
            target_idx = id_to_idx.get(tid, -1)
            if target_idx != -1 and target_idx != self_idx and float(healths[target_idx]) > 0.0:
                return True
    return False


def _patch_threshold(env: Any) -> float:
    try:
        from env.gridworld_pkg.patch_world import patch_food_presence_threshold

        return float(patch_food_presence_threshold(getattr(env, "cfg", None)))
    except Exception:
        return 0.5


def _rest_need_present(runner: Any, agent: Any, health: float, injury: float) -> bool:
    if float(injury) > 0.0:
        return True
    if float(health) < 0.95 * float(getattr(runner.env_cfg, "start_health", 1.0)):
        return True
    drives = getattr(agent, "drives", None)
    if isinstance(drives, (list, tuple)) and len(drives) >= 3:
        return bool(float(drives[2]) >= 0.20 or float(drives[1]) >= 0.20)
    return False


def _field_exposure(runner: Any, agent: Any, attack: bool, gather: bool, rest: bool, legal_moves: int) -> tuple[bool, bool, str]:
    if agent is None or not bool(getattr(getattr(runner, "sim_cfg", None), "structural_live_semantics_enable", False)):
        return False, False, ""
    try:
        expr = runner.resolve_developmental_expression_for_agent(agent)
    except Exception:
        return False, False, ""
    summary = expr.get("structural_live_semantics", {}) if isinstance(expr, dict) else {}
    changed = set(summary.get("changed_effective_fields", []) or [])
    stage = str(expr.get("stage_id", "") or "")
    field_active = bool(changed.intersection({"aggression_bias", "action_temperature", "drive_logit_gain", "drive_inertia"}))
    linked = bool(
        ("aggression_bias" in changed and attack)
        or ("drive_logit_gain" in changed and (gather or rest or legal_moves > 0))
        or ("drive_inertia" in changed and (gather or rest or legal_moves > 0))
        or ("action_temperature" in changed and (attack or gather or rest or legal_moves > 0))
    )
    return field_active, linked, stage
