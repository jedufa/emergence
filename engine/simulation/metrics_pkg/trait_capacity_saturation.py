from __future__ import annotations

"""REP-1A trait capacity saturation reporting helpers.

This module is reporting-only. It consumes already-materialized metric records
and returns deterministic JSON payloads. It must not mutate agents, simulation
configuration, or random state.
"""

import json
import math
from typing import Any, Iterable, Mapping

from evolution.capacity_contract import build_capacity_contract, capacity_headroom_summary


SCHEMA_VERSION = "trait_capacity_saturation_v1"
TRAIT_NAMES: tuple[str, ...] = (
    "memory_dim",
    "mutation_rate",
    "pred_lr",
    "sensor_radius",
    "sensor_channel_active_count",
    "module_count",
    "connection_count",
    "action_temperature",
    "drive_logit_gain",
    "reproduction_threshold",
)
PERSISTENCE_THRESHOLD_GENERATIONS = 3
SELECTED_MIN_COHORT = 2
SELECTED_MIN_AT_CAP_DELTA = 0.20
FLOAT_TOUCH_TOLERANCE = 1e-9
NEAR_CAP_FRACTION = 0.05


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def empty_trait_capacity_payload(reason: str = "reporting_disabled") -> tuple[str, str, str]:
    summary = {
        "schema_version": SCHEMA_VERSION,
        "reporting_enabled": False,
        "supported": False,
        "reason": str(reason),
        "traits": {},
    }
    lineage = {
        "schema_version": SCHEMA_VERSION,
        "reporting_enabled": False,
        "traits": {},
    }
    structural = {
        "schema_version": SCHEMA_VERSION,
        "reporting_enabled": False,
        "structural_status": "not_evaluable",
        "reasons": [str(reason)],
    }
    return stable_json(summary), stable_json(lineage), stable_json(structural)


def trait_capacity_saturation_payloads(
    alive_records: Iterable[Mapping[str, Any]],
    selected_records: Iterable[Mapping[str, Any]] | None = None,
    *,
    generation: int | None = None,
    env_cfg: Any = None,
    sim_cfg: Any = None,
    lineage_streaks: Mapping[str, Mapping[Any, int]] | None = None,
    structural_context: Mapping[str, Any] | None = None,
) -> tuple[str, str, str]:
    alive = [dict(r) for r in alive_records or []]
    selected = [dict(r) for r in selected_records or []]
    reproduced = [r for r in alive if float(r.get("offspring", 0.0) or 0.0) > 0.0]
    contract = build_capacity_contract(sim_cfg=sim_cfg, env_cfg=env_cfg)
    structural = _structural_payload(
        alive,
        selected,
        reproduced,
        sim_cfg=sim_cfg,
        structural_context=structural_context,
        contract=contract,
    )
    headroom = capacity_headroom_summary(
        alive,
        contract=contract,
        structural_context={
            **dict(structural_context or {}),
            "module_cap_drop_count": structural.get("module_cap_drop_count", 0),
            "connection_cap_drop_count": structural.get("connection_cap_drop_count", 0),
            "locality_drop_count": structural.get("locality_drop_count", 0),
        },
    )
    traits: dict[str, dict[str, Any]] = {}
    lineage_payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "reporting_enabled": True,
        "generation": _nullable_int(generation),
        "policy": _policy_payload(),
        "traits": {},
    }
    for name in TRAIT_NAMES:
        spec = _trait_spec(name, env_cfg=env_cfg, sim_cfg=sim_cfg, contract=contract)
        trait = _trait_payload(
            name,
            spec,
            alive,
            selected,
            reproduced,
            lineage_streaks=dict((lineage_streaks or {}).get(name, {}) or {}),
            structural=structural,
        )
        traits[name] = trait
        lineage_payload["traits"][name] = _lineage_entries(name, trait)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "reporting_enabled": True,
        "supported": True,
        "generation": _nullable_int(generation),
        "policy": _policy_payload(),
        "capacity_contract_ref": {
            "schema_version": contract["schema_version"],
            "capacity_epoch_id": contract["capacity_epoch_id"],
            "contract_hash": contract["contract_hash"],
        },
        "capacity_headroom": headroom,
        "traits": traits,
    }
    return stable_json(summary), stable_json(lineage_payload), stable_json(structural)


def _policy_payload() -> dict[str, Any]:
    return {
        "float_touch_tolerance": FLOAT_TOUCH_TOLERANCE,
        "near_cap_fraction": NEAR_CAP_FRACTION,
        "persistence_threshold_generations": PERSISTENCE_THRESHOLD_GENERATIONS,
        "selected_min_cohort": SELECTED_MIN_COHORT,
        "selected_min_at_cap_delta": SELECTED_MIN_AT_CAP_DELTA,
    }


def _trait_spec(name: str, *, env_cfg: Any, sim_cfg: Any, contract: Mapping[str, Any]) -> dict[str, Any]:
    view_radius = int(max(1, getattr(env_cfg, "view_radius", 3) if env_cfg is not None else 3))
    implemented_channels = int(contract["observations"]["implemented_active_channel_count"])
    if name == "memory_dim":
        return _spec("bounded_ladder", "effective_memory", float(contract["policy"]["effective_memory_max"]), "capacity_contract.policy.effective_memory_max", active=True, evolvable=True)
    if name == "mutation_rate":
        return _spec("bounded_scalar", "mutation_rate", 0.20, "_MUT_RATE_BOUNDS upper", active=True, evolvable=True)
    if name == "pred_lr":
        return _spec("bounded_scalar", "effective_pred_lr", 1.0, "Genome.pred_lr clamp upper", active=True, evolvable=True)
    if name == "sensor_radius":
        gate = _cfg_bool(sim_cfg, "enable_evolvable_sensor_radius") or any(
            _cfg_bool(sim_cfg, k)
            for k in ("ablate_sensor_radius", "force_min_sensor_radius", "force_max_sensor_radius", "force_default_sensor_radius")
        )
        return _spec("bounded_ladder", "sensor_radius", float(min(view_radius, 3)), "min(env view_radius, SENSOR_RADIUS_LADDER max)", active=gate, evolvable=_cfg_bool(sim_cfg, "enable_evolvable_sensor_radius"))
    if name == "sensor_channel_active_count":
        gate = _cfg_bool(sim_cfg, "enable_evolvable_sensor_channel_mask") or any(
            _cfg_bool(sim_cfg, k)
            for k in ("ablate_sensor_channel_mask", "force_all_sensor_channels_enabled", "force_minimal_sensor_channels_enabled")
        )
        return _spec("preallocated_mask_gated", "sensor_channel_count", float(implemented_channels), "capacity_contract.observations.implemented_active_channel_count", active=gate, evolvable=_cfg_bool(sim_cfg, "enable_evolvable_sensor_channel_mask"))
    if name == "module_count":
        return _spec("structural_payload_empty_or_bounded_count", "module_count", float(contract["structure"]["limits"]["max_modules_per_agent"]), "capacity_contract.structure.limits.max_modules_per_agent", active=True, evolvable=_cfg_bool(sim_cfg, "genome_c1_enable", True) and _cfg_bool(sim_cfg, "genome_c2_enable", True), structural=True)
    if name == "connection_count":
        return _spec("structural_payload_empty_or_bounded_count", "connection_count", float(contract["structure"]["limits"]["max_connections_per_agent"]), "capacity_contract.structure.limits.max_connections_per_agent", active=True, evolvable=_cfg_bool(sim_cfg, "genome_c1_enable", True) and _cfg_bool(sim_cfg, "genome_c2_enable", True), structural=True)
    if name == "action_temperature":
        return _spec("bounded_scalar", "action_temperature", 2.5, "_MUT_SPECS action_temperature upper", active=True, evolvable=True)
    if name == "drive_logit_gain":
        return _spec("bounded_scalar", "drive_gain", 3.0, "Genome.drive_logit_gain clamp upper", active=True, evolvable=True)
    if name == "reproduction_threshold":
        return _spec("bounded_scalar", "reproduction_threshold", 90.0, "_MUT_SPECS reproduction_threshold upper", active=True, evolvable=True)
    return _spec("unknown", name, None, "unknown", active=False, evolvable=False)


def _spec(kind: str, key: str, cap: float | None, cap_source: str, *, active: bool, evolvable: bool, structural: bool = False) -> dict[str, Any]:
    return {
        "representation_kind": kind,
        "value_key": key,
        "cap_value": cap,
        "cap_source": cap_source,
        "active": bool(active),
        "evolvable": bool(evolvable),
        "structural": bool(structural),
    }


def _trait_payload(
    name: str,
    spec: Mapping[str, Any],
    alive: list[Mapping[str, Any]],
    selected: list[Mapping[str, Any]],
    reproduced: list[Mapping[str, Any]],
    *,
    lineage_streaks: Mapping[Any, int],
    structural: Mapping[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    cap = spec.get("cap_value")
    active = bool(spec.get("active"))
    evaluable = active and cap is not None and bool(spec.get("value_key"))
    key = str(spec.get("value_key"))
    structural_empty = bool(spec.get("structural")) and float(structural.get("alive_empty_payload_fraction") or 0.0) >= 1.0
    if not active:
        evaluable = False
        reasons.append("trait_surface_not_active")
    if structural_empty:
        reasons.append("all_alive_structural_payloads_empty")
    if cap is None:
        evaluable = False
        reasons.append("cap_unavailable")

    alive_stats = _cohort_stats(alive, key, cap, evaluable=evaluable)
    selected_stats = _cohort_stats(selected, key, cap, evaluable=evaluable)
    reproduced_stats = _cohort_stats(reproduced, key, cap, evaluable=evaluable)
    persistent_lineages = _persistent_lineages(lineage_streaks)

    alive_at_cap = float(alive_stats["at_cap_fraction"] or 0.0)
    reproduced_at_cap = float(reproduced_stats["at_cap_fraction"] or 0.0)
    cap_touch = bool(evaluable and not structural_empty and (alive_at_cap > 0.0 or reproduced_at_cap > 0.0))
    cap_persistent = bool(cap_touch and persistent_lineages)
    selected_delta = _delta(selected_stats["at_cap_fraction"], alive_stats["at_cap_fraction"])
    reproduced_delta = _delta(reproduced_stats["at_cap_fraction"], alive_stats["at_cap_fraction"])
    selected_mean_delta = _delta(selected_stats["mean"], alive_stats["mean"])
    reproduced_mean_delta = _delta(reproduced_stats["mean"], alive_stats["mean"])
    selected_ok = selected_stats["count"] >= SELECTED_MIN_COHORT and selected_delta is not None and selected_delta >= SELECTED_MIN_AT_CAP_DELTA
    reproduced_ok = reproduced_stats["count"] >= SELECTED_MIN_COHORT and reproduced_delta is not None and reproduced_delta >= SELECTED_MIN_AT_CAP_DELTA
    if selected_stats["count"] and selected_stats["count"] < SELECTED_MIN_COHORT:
        reasons.append("selected_cohort_below_minimum")
    if reproduced_stats["count"] and reproduced_stats["count"] < SELECTED_MIN_COHORT:
        reasons.append("reproduced_cohort_below_minimum")
    cap_selected = bool(cap_touch and (selected_ok or reproduced_ok))
    blocked = _blocked_attempt_count(name, structural)
    binding_evaluable = bool(blocked is not None)
    if not binding_evaluable:
        reasons.append("blocked_upward_attempt_evidence_unavailable")
    cap_binding = bool(cap_touch and (cap_persistent or cap_selected) and binding_evaluable and float(blocked or 0.0) > 0.0)
    status = _status(
        active=active,
        evaluable=evaluable,
        structural_empty=structural_empty,
        cap_touch=cap_touch,
        cap_persistent=cap_persistent,
        cap_selected=cap_selected,
        cap_binding=cap_binding,
    )

    entries = []
    for lineage_id, streak in sorted(lineage_streaks.items(), key=lambda kv: str(kv[0])):
        try:
            streak_i = int(streak)
        except Exception:
            streak_i = 0
        if streak_i <= 0:
            continue
        entries.append(
            {
                "lineage_id": str(lineage_id),
                "first_cap_generation": None,
                "last_cap_generation": None,
                "longest_consecutive_cap_streak": streak_i,
                "cap_generation_count": streak_i,
                "selected_at_cap_count": None,
                "reproduced_at_cap_count": None,
                "final_alive_at_cap_count": None,
                "status": "cap_persistent" if streak_i >= PERSISTENCE_THRESHOLD_GENERATIONS else "cap_touch",
            }
        )

    return {
        "trait_name": name,
        "representation_kind": str(spec.get("representation_kind")),
        "active": bool(active),
        "evolvable": bool(spec.get("evolvable")),
        "evaluable": bool(evaluable and not structural_empty),
        "cap_value": _nullable_float(cap),
        "cap_source": str(spec.get("cap_source")),
        "comparison_mode": "exact_or_float_tolerance",
        "alive_count": alive_stats["count"],
        "alive_mean": alive_stats["mean"],
        "alive_max": alive_stats["max"],
        "alive_at_cap_fraction": alive_stats["at_cap_fraction"],
        "alive_near_cap_fraction": alive_stats["near_cap_fraction"],
        "selected_count": selected_stats["count"],
        "selected_mean": selected_stats["mean"],
        "selected_max": selected_stats["max"],
        "selected_at_cap_fraction": selected_stats["at_cap_fraction"],
        "selected_near_cap_fraction": selected_stats["near_cap_fraction"],
        "reproduced_count": reproduced_stats["count"],
        "reproduced_mean": reproduced_stats["mean"],
        "reproduced_max": reproduced_stats["max"],
        "reproduced_at_cap_fraction": reproduced_stats["at_cap_fraction"],
        "reproduced_near_cap_fraction": reproduced_stats["near_cap_fraction"],
        "selected_minus_alive_mean": selected_mean_delta,
        "selected_minus_alive_at_cap_fraction": selected_delta,
        "reproduced_minus_alive_mean": reproduced_mean_delta,
        "reproduced_minus_alive_at_cap_fraction": reproduced_delta,
        "persistent_lineage_count": len(persistent_lineages),
        "longest_lineage_cap_streak": max((int(v) for v in lineage_streaks.values()), default=0),
        "blocked_upward_attempt_count": _nullable_float(blocked),
        "cap_touch": cap_touch,
        "cap_persistent": cap_persistent,
        "cap_selected": cap_selected,
        "cap_binding": cap_binding,
        "binding_evaluable": binding_evaluable,
        "status": status,
        "reasons": sorted(set(reasons)),
        "lineages": entries,
    }


def _cohort_stats(records: list[Mapping[str, Any]], key: str, cap: Any, *, evaluable: bool) -> dict[str, Any]:
    if not evaluable:
        return {"count": len(records), "mean": None, "max": None, "at_cap_fraction": None, "near_cap_fraction": None}
    vals = [_as_float(r.get(key)) for r in records if _as_float(r.get(key)) is not None]
    if not vals:
        return {"count": 0, "mean": None, "max": None, "at_cap_fraction": None, "near_cap_fraction": None}
    cap_f = float(cap)
    at = sum(1 for v in vals if _at_cap(v, cap_f))
    near = sum(1 for v in vals if _near_cap(v, cap_f))
    n = float(len(vals))
    return {
        "count": len(vals),
        "mean": float(sum(vals) / n),
        "max": float(max(vals)),
        "at_cap_fraction": float(at / n),
        "near_cap_fraction": float(near / n),
    }


def _structural_payload(
    alive: list[Mapping[str, Any]],
    selected: list[Mapping[str, Any]],
    reproduced: list[Mapping[str, Any]],
    *,
    sim_cfg: Any,
    structural_context: Mapping[str, Any] | None,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    ctx = dict(structural_context or {})
    module_cap = float(contract["structure"]["limits"]["max_modules_per_agent"])
    connection_cap = float(contract["structure"]["limits"]["max_connections_per_agent"])

    def empty_fraction(records: list[Mapping[str, Any]]) -> float | None:
        if not records:
            return None
        return float(sum(1 for r in records if int(r.get("module_count", 0) or 0) <= 0 and int(r.get("connection_count", 0) or 0) <= 0) / float(len(records)))

    def max_value(records: list[Mapping[str, Any]], key: str) -> float | None:
        vals = [_as_float(r.get(key)) for r in records if _as_float(r.get(key)) is not None]
        return float(max(vals)) if vals else None

    def sum_value(records: list[Mapping[str, Any]], key: str) -> float | None:
        vals = [_as_float(r.get(key)) for r in records if _as_float(r.get(key)) is not None]
        return float(sum(vals)) if vals else None

    op_keys = (
        "c1_op_count__bounded_rewire",
        "c1_op_count__duplicate_with_decay",
        "c1_op_count__edge_split_insertion",
        "c1_op_count__fusion_merge_compression",
        "c1_op_count__gate_drift",
        "c1_op_count__motif_duplication",
        "c1_op_count__promoter_expression_mutation",
    )
    observed_attempts = float(sum(sum_value(alive, key) or 0.0 for key in op_keys)) if alive else None
    observed_noops = sum_value(alive, "c1_operator_noop_count")
    observed_successes = None
    if observed_attempts is not None and observed_noops is not None:
        observed_successes = float(max(0.0, float(observed_attempts) - float(observed_noops)))
    observed_locality_drops = sum_value(alive, "b3_locality_dropped_connections")
    observed_capacity_blocks = sum_value(alive, "c1_capacity_blocked_count")
    blocked_reason_counts: dict[str, int] = {}
    for record in alive:
        try:
            reason_payload = json.loads(str(record.get("c1_capacity_blocked_reason_counts_json", "{}") or "{}"))
        except (TypeError, ValueError, json.JSONDecodeError):
            reason_payload = {}
        if isinstance(reason_payload, dict):
            for reason, count in reason_payload.items():
                blocked_reason_counts[str(reason)] = int(blocked_reason_counts.get(str(reason), 0)) + int(count or 0)

    alive_empty = empty_fraction(alive)
    status = "not_evaluable"
    reasons: list[str] = []
    if alive:
        if alive_empty == 1.0:
            status = "structural_payload_empty"
            reasons.append("all_alive_structural_payloads_empty")
        else:
            status = "below_cap"
    else:
        reasons.append("no_alive_records")
    module_cap_touch = _fraction_at_cap(alive, "module_count", module_cap)
    connection_cap_touch = _fraction_at_cap(alive, "connection_count", connection_cap)
    if (module_cap_touch or 0.0) > 0.0 or (connection_cap_touch or 0.0) > 0.0:
        status = "cap_touch"
    return {
        "schema_version": SCHEMA_VERSION,
        "reporting_enabled": True,
        "alive_empty_payload_fraction": alive_empty,
        "selected_empty_payload_fraction": empty_fraction(selected),
        "reproduced_empty_payload_fraction": empty_fraction(reproduced),
        "alive_module_count_max": max_value(alive, "module_count"),
        "selected_module_count_max": max_value(selected, "module_count"),
        "reproduced_module_count_max": max_value(reproduced, "module_count"),
        "alive_connection_count_max": max_value(alive, "connection_count"),
        "selected_connection_count_max": max_value(selected, "connection_count"),
        "reproduced_connection_count_max": max_value(reproduced, "connection_count"),
        "module_cap_touch_fraction": module_cap_touch,
        "connection_cap_touch_fraction": connection_cap_touch,
        "structural_operator_attempt_count": _ctx_float(ctx, "structural_operator_attempt_count", observed_attempts),
        "structural_operator_success_count": _ctx_float(ctx, "structural_operator_success_count", observed_successes),
        "structural_operator_noop_count": _ctx_float(ctx, "structural_operator_noop_count", observed_noops),
        "module_cap_drop_count": _ctx_float(ctx, "module_cap_drop_count"),
        "connection_cap_drop_count": _ctx_float(ctx, "connection_cap_drop_count"),
        "locality_drop_count": _ctx_float(ctx, "locality_drop_count", observed_locality_drops),
        "capacity_blocked_mutation_count": _ctx_float(ctx, "capacity_blocked_mutation_count", observed_capacity_blocks),
        "capacity_blocked_reason_counts": dict(sorted(blocked_reason_counts.items())),
        "structural_reachability_classification": str(contract["structure"]["reachability"]["classification"]),
        "representation_limited": bool(
            str(contract["structure"]["reachability"]["classification"]) == "not_bootstrap_reachable"
            or float(observed_capacity_blocks or 0.0) > 0.0
        ),
        "structural_status": status,
        "reasons": sorted(set(reasons)),
        **(
            {
                "escap8_growth_attempt_count": _ctx_float(
                    ctx,
                    "escap8_growth_attempt_count",
                    sum_value(alive, "escap8_growth_attempt_count"),
                ),
                "escap8_blocked_duplication_attempt_count": _ctx_float(
                    ctx,
                    "blocked_duplication_attempts",
                    sum_value(alive, "escap8_blocked_duplication_attempt_count"),
                ),
                "escap8_blocked_prune_attempt_count": _ctx_float(
                    ctx,
                    "blocked_prune_attempts",
                    sum_value(alive, "escap8_blocked_prune_attempt_count"),
                ),
                "escap8_cap_pressure_streak_max": _ctx_float(
                    ctx,
                    "cap_pressure_streak",
                    max_value(alive, "escap8_cap_pressure_streak"),
                ),
                "escap8_cap_contact_count": _ctx_float(
                    ctx,
                    "cap_contact_count",
                    sum_value(alive, "escap8_cap_contact_count"),
                ),
                "executing_module_count_max": _ctx_float(
                    ctx,
                    "executing_module_count_max",
                    max_value(alive, "executing_modules"),
                ),
                "dormant_module_count_max": _ctx_float(
                    ctx,
                    "dormant_module_count_max",
                    max_value(alive, "dormant_modules"),
                ),
                "nonzero_output_module_count_max": _ctx_float(
                    ctx,
                    "nonzero_output_module_count_max",
                    max_value(alive, "nonzero_output_modules"),
                ),
                "causal_activity_reporting": "probe_only_not_classified_live",
                "neutral_duplicate_count_max": _ctx_float(
                    ctx,
                    "neutral_duplicate_count_max",
                    max_value(alive, "neutral_duplicate_modules"),
                ),
                "redundant_module_count_max": _ctx_float(
                    ctx,
                    "redundant_module_count_max",
                    max_value(alive, "redundant_modules"),
                ),
                "prunable_module_count_max": _ctx_float(
                    ctx,
                    "prunable_module_count_max",
                    max_value(alive, "prunable_modules"),
                ),
                "structural_cost_total_max": _ctx_float(
                    ctx,
                    "structural_cost_total_max",
                    max_value(alive, "structural_cost_total"),
                ),
                "specialization_metrics_record_count": int(
                    _ctx_float(
                        ctx,
                        "specialization_metrics_record_count",
                        sum(
                            1
                            for row in alive
                            if str(
                                row.get("escap8_specialization_metrics_json", "{}")
                                or "{}"
                            )
                            != "{}"
                        ),
                    )
                ),
                "reporting_only": True,
                "selection_input": False,
            }
            if bool(
                getattr(sim_cfg, "structural_sustainable_growth_enable", False)
            )
            else {}
        ),
    }


def _lineage_entries(name: str, trait: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries = [dict(x) for x in list(trait.get("lineages", []) or [])]
    entries.sort(key=lambda x: (str(x.get("lineage_id", "")), str(x.get("status", ""))))
    return entries


def _persistent_lineages(streaks: Mapping[Any, int]) -> list[str]:
    out = []
    for lineage_id, streak in streaks.items():
        try:
            if int(streak) >= PERSISTENCE_THRESHOLD_GENERATIONS:
                out.append(str(lineage_id))
        except Exception:
            continue
    return sorted(out)


def _blocked_attempt_count(name: str, structural: Mapping[str, Any]) -> float | None:
    if name == "module_count":
        return _as_float(structural.get("module_cap_drop_count"))
    if name == "connection_count":
        conn = _as_float(structural.get("connection_cap_drop_count"))
        loc = _as_float(structural.get("locality_drop_count"))
        if conn is None and loc is None:
            return None
        return float(conn or 0.0) + float(loc or 0.0)
    return None


def _status(
    *,
    active: bool,
    evaluable: bool,
    structural_empty: bool,
    cap_touch: bool,
    cap_persistent: bool,
    cap_selected: bool,
    cap_binding: bool,
) -> str:
    if not active:
        return "not_active"
    if structural_empty:
        return "structural_payload_empty"
    if not evaluable:
        return "not_evaluable"
    if cap_binding:
        return "cap_binding"
    if cap_selected:
        return "cap_selected"
    if cap_persistent:
        return "cap_persistent"
    if cap_touch:
        return "cap_touch"
    return "below_cap"


def _fraction_at_cap(records: list[Mapping[str, Any]], key: str, cap: float) -> float | None:
    vals = [_as_float(r.get(key)) for r in records if _as_float(r.get(key)) is not None]
    if not vals:
        return None
    return float(sum(1 for v in vals if _at_cap(v, cap)) / float(len(vals)))


def _at_cap(value: float, cap: float) -> bool:
    return bool(abs(float(value) - float(cap)) <= FLOAT_TOUCH_TOLERANCE)


def _near_cap(value: float, cap: float) -> bool:
    band = max(FLOAT_TOUCH_TOLERANCE, abs(float(cap)) * NEAR_CAP_FRACTION)
    return bool(float(value) >= float(cap) - band)


def _delta(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    return float(a) - float(b)


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _nullable_float(value: Any) -> float | None:
    out = _as_float(value)
    return None if out is None else float(out)


def _nullable_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _cfg_bool(obj: Any, name: str, default: bool = False) -> bool:
    try:
        return bool(getattr(obj, name))
    except Exception:
        return bool(default)


def _cfg_float(obj: Any, name: str, default: float) -> float:
    try:
        return float(getattr(obj, name))
    except Exception:
        return float(default)


def _ctx_float(ctx: Mapping[str, Any], name: str, default: Any = None) -> float | None:
    if name not in ctx:
        return _nullable_float(default)
    return _nullable_float(ctx.get(name))
