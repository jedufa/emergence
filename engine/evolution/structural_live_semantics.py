from __future__ import annotations

"""Default-off live structural semantics helpers for REP-2B."""

import hashlib
import json
import math
from typing import Any


SCHEMA_VERSION = "structural_live_semantics_v1"
PROFILE_NONE = "none"
PROFILE_CONNECTION_FIELD_COMPILER = "connection_field_compiler_v1"
SUPPORTED_PROFILES = (PROFILE_NONE, PROFILE_CONNECTION_FIELD_COMPILER)
STRUCTURAL_LIVE_EFFECT_SCALE_MIN = 0.0
STRUCTURAL_LIVE_EFFECT_SCALE_MAX = 2.0

FIELD_BOUNDS: dict[str, tuple[float, float]] = {
    "aggression_bias": (-2.0, 2.0),
    "action_temperature": (0.25, 2.5),
    "drive_logit_gain": (0.0, 3.0),
    "drive_inertia": (0.0, 0.999),
    "drive_noise": (0.0, 0.10),
    "pred_lr": (0.0, 1.0),
    "reproduction_threshold": (2.0, 90.0),
}
MODULE_FIELD_PARAMS: dict[str, tuple[str, ...]] = {
    "behavior_bias": ("aggression_bias", "action_temperature"),
    "identity_regulation": ("drive_logit_gain", "drive_inertia", "drive_noise", "pred_lr"),
    "life_history": ("reproduction_threshold",),
}
ACTION_REACHABLE_FIELDS = ("aggression_bias", "action_temperature", "drive_logit_gain")


def stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def structural_live_semantics_enabled(sim_cfg: Any) -> bool:
    return bool(getattr(sim_cfg, "structural_live_semantics_enable", False)) if sim_cfg is not None else False


def structural_live_semantics_profile(sim_cfg: Any) -> str:
    if sim_cfg is None:
        return PROFILE_NONE
    return str(getattr(sim_cfg, "structural_live_semantics_profile", PROFILE_NONE) or PROFILE_NONE)


def validate_structural_live_semantics_config(sim_cfg: Any) -> None:
    scale = structural_live_effect_scale(sim_cfg)
    if not math.isfinite(scale):
        raise ValueError("structural_live_effect_scale must be finite")
    if scale < STRUCTURAL_LIVE_EFFECT_SCALE_MIN or scale > STRUCTURAL_LIVE_EFFECT_SCALE_MAX:
        raise ValueError(
            "structural_live_effect_scale must be between "
            f"{STRUCTURAL_LIVE_EFFECT_SCALE_MIN} and {STRUCTURAL_LIVE_EFFECT_SCALE_MAX}, got {scale!r}"
        )
    if not structural_live_semantics_enabled(sim_cfg):
        return
    profile = structural_live_semantics_profile(sim_cfg)
    if profile not in SUPPORTED_PROFILES:
        raise ValueError(f"unsupported structural_live_semantics_profile={profile!r}")
    if profile == PROFILE_NONE:
        raise ValueError("structural_live_semantics_enable requires a non-none structural_live_semantics_profile")
    schema = str(getattr(sim_cfg, "structural_live_semantics_schema_version", SCHEMA_VERSION) or "")
    if schema != SCHEMA_VERSION:
        raise ValueError(f"structural_live_semantics_schema_version must be {SCHEMA_VERSION!r}, got {schema!r}")


def structural_live_effect_scale(sim_cfg: Any) -> float:
    if sim_cfg is None:
        return 1.0
    try:
        return float(getattr(sim_cfg, "structural_live_effect_scale", 1.0))
    except Exception:
        return float("nan")


def live_semantics_disabled_summary(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    module_count = len(list((payload or {}).get("modules", []) or []))
    connection_count = len(list((payload or {}).get("connections", []) or []))
    return {
        "schema_version": SCHEMA_VERSION,
        "enabled": False,
        "profile": PROFILE_NONE,
        "payload_empty": bool(module_count == 0 and connection_count == 0),
        "graph_validation_status": "disabled",
        "graph_supported": False,
        "graph_digest": None,
        "module_count": module_count,
        "connection_count": connection_count,
        "active_module_count": None,
        "active_connection_count": None,
        "cycle_status": None,
        "unsupported_reason": "disabled",
        "effective_fields_before": None,
        "effective_fields_after": None,
        "changed_effective_fields": [],
        "action_reachable_fields": list(ACTION_REACHABLE_FIELDS),
        "module_only_change_count": None,
        "live_connection_change_count": None,
        "live_expression_change_count": None,
        "stored_only_change_count": None,
        "invalid_rejection_count": None,
        "unsupported_graph_count": None,
        "compile_count": 0,
        "cache_hit_count": None,
        "cache_miss_count": None,
        "evaluation_count": 0,
        "nonfinite_failure_count": 0,
        "w6_support_status": None,
        "checkpoint_rebuild_status": None,
        "configured_effect_scale": 1.0,
        "structural_residual_before_scale": None,
        "structural_residual_after_scale": None,
        "effective_fields_before_clip": None,
        "clipped_fields": [],
        "saturated_fields": [],
        "scale_validation_status": "disabled",
        "reasons": ["disabled"],
    }


def apply_live_structural_semantics(
    *,
    genome: Any,
    expression: dict[str, Any],
    sim_cfg: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_structural_live_semantics_config(sim_cfg)
    payload = _payload_from_genome(genome)
    if not structural_live_semantics_enabled(sim_cfg):
        return dict(expression), live_semantics_disabled_summary(payload)
    profile = structural_live_semantics_profile(sim_cfg)
    compiled = compile_live_graph(payload, profile=profile)
    expr = dict(expression)
    before = _effective_from_expr(expr)
    effect_scale = structural_live_effect_scale(sim_cfg)
    summary = _summary_base(payload, profile=profile, before=before, compiled=compiled)
    summary["configured_effect_scale"] = float(effect_scale)
    if not bool(compiled["graph_supported"]):
        summary["effective_fields_after"] = dict(before)
        summary["changed_effective_fields"] = []
        summary["structural_residual_before_scale"] = {field: 0.0 for field in FIELD_BOUNDS}
        summary["structural_residual_after_scale"] = {field: 0.0 for field in FIELD_BOUNDS}
        summary["effective_fields_before_clip"] = dict(before)
        summary["unsupported_graph_count"] = 1
        summary["reasons"] = [str(compiled["unsupported_reason"])]
        expr["structural_live_semantics"] = summary
        return expr, summary

    active_by_id = expr.get("active_by_module_id", {}) if isinstance(expr.get("active_by_module_id", {}), dict) else {}
    unscaled = dict(before)
    expression_changed = set(_apply_expression_scale(payload, active_by_id, unscaled, clip_fields=False))
    connection_changed = set(_apply_connection_effects(payload, active_by_id, unscaled, compiled, clip_fields=False))
    residual_before_scale = {field: float(unscaled.get(field, 0.0)) - float(before.get(field, 0.0)) for field in FIELD_BOUNDS}
    residual_after_scale = {field: float(effect_scale) * float(residual_before_scale.get(field, 0.0)) for field in FIELD_BOUNDS}
    before_clip = {field: float(before.get(field, 0.0)) + float(residual_after_scale.get(field, 0.0)) for field in FIELD_BOUNDS}
    after = {field: _clip_field(field, value) for field, value in before_clip.items()}
    clipped_fields = sorted(field for field in FIELD_BOUNDS if float(after[field]) != float(before_clip[field]))
    saturated_fields = sorted(field for field in FIELD_BOUNDS if _field_saturated(field, after[field]))
    changed = sorted(field for field in FIELD_BOUNDS if float(before.get(field, 0.0)) != float(after.get(field, 0.0)))
    expr["effective"] = dict(after)
    summary.update(
        {
            "active_module_count": int(sum(1 for value in active_by_id.values() if bool(value))) if active_by_id else int(len(payload.get("modules", []) or [])),
            "active_connection_count": int(len(compiled["connections"])),
            "effective_fields_before_clip": dict(before_clip),
            "effective_fields_after": dict(after),
            "changed_effective_fields": changed,
            "structural_residual_before_scale": dict(residual_before_scale),
            "structural_residual_after_scale": dict(residual_after_scale),
            "clipped_fields": clipped_fields,
            "saturated_fields": saturated_fields,
            "live_expression_change_count": int(len(expression_changed)),
            "live_connection_change_count": int(len(connection_changed)),
            "module_only_change_count": int(len(expression_changed - connection_changed)),
            "stored_only_change_count": int(max(0, len(compiled["connections"]) - len(connection_changed))),
            "evaluation_count": 1,
            "unsupported_graph_count": 0,
            "scale_validation_status": "passed",
            "reasons": [],
        }
    )
    expr["structural_live_semantics"] = summary
    return expr, summary


def compile_live_graph(payload: dict[str, Any], *, profile: str = PROFILE_CONNECTION_FIELD_COMPILER) -> dict[str, Any]:
    modules = list(payload.get("modules", []) or [])
    connections = list(payload.get("connections", []) or [])
    base = {
        "schema_version": SCHEMA_VERSION,
        "profile": str(profile),
        "graph_validation_status": "ok",
        "graph_supported": True,
        "unsupported_reason": "",
        "cycle_status": "acyclic",
        "graph_digest": "",
        "connections": [],
        "module_ids": [],
        "reason_codes": [],
    }
    if profile != PROFILE_CONNECTION_FIELD_COMPILER:
        return _unsupported(base, "unsupported_profile")
    if len(modules) == 0 and len(connections) == 0:
        base["graph_digest"] = stable_hash({"profile": profile, "modules": [], "connections": []})
        return base
    if len(modules) > 12:
        return _unsupported(base, "module_cap_exceeded")
    if len(connections) > 32:
        return _unsupported(base, "connection_cap_exceeded")

    module_map: dict[str, dict[str, Any]] = {}
    for rec in modules:
        if not isinstance(rec, dict):
            return _unsupported(base, "module_record_not_dict")
        mid = str(rec.get("module_id", "") or "")
        if not mid:
            return _unsupported(base, "missing_module_id")
        if mid in module_map:
            return _unsupported(base, "duplicate_module_id")
        module_map[mid] = dict(rec)
    base["module_ids"] = sorted(module_map)

    out_slots, in_slots = _slot_maps(module_map)
    graph_edges: list[tuple[str, str]] = []
    active_connections: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str, str, str, str]] = set()
    for rec in sorted(connections, key=_connection_sort_key):
        if not isinstance(rec, dict):
            return _unsupported(base, "connection_record_not_dict")
        cid = str(rec.get("connection_id", "") or "")
        src = str(rec.get("src_module_id", "") or "")
        dst = str(rec.get("dst_module_id", "") or "")
        src_slot = str(rec.get("src_slot", "") or "")
        dst_slot = str(rec.get("dst_slot", "") or "")
        value_type = str(rec.get("value_type", "") or "")
        weight = _finite_float(rec.get("weight", 0.0))
        if weight is None:
            return _unsupported(base, "nonfinite_weight")
        if src not in module_map:
            return _unsupported(base, "missing_source")
        if dst not in module_map:
            return _unsupported(base, "missing_destination")
        if src_slot not in out_slots.get(src, {}):
            return _unsupported(base, "missing_source_slot")
        if dst_slot not in in_slots.get(dst, {}):
            return _unsupported(base, "missing_destination_slot")
        if str(out_slots[src][src_slot]) != value_type:
            return _unsupported(base, "source_slot_type_mismatch")
        allowed_dst = str(in_slots[dst][dst_slot])
        if not _destination_type_compatible(value_type=value_type, destination_type=allowed_dst):
            return _unsupported(base, "destination_slot_type_mismatch")
        edge_key = (src, dst, src_slot, dst_slot, value_type)
        duplicate_policy = "aggregate_sorted" if edge_key in seen_edges else "unique"
        seen_edges.add(edge_key)
        conn = dict(rec)
        conn["weight"] = _clip(float(weight), -2.0, 2.0)
        conn["duplicate_edge_policy"] = duplicate_policy
        active_connections.append(conn)
        graph_edges.append((src, dst))
    cycle = _cycle_status(sorted(module_map), graph_edges)
    if cycle != "acyclic":
        return _unsupported(base, "cycle_detected", cycle_status=cycle)
    base["connections"] = active_connections
    base["graph_digest"] = stable_hash(
        {
            "profile": profile,
            "modules": [_module_digest_record(module_map[mid]) for mid in sorted(module_map)],
            "connections": [
                {
                    "connection_id": str(c.get("connection_id", "")),
                    "src_module_id": str(c.get("src_module_id", "")),
                    "dst_module_id": str(c.get("dst_module_id", "")),
                    "src_slot": str(c.get("src_slot", "")),
                    "dst_slot": str(c.get("dst_slot", "")),
                    "value_type": str(c.get("value_type", "")),
                    "channel": str(c.get("channel", "")),
                    "weight": float(c.get("weight", 0.0)),
                }
                for c in active_connections
            ],
        }
    )
    return base


def _apply_expression_scale(
    payload: dict[str, Any],
    active_by_id: dict[str, Any],
    effective: dict[str, float],
    *,
    clip_fields: bool = True,
) -> list[str]:
    changed: list[str] = []
    for module in sorted(list(payload.get("modules", []) or []), key=lambda m: str(m.get("module_id", ""))):
        mid = str(module.get("module_id", ""))
        if active_by_id and not bool(active_by_id.get(mid, True)):
            continue
        params = module.get("params", {}) if isinstance(module.get("params", {}), dict) else {}
        scale = _clip(_finite_float(params.get("expression_scale", 1.0)) or 1.0, 0.0, 2.0)
        if scale == 1.0:
            continue
        for field in MODULE_FIELD_PARAMS.get(str(module.get("module_type", "")), ()):
            if field in params:
                before = float(effective.get(field, 0.0))
                effective[field] = _field_value(field, float(params[field]) * scale, clip_fields=clip_fields)
                if float(effective[field]) != before:
                    changed.append(field)
    return sorted(set(changed))


def _apply_connection_effects(
    payload: dict[str, Any],
    active_by_id: dict[str, Any],
    effective: dict[str, float],
    compiled: dict[str, Any],
    *,
    clip_fields: bool = True,
) -> list[str]:
    module_map = {str(m.get("module_id", "")): dict(m) for m in list(payload.get("modules", []) or [])}
    changed: list[str] = []
    for conn in list(compiled.get("connections", []) or []):
        src = str(conn.get("src_module_id", ""))
        dst = str(conn.get("dst_module_id", ""))
        if active_by_id and (not bool(active_by_id.get(src, True)) or not bool(active_by_id.get(dst, True))):
            continue
        weight = float(conn.get("weight", 0.0) or 0.0)
        if weight == 0.0:
            continue
        src_module = module_map.get(src, {})
        dst_module = module_map.get(dst, {})
        dst_type = str(dst_module.get("module_type", ""))
        if dst_type != "policy_core":
            continue
        src_type = str(src_module.get("module_type", ""))
        params = src_module.get("params", {}) if isinstance(src_module.get("params", {}), dict) else {}
        scale = _clip(_finite_float(params.get("expression_scale", 1.0)) or 1.0, 0.0, 2.0)
        if src_type == "behavior_bias" and str(conn.get("dst_slot", "")) == "bias_in":
            signal = float(params.get("aggression_bias", 0.0)) * scale
            before = float(effective.get("aggression_bias", 0.0))
            effective["aggression_bias"] = _field_value("aggression_bias", before + 0.25 * weight * signal, clip_fields=clip_fields)
            if effective["aggression_bias"] != before:
                changed.append("aggression_bias")
            temp_signal = (float(params.get("action_temperature", 1.0)) * scale) - 1.0
            before_temp = float(effective.get("action_temperature", 1.0))
            effective["action_temperature"] = _field_value("action_temperature", before_temp + 0.05 * weight * temp_signal, clip_fields=clip_fields)
            if effective["action_temperature"] != before_temp:
                changed.append("action_temperature")
        elif src_type == "identity_regulation" and str(conn.get("dst_slot", "")) == "mod_in":
            inertia = float(params.get("drive_inertia", 0.85)) * scale
            before_gain = float(effective.get("drive_logit_gain", 1.0))
            effective["drive_logit_gain"] = _field_value("drive_logit_gain", before_gain + 0.10 * weight * inertia, clip_fields=clip_fields)
            if effective["drive_logit_gain"] != before_gain:
                changed.append("drive_logit_gain")
            before_inertia = float(effective.get("drive_inertia", 0.85))
            effective["drive_inertia"] = _field_value("drive_inertia", before_inertia + 0.02 * weight * (inertia - 0.5), clip_fields=clip_fields)
            if effective["drive_inertia"] != before_inertia:
                changed.append("drive_inertia")
    return sorted(set(changed))


def _summary_base(
    payload: dict[str, Any],
    *,
    profile: str,
    before: dict[str, float],
    compiled: dict[str, Any],
) -> dict[str, Any]:
    module_count = len(list(payload.get("modules", []) or []))
    connection_count = len(list(payload.get("connections", []) or []))
    return {
        "schema_version": SCHEMA_VERSION,
        "enabled": True,
        "profile": profile,
        "payload_empty": bool(module_count == 0 and connection_count == 0),
        "graph_validation_status": str(compiled.get("graph_validation_status", "unknown")),
        "graph_supported": bool(compiled.get("graph_supported", False)),
        "graph_digest": compiled.get("graph_digest"),
        "module_count": module_count,
        "connection_count": connection_count,
        "active_module_count": None,
        "active_connection_count": None,
        "cycle_status": compiled.get("cycle_status"),
        "unsupported_reason": str(compiled.get("unsupported_reason", "")),
        "effective_fields_before": dict(before),
        "effective_fields_before_clip": None,
        "effective_fields_after": None,
        "changed_effective_fields": [],
        "configured_effect_scale": 1.0,
        "structural_residual_before_scale": None,
        "structural_residual_after_scale": None,
        "clipped_fields": [],
        "saturated_fields": [],
        "action_reachable_fields": list(ACTION_REACHABLE_FIELDS),
        "module_only_change_count": 0,
        "live_connection_change_count": 0,
        "live_expression_change_count": 0,
        "stored_only_change_count": 0,
        "invalid_rejection_count": 0 if bool(compiled.get("graph_supported", False)) else 1,
        "unsupported_graph_count": 0,
        "compile_count": 1,
        "cache_hit_count": None,
        "cache_miss_count": None,
        "evaluation_count": 0,
        "nonfinite_failure_count": 0,
        "w6_support_status": "derived_reference_path",
        "checkpoint_rebuild_status": "derived_from_genome",
        "scale_validation_status": "pending",
        "reasons": [],
    }


def _payload_from_genome(genome: Any) -> dict[str, Any]:
    if hasattr(genome, "canonical_payload"):
        return dict(genome.canonical_payload())
    if hasattr(genome, "to_dict"):
        from agents.modular_genome_v1 import canonical_payload_from_genome_dict

        return canonical_payload_from_genome_dict(dict(genome.to_dict()))
    from agents.modular_genome_v1 import canonical_payload_from_genome_dict

    return canonical_payload_from_genome_dict(dict(genome or {}))


def _effective_from_expr(expr: dict[str, Any]) -> dict[str, float]:
    raw = expr.get("effective", {}) if isinstance(expr.get("effective", {}), dict) else {}
    return {field: _clip_field(field, float(raw.get(field, 0.0))) for field in FIELD_BOUNDS}


def _slot_maps(module_map: dict[str, dict[str, Any]]) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    out_slots: dict[str, dict[str, str]] = {}
    in_slots: dict[str, dict[str, str]] = {}
    for mid, module in module_map.items():
        io = module.get("io", {}) if isinstance(module.get("io", {}), dict) else {}
        outs = io.get("outputs", []) if isinstance(io.get("outputs", []), list) else []
        ins = io.get("inputs", []) if isinstance(io.get("inputs", []), list) else []
        out_slots[mid] = {str(slot.get("slot_id", "")): str(slot.get("value_type", "")) for slot in outs if isinstance(slot, dict)}
        in_slots[mid] = {str(slot.get("slot_id", "")): str(slot.get("value_type", "")) for slot in ins if isinstance(slot, dict)}
    return out_slots, in_slots


def _destination_type_compatible(*, value_type: str, destination_type: str) -> bool:
    if value_type == destination_type:
        return True
    return bool(value_type in {"vector", "gate"} and destination_type == "scalar")


def _connection_sort_key(conn: Any) -> str:
    if isinstance(conn, dict):
        return str(conn.get("connection_id", ""))
    return ""


def _cycle_status(module_ids: list[str], edges: list[tuple[str, str]]) -> str:
    adjacency = {mid: [] for mid in module_ids}
    for src, dst in edges:
        adjacency.setdefault(src, []).append(dst)
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(mid: str) -> bool:
        if mid in visiting:
            return True
        if mid in visited:
            return False
        visiting.add(mid)
        for dst in sorted(adjacency.get(mid, [])):
            if visit(dst):
                return True
        visiting.remove(mid)
        visited.add(mid)
        return False

    return "cycle_detected" if any(visit(mid) for mid in sorted(module_ids)) else "acyclic"


def _module_digest_record(module: dict[str, Any]) -> dict[str, Any]:
    return {
        "module_id": str(module.get("module_id", "")),
        "module_type": str(module.get("module_type", "")),
        "params": module.get("params", {}) if isinstance(module.get("params", {}), dict) else {},
        "io": module.get("io", {}) if isinstance(module.get("io", {}), dict) else {},
    }


def _unsupported(base: dict[str, Any], reason: str, *, cycle_status: str = "not_evaluable") -> dict[str, Any]:
    out = dict(base)
    out["graph_validation_status"] = "unsupported"
    out["graph_supported"] = False
    out["unsupported_reason"] = str(reason)
    out["cycle_status"] = str(cycle_status)
    out["graph_digest"] = stable_hash({"unsupported_reason": str(reason), "cycle_status": str(cycle_status)})
    out["reason_codes"] = [str(reason)]
    out["connections"] = []
    return out


def _finite_float(value: Any) -> float | None:
    try:
        x = float(value)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def _clip(value: float, lo: float, hi: float) -> float:
    return float(min(float(hi), max(float(lo), float(value))))


def _clip_field(field: str, value: float) -> float:
    lo, hi = FIELD_BOUNDS[field]
    return _clip(float(value), lo, hi)


def _field_value(field: str, value: float, *, clip_fields: bool) -> float:
    return _clip_field(field, value) if bool(clip_fields) else float(value)


def _field_saturated(field: str, value: float) -> bool:
    lo, hi = FIELD_BOUNDS[field]
    span = max(1e-12, float(hi) - float(lo))
    val = float(value)
    return bool(val <= float(lo) + 0.02 * span or val >= float(hi) - 0.02 * span)
