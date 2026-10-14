from __future__ import annotations

import hashlib
import json
import math
import os
from typing import Any

import numpy as np

from agents.developmental_program_v1 import canonicalize_developmental_program
from agents.family_head_residuals import (
    FAMILY_HEAD_RESIDUAL_FIELD,
    canonicalize_optional_family_head_residual_block,
)
from agents.structural_identity_v1 import (
    STRUCTURAL_ANCESTRY_SCHEMA_VERSION,
    STRUCTURAL_IDENTITY_SCHEMA_VERSION,
    STRUCTURAL_LEGACY_MIGRATION_VERSION,
    TOPOLOGY_POSITION_SCHEMA_VERSION,
    canonical_topology_position,
    canonicalize_module_identity,
)
from evolution.structural_gated_recurrent_v1 import (
    GATED_OPERATOR,
    GATED_RECORD_KEY,
    normalize_gated_module_params,
)


MODULAR_GENOME_SCHEMA_VERSION = "modular_genome_v1"
B2_INTERFACE_SCHEMA_VERSION = "module_interfaces_v1"
B3_LOCALITY_SCHEMA_VERSION = "locality_v1"

# Phase B2 campaign-frozen constraints.
B2_ALLOWED_SLOT_TYPES = ("scalar", "vector", "gate")
B2_COMPATIBILITY_MATRIX: dict[str, tuple[str, ...]] = {
    "scalar": ("scalar",),
    "vector": ("vector", "scalar"),
    "gate": ("gate", "scalar"),
}
B2_FROZEN_CAPS: dict[str, int] = {
    "max_modules_per_agent": 12,
    "max_connections_per_agent": 32,
    "max_module_fanin": 6,
    "max_module_fanout": 6,
}

# Capacity epochs may select only one of these predeclared mechanical
# envelopes.  The legacy envelope remains the default, so payloads which do
# not opt into an epoch keep their historical canonical bytes.
B2_EXPANDED_CAPS: dict[str, int] = {
    "max_modules_per_agent": 16,
    "max_connections_per_agent": 48,
    "max_module_fanin": 6,
    "max_module_fanout": 6,
}
B2_CAPACITY_PROFILES: dict[str, dict[str, int]] = {
    "b2-12x32-v1": dict(B2_FROZEN_CAPS),
    "b2-16x48-v1": dict(B2_EXPANDED_CAPS),
}


def resolve_b2_caps(value: Any = None) -> dict[str, int]:
    """Resolve an exact predeclared capacity envelope, never an ad-hoc cap."""

    if value is None:
        return dict(B2_FROZEN_CAPS)
    if not isinstance(value, dict):
        raise ValueError("B2_CAPACITY_PROFILE_MALFORMED")
    if set(value) != set(B2_FROZEN_CAPS):
        raise ValueError("B2_CAPACITY_PROFILE_FIELDS_MISMATCH")
    if any(type(value[key]) is not int for key in B2_FROZEN_CAPS):
        raise ValueError("B2_CAPACITY_PROFILE_MALFORMED")
    candidate = {str(key): value[key] for key in B2_FROZEN_CAPS}
    for profile_caps in B2_CAPACITY_PROFILES.values():
        if candidate == profile_caps:
            return dict(candidate)
    raise ValueError(f"B2_CAPACITY_PROFILE_UNSUPPORTED caps={candidate!r}")


def b2_capacity_profile_id(value: Any = None) -> str:
    caps = resolve_b2_caps(value)
    for profile_id, profile_caps in B2_CAPACITY_PROFILES.items():
        if caps == profile_caps:
            return str(profile_id)
    raise ValueError("B2_CAPACITY_PROFILE_UNSUPPORTED")


def b2_capacity_neutral_ordering_payload(value: dict[str, Any]) -> dict[str, Any]:
    """Project epoch metadata to legacy caps for neutral ordering only."""

    out = dict(value)
    resolve_b2_caps(out.get("b2_caps", None))
    out["b2_caps"] = dict(B2_FROZEN_CAPS)
    return out

B3_FROZEN_LOCALITY: dict[str, float] = {
    "max_connection_distance": 0.65,
    "distance_cost_scale": 1.0,
    "distance_cost_max": 1.0,
}

_MODULE_IO_DEFAULTS: dict[str, dict[str, list[dict[str, str]]]] = {
    "policy_core": {
        "inputs": [
            {"slot_id": "state_in", "value_type": "vector"},
            {"slot_id": "bias_in", "value_type": "scalar"},
            {"slot_id": "mod_in", "value_type": "gate"},
        ],
        "outputs": [{"slot_id": "action_out", "value_type": "vector"}],
    },
    "behavior_bias": {
        "inputs": [{"slot_id": "state_in", "value_type": "vector"}],
        "outputs": [{"slot_id": "bias_out", "value_type": "scalar"}],
    },
    "life_history": {
        "inputs": [{"slot_id": "state_in", "value_type": "vector"}],
        "outputs": [{"slot_id": "threshold_out", "value_type": "scalar"}],
    },
    "identity_regulation": {
        "inputs": [{"slot_id": "state_in", "value_type": "vector"}],
        "outputs": [{"slot_id": "mod_out", "value_type": "gate"}],
    },
    "temperament_priors": {
        "inputs": [{"slot_id": "state_in", "value_type": "vector"}],
        "outputs": [{"slot_id": "prior_out", "value_type": "scalar"}],
    },
    "generic": {
        "inputs": [{"slot_id": "in", "value_type": "scalar"}],
        "outputs": [{"slot_id": "out", "value_type": "scalar"}],
    },
}


def strict_mode_enabled() -> bool:
    v1 = str(os.environ.get("EIEL_STRICT_DETERMINISM", "")).strip().lower()
    v2 = str(os.environ.get("DEBUG_DETERMINISM", "")).strip().lower()
    v3 = str(os.environ.get("EIEL_DETERMINISM_STRICT", "")).strip().lower()
    return (v1 in {"1", "true", "yes", "on"}) or (v2 in {"1", "true", "yes", "on"}) or (v3 in {"1", "true", "yes", "on"})


def det_fail(code: str, context: str) -> None:
    raise RuntimeError(f"[DETERMINISM_VIOLATION]\ncode={code}\ncontext={context}\n")


def canon_float(value: Any, *, field: str = "") -> float:
    try:
        x = float(value)
    except Exception:
        if strict_mode_enabled():
            det_fail("DET_GENOME_CANONICALIZATION_INVALID", f"field={field};reason=non_numeric")
        x = 0.0
    if not math.isfinite(x):
        if strict_mode_enabled():
            det_fail("DET_GENOME_CANONICALIZATION_INVALID", f"field={field};reason=non_finite")
        x = 0.0
    x32 = np.float32(x).item()
    return float(round(float(x32), 8))


def _canon_scalar(v: Any, *, field: str = "") -> Any:
    if isinstance(v, bool):
        return bool(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return canon_float(v, field=field)
    if v is None:
        return None
    if isinstance(v, str):
        return str(v)
    if strict_mode_enabled():
        det_fail("DET_GENOME_CANONICALIZATION_INVALID", f"field={field};reason=unsupported_scalar_type")
    return str(v)


def _canon_obj(v: Any, *, field: str = "") -> Any:
    if isinstance(v, dict):
        out: dict[str, Any] = {}
        for k in sorted(v.keys(), key=lambda x: str(x)):
            ks = str(k)
            out[ks] = _canon_obj(v[k], field=f"{field}.{ks}" if field else ks)
        return out
    if isinstance(v, (list, tuple)):
        return [_canon_obj(x, field=field) for x in v]
    return _canon_scalar(v, field=field)


def _canon_pos2(v: Any, *, field: str = "") -> list[float]:
    arr = []
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        arr = [canon_float(v[0], field=f"{field}[0]"), canon_float(v[1], field=f"{field}[1]")]
    elif isinstance(v, dict):
        arr = [canon_float(v.get("x", 0.0), field=f"{field}.x"), canon_float(v.get("y", 0.0), field=f"{field}.y")]
    else:
        arr = [0.0, 0.0]
    x = float(min(1.0, max(0.0, arr[0])))
    y = float(min(1.0, max(0.0, arr[1])))
    return [canon_float(x, field=f"{field}[0]"), canon_float(y, field=f"{field}[1]")]


def _latent_position_for_module(module_record: dict[str, Any]) -> list[float]:
    """Compatibility projection of the explicit stable topology position."""

    return _canon_pos2(
        canonical_topology_position(
            module_record.get("topology_pos", module_record.get("latent_pos", [0.0, 0.0]))
        ),
        field="latent_pos",
    )


def _distance_and_cost(src_pos: list[float], dst_pos: list[float]) -> tuple[float, float]:
    sx, sy = float(src_pos[0]), float(src_pos[1])
    dx, dy = float(dst_pos[0]), float(dst_pos[1])
    dd = ((sx - dx) ** 2 + (sy - dy) ** 2) ** 0.5
    # Normalize [0, sqrt(2)] to [0, 1]
    dist = min(1.0, max(0.0, dd / (2.0**0.5)))
    dist = canon_float(dist, field="locality_distance")
    cost = min(
        float(B3_FROZEN_LOCALITY["distance_cost_max"]),
        max(0.0, dist * float(B3_FROZEN_LOCALITY["distance_cost_scale"])),
    )
    cost = canon_float(cost, field="locality_cost")
    return float(dist), float(cost)


def _coerce_nonneg_int(value: Any, default: int, *, field: str = "") -> int:
    try:
        iv = int(value)
    except Exception:
        iv = int(default)
    if iv < 0:
        if strict_mode_enabled():
            det_fail("DET_GENOME_B2_INVALID_CAP", f"field={field};value={iv}")
        iv = int(default)
    return int(iv)


def _canon_slot(slot: Any, *, field: str) -> dict[str, str]:
    if isinstance(slot, str):
        sid = str(slot).strip() or "slot"
        vtype = "scalar"
    elif isinstance(slot, dict):
        sid = str(slot.get("slot_id", slot.get("name", "")) or "").strip() or "slot"
        vtype = str(slot.get("value_type", "scalar") or "scalar").strip().lower()
    else:
        if strict_mode_enabled():
            det_fail("DET_GENOME_B2_SCHEMA_MALFORMED", f"{field};reason=slot_not_dict")
        sid = "slot"
        vtype = "scalar"
    if vtype not in B2_ALLOWED_SLOT_TYPES:
        if strict_mode_enabled():
            det_fail("DET_GENOME_B2_INVALID_SLOT_TYPE", f"{field};value_type={vtype}")
        vtype = "scalar"
    return {"slot_id": sid, "value_type": vtype}


def _module_io_defaults(module_type: str) -> dict[str, list[dict[str, str]]]:
    key = str(module_type or "generic")
    if key in _MODULE_IO_DEFAULTS:
        return _MODULE_IO_DEFAULTS[key]
    return _MODULE_IO_DEFAULTS["generic"]


def _canon_module_io(module_type: str, io_obj: Any, *, field: str) -> dict[str, Any]:
    defaults = _module_io_defaults(module_type)
    io_dict = io_obj if isinstance(io_obj, dict) else {}
    if io_obj is not None and not isinstance(io_obj, dict) and strict_mode_enabled():
        det_fail("DET_GENOME_B2_SCHEMA_MALFORMED", f"{field};reason=io_not_dict")

    in_raw = io_dict.get("inputs", defaults["inputs"])
    out_raw = io_dict.get("outputs", defaults["outputs"])
    if not isinstance(in_raw, list):
        if strict_mode_enabled():
            det_fail("DET_GENOME_B2_SCHEMA_MALFORMED", f"{field}.inputs;reason=not_list")
        in_raw = list(defaults["inputs"])
    if not isinstance(out_raw, list):
        if strict_mode_enabled():
            det_fail("DET_GENOME_B2_SCHEMA_MALFORMED", f"{field}.outputs;reason=not_list")
        out_raw = list(defaults["outputs"])

    inputs = [_canon_slot(s, field=f"{field}.inputs[{i}]") for i, s in enumerate(in_raw)]
    outputs = [_canon_slot(s, field=f"{field}.outputs[{i}]") for i, s in enumerate(out_raw)]
    inputs.sort(key=lambda s: (str(s["slot_id"]), str(s["value_type"])))
    outputs.sort(key=lambda s: (str(s["slot_id"]), str(s["value_type"])))

    caps_raw = io_dict.get("caps", {}) if isinstance(io_dict.get("caps", {}), dict) else {}
    max_fanin = _coerce_nonneg_int(
        caps_raw.get("max_fanin", B2_FROZEN_CAPS["max_module_fanin"]),
        B2_FROZEN_CAPS["max_module_fanin"],
        field=f"{field}.caps.max_fanin",
    )
    max_fanout = _coerce_nonneg_int(
        caps_raw.get("max_fanout", B2_FROZEN_CAPS["max_module_fanout"]),
        B2_FROZEN_CAPS["max_module_fanout"],
        field=f"{field}.caps.max_fanout",
    )
    max_fanin = min(max_fanin, B2_FROZEN_CAPS["max_module_fanin"])
    max_fanout = min(max_fanout, B2_FROZEN_CAPS["max_module_fanout"])
    return {
        "inputs": inputs,
        "outputs": outputs,
        "caps": {"max_fanin": int(max_fanin), "max_fanout": int(max_fanout)},
    }


def _legacy_default_modules(flat: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "module_id": "m000_policy_core",
            "module_type": "policy_core",
            "params": {
                "hidden_size": int(flat.get("hidden_size", 64) or 64),
                "use_gru": bool(flat.get("use_gru", False)),
                "memory_dim": int(flat.get("memory_dim", 0) or 0),
            },
        },
        {
            "module_id": "m010_behavior_bias",
            "module_type": "behavior_bias",
            "params": {
                "exploration_bias": canon_float(flat.get("exploration_bias", 0.0), field="exploration_bias"),
                "aggression_bias": canon_float(flat.get("aggression_bias", 0.0), field="aggression_bias"),
                "action_temperature": canon_float(flat.get("action_temperature", 1.0), field="action_temperature"),
            },
        },
        {
            "module_id": "m020_life_history",
            "module_type": "life_history",
            "params": {
                "reproduction_threshold": canon_float(flat.get("reproduction_threshold", 30.0), field="reproduction_threshold"),
                "mutation_rate": canon_float(flat.get("mutation_rate", 0.05), field="mutation_rate"),
            },
        },
        {
            "module_id": "m030_identity_regulation",
            "module_type": "identity_regulation",
            "params": {
                "drive_inertia": canon_float(flat.get("drive_inertia", 0.85), field="drive_inertia"),
                "drive_noise": canon_float(flat.get("drive_noise", 0.02), field="drive_noise"),
                "drive_logit_gain": canon_float(flat.get("drive_logit_gain", 1.0), field="drive_logit_gain"),
                "pred_lr": canon_float(flat.get("pred_lr", 0.08), field="pred_lr"),
            },
        },
        {
            "module_id": "m040_temperament_priors",
            "module_type": "temperament_priors",
            "params": {
                "drive_sp_hunger": canon_float(flat.get("drive_sp_hunger", 0.55), field="drive_sp_hunger"),
                "drive_sp_pain": canon_float(flat.get("drive_sp_pain", 0.05), field="drive_sp_pain"),
                "drive_sp_fatigue": canon_float(flat.get("drive_sp_fatigue", 0.15), field="drive_sp_fatigue"),
                "drive_sp_fear": canon_float(flat.get("drive_sp_fear", 0.10), field="drive_sp_fear"),
                "exp_prior_strength": canon_float(flat.get("exp_prior_strength", 0.35), field="exp_prior_strength"),
                "exp_prior_noise": canon_float(flat.get("exp_prior_noise", 0.05), field="exp_prior_noise"),
                "drive_birth_noise": canon_float(flat.get("drive_birth_noise", 0.03), field="drive_birth_noise"),
            },
        },
    ]


def _legacy_default_connections(flat: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "connection_id": "c000_behavior_to_policy",
            "src_module_id": "m010_behavior_bias",
            "dst_module_id": "m000_policy_core",
            "src_slot": "bias_out",
            "dst_slot": "bias_in",
            "value_type": "scalar",
            "channel": "logit_bias",
            "weight": canon_float(flat.get("drive_logit_gain", 1.0), field="connection.weight"),
        },
        {
            "connection_id": "c010_identity_to_policy",
            "src_module_id": "m030_identity_regulation",
            "dst_module_id": "m000_policy_core",
            "src_slot": "mod_out",
            "dst_slot": "mod_in",
            "value_type": "gate",
            "channel": "modulation",
            "weight": canon_float(flat.get("drive_inertia", 0.85), field="connection.weight"),
        },
    ]


def normalize_modules(modules: Any, *, fallback_flat: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    src = modules
    if not isinstance(src, list):
        if src is None:
            src = _legacy_default_modules(fallback_flat or {})
        else:
            if strict_mode_enabled():
                det_fail("DET_GENOME_SCHEMA_MALFORMED", "field=modules;reason=not_list")
            src = _legacy_default_modules(fallback_flat or {})

    records: list[dict[str, Any]] = []
    for idx, rec in enumerate(src):
        if not isinstance(rec, dict):
            if strict_mode_enabled():
                det_fail("DET_GENOME_SCHEMA_MALFORMED", f"field=modules[{idx}]")
            continue
        mtype = str(rec.get("module_type", rec.get("kind", "generic")) or "generic")
        mid = rec.get("module_id", rec.get("id", ""))
        params = rec.get("params", {})
        io = rec.get("io", None)
        if not isinstance(params, dict):
            if strict_mode_enabled():
                det_fail("DET_GENOME_SCHEMA_MALFORMED", f"field=modules[{idx}].params")
            params = {}
        raw_operator = str(
            params.get("live_operator", "bounded_scalar_integrator_v1")
            or "bounded_scalar_integrator_v1"
        )
        if raw_operator == GATED_OPERATOR and mtype != "generic":
            raise ValueError(
                "ESCAP7_GATED_OPERATOR_REQUIRES_GENERIC_MODULE "
                f"module_id={mid}"
            )
        if raw_operator == GATED_OPERATOR or GATED_RECORD_KEY in params:
            # Validate before generic scalar canonicalization can coerce a
            # malformed/non-finite gated field into a plausible zero.
            params = normalize_gated_module_params(params)
        records.append(
            {
                "module_id": str(mid) if str(mid or "").strip() else "",
                "module_type": mtype,
                "params": _canon_obj(params, field=f"modules[{idx}].params"),
                "io": _canon_module_io(mtype, io, field=f"modules[{idx}].io"),
                **canonicalize_module_identity(
                    rec,
                    record_index=idx,
                    strict_partial_native=strict_mode_enabled(),
                ),
            }
        )

    records.sort(
        key=lambda r: (
            r.get("module_id") or "",
            str(r.get("module_type") or ""),
            json.dumps(r.get("params") or {}, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False),
            json.dumps(r.get("io") or {}, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False),
        )
    )
    out: list[dict[str, Any]] = []
    for idx, rec in enumerate(records):
        rid = str(rec.get("module_id", "") or "")
        if not rid:
            rid = f"m{idx:03d}_{str(rec.get('module_type', 'generic'))}"
        out.append(
            {
                "module_id": rid,
                "module_type": str(rec.get("module_type", "generic")),
                "params": _canon_obj(rec.get("params", {}), field=f"modules[{idx}].params"),
                "io": _canon_obj(rec.get("io", {}), field=f"modules[{idx}].io"),
                "identity_schema_version": str(rec["identity_schema_version"]),
                "module_instance_id": str(rec["module_instance_id"]),
                "homology_id": str(rec["homology_id"]),
                "identity_origin": str(rec["identity_origin"]),
                "ancestry": _canon_obj(rec.get("ancestry", {}), field=f"modules[{idx}].ancestry"),
                "topology_position_schema_version": str(rec["topology_position_schema_version"]),
                "topology_pos": _canon_pos2(rec.get("topology_pos", [0.0, 0.0]), field=f"modules[{idx}].topology_pos"),
                "legacy_identity_source": _canon_obj(
                    rec.get("legacy_identity_source", None),
                    field=f"modules[{idx}].legacy_identity_source",
                ),
                "latent_pos": [0.0, 0.0],
            }
        )
    # ``latent_pos`` remains a compatibility alias.  It no longer hashes
    # mutable functional parameters.
    for i, rec in enumerate(out):
        rec["latent_pos"] = _latent_position_for_module(rec)
        rec["latent_pos"] = _canon_pos2(rec.get("latent_pos", [0.0, 0.0]), field=f"modules[{i}].latent_pos")
    out.sort(key=lambda r: str(r["module_id"]))
    # Capacity enforcement belongs to apply_b2_constraints(), where the
    # caller's explicit compatibility (clip) or active-science (reject) policy
    # is available.  Clipping here used to erase overflow before reject could
    # observe it.
    return out


def normalize_connections(connections: Any, *, fallback_flat: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    src = connections
    if not isinstance(src, list):
        if src is None:
            src = _legacy_default_connections(fallback_flat or {})
        else:
            if strict_mode_enabled():
                det_fail("DET_GENOME_SCHEMA_MALFORMED", "field=connections;reason=not_list")
            src = _legacy_default_connections(fallback_flat or {})

    records: list[dict[str, Any]] = []
    for idx, rec in enumerate(src):
        if not isinstance(rec, dict):
            if strict_mode_enabled():
                det_fail("DET_GENOME_SCHEMA_MALFORMED", f"field=connections[{idx}]")
            continue
        cid = rec.get("connection_id", rec.get("id", ""))
        src_id = rec.get("src_module_id", rec.get("src", ""))
        dst_id = rec.get("dst_module_id", rec.get("dst", ""))
        src_slot = rec.get("src_slot", rec.get("src_port", "out"))
        dst_slot = rec.get("dst_slot", rec.get("dst_port", "in"))
        channel = rec.get("channel", rec.get("kind", "typed"))
        value_type = str(rec.get("value_type", "scalar") or "scalar").strip().lower()
        if value_type not in B2_ALLOWED_SLOT_TYPES:
            if strict_mode_enabled():
                det_fail("DET_GENOME_B2_INVALID_SLOT_TYPE", f"field=connections[{idx}].value_type;value={value_type}")
            value_type = "scalar"
        weight = canon_float(rec.get("weight", 1.0), field=f"connections[{idx}].weight")
        records.append(
            {
                "connection_id": str(cid) if str(cid or "").strip() else "",
                "src_module_id": str(src_id),
                "dst_module_id": str(dst_id),
                "src_slot": str(src_slot or "out"),
                "dst_slot": str(dst_slot or "in"),
                "value_type": str(value_type),
                "channel": str(channel),
                "weight": float(weight),
            }
        )

    records.sort(
        key=lambda r: (
            str(r.get("src_module_id", "")),
            str(r.get("dst_module_id", "")),
            str(r.get("src_slot", "")),
            str(r.get("dst_slot", "")),
            str(r.get("value_type", "")),
            str(r.get("channel", "")),
            float(r.get("weight", 0.0)),
            str(r.get("connection_id", "")),
        )
    )
    out: list[dict[str, Any]] = []
    for idx, rec in enumerate(records):
        cid = str(rec.get("connection_id", "") or "")
        if not cid:
            cid = f"c{idx:03d}_{str(rec.get('src_module_id', 'src'))}__{str(rec.get('dst_module_id', 'dst'))}"
        out.append(
            {
                "connection_id": cid,
                "src_module_id": str(rec.get("src_module_id", "")),
                "dst_module_id": str(rec.get("dst_module_id", "")),
                "src_slot": str(rec.get("src_slot", "out")),
                "dst_slot": str(rec.get("dst_slot", "in")),
                "value_type": str(rec.get("value_type", "scalar")),
                "channel": str(rec.get("channel", "generic")),
                "weight": canon_float(rec.get("weight", 1.0), field=f"connections[{idx}].weight"),
            }
        )
    out.sort(key=lambda r: str(r["connection_id"]))
    # See normalize_modules(): retain overflow until the policy-aware boundary.
    return out


def _compat_ok(src_type: str, dst_type: str, conn_type: str) -> bool:
    conn_type = str(conn_type or "scalar")
    src_type = str(src_type or "scalar")
    dst_type = str(dst_type or "scalar")
    allow = B2_COMPATIBILITY_MATRIX.get(src_type, ())
    return (conn_type == src_type) and (dst_type in allow)


def apply_b2_constraints(
    modules: list[dict[str, Any]],
    connections: list[dict[str, Any]],
    *,
    invalid_policy: str = "clip",
    b2_caps: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    resolved_caps = resolve_b2_caps(b2_caps)
    policy = str(invalid_policy or "clip").strip().lower()
    if policy not in {"clip", "reject"}:
        if strict_mode_enabled():
            det_fail("DET_GENOME_B2_INVALID_POLICY", f"policy={policy}")
        policy = "clip"
    if strict_mode_enabled():
        policy = "reject"

    mod_sorted = sorted(list(modules or []), key=lambda m: str(m.get("module_id", "")))
    if len(mod_sorted) > int(resolved_caps["max_modules_per_agent"]):
        if policy == "reject":
            det_fail(
                "DET_GENOME_B2_BUDGET_EXCEEDED",
                f"scope=modules;max={resolved_caps['max_modules_per_agent']};observed={len(mod_sorted)}",
            )
        mod_sorted = mod_sorted[: int(resolved_caps["max_modules_per_agent"])]

    by_id: dict[str, dict[str, Any]] = {str(m.get("module_id", "")): m for m in mod_sorted if str(m.get("module_id", ""))}
    kept_modules = sorted(by_id.values(), key=lambda m: str(m.get("module_id", "")))
    out_slots: dict[str, dict[str, str]] = {}
    in_slots: dict[str, dict[str, str]] = {}
    caps: dict[str, tuple[int, int]] = {}
    pos_by_id: dict[str, list[float]] = {}
    for m in kept_modules:
        mid = str(m.get("module_id", ""))
        io = m.get("io", {}) if isinstance(m.get("io", {}), dict) else {}
        in_defs = io.get("inputs", []) if isinstance(io.get("inputs", []), list) else []
        out_defs = io.get("outputs", []) if isinstance(io.get("outputs", []), list) else []
        out_slots[mid] = {str(s.get("slot_id", "")): str(s.get("value_type", "scalar")) for s in out_defs if isinstance(s, dict)}
        in_slots[mid] = {str(s.get("slot_id", "")): str(s.get("value_type", "scalar")) for s in in_defs if isinstance(s, dict)}
        cap_obj = io.get("caps", {}) if isinstance(io.get("caps", {}), dict) else {}
        fanin = _coerce_nonneg_int(cap_obj.get("max_fanin", B2_FROZEN_CAPS["max_module_fanin"]), B2_FROZEN_CAPS["max_module_fanin"])
        fanout = _coerce_nonneg_int(cap_obj.get("max_fanout", B2_FROZEN_CAPS["max_module_fanout"]), B2_FROZEN_CAPS["max_module_fanout"])
        caps[mid] = (
            min(fanin, int(B2_FROZEN_CAPS["max_module_fanin"])),
            min(fanout, int(B2_FROZEN_CAPS["max_module_fanout"])),
        )
        pos_by_id[mid] = _canon_pos2(m.get("latent_pos", [0.0, 0.0]), field=f"modules[{mid}].latent_pos")

    conn_sorted = sorted(list(connections or []), key=lambda c: str(c.get("connection_id", "")))
    fanin_count: dict[str, int] = {}
    fanout_count: dict[str, int] = {}
    kept_connections: list[dict[str, Any]] = []
    dropped = 0
    dropped_nonlocal = 0
    locality_cost_total = 0.0
    locality_distance_sum = 0.0
    locality_distance_max = 0.0
    max_local_dist = float(B3_FROZEN_LOCALITY["max_connection_distance"])
    for rec in conn_sorted:
        src_id = str(rec.get("src_module_id", ""))
        dst_id = str(rec.get("dst_module_id", ""))
        src_slot = str(rec.get("src_slot", "out"))
        dst_slot = str(rec.get("dst_slot", "in"))
        ctype = str(rec.get("value_type", "scalar"))
        if src_id not in by_id or dst_id not in by_id:
            if policy == "reject":
                det_fail("DET_GENOME_B2_INVALID_CONNECTION", f"reason=missing_module;src={src_id};dst={dst_id}")
            dropped += 1
            continue
        src_type = out_slots.get(src_id, {}).get(src_slot, "")
        dst_type = in_slots.get(dst_id, {}).get(dst_slot, "")
        if not src_type or not dst_type:
            if policy == "reject":
                det_fail(
                    "DET_GENOME_B2_INVALID_CONNECTION",
                    f"reason=missing_slot;src={src_id}.{src_slot};dst={dst_id}.{dst_slot}",
                )
            dropped += 1
            continue
        if not _compat_ok(src_type, dst_type, ctype):
            if policy == "reject":
                det_fail(
                    "DET_GENOME_B2_TYPE_MISMATCH",
                    f"src={src_id}.{src_slot}:{src_type};dst={dst_id}.{dst_slot}:{dst_type};conn={ctype}",
                )
            dropped += 1
            continue
        dst_cap = caps.get(dst_id, (int(B2_FROZEN_CAPS["max_module_fanin"]), int(B2_FROZEN_CAPS["max_module_fanout"])))
        src_cap = caps.get(src_id, (int(B2_FROZEN_CAPS["max_module_fanin"]), int(B2_FROZEN_CAPS["max_module_fanout"])))
        dst_fanin = int(dst_cap[0])
        src_fanout = int(src_cap[1])
        cur_in = int(fanin_count.get(dst_id, 0))
        cur_out = int(fanout_count.get(src_id, 0))
        if cur_in >= dst_fanin or cur_out >= src_fanout:
            if policy == "reject":
                det_fail(
                    "DET_GENOME_B2_BUDGET_EXCEEDED",
                    f"scope=fan;src={src_id};dst={dst_id};cur_in={cur_in};cur_out={cur_out};cap_in={dst_fanin};cap_out={src_fanout}",
                )
            dropped += 1
            continue
        if len(kept_connections) >= int(resolved_caps["max_connections_per_agent"]):
            if policy == "reject":
                det_fail(
                    "DET_GENOME_B2_BUDGET_EXCEEDED",
                    f"scope=connections;max={resolved_caps['max_connections_per_agent']};observed_gt={len(kept_connections)+1}",
                )
            dropped += 1
            continue
        s_pos = pos_by_id.get(src_id, [0.0, 0.0])
        d_pos = pos_by_id.get(dst_id, [0.0, 0.0])
        dist, cost = _distance_and_cost(s_pos, d_pos)
        if dist > max_local_dist:
            if policy == "reject":
                det_fail(
                    "DET_GENOME_B3_LOCALITY_VIOLATION",
                    f"src={src_id};dst={dst_id};distance={dist};max={max_local_dist}",
                )
            dropped += 1
            dropped_nonlocal += 1
            continue
        rec2 = dict(rec)
        rec2["locality_distance"] = float(dist)
        rec2["locality_cost"] = float(cost)
        kept_connections.append(rec2)
        locality_cost_total += float(cost)
        locality_distance_sum += float(dist)
        locality_distance_max = max(locality_distance_max, float(dist))
        fanin_count[dst_id] = cur_in + 1
        fanout_count[src_id] = cur_out + 1

    kept_connections.sort(key=lambda r: str(r.get("connection_id", "")))
    n_conn = len(kept_connections)
    meta = {
        "module_count": len(kept_modules),
        "connection_count": n_conn,
        "dropped_connections": int(dropped),
        "dropped_nonlocal_connections": int(dropped_nonlocal),
        "locality_cost_total": canon_float(locality_cost_total, field="b3.locality_cost_total"),
        "locality_mean_distance": canon_float((locality_distance_sum / float(n_conn)) if n_conn > 0 else 0.0, field="b3.locality_mean_distance"),
        "locality_max_distance": canon_float(locality_distance_max, field="b3.locality_max_distance"),
    }
    return kept_modules, kept_connections, meta


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")


def canonical_payload_from_genome_dict(raw: dict[str, Any]) -> dict[str, Any]:
    schema = str(raw.get("schema_version", "") or "")
    if schema and schema != MODULAR_GENOME_SCHEMA_VERSION:
        if strict_mode_enabled():
            det_fail("DET_GENOME_SCHEMA_UNSUPPORTED", f"schema_version={schema}")
        raise ValueError(f"unsupported schema_version={schema}")

    genes_obj = raw.get("genes", None)
    genes = dict(genes_obj) if isinstance(genes_obj, dict) else dict(raw)
    genes.pop("schema_version", None)
    genes.pop("interface_schema_version", None)
    genes.pop("b2_invalid_policy", None)
    genes.pop("b2_caps", None)
    genes.pop("b3_locality_schema_version", None)
    genes.pop("b3_locality_max_connection_distance", None)
    genes.pop("b3_distance_cost_scale", None)
    genes.pop("b3_distance_cost_max", None)
    genes.pop("b3_locality_cost_total", None)
    genes.pop("b3_locality_mean_distance", None)
    genes.pop("b3_locality_max_distance", None)
    genes.pop("b3_locality_dropped_connections", None)
    genes.pop("modules", None)
    genes.pop("connections", None)
    genes.pop("developmental_program", None)
    genes.pop("genes", None)
    canon_genes = _canon_obj(genes, field="genes")
    if not isinstance(canon_genes, dict):
        if strict_mode_enabled():
            det_fail("DET_GENOME_SCHEMA_MALFORMED", "field=genes")
        canon_genes = {}
    if FAMILY_HEAD_RESIDUAL_FIELD in canon_genes:
        canon_genes[FAMILY_HEAD_RESIDUAL_FIELD] = canonicalize_optional_family_head_residual_block(
            canon_genes.get(FAMILY_HEAD_RESIDUAL_FIELD)
        )
    canon_genes["structural_identity_schema_version"] = STRUCTURAL_IDENTITY_SCHEMA_VERSION
    canon_genes["structural_ancestry_schema_version"] = STRUCTURAL_ANCESTRY_SCHEMA_VERSION
    canon_genes["topology_position_schema_version"] = TOPOLOGY_POSITION_SCHEMA_VERSION
    canon_genes["structural_legacy_migration_version"] = STRUCTURAL_LEGACY_MIGRATION_VERSION
    modules = normalize_modules(raw.get("modules", None), fallback_flat=canon_genes)
    # ESCAP-4 bindings are hereditary module-local parameters.  Canonicalize
    # only explicit blocks so legacy/disabled payload representation remains
    # unchanged while binding-list permutation cannot affect payload bytes.
    from evolution.structural_observation_routing_v1 import normalize_agent_routing_modules

    modules = normalize_agent_routing_modules(modules)
    # ESCAP-5 output bindings are likewise optional module-local hereditary
    # records.  Preserve an absent block exactly for legacy/disabled payloads,
    # while canonicalizing explicit semantic targets independently of current
    # action-vector positions.
    from evolution.structural_output_routing_v1 import normalize_agent_output_routing_modules

    modules = normalize_agent_output_routing_modules(modules)
    # ESCAP-6 developmental records remain absent for legacy payloads.  Only
    # explicit records are canonicalized, then their bounded homology DAG is
    # validated before the payload can cross a genome boundary.
    from evolution.structural_development_v1 import normalize_agent_developmental_modules

    modules = normalize_agent_developmental_modules(modules)
    # ESCAP-7 retains the existing generic outer module and allocates only an
    # explicit fixed-field nested parameter record.  Legacy/integrator modules
    # remain byte-identical when neither the operator nor record is present.
    gated_modules: list[dict[str, Any]] = []
    for module in modules:
        next_module = dict(module)
        params = next_module.get("params", {})
        if not isinstance(params, dict):
            raise ValueError("ESCAP7_MALFORMED_PARAMS reason=not_mapping")
        operator = str(
            params.get("live_operator", "bounded_scalar_integrator_v1")
            or "bounded_scalar_integrator_v1"
        )
        if operator == GATED_OPERATOR and str(next_module.get("module_type", "")) != "generic":
            raise ValueError(
                "ESCAP7_GATED_OPERATOR_REQUIRES_GENERIC_MODULE "
                f"module_id={next_module.get('module_id', '')}"
            )
        if operator == GATED_OPERATOR or GATED_RECORD_KEY in params:
            next_module["params"] = normalize_gated_module_params(params)
        gated_modules.append(next_module)
    modules = gated_modules
    connections = normalize_connections(raw.get("connections", None), fallback_flat=canon_genes)
    developmental_program = canonicalize_developmental_program(raw.get("developmental_program", None))
    invalid_policy = str(raw.get("b2_invalid_policy", "clip") or "clip").strip().lower()
    resolved_caps = resolve_b2_caps(raw.get("b2_caps", None))
    modules, connections, _meta = apply_b2_constraints(
        modules,
        connections,
        invalid_policy=invalid_policy,
        b2_caps=resolved_caps,
    )
    return {
        "schema_version": MODULAR_GENOME_SCHEMA_VERSION,
        "interface_schema_version": B2_INTERFACE_SCHEMA_VERSION,
        "b3_locality_schema_version": B3_LOCALITY_SCHEMA_VERSION,
        "genes": canon_genes,
        "modules": modules,
        "connections": connections,
        "developmental_program": developmental_program,
        "b2_invalid_policy": invalid_policy if invalid_policy in {"clip", "reject"} else "clip",
        "b2_caps": dict(resolved_caps),
        "b3_locality_max_connection_distance": canon_float(B3_FROZEN_LOCALITY["max_connection_distance"], field="b3.max_connection_distance"),
        "b3_distance_cost_scale": canon_float(B3_FROZEN_LOCALITY["distance_cost_scale"], field="b3.distance_cost_scale"),
        "b3_distance_cost_max": canon_float(B3_FROZEN_LOCALITY["distance_cost_max"], field="b3.distance_cost_max"),
        "b3_locality_cost_total": canon_float(_meta.get("locality_cost_total", 0.0), field="b3.locality_cost_total"),
        "b3_locality_mean_distance": canon_float(_meta.get("locality_mean_distance", 0.0), field="b3.locality_mean_distance"),
        "b3_locality_max_distance": canon_float(_meta.get("locality_max_distance", 0.0), field="b3.locality_max_distance"),
        "b3_locality_dropped_connections": int(_meta.get("dropped_nonlocal_connections", 0) or 0),
    }


def canonical_bytes_from_genome_dict(raw: dict[str, Any]) -> bytes:
    return canonical_json_bytes(canonical_payload_from_genome_dict(raw))


def canonical_hash_from_genome_dict(raw: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes_from_genome_dict(raw)).hexdigest()
