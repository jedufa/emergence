from __future__ import annotations

"""Dormant centered family-head residual genome block helpers.

These helpers are pure and non-authoritative for action selection.  They own
canonicalization, hashes, deterministic recombination, and the selected
POLARCH-3 mutation fixture for the dormant genome representation.
"""

import hashlib
import json
import math
from typing import Any, Iterable


FAMILY_HEAD_RESIDUAL_SCHEMA_VERSION = "centered_family_head_residuals_v1"
FAMILY_HEAD_RESIDUAL_FIELD = "family_head_residuals"
FAMILY_HEAD_RESIDUAL_FAMILY_ORDER: tuple[str, ...] = ("movement", "gather", "rest", "attack")
FAMILY_HEAD_RESIDUAL_COMPONENT_BOUND = 1.0
FAMILY_HEAD_RESIDUAL_TOTAL_L1_CAP = 2.0
FAMILY_HEAD_RESIDUAL_MUTATION_OPERATOR_ID = "family_head_pairwise_residual_transfer_v1"
FAMILY_HEAD_RESIDUAL_RECOMBINATION_OPERATOR_ID = "family_head_midpoint_center_project_v1"
FAMILY_HEAD_RESIDUAL_TOLERANCE = 1e-12


def _canon_float(value: Any, *, ndigits: int = 12) -> float:
    value_f = float(value)
    if not math.isfinite(value_f):
        raise ValueError("family-head residual values must be finite")
    out = round(value_f, int(ndigits))
    return 0.0 if out == -0.0 else float(out)


def _stable_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")


def stable_digest(obj: Any) -> str:
    return hashlib.sha256(_stable_bytes(obj)).hexdigest()


def _vector_from_any(raw: Any) -> list[float]:
    if raw is None:
        return [0.0 for _ in FAMILY_HEAD_RESIDUAL_FAMILY_ORDER]
    if isinstance(raw, dict):
        if "stored_residual_raw" in raw:
            return _vector_from_any(raw.get("stored_residual_raw"))
        if "effective_residuals" in raw:
            return _vector_from_any(raw.get("effective_residuals"))
        return [_canon_float(raw.get(family, 0.0)) for family in FAMILY_HEAD_RESIDUAL_FAMILY_ORDER]
    seq = list(raw)
    if len(seq) != len(FAMILY_HEAD_RESIDUAL_FAMILY_ORDER):
        raise ValueError(f"expected {len(FAMILY_HEAD_RESIDUAL_FAMILY_ORDER)} family-head residuals")
    return [_canon_float(value) for value in seq]


def family_residual_map(raw: Any = None) -> dict[str, float]:
    vec = _vector_from_any(raw)
    return {family: float(vec[idx]) for idx, family in enumerate(FAMILY_HEAD_RESIDUAL_FAMILY_ORDER)}


def center_project_family_residuals(
    raw: Any = None,
    *,
    component_bound: float = FAMILY_HEAD_RESIDUAL_COMPONENT_BOUND,
    total_l1_cap: float = FAMILY_HEAD_RESIDUAL_TOTAL_L1_CAP,
) -> dict[str, Any]:
    component = abs(float(component_bound))
    total_cap = abs(float(total_l1_cap))
    vec = _vector_from_any(raw)
    mean = sum(vec) / float(len(vec))
    centered = [_canon_float(v - mean) for v in vec]
    if centered:
        centered[-1] = _canon_float(-sum(centered[:-1]))

    component_cap_touched = any(abs(v) > component + FAMILY_HEAD_RESIDUAL_TOLERANCE for v in centered)
    clamped = [max(-component, min(component, v)) for v in centered]
    mean2 = sum(clamped) / float(len(clamped))
    recentered = [_canon_float(v - mean2) for v in clamped]
    if recentered:
        recentered[-1] = _canon_float(-sum(recentered[:-1]))

    l1_before = sum(abs(v) for v in recentered)
    total_cap_touched = l1_before > total_cap + FAMILY_HEAD_RESIDUAL_TOLERANCE if total_cap > 0.0 else l1_before > FAMILY_HEAD_RESIDUAL_TOLERANCE
    if total_cap <= 0.0:
        effective = [0.0 for _ in recentered]
    elif total_cap_touched:
        scale = total_cap / max(l1_before, FAMILY_HEAD_RESIDUAL_TOLERANCE)
        effective = [_canon_float(v * scale) for v in recentered]
        if effective:
            effective[-1] = _canon_float(-sum(effective[:-1]))
    else:
        effective = list(recentered)

    changed = any(abs(float(vec[idx]) - float(effective[idx])) > FAMILY_HEAD_RESIDUAL_TOLERANCE for idx in range(len(effective)))
    return {
        "schema_version": "family_head_residual_projection_v1",
        "family_order": list(FAMILY_HEAD_RESIDUAL_FAMILY_ORDER),
        "effective_residuals": {family: float(effective[idx]) for idx, family in enumerate(FAMILY_HEAD_RESIDUAL_FAMILY_ORDER)},
        "component_bound": float(component),
        "total_l1_cap": float(total_cap),
        "sum_residual": _canon_float(sum(effective)),
        "l1_magnitude": _canon_float(sum(abs(v) for v in effective)),
        "component_cap_touched": bool(component_cap_touched),
        "total_cap_touched": bool(total_cap_touched),
        "cap_touch": bool(component_cap_touched or total_cap_touched or changed),
        "cap_binding": bool(component_cap_touched or total_cap_touched),
        "component_cap_binding": bool(component_cap_touched),
        "total_cap_binding": bool(total_cap_touched),
        "canonicalization_changed": bool(changed),
        "neutral": bool(all(abs(v) <= FAMILY_HEAD_RESIDUAL_TOLERANCE for v in effective)),
    }


def canonical_family_head_residual_block(raw: Any = None) -> dict[str, Any]:
    projection = center_project_family_residuals(raw)
    effective = dict(projection["effective_residuals"])
    block = {
        "schema_version": FAMILY_HEAD_RESIDUAL_SCHEMA_VERSION,
        "family_order": list(FAMILY_HEAD_RESIDUAL_FAMILY_ORDER),
        "stored_residual_raw": [float(effective[family]) for family in FAMILY_HEAD_RESIDUAL_FAMILY_ORDER],
        "component_bound": FAMILY_HEAD_RESIDUAL_COMPONENT_BOUND,
        "total_l1_cap": FAMILY_HEAD_RESIDUAL_TOTAL_L1_CAP,
    }
    return block


def canonicalize_optional_family_head_residual_block(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        schema = str(raw.get("schema_version", FAMILY_HEAD_RESIDUAL_SCHEMA_VERSION) or FAMILY_HEAD_RESIDUAL_SCHEMA_VERSION)
        if schema != FAMILY_HEAD_RESIDUAL_SCHEMA_VERSION:
            raise ValueError(f"unsupported family-head residual schema_version={schema}")
        order = raw.get("family_order", list(FAMILY_HEAD_RESIDUAL_FAMILY_ORDER))
        if list(order) != list(FAMILY_HEAD_RESIDUAL_FAMILY_ORDER):
            raise ValueError("family-head residual family_order mismatch")
    return canonical_family_head_residual_block(raw)


def legacy_zero_family_head_residual_block() -> dict[str, Any]:
    return canonical_family_head_residual_block(None)


def family_head_residual_effective_map(raw: Any = None) -> dict[str, float]:
    block = canonical_family_head_residual_block(raw)
    return family_residual_map(block["stored_residual_raw"])


def family_head_residual_semantic_digest(raw: Any = None) -> str:
    block = canonical_family_head_residual_block(raw)
    return stable_digest(
        {
            "schema_version": FAMILY_HEAD_RESIDUAL_SCHEMA_VERSION,
            "family_order": list(FAMILY_HEAD_RESIDUAL_FAMILY_ORDER),
            "effective_residuals": block["stored_residual_raw"],
        }
    )


def family_head_residual_canonical_digest(raw: Any = None) -> str:
    return stable_digest(canonical_family_head_residual_block(raw))


def recombine_family_head_residual_blocks(parent_a: Any = None, parent_b: Any = None) -> dict[str, Any]:
    a = family_residual_map(parent_a)
    b = family_residual_map(parent_b)
    midpoint = {family: 0.5 * (float(a[family]) + float(b[family])) for family in FAMILY_HEAD_RESIDUAL_FAMILY_ORDER}
    block = canonical_family_head_residual_block(midpoint)
    return {
        "schema_version": "family_head_residual_recombination_v1",
        "operator_id": FAMILY_HEAD_RESIDUAL_RECOMBINATION_OPERATOR_ID,
        "parent_a": canonical_family_head_residual_block(parent_a),
        "parent_b": canonical_family_head_residual_block(parent_b),
        "child": block,
        "draw_budget": {"mutation.py": 0},
        "canonical_digest": family_head_residual_canonical_digest(block),
        "semantic_digest": family_head_residual_semantic_digest(block),
    }


def _ordered_family_pair(index: int) -> tuple[str, str]:
    pairs = [
        (a, b)
        for a in FAMILY_HEAD_RESIDUAL_FAMILY_ORDER
        for b in FAMILY_HEAD_RESIDUAL_FAMILY_ORDER
        if a != b
    ]
    return pairs[int(index) % len(pairs)]


def mutate_family_head_residual_block(
    raw: Any,
    rng: Any,
    *,
    step: float = 0.1,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply the selected two-draw pairwise reciprocal mutation."""

    before = canonical_family_head_residual_block(raw)
    before_effective = family_residual_map(before["stored_residual_raw"])
    pair_raw = rng.random()
    delta_raw = rng.random()
    pair_index = min(11, int(float(pair_raw) * 12.0))
    positive_family, negative_family = _ordered_family_pair(pair_index)
    signed_delta = _canon_float((1.0 if float(delta_raw) >= 0.5 else -1.0) * abs(float(step)))
    candidate = dict(before_effective)
    candidate[positive_family] = _canon_float(candidate[positive_family] + signed_delta)
    candidate[negative_family] = _canon_float(candidate[negative_family] - signed_delta)
    projection = center_project_family_residuals(candidate)
    after = canonical_family_head_residual_block(projection["effective_residuals"])
    after_effective = family_residual_map(after["stored_residual_raw"])
    semantic_change = any(
        abs(after_effective[family] - before_effective[family]) > FAMILY_HEAD_RESIDUAL_TOLERANCE
        for family in FAMILY_HEAD_RESIDUAL_FAMILY_ORDER
    )
    event = {
        "schema_version": "family_head_residual_mutation_event_v1",
        "operator_id": FAMILY_HEAD_RESIDUAL_MUTATION_OPERATOR_ID,
        "selected_pair_index": int(pair_index),
        "positive_family": positive_family,
        "negative_family": negative_family,
        "sampled_delta": float(signed_delta),
        "parent_pre_state": before,
        "candidate_raw_state": canonical_family_head_residual_block(candidate),
        "post_project_effective_state": after,
        "cap_touch": bool(projection["cap_touch"]),
        "cap_binding": bool(projection["cap_binding"]),
        "no_op": bool(not semantic_change),
        "historical_change": bool(before != after),
        "semantic_change": bool(semantic_change),
        "canonical_digest": family_head_residual_canonical_digest(after),
        "semantic_digest": family_head_residual_semantic_digest(after),
        "draw_budget": {"mutation.py": 2},
    }
    return after, event
