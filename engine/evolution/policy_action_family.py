from __future__ import annotations

"""Shared reporting-only policy action-family helpers."""

import math
import hashlib
import json
from typing import Any, Iterable

import numpy as np

from env.gridworld_pkg.constants import (
    ACT_ATTACK,
    ACT_GATHER,
    ACT_MOVE_E,
    ACT_MOVE_N,
    ACT_MOVE_S,
    ACT_MOVE_W,
    ACT_REST,
    N_ACTIONS,
)


SCHEMA_VERSION = "structural_policy_action_family_mapping_v1"
FACTORIZED_SHADOW_SCHEMA_VERSION = "factorized_family_shadow_policy_v1"
FAMILY_HEAD_CONTRACT_VERSION = "factorized_family_shadow_family_heads_v1"
WITHIN_FAMILY_HEAD_CONTRACT_VERSION = "factorized_family_shadow_within_family_heads_v1"
FACTOR_RECONSTRUCTION_TOLERANCE = 1e-9

ACTION_ID_TO_NAME: dict[int, str] = {
    int(ACT_MOVE_N): "MOVE_N",
    int(ACT_MOVE_S): "MOVE_S",
    int(ACT_MOVE_E): "MOVE_E",
    int(ACT_MOVE_W): "MOVE_W",
    int(ACT_GATHER): "GATHER",
    int(ACT_REST): "REST",
    int(ACT_ATTACK): "ATTACK",
}

ACTION_ID_TO_FAMILY: dict[int, str] = {
    int(ACT_MOVE_N): "movement",
    int(ACT_MOVE_S): "movement",
    int(ACT_MOVE_E): "movement",
    int(ACT_MOVE_W): "movement",
    int(ACT_GATHER): "gather",
    int(ACT_REST): "rest",
    int(ACT_ATTACK): "attack",
}

FAMILY_TO_ACTION_IDS: dict[str, tuple[int, ...]] = {
    "movement": (int(ACT_MOVE_N), int(ACT_MOVE_S), int(ACT_MOVE_E), int(ACT_MOVE_W)),
    "gather": (int(ACT_GATHER),),
    "rest": (int(ACT_REST),),
    "attack": (int(ACT_ATTACK),),
}

FAMILY_ORDER: tuple[str, ...] = ("movement", "gather", "rest", "attack")
WITHIN_FAMILY_HEAD_IDS: dict[str, str] = {
    "movement": "movement.direction",
    "gather": "gather.singleton",
    "rest": "rest.singleton",
    "attack": "attack.singleton",
}
TARGET_MODES: dict[str, str] = {
    "movement": "direction",
    "gather": "environment_resolved",
    "rest": "none",
    "attack": "environment_resolved",
}


def action_name(action_id: Any) -> str:
    action = _coerce_action(action_id)
    try:
        return ACTION_ID_TO_NAME[action]
    except KeyError as exc:
        raise ValueError(f"unknown action id: {action}") from exc


def action_family(action_id: Any) -> str:
    action = _coerce_action(action_id)
    try:
        return ACTION_ID_TO_FAMILY[action]
    except KeyError as exc:
        raise ValueError(f"unknown action id: {action}") from exc


def action_family_or_unknown(action_id: Any) -> str:
    try:
        return action_family(action_id)
    except Exception:
        return "unknown"


def mapping_contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "action_count": int(N_ACTIONS),
        "actions": {
            str(action): {"name": ACTION_ID_TO_NAME[action], "family": ACTION_ID_TO_FAMILY[action]}
            for action in sorted(ACTION_ID_TO_NAME)
        },
        "families": {family: [ACTION_ID_TO_NAME[action] for action in actions] for family, actions in FAMILY_TO_ACTION_IDS.items()},
        "unknown_action_policy": "raise_value_error_in_strict_helpers",
    }


def family_head_contract() -> dict[str, Any]:
    """Return the stable POLARCH-1 family-head contract."""

    return {
        "schema_version": FAMILY_HEAD_CONTRACT_VERSION,
        "shadow_schema_version": FACTORIZED_SHADOW_SCHEMA_VERSION,
        "family_order": list(FAMILY_ORDER),
        "family_heads": {
            family: {
                "family_head_id": family,
                "action_ids": [int(action) for action in FAMILY_TO_ACTION_IDS[family]],
                "action_names": [ACTION_ID_TO_NAME[int(action)] for action in FAMILY_TO_ACTION_IDS[family]],
                "target_mode": TARGET_MODES[family],
                "authority": False,
                "trainable_parameters": "none",
                "genome_parameters": "none",
            }
            for family in FAMILY_ORDER
        },
    }


def within_family_head_contract() -> dict[str, Any]:
    """Return the stable POLARCH-1 within-family-head contract."""

    return {
        "schema_version": WITHIN_FAMILY_HEAD_CONTRACT_VERSION,
        "shadow_schema_version": FACTORIZED_SHADOW_SCHEMA_VERSION,
        "within_family_heads": {
            family: {
                "within_family_head_id": WITHIN_FAMILY_HEAD_IDS[family],
                "family_head_id": family,
                "action_ids": [int(action) for action in FAMILY_TO_ACTION_IDS[family]],
                "action_names": [ACTION_ID_TO_NAME[int(action)] for action in FAMILY_TO_ACTION_IDS[family]],
                "conditional_probability_contract": (
                    "softmax over valid primitive logits in family"
                    if len(FAMILY_TO_ACTION_IDS[family]) > 1
                    else "singleton conditional probability is 1 when valid"
                ),
                "target_mode": TARGET_MODES[family],
                "authority": False,
                "parameter_block": "absent",
                "structural_link_block": "absent",
            }
            for family in FAMILY_ORDER
        },
    }


def factorized_family_shadow_payload(
    logits: Iterable[Any],
    *,
    valid_actions: Iterable[Any] | None = None,
    chosen_action_id: Any | None = None,
    tolerance: float = FACTOR_RECONSTRUCTION_TOLERANCE,
) -> dict[str, Any]:
    """Factorize masked primitive logits into family and conditional heads.

    The helper is intentionally pure: it reads no global mutable state, performs
    no random draws, and returns deterministic JSON-compatible values. Invalid
    logits are represented as ``None`` in emitted payloads so downstream JSON
    never contains NaN or Infinity.
    """

    scores = np.asarray(list(logits), dtype=np.float64).reshape((-1,))
    if int(scores.size) != int(N_ACTIONS):
        raise ValueError(f"expected {int(N_ACTIONS)} logits, got {int(scores.size)}")
    valid = set(range(int(N_ACTIONS))) if valid_actions is None else {_coerce_action(v) for v in valid_actions}
    unknown = sorted(v for v in valid if v not in ACTION_ID_TO_NAME)
    if unknown:
        raise ValueError(f"unknown valid action ids: {unknown}")
    valid = {int(v) for v in valid if 0 <= int(v) < int(N_ACTIONS)}
    if not valid:
        raise ValueError("factorized family shadow requires at least one valid primitive action")

    masked = np.full((int(N_ACTIONS),), -math.inf, dtype=np.float64)
    for action in sorted(valid):
        value = float(scores[int(action)])
        if not math.isfinite(value):
            raise ValueError(f"non-finite valid primitive logit: action={action}")
        masked[int(action)] = value

    primitive_probs = _softmax_masked(masked)
    family_logits_raw: dict[str, float] = {}
    family_validity: dict[str, bool] = {}
    family_logits_for_softmax = np.full((len(FAMILY_ORDER),), -math.inf, dtype=np.float64)
    for idx, family in enumerate(FAMILY_ORDER):
        family_scores = [float(masked[action]) for action in FAMILY_TO_ACTION_IDS[family] if np.isfinite(masked[action])]
        family_valid = bool(family_scores)
        family_validity[family] = family_valid
        if family_valid:
            family_logit = _logsumexp(family_scores)
            family_logits_raw[family] = family_logit
            family_logits_for_softmax[idx] = family_logit
        else:
            family_logits_raw[family] = -math.inf

    family_prob_arr = _softmax_masked(family_logits_for_softmax)
    family_probabilities = {family: float(family_prob_arr[idx]) for idx, family in enumerate(FAMILY_ORDER)}

    conditional_by_family: dict[str, dict[str, float]] = {}
    movement_direction_probabilities: dict[str, float] = {}
    reconstructed = np.zeros((int(N_ACTIONS),), dtype=np.float64)
    for family in FAMILY_ORDER:
        action_ids = FAMILY_TO_ACTION_IDS[family]
        family_masked = np.full((len(action_ids),), -math.inf, dtype=np.float64)
        for idx, action in enumerate(action_ids):
            if int(action) in valid:
                family_masked[idx] = float(masked[int(action)])
        conditional = _softmax_masked(family_masked)
        conditional_payload: dict[str, float] = {}
        for idx, action in enumerate(action_ids):
            prob = float(conditional[idx]) if int(action) in valid else 0.0
            conditional_payload[ACTION_ID_TO_NAME[int(action)]] = prob
            reconstructed[int(action)] = float(family_probabilities[family]) * prob
            if family == "movement":
                movement_direction_probabilities[ACTION_ID_TO_NAME[int(action)]] = prob
        conditional_by_family[family] = conditional_payload

    error = np.abs(reconstructed - primitive_probs)
    primitive_entropy = _entropy(primitive_probs.tolist())
    family_entropy = _entropy([family_probabilities[family] for family in FAMILY_ORDER])
    movement_entropy = _entropy(list(movement_direction_probabilities.values()))
    entropy_residual = float(primitive_entropy - (family_entropy + family_probabilities["movement"] * movement_entropy))
    ranks = sorted(family_probabilities.items(), key=lambda item: (-float(item[1]), item[0]))
    chosen_action = _optional_action(chosen_action_id)
    chosen_family = action_family_or_unknown(chosen_action) if chosen_action is not None else ""

    payload = {
        "schema_version": FACTORIZED_SHADOW_SCHEMA_VERSION,
        "shadow_authority": False,
        "genome_parameters": "none",
        "trainable_parameters": "none",
        "random_draws": 0,
        "family_order": list(FAMILY_ORDER),
        "family_head_ids": {family: family for family in FAMILY_ORDER},
        "within_family_head_ids": dict(WITHIN_FAMILY_HEAD_IDS),
        "target_modes": dict(TARGET_MODES),
        "valid_action_mask": [bool(i in valid) for i in range(int(N_ACTIONS))],
        "valid_actions": [int(action) for action in sorted(valid)],
        "masked_primitive_logits": _jsonable_float_list(masked),
        "primitive_probabilities": _jsonable_float_list(primitive_probs),
        "family_validity": {family: bool(family_validity[family]) for family in FAMILY_ORDER},
        "family_logits": {family: _jsonable_float(family_logits_raw[family]) for family in FAMILY_ORDER},
        "family_probabilities": {family: _jsonable_float(family_probabilities[family]) for family in FAMILY_ORDER},
        "dominant_family": ranks[0][0] if ranks else "",
        "family_top_two_margin": float(ranks[0][1] - ranks[1][1]) if len(ranks) >= 2 else 0.0,
        "family_entropy": float(family_entropy),
        "within_family_conditional_probabilities": conditional_by_family,
        "movement_valid_direction_mask": {
            ACTION_ID_TO_NAME[int(action)]: bool(int(action) in valid)
            for action in FAMILY_TO_ACTION_IDS["movement"]
        },
        "movement_conditional_entropy": float(movement_entropy),
        "selected_movement_direction": ACTION_ID_TO_NAME[int(chosen_action)] if chosen_family == "movement" and chosen_action is not None else "",
        "singleton_conditional_probabilities": {
            family: (1.0 if family_validity[family] else 0.0)
            for family in ("gather", "rest", "attack")
        },
        "chosen_action_id": int(chosen_action) if chosen_action is not None else None,
        "chosen_family": chosen_family,
        "reconstructed_primitive_probabilities": _jsonable_float_list(reconstructed),
        "max_abs_reconstruction_error": float(np.max(error)),
        "mean_abs_reconstruction_error": float(np.mean(error)),
        "primitive_probability_sum_residual": float(np.sum(primitive_probs) - 1.0),
        "family_probability_sum_residual": float(sum(family_probabilities.values()) - 1.0),
        "conditional_probability_sum_residuals": {
            family: float(sum(conditional_by_family[family].values()) - (1.0 if family_validity[family] else 0.0))
            for family in FAMILY_ORDER
        },
        "reconstructed_probability_sum_residual": float(np.sum(reconstructed) - 1.0),
        "primitive_entropy": float(primitive_entropy),
        "entropy_decomposition_residual": float(entropy_residual),
        "factorization_evaluable": bool(float(np.max(error)) <= float(tolerance)),
        "not_evaluable_reason": "" if float(np.max(error)) <= float(tolerance) else "reconstruction_error_exceeds_tolerance",
        "tolerance": float(tolerance),
    }
    payload["shadow_distribution_id"] = _shadow_distribution_id(payload)
    return payload


def family_probability_payload(logits: Iterable[Any], *, valid_actions: Iterable[Any] | None = None) -> dict[str, Any]:
    scores = np.asarray(list(logits), dtype=np.float64).reshape((-1,))
    if int(scores.size) != int(N_ACTIONS):
        raise ValueError(f"expected {int(N_ACTIONS)} logits, got {int(scores.size)}")
    valid = set(range(int(N_ACTIONS))) if valid_actions is None else {_coerce_action(v) for v in valid_actions}
    unknown = sorted(v for v in valid if v not in ACTION_ID_TO_NAME)
    if unknown:
        raise ValueError(f"unknown valid action ids: {unknown}")
    masked = np.full((int(N_ACTIONS),), -math.inf, dtype=np.float64)
    for action in sorted(valid):
        masked[action] = scores[action]
    probs = _softmax_masked(masked)
    family_mass = {
        family: float(sum(float(probs[action]) for action in actions if action < probs.size))
        for family, actions in FAMILY_TO_ACTION_IDS.items()
    }
    primitive_top = int(np.argmax(probs)) if float(np.sum(probs)) > 0.0 else -1
    family_ranks = sorted(family_mass.items(), key=lambda item: (-float(item[1]), item[0]))
    movement_probs = [float(probs[action]) for action in FAMILY_TO_ACTION_IDS["movement"]]
    movement_best = max(movement_probs) if movement_probs else 0.0
    gather_prob = float(family_mass["gather"])
    primitive_entropy = _entropy([float(v) for v in probs.tolist()])
    family_entropy = _entropy([float(family_mass[family]) for family in FAMILY_ORDER])
    return {
        "schema_version": "structural_policy_action_family_probability_v1",
        "primitive_probabilities": [float(v) for v in probs.tolist()],
        "family_probability_mass": family_mass,
        "primitive_entropy": float(primitive_entropy),
        "family_entropy": float(family_entropy),
        "primitive_top_action": primitive_top,
        "primitive_top_family": action_family_or_unknown(primitive_top) if primitive_top >= 0 else "",
        "dominant_family": family_ranks[0][0] if family_ranks else "",
        "family_top_two_margin": float(family_ranks[0][1] - family_ranks[1][1]) if len(family_ranks) >= 2 else 0.0,
        "movement_best_direction_probability": float(movement_best),
        "movement_family_mass_minus_gather_probability": float(family_mass["movement"] - gather_prob),
        "movement_dilution_case": bool(family_mass["movement"] > gather_prob and movement_best < gather_prob),
    }


def _coerce_action(value: Any) -> int:
    try:
        action = int(value)
    except Exception as exc:
        raise ValueError(f"unknown action id: {value}") from exc
    return action


def _optional_action(value: Any) -> int | None:
    if value is None:
        return None
    try:
        action = int(value)
    except Exception:
        return None
    if action not in ACTION_ID_TO_NAME:
        return None
    return int(action)


def _softmax_masked(scores: np.ndarray) -> np.ndarray:
    finite = np.isfinite(scores)
    if not bool(np.any(finite)):
        return np.zeros_like(scores, dtype=np.float64)
    shifted = scores.copy()
    max_val = float(np.max(shifted[finite]))
    exp = np.zeros_like(shifted, dtype=np.float64)
    exp[finite] = np.exp(shifted[finite] - max_val)
    denom = float(np.sum(exp))
    return exp / denom if denom > 0.0 else np.zeros_like(scores, dtype=np.float64)


def _logsumexp(values: Iterable[Any]) -> float:
    arr = np.asarray(list(values), dtype=np.float64).reshape((-1,))
    finite = np.isfinite(arr)
    if not bool(np.any(finite)):
        return -math.inf
    max_val = float(np.max(arr[finite]))
    return float(max_val + math.log(float(np.sum(np.exp(arr[finite] - max_val)))))


def _entropy(values: Iterable[Any]) -> float:
    total = 0.0
    for value in values:
        try:
            p = float(value)
        except Exception:
            continue
        if p > 0.0 and math.isfinite(p):
            total -= p * math.log(p)
    return float(total)


def _jsonable_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return float(out)


def _jsonable_float_list(values: Iterable[Any]) -> list[float | None]:
    return [_jsonable_float(value) for value in list(values)]


def _shadow_distribution_id(payload: dict[str, Any]) -> str:
    material = {
        "schema_version": FACTORIZED_SHADOW_SCHEMA_VERSION,
        "valid_action_mask": payload.get("valid_action_mask", []),
        "masked_primitive_logits": payload.get("masked_primitive_logits", []),
        "family_head_ids": payload.get("family_head_ids", {}),
        "within_family_head_ids": payload.get("within_family_head_ids", {}),
        "target_modes": payload.get("target_modes", {}),
    }
    raw = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
