from __future__ import annotations

"""ESCAP-6 fixed-budget hereditary developmental mutation.

The caller supplies exactly two accounted draws.  This module owns no RNG and
never mutates stage boundaries, identities, topology, or routing.
"""

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

from agents.structural_identity_v1 import module_is_native_homology
from evolution.structural_development_v1 import (
    DEVELOPMENTAL_EXPRESSION_BLOCK_KEY,
    DEVELOPMENTAL_MODULE_SCHEMA_VERSION,
    DEVELOPMENTAL_STAGE_IDS,
    DEVELOPMENTAL_STAGE_PROFILE_VERSION,
    DEVELOPMENT_OPERATOR_DRAWS_PER_ATTEMPT,
    DEVELOPMENT_OPERATOR_PROFILE,
    MAX_DEPENDENCY_EDGES_PER_AGENT,
    MAX_PREREQUISITES_PER_MODULE,
    neutral_developmental_expression_record,
    normalize_agent_developmental_modules,
    normalize_developmental_expression_record,
    validate_agent_developmental_graph,
)


DEVELOPMENT_MUTATION_OPERATOR = "developmental_expression_mutation"
DEVELOPMENT_OPERATOR_DRAWS = DEVELOPMENT_OPERATOR_DRAWS_PER_ATTEMPT
DEVELOPMENT_MUTATION_RESULT_SCHEMA_VERSION = "escap6_development_mutation_result_v1"
DEVELOPMENT_OPERATION_ORDER: tuple[str, ...] = (
    "add_neutral_record",
    "mutate_stage_gain",
    "add_or_remove_dependency",
    "toggle_latch",
)


@dataclass(frozen=True)
class DevelopmentMutationResult:
    modules: list[dict[str, Any]]
    telemetry: dict[str, Any]


DevelopmentalMutationResult = DevelopmentMutationResult


def _finite_draw(value: Any) -> float | None:
    try:
        draw = float(value)
    except Exception:
        return None
    if not math.isfinite(draw) or draw < 0.0 or draw >= 1.0:
        return None
    return draw


def _select(draw: float, count: int) -> tuple[int, float]:
    if count <= 0:
        return 0, 0.0
    scaled = float(draw) * float(count)
    index = min(count - 1, int(scaled))
    return int(index), float(scaled - index)


def _sort_key(module: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(module.get("homology_id", "")),
        str(module.get("module_instance_id", "")),
        str(module.get("module_id", "")),
    )


def _record(module: Mapping[str, Any]) -> dict[str, Any] | None:
    params = module.get("params", {})
    if not isinstance(params, Mapping):
        raise ValueError("module params not mapping")
    return normalize_developmental_expression_record(
        params.get(DEVELOPMENTAL_EXPRESSION_BLOCK_KEY)
        if DEVELOPMENTAL_EXPRESSION_BLOCK_KEY in params
        else None,
        module_homology_id=str(module.get("homology_id", "")),
    )


def _replace_record(module: Mapping[str, Any], record: Mapping[str, Any]) -> dict[str, Any]:
    out = deepcopy(dict(module))
    params = out.get("params", {})
    if not isinstance(params, dict):
        raise ValueError("module params not mapping")
    next_params = deepcopy(params)
    next_params[DEVELOPMENTAL_EXPRESSION_BLOCK_KEY] = normalize_developmental_expression_record(
        record,
        module_homology_id=str(out.get("homology_id", "")),
    )
    out["params"] = next_params
    return out


def _edge_count(modules: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    for module in modules:
        record = _record(module)
        if record is not None:
            count += len(record["prerequisite_homology_ids"])
    return int(count)


def mutate_developmental_expression(
    modules: list[dict[str, Any]],
    draw0: float,
    draw1: float,
    *,
    enabled: bool,
) -> DevelopmentMutationResult:
    raw_parent = deepcopy(list(modules or []))
    telemetry: dict[str, Any] = {
        "schema_version": DEVELOPMENT_MUTATION_RESULT_SCHEMA_VERSION,
        "operator": DEVELOPMENT_MUTATION_OPERATOR,
        "profile": DEVELOPMENT_OPERATOR_PROFILE,
        "module_schema_version": DEVELOPMENTAL_MODULE_SCHEMA_VERSION,
        "stage_profile_version": DEVELOPMENTAL_STAGE_PROFILE_VERSION,
        "stage_ids": list(DEVELOPMENTAL_STAGE_IDS),
        "operation_order": list(DEVELOPMENT_OPERATION_ORDER),
        "operation_bins": {
            "add_neutral_record": [0.0, 0.25],
            "mutate_stage_gain": [0.25, 0.5],
            "add_or_remove_dependency": [0.5, 0.75],
            "toggle_latch": [0.75, 1.0],
        },
        "draw_budget": {
            "declared_supplied_draws": DEVELOPMENT_OPERATOR_DRAWS_PER_ATTEMPT,
            "internal_rng_draws": 0,
        },
        "draw0": None,
        "draw1": None,
        "enabled": bool(enabled),
        "status": "rejected",
        "reason": "",
        "operation": "",
        "selected_homology_id": "",
        "selected_module_instance_id": "",
        "selected_stage_id": "",
        "dependency_target_homology_id": "",
        "gain_before": None,
        "gain_after": None,
        "latch_before": None,
        "latch_after": None,
        "records_before": 0,
        "records_after": 0,
        "edges_before": 0,
        "edges_after": 0,
        "parent_preserved": True,
    }
    try:
        parent = sorted(normalize_agent_developmental_modules(raw_parent), key=_sort_key)
    except Exception as exc:
        telemetry["reason"] = f"invalid_parent:{type(exc).__name__}"
        return DevelopmentMutationResult(raw_parent, telemetry)
    telemetry["records_before"] = sum(1 for module in parent if _record(module) is not None)
    telemetry["records_after"] = telemetry["records_before"]
    telemetry["edges_before"] = _edge_count(parent)
    telemetry["edges_after"] = telemetry["edges_before"]

    d0 = _finite_draw(draw0)
    d1 = _finite_draw(draw1)
    telemetry["draw0"] = d0
    telemetry["draw1"] = d1
    if d0 is None or d1 is None:
        telemetry["reason"] = "draw_out_of_range"
        return DevelopmentMutationResult(parent, telemetry)
    operation_index, operation_fraction = _select(d0, len(DEVELOPMENT_OPERATION_ORDER))
    operation = DEVELOPMENT_OPERATION_ORDER[operation_index]
    telemetry["operation"] = operation
    if not bool(enabled):
        telemetry["status"] = "disabled"
        telemetry["reason"] = "operator_disabled"
        return DevelopmentMutationResult(parent, telemetry)
    try:
        validate_agent_developmental_graph(parent)
    except Exception as exc:
        telemetry["reason"] = f"invalid_parent:{type(exc).__name__}:{exc}"
        return DevelopmentMutationResult(parent, telemetry)

    native = [
        module
        for module in parent
        if module_is_native_homology(module)
        and str(module.get("module_type", "")) == "generic"
    ]
    if not native:
        telemetry["status"] = "ineligible"
        telemetry["reason"] = "no_native_homology_modules"
        return DevelopmentMutationResult(parent, telemetry)
    candidate = deepcopy(parent)

    def _selected(rows: list[dict[str, Any]], draw: float = d1) -> tuple[int, dict[str, Any], float]:
        ordinal, fraction = _select(draw, len(rows))
        chosen = rows[ordinal]
        parent_index = next(
            index
            for index, module in enumerate(candidate)
            if str(module.get("module_instance_id", "")) == str(chosen.get("module_instance_id", ""))
        )
        telemetry["selected_homology_id"] = str(chosen.get("homology_id", ""))
        telemetry["selected_module_instance_id"] = str(chosen.get("module_instance_id", ""))
        return parent_index, chosen, fraction

    if operation == "add_neutral_record":
        eligible = [module for module in native if _record(module) is None]
        if not eligible:
            telemetry["status"] = "ineligible"
            telemetry["reason"] = "all_native_modules_have_records"
            return DevelopmentMutationResult(parent, telemetry)
        index, module, _ = _selected(eligible)
        candidate[index] = _replace_record(module, neutral_developmental_expression_record())

    elif operation == "mutate_stage_gain":
        eligible = [module for module in native if _record(module) is not None]
        if not eligible:
            telemetry["status"] = "ineligible"
            telemetry["reason"] = "no_developmental_record"
            return DevelopmentMutationResult(parent, telemetry)
        # Draw 2 deterministically selects module, stage, and the bounded gain
        # through successive canonical subdivisions of the same unit interval.
        module_ordinal, module_fraction = _select(d1, len(eligible))
        module = eligible[module_ordinal]
        stage_index, gain_fraction = _select(module_fraction, len(DEVELOPMENTAL_STAGE_IDS))
        index, _, _ = _selected([module], draw=0.0)
        record = _record(module)
        assert record is not None
        before = float(record["stage_gains"][stage_index])
        after = float(gain_fraction)
        telemetry["selected_stage_id"] = DEVELOPMENTAL_STAGE_IDS[stage_index]
        telemetry["gain_before"] = before
        telemetry["gain_after"] = after
        if after == before:
            telemetry["status"] = "noop"
            telemetry["reason"] = "stage_gain_unchanged"
            return DevelopmentMutationResult(parent, telemetry)
        record["stage_gains"][stage_index] = after
        candidate[index] = _replace_record(module, record)

    elif operation == "add_or_remove_dependency":
        owners = [module for module in native if _record(module) is not None]
        if not owners:
            telemetry["status"] = "ineligible"
            telemetry["reason"] = "no_developmental_record"
            return DevelopmentMutationResult(parent, telemetry)
        proposals: list[tuple[str, dict[str, Any], str]] = []
        native_targets = sorted(native, key=_sort_key)
        for owner in sorted(owners, key=_sort_key):
            record = _record(owner)
            assert record is not None
            owner_h = str(owner.get("homology_id", ""))
            existing = set(str(value) for value in record["prerequisite_homology_ids"])
            for target in sorted(existing):
                proposals.append(("remove", owner, target))
            for target_module in native_targets:
                target = str(target_module.get("homology_id", ""))
                if target != owner_h and target not in existing:
                    proposals.append(("add", owner, target))
        if not proposals:
            telemetry["status"] = "ineligible"
            telemetry["reason"] = "no_dependency_proposal"
            return DevelopmentMutationResult(parent, telemetry)
        proposal_index, _ = _select(d1, len(proposals))
        detail, owner, target = proposals[proposal_index]
        index, _, _ = _selected([owner], draw=0.0)
        telemetry["dependency_detail"] = detail
        telemetry["dependency_target_homology_id"] = target
        record = _record(owner)
        assert record is not None
        deps = list(record["prerequisite_homology_ids"])
        if detail == "remove":
            deps.remove(target)
        else:
            if len(deps) >= MAX_PREREQUISITES_PER_MODULE:
                telemetry["status"] = "cap_blocked"
                telemetry["reason"] = "module_prerequisite_cap"
                return DevelopmentMutationResult(parent, telemetry)
            if int(telemetry["edges_before"]) >= MAX_DEPENDENCY_EDGES_PER_AGENT:
                telemetry["status"] = "cap_blocked"
                telemetry["reason"] = "agent_dependency_edge_cap"
                return DevelopmentMutationResult(parent, telemetry)
            deps.append(target)
        record["prerequisite_homology_ids"] = sorted(deps)
        candidate[index] = _replace_record(owner, record)

    else:
        eligible = [module for module in native if _record(module) is not None]
        if not eligible:
            telemetry["status"] = "ineligible"
            telemetry["reason"] = "no_developmental_record"
            return DevelopmentMutationResult(parent, telemetry)
        index, module, _ = _selected(eligible)
        record = _record(module)
        assert record is not None
        before = bool(record["latch_dependency_readiness"])
        record["latch_dependency_readiness"] = not before
        telemetry["latch_before"] = before
        telemetry["latch_after"] = not before
        candidate[index] = _replace_record(module, record)

    try:
        normalized = sorted(normalize_agent_developmental_modules(candidate), key=_sort_key)
    except Exception as exc:
        name = type(exc).__name__
        message = str(exc)
        telemetry["status"] = (
            "cycle_blocked" if "CYCLE" in message else
            "cap_blocked" if "CAP" in message else
            "rejected"
        )
        telemetry["reason"] = f"proposal_invalid:{name}:{message}"
        return DevelopmentMutationResult(parent, telemetry)
    telemetry["status"] = "accepted"
    telemetry["reason"] = "proposal_applied"
    telemetry["records_after"] = sum(1 for module in normalized if _record(module) is not None)
    telemetry["edges_after"] = _edge_count(normalized)
    telemetry["parent_preserved"] = False
    return DevelopmentMutationResult(normalized, telemetry)
