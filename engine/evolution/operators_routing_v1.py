from __future__ import annotations

"""Fixed-budget hereditary observation-binding mutation for ESCAP-4.

The operator owns no RNG.  Its caller supplies exactly two already-accounted
draws: the first selects a generic module and supplies a deterministic local
selector, while the second selects one of four frozen operations and supplies
that operation's local magnitude.  Invalid outcomes return the canonical
parent unchanged.
"""

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any, Iterable

from evolution.structural_observation_routing_v1 import (
    MAX_BINDINGS_PER_AGENT,
    MAX_BINDINGS_PER_MODULE,
    ROUTING_BLOCK_KEY,
    ROUTING_REGISTRY_VERSION,
    ROUTING_SCHEMA_VERSION,
    normalize_agent_routing_modules,
    normalize_routing_block,
)


ROUTING_MUTATION_OPERATOR = "observation_binding_mutation"
ROUTING_MUTATION_PROFILE = "escap4_routing_binding_operator_v1"
ROUTING_OPERATOR_DRAWS = 2
ROUTING_MUTATION_RESULT_SCHEMA_VERSION = "escap4_routing_binding_mutation_result_v1"

_TARGET_SLOT = "in"
_OPERATION_ORDER: tuple[str, ...] = ("add", "remove", "gain", "retarget")


@dataclass(frozen=True)
class RoutingMutationResult:
    modules: list[dict[str, Any]]
    telemetry: dict[str, Any]


def _module_sort_key(module: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(module.get("module_id", "")),
        str(module.get("module_instance_id", "")),
        str(module.get("module_type", "")),
    )


def _finite_draw(value: Any) -> float | None:
    try:
        draw = float(value)
    except Exception:
        return None
    if not math.isfinite(draw) or draw < 0.0 or draw >= 1.0:
        return None
    return draw


def _binding_sort_key(binding: dict[str, Any]) -> tuple[str, str]:
    return (
        str(binding.get("semantic_id", "")),
        str(binding.get("target_slot", "")),
    )


def _routing_block(module: dict[str, Any]) -> dict[str, Any]:
    params = module.get("params", {})
    if not isinstance(params, dict):
        raise ValueError("module params must be a mapping")
    return normalize_routing_block(params.get(ROUTING_BLOCK_KEY, {}))


def _replace_routing_block(module: dict[str, Any], block: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(module)
    params = out.get("params", {})
    if not isinstance(params, dict):
        raise ValueError("module params must be a mapping")
    params = deepcopy(params)
    params[ROUTING_BLOCK_KEY] = normalize_routing_block(block)
    out["params"] = params
    return out


def _binding_count(modules: Iterable[dict[str, Any]]) -> int:
    total = 0
    for module in modules:
        total += len(list(_routing_block(module).get("bindings", []) or []))
    return int(total)


def _select_index(unit: float, count: int) -> tuple[int, float]:
    if count <= 0:
        return 0, 0.0
    scaled = float(unit) * float(count)
    index = min(count - 1, int(scaled))
    return int(index), float(scaled - float(index))


def mutate_observation_bindings(
    modules: list[dict[str, Any]],
    draw0: float,
    draw1: float,
    *,
    enabled: bool,
    admitted_semantic_ids: tuple[str, ...] = ("self_energy", "self_health"),
    max_per_module: int = 2,
    max_per_agent: int = 8,
) -> RoutingMutationResult:
    """Mutate one module-local observation binding with a frozen two-draw map.

    Frozen mapping:

    * ``draw0`` selects a canonical generic module.  Its within-module-bin
      fraction selects a semantic or binding without an additional draw.
    * ``draw1`` selects equal-width operation bins: add, remove, gain, retarget.
      Its within-operation-bin fraction is the new gain for ``gain``.

    The add operation creates an exact-neutral zero-gain binding.  Retargeting
    explicitly changes the semantic binding key but never changes module
    instance, homology, ancestry, or topology identity.
    """

    admitted = tuple(sorted({str(value) for value in admitted_semantic_ids if str(value)}))
    module_cap = min(MAX_BINDINGS_PER_MODULE, max(0, int(max_per_module)))
    agent_cap = min(MAX_BINDINGS_PER_AGENT, max(0, int(max_per_agent)))
    raw_parent = deepcopy(list(modules or []))

    telemetry: dict[str, Any] = {
        "schema_version": ROUTING_MUTATION_RESULT_SCHEMA_VERSION,
        "operator": ROUTING_MUTATION_OPERATOR,
        "profile": ROUTING_MUTATION_PROFILE,
        "routing_schema_version": ROUTING_SCHEMA_VERSION,
        "routing_registry_version": ROUTING_REGISTRY_VERSION,
        "operation_order": list(_OPERATION_ORDER),
        "operation_bins": {
            "add": [0.0, 0.25],
            "remove": [0.25, 0.5],
            "gain": [0.5, 0.75],
            "retarget": [0.75, 1.0],
        },
        "operation_interval_rule": "lower_inclusive_upper_exclusive",
        "draw_budget": {
            "declared_supplied_draws": ROUTING_OPERATOR_DRAWS,
            "internal_rng_draws": 0,
        },
        "draw0": None,
        "draw1": None,
        "enabled": bool(enabled),
        "status": "rejected",
        "reason": "",
        "operation": "",
        "selected_module_id": "",
        "selected_module_instance_id": "",
        "selected_homology_id": "",
        "binding_key_before": "",
        "binding_key_after": "",
        "semantic_id_before": "",
        "semantic_id_after": "",
        "gain_before": None,
        "gain_after": None,
        "admitted_semantic_ids": list(admitted),
        "max_bindings_per_module": int(module_cap),
        "max_bindings_per_agent": int(agent_cap),
        "bindings_before": 0,
        "bindings_after": 0,
        "parent_preserved": True,
    }

    try:
        parent = normalize_agent_routing_modules(raw_parent)
        parent = sorted(parent, key=_module_sort_key)
        telemetry["bindings_before"] = _binding_count(parent)
        telemetry["bindings_after"] = telemetry["bindings_before"]
    except Exception as exc:
        telemetry["reason"] = f"invalid_parent:{type(exc).__name__}"
        return RoutingMutationResult(modules=raw_parent, telemetry=telemetry)

    d0 = _finite_draw(draw0)
    d1 = _finite_draw(draw1)
    telemetry["draw0"] = d0
    telemetry["draw1"] = d1
    if d0 is None or d1 is None:
        telemetry["reason"] = "draw_out_of_range"
        return RoutingMutationResult(modules=parent, telemetry=telemetry)
    if not bool(enabled):
        telemetry["status"] = "disabled"
        telemetry["reason"] = "operator_disabled"
        return RoutingMutationResult(modules=parent, telemetry=telemetry)
    if not admitted:
        telemetry["status"] = "ineligible"
        telemetry["reason"] = "no_admitted_semantics"
        return RoutingMutationResult(modules=parent, telemetry=telemetry)

    eligible_indices = [
        index
        for index, module in enumerate(parent)
        if str(module.get("module_type", "")) == "generic"
    ]
    if not eligible_indices:
        telemetry["status"] = "ineligible"
        telemetry["reason"] = "no_generic_module"
        return RoutingMutationResult(modules=parent, telemetry=telemetry)

    selected_ordinal, module_fraction = _select_index(d0, len(eligible_indices))
    module_index = eligible_indices[selected_ordinal]
    module = parent[module_index]
    telemetry["selected_module_id"] = str(module.get("module_id", ""))
    telemetry["selected_module_instance_id"] = str(module.get("module_instance_id", ""))
    telemetry["selected_homology_id"] = str(module.get("homology_id", ""))

    operation_index, operation_fraction = _select_index(d1, len(_OPERATION_ORDER))
    operation = _OPERATION_ORDER[operation_index]
    telemetry["operation"] = operation
    block = _routing_block(module)
    bindings = sorted(
        [deepcopy(binding) for binding in list(block.get("bindings", []) or [])],
        key=_binding_sort_key,
    )
    candidate_bindings = deepcopy(bindings)

    if operation == "add":
        if len(bindings) >= module_cap:
            telemetry["status"] = "cap_blocked"
            telemetry["reason"] = "module_binding_cap"
            return RoutingMutationResult(modules=parent, telemetry=telemetry)
        if int(telemetry["bindings_before"]) >= agent_cap:
            telemetry["status"] = "cap_blocked"
            telemetry["reason"] = "agent_binding_cap"
            return RoutingMutationResult(modules=parent, telemetry=telemetry)
        semantic_index, _ = _select_index(module_fraction, len(admitted))
        semantic_after = admitted[semantic_index]
        key_after = f"{semantic_after}:{_TARGET_SLOT}"
        telemetry["semantic_id_after"] = semantic_after
        telemetry["binding_key_after"] = key_after
        if any(_binding_sort_key(binding) == (semantic_after, _TARGET_SLOT) for binding in bindings):
            telemetry["reason"] = "duplicate_binding"
            return RoutingMutationResult(modules=parent, telemetry=telemetry)
        candidate_bindings.append(
            {
                "semantic_id": semantic_after,
                "target_slot": _TARGET_SLOT,
                "gain": 0.0,
                "enabled": True,
            }
        )
        telemetry["gain_after"] = 0.0

    elif operation in {"remove", "gain", "retarget"}:
        if not bindings:
            telemetry["status"] = "noop"
            telemetry["reason"] = "no_existing_binding"
            return RoutingMutationResult(modules=parent, telemetry=telemetry)
        binding_index, binding_fraction = _select_index(module_fraction, len(bindings))
        selected = deepcopy(bindings[binding_index])
        semantic_before = str(selected.get("semantic_id", ""))
        slot_before = str(selected.get("target_slot", _TARGET_SLOT) or _TARGET_SLOT)
        telemetry["semantic_id_before"] = semantic_before
        telemetry["binding_key_before"] = f"{semantic_before}:{slot_before}"
        telemetry["gain_before"] = float(selected.get("gain", 0.0) or 0.0)

        if operation == "remove":
            del candidate_bindings[binding_index]
        elif operation == "gain":
            gain_after = float(-1.0 + 2.0 * operation_fraction)
            gain_after = min(1.0, max(-1.0, gain_after))
            if gain_after == float(selected.get("gain", 0.0) or 0.0):
                telemetry["status"] = "noop"
                telemetry["reason"] = "gain_unchanged"
                return RoutingMutationResult(modules=parent, telemetry=telemetry)
            selected["gain"] = gain_after
            candidate_bindings[binding_index] = selected
            telemetry["gain_after"] = gain_after
            telemetry["semantic_id_after"] = semantic_before
            telemetry["binding_key_after"] = telemetry["binding_key_before"]
        else:
            semantic_index, _ = _select_index(binding_fraction, len(admitted))
            semantic_after = admitted[semantic_index]
            telemetry["semantic_id_after"] = semantic_after
            telemetry["binding_key_after"] = f"{semantic_after}:{slot_before}"
            if semantic_after == semantic_before:
                telemetry["status"] = "noop"
                telemetry["reason"] = "retarget_unchanged"
                return RoutingMutationResult(modules=parent, telemetry=telemetry)
            if any(
                index != binding_index
                and _binding_sort_key(binding) == (semantic_after, slot_before)
                for index, binding in enumerate(bindings)
            ):
                telemetry["reason"] = "duplicate_binding"
                return RoutingMutationResult(modules=parent, telemetry=telemetry)
            selected["semantic_id"] = semantic_after
            selected.pop("binding_key", None)
            candidate_bindings[binding_index] = selected

    candidate_block = deepcopy(block)
    candidate_block["bindings"] = sorted(candidate_bindings, key=_binding_sort_key)
    try:
        candidate_module = _replace_routing_block(module, candidate_block)
        candidate_modules = deepcopy(parent)
        candidate_modules[module_index] = candidate_module
        candidate_modules = normalize_agent_routing_modules(candidate_modules)
        candidate_modules = sorted(candidate_modules, key=_module_sort_key)
    except Exception as exc:
        telemetry["reason"] = f"candidate_rejected:{type(exc).__name__}"
        return RoutingMutationResult(modules=parent, telemetry=telemetry)

    if candidate_modules == parent:
        telemetry["status"] = "noop"
        telemetry["reason"] = "canonical_candidate_unchanged"
        return RoutingMutationResult(modules=parent, telemetry=telemetry)

    telemetry["status"] = "accepted"
    telemetry["reason"] = "binding_mutation_applied"
    telemetry["bindings_after"] = _binding_count(candidate_modules)
    telemetry["parent_preserved"] = False
    return RoutingMutationResult(modules=candidate_modules, telemetry=telemetry)


__all__ = [
    "ROUTING_MUTATION_OPERATOR",
    "ROUTING_MUTATION_PROFILE",
    "ROUTING_OPERATOR_DRAWS",
    "RoutingMutationResult",
    "mutate_observation_bindings",
]
