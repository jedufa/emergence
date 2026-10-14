"""Pure ESCAP-8 sustainable modular-growth contracts.

This module deliberately owns no random-number generator.  Mutation callers
must supply the two already-accounted unit-interval draws.  All cost and bloat
outputs are reporting-only and have no reward, fitness, energy, or selection
authority.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping

from agents.modular_genome_v1 import B2_FROZEN_CAPS, canonical_payload_from_genome_dict, resolve_b2_caps
from evolution.structural_gated_recurrent_v1 import (
    GATED_NUMERIC_FIELDS,
    GATED_OPERATOR,
    GATED_PARAMETER_BOUNDS,
    GATED_RECORD_KEY,
    normalize_gated_module_params,
)


SUSTAINABLE_GROWTH_PROFILE = "escap8_sustainable_modular_growth_v1"
ACTIVITY_RECORD_KEY = "structural_activity"
ACTIVITY_SCHEMA_VERSION = "escap8_structural_activity_v1"
PRUNE_ELIGIBILITY_VERSION = "escap8_prune_eligibility_v1"
PRUNE_CLEANUP_VERSION = "escap8_prune_cleanup_v1"
REPACK_VERSION = "escap8_biological_local_repack_v1"
FUSION_POLICY_VERSION = "escap8_fusion_deferred_historical_operator_not_safe_v1"
DUPLICATION_NEUTRALIZATION_VERSION = "escap8_exact_output_neutral_duplication_v1"
REFINEMENT_OPERATOR_VERSION = "escap8_bounded_duplicate_refinement_v1"
SPECIALIZATION_TELEMETRY_VERSION = "escap8_source_descendant_specialization_metrics_v1"
CHECKPOINT_SCHEMA_VERSION = "escap8_sustainable_growth_checkpoint_metadata_v1"

COST_SCHEMA_VERSION = "escap8_structural_cost_v1"
COST_PROFILE = "escap8_reporting_first_additive_cost_v1"
COST_MODE = "reporting_only"
COST_COEFFICIENTS: dict[str, float] = {
    "allocated_module": 1.0,
    "executing_integrator": 4.0,
    "executing_gated": 32.0,
    "evaluated_connection": 2.0,
    "active_input_binding": 1.0,
    "active_output_binding": 2.0,
    "active_dependency_edge": 1.0,
    "dormant_module_maintenance": 0.0,
    "serialized_genome_byte": 0.0,
    "saturated_or_rejected_activity": 0.0,
}

BLOAT_CLASSIFICATION_VERSION = "escap8_bloat_classification_v1"
GROWTH_MUTATION_PROFILE = "escap8_sustainable_growth_mutation_v1"
GROWTH_MUTATION_ATTEMPTS = 1
GROWTH_MUTATION_DRAWS = 2
GROWTH_MUTATION_BINS: tuple[tuple[float, float, str], ...] = (
    (0.00, 0.25, "toggle_dormancy"),
    (0.25, 0.50, "prune_eligible_module"),
    (0.50, 0.75, "refine_descendant_parameter"),
    (0.75, 1.00, "refine_descendant_routing_or_expression"),
)
SUSTAINABLE_GROWTH_OPERATOR_ATTEMPTS = GROWTH_MUTATION_ATTEMPTS
SUSTAINABLE_GROWTH_OPERATOR_DRAWS_PER_ATTEMPT = GROWTH_MUTATION_DRAWS
C2_COMPATIBILITY_VERSION = "escap8_sustainable_growth_c2_atomic_module_v1"
SUPPORTED_LANE = "python_reference_cpu"
MIGRATION_RNG_DRAWS = 0

FIXED_ENVELOPE_SHAPE: dict[str, int] = {
    "max_module_slots": int(B2_FROZEN_CAPS["max_modules_per_agent"]),
    "max_connection_slots": int(B2_FROZEN_CAPS["max_connections_per_agent"]),
    "state_width_per_module": 1,
    "observation_width": 27,
    "action_width": 7,
    "mlp_hidden_width": 64,
    "central_gru_hidden_width": 64,
    "new_arrays": 0,
}
FIXED_ENVELOPE_SHAPE_DIGEST = hashlib.sha256(
    json.dumps(
        FIXED_ENVELOPE_SHAPE,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()


def fixed_envelope_shape_for_caps(b2_caps: Mapping[str, Any] | None = None) -> dict[str, int]:
    caps = resolve_b2_caps(dict(b2_caps) if b2_caps is not None else None)
    return {
        **FIXED_ENVELOPE_SHAPE,
        "max_module_slots": int(caps["max_modules_per_agent"]),
        "max_connection_slots": int(caps["max_connections_per_agent"]),
    }


def fixed_envelope_shape_digest_for_caps(b2_caps: Mapping[str, Any] | None = None) -> str:
    shape = fixed_envelope_shape_for_caps(b2_caps)
    return hashlib.sha256(
        json.dumps(shape, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()

_DUPLICATION_ORIGINS = frozenset({"duplicate_with_decay", "motif_duplication"})
_INTEGRATOR_OPERATOR = "bounded_scalar_integrator_v1"
_INTEGRATOR_FIELDS: tuple[tuple[str, tuple[float, float]], ...] = (
    ("expression_scale", (-2.0, 2.0)),
    ("live_retention", (0.0, 1.0)),
    ("live_input_gain", (-2.0, 2.0)),
    ("live_output_gain", (-2.0, 2.0)),
)


class SustainableGrowthContractError(ValueError):
    """A strict ESCAP-8 record or pure-input contract was violated."""


@dataclass(frozen=True)
class SustainableGrowthMutationResult:
    payload: dict[str, Any]
    telemetry: dict[str, Any]


def _finite_unit(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise SustainableGrowthContractError(f"{field} must be a finite unit draw")
    try:
        result = float(value)
    except Exception as exc:
        raise SustainableGrowthContractError(f"{field} must be a finite unit draw") from exc
    if not math.isfinite(result) or not 0.0 <= result < 1.0:
        raise SustainableGrowthContractError(f"{field} must be in [0,1)")
    return result


def _nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise SustainableGrowthContractError(f"{field} must be a nonnegative integer")
    try:
        result = int(value)
    except Exception as exc:
        raise SustainableGrowthContractError(f"{field} must be a nonnegative integer") from exc
    if result < 0 or result != value:
        raise SustainableGrowthContractError(f"{field} must be a nonnegative integer")
    return result


def _finite_nonnegative(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise SustainableGrowthContractError(f"{field} must be finite and nonnegative")
    try:
        result = float(value)
    except Exception as exc:
        raise SustainableGrowthContractError(f"{field} must be finite and nonnegative") from exc
    if not math.isfinite(result) or result < 0.0:
        raise SustainableGrowthContractError(f"{field} must be finite and nonnegative")
    return result


def normalize_activity_module_params(params: Any) -> dict[str, Any]:
    """Validate the optional hereditary activity record; absence means active."""

    if not isinstance(params, Mapping):
        raise SustainableGrowthContractError("module params must be a mapping")
    normalized = deepcopy(dict(params))
    if ACTIVITY_RECORD_KEY not in normalized:
        return normalized
    record = normalized[ACTIVITY_RECORD_KEY]
    if not isinstance(record, Mapping):
        raise SustainableGrowthContractError("structural activity record must be a mapping")
    actual = {str(key) for key in record}
    expected = {"schema_version", "dormant"}
    if actual != expected:
        raise SustainableGrowthContractError(
            f"structural activity fields mismatch missing={sorted(expected-actual)!r} "
            f"extra={sorted(actual-expected)!r}"
        )
    if str(record.get("schema_version", "")) != ACTIVITY_SCHEMA_VERSION:
        raise SustainableGrowthContractError("structural activity schema mismatch")
    if type(record.get("dormant")) is not bool:
        raise SustainableGrowthContractError("structural activity dormant must be bool")
    normalized[ACTIVITY_RECORD_KEY] = {
        "schema_version": ACTIVITY_SCHEMA_VERSION,
        "dormant": bool(record["dormant"]),
    }
    return normalized


def module_is_dormant(module: Any) -> bool:
    if not isinstance(module, Mapping):
        raise SustainableGrowthContractError("module must be a mapping")
    params = normalize_activity_module_params(module.get("params", {}))
    record = params.get(ACTIVITY_RECORD_KEY)
    return bool(record is not None and record["dormant"])


def hereditary_no_output(module: Any) -> bool:
    """Return exact hereditary public-output neutrality, never runtime history."""

    if not isinstance(module, Mapping) or str(module.get("module_type", "")) != "generic":
        return False
    try:
        params = normalize_activity_module_params(module.get("params", {}))
        params = normalize_gated_module_params(params)
        if module_is_dormant({**dict(module), "params": params}):
            return True
        expression = float(params.get("expression_scale", 0.0))
        if not math.isfinite(expression):
            return False
        if expression == 0.0:
            return True
        operator = str(params.get("live_operator", _INTEGRATOR_OPERATOR) or _INTEGRATOR_OPERATOR)
        if operator == GATED_OPERATOR:
            return float(dict(params[GATED_RECORD_KEY])["output_gain"]) == 0.0
        if operator == _INTEGRATOR_OPERATOR:
            output_gain = float(params.get("live_output_gain", 0.0))
            return math.isfinite(output_gain) and output_gain == 0.0
    except (TypeError, ValueError):
        return False
    return False


def is_duplicate_descendant(module: Any) -> bool:
    if not isinstance(module, Mapping):
        return False
    ancestry = module.get("ancestry", {})
    if not isinstance(ancestry, Mapping):
        return False
    parent_ids = ancestry.get("parent_module_instance_ids", [])
    if not isinstance(parent_ids, (list, tuple)) or not any(str(value) for value in parent_ids):
        return False
    return str(ancestry.get("origin_operator", "")) in _DUPLICATION_ORIGINS


def _record_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _mapping_difference(source: Any, descendant: Any) -> dict[str, Any]:
    source_record = deepcopy(dict(source)) if isinstance(source, Mapping) else {}
    descendant_record = (
        deepcopy(dict(descendant)) if isinstance(descendant, Mapping) else {}
    )
    keys = sorted(set(source_record) | set(descendant_record))
    changed_fields = [
        key for key in keys if source_record.get(key) != descendant_record.get(key)
    ]
    return {
        "changed": bool(changed_fields),
        "changed_field_count": len(changed_fields),
        "changed_fields": changed_fields,
        "source_digest": _record_digest(source_record),
        "descendant_digest": _record_digest(descendant_record),
    }


def _incident_connection_records(
    module: Mapping[str, Any], connections: Any
) -> list[dict[str, Any]]:
    local_id = str(module.get("module_id", ""))
    records: list[dict[str, Any]] = []
    for raw in list(connections or []):
        if not isinstance(raw, Mapping):
            continue
        source = str(raw.get("src_module_id", ""))
        target = str(raw.get("dst_module_id", ""))
        if source != local_id and target != local_id:
            continue
        records.append(
            {
                "direction": "self_loop" if source == target == local_id else (
                    "outgoing" if source == local_id else "incoming"
                ),
                "other_module_id": (
                    "$self"
                    if source == target == local_id
                    else (target if source == local_id else source)
                ),
                "src_slot": str(raw.get("src_slot", "")),
                "dst_slot": str(raw.get("dst_slot", "")),
                "value_type": str(raw.get("value_type", "")),
                "channel": str(raw.get("channel", "")),
                "weight": float(raw.get("weight", 0.0) or 0.0),
            }
        )
    return sorted(records, key=lambda row: json.dumps(row, sort_keys=True))


def specialization_difference_summary(
    source_module: Mapping[str, Any],
    descendant_module: Mapping[str, Any],
    *,
    connections: Any = (),
    source_causal_output: float | None = None,
    descendant_causal_output: float | None = None,
) -> dict[str, Any]:
    """Report deterministic hereditary divergence without control authority."""

    source = deepcopy(dict(source_module))
    descendant = deepcopy(dict(descendant_module))
    source_params = normalize_activity_module_params(source.get("params", {}))
    descendant_params = normalize_activity_module_params(descendant.get("params", {}))
    excluded = {
        ACTIVITY_RECORD_KEY,
        "observation_routing",
        "output_routing",
        "developmental_expression",
        "expression_scale",
    }
    parameter = _mapping_difference(
        {key: value for key, value in source_params.items() if key not in excluded},
        {key: value for key, value in descendant_params.items() if key not in excluded},
    )
    input_routing = _mapping_difference(
        source_params.get("observation_routing", {}),
        descendant_params.get("observation_routing", {}),
    )
    output_routing = _mapping_difference(
        source_params.get("output_routing", {}),
        descendant_params.get("output_routing", {}),
    )
    source_development = source_params.get("developmental_expression", {})
    descendant_development = descendant_params.get("developmental_expression", {})
    source_dependency = (
        list(source_development.get("prerequisite_homology_ids", []) or [])
        if isinstance(source_development, Mapping)
        else []
    )
    descendant_dependency = (
        list(descendant_development.get("prerequisite_homology_ids", []) or [])
        if isinstance(descendant_development, Mapping)
        else []
    )
    development = _mapping_difference(
        {
            key: value
            for key, value in dict(source_development or {}).items()
            if key != "prerequisite_homology_ids"
        }
        if isinstance(source_development, Mapping)
        else {},
        {
            key: value
            for key, value in dict(descendant_development or {}).items()
            if key != "prerequisite_homology_ids"
        }
        if isinstance(descendant_development, Mapping)
        else {},
    )
    dependency = _mapping_difference(
        {"prerequisite_homology_ids": sorted(str(v) for v in source_dependency)},
        {"prerequisite_homology_ids": sorted(str(v) for v in descendant_dependency)},
    )
    expression = _mapping_difference(
        {"expression_scale": source_params.get("expression_scale")},
        {"expression_scale": descendant_params.get("expression_scale")},
    )
    connection = _mapping_difference(
        {"incident": _incident_connection_records(source, connections)},
        {"incident": _incident_connection_records(descendant, connections)},
    )
    causal_measured = source_causal_output is not None and descendant_causal_output is not None
    causal_output = {
        "measured": causal_measured,
        "changed": bool(
            causal_measured
            and float(source_causal_output) != float(descendant_causal_output)
        ),
        "source": float(source_causal_output) if source_causal_output is not None else None,
        "descendant": (
            float(descendant_causal_output)
            if descendant_causal_output is not None
            else None
        ),
        "basis": (
            "probe_measured_downstream_influence"
            if causal_measured
            else "not_measured_live"
        ),
    }
    dimensions = {
        "parameter": parameter,
        "input_routing": input_routing,
        "output_routing": output_routing,
        "connection": connection,
        "development": development,
        "dependency": dependency,
        "expression": expression,
        "causal_output": causal_output,
    }
    changed_dimensions = sorted(
        name for name, row in dimensions.items() if bool(row.get("changed", False))
    )
    return {
        "schema_version": SPECIALIZATION_TELEMETRY_VERSION,
        "reporting_only": True,
        "selection_input": False,
        "source_module_instance_id": str(source.get("module_instance_id", "")),
        "descendant_module_instance_id": str(descendant.get("module_instance_id", "")),
        "source_homology_id": str(source.get("homology_id", "")),
        "descendant_homology_id": str(descendant.get("homology_id", "")),
        "descendant_is_duplicate": is_duplicate_descendant(descendant),
        "dimensions": dimensions,
        "changed_dimensions": changed_dimensions,
        "changed_dimension_count": len(changed_dimensions),
        "specialized": bool(changed_dimensions),
    }


def _module_key(module: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(module.get("homology_id", "")),
        str(module.get("module_instance_id", "")),
        str(module.get("module_id", "")),
    )


def _select(draw: float, count: int) -> tuple[int, float]:
    if count <= 0:
        return 0, 0.0
    scaled = draw * float(count)
    index = min(count - 1, int(scaled))
    return index, scaled - float(index)


def _operation(draw: float) -> str:
    for lower, upper, operation in GROWTH_MUTATION_BINS:
        if lower <= draw < upper:
            return operation
    return GROWTH_MUTATION_BINS[-1][2]


def _validate_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    candidate = deepcopy(dict(payload))
    modules: list[dict[str, Any]] = []
    for raw in list(candidate.get("modules", []) or []):
        if not isinstance(raw, Mapping):
            raise SustainableGrowthContractError("module record must be a mapping")
        module = deepcopy(dict(raw))
        params = normalize_activity_module_params(module.get("params", {}))
        module["params"] = normalize_gated_module_params(params)
        modules.append(module)

    from evolution.structural_development_v1 import (
        agent_has_developmental_records,
        normalize_agent_developmental_modules,
        validate_agent_developmental_graph,
    )

    modules = normalize_agent_developmental_modules(modules)
    if agent_has_developmental_records(modules):
        validate_agent_developmental_graph(modules)
    candidate["modules"] = modules
    candidate["b2_invalid_policy"] = "reject"
    canonical = canonical_payload_from_genome_dict(candidate)

    canonical_modules = [dict(module) for module in list(canonical.get("modules", []) or [])]
    for module in canonical_modules:
        module["params"] = normalize_gated_module_params(
            normalize_activity_module_params(module.get("params", {}))
        )
    if agent_has_developmental_records(canonical_modules):
        validate_agent_developmental_graph(canonical_modules)

    from evolution.structural_live_modules_v1 import _validate_graph

    generic = [module for module in canonical_modules if str(module.get("module_type", "")) == "generic"]
    generic_ids = {str(module.get("module_id", "")) for module in generic}
    generic_connections = [
        dict(connection)
        for connection in list(canonical.get("connections", []) or [])
        if str(connection.get("src_module_id", "")) in generic_ids
        and str(connection.get("dst_module_id", "")) in generic_ids
    ]
    if generic:
        _validate_graph(generic, generic_connections)
    canonical["modules"] = canonical_modules
    return canonical


def _dependency_blockers(modules: list[dict[str, Any]], target_homology: str) -> list[str]:
    from evolution.structural_development_v1 import (
        DEVELOPMENTAL_EXPRESSION_BLOCK_KEY,
        normalize_developmental_expression_record,
    )

    blockers: list[str] = []
    for module in modules:
        if str(module.get("homology_id", "")) == target_homology:
            continue
        params = module.get("params", {})
        if not isinstance(params, Mapping) or DEVELOPMENTAL_EXPRESSION_BLOCK_KEY not in params:
            continue
        record = normalize_developmental_expression_record(
            params[DEVELOPMENTAL_EXPRESSION_BLOCK_KEY],
            module_homology_id=str(module.get("homology_id", "")),
        )
        if record is not None and target_homology in {
            str(value) for value in record["prerequisite_homology_ids"]
        }:
            blockers.append(str(module.get("homology_id", "")))
    return sorted(blockers)


def _repack_without(
    payload: Mapping[str, Any], target: Mapping[str, Any]
) -> dict[str, Any]:
    target_instance = str(target.get("module_instance_id", ""))
    target_local = str(target.get("module_id", ""))
    remaining = [
        deepcopy(dict(module))
        for module in list(payload.get("modules", []) or [])
        if not (
            str(module.get("module_instance_id", "")) == target_instance
            and str(module.get("module_id", "")) == target_local
        )
    ]
    remaining.sort(key=_module_key)
    local_map: dict[str, str] = {}
    for index, module in enumerate(remaining):
        old = str(module.get("module_id", ""))
        new = f"m{index:03d}"
        local_map[old] = new
        module["module_id"] = new

    connections: list[dict[str, Any]] = []
    for raw in list(payload.get("connections", []) or []):
        connection = deepcopy(dict(raw))
        src = str(connection.get("src_module_id", ""))
        dst = str(connection.get("dst_module_id", ""))
        if src == target_local or dst == target_local:
            continue
        if src not in local_map or dst not in local_map:
            raise SustainableGrowthContractError("prune proposal contains dangling connection")
        connection["src_module_id"] = local_map[src]
        connection["dst_module_id"] = local_map[dst]
        connection.pop("locality_distance", None)
        connection.pop("locality_cost", None)
        connections.append(connection)
    connections.sort(
        key=lambda connection: (
            str(connection.get("src_module_id", "")),
            str(connection.get("dst_module_id", "")),
            str(connection.get("src_slot", "")),
            str(connection.get("dst_slot", "")),
            str(connection.get("value_type", "")),
            str(connection.get("channel", "")),
            float(connection.get("weight", 0.0)),
            str(connection.get("connection_id", "")),
        )
    )
    for index, connection in enumerate(connections):
        connection["connection_id"] = f"c{index:03d}"

    candidate = deepcopy(dict(payload))
    candidate["modules"] = remaining
    candidate["connections"] = connections
    return candidate


def _parameter_fields(module: Mapping[str, Any]) -> tuple[tuple[str, tuple[float, float]], ...]:
    params = module.get("params", {})
    operator = str(
        (params.get("live_operator", _INTEGRATOR_OPERATOR) if isinstance(params, Mapping) else _INTEGRATOR_OPERATOR)
        or _INTEGRATOR_OPERATOR
    )
    if operator == GATED_OPERATOR:
        return tuple((field, GATED_PARAMETER_BOUNDS[field]) for field in GATED_NUMERIC_FIELDS)
    if operator == _INTEGRATOR_OPERATOR:
        return _INTEGRATOR_FIELDS
    return ()


def _replace_parameter(
    module: Mapping[str, Any], field: str, value: float
) -> dict[str, Any]:
    out = deepcopy(dict(module))
    params = normalize_gated_module_params(
        normalize_activity_module_params(out.get("params", {}))
    )
    operator = str(params.get("live_operator", _INTEGRATOR_OPERATOR) or _INTEGRATOR_OPERATOR)
    if operator == GATED_OPERATOR:
        record = deepcopy(dict(params[GATED_RECORD_KEY]))
        record[field] = float(value)
        params[GATED_RECORD_KEY] = record
        params = normalize_gated_module_params(params)
    else:
        params[field] = float(value)
    out["params"] = params
    return out


def _routing_or_expression_candidates(
    modules: list[dict[str, Any]], eligible_indices: list[int]
) -> list[tuple[int, str, int]]:
    from evolution.structural_observation_routing_v1 import ROUTING_BLOCK_KEY
    from evolution.structural_output_routing_v1 import OUTPUT_ROUTING_BLOCK_KEY

    candidates: list[tuple[int, str, int]] = []
    for module_index in eligible_indices:
        params = modules[module_index].get("params", {})
        if not isinstance(params, Mapping):
            continue
        for block_key, kind in (
            (ROUTING_BLOCK_KEY, "input_binding_gain"),
            (OUTPUT_ROUTING_BLOCK_KEY, "output_binding_gain"),
        ):
            block = params.get(block_key, {})
            bindings = list(block.get("bindings", []) or []) if isinstance(block, Mapping) else []
            for binding_index in range(len(bindings)):
                candidates.append((module_index, kind, binding_index))
    if not candidates:
        candidates.extend((module_index, "expression_scale", -1) for module_index in eligible_indices)
    return candidates


def _replace_routing_or_expression(
    module: Mapping[str, Any], kind: str, binding_index: int, value_fraction: float
) -> tuple[dict[str, Any], str, float, float]:
    from evolution.structural_observation_routing_v1 import (
        ROUTING_BLOCK_KEY,
        normalize_routing_block,
    )
    from evolution.structural_output_routing_v1 import (
        OUTPUT_ROUTING_BLOCK_KEY,
        normalize_output_routing_block,
    )

    out = deepcopy(dict(module))
    params = normalize_activity_module_params(out.get("params", {}))
    if kind == "expression_scale":
        before = float(params.get("expression_scale", 0.0))
        after = -2.0 + 4.0 * value_fraction
        params["expression_scale"] = after
        field = "expression_scale"
    else:
        block_key = ROUTING_BLOCK_KEY if kind == "input_binding_gain" else OUTPUT_ROUTING_BLOCK_KEY
        normalize = normalize_routing_block if kind == "input_binding_gain" else normalize_output_routing_block
        block = normalize(params.get(block_key))
        bindings = [deepcopy(binding) for binding in list(block.get("bindings", []) or [])]
        before = float(bindings[binding_index].get("gain", 0.0))
        after = -1.0 + 2.0 * value_fraction
        bindings[binding_index]["gain"] = after
        block["bindings"] = bindings
        params[block_key] = normalize(block)
        field = kind
    out["params"] = normalize_gated_module_params(params)
    return out, field, before, after


def mutate_sustainable_growth(
    payload: Mapping[str, Any],
    draw0: Any,
    draw1: Any,
    enabled: bool = True,
) -> SustainableGrowthMutationResult:
    """Apply one frozen ESCAP-8 mutation from exactly two supplied draws."""

    raw_parent = deepcopy(dict(payload))
    telemetry: dict[str, Any] = {
        "schema_version": "escap8_sustainable_growth_mutation_result_v1",
        "profile": GROWTH_MUTATION_PROFILE,
        "operation_bins": [list(row) for row in GROWTH_MUTATION_BINS],
        "draw_budget": {"declared_supplied_draws": 2, "internal_rng_draws": 0},
        "draw0": None,
        "draw1": None,
        "enabled": bool(enabled),
        "operation": "",
        "status": "rejected",
        "reason": "",
        "selected_module_instance_id": "",
        "selected_homology_id": "",
        "selected_field": "",
        "value_before": None,
        "value_after": None,
        "dependency_blockers": [],
        "parent_preserved": True,
        "fusion_policy": FUSION_POLICY_VERSION,
    }
    try:
        parent = _validate_payload(raw_parent)
    except Exception as exc:
        telemetry["reason"] = f"invalid_parent:{type(exc).__name__}:{exc}"
        return SustainableGrowthMutationResult(raw_parent, telemetry)
    try:
        d0 = _finite_unit(draw0, field="draw0")
        d1 = _finite_unit(draw1, field="draw1")
    except SustainableGrowthContractError as exc:
        telemetry["reason"] = f"invalid_draw:{exc}"
        return SustainableGrowthMutationResult(parent, telemetry)
    telemetry["draw0"] = d0
    telemetry["draw1"] = d1
    operation = _operation(d0)
    telemetry["operation"] = operation
    if not bool(enabled):
        telemetry["status"] = "disabled"
        telemetry["reason"] = "operator_disabled"
        return SustainableGrowthMutationResult(parent, telemetry)

    modules = [deepcopy(dict(module)) for module in list(parent.get("modules", []) or [])]
    generic_indices = [
        index
        for index, module in enumerate(modules)
        if str(module.get("module_type", "")) == "generic"
    ]
    generic_indices.sort(key=lambda index: _module_key(modules[index]))

    try:
        candidate = deepcopy(parent)
        if operation == "toggle_dormancy":
            if not generic_indices:
                telemetry.update(status="ineligible", reason="no_generic_module")
                return SustainableGrowthMutationResult(parent, telemetry)
            ordinal, _ = _select(d1, len(generic_indices))
            index = generic_indices[ordinal]
            selected = modules[index]
            dormant_before = module_is_dormant(selected)
            params = normalize_activity_module_params(selected.get("params", {}))
            params[ACTIVITY_RECORD_KEY] = {
                "schema_version": ACTIVITY_SCHEMA_VERSION,
                "dormant": not dormant_before,
            }
            selected["params"] = params
            modules[index] = selected
            candidate["modules"] = modules
            telemetry["selected_field"] = "dormant"
            telemetry["value_before"] = dormant_before
            telemetry["value_after"] = not dormant_before

        elif operation == "prune_eligible_module":
            if len(generic_indices) <= 1:
                telemetry.update(status="ineligible", reason="minimum_module_invariant")
                return SustainableGrowthMutationResult(parent, telemetry)
            eligible = [index for index in generic_indices if hereditary_no_output(modules[index])]
            if not eligible:
                telemetry.update(status="ineligible", reason="no_prunable_module")
                return SustainableGrowthMutationResult(parent, telemetry)
            ordinal, _ = _select(d1, len(eligible))
            selected = modules[eligible[ordinal]]
            blockers = _dependency_blockers(modules, str(selected.get("homology_id", "")))
            if blockers:
                telemetry["selected_module_instance_id"] = str(selected.get("module_instance_id", ""))
                telemetry["selected_homology_id"] = str(selected.get("homology_id", ""))
                telemetry["dependency_blockers"] = blockers
                telemetry.update(status="dependency_blocked", reason="required_by_development")
                return SustainableGrowthMutationResult(parent, telemetry)
            telemetry["selected_field"] = "module"
            telemetry["value_before"] = 1.0
            telemetry["value_after"] = 0.0
            candidate = _repack_without(parent, selected)

        elif operation == "refine_descendant_parameter":
            eligible = [index for index in generic_indices if is_duplicate_descendant(modules[index])]
            eligible = [index for index in eligible if _parameter_fields(modules[index])]
            if not eligible:
                telemetry.update(status="ineligible", reason="no_duplicate_descendant_parameter")
                return SustainableGrowthMutationResult(parent, telemetry)
            module_ordinal, module_fraction = _select(d1, len(eligible))
            index = eligible[module_ordinal]
            fields = _parameter_fields(modules[index])
            field_ordinal, value_fraction = _select(module_fraction, len(fields))
            field, bounds = fields[field_ordinal]
            before_params = normalize_gated_module_params(modules[index].get("params", {}))
            if str(before_params.get("live_operator", _INTEGRATOR_OPERATOR)) == GATED_OPERATOR:
                before = float(dict(before_params[GATED_RECORD_KEY])[field])
            else:
                before = float(before_params.get(field, 0.0))
            after = float(bounds[0]) + value_fraction * (float(bounds[1]) - float(bounds[0]))
            if after == before:
                telemetry.update(status="noop", reason="refined_value_unchanged")
                return SustainableGrowthMutationResult(parent, telemetry)
            modules[index] = _replace_parameter(modules[index], field, after)
            candidate["modules"] = modules
            telemetry["selected_field"] = field
            telemetry["value_before"] = before
            telemetry["value_after"] = after

        else:
            eligible = [index for index in generic_indices if is_duplicate_descendant(modules[index])]
            if not eligible:
                telemetry.update(status="ineligible", reason="no_duplicate_descendant_routing_or_expression")
                return SustainableGrowthMutationResult(parent, telemetry)
            proposals = _routing_or_expression_candidates(modules, eligible)
            proposal_index, value_fraction = _select(d1, len(proposals))
            index, kind, binding_index = proposals[proposal_index]
            replacement, field, before, after = _replace_routing_or_expression(
                modules[index], kind, binding_index, value_fraction
            )
            if after == before:
                telemetry.update(status="noop", reason="refined_value_unchanged")
                return SustainableGrowthMutationResult(parent, telemetry)
            modules[index] = replacement
            candidate["modules"] = modules
            telemetry["selected_field"] = field
            telemetry["value_before"] = before
            telemetry["value_after"] = after

        if operation != "prune_eligible_module":
            selected_module = next(
                (
                    module
                    for module in list(candidate.get("modules", []) or [])
                    if telemetry["selected_field"]
                    and (
                        telemetry["selected_module_instance_id"] == str(module.get("module_instance_id", ""))
                        or not telemetry["selected_module_instance_id"]
                    )
                ),
                None,
            )
            # Selection is set precisely below from the changed record.
            if operation == "toggle_dormancy":
                selected_module = modules[index]
            elif operation.startswith("refine_"):
                selected_module = modules[index]
            if selected_module is not None:
                telemetry["selected_module_instance_id"] = str(selected_module.get("module_instance_id", ""))
                telemetry["selected_homology_id"] = str(selected_module.get("homology_id", ""))
        else:
            telemetry["selected_module_instance_id"] = str(selected.get("module_instance_id", ""))
            telemetry["selected_homology_id"] = str(selected.get("homology_id", ""))

        validated = _validate_payload(candidate)
    except Exception as exc:
        telemetry["reason"] = f"invalid_proposal:{type(exc).__name__}:{exc}"
        return SustainableGrowthMutationResult(parent, telemetry)

    if operation.startswith("refine_") and telemetry["selected_module_instance_id"]:
        descendant = next(
            (
                module
                for module in list(validated.get("modules", []) or [])
                if str(module.get("module_instance_id", ""))
                == telemetry["selected_module_instance_id"]
            ),
            None,
        )
        if descendant is not None:
            ancestry = descendant.get("ancestry", {})
            parent_instance_ids = (
                list(ancestry.get("parent_module_instance_ids", []) or [])
                if isinstance(ancestry, Mapping)
                else []
            )
            source_instance_id = str(parent_instance_ids[0]) if parent_instance_ids else ""
            source = next(
                (
                    module
                    for module in list(parent.get("modules", []) or [])
                    if str(module.get("module_instance_id", "")) == source_instance_id
                ),
                None,
            )
            if source is not None:
                telemetry["specialization_difference"] = specialization_difference_summary(
                    source,
                    descendant,
                    connections=list(validated.get("connections", []) or []),
                )

    if validated == parent:
        telemetry.update(status="noop", reason="canonical_candidate_unchanged")
        return SustainableGrowthMutationResult(parent, telemetry)
    telemetry.update(status="accepted", reason="proposal_valid", parent_preserved=False)
    return SustainableGrowthMutationResult(validated, telemetry)


def structural_cost_summary(
    *,
    allocated_module_count: int,
    executing_integrator_count: int,
    executing_gated_count: int,
    evaluated_connection_count: int,
    active_input_binding_count: int,
    active_output_binding_count: int,
    active_dependency_edge_count: int,
    dormant_module_count: int,
    serialized_genome_bytes: int,
    saturated_or_rejected_activity_count: int,
) -> dict[str, Any]:
    raw_counts = {
        "allocated_module": allocated_module_count,
        "executing_integrator": executing_integrator_count,
        "executing_gated": executing_gated_count,
        "evaluated_connection": evaluated_connection_count,
        "active_input_binding": active_input_binding_count,
        "active_output_binding": active_output_binding_count,
        "active_dependency_edge": active_dependency_edge_count,
        "dormant_module_maintenance": dormant_module_count,
        "serialized_genome_byte": serialized_genome_bytes,
        "saturated_or_rejected_activity": saturated_or_rejected_activity_count,
    }
    counts = {
        key: _nonnegative_int(value, field=key)
        for key, value in raw_counts.items()
    }
    if counts["dormant_module_maintenance"] > counts["allocated_module"]:
        raise SustainableGrowthContractError("dormant modules exceed allocated modules")
    components = {
        key: float(counts[key]) * float(COST_COEFFICIENTS[key])
        for key in COST_COEFFICIENTS
    }
    total = math.fsum(components[key] for key in sorted(components))
    return {
        "schema_version": COST_SCHEMA_VERSION,
        "profile": COST_PROFILE,
        "mode": COST_MODE,
        "reporting_only": True,
        "coefficients": dict(COST_COEFFICIENTS),
        "counts": counts,
        "components": components,
        "total": float(total),
    }


def classify_bloat(
    *,
    allocated_module_count: int,
    module_capacity: int,
    active_module_count: int,
    dormant_module_count: int,
    causally_active_module_count: int,
    neutral_duplicate_count: int,
    redundant_structure_count: int,
    prunable_module_count: int,
    blocked_duplication_attempt_count: int,
    blocked_prune_attempt_count: int,
    cap_contact_count: int,
    sustained_cap_pressure: bool,
    structural_cost: float = 0.0,
    useful_activity_proxy: float = 0.0,
    causal_activity_measured: bool = True,
) -> dict[str, Any]:
    integer_inputs = {
        "allocated_module_count": allocated_module_count,
        "module_capacity": module_capacity,
        "active_module_count": active_module_count,
        "dormant_module_count": dormant_module_count,
        "causally_active_module_count": causally_active_module_count,
        "neutral_duplicate_count": neutral_duplicate_count,
        "redundant_structure_count": redundant_structure_count,
        "prunable_module_count": prunable_module_count,
        "blocked_duplication_attempt_count": blocked_duplication_attempt_count,
        "blocked_prune_attempt_count": blocked_prune_attempt_count,
        "cap_contact_count": cap_contact_count,
    }
    metrics = {
        key: _nonnegative_int(value, field=key)
        for key, value in integer_inputs.items()
    }
    if type(sustained_cap_pressure) is not bool:
        raise SustainableGrowthContractError("sustained_cap_pressure must be bool")
    if type(causal_activity_measured) is not bool:
        raise SustainableGrowthContractError("causal_activity_measured must be bool")
    metrics["structural_cost"] = _finite_nonnegative(structural_cost, field="structural_cost")
    metrics["useful_activity_proxy"] = _finite_nonnegative(
        useful_activity_proxy, field="useful_activity_proxy"
    )
    metrics["sustained_cap_pressure"] = bool(sustained_cap_pressure)
    metrics["causal_activity_measured"] = bool(causal_activity_measured)
    allocated = metrics["allocated_module_count"]
    capacity = metrics["module_capacity"]
    if capacity <= 0 or allocated > capacity:
        raise SustainableGrowthContractError("module capacity must be positive and not exceeded")
    for key in (
        "active_module_count",
        "dormant_module_count",
        "causally_active_module_count",
        "neutral_duplicate_count",
        "prunable_module_count",
    ):
        if metrics[key] > allocated:
            raise SustainableGrowthContractError(f"{key} exceeds allocated modules")

    waste = (
        metrics["dormant_module_count"]
        + metrics["neutral_duplicate_count"]
        + metrics["redundant_structure_count"]
    )
    low_causal_fraction = (
        allocated > 0
        and metrics["causally_active_module_count"] * 4 <= allocated
    )
    if (
        bool(causal_activity_measured)
        and bool(sustained_cap_pressure)
        and waste > 0
        and low_causal_fraction
    ):
        label = "possible_bloat"
    elif bool(sustained_cap_pressure) and (
        allocated >= capacity or metrics["blocked_duplication_attempt_count"] > 0
    ) and metrics["prunable_module_count"] == 0:
        label = "capacity_limited"
    elif metrics["prunable_module_count"] > 0 and (
        bool(sustained_cap_pressure)
        or metrics["blocked_prune_attempt_count"] > 0
        or allocated >= capacity
    ):
        label = "pruning_pressure"
    elif bool(causal_activity_measured) and allocated > 0 and metrics["causally_active_module_count"] == 0 and (
        metrics["neutral_duplicate_count"] > 0
        or metrics["active_module_count"] > 0
    ):
        label = "growth_without_activity"
    elif metrics["dormant_module_count"] > 0:
        label = "dormant_reserve"
    elif bool(causal_activity_measured) and metrics["active_module_count"] > 0 and metrics["causally_active_module_count"] > 0:
        label = "active_growth"
    else:
        label = "healthy_headroom"
    return {
        "schema_version": BLOAT_CLASSIFICATION_VERSION,
        "classification": label,
        "reporting_only": True,
        "single_cap_touch_is_pressure": False,
        "causal_activity_basis": (
            "measured_downstream_influence"
            if bool(causal_activity_measured)
            else "probe_required_not_classified_live"
        ),
        "metrics": metrics,
    }


__all__ = [
    "ACTIVITY_RECORD_KEY",
    "ACTIVITY_SCHEMA_VERSION",
    "BLOAT_CLASSIFICATION_VERSION",
    "C2_COMPATIBILITY_VERSION",
    "CHECKPOINT_SCHEMA_VERSION",
    "COST_COEFFICIENTS",
    "COST_MODE",
    "COST_PROFILE",
    "COST_SCHEMA_VERSION",
    "FIXED_ENVELOPE_SHAPE",
    "FIXED_ENVELOPE_SHAPE_DIGEST",
    "FUSION_POLICY_VERSION",
    "DUPLICATION_NEUTRALIZATION_VERSION",
    "GROWTH_MUTATION_ATTEMPTS",
    "GROWTH_MUTATION_BINS",
    "GROWTH_MUTATION_DRAWS",
    "GROWTH_MUTATION_PROFILE",
    "MIGRATION_RNG_DRAWS",
    "PRUNE_CLEANUP_VERSION",
    "PRUNE_ELIGIBILITY_VERSION",
    "REPACK_VERSION",
    "REFINEMENT_OPERATOR_VERSION",
    "SPECIALIZATION_TELEMETRY_VERSION",
    "SUPPORTED_LANE",
    "SUSTAINABLE_GROWTH_OPERATOR_ATTEMPTS",
    "SUSTAINABLE_GROWTH_OPERATOR_DRAWS_PER_ATTEMPT",
    "SUSTAINABLE_GROWTH_PROFILE",
    "SustainableGrowthContractError",
    "SustainableGrowthMutationResult",
    "classify_bloat",
    "hereditary_no_output",
    "is_duplicate_descendant",
    "module_is_dormant",
    "mutate_sustainable_growth",
    "normalize_activity_module_params",
    "specialization_difference_summary",
    "structural_cost_summary",
]
