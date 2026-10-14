from __future__ import annotations

"""Deterministic structural operator framework for Evolvability Phase C1."""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from typing import Any, Callable

from agents.developmental_program_v1 import canonicalize_developmental_program
from agents.modular_genome_v1 import (
    B2_INTERFACE_SCHEMA_VERSION,
    B3_FROZEN_LOCALITY,
    B3_LOCALITY_SCHEMA_VERSION,
    MODULAR_GENOME_SCHEMA_VERSION,
    _canon_obj,
    canonical_payload_from_genome_dict,
    canon_float,
    resolve_b2_caps,
)
from agents.structural_identity_v1 import new_native_identity
from evolution.capacity_errors import classify_structural_capacity_error
from evolution.structural_gated_recurrent_v1 import (
    GATED_CANDIDATE_FIELDS,
    GATED_DRAW1_BINS,
    GATED_DRAW_DECOMPOSITION_VERSION,
    GATED_MUTATION_BUDGET_VERSION,
    GATED_MUTATION_PROFILE,
    GATED_OPERATOR,
    GATED_OUTPUT_FIELDS,
    GATED_PARAMETER_BOUNDS,
    GATED_RECORD_KEY,
    GATED_RESET_FIELDS,
    GATED_UPDATE_FIELDS,
    GatedRecordError,
    f32,
    hereditary_neutral_integrator_eligible,
    nested_interval_choice,
    neutral_gated_recurrent_record,
    normalize_gated_module_params,
)


C1_OPERATOR_CATALOG_VERSION = "c1_operator_catalog_v2_structural_bootstrap"
C1_FROZEN_MAX_OPS_PER_OFFSPRING = 2
C1_FROZEN_INVALID_OPERATOR_POLICY = "reject"
C1_FROZEN_OPERATOR_MIX: dict[str, float] = {
    "duplicate_with_decay": 1.0,
    "gate_drift": 1.0,
    "bounded_rewire": 1.0,
    "edge_split_insertion": 1.0,
    "motif_duplication": 1.0,
    "promoter_expression_mutation": 1.0,
    "fusion_merge_compression": 1.0,
}


def c1_strict_enabled() -> bool:
    v1 = str(os.environ.get("EIEL_STRICT_DETERMINISM", "")).strip().lower()
    v2 = str(os.environ.get("DEBUG_DETERMINISM", "")).strip().lower()
    return (v1 in {"1", "true", "yes", "on"}) or (v2 in {"1", "true", "yes", "on"})


def _det_fail(code: str, context: str) -> None:
    raise RuntimeError(f"[DETERMINISM_VIOLATION]\ncode={code}\ncontext={context}\n")


def _mod_sort_key(m: dict[str, Any]) -> tuple[str, str]:
    return (str(m.get("module_id", "")), str(m.get("module_type", "")))


def _conn_sort_key(c: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(c.get("connection_id", "")),
        str(c.get("src_module_id", "")),
        str(c.get("dst_module_id", "")),
    )


def _next_id(existing_ids: set[str], prefix: str) -> str:
    i = 0
    while True:
        cid = f"{prefix}{i:03d}"
        if cid not in existing_ids:
            return cid
        i += 1


def _take_draws(rng: Any, n: int) -> list[float]:
    return [float(rng.random()) for _ in range(max(0, int(n)))]


def apply_observation_routing_operator_attempt(
    modules: list[dict[str, Any]],
    *,
    rng: Any,
    enabled: bool,
) -> Any:
    """Consume the frozen two-draw ESCAP-4 attempt budget for every outcome."""

    from evolution.operators_routing_v1 import (
        ROUTING_OPERATOR_DRAWS,
        mutate_observation_bindings,
    )

    draws = _take_draws(rng, ROUTING_OPERATOR_DRAWS)
    return mutate_observation_bindings(
        modules,
        draws[0],
        draws[1],
        enabled=bool(enabled),
    )


def apply_output_routing_operator_attempt(
    modules: list[dict[str, Any]],
    *,
    rng: Any,
    enabled: bool,
) -> Any:
    """Consume the frozen two-draw ESCAP-5 attempt budget for every outcome."""

    from evolution.operators_output_routing_v1 import (
        OUTPUT_ROUTING_OPERATOR_DRAWS,
        mutate_output_bindings,
    )

    draws = _take_draws(rng, OUTPUT_ROUTING_OPERATOR_DRAWS)
    return mutate_output_bindings(
        modules,
        draws[0],
        draws[1],
        enabled=bool(enabled),
    )


def apply_development_operator_attempt(
    modules: list[dict[str, Any]],
    *,
    rng: Any,
    enabled: bool,
) -> Any:
    """Consume the frozen two-draw ESCAP-6 attempt budget for every outcome."""

    from evolution.operators_development_v1 import mutate_developmental_expression
    from evolution.structural_development_v1 import DEVELOPMENT_OPERATOR_DRAWS_PER_ATTEMPT

    draws = _take_draws(rng, DEVELOPMENT_OPERATOR_DRAWS_PER_ATTEMPT)
    return mutate_developmental_expression(
        modules,
        draws[0],
        draws[1],
        enabled=bool(enabled),
    )


apply_developmental_operator_attempt = apply_development_operator_attempt


GATED_RECURRENT_OPERATOR_DRAWS_PER_ATTEMPT = 2
GATED_RECURRENT_OPERATOR_ATTEMPTS = 1
GATED_RECURRENT_MUTATION_RESULT_SCHEMA_VERSION = "escap7_gated_recurrent_mutation_result_v1"


@dataclass(frozen=True)
class GatedRecurrentMutationResult:
    modules: list[dict[str, Any]]
    telemetry: dict[str, Any]


def _gated_module_sort_key(module: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(module.get("homology_id", "")),
        str(module.get("module_instance_id", "")),
        str(module.get("module_id", "")),
    )


def _gated_finite_draw(value: Any) -> float | None:
    try:
        draw = float(value)
    except Exception:
        return None
    if not math.isfinite(draw) or draw < 0.0 or draw >= 1.0:
        return None
    return draw


def _gated_operation(draw: float) -> str:
    for lower, upper, operation in GATED_DRAW1_BINS:
        if float(lower) <= draw < float(upper):
            return str(operation)
    return str(GATED_DRAW1_BINS[-1][2])


def mutate_gated_recurrent_operator(
    modules: list[dict[str, Any]],
    draw0: float,
    draw1: float,
    *,
    enabled: bool,
) -> GatedRecurrentMutationResult:
    """Apply the post-ESCAP-6 one-attempt/two-draw hereditary lane."""

    raw_parent = deepcopy(list(modules or []))
    telemetry: dict[str, Any] = {
        "schema_version": GATED_RECURRENT_MUTATION_RESULT_SCHEMA_VERSION,
        "profile": GATED_MUTATION_PROFILE,
        "draw_decomposition_version": GATED_DRAW_DECOMPOSITION_VERSION,
        "mutation_budget_version": GATED_MUTATION_BUDGET_VERSION,
        "draw_budget": {
            "declared_supplied_draws": GATED_RECURRENT_OPERATOR_DRAWS_PER_ATTEMPT,
            "internal_rng_draws": 0,
        },
        "operation_bins": {
            operation: [float(lower), float(upper)]
            for lower, upper, operation in GATED_DRAW1_BINS
        },
        "draw0": None,
        "draw1": None,
        "enabled": bool(enabled),
        "status": "rejected",
        "reason": "",
        "operation": "",
        "selected_homology_id": "",
        "selected_module_instance_id": "",
        "selected_field": "",
        "value_before": None,
        "value_after": None,
        "parent_preserved": True,
    }
    try:
        parent: list[dict[str, Any]] = []
        for module in raw_parent:
            record = deepcopy(dict(module))
            record["params"] = normalize_gated_module_params(record.get("params", {}))
            if (
                str(record["params"].get("live_operator", "")) == GATED_OPERATOR
                and str(record.get("module_type", "")) != "generic"
            ):
                raise ValueError(
                    "ESCAP7_GATED_OPERATOR_REQUIRES_GENERIC_MODULE "
                    f"module_id={record.get('module_id', '')}"
                )
            parent.append(record)
    except (GatedRecordError, TypeError, ValueError) as exc:
        telemetry["reason"] = f"invalid_parent:{type(exc).__name__}:{exc}"
        return GatedRecurrentMutationResult(raw_parent, telemetry)

    d0 = _gated_finite_draw(draw0)
    d1 = _gated_finite_draw(draw1)
    telemetry["draw0"] = d0
    telemetry["draw1"] = d1
    if d0 is None or d1 is None:
        telemetry["reason"] = "draw_out_of_range"
        return GatedRecurrentMutationResult(parent, telemetry)
    operation = _gated_operation(d0)
    telemetry["operation"] = operation
    if not bool(enabled):
        telemetry["status"] = "disabled"
        telemetry["reason"] = "operator_disabled"
        return GatedRecurrentMutationResult(parent, telemetry)

    candidate = deepcopy(parent)
    if operation == "acquire_neutral":
        eligible = sorted(
            [module for module in parent if hereditary_neutral_integrator_eligible(module)],
            key=_gated_module_sort_key,
        )
        if not eligible:
            telemetry["status"] = "ineligible"
            telemetry["reason"] = "no_hereditary_output_neutral_integrator"
            return GatedRecurrentMutationResult(parent, telemetry)
        try:
            module_index, _unused_fraction = nested_interval_choice(
                d1,
                len(eligible),
                field="gated_mutation.module",
            )
        except GatedRecordError as exc:
            telemetry["reason"] = f"invalid_decomposition:{exc}"
            return GatedRecurrentMutationResult(parent, telemetry)
        selected = eligible[module_index]
        selected_instance = str(selected.get("module_instance_id", ""))
        selected_local = str(selected.get("module_id", ""))
        parent_index = next(
            index
            for index, module in enumerate(candidate)
            if str(module.get("module_instance_id", "")) == selected_instance
            and str(module.get("module_id", "")) == selected_local
        )
        proposal = deepcopy(selected)
        params = deepcopy(dict(proposal.get("params", {})))
        params["live_operator"] = GATED_OPERATOR
        params[GATED_RECORD_KEY] = neutral_gated_recurrent_record()
        proposal["params"] = normalize_gated_module_params(params)
        candidate[parent_index] = proposal
        telemetry["selected_homology_id"] = str(selected.get("homology_id", ""))
        telemetry["selected_module_instance_id"] = selected_instance

    else:
        fields_by_operation = {
            "mutate_reset_parameter": GATED_RESET_FIELDS,
            "mutate_update_parameter": GATED_UPDATE_FIELDS,
            "mutate_candidate_parameter": GATED_CANDIDATE_FIELDS,
            "mutate_output_gain": GATED_OUTPUT_FIELDS,
        }
        fields = fields_by_operation[operation]
        eligible = sorted(
            [
                module
                for module in parent
                if str(module.get("module_type", "")) == "generic"
                and str(
                    dict(module.get("params", {})).get(
                        "live_operator", "bounded_scalar_integrator_v1"
                    )
                )
                == GATED_OPERATOR
            ],
            key=_gated_module_sort_key,
        )
        if not eligible:
            telemetry["status"] = "ineligible"
            telemetry["reason"] = "no_gated_module"
            return GatedRecurrentMutationResult(parent, telemetry)
        try:
            module_index, module_fraction = nested_interval_choice(
                d1,
                len(eligible),
                field="gated_mutation.module",
            )
            field_index, value_fraction = nested_interval_choice(
                module_fraction,
                len(fields),
                field="gated_mutation.field",
            )
        except GatedRecordError as exc:
            telemetry["reason"] = f"invalid_decomposition:{exc}"
            return GatedRecurrentMutationResult(parent, telemetry)
        selected = eligible[module_index]
        selected_field = str(fields[field_index])
        lower, upper = GATED_PARAMETER_BOUNDS[selected_field]
        bounded_value = f32(
            f32(lower, field=f"{selected_field}.lower")
            + f32(
                value_fraction * f32(upper - lower, field=f"{selected_field}.span"),
                field=f"{selected_field}.offset",
            ),
            field=f"{selected_field}.value",
        )
        bounded_value = float(min(float(upper), max(float(lower), bounded_value)))
        selected_instance = str(selected.get("module_instance_id", ""))
        selected_local = str(selected.get("module_id", ""))
        parent_index = next(
            index
            for index, module in enumerate(candidate)
            if str(module.get("module_instance_id", "")) == selected_instance
            and str(module.get("module_id", "")) == selected_local
        )
        proposal = deepcopy(selected)
        params = deepcopy(dict(proposal.get("params", {})))
        gated = deepcopy(dict(params[GATED_RECORD_KEY]))
        before = float(gated[selected_field])
        gated[selected_field] = bounded_value
        params[GATED_RECORD_KEY] = gated
        proposal["params"] = normalize_gated_module_params(params)
        telemetry["selected_homology_id"] = str(selected.get("homology_id", ""))
        telemetry["selected_module_instance_id"] = selected_instance
        telemetry["selected_field"] = selected_field
        telemetry["value_before"] = before
        telemetry["value_after"] = bounded_value
        if bounded_value == before:
            telemetry["status"] = "noop"
            telemetry["reason"] = "bounded_value_unchanged"
            return GatedRecurrentMutationResult(parent, telemetry)
        candidate[parent_index] = proposal

    try:
        for module in candidate:
            params = normalize_gated_module_params(module.get("params", {}))
            if (
                str(params.get("live_operator", "")) == GATED_OPERATOR
                and str(module.get("module_type", "")) != "generic"
            ):
                raise ValueError(
                    "ESCAP7_GATED_OPERATOR_REQUIRES_GENERIC_MODULE "
                    f"module_id={module.get('module_id', '')}"
                )
    except (GatedRecordError, TypeError, ValueError) as exc:
        telemetry["reason"] = f"invalid_proposal:{type(exc).__name__}:{exc}"
        return GatedRecurrentMutationResult(parent, telemetry)
    telemetry["status"] = "accepted"
    telemetry["reason"] = "proposal_valid"
    telemetry["parent_preserved"] = True
    return GatedRecurrentMutationResult(candidate, telemetry)


def apply_gated_recurrent_operator_attempt(
    modules: list[dict[str, Any]],
    *,
    rng: Any,
    enabled: bool,
) -> GatedRecurrentMutationResult:
    """Consume exactly two supplied draws for every ESCAP-7 outcome."""

    draws = _take_draws(rng, GATED_RECURRENT_OPERATOR_DRAWS_PER_ATTEMPT)
    return mutate_gated_recurrent_operator(
        modules,
        draws[0],
        draws[1],
        enabled=bool(enabled),
    )


def _normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return canonical_payload_from_genome_dict(payload)


def _choose_index(draw: float, n: int) -> int:
    if n <= 0:
        return 0
    x = int(float(draw) * float(n))
    if x >= n:
        x = n - 1
    if x < 0:
        x = 0
    return int(x)


def _set_result_reason(context: dict[str, Any], reason: str) -> None:
    context["result_reason"] = str(reason)


def _native_identity_for_operator(
    context: dict[str, Any],
    *,
    operator_role: str,
    parent_module: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return new_native_identity(
        context=context.get("identity_context", {}),
        operator_id=str(context.get("operator_id", "")),
        attempt_index=int(context.get("attempt_index", 0)),
        operator_role=str(operator_role),
        parent_module=parent_module,
        inherited_homology=False,
    )


def _op_duplicate_with_decay(
    payload: dict[str, Any], draws: list[float], context: dict[str, Any]
) -> tuple[dict[str, Any], bool]:
    modules = sorted(list(payload.get("modules", [])), key=_mod_sort_key)
    if not modules:
        if not bool(context.get("bootstrap_enabled", False)):
            _set_result_reason(context, "bootstrap_disabled_empty_structure")
            return payload, False
        new_mid = _next_id(set(), "m")
        identity = _native_identity_for_operator(context, operator_role="empty_founder_first_module")
        payload["modules"] = [
            {
                "module_id": new_mid,
                "module_type": "generic",
                "params": {"expression_scale": canon_float(0.0, field="c1.bootstrap.expression_scale")},
                "io": {
                    "inputs": [{"slot_id": "in", "value_type": "scalar"}],
                    "outputs": [{"slot_id": "out", "value_type": "scalar"}],
                    "caps": {"max_fanin": 6, "max_fanout": 6},
                },
                **identity,
                "latent_pos": list(identity["topology_pos"]),
            }
        ]
        _set_result_reason(context, "bootstrap_first_module_created")
        return payload, True
    src_idx = _choose_index(draws[0], len(modules))
    src = deepcopy(modules[src_idx])
    decay = 0.75 + 0.20 * float(draws[1])  # [0.75, 0.95]
    jitter = 0.90 + 0.20 * float(draws[2])  # [0.90, 1.10]
    params = deepcopy(dict(src.get("params", {})))
    for k, v in list(params.items()):
        if isinstance(v, (int, float)):
            params[k] = canon_float(float(v) * decay * jitter, field=f"c1.dup.params.{k}")
    gated_duplicate = str(params.get("live_operator", "")) == GATED_OPERATOR
    if gated_duplicate:
        # Newborn lifetime state/output/latches are allocated elsewhere from
        # the new identity.  The hereditary copy is deep and public-output
        # neutral without consuming another draw.
        params = normalize_gated_module_params(params)
        gated_record = deepcopy(dict(params[GATED_RECORD_KEY]))
        gated_record["output_gain"] = 0.0
        params[GATED_RECORD_KEY] = gated_record
    elif (
        bool(context.get("sustainable_growth_enabled", False))
        and str(src.get("module_type", "")) == "generic"
    ):
        # ESCAP-8 makes integrator duplication exact-output-neutral without
        # adding a draw.  This branch is feature-gated so ESCAP-7 disabled
        # equivalence retains the historical decay/jitter payload exactly.
        params["live_operator"] = "bounded_scalar_integrator_v1"
        params["live_output_gain"] = 0.0
    existing_ids = {str(m.get("module_id", "")) for m in modules}
    new_mid = _next_id(existing_ids, "m")
    src["module_id"] = new_mid
    src["params"] = params
    identity = _native_identity_for_operator(
        context,
        operator_role="duplicated_module",
        parent_module=modules[src_idx],
    )
    src.update(identity)
    src["latent_pos"] = list(identity["topology_pos"])
    payload["modules"] = modules + [src]
    _set_result_reason(
        context,
        "gated_module_duplicated_neutral_output"
        if gated_duplicate
        else "bootstrap_second_module_created"
        if bool(context.get("bootstrap_enabled", False)) and len(modules) == 1
        else "module_duplicated_new_locus",
    )
    return payload, True


def _op_gate_drift(
    payload: dict[str, Any], draws: list[float], context: dict[str, Any]
) -> tuple[dict[str, Any], bool]:
    conns = sorted(list(payload.get("connections", [])), key=_conn_sort_key)
    if not conns:
        _set_result_reason(context, "requires_existing_connection")
        return payload, False
    gate_idxs = [i for i, c in enumerate(conns) if str(c.get("value_type", "")) == "gate"]
    pool = gate_idxs if gate_idxs else list(range(len(conns)))
    idx = pool[_choose_index(draws[0], len(pool))]
    delta = (float(draws[1]) - 0.5) * 0.40  # [-0.2, +0.2]
    c = dict(conns[idx])
    w = float(c.get("weight", 1.0) or 1.0)
    c["weight"] = canon_float(min(2.0, max(0.2, w + delta)), field="c1.gate_drift.weight")
    conns[idx] = c
    payload["connections"] = conns
    return payload, True


def _op_bounded_rewire(
    payload: dict[str, Any], draws: list[float], context: dict[str, Any]
) -> tuple[dict[str, Any], bool]:
    modules = sorted(list(payload.get("modules", [])), key=_mod_sort_key)
    conns = sorted(list(payload.get("connections", [])), key=_conn_sort_key)
    if len(modules) < 2:
        _set_result_reason(context, "requires_two_modules")
        return payload, False
    if not conns:
        if not bool(context.get("bootstrap_enabled", False)):
            _set_result_reason(context, "bootstrap_disabled_requires_existing_connection")
            return payload, False
        compatible: list[tuple[str, str, str, str, str]] = []
        for src in modules:
            src_io = src.get("io", {}) if isinstance(src.get("io", {}), dict) else {}
            outputs = src_io.get("outputs", []) if isinstance(src_io.get("outputs", []), list) else []
            for dst in modules:
                if str(src.get("module_id", "")) == str(dst.get("module_id", "")):
                    continue
                dst_io = dst.get("io", {}) if isinstance(dst.get("io", {}), dict) else {}
                inputs = dst_io.get("inputs", []) if isinstance(dst_io.get("inputs", []), list) else []
                for out_slot in outputs:
                    for in_slot in inputs:
                        if not isinstance(out_slot, dict) or not isinstance(in_slot, dict):
                            continue
                        value_type = str(out_slot.get("value_type", "scalar"))
                        if value_type != str(in_slot.get("value_type", "scalar")):
                            continue
                        compatible.append(
                            (
                                str(src.get("module_id", "")),
                                str(dst.get("module_id", "")),
                                str(out_slot.get("slot_id", "out")),
                                str(in_slot.get("slot_id", "in")),
                                value_type,
                            )
                        )
        compatible.sort()
        if not compatible:
            _set_result_reason(context, "bootstrap_no_compatible_module_slots")
            return payload, False
        edge = compatible[_choose_index(draws[0], len(compatible))]
        payload["connections"] = [
            {
                "connection_id": _next_id(set(), "c"),
                "src_module_id": edge[0],
                "dst_module_id": edge[1],
                "src_slot": edge[2],
                "dst_slot": edge[3],
                "value_type": edge[4],
                "channel": "c1_bootstrap",
                "weight": canon_float(0.0, field="c1.bootstrap.connection_weight"),
            }
        ]
        _set_result_reason(context, "bootstrap_first_connection_created")
        return payload, True
    cidx = _choose_index(draws[0], len(conns))
    didx = _choose_index(draws[1], len(modules))
    c = dict(conns[cidx])
    old_dst = str(c.get("dst_module_id", ""))
    new_dst = str(modules[didx].get("module_id", old_dst))
    if new_dst == old_dst and len(modules) > 1:
        didx = (didx + 1) % len(modules)
        new_dst = str(modules[didx].get("module_id", old_dst))
    c["dst_module_id"] = new_dst
    c["weight"] = canon_float(0.50 + float(draws[2]), field="c1.rewire.weight")  # [0.5, 1.5]
    conns[cidx] = c
    payload["connections"] = conns
    return payload, True


def _op_edge_split_insertion(
    payload: dict[str, Any], draws: list[float], context: dict[str, Any]
) -> tuple[dict[str, Any], bool]:
    modules = sorted(list(payload.get("modules", [])), key=_mod_sort_key)
    conns = sorted(list(payload.get("connections", [])), key=_conn_sort_key)
    if not conns:
        _set_result_reason(context, "requires_existing_connection")
        return payload, False
    cidx = _choose_index(draws[0], len(conns))
    target = dict(conns[cidx])
    existing_mod_ids = {str(m.get("module_id", "")) for m in modules}
    existing_conn_ids = {str(c.get("connection_id", "")) for c in conns}
    new_mid = _next_id(existing_mod_ids, "m")
    new_m = {
        "module_id": new_mid,
        "module_type": "generic",
        "params": {"expression_scale": canon_float(0.8 + 0.4 * float(draws[1]), field="c1.split.expression_scale")},
        "io": {
            "inputs": [{"slot_id": "in", "value_type": "scalar"}],
            "outputs": [{"slot_id": "out", "value_type": "scalar"}],
            "caps": {"max_fanin": 6, "max_fanout": 6},
        },
    }
    parent_module = next(
        (m for m in modules if str(m.get("module_id", "")) == str(target.get("src_module_id", ""))),
        modules[0] if modules else None,
    )
    identity = _native_identity_for_operator(
        context,
        operator_role="edge_split_inserted_module",
        parent_module=parent_module,
    )
    new_m.update(identity)
    new_m["latent_pos"] = list(identity["topology_pos"])
    c1_id = _next_id(existing_conn_ids, "c")
    existing_conn_ids.add(c1_id)
    c2_id = _next_id(existing_conn_ids, "c")
    c1 = {
        "connection_id": c1_id,
        "src_module_id": str(target.get("src_module_id", "")),
        "dst_module_id": new_mid,
        "src_slot": str(target.get("src_slot", "out")),
        "dst_slot": "in",
        "value_type": "scalar",
        "channel": "c1_split",
        "weight": canon_float(0.5 + 0.5 * float(draws[1]), field="c1.split.w1"),
    }
    c2 = {
        "connection_id": c2_id,
        "src_module_id": new_mid,
        "dst_module_id": str(target.get("dst_module_id", "")),
        "src_slot": "out",
        "dst_slot": str(target.get("dst_slot", "in")),
        "value_type": "scalar",
        "channel": "c1_split",
        "weight": canon_float(0.5 + 0.5 * float(draws[2]), field="c1.split.w2"),
    }
    out_conns = [c for i, c in enumerate(conns) if i != cidx]
    out_conns.extend([c1, c2])
    payload["modules"] = modules + [new_m]
    payload["connections"] = out_conns
    return payload, True


def _op_motif_duplication(
    payload: dict[str, Any], draws: list[float], context: dict[str, Any]
) -> tuple[dict[str, Any], bool]:
    modules = sorted(list(payload.get("modules", [])), key=_mod_sort_key)
    if len(modules) < 2:
        _set_result_reason(context, "requires_two_modules")
        return payload, False
    start = _choose_index(draws[0], len(modules))
    idxs = [start, (start + 1) % len(modules)]
    existing_mod_ids = {str(m.get("module_id", "")) for m in modules}
    new_modules: list[dict[str, Any]] = []
    gated_duplicate_count = 0
    id_map: dict[str, str] = {}
    for j, mi in enumerate(idxs):
        src = deepcopy(modules[mi])
        old = str(src.get("module_id", ""))
        new_mid = _next_id(existing_mod_ids, "m")
        existing_mod_ids.add(new_mid)
        id_map[old] = new_mid
        src["module_id"] = new_mid
        identity = _native_identity_for_operator(
            context,
            operator_role=f"motif_duplicate_{j}",
            parent_module=modules[mi],
        )
        src.update(identity)
        src["latent_pos"] = list(identity["topology_pos"])
        decay = 0.80 + 0.15 * float(draws[1 + (j % 2)])
        params = deepcopy(dict(src.get("params", {})))
        for k, v in list(params.items()):
            if isinstance(v, (int, float)):
                params[k] = canon_float(float(v) * decay, field=f"c1.motif.params.{k}")
        if str(params.get("live_operator", "")) == GATED_OPERATOR:
            params = normalize_gated_module_params(params)
            gated_record = deepcopy(dict(params[GATED_RECORD_KEY]))
            gated_record["output_gain"] = 0.0
            params[GATED_RECORD_KEY] = gated_record
            gated_duplicate_count += 1
        elif (
            bool(context.get("sustainable_growth_enabled", False))
            and str(src.get("module_type", "")) == "generic"
        ):
            params["live_operator"] = "bounded_scalar_integrator_v1"
            params["live_output_gain"] = 0.0
        src["params"] = params
        new_modules.append(src)
    conns = sorted(list(payload.get("connections", [])), key=_conn_sort_key)
    existing_conn_ids = {str(c.get("connection_id", "")) for c in conns}
    new_conns: list[dict[str, Any]] = []
    for c in conns:
        src = str(c.get("src_module_id", ""))
        dst = str(c.get("dst_module_id", ""))
        if src in id_map and dst in id_map:
            cc = dict(c)
            cc["connection_id"] = _next_id(existing_conn_ids, "c")
            existing_conn_ids.add(str(cc["connection_id"]))
            cc["src_module_id"] = id_map[src]
            cc["dst_module_id"] = id_map[dst]
            new_conns.append(cc)
    payload["modules"] = modules + new_modules
    payload["connections"] = conns + new_conns
    if gated_duplicate_count:
        _set_result_reason(
            context,
            f"gated_motif_duplicated_neutral_output_count_{gated_duplicate_count}",
        )
    return payload, True


def _op_promoter_expression_mutation(
    payload: dict[str, Any], draws: list[float], context: dict[str, Any]
) -> tuple[dict[str, Any], bool]:
    modules = sorted(list(payload.get("modules", [])), key=_mod_sort_key)
    if not modules:
        _set_result_reason(context, "requires_existing_module")
        return payload, False
    midx = _choose_index(draws[0], len(modules))
    m = deepcopy(modules[midx])
    params = deepcopy(dict(m.get("params", {})))
    params["expression_scale"] = canon_float(0.5 + float(draws[1]), field="c1.promoter.expression_scale")  # [0.5, 1.5]
    is_gated = str(params.get("live_operator", "")) == GATED_OPERATOR
    if str(m.get("module_type", "")) == "generic" and not is_gated:
        # Reuse the existing fixed two-draw promoter budget. The shared draw
        # deterministically parameterizes the bounded ESCAP-3 operator without
        # adding a scheduler bucket or changing RNG consumption.
        x = float(draws[1])
        params["live_operator"] = "bounded_scalar_integrator_v1"
        # Draw 0 chooses the module. Draw 1 is split into a parameter selector
        # and within-bin magnitude, so repeated promoter events can vary the
        # three live parameters independently without increasing RNG cost.
        field_index = min(2, int(x * 3.0))
        magnitude = min(1.0, max(0.0, x * 3.0 - float(field_index)))
        if field_index == 0:
            params["live_retention"] = canon_float(0.05 + 0.90 * magnitude, field="c1.promoter.live_retention")
        elif field_index == 1:
            params["live_input_gain"] = canon_float(-1.0 + 2.0 * magnitude, field="c1.promoter.live_input_gain")
        else:
            params["live_output_gain"] = canon_float(0.05 + 0.95 * magnitude, field="c1.promoter.live_output_gain")
    m["params"] = params
    modules[midx] = m
    payload["modules"] = modules
    return payload, True


def _op_fusion_merge_compression(
    payload: dict[str, Any], draws: list[float], context: dict[str, Any]
) -> tuple[dict[str, Any], bool]:
    modules = sorted(list(payload.get("modules", [])), key=_mod_sort_key)
    if len(modules) < 2:
        _set_result_reason(context, "requires_two_modules")
        return payload, False
    aidx = _choose_index(draws[0], len(modules))
    bidx = _choose_index(draws[1], len(modules))
    if bidx == aidx:
        bidx = (bidx + 1) % len(modules)
    keep = str(modules[aidx].get("module_id", ""))
    drop = str(modules[bidx].get("module_id", ""))
    out_modules = [deepcopy(m) for m in modules if str(m.get("module_id", "")) != drop]
    out_conns: list[dict[str, Any]] = []
    for c in sorted(list(payload.get("connections", [])), key=_conn_sort_key):
        cc = deepcopy(c)
        if str(cc.get("src_module_id", "")) == drop:
            cc["src_module_id"] = keep
        if str(cc.get("dst_module_id", "")) == drop:
            cc["dst_module_id"] = keep
        if str(cc.get("src_module_id", "")) == str(cc.get("dst_module_id", "")):
            continue
        out_conns.append(cc)
    # Validate the complete proposal before committing it.  In particular,
    # deleting a prerequisite homology must be a classified, draw-preserving
    # no-op rather than a later uncaught canonicalization exception.
    try:
        from evolution.structural_development_v1 import (
            agent_has_developmental_records,
            normalize_agent_developmental_modules,
            validate_agent_developmental_graph,
        )

        validated_modules = normalize_agent_developmental_modules(out_modules)
        if agent_has_developmental_records(validated_modules):
            validate_agent_developmental_graph(validated_modules)
        for module in validated_modules:
            normalize_gated_module_params(module.get("params", {}))
        # Validate the complete temporary proposal, including routing records,
        # endpoints, capacity, and the live graph, before touching the caller's
        # payload.  Fusion can otherwise turn an acyclic A->B->C graph into a
        # two-node cycle while remapping the dropped node.
        proposal = deepcopy(payload)
        proposal["modules"] = validated_modules
        proposal["connections"] = out_conns
        proposal["b2_invalid_policy"] = "reject"
        canonical_proposal = canonical_payload_from_genome_dict(proposal)
        from evolution.structural_live_modules_v1 import _validate_graph

        live_modules = [
            dict(module)
            for module in list(canonical_proposal.get("modules", []) or [])
            if str(module.get("module_type", "")) == "generic"
        ]
        if live_modules:
            live_ids = {str(module.get("module_id", "")) for module in live_modules}
            live_connections = [
                dict(connection)
                for connection in list(canonical_proposal.get("connections", []) or [])
                if str(connection.get("src_module_id", "")) in live_ids
                or str(connection.get("dst_module_id", "")) in live_ids
            ]
            _validate_graph(live_modules, live_connections)
    except (GatedRecordError, RuntimeError, TypeError, ValueError) as exc:
        detail = str(exc)
        reason = (
            "fusion_dependency_rejected"
            if "ESCAP6_MISSING_PREREQUISITE_HOMOLOGY" in detail
            else "fusion_contract_rejected"
        )
        _set_result_reason(context, reason)
        context["fusion_rejection_detail"] = detail
        return payload, False
    payload["modules"] = validated_modules
    payload["connections"] = out_conns
    return payload, True


@dataclass(frozen=True)
class C1OperatorSpec:
    op_id: str
    name: str
    budget_label: str
    py_draws: int
    fn: Callable[[dict[str, Any], list[float], dict[str, Any]], tuple[dict[str, Any], bool]]


_CATALOG: tuple[C1OperatorSpec, ...] = (
    C1OperatorSpec("duplicate_with_decay", "duplicate_with_decay", "mutation.c1.operator.duplicate_with_decay", 3, _op_duplicate_with_decay),
    C1OperatorSpec("gate_drift", "gate_drift", "mutation.c1.operator.gate_drift", 2, _op_gate_drift),
    C1OperatorSpec("bounded_rewire", "bounded_rewire", "mutation.c1.operator.bounded_rewire", 3, _op_bounded_rewire),
    C1OperatorSpec("edge_split_insertion", "edge_split_insertion", "mutation.c1.operator.edge_split_insertion", 3, _op_edge_split_insertion),
    C1OperatorSpec("motif_duplication", "motif_duplication", "mutation.c1.operator.motif_duplication", 3, _op_motif_duplication),
    C1OperatorSpec("promoter_expression_mutation", "promoter_expression_mutation", "mutation.c1.operator.promoter_expression_mutation", 2, _op_promoter_expression_mutation),
    C1OperatorSpec("fusion_merge_compression", "fusion_merge_compression", "mutation.c1.operator.fusion_merge_compression", 2, _op_fusion_merge_compression),
)
_CATALOG_BY_ID: dict[str, C1OperatorSpec] = {c.op_id: c for c in _CATALOG}
_ORDERED_IDS: tuple[str, ...] = tuple(sorted(_CATALOG_BY_ID.keys()))
# R9 trace localized that even scalar-looking operators can depend on
# canonicalization side effects such as derived latent positions. Keep the
# skip list empty until an operator-specific proof is added.
_SCALAR_NORMALIZE_SKIP_SAFE_OPS: frozenset[str] = frozenset()
# Current C1 operators return ok=False only from early no-op guards before
# mutating payload structure. R15 may skip per-op normalization for those
# unchanged no-op results while preserving draws, usage counts, and final
# observability annotations.
_NOOP_NORMALIZE_SKIP_SAFE_OPS: frozenset[str] = frozenset(_ORDERED_IDS)
_CANONICAL_PAYLOAD_TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {
        "schema_version",
        "interface_schema_version",
        "b2_invalid_policy",
        "b2_caps",
        "b3_locality_schema_version",
        "b3_locality_max_connection_distance",
        "b3_distance_cost_scale",
        "b3_distance_cost_max",
        "b3_locality_cost_total",
        "b3_locality_mean_distance",
        "b3_locality_max_distance",
        "b3_locality_dropped_connections",
        "modules",
        "connections",
        "developmental_program",
        "genes",
    }
)


def _payload_digest(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _empty_structure_canonical_payload(raw: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    schema = str(raw.get("schema_version", "") or "")
    if schema and schema != MODULAR_GENOME_SCHEMA_VERSION:
        return None, "unsupported_schema"
    modules = raw.get("modules", None)
    connections = raw.get("connections", None)
    if not isinstance(modules, list) or len(modules) != 0:
        return None, "non_empty_or_missing_modules"
    if not isinstance(connections, list) or len(connections) != 0:
        return None, "non_empty_or_missing_connections"

    genes_obj = raw.get("genes", None)
    genes = dict(genes_obj) if isinstance(genes_obj, dict) else dict(raw)
    for key in _CANONICAL_PAYLOAD_TOP_LEVEL_KEYS:
        genes.pop(key, None)
    canon_genes = _canon_obj(genes, field="genes")
    if not isinstance(canon_genes, dict):
        return None, "malformed_genes"

    invalid_policy = str(raw.get("b2_invalid_policy", "clip") or "clip").strip().lower()
    resolved_caps = resolve_b2_caps(raw.get("b2_caps", None))
    return (
        {
            "schema_version": MODULAR_GENOME_SCHEMA_VERSION,
            "interface_schema_version": B2_INTERFACE_SCHEMA_VERSION,
            "b3_locality_schema_version": B3_LOCALITY_SCHEMA_VERSION,
            "genes": canon_genes,
            "modules": [],
            "connections": [],
            "developmental_program": canonicalize_developmental_program(raw.get("developmental_program", None)),
            "b2_invalid_policy": invalid_policy if invalid_policy in {"clip", "reject"} else "clip",
            "b2_caps": dict(resolved_caps),
            "b3_locality_max_connection_distance": canon_float(
                B3_FROZEN_LOCALITY["max_connection_distance"],
                field="b3.max_connection_distance",
            ),
            "b3_distance_cost_scale": canon_float(
                B3_FROZEN_LOCALITY["distance_cost_scale"],
                field="b3.distance_cost_scale",
            ),
            "b3_distance_cost_max": canon_float(
                B3_FROZEN_LOCALITY["distance_cost_max"],
                field="b3.distance_cost_max",
            ),
            "b3_locality_cost_total": canon_float(0.0, field="b3.locality_cost_total"),
            "b3_locality_mean_distance": canon_float(0.0, field="b3.locality_mean_distance"),
            "b3_locality_max_distance": canon_float(0.0, field="b3.locality_max_distance"),
            "b3_locality_dropped_connections": 0,
        },
        "",
    )


def _persist_c1_annotations(
    working: dict[str, Any],
    *,
    chosen_ids: list[str],
    applied: list[str],
    noop: int,
    usage_counts: dict[str, int],
    capacity_blocked_reason_counts: dict[str, int] | None = None,
    noop_reason_counts: dict[str, int] | None = None,
    structural_bootstrap_enabled: bool = False,
    structural_bootstrap_accepted_count: int = 0,
    structural_sustainable_growth_enabled: bool = False,
    sustainable_growth_attempt_row: dict[str, Any] | None = None,
    blocked_duplication_attempts: int = 0,
) -> dict[str, Any]:
    genes = dict(working.get("genes", {})) if isinstance(working.get("genes", {}), dict) else {}
    genes["c1_catalog_version"] = str(C1_OPERATOR_CATALOG_VERSION)
    genes["c1_operator_schedule"] = ",".join(str(x) for x in chosen_ids)
    genes["c1_operator_applied"] = ",".join(str(x) for x in applied)
    genes["c1_operator_noop_count"] = int(noop)
    genes["c1_operator_budget_status"] = "ok"
    reason_counts = {
        str(reason): int(count)
        for reason, count in sorted(dict(capacity_blocked_reason_counts or {}).items())
        if int(count) > 0
    }
    genes["c1_capacity_blocked_count"] = int(sum(reason_counts.values()))
    genes["c1_capacity_blocked_reason_counts_json"] = json.dumps(
        reason_counts,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    noop_reasons = {
        str(reason): int(count)
        for reason, count in sorted(dict(noop_reason_counts or {}).items())
        if int(count) > 0
    }
    genes["c1_noop_reason_counts_json"] = json.dumps(
        noop_reasons,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    genes["structural_bootstrap_enabled"] = int(bool(structural_bootstrap_enabled))
    genes["structural_bootstrap_accepted_count"] = int(structural_bootstrap_accepted_count)
    for op_id in _ORDERED_IDS:
        genes[f"c1_op_count__{op_id}"] = int(usage_counts.get(op_id, 0))
    if bool(structural_sustainable_growth_enabled):
        row = dict(sustainable_growth_attempt_row or {})
        operation = str(row.get("operation", "") or "")
        status = str(row.get("status", "") or "")
        blocked_prune = int(
            operation == "prune_eligible_module" and status != "accepted"
        )
        prior_duplication = int(
            genes.get("escap8_blocked_duplication_attempt_count", 0) or 0
        )
        prior_prune = int(genes.get("escap8_blocked_prune_attempt_count", 0) or 0)
        prior_attempts = int(genes.get("escap8_growth_attempt_count", 0) or 0)
        working_caps = resolve_b2_caps(working.get("b2_caps", None))
        at_module_cap = int(
            len(list(working.get("modules", []) or []))
            >= int(working_caps["max_modules_per_agent"])
        )
        pressure_now = bool(int(blocked_duplication_attempts) > 0 or at_module_cap)
        prior_streak = int(genes.get("escap8_cap_pressure_streak", 0) or 0)
        genes.update(
            {
                "escap8_growth_attempt_count": prior_attempts + 1,
                "escap8_growth_last_operation": operation,
                "escap8_growth_last_status": status,
                "escap8_growth_last_reason": str(row.get("reason", "") or ""),
                "escap8_blocked_duplication_attempt_count": (
                    prior_duplication + int(blocked_duplication_attempts)
                ),
                "escap8_blocked_prune_attempt_count": prior_prune + blocked_prune,
                "escap8_cap_contact_count": int(
                    genes.get("escap8_cap_contact_count", 0) or 0
                )
                + at_module_cap,
                "escap8_cap_pressure_streak": prior_streak + 1 if pressure_now else 0,
            }
        )
        specialization = row.get("specialization_difference")
        if isinstance(specialization, dict):
            genes["escap8_specialization_metrics_json"] = json.dumps(
                specialization,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
    working["genes"] = genes
    return working


def operator_catalog() -> tuple[C1OperatorSpec, ...]:
    return _CATALOG


def structural_operator_dependency_graph(*, bootstrap_enabled: bool) -> dict[str, Any]:
    """Machine-readable reachability graph for the frozen C1 catalogue."""

    enabled = bool(bootstrap_enabled)
    operators = {
        "duplicate_with_decay": {
            "requires": "zero modules when bootstrap is enabled; otherwise one module",
            "minimum_modules": 0 if enabled else 1,
            "requires_existing_connection": False,
            "bootstrap_override": "empty_to_first_module" if enabled else "disabled",
            "eligible_from_empty": enabled,
            "transition": "0->1 modules (bootstrap) or N->N+1 modules",
            "draws": 3,
        },
        "promoter_expression_mutation": {
            "requires": "one module",
            "minimum_modules": 1,
            "requires_existing_connection": False,
            "bootstrap_override": "none",
            "eligible_from_empty": False,
            "transition": "functional scalar mutation only",
            "draws": 2,
        },
        "motif_duplication": {
            "requires": "two modules",
            "minimum_modules": 2,
            "requires_existing_connection": False,
            "bootstrap_override": "none",
            "eligible_from_empty": False,
            "transition": "N->N+2 modules plus copied internal relationships",
            "draws": 3,
        },
        "fusion_merge_compression": {
            "requires": "two modules",
            "minimum_modules": 2,
            "requires_existing_connection": False,
            "bootstrap_override": "none",
            "eligible_from_empty": False,
            "transition": "N->N-1 modules",
            "draws": 2,
        },
        "bounded_rewire": {
            "requires": "two modules when bootstrap is enabled; otherwise an existing connection",
            "minimum_modules": 2 if enabled else 1,
            "requires_existing_connection": not enabled,
            "bootstrap_override": "module_pair_to_first_connection" if enabled else "disabled",
            "eligible_from_empty": False,
            "transition": "0->1 connections (bootstrap) or rewires one connection",
            "draws": 3,
        },
        "gate_drift": {
            "requires": "one connection",
            "minimum_modules": 1,
            "requires_existing_connection": True,
            "bootstrap_override": "none",
            "eligible_from_empty": False,
            "transition": "connection weight mutation",
            "draws": 2,
        },
        "edge_split_insertion": {
            "requires": "one connection",
            "minimum_modules": 1,
            "requires_existing_connection": True,
            "bootstrap_override": "none",
            "eligible_from_empty": False,
            "transition": "adds one module and replaces one edge with two",
            "draws": 3,
        },
    }
    return {
        "schema_version": "c1_structural_operator_dependency_graph_v1",
        "catalog_version": C1_OPERATOR_CATALOG_VERSION,
        "bootstrap_enabled": enabled,
        "operators": {key: operators[key] for key in sorted(operators)},
        "requirement_classes": {
            "eligible_from_empty": ["duplicate_with_decay"] if enabled else [],
            "requires_at_least_one_module": [
                "duplicate_with_decay",
                "promoter_expression_mutation",
            ] if not enabled else ["promoter_expression_mutation"],
            "requires_two_or_more_modules": [
                *(["bounded_rewire"] if enabled else []),
                "fusion_merge_compression",
                "motif_duplication",
            ],
            "requires_existing_connection": (
                ["edge_split_insertion", "gate_drift"]
                if enabled
                else ["bounded_rewire", "edge_split_insertion", "gate_drift"]
            ),
        },
        "shortest_legal_path": (
            [
                "empty",
                "duplicate_with_decay:first_neutral_module",
                "duplicate_with_decay:second_neutral_module",
                "bounded_rewire:first_zero_weight_connection",
                "minimal_connected_structure",
            ]
            if enabled
            else []
        ),
        "impossible_transitions": [
            "empty_directly_to_first_connection",
            "one_module_directly_to_first_connection",
            *(
                []
                if enabled
                else [
                    "empty_to_first_module",
                    "module_pair_without_connection_to_first_connection",
                ]
            ),
        ],
    }


def _mix_weights() -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    for op_id in _ORDERED_IDS:
        out.append((op_id, float(C1_FROZEN_OPERATOR_MIX.get(op_id, 0.0))))
    return out


def schedule_operator_ids(
    *,
    rng: Any,
    max_ops: int,
    rng_manager: Any = None,
) -> list[str]:
    n_ops = max(0, int(max_ops))
    weights = _mix_weights()
    total = sum(max(0.0, w) for _, w in weights)
    if total <= 0.0:
        if c1_strict_enabled():
            _det_fail("DET_C1_OPERATOR_MIX_INVALID", "reason=non_positive_total_weight")
        return []

    expected = None
    if rng_manager is not None:
        expected = rng_manager.expected_delta_for_boundary(
            "mutation.c1.schedule_mix",
            ctx={"schedule_draws": int(n_ops)},
        )
    scope = rng_manager.budget_scope("mutation.c1.schedule_mix", expected_delta=expected) if rng_manager is not None else None
    out: list[str] = []
    if scope is None:
        draws = _take_draws(rng, n_ops)
    else:
        with scope:
            draws = _take_draws(rng, n_ops)
    for d in draws:
        x = float(d) * float(total)
        acc = 0.0
        chosen = weights[-1][0]
        for op_id, w in weights:
            ww = max(0.0, float(w))
            acc += ww
            if x <= acc:
                chosen = op_id
                break
        out.append(chosen)
    return out


def apply_operator_pipeline(
    *,
    payload: dict[str, Any],
    rng: Any,
    rng_manager: Any = None,
    max_ops: int = C1_FROZEN_MAX_OPS_PER_OFFSPRING,
    invalid_operator_policy: str = C1_FROZEN_INVALID_OPERATOR_POLICY,
    payload_is_canonical: bool = False,
    skip_final_normalize: bool = False,
    skip_scalar_op_normalize: bool = False,
    skip_noop_normalize: bool = False,
    empty_structure_fastpath: bool = False,
    empty_structure_fastpath_fail_closed: bool = False,
    trace_records: list[dict[str, Any]] | None = None,
    trace_payload_identity_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    emit_final_payload_digest: bool = False,
    trusted_canonical_input: bool = False,
    structural_bootstrap_enable: bool = False,
    structural_identity_context: dict[str, Any] | None = None,
    observation_routing_mutation_enable: bool = False,
    observation_routing_operator_attempts: int = 0,
    output_routing_mutation_enable: bool = False,
    output_routing_operator_attempts: int = 0,
    structural_development_mutation_enable: bool = False,
    structural_development_operator_attempts: int = 0,
    structural_gated_recurrent_mutation_enable: bool = False,
    structural_gated_recurrent_operator_attempts: int = 0,
    structural_sustainable_growth_enable: bool = False,
    structural_sustainable_growth_operator_attempts: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload_is_canonical = bool(payload_is_canonical or trusted_canonical_input)
    raw_payload = dict(payload)
    policy = str(invalid_operator_policy or C1_FROZEN_INVALID_OPERATOR_POLICY).strip().lower()
    if c1_strict_enabled():
        policy = "reject"
    if policy == "reject":
        raw_payload["b2_invalid_policy"] = "reject"
    bootstrap_enabled = bool(structural_bootstrap_enable)
    empty_structure_fastpath_requested = bool(empty_structure_fastpath)
    empty_structure_fastpath_enabled = bool(
        empty_structure_fastpath
        and not bootstrap_enabled
        and not (
            bool(observation_routing_mutation_enable)
            and int(observation_routing_operator_attempts or 0) > 0
        )
        and not (
            bool(output_routing_mutation_enable)
            and int(output_routing_operator_attempts or 0) > 0
        )
        and not (
            bool(structural_development_mutation_enable)
            and int(structural_development_operator_attempts or 0) > 0
        )
        and not (
            bool(structural_gated_recurrent_mutation_enable)
            and int(structural_gated_recurrent_operator_attempts or 0) > 0
        )
        and not (
            bool(structural_sustainable_growth_enable)
            and int(structural_sustainable_growth_operator_attempts or 0) > 0
        )
    )
    empty_structure_fastpath_hits = 0
    empty_structure_fastpath_misses = 0
    empty_structure_fastpath_miss_reason = ""
    if empty_structure_fastpath_enabled:
        if not bool(payload_is_canonical):
            empty_structure_fastpath_miss_reason = "payload_not_canonical_boundary"
        elif not bool(skip_final_normalize):
            empty_structure_fastpath_miss_reason = "final_normalize_required"
        elif trace_records is not None:
            empty_structure_fastpath_miss_reason = "trace_enabled"
        else:
            fast_payload, miss_reason = _empty_structure_canonical_payload(raw_payload)
            if fast_payload is None:
                empty_structure_fastpath_miss_reason = str(miss_reason or "unsupported_payload")
            else:
                empty_structure_fastpath_hits = 1

    if empty_structure_fastpath_enabled and empty_structure_fastpath_hits == 0:
        empty_structure_fastpath_misses = 1
        if bool(empty_structure_fastpath_fail_closed):
            _det_fail(
                "DET_C1_EMPTY_STRUCTURE_FASTPATH_UNSUPPORTED",
                f"reason={empty_structure_fastpath_miss_reason or 'unsupported_payload'}",
            )

    if empty_structure_fastpath_hits:
        working = fast_payload
    else:
        working = raw_payload if bool(payload_is_canonical) else _normalize_payload(raw_payload)
    chosen_ids = schedule_operator_ids(rng=rng, max_ops=max_ops, rng_manager=rng_manager)
    usage_counts = {op_id: 0 for op_id in _ORDERED_IDS}
    applied: list[str] = []
    noop = 0
    scalar_normalize_skipped = 0
    noop_normalize_skipped = 0
    normalize_calls = 0
    capacity_blocked_reason_counts: dict[str, int] = {}
    noop_reason_counts: dict[str, int] = {}
    operator_reason_counts: dict[str, int] = {}
    blocked_duplication_attempts = 0
    structural_bootstrap_accepted_count = 0
    identity_context = dict(structural_identity_context or {})
    child_raw = identity_context.get("child_agent_id", -1)
    child_agent_id = -1 if child_raw is None or child_raw == "" else int(child_raw)
    if child_agent_id < 0 and not str(
        identity_context.get("lineage_token", "") or ""
    ):
        # Direct unit/diagnostic calls lack population child IDs.  Keep them
        # deterministic without consuming RNG; live reproduction passes the
        # campaign-scoped context explicitly.
        identity_context["lineage_token"] = f"payload:{_payload_digest(raw_payload)}"

    if empty_structure_fastpath_hits:
        for op_id in chosen_ids:
            spec = _CATALOG_BY_ID.get(str(op_id))
            if spec is None:
                if policy == "reject":
                    _det_fail("DET_C1_UNKNOWN_OPERATOR", f"operator_id={op_id}")
                noop += 1
                continue
            usage_counts[spec.op_id] = int(usage_counts.get(spec.op_id, 0)) + 1
            expected = None
            if rng_manager is not None:
                expected = rng_manager.expected_delta_for_boundary(spec.budget_label, ctx={})
            scope = rng_manager.budget_scope(spec.budget_label, expected_delta=expected) if rng_manager is not None else None
            if scope is None:
                _take_draws(rng, int(spec.py_draws))
            else:
                with scope:
                    _take_draws(rng, int(spec.py_draws))
            applied.append(spec.op_id)
            noop += 1
            noop_normalize_skipped += 1

        telemetry = {
            "catalog_version": C1_OPERATOR_CATALOG_VERSION,
            "schedule_ids": list(chosen_ids),
            "applied_ids": list(applied),
            "usage_counts": dict(usage_counts),
            "noop_count": int(noop),
            "max_ops": int(max_ops),
            "invalid_operator_policy": str(policy),
            "budget_status": "ok",
            "normalize_calls": 0,
            "scalar_normalize_skipped": 0,
            "noop_normalize_skipped": int(noop_normalize_skipped),
            "scalar_normalize_skip_safe_ops": sorted(_SCALAR_NORMALIZE_SKIP_SAFE_OPS),
            "noop_normalize_skip_safe_ops": sorted(_NOOP_NORMALIZE_SKIP_SAFE_OPS),
            "empty_structure_fastpath_enabled": 1,
            "empty_structure_fastpath_requested": 1,
            "empty_structure_fastpath_hits": 1,
            "empty_structure_fastpath_misses": 0,
            "empty_structure_fastpath_miss_reason": "",
            "capacity_blocked_count": 0,
            "capacity_blocked_reason_counts": {},
            "noop_reason_counts": {"empty_structure_fastpath_noop": int(noop)},
            "operator_reason_counts": {},
            "structural_bootstrap_enabled": 0,
            "structural_bootstrap_accepted_count": 0,
        }
        working = _persist_c1_annotations(
            working,
            chosen_ids=chosen_ids,
            applied=applied,
            noop=noop,
            usage_counts=usage_counts,
            capacity_blocked_reason_counts={},
            noop_reason_counts={"empty_structure_fastpath_noop": int(noop)},
            structural_bootstrap_enabled=False,
            structural_bootstrap_accepted_count=0,
        )
        if bool(emit_final_payload_digest):
            telemetry["final_payload_digest"] = _payload_digest(working)
        return working, telemetry

    for attempt_index, op_id in enumerate(chosen_ids):
        spec = _CATALOG_BY_ID.get(str(op_id))
        if spec is None:
            if policy == "reject":
                _det_fail("DET_C1_UNKNOWN_OPERATOR", f"operator_id={op_id}")
            noop += 1
            continue
        usage_counts[spec.op_id] = int(usage_counts.get(spec.op_id, 0)) + 1
        blocked_reason = ""
        op_context: dict[str, Any] = {
            "bootstrap_enabled": bool(bootstrap_enabled),
            "identity_context": identity_context,
            "operator_id": str(spec.op_id),
            "attempt_index": int(attempt_index),
            "result_reason": "",
            "sustainable_growth_enabled": bool(structural_sustainable_growth_enable),
        }
        pre_working = dict(working)
        pre_digest = _payload_digest(pre_working) if trace_records is not None else ""
        expected = None
        if rng_manager is not None:
            expected = rng_manager.expected_delta_for_boundary(spec.budget_label, ctx={})
        scope = rng_manager.budget_scope(spec.budget_label, expected_delta=expected) if rng_manager is not None else None
        if scope is None:
            draws = _take_draws(rng, int(spec.py_draws))
            working, ok = spec.fn(working, draws, op_context)
        else:
            with scope:
                draws = _take_draws(rng, int(spec.py_draws))
                working, ok = spec.fn(working, draws, op_context)
        operator_reason = str(op_context.get("result_reason", "") or "")
        if operator_reason:
            operator_reason_counts[operator_reason] = int(operator_reason_counts.get(operator_reason, 0)) + 1
        post_digest = _payload_digest(working) if trace_records is not None else ""
        skipped_noop_normalize = bool(
            skip_noop_normalize
            and not ok
            and spec.op_id in _NOOP_NORMALIZE_SKIP_SAFE_OPS
            and isinstance(working.get("genes"), dict)
            and (trace_records is None or post_digest == pre_digest)
        )
        skipped_normalize = bool(
            skip_scalar_op_normalize
            and ok
            and spec.op_id in _SCALAR_NORMALIZE_SKIP_SAFE_OPS
        )
        if skipped_noop_normalize:
            noop_normalize_skipped += 1
            normalized_digest = post_digest
        elif skipped_normalize:
            scalar_normalize_skipped += 1
            normalized_digest = post_digest
        else:
            try:
                working = _normalize_payload(working)
                normalize_calls += 1
            except RuntimeError as e:
                msg = str(e)
                if ("DET_GENOME_B2_" in msg) or ("DET_GENOME_B3_" in msg):
                    # Deterministic fallback: treat structurally invalid operator output as no-op.
                    blocked_reason = classify_structural_capacity_error(msg)
                    capacity_blocked_reason_counts[blocked_reason] = (
                        int(capacity_blocked_reason_counts.get(blocked_reason, 0)) + 1
                    )
                    if spec.op_id in {"duplicate_with_decay", "motif_duplication"}:
                        blocked_duplication_attempts += 1
                    working = _normalize_payload(pre_working)
                    normalize_calls += 1
                    ok = False
                else:
                    raise
            normalized_digest = _payload_digest(working) if trace_records is not None else ""
        if trace_records is not None:
            pre_identity = trace_payload_identity_fn(pre_working) if trace_payload_identity_fn is not None else {}
            post_identity = trace_payload_identity_fn(working) if trace_payload_identity_fn is not None else {}
            normalized_identity = trace_payload_identity_fn(working) if trace_payload_identity_fn is not None else {}
            historical_changed = bool(pre_identity.get("graph_identity_digest", pre_digest) != normalized_identity.get("graph_identity_digest", normalized_digest))
            semantic_changed = bool(pre_identity.get("graph_semantic_digest", pre_digest) != normalized_identity.get("graph_semantic_digest", normalized_digest))
            status = "applied" if bool(ok) else ("capacity_blocked" if blocked_reason else "valid_noop")
            trace_records.append(
                {
                    "operator_id": str(spec.op_id),
                    "pre_payload_digest": str(pre_digest),
                    "post_payload_digest": str(post_digest),
                    "normalized_payload_digest": str(normalized_digest),
                    "pre_graph_identity_digest": str(pre_identity.get("graph_identity_digest", "")),
                    "post_graph_identity_digest": str(post_identity.get("graph_identity_digest", "")),
                    "normalized_graph_identity_digest": str(normalized_identity.get("graph_identity_digest", "")),
                    "pre_graph_semantic_digest": str(pre_identity.get("graph_semantic_digest", "")),
                    "post_graph_semantic_digest": str(post_identity.get("graph_semantic_digest", "")),
                    "normalized_graph_semantic_digest": str(normalized_identity.get("graph_semantic_digest", "")),
                    "pre_graph_supported": bool(pre_identity.get("graph_supported", False)),
                    "post_graph_supported": bool(post_identity.get("graph_supported", False)),
                    "normalized_graph_supported": bool(normalized_identity.get("graph_supported", False)),
                    "ok": bool(ok),
                    "operator_status": status,
                    "historical_changed": historical_changed,
                    "semantic_changed": semantic_changed,
                    "valid_noop": bool(status == "valid_noop" and not historical_changed and not semantic_changed),
                    "capacity_blocked_reason": str(blocked_reason),
                    "operator_reason": str(operator_reason),
                    "normalization_skipped": bool(skipped_normalize),
                    "noop_normalization_skipped": bool(skipped_noop_normalize),
                    "draw_count": int(spec.py_draws),
                }
            )
        applied.append(spec.op_id)
        if not ok:
            noop += 1
            reason = operator_reason or (f"capacity_blocked:{blocked_reason}" if blocked_reason else "unclassified_noop")
            noop_reason_counts[reason] = int(noop_reason_counts.get(reason, 0)) + 1
        elif operator_reason.startswith("bootstrap_") and operator_reason.endswith("_created"):
            structural_bootstrap_accepted_count += 1

    telemetry = {
        "catalog_version": C1_OPERATOR_CATALOG_VERSION,
        "schedule_ids": list(chosen_ids),
        "applied_ids": list(applied),
        "usage_counts": dict(usage_counts),
        "noop_count": int(noop),
        "max_ops": int(max_ops),
        "invalid_operator_policy": str(policy),
        "budget_status": "ok",
        "normalize_calls": int(normalize_calls),
        "scalar_normalize_skipped": int(scalar_normalize_skipped),
        "noop_normalize_skipped": int(noop_normalize_skipped),
        "scalar_normalize_skip_safe_ops": sorted(_SCALAR_NORMALIZE_SKIP_SAFE_OPS),
        "noop_normalize_skip_safe_ops": sorted(_NOOP_NORMALIZE_SKIP_SAFE_OPS),
        "empty_structure_fastpath_enabled": int(empty_structure_fastpath_enabled),
        "empty_structure_fastpath_requested": int(empty_structure_fastpath_requested),
        "empty_structure_fastpath_hits": int(empty_structure_fastpath_hits),
        "empty_structure_fastpath_misses": int(empty_structure_fastpath_misses),
        "empty_structure_fastpath_miss_reason": str(empty_structure_fastpath_miss_reason),
        "capacity_blocked_count": int(sum(capacity_blocked_reason_counts.values())),
        "capacity_blocked_reason_counts": dict(sorted(capacity_blocked_reason_counts.items())),
        "noop_reason_counts": dict(sorted(noop_reason_counts.items())),
        "operator_reason_counts": dict(sorted(operator_reason_counts.items())),
        "structural_bootstrap_enabled": int(bootstrap_enabled),
        "structural_bootstrap_accepted_count": int(structural_bootstrap_accepted_count),
    }

    # ESCAP-4 owns a distinct fixed-budget routing operator.  It is not part of
    # the frozen C1 catalogue/mix, so every existing profile retains identical
    # schedule probabilities and draws while this explicit gate is off.
    routing_attempts = int(observation_routing_operator_attempts or 0)
    if routing_attempts < 0 or routing_attempts > 2:
        _det_fail(
            "DET_ESCAP4_ROUTING_ATTEMPTS_INVALID",
            f"attempts={routing_attempts}",
        )
    if bool(observation_routing_mutation_enable) and routing_attempts > 0:
        from evolution.operators_routing_v1 import (
            ROUTING_OPERATOR_DRAWS,
        )

        routing_rows: list[dict[str, Any]] = []
        for _routing_attempt in range(routing_attempts):
            label = "mutation.c1.operator.observation_binding_mutation"
            expected = None
            if rng_manager is not None:
                expected = rng_manager.expected_delta_for_boundary(label, ctx={})
            scope = rng_manager.budget_scope(label, expected_delta=expected) if rng_manager is not None else None
            if scope is None:
                result = apply_observation_routing_operator_attempt(
                    list(working.get("modules", []) or []),
                    rng=rng,
                    enabled=True,
                )
            else:
                with scope:
                    result = apply_observation_routing_operator_attempt(
                        list(working.get("modules", []) or []),
                        rng=rng,
                        enabled=True,
                    )
            working["modules"] = list(result.modules)
            row = dict(result.telemetry)
            row["attempt_index"] = int(_routing_attempt)
            routing_rows.append(row)
        telemetry["observation_routing_mutation"] = {
            "enabled": True,
            "attempts": routing_attempts,
            "draws_per_attempt": ROUTING_OPERATOR_DRAWS,
            "attempt_rows": routing_rows,
            "accepted": int(sum(1 for row in routing_rows if row.get("status") == "accepted")),
        }

    # ESCAP-5 owns a second distinct fixed-budget lane.  Its placement after
    # ESCAP-4 is frozen, while both lanes remain outside the historical C1
    # catalogue and therefore add no schedule draws when disabled.
    output_attempts = int(output_routing_operator_attempts or 0)
    if output_attempts < 0 or output_attempts > 2:
        _det_fail(
            "DET_ESCAP5_OUTPUT_ROUTING_ATTEMPTS_INVALID",
            f"attempts={output_attempts}",
        )
    if bool(output_routing_mutation_enable) and output_attempts > 0:
        from evolution.operators_output_routing_v1 import OUTPUT_ROUTING_OPERATOR_DRAWS

        output_rows: list[dict[str, Any]] = []
        for output_attempt in range(output_attempts):
            label = "mutation.c1.operator.output_binding_mutation"
            expected = None
            if rng_manager is not None:
                expected = rng_manager.expected_delta_for_boundary(label, ctx={})
            scope = rng_manager.budget_scope(label, expected_delta=expected) if rng_manager is not None else None
            if scope is None:
                result = apply_output_routing_operator_attempt(
                    list(working.get("modules", []) or []),
                    rng=rng,
                    enabled=True,
                )
            else:
                with scope:
                    result = apply_output_routing_operator_attempt(
                        list(working.get("modules", []) or []),
                        rng=rng,
                        enabled=True,
                    )
            working["modules"] = list(result.modules)
            row = dict(result.telemetry)
            row["attempt_index"] = int(output_attempt)
            output_rows.append(row)
        telemetry["output_routing_mutation"] = {
            "enabled": True,
            "attempts": output_attempts,
            "draws_per_attempt": OUTPUT_ROUTING_OPERATOR_DRAWS,
            "attempt_rows": output_rows,
            "accepted": int(sum(1 for row in output_rows if row.get("status") == "accepted")),
        }

    # ESCAP-6 owns one fixed-budget lane after ESCAP-5.  It remains outside
    # the historical C1 catalogue, so all disabled schedules and draws stay
    # byte-for-byte on the old path.
    development_attempts = int(structural_development_operator_attempts or 0)
    if development_attempts < 0 or development_attempts > 1:
        _det_fail(
            "DET_ESCAP6_DEVELOPMENT_ATTEMPTS_INVALID",
            f"attempts={development_attempts}",
        )
    if bool(structural_development_mutation_enable) and development_attempts != 1:
        _det_fail(
            "DET_ESCAP6_DEVELOPMENT_ATTEMPTS_INVALID",
            f"enabled=1;attempts={development_attempts};expected=1",
        )
    if bool(structural_development_mutation_enable):
        from evolution.structural_development_v1 import DEVELOPMENT_OPERATOR_DRAWS_PER_ATTEMPT

        development_rows: list[dict[str, Any]] = []
        label = "mutation.c1.operator.developmental_expression_mutation"
        expected = None
        if rng_manager is not None:
            expected = rng_manager.expected_delta_for_boundary(label, ctx={})
        scope = rng_manager.budget_scope(label, expected_delta=expected) if rng_manager is not None else None
        if scope is None:
            result = apply_development_operator_attempt(
                list(working.get("modules", []) or []),
                rng=rng,
                enabled=True,
            )
        else:
            with scope:
                result = apply_development_operator_attempt(
                    list(working.get("modules", []) or []),
                    rng=rng,
                    enabled=True,
                )
        working["modules"] = list(result.modules)
        row = dict(result.telemetry)
        row["attempt_index"] = 0
        development_rows.append(row)
        telemetry["structural_development_mutation"] = {
            "enabled": True,
            "attempts": 1,
            "draws_per_attempt": DEVELOPMENT_OPERATOR_DRAWS_PER_ATTEMPT,
            "attempt_rows": development_rows,
            "accepted": int(row.get("status") == "accepted"),
        }

    # ESCAP-7 owns one distinct fixed-budget lane after ESCAP-6.  Disabled
    # callers never enter this branch and therefore retain the exact prior
    # schedule, payload, telemetry, and draw count.
    gated_attempts = int(structural_gated_recurrent_operator_attempts or 0)
    if gated_attempts < 0 or gated_attempts > GATED_RECURRENT_OPERATOR_ATTEMPTS:
        _det_fail(
            "DET_ESCAP7_GATED_RECURRENT_ATTEMPTS_INVALID",
            f"attempts={gated_attempts}",
        )
    if bool(structural_gated_recurrent_mutation_enable) and gated_attempts != 1:
        _det_fail(
            "DET_ESCAP7_GATED_RECURRENT_ATTEMPTS_INVALID",
            f"enabled=1;attempts={gated_attempts};expected=1",
        )
    if bool(structural_gated_recurrent_mutation_enable):
        label = "mutation.c1.operator.gated_scalar_recurrent_mutation"
        expected = None
        if rng_manager is not None:
            expected = rng_manager.expected_delta_for_boundary(label, ctx={})
        scope = (
            rng_manager.budget_scope(label, expected_delta=expected)
            if rng_manager is not None
            else None
        )
        if scope is None:
            result = apply_gated_recurrent_operator_attempt(
                list(working.get("modules", []) or []),
                rng=rng,
                enabled=True,
            )
        else:
            with scope:
                result = apply_gated_recurrent_operator_attempt(
                    list(working.get("modules", []) or []),
                    rng=rng,
                    enabled=True,
                )
        working["modules"] = list(result.modules)
        row = dict(result.telemetry)
        row["attempt_index"] = 0
        telemetry["structural_gated_recurrent_mutation"] = {
            "enabled": True,
            "attempts": 1,
            "draws_per_attempt": GATED_RECURRENT_OPERATOR_DRAWS_PER_ATTEMPT,
            "attempt_rows": [row],
            "accepted": int(row.get("status") == "accepted"),
        }

    # ESCAP-8 owns one distinct fixed-budget lane after ESCAP-7.  The pure
    # mutation helper receives both supplied draws and performs no RNG calls.
    # Disabled callers never enter this branch, retaining exact ESCAP-7
    # schedules, payloads, telemetry, and draw counts.
    from evolution.structural_sustainable_growth_v1 import (
        GROWTH_MUTATION_ATTEMPTS,
        GROWTH_MUTATION_DRAWS,
        mutate_sustainable_growth,
    )

    growth_attempts = int(structural_sustainable_growth_operator_attempts or 0)
    if growth_attempts < 0 or growth_attempts > GROWTH_MUTATION_ATTEMPTS:
        _det_fail(
            "DET_ESCAP8_SUSTAINABLE_GROWTH_ATTEMPTS_INVALID",
            f"attempts={growth_attempts}",
        )
    if bool(structural_sustainable_growth_enable) and growth_attempts != 1:
        _det_fail(
            "DET_ESCAP8_SUSTAINABLE_GROWTH_ATTEMPTS_INVALID",
            f"enabled=1;attempts={growth_attempts};expected=1",
        )
    if bool(structural_sustainable_growth_enable):
        label = "mutation.c1.operator.sustainable_growth_mutation"
        expected = None
        if rng_manager is not None:
            expected = rng_manager.expected_delta_for_boundary(label, ctx={})
        scope = (
            rng_manager.budget_scope(label, expected_delta=expected)
            if rng_manager is not None
            else None
        )
        if scope is None:
            draws = _take_draws(rng, GROWTH_MUTATION_DRAWS)
            result = mutate_sustainable_growth(
                working,
                draws[0],
                draws[1],
                enabled=True,
            )
        else:
            with scope:
                draws = _take_draws(rng, GROWTH_MUTATION_DRAWS)
                result = mutate_sustainable_growth(
                    working,
                    draws[0],
                    draws[1],
                    enabled=True,
                )
        working = dict(result.payload)
        row = dict(result.telemetry)
        row["attempt_index"] = 0
        telemetry["structural_sustainable_growth_mutation"] = {
            "enabled": True,
            "attempts": 1,
            "draws_per_attempt": GROWTH_MUTATION_DRAWS,
            "attempt_rows": [row],
            "accepted": int(row.get("status") == "accepted"),
        }
    else:
        row = {}

    # Persist C1 observability annotations into genes payload.
    working = _persist_c1_annotations(
        working,
        chosen_ids=chosen_ids,
        applied=applied,
        noop=noop,
        usage_counts=usage_counts,
        capacity_blocked_reason_counts=capacity_blocked_reason_counts,
        noop_reason_counts=noop_reason_counts,
        structural_bootstrap_enabled=bootstrap_enabled,
        structural_bootstrap_accepted_count=structural_bootstrap_accepted_count,
        structural_sustainable_growth_enabled=bool(
            structural_sustainable_growth_enable
        ),
        sustainable_growth_attempt_row=row,
        blocked_duplication_attempts=blocked_duplication_attempts,
    )

    if not bool(skip_final_normalize):
        working = _normalize_payload(working)
    if bool(emit_final_payload_digest):
        telemetry["final_payload_digest"] = _payload_digest(working)
    return working, telemetry
