from __future__ import annotations

"""ESCAP-3 bounded reference executor for inherited structural modules.

The executor is deliberately small: generic scalar modules integrate the
existing fatigue drive plus previous-tick typed connection outputs and expose
one clipped additive residual for the existing REST action logit.  It owns no
RNG and changes no reward, energy, selection, or action vocabulary semantics.
"""

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

import numpy as np

from evolution.structural_gated_recurrent_v1 import (
    GATED_EXECUTION_OPERATION_ESTIMATE,
    GATED_OPERATOR,
    GATED_RECORD_KEY,
    GatedRecordError,
    normalize_gated_recurrent_record,
    step_gated_scalar,
)
from evolution.structural_sustainable_growth_v1 import (
    ACTIVITY_RECORD_KEY,
    BLOAT_CLASSIFICATION_VERSION,
    COST_COEFFICIENTS,
    COST_MODE,
    COST_PROFILE,
    COST_SCHEMA_VERSION,
    SUSTAINABLE_GROWTH_PROFILE,
    classify_bloat,
    hereditary_no_output,
    is_duplicate_descendant,
    module_is_dormant,
    structural_cost_summary,
    fixed_envelope_shape_for_caps,
)


LIVE_MODULE_SCHEMA_VERSION = "escap3_live_module_runtime_v1"
LIVE_OPERATOR = "bounded_scalar_integrator_v1"
LIVE_PROFILE = "escap3_scalar_integrator_rest_residual_v1"
MAX_REST_RESIDUAL = 0.25
STATE_BOUND = 1.0


def live_modules_enabled(sim_cfg: Any) -> bool:
    return bool(getattr(sim_cfg, "structural_live_modules_enable", False))


def validate_live_modules_config(sim_cfg: Any, *, device: Any = "cpu", policy_stride: int = 1) -> None:
    if not live_modules_enabled(sim_cfg):
        return
    profile = str(getattr(sim_cfg, "structural_live_modules_profile", "none") or "none")
    if profile != LIVE_PROFILE:
        raise RuntimeError(f"ESCAP3_UNSUPPORTED_PROFILE profile={profile}")
    if str(device) != "cpu":
        raise RuntimeError(f"ESCAP3_CPU_REFERENCE_ONLY device={device}")
    if int(policy_stride) != 1:
        raise RuntimeError(f"ESCAP3_POLICY_STRIDE_UNSUPPORTED stride={int(policy_stride)}")
    prohibited = (
        "native_runtime_enable",
        "runnercore_native_enable",
        "cxxrewrite_native_tick_authoritative_enable",
        "cxxrewrite_resident_generation_runner_enable",
        "cxxwave1g_full_cpp_backbone_enable",
        "cxxbackbone8_resident_flat_patch_generation_island_enable",
    )
    active = [name for name in prohibited if bool(getattr(sim_cfg, name, False))]
    if active:
        raise RuntimeError(f"ESCAP3_NATIVE_LANE_UNSUPPORTED flags={','.join(active)}")


def _finite(value: Any, *, field: str) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise RuntimeError(f"ESCAP3_INVALID_NUMERIC field={field}") from exc
    if not math.isfinite(out):
        raise RuntimeError(f"ESCAP3_NONFINITE field={field}")
    return out


def _clip(value: float, bound: float = STATE_BOUND) -> float:
    return float(np.float32(min(float(bound), max(-float(bound), float(value)))))


def _module_key(module: dict[str, Any]) -> str:
    key = str(module.get("module_instance_id", "") or "")
    if not key:
        raise RuntimeError(f"ESCAP3_MISSING_INSTANCE_ID module_id={module.get('module_id', '')}")
    return key


def _stable_biological_module_key(module: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(module.get("homology_id", "") or ""),
        str(module.get("module_instance_id", "") or ""),
        str(module.get("module_id", "") or ""),
    )


def _module_params(module: dict[str, Any]) -> tuple[float, float, float, float]:
    params = module.get("params", {})
    if not isinstance(params, dict):
        raise RuntimeError(f"ESCAP3_INVALID_PARAMS module_id={module.get('module_id', '')}")
    operator = str(params.get("live_operator", LIVE_OPERATOR) or LIVE_OPERATOR)
    if operator != LIVE_OPERATOR:
        raise RuntimeError(
            f"ESCAP3_UNSUPPORTED_OPERATOR module_id={module.get('module_id', '')};operator={operator}"
        )
    expression = _finite(params.get("expression_scale", 0.0), field="expression_scale")
    retention = _finite(params.get("live_retention", 0.0), field="live_retention")
    input_gain = _finite(params.get("live_input_gain", 0.0), field="live_input_gain")
    output_gain = _finite(params.get("live_output_gain", 0.0), field="live_output_gain")
    if not 0.0 <= retention <= 1.0:
        raise RuntimeError(f"ESCAP3_PARAM_OUT_OF_RANGE field=live_retention;value={retention}")
    for name, value in (("expression_scale", expression), ("live_input_gain", input_gain), ("live_output_gain", output_gain)):
        if not -2.0 <= value <= 2.0:
            raise RuntimeError(f"ESCAP3_PARAM_OUT_OF_RANGE field={name};value={value}")
    return expression, retention, input_gain, output_gain


def _module_operator(module: dict[str, Any]) -> str:
    params = module.get("params", {})
    if not isinstance(params, dict):
        raise RuntimeError(f"ESCAP3_INVALID_PARAMS module_id={module.get('module_id', '')}")
    return str(params.get("live_operator", LIVE_OPERATOR) or LIVE_OPERATOR)


def _gated_module_params(module: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    params = module.get("params", {})
    if not isinstance(params, dict):
        raise RuntimeError(f"ESCAP7_INVALID_PARAMS module_id={module.get('module_id', '')}")
    if _module_operator(module) != GATED_OPERATOR:
        raise RuntimeError(
            f"ESCAP7_OPERATOR_MISMATCH module_id={module.get('module_id', '')}"
        )
    expression = _finite(params.get("expression_scale", 0.0), field="expression_scale")
    if not -2.0 <= expression <= 2.0:
        raise RuntimeError(
            f"ESCAP7_PARAM_OUT_OF_RANGE field=expression_scale;value={expression}"
        )
    try:
        record = normalize_gated_recurrent_record(params.get(GATED_RECORD_KEY))
    except GatedRecordError as exc:
        raise RuntimeError(str(exc)) from exc
    return expression, record


def _validate_graph(modules: list[dict[str, Any]], connections: list[dict[str, Any]]) -> None:
    by_id = {str(m.get("module_id", "")): m for m in modules}
    edges: dict[str, list[str]] = {mid: [] for mid in by_id}
    for conn in connections:
        cid = str(conn.get("connection_id", ""))
        src = str(conn.get("src_module_id", ""))
        dst = str(conn.get("dst_module_id", ""))
        if src not in by_id or dst not in by_id:
            raise RuntimeError(f"ESCAP3_MISSING_ENDPOINT connection_id={cid};src={src};dst={dst}")
        if str(conn.get("src_slot", "out")) != "out" or str(conn.get("dst_slot", "in")) != "in":
            raise RuntimeError(f"ESCAP3_MISSING_TYPED_SLOT connection_id={cid}")
        if str(conn.get("value_type", "scalar")) != "scalar":
            raise RuntimeError(f"ESCAP3_TYPE_MISMATCH connection_id={cid}")
        _finite(conn.get("weight", 0.0), field=f"connection[{cid}].weight")
        edges[src].append(dst)

    # Connections read previous-tick values, but cycles are still rejected so
    # the hereditary graph remains bounded and directly auditable.
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(mid: str) -> None:
        if mid in visiting:
            raise RuntimeError(f"ESCAP3_CYCLE_REJECTED module_id={mid}")
        if mid in visited:
            return
        visiting.add(mid)
        for nxt in sorted(edges.get(mid, [])):
            visit(nxt)
        visiting.remove(mid)
        visited.add(mid)

    for mid in sorted(edges):
        visit(mid)


def _runtime_keys(agent: Any) -> tuple[set[str], set[str]]:
    state_obj = getattr(agent, "structural_live_module_state", {}) or {}
    output_obj = getattr(agent, "structural_live_module_outputs", {}) or {}
    if not isinstance(state_obj, dict) or not isinstance(output_obj, dict):
        raise RuntimeError("ESCAP3_RUNTIME_MAP_MALFORMED")
    return {str(k) for k in state_obj}, {str(k) for k in output_obj}


def reconcile_runtime_state_for_genome(
    agent: Any,
    *,
    orphan_policy: str = "reject",
    sustainable_growth_enabled: bool = False,
) -> dict[str, Any]:
    """Reconcile lifetime state at an explicit architecture lifecycle boundary.

    Normal execution rejects orphans. ``prune`` is reserved for an explicit
    deletion boundary; it never reassigns state and reports every removed key.
    """

    modules = [
        dict(m)
        for m in list(getattr(getattr(agent, "genome", None), "modules", []) or [])
        if str(m.get("module_type", "")) == "generic"
    ]
    valid = {_module_key(m) for m in modules}
    state_keys, output_keys = _runtime_keys(agent)
    if state_keys != output_keys:
        raise RuntimeError(
            "ESCAP3_RUNTIME_MAP_KEY_MISMATCH "
            f"state_only={','.join(sorted(state_keys - output_keys))};"
            f"output_only={','.join(sorted(output_keys - state_keys))}"
        )
    latch_obj = getattr(agent, "structural_development_latch_by_instance_id", {}) or {}
    if not isinstance(latch_obj, dict):
        raise RuntimeError("ESCAP6_LATCH_MAP_MALFORMED")
    latch_keys = {str(key) for key, value in latch_obj.items() if bool(value)}
    effective_gain_obj: dict[Any, Any] = {}
    effective_gain_keys: set[str] = set()
    if bool(sustainable_growth_enabled):
        effective_gain_obj = (
            getattr(agent, "structural_development_effective_gain_by_instance_id", {}) or {}
        )
        if not isinstance(effective_gain_obj, dict):
            raise RuntimeError("ESCAP6_EFFECTIVE_GAIN_MAP_MALFORMED")
        effective_gain_keys = {str(key) for key in effective_gain_obj}
    orphaned = sorted((state_keys | output_keys) - valid)
    orphaned_latches = sorted(latch_keys - valid)
    orphaned_effective_gains = sorted(effective_gain_keys - valid)
    policy = str(orphan_policy or "reject")
    if (orphaned or orphaned_latches or orphaned_effective_gains) and policy != "prune":
        context = (
            f"keys={','.join(orphaned)};escap6_latch_keys={','.join(orphaned_latches)}"
        )
        if bool(sustainable_growth_enabled):
            context += f";escap6_effective_gain_keys={','.join(orphaned_effective_gains)}"
        raise RuntimeError(f"ESCAP3_ORPHAN_RUNTIME_STATE {context}")
    if policy not in {"reject", "prune"}:
        raise RuntimeError(f"ESCAP3_INVALID_ORPHAN_POLICY policy={policy}")
    if orphaned:
        agent.structural_live_module_state = {
            str(k): v for k, v in dict(getattr(agent, "structural_live_module_state", {}) or {}).items() if str(k) in valid
        }
        agent.structural_live_module_outputs = {
            str(k): v for k, v in dict(getattr(agent, "structural_live_module_outputs", {}) or {}).items() if str(k) in valid
        }
    if orphaned_latches:
        agent.structural_development_latch_by_instance_id = {
            str(key): bool(value)
            for key, value in latch_obj.items()
            if str(key) in valid and bool(value)
        }
    if bool(sustainable_growth_enabled) and orphaned_effective_gains:
        agent.structural_development_effective_gain_by_instance_id = {
            str(key): value
            for key, value in effective_gain_obj.items()
            if str(key) in valid
        }
    telemetry_cleared = False
    if bool(sustainable_growth_enabled) and policy == "prune":
        # Runtime telemetry is derived, not hereditary or authoritative.  It
        # contains state rows and instance-keyed routing summaries from the
        # previous architecture.  Clearing it at the explicit prune boundary
        # is mandatory even when the removed module never acquired a state,
        # output, latch, or effective-gain entry.  The next live tick
        # deterministically rebuilds the complete summary for the new graph.
        telemetry_cleared = bool(
            dict(getattr(agent, "structural_live_module_telemetry", {}) or {})
        )
        agent.structural_live_module_telemetry = {}
    result: dict[str, Any] = {
        "valid_instance_ids": sorted(valid),
        "pruned_instance_ids": orphaned,
        "pruned_development_latch_instance_ids": orphaned_latches,
    }
    if bool(sustainable_growth_enabled):
        result.update(
            {
                "pruned_development_effective_gain_instance_ids": orphaned_effective_gains,
                "stale_telemetry_cleared": telemetry_cleared,
            }
        )
    return result


@dataclass(frozen=True)
class LiveModuleStep:
    rest_residual: float
    summary: dict[str, Any]


def execute_agent_live_modules(
    agent: Any,
    *,
    fatigue: float,
    reset_instance_ids: Iterable[str] = (),
    lesion_connection_ids: Iterable[str] = (),
    lesion_expression: bool = False,
    routed_observation_by_instance_id: Mapping[str, Any] | None = None,
    routing_telemetry_by_instance_id: Mapping[str, Mapping[str, Any]] | None = None,
    routing_summary: Mapping[str, Any] | None = None,
    developmental_plan: Any = None,
    gated_recurrent_enabled: bool = False,
    sustainable_growth_enabled: bool = False,
) -> LiveModuleStep:
    genome = getattr(agent, "genome", None)
    module_sort_key = (
        _stable_biological_module_key
        if bool(gated_recurrent_enabled)
        else lambda m: str(m.get("module_id", ""))
    )
    modules_all = sorted(
        list(getattr(genome, "modules", []) or []),
        key=module_sort_key,
    )
    # Only the ESCAP-2 generic scalar substrate is executable in ESCAP-3.
    modules = [dict(m) for m in modules_all if str(m.get("module_type", "")) == "generic"]
    instance_ids = [_module_key(m) for m in modules]
    if len(set(instance_ids)) != len(instance_ids):
        raise RuntimeError("ESCAP3_DUPLICATE_INSTANCE_ID")
    live_ids = {str(m.get("module_id", "")) for m in modules}
    connections = [
        dict(c)
        for c in list(getattr(genome, "connections", []) or [])
        if str(c.get("src_module_id", "")) in live_ids or str(c.get("dst_module_id", "")) in live_ids
    ]
    if bool(gated_recurrent_enabled):
        biological_by_local = {
            str(module.get("module_id", "")): _stable_biological_module_key(module)
            for module in modules
        }
        connections.sort(
            key=lambda c: (
                biological_by_local.get(str(c.get("src_module_id", "")), ("", "", "")),
                biological_by_local.get(str(c.get("dst_module_id", "")), ("", "", "")),
                str(c.get("src_slot", "out")),
                str(c.get("dst_slot", "in")),
                str(c.get("value_type", "scalar")),
                str(c.get("channel", "generic")),
                float(c.get("weight", 0.0)),
                str(c.get("connection_id", "")),
            )
        )
    else:
        connections.sort(key=lambda c: str(c.get("connection_id", "")))
    _validate_graph(modules, connections)
    reconcile_runtime_state_for_genome(
        agent,
        orphan_policy="reject",
        sustainable_growth_enabled=bool(sustainable_growth_enabled),
    )

    fatigue_f = _finite(fatigue, field="fatigue")
    if not 0.0 <= fatigue_f <= 1.0:
        raise RuntimeError(f"ESCAP3_INPUT_OUT_OF_RANGE field=fatigue;value={fatigue_f}")
    state = dict(getattr(agent, "structural_live_module_state", {}) or {})
    previous = dict(getattr(agent, "structural_live_module_outputs", {}) or {})
    developmental_gain_by_instance: dict[str, float] = {}
    previous_for_connections = previous
    if developmental_plan is not None:
        developmental_gain_by_instance = {
            str(key): _finite(value, field=f"developmental_gain[{key}]")
            for key, value in dict(
                getattr(developmental_plan, "effective_gain_by_instance_id", {}) or {}
            ).items()
        }
        if set(developmental_gain_by_instance) != set(instance_ids):
            raise RuntimeError(
                "ESCAP6_RUNTIME_PLAN_INSTANCE_MISMATCH "
                f"missing={','.join(sorted(set(instance_ids) - set(developmental_gain_by_instance)))};"
                f"extra={','.join(sorted(set(developmental_gain_by_instance) - set(instance_ids)))}"
            )
        if any(not 0.0 <= value <= 1.0 for value in developmental_gain_by_instance.values()):
            raise RuntimeError("ESCAP6_EFFECTIVE_GAIN_OUT_OF_RANGE")
        # A source that is inactive on this tick must not leak its previously
        # committed effective output into a downstream structural connection.
        previous_for_connections = {
            str(key): (0.0 if developmental_gain_by_instance[str(key)] == 0.0 else value)
            for key, value in previous.items()
        }
    dormant_instance_ids: set[str] = set()
    dormant_local_ids: set[str] = set()
    if bool(sustainable_growth_enabled):
        dormant_instance_ids = {
            _module_key(module) for module in modules if module_is_dormant(module)
        }
        dormant_local_ids = {
            str(module.get("module_id", ""))
            for module in modules
            if _module_key(module) in dormant_instance_ids
        }
        # Dormancy is hereditary architecture, never a retained lifetime
        # output.  Mask the source before any downstream connection reads it.
        previous_for_connections = {
            str(key): (0.0 if str(key) in dormant_instance_ids else value)
            for key, value in previous_for_connections.items()
        }
    reset = {str(v) for v in reset_instance_ids}
    lesioned_connections = {str(v) for v in lesion_connection_ids}
    routed_observations = {
        str(key): _finite(value, field=f"routed_observation[{key}]")
        for key, value in dict(routed_observation_by_instance_id or {}).items()
    }
    routing_rows = {
        str(key): dict(value)
        for key, value in dict(routing_telemetry_by_instance_id or {}).items()
    }
    orphan_routing = sorted((set(routed_observations) | set(routing_rows)) - set(instance_ids))
    if orphan_routing:
        raise RuntimeError(f"ESCAP4_ORPHAN_ROUTING_BINDING keys={','.join(orphan_routing)}")
    incoming: dict[str, list[dict[str, Any]]] = {str(m.get("module_id", "")): [] for m in modules}
    for conn in connections:
        incoming[str(conn.get("dst_module_id", ""))].append(conn)

    next_state: dict[str, float] = {}
    next_outputs: dict[str, float] = {}
    instance_by_local: dict[str, str] = {}
    state_rows: list[dict[str, Any]] = []
    residual = 0.0
    active = 0
    evaluated_connections = 0
    allocated_gated = 0
    executable_gated = 0
    active_gated = 0
    causally_active_gated = 0
    saturated_reset_gates = 0
    saturated_update_gates = 0
    saturated_candidates = 0
    saturated_gated_states = 0
    saturated_gated_outputs = 0
    executing_integrators = 0
    executing_modules = 0
    active_input_bindings = 0
    active_output_bindings = 0
    active_dependency_edges = 0
    for module in modules:
        mid = str(module.get("module_id", ""))
        key = _module_key(module)
        instance_by_local[mid] = key
        operator = _module_operator(module)
        dormant = bool(sustainable_growth_enabled and key in dormant_instance_ids)
        gated_record: dict[str, Any] | None = None
        if operator == LIVE_OPERATOR:
            raw_params = module.get("params", {})
            if bool(gated_recurrent_enabled) and GATED_RECORD_KEY in raw_params:
                raise RuntimeError(
                    f"ESCAP7_STALE_OPERATOR_PAYLOAD module_id={mid};operator={operator}"
                )
            expression, retention, input_gain, output_gain = _module_params(module)
        elif operator == GATED_OPERATOR and bool(gated_recurrent_enabled):
            expression, gated_record = _gated_module_params(module)
            retention = 0.0
            input_gain = 0.0
            output_gain = float(gated_record["output_gain"])
            allocated_gated += 1
            if not bool(sustainable_growth_enabled):
                executable_gated += 1
        else:
            code = "ESCAP7_DISABLED_OPERATOR" if operator == GATED_OPERATOR else "ESCAP3_UNSUPPORTED_OPERATOR"
            raise RuntimeError(f"{code} module_id={mid};operator={operator}")
        if lesion_expression:
            expression = 0.0
        old_state = 0.0 if key in reset else _finite(state.get(key, 0.0), field=f"state[{key}]")
        developmental_gain = (
            1.0
            if developmental_plan is None
            else developmental_gain_by_instance[key]
        )
        module_executing = bool(
            not dormant
            and (developmental_plan is None or developmental_gain > 0.0)
        )
        if module_executing:
            executing_modules += 1
            if operator == GATED_OPERATOR:
                executable_gated += int(bool(sustainable_growth_enabled))
            else:
                executing_integrators += 1
            params_for_cost = module.get("params", {})
            if isinstance(params_for_cost, Mapping):
                input_block = params_for_cost.get("observation_routing", {})
                if isinstance(input_block, Mapping):
                    active_input_bindings += sum(
                        1
                        for binding in list(input_block.get("bindings", []) or [])
                        if isinstance(binding, Mapping) and bool(binding.get("enabled", True))
                    )
                output_block = params_for_cost.get("output_routing", {})
                if isinstance(output_block, Mapping):
                    active_output_bindings += sum(
                        1
                        for binding in list(output_block.get("bindings", []) or [])
                        if isinstance(binding, Mapping) and bool(binding.get("enabled", True))
                    )
                development_block = params_for_cost.get("developmental_expression", {})
                if isinstance(development_block, Mapping):
                    active_dependency_edges += len(
                        list(development_block.get("prerequisite_homology_ids", []) or [])
                    )
        routed_observation = _finite(
            routed_observations.get(key, 0.0),
            field=f"routed_observation[{key}]",
        )
        if not -1.0 <= routed_observation <= 1.0:
            raise RuntimeError(
                f"ESCAP4_ROUTED_INPUT_OUT_OF_RANGE module_instance_id={key};value={routed_observation}"
            )
        # The routed lane has already been clipped exactly once to its declared
        # aggregate bound.  Add it to fatigue at the existing typed-input
        # boundary without a second input clip; the pre-existing state bound
        # remains the next and only downstream saturation point.  The explicit
        # zero branch preserves prior ESCAP-3 arithmetic exactly.
        typed_external_input = (
            fatigue_f
            if routed_observation == 0.0
            else fatigue_f + routed_observation
        )
        routed = 0.0
        if dormant:
            # Frozen ESCAP-8 dormancy rule: preserve the owned state, publish
            # exact zero, and skip both incoming edges and the live operator.
            new_state = old_state
            routed = 0.0
            base_output = 0.0
            output = 0.0
            gated_step = None
        elif developmental_plan is None or developmental_gain > 0.0:
            for conn in incoming.get(mid, []):
                cid = str(conn.get("connection_id", ""))
                if cid in lesioned_connections:
                    continue
                src_mid = str(conn.get("src_module_id", ""))
                if bool(sustainable_growth_enabled) and src_mid in dormant_local_ids:
                    continue
                src_key = instance_by_local.get(src_mid)
                if src_key is None:
                    src_module = next((m for m in modules if str(m.get("module_id", "")) == src_mid), None)
                    if src_module is None:
                        raise RuntimeError(f"ESCAP3_MISSING_ENDPOINT connection_id={cid}")
                    src_key = _module_key(src_module)
                routed += _finite(conn.get("weight", 0.0), field=f"connection[{cid}].weight") * _finite(
                    previous_for_connections.get(src_key, 0.0), field=f"previous_output[{src_key}]"
                )
                if not math.isfinite(routed):
                    raise RuntimeError(f"ESCAP3_NONFINITE field=aggregate_input[{mid}]")
                evaluated_connections += 1
            gated_step = None
            if operator == LIVE_OPERATOR:
                # Preserve the established integrator expression and Python
                # operation order exactly.  ESCAP-7 is a separate branch.
                new_state = _clip(retention * old_state + input_gain * typed_external_input + routed)
                base_output = _clip(expression * output_gain * new_state)
                output = (
                    base_output
                    if developmental_plan is None or developmental_gain == 1.0
                    else _clip(developmental_gain * base_output)
                )
            else:
                try:
                    gated_step = step_gated_scalar(
                        external_input=typed_external_input,
                        routed_previous=routed,
                        previous_state=old_state,
                        expression_scale=expression,
                        developmental_gain=developmental_gain,
                        record=gated_record,
                    )
                except GatedRecordError as exc:
                    raise RuntimeError(str(exc)) from exc
                new_state = gated_step.state_after
                # ``base_live_output`` remains the value immediately before
                # ESCAP-6 development, matching its legacy telemetry meaning.
                base_output = gated_step.expressed_output
                output = gated_step.effective_output
                active_gated += int(expression != 0.0 and output_gain != 0.0)
                causally_active_gated += int(output != 0.0)
                saturated_reset_gates += int(abs(gated_step.reset_pre) == 16.0)
                saturated_update_gates += int(abs(gated_step.update_pre) == 16.0)
                saturated_candidates += int(abs(gated_step.candidate) == 1.0)
                saturated_gated_states += int(abs(gated_step.state_after) == 1.0)
                saturated_gated_outputs += int(abs(gated_step.base_output) == 1.0)
        else:
            # Frozen v1 inactive rule: retain owned state, publish exact zero,
            # and do not invisibly evaluate incoming connections.
            new_state = old_state
            base_output = 0.0
            output = 0.0
            gated_step = None
        next_state[key] = new_state
        next_outputs[key] = output
        residual += output
        if not math.isfinite(residual):
            raise RuntimeError("ESCAP3_NONFINITE field=rest_residual")
        if operator == GATED_OPERATOR:
            if developmental_gain > 0.0 and expression != 0.0 and output_gain != 0.0:
                active += 1
        elif developmental_gain > 0.0 and expression != 0.0 and (input_gain != 0.0 or routed != 0.0) and output_gain != 0.0:
            active += 1
        state_row = {
                "module_id": mid,
                "module_instance_id": key,
                "state_before": old_state,
                "typed_input_fatigue": fatigue_f,
                "typed_input_routed_observation": routed_observation,
                "typed_input_combined_external": typed_external_input,
                "typed_input_previous_outputs": _clip(routed),
                "state_after": new_state,
                "typed_output": output,
                "observation_routing": routing_rows.get(key, {}),
            }
        if bool(gated_recurrent_enabled):
            state_row["live_operator"] = operator
        if operator == GATED_OPERATOR:
            state_row.update(
                {
                    "gated_input": None if gated_step is None else gated_step.input_aggregate,
                    "gated_state_before": None if gated_step is None else gated_step.state_before,
                    "reset_pre": None if gated_step is None else gated_step.reset_pre,
                    "reset_gate": None if gated_step is None else gated_step.reset_gate,
                    "update_pre": None if gated_step is None else gated_step.update_pre,
                    "update_gate": None if gated_step is None else gated_step.update_gate,
                    "candidate_pre": None if gated_step is None else gated_step.candidate_pre,
                    "candidate": None if gated_step is None else gated_step.candidate,
                    "gated_base_output": None if gated_step is None else gated_step.base_output,
                    "gated_expressed_output": None if gated_step is None else gated_step.expressed_output,
                    "gated_effective_output": output,
                    "gated_evaluation_rng_draws": 0,
                }
            )
        if developmental_plan is not None:
            state_row.update(
                {
                    "developmental_expression_gain": developmental_gain,
                    "base_live_output": base_output,
                    "inactive_state_frozen": bool(developmental_gain == 0.0),
                }
            )
        if bool(sustainable_growth_enabled):
            state_row.update(
                {
                    "structural_activity_record_present": bool(
                        isinstance(module.get("params", {}), Mapping)
                        and ACTIVITY_RECORD_KEY in module.get("params", {})
                    ),
                    "structurally_dormant": dormant,
                    "executed_live_operator": module_executing,
                }
            )
        state_rows.append(state_row)

    bounded_residual = _clip(residual, MAX_REST_RESIDUAL)
    summary = {
        "schema_version": LIVE_MODULE_SCHEMA_VERSION,
        "profile": LIVE_PROFILE,
        "allocated_modules": len(modules),
        "executable_modules": len(modules),
        "active_modules": active,
        "nonzero_output_modules": int(sum(1 for value in next_outputs.values() if value != 0.0)),
        "changing_state_modules": int(
            sum(1 for row in state_rows if float(row["state_before"]) != float(row["state_after"]))
        ),
        "live_connections": len(connections),
        "evaluated_connections": evaluated_connections,
        "state_updates": len(modules),
        "module_evaluations": len(modules),
        "absolute_output": float(np.float32(sum(abs(v) for v in next_outputs.values()))),
        "reported_compute_cost": int(len(modules) * 4 + evaluated_connections * 2),
        "rest_logit_residual": bounded_residual,
        "state_instance_map": instance_by_local,
        "state_rows": state_rows,
        "random_draws": 0,
        "lane": "cpu_reference",
        "lesion_expression": bool(lesion_expression),
        "lesioned_connection_ids": sorted(lesioned_connections),
        "reset_instance_ids": sorted(reset),
        "clipped_rest_residual": bool(float(residual) != float(bounded_residual)),
        "rejected_evaluations": 0,
        "observation_routing": dict(routing_summary or {"execution": "disabled", "random_draws": 0}),
    }
    if bool(gated_recurrent_enabled):
        allocated_integrators = len(modules) - allocated_gated
        summary.update(
            {
                "gated_recurrent_enabled": True,
                "allocated_scalar_integrators": allocated_integrators,
                "allocated_integrator_modules": allocated_integrators,
                "allocated_gated_recurrent_modules": allocated_gated,
                "executable_gated_recurrent_modules": executable_gated,
                "active_gated_recurrent_modules": active_gated,
                "inactive_gated_recurrent_modules": allocated_gated - active_gated,
                "causally_active_gated_recurrent_modules": causally_active_gated,
                "saturated_reset_gates": saturated_reset_gates,
                "saturated_update_gates": saturated_update_gates,
                "saturated_candidates": saturated_candidates,
                "saturated_gated_states": saturated_gated_states,
                "saturated_gated_outputs": saturated_gated_outputs,
                "owned_state_width_per_module": 1,
                "gated_execution_rng_draws": 0,
                "gated_execution_operation_estimate": int(
                    executable_gated * GATED_EXECUTION_OPERATION_ESTIMATE
                ),
                "parameter_bound_rejections": 0,
                "reported_compute_cost": int(
                    allocated_integrators * 4
                    + executable_gated * GATED_EXECUTION_OPERATION_ESTIMATE
                    + evaluated_connections * 2
                ),
            }
        )
        summary["total_live_execution_operation_estimate"] = int(
            summary["reported_compute_cost"]
        )
    if developmental_plan is not None:
        summary["structural_development"] = dict(
            getattr(developmental_plan, "telemetry", {}) or {}
        )
    if bool(sustainable_growth_enabled):
        dependency_targets: set[str] = set()
        for module in modules:
            params = module.get("params", {})
            if not isinstance(params, Mapping):
                continue
            development = params.get("developmental_expression", {})
            if isinstance(development, Mapping):
                dependency_targets.update(
                    str(value)
                    for value in list(
                        development.get("prerequisite_homology_ids", []) or []
                    )
                )
        neutral_duplicate_modules = [
            module
            for module in modules
            if is_duplicate_descendant(module) and hereditary_no_output(module)
        ]
        generic_module_count = sum(
            1 for module in modules if str(module.get("module_type", "")) == "generic"
        )
        prunable_modules = [
            module
            for module in modules
            if generic_module_count > 1
            and str(module.get("module_type", "")) == "generic"
            and (module_is_dormant(module) or hereditary_no_output(module))
            and str(module.get("homology_id", "")) not in dependency_targets
        ]
        serialized_genome_bytes = (
            len(genome.canonical_bytes())
            if callable(getattr(genome, "canonical_bytes", None))
            else 0
        )
        saturated_or_rejected = int(
            saturated_reset_gates
            + saturated_update_gates
            + saturated_candidates
            + saturated_gated_states
            + saturated_gated_outputs
            + int(summary.get("rejected_evaluations", 0) or 0)
        )
        active_envelope = fixed_envelope_shape_for_caps(
            getattr(genome, "b2_caps", None)
        )
        cost = structural_cost_summary(
            allocated_module_count=len(modules),
            executing_integrator_count=executing_integrators,
            executing_gated_count=executable_gated,
            evaluated_connection_count=evaluated_connections,
            active_input_binding_count=active_input_bindings,
            active_output_binding_count=active_output_bindings,
            active_dependency_edge_count=active_dependency_edges,
            dormant_module_count=len(dormant_instance_ids),
            serialized_genome_bytes=serialized_genome_bytes,
            saturated_or_rejected_activity_count=saturated_or_rejected,
        )
        live_operation_estimate = int(
            executing_integrators * 4
            + executable_gated * GATED_EXECUTION_OPERATION_ESTIMATE
            + evaluated_connections * 2
        )
        blocked_duplication_attempts = int(
            getattr(genome, "escap8_blocked_duplication_attempt_count", 0) or 0
        )
        blocked_prune_attempts = int(
            getattr(genome, "escap8_blocked_prune_attempt_count", 0) or 0
        )
        cap_pressure_streak = int(
            getattr(genome, "escap8_cap_pressure_streak", 0) or 0
        )
        bloat = classify_bloat(
            allocated_module_count=len(modules),
            module_capacity=int(active_envelope["max_module_slots"]),
            active_module_count=executing_modules,
            dormant_module_count=len(dormant_instance_ids),
            causally_active_module_count=0,
            neutral_duplicate_count=len(neutral_duplicate_modules),
            redundant_structure_count=len(neutral_duplicate_modules),
            prunable_module_count=len(prunable_modules),
            blocked_duplication_attempt_count=blocked_duplication_attempts,
            blocked_prune_attempt_count=blocked_prune_attempts,
            cap_contact_count=int(
                len(modules) >= int(active_envelope["max_module_slots"])
            ),
            sustained_cap_pressure=bool(cap_pressure_streak >= 3),
            structural_cost=float(cost["total"]),
            useful_activity_proxy=float(summary["absolute_output"]),
            causal_activity_measured=False,
        )
        summary.update(
            {
                "sustainable_growth_enabled": True,
                "sustainable_growth_profile": SUSTAINABLE_GROWTH_PROFILE,
                "allocated_modules": len(modules),
                "executing_modules": executing_modules,
                "executable_modules": executing_modules,
                "state_updates": executing_modules,
                "module_evaluations": executing_modules,
                "reported_compute_cost": live_operation_estimate,
                "total_live_execution_operation_estimate": live_operation_estimate,
                "dormant_modules": len(dormant_instance_ids),
                "nonzero_output_modules": int(
                    sum(1 for value in next_outputs.values() if value != 0.0)
                ),
                "causal_activity_reporting": "probe_only_not_classified_live",
                "neutral_duplicate_modules": len(neutral_duplicate_modules),
                # V1's conservative redundancy class is exact: a duplicate
                # descendant with hereditary zero output.  It is a reporting
                # label only and is never a pruning or selection instruction.
                "redundant_modules": len(neutral_duplicate_modules),
                "prunable_modules": len(prunable_modules),
                "blocked_duplication_attempts": blocked_duplication_attempts,
                "blocked_prune_attempts": blocked_prune_attempts,
                "cap_pressure_streak": cap_pressure_streak,
                "sustained_cap_pressure": bool(cap_pressure_streak >= 3),
                "prunable_module_instance_ids": sorted(
                    _module_key(module) for module in prunable_modules
                ),
                "active_input_bindings": active_input_bindings,
                "active_output_bindings": active_output_bindings,
                "active_dependency_edges": active_dependency_edges,
                "serialized_genome_bytes": serialized_genome_bytes,
                "serialized_genome_cap": None,
                "serialized_genome_capacity_bounded": False,
                "structural_cost": cost,
                "structural_cost_total": float(cost["total"]),
                "bloat_classification": bloat,
                "cost_schema_version": COST_SCHEMA_VERSION,
                "cost_profile": COST_PROFILE,
                "cost_mode": COST_MODE,
                "cost_coefficients": dict(COST_COEFFICIENTS),
                "bloat_classification_version": BLOAT_CLASSIFICATION_VERSION,
            }
        )
    agent.structural_live_module_state = next_state
    agent.structural_live_module_outputs = next_outputs
    agent.structural_live_module_telemetry = summary
    if developmental_plan is not None:
        agent.structural_development_latch_by_instance_id = dict(
            getattr(developmental_plan, "latch_by_instance_id", {}) or {}
        )
        agent.structural_development_stage_id = str(
            getattr(developmental_plan, "stage_id", "") or ""
        )
        agent.structural_development_effective_gain_by_instance_id = dict(
            developmental_gain_by_instance
        )
    return LiveModuleStep(rest_residual=bounded_residual, summary=summary)


def execute_live_modules_batch(
    alive: list[Any],
    fatigue: np.ndarray,
    *,
    sim_cfg: Any,
    device: Any,
    routing_batches: list[Any] | None = None,
) -> np.ndarray:
    validate_live_modules_config(sim_cfg, device=device, policy_stride=1)
    out = np.zeros((len(alive),), dtype=np.float32)
    if not live_modules_enabled(sim_cfg):
        return out
    vals = np.asarray(fatigue, dtype=np.float32).reshape(-1)
    if vals.shape[0] != len(alive):
        raise RuntimeError("ESCAP3_INPUT_SHAPE_MISMATCH")
    batches = routing_batches or [None] * len(alive)
    if len(batches) != len(alive):
        raise RuntimeError("ESCAP4_ROUTING_BATCH_SHAPE_MISMATCH")
    developmental_enabled = bool(
        getattr(sim_cfg, "structural_development_enable", False)
    )
    gated_recurrent_enabled = bool(
        getattr(sim_cfg, "structural_gated_recurrent_enable", False)
    )
    sustainable_growth_enabled = bool(
        getattr(sim_cfg, "structural_sustainable_growth_enable", False)
    )
    for i, agent in enumerate(alive):
        batch = batches[i]
        kwargs: dict[str, Any] = {}
        if batch is not None:
            kwargs = {
                "routed_observation_by_instance_id": dict(getattr(batch, "aggregate_by_instance_id", {}) or {}),
                "routing_telemetry_by_instance_id": dict(getattr(batch, "telemetry_by_instance_id", {}) or {}),
                "routing_summary": dict(getattr(batch, "summary", {}) or {}),
            }
        if developmental_enabled:
            from evolution.structural_development_runtime_v1 import (
                resolve_agent_development_runtime,
            )

            kwargs["developmental_plan"] = resolve_agent_development_runtime(
                agent,
                sim_cfg=sim_cfg,
            )
        out[i] = np.float32(
            execute_agent_live_modules(
                agent,
                fatigue=float(vals[i]),
                gated_recurrent_enabled=gated_recurrent_enabled,
                sustainable_growth_enabled=sustainable_growth_enabled,
                **kwargs,
            ).rest_residual
        )
    return out


def add_rest_logit_residual_(logits: Any, residual: Any, *, rest_index: int) -> Any:
    """Apply the single ESCAP-3 controller seam exactly once, in place."""

    if len(logits.shape) != 2 or len(residual.shape) != 1 or int(logits.shape[0]) != int(residual.shape[0]):
        raise RuntimeError("ESCAP3_LOGIT_SEAM_SHAPE_MISMATCH")
    if int(rest_index) < 0 or int(rest_index) >= int(logits.shape[1]):
        raise RuntimeError("ESCAP3_LOGIT_SEAM_INDEX_INVALID")
    logits[:, int(rest_index)] = logits[:, int(rest_index)] + residual
    return logits
