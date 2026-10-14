from __future__ import annotations

"""Deterministic, JSON-ready ESCAP-6 twelve-arm technical probe.

This probe exercises bounded lifetime expression and dependency mechanics only.
It changes no environment, action, reward, fitness, selection, or physical
module/connection capacity, and it owns no random draws.
"""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agents.structural_identity_v1 import new_native_identity
from evolution.structural_development_runtime_v1 import (
    resolve_agent_development_runtime,
)
from evolution.structural_development_v1 import (
    DEVELOPMENTAL_EXPRESSION_BLOCK_KEY,
    DEVELOPMENTAL_MODULE_SCHEMA_VERSION,
    DEVELOPMENTAL_STAGE_PROFILE_VERSION,
    MATURE_START_STEP,
    LATE_START_STEP,
    neutral_developmental_expression_record,
)
from evolution.structural_live_modules_v1 import execute_agent_live_modules


PROBE_SCHEMA_VERSION = "escap6_developmental_expression_probe_v1"
EXPECTED_ARM_COUNT = 12


def _stable_hash(value: Any) -> str:
    blob = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _development(
    gains: tuple[float, float, float] = (1.0, 1.0, 1.0),
    *,
    prerequisites: tuple[str, ...] = (),
    latch: bool = False,
) -> dict[str, Any]:
    return {
        "schema_version": DEVELOPMENTAL_MODULE_SCHEMA_VERSION,
        "stage_profile_version": DEVELOPMENTAL_STAGE_PROFILE_VERSION,
        "stage_gains": [float(value) for value in gains],
        "prerequisite_homology_ids": list(prerequisites),
        "latch_dependency_readiness": bool(latch),
    }


def _module(
    index: int,
    *,
    development: dict[str, Any] | None = None,
    expression_scale: float = 1.0,
    retention: float = 0.0,
    input_gain: float = 1.0,
    output_gain: float = 1.0,
) -> dict[str, Any]:
    identity = new_native_identity(
        context={
            "campaign_id": "escap6-technical-probe",
            "branch_id": "developmental-expression",
            "generation": 1,
            "tick": 0,
            "child_agent_id": int(index),
            "parent_agent_ids": [1, 2],
            "lineage_token": "escap6-probe-lineage",
        },
        operator_id="escap6_probe_fixture",
        attempt_index=int(index),
        operator_role=f"module-{int(index)}",
    )
    params: dict[str, Any] = {
        "live_operator": "bounded_scalar_integrator_v1",
        "expression_scale": float(expression_scale),
        "live_retention": float(retention),
        "live_input_gain": float(input_gain),
        "live_output_gain": float(output_gain),
    }
    if development is not None:
        params[DEVELOPMENTAL_EXPRESSION_BLOCK_KEY] = deepcopy(development)
    return {
        "module_id": f"m{int(index):03d}",
        "module_type": "generic",
        "params": params,
        "io": {
            "inputs": [{"slot_id": "in", "value_type": "scalar"}],
            "outputs": [{"slot_id": "out", "value_type": "scalar"}],
            "caps": {"max_fanin": 6, "max_fanout": 6},
        },
        **identity,
        "latent_pos": list(identity["topology_pos"]),
    }


def _connection(index: int, source: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    return {
        "connection_id": f"c{int(index):03d}",
        "src_module_id": str(source["module_id"]),
        "src_slot": "out",
        "dst_module_id": str(target["module_id"]),
        "dst_slot": "in",
        "value_type": "scalar",
        "weight": 1.0,
        "enabled": True,
    }


def _agent(
    modules: list[dict[str, Any]],
    *,
    age: int = 0,
    connections: list[dict[str, Any]] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        genome=SimpleNamespace(
            modules=deepcopy(modules),
            connections=deepcopy(connections or []),
        ),
        stats=SimpleNamespace(survival_steps=int(age)),
        structural_live_module_state={},
        structural_live_module_outputs={},
        structural_live_module_telemetry={},
        structural_development_latch_by_instance_id={},
        structural_development_stage_id="",
        structural_development_effective_gain_by_instance_id={},
    )


def _cfg() -> SimpleNamespace:
    return SimpleNamespace(
        structural_development_mature_start_step=MATURE_START_STEP,
        structural_development_late_start_step=LATE_START_STEP,
    )


def _architecture(agent: SimpleNamespace) -> dict[str, Any]:
    return {
        "modules": deepcopy(agent.genome.modules),
        "connections": deepcopy(agent.genome.connections),
    }


def _run_enabled_step(
    agent: SimpleNamespace,
    *,
    fatigue: float,
    dependency_lesion_instance_ids: tuple[str, ...] = (),
    expression_lesion_instance_ids: tuple[str, ...] = (),
    latch_lesion_instance_ids: tuple[str, ...] = (),
) -> tuple[Any, Any]:
    plan = resolve_agent_development_runtime(
        agent,
        sim_cfg=_cfg(),
        dependency_lesion_instance_ids=dependency_lesion_instance_ids,
        expression_lesion_instance_ids=expression_lesion_instance_ids,
        latch_lesion_instance_ids=latch_lesion_instance_ids,
    )
    step = execute_agent_live_modules(
        agent,
        fatigue=float(fatigue),
        developmental_plan=plan,
    )
    return plan, step


def _close(left: float, right: float, *, tolerance: float = 1e-7) -> bool:
    return abs(float(left) - float(right)) <= float(tolerance)


def _disabled_legacy_arm() -> dict[str, Any]:
    module = _module(1, expression_scale=0.5, retention=0.5)
    implicit = _agent([module])
    explicit_none = deepcopy(implicit)
    first = execute_agent_live_modules(implicit, fatigue=0.5)
    second = execute_agent_live_modules(explicit_none, fatigue=0.5, developmental_plan=None)
    passed = bool(
        first == second
        and implicit.structural_live_module_state == explicit_none.structural_live_module_state
        and implicit.structural_live_module_outputs == explicit_none.structural_live_module_outputs
        and "structural_development" not in first.summary
    )
    return {
        "arm": "disabled_legacy",
        "passed": passed,
        "state": deepcopy(implicit.structural_live_module_state),
        "outputs": deepcopy(implicit.structural_live_module_outputs),
        "escap6_telemetry_present": "structural_development" in first.summary,
        "random_draws": int(first.summary["random_draws"]),
    }


def _enabled_neutral_arm() -> dict[str, Any]:
    absent = _module(2, expression_scale=0.5, retention=0.5)
    neutral = deepcopy(absent)
    neutral["params"][DEVELOPMENTAL_EXPRESSION_BLOCK_KEY] = (
        neutral_developmental_expression_record()
    )
    control = _agent([absent], age=0)
    treatment = _agent([neutral], age=0)
    control_step = execute_agent_live_modules(control, fatigue=0.5)
    plan, treatment_step = _run_enabled_step(treatment, fatigue=0.5)
    instance_id = str(neutral["module_instance_id"])
    passed = bool(
        control.structural_live_module_state == treatment.structural_live_module_state
        and control.structural_live_module_outputs == treatment.structural_live_module_outputs
        and control_step.rest_residual == treatment_step.rest_residual
        and plan.effective_gain_by_instance_id == {instance_id: 1.0}
    )
    return {
        "arm": "enabled_neutral",
        "passed": passed,
        "effective_gain": float(plan.effective_gain_by_instance_id[instance_id]),
        "legacy_equal": bool(control_step.rest_residual == treatment_step.rest_residual),
        "random_draws": int(plan.telemetry["random_draws"]),
    }


def _stage_expression_arm(name: str, *, age: int, expected_stage: str, expected_gain: float) -> dict[str, Any]:
    module = _module(
        10 + int(age),
        development=_development((0.25, 0.5, 0.75)),
        expression_scale=0.8,
    )
    agent = _agent([module], age=age)
    plan, step = _run_enabled_step(agent, fatigue=1.0)
    instance_id = str(module["module_instance_id"])
    row = step.summary["state_rows"][0]
    expected_output = 0.8 * float(expected_gain)
    passed = bool(
        plan.stage_id == expected_stage
        and _close(plan.effective_gain_by_instance_id[instance_id], expected_gain)
        and _close(row["base_live_output"], 0.8)
        and _close(row["typed_output"], expected_output)
    )
    return {
        "arm": name,
        "passed": passed,
        "age_steps": int(age),
        "stage_id": plan.stage_id,
        "effective_gain": float(plan.effective_gain_by_instance_id[instance_id]),
        "base_live_output": float(row["base_live_output"]),
        "typed_output": float(row["typed_output"]),
        "random_draws": int(plan.telemetry["random_draws"]),
    }


def _dependency_arm(name: str, *, root_gain: float, expected_ready: bool) -> dict[str, Any]:
    root = _module(30, development=_development((root_gain, root_gain, root_gain)))
    child = _module(
        31,
        development=_development(prerequisites=(str(root["homology_id"]),)),
    )
    agent = _agent([child, root], age=0)
    plan, step = _run_enabled_step(agent, fatigue=0.5)
    root_id = str(root["module_instance_id"])
    child_id = str(child["module_instance_id"])
    observed_ready = bool(plan.dependency_ready_by_instance_id[child_id])
    expected_child_gain = 1.0 if expected_ready else 0.0
    passed = bool(
        observed_ready is bool(expected_ready)
        and plan.effective_gain_by_instance_id[child_id] == expected_child_gain
        and plan.topological_instance_ids == (root_id, child_id)
    )
    return {
        "arm": name,
        "passed": passed,
        "topological_instance_ids": list(plan.topological_instance_ids),
        "dependency_ready": observed_ready,
        "root_effective_gain": float(plan.effective_gain_by_instance_id[root_id]),
        "child_effective_gain": float(plan.effective_gain_by_instance_id[child_id]),
        "child_output": float(agent.structural_live_module_outputs[child_id]),
        "random_draws": int(step.summary["random_draws"]),
    }


def _latch_arm() -> dict[str, Any]:
    root = _module(40, development=_development((1.0, 0.0, 0.0)))
    child = _module(
        41,
        development=_development(
            (1.0, 0.5, 0.0),
            prerequisites=(str(root["homology_id"]),),
            latch=True,
        ),
    )
    agent = _agent([child, root], age=0)
    early_plan, _ = _run_enabled_step(agent, fatigue=0.5)
    child_id = str(child["module_instance_id"])
    agent.stats.survival_steps = MATURE_START_STEP
    mature_plan, mature_step = _run_enabled_step(agent, fatigue=0.5)
    passed = bool(
        early_plan.latch_by_instance_id == {child_id: True}
        and mature_plan.effective_gain_by_instance_id[child_id] == 0.5
        and mature_plan.latch_by_instance_id == {child_id: True}
    )
    return {
        "arm": "readiness_latch",
        "passed": passed,
        "early_latches": deepcopy(early_plan.latch_by_instance_id),
        "mature_latches": deepcopy(mature_plan.latch_by_instance_id),
        "mature_child_effective_gain": float(mature_plan.effective_gain_by_instance_id[child_id]),
        "random_draws": int(mature_step.summary["random_draws"]),
    }


def _inactive_freeze_stale_arm() -> dict[str, Any]:
    source = _module(
        50,
        development=_development((0.0, 1.0, 1.0)),
        retention=1.0,
    )
    target = _module(
        51,
        development=_development(),
        retention=1.0,
        input_gain=0.0,
    )
    agent = _agent([source, target], age=0, connections=[_connection(1, source, target)])
    source_id = str(source["module_instance_id"])
    target_id = str(target["module_instance_id"])
    agent.structural_live_module_state = {source_id: 0.75, target_id: 0.0}
    agent.structural_live_module_outputs = {source_id: 0.75, target_id: 0.0}
    plan, step = _run_enabled_step(agent, fatigue=1.0)
    passed = bool(
        plan.effective_gain_by_instance_id[source_id] == 0.0
        and agent.structural_live_module_state[source_id] == 0.75
        and agent.structural_live_module_outputs[source_id] == 0.0
        and agent.structural_live_module_state[target_id] == 0.0
        and agent.structural_live_module_outputs[target_id] == 0.0
    )
    return {
        "arm": "inactive_freeze_stale_output",
        "passed": passed,
        "source_state_after": float(agent.structural_live_module_state[source_id]),
        "source_output_after": float(agent.structural_live_module_outputs[source_id]),
        "target_state_after": float(agent.structural_live_module_state[target_id]),
        "target_output_after": float(agent.structural_live_module_outputs[target_id]),
        "evaluated_connections": int(step.summary["evaluated_connections"]),
        "random_draws": int(plan.telemetry["random_draws"]),
    }


def _reactivation_arm() -> dict[str, Any]:
    module = _module(
        60,
        development=_development((0.0, 0.5, 1.0)),
        expression_scale=0.5,
        retention=1.0,
    )
    agent = _agent([module], age=0)
    instance_id = str(module["module_instance_id"])
    agent.structural_live_module_state = {instance_id: 0.4}
    agent.structural_live_module_outputs = {instance_id: 0.4}
    _run_enabled_step(agent, fatigue=1.0)
    frozen_state = float(agent.structural_live_module_state[instance_id])
    agent.stats.survival_steps = MATURE_START_STEP
    plan, step = _run_enabled_step(agent, fatigue=0.0)
    row = step.summary["state_rows"][0]
    passed = bool(
        _close(frozen_state, 0.4)
        and _close(row["state_after"], 0.4)
        and _close(row["base_live_output"], 0.2)
        and _close(row["typed_output"], 0.1)
        and plan.stage_id == "mature"
    )
    return {
        "arm": "reactivation_retained_state",
        "passed": passed,
        "frozen_state": frozen_state,
        "reactivated_state": float(row["state_after"]),
        "base_live_output": float(row["base_live_output"]),
        "effective_output": float(row["typed_output"]),
        "random_draws": int(plan.telemetry["random_draws"]),
    }


def _lesion_arm() -> dict[str, Any]:
    root = _module(70, development=_development((0.0, 0.0, 0.0)))
    child = _module(
        71,
        development=_development(prerequisites=(str(root["homology_id"]),)),
    )
    child_id = str(child["module_instance_id"])
    control = _agent([root, child], age=0)
    control_plan = resolve_agent_development_runtime(control, sim_cfg=_cfg())
    bypass = _agent([root, child], age=0)
    bypass_plan, bypass_step = _run_enabled_step(
        bypass,
        fatigue=0.5,
        dependency_lesion_instance_ids=(child_id,),
    )
    expression = _agent([root, child], age=0)
    expression_plan, expression_step = _run_enabled_step(
        expression,
        fatigue=0.5,
        dependency_lesion_instance_ids=(child_id,),
        expression_lesion_instance_ids=(child_id,),
    )
    passed = bool(
        control_plan.effective_gain_by_instance_id[child_id] == 0.0
        and bypass_plan.effective_gain_by_instance_id[child_id] == 1.0
        and expression_plan.effective_gain_by_instance_id[child_id] == 0.0
    )
    return {
        "arm": "dependency_and_expression_lesions",
        "passed": passed,
        "control_child_gain": float(control_plan.effective_gain_by_instance_id[child_id]),
        "dependency_lesion_child_gain": float(bypass_plan.effective_gain_by_instance_id[child_id]),
        "expression_lesion_child_gain": float(expression_plan.effective_gain_by_instance_id[child_id]),
        "dependency_lesion_output": float(bypass.structural_live_module_outputs[child_id]),
        "expression_lesion_output": float(expression.structural_live_module_outputs[child_id]),
        "random_draws": int(bypass_step.summary["random_draws"])
        + int(expression_step.summary["random_draws"]),
    }


def _checkpoint_fixed_shape_arm() -> dict[str, Any]:
    root = _module(80, development=_development((1.0, 0.0, 0.0)), retention=0.5)
    child = _module(
        81,
        development=_development(
            (1.0, 0.5, 0.5),
            prerequisites=(str(root["homology_id"]),),
            latch=True,
        ),
        retention=0.5,
    )
    agent = _agent([child, root], age=0, connections=[_connection(2, root, child)])
    architecture_before = _architecture(agent)
    _run_enabled_step(agent, fatigue=0.4)
    checkpoint_payload = {
        "age_steps": int(agent.stats.survival_steps),
        "state": deepcopy(agent.structural_live_module_state),
        "outputs": deepcopy(agent.structural_live_module_outputs),
        "telemetry": deepcopy(agent.structural_live_module_telemetry),
        "latches": deepcopy(agent.structural_development_latch_by_instance_id),
        "stage_id": str(agent.structural_development_stage_id),
        "effective_gains": deepcopy(agent.structural_development_effective_gain_by_instance_id),
    }
    restored = _agent(agent.genome.modules, age=checkpoint_payload["age_steps"], connections=agent.genome.connections)
    restored.structural_live_module_state = deepcopy(checkpoint_payload["state"])
    restored.structural_live_module_outputs = deepcopy(checkpoint_payload["outputs"])
    restored.structural_live_module_telemetry = deepcopy(checkpoint_payload["telemetry"])
    restored.structural_development_latch_by_instance_id = deepcopy(checkpoint_payload["latches"])
    restored.structural_development_stage_id = checkpoint_payload["stage_id"]
    restored.structural_development_effective_gain_by_instance_id = deepcopy(checkpoint_payload["effective_gains"])
    agent.stats.survival_steps = MATURE_START_STEP
    restored.stats.survival_steps = MATURE_START_STEP
    plan_a, step_a = _run_enabled_step(agent, fatigue=0.2)
    plan_b, step_b = _run_enabled_step(restored, fatigue=0.2)
    architecture_after = _architecture(agent)
    state_keys = sorted(str(module["module_instance_id"]) for module in agent.genome.modules)
    continuation_equal = bool(
        agent.structural_live_module_state == restored.structural_live_module_state
        and agent.structural_live_module_outputs == restored.structural_live_module_outputs
        and agent.structural_development_latch_by_instance_id
        == restored.structural_development_latch_by_instance_id
        and _stable_hash(step_a.summary) == _stable_hash(step_b.summary)
        and plan_a == plan_b
    )
    fixed_shape = bool(
        architecture_before == architecture_after
        and len(agent.genome.modules) == 2
        and len(agent.genome.connections) == 1
        and sorted(agent.structural_live_module_state) == state_keys
        and sorted(agent.structural_live_module_outputs) == state_keys
    )
    return {
        "arm": "checkpoint_continuation_fixed_shape",
        "passed": bool(continuation_equal and fixed_shape),
        "checkpoint_payload": checkpoint_payload,
        "continuation_equal": continuation_equal,
        "fixed_shape": fixed_shape,
        "module_count": len(agent.genome.modules),
        "connection_count": len(agent.genome.connections),
        "architecture_hash_before": _stable_hash(architecture_before),
        "architecture_hash_after": _stable_hash(architecture_after),
        "runtime_instance_ids": state_keys,
        "random_draws": int(plan_a.telemetry["random_draws"])
        + int(plan_b.telemetry["random_draws"]),
    }


def run_technical_probe() -> dict[str, Any]:
    arms = [
        _disabled_legacy_arm(),
        _enabled_neutral_arm(),
        _stage_expression_arm(
            "early_expression",
            age=0,
            expected_stage="early",
            expected_gain=0.25,
        ),
        _stage_expression_arm(
            "mature_expression",
            age=MATURE_START_STEP,
            expected_stage="mature",
            expected_gain=0.5,
        ),
        _stage_expression_arm(
            "late_expression",
            age=LATE_START_STEP,
            expected_stage="late",
            expected_gain=0.75,
        ),
        _dependency_arm("dependency_blocked", root_gain=0.0, expected_ready=False),
        _dependency_arm("dependency_ready", root_gain=1.0, expected_ready=True),
        _latch_arm(),
        _inactive_freeze_stale_arm(),
        _reactivation_arm(),
        _lesion_arm(),
        _checkpoint_fixed_shape_arm(),
    ]
    names = [str(arm["arm"]) for arm in arms]
    payload: dict[str, Any] = {
        "schema_version": PROBE_SCHEMA_VERSION,
        "purpose": "bounded developmental expression and dependency mechanics only",
        "stage_boundaries": {
            "mature_start_step": MATURE_START_STEP,
            "late_start_step": LATE_START_STEP,
        },
        "arms": arms,
        "checks": {
            "exactly_twelve_arms": len(arms) == EXPECTED_ARM_COUNT,
            "arm_names_unique": len(names) == len(set(names)),
            "all_arms_passed": all(bool(arm["passed"]) for arm in arms),
            "execution_rng_free": all(int(arm["random_draws"]) == 0 for arm in arms),
            "checkpoint_continuation_equal": bool(arms[-1]["continuation_equal"]),
            "physical_shape_fixed": bool(arms[-1]["fixed_shape"]),
        },
        "interpretation_boundary": (
            "technical mechanics only; no ecology, adaptiveness, cognition, "
            "emergence, intelligence, fitness, or selection claim"
        ),
    }
    payload["probe_hash"] = _stable_hash(payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    payload = run_technical_probe()
    text = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0 if all(payload["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
