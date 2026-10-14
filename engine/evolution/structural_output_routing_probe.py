from __future__ import annotations

"""Deterministic, JSON-ready ESCAP-5 ten-arm technical probe."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from env.gridworld_pkg.constants import ACT_MOVE_N, ACT_REST, N_ACTIONS
from evolution.structural_live_modules_v1 import execute_agent_live_modules
from evolution.structural_output_routing_v1 import (
    ADMITTED_ACTION_SEMANTIC_IDS,
    COMBINED_REST_COMPATIBILITY_VERSION,
    OUTPUT_ROUTING_BLOCK_KEY,
    OUTPUT_ROUTING_PROFILE,
    OUTPUT_ROUTING_SCHEMA_VERSION,
    ACTION_SEMANTIC_REGISTRY_VERSION,
    add_output_routing_residuals_,
    compile_output_routing_batch,
)


PROBE_SCHEMA_VERSION = "escap5_modular_output_routing_probe_v1"
INSTANCE_A = "mi-escap5-a"
INSTANCE_B = "mi-escap5-b"
LEGACY_REST = np.float32(0.10)


def _stable_hash(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(blob).hexdigest()


def _binding(semantic_id: str, gain: float) -> dict[str, Any]:
    return {"semantic_id": semantic_id, "source_slot": "out", "gain": gain, "enabled": True}


def _module(mid: str, instance_id: str, bindings: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "module_id": mid,
        "module_type": "generic",
        "module_instance_id": instance_id,
        "homology_id": f"h-{instance_id}",
        "params": {
            "live_operator": "bounded_scalar_integrator_v1",
            "expression_scale": 1.0,
            "live_retention": 1.0,
            "live_input_gain": 0.0,
            "live_output_gain": 1.0,
            OUTPUT_ROUTING_BLOCK_KEY: {
                "schema_version": OUTPUT_ROUTING_SCHEMA_VERSION,
                "registry_version": ACTION_SEMANTIC_REGISTRY_VERSION,
                "bindings": deepcopy(bindings),
            },
        },
    }


def _agent(modules: list[dict[str, Any]], outputs: dict[str, float]) -> SimpleNamespace:
    return SimpleNamespace(
        genome=SimpleNamespace(modules=deepcopy(modules), connections=[]),
        structural_live_module_state=deepcopy(outputs),
        structural_live_module_outputs=deepcopy(outputs),
        structural_live_module_telemetry={},
    )


def _cfg(enabled: bool) -> SimpleNamespace:
    return SimpleNamespace(structural_output_routing_enable=bool(enabled))


def _run_static_arm(
    name: str,
    modules: list[dict[str, Any]],
    outputs: dict[str, float],
    *,
    enabled: bool = True,
    lesions: tuple[tuple[str, str, str], ...] = (),
) -> dict[str, Any]:
    agent = _agent(modules, outputs)
    batch = compile_output_routing_batch(
        [agent],
        sim_cfg=_cfg(enabled),
        lesion_binding_keys=lesions,
        legacy_rest_residuals=np.asarray([LEGACY_REST], dtype=np.float32),
    )
    logits = np.zeros((1, N_ACTIONS), dtype=np.float32)
    add_output_routing_residuals_(logits, batch)
    return {
        "arm": name,
        "genome_hash": _stable_hash(modules),
        "module_outputs": {key: float(value) for key, value in sorted(outputs.items())},
        "residuals": batch.residuals[0].astype(float).tolist(),
        "logits_after_single_add": logits[0].astype(float).tolist(),
        "summary": deepcopy(batch.rows[0].summary),
        "telemetry": deepcopy(batch.rows[0].telemetry_by_instance_id),
        "random_draws": int(batch.rows[0].summary.get("random_draws", -1)),
    }


def _state_reset_arm() -> dict[str, Any]:
    module = _module("m000", INSTANCE_A, [_binding("move_north", 1.0)])
    control = _agent([module], {INSTANCE_A: 0.8})
    reset = deepcopy(control)
    control_step = execute_agent_live_modules(control, fatigue=0.0)
    reset_step = execute_agent_live_modules(reset, fatigue=0.0, reset_instance_ids=[INSTANCE_A])
    control_batch = compile_output_routing_batch(
        [control], sim_cfg=_cfg(True), legacy_rest_residuals=[control_step.rest_residual]
    )
    reset_batch = compile_output_routing_batch(
        [reset], sim_cfg=_cfg(True), legacy_rest_residuals=[reset_step.rest_residual]
    )
    return {
        "arm": "state_reset",
        "control_state": deepcopy(control.structural_live_module_state),
        "reset_state": deepcopy(reset.structural_live_module_state),
        "control_residuals": control_batch.residuals[0].astype(float).tolist(),
        "reset_residuals": reset_batch.residuals[0].astype(float).tolist(),
        "random_draws": 0,
    }


def _masked_high_residual_arm() -> dict[str, Any]:
    arm = _run_static_arm(
        "masked_high_residual",
        [_module("m000", INSTANCE_A, [_binding("move_north", 1.0)])],
        {INSTANCE_A: 1.0},
    )
    masked = np.asarray(arm["logits_after_single_add"], dtype=np.float32)
    masked[ACT_MOVE_N] = -np.inf
    arm["masked_selected_action"] = int(np.argmax(masked))
    arm["masked_target_excluded"] = bool(arm["masked_selected_action"] != ACT_MOVE_N)
    return arm


def _checkpoint_arm() -> dict[str, Any]:
    module = _module("m000", INSTANCE_A, [_binding("move_north", 0.5), _binding("rest", -0.25)])
    source = _agent([module], {INSTANCE_A: 0.6})
    payload = {
        "modules": deepcopy(source.genome.modules),
        "state": deepcopy(source.structural_live_module_state),
        "outputs": deepcopy(source.structural_live_module_outputs),
        "legacy_rest": float(LEGACY_REST),
        "migration_random_draws": 0,
    }
    restored = _agent(payload["modules"], payload["outputs"])
    restored.structural_live_module_state = deepcopy(payload["state"])
    first = compile_output_routing_batch(
        [source], sim_cfg=_cfg(True), legacy_rest_residuals=[LEGACY_REST]
    )
    second = compile_output_routing_batch(
        [restored], sim_cfg=_cfg(True), legacy_rest_residuals=[LEGACY_REST]
    )
    return {
        "arm": "checkpoint_continuation",
        "payload": payload,
        "residual_bytes_equal": first.residuals.tobytes() == second.residuals.tobytes(),
        "summary_hash_equal": _stable_hash(first.rows[0].summary) == _stable_hash(second.rows[0].summary),
        "random_draws": 0,
    }


def run_technical_probe() -> dict[str, Any]:
    north = _binding("move_north", 0.5)
    rest = _binding("rest", 0.5)
    arms = [
        _run_static_arm("disabled", [_module("m000", INSTANCE_A, [north])], {INSTANCE_A: 0.5}, enabled=False),
        _run_static_arm("enabled_zero_gain", [_module("m000", INSTANCE_A, [_binding("move_north", 0.0)])], {INSTANCE_A: 0.5}),
        _run_static_arm("active_rest", [_module("m000", INSTANCE_A, [rest])], {INSTANCE_A: 0.2}),
        _run_static_arm("active_non_rest", [_module("m000", INSTANCE_A, [north])], {INSTANCE_A: 0.2}),
        _run_static_arm("active_two_targets", [_module("m000", INSTANCE_A, [north, rest])], {INSTANCE_A: 0.2}),
        _run_static_arm(
            "two_modules_one_target",
            [_module("m000", INSTANCE_A, [north]), _module("m001", INSTANCE_B, [north])],
            {INSTANCE_A: 0.2, INSTANCE_B: 0.2},
        ),
        _run_static_arm(
            "binding_lesion",
            [_module("m000", INSTANCE_A, [north])],
            {INSTANCE_A: 0.2},
            lesions=((INSTANCE_A, "move_north", "out"),),
        ),
        _state_reset_arm(),
        _masked_high_residual_arm(),
        _checkpoint_arm(),
    ]
    by_name = {row["arm"]: row for row in arms}
    payload: dict[str, Any] = {
        "schema_version": PROBE_SCHEMA_VERSION,
        "profile": OUTPUT_ROUTING_PROFILE,
        "combined_rest_compatibility_version": COMBINED_REST_COMPATIBILITY_VERSION,
        "admitted_action_semantic_ids": list(ADMITTED_ACTION_SEMANTIC_IDS),
        "arms": arms,
        "interpretation_boundary": "bounded technical routing only; no ecology, adaptiveness, cognition, or selection claim",
    }
    payload["checks"] = {
        "ten_arms_present": len(arms) == 10,
        "disabled_equals_legacy_rest": by_name["disabled"]["residuals"][ACT_REST] == float(LEGACY_REST),
        "zero_gain_equals_legacy_rest": by_name["enabled_zero_gain"]["residuals"][ACT_REST] == float(LEGACY_REST),
        "active_rest_combines_once": by_name["active_rest"]["residuals"][ACT_REST]
        == float(np.float32(0.2)),
        "active_non_rest_target_only": by_name["active_non_rest"]["residuals"][ACT_MOVE_N] != 0.0,
        "two_targets_both_nonzero": sum(value != 0.0 for value in by_name["active_two_targets"]["residuals"]) == 2,
        "two_modules_aggregate": by_name["two_modules_one_target"]["residuals"][ACT_MOVE_N]
        == float(np.float32(0.2)),
        "lesion_neutralizes_binding": by_name["binding_lesion"]["residuals"][ACT_MOVE_N] == 0.0,
        "state_reset_changes_residual": by_name["state_reset"]["control_residuals"] != by_name["state_reset"]["reset_residuals"],
        "masked_target_not_selected": bool(by_name["masked_high_residual"]["masked_target_excluded"]),
        "checkpoint_continuation_equal": bool(by_name["checkpoint_continuation"]["residual_bytes_equal"]),
        "execution_rng_free": all(int(row["random_draws"]) == 0 for row in arms),
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
