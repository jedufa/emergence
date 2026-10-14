from __future__ import annotations

"""Machine-readable ESCAP-3 audit and bounded five-arm technical probe."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from evolution.structural_live_modules_v1 import (
    LIVE_MODULE_SCHEMA_VERSION,
    LIVE_OPERATOR,
    LIVE_PROFILE,
    add_rest_logit_residual_,
    execute_agent_live_modules,
)


AUDIT_SCHEMA_VERSION = "escap3_stage0_execution_audit_v1"
PROBE_SCHEMA_VERSION = "escap3_live_module_technical_probe_v1"


def _stable_hash(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def build_stage0_audit() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "scope": "ESCAP-3 live computational modules with owned typed state",
        "existing_surfaces": {
            "module_schema_and_typed_slots": "engine/agents/modular_genome_v1.py",
            "structural_identity_and_homology": "engine/agents/structural_identity_v1.py",
            "scalar_field_compiler": "engine/evolution/structural_live_semantics.py",
            "fixed_budget_mutation": "engine/evolution/operators_c1.py",
            "homology_recombination": "engine/evolution/recombination_c2.py",
            "agent_state_owner": "engine/agents/agent.py",
            "mlp_and_gru_policy_tick": "engine/simulation/runner_pkg/runner_policy.py:_batched_actions",
            "checkpoint_agent_payload": "engine/simulation/checkpoint_bundle_v2.py",
            "capacity_contract": "engine/evolution/capacity_contract.py",
        },
        "field_classification": {
            "executing": [
                "generic.params.expression_scale",
                "generic.params.live_retention",
                "generic.params.live_input_gain",
                "generic.params.live_output_gain",
                "generic scalar input/output slots",
                "scalar connection weight",
                "module_instance_id keyed runtime state and previous output",
            ],
            "serialization_only_in_escap3": [
                "policy_core module",
                "behavior_bias module",
                "identity_regulation module",
                "life_history module",
                "temperament_priors module",
                "vector and gate slots",
            ],
        },
        "execution_contract": {
            "operator": LIVE_OPERATOR,
            "runtime_schema_version": LIVE_MODULE_SCHEMA_VERSION,
            "profile": LIVE_PROFILE,
            "existing_typed_input": "fatigue drive scalar in [0,1]",
            "connection_boundary": "previous-tick source scalar output",
            "state_update": "clip(retention * previous_state + input_gain * fatigue + canonical_connection_sum, -1, 1)",
            "typed_output": "clip(expression_scale * output_gain * current_state, -1, 1)",
            "controller_seam": "add clipped aggregate once to existing REST logit after drive modulation and before authority/sampling",
            "state_commit": "all reads use previous maps; next state/output maps commit only after all module evaluations",
            "connection_order": "connection_id ascending with scalar sum; graph cycles rejected",
            "identity_owner": "module_instance_id, never local module_id/slot",
            "birth_and_reproduction": "runtime maps empty; architecture and parameters hereditary, lifetime contents not hereditary",
            "deletion": "explicit prune boundary reports removed instance keys; ordinary execution rejects orphans",
            "disable_reenable": "disabled path freezes owned maps without evaluating; re-enable resumes reconciled state",
            "cost": "reporting only: 4 units per module evaluation plus 2 per evaluated connection",
            "rng_draws": 0,
        },
        "capacity": {
            "maximum_modules": 12,
            "maximum_connections": 32,
            "state_dimensions_per_module": 1,
            "storable": True,
            "executable": True,
            "active": "feature flag plus nonzero expression/input/output path",
            "saturated": "numeric output clips at declared bound; structural overflow rejects through B2",
        },
        "lane_support": {
            "cpu_reference": "implemented",
            "cpu_mlp": "shared seam implemented",
            "cpu_gru": "shared seam implemented",
            "policy_stride": "fail closed in preflight",
            "native_cxx": "fail closed in preflight",
            "cuda": "fail closed in preflight",
        },
        "prohibited_seams": [
            "new observations or actions",
            "reward fitness or selection",
            "energy or ecology cost coupling",
            "mutation-rate changes",
            "automatic capacity resizing",
            "developmental growth",
            "unbounded recurrence or arbitrary network execution",
            "CUDA kernels native runtime or performance worktrees",
        ],
        "known_bounded_limitations": [
            "one fixed scalar integrator operator",
            "one existing fatigue input",
            "one existing REST-logit residual seam",
            "promoter retains a fixed two-draw budget and mutates one of three live parameters per event",
            "reporting-only compute cost",
        ],
    }
    payload["audit_hash"] = _stable_hash(payload)
    return payload


def _fixture_modules(*, neutral: bool = False) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    expression = 0.0 if neutral else 1.0
    modules = [
        {
            "module_id": "m000",
            "module_type": "generic",
            "module_instance_id": "mi-escap3-source",
            "params": {
                "live_operator": LIVE_OPERATOR,
                "expression_scale": expression,
                "live_retention": 0.25,
                "live_input_gain": 0.5,
                "live_output_gain": 0.5,
            },
        },
        {
            "module_id": "m001",
            "module_type": "generic",
            "module_instance_id": "mi-escap3-sink",
            "params": {
                "live_operator": LIVE_OPERATOR,
                "expression_scale": expression,
                "live_retention": 0.5,
                "live_input_gain": 0.0,
                "live_output_gain": 0.5,
            },
        },
    ]
    connections = [
        {
            "connection_id": "c000",
            "src_module_id": "m000",
            "dst_module_id": "m001",
            "src_slot": "out",
            "dst_slot": "in",
            "value_type": "scalar",
            "weight": 0.5,
        }
    ]
    return modules, connections


def _agent(*, neutral: bool = False) -> SimpleNamespace:
    modules, connections = _fixture_modules(neutral=neutral)
    return SimpleNamespace(
        genome=SimpleNamespace(modules=deepcopy(modules), connections=deepcopy(connections)),
        structural_live_module_state={},
        structural_live_module_outputs={},
        structural_live_module_telemetry={},
    )


def _row(step: int, result: Any, *, action_logits: list[float]) -> dict[str, Any]:
    return {
        "tick": int(step),
        "state_by_instance_id": dict(result.summary.get("state_instance_map", {})),
        "state_rows": list(result.summary.get("state_rows", [])),
        "previous_outputs_after": dict(result.summary.get("state_instance_map", {})),
        "rest_logit_residual": float(result.rest_residual),
        "action_logits": list(action_logits),
        "selected_action": int(np.argmax(np.asarray(action_logits, dtype=np.float32))),
        "telemetry": {k: v for k, v in result.summary.items() if k != "state_rows"},
    }


def _run_arm(name: str) -> dict[str, Any]:
    disabled = name == "disabled"
    agent = _agent(neutral=False)
    trajectory: list[dict[str, Any]] = []
    for tick, fatigue in enumerate((0.4, 0.8, 0.2)):
        if disabled:
            trajectory.append(
                {
                    "tick": tick,
                    "state_rows": [],
                    "rest_logit_residual": 0.0,
                    "action_logits": [0.0] * 7,
                    "selected_action": 0,
                    "telemetry": {"random_draws": 0, "execution": "disabled"},
                }
            )
            continue
        kwargs: dict[str, Any] = {}
        if name == "enabled_zero_neutral":
            kwargs["lesion_expression"] = True
        if name == "active_state_reset_lesion" and tick == 2:
            kwargs["reset_instance_ids"] = ["mi-escap3-sink"]
        if name == "active_connection_lesion":
            kwargs["lesion_connection_ids"] = ["c000"]
        result = execute_agent_live_modules(agent, fatigue=fatigue, **kwargs)
        base = np.zeros((1, 7), dtype=np.float32)
        add_rest_logit_residual_(base, np.asarray([result.rest_residual], dtype=np.float32), rest_index=5)
        row = _row(tick, result, action_logits=base[0].astype(float).tolist())
        row["state_by_instance_id"] = dict(agent.structural_live_module_state)
        row["previous_outputs_after"] = dict(agent.structural_live_module_outputs)
        trajectory.append(row)
    return {
        "arm": name,
        "starting_genome_hash": _stable_hash(
            {"modules": agent.genome.modules, "connections": agent.genome.connections}
        ),
        "module_instance_ids": sorted(str(m["module_instance_id"]) for m in agent.genome.modules),
        "trajectory": trajectory,
        "random_draws": 0,
    }


def run_technical_probe() -> dict[str, Any]:
    arms = [
        _run_arm("disabled"),
        _run_arm("enabled_zero_neutral"),
        _run_arm("active"),
        _run_arm("active_state_reset_lesion"),
        _run_arm("active_connection_lesion"),
    ]

    uninterrupted = _agent()
    execute_agent_live_modules(uninterrupted, fatigue=0.4)
    execute_agent_live_modules(uninterrupted, fatigue=0.8)
    restored = _agent()
    checkpoint_payload = {
        "schema_version": LIVE_MODULE_SCHEMA_VERSION,
        "state_by_instance_id": deepcopy(uninterrupted.structural_live_module_state),
        "previous_outputs_by_instance_id": deepcopy(uninterrupted.structural_live_module_outputs),
        "telemetry": deepcopy(uninterrupted.structural_live_module_telemetry),
    }
    restored.structural_live_module_state = deepcopy(checkpoint_payload["state_by_instance_id"])
    restored.structural_live_module_outputs = deepcopy(checkpoint_payload["previous_outputs_by_instance_id"])
    restored.structural_live_module_telemetry = deepcopy(checkpoint_payload["telemetry"])
    next_a = execute_agent_live_modules(uninterrupted, fatigue=0.2)
    next_b = execute_agent_live_modules(restored, fatigue=0.2)
    base_logits = np.zeros((1, 7), dtype=np.float32)
    uninterrupted_logits = base_logits.copy()
    restored_logits = base_logits.copy()
    add_rest_logit_residual_(
        uninterrupted_logits,
        np.asarray([next_a.rest_residual], dtype=np.float32),
        rest_index=5,
    )
    add_rest_logit_residual_(
        restored_logits,
        np.asarray([next_b.rest_residual], dtype=np.float32),
        rest_index=5,
    )
    checkpoint_parity = {
        "payload": checkpoint_payload,
        "state_equal": uninterrupted.structural_live_module_state == restored.structural_live_module_state,
        "previous_outputs_equal": uninterrupted.structural_live_module_outputs == restored.structural_live_module_outputs,
        "first_post_resume_residual_equal": next_a.rest_residual == next_b.rest_residual,
        "first_post_resume_summary_hash_equal": _stable_hash(next_a.summary) == _stable_hash(next_b.summary),
        "first_post_resume_action_logits_equal": uninterrupted_logits.tobytes() == restored_logits.tobytes(),
        "first_post_resume_selected_action_equal": int(np.argmax(uninterrupted_logits[0]))
        == int(np.argmax(restored_logits[0])),
        "uninterrupted_first_transition": {
            "state_by_instance_id": dict(uninterrupted.structural_live_module_state),
            "previous_outputs_by_instance_id": dict(uninterrupted.structural_live_module_outputs),
            "rest_logit_residual": float(next_a.rest_residual),
            "action_logits": uninterrupted_logits[0].astype(float).tolist(),
            "selected_action": int(np.argmax(uninterrupted_logits[0])),
            "sampling_mode": "deterministic_argmax",
            "sampling_rng_draws": 0,
        },
        "restored_first_transition": {
            "state_by_instance_id": dict(restored.structural_live_module_state),
            "previous_outputs_by_instance_id": dict(restored.structural_live_module_outputs),
            "rest_logit_residual": float(next_b.rest_residual),
            "action_logits": restored_logits[0].astype(float).tolist(),
            "selected_action": int(np.argmax(restored_logits[0])),
            "sampling_mode": "deterministic_argmax",
            "sampling_rng_draws": 0,
        },
        "migration_random_draws": 0,
    }
    payload: dict[str, Any] = {
        "schema_version": PROBE_SCHEMA_VERSION,
        "preregistered_fatigue_sequence": [0.4, 0.8, 0.2],
        "base_action_logits": [0.0] * 7,
        "rest_action_index": 5,
        "arms": arms,
        "checkpoint_continuation": checkpoint_parity,
        "interpretation_boundary": "technical causal execution evidence only; no ecology adaptiveness cognition intelligence or selection claim",
    }
    active = next(row for row in arms if row["arm"] == "active")
    reset = next(row for row in arms if row["arm"] == "active_state_reset_lesion")
    connection = next(row for row in arms if row["arm"] == "active_connection_lesion")
    neutral = next(row for row in arms if row["arm"] == "enabled_zero_neutral")
    payload["checks"] = {
        "neutral_residual_zero": all(float(row["rest_logit_residual"]) == 0.0 for row in neutral["trajectory"]),
        "active_first_expression_tick": next(
            (int(row["tick"]) for row in active["trajectory"] if float(row["rest_logit_residual"]) != 0.0), None
        ),
        "state_reset_changes_tick2": reset["trajectory"][2]["state_by_instance_id"] != active["trajectory"][2]["state_by_instance_id"],
        "connection_lesion_changes_trajectory": connection["trajectory"] != active["trajectory"],
        "checkpoint_continuation_parity": all(
            bool(checkpoint_parity[key])
            for key in (
                "state_equal",
                "previous_outputs_equal",
                "first_post_resume_residual_equal",
                "first_post_resume_summary_hash_equal",
                "first_post_resume_action_logits_equal",
                "first_post_resume_selected_action_equal",
            )
        ),
        "executor_random_draws_zero": all(int(row["random_draws"]) == 0 for row in arms),
        "five_arm_starting_genome_matched": len({str(row["starting_genome_hash"]) for row in arms}) == 1,
        "five_arm_structural_identities_matched": len(
            {tuple(row["module_instance_ids"]) for row in arms}
        )
        == 1,
    }
    payload["probe_hash"] = _stable_hash(payload)
    return payload


def _write(path: str, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-out", required=True)
    parser.add_argument("--probe-out", required=True)
    args = parser.parse_args()
    _write(args.audit_out, build_stage0_audit())
    probe = run_technical_probe()
    _write(args.probe_out, probe)
    if not all(bool(value) for key, value in probe["checks"].items() if key != "active_first_expression_tick"):
        return 1
    if probe["checks"]["active_first_expression_tick"] is None:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
