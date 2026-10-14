#!/usr/bin/env python3
from __future__ import annotations

"""Deterministic ESCAP-7 twenty-arm CPU/reference technical probe."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = PROJECT_ROOT / "engine"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from agents.agent import ACTION_VOCABULARY  # noqa: E402
from agents.structural_identity_v1 import new_native_identity  # noqa: E402
from env.gridworld_pkg.constants import ACT_MOVE_N, ACT_REST, N_ACTIONS  # noqa: E402
from evolution.operators_c1 import (  # noqa: E402
    _op_duplicate_with_decay,
    mutate_gated_recurrent_operator,
)
from evolution.structural_gated_recurrent_v1 import (  # noqa: E402
    GATED_OPERATOR,
    neutral_gated_recurrent_record,
)
from evolution.structural_live_modules_v1 import (  # noqa: E402
    LIVE_PROFILE,
    execute_agent_live_modules,
)
from evolution.structural_output_routing_v1 import (  # noqa: E402
    ACTION_SEMANTIC_REGISTRY_VERSION,
    OUTPUT_ROUTING_BLOCK_KEY,
    OUTPUT_ROUTING_SCHEMA_VERSION,
    compile_agent_output_routing,
)
from simulation.runner_pkg.runner_base import (  # noqa: E402
    _validate_escap7_gated_recurrent_config,
)
from simulation.runner_pkg.runner_config import SimConfig  # noqa: E402


PROBE_SCHEMA_VERSION = "escap7_fixed_width_gated_technical_probe_v1"


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _identity(index: int) -> dict[str, Any]:
    return new_native_identity(
        context={
            "campaign_id": "escap7-technical-probe",
            "branch_id": "fixed-width-gated",
            "generation": 1,
            "tick": 0,
            "child_agent_id": int(index),
            "parent_agent_ids": [1, 2],
            "lineage_token": "escap7-technical-probe-lineage",
        },
        operator_id="probe_fixture",
        attempt_index=int(index),
        operator_role=f"module-{index}",
    )


def _record(**updates: float) -> dict[str, Any]:
    record = neutral_gated_recurrent_record()
    record.update(updates)
    return record


def _binding(semantic_id: str, gain: float = 1.0) -> dict[str, Any]:
    return {
        "semantic_id": semantic_id,
        "source_slot": "out",
        "gain": float(gain),
        "enabled": True,
    }


def _module(
    index: int,
    *,
    gated: bool = True,
    expression: float = 1.0,
    output_gain: float = 1.0,
    bindings: list[dict[str, Any]] | None = None,
    record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    identity = _identity(index)
    params: dict[str, Any] = {
        "live_operator": GATED_OPERATOR if gated else "bounded_scalar_integrator_v1",
        "expression_scale": float(expression),
    }
    if gated:
        params["gated_recurrent"] = deepcopy(
            record
            or _record(
                candidate_bias=0.5,
                candidate_input_gain=1.0,
                update_bias=1.0,
                output_gain=output_gain,
            )
        )
    else:
        params.update(
            {
                "live_retention": 0.5,
                "live_input_gain": 1.0,
                "live_output_gain": float(output_gain),
            }
        )
    if bindings is not None:
        params[OUTPUT_ROUTING_BLOCK_KEY] = {
            "schema_version": OUTPUT_ROUTING_SCHEMA_VERSION,
            "registry_version": ACTION_SEMANTIC_REGISTRY_VERSION,
            "bindings": deepcopy(bindings),
        }
    return {
        "module_id": f"m{index:03d}",
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


def _agent(
    modules: list[dict[str, Any]],
    connections: list[dict[str, Any]] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        genome=SimpleNamespace(
            modules=deepcopy(modules),
            connections=deepcopy(connections or []),
        ),
        structural_live_module_state={},
        structural_live_module_outputs={},
        structural_live_module_telemetry={},
        structural_development_latch_by_instance_id={},
        structural_development_stage_id="",
        structural_development_effective_gain_by_instance_id={},
    )


def _cfg() -> SimConfig:
    cfg = SimConfig()
    cfg.device = "cpu"
    cfg.bank_device = "cpu"
    cfg.policy_tick_stride = 1
    cfg.structural_live_modules_enable = True
    cfg.structural_live_modules_profile = LIVE_PROFILE
    cfg.structural_gated_recurrent_enable = True
    cfg.structural_gated_recurrent_profile = "escap7_fixed_width_gated_recurrent_v1"
    return cfg


def _development_plan(modules: list[dict[str, Any]], gain: float) -> SimpleNamespace:
    return SimpleNamespace(
        effective_gain_by_instance_id={
            str(module["module_instance_id"]): float(gain) for module in modules
        },
        latch_by_instance_id={},
        stage_id="mature",
        telemetry={"stage_id": "mature", "random_draws": 0},
    )


def _logit_record(
    residuals: np.ndarray,
) -> tuple[list[float], int, str]:
    logits = np.asarray(residuals, dtype=np.float32).reshape(N_ACTIONS)
    selected = int(np.argmax(logits))
    return logits.astype(float).tolist(), selected, str(ACTION_VOCABULARY[selected])


def _arm_from_step(
    name: str,
    agent: SimpleNamespace,
    *,
    fatigue: float,
    mutation_draws: int = 0,
    developmental_plan: Any = None,
    reset_instance_ids: tuple[str, ...] = (),
    lesion_connection_ids: tuple[str, ...] = (),
    lesion_expression: bool = False,
    routed_observation_by_instance_id: dict[str, float] | None = None,
) -> dict[str, Any]:
    gated = any(
        str(module.get("params", {}).get("live_operator", "")) == GATED_OPERATOR
        for module in agent.genome.modules
    )
    step = execute_agent_live_modules(
        agent,
        fatigue=float(fatigue),
        gated_recurrent_enabled=gated,
        developmental_plan=developmental_plan,
        reset_instance_ids=reset_instance_ids,
        lesion_connection_ids=lesion_connection_ids,
        lesion_expression=lesion_expression,
        routed_observation_by_instance_id=routed_observation_by_instance_id,
    )
    residuals = np.zeros((N_ACTIONS,), dtype=np.float32)
    residuals[ACT_REST] = np.float32(step.rest_residual)
    logits, selected, selected_name = _logit_record(residuals)
    row = step.summary["state_rows"][0]
    return {
        "arm": name,
        "input": float(row["typed_input_combined_external"]),
        "prior_state": float(row["state_before"]),
        "reset": row.get("reset_gate"),
        "update": row.get("update_gate"),
        "candidate": row.get("candidate"),
        "new_state": float(row["state_after"]),
        "public_output": float(row["typed_output"]),
        "developmental_gain": float(row.get("developmental_expression_gain", 1.0)),
        "action_residual": residuals.astype(float).tolist(),
        "full_logits": logits,
        "selected_action": selected,
        "selected_action_name": selected_name,
        "execution_rng_draws": int(step.summary["random_draws"]),
        "mutation_draw_counts": int(mutation_draws),
        "state_width": int(step.summary.get("owned_state_width_per_module", 1)),
        "reported_compute_cost": int(step.summary["reported_compute_cost"]),
        "shape_digest": "",
        "details": {"summary": deepcopy(step.summary)},
    }


def _route_arm(
    name: str,
    semantic_id: str,
    *,
    lesion: bool = False,
) -> dict[str, Any]:
    module = _module(
        200 + len(name),
        record=_record(candidate_bias=1.0, update_bias=4.0, output_gain=1.0),
        bindings=[_binding(semantic_id)],
    )
    agent = _agent([module])
    base = _arm_from_step(name, agent, fatigue=0.0)
    lesion_keys = (
        [(str(module["module_instance_id"]), semantic_id, "out")]
        if lesion
        else []
    )
    routed = compile_agent_output_routing(
        agent,
        legacy_rest_residual=base["action_residual"][ACT_REST],
        lesion_binding_keys=lesion_keys,
    )
    residual = routed.residual.astype(np.float32)
    logits, selected, selected_name = _logit_record(residual)
    base.update(
        {
            "action_residual": residual.astype(float).tolist(),
            "full_logits": logits,
            "selected_action": selected,
            "selected_action_name": selected_name,
            "details": {**base["details"], "output_routing": routed.summary},
        }
    )
    return base


def run_probe() -> dict[str, Any]:
    arms: list[dict[str, Any]] = []

    arms.append(_arm_from_step("legacy_integrator", _agent([_module(1, gated=False)]), fatigue=0.4))
    arms.append(
        _arm_from_step(
            "neutral_eligible_integrator_before_acquisition",
            _agent([_module(2, gated=False, expression=0.0)]),
            fatigue=0.4,
        )
    )

    neutral_source = _module(3, gated=False, expression=0.0)
    acquisition = mutate_gated_recurrent_operator([neutral_source], 0.1, 0.5, enabled=True)
    arms.append(
        _arm_from_step(
            "neutral_gated_acquisition",
            _agent(acquisition.modules),
            fatigue=0.4,
            mutation_draws=2,
        )
    )

    hidden_module = _module(4, record=neutral_gated_recurrent_record())
    hidden_agent = _agent([hidden_module])
    hidden_agent.structural_live_module_state = {str(hidden_module["module_instance_id"]): 0.8}
    hidden_agent.structural_live_module_outputs = {str(hidden_module["module_instance_id"]): 0.0}
    arms.append(_arm_from_step("hidden_state_zero_output_gain", hidden_agent, fatigue=0.0))

    activated = _module(5, record=_record(output_gain=1.0))
    activated_agent = _agent([activated])
    activated_agent.structural_live_module_state = {str(activated["module_instance_id"]): 0.8}
    activated_agent.structural_live_module_outputs = {str(activated["module_instance_id"]): 0.0}
    arms.append(_arm_from_step("output_gain_activation", activated_agent, fatigue=0.0))

    mutation_specs = (
        ("reset_gate_mutation", 0.25),
        ("update_gate_mutation", 0.45),
        ("candidate_mutation", 0.65),
    )
    for index, (name, draw0) in enumerate(mutation_specs, start=6):
        mutation = mutate_gated_recurrent_operator([_module(index)], draw0, 0.37, enabled=True)
        arms.append(
            _arm_from_step(name, _agent(mutation.modules), fatigue=0.4, mutation_draws=2)
        )

    arms.append(_arm_from_step("self_energy_input", _agent([_module(9)]), fatigue=0.75))

    source = _module(10, record=_record(candidate_bias=1.0, update_bias=4.0, output_gain=1.0))
    sink = _module(11, record=_record(candidate_input_gain=1.0, update_bias=4.0, output_gain=1.0))
    connection = {
        "connection_id": "c000",
        "src_module_id": source["module_id"],
        "dst_module_id": sink["module_id"],
        "src_slot": "out",
        "dst_slot": "in",
        "value_type": "scalar",
        "channel": "typed",
        "weight": 0.5,
    }
    connection_agent = _agent([source, sink], [connection])
    execute_agent_live_modules(connection_agent, fatigue=0.0, gated_recurrent_enabled=True)
    connection_arm = _arm_from_step("previous_tick_connection_input", connection_agent, fatigue=0.0)
    connection_arm["details"]["sink_row"] = deepcopy(
        next(
            row
            for row in connection_agent.structural_live_module_telemetry["state_rows"]
            if row["module_instance_id"] == str(sink["module_instance_id"])
        )
    )
    arms.append(connection_arm)

    mature_module = _module(12)
    arms.append(
        _arm_from_step(
            "mature_stage_developmental_activation",
            _agent([mature_module]),
            fatigue=0.4,
            developmental_plan=_development_plan([mature_module], 0.5),
        )
    )
    arms.append(_route_arm("output_routing_movement", "move_north"))
    arms.append(_route_arm("output_routing_rest", "rest"))

    input_module = _module(15)
    input_control_agent = _agent([input_module])
    input_control = _arm_from_step(
        "input_control",
        input_control_agent,
        fatigue=0.0,
        routed_observation_by_instance_id={str(input_module["module_instance_id"]): 0.5},
    )
    input_lesion = _arm_from_step(
        "input_lesion",
        _agent([input_module]),
        fatigue=0.0,
        routed_observation_by_instance_id={str(input_module["module_instance_id"]): 0.0},
    )
    input_lesion["details"]["control_new_state"] = input_control["new_state"]
    arms.append(input_lesion)

    output_arm = _route_arm("output_lesion", "move_north", lesion=True)
    output_arm["details"]["lesion"] = "public ESCAP5 binding lesion"
    arms.append(output_arm)

    reset_module = _module(17)
    reset_agent = _agent([reset_module])
    reset_id = str(reset_module["module_instance_id"])
    reset_agent.structural_live_module_state = {reset_id: 0.75}
    reset_agent.structural_live_module_outputs = {reset_id: 0.25}
    arms.append(
        _arm_from_step(
            "state_reset_lesion",
            reset_agent,
            fatigue=0.0,
            reset_instance_ids=(reset_id,),
        )
    )

    checkpoint_module = _module(18)
    checkpoint_source = _agent([checkpoint_module])
    _arm_from_step("checkpoint_seed", checkpoint_source, fatigue=0.4)
    restored = _agent(checkpoint_source.genome.modules)
    restored.structural_live_module_state = deepcopy(checkpoint_source.structural_live_module_state)
    restored.structural_live_module_outputs = deepcopy(checkpoint_source.structural_live_module_outputs)
    source_next = _arm_from_step("checkpoint_source_next", checkpoint_source, fatigue=0.2)
    restored_next = _arm_from_step("checkpoint_continuation", restored, fatigue=0.2)
    restored_next["details"]["continuation_parity"] = source_next == {
        **restored_next,
        "arm": "checkpoint_source_next",
        "details": source_next["details"],
    }
    restored_next["details"]["source_state"] = deepcopy(checkpoint_source.structural_live_module_state)
    restored_next["details"]["restored_state"] = deepcopy(restored.structural_live_module_state)
    arms.append(restored_next)

    duplicate_source = _module(19, output_gain=1.0)
    duplicate_payload = {"modules": [duplicate_source], "connections": []}
    duplicate_context = {
        "operator_id": "probe_duplicate",
        "attempt_index": 0,
        "identity_context": {
            "campaign_id": "escap7-technical-probe",
            "branch_id": "fixed-width-gated",
            "generation": 2,
            "tick": 0,
            "child_agent_id": 19,
            "parent_agent_ids": [1, 2],
            "lineage_token": "escap7-duplicate-lineage",
        },
    }
    duplicate_payload, _ = _op_duplicate_with_decay(
        duplicate_payload, [0.0, 0.5, 0.5], duplicate_context
    )
    duplicate_arm = _arm_from_step(
        "duplication_neutralized_output",
        _agent(duplicate_payload["modules"]),
        fatigue=0.4,
        mutation_draws=3,
    )
    duplicate_arm["details"]["duplicate_output_gain"] = duplicate_payload["modules"][1]["params"]["gated_recurrent"]["output_gain"]
    arms.append(duplicate_arm)

    repack_a = _agent([_module(20), _module(21)])
    repack_b = _agent(list(reversed(repack_a.genome.modules)))
    repack_arm = _arm_from_step("local_repacking_parity", repack_a, fatigue=0.4)
    repack_other = _arm_from_step("local_repacking_parity_other", repack_b, fatigue=0.4)
    repack_arm["details"]["state_map_parity"] = (
        repack_a.structural_live_module_state == repack_b.structural_live_module_state
    )
    repack_arm["details"]["other_public_output"] = repack_other["public_output"]
    arms.append(repack_arm)

    cfg = _cfg()
    cfg.runnercore_authoritative_enable = True
    rejected = False
    error = ""
    try:
        _validate_escap7_gated_recurrent_config(
            cfg,
            policy_device=torch.device("cpu"),
            bank_device=torch.device("cpu"),
        )
    except RuntimeError as exc:
        rejected = True
        error = str(exc)
    runner_arm = _arm_from_step("runnercore_rejection_fixture", _agent([_module(22)]), fatigue=0.0)
    runner_arm["details"]["runnercore_rejected"] = rejected
    runner_arm["details"]["error"] = error
    arms.append(runner_arm)

    shape_contract = {
        "state_width_per_module": 1,
        "module_slots_change": 0,
        "connection_slots_change": 0,
        "observation_width_change": 0,
        "action_width": N_ACTIONS,
        "mlp_hidden_width": 64,
        "central_gru_hidden_width": 64,
        "central_gru_shape_change": 0,
    }
    shape_digest = _hash(shape_contract)
    for arm in arms:
        arm["shape_digest"] = shape_digest

    by_name = {row["arm"]: row for row in arms}
    checks = {
        "twenty_arms_present": len(arms) == 20,
        "all_execution_rng_free": all(row["execution_rng_draws"] == 0 for row in arms),
        "all_state_width_one": all(row["state_width"] == 1 for row in arms),
        "neutral_acquisition_output_zero": by_name["neutral_gated_acquisition"]["public_output"] == 0.0,
        "hidden_state_evolves_at_zero_output": by_name["hidden_state_zero_output_gain"]["new_state"] != by_name["hidden_state_zero_output_gain"]["prior_state"] and by_name["hidden_state_zero_output_gain"]["public_output"] == 0.0,
        "output_activation_nonzero": by_name["output_gain_activation"]["public_output"] != 0.0,
        "connection_previous_tick_nonzero": by_name["previous_tick_connection_input"]["details"]["sink_row"]["typed_input_previous_outputs"] != 0.0,
        "movement_routing_nonzero": by_name["output_routing_movement"]["action_residual"][ACT_MOVE_N] != 0.0,
        "rest_routing_nonzero": by_name["output_routing_rest"]["action_residual"][ACT_REST] != 0.0,
        "input_lesion_changes_state": by_name["input_lesion"]["new_state"] != by_name["input_lesion"]["details"]["control_new_state"],
        "output_lesion_zeroes_movement_route": by_name["output_lesion"]["action_residual"][ACT_MOVE_N] == 0.0 and by_name["output_lesion"]["details"]["output_routing"]["lesioned_binding_count"] == 1,
        "checkpoint_continuation_state_equal": by_name["checkpoint_continuation"]["details"]["source_state"] == by_name["checkpoint_continuation"]["details"]["restored_state"],
        "duplicate_neutralized": by_name["duplication_neutralized_output"]["details"]["duplicate_output_gain"] == 0.0,
        "repacking_parity": bool(by_name["local_repacking_parity"]["details"]["state_map_parity"]),
        "runnercore_rejected": bool(by_name["runnercore_rejection_fixture"]["details"]["runnercore_rejected"]),
    }
    payload: dict[str, Any] = {
        "schema_version": PROBE_SCHEMA_VERSION,
        "lane": "python_reference_cpu_stride1",
        "shape_contract": shape_contract,
        "shape_digest": shape_digest,
        "arms": arms,
        "checks": checks,
        "interpretation_boundary": "bounded technical evidence only; no adaptiveness, cognition, intelligence, learning, emergence, or selection claim",
    }
    payload["probe_hash"] = _hash(payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    payload = run_probe()
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0 if all(payload["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
