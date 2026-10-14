from __future__ import annotations

"""Machine-readable ESCAP-4 observation audit and bounded technical probe."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from agents.perception import perception_channel_registry, perception_channel_registry_dict
from evolution.structural_live_modules_v1 import add_rest_logit_residual_, execute_agent_live_modules
from evolution.structural_observation_routing_v1 import (
    ADMITTED_SEMANTIC_IDS,
    MAX_BINDINGS_PER_AGENT,
    MAX_BINDINGS_PER_MODULE,
    ROUTING_EXPOSURE_MODE,
    ROUTING_PROFILE,
    ROUTING_REGISTRY_VERSION,
    ROUTING_SCHEMA_VERSION,
    RoutingBatch,
    compile_module_routing,
    resolve_routing_values,
)


AUDIT_SCHEMA_VERSION = "escap4_observation_semantics_audit_v1"
PROBE_SCHEMA_VERSION = "escap4_modular_observation_routing_probe_v1"
FATIGUE_SEQUENCE = (0.0, 0.0, 0.0)
ENERGY_SEQUENCE = (0.2, 0.4, 0.6)
HEALTH_SEQUENCE = (0.5, 0.5, 0.5)
LOCAL_SIZE = 9
MODULE_INSTANCE_ID = "mi-escap4-routing"


def _stable_hash(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _observation(energy: float, health: float) -> np.ndarray:
    row = np.zeros((LOCAL_SIZE + 2,), dtype=np.float32)
    row[LOCAL_SIZE] = np.float32(energy)
    row[LOCAL_SIZE + 1] = np.float32(health)
    return row


def _module() -> dict[str, Any]:
    return {
        "module_id": "m000",
        "module_type": "generic",
        "module_instance_id": MODULE_INSTANCE_ID,
        "homology_id": "h-escap4-routing",
        "params": {
            "live_operator": "bounded_scalar_integrator_v1",
            "expression_scale": 1.0,
            "live_retention": 0.5,
            "live_input_gain": 1.0,
            "live_output_gain": 0.5,
            "observation_routing": {
                "schema_version": ROUTING_SCHEMA_VERSION,
                "registry_version": ROUTING_REGISTRY_VERSION,
                "bindings": [
                    {
                        "semantic_id": "self_energy",
                        "target_slot": "in",
                        "gain": 0.5,
                        "enabled": True,
                    }
                ],
            },
        },
    }


def _agent() -> SimpleNamespace:
    return SimpleNamespace(
        genome=SimpleNamespace(modules=[deepcopy(_module())], connections=[]),
        structural_live_module_state={},
        structural_live_module_outputs={},
        structural_live_module_telemetry={},
    )


def build_stage0_audit() -> dict[str, Any]:
    channels: list[dict[str, Any]] = []
    for channel in perception_channel_registry():
        row = channel.to_dict()
        row["proposed_routing_status"] = "eligible" if bool(channel.routing_eligible) else "ineligible"
        channels.append(row)
    payload: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "scope": "existing finalized observation meanings admitted to ESCAP-3 scalar module inputs",
        "registry": perception_channel_registry_dict(),
        "routing_contract": {
            "schema_version": ROUTING_SCHEMA_VERSION,
            "registry_version": ROUTING_REGISTRY_VERSION,
            "profile": ROUTING_PROFILE,
            "exposure_mode": ROUTING_EXPOSURE_MODE,
            "eligible_semantic_ids": list(ADMITTED_SEMANTIC_IDS),
            "maximum_bindings_per_module": MAX_BINDINGS_PER_MODULE,
            "maximum_bindings_per_agent": MAX_BINDINGS_PER_AGENT,
            "resolution_boundary": "completed already-masked policy observation vector",
            "direct_observation_shape_change": False,
            "execution_rng_draws": 0,
        },
        "channels": channels,
        "lane_support": {
            "cpu_reference_single_tick_mlp": "implemented",
            "cpu_reference_single_tick_gru": "implemented",
            "policy_stride": "fail closed",
            "native_cxx": "fail closed",
            "cuda": "fail closed",
        },
        "prohibited": [
            "new sensor meanings",
            "spatial vector reduction or positional binding",
            "future hidden global metric debug or lineage channels",
            "observation shape changes",
            "reward fitness selection ecology or action vocabulary changes",
            "CUDA native or performance integration",
        ],
        "interpretation_boundary": (
            "technical causal routing evidence only; allocation or activity does not establish perception, "
            "adaptiveness, cognition, intelligence, ecology, or selection value"
        ),
    }
    payload["audit_hash"] = _stable_hash(payload)
    return payload


def _zero_gain_intervention(batch: RoutingBatch) -> RoutingBatch:
    telemetry = deepcopy(batch.telemetry_by_instance_id)
    for module_row in telemetry.values():
        for binding in module_row.get("bindings", []):
            binding["hereditary_gain"] = binding.get("gain")
            binding["effective_gain"] = 0.0
            binding["contribution"] = 0.0
        module_row["aggregate_before_clip"] = 0.0
        module_row["routed_observation_aggregate"] = 0.0
        module_row["zero_gain_intervention"] = True
    summary = deepcopy(batch.summary)
    summary.update({"active_bindings": 0, "zero_gain_intervention": True})
    return RoutingBatch(
        aggregate_by_instance_id={key: 0.0 for key in batch.aggregate_by_instance_id},
        telemetry_by_instance_id=telemetry,
        summary=summary,
    )


def _routing_batch(name: str, obs: np.ndarray, modules: list[dict[str, Any]]) -> RoutingBatch | None:
    if name == "routing_disabled":
        return None
    source_lesions = ("self_energy",) if name == "source_channel_lesion" else ()
    resolved = resolve_routing_values(obs, LOCAL_SIZE, ("self_energy",), lesion_semantic_ids=source_lesions)
    binding_lesions = (
        ((MODULE_INSTANCE_ID, "self_energy", "in"),) if name == "binding_lesion" else ()
    )
    batch = compile_module_routing(modules, resolved, lesion_binding_keys=binding_lesions)
    return _zero_gain_intervention(batch) if name == "enabled_zero_gain" else batch


def _run_arm(name: str) -> dict[str, Any]:
    agent = _agent()
    starting_genome = {
        "modules": deepcopy(agent.genome.modules),
        "connections": deepcopy(agent.genome.connections),
    }
    observations = [
        _observation(energy, health)
        for energy, health in zip(ENERGY_SEQUENCE, HEALTH_SEQUENCE, strict=True)
    ]
    trajectory: list[dict[str, Any]] = []
    for tick, (obs, fatigue) in enumerate(zip(observations, FATIGUE_SEQUENCE, strict=True)):
        before_digest = hashlib.sha256(obs.tobytes()).hexdigest()
        batch = _routing_batch(name, obs, list(agent.genome.modules))
        kwargs: dict[str, Any] = {}
        if batch is not None:
            kwargs.update(
                routed_observation_by_instance_id=batch.aggregate_by_instance_id,
                routing_telemetry_by_instance_id=batch.telemetry_by_instance_id,
                routing_summary=batch.summary,
            )
        if name == "state_reset_tick2" and tick == 2:
            kwargs["reset_instance_ids"] = [MODULE_INSTANCE_ID]
        result = execute_agent_live_modules(agent, fatigue=fatigue, **kwargs)
        logits = np.zeros((1, 7), dtype=np.float32)
        add_rest_logit_residual_(
            logits,
            np.asarray([result.rest_residual], dtype=np.float32),
            rest_index=5,
        )
        trajectory.append(
            {
                "tick": tick,
                "observation": {
                    "self_energy": float(obs[LOCAL_SIZE]),
                    "self_health": float(obs[LOCAL_SIZE + 1]),
                    "direct_observation_digest": before_digest,
                    "direct_observation_unchanged": before_digest
                    == hashlib.sha256(obs.tobytes()).hexdigest(),
                },
                "fatigue": float(fatigue),
                "routing": dict(result.summary.get("observation_routing", {})),
                "state_rows": deepcopy(result.summary.get("state_rows", [])),
                "state_after": deepcopy(agent.structural_live_module_state),
                "previous_outputs_after": deepcopy(agent.structural_live_module_outputs),
                "rest_logit_residual": float(result.rest_residual),
                "action_logits": logits[0].astype(float).tolist(),
                "selected_action": int(np.argmax(logits[0])),
                "executor_random_draws": int(result.summary.get("random_draws", -1)),
            }
        )
    return {
        "arm": name,
        "starting_genome_hash": _stable_hash(starting_genome),
        "module_instance_ids": [MODULE_INSTANCE_ID],
        "direct_observation_sequence_hash": _stable_hash(
            [row["observation"]["direct_observation_digest"] for row in trajectory]
        ),
        "fatigue_sequence": list(FATIGUE_SEQUENCE),
        "mutation_operator_draws": 0,
        "c2_donor_draws": 0,
        "trajectory": trajectory,
    }


def _checkpoint_continuation() -> dict[str, Any]:
    observations = [
        _observation(energy, health)
        for energy, health in zip(ENERGY_SEQUENCE, HEALTH_SEQUENCE, strict=True)
    ]
    uninterrupted = _agent()
    for tick in (0, 1):
        batch = _routing_batch("active", observations[tick], uninterrupted.genome.modules)
        assert batch is not None
        execute_agent_live_modules(
            uninterrupted,
            fatigue=FATIGUE_SEQUENCE[tick],
            routed_observation_by_instance_id=batch.aggregate_by_instance_id,
            routing_telemetry_by_instance_id=batch.telemetry_by_instance_id,
            routing_summary=batch.summary,
        )
    checkpoint_payload = {
        "schema_version": "escap3_live_module_runtime_v1",
        "routing_schema_version": ROUTING_SCHEMA_VERSION,
        "routing_registry_version": ROUTING_REGISTRY_VERSION,
        "state_by_instance_id": deepcopy(uninterrupted.structural_live_module_state),
        "previous_outputs_by_instance_id": deepcopy(uninterrupted.structural_live_module_outputs),
        "telemetry": deepcopy(uninterrupted.structural_live_module_telemetry),
        "migration_random_draws": 0,
    }
    restored = _agent()
    restored.structural_live_module_state = deepcopy(checkpoint_payload["state_by_instance_id"])
    restored.structural_live_module_outputs = deepcopy(checkpoint_payload["previous_outputs_by_instance_id"])
    restored.structural_live_module_telemetry = deepcopy(checkpoint_payload["telemetry"])
    batch_a = _routing_batch("active", observations[2], uninterrupted.genome.modules)
    batch_b = _routing_batch("active", observations[2], restored.genome.modules)
    assert batch_a is not None and batch_b is not None
    next_a = execute_agent_live_modules(
        uninterrupted,
        fatigue=FATIGUE_SEQUENCE[2],
        routed_observation_by_instance_id=batch_a.aggregate_by_instance_id,
        routing_telemetry_by_instance_id=batch_a.telemetry_by_instance_id,
        routing_summary=batch_a.summary,
    )
    next_b = execute_agent_live_modules(
        restored,
        fatigue=FATIGUE_SEQUENCE[2],
        routed_observation_by_instance_id=batch_b.aggregate_by_instance_id,
        routing_telemetry_by_instance_id=batch_b.telemetry_by_instance_id,
        routing_summary=batch_b.summary,
    )
    return {
        "payload": checkpoint_payload,
        "state_equal": uninterrupted.structural_live_module_state == restored.structural_live_module_state,
        "previous_outputs_equal": uninterrupted.structural_live_module_outputs
        == restored.structural_live_module_outputs,
        "first_post_resume_residual_equal": next_a.rest_residual == next_b.rest_residual,
        "first_post_resume_summary_hash_equal": _stable_hash(next_a.summary) == _stable_hash(next_b.summary),
        "direct_observation_digest": hashlib.sha256(observations[2].tobytes()).hexdigest(),
        "migration_random_draws": 0,
    }


def run_technical_probe() -> dict[str, Any]:
    arm_names = (
        "routing_disabled",
        "enabled_zero_gain",
        "active",
        "source_channel_lesion",
        "binding_lesion",
        "state_reset_tick2",
    )
    arms = [_run_arm(name) for name in arm_names]
    by_name = {arm["arm"]: arm for arm in arms}
    checkpoint = _checkpoint_continuation()
    source_binding = by_name["source_channel_lesion"]["trajectory"][0]["state_rows"][0][
        "observation_routing"
    ]["bindings"][0]
    binding_binding = by_name["binding_lesion"]["trajectory"][0]["state_rows"][0][
        "observation_routing"
    ]["bindings"][0]
    payload: dict[str, Any] = {
        "schema_version": PROBE_SCHEMA_VERSION,
        "profile": ROUTING_PROFILE,
        "exposure_mode": ROUTING_EXPOSURE_MODE,
        "observation_semantic_sequence": {
            "self_energy": list(ENERGY_SEQUENCE),
            "self_health": list(HEALTH_SEQUENCE),
        },
        "fatigue_sequence": list(FATIGUE_SEQUENCE),
        "arms": arms,
        "checkpoint_continuation": checkpoint,
        "interpretation_boundary": "bounded technical causal routing fixture only; no ecology campaign or broad claim",
    }
    payload["checks"] = {
        "six_arm_starting_genome_matched": len({arm["starting_genome_hash"] for arm in arms}) == 1,
        "six_arm_identity_matched": len({tuple(arm["module_instance_ids"]) for arm in arms}) == 1,
        "six_arm_direct_observation_sequence_matched": len(
            {arm["direct_observation_sequence_hash"] for arm in arms}
        )
        == 1,
        "six_arm_fatigue_matched": all(arm["fatigue_sequence"] == list(FATIGUE_SEQUENCE) for arm in arms),
        "disabled_neutral": all(
            row["rest_logit_residual"] == 0.0 for row in by_name["routing_disabled"]["trajectory"]
        ),
        "zero_gain_neutral": all(
            row["rest_logit_residual"] == 0.0 for row in by_name["enabled_zero_gain"]["trajectory"]
        ),
        "active_nonzero": any(
            row["rest_logit_residual"] != 0.0 for row in by_name["active"]["trajectory"]
        ),
        "source_lesion_is_distinct": bool(source_binding["source_lesioned"])
        and not bool(source_binding["binding_lesioned"]),
        "binding_lesion_is_distinct": not bool(binding_binding["source_lesioned"])
        and bool(binding_binding["binding_lesioned"]),
        "state_reset_changes_tick2": by_name["state_reset_tick2"]["trajectory"][2]["state_after"]
        != by_name["active"]["trajectory"][2]["state_after"],
        "executor_random_draws_zero": all(
            row["executor_random_draws"] == 0 for arm in arms for row in arm["trajectory"]
        ),
        "checkpoint_continuation_parity": all(
            bool(checkpoint[key])
            for key in (
                "state_equal",
                "previous_outputs_equal",
                "first_post_resume_residual_equal",
                "first_post_resume_summary_hash_equal",
            )
        ),
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
    return 0 if all(bool(value) for value in probe["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
