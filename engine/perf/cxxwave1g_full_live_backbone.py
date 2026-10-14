from __future__ import annotations

"""CXXWAVE1G full live C++ backbone state and support contract.

This module is intentionally conservative. It defines the single resident state
and capability surface that a future C++/CUDA generation call must own before a
live envelope may open. The executor fails closed unless every required group is
native-authoritative for the requested cell.
"""

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

import numpy as np


SCHEMA_VERSION = "cxxwave1g_full_live_cpp_backbone_v1"
RESIDENT_STATE_SCHEMA_VERSION = "resident_runtime_state_v3"
CAPABILITY_SCHEMA_VERSION = "cpp_backbone_capability_set_v1"
EXECUTION_RESULT_SCHEMA_VERSION = "cpp_backbone_generation_result_v1"


STATE_GROUPS = (
    "execution_cell_metadata",
    "population_slots_stable_ids",
    "mlp_gru_policy_banks",
    "policy_row_mappings",
    "hmem",
    "action_cache",
    "observations_staging",
    "drives",
    "chemistry_maintenance",
    "patch_world_state",
    "jungle_state",
    "wdl_state",
    "lifecycle_reproduction_mutation",
    "lineage_founder",
    "typed_event_tape",
    "checkpoint_cursors",
    "rng_state_indices",
    "capability_support_mask",
    "error_fallback_status",
)

REQUIRED_LIVE_NATIVE_GROUPS = frozenset(STATE_GROUPS) - frozenset({"error_fallback_status"})

MAJOR_BLOCKS = {
    "policy_hmem": {
        "required": ("mlp_gru_policy_banks", "policy_row_mappings", "hmem", "action_cache", "observations_staging", "rng_state_indices"),
        "pass": "CXXWAVE1G-POLICY-A/LIVE-POLICY-NATIVE",
        "blocked": "CXXWAVE1G-POLICY-C/PARITY-BLOCKED",
    },
    "patch_wdl_environment": {
        "required": ("patch_world_state", "jungle_state", "wdl_state", "observations_staging", "rng_state_indices"),
        "pass": "CXXWAVE1G-ENV-A/PATCH-WDL-LIVE-NATIVE",
        "blocked": "CXXWAVE1G-ENV-D/PARITY-BLOCKED",
    },
    "lifecycle_reproduction": {
        "required": ("lifecycle_reproduction_mutation", "population_slots_stable_ids", "policy_row_mappings", "hmem", "rng_state_indices"),
        "pass": "CXXWAVE1G-LIFE-A/FULL-LIVE-LIFECYCLE-NATIVE",
        "blocked": "CXXWAVE1G-LIFE-D/PARITY-BLOCKED",
    },
    "typed_events_public_outputs": {
        "required": ("typed_event_tape", "lineage_founder", "capability_support_mask"),
        "pass": "CXXWAVE1G-EVENT-A/FULL-LIVE-EVENT-NATIVE",
        "blocked": "CXXWAVE1G-EVENT-C/PUBLIC-COMPATIBILITY-BLOCKED",
    },
    "resident_checkpoint": {
        "required": ("checkpoint_cursors", "rng_state_indices", "policy_row_mappings", "hmem", "action_cache", "typed_event_tape"),
        "pass": "CXXWAVE1G-CKPT-A/FULL-RESIDENT-CHECKPOINT-NATIVE",
        "blocked": "CXXWAVE1G-CKPT-D/PARITY-BLOCKED",
    },
    "multi_cell_runtime": {
        "required": (
            "execution_cell_metadata",
            "capability_support_mask",
            "rng_state_indices",
            "typed_event_tape",
            "checkpoint_cursors",
        ),
        "pass": "CXXWAVE1G-BATCH-A/FOUR-EIGHT-TWELVE-LIVE",
        "blocked": "CXXWAVE1G-BATCH-D/PARITY-OR-ISOLATION-BLOCKED",
    },
}


@dataclass(frozen=True)
class FieldAuthority:
    group: str
    authority: str
    dtype: str
    device: str
    shape: tuple[int, ...] | tuple[()]
    ordering: str
    mutation_owner: str
    checkpoint_member: bool
    deterministic_rebuild: str
    parity_comparison: str
    supported_capability: str


@dataclass(frozen=True)
class CppBackboneCapabilitySetV1:
    policy_family: str
    hmem_width: int
    action_mode: str
    action_stride: int
    patch_features: tuple[str, ...]
    wdl_features: tuple[str, ...]
    lifecycle_churn: str
    mutation_type: str
    chemistry_maintenance: str
    event_report_mode: str
    checkpoint_schema: str
    output_mode: str
    multi_cell_compatibility: str
    native_authority: Mapping[str, bool]


@dataclass(frozen=True)
class ResidentRuntimeStateV3:
    generation: int
    tick: int
    cell_id: str
    field_authority: tuple[FieldAuthority, ...]
    arrays: Mapping[str, np.ndarray]
    scalars: Mapping[str, int | float | str | bool]
    capability: CppBackboneCapabilitySetV1
    support_blockers: tuple[str, ...]
    state_digest: str


def cpp_backbone_capability_set_v1(
    *,
    policy_family: str = "unknown",
    hmem_width: int = 0,
    action_mode: str = "deterministic",
    action_stride: int = 1,
    patch_features: tuple[str, ...] = (),
    wdl_features: tuple[str, ...] = (),
    lifecycle_churn: str = "python_authoritative",
    mutation_type: str = "python_authoritative",
    chemistry_maintenance: str = "scalar_native_supported",
    event_report_mode: str = "scalar_tape_native_public_python",
    checkpoint_schema: str = "scalar_checkpoint_native_full_bundle_python",
    output_mode: str = "compatibility_python",
    multi_cell_compatibility: str = "not_supported",
    native_authority: Mapping[str, bool] | None = None,
) -> CppBackboneCapabilitySetV1:
    authority = {group: False for group in STATE_GROUPS}
    authority.update(
        {
            "drives": chemistry_maintenance == "scalar_native_supported",
            "chemistry_maintenance": chemistry_maintenance == "scalar_native_supported",
            "typed_event_tape": event_report_mode.startswith("scalar_tape_native"),
            "checkpoint_cursors": checkpoint_schema.startswith("scalar_checkpoint_native"),
            "error_fallback_status": True,
        }
    )
    if native_authority:
        authority.update({str(key): bool(value) for key, value in native_authority.items()})
    return CppBackboneCapabilitySetV1(
        policy_family=str(policy_family),
        hmem_width=int(hmem_width),
        action_mode=str(action_mode),
        action_stride=int(action_stride),
        patch_features=tuple(str(x) for x in patch_features),
        wdl_features=tuple(str(x) for x in wdl_features),
        lifecycle_churn=str(lifecycle_churn),
        mutation_type=str(mutation_type),
        chemistry_maintenance=str(chemistry_maintenance),
        event_report_mode=str(event_report_mode),
        checkpoint_schema=str(checkpoint_schema),
        output_mode=str(output_mode),
        multi_cell_compatibility=str(multi_cell_compatibility),
        native_authority=authority,
    )


def capability_payload(capability: CppBackboneCapabilitySetV1) -> dict[str, Any]:
    authority = {str(key): bool(value) for key, value in capability.native_authority.items()}
    blockers = _support_blockers(capability)
    return {
        "schema_version": CAPABILITY_SCHEMA_VERSION,
        "policy_family": capability.policy_family,
        "hmem_width": int(capability.hmem_width),
        "action_mode": capability.action_mode,
        "action_stride": int(capability.action_stride),
        "patch_features": list(capability.patch_features),
        "wdl_features": list(capability.wdl_features),
        "lifecycle_churn": capability.lifecycle_churn,
        "mutation_type": capability.mutation_type,
        "chemistry_maintenance": capability.chemistry_maintenance,
        "event_report_mode": capability.event_report_mode,
        "checkpoint_schema": capability.checkpoint_schema,
        "output_mode": capability.output_mode,
        "multi_cell_compatibility": capability.multi_cell_compatibility,
        "native_authority": authority,
        "major_block_decisions": major_block_decisions(capability),
        "supported": not blockers,
        "blockers": blockers,
        "missing_native_authority_groups": sorted(group for group in REQUIRED_LIVE_NATIVE_GROUPS if not authority.get(group, False)),
    }


def capture_resident_runtime_state_v3(
    runner: Any,
    *,
    generation: int,
    tick: int = 0,
    cell_id: str = "cell0",
    capability: CppBackboneCapabilitySetV1 | None = None,
) -> ResidentRuntimeStateV3:
    arrays = _runner_arrays(runner)
    if capability is None:
        capability = infer_capability_from_runner(runner)
    field_authority = tuple(_field_authorities(arrays=arrays, capability=capability))
    scalars = {
        "generation": int(generation),
        "tick": int(tick),
        "population_capacity": int(np.asarray(arrays.get("agent_ids", np.zeros((0,), dtype=np.int64))).shape[0]),
        "steps_per_generation": int(getattr(getattr(runner, "sim_cfg", None), "steps_per_generation", 0) or 0),
        "rng_digest": _digest_json(_rng_payload(runner)),
    }
    payload_digest = _digest_json(
        {
            "schema_version": RESIDENT_STATE_SCHEMA_VERSION,
            "generation": int(generation),
            "tick": int(tick),
            "cell_id": str(cell_id),
            "arrays": {name: _array_digest(value) for name, value in sorted(arrays.items())},
            "scalars": scalars,
            "capability": capability_payload(capability),
        }
    )
    return ResidentRuntimeStateV3(
        generation=int(generation),
        tick=int(tick),
        cell_id=str(cell_id),
        field_authority=field_authority,
        arrays=arrays,
        scalars=scalars,
        capability=capability,
        support_blockers=tuple(_support_blockers(capability)),
        state_digest=payload_digest,
    )


def resident_runtime_state_payload(state: ResidentRuntimeStateV3) -> dict[str, Any]:
    return {
        "schema_version": RESIDENT_STATE_SCHEMA_VERSION,
        "generation": int(state.generation),
        "tick": int(state.tick),
        "cell_id": state.cell_id,
        "field_authority": [authority.__dict__ for authority in state.field_authority],
        "arrays": {name: _array_payload(value) for name, value in sorted(state.arrays.items())},
        "scalars": dict(state.scalars),
        "capability": capability_payload(state.capability),
        "support_blockers": list(state.support_blockers),
        "state_digest": state.state_digest,
    }


def infer_capability_from_runner(runner: Any) -> CppBackboneCapabilitySetV1:
    cfg = getattr(runner, "sim_cfg", None)
    env_cfg = getattr(runner, "env_cfg", None)
    hmem = getattr(runner, "_Hmem", None)
    hmem_width = int(getattr(hmem, "shape", (0, 0))[1]) if hmem is not None and len(getattr(hmem, "shape", ())) >= 2 else 0
    families = _policy_families(runner)
    patch_features: list[str] = []
    if bool(getattr(env_cfg, "patch_world_enable", False)):
        patch_features.append("patch_world")
    for name in ("jungle_enable", "patch_world_observation_enable", "patch_world_observation_stacked_gather_enable"):
        if bool(getattr(env_cfg, name, False)):
            patch_features.append(name)
    wdl_features: list[str] = []
    for name in (
        "wdl_delayed_hazard_enable",
        "prediction_demand_enable",
        "delayed_resource_depletion_enable",
        "delayed_crowding_waste_enable",
    ):
        if bool(getattr(env_cfg, name, False)):
            wdl_features.append(name)
    return cpp_backbone_capability_set_v1(
        policy_family="+".join(families) if families else "unknown",
        hmem_width=hmem_width,
        action_mode="deterministic" if bool(getattr(cfg, "policy_deterministic_actions", False)) else "stochastic",
        action_stride=int(getattr(cfg, "policy_tick_stride", 1) or 1),
        patch_features=tuple(patch_features),
        wdl_features=tuple(wdl_features),
        lifecycle_churn="python_authoritative",
        mutation_type="python_authoritative",
        chemistry_maintenance=(
            "scalar_native_supported"
            if bool(getattr(cfg, "cxxwave1f_scalar_backbone_enable", False))
            else "python_authoritative"
        ),
        event_report_mode="scalar_tape_native_public_python",
        checkpoint_schema="scalar_checkpoint_native_full_bundle_python",
        output_mode="compatibility_python",
        multi_cell_compatibility="not_supported",
    )


def execute_generation(
    state: ResidentRuntimeStateV3,
    generation_config: Mapping[str, Any],
    *,
    generation_index: int,
    steps_per_generation: int,
) -> dict[str, Any]:
    blockers = list(state.support_blockers)
    if blockers:
        return {
            "schema_version": EXECUTION_RESULT_SCHEMA_VERSION,
            "executed": False,
            "decision": "CXXWAVE1G-F/LIVE-PARITY-OR-CHECKPOINT-BLOCKED",
            "support_status": "fail_closed_before_execution",
            "generation_index": int(generation_index),
            "steps_per_generation": int(steps_per_generation),
            "state_digest_before": state.state_digest,
            "state_digest_after": state.state_digest,
            "hidden_fallback": False,
            "blockers": blockers,
            "major_block_decisions": major_block_decisions(state.capability),
            "generation_config": dict(generation_config),
        }
    return {
        "schema_version": EXECUTION_RESULT_SCHEMA_VERSION,
        "executed": False,
        "decision": "CXXWAVE1G-H/PARTIAL-AUTONOMOUS-CLOSEOUT",
        "support_status": "native_backend_not_linked",
        "generation_index": int(generation_index),
        "steps_per_generation": int(steps_per_generation),
        "state_digest_before": state.state_digest,
        "state_digest_after": state.state_digest,
        "hidden_fallback": False,
        "blockers": ["cpp_backbone_execute_generation_backend_missing"],
        "major_block_decisions": major_block_decisions(state.capability),
        "generation_config": dict(generation_config),
    }


def first_divergence_proof(result: Mapping[str, Any]) -> dict[str, Any]:
    blockers = [str(x) for x in result.get("blockers", [])]
    return {
        "schema_version": "cxxwave1g_first_divergence_or_support_blocker_v1",
        "kind": "pre_execution_support_blocker" if blockers else "not_blocked",
        "executed": bool(result.get("executed", False)),
        "hidden_fallback": bool(result.get("hidden_fallback", False)),
        "blockers": blockers,
        "first_blocker": blockers[0] if blockers else "",
        "science_state_mutated": False,
    }


def major_block_decisions(capability: CppBackboneCapabilitySetV1) -> dict[str, dict[str, Any]]:
    authority = capability.native_authority
    decisions: dict[str, dict[str, Any]] = {}
    for name, spec in MAJOR_BLOCKS.items():
        missing = [group for group in spec["required"] if not bool(authority.get(group, False))]
        decision = spec["pass"] if not missing else spec["blocked"]
        blockers = [f"missing_native_authority:{group}" for group in missing]
        if name == "typed_events_public_outputs" and capability.output_mode != "native_public_compatible":
            blockers.append("public_output_compatibility_python_owned")
            decision = spec["blocked"]
        if name == "resident_checkpoint" and not capability.checkpoint_schema.startswith("full_resident_checkpoint_native"):
            blockers.append("full_resident_checkpoint_not_native")
            decision = spec["blocked"]
        if name == "lifecycle_reproduction":
            if capability.lifecycle_churn != "native_authoritative":
                blockers.append("lifecycle_reproduction_mutation_python_owned")
                decision = spec["blocked"]
            if capability.mutation_type != "native_authoritative":
                blockers.append("mutation_rng_draw_schedule_python_owned")
                decision = spec["blocked"]
        if name == "multi_cell_runtime" and capability.multi_cell_compatibility == "not_supported":
            blockers.append("multi_cell_batching_not_supported")
            decision = spec["blocked"]
        decisions[name] = {
            "decision": decision,
            "supported": not blockers,
            "required_groups": list(spec["required"]),
            "blockers": blockers,
        }
    return decisions


def _support_blockers(capability: CppBackboneCapabilitySetV1) -> list[str]:
    authority = capability.native_authority
    blockers = [f"missing_native_authority:{group}" for group in sorted(REQUIRED_LIVE_NATIVE_GROUPS) if not authority.get(group, False)]
    if capability.output_mode != "native_public_compatible":
        blockers.append("public_output_compatibility_python_owned")
    if not capability.checkpoint_schema.startswith("full_resident_checkpoint_native"):
        blockers.append("full_resident_checkpoint_not_native")
    if capability.lifecycle_churn != "native_authoritative":
        blockers.append("lifecycle_reproduction_mutation_python_owned")
    if capability.mutation_type != "native_authoritative":
        blockers.append("mutation_rng_draw_schedule_python_owned")
    if capability.multi_cell_compatibility == "not_supported":
        blockers.append("multi_cell_batching_not_supported")
    return blockers


def _runner_arrays(runner: Any) -> dict[str, np.ndarray]:
    agents = sorted(list(getattr(getattr(runner, "pop", None), "alive_agents", lambda: [])()), key=lambda a: int(getattr(a, "agent_id", -1)))
    agent_ids = np.asarray([int(getattr(agent, "agent_id", -1)) for agent in agents], dtype=np.int64)
    env = getattr(runner, "env", None)
    arrays: dict[str, np.ndarray] = {
        "agent_ids": agent_ids,
        "energy": np.asarray([float(getattr(agent, "energy", 0.0)) for agent in agents], dtype=np.float32),
        "health": np.asarray([float(getattr(agent, "health", 0.0)) for agent in agents], dtype=np.float32),
        "injury": np.asarray([float(getattr(agent, "injury", 0.0)) for agent in agents], dtype=np.float32),
        "age": np.asarray([int(getattr(agent, "age", 0)) for agent in agents], dtype=np.int32),
    }
    for name in ("agent_alive", "agent_pos_y", "agent_pos_x", "food", "patch_food_amount", "patch_cover", "patch_shelter"):
        value = getattr(env, name, None)
        if value is not None:
            arrays[name] = np.ascontiguousarray(value)
    hmem = getattr(runner, "_Hmem", None)
    if hmem is not None:
        try:
            arrays["hmem"] = np.ascontiguousarray(hmem.detach().cpu().numpy(), dtype=np.float32)
        except Exception:
            arrays["hmem"] = np.zeros((0, 0), dtype=np.float32)
    return arrays


def _field_authorities(*, arrays: Mapping[str, np.ndarray], capability: CppBackboneCapabilitySetV1) -> list[FieldAuthority]:
    rows: list[FieldAuthority] = []
    for group in STATE_GROUPS:
        owned = bool(capability.native_authority.get(group, False))
        representative = _representative_array_for_group(group, arrays)
        rows.append(
            FieldAuthority(
                group=group,
                authority="cpp_backbone" if owned else "python_live_runner",
                dtype=str(representative.dtype) if representative is not None else "object",
                device="cpu",
                shape=tuple(int(x) for x in representative.shape) if representative is not None else tuple(),
                ordering="stable_agent_id_order" if group in {"population_slots_stable_ids", "policy_row_mappings", "hmem"} else "schema_order",
                mutation_owner="cpp_backbone" if owned else "python_live_runner",
                checkpoint_member=group in REQUIRED_LIVE_NATIVE_GROUPS,
                deterministic_rebuild="native_state_replay" if owned else "python_object_rebuild_required",
                parity_comparison="exact_digest",
                supported_capability="supported" if owned else "blocked",
            )
        )
    return rows


def _representative_array_for_group(group: str, arrays: Mapping[str, np.ndarray]) -> np.ndarray | None:
    mapping = {
        "population_slots_stable_ids": "agent_ids",
        "hmem": "hmem",
        "drives": "energy",
        "chemistry_maintenance": "health",
        "patch_world_state": "patch_food_amount",
        "wdl_state": "patch_cover",
        "jungle_state": "patch_shelter",
    }
    name = mapping.get(group)
    if name:
        return arrays.get(name)
    return None


def _policy_families(runner: Any) -> tuple[str, ...]:
    families: set[str] = set()
    for agent in list(getattr(getattr(runner, "pop", None), "alive_agents", lambda: [])()):
        policy = getattr(agent, "policy", None)
        families.add(type(policy).__name__ if policy is not None else "unknown")
    return tuple(sorted(families))


def _rng_payload(runner: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for name in ("np_rng", "rng_manager"):
        value = getattr(runner, name, None)
        try:
            payload[name] = repr(value.bit_generator.state)  # numpy Generator
        except Exception:
            payload[name] = repr(value)
    return payload


def _array_payload(value: np.ndarray) -> dict[str, Any]:
    arr = np.ascontiguousarray(value)
    return {"dtype": str(arr.dtype), "shape": list(arr.shape), "sha256": _array_digest(arr)}


def _array_digest(value: Any) -> str:
    arr = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(json.dumps(list(arr.shape), separators=(",", ":")).encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()


def _digest_json(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


__all__ = [
    "CAPABILITY_SCHEMA_VERSION",
    "EXECUTION_RESULT_SCHEMA_VERSION",
    "RESIDENT_STATE_SCHEMA_VERSION",
    "CppBackboneCapabilitySetV1",
    "FieldAuthority",
    "ResidentRuntimeStateV3",
    "capability_payload",
    "capture_resident_runtime_state_v3",
    "cpp_backbone_capability_set_v1",
    "execute_generation",
    "first_divergence_proof",
    "infer_capability_from_runner",
    "major_block_decisions",
    "resident_runtime_state_payload",
]
