from __future__ import annotations

"""ESCAP-4 bounded routing of existing scalar observation meanings.

The existing perception registry is the only semantic authority.  Bindings
carry semantic IDs rather than policy-vector offsets; resolution happens from
the completed, already-masked observation frame immediately before ESCAP-3.
"""

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

import numpy as np

from agents.perception import (
    PERCEPTION_ROUTING_CONTRACT_VERSION,
    get_perception_channel,
    perception_channel_registry,
)
from evolution.structural_live_modules_v1 import LIVE_PROFILE, live_modules_enabled


ROUTING_SCHEMA_VERSION = "escap4_observation_routing_v1"
ROUTING_PROFILE = "escap4_scalar_self_state_parallel_v1"
ROUTING_REGISTRY_VERSION = PERCEPTION_ROUTING_CONTRACT_VERSION
ROUTING_EXPOSURE_MODE = "parallel"
ROUTING_BLOCK_KEY = "observation_routing"
MAX_BINDINGS_PER_MODULE = 2
MAX_BINDINGS_PER_AGENT = 8
MAX_BINDING_GAIN = 1.0
ROUTED_INPUT_BOUND = 1.0
ADMITTED_SEMANTIC_IDS = tuple(
    sorted(
        str(channel.channel_id)
        for channel in perception_channel_registry()
        if bool(getattr(channel, "routing_eligible", False))
    )
)


def routing_enabled(sim_cfg: Any) -> bool:
    return bool(getattr(sim_cfg, "structural_observation_routing_enable", False))


def validate_routing_config(sim_cfg: Any, *, device: Any = "cpu", policy_stride: int = 1) -> None:
    if not routing_enabled(sim_cfg):
        return
    if not live_modules_enabled(sim_cfg):
        raise RuntimeError("ESCAP4_REQUIRES_ESCAP3_LIVE_MODULES")
    live_profile = str(getattr(sim_cfg, "structural_live_modules_profile", "none") or "none")
    if live_profile != LIVE_PROFILE:
        raise RuntimeError(f"ESCAP4_ESCAP3_PROFILE_MISMATCH profile={live_profile}")
    expected = (
        ROUTING_PROFILE,
        ROUTING_SCHEMA_VERSION,
        ROUTING_REGISTRY_VERSION,
        MAX_BINDINGS_PER_MODULE,
        MAX_BINDINGS_PER_AGENT,
        "escap4_routing_binding_operator_v1",
    )
    actual = (
        str(getattr(sim_cfg, "structural_observation_routing_profile", "none") or "none"),
        str(getattr(sim_cfg, "structural_observation_routing_schema_version", ROUTING_SCHEMA_VERSION) or ROUTING_SCHEMA_VERSION),
        str(getattr(sim_cfg, "structural_observation_routing_registry_version", ROUTING_REGISTRY_VERSION) or ROUTING_REGISTRY_VERSION),
        int(getattr(sim_cfg, "structural_observation_routing_max_bindings_per_module", MAX_BINDINGS_PER_MODULE) or MAX_BINDINGS_PER_MODULE),
        int(getattr(sim_cfg, "structural_observation_routing_max_bindings_per_agent", MAX_BINDINGS_PER_AGENT) or MAX_BINDINGS_PER_AGENT),
        str(getattr(sim_cfg, "structural_observation_routing_operator_profile", "escap4_routing_binding_operator_v1") or "escap4_routing_binding_operator_v1"),
    )
    if actual != expected:
        raise RuntimeError(f"ESCAP4_CONFIG_CONTRACT_MISMATCH expected={expected!r};actual={actual!r}")
    attempts = int(getattr(sim_cfg, "structural_observation_routing_operator_attempts", 1) or 0)
    if attempts != 1:
        raise RuntimeError(
            f"ESCAP4_OPERATOR_BUDGET_MISMATCH attempts={attempts};expected=1"
        )
    if str(device) != "cpu":
        raise RuntimeError(f"ESCAP4_CPU_REFERENCE_ONLY device={device}")
    if int(policy_stride) != 1:
        raise RuntimeError(f"ESCAP4_POLICY_STRIDE_UNSUPPORTED stride={int(policy_stride)}")
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
        raise RuntimeError(f"ESCAP4_NATIVE_LANE_UNSUPPORTED flags={','.join(active)}")


def _finite(value: Any, *, field: str) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise RuntimeError(f"ESCAP4_INVALID_NUMERIC field={field}") from exc
    if not math.isfinite(out):
        raise RuntimeError(f"ESCAP4_NONFINITE field={field}")
    return float(out)


def _canonical_gain(value: Any) -> float:
    gain = _finite(value, field="binding.gain")
    if not -MAX_BINDING_GAIN <= gain <= MAX_BINDING_GAIN:
        raise RuntimeError(f"ESCAP4_BINDING_GAIN_OUT_OF_RANGE value={gain}")
    return float(np.float32(gain))


def binding_key(binding: Mapping[str, Any]) -> tuple[str, str]:
    return (str(binding.get("semantic_id", "") or ""), str(binding.get("target_slot", "") or ""))


def normalize_routing_block(block: Any) -> dict[str, Any]:
    if block is None:
        block = {}
    if not isinstance(block, Mapping):
        raise RuntimeError("ESCAP4_BINDING_BLOCK_MALFORMED")
    schema = str(block.get("schema_version", ROUTING_SCHEMA_VERSION) or ROUTING_SCHEMA_VERSION)
    registry = str(block.get("registry_version", ROUTING_REGISTRY_VERSION) or ROUTING_REGISTRY_VERSION)
    if schema != ROUTING_SCHEMA_VERSION:
        raise RuntimeError(f"ESCAP4_BINDING_SCHEMA_UNSUPPORTED schema={schema}")
    if registry != ROUTING_REGISTRY_VERSION:
        raise RuntimeError(f"ESCAP4_REGISTRY_VERSION_MISMATCH registry={registry}")
    raw_bindings = block.get("bindings", [])
    if not isinstance(raw_bindings, list):
        raise RuntimeError("ESCAP4_BINDINGS_MALFORMED")
    if len(raw_bindings) > MAX_BINDINGS_PER_MODULE:
        raise RuntimeError(
            f"ESCAP4_MODULE_BINDING_CAP_EXCEEDED observed={len(raw_bindings)};max={MAX_BINDINGS_PER_MODULE}"
        )
    canonical: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, raw in enumerate(raw_bindings):
        if not isinstance(raw, Mapping):
            raise RuntimeError(f"ESCAP4_BINDING_MALFORMED index={index}")
        semantic_id = str(raw.get("semantic_id", "") or "")
        target_slot = str(raw.get("target_slot", "") or "")
        if semantic_id not in ADMITTED_SEMANTIC_IDS:
            try:
                channel = get_perception_channel(semantic_id)
            except Exception as exc:
                raise RuntimeError(f"ESCAP4_UNKNOWN_SEMANTIC_ID semantic_id={semantic_id}") from exc
            reason = str(getattr(channel, "routing_ineligible_reason", "not_admitted") or "not_admitted")
            raise RuntimeError(f"ESCAP4_SEMANTIC_INELIGIBLE semantic_id={semantic_id};reason={reason}")
        channel = get_perception_channel(semantic_id)
        if str(getattr(channel, "value_type", "")) != "scalar" or str(getattr(channel, "value_shape", "")) != "scalar":
            raise RuntimeError(f"ESCAP4_SEMANTIC_TYPE_UNSUPPORTED semantic_id={semantic_id}")
        if target_slot != "in":
            raise RuntimeError(f"ESCAP4_TARGET_SLOT_UNSUPPORTED target_slot={target_slot}")
        key = (semantic_id, target_slot)
        if key in seen:
            raise RuntimeError(f"ESCAP4_DUPLICATE_BINDING semantic_id={semantic_id};target_slot={target_slot}")
        seen.add(key)
        enabled = raw.get("enabled", True)
        if not isinstance(enabled, bool):
            raise RuntimeError(f"ESCAP4_BINDING_ENABLED_NOT_BOOL semantic_id={semantic_id}")
        canonical.append(
            {
                "semantic_id": semantic_id,
                "target_slot": target_slot,
                "gain": _canonical_gain(raw.get("gain", 0.0)),
                "enabled": bool(enabled),
            }
        )
    canonical.sort(key=binding_key)
    return {
        "schema_version": ROUTING_SCHEMA_VERSION,
        "registry_version": ROUTING_REGISTRY_VERSION,
        "bindings": canonical,
    }


def normalize_agent_routing_modules(modules: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    binding_count = 0
    for raw_module in modules:
        module = dict(raw_module)
        params = dict(module.get("params", {}) or {})
        if ROUTING_BLOCK_KEY in params:
            block = normalize_routing_block(params.get(ROUTING_BLOCK_KEY))
            if block["bindings"] and str(module.get("module_type", "")) != "generic":
                raise RuntimeError(
                    f"ESCAP4_BINDING_MODULE_TYPE_UNSUPPORTED module_type={module.get('module_type', '')}"
                )
            params[ROUTING_BLOCK_KEY] = block
            binding_count += len(block["bindings"])
        module["params"] = params
        out.append(module)
    if binding_count > MAX_BINDINGS_PER_AGENT:
        raise RuntimeError(
            f"ESCAP4_AGENT_BINDING_CAP_EXCEEDED observed={binding_count};max={MAX_BINDINGS_PER_AGENT}"
        )
    return out


def resolve_routing_values(
    obs_row: Any,
    local_size: int,
    semantic_ids: Iterable[str],
    *,
    lesion_semantic_ids: Iterable[str] = (),
) -> dict[str, dict[str, Any]]:
    row = np.asarray(obs_row, dtype=np.float32).reshape(-1)
    local = int(local_size)
    if local < 0 or row.size < local + 2:
        raise RuntimeError(f"ESCAP4_OBSERVATION_SHAPE_MISMATCH size={row.size};local_size={local}")
    positions = {"self_energy": local, "self_health": local + 1}
    lesions = {str(value) for value in lesion_semantic_ids}
    resolved: dict[str, dict[str, Any]] = {}
    for semantic_id in sorted({str(value) for value in semantic_ids}):
        if semantic_id not in ADMITTED_SEMANTIC_IDS:
            raise RuntimeError(f"ESCAP4_SEMANTIC_NOT_ADMITTED semantic_id={semantic_id}")
        channel = get_perception_channel(semantic_id)
        source_value = _finite(row[positions[semantic_id]], field=f"observation[{semantic_id}]")
        lo = float(getattr(channel, "value_min", 0.0))
        hi = float(getattr(channel, "value_max", 1.0))
        if not lo <= source_value <= hi:
            raise RuntimeError(
                f"ESCAP4_OBSERVATION_OUT_OF_RANGE semantic_id={semantic_id};value={source_value};bounds={lo},{hi}"
            )
        lesioned = semantic_id in lesions
        resolved[semantic_id] = {
            "semantic_id": semantic_id,
            "source_value": float(np.float32(source_value)),
            "resolved_value": 0.0 if lesioned else float(np.float32(source_value)),
            "source_lesioned": bool(lesioned),
            "direct_policy_exposure": True,
            "feature_gate": str(getattr(channel, "feature_gate", "always")),
        }
    return resolved


@dataclass(frozen=True)
class RoutingBatch:
    aggregate_by_instance_id: dict[str, float]
    telemetry_by_instance_id: dict[str, dict[str, Any]]
    summary: dict[str, Any]


def compile_module_routing(
    modules: Iterable[Mapping[str, Any]],
    resolved: Mapping[str, Mapping[str, Any]],
    *,
    lesion_binding_keys: Iterable[tuple[str, str, str]] = (),
) -> RoutingBatch:
    canonical_modules = normalize_agent_routing_modules(modules)
    local_ids = [str(module.get("module_id", "") or "") for module in canonical_modules]
    instance_ids = [str(module.get("module_instance_id", "") or "") for module in canonical_modules]
    if len(set(local_ids)) != len(local_ids):
        raise RuntimeError("ESCAP4_DUPLICATE_MODULE_ID")
    populated_instances = [value for value in instance_ids if value]
    if len(set(populated_instances)) != len(populated_instances):
        raise RuntimeError("ESCAP4_DUPLICATE_MODULE_INSTANCE_ID")
    lesions = {(str(a), str(b), str(c)) for a, b, c in lesion_binding_keys}
    aggregates: dict[str, float] = {}
    telemetry: dict[str, dict[str, Any]] = {}
    allocated = executable = active = unavailable = clipped = 0
    for module in sorted(canonical_modules, key=lambda value: str(value.get("module_id", ""))):
        if str(module.get("module_type", "")) != "generic":
            continue
        params = dict(module.get("params", {}) or {})
        if ROUTING_BLOCK_KEY not in params:
            continue
        block = normalize_routing_block(params[ROUTING_BLOCK_KEY])
        instance_id = str(module.get("module_instance_id", "") or "")
        if not instance_id:
            raise RuntimeError(f"ESCAP4_MISSING_MODULE_INSTANCE_ID module_id={module.get('module_id', '')}")
        rows: list[dict[str, Any]] = []
        aggregate = 0.0
        for binding in block["bindings"]:
            allocated += 1
            semantic_id, target_slot = binding_key(binding)
            if semantic_id not in resolved:
                unavailable += 1
                raise RuntimeError(f"ESCAP4_RESOLVED_VALUE_MISSING semantic_id={semantic_id}")
            source = dict(resolved[semantic_id])
            binding_lesioned = (instance_id, semantic_id, target_slot) in lesions
            enabled = bool(binding["enabled"])
            gain = float(binding["gain"])
            contribution = 0.0
            if enabled:
                executable += 1
                if not binding_lesioned:
                    contribution = float(np.float32(gain * float(source["resolved_value"])))
            if contribution != 0.0:
                active += 1
            aggregate += contribution
            if not math.isfinite(aggregate):
                raise RuntimeError(f"ESCAP4_NONFINITE field=aggregate[{instance_id}]")
            rows.append(
                {
                    **source,
                    "target_slot": target_slot,
                    "gain": gain,
                    "binding_enabled": enabled,
                    "binding_lesioned": bool(binding_lesioned),
                    "contribution": contribution,
                }
            )
        bounded = float(np.float32(min(ROUTED_INPUT_BOUND, max(-ROUTED_INPUT_BOUND, aggregate))))
        was_clipped = bounded != float(aggregate)
        clipped += int(was_clipped)
        aggregates[instance_id] = bounded
        telemetry[instance_id] = {
            "module_id": str(module.get("module_id", "")),
            "module_instance_id": instance_id,
            "bindings": rows,
            "aggregate_before_clip": float(np.float32(aggregate)),
            "routed_observation_aggregate": bounded,
            "aggregate_clipped": was_clipped,
        }
    summary = {
        "schema_version": ROUTING_SCHEMA_VERSION,
        "registry_version": ROUTING_REGISTRY_VERSION,
        "profile": ROUTING_PROFILE,
        "exposure_mode": ROUTING_EXPOSURE_MODE,
        "eligible_semantic_ids": list(ADMITTED_SEMANTIC_IDS),
        "available_semantic_ids": sorted(str(key) for key in resolved),
        "allocated_bindings": allocated,
        "executable_bindings": executable,
        "active_bindings": active,
        "unavailable_bindings": unavailable,
        "clipped_module_aggregates": clipped,
        "lesioned_binding_count": len(lesions),
        "random_draws": 0,
        "lane": "cpu_reference",
    }
    return RoutingBatch(aggregates, telemetry, summary)


def route_observations_batch(
    alive: list[Any],
    obs_np: Any,
    *,
    local_size: int,
    sim_cfg: Any,
    lesion_semantic_ids: Iterable[str] = (),
    lesion_binding_keys: Iterable[tuple[str, str, str]] = (),
) -> list[RoutingBatch]:
    if not routing_enabled(sim_cfg):
        return [RoutingBatch({}, {}, {"execution": "disabled", "random_draws": 0}) for _ in alive]
    observations = np.asarray(obs_np, dtype=np.float32)
    if observations.ndim != 2 or observations.shape[0] != len(alive):
        raise RuntimeError("ESCAP4_BATCH_OBSERVATION_SHAPE_MISMATCH")
    # Validate every binding and resolve every value before ESCAP-3 mutates any
    # lifetime state.  This makes malformed batch input an atomic failure.
    normalized_modules = [
        normalize_agent_routing_modules(list(getattr(getattr(agent, "genome", None), "modules", []) or []))
        for agent in alive
    ]
    required = [
        sorted(
            {
                str(binding["semantic_id"])
                for module in modules
                for binding in (
                    dict(module.get("params", {}) or {}).get(ROUTING_BLOCK_KEY, {}).get("bindings", [])
                    if isinstance(dict(module.get("params", {}) or {}).get(ROUTING_BLOCK_KEY, {}), Mapping)
                    else []
                )
            }
        )
        for modules in normalized_modules
    ]
    resolved_rows = [
        resolve_routing_values(
            observations[index],
            local_size,
            required[index],
            lesion_semantic_ids=lesion_semantic_ids,
        )
        for index in range(len(alive))
    ]
    return [
        compile_module_routing(
            normalized_modules[index],
            resolved_rows[index],
            lesion_binding_keys=lesion_binding_keys,
        )
        for index in range(len(alive))
    ]
