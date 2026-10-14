from __future__ import annotations

"""ESCAP-5 bounded routing of live module outputs to existing action logits.

Bindings retain semantic action identifiers, never policy-vector offsets.  The
authoritative action vocabulary and numeric constants are reconciled at the
runtime boundary before any residual is compiled.  This module owns no RNG and
does not choose, mask, execute, or reinterpret actions.
"""

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

import numpy as np

from agents.agent import ACTION_VOCABULARY, ACTION_VOCABULARY_SCHEMA_VERSION
from env.gridworld_pkg.constants import (
    ACT_MOVE_E,
    ACT_MOVE_N,
    ACT_MOVE_S,
    ACT_MOVE_W,
    ACT_REST,
    N_ACTIONS,
)
from evolution.structural_live_modules_v1 import LIVE_PROFILE, live_modules_enabled


OUTPUT_ROUTING_SCHEMA_VERSION = "escap5_output_routing_v1"
OUTPUT_ROUTING_PROFILE = "escap5_modular_action_logit_residual_v1"
ACTION_SEMANTIC_REGISTRY_VERSION = "eiel_action_semantic_routing_registry_v1"
OUTPUT_ROUTING_BLOCK_KEY = "output_routing"
OUTPUT_ROUTING_ARBITRATION_VERSION = "escap5_output_arbitration_v1"
OUTPUT_ROUTING_AUTHORITY_ORDER_VERSION = "escap5_before_existing_authority_v1"
OUTPUT_ROUTING_OPERATOR_PROFILE = "escap5_output_binding_operator_v1"
OUTPUT_ROUTING_EXPOSURE_MODE = "parallel"
COMBINED_REST_COMPATIBILITY_VERSION = "escap5_combined_rest_residual_v1"

ADMITTED_ACTION_SEMANTIC_IDS: tuple[str, ...] = (
    "move_north",
    "move_south",
    "move_east",
    "move_west",
    "rest",
)
MAX_OUTPUT_BINDINGS_PER_MODULE = 2
MAX_OUTPUT_BINDINGS_PER_AGENT = 8
MAX_OUTPUT_BINDING_GAIN = 1.0
MAX_ACTION_RESIDUAL = 0.25
MODULE_OUTPUT_BOUND = 1.0

_SOURCE_SLOT = "out"


def output_routing_enabled(sim_cfg: Any) -> bool:
    return bool(getattr(sim_cfg, "structural_output_routing_enable", False))


def _runtime_action_index_by_semantic_id() -> dict[str, int]:
    """Resolve the admitted meanings through both existing action authorities."""

    vocabulary = tuple(str(value) for value in ACTION_VOCABULARY)
    if len(vocabulary) != int(N_ACTIONS):
        raise RuntimeError(
            "ESCAP5_ACTION_VOCABULARY_SIZE_MISMATCH "
            f"vocabulary={len(vocabulary)};constants={int(N_ACTIONS)}"
        )
    constant_indices = {
        "move_north": int(ACT_MOVE_N),
        "move_south": int(ACT_MOVE_S),
        "move_east": int(ACT_MOVE_E),
        "move_west": int(ACT_MOVE_W),
        "rest": int(ACT_REST),
    }
    resolved: dict[str, int] = {}
    for semantic_id in ADMITTED_ACTION_SEMANTIC_IDS:
        try:
            vocabulary_index = vocabulary.index(semantic_id)
        except ValueError as exc:
            raise RuntimeError(
                f"ESCAP5_ACTION_SEMANTIC_UNAVAILABLE semantic_id={semantic_id};"
                f"vocabulary_version={ACTION_VOCABULARY_SCHEMA_VERSION}"
            ) from exc
        constant_index = constant_indices[semantic_id]
        if vocabulary_index != constant_index:
            raise RuntimeError(
                "ESCAP5_ACTION_SEMANTIC_INDEX_MISMATCH "
                f"semantic_id={semantic_id};vocabulary_index={vocabulary_index};"
                f"constant_index={constant_index}"
            )
        resolved[semantic_id] = int(constant_index)
    return resolved


def validate_output_routing_config(
    sim_cfg: Any,
    *,
    device: Any = "cpu",
    policy_stride: int = 1,
) -> None:
    if not output_routing_enabled(sim_cfg):
        return
    if not live_modules_enabled(sim_cfg):
        raise RuntimeError("ESCAP5_REQUIRES_ESCAP3_LIVE_MODULES")
    live_profile = str(getattr(sim_cfg, "structural_live_modules_profile", "none") or "none")
    if live_profile != LIVE_PROFILE:
        raise RuntimeError(f"ESCAP5_ESCAP3_PROFILE_MISMATCH profile={live_profile}")

    expected = (
        OUTPUT_ROUTING_PROFILE,
        OUTPUT_ROUTING_SCHEMA_VERSION,
        ACTION_SEMANTIC_REGISTRY_VERSION,
        MAX_OUTPUT_BINDINGS_PER_MODULE,
        MAX_OUTPUT_BINDINGS_PER_AGENT,
        OUTPUT_ROUTING_OPERATOR_PROFILE,
        MAX_OUTPUT_BINDING_GAIN,
        MAX_ACTION_RESIDUAL,
        OUTPUT_ROUTING_ARBITRATION_VERSION,
        OUTPUT_ROUTING_AUTHORITY_ORDER_VERSION,
        COMBINED_REST_COMPATIBILITY_VERSION,
    )
    actual = (
        str(getattr(sim_cfg, "structural_output_routing_profile", "none") or "none"),
        str(
            getattr(
                sim_cfg,
                "structural_output_routing_schema_version",
                OUTPUT_ROUTING_SCHEMA_VERSION,
            )
            or OUTPUT_ROUTING_SCHEMA_VERSION
        ),
        str(
            getattr(
                sim_cfg,
                "structural_output_routing_registry_version",
                ACTION_SEMANTIC_REGISTRY_VERSION,
            )
            or ACTION_SEMANTIC_REGISTRY_VERSION
        ),
        int(
            getattr(
                sim_cfg,
                "structural_output_routing_max_bindings_per_module",
                MAX_OUTPUT_BINDINGS_PER_MODULE,
            )
            or MAX_OUTPUT_BINDINGS_PER_MODULE
        ),
        int(
            getattr(
                sim_cfg,
                "structural_output_routing_max_bindings_per_agent",
                MAX_OUTPUT_BINDINGS_PER_AGENT,
            )
            or MAX_OUTPUT_BINDINGS_PER_AGENT
        ),
        str(
            getattr(
                sim_cfg,
                "structural_output_routing_operator_profile",
                OUTPUT_ROUTING_OPERATOR_PROFILE,
            )
            or OUTPUT_ROUTING_OPERATOR_PROFILE
        ),
        float(
            getattr(
                sim_cfg,
                "structural_output_routing_max_binding_gain",
                MAX_OUTPUT_BINDING_GAIN,
            )
        ),
        float(
            getattr(
                sim_cfg,
                "structural_output_routing_max_action_residual",
                MAX_ACTION_RESIDUAL,
            )
        ),
        str(
            getattr(
                sim_cfg,
                "structural_output_routing_arbitration_version",
                OUTPUT_ROUTING_ARBITRATION_VERSION,
            )
            or OUTPUT_ROUTING_ARBITRATION_VERSION
        ),
        str(
            getattr(
                sim_cfg,
                "structural_output_routing_authority_order_version",
                OUTPUT_ROUTING_AUTHORITY_ORDER_VERSION,
            )
            or OUTPUT_ROUTING_AUTHORITY_ORDER_VERSION
        ),
        str(
            getattr(
                sim_cfg,
                "structural_output_routing_combined_rest_compatibility_version",
                COMBINED_REST_COMPATIBILITY_VERSION,
            )
            or COMBINED_REST_COMPATIBILITY_VERSION
        ),
    )
    if actual != expected:
        raise RuntimeError(
            f"ESCAP5_CONFIG_CONTRACT_MISMATCH expected={expected!r};actual={actual!r}"
        )
    attempts = int(getattr(sim_cfg, "structural_output_routing_operator_attempts", 1) or 0)
    if attempts != 1:
        raise RuntimeError(f"ESCAP5_OPERATOR_BUDGET_MISMATCH attempts={attempts};expected=1")
    if str(device) != "cpu":
        raise RuntimeError(f"ESCAP5_CPU_REFERENCE_ONLY device={device}")
    if int(policy_stride) != 1:
        raise RuntimeError(f"ESCAP5_POLICY_STRIDE_UNSUPPORTED stride={int(policy_stride)}")
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
        raise RuntimeError(f"ESCAP5_NATIVE_LANE_UNSUPPORTED flags={','.join(active)}")
    _runtime_action_index_by_semantic_id()


def _finite(value: Any, *, field: str) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise RuntimeError(f"ESCAP5_INVALID_NUMERIC field={field}") from exc
    if not math.isfinite(out):
        raise RuntimeError(f"ESCAP5_NONFINITE field={field}")
    return out


def _canonical_gain(value: Any) -> float:
    gain = _finite(value, field="binding.gain")
    if not -MAX_OUTPUT_BINDING_GAIN <= gain <= MAX_OUTPUT_BINDING_GAIN:
        raise RuntimeError(f"ESCAP5_BINDING_GAIN_OUT_OF_RANGE value={gain}")
    return float(np.float32(gain))


def output_binding_key(binding: Mapping[str, Any]) -> tuple[str, str]:
    return (
        str(binding.get("semantic_id", "") or ""),
        str(binding.get("source_slot", "") or ""),
    )


def normalize_output_routing_block(block: Any) -> dict[str, Any]:
    if block is None:
        block = {}
    if not isinstance(block, Mapping):
        raise RuntimeError("ESCAP5_BINDING_BLOCK_MALFORMED")
    schema = str(block.get("schema_version", OUTPUT_ROUTING_SCHEMA_VERSION) or OUTPUT_ROUTING_SCHEMA_VERSION)
    registry = str(
        block.get("registry_version", ACTION_SEMANTIC_REGISTRY_VERSION)
        or ACTION_SEMANTIC_REGISTRY_VERSION
    )
    if schema != OUTPUT_ROUTING_SCHEMA_VERSION:
        raise RuntimeError(f"ESCAP5_BINDING_SCHEMA_UNSUPPORTED schema={schema}")
    if registry != ACTION_SEMANTIC_REGISTRY_VERSION:
        raise RuntimeError(f"ESCAP5_REGISTRY_VERSION_MISMATCH registry={registry}")
    raw_bindings = block.get("bindings", [])
    if not isinstance(raw_bindings, list):
        raise RuntimeError("ESCAP5_BINDINGS_MALFORMED")
    if len(raw_bindings) > MAX_OUTPUT_BINDINGS_PER_MODULE:
        raise RuntimeError(
            "ESCAP5_MODULE_BINDING_CAP_EXCEEDED "
            f"observed={len(raw_bindings)};max={MAX_OUTPUT_BINDINGS_PER_MODULE}"
        )

    canonical: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, raw in enumerate(raw_bindings):
        if not isinstance(raw, Mapping):
            raise RuntimeError(f"ESCAP5_BINDING_MALFORMED index={index}")
        semantic_id = str(raw.get("semantic_id", "") or "")
        source_slot = str(raw.get("source_slot", "") or "")
        if semantic_id not in ADMITTED_ACTION_SEMANTIC_IDS:
            if semantic_id in tuple(str(value) for value in ACTION_VOCABULARY):
                raise RuntimeError(f"ESCAP5_ACTION_SEMANTIC_INELIGIBLE semantic_id={semantic_id}")
            raise RuntimeError(f"ESCAP5_UNKNOWN_ACTION_SEMANTIC_ID semantic_id={semantic_id}")
        if source_slot != _SOURCE_SLOT:
            raise RuntimeError(f"ESCAP5_SOURCE_SLOT_UNSUPPORTED source_slot={source_slot}")
        key = (semantic_id, source_slot)
        if key in seen:
            raise RuntimeError(
                f"ESCAP5_DUPLICATE_BINDING semantic_id={semantic_id};source_slot={source_slot}"
            )
        seen.add(key)
        enabled = raw.get("enabled", True)
        if not isinstance(enabled, bool):
            raise RuntimeError(f"ESCAP5_BINDING_ENABLED_NOT_BOOL semantic_id={semantic_id}")
        canonical.append(
            {
                "semantic_id": semantic_id,
                "source_slot": source_slot,
                "gain": _canonical_gain(raw.get("gain", 0.0)),
                "enabled": bool(enabled),
            }
        )
    canonical.sort(key=output_binding_key)
    return {
        "schema_version": OUTPUT_ROUTING_SCHEMA_VERSION,
        "registry_version": ACTION_SEMANTIC_REGISTRY_VERSION,
        "bindings": canonical,
    }


def normalize_agent_output_routing_modules(
    modules: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    binding_count = 0
    for raw_module in modules:
        module = dict(raw_module)
        params_obj = module.get("params", {})
        if not isinstance(params_obj, Mapping):
            raise RuntimeError("ESCAP5_MODULE_PARAMS_MALFORMED")
        params = dict(params_obj)
        if OUTPUT_ROUTING_BLOCK_KEY in params:
            block = normalize_output_routing_block(params.get(OUTPUT_ROUTING_BLOCK_KEY))
            if block["bindings"] and str(module.get("module_type", "")) != "generic":
                raise RuntimeError(
                    "ESCAP5_BINDING_MODULE_TYPE_UNSUPPORTED "
                    f"module_type={module.get('module_type', '')}"
                )
            params[OUTPUT_ROUTING_BLOCK_KEY] = block
            binding_count += len(block["bindings"])
        module["params"] = params
        out.append(module)
    if binding_count > MAX_OUTPUT_BINDINGS_PER_AGENT:
        raise RuntimeError(
            "ESCAP5_AGENT_BINDING_CAP_EXCEEDED "
            f"observed={binding_count};max={MAX_OUTPUT_BINDINGS_PER_AGENT}"
        )
    return out


@dataclass(frozen=True)
class OutputRoutingResult:
    residual: np.ndarray
    telemetry_by_instance_id: dict[str, dict[str, Any]]
    summary: dict[str, Any]


@dataclass(frozen=True)
class OutputRoutingBatch:
    residuals: np.ndarray
    rows: tuple[OutputRoutingResult, ...]


def compile_agent_output_routing(
    agent: Any = None,
    *,
    modules: Iterable[Mapping[str, Any]] | None = None,
    outputs_by_instance_id: Mapping[str, Any] | None = None,
    lesion_binding_keys: Iterable[tuple[str, str, str]] = (),
    legacy_rest_residual: Any = None,
) -> OutputRoutingResult:
    """Compile one agent's current live scalar outputs into action residuals."""

    if agent is not None:
        if modules is not None or outputs_by_instance_id is not None:
            raise RuntimeError("ESCAP5_AGENT_AND_EXPLICIT_INPUTS_AMBIGUOUS")
        modules = list(getattr(getattr(agent, "genome", None), "modules", []) or [])
        outputs_by_instance_id = dict(
            getattr(agent, "structural_live_module_outputs", {}) or {}
        )
    if modules is None:
        modules = []
    if outputs_by_instance_id is None:
        outputs_by_instance_id = {}
    if not isinstance(outputs_by_instance_id, Mapping):
        raise RuntimeError("ESCAP5_MODULE_OUTPUT_MAP_MALFORMED")

    action_indices = _runtime_action_index_by_semantic_id()
    canonical_modules = normalize_agent_output_routing_modules(modules)
    local_ids = [str(module.get("module_id", "") or "") for module in canonical_modules]
    instance_ids = [str(module.get("module_instance_id", "") or "") for module in canonical_modules]
    if len(set(local_ids)) != len(local_ids):
        raise RuntimeError("ESCAP5_DUPLICATE_MODULE_ID")
    populated_instances = [value for value in instance_ids if value]
    if len(set(populated_instances)) != len(populated_instances):
        raise RuntimeError("ESCAP5_DUPLICATE_MODULE_INSTANCE_ID")

    outputs = {
        str(key): _finite(value, field=f"module_output[{key}]")
        for key, value in outputs_by_instance_id.items()
    }
    legacy_rest_present = legacy_rest_residual is not None
    legacy_rest = (
        0.0
        if legacy_rest_residual is None
        else _finite(legacy_rest_residual, field="legacy_rest_residual")
    )
    if not -MAX_ACTION_RESIDUAL <= legacy_rest <= MAX_ACTION_RESIDUAL:
        raise RuntimeError(
            f"ESCAP5_LEGACY_REST_RESIDUAL_OUT_OF_RANGE value={legacy_rest}"
        )
    lesions = {(str(a), str(b), str(c)) for a, b, c in lesion_binding_keys}
    aggregate_by_semantic = {semantic_id: 0.0 for semantic_id in ADMITTED_ACTION_SEMANTIC_IDS}
    telemetry: dict[str, dict[str, Any]] = {}
    allocated = executable = active = unavailable = 0

    for module in sorted(canonical_modules, key=lambda value: str(value.get("module_id", ""))):
        if str(module.get("module_type", "")) != "generic":
            continue
        params = dict(module.get("params", {}) or {})
        if OUTPUT_ROUTING_BLOCK_KEY not in params:
            continue
        block = normalize_output_routing_block(params[OUTPUT_ROUTING_BLOCK_KEY])
        instance_id = str(module.get("module_instance_id", "") or "")
        if not instance_id:
            raise RuntimeError(
                f"ESCAP5_MISSING_MODULE_INSTANCE_ID module_id={module.get('module_id', '')}"
            )
        if block["bindings"] and instance_id not in outputs:
            unavailable += len(block["bindings"])
            raise RuntimeError(f"ESCAP5_MODULE_OUTPUT_MISSING module_instance_id={instance_id}")
        output = _finite(outputs.get(instance_id, 0.0), field=f"module_output[{instance_id}]")
        if not -MODULE_OUTPUT_BOUND <= output <= MODULE_OUTPUT_BOUND:
            raise RuntimeError(
                "ESCAP5_MODULE_OUTPUT_OUT_OF_RANGE "
                f"module_instance_id={instance_id};value={output}"
            )
        rows: list[dict[str, Any]] = []
        for binding in block["bindings"]:
            allocated += 1
            semantic_id, source_slot = output_binding_key(binding)
            binding_lesioned = (instance_id, semantic_id, source_slot) in lesions
            enabled = bool(binding["enabled"])
            gain = float(binding["gain"])
            contribution = 0.0
            if enabled:
                executable += 1
                if not binding_lesioned:
                    contribution = float(np.float32(gain * output))
            if contribution != 0.0:
                active += 1
            aggregate_by_semantic[semantic_id] += contribution
            if not math.isfinite(aggregate_by_semantic[semantic_id]):
                raise RuntimeError(
                    f"ESCAP5_NONFINITE field=aggregate[{semantic_id}]"
                )
            rows.append(
                {
                    "semantic_id": semantic_id,
                    "source_slot": source_slot,
                    "action_index": int(action_indices[semantic_id]),
                    "module_output": float(np.float32(output)),
                    "gain": gain,
                    "binding_enabled": enabled,
                    "binding_lesioned": bool(binding_lesioned),
                    "contribution": contribution,
                }
            )
        telemetry[instance_id] = {
            "module_id": str(module.get("module_id", "")),
            "module_instance_id": instance_id,
            "bindings": rows,
        }

    residual = np.zeros((int(N_ACTIONS),), dtype=np.float32)
    action_rows: list[dict[str, Any]] = []
    clipped_count = 0
    for semantic_id in ADMITTED_ACTION_SEMANTIC_IDS:
        routed_aggregate = _finite(
            aggregate_by_semantic[semantic_id],
            field=f"aggregate[{semantic_id}]",
        )
        aggregate = routed_aggregate
        if semantic_id == "rest":
            # Preserve the legacy float32 value without another arithmetic or
            # rounding step whenever the explicit routed aggregate is neutral.
            # Otherwise combine both REST sources before the one ESCAP-5 cap.
            if routed_aggregate == 0.0:
                bounded = legacy_rest
                aggregate = legacy_rest
            else:
                aggregate = _finite(
                    legacy_rest + routed_aggregate,
                    field="aggregate[rest_with_legacy]",
                )
                bounded = float(
                    np.float32(
                        min(MAX_ACTION_RESIDUAL, max(-MAX_ACTION_RESIDUAL, aggregate))
                    )
                )
        else:
            bounded = float(
                np.float32(
                    min(MAX_ACTION_RESIDUAL, max(-MAX_ACTION_RESIDUAL, aggregate))
                )
            )
        clipped = bounded != float(aggregate)
        clipped_count += int(clipped)
        action_index = int(action_indices[semantic_id])
        residual[action_index] = np.float32(bounded)
        action_rows.append(
            {
                "semantic_id": semantic_id,
                "action_index": action_index,
                "aggregate_before_clip": float(np.float32(aggregate)),
                "routed_aggregate_before_legacy_rest": float(np.float32(routed_aggregate)),
                "legacy_rest_residual": (
                    float(np.float32(legacy_rest)) if semantic_id == "rest" else 0.0
                ),
                "action_residual": bounded,
                "aggregate_clipped": clipped,
            }
        )

    summary = {
        "schema_version": OUTPUT_ROUTING_SCHEMA_VERSION,
        "registry_version": ACTION_SEMANTIC_REGISTRY_VERSION,
        "action_vocabulary_version": ACTION_VOCABULARY_SCHEMA_VERSION,
        "profile": OUTPUT_ROUTING_PROFILE,
        "exposure_mode": OUTPUT_ROUTING_EXPOSURE_MODE,
        "arbitration_version": OUTPUT_ROUTING_ARBITRATION_VERSION,
        "authority_order_version": OUTPUT_ROUTING_AUTHORITY_ORDER_VERSION,
        "combined_rest_compatibility_version": COMBINED_REST_COMPATIBILITY_VERSION,
        "legacy_rest_residual_present": bool(legacy_rest_present),
        "admitted_action_semantic_ids": list(ADMITTED_ACTION_SEMANTIC_IDS),
        "allocated_bindings": allocated,
        "executable_bindings": executable,
        "active_bindings": active,
        "unavailable_bindings": unavailable,
        "clipped_action_aggregates": clipped_count,
        "lesioned_binding_count": len(lesions),
        "action_rows": action_rows,
        "random_draws": 0,
        "lane": "cpu_reference",
    }
    return OutputRoutingResult(residual, telemetry, summary)


def compile_output_routing_batch(
    alive: list[Any],
    *,
    sim_cfg: Any,
    lesion_binding_keys: Iterable[tuple[str, str, str]] = (),
    legacy_rest_residuals: Any = None,
) -> OutputRoutingBatch:
    residuals = np.zeros((len(alive), int(N_ACTIONS)), dtype=np.float32)
    if legacy_rest_residuals is None:
        legacy_rest = np.zeros((len(alive),), dtype=np.float32)
        legacy_present = False
    else:
        legacy_rest = np.asarray(legacy_rest_residuals, dtype=np.float32).reshape(-1)
        legacy_present = True
        if int(legacy_rest.shape[0]) != len(alive):
            raise RuntimeError(
                "ESCAP5_LEGACY_REST_BATCH_SHAPE_MISMATCH "
                f"observed={int(legacy_rest.shape[0])};expected={len(alive)}"
            )
        if not bool(np.all(np.isfinite(legacy_rest))):
            raise RuntimeError("ESCAP5_NONFINITE field=legacy_rest_residuals")
        if bool(np.any(np.abs(legacy_rest) > np.float32(MAX_ACTION_RESIDUAL))):
            raise RuntimeError("ESCAP5_LEGACY_REST_RESIDUAL_OUT_OF_RANGE")
    if not output_routing_enabled(sim_cfg):
        if legacy_present:
            residuals[:, int(ACT_REST)] = legacy_rest
        rows = tuple(
            OutputRoutingResult(
                residuals[index].copy(),
                {},
                {
                    "execution": "disabled",
                    "combined_rest_compatibility_version": COMBINED_REST_COMPATIBILITY_VERSION,
                    "legacy_rest_residual_present": bool(legacy_present),
                    "random_draws": 0,
                },
            )
            for index in range(len(alive))
        )
        return OutputRoutingBatch(residuals, rows)
    rows = tuple(
        compile_agent_output_routing(
            agent,
            lesion_binding_keys=lesion_binding_keys,
            legacy_rest_residual=(legacy_rest[index] if legacy_present else None),
        )
        for index, agent in enumerate(alive)
    )
    for index, row in enumerate(rows):
        residuals[index, :] = row.residual
    return OutputRoutingBatch(residuals, rows)


def add_output_routing_residuals_(logits: Any, residuals: Any) -> Any:
    """Apply the ESCAP-5 residual matrix exactly once at the policy seam."""

    if isinstance(residuals, OutputRoutingBatch):
        residuals = residuals.residuals
    # The pure compiler is NumPy/CPU.  Convert explicitly at this single seam
    # when the existing policy path owns a torch tensor; this performs no RNG.
    if not isinstance(logits, np.ndarray) and isinstance(residuals, np.ndarray):
        try:
            import torch

            if torch.is_tensor(logits):
                residuals = torch.from_numpy(residuals).to(
                    device=logits.device,
                    dtype=logits.dtype,
                )
        except Exception as exc:
            raise RuntimeError("ESCAP5_LOGIT_SEAM_CONVERSION_FAILED") from exc
    if len(logits.shape) != 2 or len(residuals.shape) != 2:
        raise RuntimeError("ESCAP5_LOGIT_SEAM_SHAPE_MISMATCH")
    if tuple(int(value) for value in logits.shape) != tuple(
        int(value) for value in residuals.shape
    ):
        raise RuntimeError(
            "ESCAP5_LOGIT_SEAM_SHAPE_MISMATCH "
            f"logits={tuple(logits.shape)};residuals={tuple(residuals.shape)}"
        )
    if int(logits.shape[1]) != int(N_ACTIONS):
        raise RuntimeError(
            f"ESCAP5_LOGIT_SEAM_ACTION_WIDTH_INVALID observed={int(logits.shape[1])}"
        )
    try:
        if isinstance(logits, np.ndarray):
            if not bool(np.all(np.isfinite(logits))):
                raise RuntimeError("ESCAP5_NONFINITE field=base_logits")
            if not bool(np.all(np.isfinite(residuals))):
                raise RuntimeError("ESCAP5_NONFINITE field=logit_residuals")
            candidate = logits + residuals
            if not bool(np.all(np.isfinite(candidate))):
                raise RuntimeError("ESCAP5_NONFINITE field=final_logits")
            logits[...] = candidate
        else:
            import torch

            if not bool(torch.all(torch.isfinite(logits)).item()):
                raise RuntimeError("ESCAP5_NONFINITE field=base_logits")
            if not bool(torch.all(torch.isfinite(residuals)).item()):
                raise RuntimeError("ESCAP5_NONFINITE field=logit_residuals")
            candidate = logits + residuals
            if not bool(torch.all(torch.isfinite(candidate)).item()):
                raise RuntimeError("ESCAP5_NONFINITE field=final_logits")
            logits.copy_(candidate)
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError("ESCAP5_LOGIT_SEAM_NUMERIC_CHECK_FAILED") from exc
    return logits


__all__ = [
    "ACTION_SEMANTIC_REGISTRY_VERSION",
    "ADMITTED_ACTION_SEMANTIC_IDS",
    "COMBINED_REST_COMPATIBILITY_VERSION",
    "MAX_ACTION_RESIDUAL",
    "MAX_OUTPUT_BINDING_GAIN",
    "MAX_OUTPUT_BINDINGS_PER_AGENT",
    "MAX_OUTPUT_BINDINGS_PER_MODULE",
    "OUTPUT_ROUTING_ARBITRATION_VERSION",
    "OUTPUT_ROUTING_AUTHORITY_ORDER_VERSION",
    "OUTPUT_ROUTING_BLOCK_KEY",
    "OUTPUT_ROUTING_OPERATOR_PROFILE",
    "OUTPUT_ROUTING_PROFILE",
    "OUTPUT_ROUTING_SCHEMA_VERSION",
    "OutputRoutingBatch",
    "OutputRoutingResult",
    "add_output_routing_residuals_",
    "compile_agent_output_routing",
    "compile_output_routing_batch",
    "normalize_agent_output_routing_modules",
    "normalize_output_routing_block",
    "output_binding_key",
    "output_routing_enabled",
    "validate_output_routing_config",
]
