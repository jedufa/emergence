from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .bindings import load_prebuilt_extension, native_extension_status
from .types import ABI_VERSION, EXTENSION_VERSION, NativeEngineCounters, NativeGenerationResult, NativeRuntimeCapability


_COUNT_KEYS = (
    "native_tick_loop_count",
    "policy_loop_tick_calls",
    "policy_cache_rows_requested",
    "policy_cache_rows_hit",
    "policy_cache_rows_missed",
    "policy_cache_rows_refilled",
    "policy_cache_full_refill_ticks",
    "policy_cache_partial_refill_ticks",
    "policy_row64_probe_requested_count",
    "policy_row64_probe_rows_count",
    "policy_refill_calls",
    "policy_refill_rows",
    "cache_hit_rows",
    "policy_cache_hit_calls",
    "dev_stage_transition_count",
    "food_spawn_attempts",
    "food_spawned_rows",
    "applied_actions",
    "moved",
    "gathered",
    "action_move_n",
    "action_move_s",
    "action_move_e",
    "action_move_w",
    "action_move_total",
    "action_gather",
    "action_rest",
    "action_attack",
    "movement_attempts",
    "movement_blocked_occupied",
    "movement_blocked_bounds",
    "died",
    "death_rows",
    "committed_birth_rows",
    "failed_birth_rows",
    "public_failed_birth_rows",
    "birth_attempt_rows",
    "birth_pre_cooldown",
    "birth_pre_no_energy",
    "injury_blocked_births",
    "birth_pre_injury_gate",
    "birth_pre_no_space",
    "birth_pre_no_mate",
    "birth_fail_no_space",
    "birth_fail_no_mate",
    "birth_fail_buffer",
    "birth_fail_injury_gate",
    "birth_fail_scar_gate",
    "birth_fail_conception",
    "birth_fail_gestation",
    "alive_count",
    "capacity",
    "free_slots",
    "no_free_slot",
    "no_space",
    "buffer_blocked",
    "mutation_events",
    "typed_event_rows",
    "population_birth_noise_rows_consumed",
    "population_exp_prior_noise_rows_consumed",
    "native_population_standard_normal_rows_generated",
    "python_precomputed_child_random_rows_consumed",
    "spawn_rng_draws",
)

_FLOAT_KEYS = (
    "peak_injury",
)

_PATCH_REQUIRED_INTEGER_COUNT_KEYS = (
    "patch_movement_energy_event_count",
)

_PATCH_OPTIONAL_INTEGER_COUNT_KEYS = (
    "patch_spawn_sync_rows",
    "typed_death_event_rows",
    "env_rng_padding_draws",
)

_PATCH_REQUIRED_FLOAT_TOTAL_KEYS = (
    "patch_food_harvest_total",
    "patch_scent_emit_total",
    "patch_waste_emit_total",
    "patch_movement_energy_total",
)

_PATCH_OPTIONAL_FLOAT_TOTAL_KEYS = (
    "energy_collected_delta",
    "patch_scent_move_queued_total",
    "patch_waste_move_queued_total",
    "patch_scent_gather_queued_total",
    "patch_waste_gather_queued_total",
    "patch_movement_energy_extra",
    "patch_movement_energy_extra_total",
    "patch_scent_emitted_total",
    "patch_waste_emitted_total",
)

# These are the bounded Patch substrate ratios in the reference metrics schema.
_PATCH_BOUNDED_RATIO_KEYS = (
    "patch_food_depleted_frac",
    "patch_food_full_frac",
    "patch_density_hotspot_frac",
)

_PATCH_STRUCTURED_DIAGNOSTIC_KEYS = (
    "stage_order",
    "patch_generation_core",
    "population_death_cull",
    "population_asexual_resident_birth_commit",
    "microcycles",
)


def _runtime_capability(
    *,
    world: str,
    prebuilt_available: bool,
    extension_abi_version: str = "",
    extension_version: str = "",
    policy_family: str = "",
    device: str = "",
    lifecycle_branch: str = "",
    checkpoint_mode: str = "",
    reporting_supported: bool = False,
    reporting_qualification: str = "",
    benchmark_qualified: bool = False,
    accepted_default_native_envelope: bool = False,
) -> NativeRuntimeCapability:
    normal_world = str(world or "").strip().lower()
    normal_policy_family = str(policy_family or "").strip().lower()
    normal_device = str(device or "").strip().lower()
    normal_lifecycle_branch = str(lifecycle_branch or "").strip().lower()
    normal_checkpoint_mode = str(checkpoint_mode or "").strip().lower()
    abi_compatible = bool(
        prebuilt_available
        and extension_abi_version == ABI_VERSION
        and extension_version == EXTENSION_VERSION
    )
    flat_supported = bool(
        abi_compatible
        and normal_world == "flat"
        and normal_policy_family == "mlp"
        and normal_device == "cpu"
        and normal_lifecycle_branch == "flat_generation_birth_commit"
        and normal_checkpoint_mode in {"off", "generation_boundary_v2"}
        and reporting_supported
    )
    return NativeRuntimeCapability(
        extension_version=str(extension_version or ""),
        extension_abi_version=str(extension_abi_version or ""),
        active_world=normal_world,
        policy_family=normal_policy_family,
        device=normal_device,
        lifecycle_branch=normal_lifecycle_branch,
        checkpoint_mode=normal_checkpoint_mode,
        reporting_support=(
            str(reporting_qualification or "qualified")
            if reporting_supported
            else "unqualified"
        ),
        benchmark_qualification="qualified" if benchmark_qualified else "not_qualified",
        extension_abi_compatible=abi_compatible,
        accepted_default_native_envelope=bool(accepted_default_native_envelope),
        cpu=abi_compatible and normal_device == "cpu",
        cuda=False,
        persistent_engine=abi_compatible,
        runtime_jit_in_backend=not bool(prebuilt_available),
        production_jit_free=abi_compatible,
        supports_flat_generation=flat_supported,
        supports_patch_generation=False,
        supports_asexual_resident_birth_commit=flat_supported,
        supports_patch_churn=False,
        supports_exact_production_asexual_child_construction=flat_supported,
        supports_exact_production_spawn_rng_order=flat_supported,
        supports_exact_production_mate_selection=False,
        supports_exact_production_recombination=False,
        supports_exact_production_mutation_rng_budget=flat_supported,
        supports_exact_death_cull_slot_reuse_lineage=flat_supported,
        supports_exact_typed_event_tape=flat_supported,
        supports_exact_public_output_serialization=flat_supported,
        supports_cxx19_formal_benchmark_gate=(
            flat_supported
            and bool(benchmark_qualified)
            and bool(accepted_default_native_envelope)
        ),
        supports_python_agent_sync_in_normal_lane=False,
    )


def _runtime_capability_for_runner(
    runner: Any,
    *,
    world: str,
    safe_default_decision: Any,
) -> NativeRuntimeCapability:
    sim_cfg = getattr(runner, "sim_cfg", None)
    return _runtime_capability(
        world=world,
        prebuilt_available=bool(safe_default_decision.prebuilt_extension_available),
        extension_abi_version=safe_default_decision.extension_abi_version,
        extension_version=safe_default_decision.extension_version,
        policy_family=str(getattr(sim_cfg, "native_runtime_fixed_policy_family", "") or ""),
        device=str(getattr(sim_cfg, "device", "") or ""),
        lifecycle_branch=(
            "flat_generation_birth_commit"
            if bool(getattr(sim_cfg, "native_runtime_birth_commit_enable", True))
            else "flat_generation_preflight"
        ),
        checkpoint_mode=safe_default_decision.checkpoint_mode,
        reporting_supported=safe_default_decision.reporting_supported,
        reporting_qualification=safe_default_decision.reporting_qualification,
        benchmark_qualified=safe_default_decision.benchmark_qualified,
        accepted_default_native_envelope=bool(safe_default_decision.lifecycle_default_active),
    )

_ROW_KEYS = (
    "spawn_rows",
    "spawn_candidate_rows_debug",
    "child_hash_rows",
    "child_hash_chain_rows",
    "child_drive_rows",
    "birth_genome_trace_rows",
    "reproduction_parent_state_rows",
    "failed_parent_agent_ids",
    "failed_birth_reason_codes",
    "failed_birth_reasons",
    "failed_spawn_rows",
    "mutation_rows",
    "parent_agent_ids",
    "mate_agent_ids",
    "child_agent_ids",
    "child_slots",
    "child_positions",
    "behavior_rows",
    "policy_rows_debug",
    "policy_row64_probe",
    "native_policy_cache_trace",
    "native_rng_phase_trace",
    "food_spawn_positions",
    "food_spawn_attempt_positions",
    "food_spawn_attempt_rows",
)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _json_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _decode_packed_native_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        kind = str(value.get("kind", "") or "")
        if kind in {"tensor", "ndarray"} and "data_hex" in value and "dtype" in value and "shape" in value:
            data = bytes.fromhex(str(value["data_hex"]))
            expected_sha = str(value.get("sha256", "") or "")
            if expected_sha and hashlib.sha256(data).hexdigest() != expected_sha:
                raise ValueError(f"native checkpoint {kind} sha256 mismatch")
            dtype = np.dtype(str(value["dtype"]))
            shape = tuple(int(x) for x in list(value.get("shape") or []))
            arr = np.frombuffer(data, dtype=dtype).copy()
            expected_size = int(np.prod(shape, dtype=np.int64)) if shape else 1
            if int(arr.size) != expected_size:
                raise ValueError(
                    f"native checkpoint {kind} size mismatch: "
                    f"shape={list(shape)} dtype={dtype} bytes={len(data)}"
                )
            arr = np.ascontiguousarray(arr.reshape(shape))
            if kind == "tensor":
                return torch.from_numpy(arr).contiguous()
            return arr
        return {str(k): _decode_packed_native_value(v) for k, v in dict(value).items()}
    if isinstance(value, list):
        return [_decode_packed_native_value(v) for v in value]
    return value


def decode_bundled_native_checkpoint_projection(sidecar: Mapping[str, Any]) -> dict[str, Any]:
    """Decode a checkpoint bundle native sidecar into a restore-ready projection."""

    if str(sidecar.get("schema_version", "") or "") != "native_runtime_checkpoint_projection_v1":
        raise ValueError("unsupported native runtime checkpoint sidecar schema")
    status = str(sidecar.get("status", "") or "")
    if status != "present":
        raise ValueError(f"native runtime checkpoint sidecar is not present: {status or 'missing'}")
    projection = sidecar.get("projection")
    if not isinstance(projection, Mapping):
        raise ValueError("native runtime checkpoint sidecar has no projection object")
    decoded = _decode_packed_native_value(projection)
    if not isinstance(decoded, dict):
        raise ValueError("decoded native runtime checkpoint projection is not an object")
    tensors = decoded.get("tensors")
    if not isinstance(tensors, dict) or not all(hasattr(tensor, "detach") for tensor in tensors.values()):
        raise ValueError("decoded native runtime checkpoint projection has invalid tensors")
    metadata = decoded.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("decoded native runtime checkpoint projection has invalid metadata")
    return decoded


def _profile_world(value: Any) -> str:
    if isinstance(value, Mapping):
        if value.get("supported") is False:
            return ""
        value = value.get("world", value.get("profile", ""))
    elif not isinstance(value, (str, bytes, bytearray)) and hasattr(value, "profile"):
        if getattr(value, "supported", True) is False:
            return ""
        value = getattr(value, "profile", "")
    normal = str(value or "").strip().lower()
    if normal in {"p0", "p1", "p2"}:
        return "patch"
    if normal in {"patch", "flat"}:
        return normal
    return ""


def _finite_patch_profile(value: Any) -> str:
    if isinstance(value, Mapping):
        if value.get("supported") is False:
            return ""
        value = value.get("profile", "")
    elif not isinstance(value, (str, bytes, bytearray)) and hasattr(value, "profile"):
        if getattr(value, "supported", True) is False:
            return ""
        value = getattr(value, "profile", "")
    normal = str(value or "").strip().upper()
    return normal if normal in {"P0", "P1", "P2"} else ""


def _declared_patch_profile(raw: Mapping[str, Any], *, world: Any, profile: Any) -> bool:
    values = (
        ("world argument", world),
        ("profile argument", profile),
        ("raw world", raw.get("world")),
        ("raw profile", raw.get("profile")),
    )
    declarations: set[str] = set()
    for label, value in values:
        if value is None:
            continue
        declaration = _profile_world(value)
        if not declaration:
            raise ValueError(f"native compact result has invalid {label} declaration")
        declarations.add(declaration)
    if len(declarations) > 1:
        raise ValueError(
            "native compact result has conflicting world/profile declarations: "
            f"{sorted(declarations)}"
        )
    return declarations == {"patch"}


def _declared_finite_patch_profile(raw: Mapping[str, Any], *, profile: Any) -> str:
    declarations = {
        declaration
        for declaration in (
            _finite_patch_profile(profile),
            _finite_patch_profile(raw.get("profile")),
        )
        if declaration
    }
    if len(declarations) > 1:
        raise ValueError(
            "native compact result has conflicting finite Patch profiles: "
            f"{sorted(declarations)}"
        )
    if not declarations:
        raise ValueError("native Patch compact result is missing finite P0/P1/P2 profile metadata")
    return declarations.pop()


def _strict_patch_count(value: Any, *, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"native Patch compact field {key} must be an integer count")
    result = int(value)
    if result < 0:
        raise ValueError(f"native Patch compact field {key} must be nonnegative")
    return result


def _strict_patch_float(value: Any, *, key: str, bounded: bool = False) -> float:
    if isinstance(value, Integral) or not isinstance(value, Real):
        kind = "bounded ratio" if bounded else "finite float total"
        raise ValueError(f"native Patch compact field {key} must be a {kind}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"native Patch compact field {key} must be finite")
    if bounded and not 0.0 <= result <= 1.0:
        raise ValueError(f"native Patch compact field {key} must be in [0, 1]")
    return result


def _strict_patch_diagnostic(value: Any, *, key: str) -> Any:
    if key == "stage_order":
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes, bytearray))
            or not all(isinstance(stage, str) and stage for stage in value)
        ):
            raise ValueError("native Patch compact field stage_order must be a sequence of stage names")
    elif key == "microcycles":
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes, bytearray))
            or not all(isinstance(cycle, Mapping) for cycle in value)
        ):
            raise ValueError("native Patch compact field microcycles must be a sequence of mappings")
    elif not isinstance(value, Mapping):
        raise ValueError(f"native Patch compact field {key} must be a mapping")
    return copy.deepcopy(value)


def _add_patch_compact_fields(
    raw: Mapping[str, Any],
    compact: dict[str, Any],
    *,
    profile: Any,
) -> None:
    required = (*_PATCH_REQUIRED_INTEGER_COUNT_KEYS, *_PATCH_REQUIRED_FLOAT_TOTAL_KEYS)
    missing = [key for key in required if key not in raw or raw[key] is None]
    if missing:
        raise ValueError(
            "native Patch compact result is missing required reporting fields: "
            + ", ".join(missing)
        )

    compact["world"] = "patch"
    compact["profile"] = _declared_finite_patch_profile(raw, profile=profile)

    for key in _COUNT_KEYS:
        if key in raw:
            compact[key] = _strict_patch_count(raw[key], key=key)
    if "native_stage_count" in raw:
        compact["native_stage_count"] = _strict_patch_count(
            raw["native_stage_count"],
            key="native_stage_count",
        )
    for key in (*_PATCH_REQUIRED_INTEGER_COUNT_KEYS, *_PATCH_OPTIONAL_INTEGER_COUNT_KEYS):
        if key in raw:
            compact[key] = _strict_patch_count(raw[key], key=key)
    for key in (*_PATCH_REQUIRED_FLOAT_TOTAL_KEYS, *_PATCH_OPTIONAL_FLOAT_TOTAL_KEYS):
        if key in raw:
            compact[key] = _strict_patch_float(raw[key], key=key)
    for key in _PATCH_BOUNDED_RATIO_KEYS:
        if key in raw:
            compact[key] = _strict_patch_float(raw[key], key=key, bounded=True)

    for key in _PATCH_STRUCTURED_DIAGNOSTIC_KEYS:
        if key in raw:
            compact[key] = _strict_patch_diagnostic(raw[key], key=key)

    stage_order = tuple(compact.get("stage_order", ()))
    stage_count = int(compact.get("native_stage_count", 0))
    if stage_count:
        if not stage_order:
            raise ValueError("native Patch staged result is missing required stage_order")
        if len(stage_order) != stage_count:
            raise ValueError(
                "native Patch compact stage_order length does not match native_stage_count"
            )
    required_diagnostics = {
        "patch_generation_core"
        for stage in stage_order
        if stage in {"patch_generation_core", "patch_tick_core"}
    }
    if "population_death_cull" in stage_order:
        required_diagnostics.add("population_death_cull")
    if "population_resident_birth_commit" in stage_order:
        required_diagnostics.add("population_asexual_resident_birth_commit")
    if "native_lifecycle_microcycle_loop" in stage_order:
        required_diagnostics.add("microcycles")
    if raw.get("native_lifecycle_microcycles") is True:
        required_diagnostics.add("microcycles")
    missing_diagnostics = sorted(key for key in required_diagnostics if key not in raw)
    if missing_diagnostics:
        raise ValueError(
            "native Patch compact result is missing declared diagnostic payloads: "
            + ", ".join(missing_diagnostics)
        )


def _compact_raw_result(
    raw: dict[str, Any],
    *,
    native_batch_owner: bool | None = None,
    world: Any = None,
    profile: Any = None,
) -> dict[str, Any]:
    compact: dict[str, Any] = {
        "schema_version": "eiel_native_generation_compact_result_v2",
        "raw_result_schema_version": str(raw.get("schema_version", "")),
        "decision": str(raw.get("decision", "")),
        "native_stage_count": _as_int(raw.get("native_stage_count", 0)),
        "blocker": str(raw.get("blocker", "")),
    }
    if native_batch_owner is not None:
        compact["native_batch_owner"] = bool(native_batch_owner)
    for key in _COUNT_KEYS:
        if key in raw:
            compact[key] = _as_int(raw.get(key, 0))
    from .public_metrics_overlay import native_policy_cache_metrics

    compact.update(native_policy_cache_metrics(raw, compact))
    for key in _FLOAT_KEYS:
        if key in raw:
            compact[key] = float(raw.get(key, 0.0) or 0.0)
    for key in _ROW_KEYS:
        if key not in raw:
            continue
        rows = list(raw.get(key) or [])
        compact[f"{key}_count"] = len(rows)
        compact[f"{key}_digest"] = _json_digest(rows)
        if key == "child_hash_chain_rows":
            compact["child_hash_chain_stage_names"] = [str(row[3]) for row in rows if len(row) > 3]
    try:
        debug_limit = int(os.environ.get("EIEL_NATIVE_RUNTIME_DEBUG_ROW_SAMPLE", "0") or "0")
    except ValueError:
        debug_limit = 0
    if debug_limit > 0:
        sample_keys = (
            "action_rows",
            "food_cells_pre_action",
            "food_spawn_positions",
            "food_spawn_attempt_positions",
            "food_spawn_attempt_rows",
            "spawn_rows",
            "failed_spawn_rows",
            "spawn_candidate_rows_debug",
            "child_drive_rows",
            "birth_genome_trace_rows",
            "reproduction_parent_state_rows",
            "parent_agent_ids",
            "child_agent_ids",
            "child_positions",
            "policy_rows_debug",
            "native_policy_cache_trace",
            "native_rng_phase_trace",
            "food_spawn_positions",
            "food_spawn_attempt_positions",
            "food_spawn_attempt_rows",
        )
        compact["debug_row_sample_limit"] = int(debug_limit)
        compact["debug_row_samples"] = {
            key: list(raw.get(key) or [])[:debug_limit]
            for key in sample_keys
            if key in raw
        }
    if _declared_patch_profile(raw, world=world, profile=profile):
        _add_patch_compact_fields(raw, compact, profile=profile)
    return compact


class NativeEielEngine:
    """Persistent native runtime owner for one EIEL cell.

    This is the production-package integration point for the supported fixed
    Flat CPU MLP envelope.  With a prebuilt extension, the lifetime semantics
    are production-shaped: one construction, many generation calls, no runtime
    JIT, and no normal Python Agent synchronization.
    """

    def __init__(
        self,
        runner: Any,
        *,
        generation: int = 0,
        tick: int = 0,
        build_directory: str | Path | None = None,
    ) -> None:
        from perf.cxxbackbone10_resident_state_owner import (  # noqa: WPS433
            ResidentRuntimeState,
            build_import_projection,
        )

        patch = bool(getattr(getattr(runner, "env_cfg", None), "patch_world_enable", False))
        self.runner = runner
        self.world = "patch" if patch else "flat"
        self.capability_mask = {"flat": not patch, "patch": patch, "churn": True}
        sim_cfg = getattr(runner, "sim_cfg", None)
        env_cfg = getattr(runner, "env_cfg", None)
        from .safe_default import resolve_native_safe_default  # noqa: WPS433

        safe_default_decision = resolve_native_safe_default(sim_cfg, env_cfg, runner)
        if not safe_default_decision.native_supported:
            joined = ",".join(safe_default_decision.unsupported_reasons)
            raise RuntimeError(f"EIEL-NATIVE-RUNTIME-UNSUPPORTED: {joined}")
        if not safe_default_decision.extension_loaded:
            raise RuntimeError("EIEL-NATIVE-RUNTIME-UNSUPPORTED: native_extension_not_loaded")
        self.safe_default_decision = safe_default_decision
        self.capability = _runtime_capability_for_runner(
            runner,
            world=self.world,
            safe_default_decision=safe_default_decision,
        )
        self.counters = NativeEngineCounters(engine_constructions=1, import_projection_calls=1)
        projection = build_import_projection(
            runner,
            self.capability_mask,
            generation=int(generation),
            tick=int(tick),
        )
        self.patch_profile = (
            str(projection.metadata.get("native_runtime_patch_profile", "") or "")
            if self.world == "patch"
            else None
        )
        mod = load_prebuilt_extension(safe_default_decision.extension_path)
        native = mod.ResidentRuntimeStateV10(
            projection.tensors,
            copy.deepcopy(projection.metadata),
        )
        if projection.pending_generation_free_slots:
            native.restore_checkpoint_projection(
                {
                    "schema_version": projection.schema_version,
                    "metadata": copy.deepcopy(projection.metadata),
                    "tensors": projection.tensors,
                    "pending_generation_free_agent_ids": list(
                        projection.pending_generation_free_agent_ids
                    ),
                    "pending_generation_free_slots": list(projection.pending_generation_free_slots),
                }
            )
        self._owner = ResidentRuntimeState(native, projection)

    @classmethod
    def from_runner(
        cls,
        runner: Any,
        *,
        generation: int = 0,
        tick: int = 0,
        build_directory: str | Path | None = None,
    ) -> "NativeEielEngine":
        return cls(runner, generation=generation, tick=tick, build_directory=build_directory)

    @classmethod
    def from_checkpoint_projection(
        cls,
        checkpoint_projection: dict[str, Any],
        *,
        build_directory: str | Path | None = None,
        extension_path: str | Path | None = None,
        allow_partial_generation_for_diagnostics: bool = False,
    ) -> "NativeEielEngine":
        from perf.cxxbackbone10_resident_state_owner import (  # noqa: WPS433
            ImportProjectionV10,
            ResidentRuntimeState,
        )

        self = cls.__new__(cls)
        metadata = copy.deepcopy(dict(checkpoint_projection.get("metadata") or {}))
        tensors = dict(checkpoint_projection.get("tensors") or {})
        try:
            checkpoint_tick = int(metadata.get("tick", 0) or 0)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("native checkpoint tick is malformed") from exc
        checkpoint_phase = str(metadata.get("agent_grid_runner_phase", "") or "").strip().lower()
        if (
            not allow_partial_generation_for_diagnostics
            and (checkpoint_tick != 0 or checkpoint_phase != "generation_boundary")
        ):
            raise RuntimeError(
                "partial-generation native checkpoint continuation is not supported: "
                f"tick={checkpoint_tick} phase={checkpoint_phase or '<missing>'}"
            )
        self.runner = None
        raw_world = metadata.get("world")
        if type(raw_world) is not str or raw_world not in {"flat", "patch"}:
            raise RuntimeError("native checkpoint requires exact world metadata")
        self.world = raw_world
        self.capability_mask = dict(metadata.get("capability_mask") or {})
        if not self.capability_mask:
            self.capability_mask = {
                "flat": self.world != "patch",
                "patch": self.world == "patch",
                "churn": True,
            }
        if self.world == "patch":
            if extension_path is None or not str(extension_path):
                raise RuntimeError(
                    "EIEL-NATIVE-RUNTIME-UNSUPPORTED: patch_checkpoint_route_not_qualified: "
                    "Patch checkpoint restore requires an explicit development extension"
                )
            profile = metadata.get("native_runtime_patch_profile")
            if type(profile) is not str or profile not in {"P0", "P1", "P2"}:
                raise RuntimeError(
                    "EIEL-NATIVE-RUNTIME-UNSUPPORTED: Patch checkpoint requires exact "
                    "finite P0/P1/P2 profile metadata"
                )
            extension = native_extension_status(extension_path)
            if not (
                extension.available
                and extension.abi_compatible
                and extension.output_sha256_verified
                and not extension.runtime_jit
            ):
                reason = extension.error or "explicit development extension is not qualified"
                raise RuntimeError(f"EIEL-NATIVE-RUNTIME-UNSUPPORTED: {reason}")
            mod = load_prebuilt_extension(extension.extension_path)
            owner = ResidentRuntimeState.from_checkpoint_projection(
                checkpoint_projection,
                native_owner_module=mod,
            )
            metadata = copy.deepcopy(owner.import_projection.metadata)
            self.patch_profile = str(metadata["native_runtime_patch_profile"])
            self.capability_mask = dict(metadata["capability_mask"])
            self.capability = _runtime_capability(
                world="patch",
                prebuilt_available=True,
                extension_abi_version=extension.extension_abi_version,
                extension_version=extension.extension_version,
                policy_family=str(metadata.get("native_runtime_fixed_policy_family", "") or ""),
                device="cpu",
                lifecycle_branch="patch_checkpoint_development_owner",
                checkpoint_mode="generation_boundary_v2",
                reporting_supported=False,
                benchmark_qualified=False,
                accepted_default_native_envelope=False,
            )
            self.counters = NativeEngineCounters(
                engine_constructions=1,
                checkpoint_restore_calls=1,
            )
            self._owner = owner
            return self
        self.patch_profile = None
        from .safe_default import qualify_native_extension  # noqa: WPS433

        extension = qualify_native_extension()
        if not extension.supported or not extension.loaded:
            joined = ",".join(extension.reasons)
            raise RuntimeError(f"EIEL-NATIVE-RUNTIME-UNSUPPORTED: {joined}")
        self.capability = _runtime_capability(
            world=self.world,
            prebuilt_available=extension.available,
            extension_abi_version=extension.abi_version,
            extension_version=extension.extension_version,
            policy_family=str(metadata.get("native_runtime_fixed_policy_family", "") or ""),
            device="cpu",
            lifecycle_branch="checkpoint_continuation",
            checkpoint_mode="generation_boundary_v2",
            reporting_supported=False,
            benchmark_qualified=False,
            accepted_default_native_envelope=False,
        )
        self.counters = NativeEngineCounters(engine_constructions=1, checkpoint_restore_calls=1)
        mod = load_prebuilt_extension(extension.path)
        native = mod.ResidentRuntimeStateV10(tensors, copy.deepcopy(metadata))
        restore_projection = dict(checkpoint_projection)
        restore_projection["metadata"] = copy.deepcopy(metadata)
        native.restore_checkpoint_projection(restore_projection)
        projection = ImportProjectionV10(
            schema_version=str(checkpoint_projection.get("schema_version", "native_checkpoint_projection")),
            metadata=metadata,
            tensors=tensors,
            pending_generation_free_agent_ids=tuple(
                int(value) for value in checkpoint_projection.get("pending_generation_free_agent_ids", ())
            ),
            pending_generation_free_slots=tuple(
                int(value) for value in checkpoint_projection.get("pending_generation_free_slots", ())
            ),
        )
        self._owner = ResidentRuntimeState(native, projection)
        return self

    def bind_qualified_runner(self, runner: Any, safe_default_decision: Any) -> None:
        """Bind a restored owner only after the destination runner qualifies."""

        patch = bool(getattr(getattr(runner, "env_cfg", None), "patch_world_enable", False))
        destination_world = "patch" if patch else "flat"
        if destination_world != self.world:
            raise RuntimeError(
                "native checkpoint world does not match destination runner: "
                f"checkpoint={self.world!r} destination={destination_world!r}"
            )
        if not (
            safe_default_decision.native_supported
            and safe_default_decision.native_execution_active
            and safe_default_decision.extension_loaded
            and safe_default_decision.extension_abi_compatible
            and safe_default_decision.extension_hash_verified
            and not safe_default_decision.extension_runtime_jit
        ):
            raise RuntimeError("native checkpoint destination runner is not qualified")
        self.runner = runner
        self.safe_default_decision = safe_default_decision
        self.capability = _runtime_capability_for_runner(
            runner,
            world=self.world,
            safe_default_decision=safe_default_decision,
        )

    def execute_generation(
        self,
        *,
        generation_index: int,
        steps_per_generation: int,
        max_births: int = 1,
        birth_commit: bool = True,
    ) -> NativeGenerationResult:
        self.counters.generation_calls += 1
        if self.world == "patch":
            raw = self._owner.execute_generation_patch_cpu(int(generation_index), int(steps_per_generation))
            committed_birth_rows = 0
        elif birth_commit:
            raw = self._owner.execute_generation_flat_with_population_birth_commit_cpu(
                int(generation_index),
                int(steps_per_generation),
                int(max_births),
            )
            committed_birth_rows = int(raw.get("committed_birth_rows", 0))
        else:
            raw = self._owner.execute_generation_flat_with_population_preflight_cpu(
                int(generation_index),
                int(steps_per_generation),
            )
            committed_birth_rows = 0
        return NativeGenerationResult(
            schema_version="eiel_native_generation_result_v1",
            decision="EIEL-NATIVE-RUNTIME-PERSISTENT-GENERATION",
            executed=bool(raw.get("executed", False)),
            generation_index=int(generation_index),
            steps_per_generation=int(steps_per_generation),
            world=self.world,
            native_tick_loop_count=int(raw.get("native_tick_loop_count", steps_per_generation)),
            committed_birth_rows=committed_birth_rows,
            counters=self.counters,
            capability=self.capability,
            compact_result=_compact_raw_result(
                raw,
                world=self.world,
                profile=self.patch_profile,
            ),
            raw_result=dict(raw),
        )

    def execute_generation_boundary_turnover(self) -> dict[str, Any]:
        return self._owner.execute_population_generation_turnover_cpu()

    def export_checkpoint_projection(self) -> dict[str, Any]:
        return self._owner.export_checkpoint_projection()

    def export_validation_projection(self) -> dict[str, Any]:
        self.counters.validation_export_calls += 1
        return self._owner.export_validation_projection()

    def export_report_projection(self) -> dict[str, Any]:
        self.counters.report_export_calls += 1
        return self._owner.export_report_projection()

    def restore_checkpoint_projection(self, projection: dict[str, Any]) -> None:
        self.counters.checkpoint_restore_calls += 1
        self._owner.restore_checkpoint_projection(projection)

    def import_checkpoint_runtime_prefix(
        self,
        projection: dict[str, Any],
    ) -> dict[str, Any]:
        return self._owner.import_checkpoint_runtime_prefix(projection)


class NativeEielBatchEngine:
    """Persistent batch owner scaffold.

    This keeps cell engines alive and exposes one batch-generation call.  The
    native batch path iterates resident owners without per-cell Python runner
    lifecycle loops.
    """

    def __init__(self, engines: Iterable[NativeEielEngine]) -> None:
        self.engines = list(engines)
        self.counters = NativeEngineCounters(engine_constructions=len(self.engines))

    @classmethod
    def from_runners(
        cls,
        runners: Iterable[Any],
        *,
        build_directory: str | Path | None = None,
    ) -> "NativeEielBatchEngine":
        return cls(NativeEielEngine.from_runner(runner, build_directory=build_directory) for runner in runners)

    def execute_generation(
        self,
        *,
        generation_index: int,
        steps_per_generation: int,
        max_births: int = 1,
    ) -> list[NativeGenerationResult]:
        self.counters.batch_generation_calls += 1
        prebuilt_status = native_extension_status()
        if prebuilt_status.available and all(engine.world == "flat" for engine in self.engines):
            mod = load_prebuilt_extension(prebuilt_status.extension_path)
            native_owners = [engine._owner._native_owner for engine in self.engines]
            raw = mod.execute_native_batch_flat_with_population_birth_commit_cpu(
                native_owners,
                int(generation_index),
                int(steps_per_generation),
                int(max_births),
            )
            results: list[NativeGenerationResult] = []
            raw_results = list(raw.get("results", []))
            self.counters.native_batch_generation_calls += 1
            self.counters.batch_native_cell_loops += len(self.engines)
            for engine, cell_raw in zip(self.engines, raw_results, strict=True):
                engine.counters.generation_calls += 1
                results.append(
                    NativeGenerationResult(
                        schema_version="eiel_native_generation_result_v1",
                        decision="EIEL-NATIVE-RUNTIME-PERSISTENT-GENERATION",
                        executed=bool(cell_raw.get("executed", False)),
                        generation_index=int(generation_index),
                        steps_per_generation=int(steps_per_generation),
                        world=engine.world,
                        native_tick_loop_count=int(cell_raw.get("native_tick_loop_count", steps_per_generation)),
                        committed_birth_rows=int(cell_raw.get("committed_birth_rows", 0)),
                        counters=engine.counters,
                        capability=engine.capability,
                        compact_result=_compact_raw_result(
                            dict(cell_raw),
                            native_batch_owner=bool(raw.get("native_batch_owner", False)),
                        ),
                        raw_result=dict(cell_raw),
                    )
                )
            return results
        self.counters.batch_python_cell_loops += len(self.engines)
        return [
            engine.execute_generation(
                generation_index=generation_index,
                steps_per_generation=steps_per_generation,
                max_births=max_births,
            )
            for engine in self.engines
        ]
