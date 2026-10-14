"""Build, execute, and checkpoint the exact Jungle v4 resident CUDA overlay."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Final

import torch

from engine.native_runtime.cuda_architecture_b_backend_factory import (
    CudaArchitectureBRunnerEnvelope,
    derive_cuda_architecture_b_runner_envelope,
)
from engine.perf.cuda_architecture_b_direct_resident_execution_v2 import (
    create_run_control_v2,
)
from engine.perf.cuda_architecture_b_direct_resident_runtime_v1 import (
    resident_window_count_v1,
)
from engine.perf.cuda_architecture_b_direct_resident_gru_v3 import (
    _canonical_pred_lr_exact_cpu_v3,
    _materialize_disabled_prediction_state_v3,
)
from engine.perf.cuda_architecture_b_direct_resident_preparation_v1 import (
    DIRECT_RESIDENT_ARGUMENT_NAMES_V1,
    GENERATION_COUNT_ARGUMENT_INDEX_V1,
    PreparedDirectResidentTrajectoryV1,
    prepare_direct_resident_trajectory_v1,
)
from engine.perf.cuda_architecture_b_direct_resident_settlement_v1 import (
    _project_terminal_checkpoint,
    _with_materialized_event_history,
)
from engine.perf.cuda_architecture_b_direct_resident_source_allocator_v1 import (
    PHYSICAL_WINDOW_TICKS_V1,
    PHYSICAL_WINDOW_TURNOVER_EPISODES_V1,
    allocate_direct_resident_source_v1,
    required_physical_window_policy_episodes_v1,
)
from engine.perf.cuda_architecture_b_direct_resident_source_builder_v1 import (
    DirectResidentSourceAllocationRequestV1,
    QUALIFIED_EVENT_CAPACITY_V1,
    birth_state_capacity_overrides_v1,
)
from engine.perf.cuda_architecture_b_generation import CudaGenerationConfig
from engine.perf.cxxbackbone10_resident_state_owner import (
    _with_checkpoint_contract_metadata,
    build_import_projection,
    projection_digest,
)
from perf.cuda_architecture_b_boundary import import_cpu_owner_boundary_to_cuda
from perf.cuda_architecture_b_generation_birth_state import (
    prepare_generation_birth_state_cuda,
)


JUNGLE_ABI_VERSION_V4: Final[int] = 4
JUNGLE_TICKS_PER_GENERATION_V4: Final[int] = 80
JUNGLE_SUPPORTED_TICKS_PER_GENERATION_V4: Final[frozenset[int]] = frozenset(
    {80, 160}
)
JUNGLE_RECOVERY_STAT_COUNT_V4: Final[int] = 7
JUNGLE_TICK_INTEGER_METRIC_COUNT_V4: Final[int] = 9
JUNGLE_TICK_DOUBLE_METRIC_COUNT_V4: Final[int] = 7
JUNGLE_CAPABILITY_ENTRYPOINT_V4: Final[str] = (
    "eiel_cuda_architecture_b_flattened_jungle_capability_v4"
)
JUNGLE_PHYSICAL_ENTRYPOINT_V4: Final[str] = (
    "eiel_cuda_architecture_b_execute_direct_resident_jungle_physical_v4"
)
JUNGLE_DIRECT_ENTRYPOINT_V4: Final[str] = (
    "eiel_cuda_architecture_b_execute_direct_resident_jungle_v4"
)
JUNGLE_PHYSICAL_GENERATION_COUNTS_V4: Final[frozenset[int]] = frozenset({1, 2})
JUNGLE_RESIDENT_GENERATION_MIN_V4: Final[int] = 3
JUNGLE_RESIDENT_GENERATION_MAX_V4: Final[int] = 32
JUNGLE_RESIDENT_GENERATION_COUNTS_V4: Final[frozenset[int]] = frozenset(
    range(JUNGLE_RESIDENT_GENERATION_MIN_V4, JUNGLE_RESIDENT_GENERATION_MAX_V4 + 1)
)
JUNGLE_STATE_FIELDS_V4: Final[tuple[str, ...]] = (
    "patch_recovery_activity",
    "patch_visibility",
)
JUNGLE_CAPABILITIES_V4: Final[dict[str, bool]] = {
    "flat": False,
    "patch": True,
    "churn": True,
    "wdl": False,
    "jungle": True,
    "gru": False,
    "cuda": False,
}
_JUNGLE_ENVELOPE_FLAGS_V4: Final[tuple[str, ...]] = (
    "jungle_predator_pressure_enable",
    "jungle_reproduction_emissions_enable",
    "jungle_juvenile_predator_vulnerability_enable",
    "jungle_terrain_protection_enable",
    "jungle_ecological_recovery_enable",
)
_JUNGLE_BOOL_METADATA_FIELDS_V4: Final[tuple[str, ...]] = (
    *_JUNGLE_ENVELOPE_FLAGS_V4,
    "jungle_birth_emission_at_child_cell_enable",
    "jungle_birth_emission_clamp_enable",
    "jungle_terrain_protection_apply_to_damage_enable",
    "jungle_recovery_regrowth_boost_enable",
    "patch_density_include_self",
    "patch_density_normalize",
    "patch_visibility_from_cover_enable",
)
_JUNGLE_INT_METADATA_FIELDS_V4: Final[tuple[str, ...]] = (
    "jungle_juvenile_age_threshold",
    "patch_density_radius",
)
_JUNGLE_FLOAT_METADATA_FIELDS_V4: Final[tuple[str, ...]] = (
    "patch_scent_max",
    "patch_waste_max",
    "patch_visibility_min",
    "patch_visibility_max",
    "patch_visibility_default",
    "patch_fertility_min",
    "patch_fertility_max",
    "jungle_predator_base_risk",
    "jungle_predator_scent_weight",
    "jungle_predator_waste_weight",
    "jungle_predator_density_weight",
    "jungle_predator_visibility_weight",
    "jungle_predator_cover_protection_weight",
    "jungle_predator_shelter_protection_weight",
    "jungle_predator_risk_min",
    "jungle_predator_risk_cap",
    "jungle_predator_health_damage",
    "jungle_birth_scent_emit",
    "jungle_birth_waste_emit",
    "jungle_juvenile_predator_damage_multiplier",
    "jungle_juvenile_vulnerability_cap",
    "jungle_cover_damage_reduction_weight",
    "jungle_shelter_damage_reduction_weight",
    "jungle_terrain_damage_multiplier_min",
    "jungle_recovery_activity_min",
    "jungle_recovery_activity_max",
    "jungle_recovery_activity_decay_per_step",
    "jungle_recovery_activity_base_growth",
    "jungle_recovery_activity_cover_weight",
    "jungle_recovery_activity_fertility_weight",
    "jungle_recovery_activity_waste_penalty_weight",
    "jungle_recovery_activity_density_penalty_weight",
    "jungle_recovery_activity_scent_penalty_weight",
    "jungle_recovery_regrowth_boost_scale",
    "jungle_recovery_regrowth_boost_max",
)
PHYSICAL_RUNTIME_BOOKKEEPING_ARGUMENT_INDEX_V4: Final[int] = (
    DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index("physical_runtime_bookkeeping")
)
PHYSICAL_FULL_TICK_OUTPUTS_ARGUMENT_INDEX_V4: Final[int] = (
    DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index("physical_full_tick_output_tensors")
)
PHYSICAL_PATCH_STATS_ARGUMENT_INDEX_V4: Final[int] = (
    DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index("physical_patch_stats_tensors")
)
TICKS_ARGUMENT_INDEX_V4: Final[int] = DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index(
    "ticks_per_generation"
)


class CudaArchitectureBDirectResidentJungleV4Error(RuntimeError):
    """The exact Jungle resident source or execution boundary is unavailable."""


@dataclass(frozen=True, slots=True)
class PreparedDirectResidentJungleV4:
    patch: PreparedDirectResidentTrajectoryV1
    state_tensors: tuple[torch.Tensor, ...]
    reporting_tensors: tuple[torch.Tensor, ...]
    config: dict[str, Any]
    logical_generation_count: int
    initial_projection: Any


def _fail(message: str) -> CudaArchitectureBDirectResidentJungleV4Error:
    return CudaArchitectureBDirectResidentJungleV4Error(message)


def _validated_capability(module: Any) -> dict[str, Any]:
    entrypoint = getattr(module, JUNGLE_CAPABILITY_ENTRYPOINT_V4, None)
    if not callable(entrypoint):
        raise _fail(
            f"CUDA extension is missing callable {JUNGLE_CAPABILITY_ENTRYPOINT_V4}"
        )
    raw = entrypoint()
    if not isinstance(raw, Mapping):
        raise _fail("Jungle CUDA capability must be a mapping")
    capability = dict(raw)
    expected = {
        "abi_version": JUNGLE_ABI_VERSION_V4,
        "world": "patch_jungle",
        "resident_state": True,
        "exact_predator_pressure": True,
        "exact_birth_emissions": True,
        "exact_ecological_recovery": True,
        "native_tick_ownership": True,
        "runtime_jit_required": False,
        "production_qualified": False,
    }
    for name, expected_value in expected.items():
        actual = capability.get(name)
        if type(actual) is not type(expected_value) or actual != expected_value:
            raise _fail(
                f"Jungle CUDA capability {name} mismatch: "
                f"expected={expected_value!r} actual={actual!r}"
            )
    return capability


def _patch_envelope_with_jungle_masked(
    runner: Any,
) -> CudaArchitectureBRunnerEnvelope:
    proxy = copy.copy(runner)
    proxy.env_cfg = copy.copy(runner.env_cfg)
    for name in _JUNGLE_ENVELOPE_FLAGS_V4:
        setattr(proxy.env_cfg, name, False)
    return derive_cuda_architecture_b_runner_envelope(proxy)


def _required_jungle_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    exact = {
        "native_runtime_jungle_execution_enable": True,
        "native_runtime_fixed_policy_family": "mlp",
        "native_runtime_allow_runtime_jit_build": False,
        "native_runtime_patch_profile": "P2",
    }
    for name, expected in exact.items():
        actual = metadata.get(name)
        if type(actual) is not type(expected) or actual != expected:
            raise _fail(
                f"Jungle import metadata {name} mismatch: "
                f"expected={expected!r} actual={actual!r}"
            )
    missing = [
        name
        for name in (
            *_JUNGLE_BOOL_METADATA_FIELDS_V4,
            *_JUNGLE_INT_METADATA_FIELDS_V4,
            *_JUNGLE_FLOAT_METADATA_FIELDS_V4,
        )
        if name not in metadata
    ]
    if missing:
        raise _fail(f"Jungle import metadata is missing: {missing}")
    for name in _JUNGLE_BOOL_METADATA_FIELDS_V4:
        if type(metadata[name]) is not bool:
            raise _fail(f"Jungle import metadata {name} must be an exact bool")
    for name in _JUNGLE_INT_METADATA_FIELDS_V4:
        if type(metadata[name]) is not int:
            raise _fail(f"Jungle import metadata {name} must be an exact integer")
    for name in _JUNGLE_FLOAT_METADATA_FIELDS_V4:
        value = metadata[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise _fail(f"Jungle import metadata {name} must be finite numeric")
    if not any(bool(metadata[name]) for name in _JUNGLE_ENVELOPE_FLAGS_V4):
        raise _fail("Jungle import metadata has no active accepted mechanic")
    return dict(metadata)


def _clamp(value: Any, lower: float, upper: float | None = None) -> float:
    resolved = max(lower, float(value))
    if upper is not None:
        resolved = min(upper, resolved)
    return resolved


def _ordered(left: Any, right: Any) -> tuple[float, float]:
    first, second = float(left), float(right)
    return (first, second) if first <= second else (second, first)


def _normalized_jungle_config(
    metadata: Mapping[str, Any],
    *,
    ticks_per_generation: int = JUNGLE_TICKS_PER_GENERATION_V4,
) -> dict[str, Any]:
    if ticks_per_generation not in JUNGLE_SUPPORTED_TICKS_PER_GENERATION_V4:
        raise _fail("Jungle ticks_per_generation must be 80 or 160")
    visibility_min, visibility_max = _ordered(
        metadata["patch_visibility_min"], metadata["patch_visibility_max"]
    )
    recovery_min, recovery_max = _ordered(
        metadata["jungle_recovery_activity_min"],
        metadata["jungle_recovery_activity_max"],
    )
    fertility_min, fertility_max = _ordered(
        metadata["patch_fertility_min"], metadata["patch_fertility_max"]
    )
    risk_min = _clamp(metadata["jungle_predator_risk_min"], 0.0, 1.0)
    risk_cap = _clamp(metadata["jungle_predator_risk_cap"], 0.0, 1.0)
    risk_min, risk_cap = _ordered(risk_min, risk_cap)
    radius = max(0, int(metadata["patch_density_radius"]))
    density_denominator = float((2 * radius + 1) ** 2)
    if not metadata["patch_density_include_self"]:
        density_denominator -= 1.0
    # Radius zero without self has an identically zero density map on CPU.
    density_denominator = max(1.0, density_denominator)
    juvenile_multiplier = min(
        _clamp(metadata["jungle_juvenile_predator_damage_multiplier"], 1.0),
        _clamp(metadata["jungle_juvenile_vulnerability_cap"], 1.0),
    )
    return {
        "enabled": True,
        "ticks_per_generation": ticks_per_generation,
        "generation_base": int(metadata["generation"]),
        "predator_pressure_enabled": bool(metadata["jungle_predator_pressure_enable"]),
        "juvenile_vulnerability_enabled": bool(
            metadata["jungle_juvenile_predator_vulnerability_enable"]
        ),
        "terrain_protection_enabled": bool(
            metadata["jungle_terrain_protection_enable"]
        ),
        "terrain_protection_apply_to_damage_enabled": bool(
            metadata["jungle_terrain_protection_apply_to_damage_enable"]
        ),
        "birth_emissions_enabled": bool(
            metadata["jungle_reproduction_emissions_enable"]
            and metadata["jungle_birth_emission_at_child_cell_enable"]
        ),
        "birth_emission_clamp_enabled": bool(
            metadata["jungle_birth_emission_clamp_enable"]
        ),
        "ecological_recovery_enabled": bool(
            metadata["jungle_ecological_recovery_enable"]
        ),
        "visibility_from_cover_enabled": bool(
            metadata["patch_visibility_from_cover_enable"]
        ),
        "recovery_regrowth_boost_enabled": bool(
            metadata["jungle_recovery_regrowth_boost_enable"]
        ),
        "juvenile_age_threshold": max(
            0, int(metadata["jungle_juvenile_age_threshold"])
        ),
        "scent_max": _clamp(metadata["patch_scent_max"], 0.0),
        "waste_max": _clamp(metadata["patch_waste_max"], 0.0),
        "density_denominator": density_denominator,
        "visibility_min": visibility_min,
        "visibility_max": visibility_max,
        "visibility_default": _clamp(
            metadata["patch_visibility_default"],
            visibility_min,
            visibility_max,
        ),
        "predator_base_risk": _clamp(metadata["jungle_predator_base_risk"], 0.0),
        "predator_scent_weight": _clamp(metadata["jungle_predator_scent_weight"], 0.0),
        "predator_waste_weight": _clamp(metadata["jungle_predator_waste_weight"], 0.0),
        "predator_density_weight": _clamp(
            metadata["jungle_predator_density_weight"], 0.0
        ),
        "predator_visibility_weight": _clamp(
            metadata["jungle_predator_visibility_weight"], 0.0
        ),
        "predator_cover_protection_weight": _clamp(
            metadata["jungle_predator_cover_protection_weight"], 0.0
        ),
        "predator_shelter_protection_weight": _clamp(
            metadata["jungle_predator_shelter_protection_weight"], 0.0
        ),
        "predator_risk_min": risk_min,
        "predator_risk_cap": risk_cap,
        "predator_health_damage": _clamp(
            metadata["jungle_predator_health_damage"], 0.0
        ),
        "juvenile_damage_multiplier": juvenile_multiplier,
        "cover_damage_reduction_weight": _clamp(
            metadata["jungle_cover_damage_reduction_weight"], 0.0
        ),
        "shelter_damage_reduction_weight": _clamp(
            metadata["jungle_shelter_damage_reduction_weight"], 0.0
        ),
        "terrain_damage_multiplier_min": _clamp(
            metadata["jungle_terrain_damage_multiplier_min"], 0.0, 1.0
        ),
        "birth_scent_emit": _clamp(metadata["jungle_birth_scent_emit"], 0.0),
        "birth_waste_emit": _clamp(metadata["jungle_birth_waste_emit"], 0.0),
        "recovery_activity_min": recovery_min,
        "recovery_activity_max": recovery_max,
        "fertility_min": fertility_min,
        "fertility_max": fertility_max,
        "recovery_decay_factor": 1.0
        - _clamp(metadata["jungle_recovery_activity_decay_per_step"], 0.0, 1.0),
        "recovery_base_growth": float(metadata["jungle_recovery_activity_base_growth"]),
        "recovery_cover_weight": _clamp(
            metadata["jungle_recovery_activity_cover_weight"], 0.0
        ),
        "recovery_fertility_weight": _clamp(
            metadata["jungle_recovery_activity_fertility_weight"], 0.0
        ),
        "recovery_waste_penalty_weight": _clamp(
            metadata["jungle_recovery_activity_waste_penalty_weight"], 0.0
        ),
        "recovery_density_penalty_weight": _clamp(
            metadata["jungle_recovery_activity_density_penalty_weight"], 0.0
        ),
        "recovery_scent_penalty_weight": _clamp(
            metadata["jungle_recovery_activity_scent_penalty_weight"], 0.0
        ),
        "recovery_regrowth_boost_scale": _clamp(
            metadata["jungle_recovery_regrowth_boost_scale"], 0.0
        ),
        "recovery_regrowth_boost_max": _clamp(
            metadata["jungle_recovery_regrowth_boost_max"], 0.0
        ),
    }


def _allocate_reporting(
    *,
    device: torch.device,
    cells: int,
    height: int,
    width: int,
    capacity: int,
    logical_generation_count: int,
    ticks_per_generation: int = JUNGLE_TICKS_PER_GENERATION_V4,
) -> tuple[torch.Tensor, ...]:
    if ticks_per_generation not in JUNGLE_SUPPORTED_TICKS_PER_GENERATION_V4:
        raise _fail("Jungle reporting ticks_per_generation must be 80 or 160")
    ticks = logical_generation_count * ticks_per_generation
    return (
        torch.empty((cells, 2, height, width), dtype=torch.float32, device=device),
        torch.zeros(
            (cells, ticks, JUNGLE_RECOVERY_STAT_COUNT_V4),
            dtype=torch.float64,
            device=device,
        ),
        torch.zeros(
            (cells, ticks, JUNGLE_TICK_INTEGER_METRIC_COUNT_V4),
            dtype=torch.int64,
            device=device,
        ),
        torch.zeros(
            (cells, ticks, JUNGLE_TICK_DOUBLE_METRIC_COUNT_V4),
            dtype=torch.float64,
            device=device,
        ),
        torch.zeros((cells, ticks, capacity), dtype=torch.float32, device=device),
    )


def build_prepared_direct_resident_jungle_v4(
    runner: Any,
    *,
    qualified_cuda_module: Any,
    logical_generation_count: int = 4,
    initial_generation: int = 0,
    source_checkpoint_projection: Mapping[str, Any] | None = None,
    cuda_device: str = "cuda",
) -> PreparedDirectResidentJungleV4:
    """Build one fixed-MLP Jungle overlay around the Patch resident ABI."""

    supported = (
        JUNGLE_PHYSICAL_GENERATION_COUNTS_V4 | JUNGLE_RESIDENT_GENERATION_COUNTS_V4
    )
    if (
        type(logical_generation_count) is not int
        or logical_generation_count not in supported
    ):
        raise _fail("Jungle preparation requires logical_generation_count in [1,32]")
    if type(initial_generation) is not int or initial_generation < 0:
        raise _fail("Jungle initial_generation must be a nonnegative exact integer")
    if cuda_device not in {"cuda", "cuda:0"}:
        raise _fail("Jungle CUDA source requires cuda or cuda:0")
    capability = _validated_capability(qualified_cuda_module)
    envelope = _patch_envelope_with_jungle_masked(runner)
    fresh_projection = build_import_projection(
        runner,
        dict(JUNGLE_CAPABILITIES_V4),
        generation=initial_generation,
        tick=0,
    )
    fresh_projection.metadata[
        "native_runtime_architecture_b_compact_reporting_only_v1"
    ] = False
    fresh_projection.metadata["import_digest"] = projection_digest(
        {"metadata": fresh_projection.metadata, "tensors": fresh_projection.tensors}
    )
    projection: Any = (
        fresh_projection
        if source_checkpoint_projection is None
        else source_checkpoint_projection
    )
    projection_metadata = (
        projection.get("metadata")
        if isinstance(projection, Mapping)
        else getattr(projection, "metadata", None)
    )
    if not isinstance(projection_metadata, Mapping):
        raise _fail("Jungle source projection is missing metadata")
    if (
        projection_metadata.get("generation") != initial_generation
        or projection_metadata.get("tick") != 0
        or projection_metadata.get("native_runtime_fixed_policy_family") != "mlp"
    ):
        raise _fail("Jungle source checkpoint is not the requested fixed-MLP boundary")
    metadata = _required_jungle_metadata(projection_metadata)
    boundary = import_cpu_owner_boundary_to_cuda(
        projection,
        event_capacity=QUALIFIED_EVENT_CAPACITY_V1,
        device=cuda_device,
    )
    max_births = int(envelope.max_births_per_tick)
    ordinary_births_enabled = getattr(
        runner.sim_cfg, "ordinary_births_enabled", None
    )
    if ordinary_births_enabled is None:
        ordinary_births_enabled = max_births > 0
    elif type(ordinary_births_enabled) is not bool:
        raise _fail("runner ordinary_births_enabled must be an exact bool")
    policy_episode_capacity = required_physical_window_policy_episodes_v1(max_births)
    birth_state_kwargs = {
        "steps": PHYSICAL_WINDOW_TICKS_V1,
        "max_births_per_tick": max_births,
        "turnover_refill_capacity": PHYSICAL_WINDOW_TURNOVER_EPISODES_V1,
    }
    birth_state_kwargs.update(
        birth_state_capacity_overrides_v1(
            runner.sim_cfg,
            ordinary_births_enabled=ordinary_births_enabled,
            resident_policy_episode_capacity=policy_episode_capacity,
            turnover_refill_capacity=PHYSICAL_WINDOW_TURNOVER_EPISODES_V1,
        )
    )
    birth_state = prepare_generation_birth_state_cuda(
        boundary,
        projection,
        **birth_state_kwargs,
    )
    configuration = replace(
        CudaGenerationConfig.from_metadata(
            metadata,
            steps=JUNGLE_TICKS_PER_GENERATION_V4,
        ),
        generation=initial_generation,
        start_tick=0,
        max_births_per_tick=max_births,
        ordinary_births_enabled=ordinary_births_enabled,
        generation_turnover_enabled=True,
    )
    request = DirectResidentSourceAllocationRequestV1(
        envelope=envelope,
        qualified_cuda_module=qualified_cuda_module,
        cuda_capability=capability,
        import_projection=projection,
        cuda_boundary=boundary,
        generation_birth_state=birth_state,
        configuration=configuration,
        cuda_device=cuda_device,
        logical_generation_count=max(3, logical_generation_count),
    )
    source = allocate_direct_resident_source_v1(request)
    if metadata["native_runtime_fixed_policy_family"] != "mlp":
        raise _fail("Jungle resident source must use the fixed MLP policy family")

    tensors = boundary.tensors
    missing_state = [name for name in JUNGLE_STATE_FIELDS_V4 if name not in tensors]
    if missing_state:
        raise _fail(f"Jungle CUDA boundary is missing state tensors: {missing_state}")
    raw_state = tuple(tensors[name] for name in JUNGLE_STATE_FIELDS_V4)
    recovery, visibility = raw_state
    if (
        not isinstance(recovery, torch.Tensor)
        or not isinstance(visibility, torch.Tensor)
        or recovery.dtype != torch.float32
        or visibility.dtype != torch.float32
        or recovery.ndim != 2
        or visibility.shape != recovery.shape
    ):
        raise _fail("Jungle state tensors must be matching float32[H,W] arrays")
    state_tensors = tuple(tensor.unsqueeze(0).contiguous() for tensor in raw_state)
    device = source.live_state["energy"].device
    cells = int(source.live_state["energy"].size(0))
    capacity = int(source.live_state["energy"].size(1))
    if cells != 1 or state_tensors[0].device != device:
        raise _fail("Jungle source and imported state must share one b1 device")
    height, width = int(recovery.size(0)), int(recovery.size(1))
    reporting_tensors = _allocate_reporting(
        device=device,
        cells=cells,
        height=height,
        width=width,
        capacity=capacity,
        logical_generation_count=logical_generation_count,
    )
    return PreparedDirectResidentJungleV4(
        patch=prepare_direct_resident_trajectory_v1(source),
        state_tensors=state_tensors,
        reporting_tensors=reporting_tensors,
        config=_normalized_jungle_config(metadata),
        logical_generation_count=logical_generation_count,
        initial_projection=projection,
    )


def _validated_result(
    raw: Any,
    expected: Mapping[str, Any],
    *,
    surface: str,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise _fail(f"Jungle {surface} CUDA entrypoint returned a non-mapping result")
    result = dict(raw)
    for name, expected_value in expected.items():
        actual = result.get(name)
        if type(actual) is not type(expected_value) or actual != expected_value:
            raise _fail(
                f"Jungle {surface} result {name} mismatch: "
                f"expected={expected_value!r} actual={actual!r}"
            )
    return result


def execute_direct_resident_jungle_physical_v4(
    module: Any,
    prepared: PreparedDirectResidentJungleV4,
    *,
    generation_count: int,
    ticks_per_generation: int = JUNGLE_TICKS_PER_GENERATION_V4,
) -> dict[str, Any]:
    """Execute one native-owned physical g1 or g2 Jungle segment."""

    if not isinstance(prepared, PreparedDirectResidentJungleV4):
        raise _fail("prepared must be PreparedDirectResidentJungleV4")
    if (
        type(generation_count) is not int
        or generation_count not in JUNGLE_PHYSICAL_GENERATION_COUNTS_V4
    ):
        raise _fail("physical Jungle qualification requires generation_count 1 or 2")
    if generation_count > prepared.logical_generation_count:
        raise _fail("Jungle preparation does not cover the requested physical segment")
    if (
        type(ticks_per_generation) is not int
        or ticks_per_generation < 1
        or ticks_per_generation > JUNGLE_TICKS_PER_GENERATION_V4
    ):
        raise _fail("physical Jungle ticks_per_generation must be in [1,80]")
    entrypoint = getattr(module, JUNGLE_PHYSICAL_ENTRYPOINT_V4, None)
    if not callable(entrypoint):
        raise _fail(
            f"CUDA extension is missing callable {JUNGLE_PHYSICAL_ENTRYPOINT_V4}"
        )
    arguments = list(prepared.patch.arguments)
    physical_ticks = generation_count * ticks_per_generation
    arguments[PHYSICAL_RUNTIME_BOOKKEEPING_ARGUMENT_INDEX_V4] = arguments[
        PHYSICAL_RUNTIME_BOOKKEEPING_ARGUMENT_INDEX_V4
    ][:, :physical_ticks]
    full_tick_outputs = list(arguments[PHYSICAL_FULL_TICK_OUTPUTS_ARGUMENT_INDEX_V4])
    full_tick_outputs[-1] = full_tick_outputs[-1][:generation_count]
    arguments[PHYSICAL_FULL_TICK_OUTPUTS_ARGUMENT_INDEX_V4] = full_tick_outputs
    patch_stats = list(arguments[PHYSICAL_PATCH_STATS_ARGUMENT_INDEX_V4])
    patch_stats[0] = patch_stats[0][:, :physical_ticks]
    arguments[PHYSICAL_PATCH_STATS_ARGUMENT_INDEX_V4] = patch_stats
    arguments[TICKS_ARGUMENT_INDEX_V4] = ticks_per_generation
    arguments[GENERATION_COUNT_ARGUMENT_INDEX_V1] = generation_count
    reporting_tensors = [prepared.reporting_tensors[0]]
    reporting_tensors.extend(
        tensor[:, :physical_ticks] for tensor in prepared.reporting_tensors[1:]
    )
    config = dict(prepared.config)
    config["ticks_per_generation"] = ticks_per_generation
    raw = entrypoint(
        list(prepared.state_tensors),
        reporting_tensors,
        config,
        *arguments,
    )
    return _validated_result(
        raw,
        {
            "jungle_native_owner": True,
            "jungle_native_controller_abi_version": JUNGLE_ABI_VERSION_V4,
            "jungle_physical_gate_entrypoint": True,
            "jungle_python_tick_or_generation_ownership": False,
            "hidden_fallback": False,
            "runtime_jit": False,
        },
        surface="physical",
    )


def execute_direct_resident_jungle_v4(
    module: Any,
    prepared: PreparedDirectResidentJungleV4,
    *,
    logical_generation_count: int,
    run_id: str,
) -> dict[str, Any]:
    """Execute one complete Jungle trajectory in one Python dispatch."""

    if not isinstance(prepared, PreparedDirectResidentJungleV4):
        raise _fail("prepared must be PreparedDirectResidentJungleV4")
    if (
        type(logical_generation_count) is not int
        or not JUNGLE_RESIDENT_GENERATION_MIN_V4
        <= logical_generation_count
        <= JUNGLE_RESIDENT_GENERATION_MAX_V4
    ):
        raise _fail("resident Jungle requires logical_generation_count in [3,32]")
    if prepared.logical_generation_count != logical_generation_count:
        raise _fail(
            "Jungle preparation horizon does not match the requested trajectory"
        )
    if prepared.patch.source.logical_generation_count != logical_generation_count:
        raise _fail(
            "Jungle source identity capacity does not match the requested horizon"
        )
    entrypoint = getattr(module, JUNGLE_DIRECT_ENTRYPOINT_V4, None)
    if not callable(entrypoint):
        raise _fail(f"CUDA extension is missing callable {JUNGLE_DIRECT_ENTRYPOINT_V4}")
    control = create_run_control_v2(module, run_id=run_id)
    metadata_value = (
        prepared.initial_projection.get("metadata")
        if isinstance(prepared.initial_projection, Mapping)
        else getattr(prepared.initial_projection, "metadata", None)
    )
    if not isinstance(metadata_value, Mapping):
        raise _fail("Jungle initial projection metadata is not a mapping")
    metadata = dict(metadata_value)
    raw = entrypoint(
        control,
        list(prepared.state_tensors),
        list(prepared.reporting_tensors),
        dict(prepared.config),
        *prepared.patch.arguments,
        logical_generation_count,
        metadata.get("policy_weight_mutation_enable"),
        metadata.get("policy_weight_mutation_mode"),
        metadata.get("policy_weight_mutation_rate"),
        metadata.get("policy_weight_mutation_std"),
    )
    result = _validated_result(
        raw,
        {
            "jungle_native_owner": True,
            "jungle_native_controller_abi_version": JUNGLE_ABI_VERSION_V4,
            "jungle_python_tick_or_generation_ownership": False,
            "completed_generations": logical_generation_count,
            "logical_generation_count": logical_generation_count,
            "window_count": resident_window_count_v1(
                initial_generation=prepared.patch.source.generation_base,
                logical_generation_count=logical_generation_count,
            ),
            "physical_window_alignment": "absolute_generation_lattice_v1",
            "external_python_window_boundaries": 0,
            "direct_cpp_window_core_dispatch": True,
            "production_native_chunk_loop": True,
            "hidden_fallback": False,
            "runtime_jit": False,
        },
        surface="resident",
    )
    manifest = dict(control.manifest())
    if (
        manifest.get("status") != "generation_complete"
        or manifest.get("requested_generations") != logical_generation_count
        or manifest.get("active_generation")
        != prepared.patch.source.generation_base + logical_generation_count - 1
        or manifest.get("awaiting_final_output_ack") is not True
        or manifest.get("sealed") is not False
    ):
        raise _fail("Jungle resident run-control manifest is not at the final boundary")
    result["native_run_control_v2"] = manifest
    return result


def project_direct_resident_jungle_checkpoint_v4(
    prepared: PreparedDirectResidentJungleV4,
    *,
    direct_result: Mapping[str, Any],
    completed_generation_count: int,
) -> Mapping[str, Any]:
    """Project one complete Jungle prefix through the Patch checkpoint path."""

    if not isinstance(prepared, PreparedDirectResidentJungleV4):
        raise _fail("prepared must be PreparedDirectResidentJungleV4")
    if (
        type(completed_generation_count) is not int
        or completed_generation_count < 1
        or completed_generation_count > prepared.logical_generation_count
    ):
        raise _fail("Jungle checkpoint projection exceeds its resident horizon")
    result = dict(direct_result)
    outputs = dict(prepared.patch.source.outputs)
    generation_summary = result.get("generation_summary_scalar")
    if not isinstance(generation_summary, torch.Tensor):
        generation_summary = prepared.patch.arguments[
            PHYSICAL_FULL_TICK_OUTPUTS_ARGUMENT_INDEX_V4
        ][-1][:completed_generation_count]
    runtime_bookkeeping = result.get("runtime_bookkeeping_by_tick")
    if not isinstance(runtime_bookkeeping, torch.Tensor):
        runtime_bookkeeping = prepared.patch.arguments[
            PHYSICAL_RUNTIME_BOOKKEEPING_ARGUMENT_INDEX_V4
        ][:, : completed_generation_count * JUNGLE_TICKS_PER_GENERATION_V4]
    outputs["generation_summary_scalar"] = generation_summary
    outputs["runtime_bookkeeping_by_tick"] = runtime_bookkeeping
    persistent_state = _materialize_disabled_prediction_state_v3(
        prepared,
        _with_materialized_event_history(prepared.patch.live_banks, result),
    )
    for name, tensor in zip(
        JUNGLE_STATE_FIELDS_V4,
        prepared.state_tensors,
        strict=True,
    ):
        persistent_state[name] = tensor
    projection_inputs = prepared.patch.checkpoint_projection_inputs
    checkpoint = _project_terminal_checkpoint(
        projection_inputs["pre_generation_canonical_projection"],
        projection_inputs["resident_birth_state"],
        outputs,
        persistent_state,
        completed_generation_count=completed_generation_count,
        source_generation=prepared.patch.source.generation_base,
    )
    projection = dict(checkpoint.projection)
    metadata = dict(projection["metadata"])
    native_tick_iterations = result.get("native_tick_iterations")
    if type(native_tick_iterations) is not int or native_tick_iterations < 1:
        native_tick_iterations = (
            completed_generation_count * JUNGLE_TICKS_PER_GENERATION_V4
        )
    if native_tick_iterations % completed_generation_count != 0:
        raise _fail("Jungle native tick count does not cover complete generations")
    metadata["prediction_tick"] = (
        native_tick_iterations // completed_generation_count
    )
    projection["metadata"] = metadata
    tensors = dict(projection["tensors"])
    tensors["pred_lr_exact"] = _canonical_pred_lr_exact_cpu_v3(
        tensors["pred_lr"],
        tensors["alive_mask"],
    )
    projection["tensors"] = tensors
    return _with_checkpoint_contract_metadata(projection)


__all__ = [
    "JUNGLE_CAPABILITY_ENTRYPOINT_V4",
    "JUNGLE_DIRECT_ENTRYPOINT_V4",
    "JUNGLE_PHYSICAL_ENTRYPOINT_V4",
    "JUNGLE_RESIDENT_GENERATION_MAX_V4",
    "JUNGLE_RESIDENT_GENERATION_MIN_V4",
    "JUNGLE_STATE_FIELDS_V4",
    "CudaArchitectureBDirectResidentJungleV4Error",
    "PreparedDirectResidentJungleV4",
    "build_prepared_direct_resident_jungle_v4",
    "execute_direct_resident_jungle_physical_v4",
    "execute_direct_resident_jungle_v4",
    "project_direct_resident_jungle_checkpoint_v4",
]
