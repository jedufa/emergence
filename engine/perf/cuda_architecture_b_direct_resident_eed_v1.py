"""Build, execute, and checkpoint the candidate resident EED CUDA overlay."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
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
from engine.perf.cuda_architecture_b_direct_resident_preparation_v1 import (
    DIRECT_RESIDENT_ARGUMENT_NAMES_V1,
    GENERATION_COUNT_ARGUMENT_INDEX_V1,
    PreparedDirectResidentTrajectoryV1,
    prepare_direct_resident_trajectory_v1,
)
from engine.perf.cuda_architecture_b_direct_resident_runtime_v1 import (
    resident_window_count_v1,
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
    MAX_EVENT_ROWS_PER_ORDINARY_BIRTH_V1,
    QUALIFIED_EVENT_CAPACITY_V1,
    BASE_EVENT_ROWS_PER_SOURCE_WINDOW_V1,
    _life_history_config_from_sim_cfg_v1,
    _resident_event_capacity_generation_span_v1,
    _stage_request_tensors_to_cpu_v1,
    _validate_life_history_metadata_v1,
    birth_state_capacity_overrides_v1,
)
from engine.perf.cuda_architecture_b_disabled_prediction_state_v1 import (
    materialize_disabled_prediction_state_v1,
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


EED_ABI_VERSION_V1: Final[int] = 2
EED_OBSERVATION_SIZE_V1: Final[int] = 74
EED_TICKS_PER_GENERATION_V1: Final[int] = 80
EED_ADMITTED_TICKS_PER_GENERATION_V1: Final[tuple[int, ...]] = (80, 160)
EED_CAPABILITY_ENTRYPOINT_V1: Final[str] = (
    "eiel_cuda_architecture_b_flattened_eed_capability_v1"
)
EED_PHYSICAL_ENTRYPOINT_V1: Final[str] = (
    "eiel_cuda_architecture_b_execute_direct_resident_eed_physical_v1"
)
EED_DIRECT_ENTRYPOINT_V1: Final[str] = (
    "eiel_cuda_architecture_b_execute_direct_resident_eed_v1"
)
EED_RESIDENT_GENERATION_MIN_V1: Final[int] = 3
EED_RESIDENT_GENERATION_MAX_V1: Final[int] = 128
EED_GRID_STATE_FIELDS_V1: Final[tuple[str, ...]] = (
    "eed_resource_a_amount",
    "eed_resource_b_amount",
    "eed_resource_a_capacity",
    "eed_resource_b_capacity",
    "eed_resource_a_renewal",
    "eed_resource_b_renewal",
    "eed_resource_a_pending_impact",
    "eed_resource_b_pending_impact",
    "eed_resource_a_impact_load",
    "eed_resource_b_impact_load",
    "respec0b_resource_class",
)
EED_CURSOR_STATE_FIELDS_V1: Final[tuple[str, ...]] = (
    "respec0b_resource_active_y",
    "respec0b_resource_active_x",
    "respec0b_resource_active_class",
    "respec0b_resource_handling_progress",
)
EED_STATE_FIELDS_V1: Final[tuple[str, ...]] = (
    *EED_GRID_STATE_FIELDS_V1,
    *EED_CURSOR_STATE_FIELDS_V1,
)
EED_CAPABILITIES_V1: Final[dict[str, bool]] = {
    "flat": False,
    "patch": True,
    "churn": True,
    "wdl": False,
    "jungle": False,
    "gru": False,
    "cuda": False,
    "eed": True,
    "cross_path_overlay": True,
    "genome_v2": False,
}
PHYSICAL_RUNTIME_BOOKKEEPING_ARGUMENT_INDEX_V1: Final[int] = (
    DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index("physical_runtime_bookkeeping")
)
PHYSICAL_FULL_TICK_OUTPUTS_ARGUMENT_INDEX_V1: Final[int] = (
    DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index("physical_full_tick_output_tensors")
)
PHYSICAL_PATCH_STATS_ARGUMENT_INDEX_V1: Final[int] = (
    DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index("physical_patch_stats_tensors")
)
TICKS_ARGUMENT_INDEX_V1: Final[int] = DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index(
    "ticks_per_generation"
)


class CudaArchitectureBDirectResidentEedV1Error(RuntimeError):
    """The exact EED resident source or execution boundary is unavailable."""


@dataclass(frozen=True, slots=True)
class PreparedDirectResidentEedV1:
    patch: PreparedDirectResidentTrajectoryV1
    state_tensors: tuple[torch.Tensor, ...]
    config: dict[str, Any]
    initial_projection: Any
    cell_id: str
    cell_ids: tuple[str, ...] = ()
    source_projections: tuple[Any, ...] = ()


def _fail(message: str) -> CudaArchitectureBDirectResidentEedV1Error:
    return CudaArchitectureBDirectResidentEedV1Error(message)


def _eed_ticks_per_generation_v1(value: Any, *, name: str) -> int:
    if (
        type(value) is not int
        or value not in EED_ADMITTED_TICKS_PER_GENERATION_V1
    ):
        raise _fail(
            f"{name} must be one of "
            f"{list(EED_ADMITTED_TICKS_PER_GENERATION_V1)!r}"
        )
    return value


def _normalized_eed_source_projection_v1(projection: Any) -> Any:
    if isinstance(projection, Mapping):
        out = dict(projection)
    else:
        metadata = getattr(projection, "metadata", None)
        tensors = getattr(projection, "tensors", None)
        if not (isinstance(metadata, Mapping) and isinstance(tensors, Mapping)):
            return projection
        out = {
            "metadata": metadata,
            "tensors": tensors,
        }
        for name, child in vars(projection).items():
            if name not in out:
                out[name] = child
    metadata = out.get("metadata")
    if not isinstance(metadata, Mapping):
        return out
    active_event_count = metadata.get(
        "typed_lifecycle_event_active_count",
        metadata.get(
            "event_tape_cursor",
            metadata.get("event_tape_active_count", 0),
        ),
    )
    if type(active_event_count) is not int or active_event_count < 0:
        raise _fail("EED source projection has invalid lifecycle event count")
    event_cursor = out.get("event_cursor", active_event_count)
    runtime_event_cursor = out.get("runtime_event_cursor", event_cursor)
    if (
        type(event_cursor) is not int
        or event_cursor < 0
        or type(runtime_event_cursor) is not int
        or runtime_event_cursor < event_cursor
    ):
        raise _fail("EED source projection has invalid event cursors")
    out.setdefault("event_cursor", event_cursor)
    out.setdefault("runtime_event_cursor", runtime_event_cursor)
    if metadata.get("generation") == 0 and metadata.get("tick") == 0:
        out.setdefault("restore_count", 0)
        out.setdefault("generation_calls", 0)
        out.setdefault("last_applied_actions", 0)
        out.setdefault("last_moved", 0)
        out.setdefault("last_gathered", 0)
        out.setdefault("last_policy_refill_calls", 0)
        out.setdefault("last_policy_refill_rows", 0)
        out.setdefault("pending_generation_free_agent_ids", [])
        out.setdefault("pending_generation_free_slots", [])
        out.setdefault("last_world", "none")
    return out


def _eed_source_event_capacity_v1(
    projection: Any,
    metadata: Mapping[str, Any],
    *,
    logical_generation_count: int,
    max_births_per_tick: int,
    ticks_per_generation: int,
) -> int:
    source_generation = metadata.get("generation")
    if type(source_generation) is not int or source_generation < 0:
        raise _fail("EED source projection generation must be nonnegative")
    if type(max_births_per_tick) is not int or max_births_per_tick < 1:
        raise _fail("EED max_births_per_tick must be a positive exact integer")
    ticks_per_generation = _eed_ticks_per_generation_v1(
        ticks_per_generation,
        name="EED event capacity ticks_per_generation",
    )
    values: list[int] = []
    for field in (
        "event_tape_active_count",
        "typed_lifecycle_event_active_count",
        "event_tape_cursor",
        "typed_lifecycle_event_next_sequence",
    ):
        value = metadata.get(field)
        if value is None:
            continue
        if type(value) is not int or value < 0:
            raise _fail(f"EED source projection {field} must be nonnegative")
        values.append(value)
    if isinstance(projection, Mapping):
        event_cursor = projection.get("event_cursor")
    else:
        event_cursor = getattr(projection, "event_cursor", None)
    if event_cursor is not None:
        if type(event_cursor) is not int or event_cursor < 0:
            raise _fail("EED source projection event_cursor must be nonnegative")
        values.append(event_cursor)
    active_prefix = max(values, default=0)
    additional_births = (max_births_per_tick - 1) * ticks_per_generation
    event_rows_per_generation = (
        BASE_EVENT_ROWS_PER_SOURCE_WINDOW_V1
        + additional_births * MAX_EVENT_ROWS_PER_ORDINARY_BIRTH_V1
    )
    resident_event_generation_span = _resident_event_capacity_generation_span_v1(
        source_generation=source_generation,
        logical_generation_count=logical_generation_count,
    )
    admitted_window_rows = resident_event_generation_span * event_rows_per_generation
    return active_prefix + max(QUALIFIED_EVENT_CAPACITY_V1, admitted_window_rows)


def _eed_source_capabilities_v1(
    source_capability_mask: Mapping[str, bool] | None,
) -> dict[str, bool]:
    capabilities = dict(EED_CAPABILITIES_V1)
    if source_capability_mask is None:
        return capabilities
    if not isinstance(source_capability_mask, Mapping):
        raise _fail("EED source capability mask must be a mapping")
    allowed = set(EED_CAPABILITIES_V1)
    unknown = sorted(set(source_capability_mask) - allowed)
    if unknown:
        raise _fail(
            f"EED source capability mask has unsupported keys: {unknown}"
        )
    for name, value in source_capability_mask.items():
        if type(value) is not bool:
            raise _fail(
                f"EED source capability {name} must be an exact boolean"
            )
        if (
            name in EED_CAPABILITIES_V1
            and name not in {"cross_path_overlay", "genome_v2"}
            and value != EED_CAPABILITIES_V1[name]
        ):
            raise _fail(
                f"EED source capability {name} contradicts the EED route"
            )
        capabilities[name] = value
    if capabilities["cross_path_overlay"] == capabilities["genome_v2"]:
        raise _fail(
            "EED requires exactly one diploid authority: "
            "cross_path_overlay or genome_v2"
        )
    return capabilities


def _validated_capability(module: Any) -> dict[str, Any]:
    entrypoint = getattr(module, EED_CAPABILITY_ENTRYPOINT_V1, None)
    if not callable(entrypoint):
        raise _fail(f"CUDA extension is missing callable {EED_CAPABILITY_ENTRYPOINT_V1}")
    raw = entrypoint()
    if not isinstance(raw, Mapping):
        raise _fail("EED CUDA capability must be a mapping")
    capability = dict(raw)
    expected = {
        "abi_version": EED_ABI_VERSION_V1,
        "world": "patch_eed_resource_impact",
        "resident_state": True,
        "observation_size": EED_OBSERVATION_SIZE_V1,
        "exact_respec_handling": True,
        "exact_resource_impact_settlement": True,
        "native_tick_ownership": True,
        "runtime_jit_required": False,
        "production_qualified": False,
    }
    for name, expected_value in expected.items():
        actual = capability.get(name)
        if type(actual) is not type(expected_value) or actual != expected_value:
            raise _fail(
                f"EED CUDA capability {name} mismatch: "
                f"expected={expected_value!r} actual={actual!r}"
            )
    return capability


def _patch_envelope_with_eed_masked(
    runner: Any,
) -> CudaArchitectureBRunnerEnvelope:
    """Classify the accepted base Patch route without mutating the runner."""

    proxy = copy.copy(runner)
    proxy.env_cfg = copy.copy(runner.env_cfg)
    proxy.env_cfg.eed_resource_impact_enable = False
    proxy.env_cfg.eed_resource_impact_apply_enable = False
    proxy.env_cfg.eed_ecology_frozen_enable = False
    proxy.env_cfg.eed_phenotype_fixed_enable = False
    proxy.env_cfg.respec0b_resource_specialization_enable = False
    return derive_cuda_architecture_b_runner_envelope(proxy)


def _required_eed_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    exact = {
        "native_runtime_eed_execution_enable": True,
        "native_runtime_eed_storage_schema_version": (
            "eiel_eed_resource_impact_native_storage_v1"
        ),
        "respec0b_resource_specialization_enable": True,
        "eed_resource_impact_enable": True,
    }
    for name, expected in exact.items():
        actual = metadata.get(name)
        if type(actual) is not type(expected) or actual != expected:
            raise _fail(
                f"EED import metadata {name} mismatch: "
                f"expected={expected!r} actual={actual!r}"
            )
    required = (
        "patch_food_presence_threshold",
        "patch_food_gather_amount",
        "respec0b_resource_a_handling_threshold",
        "respec0b_resource_b_handling_threshold",
        "respec0b_resource_a_return",
        "respec0b_resource_b_return",
        "respec0b_resource_equalized_sham_enable",
        "eed_resource_impact_apply_enable",
        "eed_ecology_frozen_enable",
        "eed_phenotype_fixed_enable",
        "eed_phenotype_fixed_axis_profile",
        "eed_resource_impact_per_success",
        "eed_resource_impact_decay_per_step",
        "eed_resource_impact_max_load",
    )
    missing = [name for name in required if name not in metadata]
    if missing:
        raise _fail(f"EED import metadata is missing: {missing}")
    if float(metadata.get("food_spawn_per_step", 0.0) or 0.0) != 0.0:
        raise _fail("EED import requires legacy food spawning disabled")
    if (
        float(metadata.get("patch_scent_gather_emit", 0.0) or 0.0) != 0.0
        or float(metadata.get("patch_waste_consume_emit", 0.0) or 0.0) != 0.0
    ):
        raise _fail("EED import requires zero gather emissions")
    return dict(metadata)


def _require_successful_physical_result_v1(
    result: Mapping[str, Any],
    *,
    generation_count: int,
    generation_base: int,
) -> None:
    sentinel = result.get("first_failure_no_failure_sentinel")
    failures = result.get("first_failure_by_cell")
    if type(sentinel) is not int or not isinstance(failures, list):
        raise _fail("EED physical result has malformed native failure records")
    for failure in failures:
        if not isinstance(failure, Mapping):
            raise _fail("EED physical result has malformed native failure records")
        status = failure.get("status")
        if type(status) is not int:
            raise _fail("EED physical result has malformed native failure status")
        if status not in {0, sentinel}:
            raise _fail(
                "EED physical native execution failed: "
                f"cell={failure.get('cell')!r} "
                f"generation={failure.get('generation')!r} "
                f"tick={failure.get('tick')!r} status={status}"
            )
    completed = result.get("completed_generations")
    completed_by_cell = result.get("completed_generations_by_cell")
    if (
        type(completed) is not int
        or completed != generation_count
        or not isinstance(completed_by_cell, list)
        or not completed_by_cell
        or any(
            type(value) is not int
            or value != generation_base + generation_count
            for value in completed_by_cell
        )
    ):
        raise _fail(
            "EED physical result did not complete the requested segment: "
            f"expected_count={generation_count} "
            f"expected_terminal={generation_base + generation_count} "
            f"completed={completed!r} "
            f"by_cell={completed_by_cell!r}"
        )


def _eed_cursor_state_tensors_v1(
    tensors: Mapping[str, torch.Tensor],
    *,
    slot_capacity: int,
) -> tuple[torch.Tensor, ...]:
    cursor_state: list[torch.Tensor] = []
    for name in EED_CURSOR_STATE_FIELDS_V1:
        initial = tensors[name]
        if initial.ndim != 1 or int(initial.numel()) != int(slot_capacity):
            raise _fail(
                "EED cursor state must be capacity-bounded slot rows: "
                f"field={name};shape={tuple(initial.shape)};"
                f"slot_capacity={int(slot_capacity)}"
            )
        cursor_state.append(initial.unsqueeze(0).contiguous())
    return tuple(cursor_state)


def _validated_eed_cell_ids_v1(
    value: Sequence[str],
    *,
    count: int,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise _fail("EED cell_ids must be a sequence")
    cell_ids = tuple(value)
    if (
        len(cell_ids) != count
        or len(set(cell_ids)) != count
        or any(
            type(cell_id) is not str
            or not cell_id
            or cell_id.strip() != cell_id
            for cell_id in cell_ids
        )
    ):
        raise _fail("EED cell_ids must be unique stable nonempty strings")
    return cell_ids


def build_direct_resident_eed_source_request_v1(
    runner: Any,
    *,
    qualified_cuda_module: Any,
    logical_generation_count: int = 3,
    initial_generation: int = 0,
    source_checkpoint_projection: Mapping[str, Any] | None = None,
    source_capability_mask: Mapping[str, bool] | None = None,
    agent_id_allocation_generation_count: int | None = None,
    cuda_device: str = "cuda",
    event_capacity_override: int | None = None,
) -> DirectResidentSourceAllocationRequestV1:
    """Translate one EED runner into the public resident source request."""

    if (
        type(logical_generation_count) is not int
        or not EED_RESIDENT_GENERATION_MIN_V1
        <= logical_generation_count
        <= EED_RESIDENT_GENERATION_MAX_V1
    ):
        raise _fail("resident EED requires logical_generation_count in [3,128]")
    if type(initial_generation) is not int or initial_generation < 0:
        raise _fail("EED initial_generation must be a nonnegative exact integer")
    if cuda_device not in {"cuda", "cuda:0"}:
        raise _fail("EED CUDA source requires cuda or cuda:0")
    capability = _validated_capability(qualified_cuda_module)
    envelope = _patch_envelope_with_eed_masked(runner)
    ticks_per_generation = _eed_ticks_per_generation_v1(
        getattr(envelope, "ticks_per_generation", None),
        name="EED runner envelope ticks_per_generation",
    )
    sim_cfg = runner.sim_cfg
    lifecycle_mode = getattr(sim_cfg, "lifecycle_mode", 0)
    if type(lifecycle_mode) is not int or lifecycle_mode not in {0, 1}:
        raise _fail("runner lifecycle_mode must be exact integer 0 or 1")
    generation_turnover_enabled = getattr(
        sim_cfg, "generation_turnover_enabled", lifecycle_mode == 0
    )
    if type(generation_turnover_enabled) is not bool:
        raise _fail("runner generation_turnover_enabled must be an exact bool")
    if generation_turnover_enabled is not (lifecycle_mode == 0):
        mode_name = "legacy" if lifecycle_mode == 0 else "hybrid"
        raise _fail(
            "runner generation_turnover_enabled must match "
            f"{mode_name} lifecycle mode"
        )
    life_history_config = _life_history_config_from_sim_cfg_v1(sim_cfg)
    source_capabilities = _eed_source_capabilities_v1(source_capability_mask)
    fresh_projection = build_import_projection(
        runner,
        source_capabilities,
        generation=initial_generation,
        tick=0,
    )
    fresh_projection.metadata[
        "native_runtime_architecture_b_compact_reporting_only_v1"
    ] = False
    fresh_projection.metadata["generation_turnover_enabled"] = (
        generation_turnover_enabled
    )
    fresh_projection.metadata["lifecycle_mode_abi"] = lifecycle_mode
    fresh_projection.metadata.update(life_history_config)
    fresh_projection.metadata["import_digest"] = projection_digest(
        {
            "metadata": fresh_projection.metadata,
            "tensors": fresh_projection.tensors,
        }
    )
    projection: Any = (
        fresh_projection
        if source_checkpoint_projection is None
        else source_checkpoint_projection
    )
    projection = _normalized_eed_source_projection_v1(projection)
    projection_metadata = (
        projection.get("metadata")
        if isinstance(projection, Mapping)
        else getattr(projection, "metadata", None)
    )
    if not isinstance(projection_metadata, Mapping):
        raise _fail("EED source projection is missing metadata")
    projection_capabilities = projection_metadata.get("capability_mask")
    if (
        not isinstance(projection_capabilities, Mapping)
        or dict(projection_capabilities) != source_capabilities
    ):
        raise _fail(
            "EED source projection capability mask differs from the "
            "configured resident route"
        )
    if (
        projection_metadata.get("generation") != initial_generation
        or projection_metadata.get("tick") != 0
        or projection_metadata.get("native_runtime_fixed_policy_family") != "mlp"
    ):
        raise _fail("EED source checkpoint is not the requested fixed-MLP boundary")
    projected_turnover = projection_metadata.get("generation_turnover_enabled")
    projected_lifecycle_mode = projection_metadata.get("lifecycle_mode_abi")
    if projected_turnover is None and projected_lifecycle_mode is None:
        if lifecycle_mode != 0:
            raise _fail("hybrid EED source checkpoint lacks lifecycle metadata")
    elif (
        type(projected_turnover) is not bool
        or isinstance(projected_lifecycle_mode, bool)
        or projected_lifecycle_mode not in {0, 1}
    ):
        raise _fail("EED source checkpoint lifecycle metadata is malformed")
    elif projected_turnover is not generation_turnover_enabled:
        raise _fail("EED source checkpoint generation turnover mismatch")
    elif int(projected_lifecycle_mode) != lifecycle_mode:
        raise _fail("EED source checkpoint lifecycle mode mismatch")
    _validate_life_history_metadata_v1(
        projection_metadata,
        life_history=life_history_config,
    )
    metadata = _required_eed_metadata(projection_metadata)
    max_births = int(envelope.max_births_per_tick)
    event_capacity = _eed_source_event_capacity_v1(
        projection,
        projection_metadata,
        logical_generation_count=logical_generation_count,
        max_births_per_tick=max_births,
        ticks_per_generation=ticks_per_generation,
    )
    if event_capacity_override is not None:
        if type(event_capacity_override) is not int or event_capacity_override < 0:
            raise _fail("event_capacity_override must be a nonnegative exact integer")
        if event_capacity_override < event_capacity:
            raise _fail("event_capacity_override cannot truncate the bounded event capacity")
        event_capacity = event_capacity_override
    boundary = import_cpu_owner_boundary_to_cuda(
        projection,
        event_capacity=event_capacity,
        device=cuda_device,
    )
    ordinary_births_enabled = getattr(
        sim_cfg, "ordinary_births_enabled", None
    )
    if ordinary_births_enabled is None:
        ordinary_births_enabled = max_births > 0
    elif type(ordinary_births_enabled) is not bool:
        raise _fail("runner ordinary_births_enabled must be an exact bool")
    policy_episode_capacity = required_physical_window_policy_episodes_v1(
        max_births
    )
    birth_state_kwargs = {
        "steps": PHYSICAL_WINDOW_TICKS_V1,
        "max_births_per_tick": max_births,
        "turnover_refill_capacity": PHYSICAL_WINDOW_TURNOVER_EPISODES_V1,
    }
    birth_state_kwargs.update(
        birth_state_capacity_overrides_v1(
            sim_cfg,
            ordinary_births_enabled=ordinary_births_enabled,
            resident_policy_episode_capacity=policy_episode_capacity,
            turnover_refill_capacity=PHYSICAL_WINDOW_TURNOVER_EPISODES_V1,
            max_births_per_tick=max_births,
        )
    )
    turnover_mlp_shadow_elide_enable = getattr(
        sim_cfg,
        "cuda_architecture_b_turnover_mlp_shadow_elide_enable",
        False,
    )
    if type(turnover_mlp_shadow_elide_enable) is not bool:
        raise _fail(
            "runner cuda_architecture_b_turnover_mlp_shadow_elide_enable "
            "must be an exact bool"
        )
    cpu_request_staging_enable = getattr(
        sim_cfg,
        "cuda_architecture_b_cpu_request_staging_enable",
        False,
    )
    if type(cpu_request_staging_enable) is not bool:
        raise _fail(
            "runner cuda_architecture_b_cpu_request_staging_enable must be an exact bool"
        )
    birth_state = prepare_generation_birth_state_cuda(
        boundary,
        projection,
        **birth_state_kwargs,
    )
    if cpu_request_staging_enable:
        boundary = _stage_request_tensors_to_cpu_v1(boundary)
        birth_state = _stage_request_tensors_to_cpu_v1(birth_state)
    configuration = replace(
        CudaGenerationConfig.from_metadata(
            metadata,
            steps=ticks_per_generation,
        ),
        generation=initial_generation,
        start_tick=0,
        max_births_per_tick=max_births,
        ordinary_births_enabled=ordinary_births_enabled,
        genome_upgrade_v2_founder_profile_id=str(
            getattr(
                sim_cfg,
                "genome_upgrade_v2_founder_profile_id",
                "neutral",
            )
        ),
        genome_upgrade_v2_founder_profile_sha256=str(
            getattr(
                sim_cfg,
                "genome_upgrade_v2_founder_profile_sha256",
                "",
            )
        ),
        genome_v2_checkpoint_authority_binding=tuple(
            getattr(
                sim_cfg,
                "genome_v2_checkpoint_authority_binding",
                (),
            )
        ),
        generation_turnover_enabled=generation_turnover_enabled,
        lifecycle_mode=lifecycle_mode,
        **life_history_config,
    )
    return DirectResidentSourceAllocationRequestV1(
        envelope=envelope,
        qualified_cuda_module=qualified_cuda_module,
        cuda_capability=capability,
        import_projection=projection,
        cuda_boundary=boundary,
        generation_birth_state=birth_state,
        configuration=configuration,
        cuda_device=cuda_device,
        event_capacity=event_capacity,
        logical_generation_count=logical_generation_count,
        agent_id_allocation_generation_count=(
            agent_id_allocation_generation_count
        ),
        turnover_mlp_shadow_elide_enable=turnover_mlp_shadow_elide_enable,
        cpu_request_staging_enable=cpu_request_staging_enable,
    )


def build_prepared_direct_resident_eed_v1(
    runner: Any | None,
    *,
    qualified_cuda_module: Any,
    cell_id: str | None = None,
    logical_generation_count: int = 3,
    initial_generation: int = 0,
    source_checkpoint_projection: Mapping[str, Any] | None = None,
    source_capability_mask: Mapping[str, bool] | None = None,
    per_cell_requests: Sequence[DirectResidentSourceAllocationRequestV1]
    | None = None,
    source_checkpoint_projections: Sequence[Mapping[str, Any]] | None = None,
    cell_ids: Sequence[str] | None = None,
    agent_id_allocation_generation_count: int | None = None,
    cuda_device: str = "cuda",
) -> PreparedDirectResidentEedV1:
    """Build an ordered physical B1..B48 EED owner."""

    if (
        type(logical_generation_count) is not int
        or not EED_RESIDENT_GENERATION_MIN_V1
        <= logical_generation_count
        <= EED_RESIDENT_GENERATION_MAX_V1
    ):
        raise _fail("resident EED requires logical_generation_count in [3,128]")
    if type(initial_generation) is not int or initial_generation < 0:
        raise _fail("EED initial_generation must be a nonnegative exact integer")
    if cuda_device not in {"cuda", "cuda:0"}:
        raise _fail("EED CUDA source requires cuda or cuda:0")
    _validated_capability(qualified_cuda_module)
    if per_cell_requests is None:
        if runner is None:
            raise _fail("EED preparation requires runner or per_cell_requests")
        if source_checkpoint_projections is not None or cell_ids is not None:
            raise _fail("B1 EED preparation uses one source checkpoint and cell_id")
        if (
            type(cell_id) is not str
            or not cell_id
            or cell_id.strip() != cell_id
        ):
            raise _fail("EED cell_id must be an explicit stable nonempty string")
        requests = (
            build_direct_resident_eed_source_request_v1(
                runner,
                qualified_cuda_module=qualified_cuda_module,
                logical_generation_count=logical_generation_count,
                initial_generation=initial_generation,
                source_checkpoint_projection=source_checkpoint_projection,
                source_capability_mask=source_capability_mask,
                agent_id_allocation_generation_count=(
                    agent_id_allocation_generation_count
                ),
                cuda_device=cuda_device,
            ),
        )
        cell_ids_tuple = (cell_id,)
    else:
        if (
            runner is not None
            or source_checkpoint_projection is not None
            or cell_id is not None
        ):
            raise _fail(
                "multicell EED preparation cannot also supply runner/B1 checkpoint"
            )
        if isinstance(per_cell_requests, (str, bytes)) or not isinstance(
            per_cell_requests, Sequence
        ):
            raise _fail("per_cell_requests must be a sequence")
        requests = tuple(per_cell_requests)
        if not 1 <= len(requests) <= 48 or any(
            not isinstance(item, DirectResidentSourceAllocationRequestV1)
            for item in requests
        ):
            raise _fail("per_cell_requests must contain B1..B48 public requests")
        if cell_ids is None:
            raise _fail("multicell EED preparation requires explicit cell_ids")
        cell_ids_tuple = _validated_eed_cell_ids_v1(
            cell_ids,
            count=len(requests),
        )
        requests = tuple(
            replace(
                request,
                batch_cells=len(requests),
                logical_generation_count=logical_generation_count,
                agent_id_allocation_generation_count=(
                    agent_id_allocation_generation_count
                ),
            )
            for request in requests
        )

    projections = tuple(
        _normalized_eed_source_projection_v1(request.import_projection)
        for request in requests
    )
    if source_checkpoint_projections is not None:
        checkpoints = tuple(source_checkpoint_projections)
        if len(checkpoints) != len(requests) or any(
            request.import_projection is not checkpoint
            for request, checkpoint in zip(requests, checkpoints, strict=True)
        ):
            raise _fail(
                "ordered EED source checkpoints must be bound without substitution"
            )
    requests = tuple(
        replace(request, import_projection=projection)
        for request, projection in zip(requests, projections, strict=True)
    )
    metadata_rows: list[Mapping[str, Any]] = []
    for cell, projection in enumerate(projections):
        projection_metadata = (
            projection.get("metadata")
            if isinstance(projection, Mapping)
            else getattr(projection, "metadata", None)
        )
        if not isinstance(projection_metadata, Mapping):
            raise _fail(f"EED source projection {cell} is missing metadata")
        projection_capabilities = projection_metadata.get("capability_mask")
        if (
            not isinstance(projection_capabilities, Mapping)
            or dict(projection_capabilities)
            != _eed_source_capabilities_v1(projection_capabilities)
        ):
            raise _fail(
                f"EED source projection {cell} capability mask differs from "
                "the configured resident route"
            )
        if (
            projection_metadata.get("generation") != initial_generation
            or projection_metadata.get("tick") != 0
            or projection_metadata.get("native_runtime_fixed_policy_family")
            != "mlp"
        ):
            raise _fail(
                f"EED source projection {cell} is not the requested fixed-MLP boundary"
            )
        metadata_rows.append(_required_eed_metadata(projection_metadata))
    metadata = metadata_rows[0]
    shared_fields = (
        "patch_food_presence_threshold",
        "patch_food_gather_amount",
        "respec0b_resource_a_handling_threshold",
        "respec0b_resource_b_handling_threshold",
        "respec0b_resource_a_return",
        "respec0b_resource_b_return",
        "respec0b_resource_equalized_sham_enable",
        "eed_resource_impact_apply_enable",
        "eed_ecology_frozen_enable",
        "eed_phenotype_fixed_enable",
        "eed_resource_impact_per_success",
        "eed_resource_impact_decay_per_step",
        "eed_resource_impact_max_load",
    )
    for name in shared_fields:
        expected = metadata.get(name)
        if any(row.get(name) != expected for row in metadata_rows[1:]):
            raise _fail(
                f"mixed per-cell {name} is not representable by EED/fixed-MLP"
            )

    source = allocate_direct_resident_source_v1(
        requests[0],
        per_cell_requests=requests if len(requests) > 1 else None,
        per_cell_ids=cell_ids_tuple,
    )
    device = source.live_state["energy"].device
    if any(name not in source.live_state for name in EED_STATE_FIELDS_V1):
        missing_by_cell = {
            cell: [
                name
                for name in EED_STATE_FIELDS_V1
                if name not in request.cuda_boundary.tensors
            ]
            for cell, request in enumerate(requests)
        }
        missing_by_cell = {
            cell: names for cell, names in missing_by_cell.items() if names
        }
        if missing_by_cell:
            raise _fail(
                f"EED/fixed-MLP CUDA boundaries are missing EED state: "
                f"{missing_by_cell}"
            )
        eed_live_state = {
            name: torch.stack(
                tuple(
                    request.cuda_boundary.tensors[name] for request in requests
                ),
                dim=0,
            )
            .to(device=device)
            .contiguous()
            for name in EED_STATE_FIELDS_V1
        }
        source = replace(
            source,
            live_state={**dict(source.live_state), **eed_live_state},
            live_banks={**dict(source.live_banks), **eed_live_state},
        )
    state_tensors = tuple(
        source.live_state[name].contiguous() for name in EED_STATE_FIELDS_V1
    )
    fixed_axis_modes = {
        "neutral": 0,
        "a_high": 1,
        "b_high": 2,
        "symmetric_mix": 3,
    }
    fixed_axis_mode_by_cell = torch.tensor(
        [
            fixed_axis_modes[
                str(row.get("eed_phenotype_fixed_axis_profile", "neutral"))
            ]
            for row in metadata_rows
        ],
        dtype=torch.int32,
        device=device,
    )
    config = {
        "enabled": True,
        "schema_version": EED_ABI_VERSION_V1,
        "food_presence_threshold": float(
            metadata["patch_food_presence_threshold"]
        ),
        "gather_amount": float(metadata["patch_food_gather_amount"]),
        "resource_a_handling_threshold": float(
            metadata["respec0b_resource_a_handling_threshold"]
        ),
        "resource_b_handling_threshold": float(
            metadata["respec0b_resource_b_handling_threshold"]
        ),
        "resource_a_return": float(metadata["respec0b_resource_a_return"]),
        "resource_b_return": float(metadata["respec0b_resource_b_return"]),
        "equalized_sham": bool(
            metadata["respec0b_resource_equalized_sham_enable"]
        ),
        "impact_per_success": float(
            metadata["eed_resource_impact_per_success"]
        ),
        "decay_per_step": float(
            metadata["eed_resource_impact_decay_per_step"]
        ),
        "maximum_load": float(metadata["eed_resource_impact_max_load"]),
        "impact_applied": bool(metadata["eed_resource_impact_apply_enable"]),
        "ecology_frozen": bool(metadata["eed_ecology_frozen_enable"]),
        "phenotype_fixed": bool(metadata["eed_phenotype_fixed_enable"]),
        "fixed_axis_mode": int(fixed_axis_mode_by_cell[0].item()),
        "fixed_axis_mode_by_cell": fixed_axis_mode_by_cell,
    }
    return PreparedDirectResidentEedV1(
        patch=prepare_direct_resident_trajectory_v1(source),
        state_tensors=state_tensors,
        config=config,
        initial_projection=projections[0],
        cell_id=cell_ids_tuple[0],
        cell_ids=cell_ids_tuple,
        source_projections=projections,
    )


def execute_direct_resident_eed_physical_v1(
    module: Any,
    prepared: PreparedDirectResidentEedV1,
    *,
    generation_count: int,
    ticks_per_generation: int = EED_TICKS_PER_GENERATION_V1,
) -> dict[str, Any]:
    """Execute one native-owned physical g1 or g2 EED qualification segment."""

    if not isinstance(prepared, PreparedDirectResidentEedV1):
        raise _fail("prepared must be PreparedDirectResidentEedV1")
    if type(generation_count) is not int or generation_count not in {1, 2}:
        raise _fail("physical EED qualification requires generation_count 1 or 2")
    if (
        type(ticks_per_generation) is not int
        or ticks_per_generation < 1
        or ticks_per_generation
        not in EED_ADMITTED_TICKS_PER_GENERATION_V1
    ):
        raise _fail(
            "physical EED ticks_per_generation must be one of "
            f"{list(EED_ADMITTED_TICKS_PER_GENERATION_V1)!r}"
        )
    entrypoint = getattr(module, EED_PHYSICAL_ENTRYPOINT_V1, None)
    if not callable(entrypoint):
        raise _fail(f"CUDA extension is missing callable {EED_PHYSICAL_ENTRYPOINT_V1}")
    arguments = list(prepared.patch.arguments)
    physical_ticks = generation_count * ticks_per_generation
    arguments[PHYSICAL_RUNTIME_BOOKKEEPING_ARGUMENT_INDEX_V1] = arguments[
        PHYSICAL_RUNTIME_BOOKKEEPING_ARGUMENT_INDEX_V1
    ][:, :physical_ticks]
    full_tick_outputs = list(
        arguments[PHYSICAL_FULL_TICK_OUTPUTS_ARGUMENT_INDEX_V1]
    )
    full_tick_outputs[-1] = full_tick_outputs[-1][:generation_count]
    arguments[PHYSICAL_FULL_TICK_OUTPUTS_ARGUMENT_INDEX_V1] = full_tick_outputs
    patch_stats = list(arguments[PHYSICAL_PATCH_STATS_ARGUMENT_INDEX_V1])
    patch_stats[0] = patch_stats[0][:, :physical_ticks]
    arguments[PHYSICAL_PATCH_STATS_ARGUMENT_INDEX_V1] = patch_stats
    arguments[TICKS_ARGUMENT_INDEX_V1] = ticks_per_generation
    arguments[GENERATION_COUNT_ARGUMENT_INDEX_V1] = generation_count
    raw = entrypoint(
        list(prepared.state_tensors),
        dict(prepared.config),
        *arguments,
    )
    if not isinstance(raw, Mapping):
        raise _fail("EED physical CUDA entrypoint returned a non-mapping result")
    result = dict(raw)
    _require_successful_physical_result_v1(
        result,
        generation_count=generation_count,
        generation_base=prepared.patch.source.generation_base,
    )
    expected = {
        "eed_native_owner": True,
        "eed_native_controller_abi_version": EED_ABI_VERSION_V1,
        "eed_physical_gate_entrypoint": True,
        "eed_python_tick_or_generation_ownership": False,
        "hidden_fallback": False,
        "runtime_jit": False,
    }
    for name, expected_value in expected.items():
        actual = result.get(name)
        if type(actual) is not type(expected_value) or actual != expected_value:
            raise _fail(
                f"EED physical result {name} mismatch: "
                f"expected={expected_value!r} actual={actual!r}"
            )
    return result


def execute_direct_resident_eed_v1(
    module: Any,
    prepared: PreparedDirectResidentEedV1,
    *,
    logical_generation_count: int,
    run_id: str,
) -> dict[str, Any]:
    """Execute one complete EED trajectory through reusable native g2 windows."""

    if not isinstance(prepared, PreparedDirectResidentEedV1):
        raise _fail("prepared must be PreparedDirectResidentEedV1")
    if (
        type(logical_generation_count) is not int
        or not EED_RESIDENT_GENERATION_MIN_V1
        <= logical_generation_count
        <= EED_RESIDENT_GENERATION_MAX_V1
    ):
        raise _fail("resident EED requires logical_generation_count in [3,128]")
    if prepared.patch.source.logical_generation_count != logical_generation_count:
        raise _fail("EED source identity capacity does not match the requested horizon")
    entrypoint = getattr(module, EED_DIRECT_ENTRYPOINT_V1, None)
    if not callable(entrypoint):
        raise _fail(f"CUDA extension is missing callable {EED_DIRECT_ENTRYPOINT_V1}")
    control = create_run_control_v2(module, run_id=run_id)
    metadata_value = (
        prepared.initial_projection.get("metadata")
        if isinstance(prepared.initial_projection, Mapping)
        else getattr(prepared.initial_projection, "metadata", None)
    )
    if not isinstance(metadata_value, Mapping):
        raise _fail("EED initial projection metadata is not a mapping")
    metadata = dict(metadata_value)
    raw = entrypoint(
        control,
        list(prepared.state_tensors),
        dict(prepared.config),
        *prepared.patch.arguments,
        logical_generation_count,
        metadata.get("policy_weight_mutation_enable"),
        metadata.get("policy_weight_mutation_mode"),
        metadata.get("policy_weight_mutation_rate"),
        metadata.get("policy_weight_mutation_std"),
    )
    if not isinstance(raw, Mapping):
        raise _fail("EED resident CUDA entrypoint returned a non-mapping result")
    result = dict(raw)
    expected = {
        "eed_native_owner": True,
        "eed_native_controller_abi_version": EED_ABI_VERSION_V1,
        "eed_python_tick_or_generation_ownership": False,
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
    }
    for name, expected_value in expected.items():
        actual = result.get(name)
        if type(actual) is not type(expected_value) or actual != expected_value:
            raise _fail(
                f"EED resident result {name} mismatch: "
                f"expected={expected_value!r} actual={actual!r}"
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
        raise _fail("EED resident run-control manifest is not at the final boundary")
    result["native_run_control_v2"] = manifest
    return result


def project_direct_resident_eed_checkpoint_v1(
    prepared: PreparedDirectResidentEedV1,
    *,
    direct_result: Mapping[str, Any],
    completed_generation_count: int,
) -> Mapping[str, Any]:
    """Project an EED prefix through the accepted Patch checkpoint path."""

    if not isinstance(prepared, PreparedDirectResidentEedV1):
        raise _fail("prepared must be PreparedDirectResidentEedV1")
    if (
        type(completed_generation_count) is not int
        or completed_generation_count < 1
        or completed_generation_count
        > prepared.patch.source.logical_generation_count
    ):
        raise _fail("EED checkpoint projection exceeds its resident horizon")
    result = dict(direct_result)
    outputs = dict(prepared.patch.source.outputs)
    ticks_per_generation = _eed_ticks_per_generation_v1(
        getattr(
            prepared.patch.source,
            "ticks_per_generation",
            EED_TICKS_PER_GENERATION_V1,
        ),
        name="prepared EED source ticks_per_generation",
    )
    generation_summary = result.get("generation_summary_scalar")
    if not isinstance(generation_summary, torch.Tensor):
        generation_summary = prepared.patch.arguments[
            PHYSICAL_FULL_TICK_OUTPUTS_ARGUMENT_INDEX_V1
        ][-1][:completed_generation_count]
    runtime_bookkeeping = result.get("runtime_bookkeeping_by_tick")
    if not isinstance(runtime_bookkeeping, torch.Tensor):
        runtime_bookkeeping = prepared.patch.arguments[
            PHYSICAL_RUNTIME_BOOKKEEPING_ARGUMENT_INDEX_V1
        ][:, : completed_generation_count * ticks_per_generation]
    outputs["generation_summary_scalar"] = generation_summary
    outputs["runtime_bookkeeping_by_tick"] = runtime_bookkeeping
    persistent_state = dict(
        _with_materialized_event_history(prepared.patch.live_banks, result)
    )
    persistent_state = materialize_disabled_prediction_state_v1(
        prepared.initial_projection,
        persistent_state,
        reset_count_max=ticks_per_generation,
    )
    for name, tensor in zip(
        EED_STATE_FIELDS_V1,
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
        inheritance_sidecar_cell_id=prepared.cell_id,
        genome_v2_checkpoint_authority_binding=projection_inputs.get(
            "genome_v2_checkpoint_authority_binding", ()
        ),
        genome_v2_causal_physical_cell_ordinal=projection_inputs.get(
            "genome_v2_causal_physical_cell_ordinal", 0
        ),
    )
    return _with_checkpoint_contract_metadata(dict(checkpoint.projection))


def _validated_eed_checkpoint_state_v1(
    projection: Mapping[str, Any],
    *,
    cell: int,
) -> Mapping[str, Any]:
    from engine.native_runtime.interchange_projection import (
        validate_native_checkpoint_projection,
    )

    tensors = projection.get("tensors")
    if not isinstance(tensors, Mapping):
        raise _fail(f"EED checkpoint {cell} is missing tensors")
    missing = [name for name in EED_STATE_FIELDS_V1 if name not in tensors]
    if missing:
        raise _fail(f"EED checkpoint {cell} is missing EED state: {missing}")
    checked = _with_checkpoint_contract_metadata(dict(projection))
    validate_native_checkpoint_projection(checked)
    return checked


def project_direct_resident_eed_multicell_checkpoints_v1(
    prepared: PreparedDirectResidentEedV1,
    *,
    direct_result: Mapping[str, Any],
    completed_generation_count: int,
    reporting_enabled: bool,
) -> tuple[Mapping[str, Any], ...]:
    """Project ordered B1..B48 EED checkpoints through the Patch contract."""

    settlement = project_direct_resident_eed_multicell_settlement_v1(
        prepared,
        direct_result=direct_result,
        completed_generation_count=completed_generation_count,
        reporting_enabled=reporting_enabled,
    )
    return settlement.checkpoint_projections


def project_direct_resident_eed_multicell_settlement_v1(
    prepared: PreparedDirectResidentEedV1,
    *,
    direct_result: Mapping[str, Any],
    completed_generation_count: int,
    reporting_enabled: bool,
) -> Any:
    """Project ordered EED checkpoints while preserving Patch settlement receipts."""

    if not isinstance(prepared, PreparedDirectResidentEedV1):
        raise _fail("prepared must be PreparedDirectResidentEedV1")
    cell_ids = prepared.cell_ids or (prepared.cell_id,)
    projections = prepared.source_projections or (prepared.initial_projection,)
    if len(cell_ids) != len(projections):
        raise _fail("EED prepared owner lost ordered source projection identity")
    from engine.perf.cuda_architecture_b_direct_resident_patch_v1 import (
        DirectResidentPatchMulticellSettlementV1,
        PATCH_MULTICELL_FACADE_SCHEMA_VERSION_V1,
        PreparedDirectResidentPatchMulticellV1,
        project_direct_resident_patch_multicell_checkpoints_v1,
    )

    patch_prepared = PreparedDirectResidentPatchMulticellV1(
        schema_version=PATCH_MULTICELL_FACADE_SCHEMA_VERSION_V1,
        patch=prepared.patch,
        cell_ids=cell_ids,
        source_projections=projections,
        resumed=False,
    )
    capability = prepared.patch.checkpoint_projection_inputs.get(
        "cuda_capability"
    )
    if not isinstance(capability, Mapping):
        raise _fail("EED multicell checkpoint projection lacks CUDA capability")
    settlement = project_direct_resident_patch_multicell_checkpoints_v1(
        patch_prepared,
        direct_result=direct_result,
        capability=capability,
        reporting_enabled=reporting_enabled,
    )
    checked_projections = tuple(
        _validated_eed_checkpoint_state_v1(projection, cell=cell)
        for cell, projection in enumerate(settlement.checkpoint_projections)
    )
    return DirectResidentPatchMulticellSettlementV1(
        schema_version=settlement.schema_version,
        cell_ids=settlement.cell_ids,
        checkpoint_projections=checked_projections,
        compact_result_rows=settlement.compact_result_rows,
        completed_generation_count=settlement.completed_generation_count,
        terminal_generation=settlement.terminal_generation,
        segment_artifacts=settlement.segment_artifacts,
        analysis_source_rows=settlement.analysis_source_rows,
        lifecycle_qacct_receipts=settlement.lifecycle_qacct_receipts,
        settlement_event_rows=settlement.settlement_event_rows,
    )


def project_direct_resident_eed_generation_zero_v1(
    prepared: Any,
) -> tuple[Mapping[str, Any], ...]:
    """Project initialized EED G0 authority for durable checkpoints."""

    prepared = getattr(prepared, "physical_window_arguments", prepared)
    if not isinstance(prepared, PreparedDirectResidentEedV1):
        raise _fail("prepared must be PreparedDirectResidentEedV1")
    cell_ids = prepared.cell_ids or (prepared.cell_id,)
    projections = prepared.source_projections or (prepared.initial_projection,)
    from engine.perf.cuda_architecture_b_direct_resident_patch_v1 import (
        PATCH_MULTICELL_FACADE_SCHEMA_VERSION_V1,
        PreparedDirectResidentPatchMulticellV1,
        project_direct_resident_patch_multicell_generation_zero_v1,
    )

    projected = project_direct_resident_patch_multicell_generation_zero_v1(
        PreparedDirectResidentPatchMulticellV1(
            schema_version=PATCH_MULTICELL_FACADE_SCHEMA_VERSION_V1,
            patch=prepared.patch,
            cell_ids=cell_ids,
            source_projections=projections,
            resumed=False,
        )
    )
    return tuple(
        _validated_eed_checkpoint_state_v1(projection, cell=cell)
        for cell, projection in enumerate(projected)
    )


__all__ = [
    "EED_CAPABILITY_ENTRYPOINT_V1",
    "EED_DIRECT_ENTRYPOINT_V1",
    "EED_PHYSICAL_ENTRYPOINT_V1",
    "EED_STATE_FIELDS_V1",
    "CudaArchitectureBDirectResidentEedV1Error",
    "PreparedDirectResidentEedV1",
    "build_direct_resident_eed_source_request_v1",
    "build_prepared_direct_resident_eed_v1",
    "execute_direct_resident_eed_physical_v1",
    "execute_direct_resident_eed_v1",
    "project_direct_resident_eed_checkpoint_v1",
    "project_direct_resident_eed_generation_zero_v1",
    "project_direct_resident_eed_multicell_checkpoints_v1",
    "project_direct_resident_eed_multicell_settlement_v1",
]
