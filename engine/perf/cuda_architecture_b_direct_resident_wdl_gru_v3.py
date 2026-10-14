"""Build and execute the combined WDL plus stable-GRU resident CUDA path."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, Final

import torch

from engine.native_runtime.cuda_architecture_b_backend_factory import (
    derive_cuda_architecture_b_runner_envelope,
)
from engine.perf.cuda_architecture_b_direct_resident_execution_v2 import (
    create_run_control_v2,
)
from engine.perf.cuda_architecture_b_direct_resident_runtime_v1 import (
    resident_window_count_v1,
)
from engine.perf.cuda_architecture_b_direct_resident_gru_v3 import (
    GRU_ABI_VERSION_V3,
    GRU_PHYSICAL_GENERATION_COUNTS_V3,
    GRU_SCRATCH_WIDTH_V3,
    PHYSICAL_FULL_TICK_OUTPUTS_ARGUMENT_INDEX_V3,
    PHYSICAL_PATCH_STATS_ARGUMENT_INDEX_V3,
    PHYSICAL_RUNTIME_BOOKKEEPING_ARGUMENT_INDEX_V3,
    TICKS_ARGUMENT_INDEX_V3,
    PreparedDirectResidentGruV3,
    _canonical_pred_lr_exact_cpu_v3,
    _materialize_disabled_prediction_state_v3,
    project_direct_resident_gru_multicell_checkpoints_v3,
    _validated_capability as _validated_gru_capability,
    stable_gru_packed_width_v3,
)
from engine.perf.cuda_architecture_b_direct_resident_preparation_v1 import (
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
from engine.perf.cuda_architecture_b_direct_resident_wdl_v2 import (
    WDL_ABI_VERSION_V3,
    WDL_AGENT_FLAG_CUED_V3,
    WDL_OBSERVATION_SIZE_V2,
    WDL_STATE_FIELDS_V2,
    WDL_TICKS_PER_GENERATION_V2,
    PreparedDirectResidentWdlV2,
    _allocate_reporting,
    _decode_agent_flag_keys_v3,
    _required_wdl_metadata,
    _validated_capability as _validated_wdl_capability,
    _wdl_history_capacity_v3,
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


WDL_GRU_DIRECT_ENTRYPOINT_V3: Final[str] = (
    "eiel_cuda_architecture_b_execute_direct_resident_wdl_gru_v3"
)
WDL_GRU_PHYSICAL_ENTRYPOINT_V3: Final[str] = (
    "eiel_cuda_architecture_b_execute_direct_resident_wdl_gru_physical_v3"
)
WDL_GRU_OBSERVATION_CHANNELS_V3: Final[tuple[str, ...]] = (
    "legacy_occupancy_or_food",
    "patch_food_norm",
    "patch_cover",
    "patch_shelter",
    "patch_scent_norm",
    "patch_waste_norm",
    "patch_density_norm",
    "patch_movement_cost_norm",
    "wdl_cue",
)
WDL_GRU_OBSERVATION_COLUMNS_V3: Final[tuple[str, ...]] = tuple(
    f"{channel}[dy={dy},dx={dx}]"
    for channel in WDL_GRU_OBSERVATION_CHANNELS_V3
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
) + (
    "agent_energy_over_max_energy",
    "agent_health_over_start_health",
)
WDL_GRU_OBSERVATION_REGISTRY_SHA256_V3: Final[str] = (
    "e8a5eac546d09f302db52503a14a9ca40597e3e6bc9e488d29d9e626035f176f"
)
WDL_GRU_CAPABILITIES_V3: Final[dict[str, bool]] = {
    "flat": False,
    "patch": True,
    "churn": True,
    "wdl": True,
    "jungle": False,
    "gru": True,
    "cuda": False,
}


class CudaArchitectureBDirectResidentWdlGruV3Error(RuntimeError):
    """The exact combined WDL plus stable-GRU boundary is unavailable."""


@dataclass(frozen=True, slots=True)
class PreparedDirectResidentWdlGruV3:
    patch: PreparedDirectResidentTrajectoryV1
    gru_tensors: tuple[torch.Tensor, ...]
    wdl_state_tensors: tuple[torch.Tensor, ...]
    wdl_schedule_tensors: tuple[torch.Tensor, ...]
    wdl_reporting_tensors: tuple[torch.Tensor, ...]
    wdl_config: dict[str, Any]
    initial_projection: Any
    observation_size: int
    generation_rows: int
    source_projections: tuple[Any, ...] = ()
    cell_ids: tuple[str, ...] = ()

    @property
    def batch_cells(self) -> int:
        return int(self.patch.source.batch_cells)


def _fail(message: str) -> CudaArchitectureBDirectResidentWdlGruV3Error:
    return CudaArchitectureBDirectResidentWdlGruV3Error(message)


def wdl_gru_observation_contract_v3() -> dict[str, Any]:
    """Return the frozen 83-column CPU/CUDA composition contract."""

    return {
        "schema_version": "eiel_wdl_gru_observation_registry_v3",
        "column_count": len(WDL_GRU_OBSERVATION_COLUMNS_V3),
        "columns": list(WDL_GRU_OBSERVATION_COLUMNS_V3),
        "registry_sha256": WDL_GRU_OBSERVATION_REGISTRY_SHA256_V3,
        "spatial_order": (
            "channel_major_then_dy_minus1_to_plus1_then_dx_minus1_to_plus1"
        ),
        "gru_input": "all_83_columns_in_registry_order",
        "additional_recurrent_metadata_columns": [],
        "row_mask": "alive_slots_only",
        "disabled_fields": ["prediction_inputs", "modular_recurrent_metadata"],
        "normalization": {
            "legacy_occupancy_or_food": "exact_-1_0_1",
            "patch_food_scent_waste_density": "safe_norm_clamped_0_1",
            "patch_cover_shelter_wdl_cue": "finite_clamped_0_1",
            "patch_movement_cost": "min_max_clamped_0_1",
            "agent_energy": "energy_over_max_energy",
            "agent_health": "health_over_start_health",
        },
    }


def derive_direct_resident_wdl_gru_runner_envelope_v3(runner: Any) -> Any:
    proxy = copy.copy(runner)
    proxy.env_cfg = copy.copy(runner.env_cfg)
    proxy.env_cfg.wdl_delayed_hazard_enable = False
    proxy.env_cfg.wdl_delayed_hazard_observation_enable = False
    return derive_cuda_architecture_b_runner_envelope(
        proxy,
        policy_family="gru",
    )


def _wdl_config(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "enabled": True,
        "schema_version": int(metadata["wdl_delayed_hazard_schema_version"]),
        "delay_steps": int(metadata["wdl_delayed_hazard_delay_steps"]),
        "cue_duration_steps": int(
            metadata["wdl_delayed_hazard_cue_duration_steps"]
        ),
        "active_duration_steps": int(
            metadata["wdl_delayed_hazard_active_duration_steps"]
        ),
        "event_period_steps": int(
            metadata["wdl_delayed_hazard_event_period_steps"]
        ),
        "radius": int(metadata["wdl_delayed_hazard_radius"]),
        "intensity": float(metadata["wdl_delayed_hazard_intensity"]),
        "observation_dropout_rate": float(
            metadata["wdl_delayed_hazard_observation_dropout_rate"]
        ),
        "cue_enabled": bool(metadata["wdl_delayed_hazard_cue_enable"]),
        "observation_enabled": True,
        "damage_enabled": bool(metadata["wdl_delayed_hazard_damage_enable"]),
        "deterministic_schedule_enabled": bool(
            metadata["wdl_delayed_hazard_deterministic_schedule_enable"]
        ),
        "generation_base": 0,
    }


def build_prepared_direct_resident_wdl_gru_v3(
    runner: Any | None = None,
    *,
    qualified_cuda_module: Any,
    logical_generation_count: int = 4,
    agent_id_allocation_generation_count: int | None = None,
    initial_generation: int = 0,
    source_checkpoint_projection: Mapping[str, Any] | None = None,
    per_cell_requests: Sequence[DirectResidentSourceAllocationRequestV1] | None = None,
    source_checkpoint_projections: Sequence[Mapping[str, Any]] | None = None,
    cell_ids: Sequence[str] | None = None,
    task_seeds: Sequence[int] | None = None,
    intensities: Sequence[float] | None = None,
    cuda_device: str = "cuda",
) -> PreparedDirectResidentWdlGruV3:
    """Build B1..B72 WDL and stable-GRU cells around the Patch ABI.

    The original ``runner`` route remains the B1 convenience API.  Multicell
    callers supply complete public allocation requests so this owner does not
    duplicate runner construction or checkpoint restoration mechanics.
    """

    if (
        type(logical_generation_count) is not int
        or logical_generation_count < 1
        or logical_generation_count > 128
    ):
        raise _fail("WDL+GRU preparation requires logical_generation_count in [1,128]")
    if type(initial_generation) is not int or initial_generation < 0:
        raise _fail("WDL+GRU initial_generation must be a nonnegative exact integer")
    if cuda_device not in {"cuda", "cuda:0"}:
        raise _fail("WDL+GRU CUDA source requires cuda or cuda:0")
    gru_capability = _validated_gru_capability(qualified_cuda_module)
    _validated_wdl_capability(qualified_cuda_module)
    if per_cell_requests is None:
        if runner is None:
            raise _fail("WDL+GRU preparation requires runner or per_cell_requests")
        if source_checkpoint_projections is not None:
            raise _fail("B1 runner route uses source_checkpoint_projection")
        envelope = derive_direct_resident_wdl_gru_runner_envelope_v3(runner)
        fresh_projection = build_import_projection(
            runner,
            dict(WDL_GRU_CAPABILITIES_V3),
            generation=initial_generation,
            tick=0,
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
        projection_tensors = (
            projection.get("tensors")
            if isinstance(projection, Mapping)
            else getattr(projection, "tensors", None)
        )
        if not isinstance(projection_metadata, Mapping) or not isinstance(
            projection_tensors, Mapping
        ):
            raise _fail("WDL+GRU source projection is missing metadata or tensors")
        if (
            projection_metadata.get("generation") != initial_generation
            or projection_metadata.get("tick") != 0
        ):
            raise _fail("WDL+GRU source checkpoint is not the requested boundary")
        if not isinstance(projection, Mapping):
            projection.metadata[
                "native_runtime_architecture_b_compact_reporting_only_v1"
            ] = False
            projection.metadata["import_digest"] = projection_digest(
                {"metadata": projection.metadata, "tensors": projection.tensors}
            )
            projection_metadata = projection.metadata
        metadata = _required_wdl_metadata(projection_metadata)
        if metadata.get("native_runtime_fixed_policy_family") != "gru":
            raise _fail("WDL+GRU import projection is not fixed-family stable GRU")
        boundary = import_cpu_owner_boundary_to_cuda(
            projection,
            event_capacity=QUALIFIED_EVENT_CAPACITY_V1,
            device=cuda_device,
            policy_family="stable_gru",
        )
        max_births = int(envelope.max_births_per_tick)
        policy_episode_capacity = required_physical_window_policy_episodes_v1(max_births)
        ordinary_births_enabled = getattr(
            runner.sim_cfg, "ordinary_births_enabled", None
        )
        if ordinary_births_enabled is None:
            ordinary_births_enabled = max_births > 0
        elif type(ordinary_births_enabled) is not bool:
            raise _fail("runner ordinary_births_enabled must be an exact bool")
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
                steps=WDL_TICKS_PER_GENERATION_V2,
            ),
            generation=initial_generation,
            start_tick=0,
            max_births_per_tick=max_births,
            ordinary_births_enabled=ordinary_births_enabled,
            generation_turnover_enabled=True,
        )
        requests = (
            DirectResidentSourceAllocationRequestV1(
                envelope=envelope,
                qualified_cuda_module=qualified_cuda_module,
                cuda_capability=gru_capability,
                import_projection=projection,
                cuda_boundary=boundary,
                generation_birth_state=birth_state,
                configuration=configuration,
                cuda_device=cuda_device,
                logical_generation_count=max(3, logical_generation_count),
                agent_id_allocation_generation_count=(
                    agent_id_allocation_generation_count
                ),
            ),
        )
    else:
        if runner is not None or source_checkpoint_projection is not None:
            raise _fail("multicell request route cannot also supply runner/B1 checkpoint")
        if isinstance(per_cell_requests, (str, bytes, bytearray)) or not isinstance(
            per_cell_requests, Sequence
        ):
            raise _fail("per_cell_requests must be a sequence")
        requests = tuple(per_cell_requests)
        if not 1 <= len(requests) <= 72 or any(
            not isinstance(item, DirectResidentSourceAllocationRequestV1)
            for item in requests
        ):
            raise _fail("per_cell_requests must contain B1..B72 public requests")
        requests = tuple(
            replace(
                item,
                batch_cells=len(requests),
                logical_generation_count=max(3, logical_generation_count),
                agent_id_allocation_generation_count=(
                    agent_id_allocation_generation_count
                ),
            )
            for item in requests
        )
    projections = tuple(item.import_projection for item in requests)
    if source_checkpoint_projections is not None:
        checkpoints = tuple(source_checkpoint_projections)
        if len(checkpoints) != len(requests) or any(
            request.import_projection is not checkpoint
            for request, checkpoint in zip(requests, checkpoints, strict=True)
        ):
            raise _fail("ordered source checkpoints must be bound without substitution")
    metadata_rows = []
    for cell, projection in enumerate(projections):
        metadata = (
            projection.get("metadata")
            if isinstance(projection, Mapping)
            else getattr(projection, "metadata", None)
        )
        if not isinstance(metadata, Mapping):
            raise _fail(f"WDL+GRU source projection {cell} is missing metadata")
        metadata = _required_wdl_metadata(metadata)
        if (
            metadata.get("generation") != initial_generation
            or metadata.get("tick") != 0
            or metadata.get("native_runtime_fixed_policy_family") != "gru"
        ):
            raise _fail(f"WDL+GRU source projection {cell} is not the requested GRU boundary")
        metadata_rows.append(metadata)
    cells = len(requests)
    ids = tuple(str(index) for index in range(cells)) if cell_ids is None else tuple(cell_ids)
    if (
        len(ids) != cells
        or any(type(value) is not str or not value for value in ids)
        or len(set(ids)) != cells
    ):
        raise _fail("cell_ids must contain one unique nonempty string per cell")
    seeds = tuple(row["wdl_base_seed"] for row in metadata_rows)
    if task_seeds is not None and tuple(task_seeds) != seeds:
        raise _fail("task_seeds do not match the ordered WDL source projections")
    source_intensities = tuple(
        float(row["wdl_delayed_hazard_intensity"]) for row in metadata_rows
    )
    if intensities is not None and tuple(float(value) for value in intensities) != source_intensities:
        raise _fail("intensities do not match the ordered WDL source projections")
    if len(set(source_intensities)) != 1:
        raise _fail(
            "mixed per-cell WDL intensity is not representable: native WDL v2 "
            "owns one scalar intensity per physical batch"
        )
    shared_metadata_fields = (
        "wdl_delayed_hazard_schema_version",
        "wdl_delayed_hazard_delay_steps",
        "wdl_delayed_hazard_cue_duration_steps",
        "wdl_delayed_hazard_active_duration_steps",
        "wdl_delayed_hazard_event_period_steps",
        "wdl_delayed_hazard_radius",
        "wdl_delayed_hazard_observation_dropout_rate",
        "wdl_delayed_hazard_cue_enable",
        "wdl_delayed_hazard_damage_enable",
        "wdl_delayed_hazard_deterministic_schedule_enable",
        "gru_cost_base",
        "gru_cost_per_dim",
        "policy_weight_mutation_enable",
        "policy_weight_mutation_mode",
        "policy_weight_mutation_rate",
        "policy_weight_mutation_std",
    )
    for name in shared_metadata_fields:
        expected = metadata_rows[0].get(name)
        if any(row.get(name) != expected for row in metadata_rows[1:]):
            raise _fail(
                f"mixed per-cell {name} is not representable by the native "
                "WDL+GRU batch ABI"
            )
    source = allocate_direct_resident_source_v1(
        requests[0], per_cell_requests=requests if cells > 1 else None
    )
    if any(name not in source.live_state for name in WDL_STATE_FIELDS_V2):
        missing_wdl = {
            cell: [
                name
                for name in WDL_STATE_FIELDS_V2
                if name not in request.cuda_boundary.tensors
            ]
            for cell, request in enumerate(requests)
        }
        missing_wdl = {cell: names for cell, names in missing_wdl.items() if names}
        if missing_wdl:
            raise _fail(
                f"WDL+GRU CUDA boundaries are missing WDL state: {missing_wdl}"
            )
        wdl_live_state = {
            name: torch.stack(
                tuple(request.cuda_boundary.tensors[name] for request in requests),
                dim=0,
            )
            for name in WDL_STATE_FIELDS_V2
        }
        source = replace(
            source,
            live_state={**dict(source.live_state), **wdl_live_state},
            live_banks={**dict(source.live_banks), **wdl_live_state},
        )
    source_device = source.live_state["energy"].device
    slot_capacity = int(source.live_state["energy"].size(1))
    outputs = dict(source.outputs)
    outputs["observations"] = torch.zeros(
        (cells, 1, slot_capacity, WDL_OBSERVATION_SIZE_V2),
        dtype=torch.float32,
        device=source_device,
    )
    source = replace(source, outputs=outputs)
    observations = source.outputs["observations"]
    observation_size = int(observations.size(-1))
    if observation_size != WDL_OBSERVATION_SIZE_V2:
        raise _fail(
            f"WDL+GRU observation width must be 83, actual={observation_size}"
        )
    gru_pack = source.live_state["gru_pack"]
    expected_pack_width = stable_gru_packed_width_v3(observation_size)
    if int(gru_pack.size(-1)) != expected_pack_width:
        raise _fail(
            "WDL+GRU pack disagrees with the 83-column observation contract: "
            f"expected={expected_pack_width} actual={int(gru_pack.size(-1))}"
        )
    device = gru_pack.device
    cells = int(gru_pack.size(0))
    capacity = int(gru_pack.size(1))
    gru_tensors = (
        gru_pack,
        torch.empty(
            (cells, capacity, GRU_SCRATCH_WIDTH_V3),
            dtype=torch.float32,
            device=device,
        ),
        torch.zeros((cells,), dtype=torch.int32, device=device),
    )
    wdl_state_tensors = tuple(
        source.live_state[name] for name in WDL_STATE_FIELDS_V2
    )
    wdl_schedule_tensors = (
        torch.tensor(seeds, dtype=torch.int64, device=device),
        torch.tensor(
            [row["wdl_next_absolute_step"] for row in metadata_rows],
            dtype=torch.int64,
            device=device,
        ),
    )
    slot_capacity = int(source.live_state["energy"].size(1))
    max_births_per_tick = int(requests[0].configuration.max_births_per_tick)
    if any(
        int(request.configuration.max_births_per_tick) != max_births_per_tick
        for request in requests[1:]
    ):
        raise _fail(
            "mixed max_births_per_tick is not representable by the WDL v3 "
            "compact-history batch ABI"
        )
    history_capacity = _wdl_history_capacity_v3(
        slot_capacity=slot_capacity,
        ticks_per_generation=WDL_TICKS_PER_GENERATION_V2,
        max_births_per_tick=max_births_per_tick,
    )
    b1_reporting = _allocate_reporting(
        device=device,
        generation_rows=logical_generation_count,
        history_capacity=history_capacity,
    )
    wdl_reporting_tensors = tuple(
        torch.zeros(
            (cells, *tensor.shape[1:]), dtype=tensor.dtype, device=device
        )
        for tensor in b1_reporting
    )
    wdl_config = _wdl_config(metadata_rows[0])
    wdl_config["generation_base"] = initial_generation
    wdl_config.update(
        history_abi_version=WDL_ABI_VERSION_V3,
        history_capacity=history_capacity,
        slot_capacity=slot_capacity,
        max_births_per_tick=max_births_per_tick,
        ticks_per_generation=WDL_TICKS_PER_GENERATION_V2,
    )
    return PreparedDirectResidentWdlGruV3(
        patch=prepare_direct_resident_trajectory_v1(source),
        gru_tensors=gru_tensors,
        wdl_state_tensors=wdl_state_tensors,
        wdl_schedule_tensors=wdl_schedule_tensors,
        wdl_reporting_tensors=wdl_reporting_tensors,
        wdl_config=wdl_config,
        initial_projection=projections[0],
        observation_size=observation_size,
        generation_rows=logical_generation_count,
        source_projections=projections,
        cell_ids=ids,
    )


def _gru_config(prepared: PreparedDirectResidentWdlGruV3) -> dict[str, float]:
    metadata = (
        prepared.initial_projection.get("metadata")
        if isinstance(prepared.initial_projection, Mapping)
        else prepared.initial_projection.metadata
    )
    return {
        "upkeep_cost_base": float(metadata.get("gru_cost_base", 0.0)),
        "upkeep_cost_per_dim": float(metadata.get("gru_cost_per_dim", 0.0)),
    }


def _validate_result(
    result: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    surface: str,
) -> dict[str, Any]:
    out = dict(result)
    for name, expected_value in expected.items():
        actual = out.get(name)
        if type(actual) is not type(expected_value) or actual != expected_value:
            raise _fail(
                f"WDL+GRU {surface} result {name} mismatch: "
                f"expected={expected_value!r} actual={actual!r}"
            )
    return out


def execute_direct_resident_wdl_gru_physical_v3(
    module: Any,
    prepared: PreparedDirectResidentWdlGruV3,
    *,
    generation_count: int,
    ticks_per_generation: int = WDL_TICKS_PER_GENERATION_V2,
) -> dict[str, Any]:
    """Execute one combined native-owned physical g1 or g2 segment."""

    if not isinstance(prepared, PreparedDirectResidentWdlGruV3):
        raise _fail("prepared must be PreparedDirectResidentWdlGruV3")
    if generation_count not in GRU_PHYSICAL_GENERATION_COUNTS_V3:
        raise _fail("physical WDL+GRU qualification requires generation_count 1 or 2")
    if generation_count > prepared.generation_rows:
        raise _fail("WDL+GRU reporting allocation does not cover the request")
    if (
        type(ticks_per_generation) is not int
        or ticks_per_generation < 1
        or ticks_per_generation > WDL_TICKS_PER_GENERATION_V2
    ):
        raise _fail("physical WDL+GRU ticks_per_generation must be in [1,80]")
    entrypoint = getattr(module, WDL_GRU_PHYSICAL_ENTRYPOINT_V3, None)
    if not callable(entrypoint):
        raise _fail(f"CUDA extension is missing callable {WDL_GRU_PHYSICAL_ENTRYPOINT_V3}")
    arguments = list(prepared.patch.arguments)
    physical_ticks = generation_count * ticks_per_generation
    arguments[PHYSICAL_RUNTIME_BOOKKEEPING_ARGUMENT_INDEX_V3] = arguments[
        PHYSICAL_RUNTIME_BOOKKEEPING_ARGUMENT_INDEX_V3
    ][:, :physical_ticks]
    full_tick_outputs = list(
        arguments[PHYSICAL_FULL_TICK_OUTPUTS_ARGUMENT_INDEX_V3]
    )
    full_tick_outputs[-1] = full_tick_outputs[-1][:generation_count]
    arguments[PHYSICAL_FULL_TICK_OUTPUTS_ARGUMENT_INDEX_V3] = full_tick_outputs
    patch_stats = list(arguments[PHYSICAL_PATCH_STATS_ARGUMENT_INDEX_V3])
    patch_stats[0] = patch_stats[0][:, :physical_ticks]
    arguments[PHYSICAL_PATCH_STATS_ARGUMENT_INDEX_V3] = patch_stats
    arguments[TICKS_ARGUMENT_INDEX_V3] = ticks_per_generation
    arguments[GENERATION_COUNT_ARGUMENT_INDEX_V1] = generation_count
    raw = entrypoint(
        list(prepared.gru_tensors),
        _gru_config(prepared),
        list(prepared.wdl_state_tensors),
        list(prepared.wdl_schedule_tensors),
        list(prepared.wdl_reporting_tensors),
        dict(prepared.wdl_config),
        *arguments,
    )
    if not isinstance(raw, Mapping):
        raise _fail("WDL+GRU physical CUDA entrypoint returned a non-mapping result")
    result = _validate_result(
        raw,
        {
            "gru_native_owner": True,
            "gru_native_controller_abi_version": GRU_ABI_VERSION_V3,
            "wdl_native_owner": True,
            "wdl_native_controller_abi_version": WDL_ABI_VERSION_V3,
            "wdl_gru_physical_gate_entrypoint": True,
            "gru_python_tick_or_generation_ownership": False,
            "wdl_python_tick_or_generation_ownership": False,
            "hidden_fallback": False,
            "runtime_jit": False,
        },
        surface="physical",
    )
    if int(prepared.gru_tensors[2].detach().cpu().item()) != 0:
        raise _fail("WDL+GRU physical GRU device status is nonzero")
    if int(torch.count_nonzero(prepared.wdl_reporting_tensors[4]).item()) != 0:
        raise _fail("WDL+GRU physical WDL device status is nonzero")
    return result


def execute_direct_resident_wdl_gru_v3(
    module: Any,
    prepared: PreparedDirectResidentWdlGruV3,
    *,
    logical_generation_count: int,
    run_id: str,
    run_control: Any | None = None,
) -> dict[str, Any]:
    """Execute one combined trajectory through native reusable g2 windows."""

    if not isinstance(prepared, PreparedDirectResidentWdlGruV3):
        raise _fail("prepared must be PreparedDirectResidentWdlGruV3")
    if (
        type(logical_generation_count) is not int
        or logical_generation_count < 3
        or logical_generation_count > 128
    ):
        raise _fail("resident WDL+GRU requires logical_generation_count in [3,128]")
    if logical_generation_count != prepared.generation_rows:
        raise _fail("WDL+GRU reporting horizon does not match the request")
    if prepared.patch.source.logical_generation_count != logical_generation_count:
        raise _fail("WDL+GRU source identity capacity does not match the request")
    entrypoint = getattr(module, WDL_GRU_DIRECT_ENTRYPOINT_V3, None)
    if not callable(entrypoint):
        raise _fail(f"CUDA extension is missing callable {WDL_GRU_DIRECT_ENTRYPOINT_V3}")
    control = (
        create_run_control_v2(module, run_id=run_id)
        if run_control is None
        else run_control
    )
    fresh = dict(control.manifest())
    if (
        fresh.get("run_id") != run_id
        or fresh.get("status") != "initialized"
        or fresh.get("run_started") is not False
        or fresh.get("sealed") is not False
    ):
        raise _fail("WDL+GRU run control must be fresh and match run_id")
    metadata = dict(
        prepared.initial_projection.get("metadata")
        if isinstance(prepared.initial_projection, Mapping)
        else prepared.initial_projection.metadata
    )
    raw = entrypoint(
        control,
        list(prepared.gru_tensors),
        _gru_config(prepared),
        list(prepared.wdl_state_tensors),
        list(prepared.wdl_schedule_tensors),
        list(prepared.wdl_reporting_tensors),
        dict(prepared.wdl_config),
        *prepared.patch.arguments,
        logical_generation_count,
        metadata.get("policy_weight_mutation_enable"),
        metadata.get("policy_weight_mutation_mode"),
        metadata.get("policy_weight_mutation_rate"),
        metadata.get("policy_weight_mutation_std"),
    )
    if not isinstance(raw, Mapping):
        raise _fail("WDL+GRU resident CUDA entrypoint returned a non-mapping result")
    completed = raw.get("completed_generations")
    if (
        type(completed) is not int
        or completed < 1
        or completed > logical_generation_count
    ):
        raise _fail("WDL+GRU resident result is not an exact completed prefix")
    result = _validate_result(
        raw,
        {
            "gru_native_owner": True,
            "gru_native_controller_abi_version": GRU_ABI_VERSION_V3,
            "wdl_native_owner": True,
            "wdl_native_controller_abi_version": WDL_ABI_VERSION_V3,
            "gru_python_tick_or_generation_ownership": False,
            "wdl_python_tick_or_generation_ownership": False,
            "completed_generations": completed,
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
    if int(prepared.gru_tensors[2].detach().cpu().item()) != 0:
        raise _fail("WDL+GRU resident GRU device status is nonzero")
    if int(torch.count_nonzero(prepared.wdl_reporting_tensors[4]).item()) != 0:
        raise _fail("WDL+GRU resident WDL device status is nonzero")
    manifest = dict(control.manifest())
    if (
        manifest.get("status") != "generation_complete"
        or manifest.get("requested_generations") != logical_generation_count
        or manifest.get("active_generation")
        != prepared.patch.source.generation_base + completed - 1
        or manifest.get("sealed") is not False
    ):
        raise _fail("WDL+GRU run-control manifest is not at a completed boundary")
    if completed == logical_generation_count:
        if (
            manifest.get("awaiting_final_output_ack") is not True
            or manifest.get("awaiting_cancellation_checkpoint") is not False
            or manifest.get("cancel_requested") is not False
        ):
            raise _fail("WDL+GRU complete run is not awaiting final output ack")
    elif (
        manifest.get("awaiting_final_output_ack") is not False
        or manifest.get("awaiting_cancellation_checkpoint") is not True
        or manifest.get("cancel_requested") is not True
    ):
        raise _fail("WDL+GRU partial run is not awaiting cancellation checkpoint")
    result["native_run_control_v2"] = manifest
    result["cancelled_prefix"] = completed < logical_generation_count
    return result


def project_direct_resident_wdl_gru_checkpoint_v3(
    prepared: PreparedDirectResidentWdlGruV3,
    *,
    direct_result: Mapping[str, Any],
    completed_generation_count: int,
) -> Mapping[str, Any]:
    """Project one combined prefix through the accepted checkpoint contract."""

    if not isinstance(prepared, PreparedDirectResidentWdlGruV3):
        raise _fail("prepared must be PreparedDirectResidentWdlGruV3")
    if prepared.batch_cells != 1:
        raise _fail("multicell WDL+GRU owners require the ordered checkpoint projector")
    if (
        type(completed_generation_count) is not int
        or completed_generation_count < 1
        or completed_generation_count > prepared.generation_rows
    ):
        raise _fail("WDL+GRU checkpoint projection exceeds its resident horizon")
    result = dict(direct_result)
    outputs = dict(prepared.patch.source.outputs)
    generation_summary = result.get("generation_summary_scalar")
    if not isinstance(generation_summary, torch.Tensor):
        generation_summary = prepared.patch.arguments[
            PHYSICAL_FULL_TICK_OUTPUTS_ARGUMENT_INDEX_V3
        ][-1][:completed_generation_count]
    runtime_bookkeeping = result.get("runtime_bookkeeping_by_tick")
    if not isinstance(runtime_bookkeeping, torch.Tensor):
        runtime_bookkeeping = prepared.patch.arguments[
            PHYSICAL_RUNTIME_BOOKKEEPING_ARGUMENT_INDEX_V3
        ][:, : completed_generation_count * WDL_TICKS_PER_GENERATION_V2]
    outputs["generation_summary_scalar"] = generation_summary
    outputs["runtime_bookkeeping_by_tick"] = runtime_bookkeeping
    persistent_state = dict(
        _with_materialized_event_history(prepared.patch.live_banks, result)
    )
    for name, tensor in zip(
        WDL_STATE_FIELDS_V2,
        prepared.wdl_state_tensors,
        strict=True,
    ):
        persistent_state[name] = tensor
    initial_projection = prepared.initial_projection
    if isinstance(initial_projection, Mapping):
        initial_projection = SimpleNamespace(
            metadata=initial_projection["metadata"],
            tensors=initial_projection["tensors"],
        )
    gru_prepared = PreparedDirectResidentGruV3(
        patch=prepared.patch,
        gru_tensors=prepared.gru_tensors,
        initial_projection=initial_projection,
        observation_size=prepared.observation_size,
    )
    persistent_state = _materialize_disabled_prediction_state_v3(
        gru_prepared,
        persistent_state,
    )
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
    metadata["prediction_tick"] = WDL_TICKS_PER_GENERATION_V2
    metadata["wdl_next_absolute_step"] = int(
        prepared.wdl_schedule_tensors[1].detach().cpu().item()
    )
    metadata["native_runtime_wdl_action_context_generation"] = (
        prepared.patch.source.generation_base + completed_generation_count - 1
    )
    agent_keys = prepared.wdl_reporting_tensors[6][
        0, completed_generation_count - 1
    ]
    agent_flags = prepared.wdl_reporting_tensors[7][
        0, completed_generation_count - 1
    ]
    metadata["wdl_agents_cued_current_generation_agent_ids"] = (
        _decode_agent_flag_keys_v3(
            agent_keys.detach().cpu(),
            agent_flags.detach().cpu(),
            flag=WDL_AGENT_FLAG_CUED_V3,
        )
    )
    projection["metadata"] = metadata
    tensors = dict(projection["tensors"])
    tensors["pred_lr_exact"] = _canonical_pred_lr_exact_cpu_v3(
        tensors["pred_lr"],
        tensors["alive_mask"],
    )
    projection["tensors"] = tensors
    return _with_checkpoint_contract_metadata(projection)


def project_direct_resident_wdl_gru_multicell_checkpoints_v3(
    prepared: PreparedDirectResidentWdlGruV3,
    *,
    direct_result: Mapping[str, Any],
    completed_generation_count: int,
    ordered_source_projections: Sequence[Any] | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Project ordered B1..B72 WDL+GRU checkpoints without changing owners."""

    if not isinstance(prepared, PreparedDirectResidentWdlGruV3):
        raise _fail("prepared must be PreparedDirectResidentWdlGruV3")
    projections = (
        prepared.source_projections
        if ordered_source_projections is None
        else tuple(ordered_source_projections)
    )
    if len(projections) != prepared.batch_cells:
        raise _fail("ordered source projection count must match resident cells")
    gru_prepared = PreparedDirectResidentGruV3(
        patch=prepared.patch,
        gru_tensors=prepared.gru_tensors,
        initial_projection=prepared.initial_projection,
        observation_size=prepared.observation_size,
    )
    checkpoints = project_direct_resident_gru_multicell_checkpoints_v3(
        gru_prepared,
        direct_result=direct_result,
        completed_generation_count=completed_generation_count,
        ordered_source_projections=projections,
    )
    out: list[Mapping[str, Any]] = []
    for cell, checkpoint in enumerate(checkpoints):
        projection = dict(checkpoint)
        metadata = dict(projection["metadata"])
        metadata["wdl_next_absolute_step"] = int(
            prepared.wdl_schedule_tensors[1][cell].detach().cpu().item()
        )
        metadata["native_runtime_wdl_action_context_generation"] = (
            prepared.patch.source.generation_base
            + completed_generation_count
            - 1
        )
        agent_keys = prepared.wdl_reporting_tensors[6][
            cell, completed_generation_count - 1
        ]
        agent_flags = prepared.wdl_reporting_tensors[7][
            cell, completed_generation_count - 1
        ]
        metadata["wdl_agents_cued_current_generation_agent_ids"] = (
            _decode_agent_flag_keys_v3(
                agent_keys.detach().cpu(),
                agent_flags.detach().cpu(),
                flag=WDL_AGENT_FLAG_CUED_V3,
            )
        )
        projection["metadata"] = metadata
        tensors = projection.get("tensors")
        if not isinstance(tensors, Mapping):
            raise _fail(f"WDL+GRU checkpoint {cell} is missing tensors")
        missing = [name for name in WDL_STATE_FIELDS_V2 if name not in tensors]
        if missing:
            raise _fail(f"WDL+GRU checkpoint {cell} is missing WDL state: {missing}")
        out.append(_with_checkpoint_contract_metadata(projection))
    return tuple(out)


def as_wdl_prepared_v2(
    prepared: PreparedDirectResidentWdlGruV3,
) -> PreparedDirectResidentWdlV2:
    """Expose the combined WDL surfaces to existing bounded comparison helpers."""

    return PreparedDirectResidentWdlV2(
        patch=prepared.patch,
        state_tensors=prepared.wdl_state_tensors,
        schedule_tensors=prepared.wdl_schedule_tensors,
        reporting_tensors=prepared.wdl_reporting_tensors,
        config=prepared.wdl_config,
        generation_rows=prepared.generation_rows,
        initial_projection=prepared.initial_projection,
    )


__all__ = [
    "WDL_GRU_DIRECT_ENTRYPOINT_V3",
    "WDL_GRU_OBSERVATION_COLUMNS_V3",
    "WDL_GRU_OBSERVATION_REGISTRY_SHA256_V3",
    "WDL_GRU_PHYSICAL_ENTRYPOINT_V3",
    "CudaArchitectureBDirectResidentWdlGruV3Error",
    "PreparedDirectResidentWdlGruV3",
    "as_wdl_prepared_v2",
    "build_prepared_direct_resident_wdl_gru_v3",
    "execute_direct_resident_wdl_gru_physical_v3",
    "execute_direct_resident_wdl_gru_v3",
    "project_direct_resident_wdl_gru_checkpoint_v3",
    "project_direct_resident_wdl_gru_multicell_checkpoints_v3",
    "wdl_gru_observation_contract_v3",
]
