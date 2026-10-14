"""Build and execute the stable-GRU Patch resident CUDA controller."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, Final

import torch

from engine.perf.cuda_architecture_b_disabled_prediction_state_v1 import (
    DISABLED_PREDICTION_STATE_FIELDS_V1,
    materialize_disabled_prediction_state_v1,
)

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
from engine.perf.cuda_architecture_b_direct_resident_preparation_v1 import (
    DIRECT_RESIDENT_ARGUMENT_NAMES_V1,
    GENERATION_COUNT_ARGUMENT_INDEX_V1,
    PreparedDirectResidentTrajectoryV1,
    prepare_direct_resident_trajectory_v1,
)
from engine.perf.cuda_architecture_b_direct_resident_source_allocator_v1 import (
    RESIDENT_AGENT_ID_CAPACITY_METADATA_V1,
    CANONICAL_RUNTIME_PREFIX_FIELDS,
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
from engine.perf.cuda_architecture_b_direct_resident_settlement_v1 import (
    _project_terminal_checkpoint,
    _with_materialized_event_history,
)
from engine.perf.cuda_architecture_b_generation import CudaGenerationConfig
from engine.perf.cxxbackbone10_resident_state_owner import (
    _with_checkpoint_contract_metadata,
    build_import_projection,
    projection_digest,
)
from perf.cuda_architecture_b_boundary import (
    EVENT_COLUMN_NAMES,
    CHECKPOINT_PROJECTION_SCHEMA_VERSION,
    import_cpu_owner_boundary_to_cuda,
)
from perf.cuda_architecture_b_generation_birth_state import (
    prepare_generation_birth_state_cuda,
)


GRU_ABI_VERSION_V3: Final[int] = 3
GRU_HIDDEN_WIDTH_V3: Final[int] = 64
GRU_ACTION_COUNT_V3: Final[int] = 7
GRU_SCRATCH_WIDTH_V3: Final[int] = 519
GRU_CAPABILITY_ENTRYPOINT_V3: Final[str] = (
    "eiel_cuda_architecture_b_flattened_gru_capability_v3"
)
GRU_DIRECT_ENTRYPOINT_V3: Final[str] = (
    "eiel_cuda_architecture_b_execute_direct_resident_gru_v3"
)
GRU_PHYSICAL_ENTRYPOINT_V3: Final[str] = (
    "eiel_cuda_architecture_b_execute_direct_resident_gru_physical_v3"
)
GRU_RESIDENT_GENERATION_COUNTS_V3: Final[frozenset[int]] = frozenset(range(3, 129))
GRU_PHYSICAL_GENERATION_COUNTS_V3: Final[frozenset[int]] = frozenset({1, 2})
PHYSICAL_RUNTIME_BOOKKEEPING_ARGUMENT_INDEX_V3: Final[int] = (
    DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index("physical_runtime_bookkeeping")
)
PHYSICAL_FULL_TICK_OUTPUTS_ARGUMENT_INDEX_V3: Final[int] = (
    DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index("physical_full_tick_output_tensors")
)
PHYSICAL_PATCH_STATS_ARGUMENT_INDEX_V3: Final[int] = (
    DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index("physical_patch_stats_tensors")
)
TICKS_ARGUMENT_INDEX_V3: Final[int] = DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index(
    "ticks_per_generation"
)
GRU_CAPABILITIES_V3: Final[dict[str, bool]] = {
    "flat": False,
    "patch": True,
    "churn": True,
    "wdl": False,
    "jungle": False,
    "gru": True,
    "cuda": False,
}
GRU_DISABLED_PREDICTION_STATE_FIELDS_V3: Final[tuple[str, ...]] = (
    DISABLED_PREDICTION_STATE_FIELDS_V1
)


class CudaArchitectureBDirectResidentGruV3Error(RuntimeError):
    """The exact stable-GRU resident source or execution boundary is unavailable."""


@dataclass(frozen=True, slots=True)
class PreparedDirectResidentGruV3:
    patch: PreparedDirectResidentTrajectoryV1
    gru_tensors: tuple[torch.Tensor, ...]
    initial_projection: Any
    observation_size: int
    source_projections: tuple[Any, ...] = ()
    cell_ids: tuple[str, ...] = ()

    @property
    def batch_cells(self) -> int:
        return int(self.patch.source.batch_cells)


def _fail(message: str) -> CudaArchitectureBDirectResidentGruV3Error:
    return CudaArchitectureBDirectResidentGruV3Error(message)


def stable_gru_packed_width_v3(observation_size: int) -> int:
    if type(observation_size) is not int or observation_size < 1:
        raise _fail("stable-GRU observation size must be a positive exact integer")
    hidden = GRU_HIDDEN_WIDTH_V3
    actions = GRU_ACTION_COUNT_V3
    return (
        hidden * observation_size
        + hidden
        + 3 * hidden * hidden
        + 3 * hidden
        + 3 * hidden * hidden
        + 3 * hidden
        + actions * hidden
        + actions
    )


def _validated_capability(module: Any) -> dict[str, Any]:
    entrypoint = getattr(module, GRU_CAPABILITY_ENTRYPOINT_V3, None)
    if not callable(entrypoint):
        raise _fail(
            f"CUDA extension is missing callable {GRU_CAPABILITY_ENTRYPOINT_V3}"
        )
    raw = entrypoint()
    if not isinstance(raw, Mapping):
        raise _fail("GRU CUDA capability must be a mapping")
    capability = dict(raw)
    expected = {
        "abi_version": GRU_ABI_VERSION_V3,
        "policy_family": "stable_gru",
        "hidden_width": GRU_HIDDEN_WIDTH_V3,
        "action_count": GRU_ACTION_COUNT_V3,
        "resident_hidden_state": True,
        "cache_refill_rng_contract": "one_gumbel_vector_then_repeat",
        "native_tick_ownership": True,
        "runtime_jit_required": False,
        "production_qualified": False,
    }
    for name, expected_value in expected.items():
        actual = capability.get(name)
        if type(actual) is not type(expected_value) or actual != expected_value:
            raise _fail(
                f"GRU CUDA capability {name} mismatch: "
                f"expected={expected_value!r} actual={actual!r}"
            )
    return capability


def _checkpoint_runtime_fields_v3(
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    fields = {name: checkpoint[name] for name in CANONICAL_RUNTIME_PREFIX_FIELDS}
    for name in ("tensor_manifest", "event_cursor"):
        if name in checkpoint:
            fields[name] = checkpoint[name]
    return fields


def _gru_patch_envelope(runner: Any) -> CudaArchitectureBRunnerEnvelope:
    return derive_cuda_architecture_b_runner_envelope(
        runner,
        policy_family="gru",
    )


def _restore_checkpoint_gru_upkeep_cost_v3(
    source: Any,
    checkpoint_projection: Mapping[str, Any],
) -> None:
    """Restore the authoritative resident GRU charge lost by Agent materialization."""

    tensors = checkpoint_projection.get("tensors")
    if not isinstance(tensors, Mapping):
        raise _fail("GRU source checkpoint tensors must be a mapping")
    checkpoint = tensors.get("gru_upkeep_cost")
    live_state = getattr(source, "live_state", None)
    target = (
        live_state.get("gru_upkeep_cost") if isinstance(live_state, Mapping) else None
    )
    if not isinstance(checkpoint, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise _fail("GRU source checkpoint must contain resident gru_upkeep_cost")
    if checkpoint.device.type != "cpu" or checkpoint.requires_grad:
        raise _fail("checkpoint gru_upkeep_cost must be detached CPU storage")
    if (
        checkpoint.dtype != torch.float32
        or checkpoint.ndim != 1
        or target.dtype != checkpoint.dtype
        or tuple(target.shape) != (1, checkpoint.numel())
    ):
        raise _fail("checkpoint gru_upkeep_cost shape or dtype mismatch")
    with torch.no_grad():
        target.copy_(checkpoint.unsqueeze(0).to(device=target.device))


def build_prepared_direct_resident_gru_v3(
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
    cuda_device: str = "cuda",
    batch_cells: int = 1,
    observation_extra_columns: int = 0,
    output_ticks: int = 1,
    immutable_shared_gru_pack: torch.Tensor | None = None,
) -> PreparedDirectResidentGruV3:
    """Build one stable-GRU overlay around the unchanged Patch resident ABI."""

    if (
        type(logical_generation_count) is not int
        or logical_generation_count < 1
        or logical_generation_count > 128
    ):
        raise _fail("GRU preparation requires logical_generation_count in [1,128]")
    if type(initial_generation) is not int or initial_generation < 0:
        raise _fail("GRU initial_generation must be a nonnegative exact integer")
    if cuda_device not in {"cuda", "cuda:0"}:
        raise _fail("GRU CUDA source requires cuda or cuda:0")
    capability = _validated_capability(qualified_cuda_module)
    if per_cell_requests is None:
        if runner is None:
            raise _fail("GRU preparation requires runner or per_cell_requests")
        if source_checkpoint_projections is not None:
            raise _fail("B1 runner route uses source_checkpoint_projection")
        if type(batch_cells) is not int or batch_cells != 1:
            raise _fail("runner convenience route is B1; use per_cell_requests for multicell")
        envelope = _gru_patch_envelope(runner)
        fresh_projection = build_import_projection(
            runner,
            dict(GRU_CAPABILITIES_V3),
            generation=initial_generation,
            tick=0,
        )
        projection: Any = (
            fresh_projection
            if source_checkpoint_projection is None
            else source_checkpoint_projection
        )
        if source_checkpoint_projection is not None:
            checkpoint_metadata = source_checkpoint_projection.get("metadata")
            if (
                not isinstance(checkpoint_metadata, Mapping)
                or checkpoint_metadata.get("generation") != initial_generation
                or checkpoint_metadata.get("tick") != 0
            ):
                raise _fail("GRU source checkpoint is not the requested generation boundary")
        if isinstance(projection, Mapping):
            projection_metadata = projection.get("metadata")
            projection_tensors = projection.get("tensors")
        else:
            projection_metadata = projection.metadata
            projection_tensors = projection.tensors
        if not isinstance(projection_metadata, Mapping) or not isinstance(
            projection_tensors, Mapping
        ):
            raise _fail("GRU source projection is missing metadata or tensors")
        if not isinstance(projection, Mapping):
            projection.metadata[
                "native_runtime_architecture_b_compact_reporting_only_v1"
            ] = False
            projection.metadata["import_digest"] = projection_digest(
                {"metadata": projection.metadata, "tensors": projection.tensors}
            )
            projection_metadata = projection.metadata
        boundary = import_cpu_owner_boundary_to_cuda(
            projection,
            event_capacity=QUALIFIED_EVENT_CAPACITY_V1,
            device=cuda_device,
            policy_family="stable_gru",
        )
        max_births = int(envelope.max_births_per_tick)
        ordinary_births_enabled = getattr(
            runner.sim_cfg, "ordinary_births_enabled", None
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
                projection_metadata, steps=PHYSICAL_WINDOW_TICKS_V1
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
                cuda_capability=capability,
                import_projection=projection,
                cuda_boundary=boundary,
                generation_birth_state=birth_state,
                configuration=configuration,
                cuda_device=cuda_device,
                logical_generation_count=max(3, logical_generation_count),
                agent_id_allocation_generation_count=agent_id_allocation_generation_count,
                observation_extra_columns=observation_extra_columns,
                output_ticks=output_ticks,
                immutable_shared_gru_pack=immutable_shared_gru_pack,
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
                agent_id_allocation_generation_count=agent_id_allocation_generation_count,
                observation_extra_columns=observation_extra_columns,
                output_ticks=output_ticks,
                immutable_shared_gru_pack=immutable_shared_gru_pack,
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
    admitted_policy_families = (
        {"gru"} if per_cell_requests is None else {"mlp", "gru", "mixed"}
    )
    for cell, projection in enumerate(projections):
        metadata = (
            projection.get("metadata")
            if isinstance(projection, Mapping)
            else getattr(projection, "metadata", None)
        )
        if (
            not isinstance(metadata, Mapping)
            or metadata.get("generation") != initial_generation
            or metadata.get("tick") != 0
            or str(metadata.get("native_runtime_fixed_policy_family", ""))
            .strip()
            .lower()
            not in admitted_policy_families
        ):
            raise _fail(f"GRU source projection {cell} is not the requested boundary")
    shared_fields = (
        "gru_cost_base",
        "gru_cost_per_dim",
        "policy_weight_mutation_enable",
        "policy_weight_mutation_mode",
        "policy_weight_mutation_rate",
        "policy_weight_mutation_std",
    )
    metadata_rows = tuple(
        projection.get("metadata") if isinstance(projection, Mapping) else projection.metadata
        for projection in projections
    )
    for name in shared_fields:
        expected = metadata_rows[0].get(name)
        if any(row.get(name) != expected for row in metadata_rows[1:]):
            raise _fail(f"mixed per-cell {name} is not representable by the native GRU batch ABI")
    cells = len(requests)
    ids = tuple(str(index) for index in range(cells)) if cell_ids is None else tuple(cell_ids)
    if (
        len(ids) != cells
        or any(type(value) is not str or not value for value in ids)
        or len(set(ids)) != cells
    ):
        raise _fail("cell_ids must contain one unique nonempty string per cell")
    source = allocate_direct_resident_source_v1(
        requests[0], per_cell_requests=requests if cells > 1 else None
    )
    if source_checkpoint_projection is not None:
        _restore_checkpoint_gru_upkeep_cost_v3(source, source_checkpoint_projection)
    gru_pack = source.live_state["gru_pack"]
    observations = source.outputs["observations"]
    observation_size = int(observations.size(-1))
    expected_pack_width = stable_gru_packed_width_v3(observation_size)
    actual_pack_width = int(gru_pack.size(-1))
    if actual_pack_width != expected_pack_width:
        raise _fail(
            "GRU policy pack disagrees with the resident observation contract: "
            f"expected={expected_pack_width} actual={actual_pack_width}"
        )
    device = gru_pack.device
    cells = int(source.live_state["energy"].size(0))
    capacity = int(source.live_state["energy"].size(1))
    gru_tensors = (
        gru_pack,
        torch.empty(
            (cells, capacity, GRU_SCRATCH_WIDTH_V3),
            dtype=torch.float32,
            device=device,
        ),
        torch.zeros((cells,), dtype=torch.int32, device=device),
    )
    return PreparedDirectResidentGruV3(
        patch=prepare_direct_resident_trajectory_v1(source),
        gru_tensors=gru_tensors,
        initial_projection=projections[0],
        observation_size=observation_size,
        source_projections=projections,
        cell_ids=ids,
    )


def execute_direct_resident_gru_v3(
    module: Any,
    prepared: PreparedDirectResidentGruV3,
    *,
    logical_generation_count: int,
    run_id: str,
) -> dict[str, Any]:
    """Execute one complete stable-GRU trajectory through native g2 windows."""

    if not isinstance(prepared, PreparedDirectResidentGruV3):
        raise _fail("prepared must be PreparedDirectResidentGruV3")
    if logical_generation_count not in GRU_RESIDENT_GENERATION_COUNTS_V3:
        raise _fail("resident GRU requires logical_generation_count in [3,128]")
    if prepared.patch.source.logical_generation_count != logical_generation_count:
        raise _fail("GRU source identity capacity does not match the requested horizon")
    entrypoint = getattr(module, GRU_DIRECT_ENTRYPOINT_V3, None)
    if not callable(entrypoint):
        raise _fail(f"CUDA extension is missing callable {GRU_DIRECT_ENTRYPOINT_V3}")
    control = create_run_control_v2(module, run_id=run_id)
    metadata_value = _projection_member_v3(prepared.initial_projection, "metadata")
    if not isinstance(metadata_value, Mapping):
        raise _fail("GRU initial projection metadata is not a mapping")
    metadata = dict(metadata_value)
    raw = entrypoint(
        control,
        list(prepared.gru_tensors),
        {
            "upkeep_cost_base": float(metadata.get("gru_cost_base", 0.0)),
            "upkeep_cost_per_dim": float(metadata.get("gru_cost_per_dim", 0.0)),
        },
        *prepared.patch.arguments,
        logical_generation_count,
        metadata.get("policy_weight_mutation_enable"),
        metadata.get("policy_weight_mutation_mode"),
        metadata.get("policy_weight_mutation_rate"),
        metadata.get("policy_weight_mutation_std"),
    )
    if not isinstance(raw, Mapping):
        raise _fail("GRU resident CUDA entrypoint returned a non-mapping result")
    result = dict(raw)
    expected = {
        "gru_native_owner": True,
        "gru_native_controller_abi_version": GRU_ABI_VERSION_V3,
        "gru_python_tick_or_generation_ownership": False,
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
                f"GRU resident result {name} mismatch: "
                f"expected={expected_value!r} actual={actual!r}"
            )
    if int(prepared.gru_tensors[2].detach().cpu().item()) != 0:
        raise _fail("GRU resident device status is nonzero")
    manifest = dict(control.manifest())
    if (
        manifest.get("status") != "generation_complete"
        or manifest.get("requested_generations") != logical_generation_count
        or manifest.get("active_generation")
        != prepared.patch.source.generation_base + logical_generation_count - 1
        or manifest.get("awaiting_final_output_ack") is not True
        or manifest.get("sealed") is not False
    ):
        raise _fail("GRU resident run-control manifest is not at the final boundary")
    result["native_run_control_v2"] = manifest
    return result


def execute_direct_resident_gru_physical_v3(
    module: Any,
    prepared: PreparedDirectResidentGruV3,
    *,
    generation_count: int,
    ticks_per_generation: int = PHYSICAL_WINDOW_TICKS_V1 // 2,
) -> dict[str, Any]:
    """Execute one native-owned physical g1 or g2 stable-GRU segment."""

    if not isinstance(prepared, PreparedDirectResidentGruV3):
        raise _fail("prepared must be PreparedDirectResidentGruV3")
    if generation_count not in GRU_PHYSICAL_GENERATION_COUNTS_V3:
        raise _fail("physical GRU qualification requires generation_count 1 or 2")
    if (
        type(ticks_per_generation) is not int
        or ticks_per_generation < 1
        or ticks_per_generation > PHYSICAL_WINDOW_TICKS_V1 // 2
    ):
        raise _fail("physical GRU ticks_per_generation must be in [1,80]")
    entrypoint = getattr(module, GRU_PHYSICAL_ENTRYPOINT_V3, None)
    if not callable(entrypoint):
        raise _fail(f"CUDA extension is missing callable {GRU_PHYSICAL_ENTRYPOINT_V3}")
    arguments = list(prepared.patch.arguments)
    physical_ticks = generation_count * ticks_per_generation
    arguments[PHYSICAL_RUNTIME_BOOKKEEPING_ARGUMENT_INDEX_V3] = arguments[
        PHYSICAL_RUNTIME_BOOKKEEPING_ARGUMENT_INDEX_V3
    ][:, :physical_ticks]
    full_tick_outputs = list(arguments[PHYSICAL_FULL_TICK_OUTPUTS_ARGUMENT_INDEX_V3])
    full_tick_outputs[-1] = full_tick_outputs[-1][:generation_count]
    arguments[PHYSICAL_FULL_TICK_OUTPUTS_ARGUMENT_INDEX_V3] = full_tick_outputs
    patch_stats = list(arguments[PHYSICAL_PATCH_STATS_ARGUMENT_INDEX_V3])
    patch_stats[0] = patch_stats[0][:, :physical_ticks]
    arguments[PHYSICAL_PATCH_STATS_ARGUMENT_INDEX_V3] = patch_stats
    arguments[TICKS_ARGUMENT_INDEX_V3] = ticks_per_generation
    arguments[GENERATION_COUNT_ARGUMENT_INDEX_V1] = generation_count
    metadata_value = _projection_member_v3(prepared.initial_projection, "metadata")
    if not isinstance(metadata_value, Mapping):
        raise _fail("GRU initial projection metadata is not a mapping")
    metadata = dict(metadata_value)
    raw = entrypoint(
        list(prepared.gru_tensors),
        {
            "upkeep_cost_base": float(metadata.get("gru_cost_base", 0.0)),
            "upkeep_cost_per_dim": float(metadata.get("gru_cost_per_dim", 0.0)),
        },
        *arguments,
    )
    if not isinstance(raw, Mapping):
        raise _fail("GRU physical CUDA entrypoint returned a non-mapping result")
    result = dict(raw)
    expected = {
        "gru_native_owner": True,
        "gru_native_controller_abi_version": GRU_ABI_VERSION_V3,
        "gru_physical_gate_entrypoint": True,
        "gru_python_tick_or_generation_ownership": False,
        "hidden_fallback": False,
        "runtime_jit": False,
    }
    for name, expected_value in expected.items():
        actual = result.get(name)
        if type(actual) is not type(expected_value) or actual != expected_value:
            raise _fail(
                f"GRU physical result {name} mismatch: "
                f"expected={expected_value!r} actual={actual!r}"
            )
    if int(prepared.gru_tensors[2].detach().cpu().item()) != 0:
        raise _fail("GRU physical device status is nonzero")
    return result


def _materialize_disabled_prediction_state_v3(
    prepared: PreparedDirectResidentGruV3,
    persistent_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Settle dormant prediction rows after native lifecycle changes.

    Prediction execution is disabled in the qualified stable-GRU envelope, so
    surviving rows are immutable. Births and deaths still own canonical reset
    semantics. Reconstruct those rows once at the checkpoint boundary without
    introducing host control into the resident generation loop.
    """

    return materialize_disabled_prediction_state_v1(
        prepared.initial_projection,
        persistent_state,
        reset_count_max=PHYSICAL_WINDOW_TICKS_V1 // 2,
    )


def _canonical_pred_lr_exact_cpu_v3(
    pred_lr: torch.Tensor,
    alive_mask: torch.Tensor,
) -> torch.Tensor:
    if pred_lr.device.type != "cpu" or alive_mask.device.type != "cpu":
        raise _fail(
            "canonical prediction-rate settlement requires CPU checkpoint tensors"
        )
    if pred_lr.shape != alive_mask.shape:
        raise _fail("canonical prediction-rate settlement shape mismatch")
    canonical = torch.round(pred_lr.to(torch.float64) * 100_000_000.0)
    canonical = canonical / 100_000_000.0
    return torch.where(
        alive_mask.to(dtype=torch.bool),
        canonical,
        torch.zeros_like(canonical),
    ).contiguous()


def _projection_member_v3(projection: Any, name: str) -> Any:
    if isinstance(projection, Mapping):
        return projection.get(name)
    return getattr(projection, name, None)


def _clone_private_v3(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone().contiguous()
    if isinstance(value, Mapping):
        return {name: _clone_private_v3(member) for name, member in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_private_v3(member) for member in value)
    if isinstance(value, list):
        return [_clone_private_v3(member) for member in value]
    return copy.deepcopy(value)


def _private_projection_parts_v3(projection: Any, *, cell: int) -> SimpleNamespace:
    schema_version = _projection_member_v3(projection, "schema_version")
    metadata = _projection_member_v3(projection, "metadata")
    tensors = _projection_member_v3(projection, "tensors")
    if type(schema_version) is not str or not schema_version:
        raise _fail(f"source projection {cell} schema_version must be nonempty")
    if (
        not isinstance(metadata, Mapping)
        or not isinstance(tensors, Mapping)
        or not tensors
    ):
        raise _fail(f"source projection {cell} metadata/tensors are invalid")
    runtime_fields: dict[str, Any] = {}
    if isinstance(projection, Mapping):
        members = projection.items()
    else:
        members = vars(projection).items()
    for name, value in members:
        if name not in {"schema_version", "metadata", "tensors"}:
            runtime_fields[name] = _clone_private_v3(value)
    return SimpleNamespace(
        schema_version=schema_version,
        metadata=_clone_private_v3(dict(metadata)),
        tensors=_clone_private_v3(dict(tensors)),
        **runtime_fields,
    )


def _projection_mapping_v3(projection: SimpleNamespace) -> dict[str, Any]:
    return {name: _clone_private_v3(value) for name, value in vars(projection).items()}


def _projection_order_id_v3(projection: SimpleNamespace) -> str | None:
    for source in (vars(projection), projection.metadata):
        for name in ("cell_id", "source_projection_cell_id"):
            value = source.get(name)
            if type(value) is str and value:
                return value
    return None


def _set_checkpoint_policy_family_metadata_v3(
    metadata: Mapping[str, Any], policy_family: str
) -> dict[str, Any]:
    if policy_family not in {"mlp", "gru", "mixed"}:
        raise _fail("checkpoint policy family must be mlp, gru, or mixed")
    updated = dict(metadata)
    updated["native_runtime_fixed_policy_family"] = policy_family
    capability_mask = dict(updated.get("capability_mask", {}))
    capability_mask["gru"] = policy_family in {"gru", "mixed"}
    updated["capability_mask"] = capability_mask
    return updated


def _validated_source_projections_v3(
    projections: Sequence[Any],
    *,
    cell_count: int,
    source_generation: int,
    direct_result: Mapping[str, Any],
) -> tuple[SimpleNamespace, ...]:
    if isinstance(projections, (str, bytes, bytearray)) or not isinstance(
        projections, Sequence
    ):
        raise _fail("ordered_source_projections must be a sequence")
    if len(projections) != cell_count:
        raise _fail("source projection count must exactly equal resident cell count")
    if len({id(projection) for projection in projections}) != cell_count:
        raise _fail("source projections must be distinct ordered objects")

    private = tuple(
        _private_projection_parts_v3(projection, cell=cell)
        for cell, projection in enumerate(projections)
    )
    expected_schema = private[0].schema_version
    expected_fields = tuple(private[0].tensors)
    expected_layout: dict[str, tuple[Any, ...]] = {}
    occupied: dict[tuple[str, int], tuple[int, str]] = {}
    for cell, projection in enumerate(private):
        if projection.schema_version != expected_schema:
            raise _fail(f"source projection {cell} schema_version mismatch")
        metadata = projection.metadata
        if (
            type(metadata.get("generation")) is not int
            or metadata.get("generation") != source_generation
            or type(metadata.get("tick")) is not int
            or metadata.get("tick") != 0
            or metadata.get("agent_grid_runner_phase") != "generation_boundary"
        ):
            raise _fail(f"source projection {cell} is not the requested boundary")
        if tuple(projection.tensors) != expected_fields:
            raise _fail(f"source projection {cell} tensor field order/layout mismatch")
        for name, tensor in projection.tensors.items():
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.device.type != "cpu"
                or tensor.requires_grad
            ):
                raise _fail(
                    f"source projection {cell} tensor {name!r} must be detached CPU storage"
                )
            layout = (
                tensor.dtype,
                tensor.layout,
                tuple(tensor.shape),
                tuple(tensor.stride()),
            )
            if cell == 0:
                expected_layout[name] = layout
            elif name == "agent_id_to_slot":
                expected = expected_layout[name]
                if (
                    tensor.dtype != expected[0]
                    or tensor.dtype != torch.int32
                    or tensor.layout != expected[1]
                    or tensor.ndim != 1
                    or not tensor.is_contiguous()
                ):
                    raise _fail(
                        f"source projection {cell} agent_id_to_slot layout mismatch"
                    )
            elif name in EVENT_COLUMN_NAMES:
                expected = expected_layout[name]
                if (
                    tensor.dtype != expected[0]
                    or tensor.layout != expected[1]
                    or tensor.ndim != 1
                    or not tensor.is_contiguous()
                ):
                    raise _fail(
                        f"source projection {cell} event tensor {name!r} "
                        "layout mismatch"
                    )
            elif layout != expected_layout[name]:
                raise _fail(
                    f"source projection {cell} tensor {name!r} shape/dtype/layout mismatch"
                )
            if tensor.numel():
                key = (str(tensor.device), int(tensor.untyped_storage().data_ptr()))
                previous = occupied.get(key)
                if previous is not None:
                    raise _fail(
                        "source projection storage alias: "
                        f"cell={cell} field={name!r} aliases "
                        f"cell={previous[0]} field={previous[1]!r}"
                    )
                occupied[key] = (cell, name)

    expected_ids = direct_result.get("source_projection_cell_ids")
    if expected_ids is not None:
        if (
            isinstance(expected_ids, (str, bytes, bytearray))
            or not isinstance(expected_ids, Sequence)
            or len(expected_ids) != cell_count
        ):
            raise _fail("direct result source_projection_cell_ids layout mismatch")
        actual_ids = tuple(
            _projection_order_id_v3(projection) for projection in private
        )
        if any(value is None for value in actual_ids) or actual_ids != tuple(
            expected_ids
        ):
            raise _fail("source projection order does not match the direct result")
    return private


def _slice_cell_tensor_v3(
    value: Any,
    *,
    name: str,
    cell: int,
    cell_count: int,
    axis: int = 0,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim <= axis:
        raise _fail(f"multicell owner tensor is invalid: {name}")
    if int(value.size(axis)) != cell_count:
        raise _fail(f"multicell owner tensor has the wrong cell axis: {name}")
    return value.narrow(axis, cell, 1).detach().clone().contiguous()


def _slice_live_banks_v3(
    banks: Mapping[str, Any], *, cell: int, cell_count: int
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, value in banks.items():
        if isinstance(value, torch.Tensor):
            out[name] = _slice_cell_tensor_v3(
                value, name=f"live_banks.{name}", cell=cell, cell_count=cell_count
            )
        elif name == "event_columns":
            if not isinstance(value, Sequence) or isinstance(
                value, (str, bytes, bytearray)
            ):
                raise _fail("multicell owner event_columns must be a sequence")
            out[name] = [
                _slice_cell_tensor_v3(
                    column,
                    name=f"live_banks.event_columns[{index}]",
                    cell=cell,
                    cell_count=cell_count,
                )
                for index, column in enumerate(value)
            ]
    return out


def _slice_output_mapping_v3(
    outputs: Mapping[str, Any], *, cell: int, cell_count: int
) -> dict[str, Any]:
    required = {
        "birth_combined_status",
        "event_status_by_tick",
        "generation_summary_scalar",
        "lifecycle_status",
        "runtime_bookkeeping_by_tick",
        "turnover_counts",
        "turnover_next_generation",
        "turnover_next_tick",
        "turnover_status",
    }
    out: dict[str, Any] = {}
    for name, value in outputs.items():
        if name not in required:
            continue
        if not isinstance(value, torch.Tensor):
            out[name] = _clone_private_v3(value)
        elif name == "generation_summary_scalar":
            out[name] = _slice_cell_tensor_v3(
                value, name=name, cell=cell, cell_count=cell_count, axis=1
            )
        elif value.ndim > 0 and int(value.size(0)) == cell_count:
            out[name] = _slice_cell_tensor_v3(
                value, name=name, cell=cell, cell_count=cell_count
            )
        else:
            out[name] = value.detach().clone().contiguous()
    return out


def _materialize_active_event_histories_v3(
    prepared: PreparedDirectResidentGruV3,
    *,
    cell_count: int,
) -> tuple[tuple[tuple[torch.Tensor, ...], ...], tuple[int, ...]]:
    banks = prepared.patch.live_banks
    columns = banks.get("event_columns")
    active_count = banks.get("event_active_count")
    event_status = banks.get("event_status")
    if (
        not isinstance(columns, Sequence)
        or len(columns) != len(EVENT_COLUMN_NAMES)
        or any(not isinstance(column, torch.Tensor) for column in columns)
        or not isinstance(active_count, torch.Tensor)
        or tuple(active_count.shape) != (cell_count,)
        or not isinstance(event_status, torch.Tensor)
        or tuple(event_status.shape) != (cell_count,)
        or bool(torch.count_nonzero(event_status).item())
    ):
        raise _fail("physical multicell event bank is invalid")
    counts = tuple(int(value) for value in active_count.detach().cpu().tolist())
    capacity = int(columns[0].size(1))
    if any(
        column.ndim != 2
        or tuple(column.shape) != (cell_count, capacity)
        for column in columns
    ) or any(count < 0 or count > capacity for count in counts):
        raise _fail("physical multicell event bank layout is invalid")
    histories = tuple(
        tuple(
            column[cell, :count].detach().to(device="cpu").contiguous()
            for column in columns
        )
        for cell, count in enumerate(counts)
    )
    return histories, counts


def project_direct_resident_gru_multicell_checkpoints_v3(
    prepared: PreparedDirectResidentGruV3,
    *,
    direct_result: Mapping[str, Any],
    completed_generation_count: int,
    ordered_source_projections: Sequence[Any],
    pre_materialized_cpu_fields: frozenset[str] = frozenset(),
    checkpoint_policy_family: str | None = None,
    checkpoint_policy_families: Sequence[str] | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Project each cell of one resident owner through the unchanged B1 contract."""

    if not isinstance(prepared, PreparedDirectResidentGruV3):
        raise _fail("prepared must be PreparedDirectResidentGruV3")
    if not isinstance(direct_result, Mapping):
        raise _fail("direct_result must be a mapping")
    source = prepared.patch.source
    cell_count = getattr(source, "batch_cells", None)
    if type(cell_count) is not int or cell_count < 1:
        raise _fail("prepared owner must expose a positive exact resident cell count")
    if checkpoint_policy_family is not None and checkpoint_policy_families is not None:
        raise _fail("checkpoint policy family inputs are mutually exclusive")
    if checkpoint_policy_families is None:
        policy_families = (checkpoint_policy_family,) * cell_count
    else:
        if isinstance(checkpoint_policy_families, (str, bytes)) or not isinstance(
            checkpoint_policy_families, Sequence
        ):
            raise _fail("checkpoint_policy_families must be a sequence")
        policy_families = tuple(checkpoint_policy_families)
        if len(policy_families) != cell_count:
            raise _fail("checkpoint policy family count does not match resident cells")
    if any(family not in {None, "mlp", "gru", "mixed"} for family in policy_families):
        raise _fail("checkpoint policy families must be mlp, gru, mixed, or None")
    if (
        type(completed_generation_count) is not int
        or completed_generation_count < 1
        or completed_generation_count > source.logical_generation_count
    ):
        raise _fail("multicell checkpoint projection exceeds its resident horizon")
    reported_completed = direct_result.get("completed_generations")
    if (
        reported_completed is not None
        and reported_completed != completed_generation_count
    ):
        raise _fail("direct result completed generation count mismatch")
    projections = _validated_source_projections_v3(
        ordered_source_projections,
        cell_count=cell_count,
        source_generation=source.generation_base,
        direct_result=direct_result,
    )
    summary = direct_result.get("generation_summary_scalar")
    bookkeeping = direct_result.get("runtime_bookkeeping_by_tick")
    if not isinstance(summary, torch.Tensor):
        summary = prepared.patch.arguments[
            PHYSICAL_FULL_TICK_OUTPUTS_ARGUMENT_INDEX_V3
        ][-1][:completed_generation_count]
    if not isinstance(bookkeeping, torch.Tensor):
        bookkeeping = prepared.patch.arguments[
            PHYSICAL_RUNTIME_BOOKKEEPING_ARGUMENT_INDEX_V3
        ][:, : completed_generation_count * (PHYSICAL_WINDOW_TICKS_V1 // 2)]
    _slice_cell_tensor_v3(
        summary,
        name="direct_result.generation_summary_scalar",
        cell=0,
        cell_count=cell_count,
        axis=1,
    )
    _slice_cell_tensor_v3(
        bookkeeping,
        name="direct_result.runtime_bookkeeping_by_tick",
        cell=0,
        cell_count=cell_count,
    )
    histories = direct_result.get("event_histories")
    event_rows = direct_result.get("event_rows_by_cell")
    if histories is None and event_rows is None:
        histories, event_rows = _materialize_active_event_histories_v3(
            prepared, cell_count=cell_count
        )
    if (
        not isinstance(histories, Sequence)
        or isinstance(histories, (str, bytes, bytearray))
        or len(histories) != cell_count
        or not isinstance(event_rows, Sequence)
        or isinstance(event_rows, (str, bytes, bytearray))
        or len(event_rows) != cell_count
    ):
        raise _fail(
            "direct result must expose one event history and row count per cell"
        )

    projected: list[Mapping[str, Any]] = []
    for cell, (source_projection, cell_policy_family) in enumerate(
        zip(projections, policy_families, strict=True)
    ):
        live_banks = _slice_live_banks_v3(
            prepared.patch.live_banks, cell=cell, cell_count=cell_count
        )
        outputs = _slice_output_mapping_v3(
            source.outputs, cell=cell, cell_count=cell_count
        )
        cell_summary = _slice_cell_tensor_v3(
            summary,
            name="direct_result.generation_summary_scalar",
            cell=cell,
            cell_count=cell_count,
            axis=1,
        )
        cell_bookkeeping = _slice_cell_tensor_v3(
            bookkeeping,
            name="direct_result.runtime_bookkeeping_by_tick",
            cell=cell,
            cell_count=cell_count,
        )
        outputs["generation_summary_scalar"] = cell_summary
        outputs["runtime_bookkeeping_by_tick"] = cell_bookkeeping
        checkpoint_inputs = dict(prepared.patch.checkpoint_projection_inputs)
        canonical_projection = _projection_mapping_v3(source_projection)
        if (
            canonical_projection.get("schema_version")
            != CHECKPOINT_PROJECTION_SCHEMA_VERSION
        ):
            canonical_by_cell = checkpoint_inputs.get(
                "pre_generation_canonical_projections"
            )
            if (
                not isinstance(canonical_by_cell, Sequence)
                or len(canonical_by_cell) != cell_count
                or not isinstance(canonical_by_cell[cell], Mapping)
            ):
                raise _fail(
                    "resident source lacks ordered canonical checkpoint projections"
                )
            canonical_projection = _clone_private_v3(
                dict(canonical_by_cell[cell])
            )
        if cell_policy_family is not None:
            canonical_metadata = dict(canonical_projection["metadata"])
            # The canonical input is the common pre-arm source. Keep that
            # intermediate envelope mixed until current resident rows replace it.
            canonical_metadata["native_runtime_fixed_policy_family"] = "mixed"
            capability_mask = dict(canonical_metadata.get("capability_mask", {}))
            capability_mask["gru"] = True
            canonical_metadata["capability_mask"] = capability_mask
            canonical_projection["metadata"] = canonical_metadata
        checkpoint_inputs["pre_generation_canonical_projection"] = canonical_projection
        cell_source = replace(
            source,
            live_state=live_banks,
            outputs=outputs,
            live_banks=live_banks,
            checkpoint_projection_inputs=checkpoint_inputs,
            batch_cells=1,
        )
        cell_patch = replace(
            prepared.patch,
            source=cell_source,
            live_banks=live_banks,
            checkpoint_projection_inputs=checkpoint_inputs,
        )
        cell_prepared = replace(
            prepared,
            patch=cell_patch,
            initial_projection=source_projection,
        )
        cell_result = {
            "generation_summary_scalar": cell_summary,
            "runtime_bookkeeping_by_tick": cell_bookkeeping,
            "event_histories": [_clone_private_v3(histories[cell])],
            "event_rows_by_cell": [copy.deepcopy(event_rows[cell])],
        }
        projection = project_direct_resident_gru_checkpoint_v3(
            cell_prepared,
            direct_result=cell_result,
            completed_generation_count=completed_generation_count,
            pre_materialized_cpu_fields=pre_materialized_cpu_fields,
            checkpoint_policy_family=cell_policy_family,
        )
        projected.append(_clone_private_v3(projection))
    return tuple(projected)


def project_direct_resident_gru_checkpoint_v3(
    prepared: PreparedDirectResidentGruV3,
    *,
    direct_result: Mapping[str, Any],
    completed_generation_count: int,
    pre_materialized_cpu_fields: frozenset[str] = frozenset(),
    checkpoint_policy_family: str | None = None,
) -> Mapping[str, Any]:
    """Project a stable-GRU resident prefix through the Patch checkpoint path."""

    if not isinstance(prepared, PreparedDirectResidentGruV3):
        raise _fail("prepared must be PreparedDirectResidentGruV3")
    if checkpoint_policy_family not in {None, "mlp", "gru", "mixed"}:
        raise _fail("checkpoint_policy_family must be mlp, gru, mixed, or None")
    if (
        type(completed_generation_count) is not int
        or completed_generation_count < 1
        or completed_generation_count > prepared.patch.source.logical_generation_count
    ):
        raise _fail("GRU checkpoint projection exceeds its resident horizon")
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
        ][:, : completed_generation_count * (PHYSICAL_WINDOW_TICKS_V1 // 2)]
    outputs["generation_summary_scalar"] = generation_summary
    outputs["runtime_bookkeeping_by_tick"] = runtime_bookkeeping
    persistent_state = _materialize_disabled_prediction_state_v3(
        prepared,
        _with_materialized_event_history(prepared.patch.live_banks, result),
    )
    projection_inputs = prepared.patch.checkpoint_projection_inputs
    checkpoint = _project_terminal_checkpoint(
        projection_inputs["pre_generation_canonical_projection"],
        projection_inputs["resident_birth_state"],
        outputs,
        persistent_state,
        completed_generation_count=completed_generation_count,
        source_generation=prepared.patch.source.generation_base,
        pre_materialized_cpu_fields=pre_materialized_cpu_fields,
        inheritance_sidecar_cell_id=(
            prepared.cell_ids[0] if prepared.cell_ids else "0"
        ),
        genome_v2_checkpoint_authority_binding=projection_inputs.get(
            "genome_v2_checkpoint_authority_binding", ()
        ),
        genome_v2_causal_physical_cell_ordinal=projection_inputs.get(
            "genome_v2_causal_physical_cell_ordinal", 0
        ),
    )
    projection = dict(checkpoint.projection)
    metadata = dict(projection["metadata"])
    metadata["prediction_tick"] = PHYSICAL_WINDOW_TICKS_V1 // 2
    if checkpoint_policy_family is not None:
        metadata = _set_checkpoint_policy_family_metadata_v3(
            metadata, checkpoint_policy_family
        )
    projection["metadata"] = metadata
    tensors = dict(projection["tensors"])
    physical_id_map = prepared.patch.live_banks.get("agent_id_to_slot")
    canonical_id_map = tensors.get("agent_id_to_slot")
    if (
        not isinstance(physical_id_map, torch.Tensor)
        or physical_id_map.dtype != torch.int32
        or physical_id_map.ndim != 2
        or int(physical_id_map.size(0)) != 1
        or not isinstance(canonical_id_map, torch.Tensor)
        or canonical_id_map.dtype != physical_id_map.dtype
        or canonical_id_map.ndim != 1
        or canonical_id_map.numel() > physical_id_map.size(1)
    ):
        raise _fail("resident agent-id allocation cannot be checkpointed exactly")
    physical_id_map_cpu = physical_id_map[0].detach().to(device="cpu")
    canonical_length = int(canonical_id_map.numel())
    if not torch.equal(
        physical_id_map_cpu[:canonical_length], canonical_id_map.detach().cpu()
    ) or bool(physical_id_map_cpu[canonical_length:].ne(-1).any().item()):
        raise _fail("resident agent-id allocation has a noncanonical inactive tail")
    metadata[RESIDENT_AGENT_ID_CAPACITY_METADATA_V1] = int(physical_id_map.size(1))
    tensors["pred_lr_exact"] = _canonical_pred_lr_exact_cpu_v3(
        tensors["pred_lr"],
        tensors["alive_mask"],
    )
    projection["tensors"] = tensors
    return _with_checkpoint_contract_metadata(projection)


__all__ = [
    "GRU_CAPABILITY_ENTRYPOINT_V3",
    "GRU_DIRECT_ENTRYPOINT_V3",
    "GRU_PHYSICAL_ENTRYPOINT_V3",
    "CudaArchitectureBDirectResidentGruV3Error",
    "PreparedDirectResidentGruV3",
    "GRU_DISABLED_PREDICTION_STATE_FIELDS_V3",
    "build_prepared_direct_resident_gru_v3",
    "execute_direct_resident_gru_v3",
    "execute_direct_resident_gru_physical_v3",
    "project_direct_resident_gru_checkpoint_v3",
    "project_direct_resident_gru_multicell_checkpoints_v3",
    "stable_gru_packed_width_v3",
]
