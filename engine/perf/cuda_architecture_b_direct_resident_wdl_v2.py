"""Build and execute the exact Patch P2 plus WDL resident CUDA overlay."""

from __future__ import annotations

import copy
import dataclasses
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Callable, Final

import torch

from engine.native_runtime.cuda_architecture_b_backend_factory import (
    CudaArchitectureBRunnerEnvelope,
    derive_cuda_architecture_b_runner_envelope,
)
from engine.native_runtime.interchange_projection import (
    validate_native_checkpoint_projection,
)
from engine.perf.cuda_architecture_b_checkpoint import CudaBoundaryCheckpoint
from engine.perf.cuda_architecture_b_direct_resident_patch_v1 import (
    DetachedCellProjectionV1,
    DirectResidentPatchMulticellSettlementV1,
    _projection_mapping,
    _slice_live_banks,
)
from engine.perf.cuda_architecture_b_direct_resident_preparation_v1 import (
    DIRECT_RESIDENT_ARGUMENT_NAMES_V1,
    GENERATION_COUNT_ARGUMENT_INDEX_V1,
    DirectResidentSourceV1,
    PreparedDirectResidentTrajectoryV1,
    prepare_direct_resident_trajectory_v1,
)
from engine.perf.cuda_architecture_b_direct_resident_execution_v2 import (
    create_run_control_v2,
)
from engine.perf.cuda_architecture_b_direct_resident_runtime_v1 import (
    resident_window_count_v1,
)
from engine.perf.cuda_architecture_b_disabled_prediction_state_v1 import (
    materialize_disabled_prediction_state_v1,
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
    canonical_generation_zero_source_projection_v1,
)
from engine.perf.cuda_architecture_b_direct_resident_settlement_v1 import (
    _project_terminal_checkpoint,
    _with_materialized_event_history,
    with_resident_inheritance_sidecars_v1,
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


WDL_ABI_VERSION_V3: Final[int] = 3
WDL_OBSERVATION_SIZE_V2: Final[int] = 83
WDL_TICKS_PER_GENERATION_V2: Final[int] = 80
WDL_SUPPORTED_TICKS_PER_GENERATION_V2: Final[frozenset[int]] = frozenset(
    {80, 160}
)
WDL_GENERATION_METRIC_COUNT_V2: Final[int] = 19
WDL_TICK_INTEGER_METRIC_COUNT_V2: Final[int] = 6
WDL_EVENT_INTEGER_COUNT_V2: Final[int] = 5
WDL_CAPABILITY_ENTRYPOINT_V3: Final[str] = (
    "eiel_cuda_architecture_b_flattened_wdl_capability_v3"
)
WDL_PHYSICAL_ENTRYPOINT_V3: Final[str] = (
    "eiel_cuda_architecture_b_execute_direct_resident_wdl_physical_v3"
)
WDL_DIRECT_ENTRYPOINT_V3: Final[str] = (
    "eiel_cuda_architecture_b_execute_direct_resident_wdl_v3"
)
WDL_AGENT_FLAG_CUED_V3: Final[int] = 1
WDL_AGENT_FLAG_DAMAGED_V3: Final[int] = 2
WDL_AGENT_FLAG_MASK_V3: Final[int] = (
    WDL_AGENT_FLAG_CUED_V3 | WDL_AGENT_FLAG_DAMAGED_V3
)
WDL_RESIDENT_GENERATION_MIN_V2: Final[int] = 3
WDL_RESIDENT_GENERATION_MAX_V2: Final[int] = 128
WDL_STATE_FIELDS_V2: Final[tuple[str, ...]] = (
    "patch_delayed_hazard_cue",
    "patch_delayed_hazard_timer",
    "patch_delayed_hazard_active",
    "patch_delayed_hazard_intensity",
    "patch_delayed_hazard_age",
    "patch_delayed_hazard_event_id",
    "patch_delayed_hazard_prior_cue",
)
WDL_CAPABILITIES_V2: Final[dict[str, bool]] = {
    "flat": False,
    "patch": True,
    "churn": True,
    "wdl": True,
    "jungle": False,
    "gru": False,
    "cuda": False,
}
WDL_JUNGLE_CAPABILITIES_V1: Final[dict[str, bool]] = {
    **WDL_CAPABILITIES_V2,
    "jungle": True,
}
WDL_OPTIONAL_SOURCE_CAPABILITIES_V2: Final[frozenset[str]] = frozenset(
    {"genome_v2"}
)
_TICK_NATIVE_RUNTIME_PREFIX_FIELDS_V1: Final[tuple[str, ...]] = (
    "restore_count",
    "generation_calls",
    "event_cursor",
    "runtime_event_cursor",
    "last_applied_actions",
    "last_moved",
    "last_gathered",
    "last_policy_refill_calls",
    "last_policy_refill_rows",
    "pending_generation_free_agent_ids",
    "pending_generation_free_slots",
    "last_world",
)
PHYSICAL_RUNTIME_BOOKKEEPING_ARGUMENT_INDEX_V2: Final[int] = (
    DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index("physical_runtime_bookkeeping")
)


def _wdl_source_capabilities_v2(
    source_capability_mask: Mapping[str, bool] | None,
    *,
    jungle_overlay: bool = False,
    eed_overlay: bool = False,
    hapb_host_policy_enable: bool = False,
) -> dict[str, bool]:
    route_capabilities = dict(
        WDL_JUNGLE_CAPABILITIES_V1 if jungle_overlay else WDL_CAPABILITIES_V2
    )
    if eed_overlay:
        if not jungle_overlay:
            raise _fail("EED WDL source capability requires the shared Jungle owner")
        route_capabilities["eed"] = True
    capabilities = dict(route_capabilities)
    if source_capability_mask is None:
        return capabilities
    if not isinstance(source_capability_mask, Mapping):
        raise _fail("WDL source capability mask must be a mapping")
    allowed = set(route_capabilities) | set(
        WDL_OPTIONAL_SOURCE_CAPABILITIES_V2
    )
    unknown = sorted(set(source_capability_mask) - allowed)
    if unknown:
        raise _fail(
            f"WDL source capability mask has unsupported keys: {unknown}"
        )
    for name, value in source_capability_mask.items():
        if type(value) is not bool:
            raise _fail(
                f"WDL source capability {name} must be an exact boolean"
            )
        if (
            name in route_capabilities
            and value != route_capabilities[name]
        ):
            raise _fail(
                f"WDL source capability {name} contradicts the WDL route"
            )
        capabilities[name] = value
    return capabilities
PHYSICAL_FULL_TICK_OUTPUTS_ARGUMENT_INDEX_V2: Final[int] = (
    DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index("physical_full_tick_output_tensors")
)
PHYSICAL_PATCH_STATS_ARGUMENT_INDEX_V2: Final[int] = (
    DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index("physical_patch_stats_tensors")
)
TICKS_ARGUMENT_INDEX_V2: Final[int] = DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index(
    "ticks_per_generation"
)


class CudaArchitectureBDirectResidentWdlV2Error(RuntimeError):
    """The exact WDL resident source or execution boundary is unavailable."""


@dataclass(frozen=True, slots=True)
class PreparedDirectResidentWdlV2:
    patch: PreparedDirectResidentTrajectoryV1
    state_tensors: tuple[torch.Tensor, ...]
    schedule_tensors: tuple[torch.Tensor, ...]
    reporting_tensors: tuple[torch.Tensor, ...]
    config: dict[str, Any]
    generation_rows: int
    initial_projection: Any
    cell_ids: tuple[str, ...]
    source_projections: tuple[Any, ...] = ()
    metadata_rows: tuple[Mapping[str, Any], ...] = ()
    # Compact scalar continuation prefixes for tick-native subspans.  These
    # rows deliberately contain no tensor-bearing values; the resident CUDA
    # owner remains the sole holder of mutable simulation state.
    runtime_prefix_rows: tuple[Mapping[str, Any], ...] = ()
    # Bounded physical-B owners retain only a factory for reloading one
    # canonical source projection at checkpoint time.  The factory owns
    # descriptors/digests, never a decoded B-wide projection bank.
    source_projection_cursor_factory: Callable[[], Any] | None = None
    source_import_cursor_receipt: Mapping[str, Any] | None = None
    streamed_extra_tensors: Mapping[str, torch.Tensor] | None = None


@dataclass(frozen=True, slots=True)
class _StreamedWdlSourceAssemblyV2:
    """One-cell-at-a-time source import result for the physical owner."""

    source: DirectResidentSourceV1
    metadata_rows: tuple[Mapping[str, Any], ...]
    runtime_prefix_rows: tuple[Mapping[str, Any], ...]
    first_projection: Mapping[str, Any]
    extra_tensors: Mapping[str, torch.Tensor]
    source_projection_cursor_factory: Callable[[], Any] | None
    source_import_cursor_receipt: Mapping[str, Any] | None


def _fail(message: str) -> CudaArchitectureBDirectResidentWdlV2Error:
    return CudaArchitectureBDirectResidentWdlV2Error(message)


def _projection_generation_turnover_enabled_v2(
    projection: Any,
    *,
    default: bool | None = True,
) -> bool | None:
    if isinstance(projection, Mapping):
        metadata = projection.get("metadata")
    else:
        checkpoint_projection = getattr(projection, "projection", None)
        if isinstance(checkpoint_projection, Mapping):
            metadata = checkpoint_projection.get("metadata")
        else:
            metadata = getattr(projection, "metadata", None)
    if not isinstance(metadata, Mapping):
        return default
    turnover = metadata.get("generation_turnover_enabled")
    lifecycle_mode = metadata.get("lifecycle_mode_abi")
    if turnover is None and lifecycle_mode is None:
        return default
    if type(turnover) is not bool or isinstance(lifecycle_mode, bool):
        raise _fail("WDL checkpoint lifecycle metadata is malformed")
    if lifecycle_mode not in {0, 1}:
        raise _fail("WDL checkpoint lifecycle mode ABI is invalid")
    expected_mode = 0 if turnover else 1
    if lifecycle_mode != expected_mode:
        raise _fail("WDL checkpoint lifecycle metadata is contradictory")
    return turnover


def _prepared_generation_turnover_enabled_v2(
    prepared: "PreparedDirectResidentWdlV2",
) -> bool:
    checkpoint_inputs = prepared.patch.checkpoint_projection_inputs
    turnover = _projection_generation_turnover_enabled_v2(
        checkpoint_inputs.get("pre_generation_canonical_projection"),
        default=None,
    )
    if turnover is None:
        turnover = _projection_generation_turnover_enabled_v2(
            prepared.initial_projection,
            default=None,
        )
    if turnover is None:
        source = getattr(prepared.patch, "source", None)
        turnover = getattr(source, "generation_turnover_enabled", None)
        if turnover is None:
            configuration = getattr(source, "configuration", None)
            if isinstance(configuration, Mapping):
                turnover = configuration.get("generation_turnover_enabled")
            else:
                turnover = getattr(
                    configuration, "generation_turnover_enabled", None
                )
        if turnover is None:
            turnover = True
    if type(turnover) is not bool:
        raise _fail("WDL checkpoint lifecycle mode could not be resolved")
    return turnover


def _wdl_allocation_cell_ids_v2(
    cell_id: str | None,
) -> tuple[str, ...] | None:
    if cell_id is None:
        return None
    if type(cell_id) is not str or not cell_id or cell_id.strip() != cell_id:
        raise _fail("WDL cell_id must be a nonempty stable string")
    return (cell_id,)


def _wdl_turnover_selection_authority_v2(sim_cfg: Any) -> tuple[int, int]:
    turnover_selection_mode = getattr(sim_cfg, "turnover_selection_mode", 0)
    if type(turnover_selection_mode) is not int or turnover_selection_mode not in {
        0,
        1,
    }:
        raise _fail("runner turnover_selection_mode must be exact integer 0 or 1")
    turnover_selection_seed = getattr(
        sim_cfg,
        "turnover_selection_seed",
        getattr(sim_cfg, "seed", 0),
    )
    if (
        type(turnover_selection_seed) is not int
        or not 0 <= turnover_selection_seed <= (1 << 64) - 1
    ):
        raise _fail("runner turnover_selection_seed must be an exact uint64 integer")
    return turnover_selection_mode, turnover_selection_seed


def _validated_wdl_cell_ids_v2(
    value: Sequence[str],
    *,
    count: int,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise _fail("WDL cell_ids must be a sequence")
    cell_ids = tuple(value)
    if (
        len(cell_ids) != count
        or any(type(cell_id) is not str or not cell_id for cell_id in cell_ids)
        or len(set(cell_ids)) != count
    ):
        raise _fail("WDL cell_ids must contain one unique nonempty string per cell")
    return cell_ids


def _enable_wdl_genome_v2_causal_capture_v2(
    outputs: Mapping[str, Any],
    *,
    enabled: bool,
    hapb_host_policy_enable: bool = False,
) -> None:
    """Enable bounded event capture for the production WDL Genome V2 route."""

    if not enabled:
        return

    admitted_max = 256
    for name in (
        "birth_genome_v2_causal_capture_enabled",
        "turnover_genome_v2_causal_capture_enabled",
    ):
        flag = outputs.get(name)
        if (
            not isinstance(flag, torch.Tensor)
            or flag.dtype != torch.bool
            or flag.ndim != 1
            or not 1 <= int(flag.shape[0]) <= admitted_max
        ):
            raise _fail(f"WDL Genome V2 causal capture flag is malformed: {name}")
        flag.fill_(True)


def _validated_capability(module: Any) -> dict[str, Any]:
    entrypoint = getattr(module, WDL_CAPABILITY_ENTRYPOINT_V3, None)
    if not callable(entrypoint):
        raise _fail(f"CUDA extension is missing callable {WDL_CAPABILITY_ENTRYPOINT_V3}")
    raw = entrypoint()
    if not isinstance(raw, Mapping):
        raise _fail("WDL CUDA capability must be a mapping")
    capability = dict(raw)
    expected = {
        "abi_version": WDL_ABI_VERSION_V3,
        "world": "patch_wdl",
        "resident_state": True,
        "observation_size": WDL_OBSERVATION_SIZE_V2,
        "exact_phase_schedule": True,
        "exact_cue_observation": True,
        "exact_delayed_damage": True,
        "native_tick_ownership": True,
        "generation_history_abi": "compact_keyed_generation_history_v3",
        "generation_history_capacity_formula": (
            "slot_capacity+ticks_per_generation*max_births_per_tick"
        ),
        "generation_history_no_eviction": True,
        "runtime_jit_required": False,
        "production_qualified": False,
    }
    for name, expected_value in expected.items():
        actual = capability.get(name)
        if type(actual) is not type(expected_value) or actual != expected_value:
            raise _fail(
                f"WDL CUDA capability {name} mismatch: "
                f"expected={expected_value!r} actual={actual!r}"
            )
    return capability


def _patch_envelope_with_wdl_masked(
    runner: Any,
    *,
    allow_short_technical_gate_generation_count: bool = False,
    tick_native_control: Mapping[str, Any] | None = None,
) -> CudaArchitectureBRunnerEnvelope:
    """Reuse the qualified Patch envelope classifier without mutating the runner."""

    proxy = copy.copy(runner)
    proxy.env_cfg = copy.copy(runner.env_cfg)
    proxy.env_cfg.wdl_delayed_hazard_enable = False
    proxy.env_cfg.wdl_delayed_hazard_observation_enable = False
    return derive_cuda_architecture_b_runner_envelope(
        proxy,
        allow_short_technical_gate_generation_count=(
            allow_short_technical_gate_generation_count
        ),
        tick_native_control=tick_native_control,
    )


def _required_wdl_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    exact = {
        "native_runtime_wdl_execution_enable": True,
        "native_runtime_wdl_storage_enable": True,
        "wdl_delayed_hazard_schema_version": 1,
        "wdl_delayed_hazard_observation_enable": True,
    }
    for name, expected in exact.items():
        actual = metadata.get(name)
        if type(actual) is not type(expected) or actual != expected:
            raise _fail(
                f"WDL import metadata {name} mismatch: "
                f"expected={expected!r} actual={actual!r}"
            )
    required = (
        "wdl_base_seed",
        "wdl_next_absolute_step",
        "wdl_delayed_hazard_delay_steps",
        "wdl_delayed_hazard_cue_duration_steps",
        "wdl_delayed_hazard_active_duration_steps",
        "wdl_delayed_hazard_event_period_steps",
        "wdl_delayed_hazard_radius",
        "wdl_delayed_hazard_intensity",
        "wdl_delayed_hazard_observation_dropout_rate",
        "wdl_delayed_hazard_cue_enable",
        "wdl_delayed_hazard_damage_enable",
        "wdl_delayed_hazard_deterministic_schedule_enable",
    )
    missing = [name for name in required if name not in metadata]
    if missing:
        raise _fail(f"WDL import metadata is missing: {missing}")
    return dict(metadata)


def _tick_native_runtime_prefix_row_v1(
    projection: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract only scalar continuation authority from one source projection.

    Tick-native dispatches intentionally retain mutable banks on the resident
    CUDA owner while their 160-tick reporting intervals are compacted.  The
    next checkpoint therefore needs the source projection's small runtime
    prefix, but never its tensor payload on the host.  Keep this extraction
    explicit so a future tensor-bearing field cannot silently enter the
    continuation prefix.
    """

    if not isinstance(projection, Mapping):
        raise _fail("tick-native runtime prefix source is not a mapping")
    metadata = projection.get("metadata")
    if not isinstance(metadata, Mapping):
        raise _fail("tick-native runtime prefix source lacks metadata")
    active_event_count = int(
        metadata.get(
            "event_tape_active_count",
            metadata.get("typed_lifecycle_event_active_count", 0),
        )
        or 0
    )
    defaults: dict[str, Any] = {
        "restore_count": 0,
        "generation_calls": 0,
        "event_cursor": active_event_count,
        "runtime_event_cursor": active_event_count,
        "last_applied_actions": 0,
        "last_moved": 0,
        "last_gathered": 0,
        "last_policy_refill_calls": 0,
        "last_policy_refill_rows": 0,
        "pending_generation_free_agent_ids": (),
        "pending_generation_free_slots": (),
        "last_world": "none",
    }
    row: dict[str, Any] = {}
    for name in _TICK_NATIVE_RUNTIME_PREFIX_FIELDS_V1:
        value = projection.get(name, defaults[name])
        if hasattr(value, "device") or hasattr(value, "detach"):
            raise _fail(
                "tick-native runtime prefix retained a tensor-bearing field: "
                f"{name}"
            )
        if name in {
            "pending_generation_free_agent_ids",
            "pending_generation_free_slots",
        }:
            if not isinstance(value, (list, tuple)) or any(
                type(item) is not int for item in value
            ):
                raise _fail(f"tick-native runtime prefix {name} is malformed")
            row[name] = list(value)
        else:
            row[name] = copy.deepcopy(value)
    return row


def _stream_copy_cell_value_v2(
    target: Any,
    source: Any,
    *,
    cell: int,
    cells: int,
    path: str,
) -> Any:
    """Copy one B1 source row into a preallocated B-wide owner.

    This helper deliberately mutates only the already allocated CUDA owner.
    It never builds a list of per-cell tensors.  Mappings and small structural
    tuples are rebuilt only when their container carries a per-cell value.
    """

    if isinstance(target, torch.Tensor):
        if not isinstance(source, torch.Tensor):
            raise _fail(f"streamed source field {path} changed from tensor type")
        if path in {
            "source.event_state_tensors[26]",
            "source.event_state_tensors[27]",
            "source.event_state_tensors[28]",
            "source.live_banks.terminal_survival",
            "source.live_banks.terminal_rows",
        }:
            # Initial reporting rows are intentionally ragged at B1.  The
            # streamed B-wide owner reserves the bounded resident capacity;
            # copy the live prefix and restore the ABI-defined tail so a
            # successor with a different live count cannot alter the owner
            # shape or retain a prior cell's reporting rows.
            if (
                target.ndim != 2
                or source.ndim != 2
                or int(target.shape[0]) != cells
                or int(source.shape[0]) != 1
                or int(source.shape[1]) > int(target.shape[1])
            ):
                raise _fail(
                    f"streamed source field {path} has incompatible reporting shapes: "
                    f"target={tuple(target.shape)} source={tuple(source.shape)}"
                )
            if path.endswith("[26]"):
                target[cell].fill_(torch.iinfo(target.dtype).max)
            else:
                target[cell].zero_()
            target[cell, : int(source.shape[1])].copy_(
                source[0], non_blocking=False
            )
            return target
        if (
            target.ndim >= 1
            and source.ndim >= 1
            and int(target.shape[0]) == cells
            and int(source.shape[0]) == 1
            and tuple(target.shape[1:]) == tuple(source.shape[1:])
        ):
            target[cell].copy_(source[0], non_blocking=False)
            return target
        if target.ndim == source.ndim and target.ndim >= 2:
            for axis in range(1, target.ndim):
                target_other_shape = tuple(
                    int(value)
                    for index, value in enumerate(target.shape)
                    if index != axis
                )
                source_other_shape = tuple(
                    int(value)
                    for index, value in enumerate(source.shape)
                    if index != axis
                )
                if (
                    int(target.shape[axis]) == cells
                    and int(source.shape[axis]) == 1
                    and target_other_shape == source_other_shape
                ):
                    target_index = [slice(None)] * target.ndim
                    source_index = [slice(None)] * source.ndim
                    target_index[axis] = cell
                    source_index[axis] = 0
                    target[tuple(target_index)].copy_(
                        source[tuple(source_index)], non_blocking=False
                    )
                    return target
        # Shared zero-width/descriptor tensors are intentionally allocated
        # once and do not have a physical-cell axis.
        if tuple(target.shape) == tuple(source.shape):
            return target
        raise _fail(
            f"streamed source field {path} has incompatible B1/B{cells} shapes: "
            f"target={tuple(target.shape)} source={tuple(source.shape)}"
        )
    if isinstance(target, Mapping):
        if not isinstance(source, Mapping):
            raise _fail(f"streamed source field {path} changed from mapping type")
        out = dict(target)
        for name, source_value in source.items():
            if name not in out:
                # A one-cell source may expose a diagnostic alias that the
                # B-wide owner intentionally elides.  It is not an owner
                # authority and must not force a second allocation.
                continue
            out[name] = _stream_copy_cell_value_v2(
                out[name],
                source_value,
                cell=cell,
                cells=cells,
                path=f"{path}.{name}",
            )
        return out
    if isinstance(target, list):
        if not isinstance(source, (list, tuple)):
            raise _fail(f"streamed source field {path} changed from sequence type")
        if len(target) == cells and len(source) == 1:
            target[cell] = source[0]
            return target
        if len(target) != len(source):
            raise _fail(f"streamed source field {path} sequence lengths differ")
        for index, source_value in enumerate(source):
            target[index] = _stream_copy_cell_value_v2(
                target[index],
                source_value,
                cell=cell,
                cells=cells,
                path=f"{path}[{index}]",
            )
        return target
    if isinstance(target, tuple):
        if not isinstance(source, (list, tuple)):
            raise _fail(f"streamed source field {path} changed from sequence type")
        if len(target) == cells and len(source) == 1:
            values = list(target)
            values[cell] = source[0]
            return tuple(values)
        if len(target) != len(source):
            raise _fail(f"streamed source field {path} sequence lengths differ")
        return tuple(
            _stream_copy_cell_value_v2(
                target[index],
                source_value,
                cell=cell,
                cells=cells,
                path=f"{path}[{index}]",
            )
            for index, source_value in enumerate(source)
        )
    # Scalars and immutable configuration markers are validated separately;
    # they are shared owner authority and are not copied per cell here.
    return target


def _stream_copy_source_cell_v2(
    target: DirectResidentSourceV1,
    source: DirectResidentSourceV1,
    *,
    cell: int,
    cells: int,
) -> DirectResidentSourceV1:
    """Copy one temporary B1 source into a preallocated physical-B owner."""

    updates: dict[str, Any] = {}
    for field in dataclasses.fields(target):
        name = field.name
        if name in {
            "configuration",
            "checkpoint_projection_inputs",
            "batch_cells",
            "logical_generation_count",
            "physical_window_generations",
            "ticks_per_generation",
            "generation_base",
            "allow_short_technical_gate_generation_count",
            "allow_partial_generation_source_cursor",
            "deterministic",
            "time_authority",
            "resident_owner_span_ticks",
            "native_dispatch_window_ticks",
            "refill_chunk_ticks",
            "source_window_ticks",
            "resident_event_capacity",
        }:
            continue
        target_value = getattr(target, name)
        source_value = getattr(source, name)
        updates[name] = _stream_copy_cell_value_v2(
            target_value,
            source_value,
            cell=cell,
            cells=cells,
            path=f"source.{name}",
        )
    return replace(target, **updates)


def _streamed_configuration_signature_v2(configuration: Any) -> dict[str, Any]:
    """Return shared configuration identity without per-cell founder knobs."""

    if not dataclasses.is_dataclass(configuration):
        return {"repr": repr(configuration)}
    payload = dataclasses.asdict(configuration)
    for name in (
        "genome_upgrade_v2_founder_profile_id",
        "genome_upgrade_v2_founder_profile_sha256",
        "move_cost",
        "patch_food_gather_amount",
        "patch_food_regrow_per_step",
        "patch_movement_cost_energy_scale",
        "jungle_predator_base_risk",
        "jungle_predator_health_damage",
        "jungle_predator_risk_cap",
        "jungle_recovery_activity_base_growth",
        "jungle_recovery_activity_decay_per_step",
        "jungle_recovery_activity_fertility_weight",
        "jungle_recovery_activity_waste_penalty_weight",
        "reproduce_cooldown_steps",
        "reproduce_min_energy_buffer",
        "reproduce_parent_energy_share",
        "reproduce_try_prob",
        "diploid_mutation_enable",
        "diploid_mutation_step_q10",
        "genome_upgrade_v2_mutation_rate_q10",
        "recomb72_mutation_off",
        "turnover_selection_seed",
    ):
        payload.pop(name, None)
    return payload


def _build_streamed_wdl_source_v2(
    *,
    request_factory: Callable[[int], DirectResidentSourceAllocationRequestV1],
    qualified_cuda_module: Any,
    cell_ids: Sequence[str],
    generation_rows: int,
    initial_generation: int,
    agent_id_allocation_generation_count: int | None,
    allow_short_technical_gate_generation_count: bool,
    allow_partial_generation_source_cursor: bool,
    extra_tensor_names: Sequence[str] = (),
    expected_source_capabilities: Mapping[str, bool] | None = None,
    source_projection_cursor_factory: Callable[[], Any] | None = None,
    source_import_cursor_receipt: Mapping[str, Any] | None = None,
) -> _StreamedWdlSourceAssemblyV2:
    """Build a physical owner from one direct request at a time.

    The existing allocator remains the authority for tensor shapes and ABI
    fields.  It is invoked once for a B-wide allocation seeded from cell zero,
    then once per remaining cell for a temporary B1 import whose row is copied
    into the resident owner.  No tuple/list of direct requests or source
    projections is retained.
    """

    ordered_ids = tuple(cell_ids)
    if len(ordered_ids) < 1 or len(set(ordered_ids)) != len(ordered_ids):
        raise _fail("streamed WDL source requires unique ordered cell IDs")
    cells = len(ordered_ids)
    extra_names = tuple(extra_tensor_names)
    if len(set(extra_names)) != len(extra_names):
        raise _fail("streamed WDL extra tensor names must be unique")

    first_request = request_factory(0)
    if not isinstance(first_request, DirectResidentSourceAllocationRequestV1):
        raise _fail("streamed WDL request factory returned the wrong request type")
    first_request = _stage_wdl_policy_birth_requests_on_boundary_cuda_v2(
        (first_request,)
    )[0]
    if first_request.batch_cells != 1:
        raise _fail("streamed WDL request factory must return B1 requests")
    if first_request.cpu_request_staging_enable is not False:
        raise _fail("streamed physical B136 forbids CPU request staging")
    first_projection = first_request.import_projection
    if not isinstance(first_projection, Mapping):
        raise _fail("streamed WDL request lacks a canonical source projection")
    first_runtime_prefix = _tick_native_runtime_prefix_row_v1(first_projection)
    first_metadata = _required_wdl_metadata(
        first_projection.get("metadata", {})
        if isinstance(first_projection, Mapping)
        else {}
    )
    if first_metadata.get("generation") != initial_generation:
        raise _fail("streamed WDL source generation differs from owner boundary")
    if (
        first_metadata.get("tick") != 0
        and not allow_partial_generation_source_cursor
    ):
        raise _fail("streamed WDL source is not at an admitted owner boundary")
    first_capabilities = first_metadata.get("capability_mask")
    if not isinstance(first_capabilities, Mapping):
        raise _fail("streamed WDL source lacks its capability mask")
    if (
        expected_source_capabilities is not None
        and dict(first_capabilities) != dict(expected_source_capabilities)
    ):
        raise _fail("streamed WDL source capability mask differs from route")
    if first_metadata.get("native_runtime_fixed_policy_family") != "mlp":
        raise _fail("streamed WDL source is not fixed MLP")
    first_signature = _streamed_configuration_signature_v2(
        first_request.configuration
    )
    first_steps = getattr(first_request.configuration, "steps", None)
    if first_steps not in WDL_SUPPORTED_TICKS_PER_GENERATION_V2:
        raise _fail("streamed WDL source has unsupported ticks_per_generation")
    for name in extra_names:
        boundary_tensors = getattr(first_request.cuda_boundary, "tensors", None)
        value = boundary_tensors.get(name) if isinstance(boundary_tensors, Mapping) else None
        if not isinstance(value, torch.Tensor):
            raise _fail(f"streamed WDL boundary lacks tensor {name!r}")
        if value.device.type != "cuda":
            raise _fail(f"streamed WDL boundary tensor {name!r} is not CUDA-resident")

    target_boundary = first_request.cuda_boundary
    boundary_tensors = dict(getattr(target_boundary, "tensors", {}) or {})
    boundary_sequence_base = boundary_tensors.get("event_sequence_base")
    if cells > 1:
        if not isinstance(boundary_sequence_base, torch.Tensor) or (
            boundary_sequence_base.numel() != 1
        ):
            raise _fail(
                "streamed WDL first source event sequence frontier must be a scalar"
            )
        # The first B1 request supplies allocation shape.  Expand only this
        # bounded device-resident frontier so the allocator can construct the
        # B-wide owner; each later B1 source is copied into its physical row
        # below.  No request tuple or CPU staging bank is retained.
        boundary_tensors["event_sequence_base"] = (
            boundary_sequence_base.reshape(1).expand(cells).clone()
        )
        target_boundary = replace(target_boundary, tensors=boundary_tensors)

    target_request = replace(
        first_request,
        cuda_boundary=target_boundary,
        batch_cells=cells,
        logical_generation_count=(
            generation_rows
            if allow_short_technical_gate_generation_count
            else max(3, generation_rows)
        ),
        agent_id_allocation_generation_count=agent_id_allocation_generation_count,
        allow_short_technical_gate_generation_count=(
            allow_short_technical_gate_generation_count
        ),
        allow_partial_generation_source_cursor=(
            allow_partial_generation_source_cursor
        ),
    )
    source = allocate_direct_resident_source_v1(
        target_request,
        per_cell_ids=ordered_ids,
    )
    metadata_rows: list[Mapping[str, Any]] = [first_metadata]
    runtime_prefix_rows: list[Mapping[str, Any]] = [first_runtime_prefix]
    extra_tensors: dict[str, torch.Tensor] = {}
    target_device = source.live_state["energy"].device
    first_boundary_tensors = first_request.cuda_boundary.tensors
    for name in extra_names:
        first_value = first_boundary_tensors[name]
        extra_tensors[name] = torch.empty(
            (cells, *first_value.shape),
            dtype=first_value.dtype,
            device=target_device,
        )
        extra_tensors[name][0].copy_(first_value, non_blocking=False)

    # The B-wide seed above is allocation shape only.  Its canonical sidecars
    # are intentionally not restored as a multicell set, and its fresh seed
    # may therefore not represent cell zero.  Consume the first B1 request
    # through the same bounded row-copy path used for every successor cell so
    # the resident owner contains exact state for all ordered cells without
    # retaining a B-wide host checkpoint bank.
    first_source = allocate_direct_resident_source_v1(
        first_request,
        per_cell_ids=(ordered_ids[0],),
    )
    source = _stream_copy_source_cell_v2(
        source,
        first_source,
        cell=0,
        cells=cells,
    )
    del first_source

    expected_event_capacity = first_request.event_capacity
    expected_projection_digest = first_request.source_import_projection_digest_sha256
    for cell in range(1, cells):
        request = request_factory(cell)
        if not isinstance(request, DirectResidentSourceAllocationRequestV1):
            raise _fail("streamed WDL request factory returned the wrong request type")
        request = _stage_wdl_policy_birth_requests_on_boundary_cuda_v2((request,))[0]
        if request.batch_cells != 1:
            raise _fail("streamed WDL request factory must return B1 requests")
        if request.cpu_request_staging_enable is not False:
            raise _fail("streamed physical B136 forbids CPU request staging")
        if request.event_capacity != expected_event_capacity:
            raise _fail(
                "streamed physical B136 requires one source-derived event capacity"
            )
        if (
            expected_projection_digest is not None
            and request.source_import_projection_digest_sha256 is not None
            and len(request.source_import_projection_digest_sha256) != 64
        ):
            raise _fail("streamed source import digest is malformed")
        projection = request.import_projection
        metadata = (
            projection.get("metadata") if isinstance(projection, Mapping) else None
        )
        if not isinstance(metadata, Mapping):
            raise _fail(f"streamed WDL source {cell} lacks metadata")
        normalized_metadata = _required_wdl_metadata(metadata)
        if normalized_metadata.get("generation") != initial_generation:
            raise _fail(f"streamed WDL source {cell} generation differs")
        if (
            normalized_metadata.get("tick") != 0
            and not allow_partial_generation_source_cursor
        ):
            raise _fail(f"streamed WDL source {cell} is not at an owner boundary")
        if normalized_metadata.get("native_runtime_fixed_policy_family") != "mlp":
            raise _fail(f"streamed WDL source {cell} is not fixed MLP")
        capabilities = normalized_metadata.get("capability_mask")
        if dict(capabilities or {}) != dict(first_capabilities):
            raise _fail(f"streamed WDL source {cell} capability mask differs")
        if _streamed_configuration_signature_v2(request.configuration) != first_signature:
            raise _fail(f"streamed WDL source {cell} configuration differs")
        if getattr(request.configuration, "steps", None) != first_steps:
            raise _fail(f"streamed WDL source {cell} tick geometry differs")
        for name in (
            "wdl_delayed_hazard_schema_version",
            "wdl_delayed_hazard_delay_steps",
            "wdl_delayed_hazard_cue_duration_steps",
            "wdl_delayed_hazard_active_duration_steps",
            "wdl_delayed_hazard_event_period_steps",
            "wdl_delayed_hazard_intensity",
            "wdl_delayed_hazard_observation_dropout_rate",
            "wdl_delayed_hazard_cue_enable",
            "wdl_delayed_hazard_damage_enable",
            "wdl_delayed_hazard_deterministic_schedule_enable",
        ):
            if normalized_metadata.get(name) != first_metadata.get(name):
                raise _fail(f"streamed WDL source {cell} has mixed {name}")
        one_source = allocate_direct_resident_source_v1(
            request,
            per_cell_ids=(ordered_ids[cell],),
        )
        source = _stream_copy_source_cell_v2(
            source,
            one_source,
            cell=cell,
            cells=cells,
        )
        boundary_tensors = request.cuda_boundary.tensors
        for name in extra_names:
            value = boundary_tensors.get(name)
            if not isinstance(value, torch.Tensor):
                raise _fail(f"streamed WDL boundary {cell} lacks tensor {name!r}")
            if tuple(value.shape) != tuple(extra_tensors[name].shape[1:]):
                raise _fail(f"streamed WDL boundary {cell} tensor {name!r} shape differs")
            extra_tensors[name][cell].copy_(value, non_blocking=False)
        metadata_rows.append(normalized_metadata)
        runtime_prefix_rows.append(_tick_native_runtime_prefix_row_v1(projection))
        del one_source, request

    compact_inputs = dict(source.checkpoint_projection_inputs)
    compact_inputs["pre_generation_canonical_projection"] = first_projection
    compact_inputs.pop("pre_generation_canonical_projections", None)
    if source_projection_cursor_factory is not None:
        compact_inputs["source_projection_cursor_factory"] = (
            source_projection_cursor_factory
        )
    if source_import_cursor_receipt is not None:
        compact_inputs["source_import_cursor_receipt"] = dict(
            source_import_cursor_receipt
        )
    source = replace(source, checkpoint_projection_inputs=compact_inputs)
    return _StreamedWdlSourceAssemblyV2(
        source=source,
        metadata_rows=tuple(metadata_rows),
        runtime_prefix_rows=tuple(runtime_prefix_rows),
        first_projection=first_projection,
        extra_tensors=extra_tensors,
        source_projection_cursor_factory=source_projection_cursor_factory,
        source_import_cursor_receipt=(
            dict(source_import_cursor_receipt)
            if source_import_cursor_receipt is not None
            else None
        ),
    )


def _wdl_history_capacity_v3(
    *,
    slot_capacity: int,
    ticks_per_generation: int,
    max_births_per_tick: int,
) -> int:
    for name, value, minimum in (
        ("slot_capacity", slot_capacity, 1),
        ("ticks_per_generation", ticks_per_generation, 1),
        ("max_births_per_tick", max_births_per_tick, 0),
    ):
        if type(value) is not int or value < minimum:
            raise _fail(f"WDL {name} is outside the compact-history contract")
    return slot_capacity + ticks_per_generation * max_births_per_tick


def _prepared_ticks_per_generation_v2(prepared: "PreparedDirectResidentWdlV2") -> int:
    ticks = getattr(
        prepared.patch.source, "ticks_per_generation", WDL_TICKS_PER_GENERATION_V2
    )
    if ticks not in WDL_SUPPORTED_TICKS_PER_GENERATION_V2:
        raise _fail("prepared WDL ticks_per_generation must be 80 or 160")
    return ticks


def _wdl_history_probe_start_v3(key: int, capacity: int) -> int:
    if type(key) is not int or key < 0:
        raise _fail("WDL compact-history keys must be nonnegative exact integers")
    if type(capacity) is not int or capacity < 1:
        raise _fail("WDL compact-history capacity must be positive")
    uint64_mask = (1 << 64) - 1
    wrapped_product = (
        (key & uint64_mask) * 11400714819323198485
    ) & uint64_mask
    return wrapped_product % capacity


def _compact_flag_upsert_v3(
    keys: torch.Tensor,
    flags: torch.Tensor,
    *,
    key: int,
    flag: int,
) -> int:
    if (
        keys.dtype != torch.int64
        or flags.dtype != torch.uint8
        or keys.ndim != 1
        or flags.shape != keys.shape
    ):
        raise _fail("WDL compact agent table has malformed key/flag tensors")
    if type(flag) is not int or flag <= 0 or flag & ~WDL_AGENT_FLAG_MASK_V3:
        raise _fail("WDL compact agent flag is outside the v3 bit contract")
    capacity = int(keys.numel())
    start = _wdl_history_probe_start_v3(key, capacity)
    for probe in range(capacity):
        slot = (start + probe) % capacity
        resident = int(keys[slot].item())
        if resident == key:
            flags[slot] = int(flags[slot].item()) | flag
            return slot
        if resident == -1:
            keys[slot] = key
            flags[slot] = flag
            return slot
        if resident < -1:
            raise _fail("WDL compact agent table contains a malformed key")
    raise _fail("WDL compact agent history table is full")


def _compact_lineage_accumulate_v3(
    keys: torch.Tensor,
    damage: torch.Tensor,
    *,
    key: int,
    delta: float,
) -> int:
    if (
        keys.dtype != torch.int64
        or damage.dtype != torch.float64
        or keys.ndim != 1
        or damage.shape != keys.shape
    ):
        raise _fail("WDL compact lineage table has malformed tensors")
    if type(delta) is not float or not math.isfinite(delta):
        raise _fail("WDL compact lineage damage must be finite")
    capacity = int(keys.numel())
    start = _wdl_history_probe_start_v3(key, capacity)
    for probe in range(capacity):
        slot = (start + probe) % capacity
        resident = int(keys[slot].item())
        if resident == key:
            damage[slot] = float(damage[slot].item()) + delta
            return slot
        if resident == -1:
            keys[slot] = key
            damage[slot] = delta
            return slot
        if resident < -1:
            raise _fail("WDL compact lineage table contains a malformed key")
    raise _fail("WDL compact lineage history table is full")


def _decode_agent_flag_keys_v3(
    keys: torch.Tensor,
    flags: torch.Tensor,
    *,
    flag: int,
) -> list[int]:
    if (
        keys.dtype != torch.int64
        or flags.dtype != torch.uint8
        or keys.ndim != 1
        or flags.shape != keys.shape
    ):
        raise _fail("WDL compact checkpoint agent table is malformed")
    if type(flag) is not int or flag <= 0 or flag & ~WDL_AGENT_FLAG_MASK_V3:
        raise _fail("WDL compact checkpoint flag is invalid")
    decoded: list[int] = []
    seen: set[int] = set()
    for key_value, flag_value in zip(keys.tolist(), flags.tolist(), strict=True):
        key = int(key_value)
        bits = int(flag_value)
        if bits & ~WDL_AGENT_FLAG_MASK_V3:
            raise _fail("WDL compact checkpoint agent flags contain unknown bits")
        if key == -1:
            if bits != 0:
                raise _fail("WDL compact checkpoint empty agent slot has flags")
            continue
        if key < 0 or bits == 0 or key in seen:
            raise _fail("WDL compact checkpoint agent table contains invalid keys")
        seen.add(key)
        if bits & flag:
            decoded.append(key)
    return sorted(decoded)


def _allocate_reporting(
    *,
    device: torch.device,
    generation_rows: int,
    history_capacity: int,
    cells: int = 1,
    ticks_per_generation: int = WDL_TICKS_PER_GENERATION_V2,
) -> tuple[torch.Tensor, ...]:
    if type(cells) is not int or cells < 1:
        raise _fail("WDL reporting cells must be a positive exact integer")
    if ticks_per_generation not in WDL_SUPPORTED_TICKS_PER_GENERATION_V2:
        raise _fail("WDL reporting ticks_per_generation must be 80 or 160")
    ticks = ticks_per_generation
    return (
        torch.zeros(
            (cells, generation_rows, ticks, WDL_TICK_INTEGER_METRIC_COUNT_V2),
            dtype=torch.int64,
            device=device,
        ),
        torch.zeros(
            (cells, generation_rows, ticks), dtype=torch.float64, device=device
        ),
        torch.zeros(
            (cells, generation_rows, ticks, WDL_EVENT_INTEGER_COUNT_V2),
            dtype=torch.int64,
            device=device,
        ),
        torch.zeros(
            (cells, generation_rows, ticks), dtype=torch.float32, device=device
        ),
        torch.zeros(
            (cells, generation_rows, ticks), dtype=torch.int32, device=device
        ),
        torch.zeros(
            (cells, generation_rows, WDL_GENERATION_METRIC_COUNT_V2),
            dtype=torch.float64,
            device=device,
        ),
        torch.full(
            (cells, generation_rows, history_capacity),
            -1,
            dtype=torch.int64,
            device=device,
        ),
        torch.zeros(
            (cells, generation_rows, history_capacity),
            dtype=torch.uint8,
            device=device,
        ),
        torch.full(
            (cells, generation_rows, history_capacity),
            -1,
            dtype=torch.int64,
            device=device,
        ),
        torch.zeros(
            (cells, generation_rows, history_capacity),
            dtype=torch.float64,
            device=device,
        ),
    )


def _stage_wdl_policy_birth_mapping_on_boundary_cuda_v2(
    policy_state: Mapping[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Copy CPU-staged WDL policy birth tensors onto the exact CUDA boundary."""

    if not isinstance(policy_state, Mapping):
        raise _fail("WDL policy-birth state must be a mapping")
    staged: dict[str, Any] = {}
    for name, value in policy_state.items():
        if isinstance(value, torch.Tensor) and value.device != device:
            staged[name] = value.to(device=device, non_blocking=False)
        else:
            staged[name] = value
    return staged


def _stage_wdl_policy_birth_requests_on_boundary_cuda_v2(
    requests: Sequence[DirectResidentSourceAllocationRequestV1],
) -> tuple[DirectResidentSourceAllocationRequestV1, ...]:
    """Adapt generic CPU-staged birth tape maps for CUDA-only WDL allocation."""

    staged_requests: list[DirectResidentSourceAllocationRequestV1] = []
    for request in requests:
        if (
            getattr(request, "cpu_request_staging_enable", False) is True
            or getattr(request, "direct_batch_host_import_enable", False) is True
        ):
            staged_requests.append(request)
            continue
        birth_state = request.generation_birth_state
        policy_state = getattr(birth_state, "policy_state", None)
        turnover_policy_state = getattr(birth_state, "turnover_policy_state", None)
        if policy_state is None and turnover_policy_state is None:
            staged_requests.append(request)
            continue
        boundary_tensors = getattr(request.cuda_boundary, "tensors", None)
        if not isinstance(boundary_tensors, Mapping):
            raise _fail("WDL CUDA boundary tensors are missing")
        devices = {
            value.device
            for value in boundary_tensors.values()
            if isinstance(value, torch.Tensor) and value.is_cuda
        }
        if len(devices) != 1:
            raise _fail("WDL CUDA boundary must have one explicit CUDA device")
        device = next(iter(devices))
        staged_birth_state = replace(
            birth_state,
            policy_state=_stage_wdl_policy_birth_mapping_on_boundary_cuda_v2(
                policy_state,
                device=device,
            ),
            turnover_policy_state=_stage_wdl_policy_birth_mapping_on_boundary_cuda_v2(
                turnover_policy_state,
                device=device,
            ),
        )
        staged_requests.append(
            replace(request, generation_birth_state=staged_birth_state)
        )
    return tuple(staged_requests)


def build_prepared_direct_resident_wdl_v2(
    runner: Any | None,
    *,
    qualified_cuda_module: Any,
    generation_rows: int = 2,
    initial_generation: int = 0,
    source_checkpoint_projection: Mapping[str, Any] | None = None,
    per_cell_requests: Sequence[DirectResidentSourceAllocationRequestV1] | None = None,
    source_checkpoint_projections: Sequence[Mapping[str, Any]] | None = None,
    cell_ids: Sequence[str] | None = None,
    agent_id_allocation_generation_count: int | None = None,
    source_capability_mask: Mapping[str, bool] | None = None,
    cell_id: str | None = None,
    cuda_device: str = "cuda",
    allow_short_technical_gate_generation_count: bool = False,
    allow_partial_generation_source_cursor: bool = False,
    jungle_overlay: bool = False,
    eed_overlay: bool = False,
    hapb_host_policy_enable: bool = False,
) -> PreparedDirectResidentWdlV2:
    """Build one WDL overlay around the unchanged Patch resident ABI."""

    if type(generation_rows) is not int or generation_rows < 1:
        raise _fail("generation_rows must be a positive exact integer")
    if type(initial_generation) is not int or initial_generation < 0:
        raise _fail("WDL initial_generation must be a nonnegative exact integer")
    if type(allow_partial_generation_source_cursor) is not bool:
        raise _fail("WDL partial-generation source authority must be an exact bool")
    if cuda_device not in {"cuda", "cuda:0"}:
        raise _fail("WDL CUDA source requires cuda or cuda:0")
    if type(hapb_host_policy_enable) is not bool:
        raise _fail("HAPB host-policy enable must be an exact bool")
    capability = _validated_capability(qualified_cuda_module)
    request_generation_rows = (
        generation_rows
        if allow_short_technical_gate_generation_count
        else max(3, generation_rows)
    )
    if per_cell_requests is None:
        if runner is None:
            raise _fail("WDL preparation requires runner or per_cell_requests")
        if source_checkpoint_projections is not None:
            raise _fail("B1 WDL route uses source_checkpoint_projection")
        allocation_cell_ids = _wdl_allocation_cell_ids_v2(cell_id)
        envelope = _patch_envelope_with_wdl_masked(
            runner,
            allow_short_technical_gate_generation_count=(
                allow_short_technical_gate_generation_count
            ),
        )
        source_capabilities = _wdl_source_capabilities_v2(
            source_capability_mask,
            jungle_overlay=jungle_overlay,
            eed_overlay=eed_overlay,
        )
        fresh_projection = build_import_projection(
            runner,
            source_capabilities,
            generation=initial_generation,
            tick=0,
        )
        if source_checkpoint_projection is None:
            fresh_projection = canonical_generation_zero_source_projection_v1(
                fresh_projection
            )
            fresh_projection["metadata"][
                "native_runtime_architecture_b_compact_reporting_only_v1"
            ] = False
            fresh_projection["metadata"]["import_digest"] = projection_digest(
                {
                    "metadata": fresh_projection["metadata"],
                    "tensors": fresh_projection["tensors"],
                }
            )
            projection: Any = fresh_projection
        else:
            projection = source_checkpoint_projection
        projection_metadata = (
            projection.get("metadata")
            if isinstance(projection, Mapping)
            else getattr(projection, "metadata", None)
        )
        if not isinstance(projection_metadata, Mapping):
            raise _fail("WDL source projection is missing metadata")
        projection_capabilities = projection_metadata.get("capability_mask")
        if (
            not isinstance(projection_capabilities, Mapping)
            or dict(projection_capabilities) != source_capabilities
        ):
            raise _fail(
                "WDL source projection capability mask differs from the "
                "configured resident route"
            )
        if (
            projection_metadata.get("generation") != initial_generation
            or (
                projection_metadata.get("tick") != 0
                and not allow_partial_generation_source_cursor
            )
            or projection_metadata.get("native_runtime_fixed_policy_family") != "mlp"
        ):
            raise _fail("WDL source checkpoint is not the requested fixed-MLP boundary")
        metadata = _required_wdl_metadata(projection_metadata)
        cpu_request_staging_enable = getattr(
            runner.sim_cfg,
            "cuda_architecture_b_cpu_request_staging_enable",
            False,
        )
        if type(cpu_request_staging_enable) is not bool:
            raise _fail(
                "runner cuda_architecture_b_cpu_request_staging_enable must be an exact bool"
            )
        boundary = import_cpu_owner_boundary_to_cuda(
            projection,
            event_capacity=QUALIFIED_EVENT_CAPACITY_V1,
            device="cpu" if cpu_request_staging_enable else cuda_device,
            allow_cpu_staging=cpu_request_staging_enable,
            allow_partial_generation_source_cursor=(
                allow_partial_generation_source_cursor
            ),
            host_authoritative_tensor_names=(
                ("mlp_pack",) if hapb_host_policy_enable else ()
            ),
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
        if cpu_request_staging_enable:
            birth_state_kwargs["policy_tape_staging_device"] = "cpu"
        birth_state = prepare_generation_birth_state_cuda(
            boundary,
            projection,
            **birth_state_kwargs,
        )
        turnover_selection_mode, turnover_selection_seed = (
            _wdl_turnover_selection_authority_v2(runner.sim_cfg)
        )
        lifecycle_mode = getattr(runner.sim_cfg, "lifecycle_mode", 0)
        if type(lifecycle_mode) is not int or lifecycle_mode not in {0, 1}:
            raise _fail("runner lifecycle_mode must be exact integer 0 or 1")
        generation_turnover_enabled = lifecycle_mode == 0
        ticks_per_generation = getattr(
            runner.sim_cfg, "steps_per_generation", WDL_TICKS_PER_GENERATION_V2
        )
        if ticks_per_generation not in WDL_SUPPORTED_TICKS_PER_GENERATION_V2:
            raise _fail("WDL runner ticks_per_generation must be 80 or 160")
        configuration = replace(
            CudaGenerationConfig.from_metadata(
                metadata,
                steps=ticks_per_generation,
            ),
            generation=initial_generation,
            start_tick=int(projection_metadata["tick"]),
            max_births_per_tick=max_births,
            ordinary_births_enabled=ordinary_births_enabled,
            generation_turnover_enabled=generation_turnover_enabled,
            lifecycle_mode=lifecycle_mode,
            turnover_selection_mode=turnover_selection_mode,
            turnover_selection_seed=turnover_selection_seed,
            genome_upgrade_v2_enable=bool(
                getattr(runner.sim_cfg, "genome_upgrade_v2_enable", False)
            ),
            genome_upgrade_v2_founder_profile_id=str(
                getattr(
                    runner.sim_cfg,
                    "genome_upgrade_v2_founder_profile_id",
                    "neutral",
                )
            ),
            genome_upgrade_v2_founder_profile_sha256=str(
                getattr(
                    runner.sim_cfg,
                    "genome_upgrade_v2_founder_profile_sha256",
                    "",
                )
            ),
            genome_v2_checkpoint_authority_binding=tuple(
                getattr(
                    runner.sim_cfg,
                    "genome_v2_checkpoint_authority_binding",
                    (),
                )
            ),
            genome_v2_causal_capture_enable=bool(
                getattr(runner.sim_cfg, "genome_v2_causal_capture_enable", False)
            ),
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
                logical_generation_count=request_generation_rows,
                agent_id_allocation_generation_count=(
                    agent_id_allocation_generation_count
                ),
                allow_short_technical_gate_generation_count=(
                    allow_short_technical_gate_generation_count
                ),
                allow_partial_generation_source_cursor=(
                    allow_partial_generation_source_cursor
                ),
                cpu_request_staging_enable=cpu_request_staging_enable,
            ),
        )
        cell_ids_tuple = allocation_cell_ids or ("unbound-cell-0",)
        projections = (projection,)
        metadata_rows = [metadata]
    else:
        if runner is not None or source_checkpoint_projection is not None:
            raise _fail("multicell WDL route cannot also supply runner/B1 checkpoint")
        if isinstance(per_cell_requests, (str, bytes)) or not isinstance(
            per_cell_requests, Sequence
        ):
            raise _fail("per_cell_requests must be a sequence")
        requests = tuple(per_cell_requests)
        admitted_max = 256
        if not 1 <= len(requests) <= admitted_max or any(
            not isinstance(item, DirectResidentSourceAllocationRequestV1)
            for item in requests
        ):
            raise _fail(
                f"per_cell_requests must contain B1..B{admitted_max} public requests"
            )
        if cell_ids is None:
            cell_ids_tuple = tuple(str(index) for index in range(len(requests)))
        else:
            cell_ids_tuple = _validated_wdl_cell_ids_v2(
                cell_ids,
                count=len(requests),
            )
        requests = tuple(
            replace(
                request,
                batch_cells=len(requests),
                logical_generation_count=request_generation_rows,
                agent_id_allocation_generation_count=(
                    agent_id_allocation_generation_count
                ),
                allow_short_technical_gate_generation_count=(
                    allow_short_technical_gate_generation_count
                ),
                allow_partial_generation_source_cursor=(
                    allow_partial_generation_source_cursor
                ),
            )
            for request in requests
        )
        projections = tuple(request.import_projection for request in requests)
        if source_checkpoint_projections is not None:
            checkpoints = tuple(source_checkpoint_projections)
            if len(checkpoints) != len(requests) or any(
                request.import_projection is not checkpoint
                for request, checkpoint in zip(requests, checkpoints, strict=True)
            ):
                raise _fail("ordered source checkpoints must be bound without substitution")
        metadata_rows: list[Mapping[str, Any]] = []
        for cell, projection in enumerate(projections):
            projection_metadata = (
                projection.get("metadata")
                if isinstance(projection, Mapping)
                else getattr(projection, "metadata", None)
            )
            if not isinstance(projection_metadata, Mapping):
                raise _fail(f"WDL source projection {cell} is missing metadata")
            projection_capabilities = projection_metadata.get("capability_mask")
            if (
                not isinstance(projection_capabilities, Mapping)
                or dict(projection_capabilities)
                != _wdl_source_capabilities_v2(
                    projection_capabilities,
                    jungle_overlay=jungle_overlay,
                    eed_overlay=eed_overlay,
                )
            ):
                raise _fail(
                    f"WDL source projection {cell} capability mask differs from "
                    "the configured resident route"
                )
            if (
                projection_metadata.get("generation") != initial_generation
                or (
                    projection_metadata.get("tick") != 0
                    and not allow_partial_generation_source_cursor
                )
                or projection_metadata.get("native_runtime_fixed_policy_family")
                != "mlp"
            ):
                raise _fail(
                    f"WDL source projection {cell} is not the requested fixed-MLP boundary"
                )
            metadata_rows.append(_required_wdl_metadata(projection_metadata))
        metadata = metadata_rows[0]
        shared_fields = (
            "wdl_delayed_hazard_schema_version",
            "wdl_delayed_hazard_delay_steps",
            "wdl_delayed_hazard_cue_duration_steps",
            "wdl_delayed_hazard_active_duration_steps",
            "wdl_delayed_hazard_event_period_steps",
            "wdl_delayed_hazard_intensity",
            "wdl_delayed_hazard_observation_dropout_rate",
            "wdl_delayed_hazard_cue_enable",
            "wdl_delayed_hazard_damage_enable",
            "wdl_delayed_hazard_deterministic_schedule_enable",
        )
        for name in shared_fields:
            expected = metadata.get(name)
            if any(row.get(name) != expected for row in metadata_rows[1:]):
                raise _fail(
                    f"mixed per-cell {name} is not representable by WDL/fixed-MLP"
                )
    requests = _stage_wdl_policy_birth_requests_on_boundary_cuda_v2(requests)
    configuration = requests[0].configuration
    ticks_per_generation = getattr(configuration, "steps", None)
    if ticks_per_generation not in WDL_SUPPORTED_TICKS_PER_GENERATION_V2:
        raise _fail("WDL request ticks_per_generation must be 80 or 160")
    if any(
        getattr(request.configuration, "steps", None) != ticks_per_generation
        for request in requests[1:]
    ):
        raise _fail("mixed ticks_per_generation is not representable by WDL/fixed-MLP")
    source = allocate_direct_resident_source_v1(
        requests[0],
        per_cell_requests=requests if len(requests) > 1 else None,
        per_cell_ids=cell_ids_tuple,
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
                f"WDL/fixed-MLP CUDA boundaries are missing WDL state: {missing_wdl}"
            )
        cpu_staged = getattr(requests[0], "cpu_request_staging_enable", False)
        direct_batch_host_import = getattr(
            requests[0], "direct_batch_host_import_enable", False
        )
        if type(cpu_staged) is not bool or any(
            getattr(request, "cpu_request_staging_enable", False) is not cpu_staged
            for request in requests
        ):
            raise _fail("WDL CPU staging authority must be uniform within one batch")
        if type(direct_batch_host_import) is not bool or any(
            getattr(request, "direct_batch_host_import_enable", False)
            is not direct_batch_host_import
            for request in requests
        ):
            raise _fail(
                "WDL direct batch host-import authority must be uniform within one batch"
            )
        wdl_live_state = {}
        for name in WDL_STATE_FIELDS_V2:
            values = tuple(request.cuda_boundary.tensors[name] for request in requests)
            if cpu_staged:
                first = values[0]
                staged = torch.empty(
                    (len(values), *first.shape), dtype=first.dtype, device=source.live_state["energy"].device
                )
                for cell, value in enumerate(values):
                    staged[cell].copy_(value, non_blocking=False)
                wdl_live_state[name] = staged
            elif direct_batch_host_import:
                first = values[0]
                staged = torch.empty(
                    (len(values), *first.shape),
                    dtype=first.dtype,
                    device=source.live_state["energy"].device,
                ).contiguous()
                for cell, value in enumerate(values):
                    staged[cell].copy_(value, non_blocking=False)
                wdl_live_state[name] = staged
            else:
                wdl_live_state[name] = torch.stack(values, dim=0)
        source = replace(
            source,
            live_state={**dict(source.live_state), **wdl_live_state},
            live_banks={**dict(source.live_banks), **wdl_live_state},
        )
    if jungle_overlay:
        recovery = source.live_state.get("patch_recovery_activity")
        visibility_rows = []
        for cell, request in enumerate(requests):
            visibility = request.cuda_boundary.tensors.get("patch_visibility")
            if (
                not isinstance(visibility, torch.Tensor)
                or visibility.dtype != torch.float32
                or visibility.ndim != 2
            ):
                raise _fail(
                    f"WDL/Jungle CUDA boundary {cell} lacks patch_visibility"
                )
            visibility_rows.append(visibility)
        if getattr(requests[0], "cpu_request_staging_enable", False) is True:
            first = visibility_rows[0]
            visibility = torch.empty(
                (len(visibility_rows), *first.shape),
                dtype=first.dtype,
                device=source.live_state["energy"].device,
            )
            for cell, value in enumerate(visibility_rows):
                visibility[cell].copy_(value, non_blocking=False)
        elif getattr(requests[0], "direct_batch_host_import_enable", False) is True:
            first = visibility_rows[0]
            visibility = torch.empty(
                (len(visibility_rows), *first.shape),
                dtype=first.dtype,
                device=recovery.device,
            ).contiguous()
            for cell, value in enumerate(visibility_rows):
                visibility[cell].copy_(value, non_blocking=False)
        else:
            visibility = torch.stack(visibility_rows, dim=0).contiguous()
        if (
            not isinstance(recovery, torch.Tensor)
            or visibility.shape != recovery.shape
            or visibility.device != recovery.device
        ):
            raise _fail("WDL/Jungle visibility does not match Patch recovery state")
        source = replace(
            source,
            live_banks={**dict(source.live_banks), "patch_visibility": visibility},
        )
    device = source.live_state["energy"].device
    slot_capacity = int(source.live_state["energy"].size(1))
    outputs = dict(source.outputs)
    outputs["observations"] = torch.zeros(
        (len(requests), 1, slot_capacity, WDL_OBSERVATION_SIZE_V2),
        dtype=torch.float32,
        device=device,
    )
    _enable_wdl_genome_v2_causal_capture_v2(
        outputs,
        enabled=bool(configuration.genome_v2_causal_capture_enable),
        hapb_host_policy_enable=hapb_host_policy_enable,
    )
    source = replace(source, outputs=outputs)

    hidden_size = int(configuration.hidden_size)
    action_count = int(configuration.action_count)
    expected_pack_width = (
        hidden_size * WDL_OBSERVATION_SIZE_V2
        + hidden_size
        + hidden_size * hidden_size
        + hidden_size
        + action_count * hidden_size
        + action_count
    )
    actual_pack_width = int(source.live_state["mlp_pack"].size(-1))
    if actual_pack_width != expected_pack_width:
        raise _fail(
            "WDL policy pack does not use the 83-column observation contract: "
            f"expected={expected_pack_width} actual={actual_pack_width}"
        )

    state_tensors = tuple(
        source.live_state[name].contiguous() for name in WDL_STATE_FIELDS_V2
    )
    schedule_tensors = (
        torch.tensor(
            [row["wdl_base_seed"] for row in metadata_rows]
            if len(requests) > 1
            else [metadata["wdl_base_seed"]],
            dtype=torch.int64,
            device=device,
        ),
        torch.tensor(
            [row["wdl_next_absolute_step"] for row in metadata_rows]
            if len(requests) > 1
            else [metadata["wdl_next_absolute_step"]],
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
            "mixed max_births_per_tick is not representable by WDL/fixed-MLP"
        )
    history_capacity = _wdl_history_capacity_v3(
        slot_capacity=slot_capacity,
        ticks_per_generation=ticks_per_generation,
        max_births_per_tick=max_births_per_tick,
    )
    reporting_tensors = _allocate_reporting(
        device=device,
        generation_rows=generation_rows,
        history_capacity=history_capacity,
        cells=len(requests),
        ticks_per_generation=ticks_per_generation,
    )
    config = {
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
        "radius_by_cell": torch.tensor(
            [row["wdl_delayed_hazard_radius"] for row in metadata_rows],
            dtype=torch.int32,
            device=device,
        ),
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
        "history_abi_version": WDL_ABI_VERSION_V3,
        "history_capacity": history_capacity,
        "slot_capacity": slot_capacity,
        "max_births_per_tick": max_births_per_tick,
        "ticks_per_generation": ticks_per_generation,
        "generation_base": initial_generation,
        "cuda_capability": dict(getattr(requests[0], "cuda_capability", capability)),
        "wdl_cuda_capability": capability,
    }
    return PreparedDirectResidentWdlV2(
        patch=prepare_direct_resident_trajectory_v1(source),
        state_tensors=state_tensors,
        schedule_tensors=schedule_tensors,
        reporting_tensors=reporting_tensors,
        config=config,
        generation_rows=generation_rows,
        initial_projection=projections[0],
        cell_ids=cell_ids_tuple,
        source_projections=projections,
        metadata_rows=tuple(metadata_rows),
        runtime_prefix_rows=tuple(
            _tick_native_runtime_prefix_row_v1(projection)
            for projection in projections
        ),
    )


def build_prepared_direct_resident_wdl_v2_streaming(
    *,
    request_factory: Callable[[int], DirectResidentSourceAllocationRequestV1],
    qualified_cuda_module: Any,
    generation_rows: int = 2,
    initial_generation: int = 0,
    cell_ids: Sequence[str],
    agent_id_allocation_generation_count: int | None = None,
    allow_short_technical_gate_generation_count: bool = False,
    allow_partial_generation_source_cursor: bool = False,
    source_projection_cursor_factory: Callable[[], Any] | None = None,
    source_import_cursor_receipt: Mapping[str, Any] | None = None,
    source_capability_mask: Mapping[str, bool] | None = None,
    jungle_overlay: bool = False,
    eed_overlay: bool = False,
    extra_tensor_names: Sequence[str] = (),
) -> PreparedDirectResidentWdlV2:
    """Prepare one physical owner without a B-wide host request bank.

    This is the tick-native physical-B136 seam.  ``request_factory`` is called
    exactly once per ordered cell and every direct request is consumed before
    the next one is requested.  Legacy tuple preparation remains available to
    the bounded qualification fixtures and non-tick-native routes.
    """

    if type(generation_rows) is not int or generation_rows < 1:
        raise _fail("generation_rows must be a positive exact integer")
    if type(initial_generation) is not int or initial_generation < 0:
        raise _fail("initial_generation must be a nonnegative exact integer")
    capability = _validated_capability(qualified_cuda_module)
    expected_capabilities = _wdl_source_capabilities_v2(
        source_capability_mask,
        jungle_overlay=jungle_overlay,
        eed_overlay=eed_overlay,
    )
    assembly = _build_streamed_wdl_source_v2(
        request_factory=request_factory,
        qualified_cuda_module=qualified_cuda_module,
        cell_ids=cell_ids,
        generation_rows=generation_rows,
        initial_generation=initial_generation,
        agent_id_allocation_generation_count=agent_id_allocation_generation_count,
        allow_short_technical_gate_generation_count=(
            allow_short_technical_gate_generation_count
        ),
        allow_partial_generation_source_cursor=(
            allow_partial_generation_source_cursor
        ),
        extra_tensor_names=extra_tensor_names,
        expected_source_capabilities=expected_capabilities,
        source_projection_cursor_factory=source_projection_cursor_factory,
        source_import_cursor_receipt=source_import_cursor_receipt,
    )
    source = assembly.source
    extra = dict(assembly.extra_tensors)
    missing_wdl = [name for name in WDL_STATE_FIELDS_V2 if name not in extra]
    if missing_wdl:
        raise _fail(
            "streamed WDL preparation lacks WDL state tensors: "
            + ", ".join(missing_wdl)
        )
    device = source.live_state["energy"].device
    source_live_state = {
        **dict(source.live_state),
        **{name: extra[name] for name in WDL_STATE_FIELDS_V2},
    }
    source_live_banks = {
        **dict(source.live_banks),
        **{name: extra[name] for name in WDL_STATE_FIELDS_V2},
    }
    source = replace(
        source,
        live_state=source_live_state,
        live_banks=source_live_banks,
        outputs={
            **dict(source.outputs),
            "observations": torch.zeros(
                (
                    len(tuple(cell_ids)),
                    1,
                    int(source.live_state["energy"].size(1)),
                    WDL_OBSERVATION_SIZE_V2,
                ),
                dtype=torch.float32,
                device=device,
            ),
        },
    )
    outputs = dict(source.outputs)
    _enable_wdl_genome_v2_causal_capture_v2(
        outputs,
        enabled=bool(source.configuration.genome_v2_causal_capture_enable),
        hapb_host_policy_enable=False,
    )
    source = replace(source, outputs=outputs)
    configuration = source.configuration
    ticks_per_generation = getattr(configuration, "steps", None)
    if ticks_per_generation not in WDL_SUPPORTED_TICKS_PER_GENERATION_V2:
        raise _fail("streamed WDL source has unsupported ticks_per_generation")
    slot_capacity = int(source.live_state["energy"].size(1))
    max_births_per_tick = int(configuration.max_births_per_tick)
    history_capacity = _wdl_history_capacity_v3(
        slot_capacity=slot_capacity,
        ticks_per_generation=ticks_per_generation,
        max_births_per_tick=max_births_per_tick,
    )
    ordered_ids = tuple(cell_ids)
    metadata_rows = assembly.metadata_rows
    metadata = metadata_rows[0]
    reporting_tensors = _allocate_reporting(
        device=device,
        generation_rows=generation_rows,
        history_capacity=history_capacity,
        cells=len(ordered_ids),
        ticks_per_generation=ticks_per_generation,
    )
    config = {
        "enabled": True,
        "schema_version": int(metadata["wdl_delayed_hazard_schema_version"]),
        "delay_steps": int(metadata["wdl_delayed_hazard_delay_steps"]),
        "cue_duration_steps": int(metadata["wdl_delayed_hazard_cue_duration_steps"]),
        "active_duration_steps": int(metadata["wdl_delayed_hazard_active_duration_steps"]),
        "event_period_steps": int(metadata["wdl_delayed_hazard_event_period_steps"]),
        "radius": int(metadata["wdl_delayed_hazard_radius"]),
        "radius_by_cell": torch.tensor(
            [row["wdl_delayed_hazard_radius"] for row in metadata_rows],
            dtype=torch.int32,
            device=device,
        ),
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
        "history_abi_version": WDL_ABI_VERSION_V3,
        "history_capacity": history_capacity,
        "slot_capacity": slot_capacity,
        "max_births_per_tick": max_births_per_tick,
        "ticks_per_generation": ticks_per_generation,
        "generation_base": initial_generation,
        "cuda_capability": dict(
            assembly.source.checkpoint_projection_inputs.get(
                "cuda_capability", capability
            )
        ),
        "wdl_cuda_capability": capability,
    }
    return PreparedDirectResidentWdlV2(
        patch=prepare_direct_resident_trajectory_v1(source),
        state_tensors=tuple(
            source.live_state[name].contiguous() for name in WDL_STATE_FIELDS_V2
        ),
        schedule_tensors=(
            torch.tensor(
                [row["wdl_base_seed"] for row in metadata_rows],
                dtype=torch.int64,
                device=device,
            ),
            torch.tensor(
                [row["wdl_next_absolute_step"] for row in metadata_rows],
                dtype=torch.int64,
                device=device,
            ),
        ),
        reporting_tensors=reporting_tensors,
        config=config,
        generation_rows=generation_rows,
        initial_projection=assembly.first_projection,
        cell_ids=ordered_ids,
        source_projections=(),
        metadata_rows=tuple(metadata_rows),
        runtime_prefix_rows=assembly.runtime_prefix_rows,
        source_projection_cursor_factory=(
            assembly.source_projection_cursor_factory
        ),
        source_import_cursor_receipt=assembly.source_import_cursor_receipt,
        streamed_extra_tensors=dict(assembly.extra_tensors),
    )


def execute_direct_resident_wdl_physical_v2(
    module: Any,
    prepared: PreparedDirectResidentWdlV2,
    *,
    generation_count: int,
    ticks_per_generation: int = WDL_TICKS_PER_GENERATION_V2,
) -> dict[str, Any]:
    """Execute one native-owned physical g1 or g2 WDL qualification segment."""

    if not isinstance(prepared, PreparedDirectResidentWdlV2):
        raise _fail("prepared must be PreparedDirectResidentWdlV2")
    if type(generation_count) is not int or generation_count not in {1, 2}:
        raise _fail("physical WDL qualification requires generation_count 1 or 2")
    if generation_count > prepared.generation_rows:
        raise _fail("WDL reporting allocation does not cover the requested generations")
    if (
        type(ticks_per_generation) is not int
        or ticks_per_generation < 1
        or ticks_per_generation not in WDL_SUPPORTED_TICKS_PER_GENERATION_V2
    ):
        raise _fail("physical WDL ticks_per_generation must be 80 or 160")
    entrypoint = getattr(module, WDL_PHYSICAL_ENTRYPOINT_V3, None)
    if not callable(entrypoint):
        raise _fail(f"CUDA extension is missing callable {WDL_PHYSICAL_ENTRYPOINT_V3}")
    arguments = list(prepared.patch.arguments)
    physical_ticks = generation_count * ticks_per_generation
    arguments[PHYSICAL_RUNTIME_BOOKKEEPING_ARGUMENT_INDEX_V2] = arguments[
        PHYSICAL_RUNTIME_BOOKKEEPING_ARGUMENT_INDEX_V2
    ][:, :physical_ticks]
    full_tick_outputs = list(
        arguments[PHYSICAL_FULL_TICK_OUTPUTS_ARGUMENT_INDEX_V2]
    )
    full_tick_outputs[-1] = full_tick_outputs[-1][:generation_count]
    arguments[PHYSICAL_FULL_TICK_OUTPUTS_ARGUMENT_INDEX_V2] = full_tick_outputs
    patch_stats = list(arguments[PHYSICAL_PATCH_STATS_ARGUMENT_INDEX_V2])
    patch_stats[0] = patch_stats[0][:, :physical_ticks]
    arguments[PHYSICAL_PATCH_STATS_ARGUMENT_INDEX_V2] = patch_stats
    arguments[TICKS_ARGUMENT_INDEX_V2] = ticks_per_generation
    arguments[GENERATION_COUNT_ARGUMENT_INDEX_V1] = generation_count
    raw = entrypoint(
        list(prepared.state_tensors),
        list(prepared.schedule_tensors),
        list(prepared.reporting_tensors),
        dict(prepared.config),
        *arguments,
    )
    if not isinstance(raw, Mapping):
        raise _fail("WDL physical CUDA entrypoint returned a non-mapping result")
    result = dict(raw)
    expected = {
        "wdl_native_owner": True,
        "wdl_native_controller_abi_version": WDL_ABI_VERSION_V3,
        "wdl_physical_gate_entrypoint": True,
        "wdl_python_tick_or_generation_ownership": False,
        "hidden_fallback": False,
        "runtime_jit": False,
    }
    for name, expected_value in expected.items():
        actual = result.get(name)
        if type(actual) is not type(expected_value) or actual != expected_value:
            raise _fail(
                f"WDL physical result {name} mismatch: "
                f"expected={expected_value!r} actual={actual!r}"
            )
    return result


def execute_direct_resident_wdl_v2(
    module: Any,
    prepared: PreparedDirectResidentWdlV2,
    *,
    logical_generation_count: int,
    run_id: str,
) -> dict[str, Any]:
    """Execute one complete WDL trajectory through reusable native g2 windows."""

    if not isinstance(prepared, PreparedDirectResidentWdlV2):
        raise _fail("prepared must be PreparedDirectResidentWdlV2")
    minimum_generation_count = (
        1
        if getattr(
            prepared.patch.source,
            "allow_short_technical_gate_generation_count",
            False,
        )
        is True
        else WDL_RESIDENT_GENERATION_MIN_V2
    )
    if (
        type(logical_generation_count) is not int
        or not minimum_generation_count
        <= logical_generation_count
        <= WDL_RESIDENT_GENERATION_MAX_V2
    ):
        raise _fail(
            "resident WDL requires logical_generation_count in "
            f"[{minimum_generation_count},128]"
        )
    if logical_generation_count > prepared.generation_rows:
        raise _fail("WDL reporting allocation does not cover the requested generations")
    if prepared.patch.source.logical_generation_count != logical_generation_count:
        raise _fail("WDL source identity capacity does not match the requested horizon")
    entrypoint = getattr(module, WDL_DIRECT_ENTRYPOINT_V3, None)
    if not callable(entrypoint):
        raise _fail(f"CUDA extension is missing callable {WDL_DIRECT_ENTRYPOINT_V3}")
    control = create_run_control_v2(module, run_id=run_id)
    metadata_value = (
        prepared.initial_projection.get("metadata")
        if isinstance(prepared.initial_projection, Mapping)
        else getattr(prepared.initial_projection, "metadata", None)
    )
    if not isinstance(metadata_value, Mapping):
        raise _fail("WDL initial projection metadata is not a mapping")
    metadata = dict(metadata_value)
    raw = entrypoint(
        control,
        list(prepared.state_tensors),
        list(prepared.schedule_tensors),
        list(prepared.reporting_tensors),
        dict(prepared.config),
        *prepared.patch.arguments,
        logical_generation_count,
        metadata.get("policy_weight_mutation_enable"),
        metadata.get("policy_weight_mutation_mode"),
        metadata.get("policy_weight_mutation_rate"),
        metadata.get("policy_weight_mutation_std"),
    )
    if not isinstance(raw, Mapping):
        raise _fail("WDL resident CUDA entrypoint returned a non-mapping result")
    result = dict(raw)
    expected = {
        "wdl_native_owner": True,
        "wdl_native_controller_abi_version": WDL_ABI_VERSION_V3,
        "wdl_python_tick_or_generation_ownership": False,
        "completed_generations": logical_generation_count,
        "logical_generation_count": logical_generation_count,
        "physical_window_generations": (
            prepared.patch.source.physical_window_generations
        ),
        "window_count": resident_window_count_v1(
            initial_generation=prepared.patch.source.generation_base,
            logical_generation_count=logical_generation_count,
            physical_window_generations=(
                prepared.patch.source.physical_window_generations
            ),
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
                f"WDL resident result {name} mismatch: "
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
        raise _fail("WDL resident run-control manifest is not at the final boundary")
    result["native_run_control_v2"] = manifest
    return result


def project_direct_resident_wdl_checkpoint_v2(
    prepared: PreparedDirectResidentWdlV2,
    *,
    direct_result: Mapping[str, Any],
    completed_generation_count: int,
    inheritance_sidecar_cell_id: str | None = None,
    generation_base_override: int | None = None,
) -> Mapping[str, Any]:
    """Project one complete WDL prefix through the accepted Patch checkpoint path."""

    if not isinstance(prepared, PreparedDirectResidentWdlV2):
        raise _fail("prepared must be PreparedDirectResidentWdlV2")
    if (
        type(completed_generation_count) is not int
        or completed_generation_count < 1
        or completed_generation_count > prepared.generation_rows
    ):
        raise _fail("WDL checkpoint projection exceeds its resident reporting horizon")
    result = dict(direct_result)
    tick_span_start = result.get("tick_span_start")
    tick_span_end = result.get("tick_span_end")
    tick_span_count = result.get("tick_span_count")
    tick_span = all(
        type(value) is int
        for value in (tick_span_start, tick_span_end, tick_span_count)
    )
    if tick_span and (
        tick_span_start < 0
        or tick_span_count < 1
        or tick_span_end != tick_span_start + tick_span_count
        or tick_span_end > _prepared_ticks_per_generation_v2(prepared)
    ):
        raise _fail("WDL tick-span result has invalid exact cursor coordinates")
    terminal_tick = (
        0
        if tick_span and result.get("terminal_generation_boundary") is True
        else tick_span_end
        if tick_span
        else 0
    )
    outputs = dict(prepared.patch.source.outputs)
    generation_summary = result.get("generation_summary_scalar")
    if not isinstance(generation_summary, torch.Tensor):
        generation_summary = prepared.patch.arguments[
            PHYSICAL_FULL_TICK_OUTPUTS_ARGUMENT_INDEX_V2
        ][-1][:completed_generation_count]
    runtime_bookkeeping = result.get("runtime_bookkeeping_by_tick")
    if not isinstance(runtime_bookkeeping, torch.Tensor):
        runtime_bookkeeping = prepared.patch.arguments[
            PHYSICAL_RUNTIME_BOOKKEEPING_ARGUMENT_INDEX_V2
        ][:, : completed_generation_count * _prepared_ticks_per_generation_v2(prepared)]
    outputs["generation_summary_scalar"] = generation_summary
    outputs["runtime_bookkeeping_by_tick"] = runtime_bookkeeping
    persistent_state = dict(
        _with_materialized_event_history(prepared.patch.live_banks, result)
    )
    persistent_state = materialize_disabled_prediction_state_v1(
        prepared.initial_projection,
        persistent_state,
        reset_count_max=_prepared_ticks_per_generation_v2(prepared),
    )
    for name, tensor in zip(
        WDL_STATE_FIELDS_V2,
        prepared.state_tensors,
        strict=True,
    ):
        persistent_state[name] = tensor
    projection_inputs = prepared.patch.checkpoint_projection_inputs
    agent_keys = prepared.reporting_tensors[6][0, completed_generation_count - 1]
    agent_flags = prepared.reporting_tensors[7][0, completed_generation_count - 1]
    source_generation = prepared.patch.source.generation_base
    if generation_base_override is not None:
        if type(generation_base_override) is not int or generation_base_override < 0:
            raise _fail("generation_base_override must be a nonnegative integer")
        source_generation = generation_base_override
    checkpoint = _project_terminal_checkpoint(
        projection_inputs["pre_generation_canonical_projection"],
        projection_inputs["resident_birth_state"],
        outputs,
        persistent_state,
        completed_generation_count=completed_generation_count,
        source_tick=(tick_span_start if tick_span else 0),
        completed_tick_count=(tick_span_count if tick_span else None),
        terminal_tick=terminal_tick,
        source_generation=source_generation,
        generation_turnover_enabled=_prepared_generation_turnover_enabled_v2(prepared),
        inheritance_sidecar_cell_id=inheritance_sidecar_cell_id,
        genome_v2_checkpoint_authority_binding=projection_inputs.get(
            "genome_v2_checkpoint_authority_binding", ()
        ),
        genome_v2_causal_physical_cell_ordinal=projection_inputs.get(
            "genome_v2_causal_physical_cell_ordinal", 0
        ),
        terminal_metadata_updates={
            "wdl_next_absolute_step": int(
                prepared.schedule_tensors[1].detach().cpu().item()
            ),
            "native_runtime_wdl_action_context_generation": (
                source_generation
                + completed_generation_count
                - 1
            ),
            "wdl_agents_cued_current_generation_agent_ids": (
                _decode_agent_flag_keys_v3(
                    agent_keys.detach().cpu(),
                    agent_flags.detach().cpu(),
                    flag=WDL_AGENT_FLAG_CUED_V3,
                )
            ),
        },
    )
    return checkpoint.projection


def project_direct_resident_wdl_multicell_settlement_v2(
    prepared: PreparedDirectResidentWdlV2,
    *,
    direct_result: Mapping[str, Any],
    completed_generation_count: int,
    reporting_enabled: bool,
    generation_base_override: int | None = None,
    bounded_projection_cursor: bool = False,
) -> DirectResidentPatchMulticellSettlementV1:
    """Project WDL checkpoints without discarding Patch settlement evidence."""

    if not isinstance(prepared, PreparedDirectResidentWdlV2):
        raise _fail("prepared must be PreparedDirectResidentWdlV2")
    if (
        type(completed_generation_count) is not int
        or completed_generation_count < 1
        or completed_generation_count > prepared.generation_rows
    ):
        raise _fail("WDL checkpoint projection exceeds its resident reporting horizon")
    from engine.perf.cuda_architecture_b_direct_resident_patch_v1 import (
        PATCH_MULTICELL_FACADE_SCHEMA_VERSION_V1,
        PreparedDirectResidentPatchMulticellV1,
        project_direct_resident_patch_multicell_checkpoints_v1,
    )

    patch_prepared = PreparedDirectResidentPatchMulticellV1(
        schema_version=PATCH_MULTICELL_FACADE_SCHEMA_VERSION_V1,
        patch=prepared.patch,
        cell_ids=prepared.cell_ids,
        source_projections=(
            prepared.source_projections
            if prepared.source_projections
            else (prepared.initial_projection,)
        ),
        resumed=False,
        source_projection_cursor_factory=prepared.source_projection_cursor_factory,
    )
    capability = prepared.patch.checkpoint_projection_inputs.get("cuda_capability")
    if capability is None:
        capability = prepared.config.get("cuda_capability")
    if not isinstance(capability, Mapping):
        raise _fail("WDL multicell checkpoint projection lacks base CUDA capability")
    settlement = project_direct_resident_patch_multicell_checkpoints_v1(
        patch_prepared,
        direct_result=direct_result,
        capability=capability,
        reporting_enabled=reporting_enabled,
        generation_base_override=generation_base_override,
        bounded_projection_cursor=bounded_projection_cursor,
    )
    source_generation = prepared.patch.source.generation_base
    physical_cell_ordinals = prepared.patch.checkpoint_projection_inputs.get(
        "genome_v2_causal_physical_cell_ordinals"
    )

    def physical_cell_ordinal(cell: int) -> int:
        if (
            isinstance(physical_cell_ordinals, Sequence)
            and not isinstance(physical_cell_ordinals, (str, bytes, bytearray))
            and len(physical_cell_ordinals) == len(prepared.cell_ids)
            and type(physical_cell_ordinals[cell]) is int
            and physical_cell_ordinals[cell] >= 0
        ):
            return int(physical_cell_ordinals[cell])
        return cell
    if generation_base_override is not None:
        if type(generation_base_override) is not int or generation_base_override < 0:
            raise _fail("generation_base_override must be a nonnegative integer")
        source_generation = generation_base_override
    if bounded_projection_cursor:
        cursor = settlement.checkpoint_projection_cursor
        if cursor is None:
            raise _fail("bounded WDL projection lacks a checkpoint cursor")

        def transform(
            detached: DetachedCellProjectionV1,
        ) -> DetachedCellProjectionV1:
            projection = dict(detached.require_projection())
            metadata = dict(projection["metadata"])
            cell = detached.cell_index
            metadata["wdl_next_absolute_step"] = int(
                prepared.schedule_tensors[1][cell].detach().cpu().item()
            )
            metadata["native_runtime_wdl_action_context_generation"] = (
                source_generation + completed_generation_count - 1
            )
            agent_keys = prepared.reporting_tensors[6][
                cell, completed_generation_count - 1
            ]
            agent_flags = prepared.reporting_tensors[7][
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
            if (
                "diploid_checkpoint_projection" in projection
                or "genome_upgrade_v2_checkpoint_projection" in projection
            ):
                projection = _with_checkpoint_contract_metadata(projection)
                checkpoint = CudaBoundaryCheckpoint(
                    projection=projection,
                    projection_digest=projection_digest(projection),
                    device="cpu",
                    device_to_host_operations=0,
                    device_to_host_bytes=0,
                    device_synchronizations=0,
                )
                checkpoint = with_resident_inheritance_sidecars_v1(
                    checkpoint,
                    _slice_live_banks(
                        prepared.patch.live_banks,
                        cell=cell,
                        cells=len(prepared.cell_ids),
                    ),
                    cell_id=prepared.cell_ids[cell],
                    genome_v2_checkpoint_authority_binding=(
                        prepared.patch.checkpoint_projection_inputs.get(
                            "genome_v2_checkpoint_authority_binding", ()
                        )
                    ),
                    genome_v2_causal_physical_cell_ordinal=physical_cell_ordinal(
                        cell
                    ),
                )
                projection = checkpoint.projection
            tensors = projection.get("tensors")
            if not isinstance(tensors, Mapping):
                raise _fail(f"WDL checkpoint {cell} is missing tensors")
            missing = [name for name in WDL_STATE_FIELDS_V2 if name not in tensors]
            if missing:
                raise _fail(f"WDL checkpoint {cell} is missing WDL state: {missing}")
            projection = _with_checkpoint_contract_metadata(projection)
            validate_native_checkpoint_projection(projection)
            return DetachedCellProjectionV1(
                schema_version=detached.schema_version,
                cell_id=detached.cell_id,
                cell_index=detached.cell_index,
                projection=projection,
                compact_result_rows=detached.compact_result_rows,
                segment_artifacts=detached.segment_artifacts,
                lifecycle_qacct_receipt=detached.lifecycle_qacct_receipt,
                terminal_generation=detached.terminal_generation,
                terminal_tick=detached.terminal_tick,
                live_projection_bytes=detached.live_projection_bytes,
            )

        mapped = cursor.map(transform)
        return replace(
            settlement,
            checkpoint_projections=(),
            checkpoint_projection_cursor=mapped,
        )

    out: list[Mapping[str, Any]] = []
    for cell, checkpoint in enumerate(settlement.checkpoint_projections):
        projection = dict(checkpoint)
        metadata = dict(projection["metadata"])
        metadata["wdl_next_absolute_step"] = int(
            prepared.schedule_tensors[1][cell].detach().cpu().item()
        )
        metadata["native_runtime_wdl_action_context_generation"] = (
            source_generation + completed_generation_count - 1
        )
        agent_keys = prepared.reporting_tensors[6][
            cell, completed_generation_count - 1
        ]
        agent_flags = prepared.reporting_tensors[7][
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
        if (
            "diploid_checkpoint_projection" in projection
            or "genome_upgrade_v2_checkpoint_projection" in projection
        ):
            projection = _with_checkpoint_contract_metadata(projection)
            checkpoint = CudaBoundaryCheckpoint(
                projection=projection,
                projection_digest=projection_digest(projection),
                device="cpu",
                device_to_host_operations=0,
                device_to_host_bytes=0,
                device_synchronizations=0,
            )
            checkpoint = with_resident_inheritance_sidecars_v1(
                checkpoint,
                _slice_live_banks(
                    prepared.patch.live_banks,
                    cell=cell,
                    cells=len(prepared.cell_ids),
                ),
                cell_id=prepared.cell_ids[cell],
                genome_v2_checkpoint_authority_binding=(
                    prepared.patch.checkpoint_projection_inputs.get(
                        "genome_v2_checkpoint_authority_binding",
                        (),
                    )
                ),
                genome_v2_causal_physical_cell_ordinal=physical_cell_ordinal(cell),
            )
            projection = checkpoint.projection
        tensors = projection.get("tensors")
        if not isinstance(tensors, Mapping):
            raise _fail(f"WDL checkpoint {cell} is missing tensors")
        missing = [name for name in WDL_STATE_FIELDS_V2 if name not in tensors]
        if missing:
            raise _fail(f"WDL checkpoint {cell} is missing WDL state: {missing}")
        projection = _with_checkpoint_contract_metadata(projection)
        validate_native_checkpoint_projection(projection)
        out.append(projection)
    return replace(settlement, checkpoint_projections=tuple(out))


def project_direct_resident_wdl_multicell_checkpoints_v2(
    prepared: PreparedDirectResidentWdlV2,
    *,
    direct_result: Mapping[str, Any],
    completed_generation_count: int,
    reporting_enabled: bool,
) -> tuple[Mapping[str, Any], ...]:
    """Project ordered B1..B72 WDL/fixed-MLP checkpoints."""

    settlement = project_direct_resident_wdl_multicell_settlement_v2(
        prepared,
        direct_result=direct_result,
        completed_generation_count=completed_generation_count,
        reporting_enabled=reporting_enabled,
    )
    return tuple(settlement.checkpoint_projections)


def project_direct_resident_wdl_generation_zero_v2(
    prepared: Any,
) -> tuple[Mapping[str, Any], ...]:
    """Project initialized WDL resident G0 authority for durable checkpoints."""

    prepared = getattr(prepared, "physical_window_arguments", prepared)
    if not isinstance(prepared, PreparedDirectResidentWdlV2):
        raise _fail("prepared must be PreparedDirectResidentWdlV2")
    checkpoint_inputs = prepared.patch.checkpoint_projection_inputs
    if not isinstance(checkpoint_inputs, Mapping):
        raise _fail("WDL prepared owner omitted checkpoint projection inputs")
    if len(prepared.cell_ids) != 1:
        from engine.perf.cuda_architecture_b_direct_resident_patch_v1 import (
            PATCH_MULTICELL_FACADE_SCHEMA_VERSION_V1,
            PreparedDirectResidentPatchMulticellV1,
            project_direct_resident_patch_multicell_generation_zero_v1,
        )

        return project_direct_resident_patch_multicell_generation_zero_v1(
            PreparedDirectResidentPatchMulticellV1(
                schema_version=PATCH_MULTICELL_FACADE_SCHEMA_VERSION_V1,
                patch=prepared.patch,
                cell_ids=prepared.cell_ids,
                source_projections=(
                    prepared.source_projections
                    if prepared.source_projections
                    else (prepared.initial_projection,)
                ),
                resumed=False,
            )
        )
    if (
        len(prepared.cell_ids) != 1
        or type(prepared.cell_ids[0]) is not str
        or not prepared.cell_ids[0]
        or prepared.cell_ids[0].strip() != prepared.cell_ids[0]
    ):
        raise _fail("WDL generation-zero checkpoint requires one stable cell ID")
    canonical = _projection_mapping(
        checkpoint_inputs.get("pre_generation_canonical_projection"),
        name="checkpoint_inputs.pre_generation_canonical_projection",
    )
    checkpoint = CudaBoundaryCheckpoint(
        projection=dict(canonical),
        projection_digest=projection_digest(canonical),
        device="cpu",
        device_to_host_operations=0,
        device_to_host_bytes=0,
        device_synchronizations=0,
    )
    checkpoint = with_resident_inheritance_sidecars_v1(
        checkpoint,
        _slice_live_banks(prepared.patch.live_banks, cell=0, cells=1),
        cell_id=prepared.cell_ids[0],
        genome_v2_checkpoint_authority_binding=checkpoint_inputs.get(
            "genome_v2_checkpoint_authority_binding", ()
        ),
        genome_v2_causal_physical_cell_ordinal=0,
        include_genome_v2_causal_projection=False,
    )
    checkpoint = CudaBoundaryCheckpoint(
        **{
            **checkpoint.__dict__,
            "projection": _with_checkpoint_contract_metadata(
                checkpoint.projection
            ),
        }
    )
    validate_native_checkpoint_projection(checkpoint.projection)
    return (checkpoint.projection,)


__all__ = [
    "WDL_CAPABILITY_ENTRYPOINT_V3",
    "WDL_DIRECT_ENTRYPOINT_V3",
    "WDL_PHYSICAL_ENTRYPOINT_V3",
    "WDL_STATE_FIELDS_V2",
    "CudaArchitectureBDirectResidentWdlV2Error",
    "PreparedDirectResidentWdlV2",
    "build_prepared_direct_resident_wdl_v2",
    "build_prepared_direct_resident_wdl_v2_streaming",
    "execute_direct_resident_wdl_v2",
    "execute_direct_resident_wdl_physical_v2",
    "_enable_wdl_genome_v2_causal_capture_v2",
    "project_direct_resident_wdl_checkpoint_v2",
    "project_direct_resident_wdl_multicell_settlement_v2",
    "project_direct_resident_wdl_multicell_checkpoints_v2",
    "project_direct_resident_wdl_generation_zero_v2",
]
