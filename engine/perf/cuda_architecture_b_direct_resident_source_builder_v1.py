"""Build the exact Patch P2 configurable-birth direct-resident source boundary."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass, replace
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final

import torch

from engine.campaigns.tick_native_control import (
    SOURCE_DERIVED_BASE_EVENT_ROWS_PER_SOURCE_WINDOW_V1,
    SOURCE_DERIVED_MAX_EVENT_ROWS_PER_ORDINARY_BIRTH_V1,
    physical_b136_event_window_capacity_v1 as physical_b136_event_window_capacity_v1,
    source_derived_event_window_capacity_v1,
)
from engine.native_runtime.cuda_architecture_b_backend_factory import (
    CudaArchitectureBRunnerEnvelope,
    derive_cuda_architecture_b_runner_envelope,
)
from engine.native_runtime.interchange_projection import (
    validate_native_checkpoint_projection,
)
from perf.cuda_architecture_b_boundary import (
    import_cpu_owner_boundary_to_cuda,
)
from engine.perf.cuda_architecture_b_event_sink_v1 import (
    EVENT_SINK_SCHEMA_VERSION,
    EventSinkError,
    external_event_projection_v1,
    validate_event_sink_prefix_reference_v1,
)
from engine.perf.cuda_architecture_b_direct_resident_preparation_v1 import (
    DirectResidentSourceV1,
    PreparedDirectResidentTrajectoryV1,
    prepare_direct_resident_trajectory_v1,
)
from engine.perf.cuda_architecture_b_direct_resident_source_allocator_v1 import (
    ADMITTED_TICKS_PER_GENERATION_V1,
    PHYSICAL_WINDOW_GENERATIONS_V1,
    PHYSICAL_WINDOW_TICKS_V1,
    PHYSICAL_WINDOW_TURNOVER_EPISODES_V1,
    allocate_direct_resident_source_v1,
    physical_window_ticks_v1,
    required_physical_window_policy_episodes_v1,
    _canonical_projection,
    _runtime_turnover_refill_capacity_v1 as _allocator_runtime_turnover_refill_capacity_v1,
)
from engine.perf.cuda_architecture_b_generation import CudaGenerationConfig
from perf.cuda_architecture_b_generation_birth_state import (
    CPYTHON_GAUSS_RAW_CAPACITY_PER_BIRTH,
    MAX_CPYTHON_GAUSS_RAW_CAPACITY,
    MAX_FIXED_MLP_POLICY_TAPE_EPISODES,
    prepare_generation_birth_state_cuda,
)
from engine.perf.cxxbackbone10_resident_state_owner import (
    _checkpoint_tensor_manifest,
    build_import_projection,
    projection_digest,
)


QUALIFIED_EVENT_CAPACITY_V1: Final[int] = 65_536
BASE_EVENT_ROWS_PER_SOURCE_WINDOW_V1: Final[int] = (
    SOURCE_DERIVED_BASE_EVENT_ROWS_PER_SOURCE_WINDOW_V1
)
MAX_EVENT_ROWS_PER_ORDINARY_BIRTH_V1: Final[int] = (
    SOURCE_DERIVED_MAX_EVENT_ROWS_PER_ORDINARY_BIRTH_V1
)
QUALIFIED_GENERATION_BASE_V1: Final[int] = 0
QUALIFIED_TICK_BASE_V1: Final[int] = 0
QUALIFIED_GENERATION_COUNT_V1: Final[int] = 4
QUALIFIED_TICKS_PER_GENERATION_V1: Final[int] = 80
ADMITTED_TICKS_PER_GENERATION: Final[tuple[int, ...]] = ADMITTED_TICKS_PER_GENERATION_V1
QUALIFIED_PATCH_GEOMETRIES_V1: Final[frozenset[tuple[int, int]]] = frozenset(
    {(64, 16), (128, 16), (256, 32), (512, 32)}
)
QUALIFIED_SOURCE_STEPS_V1: Final[int] = PHYSICAL_WINDOW_TICKS_V1
QUALIFIED_MAX_BIRTHS_PER_TICK_V1: Final[int] = 1
QUALIFIED_TURNOVER_EPISODES_PER_GENERATION_V1: Final[int] = 62
QUALIFIED_TURNOVER_REFILL_CAPACITY_V1: Final[int] = PHYSICAL_WINDOW_TURNOVER_EPISODES_V1
QUALIFIED_POLICY_EPISODE_CAPACITY_V1: Final[int] = (
    required_physical_window_policy_episodes_v1(QUALIFIED_MAX_BIRTHS_PER_TICK_V1)
)
ADMITTED_INITIAL_LIVE_COUNTS_V1: Final[frozenset[int]] = frozenset(
    population for population, _grid in QUALIFIED_PATCH_GEOMETRIES_V1
)
QUALIFIED_CAPABILITY_MASK_V1: Final[dict[str, bool]] = {
    "flat": False,
    "patch": True,
    "churn": True,
    "gru": False,
}
LIFE_HISTORY_METADATA_FIELDS_V1: Final[tuple[str, ...]] = (
    "developmental_maturity_age",
    "life_history_mode",
    "life_history_maturity_start_ticks",
    "life_history_maturity_rise_ticks",
    "life_history_fertility_decline_start_ticks",
    "life_history_fertility_decline_ticks",
    "life_history_elder_fertility_floor",
    "life_history_senescence_start_ticks",
    "life_history_senescence_ramp_ticks",
    "life_history_senescence_energy_cost",
    "life_history_senescence_health_cost",
    "life_history_max_age_ticks",
)
NATIVE_CAPABILITY_ENTRYPOINT_V1: Final[str] = (
    "eiel_cuda_architecture_b_flattened_generation_count_capability_v1"
)

ENGINE_SOURCE_ALLOCATOR_MODULE_V1: Final[str] = (
    "engine.perf.cuda_architecture_b_direct_resident_source_allocator_v1"
)
ENGINE_SOURCE_ALLOCATOR_NAME_V1: Final[str] = "allocate_direct_resident_source_v1"
MISSING_ENGINE_SOURCE_DEPENDENCY_V1: Final[str] = (
    f"{ENGINE_SOURCE_ALLOCATOR_MODULE_V1}.{ENGINE_SOURCE_ALLOCATOR_NAME_V1}"
)


def required_qualified_reproduction_attempt_capacity_v1(
    *,
    segment_generations: int,
    ticks_per_generation: int,
    resident_capacity: int,
) -> int:
    """Return the exact retained-route attempt bound for one drained segment."""

    for name, value in (
        ("segment_generations", segment_generations),
        ("ticks_per_generation", ticks_per_generation),
        ("resident_capacity", resident_capacity),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive exact integer")
    # One successful ordinary birth may require scanning every resident parent
    # decision before the planner commits or concludes the tick. Exposure rows
    # therefore bind planner attempts, not only successful births.
    ordinary_attempts = resident_capacity * ticks_per_generation
    turnover_attempts = min(
        QUALIFIED_TURNOVER_EPISODES_PER_GENERATION_V1,
        resident_capacity,
    )
    # Refill and ordinary-live are mutually exclusive cell paths.
    return segment_generations * max(ordinary_attempts, turnover_attempts)


class CudaArchitectureBDirectResidentSourceBuilderError(RuntimeError):
    """The exact production source cannot be built without changing owners."""


class MissingEngineDirectResidentSourceAllocatorV1Error(
    CudaArchitectureBDirectResidentSourceBuilderError
):
    """The engine has no allocator for all retained direct-resident authorities."""

    def __init__(self) -> None:
        self.dependency = MISSING_ENGINE_SOURCE_DEPENDENCY_V1
        super().__init__(
            "missing engine-owned dependency: "
            f"{self.dependency}; it must allocate the DirectResidentSourceV1 "
            "output, scratch, turnover, Patch P2 statistics, live-bank, and "
            "checkpoint authorities"
        )


@dataclass(frozen=True, slots=True)
class DirectResidentSourceAllocationRequestV1:
    """Validated inputs for the one missing engine-owned source allocator."""

    envelope: CudaArchitectureBRunnerEnvelope
    qualified_cuda_module: Any
    cuda_capability: Mapping[str, Any]
    import_projection: Any
    cuda_boundary: Any
    generation_birth_state: Any
    configuration: CudaGenerationConfig
    cuda_device: str
    event_capacity: int = QUALIFIED_EVENT_CAPACITY_V1
    logical_generation_count: int = QUALIFIED_GENERATION_COUNT_V1
    physical_window_generations: int = PHYSICAL_WINDOW_GENERATIONS_V1
    agent_id_allocation_generation_count: int | None = None
    batch_cells: int = 1
    observation_extra_columns: int = 0
    output_ticks: int = 1
    immutable_shared_gru_pack: torch.Tensor | None = None
    allow_short_technical_gate_generation_count: bool = False
    allow_partial_generation_source_cursor: bool = False
    inheritance_sidecar_cell_id: str | None = None
    genome_v2_causal_physical_cell_ordinal: int = 0
    turnover_mlp_shadow_elide_enable: bool = False
    cpu_request_staging_enable: bool = False
    # Bounded multicell import mode: retain immutable CPU owner projections on
    # host memory and transfer fields directly into the final batch-shaped CUDA
    # owner. This is distinct from CPU-resident execution.
    direct_batch_host_import_enable: bool = False
    source_import_projection_digest_sha256: str | None = None
    event_history_sink_path: str | None = None
    event_history_sink_cell_index: int = 0
    resident_owner_span_ticks: int | None = None
    native_dispatch_window_ticks: int | None = None
    source_window_ticks: int | None = None
    event_window_capacity: int | None = None
    tick_native_control: Mapping[str, Any] = field(default_factory=dict)


DirectResidentSourceAllocatorV1 = Callable[
    [DirectResidentSourceAllocationRequestV1],
    DirectResidentSourceV1,
]


def _fail(message: str) -> CudaArchitectureBDirectResidentSourceBuilderError:
    return CudaArchitectureBDirectResidentSourceBuilderError(message)


def _validated_initial_live_count_v1(value: Any, *, path: str) -> int:
    if type(value) is not int or value not in ADMITTED_INITIAL_LIVE_COUNTS_V1:
        raise _fail(
            f"{path} must be an admitted initial live count: "
            f"{sorted(ADMITTED_INITIAL_LIVE_COUNTS_V1)!r}"
        )
    return int(value)


def _stage_request_tensors_to_cpu_v1(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        staged = value.detach()
        if staged.device.type != "cpu":
            staged = staged.to(device="cpu", non_blocking=False)
        return staged.contiguous()
    if is_dataclass(value) and not isinstance(value, type):
        return replace(
            value,
            **{
                field.name: _stage_request_tensors_to_cpu_v1(
                    getattr(value, field.name)
                )
                for field in fields(value)
            },
        )
    if isinstance(value, Mapping):
        return {
            name: _stage_request_tensors_to_cpu_v1(child)
            for name, child in value.items()
        }
    if isinstance(value, SimpleNamespace):
        return SimpleNamespace(
            **{
                name: _stage_request_tensors_to_cpu_v1(child)
                for name, child in vars(value).items()
            }
        )
    if isinstance(value, tuple):
        return tuple(_stage_request_tensors_to_cpu_v1(child) for child in value)
    if isinstance(value, list):
        return [_stage_request_tensors_to_cpu_v1(child) for child in value]
    return value


def _contains_cuda_tensor_v1(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return value.device.type == "cuda"
    if is_dataclass(value) and not isinstance(value, type):
        return any(
            _contains_cuda_tensor_v1(getattr(value, field.name))
            for field in fields(value)
        )
    if isinstance(value, Mapping):
        return any(_contains_cuda_tensor_v1(child) for child in value.values())
    if isinstance(value, SimpleNamespace):
        return any(_contains_cuda_tensor_v1(child) for child in vars(value).values())
    if isinstance(value, (tuple, list)):
        return any(_contains_cuda_tensor_v1(child) for child in value)
    return False


def _cpu_owner_tensor_manifest_stale_v1(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    tensors = value.get("tensors")
    manifest = value.get("tensor_manifest")
    if not isinstance(tensors, Mapping) or not isinstance(manifest, Mapping):
        return False
    if set(manifest) != set(tensors):
        return True
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            return True
        if manifest.get(name) != _checkpoint_tensor_manifest(tensor):
            return True
    return False


def _stage_cpu_owner_projection_v1(value: Any) -> Mapping[str, Any]:
    staged = _stage_request_tensors_to_cpu_v1(value)
    if not isinstance(staged, Mapping):
        raise _fail("staged CPU owner projection must be a mapping")
    tensors = staged.get("tensors")
    if not isinstance(tensors, Mapping) or not tensors:
        raise _fail("staged CPU owner projection tensors must be a non-empty mapping")
    refreshed = dict(staged)
    refreshed["tensor_manifest"] = {
        str(name): _checkpoint_tensor_manifest(tensor)
        for name, tensor in sorted(tensors.items())
        if isinstance(tensor, torch.Tensor)
    }
    if len(refreshed["tensor_manifest"]) != len(tensors):
        raise _fail("staged CPU owner projection contains a non-tensor field")
    return refreshed


def _life_history_config_from_sim_cfg_v1(sim_cfg: Any) -> dict[str, int | float]:
    values: dict[str, int | float] = {}
    for name in LIFE_HISTORY_METADATA_FIELDS_V1:
        default = getattr(CudaGenerationConfig, name)
        raw = getattr(sim_cfg, name, default)
        if isinstance(default, int):
            if type(raw) is not int:
                raise _fail(f"runner {name} must be an exact integer")
            values[name] = int(raw)
        else:
            if type(raw) not in {int, float}:
                raise _fail(f"runner {name} must be numeric")
            values[name] = float(raw)
    if values["life_history_mode"] not in {0, 1}:
        raise _fail("runner life_history_mode must be exact integer 0 or 1")
    for name in (
        "developmental_maturity_age",
        "life_history_maturity_start_ticks",
        "life_history_maturity_rise_ticks",
        "life_history_fertility_decline_start_ticks",
        "life_history_fertility_decline_ticks",
        "life_history_senescence_start_ticks",
        "life_history_senescence_ramp_ticks",
        "life_history_max_age_ticks",
    ):
        if values[name] < 0:
            raise _fail(f"runner {name} must be nonnegative")
    elder_floor = float(values["life_history_elder_fertility_floor"])
    if not math.isfinite(elder_floor) or not 0.0 <= elder_floor <= 1.0:
        raise _fail("runner life_history_elder_fertility_floor must be in [0, 1]")
    for name in (
        "life_history_senescence_energy_cost",
        "life_history_senescence_health_cost",
    ):
        if not math.isfinite(float(values[name])) or values[name] < 0.0:
            raise _fail(f"runner {name} must be nonnegative")
    return values


def _bind_life_history_metadata_v1(
    projection: dict[str, Any],
    *,
    life_history: Mapping[str, int | float],
) -> dict[str, Any]:
    metadata = projection.get("metadata")
    if not isinstance(metadata, dict):
        raise _fail("source projection metadata must be mutable mapping")
    metadata.update(dict(life_history))
    metadata["import_digest"] = projection_digest(
        {"metadata": metadata, "tensors": projection.get("tensors", {})}
    )
    return projection


def _validate_life_history_metadata_v1(
    metadata: Mapping[str, Any],
    *,
    life_history: Mapping[str, int | float],
) -> None:
    present = tuple(name for name in LIFE_HISTORY_METADATA_FIELDS_V1 if name in metadata)
    if not present and life_history.get("life_history_mode") == 0:
        return
    for name, expected in life_history.items():
        actual = metadata.get(name)
        if isinstance(expected, int):
            if type(actual) is not int or int(actual) != expected:
                raise _fail(f"source checkpoint life-history {name} mismatch")
        else:
            if (
                type(actual) not in {int, float}
                or not math.isfinite(float(actual))
                or float(actual) != float(expected)
            ):
                raise _fail(f"source checkpoint life-history {name} mismatch")


def _validated_source_allocator(
    source_allocator: DirectResidentSourceAllocatorV1 | None,
) -> DirectResidentSourceAllocatorV1:
    if source_allocator is None:
        source_allocator = allocate_direct_resident_source_v1
    if not callable(source_allocator):
        raise _fail("source_allocator must be callable")
    module_name = getattr(source_allocator, "__module__", None)
    function_name = getattr(source_allocator, "__name__", None)
    if (
        module_name != ENGINE_SOURCE_ALLOCATOR_MODULE_V1
        or function_name != ENGINE_SOURCE_ALLOCATOR_NAME_V1
    ):
        raise _fail(
            "source_allocator must be the exact engine-owned dependency "
            f"{MISSING_ENGINE_SOURCE_DEPENDENCY_V1}"
        )
    return source_allocator


def _validated_cuda_device(cuda_device: str) -> str:
    if type(cuda_device) is not str or cuda_device not in {"cuda", "cuda:0"}:
        raise _fail("the qualified source builder requires cuda or cuda:0")
    return cuda_device


def _validated_physical_birth_capacity_v1(
    max_births_per_tick: object,
    slot_capacity: object,
    *,
    physical_window_generations: object = PHYSICAL_WINDOW_GENERATIONS_V1,
    turnover_refill_capacity: int | None = None,
) -> tuple[int, int]:
    if type(slot_capacity) is not int or slot_capacity < 1:
        raise _fail("qualified CUDA boundary must expose a positive slot capacity")
    if (
        type(max_births_per_tick) is not int
        or max_births_per_tick < 1
        or max_births_per_tick > slot_capacity
    ):
        raise _fail(
            "qualified max_births_per_tick must be an exact integer in "
            f"[1, slot_capacity={slot_capacity}], got {max_births_per_tick!r}"
        )
    if turnover_refill_capacity is None:
        episode_capacity = required_physical_window_policy_episodes_v1(
            max_births_per_tick,
            physical_window_generations=physical_window_generations,
        )
    else:
        if type(turnover_refill_capacity) is not int or turnover_refill_capacity < 0:
            raise _fail(
                "turnover_refill_capacity must be a nonnegative exact integer"
            )
        episode_capacity = (
            physical_window_ticks_v1(physical_window_generations)
            * max_births_per_tick
            + turnover_refill_capacity
        )
    gaussian_episode_ceiling = (
        MAX_CPYTHON_GAUSS_RAW_CAPACITY // CPYTHON_GAUSS_RAW_CAPACITY_PER_BIRTH
    )
    if (
        episode_capacity > MAX_FIXED_MLP_POLICY_TAPE_EPISODES
        or episode_capacity > gaussian_episode_ceiling
    ):
        raise _fail(
            "one physical g1/g2 window exceeds resident birth tape capacity: "
            f"max_births_per_tick={max_births_per_tick} "
            f"required_policy_episodes={episode_capacity} "
            f"fixed_mlp_policy_episode_ceiling={MAX_FIXED_MLP_POLICY_TAPE_EPISODES} "
            f"cpython_gaussian_episode_ceiling={gaussian_episode_ceiling}; "
            "native boundary tape staging/refill is required"
        )
    return max_births_per_tick, episode_capacity


def _runtime_turnover_refill_capacity_v1(
    sim_cfg: Any,
    *,
    population_size: int,
    slot_capacity: int,
    ordinary_births_enabled: bool,
    physical_window_generations: int,
) -> int:
    """Derive the declared native turnover refill bound for this runner.

    The historical allocator default (62 rows per generation) remains the
    fallback for older callers that do not expose lifecycle sizing.  A live
    lifecycle envelope must instead bind the turnover segment to its current
    population/cull contract; otherwise the native plan can consume more rows
    than the resident policy and Gaussian tapes contain.
    """
    return _allocator_runtime_turnover_refill_capacity_v1(
        sim_cfg,
        population_size=population_size,
        slot_capacity=slot_capacity,
        ordinary_births_enabled=ordinary_births_enabled,
        physical_window_generations=physical_window_generations,
    )


def _scalar_gauss_raw_capacity_override_v1(
    sim_cfg: Any,
    *,
    ordinary_births_enabled: bool,
) -> int | None:
    if type(ordinary_births_enabled) is not bool:
        raise _fail("ordinary_births_enabled must be an exact bool")
    explicit_capacity = getattr(
        sim_cfg,
        "cuda_architecture_b_cpython_gauss_segment_capacity",
        -1,
    )
    if type(explicit_capacity) is not int:
        raise _fail(
            "runner cuda_architecture_b_cpython_gauss_segment_capacity "
            "must be an exact integer"
        )
    if explicit_capacity < -1:
        raise _fail(
            "runner cuda_architecture_b_cpython_gauss_segment_capacity "
            "must be -1 or a nonnegative integer"
        )
    if explicit_capacity >= 0:
        return explicit_capacity
    if ordinary_births_enabled:
        return None
    genome_v2_enabled = getattr(sim_cfg, "genome_upgrade_v2_enable", False)
    if type(genome_v2_enabled) is not bool:
        raise _fail("runner genome_upgrade_v2_enable must be an exact bool")
    mutation_rate_q10 = getattr(sim_cfg, "genome_upgrade_v2_mutation_rate_q10", 20)
    mutation_step_q10 = getattr(sim_cfg, "genome_upgrade_v2_mutation_step_q10", 64)
    if type(mutation_rate_q10) is not int or mutation_rate_q10 < 0:
        raise _fail(
            "runner genome_upgrade_v2_mutation_rate_q10 must be a "
            "nonnegative exact integer"
        )
    if type(mutation_step_q10) is not int or mutation_step_q10 < 0:
        raise _fail(
            "runner genome_upgrade_v2_mutation_step_q10 must be a "
            "nonnegative exact integer"
        )
    return None


def _compact_policy_mutation_tape_v1(
    sim_cfg: Any,
    *,
    ordinary_births_enabled: bool,
) -> bool:
    if type(ordinary_births_enabled) is not bool:
        raise _fail("ordinary_births_enabled must be an exact bool")
    policy_mutation = getattr(sim_cfg, "policy_weight_mutation_enable", False)
    if type(policy_mutation) is not bool:
        raise _fail("runner policy_weight_mutation_enable must be an exact bool")
    if policy_mutation:
        return False
    policy_mutation_std = getattr(sim_cfg, "policy_weight_mutation_std", 0.0)
    if (
        isinstance(policy_mutation_std, bool)
        or not isinstance(policy_mutation_std, (int, float))
        or float(policy_mutation_std) != 0.0
    ):
        raise _fail("runner policy_weight_mutation_std must be exactly 0.0")
    policy_mutation_rate = getattr(sim_cfg, "policy_weight_mutation_rate", 0.0)
    if isinstance(policy_mutation_rate, bool) or not isinstance(
        policy_mutation_rate, (int, float)
    ):
        raise _fail("runner policy_weight_mutation_rate must be numeric")
    if float(policy_mutation_rate) < 0.0:
        raise _fail("runner policy_weight_mutation_rate must be nonnegative")
    return True


def _policy_birth_segment_refill_capacity_v1(
    sim_cfg: Any,
    *,
    ordinary_births_enabled: bool,
    resident_policy_episode_capacity: int,
    turnover_refill_capacity: int,
    max_births_per_tick: int,
) -> int | None:
    enabled = getattr(
        sim_cfg,
        "cuda_architecture_b_policy_birth_segment_refill_enable",
        False,
    )
    if type(enabled) is not bool:
        raise _fail(
            "runner cuda_architecture_b_policy_birth_segment_refill_enable "
            "must be an exact bool"
        )
    if not enabled:
        return None
    if type(max_births_per_tick) is not int or max_births_per_tick < 1:
        raise _fail("max_births_per_tick must be a positive exact integer")
    if not _compact_policy_mutation_tape_v1(
        sim_cfg,
        ordinary_births_enabled=ordinary_births_enabled,
    ):
        raise _fail(
            "policy birth segment refill requires disabled policy-weight mutation"
        )
    minimum_segment_capacity = max(max_births_per_tick, turnover_refill_capacity)
    explicit_capacity = getattr(
        sim_cfg,
        "cuda_architecture_b_policy_birth_segment_capacity",
        -1,
    )
    if type(explicit_capacity) is not int:
        raise _fail(
            "runner cuda_architecture_b_policy_birth_segment_capacity "
            "must be an exact integer"
        )
    if explicit_capacity < -1:
        raise _fail(
            "runner cuda_architecture_b_policy_birth_segment_capacity "
            "must be -1 or a positive integer"
        )
    segment_capacity = (
        minimum_segment_capacity if explicit_capacity == -1 else explicit_capacity
    )
    if segment_capacity < minimum_segment_capacity:
        raise _fail(
            "policy birth segment capacity must cover one ordinary tick and "
            "one turnover refill segment"
        )
    if segment_capacity >= resident_policy_episode_capacity:
        return None
    return segment_capacity


def birth_state_capacity_overrides_v1(
    sim_cfg: Any,
    *,
    ordinary_births_enabled: bool,
    resident_policy_episode_capacity: int,
    turnover_refill_capacity: int,
    max_births_per_tick: int | None = None,
) -> dict[str, Any]:
    if type(ordinary_births_enabled) is not bool:
        raise _fail("ordinary_births_enabled must be an exact bool")
    if type(resident_policy_episode_capacity) is not int:
        raise _fail("resident_policy_episode_capacity must be an exact integer")
    if type(turnover_refill_capacity) is not int:
        raise _fail("turnover_refill_capacity must be an exact integer")
    overrides: dict[str, Any] = {
        "resident_policy_episode_capacity": resident_policy_episode_capacity,
    }
    if not ordinary_births_enabled:
        overrides["ordinary_policy_episode_capacity"] = 0
        overrides["resident_policy_episode_capacity"] = turnover_refill_capacity
    scalar_gauss_raw_capacity = _scalar_gauss_raw_capacity_override_v1(
        sim_cfg,
        ordinary_births_enabled=ordinary_births_enabled,
    )
    generation_turnover_enabled = getattr(
        sim_cfg, "generation_turnover_enabled", None
    )
    if generation_turnover_enabled is None:
        lifecycle_mode = getattr(sim_cfg, "lifecycle_mode", None)
        if lifecycle_mode is not None:
            if type(lifecycle_mode) is not int or lifecycle_mode not in {0, 1}:
                raise _fail("runner lifecycle_mode must be exact integer 0 or 1")
            generation_turnover_enabled = lifecycle_mode == 0
    if generation_turnover_enabled is not None and type(
        generation_turnover_enabled
    ) is not bool:
        raise _fail("runner generation_turnover_enabled must be an exact bool")
    rng_semantics_version = int(getattr(sim_cfg, "rng_semantics_version", 1) or 1)
    if rng_semantics_version not in (1, 2):
        raise _fail("runner rng_semantics_version must be 1 or 2")
    if (
        scalar_gauss_raw_capacity is not None
        and ordinary_births_enabled
        and generation_turnover_enabled is True
        and rng_semantics_version != 2
    ):
        required_gauss_raw_capacity = (
            resident_policy_episode_capacity * CPYTHON_GAUSS_RAW_CAPACITY_PER_BIRTH
        )
        if scalar_gauss_raw_capacity < required_gauss_raw_capacity:
            raise _fail(
                "runner cpython_gauss_segment_capacity must cover the complete "
                "ordinary-birth plus atomic-turnover physical window: "
                f"required={required_gauss_raw_capacity} "
                f"configured={scalar_gauss_raw_capacity}"
            )
    if scalar_gauss_raw_capacity is not None:
        overrides["scalar_gauss_raw_capacity"] = scalar_gauss_raw_capacity
    if rng_semantics_version == 2:
        # RNG V2 is coordinate-addressed on GPU. No future Gaussian,
        # crossover, mutation, or post-state rows may be materialized.
        overrides["scalar_gauss_raw_capacity"] = 0
        overrides["ordinary_policy_episode_capacity"] = 0
        overrides["resident_policy_episode_capacity"] = 0
        overrides["policy_birth_segment_refill_enabled"] = False
        overrides["compact_policy_mutation_tape"] = True
        from engine.hapb_rngv2.route import apply_hapb_rng_route

        apply_hapb_rng_route(sim_cfg, hapb_rng_semantics_version=2)
    if _compact_policy_mutation_tape_v1(
        sim_cfg,
        ordinary_births_enabled=ordinary_births_enabled,
    ):
        overrides["compact_policy_mutation_tape"] = True
    if max_births_per_tick is not None and rng_semantics_version != 2:
        segment_capacity = _policy_birth_segment_refill_capacity_v1(
            sim_cfg,
            ordinary_births_enabled=ordinary_births_enabled,
            resident_policy_episode_capacity=int(
                overrides["resident_policy_episode_capacity"]
            ),
            turnover_refill_capacity=turnover_refill_capacity,
            max_births_per_tick=max_births_per_tick,
        )
        if segment_capacity is not None:
            overrides["resident_policy_episode_capacity"] = segment_capacity
            overrides["policy_birth_segment_refill_enabled"] = True
    return overrides


def _compact_no_birth_event_rows_per_generation_v1(
    sim_cfg: Any,
    *,
    ordinary_births_enabled: bool,
    resident_capacity: object,
) -> int | None:
    _scalar_gauss_raw_capacity_override_v1(
        sim_cfg,
        ordinary_births_enabled=ordinary_births_enabled,
    )
    if ordinary_births_enabled:
        return None
    genome_v2_enabled = getattr(sim_cfg, "genome_upgrade_v2_enable", False)
    mutation_rate_q10 = getattr(sim_cfg, "genome_upgrade_v2_mutation_rate_q10", 20)
    mutation_step_q10 = getattr(sim_cfg, "genome_upgrade_v2_mutation_step_q10", 64)
    if (
        genome_v2_enabled is not True
        or mutation_rate_q10 != 0
        or mutation_step_q10 != 0
    ):
        return None
    if not _compact_policy_mutation_tape_v1(
        sim_cfg,
        ordinary_births_enabled=ordinary_births_enabled,
    ):
        return None
    if type(resident_capacity) is not int or resident_capacity < 1:
        raise _fail("resident event capacity bound requires positive resident capacity")
    turnover_birth_rows = min(
        QUALIFIED_TURNOVER_EPISODES_PER_GENERATION_V1,
        resident_capacity,
    )
    resident_death_rows = resident_capacity
    return resident_death_rows + turnover_birth_rows


def _resident_event_capacity_generation_span_v1(
    *,
    source_generation: int,
    logical_generation_count: int,
    physical_window_generations: int = PHYSICAL_WINDOW_GENERATIONS_V1,
) -> int:
    if type(source_generation) is not int or source_generation < 0:
        raise _fail("source_generation must be a nonnegative exact integer")
    if type(logical_generation_count) is not int or logical_generation_count < 1:
        raise _fail("logical_generation_count must be a positive exact integer")
    if type(physical_window_generations) is not int or physical_window_generations not in {
        1,
        2,
    }:
        raise _fail("physical_window_generations must be exact integer 1 or 2")
    return min(logical_generation_count, physical_window_generations)


def _source_event_active_prefix_bound_v1(
    projection: Any,
    metadata: Mapping[str, Any],
) -> int:
    """Return the largest declared lifecycle event active-prefix cursor.

    Checkpoint projections may carry both legacy event-tape counts and typed
    lifecycle event counts/cursors.  Event capacity is an allocation upper
    bound, so it must preserve the largest declared active prefix and leave
    strict equality/consistency checks to the CUDA boundary importer.
    """

    values: list[int] = []
    for metadata_field in (
        "event_tape_active_count",
        "typed_lifecycle_event_active_count",
        "event_tape_cursor",
    ):
        value = metadata.get(metadata_field)
        if value is None:
            continue
        if type(value) is not int or value < 0:
            raise _fail(f"source projection {metadata_field} must be nonnegative")
        values.append(value)
    if isinstance(projection, Mapping):
        top_level_values = (
            ("event_cursor", projection.get("event_cursor")),
        )
    else:
        top_level_values = (
            ("event_cursor", getattr(projection, "event_cursor", None)),
        )
    for metadata_field, value in top_level_values:
        if value is None:
            continue
        if type(value) is not int or value < 0:
            raise _fail(f"source projection {metadata_field} must be nonnegative")
        values.append(value)
    return max(values, default=0)


def _patch_geometry_supported_v1(
    qualification: Mapping[str, Any],
    *,
    population: int,
    grid_size: int,
) -> bool:
    if (population, grid_size) not in QUALIFIED_PATCH_GEOMETRIES_V1:
        return False
    supported = qualification.get("supported_geometry_envelopes")
    if isinstance(supported, Sequence) and not isinstance(supported, (str, bytes)):
        for item in supported:
            if not isinstance(item, Mapping):
                continue
            if (
                item.get("population") == population
                and item.get("grid_size") == grid_size
            ):
                return True
    return (
        qualification.get("population") == population
        and qualification.get("grid_size") == grid_size
    )


def _validated_cuda_capability(qualified_cuda_module: Any) -> Mapping[str, Any]:
    entrypoint = getattr(qualified_cuda_module, NATIVE_CAPABILITY_ENTRYPOINT_V1, None)
    if not callable(entrypoint):
        raise _fail(
            "qualified_cuda_module is missing callable "
            f"{NATIVE_CAPABILITY_ENTRYPOINT_V1}"
        )
    capability = entrypoint()
    if not isinstance(capability, Mapping):
        raise _fail("qualified CUDA capability must be a mapping")
    qualification = capability.get("patch_p2_qualification")
    if not isinstance(qualification, Mapping):
        raise _fail("qualified CUDA capability is missing patch_p2_qualification")
    expected = {
        "qualified": True,
        "profile": "P2",
        "world": "patch",
        "policy": "stochastic_fixed_mlp",
        "batch_cells": 1,
        "ticks_per_generation": QUALIFIED_TICKS_PER_GENERATION_V1,
        "generation_turnover": True,
    }
    for name, expected_value in expected.items():
        actual = qualification.get(name)
        if type(actual) is not type(expected_value) or actual != expected_value:
            raise _fail(
                f"qualified CUDA capability patch_p2_qualification.{name} "
                f"mismatch: expected={expected_value!r} actual={actual!r}"
            )
    admitted_ticks = qualification.get("admitted_ticks_per_generation")
    if (
        isinstance(admitted_ticks, (str, bytes))
        or not isinstance(admitted_ticks, Sequence)
        or tuple(admitted_ticks) != ADMITTED_TICKS_PER_GENERATION
    ):
        raise _fail(
            "qualified CUDA capability must admit ticks per generation "
            f"{list(ADMITTED_TICKS_PER_GENERATION)!r}"
        )
    supported_lifecycle_modes = qualification.get("supported_lifecycle_modes")
    if (
        isinstance(supported_lifecycle_modes, (str, bytes))
        or not isinstance(supported_lifecycle_modes, Sequence)
        or tuple(supported_lifecycle_modes) != (0, 1)
    ):
        raise _fail(
            "qualified CUDA capability must support lifecycle modes [0, 1]"
        )
    if capability.get("generation_summary_scalar_layout_width") != 31:
        raise _fail("qualified CUDA capability requires summary width 31")
    if capability.get("canonical_final_checkpoint_projection_supported") is not True:
        raise _fail("qualified CUDA capability requires terminal checkpoint projection")
    return capability


def _projection_metadata(
    projection: Any,
    *,
    expected_generation: int | None = None,
    expected_population: int = 64,
    expected_ticks_per_generation: int = QUALIFIED_TICKS_PER_GENERATION_V1,
    allow_partial_generation_source_cursor: bool = False,
) -> Mapping[str, Any]:
    metadata = (
        projection.get("metadata")
        if isinstance(projection, Mapping)
        else getattr(projection, "metadata", None)
    )
    if not isinstance(metadata, Mapping):
        raise _fail("build_import_projection returned no metadata mapping")
    expected = {
        "world": "patch",
        "native_runtime_patch_profile": "P2",
        "native_runtime_patch_profile_scope": "full_profile",
        "population_size": expected_population,
        "configured_steps_per_generation": expected_ticks_per_generation,
        "policy_deterministic_actions": False,
    }
    for name, expected_value in expected.items():
        actual = metadata.get(name)
        if type(actual) is not type(expected_value) or actual != expected_value:
            raise _fail(
                f"qualified import projection metadata {name} mismatch: "
                f"expected={expected_value!r} actual={actual!r}"
            )
    tick = metadata.get("tick")
    if (
        type(tick) is not int
        or not 0 <= tick < expected_ticks_per_generation
        or (tick != QUALIFIED_TICK_BASE_V1 and not allow_partial_generation_source_cursor)
    ):
        raise _fail(
            "qualified import projection metadata tick mismatch: "
            f"expected={QUALIFIED_TICK_BASE_V1} actual={tick!r}"
        )
    population = _validated_initial_live_count_v1(
        metadata.get("population_size"),
        path="qualified import projection metadata population_size",
    )
    capacity = metadata.get("capacity")
    if type(capacity) is not int or capacity < population:
        raise _fail(
            "qualified import projection metadata capacity must cover "
            "population_size"
        )
    generation = metadata.get("generation")
    if type(generation) is not int or generation < 0:
        raise _fail(
            "qualified import projection metadata generation must be nonnegative"
        )
    if expected_generation is not None and generation != expected_generation:
        raise _fail(
            "qualified import projection metadata generation mismatch: "
            f"expected={expected_generation} actual={generation}"
        )
    return metadata


def canonical_generation_zero_source_projection_v1(value: Any) -> dict[str, Any]:
    """Materialize the complete canonical fresh G0 source checkpoint surface."""

    if isinstance(value, Mapping):
        metadata = value.get("metadata")
        tensors = value.get("tensors")
        schema_version = value.get("schema_version")
        raw = {
            str(name): (
                tensor.detach().clone()
                if hasattr(tensor, "detach")
                else copy.deepcopy(tensor)
            )
            for name, tensor in dict(value).items()
            if name not in {"metadata", "tensors"}
        }
    else:
        metadata = getattr(value, "metadata", None)
        tensors = getattr(value, "tensors", None)
        schema_version = getattr(value, "schema_version", None)
        raw = {}
    if (
        not isinstance(metadata, Mapping)
        or not isinstance(tensors, Mapping)
        or not isinstance(schema_version, str)
    ):
        raise _fail("generation-zero source builder returned an unsupported projection")
    generation = metadata.get("generation")
    if type(generation) is not int or generation != QUALIFIED_GENERATION_BASE_V1:
        raise _fail("generation-zero source builder returned a nonzero generation")
    active_event_count = metadata.get(
        "typed_lifecycle_event_active_count",
        metadata.get("event_tape_active_count", 0),
    )
    if type(active_event_count) is not int or active_event_count < 0:
        raise _fail("generation-zero source builder returned an invalid event count")
    event_cursor = raw.get("event_cursor", getattr(value, "event_cursor", active_event_count))
    runtime_event_cursor = raw.get(
        "runtime_event_cursor",
        getattr(value, "runtime_event_cursor", event_cursor),
    )
    if (
        type(event_cursor) is not int
        or event_cursor < 0
        or type(runtime_event_cursor) is not int
        or runtime_event_cursor < event_cursor
    ):
        raise _fail("generation-zero source builder returned invalid event cursors")
    pending_ids = raw.get(
        "pending_generation_free_agent_ids",
        getattr(value, "pending_generation_free_agent_ids", ()),
    )
    pending_slots = raw.get(
        "pending_generation_free_slots",
        getattr(value, "pending_generation_free_slots", ()),
    )
    for name, items in (
        ("pending_generation_free_agent_ids", pending_ids),
        ("pending_generation_free_slots", pending_slots),
    ):
        if not isinstance(items, (list, tuple)) or any(
            type(item) is not int for item in items
        ):
            raise _fail(f"generation-zero source builder returned invalid {name}")
    for name in (
        "restore_count",
        "generation_calls",
        "last_applied_actions",
        "last_moved",
        "last_gathered",
        "last_policy_refill_calls",
        "last_policy_refill_rows",
    ):
        value = raw.get(name, 0)
        if type(value) is not int or value != 0:
            raise _fail(
                "generation-zero source builder returned prior runtime history"
            )
    last_world = raw.get("last_world", "none")
    if type(last_world) is not str or last_world != "none":
        raise _fail("generation-zero source builder returned prior runtime history")
    payload = {
        **raw,
        "schema_version": schema_version,
        "metadata": copy.deepcopy(dict(metadata)),
        "tensors": dict(tensors),
        "tensor_manifest": {
            name: _checkpoint_tensor_manifest(tensor)
            for name, tensor in sorted(dict(tensors).items())
            if hasattr(tensor, "detach")
        },
        "restore_count": 0,
        "generation_calls": 0,
        "event_cursor": event_cursor,
        "runtime_event_cursor": runtime_event_cursor,
        "last_applied_actions": 0,
        "last_moved": 0,
        "last_gathered": 0,
        "last_policy_refill_calls": 0,
        "last_policy_refill_rows": 0,
        "pending_generation_free_agent_ids": list(pending_ids),
        "pending_generation_free_slots": list(pending_slots),
        "last_world": last_world,
    }
    return payload


def _build_allocation_request_v1(
    runner: Any,
    *,
    qualified_cuda_module: Any,
    cuda_device: str,
    policy_family: str = "mlp",
    runner_envelope_deriver: Callable[[Any], Any] | None = None,
    allow_short_technical_gate_generation_count: bool = False,
    allow_partial_generation_source_cursor: bool = False,
    source_capability_mask: Mapping[str, bool] | None = None,
    logical_generation_count: int = QUALIFIED_GENERATION_COUNT_V1,
    physical_window_generations: int = PHYSICAL_WINDOW_GENERATIONS_V1,
    agent_id_allocation_generation_count: int | None = None,
    source_checkpoint_projection: Mapping[str, Any] | None = None,
    event_capacity_override: int | None = None,
    host_authoritative_tensor_names: tuple[str, ...] = (),
    event_history_sink_path: str | None = None,
    event_history_sink_cell_index: int = 0,
    direct_batch_host_import_enable: bool = False,
    resident_owner_span_ticks: int | None = None,
    native_dispatch_window_ticks: int | None = None,
    source_window_ticks: int | None = None,
    event_window_capacity: int | None = None,
    tick_native_control: Mapping[str, Any] | None = None,
) -> DirectResidentSourceAllocationRequestV1:
    native_control = dict(tick_native_control or {})
    if "refill_chunk_ticks" in native_control:
        raise _fail(
            "tick-native source allocation rejects refill_chunk_ticks; "
            "source_window_ticks is capacity-only"
        )

    def derive_envelope(*args: Any, **kwargs: Any) -> Any:
        # Keep legacy test doubles and compatibility callers source-compatible;
        # the native keyword is present only when the request actually binds
        # an absolute-tick control contract.
        if native_control:
            kwargs["tick_native_control"] = native_control
        return derive_cuda_architecture_b_runner_envelope(*args, **kwargs)

    if qualified_cuda_module is None:
        raise _fail("qualified_cuda_module is required")
    if type(allow_partial_generation_source_cursor) is not bool:
        raise _fail("allow_partial_generation_source_cursor must be an exact bool")
    if type(direct_batch_host_import_enable) is not bool:
        raise _fail("direct_batch_host_import_enable must be an exact bool")
    device = _validated_cuda_device(cuda_device)
    if event_history_sink_path is not None and (
        type(event_history_sink_path) is not str or not event_history_sink_path.strip()
    ):
        raise _fail("event_history_sink_path must be non-empty text when supplied")
    if type(event_history_sink_cell_index) is not int or event_history_sink_cell_index < 0:
        raise _fail("event_history_sink_cell_index must be a nonnegative integer")
    if type(physical_window_generations) is not int or physical_window_generations not in {
        1,
        2,
    }:
        raise _fail("physical_window_generations must be exact integer 1 or 2")
    physical_steps = physical_window_ticks_v1(physical_window_generations)
    if runner_envelope_deriver is not None:
        envelope = runner_envelope_deriver(runner)
    elif allow_short_technical_gate_generation_count:
        envelope = (
            derive_envelope(
                runner,
                allow_short_technical_gate_generation_count=True,
            )
            if policy_family == "mlp"
            else derive_envelope(
                runner,
                policy_family=policy_family,
                allow_short_technical_gate_generation_count=True,
            )
        )
    else:
        envelope = (
            derive_envelope(runner)
            if policy_family == "mlp"
            else derive_envelope(
                runner,
                policy_family=policy_family,
            )
        )
    cuda_capability = _validated_cuda_capability(qualified_cuda_module)
    envelope_population = getattr(envelope, "population", None)
    envelope_grid = getattr(envelope, "grid", None)
    if (
        type(envelope_population) is not int
        or not isinstance(envelope_grid, tuple)
        or len(envelope_grid) != 2
        or type(envelope_grid[0]) is not int
        or type(envelope_grid[1]) is not int
        or envelope_grid[0] != envelope_grid[1]
        or not _patch_geometry_supported_v1(
            cuda_capability["patch_p2_qualification"],
            population=envelope_population,
            grid_size=envelope_grid[0],
        )
    ):
        raise _fail("qualified CUDA capability does not support runner geometry")
    capability_mask = dict(source_capability_mask or QUALIFIED_CAPABILITY_MASK_V1)
    sim_cfg = getattr(runner, "sim_cfg", None)
    rng_semantics_version = getattr(sim_cfg, "rng_semantics_version", 1)
    world_uid = getattr(sim_cfg, "world_uid", 0)
    rng_namespace_key64 = getattr(
        sim_cfg, "rng_namespace_key64", 0xCA05117E450A51C0
    )
    rng_semantic_epoch = getattr(sim_cfg, "rng_semantic_epoch", 1)
    if type(rng_semantics_version) is not int or rng_semantics_version not in {1, 2}:
        raise _fail("runner RNG semantic version must be exact 1 or 2")
    if type(world_uid) is not int or not 0 <= world_uid <= 0xFFFFFFFF:
        raise _fail("runner RNG world UID must be an exact uint32")
    if (
        type(rng_namespace_key64) is not int
        or rng_namespace_key64 != 0xCA05117E450A51C0
    ):
        raise _fail("runner RNG namespace key must equal the canonical V2 key")
    if type(rng_semantic_epoch) is not int or rng_semantic_epoch != 1:
        raise _fail("runner RNG semantic epoch must equal the V1 contract epoch")
    life_history_config = _life_history_config_from_sim_cfg_v1(sim_cfg)
    if bool(getattr(sim_cfg, "genome_upgrade_v2_enable", False)):
        capability_mask["genome_v2"] = True
    fresh_projection: Mapping[str, Any] | None = None
    if source_checkpoint_projection is None:
        fresh_projection = _bind_life_history_metadata_v1(
            canonical_generation_zero_source_projection_v1(
                build_import_projection(
                    runner,
                    capability_mask,
                    generation=QUALIFIED_GENERATION_BASE_V1,
                    tick=QUALIFIED_TICK_BASE_V1,
                )
            ),
            life_history=life_history_config,
        )
        _projection_metadata(
            fresh_projection,
            expected_generation=QUALIFIED_GENERATION_BASE_V1,
            expected_population=envelope_population,
            expected_ticks_per_generation=envelope.ticks_per_generation,
        )
    projection: Any = fresh_projection
    source_projection_for_digest: Mapping[str, Any] | None = None
    if native_control and event_history_sink_path is None:
        raise _fail("tick-native execution requires an external event-history sink")
    if source_checkpoint_projection is None and event_history_sink_path is not None:
        if fresh_projection is None:
            raise _fail("fresh source projection was not constructed")
        try:
            projection = external_event_projection_v1(
                fresh_projection,
                sink_path=event_history_sink_path,
                cell_index=event_history_sink_cell_index,
            )
        except EventSinkError as exc:
            raise _fail(f"cannot establish fresh event-history authority: {exc}") from exc
    if source_checkpoint_projection is not None:
        if not isinstance(source_checkpoint_projection, Mapping):
            raise _fail("source_checkpoint_projection must be a mapping")
        try:
            validate_native_checkpoint_projection(source_checkpoint_projection)
        except Exception as exc:
            raise _fail(f"source checkpoint projection is invalid: {exc}") from exc
        projection = source_checkpoint_projection
        source_projection_for_digest = source_checkpoint_projection
        if event_history_sink_path is not None:
            source_metadata = dict(projection.get("metadata") or {})
            if source_metadata.get("event_history_storage_mode") != EVENT_SINK_SCHEMA_VERSION:
                try:
                    projection = external_event_projection_v1(
                        projection,
                        sink_path=event_history_sink_path,
                        cell_index=event_history_sink_cell_index,
                    )
                except EventSinkError as exc:
                    raise _fail(f"cannot externalize checkpoint event history: {exc}") from exc
            else:
                reference = source_metadata.get("event_history_reference")
                expected_path = str(Path(event_history_sink_path).expanduser().resolve())
                if (
                    not isinstance(reference, Mapping)
                    or str(reference.get("path")) != expected_path
                    or reference.get("cell_index") != event_history_sink_cell_index
                ):
                    raise _fail(
                        "external event-history checkpoint reference does not match "
                        "the admitted sink path"
                    )
                try:
                    validate_event_sink_prefix_reference_v1(reference)
                except EventSinkError as exc:
                    raise _fail(
                        f"event-history checkpoint prefix is not authentic: {exc}"
                    ) from exc
        if _contains_cuda_tensor_v1(projection) or _cpu_owner_tensor_manifest_stale_v1(
            projection
        ):
            # A completed native capture may retain CUDA-backed tensors in its
            # immutable settlement projection. The next owner crosses the
            # explicit CPU import boundary; refresh the manifest after staging.
            projection = _stage_cpu_owner_projection_v1(projection)
    metadata = _projection_metadata(
        projection,
        expected_population=envelope_population,
        expected_ticks_per_generation=envelope.ticks_per_generation,
        allow_partial_generation_source_cursor=(
            allow_partial_generation_source_cursor
        ),
    )
    if rng_semantics_version == 2:
        expected_rng_metadata = {
            "tw_rngv2_semantics_version": rng_semantics_version,
            "tw_rngv2_world_uid": world_uid,
            "tw_rngv2_namespace_key64": rng_namespace_key64,
            "tw_rngv2_semantic_epoch": rng_semantic_epoch,
        }
        for name, expected in expected_rng_metadata.items():
            if metadata.get(name) != expected:
                raise _fail(
                    "source checkpoint RNG identity disagrees with runner: "
                    f"{name}"
                )
    _validate_life_history_metadata_v1(
        metadata,
        life_history=life_history_config,
    )
    active_event_rows = _source_event_active_prefix_bound_v1(projection, metadata)
    source_generation = metadata.get("generation")
    max_births_per_tick = getattr(envelope, "max_births_per_tick", None)
    if type(source_generation) is not int or source_generation < 0:
        raise _fail("source projection generation must be nonnegative")
    if type(max_births_per_tick) is not int or max_births_per_tick < 1:
        raise _fail("runner envelope max_births_per_tick must be positive")
    ordinary_births_enabled = getattr(sim_cfg, "ordinary_births_enabled", None)
    if ordinary_births_enabled is None:
        ordinary_births_enabled = max_births_per_tick > 0
    elif type(ordinary_births_enabled) is not bool:
        raise _fail("runner ordinary_births_enabled must be an exact bool")
    native_source_window_ticks = source_window_ticks
    if native_control:
        control_source_window_ticks = native_control.get("source_window_ticks")
        if type(control_source_window_ticks) is not int:
            raise _fail(
                "tick-native control source_window_ticks must be an exact integer"
            )
        if (
            native_source_window_ticks is not None
            and native_source_window_ticks != control_source_window_ticks
        ):
            raise _fail(
                "source_window_ticks disagrees with tick-native control"
            )
        native_source_window_ticks = control_source_window_ticks
    if native_source_window_ticks is None:
        native_source_window_ticks = envelope.ticks_per_generation
    event_rows_per_source_window = source_derived_event_window_capacity_v1(
        max_births_per_tick=max_births_per_tick,
        source_window_ticks=native_source_window_ticks,
        base_event_rows_per_source_window=BASE_EVENT_ROWS_PER_SOURCE_WINDOW_V1,
        max_event_rows_per_ordinary_birth=MAX_EVENT_ROWS_PER_ORDINARY_BIRTH_V1,
    )
    compact_event_rows = _compact_no_birth_event_rows_per_generation_v1(
        sim_cfg,
        ordinary_births_enabled=ordinary_births_enabled,
        resident_capacity=metadata.get("capacity"),
    )
    resident_event_generation_span = _resident_event_capacity_generation_span_v1(
        source_generation=source_generation,
        logical_generation_count=logical_generation_count,
        physical_window_generations=physical_window_generations,
    )
    # Event history is a canonical checkpoint/reporting surface, not a live
    # simulation dependency.  Once a segment resumes from a nonempty
    # checkpoint, keep that history in the CPU-owned canonical projection and
    # start the new resident dispatch with an empty physical event window.
    # The segment settlement appends its newly emitted rows back to the
    # canonical prefix before exporting the next checkpoint.
    drain_event_history = active_event_rows > 0
    resident_active_event_rows = 0 if drain_event_history else active_event_rows
    if compact_event_rows is not None:
        event_rows_per_source_window = compact_event_rows
        event_capacity = resident_active_event_rows + (
            resident_event_generation_span * event_rows_per_source_window
        )
    else:
        event_capacity = (
            resident_active_event_rows
            + resident_event_generation_span * event_rows_per_source_window
        )
    native_event_capacity = event_window_capacity
    if native_event_capacity is None:
        raw_native_capacity = native_control.get("event_window_capacity")
        if type(raw_native_capacity) is int:
            native_event_capacity = raw_native_capacity
    if native_control or native_event_capacity is not None:
        if type(native_event_capacity) is not int or native_event_capacity <= 0:
            raise _fail(
                "tick-native event_window_capacity must be a positive exact integer"
            )
        # Emitted history lives in the external sink.  The CUDA owner receives
        # only a fixed pending ring, independent of total_ticks and of the
        # compatibility segment count.
        required_event_window_capacity = event_rows_per_source_window
        if native_event_capacity < required_event_window_capacity:
            raise _fail(
                "tick-native event_window_capacity is below the proved worst-case "
                "single lifecycle-window emission bound: "
                f"configured={native_event_capacity} "
                f"required={required_event_window_capacity}"
            )
        event_capacity = native_event_capacity
    if event_capacity < resident_active_event_rows:
        raise _fail("bounded resident event capacity does not cover the active prefix")
    if native_control:
        control_event_capacity = native_control.get("event_window_capacity")
        if type(control_event_capacity) is not int or control_event_capacity <= 0:
            raise _fail(
                "tick-native control event_window_capacity must be a positive "
                "exact integer"
            )
        if control_event_capacity != native_event_capacity:
            raise _fail(
                "tick-native event_window_capacity disagrees with its control"
            )
        event_capacity = control_event_capacity
    if event_capacity_override is not None:
        if type(event_capacity_override) is not int or event_capacity_override < 0:
            raise _fail("event_capacity_override must be a nonnegative exact integer")
        if native_control and event_capacity_override != event_capacity:
            raise _fail(
                "event_capacity_override cannot replace tick-native event_window_capacity"
            )
        if event_capacity_override < event_capacity:
            raise _fail("event_capacity_override cannot truncate the bounded event capacity")
        event_capacity = event_capacity_override
    cpu_request_staging_enable = getattr(
        sim_cfg,
        "cuda_architecture_b_cpu_request_staging_enable",
        False,
    )
    if type(cpu_request_staging_enable) is not bool:
        raise _fail(
            "runner cuda_architecture_b_cpu_request_staging_enable "
            "must be an exact bool"
        )
    if cpu_request_staging_enable and policy_family != "mlp":
        raise _fail("CPU request staging is admitted only for fixed MLP policy")
    if direct_batch_host_import_enable and policy_family != "mlp":
        raise _fail("direct batch host import is admitted only for fixed MLP policy")
    if direct_batch_host_import_enable and cpu_request_staging_enable:
        raise _fail(
            "direct batch host import cannot be combined with CPU resident staging"
        )
    boundary_kwargs: dict[str, Any] = {
        "event_capacity": event_capacity,
        "device": (
            "cpu"
            if cpu_request_staging_enable or direct_batch_host_import_enable
            else device
        ),
    }
    if cpu_request_staging_enable or direct_batch_host_import_enable:
        boundary_kwargs["allow_cpu_staging"] = True
    if allow_partial_generation_source_cursor:
        boundary_kwargs["allow_partial_generation_source_cursor"] = True
    if host_authoritative_tensor_names:
        boundary_kwargs["host_authoritative_tensor_names"] = (
            host_authoritative_tensor_names
        )
    if drain_event_history:
        boundary_kwargs["drain_event_history"] = True
    if policy_family == "gru":
        boundary_kwargs["policy_family"] = "stable_gru"
    boundary = import_cpu_owner_boundary_to_cuda(projection, **boundary_kwargs)
    boundary_metadata = getattr(boundary, "metadata", None)
    if not isinstance(boundary_metadata, Mapping):
        raise _fail("qualified CUDA boundary is missing metadata")
    physical_turnover_refill_capacity = _runtime_turnover_refill_capacity_v1(
        sim_cfg,
        population_size=envelope_population,
        slot_capacity=boundary_metadata.get("capacity"),
        ordinary_births_enabled=ordinary_births_enabled,
        physical_window_generations=physical_window_generations,
    )
    max_births_per_tick, policy_episode_capacity = (
        _validated_physical_birth_capacity_v1(
            getattr(envelope, "max_births_per_tick", None),
            boundary_metadata.get("capacity"),
            physical_window_generations=physical_window_generations,
            turnover_refill_capacity=physical_turnover_refill_capacity,
        )
    )
    turnover_selection_mode = getattr(sim_cfg, "turnover_selection_mode", 0)
    if type(turnover_selection_mode) is not int or turnover_selection_mode not in {
        0,
        1,
    }:
        raise _fail("runner turnover_selection_mode must be exact integer 0 or 1")
    lifecycle_mode = getattr(sim_cfg, "lifecycle_mode", 0)
    if type(lifecycle_mode) is not int or lifecycle_mode not in {0, 1}:
        raise _fail("runner lifecycle_mode must be exact integer 0 or 1")
    projected_turnover = metadata.get("generation_turnover_enabled")
    projected_lifecycle_mode = metadata.get("lifecycle_mode_abi")
    if projected_turnover is not None or projected_lifecycle_mode is not None:
        if (
            type(projected_turnover) is not bool
            or isinstance(projected_lifecycle_mode, bool)
            or projected_lifecycle_mode not in {0, 1}
        ):
            raise _fail("source checkpoint lifecycle metadata is malformed")
        expected_lifecycle_mode = 0 if projected_turnover else 1
        if projected_lifecycle_mode != expected_lifecycle_mode:
            raise _fail("source checkpoint lifecycle metadata is contradictory")
        if source_checkpoint_projection is not None:
            lifecycle_mode = expected_lifecycle_mode
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
    inheritance_sidecar_cell_id = getattr(
        runner,
        "cuda_architecture_b_inheritance_sidecar_cell_id",
        getattr(sim_cfg, "cuda_architecture_b_inheritance_sidecar_cell_id", None),
    )
    if inheritance_sidecar_cell_id is not None and (
        type(inheritance_sidecar_cell_id) is not str
        or not inheritance_sidecar_cell_id
        or inheritance_sidecar_cell_id.strip() != inheritance_sidecar_cell_id
    ):
        raise _fail("runner inheritance sidecar cell_id must be stable text")
    genome_v2_causal_physical_cell_ordinal = getattr(
        runner,
        "cuda_architecture_b_genome_v2_causal_physical_cell_ordinal",
        getattr(sim_cfg, "cuda_architecture_b_genome_v2_causal_physical_cell_ordinal", 0),
    )
    if (
        type(genome_v2_causal_physical_cell_ordinal) is not int
        or genome_v2_causal_physical_cell_ordinal < 0
    ):
        raise _fail(
            "runner Genome V2 causal physical cell ordinal must be nonnegative"
        )
    birth_state_kwargs: dict[str, Any] = {
        "steps": physical_steps,
        "max_births_per_tick": max_births_per_tick,
        "turnover_refill_capacity": physical_turnover_refill_capacity,
        "stage_cpython_gaussian_at_raw_draw_base": True,
        "rng_semantics_version": rng_semantics_version,
    }
    if rng_semantics_version == 1:
        birth_state_kwargs["policy_tape_staging_device"] = "cpu"
    birth_state_kwargs.update(
        birth_state_capacity_overrides_v1(
            sim_cfg,
            ordinary_births_enabled=ordinary_births_enabled,
            resident_policy_episode_capacity=policy_episode_capacity,
            turnover_refill_capacity=physical_turnover_refill_capacity,
            max_births_per_tick=max_births_per_tick,
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
    if turnover_mlp_shadow_elide_enable:
        if lifecycle_mode != 0:
            raise _fail("turnover MLP shadow elision requires generation turnover")
        if policy_family != "mlp":
            raise _fail("turnover MLP shadow elision requires fixed MLP policy")
        if metadata.get("policy_weight_mutation_enable") is not False:
            raise _fail(
                "turnover MLP shadow elision requires disabled policy-weight mutation"
            )
        policy_birth_segment_refill_requested = getattr(
            sim_cfg,
            "cuda_architecture_b_policy_birth_segment_refill_enable",
            False,
        )
        if type(policy_birth_segment_refill_requested) is not bool:
            raise _fail(
                "runner cuda_architecture_b_policy_birth_segment_refill_enable "
                "must be an exact bool"
            )
        if (
            birth_state_kwargs.get("policy_birth_segment_refill_enabled") is not True
            and policy_birth_segment_refill_requested is not True
        ):
            raise _fail(
                "turnover MLP shadow elision requires policy birth segment refill"
            )
    if cpu_request_staging_enable and rng_semantics_version == 1:
        birth_state_kwargs["policy_tape_staging_device"] = "cpu"
    if "mlp_pack" in host_authoritative_tensor_names:
        birth_state_kwargs["allow_host_authoritative_mlp_pack"] = True
    birth_state = prepare_generation_birth_state_cuda(
        boundary,
        projection,
        **birth_state_kwargs,
    )
    source_import_projection_digest_sha256: str | None = None
    if source_checkpoint_projection is not None:
        # A resumed bundle is a transport envelope, not a second resident
        # owner. Preserve its exact source digest, then retain only the
        # canonical view required by allocator and settlement. This allows
        # the provider to release decoded bundle storage after CUDA import.
        from engine.native_runtime.checkpoint_digests import (
            checkpoint_source_projection_authority_manifest,
        )

        try:
            source_import_projection_digest_sha256 = str(
                checkpoint_source_projection_authority_manifest(
                    source_projection_for_digest or projection
                )[
                    "source_projection_sha256"
                ]
            )
        except (KeyError, TypeError, ValueError):
            # Older synthetic/test checkpoint fixtures predate the complete
            # Bundle-v2 authority surface. Keep their legacy object intact;
            # real persisted bundles take the compact path below.
            source_import_projection_digest_sha256 = None
        else:
            projection = _canonical_projection(
                projection,
                source_generation=source_generation,
            )
    configuration = replace(
        CudaGenerationConfig.from_metadata(
            metadata,
            steps=envelope.ticks_per_generation,
        ),
        generation=int(metadata["generation"]),
        start_tick=int(metadata.get("tick", QUALIFIED_TICK_BASE_V1)),
        max_births_per_tick=max_births_per_tick,
        ordinary_births_enabled=ordinary_births_enabled,
        generation_turnover_enabled=lifecycle_mode == 0,
        lifecycle_mode=lifecycle_mode,
        turnover_selection_mode=turnover_selection_mode,
        turnover_selection_seed=turnover_selection_seed,
        rng_semantics_version=rng_semantics_version,
        world_uid=world_uid,
        rng_namespace_key64=rng_namespace_key64,
        rng_semantic_epoch=rng_semantic_epoch,
        **life_history_config,
        genome_upgrade_v2_enable=bool(
            getattr(
                getattr(runner, "sim_cfg", None),
                "genome_upgrade_v2_enable",
                False,
            )
        ),
        genome_upgrade_v2_founder_profile_id=str(
            getattr(
                getattr(runner, "sim_cfg", None),
                "genome_upgrade_v2_founder_profile_id",
                "neutral",
            )
        ),
        genome_upgrade_v2_founder_profile_sha256=str(
            getattr(
                getattr(runner, "sim_cfg", None),
                "genome_upgrade_v2_founder_profile_sha256",
                "",
            )
        ),
        genome_v2_checkpoint_authority_binding=tuple(
            getattr(
                getattr(runner, "sim_cfg", None),
                "genome_v2_checkpoint_authority_binding",
                (),
            )
        ),
        genome_v2_causal_capture_enable=bool(
            getattr(
                getattr(runner, "sim_cfg", None),
                "genome_v2_causal_capture_enable",
                False,
            )
        ),
    )
    return DirectResidentSourceAllocationRequestV1(
        envelope=envelope,
        qualified_cuda_module=qualified_cuda_module,
        cuda_capability=cuda_capability,
        import_projection=projection,
        cuda_boundary=boundary,
        generation_birth_state=birth_state,
        configuration=configuration,
        cuda_device=device,
        event_capacity=event_capacity,
        logical_generation_count=logical_generation_count,
        physical_window_generations=physical_window_generations,
        agent_id_allocation_generation_count=agent_id_allocation_generation_count,
        allow_short_technical_gate_generation_count=(
            allow_short_technical_gate_generation_count
        ),
        allow_partial_generation_source_cursor=(
            allow_partial_generation_source_cursor
        ),
        inheritance_sidecar_cell_id=inheritance_sidecar_cell_id,
        genome_v2_causal_physical_cell_ordinal=(
            genome_v2_causal_physical_cell_ordinal
        ),
        turnover_mlp_shadow_elide_enable=turnover_mlp_shadow_elide_enable,
        cpu_request_staging_enable=cpu_request_staging_enable,
        direct_batch_host_import_enable=direct_batch_host_import_enable,
        source_import_projection_digest_sha256=(
            source_import_projection_digest_sha256
        ),
        event_history_sink_path=event_history_sink_path,
        event_history_sink_cell_index=event_history_sink_cell_index,
        resident_owner_span_ticks=resident_owner_span_ticks,
        native_dispatch_window_ticks=native_dispatch_window_ticks,
        source_window_ticks=native_source_window_ticks,
        event_window_capacity=native_event_capacity,
        tick_native_control=native_control,
    )


def build_direct_resident_source_allocation_request_v1(
    runner: Any,
    *,
    qualified_cuda_module: Any,
    cuda_device: str = "cuda",
    policy_family: str = "mlp",
    runner_envelope_deriver: Callable[[Any], Any] | None = None,
    allow_short_technical_gate_generation_count: bool = False,
    allow_partial_generation_source_cursor: bool = False,
    source_capability_mask: Mapping[str, bool] | None = None,
    logical_generation_count: int = QUALIFIED_GENERATION_COUNT_V1,
    physical_window_generations: int = PHYSICAL_WINDOW_GENERATIONS_V1,
    agent_id_allocation_generation_count: int | None = None,
    source_checkpoint_projection: Mapping[str, Any] | None = None,
    event_capacity_override: int | None = None,
    host_authoritative_tensor_names: tuple[str, ...] = (),
    event_history_sink_path: str | None = None,
    event_history_sink_cell_index: int = 0,
    resident_owner_span_ticks: int | None = None,
    native_dispatch_window_ticks: int | None = None,
    source_window_ticks: int | None = None,
    event_window_capacity: int | None = None,
    tick_native_control: Mapping[str, Any] | None = None,
) -> DirectResidentSourceAllocationRequestV1:
    """Translate one real runner into the public qualified allocation request.

    Campaign/provider layers use this boundary instead of depending on the
    source builder's private assembly helper.
    """

    kwargs = {
        "qualified_cuda_module": qualified_cuda_module,
        "cuda_device": cuda_device,
        "logical_generation_count": logical_generation_count,
        "physical_window_generations": physical_window_generations,
        "agent_id_allocation_generation_count": agent_id_allocation_generation_count,
        "resident_owner_span_ticks": resident_owner_span_ticks,
        "native_dispatch_window_ticks": native_dispatch_window_ticks,
        "source_window_ticks": source_window_ticks,
        "event_window_capacity": event_window_capacity,
        "tick_native_control": tick_native_control,
    }
    if policy_family != "mlp":
        kwargs["policy_family"] = policy_family
    if runner_envelope_deriver is not None:
        kwargs["runner_envelope_deriver"] = runner_envelope_deriver
    if allow_short_technical_gate_generation_count:
        kwargs["allow_short_technical_gate_generation_count"] = True
    if allow_partial_generation_source_cursor:
        kwargs["allow_partial_generation_source_cursor"] = True
    if source_capability_mask is not None:
        kwargs["source_capability_mask"] = source_capability_mask
    if source_checkpoint_projection is not None:
        kwargs["source_checkpoint_projection"] = source_checkpoint_projection
    if event_capacity_override is not None:
        kwargs["event_capacity_override"] = event_capacity_override
    if host_authoritative_tensor_names:
        kwargs["host_authoritative_tensor_names"] = host_authoritative_tensor_names
    if event_history_sink_path is not None:
        kwargs["event_history_sink_path"] = event_history_sink_path
        kwargs["event_history_sink_cell_index"] = event_history_sink_cell_index
    return _build_allocation_request_v1(runner, **kwargs)


def build_prepared_direct_resident_trajectory_v1(
    runner: Any,
    *,
    qualified_cuda_module: Any,
    source_allocator: DirectResidentSourceAllocatorV1 | None = None,
    cuda_device: str = "cuda",
    logical_generation_count: int = QUALIFIED_GENERATION_COUNT_V1,
    physical_window_generations: int = PHYSICAL_WINDOW_GENERATIONS_V1,
    agent_id_allocation_generation_count: int | None = None,
    source_checkpoint_projection: Mapping[str, Any] | None = None,
) -> PreparedDirectResidentTrajectoryV1:
    """Build one exact production trajectory without creating another owner.

    The default allocator is engine-owned. Challenger allocators are rejected.
    """

    allocator = _validated_source_allocator(source_allocator)
    request = build_direct_resident_source_allocation_request_v1(
        runner,
        qualified_cuda_module=qualified_cuda_module,
        cuda_device=cuda_device,
        logical_generation_count=logical_generation_count,
        physical_window_generations=physical_window_generations,
        agent_id_allocation_generation_count=agent_id_allocation_generation_count,
        source_checkpoint_projection=source_checkpoint_projection,
    )
    source = allocator(request)
    if not isinstance(source, DirectResidentSourceV1):
        raise _fail("engine source allocator must return DirectResidentSourceV1")
    return prepare_direct_resident_trajectory_v1(source)


__all__ = [
    "ENGINE_SOURCE_ALLOCATOR_MODULE_V1",
    "ENGINE_SOURCE_ALLOCATOR_NAME_V1",
    "MISSING_ENGINE_SOURCE_DEPENDENCY_V1",
    "NATIVE_CAPABILITY_ENTRYPOINT_V1",
    "QUALIFIED_CAPABILITY_MASK_V1",
    "QUALIFIED_EVENT_CAPACITY_V1",
    "QUALIFIED_GENERATION_COUNT_V1",
    "QUALIFIED_MAX_BIRTHS_PER_TICK_V1",
    "QUALIFIED_POLICY_EPISODE_CAPACITY_V1",
    "QUALIFIED_SOURCE_STEPS_V1",
    "QUALIFIED_TICKS_PER_GENERATION_V1",
    "QUALIFIED_TURNOVER_REFILL_CAPACITY_V1",
    "QUALIFIED_TURNOVER_EPISODES_PER_GENERATION_V1",
    "CudaArchitectureBDirectResidentSourceBuilderError",
    "DirectResidentSourceAllocationRequestV1",
    "DirectResidentSourceAllocatorV1",
    "MissingEngineDirectResidentSourceAllocatorV1Error",
    "build_direct_resident_source_allocation_request_v1",
    "build_prepared_direct_resident_trajectory_v1",
    "canonical_generation_zero_source_projection_v1",
    "required_qualified_reproduction_attempt_capacity_v1",
]
