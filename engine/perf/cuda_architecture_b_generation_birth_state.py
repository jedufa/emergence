"""One-time generation-boundary preparation for resident CUDA births."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Final, Mapping

import torch

from perf.cuda_architecture_b_birth_transaction import (
    DRIVE_STATE_FIELDS,
    GENOME_FLOAT32_FIELDS,
    GENOME_INT32_FIELDS,
    IDENTITY_STATE_FIELDS,
    POLICY_STATE_FIELDS,
    SCALAR_RNG_STATE_FIELDS,
)
from perf.cuda_architecture_b_boundary import (
    BOUNDARY_IMPORT_SCHEMA_VERSION,
    CHECKPOINT_PROJECTION_SCHEMA_VERSION,
    EVENT_COLUMN_NAMES,
    CudaBoundaryImport,
)
from perf.cuda_architecture_b_child_policy import (
    prepare_fixed_mlp_child_policy_tape,
    prepare_fixed_mlp_turnover_child_policy_tape,
)
from perf.cuda_architecture_b_lifecycle import pack_cpython_random_state_cuda
from perf.cuda_architecture_b_torch_rng_state_pack_v1 import (
    TORCH_CPU_RNG_STATE_ENCODING_V1,
    TORCH_CPU_RNG_STATE_FULL_BYTES_V1,
    TORCH_CPU_RNG_STATE_PACKED_BYTES_V1,
    torch_cpu_rng_state_abi_receipt_v1,
)
from perf.cxxbackbone10_resident_state_owner import projection_digest


GENERATION_BIRTH_STATE_SCHEMA_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_generation_birth_state_v1"
)
CPYTHON_GAUSS_RAW_CAPACITY_PER_BIRTH: Final[int] = 512
MAX_CPYTHON_GAUSS_RAW_CAPACITY: Final[int] = 10_549_248
MAX_FIXED_MLP_POLICY_TAPE_EPISODES: Final[int] = 20_604
MAX_COMPACT_AGENT_ID: Final[int] = (1 << 31) - 1
PACKED_CROSSOVER_POLICY_FIELD: Final[str] = "policy_birth_crossover_mask_packed_tape"
ORDINARY_POLICY_STATE_FIELDS: Final[tuple[str, ...]] = POLICY_STATE_FIELDS


class CudaGenerationBirthStateError(RuntimeError):
    """Raised when a generation-boundary birth state is not exact or bounded."""


def _runtime_policy_payload(
    state: Mapping[str, torch.Tensor],
    *,
    include_packed_crossover: bool,
) -> dict[str, torch.Tensor]:
    fields = (
        POLICY_STATE_FIELDS
        if include_packed_crossover
        else ORDINARY_POLICY_STATE_FIELDS
    )
    return {name: state[name] for name in fields}


@dataclass(frozen=True)
class CudaGenerationBirthStateReceipt:
    schema_version: str
    source_projection_digest: str
    target_device: str
    world: str
    steps: int
    max_births_per_tick: int
    slot_capacity: int
    birth_episode_capacity: int
    turnover_refill_capacity: int
    agent_id_to_slot_capacity: int
    cpython_gauss_raw_capacity: int
    fixed_mlp_policy_tape_episodes: int
    fixed_mlp_policy_full_horizon_episodes: int
    policy_birth_segment_refill_enabled: bool
    cpu_torch_rng_source_reads: int
    generation_boundary_device_validation_syncs: int
    torch_rng_post_state_encoding: str
    torch_rng_post_state_logical_row_bytes: int
    torch_rng_post_state_packed_row_bytes: int
    torch_rng_post_state_dense_allocation_materialized: bool
    torch_rng_post_state_resident_rows_materialized: bool
    torch_rng_post_state_abi: Mapping[str, object]
    generation_boundary_only: bool = True
    per_tick_python: bool = False
    simulation_executed: bool = False
    generation_integration_claimed: bool = False
    checkpoint_closure_claimed: bool = False
    parity_closure_claimed: bool = False
    multicell_claimed: bool = False
    promotion_qualified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_projection_digest": self.source_projection_digest,
            "target_device": self.target_device,
            "world": self.world,
            "steps": self.steps,
            "max_births_per_tick": self.max_births_per_tick,
            "slot_capacity": self.slot_capacity,
            "birth_episode_capacity": self.birth_episode_capacity,
            "turnover_refill_capacity": self.turnover_refill_capacity,
            "agent_id_to_slot_capacity": self.agent_id_to_slot_capacity,
            "cpython_gauss_raw_capacity": self.cpython_gauss_raw_capacity,
            "fixed_mlp_policy_tape_episodes": self.fixed_mlp_policy_tape_episodes,
            "fixed_mlp_policy_full_horizon_episodes": (
                self.fixed_mlp_policy_full_horizon_episodes
            ),
            "policy_birth_segment_refill_enabled": (
                self.policy_birth_segment_refill_enabled
            ),
            "cpu_torch_rng_source_reads": self.cpu_torch_rng_source_reads,
            "generation_boundary_device_validation_syncs": (
                self.generation_boundary_device_validation_syncs
            ),
            "torch_rng_post_state_encoding": self.torch_rng_post_state_encoding,
            "torch_rng_post_state_logical_row_bytes": (
                self.torch_rng_post_state_logical_row_bytes
            ),
            "torch_rng_post_state_packed_row_bytes": (
                self.torch_rng_post_state_packed_row_bytes
            ),
            "torch_rng_post_state_dense_allocation_materialized": (
                self.torch_rng_post_state_dense_allocation_materialized
            ),
            "torch_rng_post_state_resident_rows_materialized": (
                self.torch_rng_post_state_resident_rows_materialized
            ),
            "torch_rng_post_state_abi": dict(self.torch_rng_post_state_abi),
            "generation_boundary_only": self.generation_boundary_only,
            "per_tick_python": self.per_tick_python,
            "simulation_executed": self.simulation_executed,
            "generation_integration_claimed": self.generation_integration_claimed,
            "checkpoint_closure_claimed": self.checkpoint_closure_claimed,
            "parity_closure_claimed": self.parity_closure_claimed,
            "multicell_claimed": self.multicell_claimed,
            "promotion_qualified": self.promotion_qualified,
        }


@dataclass(frozen=True)
class CudaGenerationBirthState:
    schema_version: str
    identity_state: dict[str, torch.Tensor]
    scalar_rng_state: dict[str, torch.Tensor]
    genome_state: dict[str, torch.Tensor]
    policy_mem_dim: torch.Tensor
    policy_use_gru: torch.Tensor
    drive_state: dict[str, torch.Tensor]
    policy_state: dict[str, torch.Tensor]
    turnover_policy_state: dict[str, torch.Tensor]
    planner_pcg64_state_words: torch.Tensor
    planner_pcg64_buffer_words: torch.Tensor
    policy_np_semantic_draws: torch.Tensor
    runner_pcg64_state_words: torch.Tensor
    runner_pcg64_buffer_words: torch.Tensor
    metrics_np_semantic_draws: torch.Tensor
    event_tape_columns: tuple[torch.Tensor, ...]
    event_tape_active_count: torch.Tensor
    event_tape_status: torch.Tensor
    receipt: CudaGenerationBirthStateReceipt

    def microcycle_state_kwargs(self) -> dict[str, Any]:
        """Return the state-only arguments accepted by the birth microcycle."""
        return {
            "identity_state": self.identity_state,
            "scalar_rng_state": self.scalar_rng_state,
            "genome_state": self.genome_state,
            "policy_mem_dim": self.policy_mem_dim,
            "policy_use_gru": self.policy_use_gru,
            "drive_state": self.drive_state,
            "policy_state": self.policy_state,
            "planner_pcg64_state_words": self.planner_pcg64_state_words,
            "planner_pcg64_buffer_words": self.planner_pcg64_buffer_words,
            "policy_np_semantic_draws": self.policy_np_semantic_draws,
            "runner_pcg64_state_words": self.runner_pcg64_state_words,
            "runner_pcg64_buffer_words": self.runner_pcg64_buffer_words,
            "metrics_np_semantic_draws": self.metrics_np_semantic_draws,
            "event_tape_columns": self.event_tape_columns,
            "event_tape_active_count": self.event_tape_active_count,
            "event_tape_status": self.event_tape_status,
        }


GENERATION_BIRTH_STATE_CAPABILITIES: Final[dict[str, bool | str]] = {
    "schema_version": "eiel_cuda_architecture_b_generation_birth_state_capabilities_v1",
    "decision": "ARCHB-CUDA/GENERATION-BOUNDARY-ORDINARY-BIRTH-PREPARATION",
    "generation_boundary_only": True,
    "per_tick_python": False,
    "fixed_capacity_slot_free_queue": True,
    "resident_slot_free_queue_count": True,
    "bounded_dense_agent_id_to_slot": True,
    "exact_cpython_gaussian_tape": True,
    "exact_fixed_mlp_child_policy_tape": True,
    "rngv2_formula_policy_crossover": True,
    "rngv2_future_policy_tape": False,
    "packed_torch_cpu_post_state_tape": True,
    "authoritative_tensor_aliases": True,
    "planner_pcg64_alias": "policy_birth_spawn_env",
    "child_drive_pcg64_alias": "population",
    "semantic_draw_counters": True,
    "flat": True,
    "patch_p2": True,
    "simulation_execution": False,
    "generation_integration": False,
    "checkpoint_closure": False,
    "multicell": False,
    "promotion_qualified": False,
}


def _strict_int(value: Any, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise CudaGenerationBirthStateError(
            f"{name} must be an exact integer greater than or equal to {minimum}"
        )
    return int(value)


def _strict_bool(metadata: Mapping[str, Any], name: str) -> bool:
    value = metadata.get(name)
    if type(value) is not bool:
        raise CudaGenerationBirthStateError(f"metadata.{name} must be an exact bool")
    return bool(value)


def _finite_float(metadata: Mapping[str, Any], name: str) -> float:
    value = metadata.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CudaGenerationBirthStateError(f"metadata.{name} must be numeric")
    result = float(value)
    if not isfinite(result):
        raise CudaGenerationBirthStateError(f"metadata.{name} must be finite")
    return result


def _projection_parts(projection: Any) -> tuple[str, Mapping[str, Any], Mapping[str, Any]]:
    if isinstance(projection, Mapping):
        schema_version = projection.get("schema_version")
        metadata = projection.get("metadata")
        tensors = projection.get("tensors")
    else:
        schema_version = getattr(projection, "schema_version", None)
        metadata = getattr(projection, "metadata", None)
        tensors = getattr(projection, "tensors", None)
    if type(schema_version) is not str:
        raise CudaGenerationBirthStateError(
            "original CPU owner projection schema_version must be a string"
        )
    if not isinstance(metadata, Mapping) or not isinstance(tensors, Mapping):
        raise CudaGenerationBirthStateError(
            "original CPU owner projection must expose metadata and tensors mappings"
        )
    return schema_version, metadata, tensors


def _resident_tensor(
    tensors: Mapping[str, torch.Tensor],
    name: str,
    *,
    device: torch.device,
) -> torch.Tensor:
    tensor = tensors.get(name)
    if not isinstance(tensor, torch.Tensor):
        raise CudaGenerationBirthStateError(f"CUDA boundary tensor is missing: {name}")
    if tensor.device != device or not tensor.is_contiguous():
        raise CudaGenerationBirthStateError(
            f"boundary tensor is not contiguous on {device}: {name}"
        )
    if tensor.requires_grad:
        raise CudaGenerationBirthStateError(
            f"CUDA boundary tensor must not require gradients: {name}"
        )
    return tensor


def _stream_index(
    metadata: Mapping[str, Any],
    stream: str,
    source: str,
) -> int:
    streams = metadata.get("native_runtime_rng_manager_stream_indices")
    row = streams.get(stream) if isinstance(streams, Mapping) else None
    if not isinstance(row, Mapping):
        raise CudaGenerationBirthStateError(
            f"metadata RNG stream indices are missing stream={stream}"
        )
    return _strict_int(
        row.get(source),
        name=f"metadata.native_runtime_rng_manager_stream_indices.{stream}.{source}",
    )


def _python_random_state(
    metadata: Mapping[str, Any],
) -> tuple[tuple[int, tuple[int, ...], float | None], int]:
    payload = metadata.get("native_runtime_python_random_state")
    if not isinstance(payload, Mapping):
        raise CudaGenerationBirthStateError(
            "metadata.native_runtime_python_random_state must be a mapping"
        )
    version = _strict_int(payload.get("version"), name="Python RNG version", minimum=3)
    if version != 3:
        raise CudaGenerationBirthStateError("CPython random state version 3 is required")
    words = payload.get("state")
    if not isinstance(words, (list, tuple)) or len(words) != 624:
        raise CudaGenerationBirthStateError(
            "CPython random state must contain exactly 624 words"
        )
    packed_words: list[int] = []
    for index, word in enumerate(words):
        packed_words.append(
            _strict_int(word, name=f"CPython random state word {index}")
        )
        if packed_words[-1] > 0xFFFFFFFF:
            raise CudaGenerationBirthStateError(
                f"CPython random state word {index} exceeds uint32"
            )
    state_index = _strict_int(
        payload.get("index"), name="CPython random state index"
    )
    if state_index > 624:
        raise CudaGenerationBirthStateError(
            "CPython random state index must not exceed 624"
        )
    gauss_next = payload.get("gauss_next")
    if gauss_next is not None:
        if isinstance(gauss_next, bool) or not isinstance(gauss_next, (int, float)):
            raise CudaGenerationBirthStateError(
                "CPython random state gauss_next must be finite or null"
            )
        gauss_next = float(gauss_next)
        if not isfinite(gauss_next):
            raise CudaGenerationBirthStateError(
                "CPython random state gauss_next must be finite or null"
            )
    raw_draw_value = payload.get("raw_draws")
    if raw_draw_value is None:
        generation = _strict_int(
            metadata.get("generation", 0),
            name="metadata generation",
        )
        if generation > 0:
            raise CudaGenerationBirthStateError(
                "nonzero-generation CPython random state omitted raw draw count"
            )
        raw_draw_value = 0
    raw_draws = _strict_int(
        raw_draw_value,
        name="CPython random state raw draw count",
    )
    return (version, tuple((*packed_words, state_index)), gauss_next), raw_draws


def _validate_policy_metadata(metadata: Mapping[str, Any]) -> tuple[bool, str, float, float]:
    enabled = _strict_bool(metadata, "policy_weight_mutation_enable")
    mode_value = metadata.get("policy_weight_mutation_mode")
    if type(mode_value) is not str:
        raise CudaGenerationBirthStateError(
            "metadata.policy_weight_mutation_mode must be a string"
        )
    mode = mode_value.lower()
    if mode in {"", "gaussian"}:
        mode = "dense"
    if mode not in {"dense", "sparse"}:
        raise CudaGenerationBirthStateError(
            "metadata.policy_weight_mutation_mode must be dense or sparse"
        )
    rate = _finite_float(metadata, "policy_weight_mutation_rate")
    std = _finite_float(metadata, "policy_weight_mutation_std")
    if not 0.0 <= rate <= 1.0:
        raise CudaGenerationBirthStateError(
            "metadata.policy_weight_mutation_rate must be in [0, 1]"
        )
    if std < 0.0:
        raise CudaGenerationBirthStateError(
            "metadata.policy_weight_mutation_std must be nonnegative"
        )
    return enabled, mode, rate, std


def _validation_failure_tensor(
    *,
    resident: Mapping[str, torch.Tensor],
    device: torch.device,
    slot_capacity: int,
    source_free_queue: torch.Tensor,
    compact_agent_ids: torch.Tensor,
    compact_slots: torch.Tensor,
    policy_state: Mapping[str, torch.Tensor],
    pending_slots: tuple[int, ...],
) -> torch.Tensor:
    failures: list[torch.Tensor] = []
    alive_mask = resident["alive_mask"]
    slot_to_agent_id = resident["slot_to_agent_id"]

    failures.append(
        source_free_queue.lt(0)
        .logical_or(source_free_queue.ge(slot_capacity))
        .any()
    )
    if source_free_queue.numel() > 1:
        ordered_queue = torch.sort(source_free_queue).values
        failures.append(ordered_queue[1:].eq(ordered_queue[:-1]).any())
    if source_free_queue.numel():
        queue_indices = source_free_queue.clamp(0, slot_capacity - 1).to(torch.int64)
        failures.append(alive_mask.index_select(0, queue_indices).any())
        failures.append(slot_to_agent_id.index_select(0, queue_indices).ge(0).any())
    if pending_slots and source_free_queue.numel():
        pending = torch.tensor(pending_slots, dtype=torch.int64, device=device)
        failures.append(source_free_queue.unsqueeze(1).eq(pending.unsqueeze(0)).any())

    failures.append(
        alive_mask.logical_and(
            slot_to_agent_id.lt(0).logical_or(slot_to_agent_id.gt(MAX_COMPACT_AGENT_ID))
        ).any()
    )
    failures.append(alive_mask.logical_not().logical_and(slot_to_agent_id.ne(-1)).any())
    failures.append(alive_mask.sum(dtype=torch.int64).ne(compact_agent_ids.numel()))
    if compact_agent_ids.numel():
        failures.append(
            compact_agent_ids.lt(0)
            .logical_or(compact_agent_ids.gt(MAX_COMPACT_AGENT_ID))
            .any()
        )
        failures.append(
            compact_slots.lt(0).logical_or(compact_slots.ge(slot_capacity)).any()
        )
        if compact_agent_ids.numel() > 1:
            ordered_slots = torch.sort(compact_slots).values
            failures.append(
                compact_agent_ids[1:].le(compact_agent_ids[:-1]).any()
            )
            failures.append(ordered_slots[1:].eq(ordered_slots[:-1]).any())
        safe_slots = compact_slots.to(torch.int64).clamp(0, slot_capacity - 1)
        failures.append(alive_mask.index_select(0, safe_slots).logical_not().any())
        failures.append(
            slot_to_agent_id.index_select(0, safe_slots).ne(compact_agent_ids).any()
        )

    failures.extend(
        (
            resident["event_tape_status"].ne(0).any(),
            resident["policy_pcg64_state_words"]
            .ne(resident["birth_spawn_env_pcg64_state_words"])
            .any(),
            resident["policy_pcg64_buffer_words"]
            .ne(resident["birth_spawn_env_pcg64_buffer_words"])
            .any(),
            policy_state["rng_manager_mutation_torch_cpu_state"]
            .ne(resident["rng_manager_mutation_torch_cpu_state"])
            .any(),
        )
    )
    return torch.stack(failures).any()


def _bounded_compatibility_agent_id_to_slot(
    *,
    compact_agent_ids: torch.Tensor,
    compact_slots: torch.Tensor,
    slot_capacity: int,
    device: torch.device,
) -> torch.Tensor:
    shadow = torch.full((slot_capacity,), -1, dtype=torch.int32, device=device)
    low_id_mask = (
        compact_agent_ids.ge(0)
        .logical_and(compact_agent_ids.lt(slot_capacity))
        .logical_and(compact_slots.ge(0))
        .logical_and(compact_slots.lt(slot_capacity))
    )
    shadow.index_copy_(
        0,
        compact_agent_ids[low_id_mask],
        compact_slots[low_id_mask],
    )
    return shadow


def prepare_generation_birth_state_cuda(
    boundary: CudaBoundaryImport,
    original_cpu_owner_projection: Any,
    *,
    steps: int,
    max_births_per_tick: int,
    turnover_refill_capacity: int = 0,
    resident_policy_episode_capacity: int | None = None,
    ordinary_policy_episode_capacity: int | None = None,
    scalar_gauss_raw_capacity: int | None = None,
    compact_policy_mutation_tape: bool = False,
    policy_birth_segment_refill_enabled: bool = False,
    stage_cpython_gaussian_at_raw_draw_base: bool = False,
    policy_tape_staging_device: torch.device | str | None = None,
    allow_host_authoritative_mlp_pack: bool = False,
    rng_semantics_version: int = 1,
) -> CudaGenerationBirthState:
    """Prepare all ordinary-birth state once, before resident tick execution."""
    if not isinstance(boundary, CudaBoundaryImport):
        raise CudaGenerationBirthStateError("boundary must be a CudaBoundaryImport")
    if boundary.schema_version != BOUNDARY_IMPORT_SCHEMA_VERSION:
        raise CudaGenerationBirthStateError("unsupported CUDA boundary import schema")
    if type(allow_host_authoritative_mlp_pack) is not bool:
        raise CudaGenerationBirthStateError(
            "allow_host_authoritative_mlp_pack must be an exact bool"
        )
    if type(rng_semantics_version) is not int or rng_semantics_version not in {1, 2}:
        raise CudaGenerationBirthStateError(
            "rng_semantics_version must be the exact integer 1 or 2"
        )
    if boundary.receipt.simulation_executed or boundary.metadata.get(
        "cuda_boundary_simulation_executed"
    ) is not False:
        raise CudaGenerationBirthStateError(
            "birth state preparation requires an unused generation boundary"
        )

    generation_steps = _strict_int(steps, name="steps", minimum=1)
    per_tick_limit = _strict_int(
        max_births_per_tick,
        name="max_births_per_tick",
    )
    requested_turnover_capacity = _strict_int(
        turnover_refill_capacity,
        name="turnover_refill_capacity",
    )
    if type(policy_birth_segment_refill_enabled) is not bool:
        raise CudaGenerationBirthStateError(
            "policy_birth_segment_refill_enabled must be an exact bool"
        )
    metadata = boundary.metadata
    tensors = boundary.tensors
    world = metadata.get("world")
    if world not in {"flat", "patch"}:
        raise CudaGenerationBirthStateError("CUDA boundary world must be flat or patch")
    if world == "patch" and (
        metadata.get("native_runtime_patch_profile") != "P2"
        or metadata.get("native_runtime_patch_profile_scope") != "full_profile"
    ):
        raise CudaGenerationBirthStateError(
            "ordinary CUDA birth preparation supports only the Patch P2 full profile"
        )
    capacity = _strict_int(metadata.get("capacity"), name="metadata.capacity", minimum=1)
    slot_to_agent_id = tensors.get("slot_to_agent_id")
    if not isinstance(slot_to_agent_id, torch.Tensor) or slot_to_agent_id.ndim != 1:
        raise CudaGenerationBirthStateError(
            "CUDA boundary slot_to_agent_id must be rank 1"
        )
    if int(slot_to_agent_id.numel()) != capacity:
        raise CudaGenerationBirthStateError(
            "metadata.capacity disagrees with slot_to_agent_id"
        )
    device = slot_to_agent_id.device
    if device.type not in {"cuda", "cpu"}:
        raise CudaGenerationBirthStateError(
            "CUDA boundary state must be CUDA-resident or explicitly CPU-staged"
        )
    try:
        tape_device = (
            device
            if policy_tape_staging_device is None
            else torch.device(policy_tape_staging_device)
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise CudaGenerationBirthStateError(
            "policy_tape_staging_device must resolve to CPU or CUDA"
        ) from exc
    if tape_device.type not in {"cpu", "cuda"}:
        raise CudaGenerationBirthStateError(
            "policy_tape_staging_device must resolve to CPU or CUDA"
        )

    default_ordinary_episode_capacity = generation_steps * min(
        per_tick_limit,
        capacity,
    )
    if ordinary_policy_episode_capacity is None:
        ordinary_episode_capacity = default_ordinary_episode_capacity
    else:
        ordinary_episode_capacity = _strict_int(
            ordinary_policy_episode_capacity,
            name="ordinary_policy_episode_capacity",
        )
        if ordinary_episode_capacity > default_ordinary_episode_capacity:
            raise CudaGenerationBirthStateError(
                "ordinary_policy_episode_capacity exceeds the declared ordinary horizon"
            )
    tape_free_candidate = False
    if resident_policy_episode_capacity is None:
        turnover_capacity = min(requested_turnover_capacity, capacity)
        episode_capacity = ordinary_episode_capacity + turnover_capacity
    else:
        turnover_capacity = requested_turnover_capacity
        episode_capacity = _strict_int(
            resident_policy_episode_capacity,
            name="resident_policy_episode_capacity",
        )
        minimum_resident_capacity = ordinary_episode_capacity + turnover_capacity
        segmented_policy_tape = policy_birth_segment_refill_enabled
        tape_free_candidate = (
            compact_policy_mutation_tape
            and ordinary_policy_episode_capacity in (0, 1)
            and resident_policy_episode_capacity
            == ordinary_policy_episode_capacity
        )
        if segmented_policy_tape and not compact_policy_mutation_tape:
            raise CudaGenerationBirthStateError(
                "policy_birth_segment_refill requires replay-backed compact "
                "disabled-mutation policy tape"
            )
        if segmented_policy_tape and episode_capacity < max(
            per_tick_limit,
            turnover_capacity,
        ):
            raise CudaGenerationBirthStateError(
                "policy_birth_segment_refill capacity must cover one ordinary "
                "tick and one turnover refill segment"
            )
        if (
            not tape_free_candidate
            and not segmented_policy_tape
            and episode_capacity < minimum_resident_capacity
        ):
            raise CudaGenerationBirthStateError(
                "resident_policy_episode_capacity is below the declared ordinary "
                "and turnover horizon"
            )
    full_horizon_episode_capacity = ordinary_episode_capacity + turnover_capacity
    if tape_free_candidate:
        full_horizon_episode_capacity = episode_capacity
    if episode_capacity > MAX_FIXED_MLP_POLICY_TAPE_EPISODES:
        raise CudaGenerationBirthStateError(
            "birth episode capacity exceeds the bounded fixed-MLP policy tape"
        )
    gauss_capacity = (
        full_horizon_episode_capacity * CPYTHON_GAUSS_RAW_CAPACITY_PER_BIRTH
    )
    if gauss_capacity > MAX_CPYTHON_GAUSS_RAW_CAPACITY:
        raise CudaGenerationBirthStateError(
            "birth episode capacity exceeds the bounded exact CPython Gaussian tape"
        )
    if scalar_gauss_raw_capacity is None:
        prepared_gauss_capacity = gauss_capacity
    else:
        prepared_gauss_capacity = _strict_int(
            scalar_gauss_raw_capacity,
            name="scalar_gauss_raw_capacity",
        )
        if prepared_gauss_capacity > gauss_capacity:
            raise CudaGenerationBirthStateError(
                "scalar_gauss_raw_capacity exceeds the declared birth episode horizon"
            )
    if type(stage_cpython_gaussian_at_raw_draw_base) is not bool:
        raise CudaGenerationBirthStateError(
            "stage_cpython_gaussian_at_raw_draw_base must be an exact bool"
        )

    if _strict_bool(metadata, "genome_c1_enable") or _strict_bool(
        metadata, "genome_c2_enable"
    ):
        raise CudaGenerationBirthStateError(
            "ordinary CUDA birth preparation does not support C1/C2 structural genetics"
        )
    env_policy_alias = _strict_bool(metadata, "native_runtime_env_policy_numpy_alias")
    if not env_policy_alias:
        raise CudaGenerationBirthStateError(
            "ordinary CUDA birth preparation requires the accepted env/policy RNG alias"
        )

    source_schema, source_metadata, source_tensors = _projection_parts(
        original_cpu_owner_projection
    )
    if source_schema != boundary.receipt.source_schema_version:
        raise CudaGenerationBirthStateError(
            "original CPU owner projection schema disagrees with the CUDA boundary"
        )
    if source_schema == CHECKPOINT_PROJECTION_SCHEMA_VERSION:
        source_digest = projection_digest(
            {"metadata": source_metadata, "tensors": source_tensors}
        )
    else:
        source_digest = source_metadata.get("import_digest")
    if (
        type(source_digest) is not str
        or source_digest != boundary.receipt.source_projection_digest
    ):
        raise CudaGenerationBirthStateError(
            "original CPU owner projection digest disagrees with the CUDA boundary"
        )
    if metadata.get("cuda_boundary_source_projection_digest") != source_digest:
        raise CudaGenerationBirthStateError(
            "CUDA boundary source projection digest is missing or contradictory"
        )
    if (
        source_schema != CHECKPOINT_PROJECTION_SCHEMA_VERSION
        and metadata.get("cuda_boundary_source_import_digest") != source_digest
    ):
        raise CudaGenerationBirthStateError(
            "CUDA boundary source import digest is missing or contradictory"
        )
    cpu_torch_state = source_tensors.get("rng_manager_mutation_torch_cpu_state")
    if not isinstance(cpu_torch_state, torch.Tensor):
        raise CudaGenerationBirthStateError(
            "original CPU owner projection is missing mutation Torch RNG state"
        )
    if (
        cpu_torch_state.device.type != "cpu"
        or cpu_torch_state.dtype != torch.uint8
        or cpu_torch_state.ndim != 1
        or int(cpu_torch_state.numel()) != TORCH_CPU_RNG_STATE_FULL_BYTES_V1
        or not cpu_torch_state.is_contiguous()
        or cpu_torch_state.requires_grad
    ):
        raise CudaGenerationBirthStateError(
            "original mutation Torch RNG state must be contiguous rank-1 CPU uint8"
        )

    required_names = set(IDENTITY_STATE_FIELDS)
    required_names.difference_update(
        {"agent_id_to_slot", "slot_free_queue_count", "population_next_id"}
    )
    required_names.update({"agent_id_to_slot_agent_ids", "agent_id_to_slot_slots"})
    required_names.update(GENOME_INT32_FIELDS)
    required_names.update(GENOME_FLOAT32_FIELDS)
    required_names.update(
        {
            "genome_pred_lr",
            "policy_mem_dim",
            "policy_use_gru",
            "drives",
            "birth_reset_shock",
            "chemistry",
            "mlp_pack",
            "hmem",
            "population_pcg64_state_words",
            "population_pcg64_buffer_words",
            "policy_pcg64_state_words",
            "policy_pcg64_buffer_words",
            "runner_pcg64_state_words",
            "runner_pcg64_buffer_words",
            "birth_spawn_env_pcg64_state_words",
            "birth_spawn_env_pcg64_buffer_words",
            "rng_manager_mutation_torch_cpu_state",
            "event_tape_active_count",
            "event_tape_status",
            *EVENT_COLUMN_NAMES,
        }
    )
    host_mlp_pack = None
    if allow_host_authoritative_mlp_pack:
        host_mlp_pack = boundary.host_authoritative_tensors.get("mlp_pack")
        if (
            not isinstance(host_mlp_pack, torch.Tensor)
            or host_mlp_pack.device.type != "cpu"
            or host_mlp_pack.is_pinned()
            or not host_mlp_pack.is_contiguous()
            or host_mlp_pack.requires_grad
        ):
            raise CudaGenerationBirthStateError(
                "HAPB birth preparation requires pageable host mlp_pack authority"
            )
        required_names.remove("mlp_pack")
    resident = {
        name: _resident_tensor(tensors, name, device=device)
        for name in required_names
    }
    if host_mlp_pack is not None:
        resident["mlp_pack"] = host_mlp_pack

    alive_mask = resident["alive_mask"]
    resident_slot_to_agent_id = resident["slot_to_agent_id"]
    if (
        alive_mask.dtype != torch.bool
        or alive_mask.ndim != 1
        or int(alive_mask.numel()) != capacity
        or resident_slot_to_agent_id.dtype != torch.int64
        or resident_slot_to_agent_id.ndim != 1
        or int(resident_slot_to_agent_id.numel()) != capacity
    ):
        raise CudaGenerationBirthStateError(
            "slot identity authority must be aligned capacity-sized int64/bool"
        )
    source_free_queue = resident["slot_free_queue"]
    if source_free_queue.dtype != torch.int64 or source_free_queue.ndim != 1:
        raise CudaGenerationBirthStateError("slot_free_queue must be rank-1 int64")
    if int(source_free_queue.numel()) > capacity:
        raise CudaGenerationBirthStateError("slot_free_queue exceeds slot capacity")
    compact_agent_ids = resident["agent_id_to_slot_agent_ids"]
    compact_slots = resident["agent_id_to_slot_slots"]
    if (
        compact_agent_ids.dtype != torch.int64
        or compact_agent_ids.ndim != 1
        or compact_slots.dtype != torch.int32
        or compact_slots.ndim != 1
        or compact_agent_ids.shape != compact_slots.shape
    ):
        raise CudaGenerationBirthStateError(
            "compact agent-ID/slot authority must be aligned rank-1 int64/int32"
        )

    population_next_id = _strict_int(
        metadata.get("population_next_id"),
        name="metadata.population_next_id",
        minimum=1,
    )
    if population_next_id > MAX_COMPACT_AGENT_ID + 1:
        raise CudaGenerationBirthStateError(
            "metadata.population_next_id exceeds the int32 grid identity range"
        )
    fixed_free_queue = torch.full(
        (capacity,), -1, dtype=torch.int64, device=device
    )
    fixed_free_queue[: source_free_queue.numel()].copy_(source_free_queue)
    fixed_id_to_slot = _bounded_compatibility_agent_id_to_slot(
        compact_agent_ids=compact_agent_ids,
        compact_slots=compact_slots,
        slot_capacity=capacity,
        device=device,
    )

    identity = {
        name: resident[name]
        for name in IDENTITY_STATE_FIELDS
        if name
        not in {
            "agent_id_to_slot",
            "slot_free_queue",
            "slot_free_queue_count",
            "population_next_id",
        }
    }
    identity.update(
        {
            "agent_id_to_slot": fixed_id_to_slot,
            "slot_free_queue": fixed_free_queue,
            "slot_free_queue_count": torch.tensor(
                [source_free_queue.numel()], dtype=torch.int64, device=device
            ),
            "population_next_id": torch.tensor(
                [population_next_id], dtype=torch.int64, device=device
            ),
        }
    )

    genome = {
        name: resident[name]
        for name in (*GENOME_INT32_FIELDS, *GENOME_FLOAT32_FIELDS)
    }
    genome["repro_threshold"] = identity["repro_threshold"]

    try:
        python_random_state, python_random_raw_draws = _python_random_state(metadata)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        scalar_rng = pack_cpython_random_state_cuda(
            python_random_state,
            device=device,
            semantic_draws=_stream_index(metadata, "mutation", "py"),
            gauss_tape_raw_capacity=prepared_gauss_capacity,
            raw_draws=python_random_raw_draws,
            raw_draw_base=(
                python_random_raw_draws
                if stage_cpython_gaussian_at_raw_draw_base
                else 0
            ),
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        if isinstance(exc, CudaGenerationBirthStateError):
            raise
        raise CudaGenerationBirthStateError(
            "failed to prepare exact CPython scalar-mutation state: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    policy_enabled, policy_mode, policy_rate, policy_std = _validate_policy_metadata(metadata)
    if type(compact_policy_mutation_tape) is not bool:
        raise CudaGenerationBirthStateError(
            "compact_policy_mutation_tape must be an exact bool"
        )
    if compact_policy_mutation_tape and policy_enabled:
        raise CudaGenerationBirthStateError(
            "compact_policy_mutation_tape requires disabled policy-weight mutation"
        )
    rngv2_formula_policy = rng_semantics_version == 2
    if rngv2_formula_policy:
        if policy_enabled:
            raise CudaGenerationBirthStateError(
                "RNG V2 fixed-MLP policy mutation is not implemented; future "
                "policy tapes are forbidden"
            )
        if prepared_gauss_capacity != 0:
            raise CudaGenerationBirthStateError(
                "RNG V2 requires zero CPython Gaussian future-tape capacity"
            )
        if episode_capacity != 0 or ordinary_episode_capacity != 0:
            raise CudaGenerationBirthStateError(
                "RNG V2 requires zero fixed-MLP policy future-tape capacity"
            )
        if policy_birth_segment_refill_enabled:
            raise CudaGenerationBirthStateError(
                "RNG V2 forbids policy future-tape segment refill"
            )
        if tape_device != device:
            raise CudaGenerationBirthStateError(
                "RNG V2 policy state must be created on the resident device"
            )
    mlp_pack = resident["mlp_pack"]
    if mlp_pack.ndim != 2 or int(mlp_pack.shape[0]) != capacity:
        raise CudaGenerationBirthStateError("mlp_pack must be slot-capacity rank 2")
    if rngv2_formula_policy:
        policy = {
            "mlp_pack": mlp_pack,
            "hmem": resident["hmem"],
            "rng_manager_mutation_torch_cpu_state": cpu_torch_state.clone().to(
                device=device
            ),
            "policy_birth_delta_tape": torch.empty(
                (0, 1), dtype=torch.float32, device=device
            ),
            "policy_birth_mutation_mask_tape": torch.empty(
                (0, 1), dtype=torch.bool, device=device
            ),
            "policy_birth_crossover_mask_packed_tape": torch.empty(
                (0, (int(mlp_pack.shape[1]) + 7) // 8),
                dtype=torch.uint8,
                device=device,
            ),
            "policy_birth_post_rng_state_packed_tape": torch.empty(
                (0, 0), dtype=torch.uint8, device=device
            ),
            "policy_birth_semantic_draws_tape": torch.empty(
                (0,), dtype=torch.int64, device=device
            ),
            "policy_birth_tape_cursor": torch.zeros(
                (1,), dtype=torch.int64, device=device
            ),
            "mutation_torch_cpu_semantic_draws": torch.tensor(
                [_stream_index(metadata, "mutation", "torch_cpu")],
                dtype=torch.int64,
                device=device,
            ),
        }
        policy = _runtime_policy_payload(policy, include_packed_crossover=True)
        turnover_policy = policy
    elif resident_policy_episode_capacity is not None:
        try:
            shared_policy_tape = prepare_fixed_mlp_turnover_child_policy_tape(
                cpu_torch_state,
                episode_capacity=episode_capacity,
                row_width=int(mlp_pack.shape[1]),
                device=tape_device,
                policy_weight_mutation_enable=policy_enabled,
                policy_weight_mutation_mode=policy_mode,
                policy_weight_mutation_rate=policy_rate,
                policy_weight_mutation_std=policy_std,
                semantic_draws=_stream_index(metadata, "mutation", "torch_cpu"),
                compact_no_mutation_tape=compact_policy_mutation_tape,
                include_post_state_tape=not compact_policy_mutation_tape,
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise CudaGenerationBirthStateError(
                "failed to prepare exact shared resident fixed-MLP policy tape"
            ) from exc
        policy = {
            "mlp_pack": mlp_pack,
            "hmem": resident["hmem"],
            **shared_policy_tape,
        }
        policy["rng_manager_mutation_torch_cpu_state"] = policy[
            "rng_manager_mutation_torch_cpu_state"
        ].to(device=device)
        policy = _runtime_policy_payload(policy, include_packed_crossover=True)
        turnover_policy = policy
    else:
        try:
            policy_tape = prepare_fixed_mlp_child_policy_tape(
                cpu_torch_state,
                episode_capacity=ordinary_episode_capacity,
                row_width=int(mlp_pack.shape[1]),
                device=tape_device,
                policy_weight_mutation_enable=policy_enabled,
                policy_weight_mutation_mode=policy_mode,
                policy_weight_mutation_rate=policy_rate,
                policy_weight_mutation_std=policy_std,
                semantic_draws=_stream_index(metadata, "mutation", "torch_cpu"),
                compact_no_mutation_tape=compact_policy_mutation_tape,
                include_post_state_tape=not compact_policy_mutation_tape,
                include_crossover_mask=True,
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise CudaGenerationBirthStateError(
                "failed to prepare exact fixed-MLP child-policy tape"
            ) from exc
        policy = {
            "mlp_pack": mlp_pack,
            "hmem": resident["hmem"],
            **policy_tape,
        }
        policy["rng_manager_mutation_torch_cpu_state"] = policy[
            "rng_manager_mutation_torch_cpu_state"
        ].to(device=device)
        policy = _runtime_policy_payload(policy, include_packed_crossover=True)
        try:
            turnover_policy_tape = prepare_fixed_mlp_turnover_child_policy_tape(
                cpu_torch_state,
                episode_capacity=turnover_capacity,
                row_width=int(mlp_pack.shape[1]),
                device=tape_device,
                policy_weight_mutation_enable=policy_enabled,
                policy_weight_mutation_mode=policy_mode,
                policy_weight_mutation_rate=policy_rate,
                policy_weight_mutation_std=policy_std,
                semantic_draws=_stream_index(metadata, "mutation", "torch_cpu"),
                compact_no_mutation_tape=compact_policy_mutation_tape,
                include_post_state_tape=not compact_policy_mutation_tape,
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise CudaGenerationBirthStateError(
                "failed to prepare exact fixed-MLP turnover child-policy tape"
            ) from exc
        turnover_policy = {
            "mlp_pack": mlp_pack,
            "hmem": resident["hmem"],
            **turnover_policy_tape,
        }
        # Both policy paths commit the same authoritative Torch RNG state/counter.
        turnover_policy["rng_manager_mutation_torch_cpu_state"] = policy[
            "rng_manager_mutation_torch_cpu_state"
        ]
        turnover_policy["mutation_torch_cpu_semantic_draws"] = policy[
            "mutation_torch_cpu_semantic_draws"
        ]
        turnover_policy = _runtime_policy_payload(
            turnover_policy, include_packed_crossover=True
        )
    drive = {
        "drives": resident["drives"],
        "birth_reset_shock": resident["birth_reset_shock"],
        "pcg64_state_words": resident["population_pcg64_state_words"],
        "pcg64_buffer_words": resident["population_pcg64_buffer_words"],
        "mutation_np_semantic_draws": torch.tensor(
            [_stream_index(metadata, "mutation", "np")],
            dtype=torch.int64,
            device=device,
        ),
        "chemistry": resident["chemistry"],
    }
    planner_state_words = resident["policy_pcg64_state_words"]
    planner_buffer_words = resident["policy_pcg64_buffer_words"]
    policy_np_semantic_draws = torch.tensor(
        [_stream_index(metadata, "policy", "np")],
        dtype=torch.int64,
        device=device,
    )
    metrics_np_semantic_draws = torch.tensor(
        [_stream_index(metadata, "metrics", "np")],
        dtype=torch.int64,
        device=device,
    )

    if set(identity) != set(IDENTITY_STATE_FIELDS):
        raise CudaGenerationBirthStateError("prepared identity state is incomplete")
    if set(scalar_rng) != set(SCALAR_RNG_STATE_FIELDS):
        raise CudaGenerationBirthStateError("prepared scalar RNG state is incomplete")
    if set(genome) != set((*GENOME_INT32_FIELDS, *GENOME_FLOAT32_FIELDS)):
        raise CudaGenerationBirthStateError("prepared genome state is incomplete")
    if set(drive) != set(DRIVE_STATE_FIELDS):
        raise CudaGenerationBirthStateError("prepared drive state is incomplete")
    policy_field_authority = (
        POLICY_STATE_FIELDS
        if PACKED_CROSSOVER_POLICY_FIELD in policy
        else ORDINARY_POLICY_STATE_FIELDS
    )
    if set(policy) != set(policy_field_authority):
        raise CudaGenerationBirthStateError("prepared policy state is incomplete")
    if set(turnover_policy) != set(POLICY_STATE_FIELDS):
        raise CudaGenerationBirthStateError(
            "prepared turnover policy state is incomplete"
        )

    failed = _validation_failure_tensor(
        resident=resident,
        device=device,
        slot_capacity=capacity,
        source_free_queue=source_free_queue,
        compact_agent_ids=compact_agent_ids,
        compact_slots=compact_slots,
        policy_state=policy,
        pending_slots=boundary.pending_generation_free_slots,
    )
    if bool(failed.item()):
        raise CudaGenerationBirthStateError(
            "CUDA boundary birth queues, aliases, event status, or RNG source contradict"
        )

    receipt = CudaGenerationBirthStateReceipt(
        schema_version=GENERATION_BIRTH_STATE_SCHEMA_VERSION,
        source_projection_digest=source_digest,
        target_device=str(device),
        world=str(world),
        steps=generation_steps,
        max_births_per_tick=per_tick_limit,
        slot_capacity=capacity,
        birth_episode_capacity=(0 if rngv2_formula_policy else episode_capacity),
        turnover_refill_capacity=turnover_capacity,
        agent_id_to_slot_capacity=capacity,
        cpython_gauss_raw_capacity=prepared_gauss_capacity,
        fixed_mlp_policy_tape_episodes=(
            0 if rngv2_formula_policy else episode_capacity
        ),
        fixed_mlp_policy_full_horizon_episodes=(
            0 if rngv2_formula_policy else full_horizon_episode_capacity
        ),
        policy_birth_segment_refill_enabled=policy_birth_segment_refill_enabled,
        cpu_torch_rng_source_reads=1,
        generation_boundary_device_validation_syncs=1,
        torch_rng_post_state_encoding=TORCH_CPU_RNG_STATE_ENCODING_V1,
        torch_rng_post_state_logical_row_bytes=TORCH_CPU_RNG_STATE_FULL_BYTES_V1,
        torch_rng_post_state_packed_row_bytes=TORCH_CPU_RNG_STATE_PACKED_BYTES_V1,
        torch_rng_post_state_dense_allocation_materialized=False,
        torch_rng_post_state_resident_rows_materialized=(
            not compact_policy_mutation_tape and not rngv2_formula_policy
        ),
        torch_rng_post_state_abi=torch_cpu_rng_state_abi_receipt_v1(),
    )
    return CudaGenerationBirthState(
        schema_version=GENERATION_BIRTH_STATE_SCHEMA_VERSION,
        identity_state=identity,
        scalar_rng_state=scalar_rng,
        genome_state=genome,
        policy_mem_dim=resident["policy_mem_dim"],
        policy_use_gru=resident["policy_use_gru"],
        drive_state=drive,
        policy_state=policy,
        turnover_policy_state=turnover_policy,
        planner_pcg64_state_words=planner_state_words,
        planner_pcg64_buffer_words=planner_buffer_words,
        policy_np_semantic_draws=policy_np_semantic_draws,
        runner_pcg64_state_words=resident["runner_pcg64_state_words"],
        runner_pcg64_buffer_words=resident["runner_pcg64_buffer_words"],
        metrics_np_semantic_draws=metrics_np_semantic_draws,
        event_tape_columns=tuple(resident[name] for name in EVENT_COLUMN_NAMES),
        event_tape_active_count=resident["event_tape_active_count"],
        event_tape_status=resident["event_tape_status"],
        receipt=receipt,
    )


__all__ = [
    "CPYTHON_GAUSS_RAW_CAPACITY_PER_BIRTH",
    "GENERATION_BIRTH_STATE_CAPABILITIES",
    "GENERATION_BIRTH_STATE_SCHEMA_VERSION",
    "CudaGenerationBirthState",
    "CudaGenerationBirthStateError",
    "CudaGenerationBirthStateReceipt",
    "prepare_generation_birth_state_cuda",
]
