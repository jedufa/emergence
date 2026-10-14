"""Declared generation-boundary CUDA checkpoint projection transport."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
from typing import Any, Final, Mapping, Sequence

import numpy as np
import torch

from native_runtime.checkpoint_digests import (
    checkpoint_source_projection_authority_manifest,
)
from native_runtime.interchange_projection import validate_native_checkpoint_projection
from perf.cuda_architecture_b_boundary import (
    CHECKPOINT_PROJECTION_SCHEMA_VERSION,
    EVENT_COLUMN_NAMES,
)
from perf.cuda_architecture_b_modular_state_abi import (
    MODULAR_STORAGE_METADATA_KEY,
    MODULAR_STORAGE_PROJECTION_PREFIX,
    ModularStorageError,
    PackedModularStorageV1,
    modular_storage_generation_boundary_projection_payload,
    modular_storage_lifecycle_projection_payload,
    modular_storage_from_projection,
)
from perf.cuda_architecture_b_disabled_prediction_state_v1 import (
    DISABLED_PREDICTION_STATE_FIELDS_V1,
    DisabledPredictionSettlementError,
    materialize_disabled_prediction_state_v1,
)
from engine.perf.recurrent_native.modular_recurrent_mutation_receipt_v1 import (
    ModularRecurrentMutationReceiptV1,
)
from engine.perf.recurrent_native.modular_recurrent_mutation_sequence_v1 import (
    ModularRecurrentMutationSequenceV1,
)
from perf.cxxbackbone10_resident_state_owner import (
    EED_CURSOR_ENV_ARRAYS,
    _rebuild_patch_checkpoint_derived_arrays,
    _with_checkpoint_contract_metadata,
    projection_digest,
)


CANONICAL_DISABLED_TENSOR_FIELDS: Final[frozenset[str]] = frozenset(
    {"biome_grid", "gru_pack", "hazard_rate_map", "hmem", "policy_use_gru"}
)
_GRU_RESIDENT_TENSOR_FIELDS: Final[frozenset[str]] = frozenset(
    {"gru_pack", "hmem", "policy_use_gru"}
)
REFRESH_DERIVED_TENSOR_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "agent_id_to_slot_agent_ids",
        "agent_id_to_slot_slots",
        "agent_ids_by_alive_order",
        "patch_density",
        "patch_visibility",
    }
)
OPAQUE_CPU_RNG_TENSOR_FIELDS: Final[frozenset[str]] = frozenset(
    {"rng_manager_mutation_torch_cpu_state"}
)
_SOURCE_CANONICAL_TENSOR_FIELDS: Final[frozenset[str]] = frozenset(
    CANONICAL_DISABLED_TENSOR_FIELDS
    | REFRESH_DERIVED_TENSOR_FIELDS
    | OPAQUE_CPU_RNG_TENSOR_FIELDS
)

PHASE_E_PASSIVE_NO_CHURN_GUARD_SCHEMA_VERSION: Final[str] = (
    "eiel_phase_e_passive_modular_no_churn_guard_v1"
)
PHASE_E_PASSIVE_NO_CHURN_GUARD_ERROR: Final[str] = (
    "PHASE_E_PASSIVE_MODULAR_NO_CHURN_GUARD"
)
_PHASE_E_IDENTITY_TENSOR_FIELDS: Final[tuple[str, ...]] = (
    "alive_mask",
    "slot_to_agent_id",
    "slot_tokens",
)

POST_GENERATION_REQUIRED_BOOKKEEPING_FIELDS: Final[tuple[str, ...]] = (
    "applied_actions",
    "moved",
    "gathered",
    "policy_refill_calls",
    "policy_refill_rows",
)

_POST_GENERATION_STATUS_FIELDS: Final[tuple[str, ...]] = (
    "lifecycle_status_by_tick",
    "birth_status_by_tick",
    "event_tape_status",
    "event_append_status_by_tick",
    "turnover_status",
)

# Canonical names are on the left; resident birth-bank names are on the right.
# The order below is also the explicit authority precedence used by the adapter.
_BIRTH_IDENTITY_CANONICAL_SOURCES: Final[dict[str, str]] = {
    name: name
    for name in (
        "slot_to_agent_id",
        "slot_tokens",
        "alive_mask",
        "pos_y",
        "pos_x",
        "agent_grid",
        "energy",
        "health",
        "injury",
        "scar",
        "age",
        "repro_cooldown",
        "repro_threshold",
        "offspring_count",
        "lineage_founder_id",
        "lineage_depth",
        "lineage_birth_generation",
        "lineage_birth_tick",
        "patch_habitat_lineage_id",
        "policy_energy_observation",
        "survival_steps",
        "rest_streak",
        "action_cache_token",
        "action_cache_cursor",
        "action_cache_next",
        "last_action",
        "last_last_action",
        "action_switches",
        "action_oscillations",
        "episode0_switches",
        "episode0_oscillations",
    )
}
_BIRTH_GENOME_CANONICAL_SOURCES: Final[dict[str, str]] = {
    name: name
    for name in (
        "genome_hidden_size",
        "genome_memory_dim",
        "genome_sensor_channel_mask",
        "genome_sensor_radius",
        "genome_introspection_channel_mask",
        "genome_self_regulation_control_mask",
        "genome_self_modification_target_mask",
        "genome_directed_inheritance_channel_mask",
        "genome_exploration_bias",
        "genome_aggression_bias",
        "genome_mutation_rate",
        "genome_reproduction_threshold",
        "genome_action_temperature",
        "genome_drive_inertia",
        "genome_drive_noise",
        "genome_drive_logit_gain",
        "genome_drive_sp_hunger",
        "genome_drive_sp_pain",
        "genome_drive_sp_fatigue",
        "genome_drive_sp_fear",
        "genome_exp_prior_strength",
        "genome_exp_prior_noise",
        "genome_drive_birth_noise",
        "gru_upkeep_cost",
        "policy_attack_bias",
        "policy_action_temperature",
        "policy_drive_logit_gain",
        "drive_inertia",
        "drive_noise",
    )
}
_BIRTH_GENOME_CANONICAL_SOURCES["genome_pred_lr"] = "pred_lr"
_BIRTH_DRIVE_CANONICAL_SOURCES: Final[dict[str, str]] = {
    "drives": "drives",
    "birth_reset_shock": "birth_reset_shock",
    "chemistry": "chemistry",
}
_BIRTH_POLICY_CANONICAL_SOURCES: Final[dict[str, str]] = {
    "mlp_pack": "mlp_pack",
    "rng_manager_mutation_torch_cpu_state": (
        "rng_manager_mutation_torch_cpu_state"
    ),
}


CUDA_CHECKPOINT_CAPABILITIES = {
    "schema_version": "eiel_cuda_architecture_b_checkpoint_capabilities_v1",
    "generation_boundary_only": True,
    "trajectory_host_transfer": False,
    "projection_digest_reused": "cxxbackbone10_projection_digest",
    "exact_tensor_roundtrip": True,
    "post_turnover_canonical_export": "bounded_same_host_flat_patch_stage1",
    "completed_turnover_required": True,
    "active_event_prefix_only": True,
    "opaque_torch_cpu_rng_bytes_preserved": True,
    "full_cuda_checkpoint_route_qualified": False,
    "complete_139_tensor_route_qualified": False,
    "promotion_qualified": False,
}


@dataclass(frozen=True)
class CudaBoundaryCheckpoint:
    projection: dict[str, Any]
    projection_digest: str
    device: str
    device_to_host_operations: int
    device_to_host_bytes: int
    device_synchronizations: int
    source_projection_authority: dict[str, Any] | None = None


@dataclass(frozen=True)
class CudaCheckpointRestore:
    projection: dict[str, Any]
    source_projection_digest: str
    device: str
    host_to_device_operations: int
    host_to_device_bytes: int
    device_synchronizations: int


class CudaCheckpointProjectionError(RuntimeError):
    """Raised when post-generation resident state cannot prove a checkpoint."""


_MISSING = object()


def _phase_e_guard_fail(message: str) -> CudaCheckpointProjectionError:
    return CudaCheckpointProjectionError(
        f"{PHASE_E_PASSIVE_NO_CHURN_GUARD_ERROR}: {message}"
    )


def _phase_e_tensor_fingerprint(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, torch.Tensor):
        raise _phase_e_guard_fail(f"tensor is missing or invalid: {name}")
    tensor = value.detach().cpu().contiguous()
    try:
        payload = tensor.numpy().tobytes(order="C")
    except Exception as exc:
        raise _phase_e_guard_fail(f"tensor bytes are unavailable: {name}") from exc
    return {
        "dtype": str(tensor.dtype),
        "shape": [int(part) for part in tensor.shape],
        "nbytes": int(tensor.numel() * tensor.element_size()),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def capture_phase_e_passive_modular_no_churn_guard(
    projection: Any,
    *,
    enabled: bool,
    fixed_capacity: int,
    max_births: int,
) -> dict[str, Any] | None:
    """Freeze a passive modular boundary without promoting execution support.

    The disabled return intentionally occurs before ``projection`` is
    inspected.  The enabled milestone is one generation with no births and no
    modular lifecycle changes; it only proves exact storage preservation.
    """

    if enabled is False:
        return None
    if enabled is not True:
        raise _phase_e_guard_fail("enabled must be an exact bool")
    if type(fixed_capacity) is not int or fixed_capacity <= 0:
        raise _phase_e_guard_fail("fixed_capacity must be a positive exact int")
    if type(max_births) is not int or max_births != 0:
        raise _phase_e_guard_fail("passive modular milestone requires max_births=0")
    if not isinstance(projection, Mapping):
        raise _phase_e_guard_fail("checkpoint projection must be a mapping")
    metadata = projection.get("metadata")
    tensors = projection.get("tensors")
    if not isinstance(metadata, Mapping) or not isinstance(tensors, Mapping):
        raise _phase_e_guard_fail("checkpoint metadata/tensors must be mappings")
    if type(metadata.get("capacity")) is not int or metadata["capacity"] != fixed_capacity:
        raise _phase_e_guard_fail("checkpoint capacity is not the fixed capacity")
    try:
        packed = modular_storage_from_projection(metadata, tensors)
    except Exception as exc:
        raise _phase_e_guard_fail(f"modular checkpoint overlay is invalid: {exc}") from exc
    if packed is None:
        raise _phase_e_guard_fail("modular checkpoint overlay is absent")
    alive_mask = tensors.get("alive_mask")
    slot_to_agent_id = tensors.get("slot_to_agent_id")
    slot_tokens = tensors.get("slot_tokens")
    if not all(
        isinstance(value, torch.Tensor)
        for value in (alive_mask, slot_to_agent_id, slot_tokens)
    ):
        raise _phase_e_guard_fail("authoritative slot identity tensors are absent")
    if (
        alive_mask.dtype != torch.bool
        or slot_to_agent_id.dtype != torch.int64
        or slot_tokens.dtype != torch.int64
        or tuple(alive_mask.shape) != (fixed_capacity,)
        or tuple(slot_to_agent_id.shape) != (fixed_capacity,)
        or tuple(slot_tokens.shape) != (fixed_capacity,)
    ):
        raise _phase_e_guard_fail(
            "authoritative membership/slot-token shape or dtype changed"
        )
    normalized_tokens = torch.where(
        alive_mask,
        slot_tokens,
        torch.full_like(slot_tokens, -1),
    )
    if (
        not torch.equal(packed.tensors["agent_occupied"], alive_mask)
        or not torch.equal(packed.tensors["agent_id"], slot_to_agent_id)
        or not torch.equal(packed.tensors["agent_slot_token"], normalized_tokens)
    ):
        raise _phase_e_guard_fail(
            "modular membership/slot ownership contradicts checkpoint authority"
        )
    envelope = metadata.get(MODULAR_STORAGE_METADATA_KEY)
    if not isinstance(envelope, Mapping):
        raise _phase_e_guard_fail("modular cold sidecar/profile envelope is absent")
    modular_names = tuple(
        sorted(
            str(name)
            for name in tensors
            if str(name).startswith(MODULAR_STORAGE_PROJECTION_PREFIX)
        )
    )
    if not modular_names:
        raise _phase_e_guard_fail("modular tensor set is empty")
    identity = {
        name: _phase_e_tensor_fingerprint(tensors.get(name), name=name)
        for name in _PHASE_E_IDENTITY_TENSOR_FIELDS
    }
    modular = {
        name: _phase_e_tensor_fingerprint(tensors[name], name=name)
        for name in modular_names
    }
    return {
        "schema_version": PHASE_E_PASSIVE_NO_CHURN_GUARD_SCHEMA_VERSION,
        "fixed_capacity": fixed_capacity,
        "max_births": 0,
        "profile_id": str(packed.layout.profile_id),
        "cold_sidecar_profile_envelope": deepcopy(dict(envelope)),
        "identity_tensors": identity,
        "modular_tensors": modular,
        "live_module_execution_supported": False,
        "modular_lifecycle_supported": False,
    }


def validate_phase_e_passive_modular_no_churn_guard(
    guard: Mapping[str, Any] | None,
    post_generation_projection: Any,
    *,
    completed_generation_count: int,
    fixed_capacity: int,
    max_births: int,
) -> None:
    """Fail closed unless the passive modular boundary is byte-identical."""

    if guard is None:
        return
    if not isinstance(guard, Mapping):
        raise _phase_e_guard_fail("guard snapshot must be a mapping")
    if guard.get("schema_version") != PHASE_E_PASSIVE_NO_CHURN_GUARD_SCHEMA_VERSION:
        raise _phase_e_guard_fail("guard schema mismatch")
    if type(completed_generation_count) is not int or completed_generation_count != 1:
        raise _phase_e_guard_fail("passive modular milestone requires one generation")
    observed = capture_phase_e_passive_modular_no_churn_guard(
        post_generation_projection,
        enabled=True,
        fixed_capacity=fixed_capacity,
        max_births=max_births,
    )
    if observed != dict(guard):
        before_identity = guard.get("identity_tensors")
        after_identity = observed.get("identity_tensors") if observed is not None else None
        if before_identity != after_identity:
            raise _phase_e_guard_fail(
                "agent membership, occupancy, or authoritative slot tokens changed"
            )
        before_envelope = guard.get("cold_sidecar_profile_envelope")
        after_envelope = (
            observed.get("cold_sidecar_profile_envelope")
            if observed is not None
            else None
        )
        if before_envelope != after_envelope:
            raise _phase_e_guard_fail("modular cold sidecar or profile changed")
        raise _phase_e_guard_fail(
            "modular tensor field, shape, dtype, byte count, or digest changed"
        )


def _member(value: Any, name: str, *, default: Any = _MISSING) -> Any:
    if isinstance(value, Mapping):
        result = value.get(name, _MISSING)
    else:
        result = getattr(value, name, _MISSING)
    if result is _MISSING:
        if default is not _MISSING:
            return default
        raise CudaCheckpointProjectionError(
            f"post-generation CUDA checkpoint source is missing {name}"
        )
    return result


def _has_member(value: Any, name: str) -> bool:
    return name in value if isinstance(value, Mapping) else hasattr(value, name)


def _mapping_member(value: Any, name: str) -> Mapping[str, Any]:
    result = _member(value, name)
    if not isinstance(result, Mapping):
        raise CudaCheckpointProjectionError(
            f"post-generation CUDA checkpoint source {name} must be a mapping"
        )
    return result


def _scalar_int(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1 or value.dtype == torch.bool:
            raise CudaCheckpointProjectionError(
                f"{name} must be a one-element integer tensor"
            )
        value = value.detach().cpu().item()
    if type(value) is not int or value < minimum:
        raise CudaCheckpointProjectionError(
            f"{name} must be an exact integer >= {minimum}"
        )
    return int(value)


def _require_zero_status(generation_result: Any, name: str) -> None:
    value = _member(generation_result, name)
    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        raise CudaCheckpointProjectionError(
            f"CudaGenerationResult.{name} must be a non-empty status tensor"
        )
    status = value.detach().cpu()
    failed = status.ne(0).nonzero(as_tuple=False)
    if failed.numel():
        index = tuple(int(part) for part in failed[0].tolist())
        code = int(status[index].item())
        raise CudaCheckpointProjectionError(
            "post-generation CUDA checkpoint requires successful "
            f"{name}: first_failure={index} status={code}"
        )


def _result_bookkeeping(generation_result: Any) -> dict[str, int]:
    missing = [
        name
        for name in POST_GENERATION_REQUIRED_BOOKKEEPING_FIELDS
        if not _has_member(generation_result, name)
    ]
    if missing:
        raise CudaCheckpointProjectionError(
            "CudaGenerationResult lacks required checkpoint bookkeeping counters: "
            + ", ".join(missing)
        )
    totals = {
        name: _scalar_int(
            _member(generation_result, name),
            name=f"CudaGenerationResult.{name}",
        )
        for name in POST_GENERATION_REQUIRED_BOOKKEEPING_FIELDS
    }
    runtime = _member(generation_result, "runtime_bookkeeping_by_tick")
    if (
        not isinstance(runtime, torch.Tensor)
        or runtime.dtype != torch.int64
        or runtime.ndim != 2
        or runtime.shape[0] <= 0
        or runtime.shape[1] != len(POST_GENERATION_REQUIRED_BOOKKEEPING_FIELDS)
    ):
        raise CudaCheckpointProjectionError(
            "CudaGenerationResult.runtime_bookkeeping_by_tick must be "
            "int64[steps,5]"
        )
    runtime_cpu = runtime.detach().cpu()
    for index, name in enumerate(POST_GENERATION_REQUIRED_BOOKKEEPING_FIELDS):
        if int(runtime_cpu[:, index].sum().item()) != totals[name]:
            raise CudaCheckpointProjectionError(
                "CudaGenerationResult runtime bookkeeping does not sum to "
                f"aggregate {name}"
            )
    last = runtime_cpu[-1]
    return {
        **totals,
        "generation_calls": int(runtime_cpu.shape[0]),
        "last_applied_actions": int(last[0].item()),
        "last_moved": int(last[1].item()),
        "last_gathered": int(last[2].item()),
        "last_policy_refill_calls": int(last[3].item()),
        "last_policy_refill_rows": int(last[4].item()),
    }


class _ResidentCpuSnapshot:
    def __init__(self) -> None:
        self.source_device: torch.device | None = None
        self.device_to_host_operations = 0
        self.device_to_host_bytes = 0
        self.device_synchronizations = 0

    def copy(
        self,
        value: Any,
        *,
        name: str,
        allow_pre_materialized_cpu: bool = False,
    ) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise CudaCheckpointProjectionError(
                f"resident checkpoint field is not a tensor: {name}"
            )
        if value.requires_grad or not value.is_contiguous():
            raise CudaCheckpointProjectionError(
                f"resident checkpoint tensor must be contiguous and detached: {name}"
            )
        if value.device.type == "cpu" and allow_pre_materialized_cpu:
            return value.detach().contiguous().clone()
        if self.source_device is None:
            self.source_device = value.device
        elif value.device != self.source_device:
            raise CudaCheckpointProjectionError(
                "resident checkpoint tensors must share one device: "
                f"expected={self.source_device} field={name} actual={value.device}"
            )
        if value.device.type == "cuda":
            if self.device_synchronizations == 0:
                torch.cuda.synchronize(value.device)
                self.device_synchronizations = 1
            self.device_to_host_operations += 1
            self.device_to_host_bytes += _tensor_bytes(value)
        return value.detach().to(device="cpu").contiguous().clone()

    @property
    def device(self) -> str:
        return str(self.source_device or torch.device("cpu"))


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _exact_int(value: Any, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise RuntimeError(f"{name} must be an exact integer >= {minimum}")
    return int(value)


def _exact_int_list(value: Any, *, name: str) -> list[int]:
    if not isinstance(value, (list, tuple)):
        raise RuntimeError(f"{name} must be a list or tuple")
    result = []
    for index, item in enumerate(value):
        if type(item) is not int:
            raise RuntimeError(f"{name}[{index}] must be an exact integer")
        result.append(int(item))
    return result


def _pcg64_payload(
    recorder: _ResidentCpuSnapshot,
    *,
    state_words: Any,
    buffer_words: Any,
    name: str,
) -> dict[str, object]:
    state = recorder.copy(state_words, name=f"{name}.state_words")
    buffer = recorder.copy(buffer_words, name=f"{name}.buffer_words")
    if state.dtype != torch.uint64 or tuple(state.shape) != (4,):
        raise CudaCheckpointProjectionError(
            f"{name} PCG64 state must be uint64[4]"
        )
    if buffer.dtype != torch.int64 or tuple(buffer.shape) != (2,):
        raise CudaCheckpointProjectionError(
            f"{name} PCG64 buffer must be int64[2]"
        )
    words = [int(value) for value in state.tolist()]
    buffered = [int(value) for value in buffer.tolist()]
    if buffered[0] not in {0, 1} or not 0 <= buffered[1] <= 0xFFFFFFFF:
        raise CudaCheckpointProjectionError(
            f"{name} PCG64 buffer is invalid: {buffered}"
        )
    return {
        "bit_generator": "PCG64",
        "state": {
            "state": words[0] | (words[1] << 64),
            "inc": words[2] | (words[3] << 64),
        },
        "has_uint32": buffered[0],
        "uinteger": buffered[1],
    }


def _cpython_random_payload(
    recorder: _ResidentCpuSnapshot,
    scalar_rng_state: Mapping[str, Any],
) -> dict[str, Any]:
    words = recorder.copy(
        scalar_rng_state.get("python_random_words"),
        name="scalar_rng_state.python_random_words",
    )
    index = _scalar_int(
        scalar_rng_state.get("python_random_index"),
        name="scalar_rng_state.python_random_index",
    )
    raw_draws = _scalar_int(
        scalar_rng_state.get("python_random_raw_draws"),
        name="scalar_rng_state.python_random_raw_draws",
    )
    has_gauss_tensor = recorder.copy(
        scalar_rng_state.get("python_random_has_gauss"),
        name="scalar_rng_state.python_random_has_gauss",
    )
    gauss_tensor = recorder.copy(
        scalar_rng_state.get("python_random_gauss_next"),
        name="scalar_rng_state.python_random_gauss_next",
    )
    if words.dtype != torch.uint32 or tuple(words.shape) != (624,):
        raise CudaCheckpointProjectionError(
            "scalar_rng_state.python_random_words must be uint32[624]"
        )
    if not 0 <= index <= 624:
        raise CudaCheckpointProjectionError(
            "scalar_rng_state.python_random_index must be in [0, 624]"
        )
    if raw_draws < 0:
        raise CudaCheckpointProjectionError(
            "scalar_rng_state.python_random_raw_draws must be non-negative"
        )
    if has_gauss_tensor.dtype != torch.bool or has_gauss_tensor.numel() != 1:
        raise CudaCheckpointProjectionError(
            "scalar_rng_state.python_random_has_gauss must be bool[1]"
        )
    if gauss_tensor.dtype != torch.float64 or gauss_tensor.numel() != 1:
        raise CudaCheckpointProjectionError(
            "scalar_rng_state.python_random_gauss_next must be float64[1]"
        )
    has_gauss = bool(has_gauss_tensor.item())
    gauss_next = float(gauss_tensor.item()) if has_gauss else None
    if gauss_next is not None and not torch.isfinite(gauss_tensor).all().item():
        raise CudaCheckpointProjectionError(
            "scalar_rng_state.python_random_gauss_next must be finite"
        )
    return {
        "schema_version": "python_random_state_v1",
        "version": 3,
        "state": [int(value) for value in words.tolist()],
        "index": index,
        "gauss_next": gauss_next,
        "raw_draws": raw_draws,
    }


def _set_semantic_counter(
    stream_indices: dict[str, Any],
    *,
    stream: str,
    source: str,
    value: Any,
    name: str,
) -> None:
    row = stream_indices.get(stream)
    if not isinstance(row, Mapping):
        raise CudaCheckpointProjectionError(
            f"canonical skeleton RNG stream counters are missing stream={stream}"
        )
    mutable_row = dict(row)
    mutable_row[source] = _scalar_int(value, name=name)
    stream_indices[stream] = mutable_row


def _birth_event_columns(resident_birth_state: Any) -> Sequence[Any]:
    columns = _member(resident_birth_state, "event_tape_columns")
    if isinstance(columns, Mapping):
        missing = [name for name in EVENT_COLUMN_NAMES if name not in columns]
        if missing:
            raise CudaCheckpointProjectionError(
                "resident birth event columns are incomplete: " + ", ".join(missing)
            )
        return tuple(columns[name] for name in EVENT_COLUMN_NAMES)
    if not isinstance(columns, (list, tuple)) or len(columns) != len(EVENT_COLUMN_NAMES):
        raise CudaCheckpointProjectionError(
            "resident birth event_tape_columns must match the canonical event schema"
        )
    return columns


def _resident_canonical_source(
    canonical_name: str,
    *,
    resident_boundary_tensors: Mapping[str, Any],
    identity_state: Mapping[str, Any],
    genome_state: Mapping[str, Any],
    drive_state: Mapping[str, Any],
    policy_state: Mapping[str, Any],
    resident_birth_state: Any,
) -> tuple[Any, str] | None:
    # Identity wins collisions such as repro_threshold. Derived and event fields
    # are handled separately and never enter this precedence chain.
    source_name = _BIRTH_IDENTITY_CANONICAL_SOURCES.get(canonical_name)
    if source_name is not None and source_name in identity_state:
        return identity_state[source_name], f"identity_state.{source_name}"
    source_name = _BIRTH_GENOME_CANONICAL_SOURCES.get(canonical_name)
    if source_name is not None and source_name in genome_state:
        return genome_state[source_name], f"genome_state.{source_name}"
    source_name = _BIRTH_DRIVE_CANONICAL_SOURCES.get(canonical_name)
    if source_name is not None and source_name in drive_state:
        return drive_state[source_name], f"drive_state.{source_name}"
    source_name = _BIRTH_POLICY_CANONICAL_SOURCES.get(canonical_name)
    if source_name is not None and source_name in policy_state:
        return policy_state[source_name], f"policy_state.{source_name}"
    if canonical_name == "policy_mem_dim":
        return (
            _member(resident_birth_state, "policy_mem_dim"),
            "resident_birth_state.policy_mem_dim",
        )
    if canonical_name in resident_boundary_tensors:
        return resident_boundary_tensors[canonical_name], f"boundary.{canonical_name}"
    return None


def _cpu_canonical_projection(projection: Mapping[str, Any]) -> dict[str, Any]:
    tensors_value = projection.get("tensors")
    if not isinstance(tensors_value, Mapping) or not tensors_value:
        raise RuntimeError("canonical checkpoint projection requires tensors")
    tensors: dict[str, torch.Tensor] = {}
    for name, value in tensors_value.items():
        if type(name) is not str or not name:
            raise RuntimeError("canonical checkpoint tensor names must be non-empty strings")
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
            raise RuntimeError(
                f"canonical checkpoint tensor must be CPU-resident: {name}"
            )
        if value.requires_grad:
            raise RuntimeError(
                f"canonical checkpoint tensor must not require gradients: {name}"
            )
        tensors[name] = value.detach().contiguous().clone()

    out = {
        str(name): deepcopy(value)
        for name, value in projection.items()
        if name not in {"metadata", "tensors", "tensor_manifest"}
    }
    out["metadata"] = deepcopy(dict(projection.get("metadata") or {}))
    out["tensors"] = tensors
    return _with_checkpoint_contract_metadata(out)


def _canonical_disabled_tensor_fields(
    metadata: Mapping[str, Any],
) -> frozenset[str]:
    fields = set(CANONICAL_DISABLED_TENSOR_FIELDS)
    if (
        metadata.get("ablate_prediction") is True
        or metadata.get("force_prediction_disabled") is True
    ):
        fields.update(DISABLED_PREDICTION_STATE_FIELDS_V1)
    policy_family = str(
        metadata.get("native_runtime_fixed_policy_family", "")
    ).lower()
    if policy_family in {"gru", "mixed"}:
        fields.difference_update(_GRU_RESIDENT_TENSOR_FIELDS)
    return frozenset(fields)


def _preflight_post_turnover_projection(
    canonical_projection: Mapping[str, Any],
    *,
    next_generation: int,
    turnover_completed: bool,
) -> tuple[dict[str, Any], int]:
    if turnover_completed is not True:
        raise RuntimeError("CUDA checkpoint export requires completed turnover")
    generation = _exact_int(next_generation, name="next_generation", minimum=1)
    if canonical_projection.get("schema_version") != CHECKPOINT_PROJECTION_SCHEMA_VERSION:
        raise RuntimeError("CUDA checkpoint export requires the canonical checkpoint schema")
    raw_metadata = canonical_projection.get("metadata")
    if not isinstance(raw_metadata, Mapping):
        raise RuntimeError("canonical checkpoint metadata must be a mapping")
    if type(raw_metadata.get("generation")) is not int or raw_metadata["generation"] != generation:
        raise RuntimeError("canonical checkpoint generation does not match next_generation")
    if type(raw_metadata.get("tick")) is not int or raw_metadata["tick"] != 0:
        raise RuntimeError("CUDA checkpoint export requires next_generation at tick 0")
    if raw_metadata.get("agent_grid_runner_phase") != "generation_boundary":
        raise RuntimeError("CUDA checkpoint export requires a generation boundary")
    world = raw_metadata.get("world")
    if type(world) is not str or world not in {"flat", "patch"}:
        raise RuntimeError("CUDA checkpoint export supports only Flat or Patch")
    if canonical_projection.get("last_world") != world:
        raise RuntimeError("canonical checkpoint last_world contradicts metadata.world")

    pending_agents = _exact_int_list(
        canonical_projection.get("pending_generation_free_agent_ids"),
        name="pending_generation_free_agent_ids",
    )
    pending_slots = _exact_int_list(
        canonical_projection.get("pending_generation_free_slots"),
        name="pending_generation_free_slots",
    )
    if len(pending_agents) != len(pending_slots):
        raise RuntimeError("canonical checkpoint pending-free list lengths disagree")
    if pending_agents or pending_slots:
        raise RuntimeError("completed turnover must leave no pending-generation-free rows")

    canonical = _cpu_canonical_projection(canonical_projection)
    source_tensors = canonical["tensors"]
    required_source_fields = set(_canonical_disabled_tensor_fields(raw_metadata))
    required_source_fields.update(
        {
            "agent_id_to_slot_agent_ids",
            "agent_id_to_slot_slots",
            "agent_ids_by_alive_order",
            *OPAQUE_CPU_RNG_TENSOR_FIELDS,
            *EVENT_COLUMN_NAMES,
        }
    )
    if world == "patch":
        required_source_fields.update({"patch_density", "patch_visibility"})
    missing = sorted(required_source_fields - set(source_tensors))
    if missing:
        raise RuntimeError(
            "canonical checkpoint projection is incomplete: " + ", ".join(missing)
        )
    rng_state = source_tensors["rng_manager_mutation_torch_cpu_state"]
    if rng_state.dtype != torch.uint8 or rng_state.ndim != 1:
        raise RuntimeError("opaque Torch CPU RNG state must be a rank-1 uint8 tensor")

    validate_native_checkpoint_projection(canonical)
    authority = checkpoint_source_projection_authority_manifest(canonical)
    if (
        authority["source_generation"] != generation
        or authority["source_tick"] != 0
        or authority["source_phase"] != "generation_boundary"
    ):
        raise RuntimeError("canonical checkpoint authority is not next_generation/tick 0")
    active_count = _exact_int(
        canonical["metadata"].get("event_tape_active_count"),
        name="metadata.event_tape_active_count",
    )
    if canonical.get("event_cursor") != active_count:
        raise RuntimeError("canonical checkpoint event_cursor contradicts active events")
    return canonical, active_count


def _single_cuda_device(tensors: Mapping[str, torch.Tensor]) -> torch.device:
    if not tensors:
        raise RuntimeError("CUDA checkpoint projection requires authoritative tensors")
    devices = {tensor.device for tensor in tensors.values() if isinstance(tensor, torch.Tensor)}
    if len(devices) != 1:
        raise RuntimeError("CUDA checkpoint tensors must occupy exactly one device")
    device = next(iter(devices))
    if device.type != "cuda":
        raise RuntimeError("CUDA checkpoint export requires CUDA-resident tensors")
    if len(tensors) != len(devices) and any(not isinstance(value, torch.Tensor) for value in tensors.values()):
        raise RuntimeError("CUDA checkpoint tensor mapping contains a non-tensor value")
    return device


def export_cuda_boundary_checkpoint(
    *,
    schema_version: str,
    metadata: Mapping[str, Any],
    tensors: Mapping[str, torch.Tensor],
    pending_generation_free_agent_ids: tuple[int, ...] = (),
    pending_generation_free_slots: tuple[int, ...] = (),
) -> CudaBoundaryCheckpoint:
    device = _single_cuda_device(tensors)
    if not schema_version:
        raise RuntimeError("CUDA checkpoint schema version is required")
    torch.cuda.synchronize(device)
    cpu_tensors = {
        str(name): tensor.detach().to(device="cpu").contiguous().clone()
        for name, tensor in tensors.items()
    }
    projection: dict[str, Any] = {
        "schema_version": str(schema_version),
        "metadata": deepcopy(dict(metadata)),
        "tensors": cpu_tensors,
        "pending_generation_free_agent_ids": [int(value) for value in pending_generation_free_agent_ids],
        "pending_generation_free_slots": [int(value) for value in pending_generation_free_slots],
    }
    return CudaBoundaryCheckpoint(
        projection=projection,
        projection_digest=projection_digest(projection),
        device=str(device),
        device_to_host_operations=len(cpu_tensors),
        device_to_host_bytes=sum(_tensor_bytes(tensor) for tensor in cpu_tensors.values()),
        device_synchronizations=1,
    )


def export_post_turnover_cuda_checkpoint(
    *,
    canonical_projection: Mapping[str, Any],
    resident_tensors: Mapping[str, torch.Tensor],
    next_generation: int,
    turnover_completed: bool,
) -> CudaBoundaryCheckpoint:
    """Export one strict post-turnover CPU checkpoint from resident CUDA state.

    The canonical projection remains authoritative for metadata, pending lists,
    disabled values, refreshed derived views, and opaque CPU RNG bytes. Only
    resident-authoritative tensors and active event prefixes cross devices.
    """

    canonical, active_count = _preflight_post_turnover_projection(
        canonical_projection,
        next_generation=next_generation,
        turnover_completed=turnover_completed,
    )
    canonical_tensors = dict(canonical["tensors"])
    copied_from_resident: dict[str, torch.Tensor] = {}
    missing_resident: list[str] = []
    for name in canonical_tensors:
        if name in _SOURCE_CANONICAL_TENSOR_FIELDS:
            continue
        value = resident_tensors.get(name)
        if not isinstance(value, torch.Tensor):
            missing_resident.append(name)
            continue
        if name in EVENT_COLUMN_NAMES:
            if value.ndim != 1 or int(value.shape[0]) < active_count:
                raise RuntimeError(
                    f"resident active event prefix is incomplete: {name}"
                )
            value = value[:active_count].contiguous()
        copied_from_resident[name] = value
    if missing_resident:
        raise RuntimeError(
            "resident CUDA checkpoint state is incomplete: "
            + ", ".join(sorted(missing_resident))
        )

    transport = export_cuda_boundary_checkpoint(
        schema_version=CHECKPOINT_PROJECTION_SCHEMA_VERSION,
        metadata=canonical["metadata"],
        tensors=copied_from_resident,
        pending_generation_free_agent_ids=(),
        pending_generation_free_slots=(),
    )
    final_tensors = dict(transport.projection["tensors"])
    for name in sorted(_SOURCE_CANONICAL_TENSOR_FIELDS.intersection(canonical_tensors)):
        final_tensors[name] = canonical_tensors[name].detach().contiguous().clone()

    projection = {
        str(name): deepcopy(value)
        for name, value in canonical.items()
        if name not in {"metadata", "tensors", "tensor_manifest"}
    }
    projection["metadata"] = deepcopy(canonical["metadata"])
    projection["tensors"] = {
        name: final_tensors[name] for name in sorted(final_tensors)
    }
    projection["pending_generation_free_agent_ids"] = []
    projection["pending_generation_free_slots"] = []
    projection["event_cursor"] = active_count
    projection = _with_checkpoint_contract_metadata(projection)
    validate_native_checkpoint_projection(projection)
    authority = checkpoint_source_projection_authority_manifest(projection)
    if (
        authority["source_generation"] != next_generation
        or authority["source_tick"] != 0
        or authority["source_phase"] != "generation_boundary"
    ):
        raise RuntimeError("exported CUDA checkpoint lost post-turnover authority")

    return CudaBoundaryCheckpoint(
        projection=projection,
        projection_digest=projection_digest(projection),
        device=transport.device,
        device_to_host_operations=transport.device_to_host_operations,
        device_to_host_bytes=transport.device_to_host_bytes,
        device_synchronizations=transport.device_synchronizations,
        source_projection_authority=authority,
    )


def project_post_generation_cuda_checkpoint(
    *,
    pre_generation_canonical_projection: Mapping[str, Any],
    resident_boundary_tensors: Mapping[str, torch.Tensor],
    resident_birth_state: Any,
    generation_result: Any,
    completed_generation_count: int = 1,
    completed_interval_steps: int | None = None,
    phase_e_passive_no_churn_guard: Mapping[str, Any] | None = None,
    phase_e_passive_max_births: int | None = None,
    phase_e_active_modular_authority: PackedModularStorageV1 | None = None,
    phase_e_active_completed_steps: int | None = None,
    phase_e_active_modular_lifecycle_parent_slots: Mapping[int, int] | None = None,
    phase_e_active_modular_lifecycle_mutation_receipt: (
        Mapping[str, object]
        | ModularRecurrentMutationReceiptV1
        | ModularRecurrentMutationSequenceV1
        | None
    ) = None,
    pre_materialized_cpu_fields: frozenset[str] = frozenset(),
    host_authoritative_cpu_policy_fields: frozenset[str] = frozenset(),
    generation_turnover_enabled: bool = True,
) -> CudaBoundaryCheckpoint:
    """Project an exact checkpoint from completed resident state.

    The input projection supplies only the canonical schema/configuration
    skeleton and fixed disabled tensors. Mutable state, RNG continuation, event
    prefixes, derived identity maps, and runtime bookkeeping are rebuilt from
    the completed resident banks and generation result. ``completed_interval_steps``
    publishes a legal partial-generation cut point without inventing a
    turnover. The explicit active
    Phase E path accepts a completed fixed interval with no population
    turnover, validates its source-generation/end-tick receipt, and publishes
    that interval as the next generation boundary.
    """

    if not isinstance(resident_boundary_tensors, Mapping):
        raise CudaCheckpointProjectionError(
            "resident_boundary_tensors must be a mapping"
        )
    if not isinstance(generation_turnover_enabled, bool):
        raise CudaCheckpointProjectionError(
            "generation_turnover_enabled must be a bool"
        )
    if (
        not isinstance(host_authoritative_cpu_policy_fields, frozenset)
        or any(type(name) is not str or not name for name in host_authoritative_cpu_policy_fields)
    ):
        raise CudaCheckpointProjectionError(
            "host-authoritative CPU policy fields must be non-empty names"
        )
    if completed_interval_steps is not None and (
        phase_e_active_modular_authority is not None
        or phase_e_passive_no_churn_guard is not None
        or phase_e_active_completed_steps is not None
        or phase_e_active_modular_lifecycle_parent_slots is not None
        or phase_e_active_modular_lifecycle_mutation_receipt is not None
    ):
        raise CudaCheckpointProjectionError(
            "completed interval and Phase E checkpoint modes are mutually exclusive"
        )
    if (
        phase_e_active_modular_authority is not None
        and phase_e_passive_no_churn_guard is not None
    ):
        raise CudaCheckpointProjectionError(
            "active and passive Phase E modular checkpoint modes are mutually exclusive"
        )
    active_modular_lifecycle = (
        phase_e_active_modular_lifecycle_parent_slots is not None
    )
    if (
        phase_e_active_modular_authority is None
        and phase_e_active_completed_steps is not None
    ):
        raise CudaCheckpointProjectionError(
            "active Phase E completed steps require an active modular authority"
        )
    if phase_e_active_modular_authority is None and active_modular_lifecycle:
        raise CudaCheckpointProjectionError(
            "active Phase E lifecycle parent slots require an active modular authority"
        )
    if (
        phase_e_active_modular_lifecycle_mutation_receipt is not None
        and not active_modular_lifecycle
    ):
        raise CudaCheckpointProjectionError(
            "active Phase E lifecycle mutation receipt requires lifecycle parent slots"
        )
    for name in _POST_GENERATION_STATUS_FIELDS:
        _require_zero_status(generation_result, name)
    _require_zero_status(resident_birth_state, "event_tape_status")

    skeleton = _cpu_canonical_projection(pre_generation_canonical_projection)
    if skeleton.get("schema_version") != CHECKPOINT_PROJECTION_SCHEMA_VERSION:
        raise CudaCheckpointProjectionError(
            "pre-generation skeleton must use the canonical checkpoint schema"
        )
    metadata = deepcopy(dict(skeleton.get("metadata") or {}))
    disabled_tensor_fields = _canonical_disabled_tensor_fields(metadata)
    source_generation = _exact_int(
        metadata.get("generation"),
        name="pre-generation metadata.generation",
    )
    source_tick = _exact_int(
        metadata.get("tick"), name="pre-generation metadata.tick"
    )
    source_phase = "generation_boundary" if source_tick == 0 else "partial_generation"
    if metadata.get("agent_grid_runner_phase") != source_phase:
        raise CudaCheckpointProjectionError(
            "pre-generation canonical skeleton phase contradicts its tick cursor"
        )
    world = metadata.get("world")
    if type(world) is not str or world not in {"flat", "patch"}:
        raise CudaCheckpointProjectionError(
            "pre-generation canonical skeleton world must be flat or patch"
        )
    pre_generation_calls = _exact_int(
        skeleton.get("generation_calls"), name="skeleton.generation_calls"
    )
    last_world = skeleton.get("last_world")
    pristine_boundary = pre_generation_calls == 0 and last_world == "none"
    if last_world != world and not pristine_boundary:
        raise CudaCheckpointProjectionError(
            "pre-generation canonical skeleton last_world contradicts metadata.world"
        )

    completed_generation_count = _exact_int(
        completed_generation_count,
        name="completed_generation_count",
        minimum=1,
    )
    completed_interval_steps = (
        None
        if completed_interval_steps is None
        else _exact_int(
            completed_interval_steps,
            name="completed_interval_steps",
            minimum=1,
        )
    )
    if (
        completed_interval_steps is not None
        and completed_generation_count != 1
    ):
        raise CudaCheckpointProjectionError(
            "a completed interval checkpoint cannot claim multiple generations"
        )
    if (
        completed_interval_steps is not None
        and completed_interval_steps <= source_tick
    ):
        raise CudaCheckpointProjectionError(
            "completed interval tick must advance the source cursor"
        )
    passive_modular_tensor_names: frozenset[str] = frozenset()
    active_modular_tensor_names: frozenset[str] = frozenset()
    active_completed_steps: int | None = None
    if phase_e_passive_no_churn_guard is not None:
        if phase_e_passive_max_births is None:
            raise _phase_e_guard_fail("enabled checkpoint projection lacks max_births")
        if completed_generation_count != 1:
            raise _phase_e_guard_fail("passive modular milestone requires one generation")
        raw_names = phase_e_passive_no_churn_guard.get("modular_tensors")
        if not isinstance(raw_names, Mapping):
            raise _phase_e_guard_fail("guard modular tensor manifest is malformed")
        passive_modular_tensor_names = frozenset(str(name) for name in raw_names)
    if phase_e_active_modular_authority is not None:
        if completed_generation_count != 1 and not active_modular_lifecycle:
            raise CudaCheckpointProjectionError(
                "active Phase E modular checkpoint projects exactly one completed interval"
            )
        if phase_e_active_completed_steps is None and not active_modular_lifecycle:
            raise CudaCheckpointProjectionError(
                "active Phase E modular checkpoint requires completed steps"
            )
        active_completed_steps = 0 if active_modular_lifecycle else _exact_int(
            phase_e_active_completed_steps,
            name="phase_e_active_completed_steps",
            minimum=1,
        )
        if not isinstance(
            phase_e_active_modular_authority,
            PackedModularStorageV1,
        ):
            raise CudaCheckpointProjectionError(
                "active Phase E modular authority must be packed modular storage"
            )
        try:
            source_modular_authority = modular_storage_from_projection(
                metadata,
                skeleton["tensors"],
            )
        except ModularStorageError as exc:
            raise CudaCheckpointProjectionError(
                f"active Phase E source modular projection is invalid: {exc}"
            ) from exc
        if source_modular_authority is None:
            raise CudaCheckpointProjectionError(
                "active Phase E modular authority requires a source overlay"
            )
        if (
            source_modular_authority.canonical_bytes()
            != phase_e_active_modular_authority.canonical_bytes()
        ):
            raise CudaCheckpointProjectionError(
                "active Phase E modular authority contradicts the source overlay"
            )
        active_modular_tensor_names = frozenset(
            f"{MODULAR_STORAGE_PROJECTION_PREFIX}{field.name}"
            for field in phase_e_active_modular_authority.layout.fields
        )
    result_next_generation = _scalar_int(
        _member(generation_result, "turnover_next_generation"),
        name="CudaGenerationResult.turnover_next_generation",
        minimum=(
            0
            if (
                completed_interval_steps is not None
                or phase_e_active_modular_authority is not None
            )
            else 1
        ),
    )
    result_next_tick = _scalar_int(
        _member(generation_result, "turnover_next_tick"),
        name="CudaGenerationResult.turnover_next_tick",
    )
    checkpoint_tick = 0
    if completed_interval_steps is not None:
        if (
            result_next_generation != source_generation
            or result_next_tick != completed_interval_steps
        ):
            raise CudaCheckpointProjectionError(
                "completed interval coordinates are not the exact partial boundary: "
                f"expected=({source_generation}, {completed_interval_steps}) "
                f"actual=({result_next_generation}, {result_next_tick})"
            )
        next_generation = source_generation
        checkpoint_tick = completed_interval_steps
    elif phase_e_active_modular_authority is not None:
        if active_modular_lifecycle:
            next_generation = source_generation + completed_generation_count
            if result_next_generation != next_generation or result_next_tick != 0:
                raise CudaCheckpointProjectionError(
                    "active modular lifecycle coordinates are not the exact next "
                    f"boundary: expected=({next_generation}, 0) "
                    f"actual=({result_next_generation}, {result_next_tick})"
                )
        elif (
            result_next_generation != source_generation
            or result_next_tick != active_completed_steps
        ):
            raise CudaCheckpointProjectionError(
                "active Phase E no-turnover coordinates are not the exact completed "
                f"interval: expected=({source_generation}, {active_completed_steps}) "
                f"actual=({result_next_generation}, {result_next_tick})"
            )
        else:
            next_generation = source_generation + 1
    else:
        next_generation = source_generation + completed_generation_count
        if result_next_generation != next_generation or result_next_tick != 0:
            raise CudaCheckpointProjectionError(
                "completed turnover coordinates are not the exact next boundary: "
                f"expected=({next_generation}, 0) "
                f"actual=({result_next_generation}, {result_next_tick})"
            )
    pending_free_count = _scalar_int(
        resident_boundary_tensors.get("pending_free_count"),
        name="resident_boundary_tensors.pending_free_count",
    )
    if pending_free_count != 0:
        raise CudaCheckpointProjectionError(
            "post-generation CUDA checkpoint requires pending_free_count zero"
        )
    bookkeeping = _result_bookkeeping(generation_result)
    if (
        phase_e_active_modular_authority is not None
        and not active_modular_lifecycle
        and bookkeeping["generation_calls"] != active_completed_steps
    ):
        raise CudaCheckpointProjectionError(
            "active Phase E runtime bookkeeping does not match completed steps"
        )

    identity_state = _mapping_member(resident_birth_state, "identity_state")
    scalar_rng_state = _mapping_member(resident_birth_state, "scalar_rng_state")
    genome_state = _mapping_member(resident_birth_state, "genome_state")
    drive_state = _mapping_member(resident_birth_state, "drive_state")
    policy_state = _mapping_member(resident_birth_state, "policy_state")
    recorder = _ResidentCpuSnapshot()

    source_tensors = dict(skeleton["tensors"])
    final_tensors: dict[str, torch.Tensor] = {}
    identity_derived = {
        "agent_id_to_slot_agent_ids",
        "agent_id_to_slot_slots",
        "agent_ids_by_alive_order",
    }
    patch_derived = {"patch_density", "patch_visibility"}
    special = identity_derived | patch_derived | set(EVENT_COLUMN_NAMES) | {
        "alive_mask",
        "slot_free_queue",
        "slot_to_agent_id",
    }
    for canonical_name in sorted(source_tensors):
        if canonical_name in special:
            continue
        if canonical_name in active_modular_tensor_names:
            continue
        if canonical_name in passive_modular_tensor_names:
            final_tensors[canonical_name] = source_tensors[canonical_name].clone()
            continue
        if canonical_name in disabled_tensor_fields:
            final_tensors[canonical_name] = source_tensors[canonical_name].clone()
            continue
        if canonical_name in pre_materialized_cpu_fields:
            final_tensors[canonical_name] = source_tensors[canonical_name].clone()
            continue
        resolved = _resident_canonical_source(
            canonical_name,
            resident_boundary_tensors=resident_boundary_tensors,
            identity_state=identity_state,
            genome_state=genome_state,
            drive_state=drive_state,
            policy_state=policy_state,
            resident_birth_state=resident_birth_state,
        )
        if resolved is None:
            raise CudaCheckpointProjectionError(
                "post-generation resident banks lack canonical tensor source: "
                f"{canonical_name}"
            )
        value, source_name = resolved
        allow_host_authoritative_cpu = (
            canonical_name in host_authoritative_cpu_policy_fields
            and source_name == f"policy_state.{canonical_name}"
        )
        if allow_host_authoritative_cpu and (
            not isinstance(value, torch.Tensor) or value.device.type != "cpu"
        ):
            raise CudaCheckpointProjectionError(
                "host-authoritative checkpoint policy field must remain pageable CPU: "
                f"{source_name}"
            )
        final_tensors[canonical_name] = recorder.copy(
            value,
            name=source_name,
            allow_pre_materialized_cpu=allow_host_authoritative_cpu,
        )

    slot_to_agent_id = recorder.copy(
        identity_state.get("slot_to_agent_id"),
        name="identity_state.slot_to_agent_id",
    )
    alive_mask = recorder.copy(
        identity_state.get("alive_mask"),
        name="identity_state.alive_mask",
    )
    capacity = _exact_int(metadata.get("capacity"), name="metadata.capacity", minimum=1)
    if (
        slot_to_agent_id.dtype != torch.int64
        or alive_mask.dtype != torch.bool
        or tuple(slot_to_agent_id.shape) != (capacity,)
        or tuple(alive_mask.shape) != (capacity,)
    ):
        raise CudaCheckpointProjectionError(
            "resident slot identity must be int64/bool vectors matching metadata.capacity"
        )
    if not torch.equal(alive_mask, slot_to_agent_id.ge(0)):
        raise CudaCheckpointProjectionError(
            "resident alive_mask contradicts slot_to_agent_id occupancy"
        )
    live_slots = alive_mask.nonzero(as_tuple=False).flatten().tolist()
    live_pairs = sorted(
        ((int(slot_to_agent_id[slot].item()), int(slot)) for slot in live_slots),
        key=lambda pair: pair[0],
    )
    live_agent_ids = [agent_id for agent_id, _slot in live_pairs]
    if len(live_agent_ids) != len(set(live_agent_ids)):
        raise CudaCheckpointProjectionError("resident live agent IDs are duplicated")

    result_alive_count = _scalar_int(
        _member(generation_result, "final_alive_count"),
        name="CudaGenerationResult.final_alive_count",
    )
    result_alive_ids = recorder.copy(
        _member(generation_result, "final_alive_agent_ids"),
        name="CudaGenerationResult.final_alive_agent_ids",
    )
    if result_alive_ids.dtype != torch.int64 or result_alive_ids.ndim != 1:
        raise CudaCheckpointProjectionError(
            "CudaGenerationResult.final_alive_agent_ids must be rank-1 int64"
        )
    if result_alive_count != len(live_agent_ids):
        raise CudaCheckpointProjectionError(
            "CudaGenerationResult.final_alive_count contradicts resident identity"
        )
    expected_alive_ids = torch.tensor(live_agent_ids, dtype=torch.int64)
    if not torch.equal(result_alive_ids[:result_alive_count], expected_alive_ids):
        raise CudaCheckpointProjectionError(
            "CudaGenerationResult final alive IDs are not the compact sorted resident IDs"
        )
    if bool(result_alive_ids[result_alive_count:].ne(-1).any().item()):
        raise CudaCheckpointProjectionError(
            "CudaGenerationResult final alive ID tail must contain only -1 sentinels"
        )
    turnover_counts = recorder.copy(
        _member(generation_result, "turnover_counts"),
        name="CudaGenerationResult.turnover_counts",
    )
    if turnover_counts.ndim != 1 or turnover_counts.numel() != 8:
        raise CudaCheckpointProjectionError(
            "CudaGenerationResult.turnover_counts must contain eight counters"
        )
    no_turnover_checkpoint = completed_interval_steps is not None or (
        phase_e_active_modular_authority is not None
        and not active_modular_lifecycle
    )
    if not generation_turnover_enabled:
        if bool(turnover_counts.ne(0).any().item()):
            raise CudaCheckpointProjectionError(
                "hybrid no-turnover checkpoint result must report zero "
                "turnover counts, including inactive final-live"
            )
    elif no_turnover_checkpoint:
        if bool(turnover_counts.ne(0).any().item()):
            raise CudaCheckpointProjectionError(
                "no-turnover checkpoint result must report zero turnover counts"
            )
    elif int(turnover_counts[7].item()) != result_alive_count:
        raise CudaCheckpointProjectionError(
            "CudaGenerationResult turnover final-live counter contradicts resident identity"
        )

    final_tensors["slot_to_agent_id"] = slot_to_agent_id
    final_tensors["alive_mask"] = alive_mask
    alive_id_dtype = source_tensors["agent_ids_by_alive_order"].dtype
    sparse_id_dtype = source_tensors["agent_id_to_slot_agent_ids"].dtype
    sparse_slot_dtype = source_tensors["agent_id_to_slot_slots"].dtype
    final_tensors["agent_ids_by_alive_order"] = torch.tensor(
        live_agent_ids, dtype=alive_id_dtype
    )
    final_tensors["agent_id_to_slot_agent_ids"] = torch.tensor(
        live_agent_ids, dtype=sparse_id_dtype
    )
    final_tensors["agent_id_to_slot_slots"] = torch.tensor(
        [slot for _agent_id, slot in live_pairs], dtype=sparse_slot_dtype
    )
    if (
        final_tensors["agent_id_to_slot_agent_ids"].numel() > capacity
        or final_tensors["agent_id_to_slot_slots"].numel() > capacity
    ):
        raise CudaCheckpointProjectionError(
            "compact resident identity exceeds slot capacity"
        )
    for name, numpy_dtype in EED_CURSOR_ENV_ARRAYS:
        cursor = final_tensors.get(name)
        if cursor is not None:
            expected_dtype = torch.from_numpy(
                np.empty((), dtype=numpy_dtype)
            ).dtype
            if (
                cursor.dtype != expected_dtype
                or tuple(cursor.shape) != (capacity,)
                or not cursor.is_contiguous()
            ):
                raise CudaCheckpointProjectionError(
                    "EED cursor checkpoint state must be a slot-capacity vector: "
                    f"field={name};dtype={cursor.dtype};"
                    f"shape={tuple(cursor.shape)};capacity={capacity}"
                )
            final_tensors[name] = cursor.clone()

    free_queue = recorder.copy(
        identity_state.get("slot_free_queue"),
        name="identity_state.slot_free_queue",
    )
    free_count = _scalar_int(
        identity_state.get("slot_free_queue_count"),
        name="identity_state.slot_free_queue_count",
    )
    if free_queue.dtype != torch.int64 or free_queue.ndim != 1:
        raise CudaCheckpointProjectionError(
            "resident slot_free_queue must be rank-1 int64"
        )
    if free_count > int(free_queue.numel()):
        raise CudaCheckpointProjectionError(
            "resident slot_free_queue_count exceeds queue capacity"
        )
    compact_free_queue = free_queue[:free_count].contiguous().clone()
    free_slots = [int(value) for value in compact_free_queue.tolist()]
    if (
        len(free_slots) != len(set(free_slots))
        or any(slot < 0 or slot >= capacity for slot in free_slots)
        or set(free_slots) != set(range(capacity)) - set(live_slots)
    ):
        expected_free_slots = set(range(capacity)) - set(live_slots)
        actual_free_slots = set(free_slots)
        raise CudaCheckpointProjectionError(
            "resident slot_free_queue active prefix contradicts the live/free "
            f"partition: count={free_count};expected={len(expected_free_slots)};"
            f"duplicates={len(free_slots) - len(actual_free_slots)};"
            f"missing={sorted(expected_free_slots - actual_free_slots)[:16]};"
            f"extra={sorted(actual_free_slots - expected_free_slots)[:16]}"
        )
    final_tensors["slot_free_queue"] = compact_free_queue

    active_event_count = _scalar_int(
        _member(generation_result, "event_tape_active_count"),
        name="CudaGenerationResult.event_tape_active_count",
    )
    birth_active_event_count = _scalar_int(
        _member(resident_birth_state, "event_tape_active_count"),
        name="resident_birth_state.event_tape_active_count",
    )
    if active_event_count != birth_active_event_count:
        raise CudaCheckpointProjectionError(
            "generation result and resident birth event counts disagree"
        )
    external_event_history = metadata.get("event_history_storage_mode") == (
        "eiel_resident_event_sink_v1"
    )
    raw_sequence_base = getattr(
        generation_result,
        "event_lifecycle_sequence_base",
        resident_boundary_tensors.get("event_sequence_base", 0),
    )
    sequence_base = _scalar_int(
        raw_sequence_base,
        name="CudaGenerationResult.event_lifecycle_sequence_base",
    )
    raw_frontier = getattr(
        generation_result,
        "event_lifecycle_frontier",
        sequence_base + active_event_count,
    )
    lifecycle_frontier = _scalar_int(
        raw_frontier,
        name="CudaGenerationResult.event_lifecycle_frontier",
    )
    if external_event_history and lifecycle_frontier != sequence_base + active_event_count:
        raise CudaCheckpointProjectionError(
            "external event-history frontier does not match pending resident rows"
        )
    for name, value in zip(
        EVENT_COLUMN_NAMES,
        _birth_event_columns(resident_birth_state),
        strict=True,
    ):
        column = recorder.copy(
            value,
            name=f"resident_birth_state.{name}",
            allow_pre_materialized_cpu=True,
        )
        if column.ndim != 1 or int(column.numel()) < active_event_count:
            raise CudaCheckpointProjectionError(
                f"resident event column lacks its active prefix: {name}"
            )
        final_tensors[name] = column[:active_event_count].contiguous().clone()

    if phase_e_active_modular_authority is not None:
        live_modular_tensors = {
            name: recorder.copy(
                resident_boundary_tensors.get(name),
                name=f"resident_boundary_tensors.{name}",
            )
            for name in sorted(active_modular_tensor_names)
        }
        try:
            if active_modular_lifecycle:
                active_modular_payload = modular_storage_lifecycle_projection_payload(
                    phase_e_active_modular_authority,
                    live_modular_tensors,
                    profile_id=phase_e_active_modular_authority.layout.profile_id,
                    alive_mask=alive_mask,
                    slot_to_agent_id=slot_to_agent_id,
                    slot_tokens=final_tensors["slot_tokens"],
                    selected_parent_slot_by_child_slot=(
                        phase_e_active_modular_lifecycle_parent_slots or {}
                    ),
                    mutation_receipt=(
                        phase_e_active_modular_lifecycle_mutation_receipt
                    ),
                )
            else:
                active_modular_payload = modular_storage_generation_boundary_projection_payload(
                    phase_e_active_modular_authority,
                    live_modular_tensors,
                    alive_mask=alive_mask,
                    slot_to_agent_id=slot_to_agent_id,
                    slot_tokens=final_tensors["slot_tokens"],
                )
        except ModularStorageError as exc:
            raise CudaCheckpointProjectionError(
                f"active Phase E modular generation boundary is invalid: {exc}"
            ) from exc
        metadata[MODULAR_STORAGE_METADATA_KEY] = active_modular_payload[
            "metadata"
        ][MODULAR_STORAGE_METADATA_KEY]
        final_tensors.update(active_modular_payload["tensors"])

        if (
            metadata.get("ablate_prediction") is True
            and metadata.get("force_prediction_disabled") is True
        ):
            if active_completed_steps is None:
                raise CudaCheckpointProjectionError(
                    "disabled-prediction settlement lost its completed-step count"
                )
            live_rows = alive_mask.unsqueeze(1)
            for name in ("exp_dE", "exp_dInj"):
                tensor = final_tensors.get(name)
                if (
                    not isinstance(tensor, torch.Tensor)
                    or tensor.ndim != 2
                    or int(tensor.shape[0]) != capacity
                ):
                    raise CudaCheckpointProjectionError(
                        f"active Phase E disabled prediction state is invalid: {name}"
                    )
                final_tensors[name] = torch.where(
                    live_rows,
                    torch.zeros_like(tensor),
                    tensor,
                ).contiguous()
            reset_count = final_tensors.get("prediction_reset_count")
            if (
                not isinstance(reset_count, torch.Tensor)
                or reset_count.ndim != 1
                or tuple(reset_count.shape) != (capacity,)
            ):
                raise CudaCheckpointProjectionError(
                    "active Phase E disabled prediction reset count is invalid"
                )
            final_tensors["prediction_reset_count"] = torch.where(
                alive_mask,
                torch.full_like(reset_count, active_completed_steps),
                reset_count,
            ).contiguous()

    # The resident controller may use a narrower active cache than its fixed
    # storage capacity. Canonical checkpoints define ``next`` from the stored
    # cursor so inactive storage columns cannot contradict the public cache.
    cache_actions = final_tensors["action_cache_actions"]
    cache_tokens = final_tensors["action_cache_token"]
    cache_cursors = final_tensors["action_cache_cursor"]
    if cache_actions.ndim != 2 or int(cache_actions.shape[1]) <= 0:
        raise CudaCheckpointProjectionError(
            "resident action_cache_actions must have positive rank-2 storage"
        )
    valid_cache_rows = alive_mask & cache_tokens.ge(0) & cache_cursors.ge(0)
    inactive_storage = valid_cache_rows.unsqueeze(1) & cache_actions.lt(0)
    cache_actions = torch.where(
        inactive_storage,
        cache_actions[:, :1].expand_as(cache_actions),
        cache_actions,
    ).contiguous()
    final_tensors["action_cache_actions"] = cache_actions
    cache_indices = torch.clamp(
        cache_cursors.to(torch.int64),
        0,
        int(cache_actions.shape[1]) - 1,
    ).unsqueeze(1)
    projected_cache_next = cache_actions.gather(1, cache_indices).squeeze(1)
    final_tensors["action_cache_next"] = torch.where(
        valid_cache_rows,
        projected_cache_next,
        torch.full_like(projected_cache_next, -1),
    ).contiguous()

    metadata["generation"] = next_generation
    metadata["tick"] = checkpoint_tick
    metadata["agent_grid_runner_phase"] = (
        "generation_boundary" if checkpoint_tick == 0 else "partial_generation"
    )
    if phase_e_active_modular_authority is not None:
        if active_completed_steps is None:
            raise CudaCheckpointProjectionError(
                "active Phase E modular checkpoint lost its completed-step count"
            )
        metadata["prediction_tick"] = _exact_int(
            metadata.get("prediction_tick"),
            name="pre-generation metadata.prediction_tick",
            minimum=0,
        ) + active_completed_steps
    else:
        completed_calls = bookkeeping["generation_calls"]
        if completed_calls % completed_generation_count != 0:
            raise CudaCheckpointProjectionError(
                "checkpoint cannot derive an exact per-generation prediction tick"
            )
        metadata["prediction_tick"] = (
            checkpoint_tick
            if completed_interval_steps is not None
            else completed_calls // completed_generation_count
        )
        if (
            metadata.get("ablate_prediction") is True
            or metadata.get("force_prediction_disabled") is True
        ):
            try:
                settled = materialize_disabled_prediction_state_v1(
                    skeleton,
                    {
                        name: value.unsqueeze(0)
                        for name, value in final_tensors.items()
                    },
                    reset_count_max=metadata["prediction_tick"],
                )
            except DisabledPredictionSettlementError as exc:
                raise CudaCheckpointProjectionError(
                    f"disabled prediction checkpoint settlement failed: {exc}"
                ) from exc
            for name in DISABLED_PREDICTION_STATE_FIELDS_V1:
                final_tensors[name] = settled[name].squeeze(0).contiguous()
    metadata["alive_count"] = result_alive_count
    metadata["generation_turnover_enabled"] = generation_turnover_enabled
    metadata["lifecycle_mode_abi"] = 0 if generation_turnover_enabled else 1
    metadata["lifecycle_population_authority_schema_version"] = (
        "eiel_gv2_lifecycle_v2_population_authority_v1"
    )
    if generation_turnover_enabled:
        metadata["lifecycle_population_authority_source"] = (
            "legacy_turnover_final_live_plus_resident_identity"
        )
        metadata["legacy_turnover_final_live_authority"] = (
            "authoritative_turnover_counter"
        )
        metadata["legacy_turnover_final_live_value"] = result_alive_count
    else:
        metadata["lifecycle_population_authority_source"] = (
            "resident_identity_plus_qacct"
        )
        metadata["legacy_turnover_final_live_authority"] = "inactive_zero_sentinel"
        metadata["legacy_turnover_final_live_value"] = 0
    metadata["native_runtime_architecture_b_compact_reporting_only_v1"] = False
    metadata["population_next_id"] = _scalar_int(
        identity_state.get("population_next_id"),
        name="identity_state.population_next_id",
        minimum=1,
    )
    env_policy_alias = metadata.get("native_runtime_env_policy_numpy_alias")
    if type(env_policy_alias) is not bool:
        raise CudaCheckpointProjectionError(
            "metadata.native_runtime_env_policy_numpy_alias must be an exact bool"
        )
    policy_pcg64 = _pcg64_payload(
        recorder,
        state_words=_member(generation_result, "pcg64_state_words"),
        buffer_words=_member(generation_result, "pcg64_buffer_words"),
        name="policy",
    )
    if env_policy_alias:
        env_pcg64 = deepcopy(policy_pcg64)
    else:
        env_pcg64 = _pcg64_payload(
            recorder,
            state_words=resident_boundary_tensors.get("env_pcg64_state_words"),
            buffer_words=resident_boundary_tensors.get("env_pcg64_buffer_words"),
            name="environment",
        )
    metadata["native_runtime_numpy_env_state"] = env_pcg64
    metadata["native_runtime_numpy_policy_state"] = policy_pcg64
    metadata["native_runtime_rng_state"] = int(env_pcg64["state"]["state"]) & (
        (1 << 63) - 1
    )
    metadata["native_runtime_numpy_runner_state"] = _pcg64_payload(
        recorder,
        state_words=resident_boundary_tensors.get("runner_pcg64_state_words"),
        buffer_words=resident_boundary_tensors.get("runner_pcg64_buffer_words"),
        name="runner",
    )
    metadata["native_runtime_numpy_population_state"] = _pcg64_payload(
        recorder,
        state_words=drive_state.get("pcg64_state_words"),
        buffer_words=drive_state.get("pcg64_buffer_words"),
        name="population",
    )
    metadata["native_runtime_numpy_birth_spawn_env_state"] = _pcg64_payload(
        recorder,
        state_words=_member(resident_birth_state, "planner_pcg64_state_words"),
        buffer_words=_member(resident_birth_state, "planner_pcg64_buffer_words"),
        name="birth_spawn_env",
    )
    metadata["native_runtime_python_random_state"] = _cpython_random_payload(
        recorder,
        scalar_rng_state,
    )

    raw_stream_indices = metadata.get("native_runtime_rng_manager_stream_indices")
    if not isinstance(raw_stream_indices, Mapping):
        raise CudaCheckpointProjectionError(
            "canonical skeleton lacks RNG manager stream counters"
        )
    stream_indices = deepcopy(dict(raw_stream_indices))
    _set_semantic_counter(
        stream_indices,
        stream="mutation",
        source="py",
        value=scalar_rng_state.get("mutation_py_semantic_draws"),
        name="scalar_rng_state.mutation_py_semantic_draws",
    )
    _set_semantic_counter(
        stream_indices,
        stream="mutation",
        source="np",
        value=drive_state.get("mutation_np_semantic_draws"),
        name="drive_state.mutation_np_semantic_draws",
    )
    _set_semantic_counter(
        stream_indices,
        stream="mutation",
        source="torch_cpu",
        value=policy_state.get("mutation_torch_cpu_semantic_draws"),
        name="policy_state.mutation_torch_cpu_semantic_draws",
    )
    _set_semantic_counter(
        stream_indices,
        stream="policy",
        source="np",
        value=_member(resident_birth_state, "policy_np_semantic_draws"),
        name="resident_birth_state.policy_np_semantic_draws",
    )
    policy_counters = stream_indices.get("policy")
    if not isinstance(policy_counters, Mapping):
        raise CudaCheckpointProjectionError(
            "canonical skeleton RNG stream counters are missing stream=policy"
        )
    policy_py = _exact_int(
        policy_counters.get("py"),
        name="metadata.native_runtime_rng_manager_stream_indices.policy.py",
    )
    policy_py_generation_increments = (
        0 if completed_interval_steps is not None else completed_generation_count
    )
    if completed_interval_steps is None and pre_generation_calls == 0:
        # A pristine imported boundary already includes the first generation's
        # eight policy draws. Resumed boundaries need all requested increments.
        policy_py_generation_increments -= 1
    if policy_py_generation_increments:
        mutable_policy_counters = dict(policy_counters)
        mutable_policy_counters["py"] = (
            policy_py + 8 * policy_py_generation_increments
        )
        stream_indices["policy"] = mutable_policy_counters
    _set_semantic_counter(
        stream_indices,
        stream="metrics",
        source="np",
        value=_member(generation_result, "metrics_np_semantic_draws"),
        name="CudaGenerationResult.metrics_np_semantic_draws",
    )
    metadata["native_runtime_rng_manager_stream_indices"] = stream_indices

    if world == "patch":
        final_tensors = _rebuild_patch_checkpoint_derived_arrays(
            metadata,
            final_tensors,
            surface="post-generation CUDA checkpoint projection",
        )
    elif patch_derived.intersection(source_tensors):
        raise CudaCheckpointProjectionError(
            "Flat canonical skeleton must not retain Patch derived tensors"
        )
    if set(final_tensors) != set(source_tensors):
        missing = sorted(set(source_tensors) - set(final_tensors))
        extra = sorted(set(final_tensors) - set(source_tensors))
        raise CudaCheckpointProjectionError(
            "post-generation canonical tensor names changed: "
            f"missing={missing} extra={extra}"
        )

    pre_runtime_event_cursor = _exact_int(
        skeleton.get("runtime_event_cursor"), name="skeleton.runtime_event_cursor"
    )
    pre_lifecycle_event_cursor = _exact_int(
        skeleton.get("event_cursor"), name="skeleton.event_cursor"
    )
    if active_event_count < pre_lifecycle_event_cursor and not external_event_history:
        raise CudaCheckpointProjectionError(
            "post-generation lifecycle event cursor moved backwards: "
            f"before={pre_lifecycle_event_cursor} after={active_event_count}"
        )
    if external_event_history:
        pre_frontier = _exact_int(
            metadata.get("event_history_frontier", sequence_base),
            name="metadata.event_history_frontier",
        )
        if lifecycle_frontier < pre_frontier:
            raise CudaCheckpointProjectionError(
                "external event-history frontier moved backwards"
            )
        appended_lifecycle_events = lifecycle_frontier - pre_frontier
        metadata["typed_lifecycle_event_sequence_base"] = sequence_base
        metadata["typed_lifecycle_event_next_sequence"] = lifecycle_frontier
        metadata["event_history_frontier"] = lifecycle_frontier
        metadata["typed_lifecycle_event_active_count"] = active_event_count
        metadata["event_tape_active_count"] = active_event_count
        metadata["event_tape_cursor"] = active_event_count
    else:
        appended_lifecycle_events = active_event_count - pre_lifecycle_event_cursor
    projection: dict[str, Any] = {
        "schema_version": CHECKPOINT_PROJECTION_SCHEMA_VERSION,
        "metadata": metadata,
        "tensors": {name: final_tensors[name] for name in sorted(final_tensors)},
        "restore_count": _exact_int(
            skeleton.get("restore_count"), name="skeleton.restore_count"
        ),
        "generation_calls": pre_generation_calls + bookkeeping["generation_calls"],
        "event_cursor": active_event_count,
        "runtime_event_cursor": (
            pre_runtime_event_cursor
            + bookkeeping["applied_actions"]
            + appended_lifecycle_events
        ),
        "last_applied_actions": bookkeeping["last_applied_actions"],
        "last_moved": bookkeeping["last_moved"],
        "last_gathered": bookkeeping["last_gathered"],
        "last_policy_refill_calls": bookkeeping["last_policy_refill_calls"],
        "last_policy_refill_rows": bookkeeping["last_policy_refill_rows"],
        "pending_generation_free_agent_ids": [],
        "pending_generation_free_slots": [],
        "last_world": world,
    }
    projection = _with_checkpoint_contract_metadata(projection)
    validate_native_checkpoint_projection(projection)
    validate_phase_e_passive_modular_no_churn_guard(
        phase_e_passive_no_churn_guard,
        projection,
        completed_generation_count=completed_generation_count,
        fixed_capacity=capacity,
        max_births=(
            0
            if phase_e_passive_no_churn_guard is None
            else phase_e_passive_max_births
        ),
    )
    authority = checkpoint_source_projection_authority_manifest(projection)
    if (
        authority["source_generation"] != next_generation
        or authority["source_tick"] != checkpoint_tick
        or authority["source_phase"]
        != (
            "generation_boundary"
            if checkpoint_tick == 0
            else "partial_generation"
        )
    ):
        raise CudaCheckpointProjectionError(
            "post-generation projection lost exact next-boundary authority"
        )
    return CudaBoundaryCheckpoint(
        projection=projection,
        projection_digest=projection_digest(projection),
        device=recorder.device,
        device_to_host_operations=recorder.device_to_host_operations,
        device_to_host_bytes=recorder.device_to_host_bytes,
        device_synchronizations=recorder.device_synchronizations,
        source_projection_authority=authority,
    )


export_post_generation_cuda_checkpoint = project_post_generation_cuda_checkpoint


def restore_cuda_boundary_checkpoint(
    checkpoint: CudaBoundaryCheckpoint,
    *,
    device: str | torch.device,
) -> CudaCheckpointRestore:
    target = torch.device(device)
    if target.type != "cuda":
        raise RuntimeError("CUDA checkpoint restore target must be a CUDA device")
    if projection_digest(checkpoint.projection) != checkpoint.projection_digest:
        raise RuntimeError("CUDA checkpoint projection changed after boundary export")
    cpu_tensors = dict(checkpoint.projection.get("tensors") or {})
    if not cpu_tensors or any(
        not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu"
        for tensor in cpu_tensors.values()
    ):
        raise RuntimeError("CUDA checkpoint restore requires a CPU tensor projection")
    cuda_tensors = {
        str(name): tensor.detach().to(device=target).contiguous()
        for name, tensor in cpu_tensors.items()
    }
    torch.cuda.synchronize(target)
    projection = deepcopy(checkpoint.projection)
    projection["tensors"] = cuda_tensors
    return CudaCheckpointRestore(
        projection=projection,
        source_projection_digest=checkpoint.projection_digest,
        device=str(target),
        host_to_device_operations=len(cuda_tensors),
        host_to_device_bytes=sum(_tensor_bytes(tensor) for tensor in cuda_tensors.values()),
        device_synchronizations=1,
    )


__all__ = [
    "CANONICAL_DISABLED_TENSOR_FIELDS",
    "CUDA_CHECKPOINT_CAPABILITIES",
    "CudaBoundaryCheckpoint",
    "CudaCheckpointRestore",
    "CudaCheckpointProjectionError",
    "OPAQUE_CPU_RNG_TENSOR_FIELDS",
    "PHASE_E_PASSIVE_NO_CHURN_GUARD_ERROR",
    "PHASE_E_PASSIVE_NO_CHURN_GUARD_SCHEMA_VERSION",
    "POST_GENERATION_REQUIRED_BOOKKEEPING_FIELDS",
    "REFRESH_DERIVED_TENSOR_FIELDS",
    "capture_phase_e_passive_modular_no_churn_guard",
    "export_cuda_boundary_checkpoint",
    "export_post_generation_cuda_checkpoint",
    "export_post_turnover_cuda_checkpoint",
    "project_post_generation_cuda_checkpoint",
    "restore_cuda_boundary_checkpoint",
    "validate_phase_e_passive_modular_no_churn_guard",
]
