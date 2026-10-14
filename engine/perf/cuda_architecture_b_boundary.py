"""Strict one-cell CPU-owner to CUDA lifecycle-boundary transport."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Final, Mapping

import torch

from native_runtime.lifecycle_events import FIELD_DTYPES
from perf.cuda_architecture_b_modular_state_bank import (
    ModularStateBankError,
    validate_modular_cpu_boundary_projection,
    validate_modular_resident_state_bank,
)
from perf.cuda_architecture_b_pcg64 import pcg64_state_tensors
from perf.cxxbackbone10_resident_state_owner import (
    EVENT_TAPE_TENSOR_NAMES,
    SCHEMA_VERSION as IMPORT_PROJECTION_SCHEMA_VERSION,
    TYPED_LIFECYCLE_EVENT_TENSOR_NAMES,
    projection_digest,
)


BOUNDARY_IMPORT_SCHEMA_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_boundary_import_v1"
)
CHECKPOINT_PROJECTION_SCHEMA_VERSION: Final[str] = (
    "cxxbackbone10_checkpoint_projection_v2"
)
SUPPORTED_SOURCE_SCHEMAS: Final[frozenset[str]] = frozenset(
    {
        IMPORT_PROJECTION_SCHEMA_VERSION,
        CHECKPOINT_PROJECTION_SCHEMA_VERSION,
    }
)

TYPED_EVENT_PREFIX: Final[str] = "typed_lifecycle_event_"
EVENT_COLUMN_NAMES: Final[tuple[str, ...]] = (
    *EVENT_TAPE_TENSOR_NAMES,
    *TYPED_LIFECYCLE_EVENT_TENSOR_NAMES,
)
EVENT_COLUMN_DTYPES: Final[dict[str, torch.dtype]] = {
    "event_tape_type": torch.int32,
    "event_tape_generation": torch.int64,
    "event_tape_tick": torch.int64,
    "event_tape_actor_id": torch.int64,
    "event_tape_target_id": torch.int64,
    "event_tape_value": torch.float32,
    **{
        f"{TYPED_EVENT_PREFIX}{field_name}": dtype
        for field_name, dtype in FIELD_DTYPES.items()
    },
}

POLICY_GENERATION_ALIASES: Final[dict[str, str]] = {
    "attack_bias": "policy_attack_bias",
    "action_temperature": "policy_action_temperature",
    "drive_logit_gain": "policy_drive_logit_gain",
}

GENERATION_PATCH_TENSOR_FILL_VALUES: Final[dict[str, float]] = {
    "patch_food_amount": 0.0,
    "patch_fertility": 1.0,
    "patch_cover": 0.0,
    "patch_shelter": 0.0,
    "patch_scent": 0.0,
    "patch_waste": 0.0,
    "patch_density": 0.0,
    "patch_scent_emit_pending": 0.0,
    "patch_waste_emit_pending": 0.0,
    "patch_movement_cost": 1.0,
}

PCG64_METADATA_STATES: Final[dict[str, str]] = {
    "env": "native_runtime_numpy_env_state",
    "runner": "native_runtime_numpy_runner_state",
    "policy": "native_runtime_numpy_policy_state",
    "population": "native_runtime_numpy_population_state",
    "birth_spawn_env": "native_runtime_numpy_birth_spawn_env_state",
}

_RESERVED_CONSTRUCTED_TENSOR_NAMES: Final[frozenset[str]] = frozenset(
    {
        "event_tape_active_count",
        "event_tape_cursor",
        "event_tape_status",
        "pending_free_agent_ids",
        "pending_free_slots",
        "pending_free_count",
        "pcg64_state_words",
        "pcg64_buffer_words",
        *(
            f"{stream}_pcg64_{suffix}"
            for stream in PCG64_METADATA_STATES
            for suffix in ("state_words", "buffer_words")
        ),
    }
)

_MANIFEST_DTYPE_NAMES: Final[dict[torch.dtype, frozenset[str]]] = {
    torch.bool: frozenset({"Bool", "bool", "torch.bool"}),
    torch.uint8: frozenset({"Byte", "uint8", "torch.uint8"}),
    torch.int8: frozenset({"Char", "int8", "torch.int8"}),
    torch.int16: frozenset({"Short", "int16", "torch.int16"}),
    torch.int32: frozenset({"Int", "int32", "torch.int32"}),
    torch.int64: frozenset({"Long", "int64", "torch.int64"}),
    torch.float16: frozenset({"Half", "float16", "torch.float16"}),
    torch.bfloat16: frozenset({"BFloat16", "bfloat16", "torch.bfloat16"}),
    torch.float32: frozenset({"Float", "float32", "torch.float32"}),
    torch.float64: frozenset({"Double", "float64", "torch.float64"}),
}


class CudaBoundaryImportError(RuntimeError):
    """Raised before transfer when the CPU projection is contradictory."""


@dataclass(frozen=True)
class CudaBoundaryImportReceipt:
    schema_version: str
    source_schema_version: str
    source_projection_digest: str
    target_device: str
    source_tensor_count: int
    preserved_source_tensor_count: int
    source_event_column_capacity: int
    resident_event_column_capacity: int
    active_event_count: int
    patch_tensor_mode: str
    transferred_tensor_names: tuple[str, ...]
    added_tensor_names: tuple[str, ...]
    host_to_device_operations: int
    host_to_device_bytes: int
    device_synchronizations: int
    simulation_executed: bool = False
    checkpoint_closure_claimed: bool = False
    parity_closure_claimed: bool = False
    multicell_claimed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_schema_version": self.source_schema_version,
            "source_projection_digest": self.source_projection_digest,
            "target_device": self.target_device,
            "source_tensor_count": self.source_tensor_count,
            "preserved_source_tensor_count": self.preserved_source_tensor_count,
            "source_event_column_capacity": self.source_event_column_capacity,
            "resident_event_column_capacity": self.resident_event_column_capacity,
            "active_event_count": self.active_event_count,
            "patch_tensor_mode": self.patch_tensor_mode,
            "transferred_tensor_names": list(self.transferred_tensor_names),
            "added_tensor_names": list(self.added_tensor_names),
            "host_to_device_operations": self.host_to_device_operations,
            "host_to_device_bytes": self.host_to_device_bytes,
            "device_synchronizations": self.device_synchronizations,
            "simulation_executed": self.simulation_executed,
            "checkpoint_closure_claimed": self.checkpoint_closure_claimed,
            "parity_closure_claimed": self.parity_closure_claimed,
            "multicell_claimed": self.multicell_claimed,
        }


@dataclass(frozen=True)
class CudaBoundaryImport:
    schema_version: str
    metadata: dict[str, Any]
    tensors: dict[str, torch.Tensor]
    host_authoritative_tensors: dict[str, torch.Tensor]
    pending_generation_free_agent_ids: tuple[int, ...]
    pending_generation_free_slots: tuple[int, ...]
    receipt: CudaBoundaryImportReceipt


@dataclass(frozen=True)
class _ProjectionParts:
    schema_version: str
    metadata: dict[str, Any]
    tensors: dict[str, torch.Tensor]
    pending_agent_ids: tuple[int, ...]
    pending_slots: tuple[int, ...]
    tensor_manifest: Mapping[str, Any] | None
    top_level_event_cursor: Any | None


class _TransferRecorder:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.operations = 0
        self.bytes = 0
        self.names: list[str] = []

    def copy(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device.type != "cpu":
            raise CudaBoundaryImportError(
                f"host-to-device source tensor must be CPU-resident: {name}"
            )
        self.operations += 1
        self.bytes += _tensor_bytes(tensor)
        self.names.append(name)
        return tensor.detach().to(device=self.device).contiguous()


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _strict_int(value: Any, *, name: str) -> int:
    if type(value) is not int:
        raise CudaBoundaryImportError(f"{name} must be an exact integer")
    return int(value)


def _tuple_of_ints(value: Any, *, name: str) -> tuple[int, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise CudaBoundaryImportError(f"{name} must be a list or tuple")
    result: list[int] = []
    for index, item in enumerate(value):
        if type(item) is not int:
            raise CudaBoundaryImportError(
                f"{name}[{index}] must be an exact integer"
            )
        result.append(int(item))
    return tuple(result)


def _projection_parts(projection: Any) -> _ProjectionParts:
    if isinstance(projection, Mapping):
        schema_version = projection.get("schema_version")
        metadata = projection.get("metadata")
        tensors = projection.get("tensors")
        pending_agent_ids = projection.get("pending_generation_free_agent_ids", ())
        pending_slots = projection.get("pending_generation_free_slots", ())
        tensor_manifest = projection.get("tensor_manifest")
        top_level_event_cursor = projection.get("event_cursor")
    else:
        schema_version = getattr(projection, "schema_version", None)
        metadata = getattr(projection, "metadata", None)
        tensors = getattr(projection, "tensors", None)
        pending_agent_ids = getattr(
            projection, "pending_generation_free_agent_ids", ()
        )
        pending_slots = getattr(projection, "pending_generation_free_slots", ())
        tensor_manifest = getattr(projection, "tensor_manifest", None)
        top_level_event_cursor = getattr(projection, "event_cursor", None)

    if type(schema_version) is not str or schema_version not in SUPPORTED_SOURCE_SCHEMAS:
        raise CudaBoundaryImportError(
            f"unsupported CPU owner projection schema: {schema_version!r}"
        )
    if not isinstance(metadata, Mapping):
        raise CudaBoundaryImportError("CPU owner projection metadata must be a mapping")
    if not isinstance(tensors, Mapping) or not tensors:
        raise CudaBoundaryImportError(
            "CPU owner projection tensors must be a non-empty mapping"
        )
    if tensor_manifest is not None and not isinstance(tensor_manifest, Mapping):
        raise CudaBoundaryImportError("CPU owner tensor_manifest must be a mapping")
    if schema_version == CHECKPOINT_PROJECTION_SCHEMA_VERSION and tensor_manifest is None:
        raise CudaBoundaryImportError(
            "CPU owner checkpoint projection requires tensor_manifest"
        )
    return _ProjectionParts(
        schema_version=schema_version,
        metadata=deepcopy(dict(metadata)),
        tensors=dict(tensors),
        pending_agent_ids=_tuple_of_ints(
            pending_agent_ids,
            name="pending_generation_free_agent_ids",
        ),
        pending_slots=_tuple_of_ints(
            pending_slots,
            name="pending_generation_free_slots",
        ),
        tensor_manifest=tensor_manifest,
        top_level_event_cursor=top_level_event_cursor,
    )


def _validate_source_tensors(parts: _ProjectionParts) -> None:
    for name, tensor in parts.tensors.items():
        if type(name) is not str or not name:
            raise CudaBoundaryImportError("CPU owner tensor names must be non-empty strings")
        if not isinstance(tensor, torch.Tensor):
            raise CudaBoundaryImportError(f"CPU owner field is not a tensor: {name}")
        if tensor.device.type != "cpu":
            raise CudaBoundaryImportError(
                f"CPU owner tensor must be CPU-resident before transfer: {name}"
            )
        if not tensor.is_contiguous():
            raise CudaBoundaryImportError(
                f"CPU owner tensor must be contiguous before transfer: {name}"
            )
        if tensor.requires_grad:
            raise CudaBoundaryImportError(
                f"CPU owner tensor must not require gradients: {name}"
            )
    collisions = sorted(_RESERVED_CONSTRUCTED_TENSOR_NAMES.intersection(parts.tensors))
    if collisions:
        raise CudaBoundaryImportError(
            "CPU owner projection contradicts CUDA-constructed tensor ownership: "
            + ", ".join(collisions)
        )


def _validate_tensor_manifest(parts: _ProjectionParts) -> None:
    if parts.tensor_manifest is None:
        return
    manifest = dict(parts.tensor_manifest)
    if set(manifest) != set(parts.tensors):
        missing = sorted(set(parts.tensors) - set(manifest))
        extra = sorted(set(manifest) - set(parts.tensors))
        raise CudaBoundaryImportError(
            "CPU owner tensor_manifest names disagree with tensors: "
            f"missing={missing} extra={extra}"
        )
    for name, tensor in parts.tensors.items():
        row = manifest[name]
        if not isinstance(row, Mapping):
            raise CudaBoundaryImportError(f"tensor_manifest row is invalid: {name}")
        expected_shape = [int(dim) for dim in tensor.shape]
        if list(row.get("shape", ())) != expected_shape:
            raise CudaBoundaryImportError(f"tensor_manifest shape mismatch: {name}")
        dtype_name = str(row.get("dtype", ""))
        if dtype_name not in _MANIFEST_DTYPE_NAMES.get(tensor.dtype, frozenset()):
            raise CudaBoundaryImportError(f"tensor_manifest dtype mismatch: {name}")
        exact_fields = {
            "numel": int(tensor.numel()),
            "element_size": int(tensor.element_size()),
            "bytes": _tensor_bytes(tensor),
            "is_cuda": False,
            "is_contiguous": True,
        }
        for field_name, expected in exact_fields.items():
            if row.get(field_name) != expected:
                raise CudaBoundaryImportError(
                    f"tensor_manifest {field_name} mismatch: {name}"
                )
        if str(row.get("device", "")).split(":", 1)[0] != "cpu":
            raise CudaBoundaryImportError(f"tensor_manifest device mismatch: {name}")


def _validate_boundary_state(
    parts: _ProjectionParts,
    policy_family: str,
    *,
    allow_partial_generation_source_cursor: bool,
) -> int:
    metadata = parts.metadata
    generation = _strict_int(metadata.get("generation"), name="metadata.generation")
    tick = _strict_int(metadata.get("tick"), name="metadata.tick")
    configured_ticks = _strict_int(
        metadata.get("configured_steps_per_generation"),
        name="metadata.configured_steps_per_generation",
    )
    if (
        generation < 0
        or configured_ticks < 1
        or not 0 <= tick < configured_ticks
        or (tick != 0 and not allow_partial_generation_source_cursor)
    ):
        raise CudaBoundaryImportError(
            "CUDA import accepts only a non-negative generation at tick 0"
        )
    expected_phase = "generation_boundary" if tick == 0 else "partial_generation"
    if metadata.get("agent_grid_runner_phase") != expected_phase:
        raise CudaBoundaryImportError(
            "CPU owner projection phase contradicts its exact tick cursor"
        )
    if metadata.get("agent_grid_synchronization_state") != "stored_grid_authoritative":
        raise CudaBoundaryImportError(
            "CPU owner agent_grid is not the stored authoritative boundary view"
        )
    if metadata.get("world") not in {"flat", "patch"}:
        raise CudaBoundaryImportError("CPU owner world must be flat or patch")
    if policy_family not in {"fixed_mlp", "stable_gru"}:
        raise CudaBoundaryImportError("CUDA boundary policy_family is unsupported")
    metadata_family = str(
        metadata.get("native_runtime_fixed_policy_family", "")
    ).strip().lower()
    expected_metadata_family = "mlp" if policy_family == "fixed_mlp" else "gru"
    if metadata_family != expected_metadata_family:
        raise CudaBoundaryImportError(
            "CUDA boundary policy family disagrees with CPU owner metadata: "
            f"expected={expected_metadata_family} actual={metadata_family or '<empty>'}"
        )

    capacity = _strict_int(metadata.get("capacity"), name="metadata.capacity")
    if capacity <= 0:
        raise CudaBoundaryImportError("CPU owner capacity must be positive")
    for name in (
        "slot_to_agent_id",
        "alive_mask",
        "policy_attack_bias",
        "policy_action_temperature",
        "policy_drive_logit_gain",
        "policy_use_gru",
    ):
        tensor = parts.tensors.get(name)
        if tensor is None or tensor.ndim != 1 or int(tensor.shape[0]) != capacity:
            raise CudaBoundaryImportError(
                f"CPU owner capacity tensor has an invalid shape: {name}"
            )
    alive_mask = parts.tensors["alive_mask"]
    slot_to_agent_id = parts.tensors["slot_to_agent_id"]
    if alive_mask.dtype != torch.bool or slot_to_agent_id.dtype != torch.int64:
        raise CudaBoundaryImportError("CPU owner slot identity dtypes are invalid")
    occupied = slot_to_agent_id.ge(0)
    if not torch.equal(alive_mask, occupied):
        raise CudaBoundaryImportError(
            "alive_mask and slot_to_agent_id occupancy disagree"
        )
    alive_count = _strict_int(metadata.get("alive_count"), name="metadata.alive_count")
    if alive_count != int(alive_mask.sum().item()):
        raise CudaBoundaryImportError("metadata.alive_count disagrees with alive_mask")
    live_gru = parts.tensors["policy_use_gru"][alive_mask]
    if policy_family == "fixed_mlp" and bool(torch.any(live_gru).item()):
        raise CudaBoundaryImportError("fixed-MLP boundary contains a live GRU row")
    if policy_family == "stable_gru":
        if not bool(torch.all(live_gru).item()):
            raise CudaBoundaryImportError("stable-GRU boundary contains a live non-GRU row")
        gru_pack = parts.tensors.get("gru_pack")
        hidden = parts.tensors.get("hmem")
        memory_dim = parts.tensors.get("policy_mem_dim")
        if (
            gru_pack is None or gru_pack.dtype != torch.float32
            or gru_pack.ndim != 2 or int(gru_pack.shape[0]) != capacity
            or int(gru_pack.shape[1]) <= 0
        ):
            raise CudaBoundaryImportError("stable-GRU boundary has invalid gru_pack")
        if (
            hidden is None or hidden.dtype != torch.float32
            or tuple(hidden.shape) != (capacity, 64)
        ):
            raise CudaBoundaryImportError("stable-GRU boundary has invalid hmem")
        if (
            memory_dim is None or memory_dim.dtype != torch.int64
            or tuple(memory_dim.shape) != (capacity,)
        ):
            raise CudaBoundaryImportError("stable-GRU boundary has invalid policy_mem_dim")
        accepted_width = torch.logical_or(
            memory_dim.eq(0),
            torch.logical_or(
                memory_dim.eq(8),
                torch.logical_or(
                    memory_dim.eq(16),
                    torch.logical_or(memory_dim.eq(32), memory_dim.eq(64)),
                ),
            ),
        )
        if not bool(torch.all(accepted_width[alive_mask]).item()):
            raise CudaBoundaryImportError(
                "stable-GRU boundary has unsupported live memory width"
            )

    for alias, canonical in POLICY_GENERATION_ALIASES.items():
        candidate = parts.tensors.get(alias)
        if candidate is not None and not torch.equal(candidate, parts.tensors[canonical]):
            raise CudaBoundaryImportError(
                f"CPU owner policy alias contradicts canonical tensor: {alias}"
            )
    return capacity


def _validate_generation_patch_state(
    parts: _ProjectionParts,
) -> tuple[str, tuple[int, int]]:
    agent_grid = parts.tensors.get("agent_grid")
    food_grid = parts.tensors.get("food_grid")
    if (
        agent_grid is None
        or agent_grid.dtype != torch.int32
        or agent_grid.ndim != 2
    ):
        raise CudaBoundaryImportError(
            "CPU owner agent_grid must be a rank-2 int32 tensor"
        )
    world_shape = tuple(int(dim) for dim in agent_grid.shape)
    if (
        food_grid is None
        or food_grid.dtype != torch.int32
        or tuple(food_grid.shape) != world_shape
    ):
        raise CudaBoundaryImportError(
            "CPU owner food_grid must be world-shaped int32"
        )

    present = {
        name for name in GENERATION_PATCH_TENSOR_FILL_VALUES if name in parts.tensors
    }
    world = parts.metadata["world"]
    if world == "flat":
        if present:
            raise CudaBoundaryImportError(
                "Flat CPU owner projection must not carry authoritative Patch tensors: "
                + ", ".join(sorted(present))
            )
        return "flat_neutral_device_synthesized", world_shape

    missing = sorted(set(GENERATION_PATCH_TENSOR_FILL_VALUES) - present)
    if missing:
        raise CudaBoundaryImportError(
            "Patch CPU owner projection is missing authoritative Patch tensors: "
            + ", ".join(missing)
        )
    for name in GENERATION_PATCH_TENSOR_FILL_VALUES:
        tensor = parts.tensors[name]
        if tensor.dtype != torch.float32 or tuple(tensor.shape) != world_shape:
            raise CudaBoundaryImportError(
                f"authoritative Patch tensor must be world-shaped float32: {name}"
            )
        if not bool(torch.all(torch.isfinite(tensor)).item()):
            raise CudaBoundaryImportError(
                f"authoritative Patch tensor contains non-finite values: {name}"
            )
    return "patch_authoritative_preserved", world_shape


def _validate_event_state(parts: _ProjectionParts) -> tuple[int, int, int, bool]:
    missing = [name for name in EVENT_COLUMN_NAMES if name not in parts.tensors]
    if missing:
        raise CudaBoundaryImportError(
            "CPU owner projection has incomplete lifecycle event columns: "
            + ", ".join(missing)
        )
    lengths: set[int] = set()
    for name in EVENT_COLUMN_NAMES:
        tensor = parts.tensors[name]
        if tensor.ndim != 1:
            raise CudaBoundaryImportError(f"lifecycle event column must be rank 1: {name}")
        if tensor.dtype != EVENT_COLUMN_DTYPES[name]:
            raise CudaBoundaryImportError(f"lifecycle event column dtype mismatch: {name}")
        lengths.add(int(tensor.numel()))
    if len(lengths) != 1:
        raise CudaBoundaryImportError(
            "legacy and typed lifecycle event columns have inconsistent lengths"
        )
    source_capacity = next(iter(lengths))
    metadata = parts.metadata
    active_count = _strict_int(
        metadata.get("event_tape_active_count"),
        name="metadata.event_tape_active_count",
    )
    cursor = _strict_int(
        metadata.get("event_tape_cursor"),
        name="metadata.event_tape_cursor",
    )
    typed_count = _strict_int(
        metadata.get("typed_lifecycle_event_active_count"),
        name="metadata.typed_lifecycle_event_active_count",
    )
    next_sequence = _strict_int(
        metadata.get("typed_lifecycle_event_next_sequence"),
        name="metadata.typed_lifecycle_event_next_sequence",
    )
    external_history = metadata.get("event_history_storage_mode") == (
        "eiel_resident_event_sink_v1"
    )
    sequence_base = _strict_int(
        metadata.get("typed_lifecycle_event_sequence_base", 0),
        name="metadata.typed_lifecycle_event_sequence_base",
    )
    inactive_tail = _strict_int(
        metadata.get("event_tape_inactive_tail_count"),
        name="metadata.event_tape_inactive_tail_count",
    )
    if not (
        0 <= active_count <= source_capacity
        and cursor == active_count
        and typed_count == active_count
        and next_sequence == sequence_base + active_count
        and inactive_tail == source_capacity - active_count
    ):
        raise CudaBoundaryImportError(
            "lifecycle event active count, cursor, sequence, or inactive tail disagree"
        )
    if parts.top_level_event_cursor is not None:
        top_level_cursor = _strict_int(
            parts.top_level_event_cursor,
            name="projection.event_cursor",
        )
        if top_level_cursor != active_count:
            raise CudaBoundaryImportError(
                "checkpoint event_cursor disagrees with lifecycle event active count"
            )

    active = slice(0, active_count)
    sequence = parts.tensors["typed_lifecycle_event_sequence"][active]
    if not torch.equal(
        sequence,
        torch.arange(
            sequence_base,
            sequence_base + active_count,
            dtype=torch.int64,
        ),
    ):
        raise CudaBoundaryImportError(
            "typed lifecycle active sequence is not contiguous and zero-based"
        )
    compatibility_pairs = (
        ("typed_lifecycle_event_event_type", "event_tape_type"),
        ("typed_lifecycle_event_generation", "event_tape_generation"),
        ("typed_lifecycle_event_tick", "event_tape_tick"),
        ("typed_lifecycle_event_actor_agent_id", "event_tape_actor_id"),
        ("typed_lifecycle_event_target_agent_id", "event_tape_target_id"),
        ("typed_lifecycle_event_value", "event_tape_value"),
    )
    for typed_name, legacy_name in compatibility_pairs:
        if not torch.equal(
            parts.tensors[typed_name][active],
            parts.tensors[legacy_name][active],
        ):
            raise CudaBoundaryImportError(
                f"typed and legacy active event prefixes disagree: {typed_name}"
            )
    if external_history and metadata.get("event_history_frontier") != next_sequence:
        raise CudaBoundaryImportError(
            "external event-history frontier disagrees with next sequence"
        )
    if not external_history and sequence_base != 0:
        raise CudaBoundaryImportError(
            "non-external lifecycle event history must use a zero sequence base"
        )
    return active_count, source_capacity, sequence_base, external_history


def _validate_pending_free_state(parts: _ProjectionParts, capacity: int) -> None:
    if len(parts.pending_agent_ids) != len(parts.pending_slots):
        raise CudaBoundaryImportError(
            "pending-generation-free agent and slot counts disagree"
        )
    if len(parts.pending_slots) > capacity:
        raise CudaBoundaryImportError("pending-generation-free count exceeds capacity")
    if len(set(parts.pending_agent_ids)) != len(parts.pending_agent_ids):
        raise CudaBoundaryImportError("pending-generation-free agent ids are duplicated")
    if len(set(parts.pending_slots)) != len(parts.pending_slots):
        raise CudaBoundaryImportError("pending-generation-free slots are duplicated")
    alive_mask = parts.tensors["alive_mask"]
    slot_to_agent_id = parts.tensors["slot_to_agent_id"]
    for agent_id, slot in zip(
        parts.pending_agent_ids,
        parts.pending_slots,
        strict=True,
    ):
        if agent_id < 0 or slot < 0 or slot >= capacity:
            raise CudaBoundaryImportError(
                "pending-generation-free entries must contain valid non-negative values"
            )
        if bool(alive_mask[slot].item()) or int(slot_to_agent_id[slot].item()) >= 0:
            raise CudaBoundaryImportError(
                "pending-generation-free slot contradicts live slot occupancy"
            )


def _validate_drive_noise_state(parts: _ProjectionParts, capacity: int) -> None:
    samples = parts.tensors.get("drive_noise_samples_by_tick_slot")
    valid = parts.tensors.get("drive_noise_samples_valid_by_tick_slot")
    precomputed = parts.metadata.get("native_runtime_use_precomputed_drive_noise")
    if type(precomputed) is not bool:
        raise CudaBoundaryImportError(
            "native_runtime_use_precomputed_drive_noise must be an exact bool"
        )
    if (samples is None) != (valid is None):
        raise CudaBoundaryImportError("precomputed drive-noise tensor pair is incomplete")
    if precomputed != (samples is not None):
        raise CudaBoundaryImportError(
            "precomputed drive-noise metadata and tensor presence disagree"
        )
    if samples is None or valid is None:
        return
    if (
        samples.dtype != torch.float32
        or samples.ndim != 3
        or tuple(samples.shape[1:]) != (capacity, 4)
    ):
        raise CudaBoundaryImportError(
            "drive_noise_samples_by_tick_slot must be float32[tick, capacity, 4]"
        )
    if (
        valid.dtype != torch.bool
        or valid.ndim != 2
        or tuple(valid.shape) != tuple(samples.shape[:2])
    ):
        raise CudaBoundaryImportError(
            "drive_noise_samples_valid_by_tick_slot must match sample rows"
        )


def _validate_pcg64_metadata(parts: _ProjectionParts) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    cpu_states: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for stream, metadata_name in PCG64_METADATA_STATES.items():
        state = parts.metadata.get(metadata_name)
        if not isinstance(state, Mapping):
            raise CudaBoundaryImportError(
                f"authoritative PCG64 metadata state is missing: {metadata_name}"
            )
        try:
            cpu_states[stream] = pcg64_state_tensors(state, device="cpu")
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise CudaBoundaryImportError(
                f"authoritative PCG64 metadata state is invalid: {metadata_name}"
            ) from exc
    alias = parts.metadata.get("native_runtime_env_policy_numpy_alias")
    if type(alias) is not bool:
        raise CudaBoundaryImportError(
            "native_runtime_env_policy_numpy_alias must be an exact bool"
        )
    if alias and parts.metadata["native_runtime_numpy_env_state"] != parts.metadata[
        "native_runtime_numpy_policy_state"
    ]:
        raise CudaBoundaryImportError(
            "environment/policy NumPy alias is true but authoritative states differ"
        )
    return cpu_states


def _validate_source_digest(parts: _ProjectionParts) -> str:
    digest = projection_digest(
        {
            "metadata": parts.metadata,
            "tensors": parts.tensors,
        }
    )
    declared = parts.metadata.get("import_digest")
    if (
        parts.schema_version == IMPORT_PROJECTION_SCHEMA_VERSION
        and declared is not None
        and declared != digest
    ):
        raise CudaBoundaryImportError(
            "CPU owner import projection digest does not match its tensors and metadata"
        )
    return digest


def _event_fill_value(dtype: torch.dtype) -> int | float:
    return 0.0 if dtype.is_floating_point else -1


def import_cpu_owner_boundary_to_cuda(
    projection: Any,
    *,
    event_capacity: int,
    device: str | torch.device = "cuda",
    policy_family: str = "fixed_mlp",
    allow_cpu_staging: bool = False,
    allow_partial_generation_source_cursor: bool = False,
    host_authoritative_tensor_names: tuple[str, ...] = (),
    drain_event_history: bool = False,
) -> CudaBoundaryImport:
    """Validate and transport one CPU-owner generation boundary to CUDA.

    This function performs no simulation and makes no parity or checkpoint
    closure claim. The event tape is the only accepted source tensor surface
    whose inactive capacity is intentionally rematerialized.
    """

    target = torch.device(device)
    if type(allow_cpu_staging) is not bool:
        raise CudaBoundaryImportError("allow_cpu_staging must be an exact bool")
    if type(allow_partial_generation_source_cursor) is not bool:
        raise CudaBoundaryImportError(
            "allow_partial_generation_source_cursor must be an exact bool"
        )
    if type(drain_event_history) is not bool:
        raise CudaBoundaryImportError(
            "drain_event_history must be an exact bool"
        )
    if target.type not in {"cuda", "cpu"}:
        raise CudaBoundaryImportError("CUDA boundary target must be a CUDA or CPU device")
    if target.type == "cpu" and not allow_cpu_staging:
        raise CudaBoundaryImportError("CPU boundary staging requires explicit admission")
    if target.type == "cuda" and not torch.cuda.is_available():
        raise CudaBoundaryImportError("CUDA boundary target is unavailable")
    if type(event_capacity) is not int or event_capacity < 0:
        raise CudaBoundaryImportError("event_capacity must be a non-negative integer")
    if (
        not isinstance(host_authoritative_tensor_names, tuple)
        or any(type(name) is not str or not name for name in host_authoritative_tensor_names)
        or len(set(host_authoritative_tensor_names)) != len(host_authoritative_tensor_names)
    ):
        raise CudaBoundaryImportError(
            "host-authoritative tensor names must be one unique non-empty text tuple"
        )

    parts = _projection_parts(projection)
    _validate_source_tensors(parts)
    _validate_tensor_manifest(parts)
    capacity = _validate_boundary_state(
        parts,
        policy_family,
        allow_partial_generation_source_cursor=(
            allow_partial_generation_source_cursor
        ),
    )
    try:
        modular_overlay = validate_modular_cpu_boundary_projection(
            parts.metadata,
            parts.tensors,
            agent_capacity=capacity,
        )
    except ModularStateBankError as exc:
        raise CudaBoundaryImportError(str(exc)) from exc
    patch_tensor_mode, world_shape = _validate_generation_patch_state(parts)
    (
        source_active_count,
        source_event_capacity,
        source_sequence_base,
        external_history,
    ) = _validate_event_state(parts)
    active_count = 0 if drain_event_history else source_active_count
    resident_sequence_base = (
        source_sequence_base + source_active_count
        if external_history and drain_event_history
        else source_sequence_base
    )
    if not drain_event_history and event_capacity < active_count:
        raise CudaBoundaryImportError(
            "requested resident event capacity is smaller than the active prefix"
        )
    _validate_pending_free_state(parts, capacity)
    _validate_drive_noise_state(parts, capacity)
    cpu_pcg64_states = _validate_pcg64_metadata(parts)
    source_digest = _validate_source_digest(parts)
    host_authoritative_names = set(host_authoritative_tensor_names)
    unknown_host_authoritative = host_authoritative_names - set(parts.tensors)
    if unknown_host_authoritative:
        raise CudaBoundaryImportError(
            "host-authoritative tensor names are absent from the CPU projection: "
            + ", ".join(sorted(unknown_host_authoritative))
        )

    recorder = _TransferRecorder(target)
    resident: dict[str, torch.Tensor] = {}
    host_authoritative: dict[str, torch.Tensor] = {}
    event_names = set(EVENT_COLUMN_NAMES)
    for name, tensor in parts.tensors.items():
        if name not in event_names:
            if name in host_authoritative_names:
                host_authoritative[name] = tensor
                continue
            resident[name] = recorder.copy(name, tensor)

    # RNG V2 checkpoints contain coordinates, not future random values.  The
    # fixed native full-tick ABI still has two drive-noise tensor positions, so
    # synthesize zero-sized device placeholders after the validated import
    # boundary.  No host payload is allocated or transferred.
    if parts.metadata["native_runtime_use_precomputed_drive_noise"] is False:
        resident["drive_noise_samples_by_tick_slot"] = torch.empty(
            (0, capacity, 4), dtype=torch.float32, device=target
        )
        resident["drive_noise_samples_valid_by_tick_slot"] = torch.empty(
            (0, capacity), dtype=torch.bool, device=target
        )

    # Checkpoint v2 stores the sparse agent-id map to keep the canonical
    # projection compact, while the legacy generation ABI still consumes the
    # dense compatibility view.  Reconstruct that view on the target device;
    # it is derived transport state, not another host staging allocation.
    if "agent_id_to_slot" not in resident:
        sparse_ids = resident.get("agent_id_to_slot_agent_ids")
        sparse_slots = resident.get("agent_id_to_slot_slots")
        if isinstance(sparse_ids, torch.Tensor) and isinstance(
            sparse_slots, torch.Tensor
        ):
            dense_map = torch.full(
                (capacity,),
                -1,
                dtype=torch.int32,
                device=target,
            )
            if sparse_ids.numel():
                dense_map[sparse_ids.to(torch.int64)] = sparse_slots.to(torch.int32)
            resident["agent_id_to_slot"] = dense_map

    if patch_tensor_mode == "flat_neutral_device_synthesized":
        for name, fill_value in GENERATION_PATCH_TENSOR_FILL_VALUES.items():
            resident[name] = torch.full(
                world_shape,
                fill_value,
                dtype=torch.float32,
                device=target,
            )

    active = slice(0, active_count)
    for name in EVENT_COLUMN_NAMES:
        dtype = EVENT_COLUMN_DTYPES[name]
        column = torch.full(
            (event_capacity,),
            _event_fill_value(dtype),
            dtype=dtype,
            device=target,
        )
        if active_count:
            prefix = recorder.copy(name, parts.tensors[name][active].contiguous())
            column[active].copy_(prefix)
        resident[name] = column

    active_count_cpu = torch.tensor([active_count], dtype=torch.int64)
    event_status_cpu = torch.zeros((1,), dtype=torch.int32)
    resident_active_count = recorder.copy(
        "event_tape_active_count", active_count_cpu
    )
    resident["event_tape_active_count"] = resident_active_count
    resident["event_tape_cursor"] = resident_active_count
    resident["event_tape_status"] = recorder.copy(
        "event_tape_status", event_status_cpu
    )
    resident["event_sequence_base"] = torch.tensor(
        [resident_sequence_base], dtype=torch.int64, device=target
    )

    pending_agent_ids_cpu = torch.full((capacity,), -1, dtype=torch.int64)
    pending_slots_cpu = torch.full((capacity,), -1, dtype=torch.int64)
    pending_count = len(parts.pending_slots)
    if pending_count:
        pending_agent_ids_cpu[:pending_count] = torch.tensor(
            parts.pending_agent_ids,
            dtype=torch.int64,
        )
        pending_slots_cpu[:pending_count] = torch.tensor(
            parts.pending_slots,
            dtype=torch.int64,
        )
    resident["pending_free_agent_ids"] = recorder.copy(
        "pending_free_agent_ids", pending_agent_ids_cpu
    )
    resident["pending_free_slots"] = recorder.copy(
        "pending_free_slots", pending_slots_cpu
    )
    resident["pending_free_count"] = recorder.copy(
        "pending_free_count",
        torch.tensor([pending_count], dtype=torch.int64),
    )

    for stream, (state_words, buffer_words) in cpu_pcg64_states.items():
        resident[f"{stream}_pcg64_state_words"] = recorder.copy(
            f"{stream}_pcg64_state_words",
            state_words,
        )
        resident[f"{stream}_pcg64_buffer_words"] = recorder.copy(
            f"{stream}_pcg64_buffer_words",
            buffer_words,
        )
    resident["pcg64_state_words"] = resident["policy_pcg64_state_words"]
    resident["pcg64_buffer_words"] = resident["policy_pcg64_buffer_words"]

    for alias, canonical in POLICY_GENERATION_ALIASES.items():
        resident[alias] = resident[canonical]

    modular_receipt = None
    if modular_overlay is not None:
        try:
            modular_receipt = validate_modular_resident_state_bank(
                modular_overlay,
                resident,
                device=target,
            )
        except ModularStateBankError as exc:
            raise CudaBoundaryImportError(str(exc)) from exc

    metadata = parts.metadata
    source_import_digest = metadata.pop("import_digest", None)
    if source_import_digest is not None:
        metadata["cuda_boundary_source_import_digest"] = source_import_digest
    metadata.update(
        {
            "cuda_boundary_import_schema_version": BOUNDARY_IMPORT_SCHEMA_VERSION,
            "cuda_boundary_source_projection_digest": source_digest,
            "cuda_boundary_target_device": str(target),
            "cuda_boundary_patch_tensor_contract_schema_version": (
                "eiel_cuda_architecture_b_flat_patch_generation_tensors_v1"
            ),
            "cuda_boundary_patch_tensor_mode": patch_tensor_mode,
            "event_tape_active_count": active_count,
            "event_tape_cursor": active_count,
            "event_tape_inactive_tail_count": event_capacity - active_count,
            "typed_lifecycle_event_active_count": active_count,
            "typed_lifecycle_event_sequence_base": resident_sequence_base,
            "typed_lifecycle_event_next_sequence": (
                resident_sequence_base + active_count
            ),
            "event_history_frontier": (
                resident_sequence_base + active_count
                if external_history
                else metadata.get("event_history_frontier", active_count)
            ),
            "cuda_boundary_event_history_drained": drain_event_history,
            "cuda_boundary_event_history_source_active_count": source_active_count,
            "cuda_boundary_simulation_executed": False,
            "cuda_boundary_checkpoint_closure_claimed": False,
            "cuda_boundary_parity_closure_claimed": False,
            "cuda_boundary_multicell_claimed": False,
            "cuda_boundary_host_authoritative_tensor_names": sorted(
                host_authoritative_names
            ),
        }
    )
    if modular_receipt is not None:
        metadata.update(
            {
                "cuda_boundary_modular_manifest_sha256": (
                    modular_receipt.manifest_sha256
                ),
                "cuda_boundary_modular_field_count": modular_receipt.field_count,
                "cuda_boundary_modular_profile_id": modular_receipt.profile_id,
            }
        )

    added_names = tuple(sorted(set(resident) - set(parts.tensors)))
    receipt = CudaBoundaryImportReceipt(
        schema_version=BOUNDARY_IMPORT_SCHEMA_VERSION,
        source_schema_version=parts.schema_version,
        source_projection_digest=source_digest,
        target_device=str(target),
        source_tensor_count=len(parts.tensors),
        preserved_source_tensor_count=len(parts.tensors),
        source_event_column_capacity=source_event_capacity,
        resident_event_column_capacity=event_capacity,
        active_event_count=active_count,
        patch_tensor_mode=patch_tensor_mode,
        transferred_tensor_names=tuple(recorder.names),
        added_tensor_names=added_names,
        # CPU staging deliberately performs no host-to-device transfer.  The
        # recorder still preserves the exact materialized tensor inventory;
        # final batch-shaped CUDA allocation is owned by the source allocator.
        host_to_device_operations=(recorder.operations if target.type == "cuda" else 0),
        host_to_device_bytes=(recorder.bytes if target.type == "cuda" else 0),
        device_synchronizations=0,
    )
    return CudaBoundaryImport(
        schema_version=BOUNDARY_IMPORT_SCHEMA_VERSION,
        metadata=metadata,
        tensors=resident,
        host_authoritative_tensors=host_authoritative,
        pending_generation_free_agent_ids=parts.pending_agent_ids,
        pending_generation_free_slots=parts.pending_slots,
        receipt=receipt,
    )


CUDA_BOUNDARY_IMPORT_CAPABILITIES: Final[dict[str, bool | str]] = {
    "schema_version": "eiel_cuda_architecture_b_boundary_capabilities_v1",
    "decision": "ARCHB-CUDA-STAGE1-ONE-CELL-BOUNDARY-IMPORT",
    "generation_boundary_only": False,
    "partial_generation_boundary_requires_explicit_authority": True,
    "cpu_projection_preflight": "strict_fail_closed",
    "all_source_tensors_preserved": True,
    "route_neutral_event_capacity": True,
    "event_active_prefix_preserved": True,
    "fixed_mlp_policy_alias_translation": True,
    "pending_free_state": True,
    "authoritative_pcg64_metadata_states": True,
    "precomputed_drive_noise_preserved": True,
    "flat_neutral_patch_tensors": "strict_device_synthesis",
    "patch_authoritative_tensors": "strict_preserve_without_normalization",
    "host_to_device_accounting": True,
    "simulation_execution": False,
    "checkpoint_closure": False,
    "parity_closure": False,
    "multicell": False,
    "promotion_qualified": False,
}


__all__ = [
    "BOUNDARY_IMPORT_SCHEMA_VERSION",
    "CHECKPOINT_PROJECTION_SCHEMA_VERSION",
    "CUDA_BOUNDARY_IMPORT_CAPABILITIES",
    "CudaBoundaryImport",
    "CudaBoundaryImportError",
    "CudaBoundaryImportReceipt",
    "EVENT_COLUMN_DTYPES",
    "EVENT_COLUMN_NAMES",
    "GENERATION_PATCH_TENSOR_FILL_VALUES",
    "PCG64_METADATA_STATES",
    "POLICY_GENERATION_ALIASES",
    "import_cpu_owner_boundary_to_cuda",
]
