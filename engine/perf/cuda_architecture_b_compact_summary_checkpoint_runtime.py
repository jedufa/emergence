"""Host binding plan for resident compact-checkpoint tensor digests.

The adapter in this module performs metadata-only validation.  It never reads
tensor values, projects a tensor to the host, or derives missing shape or dtype
contracts from a source tensor.  The first three columns of each pointer spec
are indices into ``tensor_table``; the native binding resolves those indices to
the corresponding ``TensorDigestPointerEntryV1`` pointers.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

import numpy as np
import torch

from engine.native_runtime.compact_summary_checkpoint_descriptor_plan import (
    CUDA_HASH_BATCH_CAPACITY,
    MODE_ACTIVE_PREFIX,
    MODE_CODES,
    MODE_DEAD_ROW_ZERO,
    MODE_EXACT,
    MODE_TORCH_RNG_PACKED_PREFIX,
    PATCH_P2_PROFILE,
    PHASE_E_MODULAR_OVERLAY_PLAN_SCHEMA_VERSION,
    PHASE_E_MODULAR_OVERLAY_STATUS,
    PLAN_SCHEMA_VERSION,
    WORLD_FLAT,
    WORLD_PATCH,
    maybe_generate_phase_e_modular_checkpoint_descriptor_overlay,
    validate_compact_summary_checkpoint_descriptor_plan,
    validate_phase_e_modular_checkpoint_descriptor_overlay,
)


CONTRACT_VERSION_V1: Final[str] = (
    "eiel_cuda_architecture_b_compact_summary_checkpoint_runtime_v1"
)
RUNTIME_SCHEMA_VERSION: Final[str] = CONTRACT_VERSION_V1
READY_FLAT_PLAN_STATUS: Final[str] = "flat_ready_patch_source_projection_pending"
READY_PATCH_P2_PLAN_STATUS: Final[str] = "flat_patch_p2_ready"
LIVE_DEVICE_EXPECTATION: Final[str] = "cuda"
DIGEST_BYTES: Final[int] = 32
NULL_TENSOR_INDEX: Final[int] = -1
PHASE_E_MODULAR_RUNTIME_SCHEMA_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_compact_summary_checkpoint_phase_e_modular_runtime_v1"
)
PHASE_E_MODULAR_STORAGE_DEVICE_EXPECTATION: Final[str] = "cpu_storage_only"

# These names and indices mirror DescriptorSpecColumnV1 in
# cuda_architecture_b_compact_summary_checkpoint_runtime_v1.hpp.  The first
# three fields contain tensor-table indices in this host-side ABI.
DESCRIPTOR_SPEC_COLUMNS_V1: Final[tuple[str, ...]] = (
    "data_tensor_index",
    "alive_mask_tensor_index",
    "active_length_tensor_index",
    "cell_stride_bytes",
    "bytes_per_cell",
    "row_capacity",
    "row_bytes",
    "alive_mask_cell_stride",
    "dtype_bytes",
    "canonicalization_mode",
    "raw_output_index",
    "canonical_output_index",
    "descriptor_index",
)
DESCRIPTOR_SPEC_COLUMN_INDEX_V1: Final[Mapping[str, int]] = (
    MappingProxyType(
        {
            name: index
            for index, name in enumerate(DESCRIPTOR_SPEC_COLUMNS_V1)
        }
    )
)
DATA_TENSOR_INDEX_V1: Final[int] = 0
ALIVE_MASK_TENSOR_INDEX_V1: Final[int] = 1
ACTIVE_LENGTH_TENSOR_INDEX_V1: Final[int] = 2
CELL_STRIDE_BYTES_V1: Final[int] = 3
BYTES_PER_CELL_V1: Final[int] = 4
ROW_CAPACITY_V1: Final[int] = 5
ROW_BYTES_V1: Final[int] = 6
ALIVE_MASK_CELL_STRIDE_V1: Final[int] = 7
DTYPE_BYTES_V1: Final[int] = 8
CANONICALIZATION_MODE_V1: Final[int] = 9
RAW_OUTPUT_INDEX_V1: Final[int] = 10
CANONICAL_OUTPUT_INDEX_V1: Final[int] = 11
DESCRIPTOR_INDEX_V1: Final[int] = 12
DESCRIPTOR_SPEC_WIDTH_V1: Final[int] = 13

# Backward-readable aliases retain the full TensorDigestPointerSpec spelling
# while the canonical registry above follows the parent header exactly.
TENSOR_DIGEST_POINTER_SPEC_COLUMNS_V1: Final[tuple[str, ...]] = (
    DESCRIPTOR_SPEC_COLUMNS_V1
)
TENSOR_DIGEST_POINTER_SPEC_COLUMN_INDEX_V1: Final[Mapping[str, int]] = (
    DESCRIPTOR_SPEC_COLUMN_INDEX_V1
)

_TORCH_DTYPE_NAMES: Final[Mapping[torch.dtype, str]] = MappingProxyType(
    {
        torch.bool: "bool",
        torch.float32: "float32",
        torch.float64: "float64",
        torch.int8: "int8",
        torch.int32: "int32",
        torch.int64: "int64",
        torch.uint8: "uint8",
    }
)
_NUMPY_DTYPES: Final[Mapping[str, np.dtype[Any]]] = MappingProxyType(
    {
        name: np.dtype(name)
        for name in (
            "bool",
            "float32",
            "float64",
            "int8",
            "int32",
            "int64",
            "uint8",
        )
    }
)
_LENGTH_PHYSICAL_SOURCES: Final[Mapping[str, tuple[str, str]]] = MappingProxyType(
    {
        "identity.slot_free_queue_count": ("resident", "slot_free_queue_count"),
        "metadata.alive_count": ("resident", "active_count_state"),
        "active_count_state": ("resident", "active_count_state"),
        "metadata.event_tape_active_count": ("resident", "event_active_count"),
        "metadata.typed_lifecycle_event_active_count": (
            "resident",
            "event_active_count",
        ),
        "policy_mutation_tape.policy_birth_tape_cursor": (
            "policy_mutation_tape",
            "policy_birth_tape_cursor",
        ),
    }
)
_EVENT_COUNT_ALIASES: Final[tuple[str, ...]] = (
    "event_tape_active_count",
    "typed_lifecycle_event_active_count",
)


class CompactSummaryCheckpointRuntimeError(ValueError):
    """Raised when a validated plan cannot bind to resident tensor metadata."""


@dataclass(frozen=True)
class CheckpointDigestOutputAllocationV1:
    """Bounded allocation metadata for generation-major digest outputs."""

    generation_count: int
    cell_count: int
    descriptor_count: int
    raw_digest_shape: tuple[int, int, int, int]
    canonical_digest_shape: tuple[int, int, int, int]
    status_shape: tuple[int, int, int]
    validity_shape: tuple[int, int, int]
    raw_digest_nbytes: int
    canonical_digest_nbytes: int
    status_nbytes: int
    validity_nbytes: int
    total_nbytes: int


@dataclass(frozen=True)
class TensorDigestPointerSpecBatchV1:
    """One immutable, contiguous native pointer-spec batch."""

    batch_index: int
    descriptor_indices: tuple[int, ...]
    descriptor_specs: np.ndarray

    @property
    def count(self) -> int:
        return len(self.descriptor_indices)


@dataclass(frozen=True)
class CompactSummaryCheckpointRuntimeV1:
    """Immutable binding metadata plus writable preallocated digest outputs."""

    schema_version: str
    plan_schema_version: str
    live_device_expectation: str
    strict_live: bool
    backend: str
    device: str
    generation_count: int
    cell_count: int
    descriptor_count: int
    descriptor_names: tuple[str, ...]
    symbol_sizes: Mapping[str, int]
    tensor_names: tuple[str, ...]
    tensor_table: tuple[Any, ...]
    tensor_aliases: Mapping[str, int]
    descriptor_spec_columns: tuple[str, ...]
    descriptor_specs: np.ndarray
    batches: tuple[TensorDigestPointerSpecBatchV1, ...]
    allocation: CheckpointDigestOutputAllocationV1
    raw_digests: Any
    canonical_digests: Any
    status: Any
    validity: Any

    @property
    def pointer_specs(self) -> np.ndarray:
        return self.descriptor_specs

    @property
    def valid(self) -> Any:
        return self.validity

    def outputs(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "raw_digests": self.raw_digests,
                "canonical_digests": self.canonical_digests,
                "status": self.status,
                "validity": self.validity,
            }
        )


@dataclass(frozen=True)
class PhaseEModularCheckpointRuntimeV1:
    """Independent exact-hash runtime for one validated modular projection."""

    schema_version: str
    plan_schema_version: str
    storage_device_expectation: str
    backend: str
    device: str
    generation_count: int
    cell_count: int
    descriptor_count: int
    descriptor_names: tuple[str, ...]
    tensor_names: tuple[str, ...]
    tensor_table: tuple[Any, ...]
    tensor_aliases: Mapping[str, int]
    descriptor_spec_columns: tuple[str, ...]
    descriptor_specs: np.ndarray
    batches: tuple[TensorDigestPointerSpecBatchV1, ...]
    allocation: CheckpointDigestOutputAllocationV1
    raw_digests: Any
    canonical_digests: Any
    status: Any
    validity: Any

    def outputs(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "raw_digests": self.raw_digests,
                "canonical_digests": self.canonical_digests,
                "status": self.status,
                "validity": self.validity,
            }
        )


@dataclass(frozen=True)
class _TensorMetadata:
    backend: str
    device: str
    dtype: str
    itemsize: int
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    contiguous: bool
    requires_grad: bool


def _fail(path: str, message: str) -> None:
    raise CompactSummaryCheckpointRuntimeError(f"{path}: {message}")


def _positive_int(value: Any, *, path: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{path} must be an int")
    if value <= 0:
        _fail(path, "must be positive")
    if value > np.iinfo(np.int64).max:
        _fail(path, "exceeds the int64 ABI range")
    return value


def _checked_product(values: Sequence[int], *, path: str) -> int:
    product = 1
    for value in values:
        if value <= 0:
            _fail(path, "contains a nonpositive extent")
        if product > np.iinfo(np.int64).max // value:
            _fail(path, "overflows the int64 ABI range")
        product *= value
    return product


def checkpoint_digest_output_allocation(
    generation_count: int,
    cell_count: int,
    descriptor_count: int,
) -> CheckpointDigestOutputAllocationV1:
    """Return output allocation metadata without allocating any tensor."""

    generations = _positive_int(generation_count, path="generation_count")
    cells = _positive_int(cell_count, path="cell_count")
    descriptors = _positive_int(descriptor_count, path="descriptor_count")
    rows = _checked_product(
        (generations, cells, descriptors), path="output rows"
    )
    raw_bytes = _checked_product((rows, DIGEST_BYTES), path="raw digests")
    canonical_bytes = _checked_product(
        (rows, DIGEST_BYTES), path="canonical digests"
    )
    status_bytes = _checked_product((rows, 4), path="status")
    validity_bytes = rows
    total = raw_bytes + canonical_bytes + status_bytes + validity_bytes
    if total > sys.maxsize:
        _fail("output allocation", "exceeds the host platform index range")
    digest_shape = (generations, cells, descriptors, DIGEST_BYTES)
    row_shape = (generations, cells, descriptors)
    return CheckpointDigestOutputAllocationV1(
        generation_count=generations,
        cell_count=cells,
        descriptor_count=descriptors,
        raw_digest_shape=digest_shape,
        canonical_digest_shape=digest_shape,
        status_shape=row_shape,
        validity_shape=row_shape,
        raw_digest_nbytes=raw_bytes,
        canonical_digest_nbytes=canonical_bytes,
        status_nbytes=status_bytes,
        validity_nbytes=validity_bytes,
        total_nbytes=total,
    )


def _tensor_metadata(value: Any, *, path: str) -> _TensorMetadata:
    if isinstance(value, np.ndarray):
        dtype = str(value.dtype)
        itemsize = int(value.dtype.itemsize)
        return _TensorMetadata(
            backend="numpy",
            device="cpu",
            dtype=dtype,
            itemsize=itemsize,
            shape=tuple(int(extent) for extent in value.shape),
            strides=tuple(int(stride) for stride in value.strides),
            contiguous=bool(value.flags.c_contiguous),
            requires_grad=False,
        )
    if isinstance(value, torch.Tensor):
        dtype = _TORCH_DTYPE_NAMES.get(value.dtype)
        if dtype is None:
            _fail(path, f"unsupported Torch dtype {value.dtype}")
        return _TensorMetadata(
            backend="torch",
            device=str(value.device),
            dtype=dtype,
            itemsize=int(value.element_size()),
            shape=tuple(int(extent) for extent in value.shape),
            strides=tuple(int(stride) for stride in value.stride()),
            contiguous=bool(value.is_contiguous()),
            requires_grad=bool(value.requires_grad),
        )
    raise TypeError(f"{path} must be a NumPy array or Torch tensor")


def _physical_key(value: Any, metadata: _TensorMetadata) -> tuple[Any, ...]:
    if metadata.backend == "numpy":
        pointer = int(value.__array_interface__["data"][0])
        return ("numpy", pointer)
    return ("torch", metadata.device, int(value.data_ptr()))


def _validate_tensor(
    value: Any,
    *,
    path: str,
    dtype: str,
    shape: tuple[int, ...],
    strict_live: bool,
) -> _TensorMetadata:
    metadata = _tensor_metadata(value, path=path)
    if dtype not in _NUMPY_DTYPES:
        _fail(path, f"unsupported descriptor dtype {dtype!r}")
    if metadata.dtype != dtype:
        _fail(path, f"dtype must be {dtype}, got {metadata.dtype}")
    expected_itemsize = int(_NUMPY_DTYPES[dtype].itemsize)
    if metadata.itemsize != expected_itemsize:
        _fail(
            path,
            f"element size must be {expected_itemsize}, got {metadata.itemsize}",
        )
    if metadata.shape != shape:
        _fail(path, f"shape must be {shape}, got {metadata.shape}")
    if not metadata.contiguous:
        _fail(path, "must be C-contiguous")
    if metadata.requires_grad:
        _fail(path, "must not require gradients")
    if strict_live and not (
        metadata.backend == "torch" and metadata.device.startswith("cuda")
    ):
        _fail(path, "strict live runtime requires a CUDA Torch tensor")
    return metadata


class _TensorTableBuilder:
    def __init__(self) -> None:
        self.values: list[Any] = []
        self.names: list[str] = []
        self.aliases: dict[str, int] = {}
        self.by_physical_key: dict[
            tuple[Any, ...], tuple[int, _TensorMetadata, str]
        ] = {}
        self.backend: str | None = None
        self.device: str | None = None

    def add(
        self,
        role: str,
        value: Any,
        metadata: _TensorMetadata,
    ) -> int:
        previous_alias = self.aliases.get(role)
        key = _physical_key(value, metadata)
        previous = self.by_physical_key.get(key)
        if previous is not None:
            index, previous_metadata, previous_role = previous
            if previous_metadata != metadata:
                _fail(
                    role,
                    "aliases physical storage with contradictory metadata from "
                    f"{previous_role}",
                )
        else:
            if self.backend is not None and (
                metadata.backend != self.backend or metadata.device != self.device
            ):
                _fail(
                    role,
                    "all resident sources must use one array backend and device",
                )
            self.backend = metadata.backend
            self.device = metadata.device
            index = len(self.values)
            self.values.append(value)
            self.names.append(role)
            self.by_physical_key[key] = (index, metadata, role)
        if previous_alias is not None and previous_alias != index:
            _fail(role, "logical alias resolves to contradictory physical sources")
        self.aliases[role] = index
        return index


def _required_mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    for key in value:
        if not isinstance(key, str):
            raise TypeError(f"{path} keys must be strings")
    return value


def _required_source(
    values: Mapping[str, Any],
    name: str,
    *,
    path: str,
) -> Any:
    if name not in values:
        _fail(path, f"missing required tensor {name!r}")
    return values[name]


def _ready_descriptors(plan: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(plan, Mapping):
        raise TypeError("plan must be a mapping")
    status = plan.get("status")
    if plan.get("world", WORLD_FLAT) != WORLD_FLAT:
        _fail(
            "plan.world",
            f"Patch pending plans are unsupported: {plan.get('world')!r}",
        )
    source_contract = plan.get("source_contract")
    descriptor_world = (
        source_contract.get("descriptor_world")
        if isinstance(source_contract, Mapping)
        else None
    )
    if descriptor_world is None:
        world = WORLD_FLAT
        expected_status = READY_FLAT_PLAN_STATUS
        ready_label = "Flat"
    elif descriptor_world == WORLD_PATCH:
        world = WORLD_PATCH
        expected_status = READY_PATCH_P2_PLAN_STATUS
        ready_label = "Patch P2"
        if source_contract.get("patch_profile") != PATCH_P2_PROFILE:
            _fail(
                "plan.source_contract.patch_profile",
                f"must equal {PATCH_P2_PROFILE!r}",
            )
    else:
        _fail(
            "plan.source_contract.descriptor_world",
            f"unsupported descriptor world {descriptor_world!r}",
        )
    if status != expected_status:
        _fail(
            "plan.status",
            f"ready {ready_label} plan required, got {status!r}",
        )
    try:
        validate_compact_summary_checkpoint_descriptor_plan(plan, world=world)
    except Exception as exc:
        raise CompactSummaryCheckpointRuntimeError(
            f"plan: frozen descriptor validation failed: {exc}"
        ) from exc
    raw_descriptors = plan.get("descriptors")
    if not isinstance(raw_descriptors, list) or not raw_descriptors:
        _fail("plan.descriptors", "must be a nonempty list")
    return tuple(raw_descriptors)


def _validated_symbol_sizes(
    descriptors: Sequence[Mapping[str, Any]],
    symbol_sizes: Mapping[str, int],
    *,
    cell_count: int,
) -> Mapping[str, int]:
    values = _required_mapping(symbol_sizes, path="symbol_sizes")
    expected = {
        symbol
        for descriptor in descriptors
        for symbol in descriptor["resident_shape_symbols"]
    }
    actual = set(values)
    if actual != expected:
        _fail(
            "symbol_sizes",
            f"symbols mismatch missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}",
        )
    resolved = {
        name: _positive_int(values[name], path=f"symbol_sizes.{name}")
        for name in sorted(expected)
    }
    if resolved.get("cell") != cell_count:
        _fail("symbol_sizes.cell", "must equal cell_count")
    return MappingProxyType(resolved)


def _descriptor_shape(
    descriptor: Mapping[str, Any],
    symbol_sizes: Mapping[str, int],
    *,
    index: int,
) -> tuple[int, ...]:
    symbols = descriptor.get("resident_shape_symbols")
    if not isinstance(symbols, list) or not all(
        isinstance(symbol, str) for symbol in symbols
    ):
        _fail(f"plan.descriptors[{index}].resident_shape_symbols", "malformed")
    shape = tuple(symbol_sizes[symbol] for symbol in symbols)
    if descriptor.get("resident_rank") != len(shape) or len(shape) < 2:
        _fail(f"plan.descriptors[{index}].resident_rank", "must match rank >= 2")
    return shape


def _register_optional_aliases(
    *,
    primary_index: int,
    primary_name: str,
    aliases: Sequence[str],
    values: Mapping[str, Any],
    table: _TensorTableBuilder,
    dtype: str,
    shape: tuple[int, ...],
    strict_live: bool,
) -> None:
    for alias in aliases:
        if alias not in values:
            continue
        role = f"resident.{alias}"
        metadata = _validate_tensor(
            values[alias],
            path=role,
            dtype=dtype,
            shape=shape,
            strict_live=strict_live,
        )
        alias_index = table.add(role, values[alias], metadata)
        if alias_index != primary_index:
            _fail(
                role,
                f"must alias resident.{primary_name}; contradictory authority supplied",
            )


def _register_data_source(
    descriptor: Mapping[str, Any],
    *,
    index: int,
    shape: tuple[int, ...],
    resident_tensors: Mapping[str, Any],
    policy_mutation_tensors: Mapping[str, Any],
    table: _TensorTableBuilder,
    strict_live: bool,
) -> tuple[int, _TensorMetadata]:
    name = descriptor.get("tensor_name")
    scope = descriptor.get("source_scope")
    physical_source_name = descriptor.get("physical_source_name")
    dtype = descriptor.get("dtype")
    if (
        not isinstance(name, str)
        or not isinstance(physical_source_name, str)
        or not isinstance(dtype, str)
    ):
        _fail(
            f"plan.descriptors[{index}]",
            "tensor_name, physical_source_name, and dtype must be strings",
        )
    aliases: tuple[str, ...] = ()
    if scope == "checkpoint_projection":
        physical_name = physical_source_name
        values = resident_tensors
        role = f"resident.{physical_name}"
        if name in {"agent_id_to_slot_agent_ids", "agent_ids_by_alive_order"}:
            if physical_name != "active_agent_ids":
                _fail(
                    f"plan.descriptors[{index}].physical_source_name",
                    f"{name} must resolve to active_agent_ids",
                )
            aliases = (name,)
        elif name == "agent_id_to_slot_slots" and physical_name != "active_agent_slots":
            _fail(
                f"plan.descriptors[{index}].physical_source_name",
                "agent_id_to_slot_slots must resolve to active_agent_slots",
            )
    elif scope == "policy_mutation_tape":
        physical_name = physical_source_name
        if (
            physical_name != name
            and not (
                name == "policy_birth_post_rng_state_tape"
                and physical_name == "policy_birth_post_rng_state_packed_tape"
            )
        ):
            _fail(
                f"plan.descriptors[{index}].physical_source_name",
                "policy tape physical source must retain its tensor name",
            )
        values = policy_mutation_tensors
        role = f"policy_mutation_tape.{physical_name}"
    else:
        _fail(
            f"plan.descriptors[{index}].source_scope",
            f"unsupported source scope {scope!r}",
        )
    value = _required_source(values, physical_name, path=role)
    metadata = _validate_tensor(
        value,
        path=role,
        dtype=dtype,
        shape=shape,
        strict_live=strict_live,
    )
    table_index = table.add(role, value, metadata)
    table.add(f"descriptor[{index}].data", value, metadata)
    if aliases:
        _register_optional_aliases(
            primary_index=table_index,
            primary_name=physical_name,
            aliases=aliases,
            values=values,
            table=table,
            dtype=dtype,
            shape=shape,
            strict_live=strict_live,
        )
    return table_index, metadata


def _register_alive_mask(
    *,
    descriptor_index: int,
    row_capacity: int,
    cell_count: int,
    resident_tensors: Mapping[str, Any],
    table: _TensorTableBuilder,
    strict_live: bool,
) -> int:
    role = "resident.alive_mask"
    value = _required_source(resident_tensors, "alive_mask", path=role)
    metadata = _validate_tensor(
        value,
        path=role,
        dtype="bool",
        shape=(cell_count, row_capacity),
        strict_live=strict_live,
    )
    table_index = table.add(role, value, metadata)
    table.add(f"descriptor[{descriptor_index}].alive_mask", value, metadata)
    return table_index


def _register_active_lengths(
    authority: str,
    *,
    descriptor_index: int,
    cell_count: int,
    resident_tensors: Mapping[str, Any],
    policy_mutation_tensors: Mapping[str, Any],
    table: _TensorTableBuilder,
    strict_live: bool,
) -> int:
    physical = _LENGTH_PHYSICAL_SOURCES.get(authority)
    if physical is None:
        _fail(
            f"plan.descriptors[{descriptor_index}].length_authority",
            f"unsupported authority {authority!r}",
        )
    scope, name = physical
    values = (
        resident_tensors if scope == "resident" else policy_mutation_tensors
    )
    role = f"{scope}.{name}"
    value = _required_source(values, name, path=role)
    metadata = _validate_tensor(
        value,
        path=role,
        dtype="int64",
        shape=(cell_count,),
        strict_live=strict_live,
    )
    table_index = table.add(role, value, metadata)
    table.add(f"descriptor[{descriptor_index}].active_lengths", value, metadata)
    if name == "event_active_count":
        _register_optional_aliases(
            primary_index=table_index,
            primary_name=name,
            aliases=_EVENT_COUNT_ALIASES,
            values=resident_tensors,
            table=table,
            dtype="int64",
            shape=(cell_count,),
            strict_live=strict_live,
        )
    if scope == "policy_mutation_tape" and name in resident_tensors:
        _register_optional_aliases(
            primary_index=table_index,
            primary_name=name,
            aliases=(name,),
            values=resident_tensors,
            table=table,
            dtype="int64",
            shape=(cell_count,),
            strict_live=strict_live,
        )
    return table_index


def _validated_batches(
    plan: Mapping[str, Any],
    specs: np.ndarray,
    descriptor_count: int,
) -> tuple[TensorDigestPointerSpecBatchV1, ...]:
    raw_batches = plan.get("batches")
    if not isinstance(raw_batches, list) or not raw_batches:
        _fail("plan.batches", "must be a nonempty list")
    result = []
    flattened: list[int] = []
    for batch_index, batch in enumerate(raw_batches):
        if not isinstance(batch, Mapping):
            _fail(f"plan.batches[{batch_index}]", "must be a mapping")
        indices = batch.get("descriptor_indices")
        if not isinstance(indices, list) or not all(type(value) is int for value in indices):
            _fail(f"plan.batches[{batch_index}].descriptor_indices", "malformed")
        expected = list(
            range(
                len(flattened),
                min(len(flattened) + CUDA_HASH_BATCH_CAPACITY, descriptor_count),
            )
        )
        if (
            batch.get("batch_index") != batch_index
            or batch.get("count") != len(indices)
            or indices != expected
            or not 1 <= len(indices) <= CUDA_HASH_BATCH_CAPACITY
        ):
            _fail(
                f"plan.batches[{batch_index}]",
                "must be the exact contiguous descriptor chunk with count <= 16",
            )
        flattened.extend(indices)
        batch_specs = specs[indices[0] : indices[-1] + 1]
        result.append(
            TensorDigestPointerSpecBatchV1(
                batch_index=batch_index,
                descriptor_indices=tuple(indices),
                descriptor_specs=batch_specs,
            )
        )
    if flattened != list(range(descriptor_count)):
        _fail("plan.batches", "do not cover every descriptor exactly once")
    return tuple(result)


def _allocate_outputs(
    allocation: CheckpointDigestOutputAllocationV1,
    *,
    backend: str,
    device: str,
) -> tuple[Any, Any, Any, Any]:
    try:
        if backend == "numpy":
            return (
                np.zeros(allocation.raw_digest_shape, dtype=np.uint8),
                np.zeros(allocation.canonical_digest_shape, dtype=np.uint8),
                np.zeros(allocation.status_shape, dtype=np.int32),
                np.zeros(allocation.validity_shape, dtype=np.uint8),
            )
        if backend == "torch":
            target = torch.device(device)
            return (
                torch.zeros(
                    allocation.raw_digest_shape, dtype=torch.uint8, device=target
                ),
                torch.zeros(
                    allocation.canonical_digest_shape,
                    dtype=torch.uint8,
                    device=target,
                ),
                torch.zeros(
                    allocation.status_shape, dtype=torch.int32, device=target
                ),
                torch.zeros(
                    allocation.validity_shape, dtype=torch.uint8, device=target
                ),
            )
    except (MemoryError, RuntimeError) as exc:
        raise CompactSummaryCheckpointRuntimeError(
            f"output allocation failed for {allocation.total_nbytes} bytes: {exc}"
        ) from exc
    _fail("output backend", f"unsupported backend {backend!r}")
    raise AssertionError("unreachable")


def bind_compact_summary_checkpoint_runtime(
    plan: Mapping[str, Any],
    resident_tensors: Mapping[str, Any],
    policy_mutation_tensors: Mapping[str, Any],
    *,
    symbol_sizes: Mapping[str, int],
    generation_count: int,
    cell_count: int,
    strict_live: bool = False,
) -> CompactSummaryCheckpointRuntimeV1:
    """Bind one validated Flat or Patch P2 plan to resident tensor authorities.

    ``strict_live=False`` permits CPU NumPy fixtures.  Production callers must
    pass ``strict_live=True`` so every source and output is a CUDA Torch tensor.
    All symbolic extents are mandatory; no shape or dtype is inferred from a
    source tensor.
    """

    if type(strict_live) is not bool:
        raise TypeError("strict_live must be a bool")
    generations = _positive_int(generation_count, path="generation_count")
    cells = _positive_int(cell_count, path="cell_count")
    descriptors = _ready_descriptors(plan)
    descriptor_count = len(descriptors)
    sizes = _validated_symbol_sizes(descriptors, symbol_sizes, cell_count=cells)
    resident = _required_mapping(resident_tensors, path="resident_tensors")
    policy = _required_mapping(
        policy_mutation_tensors, path="policy_mutation_tensors"
    )

    table = _TensorTableBuilder()
    data_rows: list[tuple[int, _TensorMetadata, tuple[int, ...]]] = []
    descriptor_names: list[str] = []
    for index, descriptor in enumerate(descriptors):
        if descriptor.get("descriptor_index") != index:
            _fail(f"plan.descriptors[{index}].descriptor_index", "out of order")
        shape = _descriptor_shape(descriptor, sizes, index=index)
        data_index, metadata = _register_data_source(
            descriptor,
            index=index,
            shape=shape,
            resident_tensors=resident,
            policy_mutation_tensors=policy,
            table=table,
            strict_live=strict_live,
        )
        data_rows.append((data_index, metadata, shape))
        descriptor_names.append(str(descriptor["tensor_name"]))

    specs = np.empty(
        (descriptor_count, DESCRIPTOR_SPEC_WIDTH_V1),
        dtype=np.int64,
    )
    raw_indices: set[int] = set()
    canonical_indices: set[int] = set()
    for index, descriptor in enumerate(descriptors):
        data_index, metadata, shape = data_rows[index]
        dtype_bytes = descriptor.get("dtype_bytes")
        if type(dtype_bytes) is not int or dtype_bytes != metadata.itemsize:
            _fail(f"plan.descriptors[{index}].dtype_bytes", "contradicts source")
        row_capacity = shape[1]
        row_elements = _checked_product(
            shape[2:] or (1,), path=f"plan.descriptors[{index}].row_bytes"
        )
        row_bytes = _checked_product(
            (row_elements, dtype_bytes),
            path=f"plan.descriptors[{index}].row_bytes",
        )
        bytes_per_cell = _checked_product(
            (row_capacity, row_bytes),
            path=f"plan.descriptors[{index}].bytes_per_cell",
        )
        mode = descriptor.get("raw_canonical_mode")
        mode_code = descriptor.get("raw_canonical_mode_code")
        if mode not in (
            MODE_EXACT,
            MODE_DEAD_ROW_ZERO,
            MODE_ACTIVE_PREFIX,
            MODE_TORCH_RNG_PACKED_PREFIX,
        ):
            _fail(f"plan.descriptors[{index}].raw_canonical_mode", "unsupported")
        if mode_code != MODE_CODES[mode]:
            _fail(
                f"plan.descriptors[{index}].raw_canonical_mode_code",
                "contradicts mode",
            )

        alive_index = NULL_TENSOR_INDEX
        active_index = NULL_TENSOR_INDEX
        alive_stride = 0
        alive_authority = descriptor.get("alive_authority")
        length_authority = descriptor.get("length_authority")
        active_length_source = descriptor.get("active_length_source")
        if mode == MODE_EXACT:
            if (
                alive_authority is not None
                or length_authority is not None
                or active_length_source is not None
            ):
                _fail(f"plan.descriptors[{index}]", "exact mode has an active source")
        elif mode == MODE_DEAD_ROW_ZERO:
            if (
                alive_authority != "tensors.alive_mask"
                or length_authority is not None
                or active_length_source is not None
            ):
                _fail(f"plan.descriptors[{index}]", "malformed live-row authority")
            alive_index = _register_alive_mask(
                descriptor_index=index,
                row_capacity=row_capacity,
                cell_count=cells,
                resident_tensors=resident,
                table=table,
                strict_live=strict_live,
            )
            alive_stride = row_capacity
        else:
            if (
                alive_authority is not None
                or not isinstance(length_authority, str)
                or not isinstance(active_length_source, str)
                or active_length_source != length_authority
            ):
                _fail(f"plan.descriptors[{index}]", "malformed active-prefix authority")
            active_index = _register_active_lengths(
                active_length_source,
                descriptor_index=index,
                cell_count=cells,
                resident_tensors=resident,
                policy_mutation_tensors=policy,
                table=table,
                strict_live=strict_live,
            )

        raw_output = descriptor.get("raw_output_index")
        canonical_output = descriptor.get("canonical_output_index")
        if (
            type(raw_output) is not int
            or raw_output < 0
            or raw_output >= descriptor_count
            or raw_output in raw_indices
        ):
            _fail(f"plan.descriptors[{index}].raw_output_index", "invalid or duplicate")
        if (
            type(canonical_output) is not int
            or canonical_output < 0
            or canonical_output >= descriptor_count
            or canonical_output in canonical_indices
        ):
            _fail(
                f"plan.descriptors[{index}].canonical_output_index",
                "invalid or duplicate",
            )
        raw_indices.add(raw_output)
        canonical_indices.add(canonical_output)
        specs[index] = (
            data_index,
            alive_index,
            active_index,
            bytes_per_cell,
            bytes_per_cell,
            row_capacity,
            row_bytes,
            alive_stride,
            dtype_bytes,
            mode_code,
            raw_output,
            canonical_output,
            index,
        )

    specs.setflags(write=False)
    batches = _validated_batches(plan, specs, descriptor_count)
    allocation = checkpoint_digest_output_allocation(
        generations, cells, descriptor_count
    )
    if table.backend is None or table.device is None:
        _fail("tensor_table", "must not be empty")
    raw, canonical, status, validity = _allocate_outputs(
        allocation, backend=table.backend, device=table.device
    )
    runtime = CompactSummaryCheckpointRuntimeV1(
        schema_version=RUNTIME_SCHEMA_VERSION,
        plan_schema_version=str(plan.get("schema_version")),
        live_device_expectation=LIVE_DEVICE_EXPECTATION,
        strict_live=strict_live,
        backend=table.backend,
        device=table.device,
        generation_count=generations,
        cell_count=cells,
        descriptor_count=descriptor_count,
        descriptor_names=tuple(descriptor_names),
        symbol_sizes=sizes,
        tensor_names=tuple(table.names),
        tensor_table=tuple(table.values),
        tensor_aliases=MappingProxyType(dict(table.aliases)),
        descriptor_spec_columns=DESCRIPTOR_SPEC_COLUMNS_V1,
        descriptor_specs=specs,
        batches=batches,
        allocation=allocation,
        raw_digests=raw,
        canonical_digests=canonical,
        status=status,
        validity=validity,
    )
    validate_compact_summary_checkpoint_runtime(runtime, strict_live=strict_live)
    return runtime


def _validate_output_metadata(
    value: Any,
    *,
    path: str,
    backend: str,
    device: str,
    dtype: str,
    shape: tuple[int, ...],
    strict_live: bool,
) -> None:
    metadata = _validate_tensor(
        value,
        path=path,
        dtype=dtype,
        shape=shape,
        strict_live=strict_live,
    )
    if metadata.backend != backend or metadata.device != device:
        _fail(path, "backend or device contradicts runtime metadata")


def validate_compact_summary_checkpoint_runtime(
    runtime: CompactSummaryCheckpointRuntimeV1,
    *,
    strict_live: bool = False,
) -> None:
    """Validate runtime metadata without reading source or output tensor bytes."""

    if not isinstance(runtime, CompactSummaryCheckpointRuntimeV1):
        raise TypeError("runtime must be CompactSummaryCheckpointRuntimeV1")
    if type(strict_live) is not bool:
        raise TypeError("strict_live must be a bool")
    enforce_live = strict_live or runtime.strict_live
    if runtime.schema_version != CONTRACT_VERSION_V1:
        _fail("runtime.schema_version", "unsupported")
    if runtime.plan_schema_version != PLAN_SCHEMA_VERSION:
        _fail("runtime.plan_schema_version", "unsupported")
    if runtime.live_device_expectation != LIVE_DEVICE_EXPECTATION:
        _fail("runtime.live_device_expectation", "must be cuda")
    if runtime.descriptor_spec_columns != DESCRIPTOR_SPEC_COLUMNS_V1:
        _fail("runtime.descriptor_spec_columns", "ABI column order drift")
    if len(runtime.descriptor_spec_columns) != DESCRIPTOR_SPEC_WIDTH_V1:
        _fail("runtime.descriptor_spec_columns", "must contain exactly 13 columns")
    if (
        runtime.descriptor_specs.dtype != np.dtype(np.int64)
        or runtime.descriptor_specs.shape
        != (runtime.descriptor_count, DESCRIPTOR_SPEC_WIDTH_V1)
        or not runtime.descriptor_specs.flags.c_contiguous
        or runtime.descriptor_specs.flags.writeable
    ):
        _fail("runtime.descriptor_specs", "must be read-only contiguous int64[D,13]")
    if len(runtime.descriptor_names) != runtime.descriptor_count:
        _fail("runtime.descriptor_names", "count mismatch")
    if len(runtime.tensor_names) != len(runtime.tensor_table):
        _fail("runtime.tensor_table", "name count mismatch")
    if not runtime.tensor_table:
        _fail("runtime.tensor_table", "must not be empty")
    for alias, index in runtime.tensor_aliases.items():
        if not isinstance(alias, str) or type(index) is not int:
            _fail("runtime.tensor_aliases", "malformed entry")
        if index < 0 or index >= len(runtime.tensor_table):
            _fail("runtime.tensor_aliases", "index out of range")
    expected_allocation = checkpoint_digest_output_allocation(
        runtime.generation_count, runtime.cell_count, runtime.descriptor_count
    )
    if runtime.allocation != expected_allocation:
        _fail("runtime.allocation", "metadata contradiction")
    _validate_output_metadata(
        runtime.raw_digests,
        path="runtime.raw_digests",
        backend=runtime.backend,
        device=runtime.device,
        dtype="uint8",
        shape=runtime.allocation.raw_digest_shape,
        strict_live=enforce_live,
    )
    _validate_output_metadata(
        runtime.canonical_digests,
        path="runtime.canonical_digests",
        backend=runtime.backend,
        device=runtime.device,
        dtype="uint8",
        shape=runtime.allocation.canonical_digest_shape,
        strict_live=enforce_live,
    )
    _validate_output_metadata(
        runtime.status,
        path="runtime.status",
        backend=runtime.backend,
        device=runtime.device,
        dtype="int32",
        shape=runtime.allocation.status_shape,
        strict_live=enforce_live,
    )
    _validate_output_metadata(
        runtime.validity,
        path="runtime.validity",
        backend=runtime.backend,
        device=runtime.device,
        dtype="uint8",
        shape=runtime.allocation.validity_shape,
        strict_live=enforce_live,
    )
    flattened = []
    for batch_index, batch in enumerate(runtime.batches):
        if batch.batch_index != batch_index or not 1 <= batch.count <= 16:
            _fail(f"runtime.batches[{batch_index}]", "invalid batch metadata")
        expected_specs = runtime.descriptor_specs[
            batch.descriptor_indices[0] : batch.descriptor_indices[-1] + 1
        ]
        if batch.descriptor_specs is not expected_specs and (
            batch.descriptor_specs.shape != expected_specs.shape
            or batch.descriptor_specs.ctypes.data != expected_specs.ctypes.data
        ):
            _fail(f"runtime.batches[{batch_index}]", "not a contiguous spec view")
        flattened.extend(batch.descriptor_indices)
    if flattened != list(range(runtime.descriptor_count)):
        _fail("runtime.batches", "descriptor coverage mismatch")
    if enforce_live and not (
        runtime.backend == "torch" and runtime.device.startswith("cuda")
    ):
        _fail("runtime.device", "strict live runtime requires CUDA")


def maybe_bind_phase_e_modular_checkpoint_runtime(
    overlay_plan: Mapping[str, Any] | None,
    projection_metadata: Mapping[str, Any],
    projection_tensors: Mapping[str, Any],
    *,
    generation_count: int,
) -> PhaseEModularCheckpointRuntimeV1 | None:
    """Bind the optional modular overlay without changing the frozen v1 runtime.

    The committed Phase E storage projection is one checkpoint/cell and is CPU
    authoritative.  The generic descriptor ABI is retained, including batches
    of at most 16, but this passive path does not claim CUDA residency.
    """

    generated = maybe_generate_phase_e_modular_checkpoint_descriptor_overlay(
        projection_metadata,
        projection_tensors,
    )
    if generated is None:
        if overlay_plan is not None:
            _fail(
                "phase_e_modular_overlay_plan",
                "plan exists while modular projection authority is absent",
            )
        return None
    if overlay_plan is None:
        _fail(
            "phase_e_modular_overlay_plan",
            "modular projection authority exists without its descriptor overlay",
        )
    validate_phase_e_modular_checkpoint_descriptor_overlay(
        overlay_plan,
        projection_metadata,
        projection_tensors,
    )
    if overlay_plan.get("schema_version") != PHASE_E_MODULAR_OVERLAY_PLAN_SCHEMA_VERSION:
        _fail("phase_e_modular_overlay_plan.schema_version", "unsupported")
    if overlay_plan.get("status") != PHASE_E_MODULAR_OVERLAY_STATUS:
        _fail("phase_e_modular_overlay_plan.status", "not ready")
    descriptors = overlay_plan.get("descriptors")
    if not isinstance(descriptors, list) or not descriptors:
        _fail("phase_e_modular_overlay_plan.descriptors", "must be nonempty")
    generations = _positive_int(generation_count, path="generation_count")

    from perf.cuda_architecture_b_modular_state_abi import (  # noqa: WPS433
        MODULAR_STORAGE_PROJECTION_PREFIX,
    )

    modular_names = sorted(
        str(name)
        for name in projection_tensors
        if str(name).startswith(MODULAR_STORAGE_PROJECTION_PREFIX)
    )
    expected_names = [str(row["tensor_name"]) for row in descriptors]
    if modular_names != expected_names:
        _fail(
            "projection_tensors",
            f"modular fields mismatch expected={expected_names!r} actual={modular_names!r}",
        )

    table = _TensorTableBuilder()
    specs = np.empty((len(descriptors), DESCRIPTOR_SPEC_WIDTH_V1), dtype=np.int64)
    descriptor_names: list[str] = []
    for index, descriptor in enumerate(descriptors):
        path = f"phase_e_modular_overlay_plan.descriptors[{index}]"
        if descriptor.get("descriptor_index") != index:
            _fail(f"{path}.descriptor_index", "out of order")
        name = descriptor.get("tensor_name")
        dtype = descriptor.get("dtype")
        raw_shape = descriptor.get("fixed_shape")
        if (
            not isinstance(name, str)
            or not isinstance(dtype, str)
            or not isinstance(raw_shape, list)
            or not all(type(extent) is int and extent > 0 for extent in raw_shape)
        ):
            _fail(path, "malformed name, dtype, or fixed shape")
        shape = tuple(raw_shape)
        value = _required_source(projection_tensors, name, path=f"projection.{name}")
        metadata = _validate_tensor(
            value,
            path=f"projection.{name}",
            dtype=dtype,
            shape=shape,
            strict_live=False,
        )
        data_index = table.add(f"projection.{name}", value, metadata)
        table.add(f"descriptor[{index}].data", value, metadata)
        dtype_bytes = descriptor.get("dtype_bytes")
        if type(dtype_bytes) is not int or dtype_bytes != metadata.itemsize:
            _fail(f"{path}.dtype_bytes", "contradicts projection tensor")
        if (
            descriptor.get("raw_canonical_mode") != MODE_EXACT
            or descriptor.get("raw_canonical_mode_code") != MODE_CODES[MODE_EXACT]
            or descriptor.get("alive_authority") is not None
            or descriptor.get("length_authority") is not None
            or descriptor.get("active_length_source") is not None
        ):
            _fail(path, "modular overlay descriptors must be exact mode")
        row_capacity = shape[0]
        row_elements = _checked_product(
            shape[1:] or (1,), path=f"{path}.row_bytes"
        )
        row_bytes = _checked_product(
            (row_elements, dtype_bytes), path=f"{path}.row_bytes"
        )
        bytes_per_cell = _checked_product(
            (row_capacity, row_bytes), path=f"{path}.bytes_per_cell"
        )
        raw_output = descriptor.get("raw_output_index")
        canonical_output = descriptor.get("canonical_output_index")
        if raw_output != index or canonical_output != index:
            _fail(path, "output indices must equal descriptor index")
        specs[index] = (
            data_index,
            NULL_TENSOR_INDEX,
            NULL_TENSOR_INDEX,
            bytes_per_cell,
            bytes_per_cell,
            row_capacity,
            row_bytes,
            0,
            dtype_bytes,
            MODE_CODES[MODE_EXACT],
            index,
            index,
            index,
        )
        descriptor_names.append(name)

    specs.setflags(write=False)
    batches = _validated_batches(overlay_plan, specs, len(descriptors))
    allocation = checkpoint_digest_output_allocation(
        generations,
        1,
        len(descriptors),
    )
    if table.backend is None or table.device is None:
        _fail("phase_e_modular_tensor_table", "must not be empty")
    if table.device != "cpu":
        _fail(
            "phase_e_modular_tensor_table.device",
            "passive modular storage authority must remain on CPU",
        )
    raw, canonical, status, validity = _allocate_outputs(
        allocation,
        backend=table.backend,
        device=table.device,
    )
    runtime = PhaseEModularCheckpointRuntimeV1(
        schema_version=PHASE_E_MODULAR_RUNTIME_SCHEMA_VERSION,
        plan_schema_version=PHASE_E_MODULAR_OVERLAY_PLAN_SCHEMA_VERSION,
        storage_device_expectation=PHASE_E_MODULAR_STORAGE_DEVICE_EXPECTATION,
        backend=table.backend,
        device=table.device,
        generation_count=generations,
        cell_count=1,
        descriptor_count=len(descriptors),
        descriptor_names=tuple(descriptor_names),
        tensor_names=tuple(table.names),
        tensor_table=tuple(table.values),
        tensor_aliases=MappingProxyType(dict(table.aliases)),
        descriptor_spec_columns=DESCRIPTOR_SPEC_COLUMNS_V1,
        descriptor_specs=specs,
        batches=batches,
        allocation=allocation,
        raw_digests=raw,
        canonical_digests=canonical,
        status=status,
        validity=validity,
    )
    validate_phase_e_modular_checkpoint_runtime(runtime)
    return runtime


def bind_phase_e_modular_checkpoint_runtime(
    overlay_plan: Mapping[str, Any],
    projection_metadata: Mapping[str, Any],
    projection_tensors: Mapping[str, Any],
    *,
    generation_count: int,
) -> PhaseEModularCheckpointRuntimeV1:
    """Bind a required Phase E modular descriptor overlay."""

    runtime = maybe_bind_phase_e_modular_checkpoint_runtime(
        overlay_plan,
        projection_metadata,
        projection_tensors,
        generation_count=generation_count,
    )
    if runtime is None:
        _fail("phase_e_modular_storage", "projection is absent")
    return runtime


def validate_phase_e_modular_checkpoint_runtime(
    runtime: PhaseEModularCheckpointRuntimeV1,
) -> None:
    """Validate immutable modular runtime metadata without reading plane values."""

    if not isinstance(runtime, PhaseEModularCheckpointRuntimeV1):
        raise TypeError("runtime must be PhaseEModularCheckpointRuntimeV1")
    if runtime.schema_version != PHASE_E_MODULAR_RUNTIME_SCHEMA_VERSION:
        _fail("phase_e_modular_runtime.schema_version", "unsupported")
    if runtime.plan_schema_version != PHASE_E_MODULAR_OVERLAY_PLAN_SCHEMA_VERSION:
        _fail("phase_e_modular_runtime.plan_schema_version", "unsupported")
    if runtime.storage_device_expectation != PHASE_E_MODULAR_STORAGE_DEVICE_EXPECTATION:
        _fail("phase_e_modular_runtime.storage_device_expectation", "unsupported")
    if runtime.cell_count != 1:
        _fail("phase_e_modular_runtime.cell_count", "must equal one checkpoint")
    if runtime.backend not in {"numpy", "torch"} or runtime.device != "cpu":
        _fail("phase_e_modular_runtime.device", "must be CPU storage")
    if runtime.descriptor_spec_columns != DESCRIPTOR_SPEC_COLUMNS_V1:
        _fail("phase_e_modular_runtime.descriptor_spec_columns", "ABI drift")
    if (
        runtime.descriptor_specs.dtype != np.dtype(np.int64)
        or runtime.descriptor_specs.shape
        != (runtime.descriptor_count, DESCRIPTOR_SPEC_WIDTH_V1)
        or not runtime.descriptor_specs.flags.c_contiguous
        or runtime.descriptor_specs.flags.writeable
    ):
        _fail(
            "phase_e_modular_runtime.descriptor_specs",
            "must be read-only contiguous int64[D,13]",
        )
    if tuple(sorted(runtime.descriptor_names)) != runtime.descriptor_names:
        _fail("phase_e_modular_runtime.descriptor_names", "must be lexicographic")
    if len(runtime.descriptor_names) != runtime.descriptor_count:
        _fail("phase_e_modular_runtime.descriptor_names", "count mismatch")
    expected_allocation = checkpoint_digest_output_allocation(
        runtime.generation_count,
        runtime.cell_count,
        runtime.descriptor_count,
    )
    if runtime.allocation != expected_allocation:
        _fail("phase_e_modular_runtime.allocation", "metadata contradiction")
    for value, name, dtype, shape in (
        (
            runtime.raw_digests,
            "raw_digests",
            "uint8",
            runtime.allocation.raw_digest_shape,
        ),
        (
            runtime.canonical_digests,
            "canonical_digests",
            "uint8",
            runtime.allocation.canonical_digest_shape,
        ),
        (runtime.status, "status", "int32", runtime.allocation.status_shape),
        (runtime.validity, "validity", "uint8", runtime.allocation.validity_shape),
    ):
        _validate_output_metadata(
            value,
            path=f"phase_e_modular_runtime.{name}",
            backend=runtime.backend,
            device=runtime.device,
            dtype=dtype,
            shape=shape,
            strict_live=False,
        )
    flattened: list[int] = []
    for batch_index, batch in enumerate(runtime.batches):
        if batch.batch_index != batch_index or not 1 <= batch.count <= 16:
            _fail(f"phase_e_modular_runtime.batches[{batch_index}]", "invalid")
        flattened.extend(batch.descriptor_indices)
    if flattened != list(range(runtime.descriptor_count)):
        _fail("phase_e_modular_runtime.batches", "descriptor coverage mismatch")


build_compact_summary_checkpoint_runtime = bind_compact_summary_checkpoint_runtime


__all__ = [
    "ACTIVE_LENGTH_TENSOR_INDEX_V1",
    "ALIVE_MASK_CELL_STRIDE_V1",
    "ALIVE_MASK_TENSOR_INDEX_V1",
    "BYTES_PER_CELL_V1",
    "CANONICALIZATION_MODE_V1",
    "CANONICAL_OUTPUT_INDEX_V1",
    "CELL_STRIDE_BYTES_V1",
    "CONTRACT_VERSION_V1",
    "DATA_TENSOR_INDEX_V1",
    "DESCRIPTOR_INDEX_V1",
    "DESCRIPTOR_SPEC_COLUMNS_V1",
    "DESCRIPTOR_SPEC_COLUMN_INDEX_V1",
    "DESCRIPTOR_SPEC_WIDTH_V1",
    "DIGEST_BYTES",
    "DTYPE_BYTES_V1",
    "LIVE_DEVICE_EXPECTATION",
    "NULL_TENSOR_INDEX",
    "RAW_OUTPUT_INDEX_V1",
    "READY_FLAT_PLAN_STATUS",
    "READY_PATCH_P2_PLAN_STATUS",
    "ROW_BYTES_V1",
    "ROW_CAPACITY_V1",
    "RUNTIME_SCHEMA_VERSION",
    "TENSOR_DIGEST_POINTER_SPEC_COLUMNS_V1",
    "TENSOR_DIGEST_POINTER_SPEC_COLUMN_INDEX_V1",
    "CheckpointDigestOutputAllocationV1",
    "CompactSummaryCheckpointRuntimeError",
    "CompactSummaryCheckpointRuntimeV1",
    "PHASE_E_MODULAR_RUNTIME_SCHEMA_VERSION",
    "PHASE_E_MODULAR_STORAGE_DEVICE_EXPECTATION",
    "PhaseEModularCheckpointRuntimeV1",
    "TensorDigestPointerSpecBatchV1",
    "bind_compact_summary_checkpoint_runtime",
    "bind_phase_e_modular_checkpoint_runtime",
    "build_compact_summary_checkpoint_runtime",
    "checkpoint_digest_output_allocation",
    "maybe_bind_phase_e_modular_checkpoint_runtime",
    "validate_compact_summary_checkpoint_runtime",
    "validate_phase_e_modular_checkpoint_runtime",
]
