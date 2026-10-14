"""Generic packed transport for one physical-batch checkpoint.

This module is deliberately not wired into a provider.  It is a transport
contract that can be adopted at an explicit producer seam after its physical
qualification.  The contract has three separate operations:

``capture_physical_batch_packed_checkpoint_v1``
    Copies already-batched tensors to an unsealed file through one shared
    two-slab copier.  It never returns a tensor, stream, owner, callback, or
    other live object.

``finalize_physical_batch_packed_checkpoint_v1``
    Runs on host-only file/scalar state.  It hashes exact byte ranges, writes
    the seal, fsyncs the file and publication directory, and atomically
    publishes the result.

``load_physical_batch_packed_checkpoint_v1``
    Strictly validates the seal and every byte range before reconstructing the
    complete batch or one selected cell.

The payload is field-major.  A common field is supplied as one contiguous
``(B, ...)`` tensor and is copied by one tensor-transfer operation.  Ragged
fields must be declared with :class:`RaggedFieldV1`; each declared tensor is
copied as one sequence item.  No per-cell projection callback is used for a
common field, and the producer never creates a full host snapshot.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Sequence
from uuid import uuid4

import torch

from . import acs_native_aggregate_checkpoint_v1 as aggregate
from . import acs_native_streamed_checkpoint_v1 as streamed


SCHEMA_VERSION = "eiel_acs_physical_batch_packed_checkpoint_v1"
MAGIC = b"ACSPBV1\x00"
FORMAT_VERSION = 1
LITTLE_ENDIAN_MARKER = 0x0102

# The layout is intentionally payload-first.  The producer can append the
# scalar header after streaming payload bytes without reserving a large header
# or retaining a host copy of any tensor.
PREFIX = struct.Struct("<8sHHQQQ")
MAX_HEADER_BYTES = 64 * 1024 * 1024
MAX_FOOTER_BYTES = 4 * 1024 * 1024
MAX_FIELD_COUNT = 100_000
MAX_BATCH_CELLS = 100_000
MAX_FIELD_BYTES = 1 << 42
MAX_PHYSICAL_PAYLOAD_BYTES = 8 * 1024**3
HASH_CHUNK_BYTES = 8 * 1024 * 1024
FOOTER_SCHEMA_VERSION = "eiel_acs_physical_batch_packed_checkpoint_footer_v1"
PAYLOAD_LAYOUT = "field_major_physical_batch_v1"
ZERO_FILL_DESCRIPTOR_SCHEMA_VERSION = "eiel_acs_physical_batch_zero_fill_descriptor_v1"
ZERO_FILL_CONTRACT = "fixed_mlp_exact_zero_v1"
CAPTURE_AUTHORITY_SCHEMA_VERSION = (
    "eiel_acs_physical_batch_transient_capture_authority_v1"
)

_DTYPE_ITEMSIZE: dict[str, int] = {
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
    "float16": 2,
    "float32": 4,
    "float64": 8,
    "bfloat16": 2,
    "complex64": 8,
    "complex128": 16,
    "uint16": 2,
    "uint32": 4,
    "uint64": 8,
}


class PhysicalBatchPackedCheckpointError(RuntimeError):
    """The packed physical-batch transport is malformed or not durable."""


@dataclass(frozen=True, slots=True)
class PackedCheckpointFileFingerprintV1:
    """The exact filesystem identity captured for an open packed file."""

    device: int
    inode: int
    mode: int
    nlink: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def from_stat(
        cls, stat_result: os.stat_result
    ) -> "PackedCheckpointFileFingerprintV1":
        return cls(
            device=int(stat_result.st_dev),
            inode=int(stat_result.st_ino),
            mode=int(stat_result.st_mode),
            nlink=int(stat_result.st_nlink),
            size=int(stat_result.st_size),
            mtime_ns=int(stat_result.st_mtime_ns),
            ctime_ns=int(stat_result.st_ctime_ns),
        )


@dataclass(frozen=True, slots=True)
class RaggedFieldV1:
    """An explicitly declared per-cell sequence for one ragged field."""

    values: tuple[torch.Tensor, ...]

    def __init__(self, values: Sequence[torch.Tensor]) -> None:
        if isinstance(values, (str, bytes, bytearray)):
            raise PhysicalBatchPackedCheckpointError(
                "ragged field values must be a tensor sequence"
            )
        normalized = tuple(values)
        if any(not isinstance(value, torch.Tensor) for value in normalized):
            raise PhysicalBatchPackedCheckpointError(
                "ragged field values must contain only tensors"
            )
        object.__setattr__(self, "values", normalized)


@dataclass(frozen=True, slots=True)
class ZeroFillFieldV1:
    """Declare an exact-zero B-wide field without allocating its tensor.

    Authority remains the existing fixed-MLP exact-zero contract.  The shape
    is the full physical-batch shape, including the leading B dimension.
    """

    dtype: str
    shape: tuple[int, ...]
    contract: str

    def __init__(
        self,
        dtype: str | torch.dtype,
        shape: Sequence[int],
        *,
        contract: str = ZERO_FILL_CONTRACT,
    ) -> None:
        dtype_name = str(dtype).removeprefix("torch.")
        if dtype_name not in streamed._DTYPE_BY_NAME:
            raise PhysicalBatchPackedCheckpointError(
                f"unsupported zero-fill dtype: {dtype_name}"
            )
        normalized_shape = tuple(shape)
        if not normalized_shape or any(
            type(dimension) is not int or dimension < 0
            for dimension in normalized_shape
        ):
            raise PhysicalBatchPackedCheckpointError(
                "zero-fill shape must be a nonempty sequence of nonnegative integers"
            )
        if contract != ZERO_FILL_CONTRACT:
            raise PhysicalBatchPackedCheckpointError(
                "zero-fill contract must be fixed_mlp_exact_zero_v1"
            )
        object.__setattr__(self, "dtype", dtype_name)
        object.__setattr__(self, "shape", normalized_shape)
        object.__setattr__(self, "contract", contract)


# A short alias is useful at call sites while the versioned name remains the
# public contract name.
RaggedField = RaggedFieldV1
ZeroFillField = ZeroFillFieldV1


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _fail(message: str) -> PhysicalBatchPackedCheckpointError:
    return PhysicalBatchPackedCheckpointError(message)


def _validate_scalar(value: Any, *, path: str = "scalar_metadata") -> Any:
    """Validate JSON scalar metadata without allowing tensor-bearing values."""

    if value is None or type(value) is bool or type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise _fail(f"{path} contains a non-finite float")
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise _fail(f"{path} mapping keys must be nonempty strings")
            normalized[key] = _validate_scalar(item, path=f"{path}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            _validate_scalar(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, torch.Tensor) or callable(value):
        raise _fail(f"{path} contains a tensor or callable")
    raise _fail(f"{path} contains unsupported type {type(value).__name__}")


def _assert_descriptor_scalars(value: Any, *, path: str = "descriptor") -> None:
    """Reject live objects recursively at the producer/host boundary."""

    if isinstance(value, torch.Tensor) or callable(value):
        raise _fail(f"{path} contains a live tensor or callable")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise _fail(f"{path} contains a non-string mapping key")
            _assert_descriptor_scalars(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_descriptor_scalars(item, path=f"{path}[{index}]")


def capture_descriptor_authority_sha256_v1(
    descriptor: Mapping[str, Any],
) -> str:
    """Digest the exact closed-file authority used by direct world sharding."""

    fingerprint = descriptor.get("captured_file_fingerprint")
    fields = (
        "device",
        "inode",
        "mode",
        "nlink",
        "size",
        "mtime_ns",
        "ctime_ns",
    )
    if (
        descriptor.get("schema_version") != SCHEMA_VERSION
        or descriptor.get("descriptor_status") != "captured_unsealed"
        or not isinstance(descriptor.get("capture_path"), str)
        or not descriptor.get("capture_path")
        or not isinstance(fingerprint, Mapping)
        or set(fingerprint) != set(fields)
        or any(type(fingerprint.get(field)) is not int for field in fields)
        or not _is_sha256(descriptor.get("header_sha256"))
    ):
        raise _fail("packed transient capture authority is malformed")
    return _sha256(
        _canonical_json(
            {
                "schema_version": CAPTURE_AUTHORITY_SCHEMA_VERSION,
                "capture_path": str(descriptor["capture_path"]),
                "batch_id": descriptor.get("batch_id"),
                "cell_ids": descriptor.get("cell_ids"),
                "field_order": descriptor.get("field_order"),
                "payload_bytes": descriptor.get("payload_bytes"),
                "header_bytes": descriptor.get("header_bytes"),
                "header_sha256": descriptor.get("header_sha256"),
                "captured_file_fingerprint": dict(fingerprint),
            }
        )
    )


def _validate_chunk_bytes(chunk_bytes: Any) -> int:
    if type(chunk_bytes) is not int or not 0 < chunk_bytes <= streamed.MAX_CHUNK_BYTES:
        raise _fail(f"chunk_bytes must be an integer in 1..{streamed.MAX_CHUNK_BYTES}")
    return chunk_bytes


def _dtype_name(tensor: torch.Tensor) -> str:
    try:
        return streamed._dtype_name(tensor)
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise _fail(f"unsupported tensor dtype: {tensor.dtype}") from exc


def _dtype_itemsize(dtype_name: str) -> int:
    try:
        return _DTYPE_ITEMSIZE[dtype_name]
    except KeyError as exc:
        raise _fail(f"unsupported tensor dtype: {dtype_name}") from exc


def _shape_bytes(dtype_name: str, shape: Sequence[int]) -> int:
    itemsize = _dtype_itemsize(dtype_name)
    elements = 1
    for dimension in shape:
        if type(dimension) is not int or dimension < 0:
            raise _fail("tensor shapes must contain nonnegative integers")
        elements *= dimension
        if elements > MAX_FIELD_BYTES // max(1, itemsize):
            raise _fail("tensor field exceeds the bounded byte limit")
    total = elements * itemsize
    if total > MAX_FIELD_BYTES:
        raise _fail("tensor field exceeds the bounded byte limit")
    return total


def _validate_physical_payload_bytes(
    payload_bytes: Any, *, label: str = "physical payload"
) -> int:
    if type(payload_bytes) is not int or payload_bytes < 0:
        raise _fail(f"{label} byte count is invalid")
    if payload_bytes > MAX_PHYSICAL_PAYLOAD_BYTES:
        raise _fail(
            f"{label} exceeds the {MAX_PHYSICAL_PAYLOAD_BYTES} byte physical cap"
        )
    return payload_bytes


def _validate_tensor_source(
    tensor: torch.Tensor,
    *,
    field: str,
    cell_index: int | None = None,
) -> tuple[str, tuple[int, ...], int]:
    if not isinstance(tensor, torch.Tensor):
        suffix = "" if cell_index is None else f" cell {cell_index}"
        raise _fail(f"field {field!r}{suffix} is not a tensor")
    if not tensor.is_contiguous():
        raise _fail(
            f"field {field!r} must be contiguous; refusing an implicit full tensor copy"
        )
    dtype_name = _dtype_name(tensor)
    shape = tuple(int(value) for value in tensor.shape)
    logical_bytes = _shape_bytes(dtype_name, shape)
    return dtype_name, shape, logical_bytes


@dataclass(frozen=True, slots=True)
class _CommonFieldPlan:
    name: str
    tensor: torch.Tensor
    dtype: str
    full_shape: tuple[int, ...]
    cell_shape: tuple[int, ...]
    logical_bytes: int
    cell_bytes: int


@dataclass(frozen=True, slots=True)
class _RaggedFieldPlan:
    name: str
    values: tuple[torch.Tensor, ...]
    dtype: str
    shapes: tuple[tuple[int, ...], ...]
    logical_bytes: int


@dataclass(frozen=True, slots=True)
class _ZeroFillFieldPlan:
    name: str
    dtype: str
    full_shape: tuple[int, ...]
    cell_shape: tuple[int, ...]
    logical_bytes: int
    cell_bytes: int
    contract: str


def _planned_physical_payload_bytes(
    plans: Sequence[_CommonFieldPlan | _RaggedFieldPlan | _ZeroFillFieldPlan],
) -> int:
    total = 0
    for plan in plans:
        if isinstance(plan, _ZeroFillFieldPlan):
            continue
        total += plan.logical_bytes
        _validate_physical_payload_bytes(total, label="planned physical payload")
    return total


def _prepare_field_plans(
    fields: Mapping[str, Any],
    *,
    batch_cells: int,
) -> tuple[tuple[_CommonFieldPlan | _RaggedFieldPlan | _ZeroFillFieldPlan, ...], bool]:
    if not isinstance(fields, Mapping) or not fields:
        raise _fail("fields must be a nonempty mapping")
    if len(fields) > MAX_FIELD_COUNT:
        raise _fail("field count exceeds the bounded limit")
    raw_names = list(fields)
    if any(not isinstance(name, str) or not name for name in raw_names):
        raise _fail("field names must be nonempty strings")
    names = sorted(raw_names)
    if len(set(names)) != len(names):
        raise _fail("field names are duplicated")

    plans: list[_CommonFieldPlan | _RaggedFieldPlan | _ZeroFillFieldPlan] = []
    has_cuda = False
    for name in names:
        value = fields[name]
        if isinstance(value, ZeroFillFieldV1):
            if name not in aggregate.ZERO_FILL_CANDIDATE_FIELDS:
                raise _fail(
                    f"zero-fill field {name!r} is outside the fixed-MLP authority"
                )
            if value.shape[0] != batch_cells:
                raise _fail(
                    f"zero-fill field {name!r} must have shape (B, ...), got {value.shape}"
                )
            logical = _shape_bytes(value.dtype, value.shape)
            cell_shape = value.shape[1:]
            cell_bytes = _shape_bytes(value.dtype, cell_shape)
            if logical != batch_cells * cell_bytes:
                raise _fail(f"zero-fill field {name!r} byte coverage is inconsistent")
            plans.append(
                _ZeroFillFieldPlan(
                    name=name,
                    dtype=value.dtype,
                    full_shape=value.shape,
                    cell_shape=cell_shape,
                    logical_bytes=logical,
                    cell_bytes=cell_bytes,
                    contract=value.contract,
                )
            )
            continue
        if isinstance(value, RaggedFieldV1):
            values = value.values
            if len(values) != batch_cells:
                raise _fail(
                    f"ragged field {name!r} has {len(values)} values for B={batch_cells}"
                )
            if not values:
                raise _fail(f"ragged field {name!r} has no cell values")
            first_dtype: str | None = None
            shapes: list[tuple[int, ...]] = []
            total = 0
            for cell_index, tensor in enumerate(values):
                dtype, shape, logical = _validate_tensor_source(
                    tensor, field=name, cell_index=cell_index
                )
                if first_dtype is None:
                    first_dtype = dtype
                elif dtype != first_dtype:
                    raise _fail(
                        f"ragged field {name!r} changes dtype at cell {cell_index}"
                    )
                shapes.append(shape)
                total += logical
                has_cuda = has_cuda or tensor.device.type == "cuda"
            assert first_dtype is not None
            plans.append(
                _RaggedFieldPlan(
                    name=name,
                    values=values,
                    dtype=first_dtype,
                    shapes=tuple(shapes),
                    logical_bytes=total,
                )
            )
            continue

        if not isinstance(value, torch.Tensor):
            raise _fail(
                f"field {name!r} must be a B-wide tensor, RaggedFieldV1, "
                "or ZeroFillFieldV1"
            )
        dtype, full_shape, logical = _validate_tensor_source(value, field=name)
        if not full_shape or full_shape[0] != batch_cells:
            raise _fail(
                f"common field {name!r} must have shape (B, ...), got {full_shape}"
            )
        cell_shape = full_shape[1:]
        cell_bytes = _shape_bytes(dtype, cell_shape)
        if logical != batch_cells * cell_bytes:
            raise _fail(f"common field {name!r} byte coverage is inconsistent")
        plans.append(
            _CommonFieldPlan(
                name=name,
                tensor=value,
                dtype=dtype,
                full_shape=full_shape,
                cell_shape=cell_shape,
                logical_bytes=logical,
                cell_bytes=cell_bytes,
            )
        )
        has_cuda = has_cuda or value.device.type == "cuda"
    return tuple(plans), has_cuda


class _PayloadSink:
    """Write bounded copier output directly to the producer file."""

    def __init__(self, stream: BinaryIO, *, payload_offset: int = 0) -> None:
        self.stream = stream
        self.bytes = 0
        self.write_operations = 0
        _validate_physical_payload_bytes(payload_offset, label="payload write offset")
        self.payload_offset = payload_offset

    def write(self, value: bytes | bytearray | memoryview) -> int:
        raw = bytes(value)
        _validate_physical_payload_bytes(
            self.payload_offset + self.bytes + len(raw),
            label="physical payload write",
        )
        self.stream.write(raw)
        self.bytes += len(raw)
        self.write_operations += 1
        return len(raw)


class _SharedTwoSlabCopier:
    """One reusable bounded copier shared by every field in one capture."""

    def __init__(
        self,
        *,
        chunk_bytes: int,
        has_cuda_fields: bool,
        require_native_pool: bool = False,
    ) -> None:
        self.chunk_bytes = chunk_bytes
        self.transfer_operations = 0
        self.chunk_operations = 0
        self.native_pool = None
        self.fallback_pool = None
        native_type = streamed._native_slab_pool_type()
        if native_type is not None:
            try:
                self.native_pool = native_type(chunk_bytes, has_cuda_fields)
            except (RuntimeError, TypeError, ValueError):
                self.native_pool = None
        if self.native_pool is None:
            if require_native_pool:
                raise _fail(
                    "packed transport requires the native preallocated two-slab pool"
                )
            self.fallback_pool = streamed._TwoSlabPool(
                chunk_bytes, cuda=has_cuda_fields
            )

    def copy_tensor(self, tensor: torch.Tensor, sink: _PayloadSink) -> int:
        if not tensor.is_contiguous():
            raise _fail("packed transport copier received a non-contiguous tensor")
        start_bytes = sink.bytes
        self.transfer_operations += 1
        if self.native_pool is not None:
            try:
                count = int(self.native_pool.copy_tensor_chunks(tensor, sink))
            except (RuntimeError, TypeError, ValueError) as exc:
                raise _fail("native two-slab tensor copy failed") from exc
        else:
            count = 0
            assert self.fallback_pool is not None
            for raw in streamed._iter_tensor_chunks(
                tensor,
                chunk_bytes=self.chunk_bytes,
                slab_pool=self.fallback_pool,
            ):
                sink.write(raw)
                count += 1
        self.chunk_operations += count
        expected_bytes = _shape_bytes(_dtype_name(tensor), tuple(tensor.shape))
        expected_chunks = (
            (expected_bytes + self.chunk_bytes - 1) // self.chunk_bytes
            if expected_bytes
            else 0
        )
        written_bytes = sink.bytes - start_bytes
        if written_bytes != expected_bytes:
            raise _fail(
                "packed transport copier tensor byte delta differs: "
                f"{written_bytes} != {expected_bytes}"
            )
        if count != expected_chunks:
            raise _fail(
                f"packed transport copier chunk coverage differs: {count} != {expected_chunks}"
            )
        return count

    def telemetry(self) -> dict[str, Any]:
        if self.native_pool is not None:
            try:
                raw = dict(self.native_pool.telemetry())
            except (AttributeError, TypeError, ValueError):
                raw = {}
            return {
                "schema_version": "eiel_acs_physical_batch_packed_slab_pool_v1",
                "slab_count": 2,
                "native_pool": True,
                "pinned": bool(raw.get("pinned", False)),
                "preallocated": bool(raw.get("preallocated", True)),
                "chunk_bytes": self.chunk_bytes,
                "chunk_count": int(raw.get("chunk_count", 0)),
                "copy_operations": int(raw.get("copy_operations", 0)),
                "copied_bytes": int(raw.get("copied_bytes", 0)),
                "field_count": int(raw.get("field_count", 0)),
                "cuda_event_wait_count": int(raw.get("cuda_event_wait_count", 0)),
                "final_cuda_fence_count": int(raw.get("final_cuda_fence_count", 0)),
                "per_chunk_stream_synchronizations": int(
                    raw.get("per_chunk_stream_synchronizations", 0)
                ),
                "source_device": str(raw.get("source_device", "unknown")),
            }
        return {
            "schema_version": "eiel_acs_physical_batch_packed_slab_pool_v1",
            "slab_count": 2,
            "native_pool": False,
            "pinned": False,
            "preallocated": True,
            "chunk_bytes": self.chunk_bytes,
        }


def _field_descriptor_common(plan: _CommonFieldPlan, *, offset: int) -> dict[str, Any]:
    return {
        "name": plan.name,
        "kind": "common",
        "storage": "payload",
        "dtype": plan.dtype,
        "dtype_code": int(streamed._DTYPE_CODE[plan.dtype]),
        "shape": list(plan.full_shape),
        "cell_shape": list(plan.cell_shape),
        "cell_bytes": plan.cell_bytes,
        "cell_order": "cell_ids_order_v1",
        "offset": offset,
        "bytes": plan.logical_bytes,
        "logical_bytes": plan.logical_bytes,
    }


def _field_descriptor_ragged(
    plan: _RaggedFieldPlan, *, entries: Sequence[Mapping[str, Any]], offset: int
) -> dict[str, Any]:
    return {
        "name": plan.name,
        "kind": "ragged",
        "storage": "payload",
        "dtype": plan.dtype,
        "dtype_code": int(streamed._DTYPE_CODE[plan.dtype]),
        "sequence": "declared_cell_sequence_v1",
        "offset": offset,
        "bytes": plan.logical_bytes,
        "logical_bytes": plan.logical_bytes,
        "entries": [dict(entry) for entry in entries],
    }


def _zero_fill_descriptor(
    plan: _ZeroFillFieldPlan, *, offset: int, field_index: int
) -> dict[str, Any]:
    return {
        "schema_version": ZERO_FILL_DESCRIPTOR_SCHEMA_VERSION,
        "field": plan.name,
        "dtype": plan.dtype,
        "shape": list(plan.full_shape),
        "cell_shape": list(plan.cell_shape),
        "cell_bytes": plan.cell_bytes,
        "layout": "C_contiguous",
        "order": {
            "scheme": PAYLOAD_LAYOUT,
            "field_index": field_index,
            "cell_order": "cell_ids_order_v1",
        },
        "offset": offset,
        "logical_bytes": plan.logical_bytes,
        "physical_bytes": 0,
        "zero_fill": {"kind": "exact_zero", "value": 0},
        "zero_fill_contract": plan.contract,
        "logical_digest": "deferred_sha256_exact_zero_bytes_v1",
    }


def _field_descriptor_zero_fill(
    plan: _ZeroFillFieldPlan, *, offset: int, field_index: int
) -> dict[str, Any]:
    return {
        "name": plan.name,
        "kind": "zero_fill",
        "storage": "zero_fill",
        "dtype": plan.dtype,
        "dtype_code": int(streamed._DTYPE_CODE[plan.dtype]),
        "shape": list(plan.full_shape),
        "cell_shape": list(plan.cell_shape),
        "cell_bytes": plan.cell_bytes,
        "cell_order": "cell_ids_order_v1",
        "offset": offset,
        "bytes": 0,
        "logical_bytes": plan.logical_bytes,
        "zero_fill_contract": plan.contract,
        "zero_fill_descriptor": _zero_fill_descriptor(
            plan, offset=offset, field_index=field_index
        ),
    }


def _validate_container_identity(
    header: Mapping[str, Any], *, payload_bytes: int
) -> tuple[list[str], list[dict[str, Any]]]:
    _validate_physical_payload_bytes(payload_bytes, label="container physical payload")
    if header.get("schema_version") != SCHEMA_VERSION:
        raise _fail("packed checkpoint schema version is invalid")
    if header.get("format_version") != FORMAT_VERSION:
        raise _fail("packed checkpoint format version is invalid")
    if (
        header.get("endianness") != "little"
        or header.get("payload_layout") != PAYLOAD_LAYOUT
    ):
        raise _fail("packed checkpoint byte layout is invalid")
    if header.get("capture_status") != "captured_unsealed":
        raise _fail("packed checkpoint capture header status is invalid")
    batch_id = header.get("batch_id")
    if not isinstance(batch_id, str) or not batch_id:
        raise _fail("packed checkpoint batch_id is invalid")
    cell_ids = header.get("cell_ids")
    if (
        not isinstance(cell_ids, list)
        or not cell_ids
        or len(cell_ids) > MAX_BATCH_CELLS
        or any(not isinstance(cell_id, str) or not cell_id for cell_id in cell_ids)
        or len(set(cell_ids)) != len(cell_ids)
    ):
        raise _fail("packed checkpoint cell order is invalid")
    if header.get("batch_cells") != len(cell_ids):
        raise _fail("packed checkpoint batch cell count is invalid")
    _validate_physical_payload_bytes(
        header.get("payload_bytes"), label="header physical payload"
    )
    if header.get("payload_bytes") != payload_bytes:
        raise _fail("packed checkpoint payload byte count is invalid")
    field_order = header.get("field_order")
    fields = header.get("fields")
    if (
        not isinstance(field_order, list)
        or not field_order
        or field_order != sorted(field_order)
        or any(not isinstance(name, str) or not name for name in field_order)
        or not isinstance(fields, list)
        or len(fields) != len(field_order)
        or len(fields) > MAX_FIELD_COUNT
    ):
        raise _fail("packed checkpoint field order is invalid")
    if [
        field.get("name") for field in fields if isinstance(field, Mapping)
    ] != field_order:
        raise _fail("packed checkpoint field descriptors do not match field order")
    _validate_scalar(header.get("scalar_metadata", {}), path="header.scalar_metadata")
    operation_counts = header.get("operation_counts")
    if not isinstance(operation_counts, Mapping) or any(
        type(value) is not int or value < 0 for value in operation_counts.values()
    ):
        raise _fail("packed checkpoint operation counts are invalid")
    return list(cell_ids), [dict(field) for field in fields]


def _validate_ranges(
    fields: Sequence[Mapping[str, Any]], *, batch_cells: int, payload_bytes: int
) -> tuple[int, list[str]]:
    _validate_physical_payload_bytes(payload_bytes, label="range physical payload")
    cursor = 0
    zero_fill_logical_bytes = 0
    zero_fill_fields: list[str] = []
    for field_index, field in enumerate(fields):
        name = field.get("name")
        kind = field.get("kind")
        dtype = field.get("dtype")
        if not isinstance(name, str) or kind not in {"common", "ragged", "zero_fill"}:
            raise _fail("packed checkpoint field descriptor is malformed")
        if not isinstance(dtype, str) or dtype not in streamed._DTYPE_BY_NAME:
            raise _fail(f"packed checkpoint dtype is invalid for {name!r}")
        if field.get("dtype_code") != streamed._DTYPE_CODE[dtype]:
            raise _fail(f"packed checkpoint dtype code is invalid for {name!r}")
        offset = field.get("offset")
        byte_count = field.get("bytes")
        logical_bytes = field.get("logical_bytes")
        if (
            type(offset) is not int
            or type(byte_count) is not int
            or type(logical_bytes) is not int
            or offset < 0
            or byte_count < 0
            or logical_bytes < 0
            or offset != cursor
        ):
            raise _fail(f"packed checkpoint byte coverage is invalid for {name!r}")
        if kind == "common":
            if field.get("storage") != "payload" or logical_bytes != byte_count:
                raise _fail(f"packed common field storage is invalid for {name!r}")
            shape = field.get("shape")
            cell_shape = field.get("cell_shape")
            cell_bytes = field.get("cell_bytes")
            if (
                not isinstance(shape, list)
                or not isinstance(cell_shape, list)
                or shape != [batch_cells, *cell_shape]
                or type(cell_bytes) is not int
                or cell_bytes < 0
                or byte_count != batch_cells * cell_bytes
                or _shape_bytes(dtype, shape) != byte_count
            ):
                raise _fail(f"packed common field shape/bytes disagree for {name!r}")
            if field.get("cell_order") != "cell_ids_order_v1":
                raise _fail(f"packed common field cell order is invalid for {name!r}")
            if field.get("zero_fill_descriptor") is not None:
                raise _fail(f"packed common field has zero-fill authority for {name!r}")
            cursor += byte_count
        elif kind == "ragged":
            if field.get("storage") != "payload" or logical_bytes != byte_count:
                raise _fail(f"packed ragged field storage is invalid for {name!r}")
            entries = field.get("entries")
            if (
                not isinstance(entries, list)
                or len(entries) != batch_cells
                or field.get("sequence") != "declared_cell_sequence_v1"
            ):
                raise _fail(f"packed ragged field sequence is invalid for {name!r}")
            entry_total = 0
            for cell_index, entry in enumerate(entries):
                if (
                    not isinstance(entry, Mapping)
                    or entry.get("cell_index") != cell_index
                ):
                    raise _fail(f"packed ragged cell order is invalid for {name!r}")
                entry_offset = entry.get("offset")
                entry_bytes = entry.get("bytes")
                entry_shape = entry.get("shape")
                if (
                    type(entry_offset) is not int
                    or type(entry_bytes) is not int
                    or entry_offset < 0
                    or entry_bytes < 0
                    or not isinstance(entry_shape, list)
                    or _shape_bytes(dtype, entry_shape) != entry_bytes
                    or entry.get("logical_bytes") != entry_bytes
                ):
                    raise _fail(f"packed ragged entry bytes are invalid for {name!r}")
                if entry_offset != cursor:
                    raise _fail(f"packed ragged entry order is invalid for {name!r}")
                entry_total += entry_bytes
                cursor += entry_bytes
            if entry_total != byte_count:
                raise _fail(f"packed ragged field byte total differs for {name!r}")
            if entries and offset != entries[0]["offset"]:
                raise _fail(f"packed ragged field offset is invalid for {name!r}")
            if field.get("zero_fill_descriptor") is not None:
                raise _fail(f"packed ragged field has zero-fill authority for {name!r}")
        else:
            if (
                name not in aggregate.ZERO_FILL_CANDIDATE_FIELDS
                or field.get("storage") != "zero_fill"
                or byte_count != 0
                or field.get("zero_fill_contract") != ZERO_FILL_CONTRACT
                or field.get("cell_order") != "cell_ids_order_v1"
            ):
                raise _fail(f"packed zero-fill authority is invalid for {name!r}")
            shape = field.get("shape")
            cell_shape = field.get("cell_shape")
            cell_bytes = field.get("cell_bytes")
            if (
                not isinstance(shape, list)
                or not isinstance(cell_shape, list)
                or shape != [batch_cells, *cell_shape]
                or type(cell_bytes) is not int
                or cell_bytes < 0
                or logical_bytes != batch_cells * cell_bytes
                or _shape_bytes(dtype, shape) != logical_bytes
            ):
                raise _fail(f"packed zero-fill shape/bytes disagree for {name!r}")
            expected_plan = _ZeroFillFieldPlan(
                name=name,
                dtype=dtype,
                full_shape=tuple(shape),
                cell_shape=tuple(cell_shape),
                logical_bytes=logical_bytes,
                cell_bytes=cell_bytes,
                contract=ZERO_FILL_CONTRACT,
            )
            expected_descriptor = _zero_fill_descriptor(
                expected_plan, offset=cursor, field_index=field_index
            )
            if field.get("zero_fill_descriptor") != expected_descriptor:
                raise _fail(f"packed zero-fill descriptor differs for {name!r}")
            zero_fill_logical_bytes += logical_bytes
            zero_fill_fields.append(name)
    if cursor != payload_bytes:
        raise _fail("packed checkpoint payload does not have exact byte coverage")
    return zero_fill_logical_bytes, zero_fill_fields


def _validate_zero_fill_summary(
    header: Mapping[str, Any],
    fields: Sequence[Mapping[str, Any]],
    *,
    batch_cells: int,
    payload_bytes: int,
) -> tuple[int, list[str]]:
    logical_bytes, names = _validate_ranges(
        fields, batch_cells=batch_cells, payload_bytes=payload_bytes
    )
    expected_contract = (
        aggregate._canonical_fixed_mlp_zero_fill_contract() if names else None
    )
    if (
        header.get("zero_fill_logical_bytes") != logical_bytes
        or header.get("zero_fill_fields") != names
        or header.get("zero_fill_contract") != expected_contract
        or header["operation_counts"].get("zero_fill_field_count") != len(names)
        or header["operation_counts"].get("zero_fill_transfer_operations") != 0
    ):
        raise _fail("packed checkpoint zero-fill summary is invalid")
    return logical_bytes, names


def _read_exact(stream: BinaryIO, length: int, *, label: str) -> bytes:
    if length < 0:
        raise _fail(f"negative read length for {label}")
    raw = stream.read(length)
    if len(raw) != length:
        raise _fail(f"packed checkpoint {label} is truncated")
    return raw


def _read_parts_from_stream(
    stream: BinaryIO,
    path: Path,
    *,
    require_sealed: bool,
) -> tuple[bytes, dict[str, Any], bytes | None, int, int]:
    prefix = _read_exact(stream, PREFIX.size, label="prefix")
    magic, version, endian, header_len, payload_len, footer_len = PREFIX.unpack(prefix)
    if magic != MAGIC or version != FORMAT_VERSION or endian != LITTLE_ENDIAN_MARKER:
        raise _fail("packed checkpoint prefix identity is invalid")
    if not 0 < header_len <= MAX_HEADER_BYTES:
        raise _fail("packed checkpoint header length is invalid")
    _validate_physical_payload_bytes(payload_len, label="prefix physical payload")
    if footer_len < 0 or footer_len > MAX_FOOTER_BYTES:
        raise _fail("packed checkpoint footer length is invalid")
    expected_size = PREFIX.size + payload_len + header_len + footer_len
    try:
        actual_size = os.fstat(stream.fileno()).st_size
    except OSError as exc:
        raise _fail(f"packed checkpoint cannot be read: {path}") from exc
    if actual_size != expected_size:
        raise _fail("packed checkpoint file size does not match its prefix")
    stream.seek(PREFIX.size + payload_len)
    header_bytes = _read_exact(stream, header_len, label="header")
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _fail("packed checkpoint header JSON is invalid") from exc
    if not isinstance(header, dict):
        raise _fail("packed checkpoint header is not an object")
    if require_sealed and footer_len == 0:
        raise _fail("packed checkpoint is not sealed")
    footer_bytes: bytes | None = None
    if footer_len:
        footer_bytes = _read_exact(stream, footer_len, label="footer")
    return prefix, header, footer_bytes, payload_len, header_len


def _read_parts(
    path: Path, *, require_sealed: bool
) -> tuple[bytes, dict[str, Any], bytes | None, int, int]:
    try:
        with path.open("rb") as stream:
            return _read_parts_from_stream(stream, path, require_sealed=require_sealed)
    except FileNotFoundError as exc:
        raise _fail(f"packed checkpoint file is missing: {path}") from exc
    except (OSError, struct.error) as exc:
        raise _fail(f"packed checkpoint cannot be read: {path}") from exc


def _hash_ranges(
    path: Path | None,
    *,
    payload_bytes: int,
    fields: Sequence[Mapping[str, Any]],
    stream: BinaryIO | None = None,
) -> tuple[str, dict[str, str], dict[str, list[str]], dict[str, str]]:
    payload_digest = hashlib.sha256()
    field_hashes: dict[str, str] = {}
    entry_hashes: dict[str, list[str]] = {}
    zero_fill_hashes: dict[str, str] = {}
    owned_stream: BinaryIO | None = None
    if stream is None:
        if path is None:
            raise _fail("packed checkpoint hash source is missing")
        owned_stream = path.open("rb")
        stream = owned_stream
    try:
        for field in fields:
            name = str(field["name"])
            if field["kind"] == "zero_fill":
                # This is the aggregate authority's exact bounded zero-stream
                # digest.  No logical zero bytes enter the physical payload.
                logical_digest = aggregate._zero_digest(int(field["logical_bytes"]))
                field_hashes[name] = logical_digest
                zero_fill_hashes[name] = logical_digest
                continue
            field_digest = hashlib.sha256()
            if field["kind"] == "common":
                ranges = [(int(field["offset"]), int(field["bytes"]))]
            else:
                ranges = [
                    (int(entry["offset"]), int(entry["bytes"]))
                    for entry in field["entries"]
                ]
            hashes: list[str] = []
            for offset, length in ranges:
                stream.seek(PREFIX.size + offset)
                remaining = length
                entry_digest = hashlib.sha256()
                while remaining:
                    raw = stream.read(min(HASH_CHUNK_BYTES, remaining))
                    if not raw:
                        raise _fail("packed checkpoint payload is truncated")
                    payload_digest.update(raw)
                    field_digest.update(raw)
                    entry_digest.update(raw)
                    remaining -= len(raw)
                hashes.append(entry_digest.hexdigest())
            field_hashes[name] = field_digest.hexdigest()
            if field["kind"] == "ragged":
                entry_hashes[name] = hashes
    finally:
        if owned_stream is not None:
            owned_stream.close()
    if payload_digest.digest_size != hashlib.sha256().digest_size:
        raise _fail("payload digest initialization failed")
    return (
        payload_digest.hexdigest(),
        field_hashes,
        entry_hashes,
        zero_fill_hashes,
    )


def _footer_seal_digest(footer: Mapping[str, Any]) -> str:
    body = {
        key: value
        for key, value in footer.items()
        if key not in {"seal_sha256", "artifact_sha256"}
    }
    return _sha256(_canonical_json(body))


def _footer_artifact_digest(
    *, prefix: bytes, header_sha256: str, payload_sha256: str, footer: Mapping[str, Any]
) -> str:
    body = dict(footer)
    body["artifact_sha256"] = "0" * 64
    return _sha256(
        _canonical_json(
            {
                "prefix_sha256": _sha256(prefix),
                "header_sha256": header_sha256,
                "payload_sha256": payload_sha256,
                "footer": body,
            }
        )
    )


def capture_physical_batch_packed_checkpoint_v1(
    fields: Mapping[str, Any],
    destination: str | os.PathLike[str],
    *,
    cell_ids: Sequence[str],
    batch_id: str,
    scalar_metadata: Mapping[str, Any] | None = None,
    chunk_bytes: int = streamed.DEFAULT_CHUNK_BYTES,
    require_native_pool: bool = False,
) -> dict[str, Any]:
    """Stream one physical batch into an unsealed, host-file descriptor.

    ``fields`` values that are tensors must already be batched as ``(B, ...)``.
    :class:`RaggedFieldV1` is the only accepted per-cell sequence, while
    :class:`ZeroFillFieldV1` declares an authoritative exact-zero field without
    allocating or copying a tensor.  The returned mapping is JSON-safe and
    contains no tensor-bearing object.
    """

    chunk_bytes = _validate_chunk_bytes(chunk_bytes)
    if not isinstance(cell_ids, Sequence) or isinstance(cell_ids, (str, bytes)):
        raise _fail("cell_ids must be a nonempty sequence")
    normalized_cell_ids = tuple(cell_ids)
    if (
        not normalized_cell_ids
        or len(normalized_cell_ids) > MAX_BATCH_CELLS
        or any(
            not isinstance(cell_id, str) or not cell_id
            for cell_id in normalized_cell_ids
        )
        or len(set(normalized_cell_ids)) != len(normalized_cell_ids)
    ):
        raise _fail("cell_ids must be unique nonempty strings in declared order")
    if not isinstance(batch_id, str) or not batch_id:
        raise _fail("batch_id must be a nonempty string")
    normalized_scalars = _validate_scalar(
        {} if scalar_metadata is None else scalar_metadata
    )
    if not isinstance(normalized_scalars, dict):
        raise _fail("scalar_metadata must be a mapping")
    plans, has_cuda_fields = _prepare_field_plans(
        fields, batch_cells=len(normalized_cell_ids)
    )
    _planned_physical_payload_bytes(plans)
    target = Path(destination).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    capture_path = target.with_name(
        f".{target.name}.{os.getpid()}.{uuid4().hex}.capture"
    )
    has_physical_tensor_fields = any(
        not isinstance(plan, _ZeroFillFieldPlan) for plan in plans
    )
    copier = (
        _SharedTwoSlabCopier(
            chunk_bytes=chunk_bytes,
            has_cuda_fields=has_cuda_fields,
            require_native_pool=require_native_pool,
        )
        if has_physical_tensor_fields
        else None
    )
    slab_telemetry = (
        copier.telemetry()
        if copier is not None
        else {
            "schema_version": "eiel_acs_physical_batch_packed_slab_pool_v1",
            "slab_count": 0,
            "native_pool": False,
            "pinned": False,
            "preallocated": False,
            "chunk_bytes": chunk_bytes,
            "disposition": "not_allocated_no_physical_payload",
        }
    )
    field_descriptors: list[dict[str, Any]] = []
    payload_bytes = 0
    common_transfers = 0
    ragged_transfers = 0
    zero_fill_logical_bytes = 0
    zero_fill_fields: list[str] = []
    try:
        with capture_path.open("xb") as stream:
            stream.write(
                PREFIX.pack(MAGIC, FORMAT_VERSION, LITTLE_ENDIAN_MARKER, 0, 0, 0)
            )
            for field_index, plan in enumerate(plans):
                if isinstance(plan, _ZeroFillFieldPlan):
                    field_descriptors.append(
                        _field_descriptor_zero_fill(
                            plan, offset=payload_bytes, field_index=field_index
                        )
                    )
                    zero_fill_logical_bytes += plan.logical_bytes
                    zero_fill_fields.append(plan.name)
                    continue
                if isinstance(plan, _CommonFieldPlan):
                    assert copier is not None
                    sink = _PayloadSink(stream, payload_offset=payload_bytes)
                    start = payload_bytes
                    copier.copy_tensor(plan.tensor, sink)
                    if sink.bytes != plan.logical_bytes:
                        raise _fail(f"common field {plan.name!r} byte coverage differs")
                    payload_bytes += sink.bytes
                    _validate_physical_payload_bytes(payload_bytes)
                    common_transfers += 1
                    field_descriptors.append(
                        _field_descriptor_common(plan, offset=start)
                    )
                    continue

                assert isinstance(plan, _RaggedFieldPlan)
                assert copier is not None
                entries: list[dict[str, Any]] = []
                start = payload_bytes
                for cell_index, (tensor, shape) in enumerate(
                    zip(plan.values, plan.shapes, strict=True)
                ):
                    sink = _PayloadSink(stream, payload_offset=payload_bytes)
                    entry_offset = payload_bytes
                    copier.copy_tensor(tensor, sink)
                    expected = _shape_bytes(plan.dtype, shape)
                    if sink.bytes != expected:
                        raise _fail(
                            f"ragged field {plan.name!r} cell {cell_index} byte coverage differs"
                        )
                    payload_bytes += sink.bytes
                    _validate_physical_payload_bytes(payload_bytes)
                    ragged_transfers += 1
                    entries.append(
                        {
                            "cell_index": cell_index,
                            "offset": entry_offset,
                            "bytes": sink.bytes,
                            "logical_bytes": expected,
                            "shape": list(shape),
                        }
                    )
                field_descriptors.append(
                    _field_descriptor_ragged(plan, entries=entries, offset=start)
                )

            operation_counts = {
                "common_field_transfer_operations": common_transfers,
                "ragged_item_transfer_operations": ragged_transfers,
                "zero_fill_field_count": len(zero_fill_fields),
                "zero_fill_transfer_operations": 0,
                "total_tensor_transfer_operations": (
                    copier.transfer_operations if copier is not None else 0
                ),
                "payload_chunk_write_operations": (
                    copier.chunk_operations if copier is not None else 0
                ),
                "batch_cells": len(normalized_cell_ids),
                "field_count": len(plans),
            }
            zero_fill_contract = (
                aggregate._canonical_fixed_mlp_zero_fill_contract()
                if zero_fill_fields
                else None
            )
            header = {
                "schema_version": SCHEMA_VERSION,
                "format_version": FORMAT_VERSION,
                "endianness": "little",
                "payload_layout": PAYLOAD_LAYOUT,
                "capture_status": "captured_unsealed",
                "batch_id": batch_id,
                "cell_ids": list(normalized_cell_ids),
                "batch_cells": len(normalized_cell_ids),
                "field_order": [plan.name for plan in plans],
                "fields": field_descriptors,
                "scalar_metadata": normalized_scalars,
                "payload_bytes": payload_bytes,
                "zero_fill_logical_bytes": zero_fill_logical_bytes,
                "zero_fill_fields": zero_fill_fields,
                "zero_fill_contract": zero_fill_contract,
                "operation_counts": operation_counts,
                "slab_pool": slab_telemetry,
                "producer_hashing": "deferred_to_host_finalize_v1",
                "producer_fsync": "deferred_to_host_finalize_v1",
                "producer_publication": "deferred_to_host_finalize_v1",
            }
            _validate_container_identity(header, payload_bytes=payload_bytes)
            _validate_zero_fill_summary(
                header,
                field_descriptors,
                batch_cells=len(normalized_cell_ids),
                payload_bytes=payload_bytes,
            )
            header_bytes = _canonical_json(header)
            if len(header_bytes) > MAX_HEADER_BYTES:
                raise _fail("packed checkpoint header exceeds the bounded limit")
            stream.write(header_bytes)
            stream.flush()
            stream.seek(0)
            stream.write(
                PREFIX.pack(
                    MAGIC,
                    FORMAT_VERSION,
                    LITTLE_ENDIAN_MARKER,
                    len(header_bytes),
                    payload_bytes,
                    0,
                )
            )
            stream.flush()
    except BaseException:
        capture_path.unlink(missing_ok=True)
        raise

    descriptor = {
        "schema_version": SCHEMA_VERSION,
        "format_version": FORMAT_VERSION,
        "descriptor_status": "captured_unsealed",
        "capture_path": str(capture_path),
        "requested_publication_path": str(target),
        "batch_id": batch_id,
        "cell_ids": list(normalized_cell_ids),
        "field_order": [plan.name for plan in plans],
        "payload_bytes": payload_bytes,
        "zero_fill_logical_bytes": zero_fill_logical_bytes,
        "zero_fill_fields": zero_fill_fields,
        "zero_fill_contract_sha256": (
            _sha256(_canonical_json(zero_fill_contract))
            if zero_fill_contract is not None
            else None
        ),
        "header_bytes": len(header_bytes),
        "header_sha256": _sha256(header_bytes),
        "operation_counts": operation_counts,
        "slab_pool": slab_telemetry,
        "sealed": False,
    }
    captured = PackedCheckpointFileFingerprintV1.from_stat(capture_path.stat())
    descriptor["captured_file_fingerprint"] = {
        "device": captured.device,
        "inode": captured.inode,
        "mode": captured.mode,
        "nlink": captured.nlink,
        "size": captured.size,
        "mtime_ns": captured.mtime_ns,
        "ctime_ns": captured.ctime_ns,
    }
    descriptor["capture_authority_sha256"] = capture_descriptor_authority_sha256_v1(
        descriptor
    )
    _assert_descriptor_scalars(descriptor)
    return descriptor


def finalize_physical_batch_packed_checkpoint_v1(
    descriptor: Mapping[str, Any],
    *,
    publication_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Hash, seal, fsync, and publish a producer descriptor on the host."""

    _assert_descriptor_scalars(descriptor)
    if descriptor.get("schema_version") != SCHEMA_VERSION:
        raise _fail("packed checkpoint descriptor schema is invalid")
    if descriptor.get("descriptor_status") != "captured_unsealed":
        raise _fail("packed checkpoint descriptor is not an unsealed capture")
    capture_path = Path(str(descriptor.get("capture_path", ""))).expanduser().resolve()
    prefix, header, footer_bytes, payload_bytes, header_len = _read_parts(
        capture_path, require_sealed=False
    )
    if footer_bytes is not None:
        raise _fail("packed checkpoint capture already contains a footer")
    cell_ids, fields = _validate_container_identity(header, payload_bytes=payload_bytes)
    if (
        descriptor.get("batch_id") != header.get("batch_id")
        or descriptor.get("cell_ids") != cell_ids
        or descriptor.get("field_order") != header.get("field_order")
        or descriptor.get("payload_bytes") != payload_bytes
        or descriptor.get("zero_fill_logical_bytes")
        != header.get("zero_fill_logical_bytes")
        or descriptor.get("zero_fill_fields") != header.get("zero_fill_fields")
        or descriptor.get("zero_fill_contract_sha256")
        != (
            _sha256(_canonical_json(header["zero_fill_contract"]))
            if header.get("zero_fill_contract") is not None
            else None
        )
    ):
        raise _fail("packed checkpoint descriptor does not match its capture file")
    _validate_zero_fill_summary(
        header,
        fields,
        batch_cells=len(cell_ids),
        payload_bytes=payload_bytes,
    )
    with capture_path.open("rb") as stream:
        stream.seek(PREFIX.size + payload_bytes)
        header_bytes = _read_exact(stream, header_len, label="header")
    payload_sha256, field_hashes, entry_hashes, zero_fill_hashes = _hash_ranges(
        capture_path, payload_bytes=payload_bytes, fields=fields
    )
    header_sha256 = _sha256(header_bytes)
    target = (
        capture_path
        if publication_path is None
        else Path(publication_path).expanduser().resolve()
    )
    if target != capture_path and target.exists():
        raise _fail(f"publication target already exists: {target}")

    footer_common: dict[str, Any] = {
        "schema_version": FOOTER_SCHEMA_VERSION,
        "status": "complete",
        "format_version": FORMAT_VERSION,
        "batch_id": header["batch_id"],
        "cell_ids": cell_ids,
        "payload_bytes": payload_bytes,
        "header_bytes": header_len,
        "header_sha256": header_sha256,
        "payload_sha256": payload_sha256,
        "field_sha256": field_hashes,
        "ragged_entry_sha256": entry_hashes,
        "zero_fill_logical_sha256": zero_fill_hashes,
        "zero_fill_logical_bytes": header["zero_fill_logical_bytes"],
        "zero_fill_fields": header["zero_fill_fields"],
        "zero_fill_contract_sha256": (
            _sha256(_canonical_json(header["zero_fill_contract"]))
            if header["zero_fill_contract"] is not None
            else None
        ),
        "operation_counts": header["operation_counts"],
        "prefix_sha256": "0" * 64,
        "seal_sha256": "0" * 64,
        "artifact_sha256": "0" * 64,
    }
    footer_length = len(_canonical_json(footer_common))
    if footer_length > MAX_FOOTER_BYTES:
        raise _fail("packed checkpoint footer exceeds the bounded limit")
    final_prefix = PREFIX.pack(
        MAGIC,
        FORMAT_VERSION,
        LITTLE_ENDIAN_MARKER,
        header_len,
        payload_bytes,
        footer_length,
    )
    footer_common["prefix_sha256"] = _sha256(final_prefix)
    footer_common["seal_sha256"] = _footer_seal_digest(footer_common)
    footer_common["artifact_sha256"] = _footer_artifact_digest(
        prefix=final_prefix,
        header_sha256=header_sha256,
        payload_sha256=payload_sha256,
        footer=footer_common,
    )
    final_footer = _canonical_json(footer_common)
    if len(final_footer) != footer_length:
        raise _fail("packed checkpoint footer length did not converge")

    with capture_path.open("r+b") as stream:
        stream.seek(0, os.SEEK_END)
        stream.write(final_footer)
        stream.flush()
        os.fsync(stream.fileno())
        stream.seek(0)
        stream.write(final_prefix)
        stream.flush()
        os.fsync(stream.fileno())

    if target != capture_path:
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(capture_path, target)
    directory_fd = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    published_stat = target.stat()
    published_fingerprint = PackedCheckpointFileFingerprintV1.from_stat(published_stat)
    return {
        "schema_version": SCHEMA_VERSION,
        "descriptor_status": "published_sealed",
        "published_path": str(target),
        "batch_id": header["batch_id"],
        "cell_ids": cell_ids,
        "payload_bytes": payload_bytes,
        "zero_fill_logical_bytes": header["zero_fill_logical_bytes"],
        "zero_fill_fields": header["zero_fill_fields"],
        "zero_fill_contract_sha256": footer_common["zero_fill_contract_sha256"],
        "header_bytes": header_len,
        "footer_bytes": len(final_footer),
        "payload_sha256": payload_sha256,
        "artifact_sha256": footer_common["artifact_sha256"],
        "published_file_fingerprint": {
            "device": published_fingerprint.device,
            "inode": published_fingerprint.inode,
            "mode": published_fingerprint.mode,
            "nlink": published_fingerprint.nlink,
            "size": published_fingerprint.size,
            "mtime_ns": published_fingerprint.mtime_ns,
            "ctime_ns": published_fingerprint.ctime_ns,
        },
        "sealed": True,
    }


def _load_tensor_range(
    stream: BinaryIO,
    *,
    offset: int,
    byte_count: int,
    dtype: str,
    shape: Sequence[int],
    device: str | torch.device,
) -> torch.Tensor:
    stream.seek(PREFIX.size + offset)
    raw = _read_exact(stream, byte_count, label="tensor payload")
    try:
        tensor = streamed._tensor_from_bytes(
            raw, dtype_name=dtype, shape=tuple(int(value) for value in shape)
        )
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise _fail("packed checkpoint tensor reconstruction failed") from exc
    if str(device) != "cpu":
        try:
            tensor = tensor.to(device)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise _fail(f"packed checkpoint cannot move tensor to {device}") from exc
    return tensor


def _load_zero_tensor(
    *, dtype: str, shape: Sequence[int], device: str | torch.device
) -> torch.Tensor:
    dtype_value = streamed._DTYPE_BY_NAME.get(dtype)
    if dtype_value is None:
        raise _fail(f"packed zero-fill dtype is invalid: {dtype}")
    try:
        return torch.zeros(
            tuple(int(value) for value in shape), dtype=dtype_value, device=device
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise _fail("packed zero-fill tensor reconstruction failed") from exc


def _validate_footer(
    *,
    prefix: bytes,
    header: Mapping[str, Any],
    header_bytes: bytes,
    footer_bytes: bytes,
    payload_bytes: int,
    path: Path | None,
    stream: BinaryIO | None = None,
) -> dict[str, Any]:
    try:
        footer = json.loads(footer_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _fail("packed checkpoint footer JSON is invalid or torn") from exc
    if not isinstance(footer, dict):
        raise _fail("packed checkpoint footer is not an object")
    if (
        footer.get("schema_version") != FOOTER_SCHEMA_VERSION
        or footer.get("status") != "complete"
    ):
        raise _fail("packed checkpoint footer is not a complete seal")
    cell_ids, fields = _validate_container_identity(header, payload_bytes=payload_bytes)
    _validate_zero_fill_summary(
        header,
        fields,
        batch_cells=len(cell_ids),
        payload_bytes=payload_bytes,
    )
    if (
        footer.get("batch_id") != header.get("batch_id")
        or footer.get("cell_ids") != cell_ids
        or footer.get("payload_bytes") != payload_bytes
        or footer.get("header_bytes") != len(header_bytes)
        or footer.get("header_sha256") != _sha256(header_bytes)
        or footer.get("prefix_sha256") != _sha256(prefix)
        or footer.get("operation_counts") != header.get("operation_counts")
        or footer.get("zero_fill_logical_bytes")
        != header.get("zero_fill_logical_bytes")
        or footer.get("zero_fill_fields") != header.get("zero_fill_fields")
        or footer.get("zero_fill_contract_sha256")
        != (
            _sha256(_canonical_json(header["zero_fill_contract"]))
            if header.get("zero_fill_contract") is not None
            else None
        )
    ):
        raise _fail("packed checkpoint seal identity does not match the container")
    payload_sha256, field_hashes, entry_hashes, zero_fill_hashes = _hash_ranges(
        path,
        payload_bytes=payload_bytes,
        fields=fields,
        stream=stream,
    )
    if footer.get("payload_sha256") != payload_sha256:
        raise _fail("packed checkpoint payload digest differs")
    if footer.get("field_sha256") != field_hashes:
        raise _fail("packed checkpoint field digest differs")
    if footer.get("ragged_entry_sha256") != entry_hashes:
        raise _fail("packed checkpoint ragged entry digest differs")
    if footer.get("zero_fill_logical_sha256") != zero_fill_hashes:
        raise _fail("packed checkpoint zero-fill logical digest differs")
    if footer.get("seal_sha256") != _footer_seal_digest(footer):
        raise _fail("packed checkpoint seal digest differs")
    if footer.get("artifact_sha256") != _footer_artifact_digest(
        prefix=prefix,
        header_sha256=_sha256(header_bytes),
        payload_sha256=payload_sha256,
        footer=footer,
    ):
        raise _fail("packed checkpoint artifact digest differs")
    return footer


def _capture_file_fingerprint(
    path: Path, stream: BinaryIO
) -> PackedCheckpointFileFingerprintV1:
    try:
        descriptor_fingerprint = PackedCheckpointFileFingerprintV1.from_stat(
            os.fstat(stream.fileno())
        )
        path_fingerprint = PackedCheckpointFileFingerprintV1.from_stat(path.stat())
    except OSError as exc:
        raise _fail(f"packed checkpoint identity cannot be read: {path}") from exc
    if descriptor_fingerprint != path_fingerprint:
        raise _fail("packed checkpoint path identity differs from its open file")
    return descriptor_fingerprint


class PhysicalBatchPackedCheckpointReaderV1:
    """A strict context for repeated reads from one sealed packed checkpoint.

    Opening the context validates the prefix, header, footer, and complete
    physical payload exactly once.  Subsequent loads read only their selected
    field ranges and are fenced by the captured file identity.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        target = Path(path).expanduser().resolve()
        stream: BinaryIO | None = None
        try:
            stream = target.open("rb")
            fingerprint = _capture_file_fingerprint(target, stream)
            prefix, header, footer_bytes, payload_bytes, header_len = (
                _read_parts_from_stream(stream, target, require_sealed=True)
            )
            if footer_bytes is None:
                raise _fail("packed checkpoint has no seal")
            stream.seek(PREFIX.size + payload_bytes)
            header_bytes = _read_exact(stream, header_len, label="header")
            footer = _validate_footer(
                prefix=prefix,
                header=header,
                header_bytes=header_bytes,
                footer_bytes=footer_bytes,
                payload_bytes=payload_bytes,
                path=None,
                stream=stream,
            )
            cell_ids, fields = _validate_container_identity(
                header, payload_bytes=payload_bytes
            )
            _validate_zero_fill_summary(
                header,
                fields,
                batch_cells=len(cell_ids),
                payload_bytes=payload_bytes,
            )
            if _capture_file_fingerprint(target, stream) != fingerprint:
                raise _fail("packed checkpoint changed during open verification")
        except (OSError, struct.error) as exc:
            if stream is not None:
                stream.close()
            raise _fail(f"packed checkpoint cannot be read: {target}") from exc
        except Exception:
            if stream is not None:
                stream.close()
            raise

        self._path = target
        self._stream = stream
        self._fingerprint = fingerprint
        self._header = header
        self._footer = footer
        self._cell_ids = cell_ids
        self._fields = fields
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def _ensure_open(self) -> BinaryIO:
        if self._closed or self._stream is None:
            raise _fail("packed checkpoint reader is closed")
        return self._stream

    def _close_after_failure(self) -> None:
        stream = self._stream
        self._stream = None
        self._closed = True
        if stream is not None:
            stream.close()

    def _assert_unchanged(self, *, phase: str) -> None:
        stream = self._ensure_open()
        try:
            current = _capture_file_fingerprint(self._path, stream)
        except PhysicalBatchPackedCheckpointError:
            self._close_after_failure()
            raise
        if current != self._fingerprint:
            self._close_after_failure()
            raise _fail(f"packed checkpoint changed {phase}")

    @property
    def path(self) -> str:
        self._ensure_open()
        return str(self._path)

    @property
    def file_fingerprint(self) -> PackedCheckpointFileFingerprintV1:
        self._ensure_open()
        return self._fingerprint

    @property
    def fingerprint(self) -> PackedCheckpointFileFingerprintV1:
        """Compatibility spelling for the captured file fingerprint."""

        return self.file_fingerprint

    @property
    def scalar_metadata(self) -> Mapping[str, Any]:
        self._ensure_open()
        return self._header["scalar_metadata"]

    @property
    def header(self) -> Mapping[str, Any]:
        self._ensure_open()
        return self._header

    @property
    def footer(self) -> Mapping[str, Any]:
        self._ensure_open()
        return self._footer

    def _selected_index(self, selected_cell: int | str | None) -> int | None:
        if selected_cell is None:
            return None
        if type(selected_cell) is int:
            if not 0 <= selected_cell < len(self._cell_ids):
                raise _fail("selected cell index is outside the physical batch")
            return selected_cell
        if isinstance(selected_cell, str):
            if selected_cell not in self._cell_ids:
                raise _fail("selected cell id is outside the physical batch")
            return self._cell_ids.index(selected_cell)
        raise _fail("selected_cell must be an integer, string, or None")

    def load(
        self,
        *,
        selected_cell: int | str | None = None,
        device: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        """Load the complete batch or one selected cell from this context."""

        stream = self._ensure_open()
        self._assert_unchanged(phase="before selected-cell load")
        selected_index = self._selected_index(selected_cell)
        try:
            loaded_fields: dict[str, Any] = {}
            for field in self._fields:
                name = str(field["name"])
                if field["kind"] == "common":
                    if selected_index is None:
                        offset = int(field["offset"])
                        shape = field["shape"]
                        byte_count = int(field["bytes"])
                    else:
                        offset = int(field["offset"]) + selected_index * int(
                            field["cell_bytes"]
                        )
                        shape = field["cell_shape"]
                        byte_count = int(field["cell_bytes"])
                    loaded_fields[name] = _load_tensor_range(
                        stream,
                        offset=offset,
                        byte_count=byte_count,
                        dtype=str(field["dtype"]),
                        shape=shape,
                        device=device,
                    )
                elif field["kind"] == "zero_fill":
                    shape = (
                        field["shape"]
                        if selected_index is None
                        else field["cell_shape"]
                    )
                    loaded_fields[name] = _load_zero_tensor(
                        dtype=str(field["dtype"]), shape=shape, device=device
                    )
                else:
                    entries = field["entries"]
                    chosen = (
                        entries if selected_index is None else [entries[selected_index]]
                    )
                    values = tuple(
                        _load_tensor_range(
                            stream,
                            offset=int(entry["offset"]),
                            byte_count=int(entry["bytes"]),
                            dtype=str(field["dtype"]),
                            shape=entry["shape"],
                            device=device,
                        )
                        for entry in chosen
                    )
                    loaded_fields[name] = (
                        values if selected_index is None else values[0]
                    )
            result = {
                "schema_version": SCHEMA_VERSION,
                "batch_id": self._header["batch_id"],
                "cell_ids": self._cell_ids,
                "selected_cell_index": selected_index,
                "selected_cell_id": (
                    None if selected_index is None else self._cell_ids[selected_index]
                ),
                "field_order": list(self._header["field_order"]),
                "fields": loaded_fields,
                "scalar_metadata": self._header["scalar_metadata"],
                "header": self._header,
                "footer": self._footer,
                "path": str(self._path),
            }
            _assert_descriptor_scalars(
                {key: value for key, value in result.items() if key != "fields"}
            )
            return result
        finally:
            self._assert_unchanged(phase="after selected-cell load")

    def load_selected_cell(
        self,
        selected_cell: int | str,
        *,
        device: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        """Load exactly one cell's field ranges."""

        return self.load(selected_cell=selected_cell, device=device)

    load_cell = load_selected_cell

    def close(self) -> None:
        """Close the context; all further reads and metadata access fail."""

        if self._closed:
            return
        self._close_after_failure()

    def __enter__(self) -> "PhysicalBatchPackedCheckpointReaderV1":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()


PackedCheckpointReader = PhysicalBatchPackedCheckpointReaderV1


def open_physical_batch_packed_checkpoint_v1(
    path: str | os.PathLike[str],
) -> PhysicalBatchPackedCheckpointReaderV1:
    """Open and fully verify one sealed packed checkpoint for repeated reads."""

    return PhysicalBatchPackedCheckpointReaderV1(path)


def load_physical_batch_packed_checkpoint_v1(
    path: str | os.PathLike[str],
    *,
    selected_cell: int | str | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Strictly load all fields or one selected cell from a sealed file."""

    with PhysicalBatchPackedCheckpointReaderV1(path) as reader:
        return reader.load(selected_cell=selected_cell, device=device)


strict_load_physical_batch_packed_checkpoint_v1 = (
    load_physical_batch_packed_checkpoint_v1
)


__all__ = [
    "FOOTER_SCHEMA_VERSION",
    "MAGIC",
    "MAX_PHYSICAL_PAYLOAD_BYTES",
    "PAYLOAD_LAYOUT",
    "PackedCheckpointFileFingerprintV1",
    "PackedCheckpointReader",
    "PhysicalBatchPackedCheckpointError",
    "PhysicalBatchPackedCheckpointReaderV1",
    "RaggedField",
    "RaggedFieldV1",
    "ZeroFillField",
    "ZeroFillFieldV1",
    "capture_physical_batch_packed_checkpoint_v1",
    "finalize_physical_batch_packed_checkpoint_v1",
    "load_physical_batch_packed_checkpoint_v1",
    "open_physical_batch_packed_checkpoint_v1",
    "strict_load_physical_batch_packed_checkpoint_v1",
]
