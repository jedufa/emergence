"""Aggregate native checkpoint container for one physical ACS batch.

The ACS lifecycle owns slot admission, FIFO backpressure, and durable
acknowledgement.  This module owns only the physical representation of one
multi-cell snapshot.  All tensor fields are copied in field-major order into a
single immutable file.  The header is an index over the field ranges and the
footer seals the header, payload, and publication identity.

The implementation deliberately reuses the validated scalar packing and
native two-slab primitives from the single-cell native format.  The aggregate
format has a distinct magic and schema so a loader can never silently confuse
the two representations.
"""

from __future__ import annotations

import hashlib
import os
import struct
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

import torch

from . import acs_native_streamed_checkpoint_v1 as streamed


SCHEMA_VERSION = "acs100x_native_aggregate_checkpoint_v2"
MAGIC = b"ACSAGV1\x00"
FORMAT_VERSION = 2
LITTLE_ENDIAN_MARKER = 0x0102
PREFIX = struct.Struct("<8sHHQQQ")
FOOTER_SCHEMA_VERSION = "acs100x_native_aggregate_checkpoint_footer_v1"
DEFAULT_CHUNK_BYTES = streamed.DEFAULT_CHUNK_BYTES
MAX_CHUNK_BYTES = streamed.MAX_CHUNK_BYTES
MAX_HEADER_BYTES = streamed.MAX_HEADER_BYTES
MAX_FOOTER_BYTES = streamed.MAX_FOOTER_BYTES
MAX_FIELD_COUNT = streamed.MAX_FIELD_COUNT
MAX_FIELD_BYTES = streamed.MAX_FIELD_BYTES
METADATA_PAYLOAD_LAYOUT = "field_major_cells_v1_metadata_payload_v1"
METADATA_SCHEMA_VERSION = "acs100x_native_aggregate_metadata_payload_v1"
LEGACY_PAYLOAD_LAYOUT = "field_major_cells_v1"
# A metadata record is the bounded scalar projection for one physical cell.
# It is deliberately bounded independently of the aggregate so the cursor
# route can serialize one cell at a time without recreating host staging.
MAX_METADATA_RECORD_BYTES = 64 * 1024 * 1024
MAX_METADATA_BYTES = 16 * 1024 * 1024 * 1024

ZERO_FILL_CANDIDATE_FIELDS = frozenset(
    {
        "gru_pack",
        "hmem",
        "policy_use_gru",
    }
)
ZERO_FILL_CONTRACT_SCHEMA_VERSION = "acs100x_fixed_mlp_zero_fill_contract_v1"
ZERO_FILL_DESCRIPTOR_SCHEMA_VERSION = "acs100x_zero_fill_descriptor_v1"


class ACSNativeAggregateCheckpointError(RuntimeError):
    """The aggregate checkpoint is malformed, conflicting, or not durable."""


def _canonical_json(value: Any) -> bytes:
    return streamed._canonical_json(value)


def _sha256(data: bytes) -> str:
    return streamed._sha256(data)


def _is_sha256(value: Any) -> bool:
    return streamed._is_sha256(value)


def _validate_int(value: Any, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ACSNativeAggregateCheckpointError(
            f"{field} must be an integer >= {minimum}"
        )
    return value


def _logical_bytes(tensor: torch.Tensor) -> int:
    value = int(tensor.numel()) * int(tensor.element_size())
    if value < 0 or value > MAX_FIELD_BYTES:
        raise ACSNativeAggregateCheckpointError(
            "aggregate checkpoint tensor field exceeds the byte bound"
        )
    return value


@lru_cache(maxsize=256)
def _zero_digest(logical_bytes: int) -> str:
    """Return the exact zero-stream digest once per logical byte length.

    Aggregate cells share tensor schemas. Re-hashing the same logical zero
    field for every cell scales with logical B rather than physical payload
    bytes and dominated both capture and restore. The digest is a pure
    function of ``logical_bytes``; this bounded process cache changes no
    checkpoint authority or corruption check.
    """

    digest = hashlib.sha256()
    block = bytes(min(8 * 1024 * 1024, max(1, logical_bytes)))
    remaining = logical_bytes
    while remaining:
        length = min(len(block), remaining)
        digest.update(block[:length])
        remaining -= length
    return digest.hexdigest()


def _zero_digest_from_cache(cache: dict[int, str], logical_bytes: int) -> str:
    value = cache.get(logical_bytes)
    if value is None:
        value = _zero_digest(logical_bytes)
        cache[logical_bytes] = value
    return value


class _AggregateFieldSink:
    """Write one field while retaining payload and field-local hashes."""

    def __init__(self, stream: Any, *, offset: int, payload_digest: Any) -> None:
        self.stream = stream
        self.offset = offset
        self.payload_digest = payload_digest
        self.bytes = 0
        self.chunks: list[dict[str, Any]] = []
        self.digest = hashlib.sha256()
        self.write_seconds = 0.0
        self.hash_seconds = 0.0
        self.chunk_descriptor_seconds = 0.0

    def write(self, value: bytes | bytearray | memoryview) -> int:
        raw = bytes(value)
        write_started = time.perf_counter()
        self.stream.write(raw)
        self.write_seconds += time.perf_counter() - write_started
        hash_started = time.perf_counter()
        self.payload_digest.update(raw)
        self.digest.update(raw)
        self.hash_seconds += time.perf_counter() - hash_started
        descriptor_started = time.perf_counter()
        self.chunks.append(
            {
                "offset": self.offset + self.bytes,
                "bytes": len(raw),
                "sha256": _sha256(raw),
            }
        )
        self.chunk_descriptor_seconds += time.perf_counter() - descriptor_started
        self.bytes += len(raw)
        return len(raw)


def _copy_tensor(
    tensor: torch.Tensor,
    sink: _AggregateFieldSink,
    *,
    chunk_bytes: int,
    fallback_pool: Any,
    native_pool: Any | None,
    dedicated_stream: Any | None,
    producer_event: Any | None = None,
) -> int:
    """Copy one tensor through the shared pool and return chunk count."""

    if native_pool is not None:
        if dedicated_stream is not None:
            if producer_event is not None:
                dedicated_stream.wait_event(producer_event)
            with torch.cuda.stream(dedicated_stream):
                count = native_pool.copy_tensor_chunks(tensor, sink)
            dedicated_stream.synchronize()
        else:
            count = native_pool.copy_tensor_chunks(tensor, sink)
        if type(count) is not int or count < 0:
            raise ACSNativeAggregateCheckpointError(
                "native aggregate tensor copy returned an invalid chunk count"
            )
        return count

    native_copy = streamed._native_chunk_copy()
    if dedicated_stream is not None and native_copy is not None:
        if producer_event is not None:
            dedicated_stream.wait_event(producer_event)
        with torch.cuda.stream(dedicated_stream):
            count = native_copy(tensor, sink, chunk_bytes)
        dedicated_stream.synchronize()
        if type(count) is not int or count < 0:
            raise ACSNativeAggregateCheckpointError(
                "native aggregate tensor copy returned an invalid chunk count"
            )
        return count

    count = 0
    for raw in streamed._iter_tensor_chunks(
        tensor,
        chunk_bytes=chunk_bytes,
        slab_pool=fallback_pool,
    ):
        sink.write(raw)
        count += 1
    return count


def _route_policy_family(non_tensors: Mapping[str, Any]) -> str:
    direct = non_tensors.get("native_runtime_fixed_policy_family")
    metadata = non_tensors.get("metadata")
    nested = metadata.get("native_runtime_fixed_policy_family") if isinstance(metadata, Mapping) else None
    value = direct if direct is not None else nested
    return str(value or "").strip().lower()


def _fixed_mlp_zero_fill_contract(
    non_tensors: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return the exact route contract that permits zero-fill elision."""

    metadata = non_tensors.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    required = {
        "world": "patch",
        "native_runtime_fixed_policy_family": "mlp",
        "native_runtime_patch_profile": "P2",
        "native_runtime_patch_profile_scope": "full_profile",
        "policy_weight_mutation_enable": False,
    }
    if any(metadata.get(name) != expected for name, expected in required.items()):
        return None
    return _canonical_fixed_mlp_zero_fill_contract()


def _canonical_fixed_mlp_zero_fill_contract() -> dict[str, Any]:
    return {
        "schema_version": ZERO_FILL_CONTRACT_SCHEMA_VERSION,
        "route_id": "acs100x_patch_p2_fixed_mlp_v1",
        "policy_family": "mlp",
        "fields": [
            {
                "field": field,
                "guaranteed_zero": True,
                "future_read": False,
                "restore": "exact_zero_v1",
            }
            for field in sorted(ZERO_FILL_CANDIDATE_FIELDS)
        ],
    }


def _zero_fill_descriptor(
    *,
    field: str,
    dtype: str,
    shape: Sequence[int],
    field_index: int,
    cell_index: int,
    logical_bytes: int,
    logical_digest_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": ZERO_FILL_DESCRIPTOR_SCHEMA_VERSION,
        "field": field,
        "dtype": dtype,
        "shape": list(shape),
        "layout": "C_contiguous",
        "order": {
            "scheme": "field_major_cells_v1",
            "field_index": field_index,
            "cell_index": cell_index,
        },
        "logical_bytes": logical_bytes,
        "zero_fill": {"kind": "exact_zero", "value": 0},
        "logical_digest_sha256": logical_digest_sha256,
    }


def _value_pool_add(
    value_pool: list[Any],
    value_pool_index: dict[bytes, int],
    value: Any,
) -> int:
    packed = streamed._pack_value(value)
    key = _canonical_json(packed)
    existing = value_pool_index.get(key)
    if existing is not None:
        return existing
    index = len(value_pool)
    value_pool_index[key] = index
    value_pool.append(packed)
    return index


def _metadata_record_bytes(
    *,
    projection: Mapping[str, Any],
    identity: Mapping[str, Any],
    observer: Mapping[str, Any],
) -> bytes:
    """Pack one cell's scalar checkpoint metadata for the payload region.

    The old aggregate index kept these packed values in ``value_pool``.  That
    made the JSON header grow with the complete causal/event projection for
    every cell.  The v2 metadata payload keeps the same lossless packing and
    digest authority, but materializes only the current cell's record while
    the producer advances.
    """

    raw = _canonical_json(
        {
            "schema_version": METADATA_SCHEMA_VERSION,
            "projection": streamed._pack_value(projection),
            "architecture_b_identity": streamed._pack_value(identity),
            "observer_continuation": streamed._pack_value(observer),
        }
    )
    if not 0 < len(raw) <= MAX_METADATA_RECORD_BYTES:
        raise ACSNativeAggregateCheckpointError(
            "aggregate cell metadata record exceeds its bounded scratch limit"
        )
    return raw


def _metadata_ref(raw: bytes, *, offset: int) -> dict[str, Any]:
    if type(offset) is not int or offset < 0:
        raise ACSNativeAggregateCheckpointError(
            "aggregate metadata payload offset is invalid"
        )
    return {
        "offset": offset,
        "bytes": len(raw),
        "sha256": _sha256(raw),
    }


def _content_digest(*, header_sha256: str, payload_sha256: str) -> str:
    return _sha256(
        _canonical_json(
            {
                "header_sha256": header_sha256,
                "payload_sha256": payload_sha256,
            }
        )
    )


def _seal_digest(
    *,
    content_sha256: str,
    header_sha256: str,
    payload_sha256: str,
    batch_id: str,
    cell_ids: Sequence[str],
) -> str:
    return _sha256(
        _canonical_json(
            {
                "batch_id": batch_id,
                "cell_ids": list(cell_ids),
                "content_sha256": content_sha256,
                "header_sha256": header_sha256,
                "payload_sha256": payload_sha256,
                "schema_version": SCHEMA_VERSION,
            }
        )
    )


def _artifact_digest(
    *,
    prefix: bytes,
    header_sha256: str,
    payload_sha256: str,
    content_sha256: str,
    footer_without_artifact: Mapping[str, Any],
) -> str:
    return _sha256(
        _canonical_json(
            {
                "content_sha256": content_sha256,
                "footer": dict(footer_without_artifact),
                "header_sha256": header_sha256,
                "payload_sha256": payload_sha256,
                "prefix_sha256": _sha256(prefix),
            }
        )
    )


def _footer_without_artifact(
    *,
    prefix: bytes,
    header_bytes: bytes,
    payload_sha256: str,
    content_sha256: str,
    batch_id: str,
    cell_ids: Sequence[str],
    payload_bytes: int,
) -> dict[str, Any]:
    header_sha256 = _sha256(header_bytes)
    return {
        "schema_version": FOOTER_SCHEMA_VERSION,
        "status": "complete",
        "batch_id": batch_id,
        "cell_ids": list(cell_ids),
        "payload_bytes": payload_bytes,
        "header_bytes": len(header_bytes),
        "prefix_sha256": _sha256(prefix),
        "header_sha256": header_sha256,
        "payload_sha256": payload_sha256,
        "content_sha256": content_sha256,
        "seal_sha256": _seal_digest(
            content_sha256=content_sha256,
            header_sha256=header_sha256,
            payload_sha256=payload_sha256,
            batch_id=batch_id,
            cell_ids=cell_ids,
        ),
    }


def write_acs_native_aggregate_checkpoint(
    projections: Sequence[Mapping[str, Any]],
    destination: str | os.PathLike[str],
    *,
    cell_ids: Sequence[str],
    generation: int,
    tick: int,
    absolute_next_tick: int | None,
    batch_id: str,
    segment_ordinal: int,
    source_projection_sha256s: Mapping[str, str],
    architecture_b_identities: Mapping[str, Mapping[str, Any]],
    observer_continuations: Mapping[str, Mapping[str, Any]],
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    require_native_extension: bool = False,
    performance_telemetry: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Write one field-major aggregate checkpoint and return its receipt."""

    if not projections:
        raise ACSNativeAggregateCheckpointError("aggregate checkpoint has no cells")
    if len(projections) != len(cell_ids):
        raise ACSNativeAggregateCheckpointError(
            "aggregate checkpoint projection and cell counts differ"
        )
    if any(not isinstance(cell_id, str) or not cell_id for cell_id in cell_ids):
        raise ACSNativeAggregateCheckpointError("aggregate checkpoint cell ids are invalid")
    if len(set(cell_ids)) != len(cell_ids):
        raise ACSNativeAggregateCheckpointError("aggregate checkpoint cell ids are duplicated")
    if type(chunk_bytes) is not int or not 0 < chunk_bytes <= MAX_CHUNK_BYTES:
        raise ACSNativeAggregateCheckpointError(
            f"chunk_bytes must be in the range 1..{MAX_CHUNK_BYTES}"
        )
    _validate_int(generation, field="generation")
    _validate_int(tick, field="tick")
    if absolute_next_tick is not None:
        _validate_int(absolute_next_tick, field="absolute_next_tick")
    _validate_int(segment_ordinal, field="segment_ordinal")
    if not isinstance(batch_id, str) or not batch_id:
        raise ACSNativeAggregateCheckpointError("batch_id is invalid")
    if set(source_projection_sha256s) != set(cell_ids):
        raise ACSNativeAggregateCheckpointError(
            "source digest authority does not cover every aggregate cell"
        )
    if set(architecture_b_identities) != set(cell_ids):
        raise ACSNativeAggregateCheckpointError(
            "architecture identity authority does not cover every aggregate cell"
        )
    if set(observer_continuations) != set(cell_ids):
        raise ACSNativeAggregateCheckpointError(
            "observer continuation authority does not cover every aggregate cell"
        )
    for cell_id in cell_ids:
        if not _is_sha256(source_projection_sha256s[cell_id]):
            raise ACSNativeAggregateCheckpointError(
                f"source digest for {cell_id} is not a SHA-256 digest"
            )
        if not isinstance(architecture_b_identities[cell_id], Mapping):
            raise ACSNativeAggregateCheckpointError(
                f"architecture identity for {cell_id} is invalid"
            )
        if not isinstance(observer_continuations[cell_id], Mapping):
            raise ACSNativeAggregateCheckpointError(
                f"observer continuation for {cell_id} is invalid"
            )

    non_tensors_by_cell: list[dict[str, Any]] = []
    tensors_by_cell: list[dict[str, torch.Tensor]] = []
    for projection in projections:
        non_tensors, tensors = streamed._validate_projection(projection)
        non_tensors_by_cell.append(non_tensors)
        tensors_by_cell.append(tensors)
    field_order = sorted(tensors_by_cell[0])
    if not field_order:
        raise ACSNativeAggregateCheckpointError("aggregate checkpoint has no tensor fields")
    if len(field_order) > MAX_FIELD_COUNT:
        raise ACSNativeAggregateCheckpointError("aggregate tensor field count exceeds the bound")
    expected_fields = set(field_order)
    for tensors in tensors_by_cell:
        if set(tensors) != expected_fields:
            raise ACSNativeAggregateCheckpointError(
                "aggregate cells do not expose one identical tensor field schema"
            )

    field_schema: dict[str, dict[str, Any]] = {}
    per_cell_field_schema = False
    for field in field_order:
        first = tensors_by_cell[0][field]
        dtype_name = streamed._dtype_name(first)
        shape = [int(value) for value in first.shape]
        logical = _logical_bytes(first)
        field_schema[field] = {
            "field": field,
            "dtype": dtype_name,
            "dtype_code": streamed._DTYPE_CODE[dtype_name],
            "shape": shape,
            "logical_bytes": logical,
        }
        for tensors in tensors_by_cell[1:]:
            tensor = tensors[field]
            if (
                streamed._dtype_name(tensor) != dtype_name
                or [int(value) for value in tensor.shape] != shape
            ):
                per_cell_field_schema = True

    cells_header: list[dict[str, Any]] = []
    for cell_id, non_tensors in zip(cell_ids, non_tensors_by_cell, strict=True):
        identity = dict(architecture_b_identities[cell_id])
        source_digest = source_projection_sha256s[cell_id]
        for identity_field in ("snapshot_digest_sha256", "source_projection_sha256"):
            identity_digest = identity.get(identity_field)
            if identity_digest is not None and identity_digest != source_digest:
                raise ACSNativeAggregateCheckpointError(
                    f"architecture identity digest binding mismatch for {cell_id}"
                )
        cells_header.append(
            {
                "cell_id": cell_id,
                "source_projection_sha256": source_digest,
                "metadata_ref": None,
                "fields": [],
            }
        )

    zero_contracts = [
        _fixed_mlp_zero_fill_contract(non_tensors)
        for non_tensors in non_tensors_by_cell
    ]
    zero_contract_families = [
        _route_policy_family(non_tensors) == "mlp"
        for non_tensors in non_tensors_by_cell
    ]
    if any(zero_contract_families) and not all(zero_contract_families):
        raise ACSNativeAggregateCheckpointError(
            "aggregate cells disagree about the fixed-MLP zero-fill contract"
        )
    if any(zero_contract_families) and any(
        contract is None for contract in zero_contracts
    ):
        raise ACSNativeAggregateCheckpointError(
            "fixed-MLP zero-fill contract is missing or incompatible"
        )
    if any(contract != zero_contracts[0] for contract in zero_contracts[1:]):
        raise ACSNativeAggregateCheckpointError(
            "aggregate cells disagree about the fixed-MLP zero-fill contract"
        )
    zero_contract = zero_contracts[0] if zero_contracts else None
    if zero_contract is not None:
        missing = sorted(ZERO_FILL_CANDIDATE_FIELDS - expected_fields)
        if missing:
            raise ACSNativeAggregateCheckpointError(
                "fixed-MLP zero-fill contract is missing candidate fields: "
                + ", ".join(missing)
            )

    target = Path(destination).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{uuid4().hex}.tmp")
    payload_digest = hashlib.sha256()
    payload_bytes = 0
    total_logical_bytes = 0
    zero_fill_logical_bytes = 0
    chunk_count = 0
    zero_digest_cache: dict[int, str] = {}
    aggregate_copy_wall_seconds = 0.0
    aggregate_payload_write_seconds = 0.0
    aggregate_hash_seconds = 0.0
    aggregate_chunk_descriptor_seconds = 0.0
    aggregate_zero_validation_seconds = 0.0
    aggregate_fsync_seconds = 0.0
    aggregate_seal_seconds = 0.0
    aggregate_publication_seconds = 0.0
    cuda_fields = any(
        tensor.device.type == "cuda"
        for tensors in tensors_by_cell
        for tensor in tensors.values()
    )
    native_pool_type = streamed._native_slab_pool_type()
    if require_native_extension and native_pool_type is None:
        raise ACSNativeAggregateCheckpointError(
            "aggregate production route requires the reusable native two-slab pool"
        )
    native_pool = native_pool_type(chunk_bytes, cuda_fields) if native_pool_type is not None else None
    fallback_pool = None
    dedicated_stream = None
    if native_pool is None:
        fallback_pool = streamed._TwoSlabPool(chunk_bytes, cuda=cuda_fields)
    if cuda_fields:
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable")
            dedicated_stream = torch.cuda.Stream()
        except (RuntimeError, ValueError) as exc:
            if require_native_extension:
                raise ACSNativeAggregateCheckpointError(
                    f"aggregate dedicated CUDA stream unavailable: {exc}"
                ) from exc
            cuda_fields = False

    if native_pool is not None:
        try:
            native_pool_telemetry = dict(native_pool.telemetry())
        except (AttributeError, TypeError, ValueError) as exc:
            if require_native_extension:
                raise ACSNativeAggregateCheckpointError(
                    f"aggregate native slab pool telemetry is unavailable: {exc}"
                ) from exc
            native_pool_telemetry = {}
    else:
        native_pool_telemetry = {
            "schema_version": "eiel_acs_native_slab_pool_fallback_v1",
            "chunk_bytes": chunk_bytes,
            "slab_count": 2,
            "pinned": False,
            "preallocated": False,
        }
    if require_native_extension and cuda_fields and (
        native_pool is None
        or native_pool_telemetry.get("pinned") is not True
        or native_pool_telemetry.get("slab_count") != 2
        or native_pool_telemetry.get("preallocated") is not True
    ):
        raise ACSNativeAggregateCheckpointError(
            "aggregate CUDA production capture requires two preallocated pinned slabs"
        )

    started = time.perf_counter()
    try:
        with temporary.open("xb") as stream:
            stream.write(b"\x00" * PREFIX.size)
            payload_offset = 0
            for field_index, field in enumerate(field_order):
                schema = field_schema[field]
                for cell_index, (cell_id, tensors) in enumerate(
                    zip(cell_ids, tensors_by_cell, strict=True)
                ):
                    tensor = tensors[field]
                    tensor_dtype = streamed._dtype_name(tensor)
                    tensor_shape = [int(value) for value in tensor.shape]
                    logical = _logical_bytes(tensor)
                    total_logical_bytes += logical
                    zero_fill = (
                        zero_contract is not None
                        and field in ZERO_FILL_CANDIDATE_FIELDS
                    )
                    if zero_fill:
                        zero_validation_started = time.perf_counter()
                        try:
                            nonzero = int(torch.count_nonzero(tensor.detach()).item())
                        except (RuntimeError, TypeError, ValueError) as exc:
                            raise ACSNativeAggregateCheckpointError(
                                f"cannot prove zero-fill field {field} for {cell_id}"
                            ) from exc
                        if nonzero != 0:
                            raise ACSNativeAggregateCheckpointError(
                                f"fixed-MLP zero-fill field is nonzero: {field}/{cell_id}"
                            )
                        aggregate_zero_validation_seconds += (
                            time.perf_counter() - zero_validation_started
                        )
                        zero_fill_logical_bytes += logical
                        zero_digest = _zero_digest_from_cache(
                            zero_digest_cache, logical
                        )
                        row = {
                            "field": field,
                            "storage": "zero_fill",
                            "offset": None,
                            "bytes": 0,
                            "logical_bytes": logical,
                            "dtype": tensor_dtype,
                            "shape": tensor_shape,
                            "sha256": zero_digest,
                            "chunks": [],
                            "zero_fill_contract": "fixed_mlp_exact_zero_v1",
                            "zero_fill_descriptor": _zero_fill_descriptor(
                                field=field,
                                dtype=tensor_dtype,
                                shape=tensor_shape,
                                field_index=field_index,
                                cell_index=cell_index,
                                logical_bytes=logical,
                                logical_digest_sha256=zero_digest,
                            ),
                        }
                    else:
                        sink = _AggregateFieldSink(
                            stream,
                            offset=payload_offset,
                            payload_digest=payload_digest,
                        )
                        copy_started = time.perf_counter()
                        copied_chunks = _copy_tensor(
                            tensor,
                            sink,
                            chunk_bytes=chunk_bytes,
                            fallback_pool=fallback_pool,
                            native_pool=native_pool,
                            dedicated_stream=dedicated_stream,
                        )
                        aggregate_copy_wall_seconds += (
                            time.perf_counter() - copy_started
                        )
                        aggregate_payload_write_seconds += sink.write_seconds
                        aggregate_hash_seconds += sink.hash_seconds
                        aggregate_chunk_descriptor_seconds += (
                            sink.chunk_descriptor_seconds
                        )
                        if sink.bytes != logical:
                            raise ACSNativeAggregateCheckpointError(
                                f"aggregate field byte count mismatch: {field}/{cell_id}"
                            )
                        expected_chunks = (logical + chunk_bytes - 1) // chunk_bytes if logical else 0
                        if copied_chunks != expected_chunks:
                            raise ACSNativeAggregateCheckpointError(
                                f"aggregate field chunk count mismatch: {field}/{cell_id}"
                            )
                        row = {
                            "field": field,
                            "storage": "payload",
                            "offset": payload_offset,
                            "bytes": sink.bytes,
                            "logical_bytes": logical,
                            "dtype": tensor_dtype,
                            "shape": tensor_shape,
                            "sha256": sink.digest.hexdigest(),
                            "chunks": sink.chunks,
                            "zero_fill_contract": None,
                            "zero_fill_descriptor": None,
                        }
                        payload_offset += sink.bytes
                        payload_bytes += sink.bytes
                        chunk_count += len(sink.chunks)
                    cells_header[cell_index]["fields"].append(row)

            metadata_offset = payload_bytes
            metadata_bytes = 0
            metadata_write_started = time.perf_counter()
            for cell_index, (cell_id, non_tensors) in enumerate(
                zip(cell_ids, non_tensors_by_cell, strict=True)
            ):
                identity = dict(architecture_b_identities[cell_id])
                observer = dict(observer_continuations[cell_id])
                raw = _metadata_record_bytes(
                    projection=non_tensors,
                    identity=identity,
                    observer=observer,
                )
                stream.write(raw)
                payload_digest.update(raw)
                cells_header[cell_index]["metadata_ref"] = _metadata_ref(
                    raw, offset=metadata_offset + metadata_bytes
                )
                metadata_bytes += len(raw)
            aggregate_payload_write_seconds += (
                time.perf_counter() - metadata_write_started
            )
            file_payload_bytes = payload_bytes + metadata_bytes
            header = {
                "schema_version": SCHEMA_VERSION,
                "format_version": FORMAT_VERSION,
                "endianness": "little",
                "payload_layout": METADATA_PAYLOAD_LAYOUT,
                "metadata_schema_version": METADATA_SCHEMA_VERSION,
                "batch_id": batch_id,
                "generation": generation,
                "tick": tick,
                "absolute_next_tick": absolute_next_tick,
                "segment_ordinal": segment_ordinal,
                "cell_ids": list(cell_ids),
                "field_order": list(field_order),
                "field_schema": [field_schema[field] for field in field_order],
                "field_schema_mode": (
                    "per_cell_v1" if per_cell_field_schema else "shared_v1"
                ),
                "cells": cells_header,
                "chunk_bytes": chunk_bytes,
                "chunk_count": chunk_count,
                "payload_bytes": payload_bytes,
                "file_payload_bytes": file_payload_bytes,
                "metadata_offset": metadata_offset,
                "metadata_bytes": metadata_bytes,
                "logical_bytes": total_logical_bytes,
                "zero_fill_logical_bytes": zero_fill_logical_bytes,
                "zero_fill_contract": zero_contract,
                "zero_fill_fields": sorted(
                    field
                    for field in field_order
                    if zero_contract is not None
                    and field in ZERO_FILL_CANDIDATE_FIELDS
                ),
            }
            header_bytes = _canonical_json(header)
            if len(header_bytes) > MAX_HEADER_BYTES:
                raise ACSNativeAggregateCheckpointError(
                    "aggregate checkpoint header exceeds the bound"
                )
            seal_started = time.perf_counter()
            payload_sha256 = payload_digest.hexdigest()
            header_sha256 = _sha256(header_bytes)
            content_sha256 = _content_digest(
                header_sha256=header_sha256,
                payload_sha256=payload_sha256,
            )
            provisional_prefix = PREFIX.pack(
                MAGIC,
                FORMAT_VERSION,
                LITTLE_ENDIAN_MARKER,
                len(header_bytes),
                file_payload_bytes,
                0,
            )
            provisional_footer = _footer_without_artifact(
                prefix=provisional_prefix,
                header_bytes=header_bytes,
                payload_sha256=payload_sha256,
                content_sha256=content_sha256,
                batch_id=batch_id,
                cell_ids=cell_ids,
                payload_bytes=file_payload_bytes,
            )
            footer_length = len(_canonical_json({**provisional_footer, "artifact_sha256": "0" * 64}))
            prefix = PREFIX.pack(
                MAGIC,
                FORMAT_VERSION,
                LITTLE_ENDIAN_MARKER,
                len(header_bytes),
                file_payload_bytes,
                footer_length,
            )
            footer_without_artifact = _footer_without_artifact(
                prefix=prefix,
                header_bytes=header_bytes,
                payload_sha256=payload_sha256,
                content_sha256=content_sha256,
                batch_id=batch_id,
                cell_ids=cell_ids,
                payload_bytes=file_payload_bytes,
            )
            footer = {
                **footer_without_artifact,
                "artifact_sha256": _artifact_digest(
                    prefix=prefix,
                    header_sha256=header_sha256,
                    payload_sha256=payload_sha256,
                    content_sha256=content_sha256,
                    footer_without_artifact=footer_without_artifact,
                ),
            }
            footer_bytes = _canonical_json(footer)
            if len(footer_bytes) != footer_length or len(footer_bytes) > MAX_FOOTER_BYTES:
                raise ACSNativeAggregateCheckpointError(
                    "aggregate checkpoint footer length did not converge"
                )
            aggregate_seal_seconds += time.perf_counter() - seal_started
            tail_write_started = time.perf_counter()
            stream.write(header_bytes)
            stream.write(footer_bytes)
            aggregate_payload_write_seconds += time.perf_counter() - tail_write_started
            stream.flush()
            stream.seek(0)
            stream.write(prefix)
            stream.flush()
            fsync_started = time.perf_counter()
            os.fsync(stream.fileno())
            aggregate_fsync_seconds += time.perf_counter() - fsync_started
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    if native_pool is not None:
        try:
            native_pool_telemetry = dict(native_pool.telemetry())
        except (AttributeError, TypeError, ValueError) as exc:
            temporary.unlink(missing_ok=True)
            raise ACSNativeAggregateCheckpointError(
                f"aggregate native slab pool telemetry is unavailable after capture: {exc}"
            ) from exc

    publication_started = time.perf_counter()
    try:
        try:
            os.link(temporary, target)
        except FileExistsError:
            try:
                existing = load_acs_native_aggregate_checkpoint(
                    target, cell_id=cell_ids[0]
                )
            except ACSNativeAggregateCheckpointError as exc:
                raise ACSNativeAggregateCheckpointError(
                    "conflicting aggregate checkpoint publication already exists"
                ) from exc
            if existing["footer"].get("artifact_sha256") != footer["artifact_sha256"]:
                raise ACSNativeAggregateCheckpointError(
                    "conflicting aggregate checkpoint publication already exists"
                )
        else:
            directory_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    else:
        temporary.unlink(missing_ok=True)
    aggregate_publication_seconds = time.perf_counter() - publication_started

    cell_receipts: dict[str, dict[str, Any]] = {}
    for cell in cells_header:
        physical = sum(
            int(row["bytes"])
            for row in cell["fields"]
            if row["storage"] == "payload"
        )
        logical = sum(int(row["logical_bytes"]) for row in cell["fields"])
        zero_logical = sum(
            int(row["logical_bytes"])
            for row in cell["fields"]
            if row["storage"] == "zero_fill"
        )
        cell_receipts[str(cell["cell_id"])] = {
            "cell_id": str(cell["cell_id"]),
            "source_projection_sha256": str(cell["source_projection_sha256"]),
            "logical_bytes": logical,
            "physical_bytes": physical,
            "zero_fill_logical_bytes": zero_logical,
            "chunk_count": sum(len(row["chunks"]) for row in cell["fields"]),
            "field_count": len(cell["fields"]),
        }
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "path": str(target),
        "batch_id": batch_id,
        "generation": generation,
        "tick": tick,
        "absolute_next_tick": absolute_next_tick,
        "segment_ordinal": segment_ordinal,
        "header_sha256": footer["header_sha256"],
        "payload_sha256": footer["payload_sha256"],
        "content_sha256": footer["content_sha256"],
        "seal_sha256": footer["seal_sha256"],
        "artifact_sha256": footer["artifact_sha256"],
        "header_bytes": len(header_bytes),
        "payload_bytes": payload_bytes,
        "metadata_bytes": metadata_bytes,
        "file_payload_bytes": file_payload_bytes,
        "logical_bytes": total_logical_bytes,
        "zero_fill_logical_bytes": zero_fill_logical_bytes,
        "chunk_count": chunk_count,
        "cell_count": len(cell_ids),
        "cells": cell_receipts,
        "zero_fill_fields": header["zero_fill_fields"],
        "native_slab_pool": native_pool_telemetry,
        "physical_file_bytes": target.stat().st_size,
    }
    if performance_telemetry is not None:
        performance_telemetry.clear()
        performance_telemetry.update(
            {
                "aggregate_native_pack_payload_bytes": float(payload_bytes),
                "aggregate_native_pack_file_payload_bytes": float(file_payload_bytes),
                "aggregate_native_pack_metadata_payload_bytes": float(metadata_bytes),
                "aggregate_native_pack_logical_bytes": float(total_logical_bytes),
                "aggregate_native_pack_zero_fill_logical_bytes": float(zero_fill_logical_bytes),
                "aggregate_native_pack_metadata_bytes": float(len(header_bytes)),
                "aggregate_native_pack_chunk_count": float(chunk_count),
                "aggregate_native_pack_field_count": float(len(field_order) * len(cell_ids)),
                "aggregate_native_pack_two_slab_pool": 1.0,
                "aggregate_native_pack_pinned_slabs": float(
                    native_pool_telemetry.get("pinned") is True
                ),
                "aggregate_native_pack_prebuilt_extension": float(native_pool is not None),
                "aggregate_native_pack_dedicated_d2h_stream": float(dedicated_stream is not None),
                "aggregate_native_pack_copy_wall_seconds": aggregate_copy_wall_seconds,
                "aggregate_native_pack_payload_write_seconds": aggregate_payload_write_seconds,
                "aggregate_native_pack_hash_seconds": aggregate_hash_seconds,
                "aggregate_native_pack_chunk_descriptor_seconds": aggregate_chunk_descriptor_seconds,
                "aggregate_native_pack_zero_validation_seconds": aggregate_zero_validation_seconds,
                "aggregate_native_pack_fsync_seconds": aggregate_fsync_seconds,
                "aggregate_native_pack_seal_seconds": aggregate_seal_seconds,
                "aggregate_native_pack_publication_seconds": aggregate_publication_seconds,
                "aggregate_native_pack_physical_file_bytes": float(target.stat().st_size),
                "aggregate_native_pack_seconds": time.perf_counter() - started,
            }
        )
    return receipt


def write_acs_native_aggregate_checkpoint_from_cursor(
    cursor: Any,
    destination: str | os.PathLike[str],
    *,
    cell_ids: Sequence[str],
    generation: int,
    tick: int,
    absolute_next_tick: int | None,
    batch_id: str,
    segment_ordinal: int,
    authority_factory: Callable[[str, Mapping[str, Any]], Mapping[str, Any]],
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    require_native_extension: bool = False,
    performance_telemetry: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Write one aggregate checkpoint from a strict one-cell cursor.

    The ordinary aggregate writer receives a ``Sequence`` because its
    historical callers already own every projection.  Tick-native settlement
    intentionally exposes only ``CheckpointProjectionCursorV1``: one
    tensor-bearing cell may be live at a time.  This writer preserves the
    aggregate field-major wire format without widening that bound.  It spools
    field bytes to one temporary file while the cursor advances, then copies
    those ranges into the final field-major payload.

    ``authority_factory`` is called before the current cursor cell is
    released.  It must return the scalar source digest, architecture identity,
    observer continuation, and layered checkpoint digests for that cell.  The
    returned object is never allowed to contain tensor-bearing state.
    """

    normalized_cell_ids = tuple(cell_ids)
    if not normalized_cell_ids:
        raise ACSNativeAggregateCheckpointError(
            "aggregate checkpoint has no cells"
        )
    if any(not isinstance(cell_id, str) or not cell_id for cell_id in normalized_cell_ids):
        raise ACSNativeAggregateCheckpointError(
            "aggregate checkpoint cell ids are invalid"
        )
    if len(set(normalized_cell_ids)) != len(normalized_cell_ids):
        raise ACSNativeAggregateCheckpointError(
            "aggregate checkpoint cell ids are duplicated"
        )
    if tuple(getattr(cursor, "cell_ids", ())) != normalized_cell_ids:
        raise ACSNativeAggregateCheckpointError(
            "aggregate checkpoint cursor cell order differs from the batch"
        )
    if not callable(authority_factory):
        raise ACSNativeAggregateCheckpointError(
            "aggregate checkpoint authority factory is not callable"
        )
    if type(chunk_bytes) is not int or not 0 < chunk_bytes <= MAX_CHUNK_BYTES:
        raise ACSNativeAggregateCheckpointError(
            f"chunk_bytes must be in the range 1..{MAX_CHUNK_BYTES}"
        )
    _validate_int(generation, field="generation")
    _validate_int(tick, field="tick")
    if absolute_next_tick is not None:
        _validate_int(absolute_next_tick, field="absolute_next_tick")
    _validate_int(segment_ordinal, field="segment_ordinal")
    if not isinstance(batch_id, str) or not batch_id:
        raise ACSNativeAggregateCheckpointError("batch_id is invalid")

    target = Path(destination).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    token = f".{target.name}.{os.getpid()}.{uuid4().hex}"
    spool_path = target.with_name(f"{token}.spool.tmp")
    temporary = target.with_name(f"{token}.tmp")

    started = time.perf_counter()
    started_cpu = time.process_time_ns()
    field_order: list[str] | None = None
    field_schema: dict[str, dict[str, Any]] = {}
    per_cell_field_schema = False
    cells_header: list[dict[str, Any]] = []
    cell_field_rows: list[list[dict[str, Any]]] = []
    metadata_spool_refs: list[dict[str, Any]] = []
    authorities_by_cell: dict[str, dict[str, Any]] = {}
    spool_digest = hashlib.sha256()
    spool_bytes = 0
    total_logical_bytes = 0
    zero_fill_logical_bytes = 0
    zero_contracts: list[dict[str, Any] | None] = []
    zero_fill_families: list[bool] = []
    copy_wall_seconds = 0.0
    payload_write_seconds = 0.0
    hash_seconds = 0.0
    chunk_descriptor_seconds = 0.0
    zero_validation_seconds = 0.0
    zero_digest_cache: dict[int, str] = {}
    native_pool: Any | None = None
    fallback_pool: Any | None = None
    dedicated_stream: Any | None = None
    cuda_fields: bool | None = None
    cursor_receipt: Any | None = None
    bounded_projection_aggregate: Mapping[str, Any] | None = None

    def initialize_pool(has_cuda_fields: bool) -> None:
        nonlocal native_pool, fallback_pool, dedicated_stream, cuda_fields
        if cuda_fields is not None:
            if cuda_fields is not has_cuda_fields:
                raise ACSNativeAggregateCheckpointError(
                    "aggregate cursor cells disagree about CUDA-backed fields"
                )
            return
        cuda_fields = has_cuda_fields
        native_pool_type = streamed._native_slab_pool_type()
        if require_native_extension and native_pool_type is None:
            raise ACSNativeAggregateCheckpointError(
                "aggregate production route requires the reusable native two-slab pool"
            )
        if native_pool_type is not None:
            native_pool = native_pool_type(chunk_bytes, has_cuda_fields)
        else:
            fallback_pool = streamed._TwoSlabPool(chunk_bytes, cuda=has_cuda_fields)
        if has_cuda_fields:
            try:
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA is unavailable")
                dedicated_stream = torch.cuda.Stream()
            except (RuntimeError, ValueError) as exc:
                if require_native_extension:
                    raise ACSNativeAggregateCheckpointError(
                        f"aggregate dedicated CUDA stream unavailable: {exc}"
                    ) from exc
                cuda_fields = False

        if native_pool is not None:
            try:
                telemetry = dict(native_pool.telemetry())
            except (AttributeError, TypeError, ValueError) as exc:
                if require_native_extension:
                    raise ACSNativeAggregateCheckpointError(
                        f"aggregate native slab pool telemetry is unavailable: {exc}"
                    ) from exc
                telemetry = {}
        else:
            telemetry = {
                "schema_version": "eiel_acs_native_slab_pool_fallback_v1",
                "chunk_bytes": chunk_bytes,
                "slab_count": 2,
                "pinned": False,
                "preallocated": False,
            }
        if require_native_extension and cuda_fields and (
            native_pool is None
            or telemetry.get("pinned") is not True
            or telemetry.get("slab_count") != 2
            or telemetry.get("preallocated") is not True
        ):
            raise ACSNativeAggregateCheckpointError(
                "aggregate CUDA production capture requires two preallocated pinned slabs"
            )

    try:
        with spool_path.open("xb") as spool_stream:
            expected_index = 0
            while True:
                detached = cursor.next_cell()
                if detached is None:
                    break
                if (
                    expected_index >= len(normalized_cell_ids)
                    or detached.cell_id != normalized_cell_ids[expected_index]
                    or detached.cell_index != expected_index
                ):
                    raise ACSNativeAggregateCheckpointError(
                        "aggregate checkpoint cursor emitted an unordered cell"
                    )
                cell_id = normalized_cell_ids[expected_index]
                projection = detached.require_projection()
                authority = dict(authority_factory(cell_id, projection))
                required_authority = {
                    "source_projection_sha256",
                    "architecture_b_identity",
                    "observer_continuation",
                    "mechanics_checkpoint_digest",
                    "canonical_checkpoint_storage_digest",
                }
                if set(authority) < required_authority:
                    raise ACSNativeAggregateCheckpointError(
                        "aggregate checkpoint cell authority is incomplete"
                    )
                source_digest = authority["source_projection_sha256"]
                identity = authority["architecture_b_identity"]
                observer = authority["observer_continuation"]
                if not _is_sha256(source_digest):
                    raise ACSNativeAggregateCheckpointError(
                        f"source digest for {cell_id} is invalid"
                    )
                if not isinstance(identity, Mapping):
                    raise ACSNativeAggregateCheckpointError(
                        f"architecture identity for {cell_id} is invalid"
                    )
                if not isinstance(observer, Mapping):
                    raise ACSNativeAggregateCheckpointError(
                        f"observer continuation for {cell_id} is invalid"
                    )
                for digest_name in (
                    "mechanics_checkpoint_digest",
                    "canonical_checkpoint_storage_digest",
                ):
                    if not _is_sha256(authority[digest_name]):
                        raise ACSNativeAggregateCheckpointError(
                            f"{digest_name} for {cell_id} is invalid"
                        )
                if identity.get("snapshot_digest_sha256") not in {None, source_digest}:
                    raise ACSNativeAggregateCheckpointError(
                        f"architecture identity digest binding mismatch for {cell_id}"
                    )
                if identity.get("source_projection_sha256") not in {None, source_digest}:
                    raise ACSNativeAggregateCheckpointError(
                        f"architecture source digest binding mismatch for {cell_id}"
                    )

                non_tensors, tensors = streamed._validate_projection(projection)
                current_fields = sorted(tensors)
                if not current_fields:
                    raise ACSNativeAggregateCheckpointError(
                        "aggregate checkpoint has no tensor fields"
                    )
                if len(current_fields) > MAX_FIELD_COUNT:
                    raise ACSNativeAggregateCheckpointError(
                        "aggregate tensor field count exceeds the bound"
                    )
                if field_order is None:
                    field_order = current_fields
                    for field in field_order:
                        tensor = tensors[field]
                        dtype_name = streamed._dtype_name(tensor)
                        shape = [int(value) for value in tensor.shape]
                        logical = _logical_bytes(tensor)
                        field_schema[field] = {
                            "field": field,
                            "dtype": dtype_name,
                            "dtype_code": streamed._DTYPE_CODE[dtype_name],
                            "shape": shape,
                            "logical_bytes": logical,
                        }
                elif current_fields != field_order:
                    raise ACSNativeAggregateCheckpointError(
                        "aggregate cells do not expose one identical tensor field schema"
                    )
                for field in field_order:
                    schema = field_schema[field]
                    tensor = tensors[field]
                    if (
                        streamed._dtype_name(tensor) != schema["dtype"]
                        or [int(value) for value in tensor.shape] != schema["shape"]
                    ):
                        per_cell_field_schema = True

                contract = _fixed_mlp_zero_fill_contract(non_tensors)
                family = _route_policy_family(non_tensors) == "mlp"
                zero_contracts.append(contract)
                zero_fill_families.append(family)
                cells_header.append(
                    {
                        "cell_id": cell_id,
                        "source_projection_sha256": source_digest,
                        "metadata_ref": None,
                        "fields": [],
                    }
                )
                authorities_by_cell[cell_id] = authority
                rows: list[dict[str, Any]] = []
                for field_index, field in enumerate(field_order):
                    schema = field_schema[field]
                    tensor = tensors[field]
                    tensor_dtype = streamed._dtype_name(tensor)
                    tensor_shape = [int(value) for value in tensor.shape]
                    logical = _logical_bytes(tensor)
                    total_logical_bytes += logical
                    zero_fill = contract is not None and field in ZERO_FILL_CANDIDATE_FIELDS
                    if zero_fill:
                        zero_started = time.perf_counter()
                        try:
                            nonzero = int(torch.count_nonzero(tensor.detach()).item())
                        except (RuntimeError, TypeError, ValueError) as exc:
                            raise ACSNativeAggregateCheckpointError(
                                f"cannot prove zero-fill field {field} for {cell_id}"
                            ) from exc
                        if nonzero != 0:
                            raise ACSNativeAggregateCheckpointError(
                                f"fixed-MLP zero-fill field is nonzero: {field}/{cell_id}"
                            )
                        zero_validation_seconds += time.perf_counter() - zero_started
                        zero_fill_logical_bytes += logical
                        zero_digest = _zero_digest_from_cache(
                            zero_digest_cache, logical
                        )
                        rows.append(
                            {
                                "field": field,
                                "storage": "zero_fill",
                                "bytes": 0,
                                "logical_bytes": logical,
                                "dtype": tensor_dtype,
                                "shape": tensor_shape,
                                "sha256": zero_digest,
                                "chunks": [],
                                "zero_fill_contract": "fixed_mlp_exact_zero_v1",
                                "zero_fill_descriptor": _zero_fill_descriptor(
                                    field=field,
                                    dtype=tensor_dtype,
                                    shape=tensor_shape,
                                    field_index=field_index,
                                    cell_index=expected_index,
                                    logical_bytes=logical,
                                    logical_digest_sha256=zero_digest,
                                ),
                            }
                        )
                    else:
                        has_cuda = getattr(
                            getattr(tensor, "device", None), "type", None
                        ) == "cuda"
                        initialize_pool(has_cuda)
                        sink = _AggregateFieldSink(
                            spool_stream,
                            offset=spool_stream.tell(),
                            payload_digest=spool_digest,
                        )
                        copy_started = time.perf_counter()
                        producer_event = None
                        if has_cuda and dedicated_stream is not None:
                            try:
                                producer_event = torch.cuda.Event()
                                producer_event.record(
                                    torch.cuda.current_stream(device=tensor.device)
                                )
                            except (RuntimeError, ValueError) as exc:
                                raise ACSNativeAggregateCheckpointError(
                                    "aggregate CUDA producer event could not be recorded"
                                ) from exc
                        copied_chunks = _copy_tensor(
                            tensor,
                            sink,
                            chunk_bytes=chunk_bytes,
                            fallback_pool=fallback_pool,
                            native_pool=native_pool,
                            dedicated_stream=dedicated_stream,
                            producer_event=producer_event,
                        )
                        copy_wall_seconds += time.perf_counter() - copy_started
                        payload_write_seconds += sink.write_seconds
                        hash_seconds += sink.hash_seconds
                        chunk_descriptor_seconds += sink.chunk_descriptor_seconds
                        if sink.bytes != logical:
                            raise ACSNativeAggregateCheckpointError(
                                f"aggregate field byte count mismatch: {field}/{cell_id}"
                            )
                        expected_chunks = (logical + chunk_bytes - 1) // chunk_bytes if logical else 0
                        if copied_chunks != expected_chunks:
                            raise ACSNativeAggregateCheckpointError(
                                f"aggregate field chunk count mismatch: {field}/{cell_id}"
                            )
                        rows.append(
                            {
                                "field": field,
                                "storage": "spool",
                                "spool_offset": sink.offset,
                                "bytes": sink.bytes,
                                "logical_bytes": logical,
                                "dtype": tensor_dtype,
                                "shape": tensor_shape,
                                "sha256": sink.digest.hexdigest(),
                                "spool_chunks": list(sink.chunks),
                            }
                        )
                        spool_bytes += sink.bytes
                metadata_raw = _metadata_record_bytes(
                    projection=non_tensors,
                    identity=identity,
                    observer=observer,
                )
                metadata_spool_offset = spool_stream.tell()
                spool_stream.write(metadata_raw)
                metadata_spool_refs.append(
                    {
                        "spool_offset": metadata_spool_offset,
                        "bytes": len(metadata_raw),
                        "sha256": _sha256(metadata_raw),
                    }
                )
                cell_field_rows.append(rows)
                cursor.release_cell(cell_id)
                expected_index += 1
                # The cursor has released the tensor-bearing mapping.  These
                # aliases must also disappear before the next cell is opened.
                del tensors, non_tensors, projection, authority, detached
            if expected_index != len(normalized_cell_ids):
                raise ACSNativeAggregateCheckpointError(
                    "aggregate checkpoint cursor ended before every cell"
                )
            cursor_receipt = cursor.close()
            if getattr(cursor_receipt, "state", None) != "SEALED":
                raise ACSNativeAggregateCheckpointError(
                    "aggregate checkpoint cursor did not seal"
                )
            bounded_projection_aggregate = cursor.aggregate_payload()
            spool_stream.flush()
    except BaseException as exc:
        try:
            cursor.fail(str(exc))
        except Exception:
            pass
        spool_path.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
        raise

    if field_order is None or not cell_field_rows:
        spool_path.unlink(missing_ok=True)
        raise ACSNativeAggregateCheckpointError(
            "aggregate checkpoint cursor produced no cells"
        )
    if any(zero_fill_families) and not all(zero_fill_families):
        spool_path.unlink(missing_ok=True)
        raise ACSNativeAggregateCheckpointError(
            "aggregate cells disagree about the fixed-MLP zero-fill contract"
        )
    if any(zero_fill_families) and any(contract is None for contract in zero_contracts):
        spool_path.unlink(missing_ok=True)
        raise ACSNativeAggregateCheckpointError(
            "fixed-MLP zero-fill contract is missing or incompatible"
        )
    if any(contract != zero_contracts[0] for contract in zero_contracts[1:]):
        spool_path.unlink(missing_ok=True)
        raise ACSNativeAggregateCheckpointError(
            "aggregate cells disagree about the fixed-MLP zero-fill contract"
        )
    zero_contract = zero_contracts[0] if zero_contracts else None
    if zero_contract is not None:
        missing = sorted(ZERO_FILL_CANDIDATE_FIELDS - set(field_order))
        if missing:
            spool_path.unlink(missing_ok=True)
            raise ACSNativeAggregateCheckpointError(
                "fixed-MLP zero-fill contract is missing candidate fields: "
                + ", ".join(missing)
            )

    payload_bytes = 0
    chunk_count = 0
    aggregate_payload_write_seconds = 0.0
    aggregate_hash_seconds = 0.0
    aggregate_chunk_descriptor_seconds = 0.0
    aggregate_fsync_seconds = 0.0
    aggregate_seal_seconds = 0.0
    aggregate_publication_seconds = 0.0
    final_payload_digest = hashlib.sha256()
    try:
        with spool_path.open("rb") as spool_stream, temporary.open("xb") as stream:
            stream.write(b"\x00" * PREFIX.size)
            payload_offset = 0
            for field_index, field in enumerate(field_order):
                for cell_index, rows in enumerate(cell_field_rows):
                    row = rows[field_index]
                    if row["storage"] == "zero_fill":
                        final_row = dict(row)
                        final_row["field"] = field
                    else:
                        sink = _AggregateFieldSink(
                            stream,
                            offset=payload_offset,
                            payload_digest=final_payload_digest,
                        )
                        for descriptor in row["spool_chunks"]:
                            offset = int(descriptor["offset"])
                            length = int(descriptor["bytes"])
                            spool_stream.seek(offset)
                            raw = spool_stream.read(length)
                            if len(raw) != length or _sha256(raw) != descriptor["sha256"]:
                                raise ACSNativeAggregateCheckpointError(
                                    "aggregate cursor spool chunk is truncated or corrupted"
                                )
                            sink.write(raw)
                        if sink.bytes != row["bytes"] or sink.digest.hexdigest() != row["sha256"]:
                            raise ACSNativeAggregateCheckpointError(
                                f"aggregate cursor field spool digest differs: {field}/{cell_index}"
                            )
                        aggregate_payload_write_seconds += sink.write_seconds
                        aggregate_hash_seconds += sink.hash_seconds
                        aggregate_chunk_descriptor_seconds += sink.chunk_descriptor_seconds
                        final_row = {
                            "field": field,
                            "storage": "payload",
                            "offset": payload_offset,
                            "bytes": sink.bytes,
                            "logical_bytes": int(row["logical_bytes"]),
                            "dtype": row["dtype"],
                            "shape": list(row["shape"]),
                            "sha256": sink.digest.hexdigest(),
                            "chunks": sink.chunks,
                            "zero_fill_contract": None,
                            "zero_fill_descriptor": None,
                        }
                        payload_offset += sink.bytes
                        payload_bytes += sink.bytes
                        chunk_count += len(sink.chunks)
                    cells_header[cell_index]["fields"].append(final_row)

            if len(metadata_spool_refs) != len(normalized_cell_ids):
                raise ACSNativeAggregateCheckpointError(
                    "aggregate cursor metadata index does not cover every cell"
                )
            metadata_offset = payload_offset
            metadata_bytes = 0
            metadata_write_started = time.perf_counter()
            for cell_index, spool_ref in enumerate(metadata_spool_refs):
                spool_offset = int(spool_ref["spool_offset"])
                length = int(spool_ref["bytes"])
                spool_stream.seek(spool_offset)
                raw = spool_stream.read(length)
                if len(raw) != length or _sha256(raw) != spool_ref["sha256"]:
                    raise ACSNativeAggregateCheckpointError(
                        "aggregate cursor metadata spool is truncated or corrupted"
                    )
                output_offset = metadata_offset + metadata_bytes
                stream.write(raw)
                final_payload_digest.update(raw)
                cells_header[cell_index]["metadata_ref"] = _metadata_ref(
                    raw, offset=output_offset
                )
                metadata_bytes += length
            aggregate_payload_write_seconds += (
                time.perf_counter() - metadata_write_started
            )
            file_payload_bytes = payload_offset + metadata_bytes
            header = {
                "schema_version": SCHEMA_VERSION,
                "format_version": FORMAT_VERSION,
                "endianness": "little",
                "payload_layout": METADATA_PAYLOAD_LAYOUT,
                "metadata_schema_version": METADATA_SCHEMA_VERSION,
                "batch_id": batch_id,
                "generation": generation,
                "tick": tick,
                "absolute_next_tick": absolute_next_tick,
                "segment_ordinal": segment_ordinal,
                "cell_ids": list(normalized_cell_ids),
                "field_order": list(field_order),
                "field_schema": [field_schema[field] for field in field_order],
                "field_schema_mode": (
                    "per_cell_v1" if per_cell_field_schema else "shared_v1"
                ),
                "cells": cells_header,
                "chunk_bytes": chunk_bytes,
                "chunk_count": chunk_count,
                "payload_bytes": payload_bytes,
                "file_payload_bytes": file_payload_bytes,
                "metadata_offset": metadata_offset,
                "metadata_bytes": metadata_bytes,
                "logical_bytes": total_logical_bytes,
                "zero_fill_logical_bytes": zero_fill_logical_bytes,
                "zero_fill_contract": zero_contract,
                "zero_fill_fields": sorted(
                    field
                    for field in field_order
                    if zero_contract is not None
                    and field in ZERO_FILL_CANDIDATE_FIELDS
                ),
            }
            header_bytes = _canonical_json(header)
            if len(header_bytes) > MAX_HEADER_BYTES:
                raise ACSNativeAggregateCheckpointError(
                    "aggregate checkpoint header exceeds the bound"
                )
            payload_sha256 = final_payload_digest.hexdigest()
            header_sha256 = _sha256(header_bytes)
            content_sha256 = _content_digest(
                header_sha256=header_sha256,
                payload_sha256=payload_sha256,
            )
            provisional_prefix = PREFIX.pack(
                MAGIC,
                FORMAT_VERSION,
                LITTLE_ENDIAN_MARKER,
                len(header_bytes),
                file_payload_bytes,
                0,
            )
            provisional_footer = _footer_without_artifact(
                prefix=provisional_prefix,
                header_bytes=header_bytes,
                payload_sha256=payload_sha256,
                content_sha256=content_sha256,
                batch_id=batch_id,
                cell_ids=normalized_cell_ids,
                payload_bytes=file_payload_bytes,
            )
            footer_length = len(
                _canonical_json({**provisional_footer, "artifact_sha256": "0" * 64})
            )
            prefix = PREFIX.pack(
                MAGIC,
                FORMAT_VERSION,
                LITTLE_ENDIAN_MARKER,
                len(header_bytes),
                file_payload_bytes,
                footer_length,
            )
            footer_without_artifact = _footer_without_artifact(
                prefix=prefix,
                header_bytes=header_bytes,
                payload_sha256=payload_sha256,
                content_sha256=content_sha256,
                batch_id=batch_id,
                cell_ids=normalized_cell_ids,
                payload_bytes=file_payload_bytes,
            )
            footer = {
                **footer_without_artifact,
                "artifact_sha256": _artifact_digest(
                    prefix=prefix,
                    header_sha256=header_sha256,
                    payload_sha256=payload_sha256,
                    content_sha256=content_sha256,
                    footer_without_artifact=footer_without_artifact,
                ),
            }
            footer_bytes = _canonical_json(footer)
            if len(footer_bytes) != footer_length or len(footer_bytes) > MAX_FOOTER_BYTES:
                raise ACSNativeAggregateCheckpointError(
                    "aggregate checkpoint footer length did not converge"
                )
            stream.write(header_bytes)
            stream.write(footer_bytes)
            stream.flush()
            stream.seek(0)
            stream.write(prefix)
            stream.flush()
            fsync_started = time.perf_counter()
            os.fsync(stream.fileno())
            aggregate_fsync_seconds += time.perf_counter() - fsync_started
    except BaseException:
        spool_path.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
        raise

    try:
        try:
            existing_target = load_acs_native_aggregate_checkpoint(
                target, cell_id=normalized_cell_ids[0]
            )
        except FileNotFoundError:
            existing_target = None
        except ACSNativeAggregateCheckpointError:
            if target.exists():
                raise ACSNativeAggregateCheckpointError(
                    "conflicting aggregate checkpoint publication already exists"
                )
            existing_target = None
        if existing_target is not None:
            if existing_target["footer"].get("artifact_sha256") != footer["artifact_sha256"]:
                raise ACSNativeAggregateCheckpointError(
                    "conflicting aggregate checkpoint publication already exists"
                )
        else:
            os.link(temporary, target)
            directory_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        spool_path.unlink(missing_ok=True)
        raise
    else:
        temporary.unlink(missing_ok=True)
        spool_path.unlink(missing_ok=True)
    aggregate_publication_seconds = time.perf_counter() - started

    if native_pool is not None:
        try:
            native_pool_telemetry = dict(native_pool.telemetry())
        except (AttributeError, TypeError, ValueError) as exc:
            raise ACSNativeAggregateCheckpointError(
                f"aggregate native slab pool telemetry is unavailable after capture: {exc}"
            ) from exc
    else:
        native_pool_telemetry = {
            "schema_version": "eiel_acs_native_slab_pool_fallback_v1",
            "chunk_bytes": chunk_bytes,
            "slab_count": 2,
            "pinned": False,
            "preallocated": False,
        }

    cell_receipts: dict[str, dict[str, Any]] = {}
    for cell in cells_header:
        physical = sum(
            int(row["bytes"])
            for row in cell["fields"]
            if row["storage"] == "payload"
        )
        logical = sum(int(row["logical_bytes"]) for row in cell["fields"])
        zero_logical = sum(
            int(row["logical_bytes"])
            for row in cell["fields"]
            if row["storage"] == "zero_fill"
        )
        cell_receipts[str(cell["cell_id"])] = {
            "cell_id": str(cell["cell_id"]),
            "source_projection_sha256": str(cell["source_projection_sha256"]),
            "logical_bytes": logical,
            "physical_bytes": physical,
            "zero_fill_logical_bytes": zero_logical,
            "chunk_count": sum(len(row["chunks"]) for row in cell["fields"]),
            "field_count": len(cell["fields"]),
        }
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "path": str(target),
        "batch_id": batch_id,
        "generation": generation,
        "tick": tick,
        "absolute_next_tick": absolute_next_tick,
        "segment_ordinal": segment_ordinal,
        "header_sha256": footer["header_sha256"],
        "payload_sha256": footer["payload_sha256"],
        "content_sha256": footer["content_sha256"],
        "seal_sha256": footer["seal_sha256"],
        "artifact_sha256": footer["artifact_sha256"],
        "header_bytes": len(header_bytes),
        "payload_bytes": payload_bytes,
        "metadata_bytes": metadata_bytes,
        "file_payload_bytes": file_payload_bytes,
        "logical_bytes": total_logical_bytes,
        "zero_fill_logical_bytes": zero_fill_logical_bytes,
        "chunk_count": chunk_count,
        "cell_count": len(normalized_cell_ids),
        "cells": cell_receipts,
        "zero_fill_fields": header["zero_fill_fields"],
        "native_slab_pool": native_pool_telemetry,
        "physical_file_bytes": target.stat().st_size,
        "cursor_receipt": (
            cursor_receipt.payload()
            if cursor_receipt is not None and hasattr(cursor_receipt, "payload")
            else None
        ),
    }
    if performance_telemetry is not None:
        performance_telemetry.clear()
        performance_telemetry.update(
            {
                "aggregate_native_pack_payload_bytes": float(payload_bytes),
                "aggregate_native_pack_file_payload_bytes": float(file_payload_bytes),
                "aggregate_native_pack_metadata_payload_bytes": float(metadata_bytes),
                "aggregate_native_pack_logical_bytes": float(total_logical_bytes),
                "aggregate_native_pack_zero_fill_logical_bytes": float(zero_fill_logical_bytes),
                "aggregate_native_pack_metadata_bytes": float(len(header_bytes)),
                "aggregate_native_pack_chunk_count": float(chunk_count),
                "aggregate_native_pack_field_count": float(len(field_order) * len(normalized_cell_ids)),
                "aggregate_native_pack_two_slab_pool": 1.0,
                "aggregate_native_pack_pinned_slabs": float(
                    native_pool_telemetry.get("pinned") is True
                ),
                "aggregate_native_pack_prebuilt_extension": float(native_pool is not None),
                "aggregate_native_pack_dedicated_d2h_stream": float(dedicated_stream is not None),
                "aggregate_native_pack_copy_wall_seconds": copy_wall_seconds,
                "aggregate_native_pack_payload_write_seconds": (
                    payload_write_seconds + aggregate_payload_write_seconds
                ),
                "aggregate_native_pack_hash_seconds": hash_seconds + aggregate_hash_seconds,
                "aggregate_native_pack_chunk_descriptor_seconds": (
                    chunk_descriptor_seconds + aggregate_chunk_descriptor_seconds
                ),
                "aggregate_native_pack_zero_validation_seconds": zero_validation_seconds,
                "aggregate_native_pack_fsync_seconds": aggregate_fsync_seconds,
                "aggregate_native_pack_publication_seconds": aggregate_publication_seconds,
                "aggregate_native_pack_physical_file_bytes": float(target.stat().st_size),
                "aggregate_native_pack_seconds": time.perf_counter() - started,
                "aggregate_native_pack_cursor_bounded": 1.0,
                "aggregate_native_pack_cursor_max_live_cells": float(
                    (receipt.get("cursor_receipt") or {}).get(
                        "max_live_tensor_bearing_projections", 0
                    )
                ),
                "aggregate_native_pack_process_cpu_seconds": (
                    time.process_time_ns() - started_cpu
                )
                / 1_000_000_000.0,
            }
        )
    return {
        "receipt": receipt,
        "authorities_by_cell": authorities_by_cell,
        "bounded_projection_aggregate": bounded_projection_aggregate,
        "checkpoint_projection_cursor_receipt": receipt.get("cursor_receipt"),
    }


def _read_exact(stream: Any, length: int, *, label: str) -> bytes:
    value = stream.read(length)
    if len(value) != length:
        raise ACSNativeAggregateCheckpointError(f"truncated aggregate checkpoint {label}")
    return value


def _strict_json(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = streamed._strict_json(raw, label=label)
    except streamed.ACSNativeCheckpointError as exc:
        raise ACSNativeAggregateCheckpointError(str(exc)) from exc
    return value


def _payload_hash(path: Path, *, payload_bytes: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(PREFIX.size)
        remaining = payload_bytes
        while remaining:
            chunk = stream.read(min(8 * 1024 * 1024, remaining))
            if not chunk:
                raise ACSNativeAggregateCheckpointError(
                    "truncated aggregate checkpoint payload"
                )
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _read_payload_range(path: Path, *, offset: int, length: int) -> bytes:
    if type(offset) is not int or offset < 0 or type(length) is not int or length < 0:
        raise ACSNativeAggregateCheckpointError("aggregate payload range is invalid")
    with path.open("rb") as stream:
        stream.seek(PREFIX.size + offset)
        raw = stream.read(length)
    if len(raw) != length:
        raise ACSNativeAggregateCheckpointError("truncated aggregate payload field")
    return raw


def load_acs_native_aggregate_checkpoint(
    source: str | os.PathLike[str],
    *,
    cell_id: str | None = None,
    expected_source_projection_sha256: str | None = None,
    verify_payload: bool = True,
) -> dict[str, Any]:
    """Strictly load one selected cell from an aggregate checkpoint."""

    path = Path(source).expanduser().resolve()
    try:
        with path.open("rb") as stream:
            prefix_bytes = _read_exact(stream, PREFIX.size, label="prefix")
            magic, version, endian, header_len, payload_len, footer_len = PREFIX.unpack(prefix_bytes)
            if magic != MAGIC or version != FORMAT_VERSION or endian != LITTLE_ENDIAN_MARKER:
                raise ACSNativeAggregateCheckpointError("aggregate checkpoint prefix identity mismatch")
            if not 0 < header_len <= MAX_HEADER_BYTES:
                raise ACSNativeAggregateCheckpointError("aggregate checkpoint header length is outside the bound")
            if payload_len < 0 or payload_len > MAX_FIELD_BYTES * MAX_FIELD_COUNT:
                raise ACSNativeAggregateCheckpointError("aggregate checkpoint payload length is outside the bound")
            if not 0 < footer_len <= MAX_FOOTER_BYTES:
                raise ACSNativeAggregateCheckpointError("aggregate checkpoint footer length is outside the bound")
            stream.seek(payload_len, os.SEEK_CUR)
            header_bytes = _read_exact(stream, header_len, label="header")
            footer_bytes = _read_exact(stream, footer_len, label="footer")
            if stream.read(1):
                raise ACSNativeAggregateCheckpointError("aggregate checkpoint has trailing bytes")
    except OSError as exc:
        raise ACSNativeAggregateCheckpointError(f"aggregate checkpoint is unreadable: {path}: {exc}") from exc

    header = _strict_json(header_bytes, label="header")
    footer = _strict_json(footer_bytes, label="footer")
    if header.get("schema_version") != SCHEMA_VERSION or header.get("format_version") != FORMAT_VERSION:
        raise ACSNativeAggregateCheckpointError("aggregate checkpoint header identity mismatch")
    payload_layout = header.get("payload_layout")
    if payload_layout not in {LEGACY_PAYLOAD_LAYOUT, METADATA_PAYLOAD_LAYOUT}:
        raise ACSNativeAggregateCheckpointError("aggregate checkpoint payload layout is invalid")
    if footer.get("schema_version") != FOOTER_SCHEMA_VERSION or footer.get("status") != "complete":
        raise ACSNativeAggregateCheckpointError("aggregate checkpoint footer identity mismatch")
    if footer.get("header_bytes") != header_len or footer.get("payload_bytes") != payload_len:
        raise ACSNativeAggregateCheckpointError("aggregate checkpoint envelope lengths disagree")
    header_sha256 = _sha256(header_bytes)
    if type(verify_payload) is not bool:
        raise ACSNativeAggregateCheckpointError("verify_payload must be an exact bool")
    if verify_payload:
        payload_sha256 = _payload_hash(path, payload_bytes=payload_len)
    else:
        payload_sha256 = footer.get("payload_sha256")
        if not _is_sha256(payload_sha256):
            raise ACSNativeAggregateCheckpointError(
                "aggregate checkpoint payload digest is invalid"
            )
    if footer.get("header_sha256") != header_sha256 or footer.get(
        "payload_sha256"
    ) != payload_sha256:
        raise ACSNativeAggregateCheckpointError("aggregate checkpoint content digest mismatch")
    content_sha256 = _content_digest(
        header_sha256=header_sha256,
        payload_sha256=payload_sha256,
    )
    if footer.get("content_sha256") != content_sha256:
        raise ACSNativeAggregateCheckpointError("aggregate checkpoint content seal mismatch")
    cell_ids = header.get("cell_ids")
    if not isinstance(cell_ids, list) or not cell_ids or not all(isinstance(value, str) and value for value in cell_ids):
        raise ACSNativeAggregateCheckpointError("aggregate checkpoint cell index is invalid")
    if len(set(cell_ids)) != len(cell_ids):
        raise ACSNativeAggregateCheckpointError("aggregate checkpoint cell index is duplicated")
    if footer.get("batch_id") != header.get("batch_id") or footer.get("cell_ids") != cell_ids:
        raise ACSNativeAggregateCheckpointError("aggregate checkpoint footer identity binding mismatch")
    expected_seal = _seal_digest(
        content_sha256=content_sha256,
        header_sha256=header_sha256,
        payload_sha256=payload_sha256,
        batch_id=str(header.get("batch_id")),
        cell_ids=cell_ids,
    )
    if footer.get("seal_sha256") != expected_seal:
        raise ACSNativeAggregateCheckpointError(
            "aggregate checkpoint seal digest mismatch"
        )
    footer_without_artifact = dict(footer)
    artifact_value = footer_without_artifact.pop("artifact_sha256", None)
    if not _is_sha256(artifact_value):
        raise ACSNativeAggregateCheckpointError("aggregate checkpoint artifact digest is invalid")
    expected_artifact = _artifact_digest(
        prefix=prefix_bytes,
        header_sha256=header_sha256,
        payload_sha256=payload_sha256,
        content_sha256=content_sha256,
        footer_without_artifact=footer_without_artifact,
    )
    if artifact_value != expected_artifact:
        raise ACSNativeAggregateCheckpointError(
            "aggregate checkpoint artifact digest mismatch"
        )

    field_order = header.get("field_order")
    field_schema_rows = header.get("field_schema")
    cells = header.get("cells")
    value_pool = header.get("value_pool")
    field_schema_mode = header.get("field_schema_mode", "shared_v1")
    if payload_layout == LEGACY_PAYLOAD_LAYOUT:
        tensor_payload_bytes = payload_len
        metadata_offset = payload_len
        metadata_bytes = 0
        if not isinstance(value_pool, list):
            raise ACSNativeAggregateCheckpointError(
                "legacy aggregate checkpoint value pool is malformed"
            )
    else:
        if value_pool is not None:
            raise ACSNativeAggregateCheckpointError(
                "metadata-payload aggregate unexpectedly carries a value pool"
            )
        tensor_payload_bytes = header.get("payload_bytes")
        metadata_offset = header.get("metadata_offset")
        metadata_bytes = header.get("metadata_bytes")
        if (
            type(tensor_payload_bytes) is not int
            or tensor_payload_bytes < 0
            or type(metadata_offset) is not int
            or metadata_offset != tensor_payload_bytes
            or type(metadata_bytes) is not int
            or metadata_bytes <= 0
            or metadata_bytes > MAX_METADATA_BYTES
            or tensor_payload_bytes + metadata_bytes != payload_len
            or header.get("file_payload_bytes") != payload_len
            or header.get("metadata_schema_version") != METADATA_SCHEMA_VERSION
        ):
            raise ACSNativeAggregateCheckpointError(
                "aggregate metadata payload envelope is malformed"
            )
    if (
        not isinstance(field_order, list)
        or not field_order
        or not all(isinstance(value, str) and value for value in field_order)
        or not isinstance(field_schema_rows, list)
        or len(field_schema_rows) != len(field_order)
        or not isinstance(cells, list)
        or len(cells) != len(cell_ids)
        or not isinstance(field_schema_mode, str)
        or field_schema_mode not in {"shared_v1", "per_cell_v1"}
    ):
        raise ACSNativeAggregateCheckpointError("aggregate checkpoint index is malformed")
    header_zero_fill_fields = header.get("zero_fill_fields")
    if (
        not isinstance(header_zero_fill_fields, list)
        or any(
            not isinstance(field, str) or field not in ZERO_FILL_CANDIDATE_FIELDS
            for field in header_zero_fill_fields
        )
        or header_zero_fill_fields != sorted(set(header_zero_fill_fields))
    ):
        raise ACSNativeAggregateCheckpointError(
            "aggregate checkpoint zero-fill field index is malformed"
        )
    header_zero_fill_contract = header.get("zero_fill_contract")
    if header_zero_fill_contract is not None and (
        not isinstance(header_zero_fill_contract, Mapping)
        or dict(header_zero_fill_contract)
        != _canonical_fixed_mlp_zero_fill_contract()
    ):
        raise ACSNativeAggregateCheckpointError(
            "aggregate checkpoint zero-fill route contract is invalid"
        )
    schema_by_field: dict[str, Mapping[str, Any]] = {}
    for field, row in zip(field_order, field_schema_rows, strict=True):
        if not isinstance(row, Mapping) or row.get("field") != field:
            raise ACSNativeAggregateCheckpointError("aggregate field schema order is invalid")
        dtype_name = row.get("dtype")
        shape = row.get("shape")
        logical = row.get("logical_bytes")
        if not isinstance(dtype_name, str) or dtype_name not in streamed._DTYPE_BY_NAME:
            raise ACSNativeAggregateCheckpointError(f"aggregate field dtype is invalid: {field}")
        if not isinstance(shape, list) or not all(type(value) is int and value >= 0 for value in shape):
            raise ACSNativeAggregateCheckpointError(f"aggregate field shape is invalid: {field}")
        if type(logical) is not int or logical < 0 or logical > MAX_FIELD_BYTES:
            raise ACSNativeAggregateCheckpointError(f"aggregate field logical size is invalid: {field}")
        element_size = torch.tensor([], dtype=streamed._DTYPE_BY_NAME[dtype_name]).element_size()
        expected_logical = element_size
        for dimension in shape:
            expected_logical *= dimension
        if logical != expected_logical:
            raise ACSNativeAggregateCheckpointError(f"aggregate field logical size disagrees: {field}")
        schema_by_field[field] = row

    if len(set(field_order)) != len(field_order) or not all(isinstance(row, Mapping) for row in cells):
        raise ACSNativeAggregateCheckpointError("aggregate checkpoint field or cell order is duplicated")
    cell_by_id = {str(row["cell_id"]): row for row in cells if isinstance(row.get("cell_id"), str)}
    if tuple(cell_by_id) != tuple(cell_ids):
        raise ACSNativeAggregateCheckpointError("aggregate checkpoint cell order is invalid")
    if payload_layout == METADATA_PAYLOAD_LAYOUT:
        expected_metadata_offset = metadata_offset
        for row in cells:
            metadata_ref = row.get("metadata_ref")
            if not isinstance(metadata_ref, Mapping):
                raise ACSNativeAggregateCheckpointError(
                    "aggregate metadata payload cell reference is malformed"
                )
            ref_offset = metadata_ref.get("offset")
            ref_bytes = metadata_ref.get("bytes")
            ref_sha256 = metadata_ref.get("sha256")
            if (
                type(ref_offset) is not int
                or ref_offset != expected_metadata_offset
                or type(ref_bytes) is not int
                or ref_bytes <= 0
                or ref_offset + ref_bytes > metadata_offset + metadata_bytes
                or not _is_sha256(ref_sha256)
            ):
                raise ACSNativeAggregateCheckpointError(
                    "aggregate metadata payload cell range is invalid"
                )
            expected_metadata_offset += ref_bytes
        if expected_metadata_offset != metadata_offset + metadata_bytes:
            raise ACSNativeAggregateCheckpointError(
                "aggregate metadata payload index does not cover the payload"
            )

    def resolve_field_schema(
        field: str, field_row: Mapping[str, Any]
    ) -> tuple[str, list[int], int]:
        base = schema_by_field[field]
        dtype_name = field_row.get("dtype", base["dtype"])
        shape = field_row.get("shape", base["shape"])
        logical = field_row.get("logical_bytes", base["logical_bytes"])
        if (
            field_schema_mode == "per_cell_v1"
            and not all(name in field_row for name in ("dtype", "shape", "logical_bytes"))
        ):
            raise ACSNativeAggregateCheckpointError(
                "aggregate per-cell field schema is incomplete"
            )
        if field_schema_mode == "shared_v1" and (
            dtype_name != base["dtype"]
            or shape != base["shape"]
            or logical != base["logical_bytes"]
        ):
            raise ACSNativeAggregateCheckpointError(
                f"aggregate field schema differs from shared schema: {field}"
            )
        if (
            not isinstance(dtype_name, str)
            or dtype_name not in streamed._DTYPE_BY_NAME
            or not isinstance(shape, list)
            or not all(type(value) is int and value >= 0 for value in shape)
            or type(logical) is not int
            or logical < 0
            or logical > MAX_FIELD_BYTES
        ):
            raise ACSNativeAggregateCheckpointError(
                f"aggregate cell field schema is invalid: {field}"
            )
        element_size = torch.tensor(
            [], dtype=streamed._DTYPE_BY_NAME[dtype_name]
        ).element_size()
        expected_logical = element_size
        for dimension in shape:
            expected_logical *= dimension
        if logical != expected_logical:
            raise ACSNativeAggregateCheckpointError(
                f"aggregate cell field logical size disagrees: {field}"
            )
        return dtype_name, list(shape), logical

    expected_offset = 0
    zero_digest_cache: dict[int, str] = {}
    for field_index, field in enumerate(field_order):
        for cell_index, row in enumerate(cells):
            fields = row.get("fields")
            if not isinstance(fields, list) or len(fields) != len(field_order):
                raise ACSNativeAggregateCheckpointError("aggregate checkpoint cell field index is malformed")
            field_row = fields[field_index]
            if not isinstance(field_row, Mapping) or field_row.get("field") != field:
                raise ACSNativeAggregateCheckpointError("aggregate checkpoint field-major order is invalid")
            storage = field_row.get("storage")
            dtype_name, shape, logical = resolve_field_schema(field, field_row)
            if storage == "payload":
                offset = field_row.get("offset")
                size = field_row.get("bytes")
                if type(offset) is not int or offset != expected_offset or type(size) is not int or size != logical or size < 0:
                    raise ACSNativeAggregateCheckpointError("aggregate checkpoint payload field range is invalid")
                expected_offset += size
            elif storage == "zero_fill":
                if (
                    field_row.get("offset") is not None
                    or field_row.get("bytes") != 0
                    or field_row.get("zero_fill_contract")
                    != "fixed_mlp_exact_zero_v1"
                ):
                    raise ACSNativeAggregateCheckpointError("aggregate checkpoint zero-fill descriptor is invalid")
                if not _is_sha256(field_row.get("sha256")):
                    raise ACSNativeAggregateCheckpointError("aggregate checkpoint zero-fill digest is invalid")
                expected_zero = _zero_digest_from_cache(
                    zero_digest_cache, logical
                )
                expected_descriptor = _zero_fill_descriptor(
                    field=field,
                    dtype=dtype_name,
                    shape=shape,
                    field_index=field_index,
                    cell_index=cell_index,
                    logical_bytes=logical,
                    logical_digest_sha256=expected_zero,
                )
                if field_row.get("sha256") != expected_zero or field_row.get(
                    "zero_fill_descriptor"
                ) != expected_descriptor:
                    raise ACSNativeAggregateCheckpointError(
                        "aggregate checkpoint zero-fill descriptor does not match its field"
                    )
            else:
                raise ACSNativeAggregateCheckpointError("aggregate checkpoint field storage kind is invalid")
            if storage == "payload" and field_row.get("zero_fill_descriptor") is not None:
                raise ACSNativeAggregateCheckpointError(
                    "aggregate payload field has an unexpected zero-fill descriptor"
                )
    if expected_offset != tensor_payload_bytes:
        raise ACSNativeAggregateCheckpointError("aggregate checkpoint payload index does not cover the payload")
    if header.get("chunk_count") != sum(
        len(field_row.get("chunks", []))
        for row in cells
        for field_row in row["fields"]
        if field_row.get("storage") == "payload"
    ):
        raise ACSNativeAggregateCheckpointError("aggregate checkpoint chunk count is inconsistent")

    selected_id = cell_id
    if selected_id is None:
        if len(cell_ids) != 1:
            raise ACSNativeAggregateCheckpointError("aggregate checkpoint cell_id is required for a multicell load")
        selected_id = cell_ids[0]
    if selected_id not in cell_by_id:
        raise ACSNativeAggregateCheckpointError(f"aggregate checkpoint cell is absent: {selected_id}")
    selected = cell_by_id[selected_id]
    source_digest = selected.get("source_projection_sha256")
    if not _is_sha256(source_digest):
        raise ACSNativeAggregateCheckpointError("aggregate checkpoint source digest is invalid")
    if expected_source_projection_sha256 is not None and source_digest != expected_source_projection_sha256:
        raise ACSNativeAggregateCheckpointError("aggregate checkpoint source digest does not match expectation")

    if payload_layout == LEGACY_PAYLOAD_LAYOUT:
        def unpack_ref(field: str) -> Any:
            ref = selected.get(field)
            if type(ref) is not int or ref < 0 or ref >= len(value_pool):
                raise ACSNativeAggregateCheckpointError(
                    f"aggregate checkpoint {field} is invalid"
                )
            try:
                return streamed._unpack_value(value_pool[ref])
            except (KeyError, TypeError, ValueError, OverflowError, streamed.ACSNativeCheckpointError) as exc:
                raise ACSNativeAggregateCheckpointError(
                    f"aggregate checkpoint {field} value is malformed"
                ) from exc

        projection_value = unpack_ref("projection_ref")
        identity = unpack_ref("architecture_b_identity_ref")
        observer = unpack_ref("observer_continuation_ref")
    else:
        metadata_ref = selected["metadata_ref"]
        raw_metadata = _read_payload_range(
            path,
            offset=int(metadata_ref["offset"]),
            length=int(metadata_ref["bytes"]),
        )
        if _sha256(raw_metadata) != metadata_ref["sha256"]:
            raise ACSNativeAggregateCheckpointError(
                "aggregate metadata payload digest mismatch"
            )
        try:
            metadata_record = _strict_json(raw_metadata, label="metadata payload")
            if metadata_record.get("schema_version") != METADATA_SCHEMA_VERSION:
                raise ACSNativeAggregateCheckpointError(
                    "aggregate metadata payload identity mismatch"
                )
            projection_value = streamed._unpack_value(metadata_record["projection"])
            identity = streamed._unpack_value(
                metadata_record["architecture_b_identity"]
            )
            observer = streamed._unpack_value(
                metadata_record["observer_continuation"]
            )
        except (KeyError, TypeError, ValueError, OverflowError, streamed.ACSNativeCheckpointError) as exc:
            raise ACSNativeAggregateCheckpointError(
                "aggregate metadata payload is malformed"
            ) from exc
    if not isinstance(projection_value, Mapping) or not isinstance(identity, Mapping) or not isinstance(observer, Mapping):
        raise ACSNativeAggregateCheckpointError("aggregate checkpoint metadata references are invalid")
    zero_fill_fields = {
        str(row["field"])
        for row in selected["fields"]
        if row.get("storage") == "zero_fill"
    }
    if zero_fill_fields:
        expected_contract = _fixed_mlp_zero_fill_contract(projection_value)
        expected_zero_fields = sorted(
            zero_fill_fields | (set(field_order) & ZERO_FILL_CANDIDATE_FIELDS)
        )
        if expected_contract is None or header_zero_fill_contract != expected_contract:
            raise ACSNativeAggregateCheckpointError(
                "aggregate zero-fill restore lacks the fixed-MLP route contract"
            )
        if header_zero_fill_fields != expected_zero_fields or zero_fill_fields != set(
            header_zero_fill_fields
        ):
            raise ACSNativeAggregateCheckpointError(
                "aggregate zero-fill field index does not match the route contract"
            )
        if not zero_fill_fields.issubset(ZERO_FILL_CANDIDATE_FIELDS):
            raise ACSNativeAggregateCheckpointError(
                "aggregate zero-fill restore contains an inadmissible field"
            )
    elif header_zero_fill_fields or header_zero_fill_contract is not None:
        raise ACSNativeAggregateCheckpointError(
            "aggregate checkpoint carries an unused zero-fill contract"
        )
    for identity_field in ("snapshot_digest_sha256", "source_projection_sha256"):
        identity_digest = identity.get(identity_field)
        if identity_digest is not None and identity_digest != source_digest:
            raise ACSNativeAggregateCheckpointError("aggregate checkpoint architecture identity binding mismatch")

    fields_by_name = {str(row["field"]): row for row in selected["fields"]}
    tensors: dict[str, torch.Tensor] = {}
    for field in field_order:
        row = fields_by_name[field]
        dtype_name, shape_list, logical = resolve_field_schema(field, row)
        shape = tuple(int(value) for value in shape_list)
        if row["storage"] == "zero_fill":
            expected_zero = _zero_digest_from_cache(
                zero_digest_cache, logical
            )
            if row["sha256"] != expected_zero:
                raise ACSNativeAggregateCheckpointError("aggregate checkpoint zero-fill digest mismatch")
            tensors[field] = torch.zeros(shape, dtype=streamed._DTYPE_BY_NAME[dtype_name])
            continue
        raw = _read_payload_range(path, offset=int(row["offset"]), length=int(row["bytes"]))
        if _sha256(raw) != row["sha256"]:
            raise ACSNativeAggregateCheckpointError(f"aggregate checkpoint field digest mismatch: {field}")
        chunks = row.get("chunks")
        if not isinstance(chunks, list):
            raise ACSNativeAggregateCheckpointError("aggregate checkpoint chunk index is invalid")
        for chunk in chunks:
            if not isinstance(chunk, Mapping) or type(chunk.get("offset")) is not int or type(chunk.get("bytes")) is not int:
                raise ACSNativeAggregateCheckpointError("aggregate checkpoint chunk descriptor is invalid")
            local_start = int(chunk["offset"]) - int(row["offset"])
            length = int(chunk["bytes"])
            if local_start < 0 or length < 0 or local_start + length > len(raw):
                raise ACSNativeAggregateCheckpointError("aggregate checkpoint chunk range is invalid")
            if _sha256(raw[local_start : local_start + length]) != chunk.get("sha256"):
                raise ACSNativeAggregateCheckpointError("aggregate checkpoint chunk digest mismatch")
        try:
            tensors[field] = streamed._tensor_from_bytes(raw, dtype_name=dtype_name, shape=shape)
        except streamed.ACSNativeCheckpointError as exc:
            raise ACSNativeAggregateCheckpointError(str(exc)) from exc

    projection = dict(projection_value)
    projection["tensors"] = tensors
    return {
        "projection": projection,
        "header": header,
        "footer": footer,
        "architecture_b_identity": dict(identity),
        "observer_continuation": dict(observer),
        "cell_id": selected_id,
        "artifact_sha256": str(artifact_value),
    }


__all__ = [
    "ACSNativeAggregateCheckpointError",
    "DEFAULT_CHUNK_BYTES",
    "FOOTER_SCHEMA_VERSION",
    "MAGIC",
    "MAX_CHUNK_BYTES",
    "SCHEMA_VERSION",
    "ZERO_FILL_CANDIDATE_FIELDS",
    "load_acs_native_aggregate_checkpoint",
    "write_acs_native_aggregate_checkpoint",
    "write_acs_native_aggregate_checkpoint_from_cursor",
]
