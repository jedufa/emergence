"""Detached native ACS capture and bounded asynchronous durability slots.

The producer-side operation is deliberately narrow: copy the live owner through
the existing reusable two-slab pool into a unique cell-major file-backed spool
and return only scalar metadata.  The worker never receives a projection,
tensor, CUDA stream, owner, or callback.  It materializes the canonical payload,
hashes, seals, fsyncs, and publishes after the producer has released the resident
owner.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field as dataclass_field
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import threading
import time
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

import torch

from . import acs_native_aggregate_checkpoint_v1 as aggregate
from . import acs_native_streamed_checkpoint_v1 as streamed


DETACHED_SLOTS_SCHEMA_VERSION_V1 = "eiel_acs_native_detached_slots_v1"
DETACHED_CAPTURE_SCHEMA_VERSION_V1 = "eiel_acs_native_detached_capture_v1"
DETACHED_SPOOL_CAPTURE_SCHEMA_VERSION_V2 = "eiel_acs_native_detached_spool_capture_v2"
DETACHED_SPOOL_SEAL_SCHEMA_VERSION_V1 = "eiel_acs_native_detached_spool_seal_v1"
DETACHED_SLOT_TOKEN_SCHEMA_VERSION_V1 = "eiel_acs_native_detached_slot_token_v1"
DETACHED_CAPTURE_HOST_ONLY_VALIDATION_SCHEMA_VERSION_V1 = (
    "eiel_acs_native_detached_capture_host_only_validation_v1"
)
DETACHED_SLOT_STATES_V1 = (
    "FREE",
    "CAPTURING",
    "CAPTURED_HOST_ONLY",
    "PERSISTING",
    "DURABLE",
    "FAILED",
)
DETACHED_ASYNC_SLOT_CAPACITIES_V1 = frozenset({1, 2})
DETACHED_COPY_BUFFER_BYTES_V1 = 8 * 1024 * 1024


class _DiscardDigestV1:
    """Sink-compatible digest used when canonical payload order does not exist yet."""

    @staticmethod
    def update(_value: bytes | bytearray | memoryview) -> None:
        return None


class ACSNativeDetachedSlotError(RuntimeError):
    """A detached ACS slot failed closed."""


class _DetachedFinalizeCancelledV1(ACSNativeDetachedSlotError):
    """Internal cancellation fence for an unacknowledged detached publication."""


def _raise_if_cancelled_v1(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise _DetachedFinalizeCancelledV1(
            "detached ACS finalization was cancelled before acknowledgement"
        )


def _canonical_json(value: Any) -> bytes:
    return streamed._canonical_json(value)


def _sha256(data: bytes) -> str:
    return streamed._sha256(data)


def _assert_no_tensors(value: Any, *, path: str = "metadata") -> None:
    """Reject any live tensor/device object crossing into the worker queue."""

    if isinstance(value, torch.Tensor) or hasattr(value, "cuda_stream"):
        raise ACSNativeDetachedSlotError(
            f"detached ACS worker metadata retains a device object: {path}"
        )
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_no_tensors(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_no_tensors(item, path=f"{path}[{index}]")


def _assert_no_callables(value: Any, *, path: str = "metadata") -> None:
    """Reject executable closures from the detached worker boundary."""

    if callable(value) and not isinstance(value, type):
        raise ACSNativeDetachedSlotError(
            f"detached ACS worker metadata retains executable state: {path}"
        )
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_no_callables(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_no_callables(item, path=f"{path}[{index}]")


def validate_detached_capture_host_only_v1(
    capture: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove a detached capture is inert, deterministic host-only metadata."""

    if not isinstance(capture, Mapping):
        raise ACSNativeDetachedSlotError("detached capture is not a mapping")
    _assert_no_tensors(capture)
    _assert_no_callables(capture)
    try:
        canonical_capture = _canonical_json(capture)
    except (TypeError, ValueError) as exc:
        raise ACSNativeDetachedSlotError(
            "detached capture is not deterministic JSON"
        ) from exc
    return {
        "schema_version": (
            DETACHED_CAPTURE_HOST_ONLY_VALIDATION_SCHEMA_VERSION_V1
        ),
        "capture_schema_version": capture.get("schema_version"),
        "capture_sha256": _sha256(canonical_capture),
        "tensor_free": True,
        "callable_free": True,
    }


def _file_sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as stream_handle:
        for chunk in iter(
            lambda: stream_handle.read(DETACHED_COPY_BUFFER_BYTES_V1), b""
        ):
            digest.update(chunk)
            total += len(chunk)
    return digest.hexdigest(), total


def _spool_file_identity_v1(path: Path) -> dict[str, int]:
    try:
        value = path.lstat()
    except OSError as exc:
        raise ACSNativeDetachedSlotError(
            f"detached cursor spool stat failed: {path}: {exc}"
        ) from exc
    if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
        raise ACSNativeDetachedSlotError(
            "detached cursor spool must be one regular unlinked-alias file"
        )
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "bytes": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
        "links": int(value.st_nlink),
    }


def _spool_descriptor_digest_v1(capture: Mapping[str, Any]) -> str:
    return _sha256(
        _canonical_json(
            {
                "schema_version": DETACHED_SPOOL_CAPTURE_SCHEMA_VERSION_V2,
                "header": capture.get("header"),
                "spool_index": capture.get("spool_index"),
                "spool_bytes": capture.get("spool_bytes"),
                "source_projection_sha256s": capture.get(
                    "source_projection_sha256s"
                ),
            }
        )
    )


def _validate_spool_descriptor_v1(
    capture: Mapping[str, Any],
) -> tuple[Path, Mapping[str, Any], Mapping[str, Any]]:
    """Validate the immutable cell-major spool and its exact byte coverage."""

    _assert_no_tensors(capture)
    _assert_no_callables(capture)
    if capture.get("schema_version") != DETACHED_SPOOL_CAPTURE_SCHEMA_VERSION_V2:
        raise ACSNativeDetachedSlotError("detached cursor spool schema is invalid")
    spool_path = Path(str(capture.get("spool_path") or "")).resolve()
    header = capture.get("header")
    spool_index = capture.get("spool_index")
    seal = capture.get("spool_seal")
    if (
        not isinstance(header, Mapping)
        or not isinstance(spool_index, Mapping)
        or not isinstance(seal, Mapping)
        or seal.get("schema_version") != DETACHED_SPOOL_SEAL_SCHEMA_VERSION_V1
    ):
        raise ACSNativeDetachedSlotError(
            "detached cursor spool authority is incomplete"
        )
    expected_bytes = capture.get("spool_bytes")
    if type(expected_bytes) is not int or expected_bytes < 0:
        raise ACSNativeDetachedSlotError("detached cursor spool size is invalid")
    identity = _spool_file_identity_v1(spool_path)
    if identity != seal.get("file_identity") or identity["bytes"] != expected_bytes:
        raise ACSNativeDetachedSlotError(
            "detached cursor spool file identity changed"
        )
    descriptor_digest = _spool_descriptor_digest_v1(capture)
    if seal.get("descriptor_sha256") != descriptor_digest:
        raise ACSNativeDetachedSlotError(
            "detached cursor spool descriptor digest differs"
        )

    cell_ids = header.get("cell_ids")
    cells = header.get("cells")
    field_order = header.get("field_order")
    cell_rows = spool_index.get("cell_field_rows")
    metadata_refs = spool_index.get("metadata_spool_refs")
    source_digests = capture.get("source_projection_sha256s")
    if (
        not isinstance(cell_ids, list)
        or not cell_ids
        or len(set(cell_ids)) != len(cell_ids)
        or not all(isinstance(value, str) and value for value in cell_ids)
        or not isinstance(cells, list)
        or len(cells) != len(cell_ids)
        or not isinstance(field_order, list)
        or not field_order
        or len(set(field_order)) != len(field_order)
        or not isinstance(cell_rows, list)
        or len(cell_rows) != len(cell_ids)
        or not isinstance(metadata_refs, list)
        or len(metadata_refs) != len(cell_ids)
        or not isinstance(source_digests, Mapping)
        or set(source_digests) != set(cell_ids)
    ):
        raise ACSNativeDetachedSlotError("detached cursor spool index is malformed")

    expected_offset = 0
    for cell_index, cell_id in enumerate(cell_ids):
        cell = cells[cell_index]
        rows = cell_rows[cell_index]
        if (
            not isinstance(cell, Mapping)
            or cell.get("cell_id") != cell_id
            or cell.get("source_projection_sha256") != source_digests[cell_id]
            or not aggregate._is_sha256(source_digests[cell_id])
            or not isinstance(rows, list)
            or len(rows) != len(field_order)
        ):
            raise ACSNativeDetachedSlotError(
                "detached cursor spool cell authority is malformed"
            )
        final_rows = cell.get("fields")
        if not isinstance(final_rows, list) or len(final_rows) != len(field_order):
            raise ACSNativeDetachedSlotError(
                "detached cursor spool final field index is malformed"
            )
        for field_index, field in enumerate(field_order):
            row = rows[field_index]
            final_row = final_rows[field_index]
            if (
                not isinstance(row, Mapping)
                or not isinstance(final_row, Mapping)
                or row.get("field") != field
                or final_row.get("field") != field
            ):
                raise ACSNativeDetachedSlotError(
                    "detached cursor spool field order differs"
                )
            if row.get("storage") == "zero_fill":
                if final_row.get("storage") != "zero_fill" or row.get("bytes") != 0:
                    raise ACSNativeDetachedSlotError(
                        "detached cursor zero-fill index differs"
                    )
                continue
            chunks = row.get("spool_chunks")
            if (
                row.get("storage") != "spool"
                or final_row.get("storage") != "payload"
                or not isinstance(chunks, list)
                or type(row.get("bytes")) is not int
                or row["bytes"] < 0
                or row.get("spool_offset") != expected_offset
                or not aggregate._is_sha256(row.get("sha256"))
            ):
                raise ACSNativeDetachedSlotError(
                    "detached cursor spool field range is invalid"
                )
            field_bytes = 0
            for chunk in chunks:
                if (
                    not isinstance(chunk, Mapping)
                    or chunk.get("offset") != expected_offset + field_bytes
                    or type(chunk.get("bytes")) is not int
                    or chunk["bytes"] < 0
                    or not aggregate._is_sha256(chunk.get("sha256"))
                ):
                    raise ACSNativeDetachedSlotError(
                        "detached cursor spool chunk coverage is invalid"
                    )
                field_bytes += int(chunk["bytes"])
            if field_bytes != row["bytes"]:
                raise ACSNativeDetachedSlotError(
                    "detached cursor spool field coverage differs"
                )
            expected_offset += field_bytes
        metadata_ref = metadata_refs[cell_index]
        if (
            not isinstance(metadata_ref, Mapping)
            or metadata_ref.get("spool_offset") != expected_offset
            or type(metadata_ref.get("bytes")) is not int
            or metadata_ref["bytes"] <= 0
            or not aggregate._is_sha256(metadata_ref.get("sha256"))
        ):
            raise ACSNativeDetachedSlotError(
                "detached cursor spool metadata coverage is invalid"
            )
        expected_offset += int(metadata_ref["bytes"])
    if expected_offset != expected_bytes:
        raise ACSNativeDetachedSlotError(
            "detached cursor spool index does not cover the file exactly"
        )
    return spool_path, header, spool_index


def load_acs_native_detached_spool_cell_v1(
    capture: Mapping[str, Any],
    *,
    cell_id: str,
    expected_source_projection_sha256: str | None = None,
) -> dict[str, Any]:
    """Restore one cell from a sealed producer spool without waiting for ACS."""

    spool_path, header, spool_index = _validate_spool_descriptor_v1(capture)
    identity_before = _spool_file_identity_v1(spool_path)
    cell_ids = list(header["cell_ids"])
    if cell_id not in cell_ids:
        raise ACSNativeDetachedSlotError(
            f"detached cursor spool cell is absent: {cell_id}"
        )
    cell_index = cell_ids.index(cell_id)
    cell = header["cells"][cell_index]
    source_digest = cell.get("source_projection_sha256")
    if (
        not aggregate._is_sha256(source_digest)
        or expected_source_projection_sha256 is not None
        and source_digest != expected_source_projection_sha256
    ):
        raise ACSNativeDetachedSlotError(
            "detached cursor spool source digest differs"
        )
    rows = spool_index["cell_field_rows"][cell_index]
    metadata_ref = spool_index["metadata_spool_refs"][cell_index]
    tensors: dict[str, torch.Tensor] = {}
    try:
        with spool_path.open("rb") as stream_handle:
            for field_index, (field, row) in enumerate(
                zip(header["field_order"], rows, strict=True)
            ):
                dtype_name = row.get("dtype")
                shape_list = row.get("shape")
                logical = row.get("logical_bytes")
                if (
                    not isinstance(dtype_name, str)
                    or dtype_name not in streamed._DTYPE_BY_NAME
                    or not isinstance(shape_list, list)
                    or not all(type(value) is int and value >= 0 for value in shape_list)
                    or type(logical) is not int
                    or logical < 0
                    or logical > aggregate.MAX_FIELD_BYTES
                ):
                    raise ACSNativeDetachedSlotError(
                        f"detached cursor spool field schema is invalid: {field}"
                    )
                shape = tuple(shape_list)
                element_bytes = torch.tensor(
                    [], dtype=streamed._DTYPE_BY_NAME[dtype_name]
                ).element_size()
                for dimension in shape:
                    element_bytes *= dimension
                if element_bytes != logical:
                    raise ACSNativeDetachedSlotError(
                        f"detached cursor spool logical size differs: {field}"
                    )
                if row.get("storage") == "zero_fill":
                    zero_digest = aggregate._zero_digest(logical)
                    if (
                        row.get("sha256") != zero_digest
                        or row.get("zero_fill_descriptor")
                        != aggregate._zero_fill_descriptor(
                            field=field,
                            dtype=dtype_name,
                            shape=shape_list,
                            field_index=field_index,
                            cell_index=cell_index,
                            logical_bytes=logical,
                            logical_digest_sha256=zero_digest,
                        )
                    ):
                        raise ACSNativeDetachedSlotError(
                            f"detached cursor spool zero-fill authority differs: {field}"
                        )
                    tensors[field] = torch.zeros(
                        shape, dtype=streamed._DTYPE_BY_NAME[dtype_name]
                    )
                    continue
                raw_parts: list[bytes] = []
                for chunk in row["spool_chunks"]:
                    stream_handle.seek(int(chunk["offset"]))
                    raw = stream_handle.read(int(chunk["bytes"]))
                    if (
                        len(raw) != int(chunk["bytes"])
                        or _sha256(raw) != chunk["sha256"]
                    ):
                        raise ACSNativeDetachedSlotError(
                            f"detached cursor spool chunk is corrupted: {field}"
                        )
                    raw_parts.append(raw)
                raw_field = b"".join(raw_parts)
                if len(raw_field) != logical or _sha256(raw_field) != row["sha256"]:
                    raise ACSNativeDetachedSlotError(
                        f"detached cursor spool field digest differs: {field}"
                    )
                tensors[field] = streamed._tensor_from_bytes(
                    raw_field, dtype_name=dtype_name, shape=shape
                )
            stream_handle.seek(int(metadata_ref["spool_offset"]))
            raw_metadata = stream_handle.read(int(metadata_ref["bytes"]))
    except OSError as exc:
        raise ACSNativeDetachedSlotError(
            f"detached cursor spool read failed: {spool_path}: {exc}"
        ) from exc
    if (
        len(raw_metadata) != int(metadata_ref["bytes"])
        or _sha256(raw_metadata) != metadata_ref["sha256"]
    ):
        raise ACSNativeDetachedSlotError(
            "detached cursor spool metadata is corrupted"
        )
    try:
        metadata = aggregate._strict_json(raw_metadata, label="metadata payload")
        if metadata.get("schema_version") != aggregate.METADATA_SCHEMA_VERSION:
            raise ACSNativeDetachedSlotError(
                "detached cursor spool metadata identity differs"
            )
        projection_value = streamed._unpack_value(metadata["projection"])
        architecture_identity = streamed._unpack_value(
            metadata["architecture_b_identity"]
        )
        observer = streamed._unpack_value(metadata["observer_continuation"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ACSNativeDetachedSlotError(
            "detached cursor spool metadata is malformed"
        ) from exc
    if (
        not isinstance(projection_value, Mapping)
        or not isinstance(architecture_identity, Mapping)
        or not isinstance(observer, Mapping)
    ):
        raise ACSNativeDetachedSlotError(
            "detached cursor spool metadata authorities are malformed"
        )
    for identity_field in ("snapshot_digest_sha256", "source_projection_sha256"):
        identity_digest = architecture_identity.get(identity_field)
        if identity_digest is not None and identity_digest != source_digest:
            raise ACSNativeDetachedSlotError(
                "detached cursor spool architecture identity binding differs"
            )
    if _spool_file_identity_v1(spool_path) != identity_before:
        raise ACSNativeDetachedSlotError(
            "detached cursor spool changed during cell restore"
        )
    projection = dict(projection_value)
    projection["tensors"] = tensors
    return {
        "projection": projection,
        "architecture_b_identity": dict(architecture_identity),
        "observer_continuation": dict(observer),
        "cell_id": cell_id,
        "source_projection_sha256": str(source_digest),
    }


def _record_producer_event_v1(
    tensor: Any,
    dedicated_stream: Any | None,
) -> Any | None:
    """Record the producer-stream frontier before a detached D2H copy.

    The provider establishes a device-wide readiness fence before opening the
    bounded cursor.  This event is still required at the copy seam: it makes
    the producer-to-dedicated-copy-stream dependency explicit and prevents a
    future caller from relying on an accidental current-stream ordering.
    """

    if dedicated_stream is None or getattr(getattr(tensor, "device", None), "type", None) != "cuda":
        return None
    try:
        producer_event = torch.cuda.Event()
        producer_event.record(torch.cuda.current_stream(device=tensor.device))
        return producer_event
    except (RuntimeError, ValueError) as exc:
        raise ACSNativeDetachedSlotError(
            "detached CUDA producer event could not be recorded"
        ) from exc


def _quarantine_capture_file_v1(capture: Mapping[str, Any]) -> str | None:
    """Move an unsealed capture out of the live namespace after failure."""

    selected_key: str | None = None
    capture_path: Path | None = None
    keys = (
        ("spool_path", "capture_path", "materialization_path")
        if capture.get("source_consumer_required") is True
        else ("capture_path", "materialization_path", "spool_path")
    )
    for key in keys:
        raw_path = capture.get(key)
        if not isinstance(raw_path, str) or not raw_path:
            continue
        candidate = Path(raw_path).expanduser().resolve()
        if candidate.is_file():
            selected_key = key
            capture_path = candidate
            break
    if capture_path is None:
        return None
    is_capture = selected_key in {"capture_path", "materialization_path"}
    suffix = ".capture.tmp" if is_capture else ".spool.tmp"
    failed_suffix = ".capture.failed" if is_capture else ".spool.failed"
    if capture_path.name.endswith(suffix):
        failed_name = capture_path.name[: -len(suffix)] + failed_suffix
    else:
        failed_name = f"{capture_path.name}.failed"
    failed_path = capture_path.with_name(failed_name)
    if failed_path.exists():
        failed_path = capture_path.with_name(
            f"{failed_name}.{os.getpid()}.{uuid4().hex}"
        )
    try:
        os.replace(capture_path, failed_path)
    except OSError as exc:
        raise ACSNativeDetachedSlotError(
            "failed detached capture could not leave the live namespace"
        ) from exc
    return str(failed_path)


def _quarantine_owned_publication_v1(path: str | Path) -> str | None:
    """Remove a cancelled worker-owned publication from the canonical namespace."""

    publication = Path(path).expanduser().resolve()
    if (
        (not publication.exists() and not publication.is_symlink())
        or publication.is_symlink()
        or (not publication.is_file() and not publication.is_dir())
    ):
        return None
    failed_path = publication.with_name(f"{publication.name}.publication.failed")
    if failed_path.exists():
        failed_path = publication.with_name(
            f"{failed_path.name}.{os.getpid()}.{uuid4().hex}"
        )
    try:
        os.replace(publication, failed_path)
        directory_fd = os.open(publication.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as exc:
        raise ACSNativeDetachedSlotError(
            "cancelled detached publication could not leave the canonical namespace"
        ) from exc
    return str(failed_path)


def _validate_capture_inputs(
    *,
    projections: Sequence[Mapping[str, Any]],
    cell_ids: Sequence[str],
    source_projection_sha256s: Mapping[str, str],
    architecture_b_identities: Mapping[str, Mapping[str, Any]],
    observer_continuations: Mapping[str, Mapping[str, Any]],
    generation: int,
    tick: int,
    absolute_next_tick: int | None,
    batch_id: str,
    segment_ordinal: int,
    chunk_bytes: int,
) -> None:
    if not projections or len(projections) != len(cell_ids):
        raise ACSNativeDetachedSlotError(
            "detached aggregate capture projection and cell counts differ"
        )
    if any(not isinstance(cell_id, str) or not cell_id for cell_id in cell_ids):
        raise ACSNativeDetachedSlotError("detached aggregate cell IDs are invalid")
    if len(set(cell_ids)) != len(cell_ids):
        raise ACSNativeDetachedSlotError("detached aggregate cell IDs are duplicated")
    if set(source_projection_sha256s) != set(cell_ids):
        raise ACSNativeDetachedSlotError(
            "detached aggregate source digest authority does not cover the batch"
        )
    if set(architecture_b_identities) != set(cell_ids):
        raise ACSNativeDetachedSlotError(
            "detached aggregate identity authority does not cover the batch"
        )
    if set(observer_continuations) != set(cell_ids):
        raise ACSNativeDetachedSlotError(
            "detached aggregate observer authority does not cover the batch"
        )
    for cell_id in cell_ids:
        if not aggregate._is_sha256(source_projection_sha256s[cell_id]):
            raise ACSNativeDetachedSlotError(
                f"detached source digest is invalid for {cell_id}"
            )
        _assert_no_tensors(architecture_b_identities[cell_id], path=f"identity.{cell_id}")
        _assert_no_tensors(observer_continuations[cell_id], path=f"observer.{cell_id}")
    if type(generation) is not int or generation < 0:
        raise ACSNativeDetachedSlotError("generation is invalid")
    if type(tick) is not int or tick < 0:
        raise ACSNativeDetachedSlotError("tick is invalid")
    if absolute_next_tick is not None and (
        type(absolute_next_tick) is not int or absolute_next_tick < 0
    ):
        raise ACSNativeDetachedSlotError("absolute_next_tick is invalid")
    if not isinstance(batch_id, str) or not batch_id:
        raise ACSNativeDetachedSlotError("batch_id is invalid")
    if type(segment_ordinal) is not int or segment_ordinal < 0:
        raise ACSNativeDetachedSlotError("segment_ordinal is invalid")
    if type(chunk_bytes) is not int or not 0 < chunk_bytes <= aggregate.MAX_CHUNK_BYTES:
        raise ACSNativeDetachedSlotError("chunk_bytes is outside the native bound")


def _aggregate_capture_header(
    *,
    projections: Sequence[Mapping[str, Any]],
    cell_ids: Sequence[str],
    source_projection_sha256s: Mapping[str, str],
    architecture_b_identities: Mapping[str, Mapping[str, Any]],
    observer_continuations: Mapping[str, Mapping[str, Any]],
    generation: int,
    tick: int,
    absolute_next_tick: int | None,
    batch_id: str,
    segment_ordinal: int,
    chunk_bytes: int,
    destination: Path,
    require_native_extension: bool,
) -> tuple[dict[str, Any], dict[str, Any], int, int, int, int, float]:
    """Capture payload bytes and return a worker-safe header plus telemetry."""

    non_tensors_by_cell: list[dict[str, Any]] = []
    tensors_by_cell: list[dict[str, torch.Tensor]] = []
    for projection in projections:
        try:
            non_tensors, tensors = streamed._validate_projection(projection)
        except streamed.ACSNativeCheckpointError as exc:
            raise ACSNativeDetachedSlotError(str(exc)) from exc
        non_tensors_by_cell.append(non_tensors)
        tensors_by_cell.append(tensors)
    field_order = sorted(tensors_by_cell[0])
    if not field_order:
        raise ACSNativeDetachedSlotError("detached aggregate has no tensor fields")
    expected_fields = set(field_order)
    for tensors in tensors_by_cell:
        if set(tensors) != expected_fields:
            raise ACSNativeDetachedSlotError(
                "detached aggregate tensor field schemas differ"
            )

    field_schema: dict[str, dict[str, Any]] = {}
    for field in field_order:
        first = tensors_by_cell[0][field]
        dtype_name = streamed._dtype_name(first)
        shape = [int(value) for value in first.shape]
        logical = aggregate._logical_bytes(first)
        field_schema[field] = {
            "field": field,
            "dtype": dtype_name,
            "dtype_code": streamed._DTYPE_CODE[dtype_name],
            "shape": shape,
            "logical_bytes": logical,
        }
        for tensors in tensors_by_cell[1:]:
            if (
                streamed._dtype_name(tensors[field]) != dtype_name
                or [int(value) for value in tensors[field].shape] != shape
            ):
                raise ACSNativeDetachedSlotError(
                    f"detached aggregate field schema differs: {field}"
                )

    cells_header: list[dict[str, Any]] = []
    for cell_id, non_tensors in zip(cell_ids, non_tensors_by_cell, strict=True):
        identity = dict(architecture_b_identities[cell_id])
        source_digest = source_projection_sha256s[cell_id]
        for identity_field in ("snapshot_digest_sha256", "source_projection_sha256"):
            if identity.get(identity_field) not in {None, source_digest}:
                raise ACSNativeDetachedSlotError(
                    f"detached identity digest differs for {cell_id}"
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
        aggregate._fixed_mlp_zero_fill_contract(non_tensors)
        for non_tensors in non_tensors_by_cell
    ]
    zero_families = [
        aggregate._route_policy_family(non_tensors) == "mlp"
        for non_tensors in non_tensors_by_cell
    ]
    if any(zero_families) and not all(zero_families):
        raise ACSNativeDetachedSlotError(
            "detached aggregate cells disagree about zero-fill eligibility"
        )
    if any(zero_families) and any(contract is None for contract in zero_contracts):
        raise ACSNativeDetachedSlotError("detached aggregate zero-fill contract is missing")
    if any(contract != zero_contracts[0] for contract in zero_contracts[1:]):
        raise ACSNativeDetachedSlotError("detached aggregate zero-fill contracts differ")
    zero_contract = zero_contracts[0] if zero_contracts else None
    if zero_contract is not None:
        missing = sorted(aggregate.ZERO_FILL_CANDIDATE_FIELDS - expected_fields)
        if missing:
            raise ACSNativeDetachedSlotError(
                "detached aggregate zero-fill fields are missing: " + ", ".join(missing)
            )

    destination.parent.mkdir(parents=True, exist_ok=True)
    capture_path = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid4().hex}.capture.tmp"
    )
    payload_digest = hashlib.sha256()
    payload_bytes = 0
    metadata_bytes = 0
    logical_bytes = 0
    zero_fill_logical_bytes = 0
    chunk_count = 0
    zero_digest_cache: dict[int, str] = {}
    cuda_fields = any(
        tensor.device.type == "cuda"
        for tensors in tensors_by_cell
        for tensor in tensors.values()
    )
    native_pool_type = streamed._native_slab_pool_type()
    if require_native_extension and native_pool_type is None:
        raise ACSNativeDetachedSlotError(
            "detached production capture requires the native two-slab pool"
        )
    native_pool = (
        native_pool_type(chunk_bytes, cuda_fields)
        if native_pool_type is not None
        else None
    )
    fallback_pool = (
        None
        if native_pool is not None
        else streamed._TwoSlabPool(chunk_bytes, cuda=cuda_fields)
    )
    dedicated_stream = None
    if cuda_fields:
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable")
            dedicated_stream = torch.cuda.Stream()
        except (RuntimeError, ValueError) as exc:
            if require_native_extension:
                raise ACSNativeDetachedSlotError(
                    f"detached dedicated CUDA stream unavailable: {exc}"
                ) from exc
            raise ACSNativeDetachedSlotError(str(exc)) from exc
    native_pool_telemetry = (
        dict(native_pool.telemetry())
        if native_pool is not None
        else {
            "schema_version": "eiel_acs_native_slab_pool_fallback_v1",
            "chunk_bytes": chunk_bytes,
            "slab_count": 2,
            "pinned": False,
            "preallocated": False,
        }
    )
    if require_native_extension and (
        native_pool_telemetry.get("pinned") is not True
        or native_pool_telemetry.get("slab_count") != 2
        or native_pool_telemetry.get("preallocated") is not True
    ):
        raise ACSNativeDetachedSlotError(
            "detached production capture did not obtain two preallocated pinned slabs"
        )

    started = time.perf_counter()
    try:
        with capture_path.open("xb") as stream_handle:
            payload_offset = 0
            for field_index, field in enumerate(field_order):
                schema = field_schema[field]
                for cell_index, (cell_id, tensors) in enumerate(
                    zip(cell_ids, tensors_by_cell, strict=True)
                ):
                    tensor = tensors[field]
                    logical = int(schema["logical_bytes"])
                    logical_bytes += logical
                    zero_fill = zero_contract is not None and field in aggregate.ZERO_FILL_CANDIDATE_FIELDS
                    if zero_fill:
                        if int(torch.count_nonzero(tensor.detach()).item()) != 0:
                            raise ACSNativeDetachedSlotError(
                                f"detached zero-fill field is nonzero: {field}/{cell_id}"
                            )
                        zero_fill_logical_bytes += logical
                        zero_digest = zero_digest_cache.setdefault(
                            logical, aggregate._zero_digest(logical)
                        )
                        row = {
                            "field": field,
                            "storage": "zero_fill",
                            "offset": None,
                            "bytes": 0,
                            "logical_bytes": logical,
                            "sha256": zero_digest,
                            "chunks": [],
                            "zero_fill_contract": "fixed_mlp_exact_zero_v1",
                            "zero_fill_descriptor": aggregate._zero_fill_descriptor(
                                field=field,
                                dtype=str(schema["dtype"]),
                                shape=schema["shape"],
                                field_index=field_index,
                                cell_index=cell_index,
                                logical_bytes=logical,
                                logical_digest_sha256=zero_digest,
                            ),
                        }
                    else:
                        sink = aggregate._AggregateFieldSink(
                            stream_handle,
                            offset=payload_offset,
                            payload_digest=payload_digest,
                        )
                        producer_event = _record_producer_event_v1(
                            tensor, dedicated_stream
                        )
                        copied = aggregate._copy_tensor(
                            tensor,
                            sink,
                            chunk_bytes=chunk_bytes,
                            fallback_pool=fallback_pool,
                            native_pool=native_pool,
                            dedicated_stream=dedicated_stream,
                            producer_event=producer_event,
                        )
                        expected_chunks = (logical + chunk_bytes - 1) // chunk_bytes
                        if sink.bytes != logical or copied != expected_chunks:
                            raise ACSNativeDetachedSlotError(
                                f"detached aggregate field coverage differs: {field}/{cell_id}"
                            )
                        row = {
                            "field": field,
                            "storage": "payload",
                            "offset": payload_offset,
                            "bytes": sink.bytes,
                            "logical_bytes": logical,
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
            for cell_index, (cell_id, non_tensors) in enumerate(
                zip(cell_ids, non_tensors_by_cell, strict=True)
            ):
                raw = aggregate._metadata_record_bytes(
                    projection=non_tensors,
                    identity=dict(architecture_b_identities[cell_id]),
                    observer=dict(observer_continuations[cell_id]),
                )
                stream_handle.write(raw)
                payload_digest.update(raw)
                cells_header[cell_index]["metadata_ref"] = aggregate._metadata_ref(
                    raw, offset=metadata_offset + metadata_bytes
                )
                metadata_bytes += len(raw)
            if metadata_bytes > aggregate.MAX_METADATA_BYTES:
                raise ACSNativeDetachedSlotError(
                    "detached aggregate metadata payload exceeds its bound"
                )
            file_payload_bytes = payload_bytes + metadata_bytes
            header = {
                "schema_version": aggregate.SCHEMA_VERSION,
                "format_version": aggregate.FORMAT_VERSION,
                "endianness": "little",
                "payload_layout": aggregate.METADATA_PAYLOAD_LAYOUT,
                "metadata_schema_version": aggregate.METADATA_SCHEMA_VERSION,
                "batch_id": batch_id,
                "generation": generation,
                "tick": tick,
                "absolute_next_tick": absolute_next_tick,
                "segment_ordinal": segment_ordinal,
                "cell_ids": list(cell_ids),
                "field_order": list(field_order),
                "field_schema": [field_schema[field] for field in field_order],
                "cells": cells_header,
                "chunk_bytes": chunk_bytes,
                "chunk_count": chunk_count,
                "payload_bytes": payload_bytes,
                "file_payload_bytes": file_payload_bytes,
                "metadata_offset": metadata_offset,
                "metadata_bytes": metadata_bytes,
                "logical_bytes": logical_bytes,
                "zero_fill_logical_bytes": zero_fill_logical_bytes,
                "zero_fill_contract": zero_contract,
                "zero_fill_fields": sorted(
                    field
                    for field in field_order
                    if zero_contract is not None
                    and field in aggregate.ZERO_FILL_CANDIDATE_FIELDS
                ),
            }
            header_bytes = _canonical_json(header)
            if len(header_bytes) > aggregate.MAX_HEADER_BYTES:
                raise ACSNativeDetachedSlotError("detached aggregate header exceeds its bound")
            stream_handle.flush()
    except BaseException:
        capture_path.unlink(missing_ok=True)
        raise
    finally:
        # These are deliberately not put into the returned receipt.  The only
        # live objects crossing the slot boundary are the file path and JSON
        # scalars above.
        del tensors_by_cell, non_tensors_by_cell

    _assert_no_tensors(header)
    return (
        {
            "schema_version": DETACHED_CAPTURE_SCHEMA_VERSION_V1,
            "capture_path": str(capture_path),
            "destination": str(destination),
            "header": header,
            "cell_ids": tuple(cell_ids),
            "batch_id": batch_id,
            "generation": generation,
            "tick": tick,
            "absolute_next_tick": absolute_next_tick,
            "segment_ordinal": segment_ordinal,
            "source_projection_sha256s": dict(source_projection_sha256s),
            "native_slab_pool": native_pool_telemetry,
            "capture_seconds": time.perf_counter() - started,
            "payload_bytes": payload_bytes,
            "metadata_bytes": metadata_bytes,
            "file_payload_bytes": file_payload_bytes,
            "logical_bytes": logical_bytes,
            "zero_fill_logical_bytes": zero_fill_logical_bytes,
            "chunk_count": chunk_count,
            "dedicated_d2h_stream": dedicated_stream is not None,
        },
        header,
        payload_bytes,
        logical_bytes,
        zero_fill_logical_bytes,
        chunk_count,
        time.perf_counter() - started,
    )


def capture_acs_native_aggregate_payload_v1(
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
    chunk_bytes: int = aggregate.DEFAULT_CHUNK_BYTES,
    require_native_extension: bool = False,
) -> dict[str, Any]:
    """Capture GPU-backed fields into one unique file-backed payload."""

    _validate_capture_inputs(
        projections=projections,
        cell_ids=cell_ids,
        source_projection_sha256s=source_projection_sha256s,
        architecture_b_identities=architecture_b_identities,
        observer_continuations=observer_continuations,
        generation=generation,
        tick=tick,
        absolute_next_tick=absolute_next_tick,
        batch_id=batch_id,
        segment_ordinal=segment_ordinal,
        chunk_bytes=chunk_bytes,
    )
    capture, _header, _payload, _logical, _zero, _chunks, _seconds = _aggregate_capture_header(
        projections=projections,
        cell_ids=cell_ids,
        source_projection_sha256s=source_projection_sha256s,
        architecture_b_identities=architecture_b_identities,
        observer_continuations=observer_continuations,
        generation=generation,
        tick=tick,
        absolute_next_tick=absolute_next_tick,
        batch_id=batch_id,
        segment_ordinal=segment_ordinal,
        chunk_bytes=chunk_bytes,
        destination=Path(destination).expanduser().resolve(),
        require_native_extension=require_native_extension,
    )
    return capture


def capture_acs_native_aggregate_payload_from_cursor_v1(
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
    chunk_bytes: int = aggregate.DEFAULT_CHUNK_BYTES,
    require_native_extension: bool = False,
) -> dict[str, Any]:
    """Capture one aggregate payload from a strict one-cell projection cursor.

    The detached route cannot turn a tick-native cursor into a batch tuple:
    that would recreate the long-horizon host staging problem.  This adapter
    therefore consumes one cursor cell at a time, spools only bounded field
    bytes to disk, and assembles the same field-major aggregate payload that
    the worker later seals.  No tensor-bearing object crosses the cursor
    release or detached worker boundary.
    """

    normalized_cell_ids = tuple(cell_ids)
    if not normalized_cell_ids:
        raise ACSNativeDetachedSlotError("detached cursor capture has no cells")
    if any(not isinstance(cell_id, str) or not cell_id for cell_id in normalized_cell_ids):
        raise ACSNativeDetachedSlotError("detached cursor capture cell IDs are invalid")
    if len(set(normalized_cell_ids)) != len(normalized_cell_ids):
        raise ACSNativeDetachedSlotError("detached cursor capture cell IDs are duplicated")
    if tuple(getattr(cursor, "cell_ids", ())) != normalized_cell_ids:
        raise ACSNativeDetachedSlotError(
            "detached cursor capture cell order differs from the batch"
        )
    if not callable(authority_factory):
        raise ACSNativeDetachedSlotError(
            "detached cursor capture authority factory is not callable"
        )
    if type(chunk_bytes) is not int or not 0 < chunk_bytes <= aggregate.MAX_CHUNK_BYTES:
        raise ACSNativeDetachedSlotError("detached cursor capture chunk size is invalid")
    if type(generation) is not int or generation < 0:
        raise ACSNativeDetachedSlotError("detached cursor capture generation is invalid")
    if type(tick) is not int or tick < 0:
        raise ACSNativeDetachedSlotError("detached cursor capture tick is invalid")
    if absolute_next_tick is not None and (
        type(absolute_next_tick) is not int or absolute_next_tick < 0
    ):
        raise ACSNativeDetachedSlotError(
            "detached cursor capture absolute tick is invalid"
        )
    if not isinstance(batch_id, str) or not batch_id:
        raise ACSNativeDetachedSlotError("detached cursor capture batch ID is invalid")
    if type(segment_ordinal) is not int or segment_ordinal < 0:
        raise ACSNativeDetachedSlotError(
            "detached cursor capture segment ordinal is invalid"
        )

    target = Path(destination).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    token = f".{target.name}.{os.getpid()}.{uuid4().hex}"
    capture_path = target.with_name(f"{token}.capture.tmp")
    spool_path = target.with_name(f"{token}.spool.tmp")
    started = time.perf_counter()
    started_perf_ns = time.perf_counter_ns()
    started_cpu = time.process_time_ns()
    field_order: list[str] | None = None
    field_schema: dict[str, dict[str, Any]] = {}
    per_cell_field_schema = False
    cells_header: list[dict[str, Any]] = []
    cell_field_rows: list[list[dict[str, Any]]] = []
    authorities_by_cell: dict[str, dict[str, Any]] = {}
    source_projection_sha256s: dict[str, str] = {}
    compact_result_rows: list[dict[str, Any]] = []
    # The producer spool is cell-major while the canonical aggregate is
    # field-major, so a spool-wide digest has no checkpoint authority. Field
    # and chunk digests still bind every byte; the worker computes the one
    # authoritative payload digest after canonical materialization.
    spool_digest = _DiscardDigestV1()
    metadata_spool_refs: list[dict[str, Any]] = []
    total_logical_bytes = 0
    zero_fill_logical_bytes = 0
    zero_contracts: list[dict[str, Any] | None] = []
    zero_fill_families: list[bool] = []
    native_pool: Any | None = None
    fallback_pool: Any | None = None
    dedicated_stream: Any | None = None
    cuda_fields: bool | None = None

    def initialize_pool(has_cuda_fields: bool) -> None:
        nonlocal native_pool, fallback_pool, dedicated_stream, cuda_fields
        if cuda_fields is not None:
            if cuda_fields is not has_cuda_fields:
                raise ACSNativeDetachedSlotError(
                    "detached cursor cells disagree about CUDA-backed fields"
                )
            return
        cuda_fields = has_cuda_fields
        native_pool_type = streamed._native_slab_pool_type()
        if require_native_extension and native_pool_type is None:
            raise ACSNativeDetachedSlotError(
                "detached production capture requires the native two-slab pool"
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
                    raise ACSNativeDetachedSlotError(
                        f"detached dedicated CUDA stream unavailable: {exc}"
                    ) from exc
                raise ACSNativeDetachedSlotError(str(exc)) from exc
        if native_pool is not None:
            try:
                telemetry = dict(native_pool.telemetry())
            except (AttributeError, TypeError, ValueError) as exc:
                if require_native_extension:
                    raise ACSNativeDetachedSlotError(
                        f"detached native slab pool telemetry is unavailable: {exc}"
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
        if require_native_extension and (
            telemetry.get("slab_count") != 2
            or telemetry.get("preallocated") is not True
            or (has_cuda_fields and telemetry.get("pinned") is not True)
        ):
            requirement = (
                "two preallocated pinned slabs"
                if has_cuda_fields
                else "two preallocated native slabs"
            )
            raise ACSNativeDetachedSlotError(
                "detached production capture did not obtain " + requirement
            )

    required_authority = {
        "source_projection_sha256",
        "architecture_b_identity",
        "observer_continuation",
        "mechanics_checkpoint_digest",
        "canonical_checkpoint_storage_digest",
    }
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
                    raise ACSNativeDetachedSlotError(
                        "detached cursor capture emitted an unordered cell"
                    )
                cell_id = normalized_cell_ids[expected_index]
                projection = detached.require_projection()
                authority = dict(authority_factory(cell_id, projection))
                _assert_no_tensors(authority, path=f"authority.{cell_id}")
                _assert_no_callables(authority, path=f"authority.{cell_id}")
                if not required_authority.issubset(authority):
                    raise ACSNativeDetachedSlotError(
                        f"detached cursor authority is incomplete for {cell_id}"
                    )
                source_digest = authority["source_projection_sha256"]
                identity = authority["architecture_b_identity"]
                observer = authority["observer_continuation"]
                if not aggregate._is_sha256(source_digest):
                    raise ACSNativeDetachedSlotError(
                        f"detached cursor source digest is invalid for {cell_id}"
                    )
                if not isinstance(identity, Mapping) or not isinstance(observer, Mapping):
                    raise ACSNativeDetachedSlotError(
                        f"detached cursor scalar authorities are invalid for {cell_id}"
                    )
                for digest_name in (
                    "mechanics_checkpoint_digest",
                    "canonical_checkpoint_storage_digest",
                ):
                    if not aggregate._is_sha256(authority[digest_name]):
                        raise ACSNativeDetachedSlotError(
                            f"detached cursor {digest_name} is invalid for {cell_id}"
                        )

                non_tensors, tensors = streamed._validate_projection(projection)
                current_fields = sorted(tensors)
                if not current_fields:
                    raise ACSNativeDetachedSlotError(
                        "detached cursor aggregate has no tensor fields"
                    )
                if len(current_fields) > aggregate.MAX_FIELD_COUNT:
                    raise ACSNativeDetachedSlotError(
                        "detached cursor aggregate field count exceeds the bound"
                    )
                if field_order is None:
                    field_order = current_fields
                    for field in field_order:
                        tensor = tensors[field]
                        dtype_name = streamed._dtype_name(tensor)
                        shape = [int(value) for value in tensor.shape]
                        field_schema[field] = {
                            "field": field,
                            "dtype": dtype_name,
                            "dtype_code": streamed._DTYPE_CODE[dtype_name],
                            "shape": shape,
                            "logical_bytes": aggregate._logical_bytes(tensor),
                        }
                elif current_fields != field_order:
                    raise ACSNativeDetachedSlotError(
                        "detached cursor aggregate cells do not expose one field schema"
                    )
                for field in field_order:
                    schema = field_schema[field]
                    tensor = tensors[field]
                    if (
                        streamed._dtype_name(tensor) != schema["dtype"]
                        or [int(value) for value in tensor.shape] != schema["shape"]
                    ):
                        per_cell_field_schema = True

                cells_header.append(
                    {
                        "cell_id": cell_id,
                        "source_projection_sha256": source_digest,
                        "metadata_ref": None,
                        "fields": [],
                    }
                )
                authorities_by_cell[cell_id] = authority
                source_projection_sha256s[cell_id] = str(source_digest)
                contract = aggregate._fixed_mlp_zero_fill_contract(non_tensors)
                family = aggregate._route_policy_family(non_tensors) == "mlp"
                zero_contracts.append(contract)
                zero_fill_families.append(family)
                rows: list[dict[str, Any]] = []
                for field_index, field in enumerate(field_order):
                    schema = field_schema[field]
                    tensor = tensors[field]
                    tensor_dtype = streamed._dtype_name(tensor)
                    tensor_shape = [int(value) for value in tensor.shape]
                    logical = aggregate._logical_bytes(tensor)
                    total_logical_bytes += logical
                    zero_fill = contract is not None and field in aggregate.ZERO_FILL_CANDIDATE_FIELDS
                    if zero_fill:
                        try:
                            nonzero = int(torch.count_nonzero(tensor.detach()).item())
                        except (RuntimeError, TypeError, ValueError) as exc:
                            raise ACSNativeDetachedSlotError(
                                f"cannot prove detached zero-fill field {field} for {cell_id}"
                            ) from exc
                        if nonzero != 0:
                            raise ACSNativeDetachedSlotError(
                                f"detached zero-fill field is nonzero: {field}/{cell_id}"
                            )
                        zero_fill_logical_bytes += logical
                        zero_digest = aggregate._zero_digest(logical)
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
                                "zero_fill_descriptor": aggregate._zero_fill_descriptor(
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
                        sink = aggregate._AggregateFieldSink(
                            spool_stream,
                            offset=spool_stream.tell(),
                            payload_digest=spool_digest,
                        )
                        producer_event = _record_producer_event_v1(
                            tensor, dedicated_stream
                        )
                        copied_chunks = aggregate._copy_tensor(
                            tensor,
                            sink,
                            chunk_bytes=chunk_bytes,
                            fallback_pool=fallback_pool,
                            native_pool=native_pool,
                            dedicated_stream=dedicated_stream,
                            producer_event=producer_event,
                        )
                        expected_chunks = (
                            (logical + chunk_bytes - 1) // chunk_bytes if logical else 0
                        )
                        if sink.bytes != logical or copied_chunks != expected_chunks:
                            raise ACSNativeDetachedSlotError(
                                f"detached cursor field coverage differs: {field}/{cell_id}"
                            )
                        rows.append(
                            {
                                "field": field,
                                "storage": "spool",
                                "spool_offset": spool_stream.tell() - sink.bytes,
                                "bytes": sink.bytes,
                                "logical_bytes": logical,
                                "dtype": tensor_dtype,
                                "shape": tensor_shape,
                                "sha256": sink.digest.hexdigest(),
                                "spool_chunks": list(sink.chunks),
                            }
                        )
                metadata_raw = aggregate._metadata_record_bytes(
                    projection=non_tensors,
                    identity=dict(identity),
                    observer=dict(observer),
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
                for raw_row in detached.compact_result_rows:
                    if not isinstance(raw_row, Mapping):
                        raise ACSNativeDetachedSlotError(
                            f"detached cursor compact metric row is not a mapping: {cell_id}"
                        )
                    _assert_no_tensors(raw_row)
                    _assert_no_callables(raw_row)
                    compact_result_rows.append(dict(raw_row))
                cursor.release_cell(cell_id)
                expected_index += 1
                del tensors, non_tensors, projection, authority, detached
            if expected_index != len(normalized_cell_ids):
                raise ACSNativeDetachedSlotError(
                    "detached cursor capture ended before every cell"
                )
            cursor_receipt = cursor.close()
            if getattr(cursor_receipt, "state", None) != "SEALED":
                raise ACSNativeDetachedSlotError(
                    "detached cursor capture cursor did not seal"
                )
            spool_stream.flush()
    except BaseException as exc:
        try:
            cursor.fail(str(exc))
        except Exception:
            pass
        capture_path.unlink(missing_ok=True)
        spool_path.unlink(missing_ok=True)
        raise

    if field_order is None or not cell_field_rows:
        capture_path.unlink(missing_ok=True)
        spool_path.unlink(missing_ok=True)
        raise ACSNativeDetachedSlotError("detached cursor capture produced no cells")
    if any(zero_fill_families) and not all(zero_fill_families):
        capture_path.unlink(missing_ok=True)
        spool_path.unlink(missing_ok=True)
        raise ACSNativeDetachedSlotError(
            "detached cursor cells disagree about zero-fill eligibility"
        )
    if any(zero_fill_families) and any(contract is None for contract in zero_contracts):
        capture_path.unlink(missing_ok=True)
        spool_path.unlink(missing_ok=True)
        raise ACSNativeDetachedSlotError(
            "detached cursor zero-fill contract is missing"
        )
    if any(contract != zero_contracts[0] for contract in zero_contracts[1:]):
        capture_path.unlink(missing_ok=True)
        spool_path.unlink(missing_ok=True)
        raise ACSNativeDetachedSlotError(
            "detached cursor zero-fill contracts differ"
        )
    zero_contract = zero_contracts[0] if zero_contracts else None
    if zero_contract is not None:
        missing = sorted(aggregate.ZERO_FILL_CANDIDATE_FIELDS - set(field_order))
        if missing:
            capture_path.unlink(missing_ok=True)
            spool_path.unlink(missing_ok=True)
            raise ACSNativeDetachedSlotError(
                "detached cursor zero-fill fields are missing: " + ", ".join(missing)
            )

    payload_bytes = 0
    chunk_count = 0
    for field_index, field in enumerate(field_order):
        for cell_index, rows in enumerate(cell_field_rows):
            row = rows[field_index]
            if row["storage"] == "zero_fill":
                final_row = {
                    "field": field,
                    "storage": "zero_fill",
                    "offset": None,
                    "bytes": 0,
                    "logical_bytes": int(row["logical_bytes"]),
                    "dtype": row["dtype"],
                    "shape": list(row["shape"]),
                    "sha256": row["sha256"],
                    "chunks": [],
                    "zero_fill_contract": "fixed_mlp_exact_zero_v1",
                    "zero_fill_descriptor": aggregate._zero_fill_descriptor(
                        field=field,
                        dtype=str(row["dtype"]),
                        shape=row["shape"],
                        field_index=field_index,
                        cell_index=cell_index,
                        logical_bytes=int(row["logical_bytes"]),
                        logical_digest_sha256=str(row["sha256"]),
                    ),
                }
            else:
                final_chunks: list[dict[str, Any]] = []
                field_offset = payload_bytes
                relative_offset = 0
                for descriptor in row["spool_chunks"]:
                    length = int(descriptor["bytes"])
                    final_chunks.append(
                        {
                            "offset": field_offset + relative_offset,
                            "bytes": length,
                            "sha256": descriptor["sha256"],
                        }
                    )
                    relative_offset += length
                if relative_offset != int(row["bytes"]):
                    spool_path.unlink(missing_ok=True)
                    raise ACSNativeDetachedSlotError(
                        f"detached cursor field spool coverage differs: {field}/{cell_index}"
                    )
                final_row = {
                    "field": field,
                    "storage": "payload",
                    "offset": field_offset,
                    "bytes": relative_offset,
                    "logical_bytes": int(row["logical_bytes"]),
                    "dtype": row["dtype"],
                    "shape": list(row["shape"]),
                    "sha256": row["sha256"],
                    "chunks": final_chunks,
                    "zero_fill_contract": None,
                    "zero_fill_descriptor": None,
                }
                payload_bytes += relative_offset
                chunk_count += len(final_chunks)
            cells_header[cell_index]["fields"].append(final_row)
    if len(metadata_spool_refs) != len(normalized_cell_ids):
        spool_path.unlink(missing_ok=True)
        raise ACSNativeDetachedSlotError(
            "detached cursor metadata index does not cover every cell"
        )
    metadata_offset = payload_bytes
    metadata_bytes = 0
    for cell_index, spool_ref in enumerate(metadata_spool_refs):
        length = int(spool_ref["bytes"])
        cells_header[cell_index]["metadata_ref"] = {
            "offset": metadata_offset + metadata_bytes,
            "bytes": length,
            "sha256": spool_ref["sha256"],
        }
        metadata_bytes += length
    if metadata_bytes > aggregate.MAX_METADATA_BYTES:
        spool_path.unlink(missing_ok=True)
        raise ACSNativeDetachedSlotError(
            "detached cursor metadata payload exceeds its bound"
        )
    file_payload_bytes = payload_bytes + metadata_bytes
    header = {
        "schema_version": aggregate.SCHEMA_VERSION,
        "format_version": aggregate.FORMAT_VERSION,
        "endianness": "little",
        "payload_layout": aggregate.METADATA_PAYLOAD_LAYOUT,
        "metadata_schema_version": aggregate.METADATA_SCHEMA_VERSION,
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
            and field in aggregate.ZERO_FILL_CANDIDATE_FIELDS
        ),
        "cursor_receipt": (
            cursor_receipt.payload() if hasattr(cursor_receipt, "payload") else None
        ),
    }
    if len(_canonical_json(header)) > aggregate.MAX_HEADER_BYTES:
        spool_path.unlink(missing_ok=True)
        raise ACSNativeDetachedSlotError(
            "detached cursor capture header exceeds its bound"
        )

    if native_pool is not None:
        native_pool_telemetry = dict(native_pool.telemetry())
    else:
        native_pool_telemetry = {
            "schema_version": "eiel_acs_native_slab_pool_fallback_v1",
            "chunk_bytes": chunk_bytes,
            "slab_count": 2,
            "pinned": False,
            "preallocated": False,
        }
    spool_identity = _spool_file_identity_v1(spool_path)
    capture = {
        "schema_version": DETACHED_SPOOL_CAPTURE_SCHEMA_VERSION_V2,
        "spool_path": str(spool_path),
        "spool_bytes": spool_path.stat().st_size,
        "materialization_path": str(capture_path),
        "destination": str(target),
        "header": header,
        "cell_ids": normalized_cell_ids,
        "batch_id": batch_id,
        "generation": generation,
        "tick": tick,
        "absolute_next_tick": absolute_next_tick,
        "segment_ordinal": segment_ordinal,
        "source_projection_sha256s": dict(source_projection_sha256s),
        "native_slab_pool": native_pool_telemetry,
        "capture_seconds": time.perf_counter() - started,
        "producer_spool_started_perf_ns": started_perf_ns,
        "producer_spool_completed_perf_ns": time.perf_counter_ns(),
        "capture_cpu_seconds": (
            time.process_time_ns() - started_cpu
        )
        / 1_000_000_000.0,
        "payload_bytes": payload_bytes,
        "metadata_bytes": metadata_bytes,
        "file_payload_bytes": file_payload_bytes,
        "logical_bytes": total_logical_bytes,
        "zero_fill_logical_bytes": zero_fill_logical_bytes,
        "chunk_count": chunk_count,
        "spool_index": {
            "cell_field_rows": cell_field_rows,
            "metadata_spool_refs": metadata_spool_refs,
        },
        "cursor_receipt": (
            cursor_receipt.payload() if hasattr(cursor_receipt, "payload") else None
        ),
        "capture_pool_count": 1,
        "capture_pool_model": "one_shared_two_slab_pool_not_per_slot",
    }
    capture["spool_seal"] = {
        "schema_version": DETACHED_SPOOL_SEAL_SCHEMA_VERSION_V1,
        "file_identity": spool_identity,
        "descriptor_sha256": _spool_descriptor_digest_v1(capture),
        "byte_coverage": "exact_contiguous_cell_major_v1",
    }
    _validate_spool_descriptor_v1(capture)
    _assert_no_tensors(capture)
    _assert_no_callables(capture)
    return {
        "capture": capture,
        "authorities_by_cell": authorities_by_cell,
        "source_projection_sha256s": source_projection_sha256s,
        "cursor_receipt": capture["cursor_receipt"],
        # Compact terminal rows are scalar producer output.  Keep them in the
        # producer-side return envelope so the detached worker receives only
        # the file-backed capture, while the provider can publish the exact
        # terminal metric contract after the cursor has sealed.
        "compact_result_rows": tuple(compact_result_rows),
    }


def _materialize_cursor_spool_v2(
    capture: Mapping[str, Any],
    *,
    cancel_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Convert a host/file-only cell-major spool on the detached worker."""

    _raise_if_cancelled_v1(cancel_event)
    _assert_no_tensors(capture)
    _assert_no_callables(capture)
    spool_path, header, spool_index = _validate_spool_descriptor_v1(capture)
    spool_identity_before = _spool_file_identity_v1(spool_path)
    capture_path = Path(str(capture.get("materialization_path") or "")).resolve()
    expected_spool_bytes = capture.get("spool_bytes")
    if (
        capture_path == spool_path
        or type(expected_spool_bytes) is not int
        or expected_spool_bytes < 0
        or spool_path.stat().st_size != expected_spool_bytes
    ):
        raise ACSNativeDetachedSlotError(
            "detached cursor spool descriptor is incomplete"
        )
    field_order = header.get("field_order")
    cells = header.get("cells")
    cell_field_rows = spool_index.get("cell_field_rows")
    metadata_spool_refs = spool_index.get("metadata_spool_refs")
    if (
        not isinstance(field_order, list)
        or not field_order
        or not isinstance(cells, list)
        or not isinstance(cell_field_rows, list)
        or len(cell_field_rows) != len(cells)
        or not isinstance(metadata_spool_refs, list)
        or len(metadata_spool_refs) != len(cells)
    ):
        raise ACSNativeDetachedSlotError("detached cursor spool index is malformed")

    started = time.perf_counter()
    written = 0
    try:
        with spool_path.open("rb") as spool_stream, capture_path.open("xb") as output:
            for field_index, field in enumerate(field_order):
                _raise_if_cancelled_v1(cancel_event)
                for cell_index, rows in enumerate(cell_field_rows):
                    if not isinstance(rows, list) or len(rows) != len(field_order):
                        raise ACSNativeDetachedSlotError(
                            "detached cursor spool field index is malformed"
                        )
                    source_row = rows[field_index]
                    final_fields = cells[cell_index].get("fields")
                    if (
                        not isinstance(source_row, Mapping)
                        or source_row.get("field") != field
                        or not isinstance(final_fields, list)
                        or len(final_fields) != len(field_order)
                    ):
                        raise ACSNativeDetachedSlotError(
                            "detached cursor spool field identity differs"
                        )
                    final_row = final_fields[field_index]
                    if not isinstance(final_row, Mapping):
                        raise ACSNativeDetachedSlotError(
                            "detached cursor final field index is malformed"
                        )
                    if source_row.get("storage") == "zero_fill":
                        if final_row.get("storage") != "zero_fill":
                            raise ACSNativeDetachedSlotError(
                                "detached cursor zero-fill storage differs"
                            )
                        continue
                    descriptors = source_row.get("spool_chunks")
                    if (
                        source_row.get("storage") != "spool"
                        or final_row.get("storage") != "payload"
                        or not isinstance(descriptors, list)
                    ):
                        raise ACSNativeDetachedSlotError(
                            "detached cursor payload storage differs"
                        )
                    field_digest = hashlib.sha256()
                    field_written = 0
                    for descriptor in descriptors:
                        _raise_if_cancelled_v1(cancel_event)
                        if not isinstance(descriptor, Mapping):
                            raise ACSNativeDetachedSlotError(
                                "detached cursor spool chunk is malformed"
                            )
                        offset = descriptor.get("offset")
                        length = descriptor.get("bytes")
                        expected_sha256 = descriptor.get("sha256")
                        if (
                            type(offset) is not int
                            or offset < 0
                            or type(length) is not int
                            or length < 0
                            or not aggregate._is_sha256(expected_sha256)
                        ):
                            raise ACSNativeDetachedSlotError(
                                "detached cursor spool chunk range is invalid"
                            )
                        spool_stream.seek(offset)
                        raw = spool_stream.read(length)
                        if len(raw) != length or _sha256(raw) != expected_sha256:
                            raise ACSNativeDetachedSlotError(
                                "detached cursor spool chunk is truncated or corrupted"
                            )
                        output.write(raw)
                        field_digest.update(raw)
                        field_written += length
                        written += length
                    if (
                        field_written != source_row.get("bytes")
                        or field_written != final_row.get("bytes")
                        or field_digest.hexdigest() != source_row.get("sha256")
                        or field_digest.hexdigest() != final_row.get("sha256")
                    ):
                        raise ACSNativeDetachedSlotError(
                            "detached cursor spool field digest differs"
                        )
            for cell_index, spool_ref in enumerate(metadata_spool_refs):
                _raise_if_cancelled_v1(cancel_event)
                if not isinstance(spool_ref, Mapping):
                    raise ACSNativeDetachedSlotError(
                        "detached cursor metadata spool reference is malformed"
                    )
                offset = spool_ref.get("spool_offset")
                length = spool_ref.get("bytes")
                expected_sha256 = spool_ref.get("sha256")
                metadata_ref = cells[cell_index].get("metadata_ref")
                if (
                    type(offset) is not int
                    or offset < 0
                    or type(length) is not int
                    or length <= 0
                    or not aggregate._is_sha256(expected_sha256)
                    or not isinstance(metadata_ref, Mapping)
                    or metadata_ref.get("bytes") != length
                    or metadata_ref.get("sha256") != expected_sha256
                ):
                    raise ACSNativeDetachedSlotError(
                        "detached cursor metadata spool range is invalid"
                    )
                spool_stream.seek(offset)
                raw = spool_stream.read(length)
                if len(raw) != length or _sha256(raw) != expected_sha256:
                    raise ACSNativeDetachedSlotError(
                        "detached cursor metadata spool is truncated or corrupted"
                    )
                output.write(raw)
                written += length
            if written != header.get("file_payload_bytes"):
                raise ACSNativeDetachedSlotError(
                    "detached cursor materialized payload size differs"
                )
            output.flush()
            _raise_if_cancelled_v1(cancel_event)
    except BaseException:
        capture_path.unlink(missing_ok=True)
        raise
    _raise_if_cancelled_v1(cancel_event)
    if _spool_file_identity_v1(spool_path) != spool_identity_before:
        capture_path.unlink(missing_ok=True)
        raise ACSNativeDetachedSlotError(
            "detached cursor spool changed during worker materialization"
        )
    if capture.get("source_consumer_required", False) is False:
        spool_path.unlink()

    materialized = {
        key: value
        for key, value in capture.items()
        if key
        not in {
            "materialization_path",
            "spool_bytes",
            "spool_index",
        }
    }
    materialized.update(
        {
            "schema_version": DETACHED_CAPTURE_SCHEMA_VERSION_V1,
            "capture_path": str(capture_path),
            "detached_spool_materialization_seconds": time.perf_counter()
            - started,
        }
    )
    _assert_no_tensors(materialized)
    _assert_no_callables(materialized)
    return materialized


def _physical_batch_checkpoint_coordinates_v1(
    capture: Mapping[str, Any],
    recipe: Mapping[str, Any],
) -> tuple[int, int, int | None]:
    """Resolve private projector coordinates without restoring public authority.

    Current tick-native captures expose only ``absolute_next_tick``.  The
    detached BundleV2 compatibility factory still needs a private
    generation/tick pair, so derive it at this boundary from the absolute
    frontier and the declared source-window width.  Optional compatibility
    metadata may corroborate the derivation, but can never supply it.
    """

    capture_time_authority = capture.get("time_authority")
    recipe_time_authority = recipe.get("time_authority")
    tick_native = (
        capture_time_authority == "absolute_tick_v1"
        or recipe_time_authority == "absolute_tick_v1"
    )
    if not tick_native and (
        capture_time_authority is not None or recipe_time_authority is not None
    ):
        raise ACSNativeDetachedSlotError(
            "detached physical checkpoint time authority is invalid"
        )
    if tick_native:
        if (
            capture_time_authority != "absolute_tick_v1"
            or recipe_time_authority != "absolute_tick_v1"
        ):
            raise ACSNativeDetachedSlotError(
                "detached physical checkpoint absolute-tick authority differs"
            )
        if any(name in capture or name in recipe for name in ("generation", "tick")):
            raise ACSNativeDetachedSlotError(
                "detached physical checkpoint public authority leaked legacy coordinates"
            )
        absolute_next_tick = capture.get("absolute_next_tick")
        authority = recipe.get("absolute_tick_contract")
        if (
            type(absolute_next_tick) is not int
            or absolute_next_tick < 0
            or recipe.get("absolute_next_tick") != absolute_next_tick
            or not isinstance(authority, Mapping)
            or authority.get("schema_version")
            != "eiel_absolute_tick_physical_capture_contract_v1"
            or authority.get("time_authority") != "absolute_tick_v1"
            or authority.get("segment_end_next_tick") != absolute_next_tick
        ):
            raise ACSNativeDetachedSlotError(
                "detached physical checkpoint absolute-tick coordinates are invalid"
            )
        source_window_ticks = authority.get("source_window_ticks")
        if type(source_window_ticks) is not int or source_window_ticks < 1:
            raise ACSNativeDetachedSlotError(
                "detached physical checkpoint source-window authority is invalid"
            )
        generation, tick = divmod(absolute_next_tick, source_window_ticks)
        compatibility = recipe.get("legacy_compatibility")
        if compatibility is not None:
            if (
                not isinstance(compatibility, Mapping)
                or compatibility.get("schema_version")
                != "eiel_absolute_tick_private_settlement_adapter_v1"
                or compatibility.get("authority")
                != "private_legacy_compatibility_only"
                or compatibility.get("derived_from_absolute_next_tick")
                not in (None, absolute_next_tick)
            ):
                raise ACSNativeDetachedSlotError(
                    "detached physical checkpoint compatibility authority is invalid"
                )
            provider_generation = compatibility.get("provider_generation")
            provider_tick = compatibility.get("provider_tick")
            if (provider_generation is None) != (provider_tick is None):
                raise ACSNativeDetachedSlotError(
                    "detached physical checkpoint compatibility coordinates are incomplete"
                )
            if provider_generation is not None and (
                type(provider_generation) is not int
                or type(provider_tick) is not int
                or provider_generation != generation
                or provider_tick != tick
            ):
                raise ACSNativeDetachedSlotError(
                    "detached physical checkpoint compatibility coordinates contradict absolute ticks"
                )
        return generation, tick, absolute_next_tick

    generation = capture.get("generation")
    tick = capture.get("tick")
    absolute_next_tick = capture.get("absolute_next_tick")
    if (
        type(generation) is not int
        or generation < 0
        or type(tick) is not int
        or tick < 0
        or (
            absolute_next_tick is not None
            and (type(absolute_next_tick) is not int or absolute_next_tick < 0)
        )
        or ("generation" in recipe and recipe.get("generation") != generation)
        or ("tick" in recipe and recipe.get("tick") != tick)
    ):
        raise ACSNativeDetachedSlotError(
            "detached physical checkpoint legacy coordinates are invalid"
        )
    return generation, tick, absolute_next_tick


def _materialize_physical_batch_capture_v1(
    capture: Mapping[str, Any],
    *,
    cancel_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Settle one file-backed physical B136 snapshot on the host worker."""

    from engine.campaigns.acs_direct_resident_physical_capture_v1 import (
        SCHEMA_VERSION as PHYSICAL_CAPTURE_SCHEMA_VERSION,
    )
    from engine.campaigns.acs_physical_batch_packed_checkpoint_v1 import (
        finalize_physical_batch_packed_checkpoint_v1,
        open_physical_batch_packed_checkpoint_v1,
    )
    from engine.campaigns.providers import (
        direct_resident_detached_checkpoint_authority_factory_v1,
    )

    _raise_if_cancelled_v1(cancel_event)
    _assert_no_tensors(capture)
    _assert_no_callables(capture)
    if capture.get("schema_version") != PHYSICAL_CAPTURE_SCHEMA_VERSION:
        raise ACSNativeDetachedSlotError(
            "detached physical-batch capture schema is invalid"
        )
    if capture.get("source_consumer_required") is not False:
        raise ACSNativeDetachedSlotError(
            "detached physical-batch capture cannot retain a source lease"
        )
    packed_descriptor = capture.get("packed_descriptor")
    recipe = capture.get("settlement_recipe")
    raw_publication_path = capture.get("raw_publication_path")
    if (
        not isinstance(packed_descriptor, Mapping)
        or not isinstance(recipe, Mapping)
        or not isinstance(raw_publication_path, str)
        or not raw_publication_path
        or tuple(capture.get("cell_ids") or ())
        != tuple(recipe.get("cell_ids") or ())
    ):
        raise ACSNativeDetachedSlotError(
            "detached physical-batch capture envelope is incomplete"
        )
    generation, tick, absolute_next_tick = (
        _physical_batch_checkpoint_coordinates_v1(capture, recipe)
    )
    world_sharded = capture.get("world_sharded_checkpoint", False)
    if type(world_sharded) is not bool:
        raise ACSNativeDetachedSlotError(
            "detached physical-batch sharded selection is invalid"
        )
    started = time.perf_counter()
    raw_path = Path(raw_publication_path).expanduser().resolve()
    cursor_capture: Mapping[str, Any] | None = None
    try:
        if world_sharded:
            raw_path = Path(
                str(packed_descriptor.get("capture_path") or "")
            ).expanduser().resolve()
            destination_root = capture.get("world_sharded_destination_root")
            if (
                not isinstance(destination_root, str)
                or not destination_root
                or not raw_path.is_file()
            ):
                raise ACSNativeDetachedSlotError(
                    "world-sharded capture lacks its destination root"
                )
            from engine.campaigns.acs_world_sharded_checkpoint_v1 import (
                write_world_sharded_checkpoint_set_from_packed_v1,
            )

            sharded = write_world_sharded_checkpoint_set_from_packed_v1(
                raw_path,
                destination_root,
                raw_capture_descriptor=packed_descriptor,
                worker_count=int(capture.get("world_sharded_worker_count", 4)),
                required_world_count=len(tuple(capture.get("cell_ids") or ())),
                cancel_event=cancel_event,
            )
            if sharded.get("source_recipe_sha256") != recipe.get("recipe_sha256"):
                raise ACSNativeDetachedSlotError(
                    "world-sharded set recipe differs from its capture envelope"
                )
            materialized = {
                **dict(sharded),
                "source_consumer_required": False,
                "cell_ids": list(capture.get("cell_ids") or ()),
                "batch_id": str(capture.get("batch_id") or ""),
                "generation": generation,
                "tick": tick,
                "absolute_next_tick": absolute_next_tick,
                "segment_ordinal": int(capture.get("segment_ordinal", -1)),
                "physical_batch_raw_payload_bytes": int(
                    packed_descriptor.get("payload_bytes", 0)
                ),
                "physical_batch_source_authority_sha256": str(
                    packed_descriptor.get("capture_authority_sha256") or ""
                ),
                "physical_batch_raw_artifact_sha256": None,
                "physical_batch_raw_finalize_bypassed": True,
                "physical_batch_settlement_seconds": 0.0,
                "world_sharded_materialization_seconds": (
                    time.perf_counter() - started
                ),
            }
            _assert_no_tensors(materialized)
            _assert_no_callables(materialized)
            raw_path.unlink(missing_ok=True)
            return materialized
        raw_receipt = finalize_physical_batch_packed_checkpoint_v1(
            packed_descriptor,
            publication_path=raw_path,
        )
        _raise_if_cancelled_v1(cancel_event)
        from engine.campaigns.acs_direct_resident_physical_settlement_v1 import (
            build_physical_batch_settlement_cursor_v1,
        )

        with open_physical_batch_packed_checkpoint_v1(raw_path) as reader:
            scalar_metadata = reader.scalar_metadata
            if (
                not isinstance(scalar_metadata, Mapping)
                or scalar_metadata.get("settlement_recipe") != recipe
            ):
                raise ACSNativeDetachedSlotError(
                    "packed physical settlement recipe differs from its envelope"
                )
            cursor = build_physical_batch_settlement_cursor_v1(
                capture,
                reader=reader,
                cancel_event=cancel_event,
            )
            authority_factory = (
                direct_resident_detached_checkpoint_authority_factory_v1(
                    provider_id=str(recipe.get("provider_id") or ""),
                    batch_id=str(capture.get("batch_id") or ""),
                    cell_ids=tuple(capture.get("cell_ids") or ()),
                    generation=generation,
                    tick=tick,
                    absolute_next_tick=absolute_next_tick,
                    segment_ordinal=int(capture.get("segment_ordinal", -1)),
                )
            )
            cursor_capture = capture_acs_native_aggregate_payload_from_cursor_v1(
                cursor,
                str(capture.get("destination") or ""),
                cell_ids=tuple(capture.get("cell_ids") or ()),
                generation=generation,
                tick=tick,
                absolute_next_tick=absolute_next_tick,
                batch_id=str(capture.get("batch_id") or ""),
                segment_ordinal=int(capture.get("segment_ordinal", -1)),
                authority_factory=authority_factory,
                require_native_extension=bool(
                    capture.get("require_native_extension", True)
                ),
            )
        _raise_if_cancelled_v1(cancel_event)
    except BaseException:
        _quarantine_capture_file_v1(
            {
                "capture_path": str(raw_path),
                "source_consumer_required": False,
            }
        )
        raise
    else:
        raw_path.unlink(missing_ok=True)

    if not isinstance(cursor_capture, Mapping):
        raise ACSNativeDetachedSlotError(
            "physical host settlement returned no cursor capture"
        )
    detached = cursor_capture.get("capture")
    authorities = cursor_capture.get("authorities_by_cell")
    source_digests = cursor_capture.get("source_projection_sha256s")
    if (
        not isinstance(detached, Mapping)
        or not isinstance(authorities, Mapping)
        or not isinstance(source_digests, Mapping)
    ):
        raise ACSNativeDetachedSlotError(
            "physical host settlement returned incomplete authorities"
        )
    cell_ids = tuple(capture.get("cell_ids") or ())
    if set(authorities) != set(cell_ids) or set(source_digests) != set(cell_ids):
        raise ACSNativeDetachedSlotError(
            "physical host settlement authorities do not cover B136"
        )
    detached_authorities = {
        "source_projection_sha256s": {
            cell_id: str(authorities[cell_id]["source_projection_sha256"])
            for cell_id in cell_ids
        },
        "architecture_b_identities": {
            cell_id: dict(authorities[cell_id]["architecture_b_identity"])
            for cell_id in cell_ids
        },
        "observer_continuations": {
            cell_id: dict(authorities[cell_id]["observer_continuation"])
            for cell_id in cell_ids
        },
        "mechanics_checkpoint_digests": {
            cell_id: str(authorities[cell_id]["mechanics_checkpoint_digest"])
            for cell_id in cell_ids
        },
        "canonical_checkpoint_storage_digests": {
            cell_id: str(
                authorities[cell_id]["canonical_checkpoint_storage_digest"]
            )
            for cell_id in cell_ids
        },
        "cuda_source_field_counts": {
            cell_id: int(authorities[cell_id].get("cuda_source_field_count", 0))
            for cell_id in cell_ids
        },
    }
    materialized = dict(detached)
    materialized.update(
        {
            "source_consumer_required": False,
            "detached_authorities": detached_authorities,
            "physical_compact_result_rows": [
                dict(row)
                for row in cursor_capture.get("compact_result_rows", ())
                if isinstance(row, Mapping)
            ],
            "physical_batch_raw_payload_bytes": int(
                raw_receipt.get("payload_bytes", 0)
            ),
            "physical_batch_raw_artifact_sha256": str(
                raw_receipt.get("artifact_sha256") or ""
            ),
            "physical_batch_settlement_seconds": time.perf_counter() - started,
        }
    )
    _assert_no_tensors(materialized)
    _assert_no_callables(materialized)
    return materialized


def _finalize_capture_v1(
    capture: Mapping[str, Any],
    *,
    cancel_event: threading.Event | None = None,
) -> dict[str, Any]:
    _raise_if_cancelled_v1(cancel_event)
    _assert_no_tensors(capture)
    _assert_no_callables(capture)
    from engine.campaigns.cuda_full_evolution_v2_acs_adapter import (
        CUDA_NATIVE_DETACHED_CAPTURE_SCHEMA_V1,
        WORKHORSE_DETACHED_CAPTURE_SCHEMA_V1,
        finalize_native_checkpoint_capture_v1,
        finalize_workhorse_checkpoint_capture_v1,
    )

    if capture.get("schema_version") == WORKHORSE_DETACHED_CAPTURE_SCHEMA_V1:
        return finalize_workhorse_checkpoint_capture_v1(
            capture,
            cancel_event=cancel_event,
        )
    if capture.get("schema_version") == CUDA_NATIVE_DETACHED_CAPTURE_SCHEMA_V1:
        return finalize_native_checkpoint_capture_v1(
            capture,
            cancel_event=cancel_event,
        )
    from engine.campaigns.acs_direct_resident_physical_capture_v1 import (
        SCHEMA_VERSION as PHYSICAL_CAPTURE_SCHEMA_VERSION,
    )

    if capture.get("schema_version") == PHYSICAL_CAPTURE_SCHEMA_VERSION:
        capture = _materialize_physical_batch_capture_v1(
            capture,
            cancel_event=cancel_event,
        )
    from engine.campaigns.acs_world_sharded_checkpoint_v1 import (
        SET_MANIFEST_SCHEMA_VERSION as WORLD_SET_MANIFEST_SCHEMA_VERSION,
    )

    if capture.get("schema_version") == WORLD_SET_MANIFEST_SCHEMA_VERSION:
        if capture.get("status") != "complete":
            raise ACSNativeDetachedSlotError(
                "world-sharded checkpoint set is not complete"
            )
        _assert_no_tensors(capture)
        _assert_no_callables(capture)
        return dict(capture)
    if capture.get("schema_version") == DETACHED_SPOOL_CAPTURE_SCHEMA_VERSION_V2:
        capture = _materialize_cursor_spool_v2(
            capture,
            cancel_event=cancel_event,
        )
    _raise_if_cancelled_v1(cancel_event)
    if capture.get("schema_version") not in {None, DETACHED_CAPTURE_SCHEMA_VERSION_V1}:
        raise ACSNativeDetachedSlotError("detached capture schema is invalid")
    capture_path = Path(str(capture.get("capture_path") or "")).resolve()
    destination = Path(str(capture.get("destination") or "")).resolve()
    header = capture.get("header")
    cell_ids = tuple(capture.get("cell_ids") or ())
    if not capture_path.is_file() or not isinstance(header, Mapping) or not cell_ids:
        raise ACSNativeDetachedSlotError("detached capture receipt is incomplete")
    payload_bytes_expected = capture.get(
        "file_payload_bytes", capture.get("payload_bytes")
    )
    if type(payload_bytes_expected) is not int or payload_bytes_expected < 0:
        raise ACSNativeDetachedSlotError("detached capture payload bytes are invalid")
    header_bytes = _canonical_json(header)
    payload_sha256, payload_bytes = _file_sha256(capture_path)
    if payload_bytes != payload_bytes_expected:
        raise ACSNativeDetachedSlotError("detached capture payload size changed")
    if header.get("payload_layout") == aggregate.METADATA_PAYLOAD_LAYOUT:
        if header.get("file_payload_bytes") != payload_bytes:
            raise ACSNativeDetachedSlotError(
                "detached aggregate header file payload size changed"
            )
        if header.get("metadata_offset") != header.get("payload_bytes"):
            raise ACSNativeDetachedSlotError(
                "detached aggregate metadata offset is inconsistent"
            )
        if (
            type(header.get("metadata_bytes")) is not int
            or header["metadata_bytes"] <= 0
            or header["metadata_bytes"] > aggregate.MAX_METADATA_BYTES
            or header["payload_bytes"] + header["metadata_bytes"] != payload_bytes
        ):
            raise ACSNativeDetachedSlotError(
                "detached aggregate metadata payload size is inconsistent"
            )
    header_sha256 = _sha256(header_bytes)
    content_sha256 = aggregate._content_digest(
        header_sha256=header_sha256,
        payload_sha256=payload_sha256,
    )
    provisional_prefix = aggregate.PREFIX.pack(
        aggregate.MAGIC,
        aggregate.FORMAT_VERSION,
        aggregate.LITTLE_ENDIAN_MARKER,
        len(header_bytes),
        payload_bytes,
        0,
    )
    provisional_footer = aggregate._footer_without_artifact(
        prefix=provisional_prefix,
        header_bytes=header_bytes,
        payload_sha256=payload_sha256,
        content_sha256=content_sha256,
        batch_id=str(capture["batch_id"]),
        cell_ids=cell_ids,
        payload_bytes=payload_bytes,
    )
    footer_length = len(
        _canonical_json({**provisional_footer, "artifact_sha256": "0" * 64})
    )
    prefix = aggregate.PREFIX.pack(
        aggregate.MAGIC,
        aggregate.FORMAT_VERSION,
        aggregate.LITTLE_ENDIAN_MARKER,
        len(header_bytes),
        payload_bytes,
        footer_length,
    )
    footer_without_artifact = aggregate._footer_without_artifact(
        prefix=prefix,
        header_bytes=header_bytes,
        payload_sha256=payload_sha256,
        content_sha256=content_sha256,
        batch_id=str(capture["batch_id"]),
        cell_ids=cell_ids,
        payload_bytes=payload_bytes,
    )
    footer = {
        **footer_without_artifact,
        "artifact_sha256": aggregate._artifact_digest(
            prefix=prefix,
            header_sha256=header_sha256,
            payload_sha256=payload_sha256,
            content_sha256=content_sha256,
            footer_without_artifact=footer_without_artifact,
        ),
    }
    footer_bytes = _canonical_json(footer)
    if len(footer_bytes) != footer_length or len(footer_bytes) > aggregate.MAX_FOOTER_BYTES:
        raise ACSNativeDetachedSlotError("detached capture footer length did not converge")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid4().hex}.finalize.tmp"
    )
    started = time.perf_counter()
    publication_created = False
    try:
        with temporary.open("xb") as output:
            output.write(prefix)
            with capture_path.open("rb") as payload:
                for chunk in iter(
                    lambda: payload.read(DETACHED_COPY_BUFFER_BYTES_V1), b""
                ):
                    _raise_if_cancelled_v1(cancel_event)
                    output.write(chunk)
            output.write(header_bytes)
            output.write(footer_bytes)
            output.flush()
            os.fsync(output.fileno())
        _raise_if_cancelled_v1(cancel_event)
        try:
            os.link(temporary, destination)
        except FileExistsError:
            existing = aggregate.load_acs_native_aggregate_checkpoint(
                destination, cell_id=cell_ids[0]
            )
            if existing["footer"].get("artifact_sha256") != footer["artifact_sha256"]:
                raise ACSNativeDetachedSlotError(
                    "conflicting detached aggregate publication already exists"
                )
        else:
            publication_created = True
            _raise_if_cancelled_v1(cancel_event)
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        _raise_if_cancelled_v1(cancel_event)
    except BaseException:
        temporary.unlink(missing_ok=True)
        if publication_created:
            _quarantine_owned_publication_v1(destination)
        _quarantine_capture_file_v1(
            {
                **dict(capture),
                "source_consumer_required": False,
                "spool_path": None,
                "materialization_path": None,
            }
        )
        raise
    else:
        temporary.unlink(missing_ok=True)
        capture_path.unlink(missing_ok=True)

    cells = header.get("cells")
    if not isinstance(cells, list) or len(cells) != len(cell_ids):
        raise ACSNativeDetachedSlotError("detached aggregate header cell index is invalid")
    cell_receipts: dict[str, dict[str, Any]] = {}
    for cell in cells:
        if not isinstance(cell, Mapping) or not isinstance(cell.get("cell_id"), str):
            raise ACSNativeDetachedSlotError("detached aggregate cell receipt is invalid")
        fields = cell.get("fields")
        if not isinstance(fields, list):
            raise ACSNativeDetachedSlotError("detached aggregate field index is invalid")
        physical = sum(
            int(row.get("bytes", 0))
            for row in fields
            if isinstance(row, Mapping) and row.get("storage") == "payload"
        )
        logical = sum(
            int(row.get("logical_bytes", 0))
            for row in fields
            if isinstance(row, Mapping)
        )
        zero_logical = sum(
            int(row.get("logical_bytes", 0))
            for row in fields
            if isinstance(row, Mapping) and row.get("storage") == "zero_fill"
        )
        cell_id = str(cell["cell_id"])
        cell_receipts[cell_id] = {
            "cell_id": cell_id,
            "source_projection_sha256": str(cell["source_projection_sha256"]),
            "logical_bytes": logical,
            "physical_bytes": physical,
            "zero_fill_logical_bytes": zero_logical,
            "chunk_count": sum(
                len(row.get("chunks", ()))
                for row in fields
                if isinstance(row, Mapping)
            ),
            "field_count": len(fields),
        }
    header_payload_bytes = header.get("payload_bytes")
    header_metadata_bytes = header.get("metadata_bytes", 0)
    header_file_payload_bytes = header.get("file_payload_bytes", payload_bytes)
    if (
        type(header_payload_bytes) is not int
        or header_payload_bytes < 0
        or type(header_metadata_bytes) is not int
        or header_metadata_bytes < 0
        or type(header_file_payload_bytes) is not int
        or header_file_payload_bytes != payload_bytes
    ):
        raise ACSNativeDetachedSlotError(
            "detached aggregate payload accounting is malformed"
        )
    return {
        "schema_version": aggregate.SCHEMA_VERSION,
        "status": "complete",
        "path": str(destination),
        "batch_id": str(capture["batch_id"]),
        "generation": int(capture["generation"]),
        "tick": int(capture["tick"]),
        "absolute_next_tick": capture.get("absolute_next_tick"),
        "segment_ordinal": int(capture["segment_ordinal"]),
        "header_sha256": footer["header_sha256"],
        "payload_sha256": footer["payload_sha256"],
        "content_sha256": footer["content_sha256"],
        "seal_sha256": footer["seal_sha256"],
        "artifact_sha256": footer["artifact_sha256"],
        "header_bytes": len(header_bytes),
        "payload_bytes": header_payload_bytes,
        "metadata_bytes": header_metadata_bytes,
        "file_payload_bytes": header_file_payload_bytes,
        "logical_bytes": int(capture["logical_bytes"]),
        "zero_fill_logical_bytes": int(capture["zero_fill_logical_bytes"]),
        "chunk_count": int(capture["chunk_count"]),
        "cell_count": len(cell_ids),
        "cells": cell_receipts,
        "zero_fill_fields": list(header.get("zero_fill_fields", [])),
        "native_slab_pool": dict(capture.get("native_slab_pool") or {}),
        "physical_file_bytes": destination.stat().st_size,
        "detached_finalize_seconds": time.perf_counter() - started,
        "detached_capture_seconds": float(capture.get("capture_seconds", 0.0)),
        "detached_spool_materialization_seconds": float(
            capture.get("detached_spool_materialization_seconds", 0.0)
        ),
        "detached_capture_path": str(capture_path),
        "detached_authorities": (
            dict(capture["detached_authorities"])
            if isinstance(capture.get("detached_authorities"), Mapping)
            else None
        ),
        "physical_compact_result_rows": [
            dict(row)
            for row in capture.get("physical_compact_result_rows", ())
            if isinstance(row, Mapping)
        ],
        "physical_batch_raw_payload_bytes": int(
            capture.get("physical_batch_raw_payload_bytes", 0)
        ),
        "physical_batch_raw_artifact_sha256": capture.get(
            "physical_batch_raw_artifact_sha256"
        ),
        "physical_batch_settlement_seconds": float(
            capture.get("physical_batch_settlement_seconds", 0.0)
        ),
        "_detached_publication_created_by_worker": publication_created,
    }


def finalize_direct_resident_physical_capture_v1(
    capture: Mapping[str, Any],
    *,
    cancel_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Finalize one tensor-free physical capture on the calling host thread.

    The producer-side capture and owner release are intentionally separate from
    this host-only step.  Keep the timing fields identical to the detached-slot
    worker so the provider can prove that persistence began only after the GPU
    owner was released.
    """

    writer_started_perf_ns = time.perf_counter_ns()
    writer_started_cpu_ns = time.thread_time_ns()
    result = _finalize_capture_v1(capture, cancel_event=cancel_event)
    writer_finished_perf_ns = time.perf_counter_ns()
    return {
        **dict(result),
        "detached_writer_started_perf_ns": writer_started_perf_ns,
        "detached_writer_finished_perf_ns": writer_finished_perf_ns,
        "detached_writer_cpu_time_ns": (
            time.thread_time_ns() - writer_started_cpu_ns
        ),
        "detached_writer_wall_time_ns": (
            writer_finished_perf_ns - writer_started_perf_ns
        ),
    }


@dataclass(frozen=True, slots=True)
class ACSNativeDetachedSlotTokenV1:
    sequence: int
    compute_frontier: int

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": DETACHED_SLOT_TOKEN_SCHEMA_VERSION_V1,
            "sequence": self.sequence,
            "compute_frontier": self.compute_frontier,
        }


@dataclass(slots=True)
class _DetachedSlotJobV1:
    token: ACSNativeDetachedSlotTokenV1
    state: str = "CAPTURING"
    capture: Mapping[str, Any] | None = None
    result: Mapping[str, Any] | None = None
    failure: BaseException | None = None
    failure_capture_path: str | None = None
    consumed: bool = False
    captured_bytes: int = 0
    enqueued_ns: int = 0
    submitted_ns: int | None = None
    started_ns: int | None = None
    finished_ns: int | None = None
    source_consumer_required: bool = False
    source_released: bool = True
    source_released_ns: int | None = None
    cancel_event: threading.Event = dataclass_field(default_factory=threading.Event)
    publication_path: str | None = None
    publication_created: bool = False


@dataclass(slots=True)
class _DetachedSourceLeaseV1:
    token: ACSNativeDetachedSlotTokenV1
    capture: Mapping[str, Any]
    detached_ns: int


class ACSNativeDetachedSlotsV1:
    """One FIFO worker with one or two detached host/file-only slots."""

    def __init__(self, *, slots: int, clock_ns: Any = time.perf_counter_ns) -> None:
        if type(slots) is not int or slots not in DETACHED_ASYNC_SLOT_CAPACITIES_V1:
            raise ACSNativeDetachedSlotError("detached ACS slots must be exactly one or two")
        self.slots = slots
        self._clock_ns = clock_ns
        self._condition = threading.Condition()
        self._jobs: deque[_DetachedSlotJobV1] = deque()
        self._by_sequence: dict[int, _DetachedSlotJobV1] = {}
        self._source_leases: dict[int, _DetachedSourceLeaseV1] = {}
        self._next_sequence = 0
        self._compute_frontier = -1
        self._durable_checkpoint_frontier = -1
        self._closed = False
        self._failure: BaseException | None = None
        self._producer_blocked_ns = 0
        self._writer_busy_ns = 0
        self._occupancy_high_water = 0
        self._temporary_capture_file_high_water = 0
        self._abandoned_snapshot_count = 0
        self._aborted = False
        self._failure_capture_paths: list[str] = []
        self._worker = threading.Thread(
            target=self._worker_main,
            name="eiel-acs-native-detached-writer",
            daemon=True,
        )
        self._worker.start()

    def begin_capture(self, *, compute_frontier: int) -> ACSNativeDetachedSlotTokenV1:
        if type(compute_frontier) is not int or compute_frontier < 0:
            raise ACSNativeDetachedSlotError("compute frontier is invalid")
        blocked_started: int | None = None
        with self._condition:
            while len(self._jobs) >= self.slots:
                self._raise_if_failed_locked()
                if self._closed:
                    raise ACSNativeDetachedSlotError("detached ACS slots are closed")
                if blocked_started is None:
                    blocked_started = self._clock_ns()
                self._condition.wait()
            if blocked_started is not None:
                self._producer_blocked_ns += self._clock_ns() - blocked_started
            self._raise_if_failed_locked()
            if self._closed:
                raise ACSNativeDetachedSlotError("detached ACS slots are closed")
            if compute_frontier <= self._compute_frontier:
                raise ACSNativeDetachedSlotError(
                    "detached ACS compute frontiers must be strictly increasing"
                )
            token = ACSNativeDetachedSlotTokenV1(
                sequence=self._next_sequence,
                compute_frontier=compute_frontier,
            )
            self._next_sequence += 1
            self._compute_frontier = compute_frontier
            job = _DetachedSlotJobV1(
                token=token,
                enqueued_ns=self._clock_ns(),
            )
            self._jobs.append(job)
            self._by_sequence[token.sequence] = job
            self._occupancy_high_water = max(self._occupancy_high_water, len(self._jobs))
            self._condition.notify_all()
            return token

    def submit_capture(
        self,
        token: ACSNativeDetachedSlotTokenV1,
        capture: Mapping[str, Any],
    ) -> None:
        if not isinstance(capture, Mapping):
            raise ACSNativeDetachedSlotError("detached capture is not a mapping")
        source_consumer_required = capture.get("source_consumer_required", False)
        if type(source_consumer_required) is not bool:
            raise ACSNativeDetachedSlotError(
                "detached ACS source-consumer lease flag must be an exact bool"
            )
        if source_consumer_required and capture.get(
            "schema_version"
        ) != DETACHED_SPOOL_CAPTURE_SCHEMA_VERSION_V2:
            raise ACSNativeDetachedSlotError(
                "detached ACS successor-source lease requires a sealed cursor spool"
            )
        try:
            validate_detached_capture_host_only_v1(capture)
        except BaseException:
            _quarantine_capture_file_v1(capture)
            raise
        with self._condition:
            try:
                self._raise_if_failed_locked()
                job = self._job_for_token_locked(token)
                if job.state != "CAPTURING":
                    raise ACSNativeDetachedSlotError(
                        "detached capture was submitted twice"
                    )
                if (
                    (
                        capture.get("capture_path") is None
                        and capture.get("spool_path") is None
                    )
                    or capture.get("destination") is None
                ):
                    raise ACSNativeDetachedSlotError(
                        "detached capture lacks file identity"
                    )
            except BaseException:
                _quarantine_capture_file_v1(capture)
                raise
            # Freeze nested JSON as well as the outer mapping.  Callers retain
            # no mutable authority that can change after worker admission.
            job.capture = json.loads(_canonical_json(capture).decode("utf-8"))
            job.captured_bytes = int(capture.get("logical_bytes", 0))
            job.source_consumer_required = source_consumer_required
            job.source_released = not source_consumer_required
            job.submitted_ns = self._clock_ns()
            job.state = "CAPTURED_HOST_ONLY"
            self._temporary_capture_file_high_water = max(
                self._temporary_capture_file_high_water,
                len(self._source_leases)
                + sum(
                    1
                    for live_job in self._jobs
                    if live_job.capture is not None
                ),
            )
            self._condition.notify_all()

    def fail_capture(
        self,
        token: ACSNativeDetachedSlotTokenV1,
        reason: BaseException | str,
    ) -> None:
        if not isinstance(reason, BaseException):
            reason = ACSNativeDetachedSlotError(str(reason))
        with self._condition:
            job = self._job_for_token_locked(token)
            for live_job in self._jobs:
                live_job.cancel_event.set()
            job.failure = reason
            job.state = "FAILED"
            self._failure = reason
            self._closed = True
            self._condition.notify_all()

    def ready(self, token: ACSNativeDetachedSlotTokenV1) -> bool:
        with self._condition:
            self._raise_if_aborted_locked()
            job = self._job_for_token_locked(token)
            return job.state in {"FAILED", "DURABLE"}

    def release_source(self, token: ACSNativeDetachedSlotTokenV1) -> int:
        """Release only the resident owner's canonical-source spool lease."""

        with self._condition:
            self._raise_if_aborted_locked()
            if not isinstance(token, ACSNativeDetachedSlotTokenV1):
                raise ACSNativeDetachedSlotError(
                    "detached ACS source lease token has the wrong type"
                )
            job = self._by_sequence.get(token.sequence)
            if job is None:
                lease = self._source_leases.get(token.sequence)
                if lease is None or lease.token != token:
                    raise ACSNativeDetachedSlotError(
                        "detached ACS source lease token is unknown"
                    )
                spool_path, _, _ = _validate_spool_descriptor_v1(lease.capture)
                spool_path.unlink()
                released_ns = self._clock_ns()
                self._source_leases.pop(token.sequence)
                self._condition.notify_all()
                return released_ns
            if job.token != token:
                raise ACSNativeDetachedSlotError(
                    "detached ACS source lease token differs"
                )
            if not job.source_consumer_required:
                raise ACSNativeDetachedSlotError(
                    "detached ACS job has no successor-source lease"
                )
            if job.source_released:
                raise ACSNativeDetachedSlotError(
                    "detached ACS successor-source lease was released twice"
                )
            if job.state in {"CAPTURING", "FAILED", "FREE"}:
                raise ACSNativeDetachedSlotError(
                    "detached ACS successor-source lease is not releasable"
                )
            job.source_released = True
            job.source_released_ns = self._clock_ns()
            self._condition.notify_all()
            return job.source_released_ns

    def result(
        self,
        token: ACSNativeDetachedSlotTokenV1,
        *,
        producer_backpressure: bool = False,
    ) -> Mapping[str, Any]:
        with self._condition:
            self._raise_if_aborted_locked()
            job = self._job_for_token_locked(token)
            if not self._jobs or self._jobs[0] is not job:
                raise ACSNativeDetachedSlotError("detached ACS results are FIFO")
            blocked_started = self._clock_ns() if producer_backpressure else None
            while job.state not in {"FAILED", "DURABLE"}:
                self._condition.wait()
                self._raise_if_aborted_locked()
            if blocked_started is not None:
                self._producer_blocked_ns += self._clock_ns() - blocked_started
            if job.failure is not None:
                raise ACSNativeDetachedSlotError(
                    f"detached ACS worker failed: {job.failure}"
                ) from job.failure
            if job.consumed or job.result is None:
                raise ACSNativeDetachedSlotError("detached ACS result was already consumed")
            job.consumed = True
            return {
                **dict(job.result),
                "detached_source_consumer_required": (
                    job.source_consumer_required
                ),
                "detached_source_released_perf_ns": job.source_released_ns,
            }

    def acknowledge_durable(self, token: ACSNativeDetachedSlotTokenV1) -> None:
        with self._condition:
            self._raise_if_aborted_locked()
            job = self._job_for_token_locked(token)
            if not job.consumed or job.state != "DURABLE":
                raise ACSNativeDetachedSlotError(
                    "detached ACS durability must be acknowledged after result consumption"
                )
            if not self._jobs or self._jobs[0] is not job:
                raise ACSNativeDetachedSlotError("detached ACS acknowledgements are FIFO")
            if job.capture is not None and job.capture.get("spool_path") is not None:
                raw_spool_path = Path(str(job.capture["spool_path"])).resolve()
                if job.source_consumer_required and not job.source_released:
                    if not raw_spool_path.exists():
                        raise ACSNativeDetachedSlotError(
                            "detached ACS successor spool disappeared before lease detachment"
                        )
                    _validate_spool_descriptor_v1(job.capture)
                    self._source_leases[token.sequence] = _DetachedSourceLeaseV1(
                        token=token,
                        capture=dict(job.capture),
                        detached_ns=self._clock_ns(),
                    )
                elif raw_spool_path.exists():
                    spool_path, _, _ = _validate_spool_descriptor_v1(job.capture)
                    spool_path.unlink()
                elif job.source_consumer_required:
                    raise ACSNativeDetachedSlotError(
                        "detached ACS successor spool disappeared before acknowledgement"
                    )
            job.state = "FREE"
            self._jobs.popleft()
            self._by_sequence.pop(token.sequence, None)
            self._durable_checkpoint_frontier = token.compute_frontier
            self._condition.notify_all()

    def poll_failure(self) -> None:
        with self._condition:
            self._raise_if_failed_locked()

    def close(self) -> None:
        with self._condition:
            if self._jobs or self._source_leases:
                raise ACSNativeDetachedSlotError(
                    "detached ACS slots cannot close with live jobs or source leases"
                )
            self._closed = True
            self._condition.notify_all()
        self._worker.join()

    def abort(self) -> None:
        with self._condition:
            self._closed = True
            self._aborted = True
            self._abandoned_snapshot_count += len(self._jobs)
            for job in self._jobs:
                job.cancel_event.set()
            self._condition.notify_all()
        self._worker.join()
        with self._condition:
            for job in self._jobs:
                if job.publication_created and job.publication_path is not None:
                    failed_path = _quarantine_owned_publication_v1(
                        job.publication_path
                    )
                    if failed_path is not None:
                        self._failure_capture_paths.append(failed_path)
                    job.publication_created = False
                if job.capture is not None:
                    failed_path = _quarantine_capture_file_v1(job.capture)
                    if failed_path is not None:
                        self._failure_capture_paths.append(failed_path)
            for lease in self._source_leases.values():
                failed_path = _quarantine_capture_file_v1(lease.capture)
                if failed_path is not None:
                    self._failure_capture_paths.append(failed_path)
            self._jobs.clear()
            self._by_sequence.clear()
            self._source_leases.clear()
            self._condition.notify_all()

    def telemetry(self) -> dict[str, Any]:
        with self._condition:
            now = self._clock_ns()
            states = [
                job.state
                for job in list(self._jobs)[: self.slots]
            ]
            states.extend("FREE" for _ in range(self.slots - len(states)))
            return {
                "schema_version": DETACHED_SLOTS_SCHEMA_VERSION_V1,
                "slots": self.slots,
                "writer_count": 1,
                "slot_states": states,
                "queue_occupancy": len(self._jobs),
                "queue_occupancy_high_water": self._occupancy_high_water,
                "captured_bytes_active": sum(job.captured_bytes for job in self._jobs),
                "source_consumer_leases_active": len(self._source_leases) + sum(
                    1
                    for job in self._jobs
                    if job.source_consumer_required and not job.source_released
                ),
                "detached_source_lease_count": len(self._source_leases),
                "source_lease_spool_bytes_active": sum(
                    int(lease.capture.get("spool_bytes", 0))
                    for lease in self._source_leases.values()
                ),
                "temporary_capture_files_active": len(self._source_leases)
                + sum(1 for job in self._jobs if job.capture is not None),
                "temporary_capture_file_high_water": (
                    self._temporary_capture_file_high_water
                ),
                "producer_blocked_time_ns": self._producer_blocked_ns,
                "writer_busy_time_ns": self._writer_busy_ns,
                "backlog_age_ns": max(
                    (now - job.enqueued_ns for job in self._jobs),
                    default=0,
                ),
                "compute_frontier": self._compute_frontier,
                "durable_checkpoint_frontier": self._durable_checkpoint_frontier,
                "frontier_lag": max(
                    0,
                    self._compute_frontier - self._durable_checkpoint_frontier,
                ),
                "failure_latched": self._failure is not None,
                "abandoned_snapshot_count": self._abandoned_snapshot_count,
                "aborted": self._aborted,
                "failure_capture_paths": list(self._failure_capture_paths),
                "closed": self._closed,
                "writer_alive": self._worker.is_alive(),
                "worker_accepts_cuda_objects": False,
                "worker_accepts_callbacks": False,
                "worker_tensor_device_policy": "forced_cpu_context_v1",
                "queued_cuda_references": 0,
                "queued_cuda_callbacks": 0,
                "capture_pool_count": 1,
                "capture_pool_model": "one_shared_two_slab_pool_not_per_slot",
            }

    def _worker_main(self) -> None:
        while True:
            with self._condition:
                job = self._jobs[0] if self._jobs and self._jobs[0].state == "CAPTURED_HOST_ONLY" else None
                while job is None:
                    if self._closed:
                        return
                    self._condition.wait()
                    job = self._jobs[0] if self._jobs and self._jobs[0].state == "CAPTURED_HOST_ONLY" else None
                job.state = "PERSISTING"
                job.started_ns = self._clock_ns()
                capture = job.capture
            assert capture is not None
            writer_cpu_started = time.thread_time_ns()
            writer_wall_started = self._clock_ns()
            publication_failure_path: str | None = None
            try:
                # The detached writer is a host-persistence owner.  Force a
                # thread-local CPU default even if the producer configured a
                # process-wide CUDA default: unqualified Torch allocations in
                # settlement must never recreate GPU residency after terminal
                # cudaDeviceReset.
                with torch.device("cpu"):
                    result = _finalize_capture_v1(
                        capture,
                        cancel_event=job.cancel_event,
                    )
                publication_created = bool(
                    result.pop("_detached_publication_created_by_worker", False)
                )
                publication_path = str(result.get("path") or "")
                if job.cancel_event.is_set():
                    if publication_created and publication_path:
                        publication_failure_path = (
                            _quarantine_owned_publication_v1(publication_path)
                        )
                    raise _DetachedFinalizeCancelledV1(
                        "detached ACS publication was cancelled before durability"
                    )
            except BaseException as exc:
                failure_capture_path = _quarantine_capture_file_v1(capture)
                exception_publication_failure_path = getattr(
                    exc, "publication_failure_path", None
                )
                if exception_publication_failure_path is not None:
                    publication_failure_path = exception_publication_failure_path
                with self._condition:
                    job.failure = exc
                    job.failure_capture_path = failure_capture_path
                    if failure_capture_path is not None:
                        self._failure_capture_paths.append(failure_capture_path)
                    if publication_failure_path is not None:
                        self._failure_capture_paths.append(
                            str(publication_failure_path)
                        )
                    job.state = "FAILED"
                    job.finished_ns = self._clock_ns()
                    self._writer_busy_ns += job.finished_ns - int(job.started_ns or job.finished_ns)
                    self._failure = exc
                    self._closed = True
                    self._condition.notify_all()
                return
            with self._condition:
                job.publication_created = publication_created
                job.publication_path = publication_path
                if job.cancel_event.is_set() or self._aborted:
                    failed_path = (
                        _quarantine_owned_publication_v1(publication_path)
                        if publication_created and publication_path
                        else None
                    )
                    if failed_path is not None:
                        self._failure_capture_paths.append(failed_path)
                    job.publication_created = False
                    job.failure = _DetachedFinalizeCancelledV1(
                        "detached ACS publication was cancelled before durability"
                    )
                    job.state = "FAILED"
                    job.finished_ns = self._clock_ns()
                    self._writer_busy_ns += job.finished_ns - int(
                        job.started_ns or job.finished_ns
                    )
                    self._failure = job.failure
                    self._closed = True
                    self._condition.notify_all()
                    return
                finished_ns = self._clock_ns()
                result = {
                    **dict(result),
                    "detached_slot_reserved_perf_ns": job.enqueued_ns,
                    "detached_capture_submitted_perf_ns": job.submitted_ns,
                    "detached_writer_started_perf_ns": job.started_ns,
                    "detached_writer_finished_perf_ns": finished_ns,
                    "detached_writer_cpu_time_ns": (
                        time.thread_time_ns() - writer_cpu_started
                    ),
                    "detached_writer_wall_time_ns": (
                        finished_ns - writer_wall_started
                    ),
                }
                job.result = result
                job.state = "DURABLE"
                job.finished_ns = finished_ns
                self._writer_busy_ns += job.finished_ns - int(job.started_ns or job.finished_ns)
                self._condition.notify_all()

    def _job_for_token_locked(
        self, token: ACSNativeDetachedSlotTokenV1
    ) -> _DetachedSlotJobV1:
        if not isinstance(token, ACSNativeDetachedSlotTokenV1):
            raise ACSNativeDetachedSlotError("detached ACS token has the wrong type")
        job = self._by_sequence.get(token.sequence)
        if job is None or job.token != token:
            raise ACSNativeDetachedSlotError("detached ACS token is unknown")
        return job

    def _raise_if_failed_locked(self) -> None:
        if self._failure is not None:
            raise ACSNativeDetachedSlotError(
                f"detached ACS failure is latched: {self._failure}"
            ) from self._failure

    def _raise_if_aborted_locked(self) -> None:
        if self._aborted:
            raise ACSNativeDetachedSlotError("detached ACS slots were aborted")


def _focused_probe_projection() -> dict[str, Any]:
    """Return a CPU-only projection for route-level detached-slot probing."""

    return {
        "schema_version": "cxxbackbone10_checkpoint_projection_v2",
        "metadata": {
            "generation": 0,
            "tick": 0,
            "world": "detached_slot_probe",
            "native_runtime_fixed_policy_family": "probe",
        },
        "event_cursor": 0,
        "runtime_event_cursor": 0,
        "tensors": {"probe": torch.zeros(4, dtype=torch.float32)},
    }


def _focused_probe_capture(root: Path, ordinal: int) -> dict[str, Any]:
    cell_id = "detached-probe-cell"
    digest = "b" * 64
    return capture_acs_native_aggregate_payload_v1(
        (_focused_probe_projection(),),
        root / f"detached-probe-{ordinal}.acsaggregate",
        cell_ids=(cell_id,),
        generation=0,
        tick=ordinal,
        absolute_next_tick=ordinal + 1,
        batch_id="detached-slot-focused-probe",
        segment_ordinal=ordinal + 1,
        source_projection_sha256s={cell_id: digest},
        architecture_b_identities={
            cell_id: {
                "snapshot_digest_sha256": digest,
                "source_projection_sha256": digest,
                "snapshot_id": f"probe-{ordinal}",
            }
        },
        observer_continuations={
            cell_id: {"event_frontier": ordinal, "cell_id": cell_id}
        },
        chunk_bytes=7,
        require_native_extension=False,
    )


def _focused_probe_source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def execute_detached_slots_focused_probe_v1(
    *,
    execution_route_sha256: str,
    obligations: frozenset[str],
    immutable_snapshot_slots: int,
) -> dict[str, Any]:
    """Exercise detached-slot backpressure and failure closure without CUDA.

    The probe uses CPU tensors only to create valid native payload receipts.  It
    proves the detached queue/state-machine boundary; physical B136 execution
    remains the authority for device residency and production timing.
    """

    if type(immutable_snapshot_slots) is not int or immutable_snapshot_slots not in {
        1,
        2,
    }:
        raise ACSNativeDetachedSlotError(
            "detached focused probe requires one or two immutable slots"
        )
    requested = set(obligations) & {
        "async_checkpoint_backpressure",
        "async_checkpoint_failure_latch",
    }
    result: dict[str, Any] = {
        "schema_version": "eiel_acs_checkpoint_focused_route_evidence_v1",
        "source_kind": "detached_file_backed_native_probe_v1",
        "implementation": (
            "engine.campaigns.acs_native_detached_slots_v1.ACSNativeDetachedSlotsV1"
        ),
        "implementation_source_sha256": _focused_probe_source_sha256(),
        "execution_route_sha256": execution_route_sha256,
        "target_immutable_snapshot_slots": immutable_snapshot_slots,
        "covered_obligations": [],
        "missing_obligations": sorted(requested),
        "exact": False,
    }

    if not requested:
        result["exact"] = True
        return result

    with tempfile.TemporaryDirectory(prefix="eiel-acs-detached-focused-") as raw_root:
        root = Path(raw_root)
        if "async_checkpoint_backpressure" in requested:
            pipeline = ACSNativeDetachedSlotsV1(slots=immutable_snapshot_slots)
            blocked_started = threading.Event()
            blocked_resumed = threading.Event()
            blocked_token: list[ACSNativeDetachedSlotTokenV1] = []
            blocked_errors: list[BaseException] = []
            tokens: list[ACSNativeDetachedSlotTokenV1] = []
            blocked_thread: threading.Thread | None = None
            try:
                for frontier in range(1, immutable_snapshot_slots + 1):
                    tokens.append(pipeline.begin_capture(compute_frontier=frontier))

                def submit_blocked() -> None:
                    blocked_started.set()
                    try:
                        blocked_token.append(
                            pipeline.begin_capture(
                                compute_frontier=immutable_snapshot_slots + 1
                            )
                        )
                    except BaseException as exc:
                        blocked_errors.append(exc)
                    finally:
                        blocked_resumed.set()

                blocked_thread = threading.Thread(
                    target=submit_blocked,
                    name="eiel-acs-detached-focused-backpressure",
                    daemon=True,
                )
                blocked_thread.start()
                if not blocked_started.wait(2):
                    raise ACSNativeDetachedSlotError(
                        "detached focused backpressure producer did not start"
                    )
                blocked_before_release = not blocked_resumed.wait(0.05)
                for ordinal, token in enumerate(tokens):
                    pipeline.submit_capture(token, _focused_probe_capture(root, ordinal))

                first = pipeline.result(tokens[0])
                if not Path(str(first["path"])).is_file():
                    raise ACSNativeDetachedSlotError(
                        "detached focused first publication is missing"
                    )
                pipeline.acknowledge_durable(tokens[0])
                if not blocked_resumed.wait(2):
                    raise ACSNativeDetachedSlotError(
                        "detached focused backpressure producer did not resume"
                    )
                blocked_thread.join(timeout=2)
                if blocked_thread.is_alive() or blocked_errors or len(blocked_token) != 1:
                    raise ACSNativeDetachedSlotError(
                        "detached focused backpressure producer did not settle"
                    )
                for token in tokens[1:]:
                    pipeline.result(token)
                    pipeline.acknowledge_durable(token)
                pipeline.submit_capture(
                    blocked_token[0], _focused_probe_capture(root, immutable_snapshot_slots)
                )
                pipeline.result(blocked_token[0])
                pipeline.acknowledge_durable(blocked_token[0])
                pipeline.close()
                telemetry = pipeline.telemetry()
                expected = [
                    "first",
                    *[
                        f"queued-{frontier}"
                        for frontier in range(2, immutable_snapshot_slots + 1)
                    ],
                    "blocked",
                ]
                exact = (
                    blocked_before_release
                    and type(telemetry.get("producer_blocked_time_ns")) is int
                    and telemetry["producer_blocked_time_ns"] > 0
                    and telemetry.get("queue_occupancy_high_water")
                    == immutable_snapshot_slots
                    and telemetry.get("compute_frontier")
                    == immutable_snapshot_slots + 1
                    and telemetry.get("durable_checkpoint_frontier")
                    == immutable_snapshot_slots + 1
                    and telemetry.get("frontier_lag") == 0
                    and telemetry.get("queue_occupancy") == 0
                    and telemetry.get("slot_states")
                    == ["FREE"] * immutable_snapshot_slots
                    and telemetry.get("failure_latched") is False
                    and telemetry.get("abandoned_snapshot_count") == 0
                    and telemetry.get("aborted") is False
                    and telemetry.get("closed") is True
                    and telemetry.get("writer_alive") is False
                )
                if not exact:
                    raise ACSNativeDetachedSlotError(
                        "detached focused backpressure evidence is not exact"
                    )
                result["backpressure"] = {
                    "blocked_before_release": True,
                    "post_release_submit_completed": True,
                    "producer_thread_joined": True,
                    "all_slots_occupied_before_block": True,
                    "blocked_submission_compute_frontier": immutable_snapshot_slots + 1,
                    "fifo_results": expected,
                    "fifo_persist_order": expected,
                    "no_overwrite_or_drop": True,
                    "producer_blocked_time_ns": telemetry[
                        "producer_blocked_time_ns"
                    ],
                    "terminal_telemetry": telemetry,
                    "exact": True,
                }
            finally:
                if blocked_thread is not None and blocked_thread.is_alive():
                    pipeline.abort()

        if "async_checkpoint_failure_latch" in requested:
            pipeline = ACSNativeDetachedSlotsV1(slots=immutable_snapshot_slots)
            tokens = []
            failure_root = root / "failure"
            failure_root.mkdir()
            failure_started = threading.Event()
            release_failure = threading.Event()
            original_finalize = _finalize_capture_v1

            def fail_first_capture(
                capture: Mapping[str, Any],
                *,
                cancel_event: threading.Event | None = None,
            ) -> dict[str, Any]:
                if capture.get("batch_id") == "detached-slot-focused-failure":
                    failure_started.set()
                    if not release_failure.wait(2):
                        raise TimeoutError(
                            "detached focused failure worker release timed out"
                        )
                    raise OSError("eiel-detached-focused-injected-writer-failure")
                return original_finalize(capture, cancel_event=cancel_event)

            try:
                globals()["_finalize_capture_v1"] = fail_first_capture
                for frontier in range(1, immutable_snapshot_slots + 1):
                    tokens.append(pipeline.begin_capture(compute_frontier=frontier))
                first_capture = _focused_probe_capture(failure_root, 0)
                first_capture["batch_id"] = "detached-slot-focused-failure"
                pipeline.submit_capture(tokens[0], first_capture)
                if not failure_started.wait(2):
                    raise ACSNativeDetachedSlotError(
                        "detached focused failure worker did not start"
                    )
                for ordinal, token in enumerate(tokens[1:], start=1):
                    pipeline.submit_capture(
                        token, _focused_probe_capture(failure_root, ordinal)
                    )
                release_failure.set()
                failure_observed = False
                try:
                    pipeline.result(tokens[0])
                except ACSNativeDetachedSlotError:
                    failure_observed = True
                post_latch_rejected = False
                try:
                    pipeline.begin_capture(compute_frontier=immutable_snapshot_slots + 1)
                except ACSNativeDetachedSlotError:
                    post_latch_rejected = True
                pipeline.abort()
                telemetry = pipeline.telemetry()
                inaccessible = True
                for token in tokens:
                    for operation in (
                        lambda token=token: pipeline.ready(token),
                        lambda token=token: pipeline.result(token),
                        lambda token=token: pipeline.acknowledge_durable(token),
                    ):
                        try:
                            operation()
                        except ACSNativeDetachedSlotError:
                            continue
                        inaccessible = False
                queued_never_started = not list(failure_root.glob("*.acsaggregate"))
                exact = (
                    failure_observed
                    and post_latch_rejected
                    and queued_never_started
                    and inaccessible
                    and telemetry.get("slots") == immutable_snapshot_slots
                    and telemetry.get("queue_occupancy_high_water")
                    == immutable_snapshot_slots
                    and telemetry.get("abandoned_snapshot_count")
                    == immutable_snapshot_slots
                    and telemetry.get("queue_occupancy") == 0
                    and telemetry.get("slot_states")
                    == ["FREE"] * immutable_snapshot_slots
                    and telemetry.get("failure_latched") is True
                    and telemetry.get("closed") is True
                    and telemetry.get("aborted") is True
                    and telemetry.get("writer_alive") is False
                    and telemetry.get("durable_checkpoint_frontier") == -1
                )
                if not exact:
                    raise ACSNativeDetachedSlotError(
                        "detached focused failure-latch evidence is not exact"
                    )
                result["failure_latch"] = {
                    "injected_failure": "eiel-detached-focused-injected-writer-failure",
                    "failure_observed": True,
                    "post_latch_capture_rejected": True,
                    "abort_joined_writer": True,
                    "all_slots_occupied_before_failure": True,
                    "queued_token_count": immutable_snapshot_slots - 1,
                    "queued_callbacks_never_started": True,
                    "all_tokens_inaccessible_after_abort": True,
                    "terminal_telemetry": telemetry,
                    "exact": True,
                }
            finally:
                release_failure.set()
                globals()["_finalize_capture_v1"] = original_finalize
                if not pipeline.telemetry().get("closed"):
                    pipeline.abort()

    result["covered_obligations"] = sorted(requested)
    result["missing_obligations"] = []
    result["exact"] = True
    return result


__all__ = [
    "ACSNativeDetachedSlotError",
    "ACSNativeDetachedSlotTokenV1",
    "ACSNativeDetachedSlotsV1",
    "DETACHED_ASYNC_SLOT_CAPACITIES_V1",
    "DETACHED_CAPTURE_HOST_ONLY_VALIDATION_SCHEMA_VERSION_V1",
    "DETACHED_CAPTURE_SCHEMA_VERSION_V1",
    "DETACHED_SPOOL_CAPTURE_SCHEMA_VERSION_V2",
    "DETACHED_SPOOL_SEAL_SCHEMA_VERSION_V1",
    "DETACHED_SLOTS_SCHEMA_VERSION_V1",
    "capture_acs_native_aggregate_payload_v1",
    "capture_acs_native_aggregate_payload_from_cursor_v1",
    "execute_detached_slots_focused_probe_v1",
    "load_acs_native_detached_spool_cell_v1",
    "validate_detached_capture_host_only_v1",
]
