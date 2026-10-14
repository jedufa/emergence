"""ACS-100X native packed streamed checkpoint format.

The outer ACS lifecycle deliberately remains in :mod:`acs_checkpoint_staging`.
This module owns only the physical representation of one detached cell
projection.  The format is a single immutable file containing a canonical
metadata/index header, field-major tensor chunks, and a sealed footer.  It is
strictly opt-in; BundleV2 remains the default route.

The Python implementation is the reference contract used by CPU parity and
failure-matrix tests.  The writer is intentionally chunked and bounded so a
native producer can replace the chunk-copy primitive without changing the
container or loader contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import time
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
from uuid import uuid4

import numpy as np
import torch


SCHEMA_VERSION = "acs100x_native_streamed_checkpoint_v1"
MAGIC = b"ACSNCV1\x00"
FORMAT_VERSION = 1
LITTLE_ENDIAN_MARKER = 0x0102
PREFIX = struct.Struct("<8sHHQQQ")
FOOTER_SCHEMA_VERSION = "acs100x_native_streamed_checkpoint_footer_v1"
DEFAULT_CHUNK_BYTES = 64 * 1024 * 1024
MAX_CHUNK_BYTES = 256 * 1024 * 1024
MAX_HEADER_BYTES = 64 * 1024 * 1024
MAX_FOOTER_BYTES = 4 * 1024 * 1024
MAX_FIELD_COUNT = 100_000
MAX_FIELD_BYTES = 1 << 42

_DTYPE_BY_NAME: dict[str, torch.dtype] = {
    "bool": torch.bool,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
    # Appended to preserve every existing on-disk dtype code.  Resident RNG
    # and identity authorities use unsigned wide integers and must round-trip
    # bit-exactly rather than being cast through a signed representation.
    "uint16": torch.uint16,
    "uint32": torch.uint32,
    "uint64": torch.uint64,
}
_DTYPE_CODE = {dtype_name: index + 1 for index, dtype_name in enumerate(_DTYPE_BY_NAME)}
_NATIVE_CHUNK_COPY: Any | None = None
_NATIVE_CHUNK_LOOKUP_DONE = False
_NATIVE_CHUNK_LOOKUP_PATH: str | None = None
_NATIVE_SLAB_POOL_TYPE: Any | None = None
_NATIVE_SLAB_POOL_LOOKUP_DONE = False
_NATIVE_SLAB_POOL_LOOKUP_PATH: str | None = None

NATIVE_CAPTURE_GROUP_SIZE = 8
NATIVE_CAPTURE_GROUP_SCHEMA_VERSION = "eiel_acs_native_capture_group_v1"


class ACSNativeCheckpointError(RuntimeError):
    """The packed checkpoint is malformed, conflicting, or not durable."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _pack_value(value: Any) -> Any:
    """Encode non-tensor projection values without lossy tuple/array casts."""

    if value is None:
        return {"kind": "none"}
    if type(value) is bool:
        return {"kind": "bool", "value": value}
    if type(value) is int:
        return {"kind": "int", "value": value}
    if type(value) is float:
        if not math.isfinite(value):
            raise ACSNativeCheckpointError("non-finite float in checkpoint metadata")
        return {"kind": "float", "value": value}
    if isinstance(value, str):
        return {"kind": "str", "value": value}
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"kind": "bytes", "hex": bytes(value).hex()}
    if isinstance(value, np.generic):
        return {
            "kind": "numpy_scalar",
            "dtype": str(value.dtype),
            "value": _pack_value(value.item()),
        }
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "kind": "ndarray",
            "dtype": str(array.dtype),
            "shape": [int(item) for item in array.shape],
            "hex": array.tobytes().hex(),
        }
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to("cpu").contiguous()
        return {
            "kind": "torch_tensor",
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": [int(item) for item in tensor.shape],
            "hex": _tensor_bytes(tensor).hex(),
        }
    if isinstance(value, Mapping):
        items = [
            [_pack_value(key), _pack_value(item)]
            for key, item in value.items()
        ]
        items.sort(key=lambda row: _canonical_json(row[0]))
        return {"kind": "mapping", "items": items}
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [_pack_value(item) for item in value]}
    if isinstance(value, list):
        return {"kind": "list", "items": [_pack_value(item) for item in value]}
    raise ACSNativeCheckpointError(
        f"unsupported checkpoint metadata value type: {type(value).__name__}"
    )


def _unpack_value(value: Any) -> Any:
    if not isinstance(value, Mapping) or not isinstance(value.get("kind"), str):
        raise ACSNativeCheckpointError("packed metadata value is malformed")
    kind = value["kind"]
    if kind == "none":
        return None
    if kind in {"bool", "int", "float", "str"}:
        return value.get("value")
    if kind == "bytes":
        return bytes.fromhex(str(value["hex"]))
    if kind == "numpy_scalar":
        return np.asarray(
            _unpack_value(value["value"]), dtype=np.dtype(str(value["dtype"]))
        ).item()
    if kind == "ndarray":
        dtype = np.dtype(str(value["dtype"]))
        shape = tuple(int(item) for item in value["shape"])
        raw = bytes.fromhex(str(value["hex"]))
        expected = int(dtype.itemsize)
        for dim in shape:
            expected *= dim
        if expected != len(raw):
            raise ACSNativeCheckpointError("packed ndarray byte length mismatch")
        return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    if kind == "torch_tensor":
        return _tensor_from_bytes(
            bytes.fromhex(str(value["hex"])),
            dtype_name=str(value["dtype"]),
            shape=tuple(int(item) for item in value["shape"]),
        )
    if kind in {"tuple", "list"}:
        items = [_unpack_value(item) for item in value["items"]]
        return tuple(items) if kind == "tuple" else items
    if kind == "mapping":
        return {
            _unpack_value(key): _unpack_value(item)
            for key, item in value["items"]
        }
    raise ACSNativeCheckpointError(f"unsupported packed metadata kind: {kind!r}")


def _dtype_name(tensor: torch.Tensor) -> str:
    name = str(tensor.dtype).removeprefix("torch.")
    if name not in _DTYPE_BY_NAME:
        raise ACSNativeCheckpointError(f"unsupported checkpoint tensor dtype: {name}")
    return name


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    if tensor.device.type != "cpu":
        tensor = tensor.to("cpu")
    tensor = tensor.detach().contiguous()
    if tensor.dtype is torch.bool:
        return tensor.to(torch.uint8).numpy().tobytes()
    return tensor.numpy().tobytes()


class _TwoSlabPool:
    """Two reusable staging slabs; pinned when the active runtime supports it."""

    def __init__(self, chunk_bytes: int, *, cuda: bool) -> None:
        self.pinned = False
        self._index = 0
        if cuda:
            try:
                self._slabs: tuple[Any, Any] = (
                    torch.empty(chunk_bytes, dtype=torch.uint8, pin_memory=True),
                    torch.empty(chunk_bytes, dtype=torch.uint8, pin_memory=True),
                )
                self.pinned = True
            except (RuntimeError, TypeError):
                self._slabs = (bytearray(chunk_bytes), bytearray(chunk_bytes))
        else:
            self._slabs = (bytearray(chunk_bytes), bytearray(chunk_bytes))

    def next(self) -> Any:
        slab = self._slabs[self._index]
        self._index = 1 - self._index
        return slab


class _ChunkSink:
    def __init__(self, stream: Any, *, offset: int) -> None:
        self.stream = stream
        self.offset = offset
        self.bytes = 0
        self.chunks: list[dict[str, Any]] = []
        self.digest = hashlib.sha256()

    def write(self, value: bytes | bytearray | memoryview) -> int:
        raw = bytes(value)
        self.stream.write(raw)
        self.digest.update(raw)
        self.chunks.append(
            {
                "offset": self.offset + self.bytes,
                "bytes": len(raw),
                "sha256": _sha256(raw),
            }
        )
        self.bytes += len(raw)
        return len(raw)


class _HashSink:
    """A bounded sink used when hashing a device-resident projection."""

    def __init__(self, digest: Any) -> None:
        self._digest = digest
        self.bytes = 0

    def write(self, value: bytes | bytearray | memoryview) -> int:
        raw = bytes(value)
        self._digest.update(raw)
        self.bytes += len(raw)
        return len(raw)


def _native_chunk_copy() -> Any | None:
    """Resolve the optional verified prebuilt producer per configured path."""

    global _NATIVE_CHUNK_COPY, _NATIVE_CHUNK_LOOKUP_DONE, _NATIVE_CHUNK_LOOKUP_PATH
    configured_path = str(os.environ.get("EIEL_NATIVE_RUNTIME_EXTENSION", "") or "")
    if _NATIVE_CHUNK_LOOKUP_DONE and (
        _NATIVE_CHUNK_LOOKUP_PATH == configured_path
        or (_NATIVE_CHUNK_LOOKUP_PATH is None and not configured_path)
    ):
        return _NATIVE_CHUNK_COPY
    _NATIVE_CHUNK_LOOKUP_PATH = configured_path
    _NATIVE_CHUNK_LOOKUP_DONE = True
    try:
        from engine.native_runtime.bindings import load_prebuilt_extension

        module = load_prebuilt_extension()
        candidate = getattr(
            module, "acs_native_streamed_checkpoint_copy_tensor_chunks", None
        )
        if callable(candidate):
            _NATIVE_CHUNK_COPY = candidate
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        _NATIVE_CHUNK_COPY = None
    return _NATIVE_CHUNK_COPY


def _native_slab_pool_type() -> Any | None:
    """Resolve the verified reusable native two-slab owner per configured path."""

    global _NATIVE_SLAB_POOL_TYPE, _NATIVE_SLAB_POOL_LOOKUP_DONE
    global _NATIVE_SLAB_POOL_LOOKUP_PATH
    configured_path = str(os.environ.get("EIEL_NATIVE_RUNTIME_EXTENSION", "") or "")
    if _NATIVE_SLAB_POOL_LOOKUP_DONE and (
        _NATIVE_SLAB_POOL_LOOKUP_PATH == configured_path
        or (_NATIVE_SLAB_POOL_LOOKUP_PATH is None and not configured_path)
    ):
        return _NATIVE_SLAB_POOL_TYPE
    _NATIVE_SLAB_POOL_LOOKUP_PATH = configured_path
    _NATIVE_SLAB_POOL_LOOKUP_DONE = True
    try:
        from engine.native_runtime.bindings import load_prebuilt_extension

        module = load_prebuilt_extension()
        candidate = getattr(
            module,
            "ACSNativeStreamedCheckpointSlabPool",
            None,
        )
        if candidate is not None:
            _NATIVE_SLAB_POOL_TYPE = candidate
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        _NATIVE_SLAB_POOL_TYPE = None
    return _NATIVE_SLAB_POOL_TYPE


class NativeCaptureGroupSession:
    """Own one reusable two-slab capture group and its boundary receipts.

    The session is intentionally created outside the per-cell writer loop. A
    group contains at most eight cells, all fields are copied in canonical
    field-major order inside each immutable cell projection, and the same
    native slab owner survives across the complete group. CUDA copies are
    ordered on a dedicated stream and the writer callback is invoked only
    after the copy boundary has completed.
    """

    def __init__(
        self,
        *,
        cell_ids: Sequence[str],
        group_start: int,
        chunk_bytes: int = DEFAULT_CHUNK_BYTES,
        cuda: bool,
        require_native_extension: bool,
    ) -> None:
        if not cell_ids or len(cell_ids) > NATIVE_CAPTURE_GROUP_SIZE:
            raise ACSNativeCheckpointError(
                "native capture group must contain one to eight cells"
            )
        if any(not isinstance(cell_id, str) or not cell_id for cell_id in cell_ids):
            raise ACSNativeCheckpointError("native capture group cell ids are invalid")
        if type(group_start) is not int or group_start < 0:
            raise ACSNativeCheckpointError("native capture group start is invalid")
        if type(cuda) is not bool or type(require_native_extension) is not bool:
            raise ACSNativeCheckpointError("native capture group flags are invalid")
        self.cell_ids = tuple(cell_ids)
        self.group_start = group_start
        self.group_end = group_start + len(self.cell_ids)
        self.chunk_bytes = chunk_bytes
        self.cuda = cuda
        self.cancelled = False
        self.cancel_reason: str | None = None
        self._cells: list[dict[str, Any]] = []
        self._current_cell: dict[str, Any] | None = None
        self._native_pool = None
        self._fallback_pool = None
        self._stream = None
        self._native_pool_type = _native_slab_pool_type()
        if self._native_pool_type is not None:
            self._native_pool = self._native_pool_type(chunk_bytes, cuda)
        elif require_native_extension:
            raise ACSNativeCheckpointError(
                "native capture production route requires the reusable native two-slab pool"
            )
        else:
            self._fallback_pool = _TwoSlabPool(chunk_bytes, cuda=cuda)
        if cuda:
            try:
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA is unavailable")
                self._stream = torch.cuda.Stream()
            except (RuntimeError, ValueError) as exc:
                if require_native_extension:
                    raise ACSNativeCheckpointError(
                        f"native capture dedicated CUDA stream unavailable: {exc}"
                    ) from exc
                self.cuda = False

    @property
    def native_pool_available(self) -> bool:
        return self._native_pool is not None

    @property
    def dedicated_stream_id(self) -> int | None:
        if self._stream is None:
            return None
        try:
            return int(self._stream.cuda_stream)
        except (AttributeError, TypeError, ValueError):
            return None

    def binding_payload(self) -> dict[str, Any]:
        """Return the immutable group binding copied into each artifact."""

        return {
            "schema_version": NATIVE_CAPTURE_GROUP_SCHEMA_VERSION,
            "group_start": self.group_start,
            "group_end": self.group_end,
            "group_size": len(self.cell_ids),
            "cell_ids": list(self.cell_ids),
            "field_order": "field_major_cell_ranges_v1",
            "chunk_bytes": self.chunk_bytes,
            "native_two_slab_pool": self.native_pool_available,
            "dedicated_d2h_stream": self._stream is not None,
            "dedicated_stream_id": self.dedicated_stream_id,
        }

    def begin_cell(
        self,
        *,
        cell_id: str,
        field_count: int,
        owner_identity: Mapping[str, Any] | None = None,
    ) -> None:
        if self.cancelled:
            raise ACSNativeCheckpointError(
                f"native capture group is cancelled: {self.cancel_reason}"
            )
        if self._current_cell is not None:
            raise ACSNativeCheckpointError("native capture group cell is already open")
        if cell_id not in self.cell_ids or any(
            row["cell_id"] == cell_id for row in self._cells
        ):
            raise ACSNativeCheckpointError("native capture group cell identity is invalid")
        row: dict[str, Any] = {
            "cell_id": cell_id,
            "field_count": field_count,
            "field_order": "lexicographic_field_major",
            "capture_started_host_ns": time.perf_counter_ns(),
            "capture_completed_host_ns": None,
            "slab_ownership_events": [],
            "status": "capturing",
            "owner_snapshot_id": (
                owner_identity.get("snapshot_id")
                if isinstance(owner_identity, Mapping)
                else None
            ),
            "owner_epoch": (
                owner_identity.get("owner_epoch")
                if isinstance(owner_identity, Mapping)
                else None
            ),
            "owner_boundary": "resident_owner_settlement_snapshot_v1",
        }
        if self._stream is not None:
            start_event = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(self._stream):
                start_event.record(self._stream)
            row["owner_boundary_event"] = start_event
        self._current_cell = row

    def copy_tensor_chunks(self, tensor: torch.Tensor, sink: Any) -> int:
        if self._current_cell is None:
            raise ACSNativeCheckpointError("native capture field has no owning cell")
        if self.cancelled:
            raise ACSNativeCheckpointError(
                f"native capture group is cancelled: {self.cancel_reason}"
            )
        if self._native_pool is not None:
            if self._stream is not None:
                with torch.cuda.stream(self._stream):
                    count = self._native_pool.copy_tensor_chunks(tensor, sink)
                self._stream.synchronize()
            else:
                count = self._native_pool.copy_tensor_chunks(tensor, sink)
        else:
            native_copy = _native_chunk_copy()
            if native_copy is None and self._fallback_pool is not None:
                count = 0
                for raw in _iter_tensor_chunks(
                    tensor,
                    chunk_bytes=self.chunk_bytes,
                    slab_pool=self._fallback_pool,
                ):
                    sink.write(raw)
                    count += 1
            elif native_copy is None:
                raise ACSNativeCheckpointError(
                    "native capture group has no verified native copy primitive"
                )
            elif self._stream is not None:
                with torch.cuda.stream(self._stream):
                    count = native_copy(tensor, sink, self.chunk_bytes)
                self._stream.synchronize()
            else:
                count = native_copy(tensor, sink, self.chunk_bytes)
        pool_telemetry = (
            self._native_pool.telemetry() if self._native_pool is not None else {}
        )
        self._current_cell["slab_ownership_events"].append(
            {
                "sequence": len(self._current_cell["slab_ownership_events"]),
                "event": "slab_returned_to_writer",
                "last_slab_index": pool_telemetry.get("last_slab_index"),
                "chunk_count": int(count),
            }
        )
        return int(count)

    def end_cell(self) -> dict[str, Any]:
        if self._current_cell is None:
            raise ACSNativeCheckpointError("native capture group cell is not open")
        row = self._current_cell
        if self._stream is not None:
            end_event = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(self._stream):
                end_event.record(self._stream)
            self._stream.synchronize()
            row["owner_boundary_event_elapsed_ms"] = float(
                row["owner_boundary_event"].elapsed_time(end_event)
            )
            row.pop("owner_boundary_event", None)
            row["dedicated_stream_id"] = self.dedicated_stream_id
        row["capture_completed_host_ns"] = time.perf_counter_ns()
        row["status"] = "complete"
        self._cells.append(row)
        self._current_cell = None
        return dict(row)

    def cancel(self, reason: str) -> None:
        if not isinstance(reason, str) or not reason:
            reason = "unspecified_native_capture_failure"
        self.cancelled = True
        self.cancel_reason = reason
        if self._current_cell is not None:
            self._current_cell["status"] = "cancelled"

    def receipt(self) -> dict[str, Any]:
        if self._current_cell is not None:
            raise ACSNativeCheckpointError("native capture group has an open cell")
        if not self.cancelled and tuple(row["cell_id"] for row in self._cells) != self.cell_ids:
            raise ACSNativeCheckpointError("native capture group did not complete every cell")
        pool_telemetry = (
            dict(self._native_pool.telemetry())
            if self._native_pool is not None
            else {
                "schema_version": "eiel_acs_native_slab_pool_fallback_v1",
                "chunk_bytes": self.chunk_bytes,
                "slab_count": 2,
                "pinned": False,
                "preallocated": False,
            }
        )
        return {
            "schema_version": NATIVE_CAPTURE_GROUP_SCHEMA_VERSION,
            "group_start": self.group_start,
            "group_end": self.group_end,
            "group_size": len(self.cell_ids),
            "cell_ids": list(self.cell_ids),
            "field_order": "field_major_cell_ranges_v1",
            "chunk_bytes": self.chunk_bytes,
            "native_two_slab_pool": self.native_pool_available,
            "dedicated_d2h_stream": self._stream is not None,
            "dedicated_stream_id": self.dedicated_stream_id,
            "slab_pool": pool_telemetry,
            "ownership_events": [
                {
                    key: value
                    for key, value in row.items()
                    if key != "owner_boundary_event"
                }
                for row in self._cells
            ],
            "failure_cancellation": {
                "status": "cancelled" if self.cancelled else "armed",
                "reason": self.cancel_reason,
            },
        }


def _tensor_raw_view(tensor: torch.Tensor) -> Any:
    """Return a one-dimensional byte view without materializing the field."""

    try:
        return tensor.numpy().view(np.uint8).reshape(-1)
    except (TypeError, RuntimeError, ValueError):
        return tensor.view(torch.uint8).reshape(-1)


def _tensor_from_bytes(raw: bytes, *, dtype_name: str, shape: tuple[int, ...]) -> torch.Tensor:
    dtype = _DTYPE_BY_NAME.get(dtype_name)
    if dtype is None:
        raise ACSNativeCheckpointError(f"unsupported checkpoint tensor dtype: {dtype_name}")
    element_count = 1
    for dim in shape:
        if dim < 0:
            raise ACSNativeCheckpointError("checkpoint tensor shape contains a negative dimension")
        element_count *= dim
    itemsize = torch.tensor([], dtype=dtype).element_size()
    expected = element_count * itemsize
    if len(raw) != expected:
        raise ACSNativeCheckpointError(
            f"checkpoint tensor byte length mismatch: expected {expected}, got {len(raw)}"
        )
    if element_count == 0:
        return torch.empty(shape, dtype=dtype)
    if dtype is torch.bool:
        tensor = torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone().to(torch.bool)
    else:
        try:
            tensor = torch.frombuffer(bytearray(raw), dtype=dtype).clone()
        except (RuntimeError, TypeError):
            tensor = torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone().view(dtype)
    return tensor.reshape(shape)


def _iter_tensor_chunks(
    tensor: torch.Tensor,
    *,
    chunk_bytes: int,
    slab_pool: _TwoSlabPool | None = None,
) -> Iterator[bytes]:
    """Yield bounded raw chunks; the producer can replace this with native slabs."""

    if type(chunk_bytes) is not int or chunk_bytes <= 0:
        raise ACSNativeCheckpointError("chunk_bytes must be a positive integer")
    source = tensor.detach()
    if not source.is_contiguous():
        source = source.contiguous()
    total_bytes = int(source.numel()) * int(source.element_size())
    if total_bytes == 0:
        return
    pool = slab_pool or _TwoSlabPool(chunk_bytes, cuda=source.device.type == "cuda")
    byte_view = (
        _tensor_raw_view(source)
        if source.device.type == "cpu"
        else source.view(torch.uint8).reshape(-1)
    )
    for offset in range(0, total_bytes, chunk_bytes):
        length = min(chunk_bytes, total_bytes - offset)
        slab = pool.next()
        if source.device.type == "cuda":
            if not isinstance(slab, torch.Tensor):
                raise ACSNativeCheckpointError("CUDA checkpoint staging slab is not pinned")
            slab[:length].copy_(byte_view[offset : offset + length], non_blocking=True)
            torch.cuda.current_stream(source.device).synchronize()
            yield slab[:length].numpy().tobytes()
        elif isinstance(slab, bytearray):
            slab[:length] = byte_view[offset : offset + length].tobytes()
            yield bytes(slab[:length])
        else:
            slab[:length].copy_(
                torch.from_numpy(byte_view[offset : offset + length]),
                non_blocking=False,
            )
            yield slab[:length].numpy().tobytes()


def _validate_projection(projection: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    if not isinstance(projection, Mapping):
        raise ACSNativeCheckpointError("checkpoint projection must be a mapping")
    tensors = projection.get("tensors")
    if not isinstance(tensors, Mapping) or not tensors:
        raise ACSNativeCheckpointError("checkpoint projection tensors must be a nonempty mapping")
    normalized: dict[str, torch.Tensor] = {}
    for name, value in tensors.items():
        if not isinstance(name, str) or not name:
            raise ACSNativeCheckpointError("checkpoint tensor field names must be nonempty strings")
        if not isinstance(value, torch.Tensor):
            raise ACSNativeCheckpointError(f"checkpoint tensor field is not tensor-like: {name}")
        normalized[name] = value
    if len(normalized) > MAX_FIELD_COUNT:
        raise ACSNativeCheckpointError("checkpoint tensor field count exceeds the bound")
    non_tensors = {str(name): value for name, value in projection.items() if name != "tensors"}
    return non_tensors, normalized


def streamed_projection_digest(
    projection: Mapping[str, Any],
    *,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> str:
    """Compute the legacy logical projection digest with bounded tensor staging.

    The digest byte contract matches ``projection_digest`` from the resident
    owner.  Unlike that historical helper, this implementation never creates
    a whole CPU copy of a device-resident tensor.
    """

    if type(chunk_bytes) is not int or not 0 < chunk_bytes <= MAX_CHUNK_BYTES:
        raise ACSNativeCheckpointError(
            f"chunk_bytes must be in the range 1..{MAX_CHUNK_BYTES}"
        )
    _non_tensors, tensors = _validate_projection(projection)
    metadata = dict(projection.get("metadata") or {})
    metadata.pop("import_digest", None)
    metadata.pop("native_policy_cache_served", None)
    metadata.pop("native_next_segment_index", None)
    metadata.pop("native_chain_sha256", None)
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    )
    native_copy = _native_chunk_copy()
    slab_pool = None if native_copy is not None else _TwoSlabPool(
        chunk_bytes,
        cuda=any(value.device.type == "cuda" for value in tensors.values()),
    )
    for name in sorted(tensors):
        tensor = tensors[name]
        digest.update(str(name).encode("utf-8"))
        digest.update(
            str(tensor.dtype).removeprefix("torch.").encode("utf-8")
        )
        digest.update(
            json.dumps(
                [int(value) for value in tensor.shape],
                separators=(",", ":"),
            ).encode("utf-8")
        )
        sink = _HashSink(digest)
        if native_copy is not None:
            native_count = native_copy(tensor, sink, chunk_bytes)
            expected_count = (
                int(tensor.numel()) * int(tensor.element_size()) + chunk_bytes - 1
            ) // chunk_bytes
            if type(native_count) is not int or native_count != expected_count:
                raise ACSNativeCheckpointError(
                    f"native projection digest chunk count is invalid: {name}"
                )
        else:
            for raw_chunk in _iter_tensor_chunks(
                tensor,
                chunk_bytes=chunk_bytes,
                slab_pool=slab_pool,
            ):
                sink.write(raw_chunk)
    return digest.hexdigest()


def _prefix(*, header_bytes: int, payload_bytes: int, footer_bytes: int) -> bytes:
    return PREFIX.pack(
        MAGIC,
        FORMAT_VERSION,
        LITTLE_ENDIAN_MARKER,
        header_bytes,
        payload_bytes,
        footer_bytes,
    )


def _read_exact(stream: Any, length: int, *, label: str) -> bytes:
    value = stream.read(length)
    if len(value) != length:
        raise ACSNativeCheckpointError(f"truncated native checkpoint {label}")
    return value


def _strict_json(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ACSNativeCheckpointError(f"invalid native checkpoint {label} JSON") from exc
    if not isinstance(value, dict):
        raise ACSNativeCheckpointError(f"native checkpoint {label} must be an object")
    if _canonical_json(value) != raw:
        raise ACSNativeCheckpointError(f"native checkpoint {label} is not canonical JSON")
    return value


def _footer_without_digest(
    *,
    content_sha256: str,
    header_sha256: str,
    payload_sha256: str,
    cell_id: str,
    source_projection_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": FOOTER_SCHEMA_VERSION,
        "status": "complete",
        "cell_id": cell_id,
        "source_projection_sha256": source_projection_sha256,
        "content_sha256": content_sha256,
        "header_sha256": header_sha256,
        "payload_sha256": payload_sha256,
        "seal_sha256": _sha256(
            _canonical_json(
                {
                    "content_sha256": content_sha256,
                    "header_sha256": header_sha256,
                    "payload_sha256": payload_sha256,
                    "cell_id": cell_id,
                    "source_projection_sha256": source_projection_sha256,
                }
            )
        ),
    }


def _artifact_digest(prefix: bytes, header: bytes, payload: bytes, footer: Mapping[str, Any]) -> str:
    return _sha256(prefix + header + payload + _canonical_json(dict(footer)))


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_file_range(path: Path, *, offset: int, length: int) -> str:
    if type(offset) is not int or offset < 0 or type(length) is not int or length < 0:
        raise ACSNativeCheckpointError("native checkpoint hash range is invalid")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            stream.seek(offset)
            remaining = length
            while remaining:
                chunk = stream.read(min(8 * 1024 * 1024, remaining))
                if not chunk:
                    raise ACSNativeCheckpointError("native checkpoint hash range is truncated")
                digest.update(chunk)
                remaining -= len(chunk)
    except OSError as exc:
        raise ACSNativeCheckpointError(f"native checkpoint is unreadable: {path}: {exc}") from exc
    return digest.hexdigest()


def _hash_file_range_with_suffix(
    path: Path,
    *,
    offset: int,
    length: int,
    suffix: bytes,
) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            stream.seek(offset)
            remaining = length
            while remaining:
                chunk = stream.read(min(8 * 1024 * 1024, remaining))
                if not chunk:
                    raise ACSNativeCheckpointError("native checkpoint hash range is truncated")
                digest.update(chunk)
                remaining -= len(chunk)
    except OSError as exc:
        raise ACSNativeCheckpointError(f"native checkpoint is unreadable: {path}: {exc}") from exc
    digest.update(suffix)
    return digest.hexdigest()


def _read_file_range(path: Path, *, offset: int, length: int, label: str) -> bytes:
    if type(offset) is not int or offset < 0 or type(length) is not int or length < 0:
        raise ACSNativeCheckpointError(f"native checkpoint {label} range is invalid")
    try:
        with path.open("rb") as stream:
            stream.seek(offset)
            raw = stream.read(length)
    except OSError as exc:
        raise ACSNativeCheckpointError(f"native checkpoint is unreadable: {path}: {exc}") from exc
    if len(raw) != length:
        raise ACSNativeCheckpointError(f"truncated native checkpoint {label}")
    return raw


def _hash_prefix_and_file(prefix: bytes, path: Path) -> str:
    digest = hashlib.sha256(prefix)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_digest_file(
    prefix: bytes,
    header: bytes,
    payload_path: Path,
    footer: Mapping[str, Any],
) -> str:
    digest = hashlib.sha256(prefix + header)
    with payload_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    digest.update(_canonical_json(dict(footer)))
    return digest.hexdigest()


def write_acs_native_streamed_checkpoint(
    projection: Mapping[str, Any],
    destination: str | os.PathLike[str],
    *,
    cell_id: str,
    generation: int,
    tick: int,
    absolute_next_tick: int | None,
    batch_id: str,
    segment_ordinal: int,
    source_projection_sha256: str,
    architecture_b_identity: Mapping[str, Any],
    observer_continuation: Mapping[str, Any],
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    performance_telemetry: dict[str, float] | None = None,
    require_native_extension: bool = False,
    native_receipt: dict[str, Any] | None = None,
    native_capture_session: NativeCaptureGroupSession | None = None,
) -> str:
    """Write one packed projection and publish it without replacing an owner."""

    if not isinstance(cell_id, str) or not cell_id:
        raise ACSNativeCheckpointError("cell_id must be a nonempty string")
    if not _is_sha256(source_projection_sha256):
        raise ACSNativeCheckpointError("source_projection_sha256 must be a SHA-256")
    if not isinstance(batch_id, str) or not batch_id:
        raise ACSNativeCheckpointError("batch_id must be a nonempty string")
    if type(generation) is not int or generation < 0:
        raise ACSNativeCheckpointError("generation must be a nonnegative integer")
    if type(tick) is not int or tick < 0:
        raise ACSNativeCheckpointError("tick must be a nonnegative integer")
    if absolute_next_tick is not None and (
        type(absolute_next_tick) is not int or absolute_next_tick < 0
    ):
        raise ACSNativeCheckpointError(
            "absolute_next_tick must be null or a nonnegative integer"
        )
    if type(segment_ordinal) is not int or segment_ordinal < 0:
        raise ACSNativeCheckpointError(
            "segment_ordinal must be a nonnegative integer"
        )
    if not isinstance(architecture_b_identity, Mapping):
        raise ACSNativeCheckpointError("architecture_b_identity must be a mapping")
    if not isinstance(observer_continuation, Mapping):
        raise ACSNativeCheckpointError("observer_continuation must be a mapping")
    if type(require_native_extension) is not bool:
        raise ACSNativeCheckpointError("require_native_extension must be boolean")
    if type(chunk_bytes) is not int or not 0 < chunk_bytes <= MAX_CHUNK_BYTES:
        raise ACSNativeCheckpointError(
            f"chunk_bytes must be in the range 1..{MAX_CHUNK_BYTES}"
        )
    non_tensors, tensors = _validate_projection(projection)
    target = Path(destination).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    field_rows: list[dict[str, Any]] = []
    payload_offset = 0
    chunk_count = 0
    cuda_fields = any(value.device.type == "cuda" for value in tensors.values())
    native_copy = _native_chunk_copy()
    if require_native_extension and native_capture_session is None:
        if native_copy is None:
            raise ACSNativeCheckpointError(
                "native streamed checkpoint route requires a verified prebuilt extension"
            )
        native_capture_session = NativeCaptureGroupSession(
            cell_ids=(cell_id,),
            group_start=0,
            chunk_bytes=chunk_bytes,
            cuda=cuda_fields,
            require_native_extension=True,
        )
    if native_capture_session is not None and not isinstance(
        native_capture_session,
        NativeCaptureGroupSession,
    ):
        raise ACSNativeCheckpointError("native capture session has the wrong type")
    if native_capture_session is not None and (
        native_capture_session.cuda != cuda_fields
        and cuda_fields
    ):
        raise ACSNativeCheckpointError(
            "native capture session device binding differs from the projection"
        )
    if require_native_extension and native_copy is None:
        if native_capture_session is None or not native_capture_session.native_pool_available:
            raise ACSNativeCheckpointError(
                "native streamed checkpoint route requires a verified prebuilt extension"
            )
    if require_native_extension and native_capture_session is not None and not native_capture_session.native_pool_available:
        raise ACSNativeCheckpointError(
            "native streamed checkpoint production route requires the reusable native two-slab pool"
        )
    metadata_payload = {
        "projection": _pack_value(non_tensors),
        "architecture_b_identity": _pack_value(dict(architecture_b_identity)),
        "observer_continuation": _pack_value(dict(observer_continuation)),
    }
    if native_capture_session is not None:
        metadata_payload["native_capture_group"] = _pack_value(
            native_capture_session.binding_payload()
        )
    slab_pool = (
        None if native_copy is not None else _TwoSlabPool(chunk_bytes, cuda=cuda_fields)
    )
    payload_temporary = target.with_name(
        f".{target.name}.{os.getpid()}.{uuid4().hex}.payload.tmp"
    )
    try:
        if native_capture_session is not None:
            native_capture_session.begin_cell(
                cell_id=cell_id,
                field_count=len(tensors),
                owner_identity=architecture_b_identity,
            )
        with payload_temporary.open("xb") as payload_stream:
            for name in sorted(tensors):
                tensor = tensors[name]
                dtype_name = _dtype_name(tensor)
                shape = [int(dim) for dim in tensor.shape]
                field_offset = payload_offset
                sink = _ChunkSink(payload_stream, offset=field_offset)
                if native_capture_session is not None:
                    native_chunk_count = native_capture_session.copy_tensor_chunks(
                        tensor,
                        sink,
                    )
                    if (
                        type(native_chunk_count) is not int
                        or native_chunk_count != len(sink.chunks)
                    ):
                        raise ACSNativeCheckpointError(
                            f"native capture group chunk count is invalid: {name}"
                        )
                elif native_copy is not None:
                    native_chunk_count = native_copy(tensor, sink, chunk_bytes)
                    if (
                        type(native_chunk_count) is not int
                        or native_chunk_count != len(sink.chunks)
                    ):
                        raise ACSNativeCheckpointError(
                            f"native checkpoint producer chunk count is invalid: {name}"
                        )
                else:
                    for raw_chunk in _iter_tensor_chunks(
                        tensor, chunk_bytes=chunk_bytes, slab_pool=slab_pool
                    ):
                        sink.write(raw_chunk)
                field_chunks = sink.chunks
                field_bytes = sink.bytes
                expected_field_bytes = int(tensor.numel()) * int(
                    tensor.element_size()
                )
                if field_bytes != expected_field_bytes or any(
                    row["bytes"] <= 0 or row["bytes"] > chunk_bytes
                    for row in field_chunks
                ):
                    raise ACSNativeCheckpointError(
                        f"native checkpoint field byte coverage is invalid: {name}"
                    )
                payload_offset += field_bytes
                chunk_count += len(field_chunks)
                if field_bytes > MAX_FIELD_BYTES:
                    raise ACSNativeCheckpointError(
                        f"checkpoint tensor field is too large: {name}"
                    )
                field_rows.append(
                    {
                        "field": name,
                        "dtype": dtype_name,
                        "dtype_code": _DTYPE_CODE[dtype_name],
                        "shape": shape,
                        "offset": field_offset,
                        "bytes": field_bytes,
                        "sha256": sink.digest.hexdigest(),
                        "chunks": field_chunks,
                    }
                )
            payload_stream.flush()
            os.fsync(payload_stream.fileno())
    except Exception:
        payload_temporary.unlink(missing_ok=True)
        raise

    payload_bytes = payload_offset
    header = {
        "schema_version": SCHEMA_VERSION,
        "format_version": FORMAT_VERSION,
        "endianness": "little",
        "batch_id": batch_id,
        "cell_id": cell_id,
        "generation": int(generation),
        "tick": int(tick),
        "absolute_next_tick": absolute_next_tick,
        "segment_ordinal": int(segment_ordinal),
        "source_projection_sha256": source_projection_sha256,
        "metadata": metadata_payload,
        "field_order": [row["field"] for row in field_rows],
        "fields": field_rows,
        "cell_index": [
            {
                "cell_id": cell_id,
                "field_start": field_rows[0]["offset"] if field_rows else 0,
                "field_end": payload_offset,
            }
        ],
        "chunk_bytes": int(chunk_bytes),
        "chunk_count": chunk_count,
        "payload_bytes": payload_bytes,
        "dtype_registry": dict(sorted(_DTYPE_CODE.items(), key=lambda row: row[1])),
    }
    header_bytes = _canonical_json(header)
    if len(header_bytes) > MAX_HEADER_BYTES:
        payload_temporary.unlink(missing_ok=True)
        raise ACSNativeCheckpointError("native checkpoint header exceeds the bound")
    payload_sha256 = _hash_file(payload_temporary)
    content_prefix = _prefix(header_bytes=len(header_bytes), payload_bytes=payload_bytes, footer_bytes=0)
    # Footer size is part of the fixed prefix.  Iterate once to resolve it
    # without introducing a self-referential digest.
    footer = _footer_without_digest(
        content_sha256=_hash_prefix_and_file(content_prefix + header_bytes, payload_temporary),
        header_sha256=_sha256(header_bytes),
        payload_sha256=payload_sha256,
        cell_id=cell_id,
        source_projection_sha256=source_projection_sha256,
    )
    footer["artifact_sha256"] = "0" * 64
    footer_bytes = _canonical_json(footer)
    for _ in range(3):
        prefix = _prefix(
            header_bytes=len(header_bytes),
            payload_bytes=payload_bytes,
            footer_bytes=len(footer_bytes),
        )
        footer = _footer_without_digest(
            content_sha256=_hash_prefix_and_file(prefix + header_bytes, payload_temporary),
            header_sha256=_sha256(header_bytes),
            payload_sha256=payload_sha256,
            cell_id=cell_id,
            source_projection_sha256=source_projection_sha256,
        )
        footer["artifact_sha256"] = _artifact_digest_file(
            prefix, header_bytes, payload_temporary, footer
        )
        next_footer_bytes = _canonical_json(footer)
        if len(next_footer_bytes) == len(footer_bytes):
            footer_bytes = next_footer_bytes
            break
        footer_bytes = next_footer_bytes
    prefix = _prefix(
        header_bytes=len(header_bytes), payload_bytes=payload_bytes, footer_bytes=len(footer_bytes)
    )
    # Recompute after the final prefix/length is known.
    footer = _footer_without_digest(
        content_sha256=_hash_prefix_and_file(prefix + header_bytes, payload_temporary),
        header_sha256=_sha256(header_bytes),
        payload_sha256=payload_sha256,
        cell_id=cell_id,
        source_projection_sha256=source_projection_sha256,
    )
    footer["artifact_sha256"] = _artifact_digest_file(
        prefix, header_bytes, payload_temporary, footer
    )
    footer_bytes = _canonical_json(footer)
    if len(footer_bytes) != PREFIX.unpack(prefix)[5]:
        raise ACSNativeCheckpointError("native checkpoint footer length did not converge")
    if len(footer_bytes) > MAX_FOOTER_BYTES:
        payload_temporary.unlink(missing_ok=True)
        raise ACSNativeCheckpointError("native checkpoint footer exceeds the bound")

    temporary = target.with_name(f".{target.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(prefix)
            stream.write(header_bytes)
            with payload_temporary.open("rb") as payload_stream:
                for chunk in iter(lambda: payload_stream.read(8 * 1024 * 1024), b""):
                    stream.write(chunk)
            stream.write(footer_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            existing = load_acs_native_streamed_checkpoint(target)
            if existing["footer"].get("artifact_sha256") != footer.get("artifact_sha256"):
                raise ACSNativeCheckpointError(
                    "conflicting native checkpoint publication already exists"
                )
        else:
            directory_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except Exception:
        temporary.unlink(missing_ok=True)
        payload_temporary.unlink(missing_ok=True)
        raise
    else:
        temporary.unlink(missing_ok=True)
        payload_temporary.unlink(missing_ok=True)
    if native_receipt is not None:
        native_receipt.update(
            {
                "schema_version": SCHEMA_VERSION,
                "path": str(target),
                "cell_id": cell_id,
                "batch_id": batch_id,
                "generation": generation,
                "tick": tick,
                "absolute_next_tick": absolute_next_tick,
                "segment_ordinal": segment_ordinal,
                "header_sha256": str(footer["header_sha256"]),
                "payload_sha256": str(footer["payload_sha256"]),
                "content_sha256": str(footer["content_sha256"]),
                "seal_sha256": str(footer["seal_sha256"]),
                "artifact_sha256": str(footer["artifact_sha256"]),
                "payload_bytes": payload_bytes,
                "field_count": len(field_rows),
                "chunk_count": chunk_count,
                "status": "complete",
            }
        )
        if native_capture_session is not None:
            native_receipt["native_capture_group"] = (
                native_capture_session.binding_payload()
            )
            native_receipt["native_capture_cell"] = native_capture_session.end_cell()
    elif native_capture_session is not None:
        native_capture_session.end_cell()
    if performance_telemetry is not None:
        performance_telemetry.update(
            {
                "native_pack_seconds": time.perf_counter() - started,
                "native_pack_payload_bytes": float(payload_bytes),
                "native_pack_field_count": float(len(field_rows)),
                "native_pack_chunk_count": float(chunk_count),
                "native_pack_two_slab_pool": 1.0,
                "native_pack_pinned_slabs": float(
                    bool(
                        native_capture_session is not None
                        and native_capture_session.native_pool_available
                    )
                    or bool(cuda_fields and native_copy is not None)
                    or bool(slab_pool is not None and slab_pool.pinned)
                ),
                "native_pack_prebuilt_extension": float(
                    native_capture_session is not None
                    and native_capture_session.native_pool_available
                )
                if native_capture_session is not None
                else float(native_copy is not None),
                "native_pack_reusable_group_pool": float(
                    native_capture_session is not None
                    and native_capture_session.native_pool_available
                ),
                "native_pack_compression": 0.0,
            }
        )
    return str(target)


def load_acs_native_streamed_checkpoint(
    source: str | os.PathLike[str],
    *,
    expected_source_projection_sha256: str | None = None,
) -> dict[str, Any]:
    """Strictly load and verify one sealed packed projection."""

    path = Path(source).expanduser().resolve()
    try:
        with path.open("rb") as stream:
            prefix_bytes = _read_exact(stream, PREFIX.size, label="prefix")
            magic, version, endian, header_len, payload_len, footer_len = PREFIX.unpack(prefix_bytes)
            if magic != MAGIC or version != FORMAT_VERSION or endian != LITTLE_ENDIAN_MARKER:
                raise ACSNativeCheckpointError("native checkpoint prefix identity mismatch")
            if not 0 < header_len <= MAX_HEADER_BYTES:
                raise ACSNativeCheckpointError("native checkpoint header length is outside the bound")
            if payload_len < 0 or payload_len > MAX_FIELD_BYTES * MAX_FIELD_COUNT:
                raise ACSNativeCheckpointError("native checkpoint payload length is outside the bound")
            if not 0 < footer_len <= MAX_FOOTER_BYTES:
                raise ACSNativeCheckpointError("native checkpoint footer length is outside the bound")
            header_bytes = _read_exact(stream, header_len, label="header")
            payload_offset = stream.tell()
            stream.seek(payload_len, os.SEEK_CUR)
            footer_bytes = _read_exact(stream, footer_len, label="footer")
            if stream.read(1):
                raise ACSNativeCheckpointError("native checkpoint has trailing bytes")
    except OSError as exc:
        raise ACSNativeCheckpointError(f"native checkpoint is unreadable: {path}: {exc}") from exc

    header = _strict_json(header_bytes, label="header")
    footer = _strict_json(footer_bytes, label="footer")
    if header.get("schema_version") != SCHEMA_VERSION:
        raise ACSNativeCheckpointError("native checkpoint schema mismatch")
    if (
        header.get("format_version") != FORMAT_VERSION
        or header.get("endianness") != "little"
        or type(header.get("payload_bytes")) is not int
        or header.get("payload_bytes") != payload_len
        or not isinstance(header.get("batch_id"), str)
        or not isinstance(header.get("cell_id"), str)
        or not header.get("cell_id")
        or not _is_sha256(header.get("source_projection_sha256"))
    ):
        raise ACSNativeCheckpointError("native checkpoint header identity mismatch")
    if footer.get("schema_version") != FOOTER_SCHEMA_VERSION or footer.get("status") != "complete":
        raise ACSNativeCheckpointError("native checkpoint is not sealed complete")
    if (
        footer.get("cell_id") != header.get("cell_id")
        or footer.get("source_projection_sha256")
        != header.get("source_projection_sha256")
        or not _is_sha256(footer.get("source_projection_sha256"))
        or not _is_sha256(footer.get("content_sha256"))
        or not _is_sha256(footer.get("header_sha256"))
        or not _is_sha256(footer.get("payload_sha256"))
        or not _is_sha256(footer.get("seal_sha256"))
        or not _is_sha256(footer.get("artifact_sha256"))
    ):
        raise ACSNativeCheckpointError("native checkpoint footer identity mismatch")
    if _hash_file_range(path, offset=payload_offset, length=payload_len) != footer.get(
        "payload_sha256"
    ):
        raise ACSNativeCheckpointError("native checkpoint payload hash mismatch")
    if _sha256(header_bytes) != footer.get("header_sha256"):
        raise ACSNativeCheckpointError("native checkpoint header hash mismatch")
    content_sha256 = _hash_file_range(
        path, offset=0, length=PREFIX.size + header_len + payload_len
    )
    if content_sha256 != footer.get("content_sha256"):
        raise ACSNativeCheckpointError("native checkpoint content hash mismatch")
    expected_seal = _sha256(
        _canonical_json(
            {
                "content_sha256": footer.get("content_sha256"),
                "header_sha256": footer.get("header_sha256"),
                "payload_sha256": footer.get("payload_sha256"),
                "cell_id": footer.get("cell_id"),
                "source_projection_sha256": footer.get("source_projection_sha256"),
            }
        )
    )
    if expected_seal != footer.get("seal_sha256"):
        raise ACSNativeCheckpointError("native checkpoint seal mismatch")
    artifact_footer = dict(footer)
    artifact_digest = artifact_footer.pop("artifact_sha256", None)
    if artifact_digest != _hash_file_range_with_suffix(
        path,
        offset=0,
        length=PREFIX.size + header_len + payload_len,
        suffix=_canonical_json(artifact_footer),
    ):
        raise ACSNativeCheckpointError("native checkpoint artifact hash mismatch")
    if expected_source_projection_sha256 is not None and (
        not _is_sha256(expected_source_projection_sha256)
        or footer.get("source_projection_sha256")
        != expected_source_projection_sha256
    ):
        raise ACSNativeCheckpointError("native checkpoint source digest binding mismatch")

    fields = header.get("fields")
    if not isinstance(fields, list) or not fields or len(fields) > MAX_FIELD_COUNT:
        raise ACSNativeCheckpointError("native checkpoint field index is invalid")
    field_order = header.get("field_order")
    if (
        not isinstance(field_order, list)
        or not all(isinstance(item, str) and item for item in field_order)
        or field_order != [row.get("field") for row in fields if isinstance(row, Mapping)]
        or field_order != sorted(set(field_order))
    ):
        raise ACSNativeCheckpointError("native checkpoint field order is not canonical")
    chunk_bytes = header.get("chunk_bytes")
    chunk_count = header.get("chunk_count")
    if (
        type(chunk_bytes) is not int
        or not 0 < chunk_bytes <= MAX_CHUNK_BYTES
        or type(chunk_count) is not int
        or chunk_count < 0
        or header.get("dtype_registry")
        != dict(sorted(_DTYPE_CODE.items(), key=lambda row: row[1]))
    ):
        raise ACSNativeCheckpointError("native checkpoint index contract is invalid")
    cell_index = header.get("cell_index")
    if (
        not isinstance(cell_index, list)
        or len(cell_index) != 1
        or not isinstance(cell_index[0], Mapping)
        or cell_index[0].get("cell_id") != header.get("cell_id")
        or cell_index[0].get("field_start") != 0
        or cell_index[0].get("field_end") != payload_len
    ):
        raise ACSNativeCheckpointError("native checkpoint cell index is invalid")
    tensors: dict[str, torch.Tensor] = {}
    previous_end = 0
    observed_chunk_count = 0
    for row in fields:
        if not isinstance(row, Mapping):
            raise ACSNativeCheckpointError("native checkpoint field row is invalid")
        name = row.get("field")
        if not isinstance(name, str) or not name or name in tensors:
            raise ACSNativeCheckpointError("native checkpoint field name is invalid")
        dtype_name = row.get("dtype")
        dtype_code = row.get("dtype_code")
        shape_raw = row.get("shape")
        offset = row.get("offset")
        size = row.get("bytes")
        chunks = row.get("chunks")
        if (
            not isinstance(dtype_name, str)
            or dtype_name not in _DTYPE_BY_NAME
            or type(dtype_code) is not int
            or dtype_code != _DTYPE_CODE[dtype_name]
            or not isinstance(shape_raw, list)
            or type(offset) is not int
            or type(size) is not int
            or size < 0
            or type(chunks) is not list
        ):
            raise ACSNativeCheckpointError(f"native checkpoint field row is malformed: {name}")
        if any(type(item) is not int or item < 0 for item in shape_raw):
            raise ACSNativeCheckpointError(f"native checkpoint field shape is invalid: {name}")
        expected_size = torch.tensor([], dtype=_DTYPE_BY_NAME[dtype_name]).element_size()
        for dimension in shape_raw:
            expected_size *= dimension
        if expected_size != size or expected_size > MAX_FIELD_BYTES:
            raise ACSNativeCheckpointError(f"native checkpoint field byte length is invalid: {name}")
        if offset != previous_end or offset < 0 or offset + size > payload_len:
            raise ACSNativeCheckpointError(f"native checkpoint field offsets are invalid: {name}")
        field_raw = _read_file_range(
            path,
            offset=payload_offset + offset,
            length=size,
            label=f"field {name}",
        )
        if not _is_sha256(row.get("sha256")) or _sha256(field_raw) != row.get("sha256"):
            raise ACSNativeCheckpointError(f"native checkpoint field hash mismatch: {name}")
        cursor = offset
        for chunk in chunks:
            if (
                not isinstance(chunk, Mapping)
                or type(chunk.get("offset")) is not int
                or chunk.get("offset") != cursor
            ):
                raise ACSNativeCheckpointError(f"native checkpoint chunk index is invalid: {name}")
            chunk_size = chunk.get("bytes")
            if (
                type(chunk_size) is not int
                or chunk_size <= 0
                or chunk_size > chunk_bytes
                or cursor + chunk_size > offset + size
                or not _is_sha256(chunk.get("sha256"))
            ):
                raise ACSNativeCheckpointError(f"native checkpoint chunk bounds are invalid: {name}")
            chunk_start = cursor - offset
            chunk_raw = field_raw[chunk_start : chunk_start + chunk_size]
            if _sha256(chunk_raw) != chunk.get("sha256"):
                raise ACSNativeCheckpointError(f"native checkpoint chunk hash mismatch: {name}")
            cursor += chunk_size
            observed_chunk_count += 1
        if cursor != offset + size:
            raise ACSNativeCheckpointError(f"native checkpoint chunk coverage is incomplete: {name}")
        tensors[name] = _tensor_from_bytes(
            field_raw,
            dtype_name=dtype_name,
            shape=tuple(int(item) for item in shape_raw),
        )
        previous_end = offset + size
    if previous_end != payload_len:
        raise ACSNativeCheckpointError("native checkpoint field index does not cover payload")
    if observed_chunk_count != chunk_count:
        raise ACSNativeCheckpointError("native checkpoint chunk count is inconsistent")

    metadata = header.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ACSNativeCheckpointError("native checkpoint metadata envelope is invalid")
    try:
        projection_value = _unpack_value(metadata.get("projection"))
        architecture_b_identity = _unpack_value(
            metadata.get("architecture_b_identity")
        )
        observer_continuation = _unpack_value(
            metadata.get("observer_continuation")
        )
        native_capture_group = (
            _unpack_value(metadata["native_capture_group"])
            if "native_capture_group" in metadata
            else None
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ACSNativeCheckpointError(
            "native checkpoint metadata value is malformed"
        ) from exc
    if not isinstance(projection_value, Mapping):
        raise ACSNativeCheckpointError("native checkpoint projection envelope is invalid")
    if not isinstance(architecture_b_identity, Mapping):
        raise ACSNativeCheckpointError(
            "native checkpoint architecture identity envelope is invalid"
        )
    if not isinstance(observer_continuation, Mapping):
        raise ACSNativeCheckpointError(
            "native checkpoint observer continuation envelope is invalid"
        )
    if native_capture_group is not None and not isinstance(
        native_capture_group, Mapping
    ):
        raise ACSNativeCheckpointError(
            "native checkpoint capture-group envelope is invalid"
        )
    for identity_field in ("snapshot_digest_sha256", "source_projection_sha256"):
        identity_digest = architecture_b_identity.get(identity_field)
        if identity_digest is not None and identity_digest != footer.get(
            "source_projection_sha256"
        ):
            raise ACSNativeCheckpointError(
                "native checkpoint architecture identity digest binding mismatch"
            )
    projection = dict(projection_value)
    projection["tensors"] = tensors
    return {
        "projection": projection,
        "header": header,
        "footer": footer,
        "architecture_b_identity": architecture_b_identity,
        "observer_continuation": observer_continuation,
        "native_capture_group": native_capture_group,
        "artifact_sha256": str(footer["artifact_sha256"]),
    }


__all__ = [
    "ACSNativeCheckpointError",
    "DEFAULT_CHUNK_BYTES",
    "FOOTER_SCHEMA_VERSION",
    "MAGIC",
    "MAX_CHUNK_BYTES",
    "NATIVE_CAPTURE_GROUP_SCHEMA_VERSION",
    "NATIVE_CAPTURE_GROUP_SIZE",
    "NativeCaptureGroupSession",
    "SCHEMA_VERSION",
    "load_acs_native_streamed_checkpoint",
    "write_acs_native_streamed_checkpoint",
]
