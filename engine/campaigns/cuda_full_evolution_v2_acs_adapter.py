"""Bounded host/file-only ACS2 adapter for CUDA Full Evolution V2.

The producer owns only the device-to-host capture barrier.  It streams one
contiguous canonical state through the existing reusable two-slab copier into
an unsealed file, then queues only deterministic JSON metadata.  The detached
worker validates and durably publishes the unchanged Workhorse V2 checkpoint
bundle without retaining a CUDA tensor, callback, owner, or full-state bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import struct
import tempfile
import threading
import time
import warnings
from collections import deque
from typing import Any, Final, Mapping
from uuid import uuid4

import torch

from .canonical import canonical_bytes
from . import cuda_full_evolution_v2_perf_surface as _perf_surface

from . import acs_native_streamed_checkpoint_v1 as streamed
from .acs_physical_batch_packed_checkpoint_v1 import (
    _PayloadSink,
    _SharedTwoSlabCopier,
)


_cuda_v2_contract = _perf_surface.contract_v2
_direct_sparse_runtime = _perf_surface.direct_sparse_runtime_v2
_cuda_v2_layout = _perf_surface.layout_v2
_cuda_v2_native_loader = _perf_surface.native_loader_v2
_cuda_v2_runtime = _perf_surface.runtime_v2
_workhorse = _perf_surface.workhorse_v2


FullEvolutionWorkhorseAuthorityError = (
    _workhorse.FullEvolutionWorkhorseAuthorityError
)
inspect_workhorse_checkpoint_bundle_v2 = (
    _workhorse.inspect_workhorse_checkpoint_bundle_v2
)
reopen_workhorse_checkpoint_bundle_v2 = (
    _workhorse.reopen_workhorse_checkpoint_bundle_v2
)
workhorse_checkpoint_metadata_from_state_identity_v2 = (
    _workhorse.workhorse_checkpoint_metadata_from_state_identity_v2
)
WORKHORSE_TIME_AUTHORITY_V2 = _workhorse.contract.WORKHORSE_TIME_AUTHORITY_V2
FullEvolutionCudaWorkhorseContractV2 = (
    _workhorse.FullEvolutionCudaWorkhorseContractV2
)
FullEvolutionWorkhorseContractError = _workhorse.FullEvolutionWorkhorseContractError
workhorse_layout_v2 = _workhorse.workhorse_layout_v2
CudaFullEvolutionNativeContractV2 = (
    _cuda_v2_contract.CudaFullEvolutionNativeContractV2
)
CudaFullEvolutionCheckpointV2 = _cuda_v2_runtime.CudaFullEvolutionCheckpointV2
CudaFullEvolutionCheckpointCaptureSourceV1 = (
    _cuda_v2_runtime.CudaFullEvolutionCheckpointCaptureSourceV1
)
_validate_checkpoint_envelope = _cuda_v2_runtime._validate_checkpoint_envelope
checkpoint_runtime_projection_from_capture_source_v1 = (
    _cuda_v2_runtime.checkpoint_runtime_projection_from_capture_source_v1
)
# Keep the campaign layer's V2 runtime dependency on one explicit architecture
# edge.  The native provider consumes these module handles instead of adding a
# second direct campaign-to-perf import.
provider_direct_sparse_runtime_v2 = _direct_sparse_runtime
provider_layout_v2 = _cuda_v2_layout
provider_native_loader_v2 = _cuda_v2_native_loader


WORKHORSE_DETACHED_CAPTURE_SCHEMA_V1: Final[str] = (
    "eiel_cuda_full_evolution_v2_workhorse_detached_checkpoint_capture_v1"
)
WORKHORSE_DETACHED_RESULT_SCHEMA_V1: Final[str] = (
    "eiel_cuda_full_evolution_v2_workhorse_detached_checkpoint_result_v1"
)
CUDA_NATIVE_DETACHED_CAPTURE_SCHEMA_V1: Final[str] = (
    "eiel_cuda_full_evolution_v2_native_detached_checkpoint_capture_v1"
)
CUDA_NATIVE_DETACHED_RESULT_SCHEMA_V1: Final[str] = (
    "eiel_cuda_full_evolution_v2_native_detached_checkpoint_result_v1"
)
CUDA_NATIVE_CHECKPOINT_METADATA_SCHEMA_V1: Final[str] = (
    "eiel_absolute_tick_checkpoint_metadata_v1"
)
CUDA_NATIVE_CHECKPOINT_COMPLETION_SCHEMA_V1: Final[str] = (
    "eiel_absolute_tick_checkpoint_bundle_complete_v1"
)
CUDA_NATIVE_CHECKPOINT_SCHEMA_V2: Final[str] = (
    "eiel_full_evolution_cuda_checkpoint_v2"
)
_COPY_BUFFER_BYTES: Final[int] = 8 * 1024 * 1024
_FINGERPRINT_FIELDS: Final[tuple[str, ...]] = (
    "device",
    "inode",
    "mode",
    "nlink",
    "size",
    "mtime_ns",
    "ctime_ns",
)
_NATIVE_RUNTIME_CHECKPOINT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "time_authority",
        "contract",
        "canonical_state_shape",
        "canonical_state_nbytes",
        "canonical_state_sha256",
        "canonical_state_layout_sha256",
        "initial_next_absolute_tick",
        "advanced_ticks",
        "acknowledged_event_next_tick",
        "last_sink_receipt_sha256",
        "world_uids",
        "rng_keys",
        "rng_semantic_epoch",
        "artifact_sha256",
        "manifest_sha256",
        "device_identity",
        "population_fixture_id",
        "metadata_seal_sha256",
    }
)


class CudaFullEvolutionV2ACSError(RuntimeError):
    """The V2 ACS capture or host-only publication failed closed."""

    def __init__(
        self,
        message: str,
        *,
        publication_failure_path: str | None = None,
    ) -> None:
        super().__init__(message)
        self.publication_failure_path = publication_failure_path


class _DigestingPayloadSink(_PayloadSink):
    """Bounded file sink that seals bytes during the producer D2H stream."""

    def __init__(self, stream: Any) -> None:
        super().__init__(stream)
        self.digest = hashlib.sha256()

    def write(self, value: bytes | bytearray | memoryview) -> int:
        raw = bytes(value)
        self.digest.update(raw)
        return super().write(raw)


class CudaFullEvolutionV2CapturePoolV1:
    """One producer-owned reusable two-slab copier for a complete ACS route."""

    def __init__(
        self,
        *,
        chunk_bytes: int = streamed.DEFAULT_CHUNK_BYTES,
        has_cuda_fields: bool,
        require_native_pool: bool = True,
    ) -> None:
        self.chunk_bytes = _validate_chunk_bytes(chunk_bytes)
        if type(has_cuda_fields) is not bool:
            raise CudaFullEvolutionV2ACSError(
                "V2 ACS capture-pool device class must be an exact bool"
            )
        self.has_cuda_fields = has_cuda_fields
        self.require_native_pool = require_native_pool
        self._owner_thread = threading.get_ident()
        self._capture_count = 0
        self._copier = _SharedTwoSlabCopier(
            chunk_bytes=self.chunk_bytes,
            has_cuda_fields=has_cuda_fields,
            require_native_pool=require_native_pool,
        )
        telemetry = self._copier.telemetry()
        if require_native_pool and (
            telemetry.get("native_pool") is not True
            or telemetry.get("slab_count") != 2
            or telemetry.get("preallocated") is not True
            or has_cuda_fields and telemetry.get("pinned") is not True
        ):
            raise CudaFullEvolutionV2ACSError(
                "V2 ACS production capture lacks its native two-slab pool"
            )

    def copy_tensor(self, tensor: torch.Tensor, sink: _PayloadSink) -> None:
        if threading.get_ident() != self._owner_thread:
            raise CudaFullEvolutionV2ACSError(
                "V2 ACS capture pool crossed its producer thread"
            )
        if (tensor.device.type == "cuda") is not self.has_cuda_fields:
            raise CudaFullEvolutionV2ACSError(
                "V2 ACS capture pool device class changed"
            )
        self._copier.copy_tensor(tensor, sink)
        self._capture_count += 1

    def telemetry(self) -> dict[str, Any]:
        return {
            **self._copier.telemetry(),
            "route_capture_pool": True,
            "route_capture_count": self._capture_count,
            "owner_thread_id": self._owner_thread,
        }


def _raise_if_cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise CudaFullEvolutionV2ACSError(
            "CUDA Full Evolution V2 checkpoint publication was cancelled"
        )


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CudaFullEvolutionV2ACSError(
            "V2 ACS descriptor is not deterministic JSON"
        ) from exc


def _fingerprint(path: Path) -> dict[str, int]:
    try:
        value = path.lstat()
    except OSError as exc:
        raise CudaFullEvolutionV2ACSError(
            f"V2 ACS capture stat failed: {path}: {exc}"
        ) from exc
    if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
        raise CudaFullEvolutionV2ACSError(
            "V2 ACS capture must be one regular unaliased file"
        )
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "mode": int(value.st_mode),
        "nlink": int(value.st_nlink),
        "size": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
    }


def _validate_chunk_bytes(value: Any) -> int:
    if type(value) is not int or not 0 < value <= streamed.MAX_CHUNK_BYTES:
        raise CudaFullEvolutionV2ACSError(
            f"V2 ACS chunk_bytes must be in 1..{streamed.MAX_CHUNK_BYTES}"
        )
    return value


def _validate_destination(destination: str | Path) -> Path:
    raw = Path(destination).expanduser()
    if raw.is_symlink():
        raise CudaFullEvolutionV2ACSError("V2 ACS destination is a symlink")
    target = raw.resolve()
    if target.exists() or target.is_symlink():
        raise CudaFullEvolutionV2ACSError(
            "V2 ACS checkpoint destination must be fresh"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.is_symlink() or not target.parent.is_dir():
        raise CudaFullEvolutionV2ACSError(
            "V2 ACS checkpoint parent is not a stable directory"
        )
    return target


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _file_sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_COPY_BUFFER_BYTES), b""):
            digest.update(chunk)
            total += len(chunk)
    return digest.hexdigest(), total


def _read_small_file(path: Path, *, maximum_bytes: int = 2 * 1024 * 1024) -> bytes:
    """Read only bounded JSON control files during post-publication readback."""

    try:
        size = path.stat().st_size
    except OSError as exc:
        raise CudaFullEvolutionV2ACSError(
            f"V2 ACS control-file stat failed: {path}: {exc}"
        ) from exc
    if size < 0 or size > maximum_bytes:
        raise CudaFullEvolutionV2ACSError(
            f"V2 ACS control file exceeds the bounded readback size: {path}"
        )
    try:
        return path.read_bytes()
    except OSError as exc:
        raise CudaFullEvolutionV2ACSError(
            f"V2 ACS control-file read failed: {path}: {exc}"
        ) from exc


def _capture_authority_sha256(capture: Mapping[str, Any]) -> str:
    body = {
        str(key): value
        for key, value in capture.items()
        if key != "capture_authority_sha256"
    }
    return hashlib.sha256(_canonical_json(body)).hexdigest()


_COMPACT_SCIENCE_CONTINUATION_FRONTIER_SCHEMA_V1 = (
    "eiel_cuda_v2_compact_science_continuation_frontier_v1"
)


def _validated_compact_science_frontier_v1(
    value: Mapping[str, Any] | None,
    *,
    checkpoint_at_tick: int,
) -> dict[str, Any] | None:
    """Validate the compact continuation bound carried by an ACS2 capture."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise CudaFullEvolutionV2ACSError(
            "V2 ACS compact science frontier binding is not a mapping"
        )
    required = {
        "schema_version",
        "next_absolute_tick",
        "native_world_control_set_sha256",
        "counter_frontier",
        "dispatch_artifacts",
        "snapshot_artifacts",
        "frontier_sha256",
    }
    if set(value) != required:
        raise CudaFullEvolutionV2ACSError(
            "V2 ACS compact science frontier binding has the wrong fields"
        )
    body = {str(key): item for key, item in value.items() if key != "frontier_sha256"}
    frontier_digest = value.get("frontier_sha256")
    control_digest = value.get("native_world_control_set_sha256")
    if (
        value.get("schema_version")
        != _COMPACT_SCIENCE_CONTINUATION_FRONTIER_SCHEMA_V1
        or value.get("next_absolute_tick") != checkpoint_at_tick
        or not isinstance(control_digest, str)
        or len(control_digest) != 64
        or any(character not in "0123456789abcdef" for character in control_digest)
        or not isinstance(value.get("counter_frontier"), Mapping)
        or value["counter_frontier"].get("next_absolute_tick")
        != checkpoint_at_tick
        or not isinstance(value.get("dispatch_artifacts"), list)
        or not isinstance(value.get("snapshot_artifacts"), list)
        or not isinstance(frontier_digest, str)
        or len(frontier_digest) != 64
        or any(character not in "0123456789abcdef" for character in frontier_digest)
        or frontier_digest != hashlib.sha256(_canonical_json(body)).hexdigest()
    ):
        raise CudaFullEvolutionV2ACSError(
            "V2 ACS compact science frontier binding is inconsistent"
        )
    try:
        return json.loads(_canonical_json(dict(value)).decode("utf-8"))
    except (TypeError, ValueError) as exc:
        raise CudaFullEvolutionV2ACSError(
            "V2 ACS compact science frontier binding is not deterministic JSON"
        ) from exc


def _renameat2_noreplace_errno(source: Path, destination: Path) -> int:
    import ctypes
    import errno

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        return errno.ENOSYS
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(str(source)),
        -100,
        os.fsencode(str(destination)),
        1,
    )
    return 0 if result == 0 else int(ctypes.get_errno())


def _publish_directory_noreplace(source: Path, destination: Path) -> None:
    """Publish a bundle directory without replacing a competing final."""

    import errno

    error_number = _renameat2_noreplace_errno(source, destination)
    if error_number == 0:
        return
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(error_number, os.strerror(error_number), destination)
    if error_number not in {errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP}:
        raise OSError(error_number, os.strerror(error_number), destination)
    import fcntl

    descriptor = os.open(
        destination.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        if os.path.lexists(destination):
            raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), destination)
        os.rename(source, destination)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _workhorse_headers(
    path: Path,
    *,
    B: int,
    state_bytes_per_world: int,
) -> tuple[tuple[int, ...], ...]:
    headers: list[tuple[int, ...]] = []
    with path.open("rb") as handle:
        for world_index in range(B):
            handle.seek(world_index * state_bytes_per_world)
            raw = handle.read(32 * 8)
            if len(raw) != 32 * 8:
                raise CudaFullEvolutionV2ACSError(
                    "V2 ACS workhorse capture has a truncated world header"
                )
            headers.append(struct.unpack("<32Q", raw))
    return tuple(headers)


def capture_workhorse_checkpoint_to_file_v1(
    state: torch.Tensor,
    destination: str | Path,
    *,
    contract: FullEvolutionCudaWorkhorseContractV2,
    checkpoint_at_tick: int,
    artifact_path: str | Path,
    event_sink_receipt_path: str | Path,
    attempt_authority_binding: Mapping[str, Any],
    chunk_bytes: int = streamed.DEFAULT_CHUNK_BYTES,
    require_native_pool: bool = True,
    capture_pool: CudaFullEvolutionV2CapturePoolV1 | None = None,
) -> dict[str, Any]:
    """Capture one live canonical Workhorse state into inert file authority."""

    if type(contract) is not FullEvolutionCudaWorkhorseContractV2:
        raise CudaFullEvolutionV2ACSError(
            "V2 ACS workhorse capture requires the exact contract"
        )
    if type(checkpoint_at_tick) is not int or checkpoint_at_tick < 0:
        raise CudaFullEvolutionV2ACSError(
            "V2 ACS checkpoint frontier is invalid"
        )
    if not isinstance(state, torch.Tensor):
        raise CudaFullEvolutionV2ACSError("V2 ACS state source is not a tensor")
    state_bytes_per_world = workhorse_layout_v2(
        contract.shape
    ).state_bytes_per_world
    expected_shape = (contract.B, state_bytes_per_world)
    if (
        state.dtype != torch.uint8
        or tuple(int(value) for value in state.shape) != expected_shape
        or not state.is_contiguous()
        or state.device.type not in {"cpu", "cuda"}
    ):
        raise CudaFullEvolutionV2ACSError(
            "V2 ACS state source differs from the contiguous uint8 B/layout"
        )
    if type(require_native_pool) is not bool:
        raise CudaFullEvolutionV2ACSError(
            "V2 ACS native-pool requirement must be an exact bool"
        )
    chunk_bytes = _validate_chunk_bytes(chunk_bytes)
    target = _validate_destination(destination)
    artifact = Path(artifact_path).expanduser().resolve()
    event_receipt = Path(event_sink_receipt_path).expanduser().resolve()
    artifact_fingerprint = _fingerprint(artifact)
    event_receipt_fingerprint = _fingerprint(event_receipt)
    capture_path = target.parent / (
        f".{target.name}.{os.getpid()}.{uuid4().hex}.capture.tmp"
    )
    pool = capture_pool or CudaFullEvolutionV2CapturePoolV1(
        chunk_bytes=chunk_bytes,
        has_cuda_fields=state.device.type == "cuda",
        require_native_pool=require_native_pool,
    )
    if (
        pool.chunk_bytes != chunk_bytes
        or pool.require_native_pool is not require_native_pool
        or pool.has_cuda_fields is not (state.device.type == "cuda")
    ):
        raise CudaFullEvolutionV2ACSError(
            "V2 ACS workhorse capture pool differs from the route"
        )
    native_pool_telemetry = pool.telemetry()
    if require_native_pool and (
        native_pool_telemetry.get("native_pool") is not True
        or native_pool_telemetry.get("slab_count") != 2
        or native_pool_telemetry.get("preallocated") is not True
        or (
            state.device.type == "cuda"
            and native_pool_telemetry.get("pinned") is not True
        )
    ):
        raise CudaFullEvolutionV2ACSError(
            "V2 ACS production capture lacks the native pinned two-slab pool"
        )
    started = time.perf_counter()
    try:
        with capture_path.open("xb") as handle:
            sink = _PayloadSink(handle)
            pool.copy_tensor(state, sink)
            handle.flush()
        logical_bytes = int(state.numel()) * int(state.element_size())
        if sink.bytes != logical_bytes:
            raise CudaFullEvolutionV2ACSError(
                "V2 ACS state capture byte coverage changed"
            )
        capture_path.chmod(0o400)
        fingerprint = _fingerprint(capture_path)
        if fingerprint["size"] != logical_bytes:
            raise CudaFullEvolutionV2ACSError(
                "V2 ACS state capture size differs after close"
            )
        native_pool_telemetry = pool.telemetry()
        descriptor = {
            "schema_version": WORKHORSE_DETACHED_CAPTURE_SCHEMA_V1,
            "descriptor_status": "CAPTURED_HOST_ONLY",
            "time_authority": WORKHORSE_TIME_AUTHORITY_V2,
            "source_consumer_required": False,
            "capture_path": str(capture_path),
            "destination": str(target),
            "contract": contract.to_dict(),
            "checkpoint_at_tick": checkpoint_at_tick,
            "artifact_path": str(artifact),
            "artifact_file_fingerprint": artifact_fingerprint,
            "event_sink_receipt_path": str(event_receipt),
            "event_sink_receipt_file_fingerprint": event_receipt_fingerprint,
            "attempt_authority_binding": dict(attempt_authority_binding),
            "logical_bytes": logical_bytes,
            "captured_file_fingerprint": fingerprint,
            "chunk_bytes": chunk_bytes,
            "source_device_type": state.device.type,
            "capture_seconds": time.perf_counter() - started,
            "native_slab_pool": native_pool_telemetry,
            "capture_digest_deferred_to_host_worker": True,
        }
        descriptor["capture_authority_sha256"] = _capture_authority_sha256(
            descriptor
        )
        _canonical_json(descriptor)
        return descriptor
    except BaseException:
        if capture_path.exists() and not capture_path.is_symlink():
            failed = capture_path.with_name(
                f"{capture_path.name.removesuffix('.capture.tmp')}.capture.failed"
            )
            if failed.exists():
                failed = failed.with_name(f"{failed.name}.{uuid4().hex}")
            os.replace(capture_path, failed)
        raise


def _validate_workhorse_capture(
    capture: Mapping[str, Any],
) -> tuple[Path, Path, FullEvolutionCudaWorkhorseContractV2]:
    if capture.get("schema_version") != WORKHORSE_DETACHED_CAPTURE_SCHEMA_V1:
        raise CudaFullEvolutionV2ACSError(
            "V2 ACS workhorse capture schema differs"
        )
    if capture.get("time_authority") != WORKHORSE_TIME_AUTHORITY_V2:
        raise CudaFullEvolutionV2ACSError(
            "V2 ACS workhorse capture is not absolute-tick authoritative"
        )
    if capture.get("source_consumer_required") is not False:
        raise CudaFullEvolutionV2ACSError(
            "V2 ACS workhorse capture gained a source-owner lease"
        )
    if capture.get("descriptor_status") != "CAPTURED_HOST_ONLY":
        raise CudaFullEvolutionV2ACSError(
            "V2 ACS workhorse capture status is not host-only"
        )
    _canonical_json(capture)
    authority_digest = capture.get("capture_authority_sha256")
    if (
        not isinstance(authority_digest, str)
        or len(authority_digest) != 64
        or authority_digest != _capture_authority_sha256(capture)
    ):
        raise CudaFullEvolutionV2ACSError(
            "V2 ACS workhorse capture authority seal differs"
        )
    capture_path = Path(str(capture.get("capture_path") or "")).resolve()
    destination = Path(str(capture.get("destination") or "")).resolve()
    if (
        destination.exists()
        or destination.is_symlink()
        or capture_path.parent != destination.parent
    ):
        raise CudaFullEvolutionV2ACSError(
            "V2 ACS workhorse destination is not fresh and same-filesystem"
        )
    expected_fingerprint = capture.get("captured_file_fingerprint")
    if (
        not isinstance(expected_fingerprint, Mapping)
        or set(expected_fingerprint) != set(_FINGERPRINT_FIELDS)
        or any(type(expected_fingerprint.get(field)) is not int for field in _FINGERPRINT_FIELDS)
        or _fingerprint(capture_path) != dict(expected_fingerprint)
    ):
        raise CudaFullEvolutionV2ACSError(
            "V2 ACS workhorse capture file identity changed"
        )
    for path_field, fingerprint_field, label in (
        ("artifact_path", "artifact_file_fingerprint", "artifact"),
        (
            "event_sink_receipt_path",
            "event_sink_receipt_file_fingerprint",
            "event receipt",
        ),
    ):
        expected = capture.get(fingerprint_field)
        path = Path(str(capture.get(path_field) or "")).resolve()
        if (
            not isinstance(expected, Mapping)
            or set(expected) != set(_FINGERPRINT_FIELDS)
            or _fingerprint(path) != dict(expected)
        ):
            raise CudaFullEvolutionV2ACSError(
                f"V2 ACS workhorse {label} file identity changed"
            )
    try:
        contract = FullEvolutionCudaWorkhorseContractV2.from_mapping(
            capture.get("contract")
        )
    except (TypeError, FullEvolutionWorkhorseContractError) as exc:
        raise CudaFullEvolutionV2ACSError(
            f"V2 ACS workhorse contract is invalid: {exc}"
        ) from exc
    expected_bytes = contract.B * workhorse_layout_v2(
        contract.shape
    ).state_bytes_per_world
    if (
        capture.get("logical_bytes") != expected_bytes
        or expected_fingerprint.get("size") != expected_bytes
        or type(capture.get("checkpoint_at_tick")) is not int
    ):
        raise CudaFullEvolutionV2ACSError(
            "V2 ACS workhorse byte/frontier binding differs"
        )
    return capture_path, destination, contract


def finalize_workhorse_checkpoint_capture_v1(
    capture: Mapping[str, Any],
    *,
    cancel_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Durably publish the unchanged Workhorse bundle on a host-only worker."""

    _raise_if_cancelled(cancel_event)
    capture_path, destination, contract = _validate_workhorse_capture(capture)
    started = time.perf_counter()
    state_sha256, state_nbytes = _file_sha256(capture_path)
    _raise_if_cancelled(cancel_event)
    headers = _workhorse_headers(
        capture_path,
        B=contract.B,
        state_bytes_per_world=workhorse_layout_v2(
            contract.shape
        ).state_bytes_per_world,
    )
    try:
        metadata = workhorse_checkpoint_metadata_from_state_identity_v2(
            contract=contract,
            checkpoint_at_tick=int(capture["checkpoint_at_tick"]),
            state_nbytes=state_nbytes,
            state_sha256=state_sha256,
            state_headers=headers,
            artifact_path=str(capture["artifact_path"]),
            event_sink_receipt_path=str(capture["event_sink_receipt_path"]),
            attempt_authority_binding=capture["attempt_authority_binding"],
        )
    except FullEvolutionWorkhorseAuthorityError as exc:
        raise CudaFullEvolutionV2ACSError(
            f"V2 ACS workhorse checkpoint authority rejected capture: {exc}"
        ) from exc
    _raise_if_cancelled(cancel_event)

    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            suffix=".bundle.tmp",
            dir=destination.parent,
        )
    )
    publication_created = False
    failed_publication: Path | None = None
    try:
        state_path = stage / "canonical_state.bin"
        os.link(capture_path, state_path, follow_symlinks=False)
        state_path.chmod(0o444)
        descriptor = os.open(state_path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        metadata_path = stage / "checkpoint.json"
        with metadata_path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(metadata, sort_keys=True, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        metadata_path.chmod(0o444)
        _fsync_directory(stage)
        _raise_if_cancelled(cancel_event)
        if destination.exists() or destination.is_symlink():
            raise CudaFullEvolutionV2ACSError(
                "V2 ACS workhorse destination appeared during publication"
            )
        _publish_directory_noreplace(stage, destination)
        publication_created = True
        destination.chmod(0o555)
        _fsync_directory(destination.parent)
        readback_sha256, readback_nbytes = _file_sha256(
            destination / "canonical_state.bin"
        )
        readback_headers = _workhorse_headers(
            destination / "canonical_state.bin",
            B=contract.B,
            state_bytes_per_world=workhorse_layout_v2(
                contract.shape
            ).state_bytes_per_world,
        )
        if (
            readback_sha256 != state_sha256
            or readback_nbytes != state_nbytes
            or readback_headers != headers
        ):
            raise CudaFullEvolutionV2ACSError(
                "V2 ACS workhorse durable state readback differs"
            )
        expected_metadata_bytes = (
            json.dumps(metadata, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8")
        if _read_small_file(destination / "checkpoint.json") != expected_metadata_bytes:
            raise CudaFullEvolutionV2ACSError(
                "V2 ACS workhorse durable metadata readback differs"
            )
        _raise_if_cancelled(cancel_event)
        capture_path.unlink()
        _fsync_directory(destination.parent)
    except BaseException as exc:
        if stage.exists() and not stage.is_symlink():
            shutil.rmtree(stage)
        if publication_created and destination.exists() and not destination.is_symlink():
            failed_publication = destination.with_name(
                f"{destination.name}.publication.failed.{uuid4().hex}"
            )
            os.replace(destination, failed_publication)
            _fsync_directory(destination.parent)
        if failed_publication is not None:
            raise CudaFullEvolutionV2ACSError(
                "V2 ACS workhorse publication was quarantined after failure",
                publication_failure_path=str(failed_publication),
            ) from exc
        raise

    return {
        "schema_version": WORKHORSE_DETACHED_RESULT_SCHEMA_V1,
        "status": "complete",
        "path": str(destination),
        "manifest_path": str(destination / "checkpoint.json"),
        "state_path": str(destination / "canonical_state.bin"),
        "checkpoint_at_tick": int(capture["checkpoint_at_tick"]),
        "canonical_state_bytes": state_nbytes,
        "canonical_state_sha256": state_sha256,
        "bundle_body_sha256": metadata["bundle_body_sha256"],
        "capture_seconds": float(capture.get("capture_seconds", 0.0)),
        "detached_finalize_seconds": time.perf_counter() - started,
        "native_slab_pool": dict(capture.get("native_slab_pool") or {}),
        "worker_state_materialization": "stream_hash_and_256_byte_world_headers_v1",
        "worker_full_state_bytes_materialized": 0,
        "worker_cuda_references": 0,
        "_detached_publication_created_by_worker": True,
    }


class CudaFullEvolutionV2WorkhorseACSCoordinatorV1:
    """Producer-side FIFO coordinator shared by every Workhorse V2 shape."""

    def __init__(
        self,
        *,
        slots: int,
        checkpoint_root: str | Path,
        artifact_path: str | Path,
        attempt_authority_binding: Mapping[str, Any],
        chunk_bytes: int = streamed.DEFAULT_CHUNK_BYTES,
        require_native_pool: bool = True,
    ) -> None:
        from .acs_native_detached_slots_v1 import ACSNativeDetachedSlotsV1

        self.slots = slots
        self.checkpoint_root = Path(checkpoint_root).expanduser().resolve()
        if self.checkpoint_root.exists() or self.checkpoint_root.is_symlink():
            raise CudaFullEvolutionV2ACSError(
                "V2 Workhorse ACS checkpoint root must be fresh"
            )
        self.checkpoint_root.mkdir(parents=True)
        self.artifact_path = Path(artifact_path).expanduser().resolve()
        self.attempt_authority_binding = json.loads(
            _canonical_json(attempt_authority_binding).decode("utf-8")
        )
        self.chunk_bytes = _validate_chunk_bytes(chunk_bytes)
        self.require_native_pool = require_native_pool
        self._pipeline = ACSNativeDetachedSlotsV1(slots=slots)
        self._capture_pool: CudaFullEvolutionV2CapturePoolV1 | None = None
        self._pending: deque[Any] = deque()
        self._receipts: list[dict[str, Any]] = []
        self._settled_by_sequence: dict[int, dict[str, Any]] = {}
        self._acknowledged_sequences: set[int] = set()
        self._closed = False

    def capture_workhorse_checkpoint(
        self,
        state: torch.Tensor,
        *,
        contract: FullEvolutionCudaWorkhorseContractV2,
        checkpoint_at_tick: int,
        event_sink_receipt_path: str | Path,
    ) -> Any:
        if self._closed:
            raise CudaFullEvolutionV2ACSError(
                "V2 Workhorse ACS coordinator is closed"
            )
        self.settle_ready()
        while len(self._pending) >= self.slots:
            self._settle_oldest(producer_backpressure=True)
            raise CudaFullEvolutionV2ACSError(
                "V2 Workhorse ACS capacity requires the FIFO head to complete "
                "its external occurrence/lifecycle commit and acknowledgement"
            )
        if self._capture_pool is None:
            self._capture_pool = CudaFullEvolutionV2CapturePoolV1(
                chunk_bytes=self.chunk_bytes,
                has_cuda_fields=state.device.type == "cuda",
                require_native_pool=self.require_native_pool,
            )
        token = self._pipeline.begin_capture(compute_frontier=checkpoint_at_tick)
        try:
            capture = capture_workhorse_checkpoint_to_file_v1(
                state,
                self.checkpoint_root
                / f"checkpoint-next-tick-{checkpoint_at_tick:012d}",
                contract=contract,
                checkpoint_at_tick=checkpoint_at_tick,
                artifact_path=self.artifact_path,
                event_sink_receipt_path=event_sink_receipt_path,
                attempt_authority_binding=self.attempt_authority_binding,
                chunk_bytes=self.chunk_bytes,
                require_native_pool=self.require_native_pool,
                capture_pool=self._capture_pool,
            )
            self._pipeline.submit_capture(token, capture)
        except BaseException as exc:
            self._pipeline.fail_capture(token, exc)
            raise
        self._pending.append(token)
        return token

    def _settle_oldest(self, *, producer_backpressure: bool) -> dict[str, Any]:
        if not self._pending:
            raise CudaFullEvolutionV2ACSError(
                "V2 Workhorse ACS settlement has no FIFO head"
            )
        token = self._pending[0]
        existing = self._settled_by_sequence.get(token.sequence)
        if existing is not None:
            return dict(existing)
        result = dict(
            self._pipeline.result(
                token, producer_backpressure=producer_backpressure
            )
        )
        self._settled_by_sequence[token.sequence] = result
        return result

    def settle_ready(self) -> tuple[dict[str, Any], ...]:
        if (
            self._pending
            and self._pending[0].sequence not in self._settled_by_sequence
            and self._pipeline.ready(self._pending[0])
        ):
            return (self._settle_oldest(producer_backpressure=False),)
        return ()

    def settle_checkpoint(self, token: Any) -> dict[str, Any]:
        sequence = getattr(token, "sequence", -1)
        existing = self._settled_by_sequence.get(sequence)
        if existing is not None:
            return dict(existing)
        if self._pending and self._pending[0] == token:
            return self._settle_oldest(producer_backpressure=True)
        raise CudaFullEvolutionV2ACSError(
            "V2 Workhorse ACS settlement is FIFO and requires the prior "
            "checkpoint's external commit/acknowledgement"
        )

    def acknowledge_checkpoint(self, token: Any) -> None:
        sequence = getattr(token, "sequence", -1)
        if sequence in self._acknowledged_sequences:
            raise CudaFullEvolutionV2ACSError(
                "V2 Workhorse ACS checkpoint was acknowledged twice"
            )
        if not self._pending or self._pending[0] != token:
            raise CudaFullEvolutionV2ACSError(
                "V2 Workhorse ACS acknowledgement is not the FIFO head"
            )
        result = self._settled_by_sequence.get(sequence)
        if result is None:
            raise CudaFullEvolutionV2ACSError(
                "V2 Workhorse ACS acknowledgement precedes detached settlement"
            )
        self._pipeline.acknowledge_durable(token)
        self._pending.popleft()
        self._receipts.append(dict(result))
        self._acknowledged_sequences.add(sequence)

    def pending_tokens(self) -> tuple[Any, ...]:
        return tuple(self._pending)

    def close(self) -> None:
        if self._closed:
            return
        if self._pending:
            raise CudaFullEvolutionV2ACSError(
                "V2 Workhorse ACS cannot close before every detached result "
                "completes its external commit and acknowledgement"
            )
        self._pipeline.close()
        self._closed = True

    def abort(self) -> None:
        if self._closed:
            return
        self._pipeline.abort()
        self._pending.clear()
        self._closed = True

    def telemetry(self) -> dict[str, Any]:
        return {
            **self._pipeline.telemetry(),
            "capture_mode": "direct_resident_cuda_state_v1",
            "capture_pool": (
                self._capture_pool.telemetry()
                if self._capture_pool is not None
                else None
            ),
            "checkpoint_receipts": [dict(row) for row in self._receipts],
            "settled_unacknowledged_checkpoint_count": sum(
                1
                for token in self._pending
                if token.sequence in self._settled_by_sequence
            ),
        }


def reopen_workhorse_checkpoint_set_v1(
    metadata_path: str | Path,
    *,
    require_current_source_match: bool = True,
) -> tuple[dict[str, Any], bytes]:
    """Reopen the exact complete logical world set through canonical authority."""

    try:
        return reopen_workhorse_checkpoint_bundle_v2(
            metadata_path,
            require_current_source_match=require_current_source_match,
        )
    except FullEvolutionWorkhorseAuthorityError as exc:
        raise CudaFullEvolutionV2ACSError(
            f"V2 Workhorse full-set checkpoint cannot reopen: {exc}"
        ) from exc


def reopen_workhorse_checkpoint_world_v1(
    metadata_path: str | Path,
    *,
    world_index: int | None = None,
    world_uid: int | None = None,
    require_current_source_match: bool = True,
) -> dict[str, Any]:
    """Reopen one world with streaming aggregate validation and one row read."""

    try:
        metadata, state_path = inspect_workhorse_checkpoint_bundle_v2(
            metadata_path,
            require_current_source_match=require_current_source_match,
        )
        contract = FullEvolutionCudaWorkhorseContractV2.from_mapping(
            metadata["contract"]
        )
    except (FullEvolutionWorkhorseAuthorityError, FullEvolutionWorkhorseContractError) as exc:
        raise CudaFullEvolutionV2ACSError(
            f"V2 Workhorse checkpoint cannot be selected: {exc}"
        ) from exc
    uids = metadata.get("world_uids")
    keys = metadata.get("rng_keys")
    if not isinstance(uids, list) or not isinstance(keys, list):
        raise CudaFullEvolutionV2ACSError(
            "V2 Workhorse checkpoint world identity is absent"
        )
    if (world_index is None) == (world_uid is None):
        raise CudaFullEvolutionV2ACSError(
            "select exactly one Workhorse checkpoint world index or UID"
        )
    if world_uid is not None:
        if type(world_uid) is not int or world_uid not in uids:
            raise CudaFullEvolutionV2ACSError(
                "selected Workhorse checkpoint world UID is absent"
            )
        selected_index = uids.index(world_uid)
    else:
        if type(world_index) is not int or not 0 <= world_index < contract.B:
            raise CudaFullEvolutionV2ACSError(
                "selected Workhorse checkpoint world index is outside B"
            )
        selected_index = world_index
    row_bytes = workhorse_layout_v2(contract.shape).state_bytes_per_world
    with state_path.open("rb") as handle:
        handle.seek(selected_index * row_bytes)
        state = handle.read(row_bytes)
    if len(state) != row_bytes:
        raise CudaFullEvolutionV2ACSError(
            "selected Workhorse checkpoint world row is truncated"
        )
    state_sha256 = hashlib.sha256(state).hexdigest()
    selected_contract = FullEvolutionCudaWorkhorseContractV2(
        shape_id=contract.shape_id,
        backend_id=contract.backend_id,
        B=1,
        temporal_k=contract.temporal_k,
        total_ticks=contract.total_ticks,
        start_next_tick=contract.start_next_tick,
    )
    projection_body = {
        "schema_version": "eiel_cuda_full_evolution_v2_workhorse_world_restore_v1",
        "time_authority": WORKHORSE_TIME_AUTHORITY_V2,
        "source_checkpoint_identity_sha256": metadata["checkpoint_identity"][
            "checkpoint_identity_sha256"
        ],
        "source_aggregate_state_sha256": metadata["canonical_state"]["sha256"],
        "source_world_count": contract.B,
        "selected_world_index": selected_index,
        "selected_world_uid": uids[selected_index],
        "selected_rng_key": keys[selected_index],
        "selected_state_offset": selected_index * row_bytes,
        "selected_state_nbytes": row_bytes,
        "selected_state_sha256": state_sha256,
        "next_absolute_tick": metadata["checkpoint_at_tick"],
        "selected_contract": selected_contract.to_dict(),
    }
    return {
        **projection_body,
        "canonical_state": state,
        "projection_sha256": hashlib.sha256(
            _canonical_json(projection_body)
        ).hexdigest(),
    }


def _native_checkpoint_id_v1(*, checkpoint_at_tick: int, state_sha256: str) -> str:
    if type(checkpoint_at_tick) is not int or checkpoint_at_tick < 0:
        raise CudaFullEvolutionV2ACSError(
            "V2 native checkpoint frontier is invalid"
        )
    if len(state_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in state_sha256
    ):
        raise CudaFullEvolutionV2ACSError("V2 native checkpoint state digest is invalid")
    return (
        f"cuda-native-checkpoint-tick-{checkpoint_at_tick:012d}-"
        f"{state_sha256[:16]}"
    )


def _native_checkpoint_from_runtime_v1(
    runtime_checkpoint: Mapping[str, Any],
    *,
    canonical_state: bytes,
    artifact_path: str | Path,
) -> CudaFullEvolutionCheckpointV2:
    if set(runtime_checkpoint) != _NATIVE_RUNTIME_CHECKPOINT_FIELDS:
        raise CudaFullEvolutionV2ACSError(
            "V2 native runtime checkpoint fields are inexact"
        )
    try:
        contract = CudaFullEvolutionNativeContractV2.from_mapping(
            runtime_checkpoint["contract"]
        )
        checkpoint = CudaFullEvolutionCheckpointV2(
            contract=contract,
            canonical_state=canonical_state,
            canonical_state_shape=tuple(runtime_checkpoint["canonical_state_shape"]),
            canonical_state_nbytes=int(runtime_checkpoint["canonical_state_nbytes"]),
            canonical_state_sha256=str(runtime_checkpoint["canonical_state_sha256"]),
            canonical_state_layout_sha256=str(
                runtime_checkpoint["canonical_state_layout_sha256"]
            ),
            initial_next_absolute_tick=int(
                runtime_checkpoint["initial_next_absolute_tick"]
            ),
            advanced_ticks=int(runtime_checkpoint["advanced_ticks"]),
            acknowledged_event_next_tick=int(
                runtime_checkpoint["acknowledged_event_next_tick"]
            ),
            last_sink_receipt_sha256=runtime_checkpoint[
                "last_sink_receipt_sha256"
            ],
            world_uids=tuple(runtime_checkpoint["world_uids"]),
            rng_keys=tuple(runtime_checkpoint["rng_keys"]),
            rng_semantic_epoch=int(runtime_checkpoint["rng_semantic_epoch"]),
            artifact_path=Path(artifact_path).expanduser().resolve(),
            artifact_sha256=str(runtime_checkpoint["artifact_sha256"]),
            manifest_sha256=str(runtime_checkpoint["manifest_sha256"]),
            device_identity=dict(runtime_checkpoint["device_identity"]),
            population_fixture_id=runtime_checkpoint["population_fixture_id"],
            metadata_seal_sha256=str(runtime_checkpoint["metadata_seal_sha256"]),
            schema_version=str(runtime_checkpoint["schema_version"]),
            time_authority=str(runtime_checkpoint["time_authority"]),
        )
        _validate_checkpoint_envelope(checkpoint)
    except Exception as exc:
        raise CudaFullEvolutionV2ACSError(
            f"V2 native checkpoint envelope cannot reopen: {exc}"
        ) from exc
    return checkpoint


def _reopen_native_bundle_controls_v1(
    bundle: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, Any], Path]:
    root = Path(bundle).expanduser()
    if root.is_symlink() or not root.is_dir():
        raise CudaFullEvolutionV2ACSError(
            "V2 native checkpoint bundle is not a stable directory"
        )
    root = root.resolve()
    metadata_path = root / "metadata.json"
    completion_path = root / "bundle_complete.json"
    state_path = root / "canonical_state.bin"
    for path in (metadata_path, completion_path, state_path):
        if path.is_symlink() or not path.is_file():
            raise CudaFullEvolutionV2ACSError(
                "V2 native checkpoint bundle inventory is incomplete"
            )
    try:
        metadata_bytes = _read_small_file(metadata_path)
        completion_bytes = _read_small_file(completion_path)
        metadata = json.loads(metadata_bytes)
        completion = json.loads(completion_bytes)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CudaFullEvolutionV2ACSError(
            f"V2 native checkpoint controls cannot reopen: {exc}"
        ) from exc
    if not isinstance(metadata, dict) or not isinstance(completion, dict):
        raise CudaFullEvolutionV2ACSError(
            "V2 native checkpoint controls are not objects"
        )
    runtime_checkpoint = metadata.get("runtime_checkpoint")
    if (
        metadata.get("schema_version") != CUDA_NATIVE_CHECKPOINT_METADATA_SCHEMA_V1
        or metadata.get("checkpoint_format_id") != CUDA_NATIVE_CHECKPOINT_SCHEMA_V2
        or metadata.get("time_authority") != "absolute_tick_v1"
        or not isinstance(runtime_checkpoint, dict)
        or completion.get("schema_version")
        != CUDA_NATIVE_CHECKPOINT_COMPLETION_SCHEMA_V1
        or completion.get("status") != "complete"
        or completion.get("checkpoint_format_id")
        != CUDA_NATIVE_CHECKPOINT_SCHEMA_V2
        or completion.get("time_authority") != "absolute_tick_v1"
        or completion.get("canonical_state_file") != "canonical_state.bin"
        or completion.get("metadata_sha256")
        != hashlib.sha256(metadata_bytes).hexdigest()
        or completion.get("checkpoint_id") != metadata.get("checkpoint_id")
        or completion.get("absolute_next_tick")
        != metadata.get("absolute_next_tick")
        or metadata.get("absolute_next_tick")
        != runtime_checkpoint.get("acknowledged_event_next_tick")
    ):
        raise CudaFullEvolutionV2ACSError(
            "V2 native checkpoint control identity differs"
        )
    state_sha256, state_nbytes = _file_sha256(state_path)
    if (
        state_nbytes != completion.get("canonical_state_nbytes")
        or state_sha256 != completion.get("canonical_state_sha256")
        or state_nbytes != runtime_checkpoint.get("canonical_state_nbytes")
        or state_sha256 != runtime_checkpoint.get("canonical_state_sha256")
        or metadata.get("checkpoint_id")
        != _native_checkpoint_id_v1(
            checkpoint_at_tick=int(metadata["absolute_next_tick"]),
            state_sha256=state_sha256,
        )
    ):
        raise CudaFullEvolutionV2ACSError(
            "V2 native checkpoint state identity differs"
        )
    return root, metadata, completion, state_path


def reopen_native_checkpoint_bundle_v1(
    bundle: str | Path,
) -> CudaFullEvolutionCheckpointV2:
    """Reopen and validate the complete B-world native ACS2 checkpoint."""

    _root, metadata, _completion, state_path = _reopen_native_bundle_controls_v1(
        bundle
    )
    state = state_path.read_bytes()
    return _native_checkpoint_from_runtime_v1(
        metadata["runtime_checkpoint"],
        canonical_state=state,
        artifact_path=metadata["artifact_path"],
    )


def reopen_native_checkpoint_world_v1(
    bundle: str | Path,
    *,
    world_index: int | None = None,
    world_uid: int | None = None,
) -> tuple[CudaFullEvolutionCheckpointV2, dict[str, Any]]:
    """Derive one B1 fresh-owner checkpoint by a bounded row read."""

    root, metadata, _completion, state_path = _reopen_native_bundle_controls_v1(
        bundle
    )
    runtime = dict(metadata["runtime_checkpoint"])
    shape = runtime.get("canonical_state_shape")
    uids = runtime.get("world_uids")
    keys = runtime.get("rng_keys")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or not isinstance(uids, list)
        or not isinstance(keys, list)
        or len(uids) != shape[0]
        or len(keys) != shape[0]
    ):
        raise CudaFullEvolutionV2ACSError(
            "V2 native checkpoint world index is malformed"
        )
    if (world_index is None) == (world_uid is None):
        raise CudaFullEvolutionV2ACSError(
            "select exactly one native checkpoint world index or UID"
        )
    if world_uid is not None:
        if type(world_uid) is not int or world_uid not in uids:
            raise CudaFullEvolutionV2ACSError(
                "selected native checkpoint world UID is absent"
            )
        selected_index = uids.index(world_uid)
    else:
        if type(world_index) is not int or not 0 <= world_index < len(uids):
            raise CudaFullEvolutionV2ACSError(
                "selected native checkpoint world index is outside B"
            )
        selected_index = world_index
    row_bytes = int(shape[1])
    with state_path.open("rb") as handle:
        handle.seek(selected_index * row_bytes)
        selected_state = handle.read(row_bytes)
    if len(selected_state) != row_bytes:
        raise CudaFullEvolutionV2ACSError(
            "selected native checkpoint world row is truncated"
        )
    selected_sha = hashlib.sha256(selected_state).hexdigest()
    contract_body = dict(runtime["contract"])
    contract_body["B"] = 1
    selected_contract = CudaFullEvolutionNativeContractV2.from_mapping(contract_body)
    selected_runtime = {
        **runtime,
        "contract": selected_contract.to_dict(),
        "canonical_state_shape": [1, row_bytes],
        "canonical_state_nbytes": row_bytes,
        "canonical_state_sha256": selected_sha,
        "canonical_state_layout_sha256": hashlib.sha256(
            _canonical_json(
                {
                    "dtype": "uint8",
                    "shape": [1, row_bytes],
                    "nbytes": row_bytes,
                    "content_sha256": selected_sha,
                }
            )
        ).hexdigest(),
        "world_uids": [uids[selected_index]],
        "rng_keys": [keys[selected_index]],
    }
    selected_runtime["metadata_seal_sha256"] = hashlib.sha256(
        _canonical_json(
            {
                key: value
                for key, value in selected_runtime.items()
                if key != "metadata_seal_sha256"
            }
        )
    ).hexdigest()
    checkpoint = _native_checkpoint_from_runtime_v1(
        selected_runtime,
        canonical_state=selected_state,
        artifact_path=metadata["artifact_path"],
    )
    projection_body = {
        "schema_version": "eiel_cuda_full_evolution_v2_world_restore_projection_v1",
        "time_authority": "absolute_tick_v1",
        "source_bundle": str(root),
        "source_checkpoint_id": metadata["checkpoint_id"],
        "source_aggregate_state_sha256": runtime["canonical_state_sha256"],
        "source_world_count": shape[0],
        "selected_world_index": selected_index,
        "selected_world_uid": uids[selected_index],
        "selected_rng_key": keys[selected_index],
        "selected_state_offset": selected_index * row_bytes,
        "selected_state_nbytes": row_bytes,
        "selected_state_sha256": selected_sha,
        "next_absolute_tick": runtime["acknowledged_event_next_tick"],
    }
    projection = {
        **projection_body,
        "projection_sha256": hashlib.sha256(
            _canonical_json(projection_body)
        ).hexdigest(),
    }
    return checkpoint, projection


def capture_native_resident_checkpoint_to_file_v1(
    source: CudaFullEvolutionCheckpointCaptureSourceV1,
    destination: str | Path,
    *,
    artifact_path: str | Path,
    checkpoint_at_tick: int,
    compact_science_frontier: Mapping[str, Any] | None = None,
    chunk_bytes: int = streamed.DEFAULT_CHUNK_BYTES,
    require_native_pool: bool = True,
    capture_pool: CudaFullEvolutionV2CapturePoolV1 | None = None,
) -> dict[str, Any]:
    """Capture canonical rows directly from the resident CUDA allocation.

    Each world's canonical prefix is contiguous even though the aggregate
    B-by-canonical-width view is strided. Row capture avoids both a compacted
    device clone and a full resident-state CPU projection.
    """

    if type(source) is not CudaFullEvolutionCheckpointCaptureSourceV1:
        raise CudaFullEvolutionV2ACSError(
            "V2 native direct capture requires the exact producer source"
        )
    if (
        type(checkpoint_at_tick) is not int
        or checkpoint_at_tick < 0
        or checkpoint_at_tick != source.acknowledged_event_next_tick
    ):
        raise CudaFullEvolutionV2ACSError(
            "V2 native direct-capture frontier differs"
        )
    validated_compact_frontier = _validated_compact_science_frontier_v1(
        compact_science_frontier,
        checkpoint_at_tick=checkpoint_at_tick,
    )
    state = source.resident_state
    canonical_row_bytes = source.canonical_bytes_per_world
    if (
        not isinstance(state, torch.Tensor)
        or state.dtype != torch.uint8
        or not state.is_cuda
        or not state.is_contiguous()
        or len(state.shape) != 2
        or int(state.shape[0]) != source.contract.B
        or type(canonical_row_bytes) is not int
        or not 0 < canonical_row_bytes <= int(state.shape[1])
    ):
        raise CudaFullEvolutionV2ACSError(
            "V2 native direct-capture source is not contiguous resident CUDA state"
        )
    try:
        checkpoint_runtime_projection_from_capture_source_v1(
            source,
            canonical_state_sha256="0" * 64,
        )
    except Exception as exc:
        raise CudaFullEvolutionV2ACSError(
            f"V2 native direct-capture source authority is invalid: {exc}"
        ) from exc
    artifact = Path(artifact_path).expanduser().resolve()
    if artifact != source.artifact_path.expanduser().resolve():
        raise CudaFullEvolutionV2ACSError(
            "V2 native direct-capture artifact path differs"
        )
    artifact_fingerprint = _fingerprint(artifact)
    chunk_bytes = _validate_chunk_bytes(chunk_bytes)
    target = _validate_destination(destination)
    capture_path = target.parent / (
        f".{target.name}.{os.getpid()}.{uuid4().hex}.capture.tmp"
    )
    pool = capture_pool or CudaFullEvolutionV2CapturePoolV1(
        chunk_bytes=chunk_bytes,
        has_cuda_fields=True,
        require_native_pool=require_native_pool,
    )
    if (
        pool.chunk_bytes != chunk_bytes
        or pool.require_native_pool is not require_native_pool
        or pool.has_cuda_fields is not True
    ):
        raise CudaFullEvolutionV2ACSError(
            "V2 native direct-capture pool differs from the CUDA route"
        )
    native_pool_telemetry = pool.telemetry()
    if require_native_pool and (
        native_pool_telemetry.get("native_pool") is not True
        or native_pool_telemetry.get("slab_count") != 2
        or native_pool_telemetry.get("preallocated") is not True
        or native_pool_telemetry.get("pinned") is not True
    ):
        raise CudaFullEvolutionV2ACSError(
            "V2 native direct capture lacks the pinned two-slab pool"
        )
    logical_bytes = source.contract.B * canonical_row_bytes
    started = time.perf_counter()
    try:
        with capture_path.open("xb") as handle:
            sink = _DigestingPayloadSink(handle)
            for world_index in range(source.contract.B):
                row = state[world_index, :canonical_row_bytes]
                if not row.is_contiguous():
                    raise CudaFullEvolutionV2ACSError(
                        "V2 native canonical resident row is not contiguous"
                    )
                pool.copy_tensor(row, sink)
            handle.flush()
        if sink.bytes != logical_bytes:
            raise CudaFullEvolutionV2ACSError(
                "V2 native direct state capture byte coverage changed"
            )
        state_sha256 = sink.digest.hexdigest()
        try:
            runtime_checkpoint = checkpoint_runtime_projection_from_capture_source_v1(
                source,
                canonical_state_sha256=state_sha256,
            )
        except Exception as exc:
            raise CudaFullEvolutionV2ACSError(
                f"V2 native direct-capture authority is invalid: {exc}"
            ) from exc
        capture_path.chmod(0o400)
        fingerprint = _fingerprint(capture_path)
        if fingerprint["size"] != logical_bytes:
            raise CudaFullEvolutionV2ACSError(
                "V2 native direct state capture size differs after close"
            )
        native_pool_telemetry = pool.telemetry()
        descriptor = {
            "schema_version": CUDA_NATIVE_DETACHED_CAPTURE_SCHEMA_V1,
            "descriptor_status": "CAPTURED_HOST_ONLY",
            "time_authority": "absolute_tick_v1",
            "source_consumer_required": False,
            "capture_path": str(capture_path),
            "destination": str(target),
            "artifact_path": str(artifact),
            "artifact_file_fingerprint": artifact_fingerprint,
            "checkpoint_at_tick": checkpoint_at_tick,
            "checkpoint_id": _native_checkpoint_id_v1(
                checkpoint_at_tick=checkpoint_at_tick,
                state_sha256=state_sha256,
            ),
            "runtime_checkpoint": runtime_checkpoint,
            "logical_bytes": logical_bytes,
            "captured_file_fingerprint": fingerprint,
            "chunk_bytes": chunk_bytes,
            "source_device_type": "cuda_resident_canonical_rows_v1",
            "canonical_row_count": source.contract.B,
            "capture_seconds": time.perf_counter() - started,
            "native_slab_pool": native_pool_telemetry,
            "capture_digest_deferred_to_host_worker": False,
            "producer_full_state_host_bytes_materialized": 0,
            "producer_compacted_cuda_bytes_materialized": 0,
            "worker_cuda_reference_allowed": False,
        }
        if validated_compact_frontier is not None:
            descriptor["compact_science_frontier"] = validated_compact_frontier
        descriptor["capture_authority_sha256"] = _capture_authority_sha256(
            descriptor
        )
        _canonical_json(descriptor)
        return descriptor
    except BaseException:
        if capture_path.exists() and not capture_path.is_symlink():
            failed = capture_path.with_name(
                f"{capture_path.name.removesuffix('.capture.tmp')}.capture.failed"
            )
            if failed.exists():
                failed = failed.with_name(f"{failed.name}.{uuid4().hex}")
            os.replace(capture_path, failed)
        raise


def capture_native_checkpoint_to_file_v1(
    checkpoint: Any,
    destination: str | Path,
    *,
    runtime_checkpoint: Mapping[str, Any],
    artifact_path: str | Path,
    checkpoint_at_tick: int,
    compact_science_frontier: Mapping[str, Any] | None = None,
    chunk_bytes: int = streamed.DEFAULT_CHUNK_BYTES,
    require_native_pool: bool = True,
    capture_pool: CudaFullEvolutionV2CapturePoolV1 | None = None,
) -> dict[str, Any]:
    """Capture an already-sealed host checkpoint for non-production test doubles.

    Real CUDA owners use :func:`capture_native_resident_checkpoint_to_file_v1`.
    This compatibility seam remains for injected host-only provider tests; it
    never constitutes production ACS evidence.
    """

    if not isinstance(runtime_checkpoint, Mapping):
        raise CudaFullEvolutionV2ACSError(
            "V2 native runtime checkpoint metadata is not a mapping"
        )
    if type(checkpoint_at_tick) is not int or checkpoint_at_tick < 0:
        raise CudaFullEvolutionV2ACSError(
            "V2 native checkpoint frontier is invalid"
        )
    validated_compact_frontier = _validated_compact_science_frontier_v1(
        compact_science_frontier,
        checkpoint_at_tick=checkpoint_at_tick,
    )
    state_bytes = getattr(checkpoint, "canonical_state", None)
    state_shape = getattr(checkpoint, "canonical_state_shape", None)
    if type(state_bytes) is not bytes or not isinstance(state_shape, tuple):
        raise CudaFullEvolutionV2ACSError(
            "V2 native checkpoint canonical state is not immutable bytes"
        )
    if any(type(value) is not int or value < 0 for value in state_shape):
        raise CudaFullEvolutionV2ACSError(
            "V2 native checkpoint canonical state shape is invalid"
        )
    if len(state_shape) != 2 or len(state_bytes) != state_shape[0] * state_shape[1]:
        raise CudaFullEvolutionV2ACSError(
            "V2 native checkpoint canonical state shape/bytes differ"
        )
    state_sha256 = getattr(checkpoint, "canonical_state_sha256", None)
    if (
        type(state_sha256) is not str
        or len(state_sha256) != 64
        or state_sha256 != hashlib.sha256(state_bytes).hexdigest()
    ):
        raise CudaFullEvolutionV2ACSError(
            "V2 native checkpoint canonical state digest differs"
        )
    expected_runtime = dict(runtime_checkpoint)
    if (
        expected_runtime.get("schema_version") != getattr(checkpoint, "schema_version", None)
        or expected_runtime.get("canonical_state_shape") != list(state_shape)
        or expected_runtime.get("canonical_state_nbytes") != len(state_bytes)
        or expected_runtime.get("canonical_state_sha256") != state_sha256
        or expected_runtime.get("metadata_seal_sha256")
        != getattr(checkpoint, "metadata_seal_sha256", None)
    ):
        raise CudaFullEvolutionV2ACSError(
            "V2 native runtime metadata is not bound to the checkpoint"
        )
    if expected_runtime.get("acknowledged_event_next_tick") != checkpoint_at_tick:
        raise CudaFullEvolutionV2ACSError(
            "V2 native checkpoint metadata frontier differs"
        )
    if Path(artifact_path).expanduser().resolve() != Path(
        getattr(checkpoint, "artifact_path", "")
    ).expanduser().resolve():
        raise CudaFullEvolutionV2ACSError(
            "V2 native checkpoint artifact path differs"
        )
    artifact = Path(artifact_path).expanduser().resolve()
    artifact_fingerprint = _fingerprint(artifact) if artifact.is_file() else None
    chunk_bytes = _validate_chunk_bytes(chunk_bytes)
    target = _validate_destination(destination)
    capture_path = target.parent / (
        f".{target.name}.{os.getpid()}.{uuid4().hex}.capture.tmp"
    )
    # The copier is read-only.  Keeping the immutable bytes view avoids a full
    # B-world duplicate solely to satisfy PyTorch's writable-buffer warning.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The given buffer is not writable",
            category=UserWarning,
        )
        source = torch.frombuffer(memoryview(state_bytes), dtype=torch.uint8).reshape(
            state_shape
        )
    pool = capture_pool or CudaFullEvolutionV2CapturePoolV1(
        chunk_bytes=chunk_bytes,
        has_cuda_fields=False,
        require_native_pool=require_native_pool,
    )
    if (
        pool.chunk_bytes != chunk_bytes
        or pool.require_native_pool is not require_native_pool
        or pool.has_cuda_fields is not False
    ):
        raise CudaFullEvolutionV2ACSError(
            "V2 native ACS capture pool differs from the route"
        )
    native_pool_telemetry = pool.telemetry()
    if require_native_pool and (
        native_pool_telemetry.get("native_pool") is not True
        or native_pool_telemetry.get("slab_count") != 2
        or native_pool_telemetry.get("preallocated") is not True
    ):
        raise CudaFullEvolutionV2ACSError(
            "V2 native production capture lacks the reusable two-slab pool"
        )
    started = time.perf_counter()
    try:
        with capture_path.open("xb") as handle:
            sink = _PayloadSink(handle)
            pool.copy_tensor(source, sink)
            handle.flush()
        logical_bytes = len(state_bytes)
        if sink.bytes != logical_bytes:
            raise CudaFullEvolutionV2ACSError(
                "V2 native state capture byte coverage changed"
            )
        capture_path.chmod(0o400)
        fingerprint = _fingerprint(capture_path)
        if fingerprint["size"] != logical_bytes:
            raise CudaFullEvolutionV2ACSError(
                "V2 native state capture size differs after close"
            )
        native_pool_telemetry = pool.telemetry()
        descriptor = {
            "schema_version": CUDA_NATIVE_DETACHED_CAPTURE_SCHEMA_V1,
            "descriptor_status": "CAPTURED_HOST_ONLY",
            "time_authority": "absolute_tick_v1",
            "source_consumer_required": False,
            "capture_path": str(capture_path),
            "destination": str(target),
            "artifact_path": str(Path(artifact_path).expanduser().resolve()),
            "artifact_file_fingerprint": artifact_fingerprint,
            "checkpoint_at_tick": checkpoint_at_tick,
            "checkpoint_id": _native_checkpoint_id_v1(
                checkpoint_at_tick=checkpoint_at_tick,
                state_sha256=state_sha256,
            ),
            "runtime_checkpoint": expected_runtime,
            "logical_bytes": logical_bytes,
            "captured_file_fingerprint": fingerprint,
            "chunk_bytes": chunk_bytes,
            "source_device_type": "cpu_checkpoint_projection",
            "capture_seconds": time.perf_counter() - started,
            "native_slab_pool": native_pool_telemetry,
            "capture_digest_deferred_to_host_worker": True,
            "producer_full_state_host_bytes_materialized": logical_bytes,
            "producer_compacted_cuda_bytes_materialized": 0,
            "worker_cuda_reference_allowed": False,
        }
        if validated_compact_frontier is not None:
            descriptor["compact_science_frontier"] = validated_compact_frontier
        descriptor["capture_authority_sha256"] = _capture_authority_sha256(
            descriptor
        )
        _canonical_json(descriptor)
        return descriptor
    except BaseException:
        if capture_path.exists() and not capture_path.is_symlink():
            failed = capture_path.with_name(
                f"{capture_path.name.removesuffix('.capture.tmp')}.capture.failed"
            )
            if failed.exists():
                failed = failed.with_name(f"{failed.name}.{uuid4().hex}")
            os.replace(capture_path, failed)
        raise


def _validate_native_capture(
    capture: Mapping[str, Any],
) -> tuple[Path, Path, dict[str, Any]]:
    if capture.get("schema_version") != CUDA_NATIVE_DETACHED_CAPTURE_SCHEMA_V1:
        raise CudaFullEvolutionV2ACSError("V2 native capture schema differs")
    if capture.get("descriptor_status") != "CAPTURED_HOST_ONLY":
        raise CudaFullEvolutionV2ACSError("V2 native capture is not host-only")
    if capture.get("time_authority") != "absolute_tick_v1":
        raise CudaFullEvolutionV2ACSError(
            "V2 native capture is not absolute-tick authoritative"
        )
    if capture.get("source_consumer_required") is not False:
        raise CudaFullEvolutionV2ACSError(
            "V2 native capture gained a source-owner lease"
        )
    _canonical_json(capture)
    if capture.get("capture_authority_sha256") != _capture_authority_sha256(capture):
        raise CudaFullEvolutionV2ACSError("V2 native capture authority seal differs")
    capture_path = Path(str(capture.get("capture_path") or "")).resolve()
    destination = Path(str(capture.get("destination") or "")).resolve()
    if (
        destination.exists()
        or destination.is_symlink()
        or capture_path.parent != destination.parent
    ):
        raise CudaFullEvolutionV2ACSError(
            "V2 native destination is not fresh and same-filesystem"
        )
    expected_fingerprint = capture.get("captured_file_fingerprint")
    if (
        not isinstance(expected_fingerprint, Mapping)
        or set(expected_fingerprint) != set(_FINGERPRINT_FIELDS)
        or any(type(expected_fingerprint.get(field)) is not int for field in _FINGERPRINT_FIELDS)
        or _fingerprint(capture_path) != dict(expected_fingerprint)
    ):
        raise CudaFullEvolutionV2ACSError("V2 native capture file identity changed")
    expected_artifact_fingerprint = capture.get("artifact_file_fingerprint")
    artifact_path = Path(str(capture.get("artifact_path") or "")).resolve()
    source_device_type = capture.get("source_device_type")
    if source_device_type not in {
        "cuda_resident_canonical_rows_v1",
        "cpu_checkpoint_projection",
    }:
        raise CudaFullEvolutionV2ACSError(
            "V2 native capture source device class differs"
        )
    if source_device_type == "cuda_resident_canonical_rows_v1":
        if (
            not isinstance(expected_artifact_fingerprint, Mapping)
            or set(expected_artifact_fingerprint) != set(_FINGERPRINT_FIELDS)
            or _fingerprint(artifact_path) != dict(expected_artifact_fingerprint)
        ):
            raise CudaFullEvolutionV2ACSError(
                "V2 native capture artifact file identity changed"
            )
    elif expected_artifact_fingerprint is not None and (
        not isinstance(expected_artifact_fingerprint, Mapping)
        or set(expected_artifact_fingerprint) != set(_FINGERPRINT_FIELDS)
        or _fingerprint(artifact_path) != dict(expected_artifact_fingerprint)
    ):
        raise CudaFullEvolutionV2ACSError(
            "V2 native host-fixture artifact file identity changed"
        )
    if (
        capture.get("worker_cuda_reference_allowed") is not False
        or type(capture.get("producer_full_state_host_bytes_materialized")) is not int
        or type(capture.get("producer_compacted_cuda_bytes_materialized")) is not int
    ):
        raise CudaFullEvolutionV2ACSError(
            "V2 native capture memory-boundary declaration differs"
        )
    runtime_checkpoint = capture.get("runtime_checkpoint")
    if not isinstance(runtime_checkpoint, Mapping):
        raise CudaFullEvolutionV2ACSError(
            "V2 native capture runtime metadata is absent"
        )
    runtime_body = {
        str(key): value
        for key, value in runtime_checkpoint.items()
        if key != "metadata_seal_sha256"
    }
    metadata_seal = runtime_checkpoint.get("metadata_seal_sha256")
    if (
        runtime_checkpoint.get("schema_version") != CUDA_NATIVE_CHECKPOINT_SCHEMA_V2
        or runtime_checkpoint.get("time_authority") != "absolute_tick_v1"
        or type(metadata_seal) is not str
        or len(metadata_seal) != 64
        or metadata_seal
        != hashlib.sha256(_canonical_json(runtime_body)).hexdigest()
    ):
        raise CudaFullEvolutionV2ACSError(
            "V2 native runtime metadata seal differs"
        )
    expected_bytes = runtime_checkpoint.get("canonical_state_nbytes")
    expected_sha = runtime_checkpoint.get("canonical_state_sha256")
    expected_shape = runtime_checkpoint.get("canonical_state_shape")
    contract = runtime_checkpoint.get("contract")
    initial_tick = runtime_checkpoint.get("initial_next_absolute_tick")
    advanced_ticks = runtime_checkpoint.get("advanced_ticks")
    acknowledged_tick = runtime_checkpoint.get("acknowledged_event_next_tick")
    if (
        not isinstance(expected_shape, list)
        or len(expected_shape) != 2
        or any(type(value) is not int or value <= 0 for value in expected_shape)
        or not isinstance(contract, Mapping)
        or contract.get("B") != expected_shape[0]
        or expected_bytes != expected_shape[0] * expected_shape[1]
        or runtime_checkpoint.get("canonical_state_layout_sha256")
        != hashlib.sha256(
            _canonical_json(
                {
                    "dtype": "uint8",
                    "shape": expected_shape,
                    "nbytes": expected_bytes,
                    "content_sha256": expected_sha,
                }
            )
        ).hexdigest()
        or type(initial_tick) is not int
        or type(advanced_ticks) is not int
        or type(acknowledged_tick) is not int
        or initial_tick != contract.get("start_next_tick")
        or advanced_ticks != acknowledged_tick - initial_tick
        or not isinstance(runtime_checkpoint.get("world_uids"), list)
        or len(runtime_checkpoint["world_uids"]) != expected_shape[0]
        or not isinstance(runtime_checkpoint.get("rng_keys"), list)
        or len(runtime_checkpoint["rng_keys"]) != expected_shape[0]
    ):
        raise CudaFullEvolutionV2ACSError(
            "V2 native runtime checkpoint envelope differs"
        )
    if (
        type(expected_bytes) is not int
        or expected_bytes <= 0
        or type(expected_sha) is not str
        or len(expected_sha) != 64
        or capture.get("logical_bytes") != expected_bytes
        or expected_fingerprint.get("size") != expected_bytes
        or type(capture.get("checkpoint_at_tick")) is not int
        or runtime_checkpoint.get("acknowledged_event_next_tick")
        != capture.get("checkpoint_at_tick")
        or capture.get("checkpoint_id")
        != _native_checkpoint_id_v1(
            checkpoint_at_tick=int(capture["checkpoint_at_tick"]),
            state_sha256=expected_sha,
        )
    ):
        raise CudaFullEvolutionV2ACSError(
            "V2 native capture byte/frontier binding differs"
        )
    _validated_compact_science_frontier_v1(
        capture.get("compact_science_frontier"),
        checkpoint_at_tick=int(capture["checkpoint_at_tick"]),
    )
    return capture_path, destination, dict(runtime_checkpoint)


def finalize_native_checkpoint_capture_v1(
    capture: Mapping[str, Any],
    *,
    cancel_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Publish one canonical native checkpoint without worker full-state bytes."""

    _raise_if_cancelled(cancel_event)
    capture_path, destination, runtime_checkpoint = _validate_native_capture(capture)
    started = time.perf_counter()
    state_sha256, state_nbytes = _file_sha256(capture_path)
    expected_sha = str(runtime_checkpoint["canonical_state_sha256"])
    expected_nbytes = int(runtime_checkpoint["canonical_state_nbytes"])
    if state_sha256 != expected_sha or state_nbytes != expected_nbytes:
        raise CudaFullEvolutionV2ACSError(
            "V2 native capture digest or size differs during worker validation"
        )
    _raise_if_cancelled(cancel_event)
    checkpoint_at_tick = int(capture["checkpoint_at_tick"])
    checkpoint_id = str(capture["checkpoint_id"])
    metadata = {
        "schema_version": CUDA_NATIVE_CHECKPOINT_METADATA_SCHEMA_V1,
        "checkpoint_format_id": CUDA_NATIVE_CHECKPOINT_SCHEMA_V2,
        "time_authority": "absolute_tick_v1",
        "checkpoint_id": checkpoint_id,
        "absolute_next_tick": checkpoint_at_tick,
        "artifact_path": str(Path(str(capture["artifact_path"])).resolve()),
        "runtime_checkpoint": {
            **runtime_checkpoint,
            "metadata_seal_sha256": runtime_checkpoint.get("metadata_seal_sha256"),
        },
    }
    compact_science_frontier = _validated_compact_science_frontier_v1(
        capture.get("compact_science_frontier"),
        checkpoint_at_tick=checkpoint_at_tick,
    )
    metadata_bytes = canonical_bytes(metadata)
    completion = {
        "schema_version": CUDA_NATIVE_CHECKPOINT_COMPLETION_SCHEMA_V1,
        "status": "complete",
        "checkpoint_format_id": CUDA_NATIVE_CHECKPOINT_SCHEMA_V2,
        "time_authority": "absolute_tick_v1",
        "checkpoint_id": checkpoint_id,
        "absolute_next_tick": checkpoint_at_tick,
        "canonical_state_file": "canonical_state.bin",
        "canonical_state_nbytes": state_nbytes,
        "canonical_state_sha256": state_sha256,
        "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
    }
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            suffix=".bundle.tmp",
            dir=destination.parent,
        )
    )
    publication_created = False
    failed_publication: Path | None = None
    try:
        state_path = stage / "canonical_state.bin"
        os.link(capture_path, state_path, follow_symlinks=False)
        state_path.chmod(0o444)
        state_fd = os.open(state_path, os.O_RDONLY)
        try:
            os.fsync(state_fd)
        finally:
            os.close(state_fd)
        metadata_path = stage / "metadata.json"
        with metadata_path.open("xb") as handle:
            handle.write(metadata_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        metadata_path.chmod(0o444)
        completion_path = stage / "bundle_complete.json"
        with completion_path.open("xb") as handle:
            handle.write(canonical_bytes(completion))
            handle.flush()
            os.fsync(handle.fileno())
        completion_path.chmod(0o444)
        _fsync_directory(stage)
        _raise_if_cancelled(cancel_event)
        _publish_directory_noreplace(stage, destination)
        publication_created = True
        destination.chmod(0o555)
        _fsync_directory(destination.parent)
        readback_sha256, readback_nbytes = _file_sha256(
            destination / "canonical_state.bin"
        )
        if readback_sha256 != state_sha256 or readback_nbytes != state_nbytes:
            raise CudaFullEvolutionV2ACSError(
                "V2 native durable state readback differs"
            )
        if _read_small_file(destination / "metadata.json") != metadata_bytes:
            raise CudaFullEvolutionV2ACSError(
                "V2 native durable metadata readback differs"
            )
        if _read_small_file(destination / "bundle_complete.json") != canonical_bytes(
            completion
        ):
            raise CudaFullEvolutionV2ACSError(
                "V2 native durable completion readback differs"
            )
        _raise_if_cancelled(cancel_event)
        capture_path.unlink()
        _fsync_directory(destination.parent)
    except BaseException as exc:
        if stage.exists() and not stage.is_symlink():
            shutil.rmtree(stage)
        if publication_created and destination.exists() and not destination.is_symlink():
            failed_publication = destination.with_name(
                f"{destination.name}.publication.failed.{uuid4().hex}"
            )
            os.replace(destination, failed_publication)
            _fsync_directory(destination.parent)
        if failed_publication is not None:
            raise CudaFullEvolutionV2ACSError(
                "V2 native publication was quarantined after failure",
                publication_failure_path=str(failed_publication),
            ) from exc
        raise
    result = {
        "schema_version": CUDA_NATIVE_DETACHED_RESULT_SCHEMA_V1,
        "status": "complete",
        "path": str(destination),
        "metadata_path": str(destination / "metadata.json"),
        "completion_path": str(destination / "bundle_complete.json"),
        "state_path": str(destination / "canonical_state.bin"),
        "checkpoint_id": checkpoint_id,
        "checkpoint_at_tick": checkpoint_at_tick,
        "canonical_state_bytes": state_nbytes,
        "canonical_state_sha256": state_sha256,
        "capture_seconds": float(capture.get("capture_seconds", 0.0)),
        "detached_finalize_seconds": time.perf_counter() - started,
        "native_slab_pool": dict(capture.get("native_slab_pool") or {}),
        "worker_state_materialization": "stream_hash_and_metadata_only_v1",
        "worker_full_state_bytes_materialized": 0,
        "worker_cuda_references": 0,
        "source_device_type": capture["source_device_type"],
        "producer_full_state_host_bytes_materialized": int(
            capture["producer_full_state_host_bytes_materialized"]
        ),
        "producer_compacted_cuda_bytes_materialized": int(
            capture["producer_compacted_cuda_bytes_materialized"]
        ),
        "_detached_publication_created_by_worker": True,
    }
    if compact_science_frontier is not None:
        result["compact_science_frontier"] = compact_science_frontier
    return result


__all__ = [
    "CudaFullEvolutionV2ACSError",
    "CudaFullEvolutionV2CapturePoolV1",
    "CudaFullEvolutionV2WorkhorseACSCoordinatorV1",
    "WORKHORSE_DETACHED_CAPTURE_SCHEMA_V1",
    "WORKHORSE_DETACHED_RESULT_SCHEMA_V1",
    "CUDA_NATIVE_DETACHED_CAPTURE_SCHEMA_V1",
    "CUDA_NATIVE_DETACHED_RESULT_SCHEMA_V1",
    "capture_workhorse_checkpoint_to_file_v1",
    "finalize_workhorse_checkpoint_capture_v1",
    "reopen_workhorse_checkpoint_set_v1",
    "reopen_workhorse_checkpoint_world_v1",
    "capture_native_resident_checkpoint_to_file_v1",
    "capture_native_checkpoint_to_file_v1",
    "finalize_native_checkpoint_capture_v1",
    "reopen_native_checkpoint_bundle_v1",
    "reopen_native_checkpoint_world_v1",
]
