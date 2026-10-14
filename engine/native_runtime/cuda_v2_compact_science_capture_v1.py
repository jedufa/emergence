"""Canonical compact science artifacts for heterogeneous CUDA V2 worlds.

The format is campaign-neutral.  Sparse birth/death events retain their native
ordering and are paired row-for-row with lifetime action, ecology, and
reproduction counters.  Declared absolute-tick snapshots retain logical-world
ownership and active-agent state without copying dense per-tick action grids.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import gzip
import hashlib
import io
import json
import os
import ctypes
import errno
from pathlib import Path
import shutil
import stat
import tempfile
from threading import RLock
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, MutableMapping, Sequence
from uuid import uuid4
from weakref import WeakValueDictionary

from .cuda_v2_science_capture_v1 import (
    COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1,
    COMPACT_SCIENCE_BIRTH_PROVENANCE_ENCODING_V1,
    COMPACT_SCIENCE_BIRTH_PROVENANCE_WIDTH_V1,
    COMPACT_SCIENCE_CAPTURE_SCHEMA_V1,
    COMPACT_SCIENCE_CHECKPOINT_FRONTIER_SCHEMA_V1,
    COMPACT_SCIENCE_DEATH_METRIC_ENCODING_V1,
    COMPACT_SCIENCE_DEATH_METRIC_ROW_WIDTH_V1,
    COMPACT_SCIENCE_DISPATCH_GZIP_ENCODING_V1,
    COMPACT_SCIENCE_DISPATCH_SCHEMA_V1,
    COMPACT_SCIENCE_PACKED_BIRTH_PROVENANCE_ENCODING_V1,
    COMPACT_SCIENCE_PACKED_BIRTH_PROVENANCE_WIDTH_V1,
    COMPACT_SCIENCE_SNAPSHOT_SCHEMA_V1,
    ScienceCompactCaptureContractError,
    _CANONICAL_STATE_BYTES,
    _agent_rows,
    _state_summary,
    _tensor_bytes,
    _world_segment,
    contract_canonical_sha256,
    contract_reconstruct_compact_birth_genome_v1,
)

class CompactScienceCaptureError(ValueError):
    """Compact acquisition inputs or materialized bytes are inconsistent."""


FROZEN_COMPACT_SCIENCE_DISPATCH_HOST_SCHEMA_V1 = (
    "eiel_frozen_compact_science_dispatch_host_v1"
)
COMPACT_SCIENCE_LOGICAL_WORLD_FIXED_BYTES_UPPER_BOUND_V1 = 1024 * 1024
COMPACT_SCIENCE_LOGICAL_WORLD_BYTES_PER_WORLD_UPPER_BOUND_V1 = 16 * 1024
COMPACT_SCIENCE_SOURCE_SINK_RECEIPT_SHA256_BYTES_V1 = 64
_COMPACT_SCIENCE_STAGE_SEAL_KEY_V1 = "_stage_seal_v1"
_COMPACT_SCIENCE_STAGE_DIRECTORY_SEAL_KEY_V1 = "."
COMPACT_SCIENCE_SNAPSHOT_TIMING_SCHEMA_V1 = (
    "eiel_compact_science_snapshot_timing_v1"
)

_COMPACT_STAGE_LOCKS_GUARD_V1 = RLock()
_COMPACT_STAGE_LOCKS_V1: WeakValueDictionary[str, RLock] = WeakValueDictionary()


def _compact_stage_lifecycle_lock_v1(path: Path) -> RLock:
    """Return the process-local lock for one exact unpublished stage path."""

    key = os.path.abspath(os.fspath(path))
    with _COMPACT_STAGE_LOCKS_GUARD_V1:
        lock = _COMPACT_STAGE_LOCKS_V1.get(key)
        if lock is None:
            lock = RLock()
            _COMPACT_STAGE_LOCKS_V1[key] = lock
        return lock


def _timing_started_v1(
    diagnostics: MutableMapping[str, Any] | None,
) -> tuple[int, int] | None:
    if diagnostics is None:
        return None
    return time.perf_counter_ns(), time.process_time_ns()


def _record_timing_v1(
    diagnostics: MutableMapping[str, Any] | None,
    phase: str,
    started: tuple[int, int] | None,
) -> None:
    if diagnostics is None or started is None:
        return
    wall_ns = time.perf_counter_ns() - started[0]
    cpu_ns = time.process_time_ns() - started[1]
    phases = diagnostics.setdefault("phases", {})
    prior = phases.get(phase)
    if prior is None:
        phases[phase] = {
            "calls": 1,
            "wall_time_ns": wall_ns,
            "cpu_time_ns": cpu_ns,
        }
        return
    prior["calls"] = int(prior["calls"]) + 1
    prior["wall_time_ns"] = int(prior["wall_time_ns"]) + wall_ns
    prior["cpu_time_ns"] = int(prior["cpu_time_ns"]) + cpu_ns


def _snapshot_tensor_nbytes_v1(value: Any) -> int | None:
    try:
        return int(value.numel()) * int(value.element_size())
    except (AttributeError, TypeError, ValueError):
        return len(value) if isinstance(value, bytes) else None


def _timed_snapshot_state_bytes_v1(
    value: Any,
    *,
    diagnostics: MutableMapping[str, Any],
) -> bytes:
    """Separate physical host transfer from host contiguity/byte conversion."""

    if isinstance(value, bytes):
        diagnostics["phases"]["state_d2h_transfer_and_wait"] = {
            "calls": 1,
            "wall_time_ns": 0,
            "cpu_time_ns": 0,
            "source_device": "cpu_bytes",
            "cuda_event_time_ns": None,
        }
        return value
    try:
        tensor = value.detach()
        cuda_events: tuple[Any, Any] | None = None
        if str(tensor.device).startswith("cuda"):
            import torch

            cuda_events = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            cuda_events[0].record()
        phase_started = _timing_started_v1(diagnostics)
        host = tensor.cpu()
        if cuda_events is not None:
            cuda_events[1].record()
            cuda_events[1].synchronize()
        _record_timing_v1(
            diagnostics,
            "state_d2h_transfer_and_wait",
            phase_started,
        )
        transfer = diagnostics["phases"]["state_d2h_transfer_and_wait"]
        transfer["source_device"] = str(tensor.device)
        transfer["cuda_event_time_ns"] = (
            round(float(cuda_events[0].elapsed_time(cuda_events[1])) * 1_000_000)
            if cuda_events is not None
            else None
        )
        phase_started = _timing_started_v1(diagnostics)
        host = host.contiguous()
        _record_timing_v1(
            diagnostics,
            "state_host_contiguous",
            phase_started,
        )
        phase_started = _timing_started_v1(diagnostics)
        raw = host.numpy().tobytes(order="C")
        _record_timing_v1(
            diagnostics,
            "state_host_bytes_conversion",
            phase_started,
        )
        return raw
    except Exception as exc:  # pragma: no cover - defensive provider boundary
        raise CompactScienceCaptureError(
            "compact snapshot state is not a detached tensor projection"
        ) from exc


@dataclass(frozen=True, slots=True)
class FrozenCompactScienceDispatchHostV1:
    """Immutable, tensor-free acquisition boundary for one CUDA dispatch."""

    schema_version: str
    packed_events: bytes
    event_agent_metrics_i64le: bytes
    event_birth_provenance_u8: bytes
    metric_fields: tuple[str, ...]
    birth_provenance_encoding: str
    start_next_absolute_tick: int
    end_next_absolute_tick: int
    tick_count: int
    event_wire_abi: str
    event_count: int
    event_content_sha256: str
    world_uids: tuple[int, ...]
    world_offsets: tuple[int, ...]
    logical_worlds_json: bytes
    native_world_control_set_sha256: str
    # Optional binding to the already-sealed event sink receipt.  The
    # asynchronous science artifact carries this identity in its own sealed
    # manifest, so a preserved compact root remains self-describing even if
    # the owner fails before returning the mutable event receipt.
    source_sink_receipt_sha256: str = ""

    @property
    def captured_host_bytes(self) -> int:
        return (
            len(self.packed_events)
            + len(self.event_agent_metrics_i64le)
            + len(self.event_birth_provenance_u8)
            + len(self.logical_worlds_json)
            + len(self.source_sink_receipt_sha256)
        )


def _validate_frozen_compact_science_dispatch_host_v1(
    frozen: FrozenCompactScienceDispatchHostV1,
) -> None:
    """Prove the exported async boundary contains immutable host primitives."""

    if (
        type(frozen) is not FrozenCompactScienceDispatchHostV1
        or frozen.schema_version
        != FROZEN_COMPACT_SCIENCE_DISPATCH_HOST_SCHEMA_V1
    ):
        raise CompactScienceCaptureError(
            "compact science workers accept only exact frozen host capsules"
        )
    if any(
        type(value) is not bytes
        for value in (
            frozen.packed_events,
            frozen.event_agent_metrics_i64le,
            frozen.event_birth_provenance_u8,
            frozen.logical_worlds_json,
        )
    ):
        raise CompactScienceCaptureError(
            "compact science frozen payloads must be immutable host bytes"
        )
    if (
        type(frozen.metric_fields) is not tuple
        or any(type(value) is not str for value in frozen.metric_fields)
        or type(frozen.world_uids) is not tuple
        or any(type(value) is not int for value in frozen.world_uids)
        or type(frozen.world_offsets) is not tuple
        or any(type(value) is not int for value in frozen.world_offsets)
    ):
        raise CompactScienceCaptureError(
            "compact science frozen sequences must contain host primitives"
        )
    if any(
        type(value) is not int
        for value in (
            frozen.start_next_absolute_tick,
            frozen.end_next_absolute_tick,
            frozen.tick_count,
            frozen.event_count,
        )
    ):
        raise CompactScienceCaptureError(
            "compact science frozen counters must be exact integers"
        )
    if any(
        type(value) is not str
        for value in (
            frozen.birth_provenance_encoding,
            frozen.event_wire_abi,
            frozen.event_content_sha256,
            frozen.native_world_control_set_sha256,
        )
    ):
        raise CompactScienceCaptureError(
            "compact science frozen identities must be exact strings"
        )
    if type(frozen.source_sink_receipt_sha256) is not str:
        raise CompactScienceCaptureError(
            "compact science source sink identity must be an exact string"
        )
    if any(
        len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in (
            frozen.event_content_sha256,
            frozen.native_world_control_set_sha256,
        )
    ):
        raise CompactScienceCaptureError(
            "compact science frozen digests are invalid"
        )
    if frozen.source_sink_receipt_sha256 and (
        len(frozen.source_sink_receipt_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in frozen.source_sink_receipt_sha256
        )
    ):
        raise CompactScienceCaptureError(
            "compact science source sink digest is invalid"
        )
    if (
        frozen.start_next_absolute_tick < 0
        or frozen.end_next_absolute_tick <= frozen.start_next_absolute_tick
        or frozen.end_next_absolute_tick
        != frozen.start_next_absolute_tick + frozen.tick_count
        or frozen.event_count < 0
        or not frozen.world_uids
        or any(value <= 0 for value in frozen.world_uids)
        or len(set(frozen.world_uids)) != len(frozen.world_uids)
        or len(frozen.world_offsets) != len(frozen.world_uids) + 1
        or frozen.world_offsets[0] != 0
        or frozen.world_offsets[-1] != frozen.event_count
        or any(
            left > right
            for left, right in zip(
                frozen.world_offsets,
                frozen.world_offsets[1:],
            )
        )
    ):
        raise CompactScienceCaptureError(
            "compact science frozen span or world offsets are invalid"
        )


def canonical_sha256(value: object) -> str:
    """Return the shared canonical contract digest as plain hexadecimal."""

    return contract_canonical_sha256(value).hex


def reconstruct_compact_birth_genome_v1(
    *,
    parent_a_genome: Mapping[str, Any],
    parent_b_genome: Mapping[str, Any],
    birth_provenance: Sequence[int] | Any,
    mutation_step: int,
) -> dict[str, Any]:
    """Compatibility wrapper over the canonical compact science contract."""

    try:
        return contract_reconstruct_compact_birth_genome_v1(
            parent_a_genome=parent_a_genome,
            parent_b_genome=parent_b_genome,
            birth_provenance=birth_provenance,
            mutation_step=mutation_step,
        )
    except ScienceCompactCaptureContractError as exc:
        raise CompactScienceCaptureError(
            str(exc)
        ) from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_compact_science_publication_path_v1(
    value: str | Path, *, label: str
) -> Path:
    """Reject symlinked parents without dereferencing the final path."""

    path = Path(os.path.abspath(os.fspath(Path(value).expanduser())))
    try:
        resolved_parent = path.parent.resolve(strict=True)
    except OSError as exc:
        raise CompactScienceCaptureError(
            f"{label} parent is unavailable: {exc}"
        ) from exc
    if resolved_parent != path.parent or path.parent.is_symlink():
        raise CompactScienceCaptureError(
            f"{label} parent must be canonical and symlink-free"
        )
    return path


def ensure_compact_science_publication_parent_v1(
    value: str | Path, *, label: str
) -> Path:
    """Create a parent chain without ever following a symbolic link."""

    path = Path(os.path.abspath(os.fspath(Path(value).expanduser())))
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open("/", flags)
    try:
        for component in path.parent.parts[1:]:
            created = False
            try:
                os.mkdir(component, mode=0o755, dir_fd=descriptor)
                created = True
            except FileExistsError:
                pass
            if created:
                os.fsync(descriptor)
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except OSError as exc:
                raise CompactScienceCaptureError(
                    f"{label} parent chain is not a canonical directory: {exc}"
                ) from exc
            os.close(descriptor)
            descriptor = child
    finally:
        os.close(descriptor)
    return canonical_compact_science_publication_path_v1(path, label=label)


def _regular_file_identity_v1(path: Path) -> dict[str, int]:
    try:
        value = os.lstat(path)
    except OSError as exc:
        raise CompactScienceCaptureError(
            f"compact dispatch staged file is unavailable: {path.name}: {exc}"
        ) from exc
    if not stat.S_ISREG(value.st_mode) or path.is_symlink():
        raise CompactScienceCaptureError(
            f"compact dispatch staged file is not regular: {path.name}"
        )
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "link_count": int(value.st_nlink),
        "mode": int(stat.S_IMODE(value.st_mode)),
        "bytes": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
    }


def _directory_identity_v1(path: Path) -> dict[str, int]:
    try:
        value = os.lstat(path)
    except OSError as exc:
        raise CompactScienceCaptureError(
            f"compact dispatch stage directory is unavailable: {exc}"
        ) from exc
    if not stat.S_ISDIR(value.st_mode) or path.is_symlink():
        raise CompactScienceCaptureError(
            "compact dispatch stage is not a regular directory"
        )
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "link_count": int(value.st_nlink),
        "mode": int(stat.S_IMODE(value.st_mode)),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
    }


def remove_compact_science_stage_v1(value: str | Path) -> None:
    """Remove one exact unpublished stage without following symbolic links."""

    root = canonical_compact_science_publication_path_v1(
        value,
        label="compact dispatch stage cleanup",
    )
    # Materialization and cleanup can be entered by the provider thread and a
    # worker/finalizer on adjacent failure paths.  Serialize only operations
    # for this exact stage; distinct dispatches retain parallel publication.
    with _compact_stage_lifecycle_lock_v1(root):
        if not os.path.lexists(root):
            return
        state = os.lstat(root)
        if not stat.S_ISDIR(state.st_mode) or root.is_symlink():
            raise CompactScienceCaptureError(
                "compact dispatch stage cleanup target is not a directory"
            )
        root.chmod(stat.S_IRWXU)
        shutil.rmtree(root)
        # Make the directory-entry removal durable.  Otherwise a crash between
        # this cleanup and the next parent fsync could resurrect an unpublished
        # canonical root.
        descriptor = os.open(
            root.parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _rename_directory_noreplace_v1(source: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing an existing name."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise CompactScienceCaptureError(
            "strict compact publication requires Linux renameat2"
        )
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
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result == 0:
        return
    failure = ctypes.get_errno()
    if failure == errno.EEXIST:
        raise CompactScienceCaptureError(
            "compact dispatch final publication root already exists"
        )
    raise CompactScienceCaptureError(
        "strict compact dispatch publication failed: "
        f"{os.strerror(failure)}"
    )


def _atomic_bytes(
    path: Path,
    value: bytes,
    *,
    timing_diagnostics: MutableMapping[str, Any] | None = None,
    timing_prefix: str = "atomic_bytes",
) -> None:
    if timing_diagnostics is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(value)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return
    phase_started = _timing_started_v1(timing_diagnostics)
    path.parent.mkdir(parents=True, exist_ok=True)
    _record_timing_v1(
        timing_diagnostics, f"{timing_prefix}_parent_prepare", phase_started
    )
    phase_started = _timing_started_v1(timing_diagnostics)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    _record_timing_v1(
        timing_diagnostics, f"{timing_prefix}_temporary_open", phase_started
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            phase_started = _timing_started_v1(timing_diagnostics)
            handle.write(value)
            _record_timing_v1(
                timing_diagnostics, f"{timing_prefix}_write", phase_started
            )
            phase_started = _timing_started_v1(timing_diagnostics)
            handle.flush()
            _record_timing_v1(
                timing_diagnostics, f"{timing_prefix}_flush", phase_started
            )
            phase_started = _timing_started_v1(timing_diagnostics)
            os.fsync(handle.fileno())
            _record_timing_v1(
                timing_diagnostics, f"{timing_prefix}_fsync", phase_started
            )
        phase_started = _timing_started_v1(timing_diagnostics)
        os.replace(temporary, path)
        _record_timing_v1(
            timing_diagnostics, f"{timing_prefix}_replace", phase_started
        )
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    timing_diagnostics: MutableMapping[str, Any] | None = None,
    timing_prefix: str = "atomic_json",
) -> None:
    if timing_diagnostics is None:
        _atomic_bytes(
            path,
            (
                json.dumps(value, sort_keys=True, separators=(",", ":"))
                + "\n"
            ).encode("utf-8"),
        )
        return
    phase_started = _timing_started_v1(timing_diagnostics)
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    _record_timing_v1(
        timing_diagnostics, f"{timing_prefix}_encode", phase_started
    )
    _atomic_bytes(
        path,
        payload,
        timing_diagnostics=timing_diagnostics,
        timing_prefix=timing_prefix,
    )


def _write_exclusive_bytes_v1(path: Path, value: bytes) -> None:
    """Write one unpublished stage member without exposing a temp pathname."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | os.O_CLOEXEC
        | os.O_NOFOLLOW,
        0o600,
    )
    try:
        view = memoryview(value)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:  # pragma: no cover - POSIX write invariant
                raise OSError("compact stage write made no progress")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive_json_v1(
    path: Path, value: Mapping[str, Any]
) -> None:
    _write_exclusive_bytes_v1(
        path,
        (
            json.dumps(value, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8"),
    )


def _gzip_json_lines(
    rows: Sequence[Mapping[str, Any]],
    *,
    timing_diagnostics: MutableMapping[str, Any] | None = None,
) -> bytes:
    if timing_diagnostics is None:
        holder = io.BytesIO()
        with gzip.GzipFile(fileobj=holder, mode="wb", mtime=0) as stream:
            for row in rows:
                stream.write(
                    (
                        json.dumps(
                            row,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    ).encode("utf-8")
                )
        return holder.getvalue()
    holder = io.BytesIO()
    phase_started = _timing_started_v1(timing_diagnostics)
    stream = gzip.GzipFile(fileobj=holder, mode="wb", mtime=0)
    _record_timing_v1(
        timing_diagnostics, "gzip_stream_initialization", phase_started
    )
    try:
        for row in rows:
            phase_started = _timing_started_v1(timing_diagnostics)
            encoded = (
                json.dumps(row, sort_keys=True, separators=(",", ":"))
                + "\n"
            ).encode("utf-8")
            _record_timing_v1(
                timing_diagnostics,
                "canonical_json_encoding_pass_2",
                phase_started,
            )
            phase_started = _timing_started_v1(timing_diagnostics)
            stream.write(encoded)
            _record_timing_v1(
                timing_diagnostics, "gzip_compression_write", phase_started
            )
    finally:
        phase_started = _timing_started_v1(timing_diagnostics)
        stream.close()
        _record_timing_v1(
            timing_diagnostics, "gzip_finalization", phase_started
        )
    return holder.getvalue()


def _gzip_dispatch_bytes_v1(value: bytes) -> bytes:
    """Encode one dispatch payload deterministically with bounded CPU cost."""

    holder = io.BytesIO()
    with gzip.GzipFile(
        fileobj=holder,
        mode="wb",
        compresslevel=1,
        mtime=0,
    ) as stream:
        stream.write(value)
    return holder.getvalue()


def _validate_control_set_sha256(value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CompactScienceCaptureError(
            "native world-control set digest is invalid"
        )


def _cell_by_world_uid(
    world_cells: Mapping[str, Mapping[str, Any]],
) -> dict[int, tuple[str, Mapping[str, Any]]]:
    result: dict[int, tuple[str, Mapping[str, Any]]] = {}
    for cell_id, cell in world_cells.items():
        value = cell.get("science_logical_world_id")
        if type(value) is not int or value <= 0 or value in result:
            raise CompactScienceCaptureError(
                "compact science cells do not carry unique logical-world identities"
            )
        result[value] = (cell_id, cell)
    return result


def freeze_compact_science_dispatch_host_v1(
    *,
    dispatch: Any,
    capture: Any,
    world_cells: Mapping[str, Mapping[str, Any]],
    native_world_control_set_sha256: str,
    source_sink_receipt_sha256: str = "",
) -> FrozenCompactScienceDispatchHostV1:
    """Detach one exact dispatch into immutable host bytes.

    This is the producer-side ownership boundary used by asynchronous
    materialization.  Nothing returned by this function retains a tensor,
    CUDA stream/event, runtime owner, callback, or mutable campaign mapping.
    """

    if (
        capture.event_content_sha256 != dispatch.content_sha256
        or capture.start_next_absolute_tick != dispatch.start_next_absolute_tick
        or capture.end_next_absolute_tick != dispatch.end_next_absolute_tick
        or capture.world_uids != dispatch.world_uids
        or capture.world_offsets != dispatch.world_offsets
    ):
        raise CompactScienceCaptureError(
            "compact science capture differs from its sparse event dispatch"
        )
    if (
        capture.birth_provenance_encoding
        != COMPACT_SCIENCE_BIRTH_PROVENANCE_ENCODING_V1
        or tuple(capture.event_birth_provenance.shape)
        != (dispatch.event_count, COMPACT_SCIENCE_BIRTH_PROVENANCE_WIDTH_V1)
    ):
        raise CompactScienceCaptureError(
            "compact birth provenance differs from its canonical encoding"
        )
    _validate_control_set_sha256(native_world_control_set_sha256)
    if source_sink_receipt_sha256 is None:
        source_sink_receipt_sha256 = ""
    if type(source_sink_receipt_sha256) is not str:
        raise CompactScienceCaptureError(
            "compact science source sink identity must be an exact string"
        )
    if source_sink_receipt_sha256:
        _validate_control_set_sha256(source_sink_receipt_sha256)
    cells = _cell_by_world_uid(world_cells)
    if set(dispatch.world_uids) != set(cells):
        raise CompactScienceCaptureError(
            "compact science dispatch does not exactly cover logical worlds"
        )
    events = bytes(dispatch.packed_events)
    event_count = int(dispatch.event_count)
    if len(events) != event_count * 32:
        raise CompactScienceCaptureError(
            "compact event stream byte count differs from its dispatch"
        )
    metric_tensor = capture.event_agent_metrics.detach().cpu().contiguous()
    provenance_tensor = (
        capture.event_birth_provenance.detach().cpu().contiguous()
    )
    metric_shape = (event_count, len(COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1))
    provenance_shape = (
        event_count,
        COMPACT_SCIENCE_BIRTH_PROVENANCE_WIDTH_V1,
    )
    if tuple(metric_tensor.shape) != metric_shape:
        raise CompactScienceCaptureError(
            "compact lifetime metric shape differs from its dispatch"
        )
    if tuple(provenance_tensor.shape) != provenance_shape:
        raise CompactScienceCaptureError(
            "compact birth provenance shape differs from its dispatch"
        )
    metric_bytes = metric_tensor.numpy().astype("<i8", copy=False).tobytes(
        order="C"
    )
    provenance_bytes = provenance_tensor.numpy().astype(
        "u1", copy=False
    ).tobytes(order="C")
    logical_worlds = [
        {
            "world_index": index,
            "world_uid": uid,
            "cell_id": cells[uid][0],
            "science_ownership": cells[uid][1].get("science_ownership"),
            "science_world_sha256": cells[uid][1].get("science_world_sha256"),
            "science_acquisition_sha256": cells[uid][1].get(
                "science_acquisition_sha256"
            ),
        }
        for index, uid in enumerate(dispatch.world_uids)
    ]
    # Prove JSON serializability and freeze every nested ownership object now,
    # before a worker can observe later caller mutation.
    logical_worlds_json = json.dumps(
        logical_worlds, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    logical_worlds_upper_bound = (
        COMPACT_SCIENCE_LOGICAL_WORLD_FIXED_BYTES_UPPER_BOUND_V1
        + len(dispatch.world_uids)
        * COMPACT_SCIENCE_LOGICAL_WORLD_BYTES_PER_WORLD_UPPER_BOUND_V1
    )
    if len(logical_worlds_json) > logical_worlds_upper_bound:
        raise CompactScienceCaptureError(
            "compact science logical-world metadata exceeds its route bound"
        )
    return FrozenCompactScienceDispatchHostV1(
        schema_version=FROZEN_COMPACT_SCIENCE_DISPATCH_HOST_SCHEMA_V1,
        packed_events=events,
        event_agent_metrics_i64le=metric_bytes,
        event_birth_provenance_u8=provenance_bytes,
        metric_fields=tuple(str(value) for value in capture.metric_fields),
        birth_provenance_encoding=str(capture.birth_provenance_encoding),
        start_next_absolute_tick=int(dispatch.start_next_absolute_tick),
        end_next_absolute_tick=int(dispatch.end_next_absolute_tick),
        tick_count=int(dispatch.tick_count),
        event_wire_abi=str(dispatch.wire_abi),
        event_count=event_count,
        event_content_sha256=str(dispatch.content_sha256),
        world_uids=tuple(int(value) for value in dispatch.world_uids),
        world_offsets=tuple(int(value) for value in dispatch.world_offsets),
        logical_worlds_json=logical_worlds_json,
        native_world_control_set_sha256=native_world_control_set_sha256,
        source_sink_receipt_sha256=source_sink_receipt_sha256,
    )


def _compact_dispatch_payloads_v1(
    frozen: FrozenCompactScienceDispatchHostV1,
) -> tuple[bytes, bytes, int, int]:
    """Vectorize exact death/birth projection without Python row expansion."""

    if (
        type(frozen) is not FrozenCompactScienceDispatchHostV1
        or frozen.schema_version
        != FROZEN_COMPACT_SCIENCE_DISPATCH_HOST_SCHEMA_V1
    ):
        raise CompactScienceCaptureError(
            "compact materialization requires an exact frozen host capsule"
        )
    import numpy as np

    event_count = frozen.event_count
    metric_width = len(COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1)
    if (
        len(frozen.packed_events) != event_count * 32
        or len(frozen.event_agent_metrics_i64le)
        != event_count * metric_width * 8
        or len(frozen.event_birth_provenance_u8)
        != event_count * COMPACT_SCIENCE_BIRTH_PROVENANCE_WIDTH_V1
    ):
        raise CompactScienceCaptureError(
            "frozen compact dispatch byte shapes differ"
        )
    metrics = np.frombuffer(
        frozen.event_agent_metrics_i64le, dtype="<i8"
    ).reshape((event_count, metric_width))
    provenance = np.frombuffer(
        frozen.event_birth_provenance_u8, dtype="u1"
    ).reshape((event_count, COMPACT_SCIENCE_BIRTH_PROVENANCE_WIDTH_V1))
    keys = np.frombuffer(frozen.packed_events, dtype="<u8").reshape(
        (event_count, 4)
    )[:, 3]
    kinds = (keys >> np.uint64(6)) & np.uint64(0x3)
    death_mask = kinds == 1
    birth_mask = kinds == 2
    if bool(np.any(~(death_mask | birth_mask))):
        raise CompactScienceCaptureError(
            "compact event stream contains an unsupported lifecycle kind"
        )
    death_rows = metrics[death_mask]
    death_provenance = provenance[death_mask]
    if death_provenance.size and bool(np.any(death_provenance)):
        raise CompactScienceCaptureError(
            "death event carries nonzero birth provenance"
        )
    unsigned_metric_indexes = tuple(
        index for index in range(metric_width) if index != 17
    )
    if death_rows.size:
        unsigned_metrics = death_rows[:, unsigned_metric_indexes]
        if bool(
            np.any(unsigned_metrics < 0)
            or np.any(unsigned_metrics >= (1 << 32))
        ):
            raise CompactScienceCaptureError(
                "death lifetime metric exceeds unsigned int32"
            )
    death_dtype = np.dtype(
        [
            (f"field_{index}", "<i8" if index == 17 else "<u4")
            for index in range(metric_width)
        ],
        align=False,
    )
    if death_dtype.itemsize != COMPACT_SCIENCE_DEATH_METRIC_ROW_WIDTH_V1:
        raise CompactScienceCaptureError(
            "compact death metric storage layout differs"
        )
    packed_deaths = np.empty(int(death_rows.shape[0]), dtype=death_dtype)
    for index in range(metric_width):
        packed_deaths[f"field_{index}"] = death_rows[:, index]
    birth_rows = metrics[birth_mask]
    if birth_rows.size and bool(np.any(birth_rows)):
        raise CompactScienceCaptureError(
            "newborn lifetime metrics are not zero at the birth boundary"
        )
    birth_provenance = provenance[birth_mask]
    mutation = birth_provenance[:, :128]
    if mutation.size and bool(np.any(mutation & np.uint8(0xF0))):
        raise CompactScienceCaptureError(
            "birth provenance uses reserved mutation bits"
        )
    packed_births = np.empty(
        (
            int(birth_provenance.shape[0]),
            COMPACT_SCIENCE_PACKED_BIRTH_PROVENANCE_WIDTH_V1,
        ),
        dtype="u1",
    )
    if packed_births.size:
        packed_births[:, :64] = mutation[:, 0::2] | (
            mutation[:, 1::2] << np.uint8(4)
        )
        packed_births[:, 64:68] = birth_provenance[:, 128:132]
    return (
        packed_deaths.tobytes(order="C"),
        packed_births.tobytes(order="C"),
        int(death_rows.shape[0]),
        int(birth_provenance.shape[0]),
    )


def _materialize_frozen_compact_science_dispatch_unchecked_v1(
    staging_root: str | Path,
    *,
    frozen: FrozenCompactScienceDispatchHostV1,
) -> dict[str, Any]:
    """Materialize one capsule into an unpublished, fully fsynced stage."""

    root = Path(staging_root).resolve()
    if root.exists():
        raise CompactScienceCaptureError(
            "compact dispatch staging root already exists"
        )
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir()
    try:
        metrics, birth_provenance, death_count, birth_count = (
            _compact_dispatch_payloads_v1(frozen)
        )
        events = frozen.packed_events
        stored_events = _gzip_dispatch_bytes_v1(events)
        stored_metrics = _gzip_dispatch_bytes_v1(metrics)
        stored_birth_provenance = _gzip_dispatch_bytes_v1(birth_provenance)
        event_path = root / "sparse_lifecycle_events_v3.bin.gz"
        metric_path = root / "event_agent_metrics_i64le.bin.gz"
        birth_provenance_path = root / "birth_provenance_packed_u4.bin.gz"
        # The stage is unpublished until its parent directory is renamed.
        # Write members directly and exclusively so cleanup actors cannot
        # observe or remove a named tempfile during the stage seal window.
        _write_exclusive_bytes_v1(event_path, stored_events)
        _write_exclusive_bytes_v1(metric_path, stored_metrics)
        _write_exclusive_bytes_v1(
            birth_provenance_path, stored_birth_provenance
        )
        logical_worlds = json.loads(frozen.logical_worlds_json)
        if not isinstance(logical_worlds, list):
            raise CompactScienceCaptureError(
                "frozen compact logical-world projection is invalid"
            )
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise
    body: dict[str, Any] = {
        "schema_version": COMPACT_SCIENCE_DISPATCH_SCHEMA_V1,
        "native_capture_schema_version": COMPACT_SCIENCE_CAPTURE_SCHEMA_V1,
        "start_next_absolute_tick": frozen.start_next_absolute_tick,
        "end_next_absolute_tick": frozen.end_next_absolute_tick,
        "tick_count": frozen.tick_count,
        "event_wire_abi": frozen.event_wire_abi,
        "event_count": frozen.event_count,
        "birth_event_count": birth_count,
        "death_event_count": death_count,
        "event_content_sha256": frozen.event_content_sha256,
        "dispatch_payload_storage_encoding": (
            COMPACT_SCIENCE_DISPATCH_GZIP_ENCODING_V1
        ),
        "event_bytes": len(events),
        "event_uncompressed_sha256": _sha256_bytes(events),
        "event_stored_bytes": len(stored_events),
        "event_file_sha256": _sha256_bytes(stored_events),
        "event_file": event_path.name,
        "metric_fields": list(frozen.metric_fields),
        "metric_source_shape": [frozen.event_count, len(frozen.metric_fields)],
        "metric_shape": [death_count, len(frozen.metric_fields)],
        "metric_storage_encoding": COMPACT_SCIENCE_DEATH_METRIC_ENCODING_V1,
        "metric_row_alignment": "death_events_in_sparse_event_order",
        "metric_row_bytes": COMPACT_SCIENCE_DEATH_METRIC_ROW_WIDTH_V1,
        "metric_bytes": len(metrics),
        "metric_uncompressed_sha256": _sha256_bytes(metrics),
        "metric_stored_bytes": len(stored_metrics),
        "metric_file_sha256": _sha256_bytes(stored_metrics),
        "metric_file": metric_path.name,
        "birth_provenance_encoding": frozen.birth_provenance_encoding,
        "birth_provenance_source_shape": [
            frozen.event_count,
            COMPACT_SCIENCE_BIRTH_PROVENANCE_WIDTH_V1,
        ],
        "birth_provenance_shape": [
            birth_count,
            COMPACT_SCIENCE_PACKED_BIRTH_PROVENANCE_WIDTH_V1,
        ],
        "birth_provenance_storage_encoding": (
            COMPACT_SCIENCE_PACKED_BIRTH_PROVENANCE_ENCODING_V1
        ),
        "birth_provenance_bytes": len(birth_provenance),
        "birth_provenance_uncompressed_sha256": _sha256_bytes(
            birth_provenance
        ),
        "birth_provenance_stored_bytes": len(stored_birth_provenance),
        "birth_provenance_file_sha256": _sha256_bytes(
            stored_birth_provenance
        ),
        "birth_provenance_file": birth_provenance_path.name,
        "birth_provenance_contract": {
            "row_alignment": "birth_events_in_sparse_event_order",
            "death_rows": "omitted_after_exact_zero_validation",
            "birth_rows": {
                "mutation_direction_bytes": {
                    "offset": 0,
                    "count": 64,
                    "packing": "two_linked_locus_nibbles_per_byte_low_then_high",
                    "linked_locus_order": (
                        "policy_locus_0_through_111_then_trait_locus_0_through_15"
                    ),
                    "allele_0_bits": [0, 1],
                    "allele_1_bits": [2, 3],
                    "code": {"0": "none", "1": "positive", "2": "negative"},
                    "reserved_code": 3,
                    "reserved_high_bits": [4, 5, 6, 7],
                },
                "recombination_choice_u32le": {
                    "offset": 64,
                    "bytes": 4,
                    "block_count": 16,
                    "bits_per_block": 2,
                    "low_bit": "parent_a_haplotype_for_child_allele_0",
                    "high_bit": "parent_b_haplotype_for_child_allele_1",
                },
            },
            "reconstruction": (
                "founder_genome_catalog_plus_prior_birth_rows_plus_resolved_"
                "mutation_step_reconstructs_exact_child_genome"
            ),
        },
        "world_offsets": list(frozen.world_offsets),
        "logical_worlds": logical_worlds,
        "native_world_control_set_sha256": (
            frozen.native_world_control_set_sha256
        ),
        "dropped_record_count": 0,
    }
    if frozen.source_sink_receipt_sha256:
        body["source_sink_receipt_sha256"] = frozen.source_sink_receipt_sha256
    manifest = {
        **body,
        "artifact_sha256": canonical_sha256(body),
    }
    manifest_path = root / "manifest.json"
    _write_exclusive_json_v1(manifest_path, manifest)
    stage_files = {
        event_path.name: manifest["event_file_sha256"],
        metric_path.name: manifest["metric_file_sha256"],
        birth_provenance_path.name: manifest[
            "birth_provenance_file_sha256"
        ],
    }
    stage_seal: dict[str, dict[str, int]] = {}
    for name, expected_sha256 in stage_files.items():
        path = root / name
        if _sha256_file(path) != expected_sha256:
            raise CompactScienceCaptureError(
                f"compact dispatch staged bytes differ after fsync: {name}"
            )
        path.chmod(stat.S_IRUSR)
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        stage_seal[name] = _regular_file_identity_v1(path)
    manifest_path.chmod(stat.S_IRUSR)
    manifest_descriptor = os.open(
        manifest_path, os.O_RDONLY | os.O_CLOEXEC
    )
    try:
        os.fsync(manifest_descriptor)
    finally:
        os.close(manifest_descriptor)
    stage_seal[manifest_path.name] = _regular_file_identity_v1(manifest_path)
    expected_members = {*stage_files, manifest_path.name}
    actual_members = {entry.name for entry in os.scandir(root)}
    if actual_members != expected_members:
        raise CompactScienceCaptureError(
            "compact dispatch stage member set differs before sealing"
        )
    root.chmod(stat.S_IRUSR | stat.S_IXUSR)
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    stage_seal[_COMPACT_SCIENCE_STAGE_DIRECTORY_SEAL_KEY_V1] = (
        _directory_identity_v1(root)
    )
    return {
        **manifest,
        "path": str(root),
        _COMPACT_SCIENCE_STAGE_SEAL_KEY_V1: stage_seal,
    }


def materialize_frozen_compact_science_dispatch_v1(
    staging_root: str | Path,
    *,
    frozen: FrozenCompactScienceDispatchHostV1,
) -> dict[str, Any]:
    """Materialize a stage and remove every partial result on failure."""

    root = ensure_compact_science_publication_parent_v1(
        staging_root,
        label="compact dispatch stage",
    )
    # Keep the stage name owned from creation through sealing.  The matching
    # cleanup lock is per-stage, so unrelated async dispatches still overlap.
    with _compact_stage_lifecycle_lock_v1(root):
        if os.path.lexists(root):
            raise CompactScienceCaptureError(
                "compact dispatch staging root already exists"
            )
        try:
            return _materialize_frozen_compact_science_dispatch_unchecked_v1(
                root,
                frozen=frozen,
            )
        except BaseException:
            if os.path.lexists(root) and not root.is_symlink():
                remove_compact_science_stage_v1(root)
            raise


def publish_frozen_compact_science_dispatch_v1(
    staged_receipt: Mapping[str, Any],
    *,
    destination: str | Path,
) -> dict[str, Any]:
    """Atomically expose a completed stage at its canonical final root."""

    staged = canonical_compact_science_publication_path_v1(
        str(staged_receipt.get("path") or ""),
        label="compact dispatch stage",
    )
    root = canonical_compact_science_publication_path_v1(
        destination,
        label="compact dispatch destination",
    )
    if staged == root or staged.parent != root.parent:
        raise CompactScienceCaptureError(
            "compact dispatch stage is outside its final publication parent"
        )
    try:
        staged_state = os.lstat(staged)
    except OSError as exc:
        raise CompactScienceCaptureError(
            f"compact dispatch stage is unavailable: {exc}"
        ) from exc
    if staged.is_symlink() or not stat.S_ISDIR(staged_state.st_mode):
        raise CompactScienceCaptureError("compact dispatch stage is unavailable")
    if os.path.lexists(root):
        raise CompactScienceCaptureError(
            "compact dispatch final publication root already exists"
        )
    manifest_path = staged / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise CompactScienceCaptureError(
            "compact dispatch stage has no sealed manifest"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    public_receipt = {
        key: value
        for key, value in staged_receipt.items()
        if key not in {"path", _COMPACT_SCIENCE_STAGE_SEAL_KEY_V1}
    }
    if manifest != public_receipt:
        raise CompactScienceCaptureError(
            "compact dispatch staged receipt differs from its manifest"
        )
    stage_seal = staged_receipt.get(_COMPACT_SCIENCE_STAGE_SEAL_KEY_V1)
    if not isinstance(stage_seal, Mapping):
        raise CompactScienceCaptureError(
            "compact dispatch stage lacks its worker verification seal"
        )
    expected_members = {
        "manifest.json",
        *(
            str(staged_receipt.get(f"{prefix}_file") or "")
            for prefix in ("event", "metric", "birth_provenance")
        ),
    }
    entries = tuple(os.scandir(staged))
    if (
        {entry.name for entry in entries} != expected_members
        or any(not entry.is_file(follow_symlinks=False) for entry in entries)
    ):
        raise CompactScienceCaptureError(
            "compact dispatch staged member set differs from its seal"
        )
    expected_directory_identity = stage_seal.get(
        _COMPACT_SCIENCE_STAGE_DIRECTORY_SEAL_KEY_V1
    )
    actual_directory_identity = _directory_identity_v1(staged)
    if (
        not isinstance(expected_directory_identity, Mapping)
        or dict(expected_directory_identity) != actual_directory_identity
        or actual_directory_identity["mode"]
        != (stat.S_IRUSR | stat.S_IXUSR)
    ):
        raise CompactScienceCaptureError(
            "compact dispatch staged directory identity changed"
        )
    for prefix in ("event", "metric", "birth_provenance"):
        name = staged_receipt.get(f"{prefix}_file")
        expected_bytes = staged_receipt.get(f"{prefix}_stored_bytes")
        expected_sha256 = staged_receipt.get(f"{prefix}_file_sha256")
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or type(expected_bytes) is not int
            or expected_bytes < 0
            or not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
        ):
            raise CompactScienceCaptureError(
                f"compact dispatch staged {prefix} identity is invalid"
            )
        payload = staged / name
        expected_identity = stage_seal.get(name)
        actual_identity = _regular_file_identity_v1(payload)
        if (
            not isinstance(expected_identity, Mapping)
            or dict(expected_identity) != actual_identity
            or actual_identity["bytes"] != expected_bytes
            or actual_identity["mode"] != stat.S_IRUSR
            or actual_identity["link_count"] != 1
        ):
            raise CompactScienceCaptureError(
                f"compact dispatch staged {prefix} bytes differ"
            )
    manifest_identity = stage_seal.get(manifest_path.name)
    if (
        not isinstance(manifest_identity, Mapping)
        or dict(manifest_identity)
        != _regular_file_identity_v1(manifest_path)
        or int(manifest_identity.get("mode", -1)) != stat.S_IRUSR
        or int(manifest_identity.get("link_count", -1)) != 1
    ):
        raise CompactScienceCaptureError(
            "compact dispatch staged manifest identity changed"
        )
    _rename_directory_noreplace_v1(staged, root)
    try:
        descriptor = os.open(root.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException as publish_error:
        rollback_error: BaseException | None = None
        if os.path.lexists(root):
            try:
                _rename_directory_noreplace_v1(root, staged)
                descriptor = os.open(
                    root.parent,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                )
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            except BaseException as exc:
                rollback_error = exc
                try:
                    remove_compact_science_stage_v1(root)
                except BaseException as cleanup_error:
                    rollback_error = cleanup_error
        if os.path.lexists(root):
            raise CompactScienceCaptureError(
                "compact dispatch publication failed after rename and the "
                "canonical root could not be quarantined or removed: "
                f"{type(rollback_error).__name__}: {rollback_error}"
            ) from publish_error
        raise CompactScienceCaptureError(
            "compact dispatch publication durability failed; canonical root "
            "was rolled back before returning failure"
        ) from publish_error
    return {**public_receipt, "path": str(root)}


COMPACT_SCIENCE_ASYNC_PUBLICATION_SCHEMA_V1 = (
    "eiel_compact_science_async_publication_v1"
)
COMPACT_SCIENCE_ASYNC_WORKERS_DEFAULT_V1 = 2
COMPACT_SCIENCE_ASYNC_PENDING_SLOTS_DEFAULT_V1 = 2


class CompactScienceAsyncPublicationError(RuntimeError):
    """Raised when bounded compact-science materialization fails closed."""


@dataclass(frozen=True, slots=True)
class CompactSciencePublicationTokenV1:
    sequence: int
    start_next_absolute_tick: int
    end_next_absolute_tick: int
    event_content_sha256: str
    captured_host_bytes: int
    staging_root: str
    destination: str
    source_sink_receipt_sha256: str = ""


@dataclass(slots=True)
class _PublicationJobV1:
    token: CompactSciencePublicationTokenV1
    future: Future[dict[str, Any]]
    enqueued_ns: int
    consumed: bool = False
    worker_elapsed_ns: int = 0


class CompactScienceAsyncPublicationV1:
    """One-or-two-worker bounded host publication lane with FIFO authority."""

    def __init__(
        self,
        *,
        workers: int = COMPACT_SCIENCE_ASYNC_WORKERS_DEFAULT_V1,
        pending_slots: int = COMPACT_SCIENCE_ASYNC_PENDING_SLOTS_DEFAULT_V1,
        materialize: Callable[..., dict[str, Any]] = (
            materialize_frozen_compact_science_dispatch_v1
        ),
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        if type(workers) is not int or workers not in {1, 2}:
            raise CompactScienceAsyncPublicationError(
                "compact science async workers must be exactly one or two"
            )
        if type(pending_slots) is not int or pending_slots not in {1, 2}:
            raise CompactScienceAsyncPublicationError(
                "compact science async pending slots must be exactly one or two"
            )
        if workers > pending_slots:
            raise CompactScienceAsyncPublicationError(
                "compact science async workers cannot exceed pending slots"
            )
        if not callable(materialize):
            raise CompactScienceAsyncPublicationError(
                "compact science materializer is unavailable"
            )
        self._workers = workers
        self._pending_slots = pending_slots
        self._materialize = materialize
        self._clock_ns = clock_ns
        self._executor = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="eiel-compact-science-writer",
        )
        self._jobs: deque[_PublicationJobV1] = deque()
        self._next_sequence = 1
        self._compute_frontier: int | None = None
        self._durable_frontier: int | None = None
        self._occupancy_high_water = 0
        self._captured_host_bytes = 0
        self._captured_host_bytes_high_water = 0
        self._producer_blocked_ns = 0
        self._worker_elapsed_ns = 0
        self._worker_elapsed_ns_max = 0
        self._submitted = 0
        self._completed = 0
        self._failed = 0
        self._failure: BaseException | None = None
        self._closed = False
        self._aborted = False

    @property
    def workers(self) -> int:
        return self._workers

    @property
    def pending_slots(self) -> int:
        return self._pending_slots

    @property
    def has_capacity(self) -> bool:
        return not self._closed and len(self._jobs) < self._pending_slots

    @property
    def has_pending(self) -> bool:
        return bool(self._jobs)

    @property
    def head_token(self) -> CompactSciencePublicationTokenV1:
        self.poll_failure()
        if not self._jobs:
            raise CompactScienceAsyncPublicationError(
                "compact science async publication has no FIFO head"
            )
        return self._jobs[0].token

    def submit(
        self,
        *,
        frozen: FrozenCompactScienceDispatchHostV1,
        destination: str | Path,
    ) -> CompactSciencePublicationTokenV1:
        self.poll_failure()
        if self._closed:
            raise CompactScienceAsyncPublicationError(
                "compact science async publication is closed"
            )
        if len(self._jobs) >= self._pending_slots:
            raise CompactScienceAsyncPublicationError(
                "compact science async publication requires producer backpressure"
            )
        try:
            _validate_frozen_compact_science_dispatch_host_v1(frozen)
        except CompactScienceCaptureError as exc:
            raise CompactScienceAsyncPublicationError(
                f"compact science frozen host capsule is invalid: {exc}"
            ) from exc
        start = frozen.start_next_absolute_tick
        end = frozen.end_next_absolute_tick
        if start < 0 or end <= start or end != start + frozen.tick_count:
            raise CompactScienceAsyncPublicationError(
                "compact science async dispatch span is invalid"
            )
        if self._compute_frontier is not None and start != self._compute_frontier:
            raise CompactScienceAsyncPublicationError(
                "compact science async dispatches must be contiguous"
            )
        try:
            root = ensure_compact_science_publication_parent_v1(
                destination,
                label="compact dispatch destination",
            )
        except CompactScienceCaptureError as exc:
            raise CompactScienceAsyncPublicationError(
                f"compact science destination is unsafe: {exc}"
            ) from exc
        if os.path.lexists(root):
            raise CompactScienceAsyncPublicationError(
                "compact science destination already exists"
            )
        staging = root.parent / f".{root.name}.async-stage-{uuid4().hex}"
        token = CompactSciencePublicationTokenV1(
            sequence=self._next_sequence,
            start_next_absolute_tick=start,
            end_next_absolute_tick=end,
            event_content_sha256=frozen.event_content_sha256,
            captured_host_bytes=frozen.captured_host_bytes,
            staging_root=str(staging),
            destination=str(root),
            source_sink_receipt_sha256=frozen.source_sink_receipt_sha256,
        )
        future = self._executor.submit(
            self._timed_materialize,
            staging,
            frozen,
        )
        self._jobs.append(
            _PublicationJobV1(
                token=token,
                future=future,
                enqueued_ns=self._clock_ns(),
            )
        )
        self._next_sequence += 1
        self._compute_frontier = end
        if self._durable_frontier is None:
            self._durable_frontier = start
        self._submitted += 1
        self._captured_host_bytes += frozen.captured_host_bytes
        self._captured_host_bytes_high_water = max(
            self._captured_host_bytes_high_water,
            self._captured_host_bytes,
        )
        self._occupancy_high_water = max(
            self._occupancy_high_water, len(self._jobs)
        )
        return token

    def ready(self, token: CompactSciencePublicationTokenV1) -> bool:
        self.poll_failure()
        job = self._job_for_token(token)
        return bool(self._jobs and self._jobs[0] is job and job.future.done())

    def result(
        self,
        token: CompactSciencePublicationTokenV1,
        *,
        producer_backpressure: bool = False,
    ) -> Mapping[str, Any]:
        self._raise_if_unusable()
        job = self._job_for_token(token)
        if not self._jobs or self._jobs[0] is not job:
            raise CompactScienceAsyncPublicationError(
                "compact science results must be consumed in FIFO order"
            )
        if job.consumed:
            raise CompactScienceAsyncPublicationError(
                "compact science async result was already consumed"
            )
        blocked_started = (
            self._clock_ns()
            if producer_backpressure and not job.future.done()
            else None
        )
        try:
            completed = job.future.result()
        except BaseException as exc:
            self._latch_failure(exc)
            raise CompactScienceAsyncPublicationError(
                "compact science async materialization failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if blocked_started is not None:
            self._producer_blocked_ns += self._clock_ns() - blocked_started
        receipt = completed["receipt"]
        job.worker_elapsed_ns = int(completed["worker_elapsed_ns"])
        self._worker_elapsed_ns += job.worker_elapsed_ns
        self._worker_elapsed_ns_max = max(
            self._worker_elapsed_ns_max,
            job.worker_elapsed_ns,
        )
        if (
            receipt.get("path") != token.staging_root
            or receipt.get("start_next_absolute_tick")
            != token.start_next_absolute_tick
            or receipt.get("end_next_absolute_tick")
            != token.end_next_absolute_tick
            or receipt.get("event_content_sha256")
            != token.event_content_sha256
            or receipt.get("source_sink_receipt_sha256", "")
            != token.source_sink_receipt_sha256
        ):
            mismatch = CompactScienceAsyncPublicationError(
                "compact science async receipt differs from its admitted token"
            )
            self._latch_failure(mismatch)
            raise mismatch
        job.consumed = True
        return dict(receipt)

    def acknowledge_published(
        self,
        token: CompactSciencePublicationTokenV1,
        *,
        published_receipt: Mapping[str, Any],
    ) -> None:
        self._raise_if_unusable()
        job = self._job_for_token(token)
        if not self._jobs or self._jobs[0] is not job or not job.consumed:
            raise CompactScienceAsyncPublicationError(
                "compact science publication acknowledgements must follow "
                "FIFO result consumption"
            )
        if (
            published_receipt.get("path") != token.destination
            or published_receipt.get("start_next_absolute_tick")
            != token.start_next_absolute_tick
            or published_receipt.get("end_next_absolute_tick")
            != token.end_next_absolute_tick
            or published_receipt.get("event_content_sha256")
            != token.event_content_sha256
            or published_receipt.get("source_sink_receipt_sha256", "")
            != token.source_sink_receipt_sha256
        ):
            raise CompactScienceAsyncPublicationError(
                "compact science published receipt differs from its token"
            )
        self._jobs.popleft()
        self._durable_frontier = token.end_next_absolute_tick
        self._captured_host_bytes -= token.captured_host_bytes
        self._completed += 1

    def poll_failure(self) -> None:
        self._raise_if_unusable()
        for job in self._jobs:
            if job.future.done() and not job.future.cancelled():
                failure = job.future.exception()
                if failure is not None:
                    self._latch_failure(failure)
                    raise CompactScienceAsyncPublicationError(
                        "compact science async worker failed: "
                        f"{type(failure).__name__}: {failure}"
                    ) from failure

    def close(self) -> None:
        self.poll_failure()
        if self._jobs:
            raise CompactScienceAsyncPublicationError(
                "compact science async publication cannot close with pending jobs"
            )
        if not self._closed:
            self._closed = True
            self._executor.shutdown(wait=True, cancel_futures=False)

    def abort(self) -> None:
        if self._closed:
            return
        self._aborted = True
        self._closed = True
        for job in self._jobs:
            job.future.cancel()
        self._executor.shutdown(wait=True, cancel_futures=True)
        for job in self._jobs:
            staging = Path(job.token.staging_root)
            if os.path.lexists(staging):
                remove_compact_science_stage_v1(staging)

    def telemetry(self) -> dict[str, Any]:
        return {
            "schema_version": COMPACT_SCIENCE_ASYNC_PUBLICATION_SCHEMA_V1,
            "workers": self._workers,
            "pending_slots": self._pending_slots,
            "compute_frontier": self._compute_frontier,
            "durable_frontier": self._durable_frontier,
            "pending_jobs": len(self._jobs),
            "occupancy_high_water": self._occupancy_high_water,
            "captured_host_bytes": self._captured_host_bytes,
            "captured_host_bytes_high_water": (
                self._captured_host_bytes_high_water
            ),
            "producer_blocked_seconds": self._producer_blocked_ns / 1_000_000_000,
            "worker_materialization_seconds_sum": (
                self._worker_elapsed_ns / 1_000_000_000
            ),
            "worker_materialization_seconds_max": (
                self._worker_elapsed_ns_max / 1_000_000_000
            ),
            "submitted_jobs": self._submitted,
            "completed_jobs": self._completed,
            "failed_jobs": self._failed,
            "queued_cuda_references": 0,
            "fifo_publication": True,
            "closed": self._closed,
            "aborted": self._aborted,
        }

    def _job_for_token(
        self, token: CompactSciencePublicationTokenV1
    ) -> _PublicationJobV1:
        for job in self._jobs:
            if job.token == token:
                return job
        raise CompactScienceAsyncPublicationError(
            "compact science async token is unknown"
        )

    def _timed_materialize(
        self,
        staging_root: str | Path,
        frozen: FrozenCompactScienceDispatchHostV1,
    ) -> dict[str, Any]:
        started_ns = self._clock_ns()
        receipt = self._materialize(staging_root, frozen=frozen)
        return {
            "receipt": receipt,
            "worker_elapsed_ns": self._clock_ns() - started_ns,
        }

    def _latch_failure(self, failure: BaseException) -> None:
        if self._failure is None:
            self._failure = failure
            self._failed += 1
            for job in self._jobs:
                if not job.future.done():
                    job.future.cancel()

    def _raise_if_unusable(self) -> None:
        if self._aborted:
            raise CompactScienceAsyncPublicationError(
                "compact science async publication was aborted"
            )
        if self._failure is not None:
            raise CompactScienceAsyncPublicationError(
                "compact science async publication is failed: "
                f"{type(self._failure).__name__}: {self._failure}"
            ) from self._failure


def write_compact_science_dispatch_v1(
    destination: str | Path,
    *,
    dispatch: Any,
    capture: Any,
    world_cells: Mapping[str, Mapping[str, Any]],
    native_world_control_set_sha256: str,
    source_sink_receipt_sha256: str = "",
) -> dict[str, Any]:
    """Compatibility wrapper over freeze, materialize, and atomic publish."""

    root = ensure_compact_science_publication_parent_v1(
        destination,
        label="compact dispatch destination",
    )
    if os.path.lexists(root):
        raise CompactScienceCaptureError(
            "compact dispatch final publication root already exists"
        )
    frozen = freeze_compact_science_dispatch_host_v1(
        dispatch=dispatch,
        capture=capture,
        world_cells=world_cells,
        native_world_control_set_sha256=native_world_control_set_sha256,
        source_sink_receipt_sha256=source_sink_receipt_sha256,
    )
    staging = root.parent / f".{root.name}.stage-{uuid4().hex}"
    staged_receipt: dict[str, Any] | None = None
    try:
        staged_receipt = materialize_frozen_compact_science_dispatch_v1(
            staging, frozen=frozen
        )
        return publish_frozen_compact_science_dispatch_v1(
            staged_receipt, destination=root
        )
    finally:
        if os.path.lexists(staging):
            remove_compact_science_stage_v1(staging)


def _snapshot_requested(cell: Mapping[str, Any], absolute_tick: int) -> bool:
    acquisition = cell.get("science_acquisition")
    if not isinstance(acquisition, Mapping):
        return False
    ticks = acquisition.get("snapshot_ticks", ())
    return (
        isinstance(ticks, Sequence)
        and not isinstance(ticks, (str, bytes))
        and absolute_tick in ticks
    )


def _write_compact_science_snapshot_v1_untimed(
    destination: str | Path,
    *,
    state: Any,
    active_metrics_projection: Mapping[str, Any],
    world_cells: Mapping[str, Mapping[str, Any]],
    absolute_tick: int,
    native_world_control_set_sha256: str,
    include_genome_catalog: bool,
) -> dict[str, Any]:
    """Frozen V1 writer used for the unperturbed performance denominator."""

    if type(absolute_tick) is not int or absolute_tick < 0:
        raise CompactScienceCaptureError("snapshot absolute tick is invalid")
    _validate_control_set_sha256(native_world_control_set_sha256)
    if active_metrics_projection.get("next_absolute_tick") != absolute_tick:
        raise CompactScienceCaptureError(
            "active metrics frontier differs from snapshot tick"
        )
    world_uids = tuple(active_metrics_projection.get("world_uids", ()))
    metric_fields = tuple(active_metrics_projection.get("metric_fields", ()))
    if metric_fields != COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1:
        raise CompactScienceCaptureError("active metric field registry differs")
    metrics_tensor = active_metrics_projection.get("tensor")
    if metrics_tensor is None or tuple(metrics_tensor.shape) != (
        len(world_uids),
        512,
        len(metric_fields),
    ):
        raise CompactScienceCaptureError("active metric tensor shape differs")
    cells = _cell_by_world_uid(world_cells)
    if set(world_uids) != set(cells):
        raise CompactScienceCaptureError(
            "compact science snapshot does not exactly cover logical worlds"
        )
    raw_state = _tensor_bytes(state, field="compact snapshot state")
    rows: list[dict[str, Any]] = []
    captured_worlds: list[dict[str, Any]] = []
    for world_index, world_uid in enumerate(world_uids):
        cell_id, cell = cells[world_uid]
        if not _snapshot_requested(cell, absolute_tick):
            continue
        world_raw = _world_segment(raw_state, world_index=world_index)
        summary = _state_summary(
            world_raw, world_uid=world_uid, tick=absolute_tick
        )
        agents = _agent_rows(
            world_raw, world_uid=world_uid, tick=absolute_tick
        )
        active = [row for row in agents.values() if row["alive"] is True]
        captured_worlds.append(
            {
                "world_index": world_index,
                "world_uid": world_uid,
                "cell_id": cell_id,
                "science_ownership": cell.get("science_ownership"),
                "science_world_sha256": cell.get("science_world_sha256"),
                "science_acquisition_sha256": cell.get(
                    "science_acquisition_sha256"
                ),
                "active_agent_count": len(active),
                "canonical_state_sha256": _sha256_bytes(
                    world_raw[:_CANONICAL_STATE_BYTES]
                ),
            }
        )
        rows.append(
            {
                "row_kind": "world_state",
                "cell_id": cell_id,
                "world_uid": world_uid,
                "absolute_tick": absolute_tick,
                **summary,
            }
        )
        genome_catalog: dict[str, Mapping[str, Any]] = {}
        for agent in sorted(active, key=lambda value: int(value["agent_id"])):
            slot = int(agent["slot"])
            genome_projection = agent.pop("_native_genome_projection")
            if include_genome_catalog:
                genome_catalog.setdefault(
                    str(agent["genome_state_id"]), genome_projection
                )
            row = {
                key: value
                for key, value in agent.items()
                if key not in {"slot"}
            }
            rows.append(
                {
                    "row_kind": "active_agent_state",
                    "cell_id": cell_id,
                    "world_uid": world_uid,
                    "absolute_tick": absolute_tick,
                    **row,
                    "lifetime_metrics": dict(
                        zip(
                            metric_fields,
                            (
                                int(value)
                                for value in metrics_tensor[
                                    world_index, slot
                                ].tolist()
                            ),
                            strict=True,
                        )
                    ),
                }
            )
        for genome_state_id, projection in sorted(genome_catalog.items()):
            rows.append(
                {
                    "row_kind": "genome_catalog",
                    "cell_id": cell_id,
                    "world_uid": world_uid,
                    "absolute_tick": absolute_tick,
                    "genome_state_id": genome_state_id,
                    "policy": list(projection["policy"]),
                    "trait": list(projection["trait"]),
                }
            )
    if not captured_worlds:
        raise CompactScienceCaptureError(
            "compact snapshot tick is not requested by any logical world"
        )
    uncompressed = b"".join(
        (
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        for row in rows
    )
    compressed = _gzip_json_lines(rows)
    root = Path(destination).resolve()
    data_path = root / "snapshot.jsonl.gz"
    _atomic_bytes(data_path, compressed)
    body: dict[str, Any] = {
        "schema_version": COMPACT_SCIENCE_SNAPSHOT_SCHEMA_V1,
        "absolute_tick": absolute_tick,
        "row_count": len(rows),
        "captured_world_count": len(captured_worlds),
        "captured_worlds": captured_worlds,
        "metric_fields": list(metric_fields),
        "genome_catalog_included": include_genome_catalog,
        "uncompressed_content_sha256": _sha256_bytes(uncompressed),
        "compressed_file_sha256": _sha256_bytes(compressed),
        "compressed_bytes": len(compressed),
        "data_file": data_path.name,
        "native_world_control_set_sha256": native_world_control_set_sha256,
        "dropped_record_count": 0,
    }
    manifest = {**body, "artifact_sha256": canonical_sha256(body)}
    _atomic_json(root / "manifest.json", manifest)
    return {**manifest, "path": str(root)}


def write_compact_science_snapshot_v1(
    destination: str | Path,
    *,
    state: Any,
    active_metrics_projection: Mapping[str, Any],
    world_cells: Mapping[str, Mapping[str, Any]],
    absolute_tick: int,
    native_world_control_set_sha256: str,
    include_genome_catalog: bool = False,
    timing_diagnostics: MutableMapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Seal requested active-agent/world snapshots at one absolute tick."""

    if timing_diagnostics is None:
        return _write_compact_science_snapshot_v1_untimed(
            destination,
            state=state,
            active_metrics_projection=active_metrics_projection,
            world_cells=world_cells,
            absolute_tick=absolute_tick,
            native_world_control_set_sha256=(
                native_world_control_set_sha256
            ),
            include_genome_catalog=include_genome_catalog,
        )
    total_started = _timing_started_v1(timing_diagnostics)
    validation_started = _timing_started_v1(timing_diagnostics)
    if timing_diagnostics is not None:
        timing_diagnostics.clear()
        timing_diagnostics.update(
            {
                "schema_version": COMPACT_SCIENCE_SNAPSHOT_TIMING_SCHEMA_V1,
                "claim_class": "TECHNICAL_ONLY",
                "implementation": "python_v1",
                "absolute_tick": absolute_tick,
                "clock": "time.perf_counter_ns",
                "cpu_clock": "time.process_time_ns",
                "phases": {},
                "counters": {},
                "success": False,
            }
        )
    if type(absolute_tick) is not int or absolute_tick < 0:
        raise CompactScienceCaptureError("snapshot absolute tick is invalid")
    _validate_control_set_sha256(native_world_control_set_sha256)
    if active_metrics_projection.get("next_absolute_tick") != absolute_tick:
        raise CompactScienceCaptureError(
            "active metrics frontier differs from snapshot tick"
        )
    world_uids = tuple(active_metrics_projection.get("world_uids", ()))
    metric_fields = tuple(active_metrics_projection.get("metric_fields", ()))
    if metric_fields != COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1:
        raise CompactScienceCaptureError("active metric field registry differs")
    metrics_tensor = active_metrics_projection.get("tensor")
    if metrics_tensor is None or tuple(metrics_tensor.shape) != (
        len(world_uids),
        512,
        len(metric_fields),
    ):
        raise CompactScienceCaptureError("active metric tensor shape differs")
    cells = _cell_by_world_uid(world_cells)
    if set(world_uids) != set(cells):
        raise CompactScienceCaptureError(
            "compact science snapshot does not exactly cover logical worlds"
        )
    _record_timing_v1(timing_diagnostics, "input_validation", validation_started)
    state_input_bytes = _snapshot_tensor_nbytes_v1(state)
    metrics_input_bytes = _snapshot_tensor_nbytes_v1(metrics_tensor)
    raw_state = _timed_snapshot_state_bytes_v1(
        state,
        diagnostics=timing_diagnostics,
    )
    rows: list[dict[str, Any]] = []
    captured_worlds: list[dict[str, Any]] = []
    active_agent_count = 0
    for world_index, world_uid in enumerate(world_uids):
        cell_id, cell = cells[world_uid]
        if not _snapshot_requested(cell, absolute_tick):
            continue
        world_raw = _world_segment(raw_state, world_index=world_index)
        phase_started = _timing_started_v1(timing_diagnostics)
        summary = _state_summary(
            world_raw, world_uid=world_uid, tick=absolute_tick
        )
        _record_timing_v1(
            timing_diagnostics,
            "world_summary_and_semantic_hashing",
            phase_started,
        )
        phase_started = _timing_started_v1(timing_diagnostics)
        agents = _agent_rows(
            world_raw, world_uid=world_uid, tick=absolute_tick
        )
        _record_timing_v1(
            timing_diagnostics,
            "agent_decode_and_semantic_hashing",
            phase_started,
        )
        phase_started = _timing_started_v1(timing_diagnostics)
        active = [row for row in agents.values() if row["alive"] is True]
        _record_timing_v1(timing_diagnostics, "alive_filtering", phase_started)
        active_agent_count += len(active)
        phase_started = _timing_started_v1(timing_diagnostics)
        canonical_state_sha256 = _sha256_bytes(
            world_raw[:_CANONICAL_STATE_BYTES]
        )
        _record_timing_v1(
            timing_diagnostics, "canonical_state_hashing", phase_started
        )
        phase_started = _timing_started_v1(timing_diagnostics)
        captured_worlds.append(
            {
                "world_index": world_index,
                "world_uid": world_uid,
                "cell_id": cell_id,
                "science_ownership": cell.get("science_ownership"),
                "science_world_sha256": cell.get("science_world_sha256"),
                "science_acquisition_sha256": cell.get(
                    "science_acquisition_sha256"
                ),
                "active_agent_count": len(active),
                "canonical_state_sha256": canonical_state_sha256,
            }
        )
        rows.append(
            {
                "row_kind": "world_state",
                "cell_id": cell_id,
                "world_uid": world_uid,
                "absolute_tick": absolute_tick,
                **summary,
            }
        )
        _record_timing_v1(
            timing_diagnostics, "world_row_materialization", phase_started
        )
        genome_catalog: dict[str, Mapping[str, Any]] = {}
        phase_started = _timing_started_v1(timing_diagnostics)
        ordered_active = sorted(active, key=lambda value: int(value["agent_id"]))
        _record_timing_v1(timing_diagnostics, "agent_sorting", phase_started)
        for agent in ordered_active:
            slot = int(agent["slot"])
            genome_projection = agent.pop("_native_genome_projection")
            if include_genome_catalog:
                genome_catalog.setdefault(
                    str(agent["genome_state_id"]), genome_projection
                )
            phase_started = _timing_started_v1(timing_diagnostics)
            row = {
                key: value
                for key, value in agent.items()
                if key not in {"slot"}
            }
            _record_timing_v1(
                timing_diagnostics,
                "active_agent_row_projection",
                phase_started,
            )
            phase_started = _timing_started_v1(timing_diagnostics)
            metric_values = metrics_tensor[world_index, slot].tolist()
            _record_timing_v1(
                timing_diagnostics, "metrics_tolist", phase_started
            )
            phase_started = _timing_started_v1(timing_diagnostics)
            rows.append(
                {
                    "row_kind": "active_agent_state",
                    "cell_id": cell_id,
                    "world_uid": world_uid,
                    "absolute_tick": absolute_tick,
                    **row,
                    "lifetime_metrics": dict(
                        zip(
                            metric_fields,
                            (int(value) for value in metric_values),
                            strict=True,
                        )
                    ),
                }
            )
            _record_timing_v1(
                timing_diagnostics,
                "active_agent_row_materialization",
                phase_started,
            )
        phase_started = _timing_started_v1(timing_diagnostics)
        for genome_state_id, projection in sorted(genome_catalog.items()):
            rows.append(
                {
                    "row_kind": "genome_catalog",
                    "cell_id": cell_id,
                    "world_uid": world_uid,
                    "absolute_tick": absolute_tick,
                    "genome_state_id": genome_state_id,
                    "policy": list(projection["policy"]),
                    "trait": list(projection["trait"]),
                }
            )
        _record_timing_v1(
            timing_diagnostics,
            "genome_catalog_sort_and_materialization",
            phase_started,
        )
    if not captured_worlds:
        raise CompactScienceCaptureError(
            "compact snapshot tick is not requested by any logical world"
        )
    phase_started = _timing_started_v1(timing_diagnostics)
    uncompressed = b"".join(
        (
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        for row in rows
    )
    _record_timing_v1(
        timing_diagnostics, "canonical_json_encoding_pass_1", phase_started
    )
    compressed = _gzip_json_lines(
        rows, timing_diagnostics=timing_diagnostics
    )
    root = Path(destination).resolve()
    data_path = root / "snapshot.jsonl.gz"
    _atomic_bytes(
        data_path,
        compressed,
        timing_diagnostics=timing_diagnostics,
        timing_prefix="data_file",
    )
    phase_started = _timing_started_v1(timing_diagnostics)
    uncompressed_content_sha256 = _sha256_bytes(uncompressed)
    _record_timing_v1(
        timing_diagnostics, "uncompressed_content_hashing", phase_started
    )
    phase_started = _timing_started_v1(timing_diagnostics)
    compressed_file_sha256 = _sha256_bytes(compressed)
    _record_timing_v1(
        timing_diagnostics, "compressed_content_hashing", phase_started
    )
    phase_started = _timing_started_v1(timing_diagnostics)
    body: dict[str, Any] = {
        "schema_version": COMPACT_SCIENCE_SNAPSHOT_SCHEMA_V1,
        "absolute_tick": absolute_tick,
        "row_count": len(rows),
        "captured_world_count": len(captured_worlds),
        "captured_worlds": captured_worlds,
        "metric_fields": list(metric_fields),
        "genome_catalog_included": include_genome_catalog,
        "uncompressed_content_sha256": uncompressed_content_sha256,
        "compressed_file_sha256": compressed_file_sha256,
        "compressed_bytes": len(compressed),
        "data_file": data_path.name,
        "native_world_control_set_sha256": native_world_control_set_sha256,
        "dropped_record_count": 0,
    }
    manifest = {**body, "artifact_sha256": canonical_sha256(body)}
    _record_timing_v1(
        timing_diagnostics, "manifest_build_and_artifact_hashing", phase_started
    )
    _atomic_json(
        root / "manifest.json",
        manifest,
        timing_diagnostics=timing_diagnostics,
        timing_prefix="manifest_file",
    )
    _record_timing_v1(timing_diagnostics, "writer_total", total_started)
    if timing_diagnostics is not None:
        phases = timing_diagnostics["phases"]
        materialization_phases = (
            "world_summary_and_semantic_hashing",
            "agent_decode_and_semantic_hashing",
            "alive_filtering",
            "canonical_state_hashing",
            "world_row_materialization",
            "agent_sorting",
            "active_agent_row_projection",
            "metrics_tolist",
            "active_agent_row_materialization",
            "genome_catalog_sort_and_materialization",
        )
        python_materialization_wall_ns = sum(
            int(phases.get(name, {}).get("wall_time_ns", 0))
            for name in materialization_phases
        )
        python_materialization_cpu_ns = sum(
            int(phases.get(name, {}).get("cpu_time_ns", 0))
            for name in materialization_phases
        )
        timing_diagnostics["derived"] = {
            "python_materialization_wall_time_ns": (
                python_materialization_wall_ns
            ),
            "python_materialization_cpu_time_ns": (
                python_materialization_cpu_ns
            ),
            "json_encoding_wall_time_ns": sum(
                int(
                    phases.get(name, {}).get("wall_time_ns", 0)
                )
                for name in (
                    "canonical_json_encoding_pass_1",
                    "canonical_json_encoding_pass_2",
                )
            ),
            "semantic_decode_and_hashing_composite_wall_time_ns": sum(
                int(phases.get(name, {}).get("wall_time_ns", 0))
                for name in (
                    "world_summary_and_semantic_hashing",
                    "agent_decode_and_semantic_hashing",
                )
            ),
            "standalone_hashing_wall_time_ns": sum(
                int(phases.get(name, {}).get("wall_time_ns", 0))
                for name in (
                    "canonical_state_hashing",
                    "uncompressed_content_hashing",
                    "compressed_content_hashing",
                    "manifest_build_and_artifact_hashing",
                )
            ),
            "compression_wall_time_ns": sum(
                int(phases.get(name, {}).get("wall_time_ns", 0))
                for name in (
                    "gzip_stream_initialization",
                    "gzip_compression_write",
                    "gzip_finalization",
                )
            ),
            "filesystem_publication_wall_time_ns": sum(
                int(phases.get(name, {}).get("wall_time_ns", 0))
                for name in (
                    "data_file_parent_prepare",
                    "data_file_temporary_open",
                    "data_file_write",
                    "data_file_flush",
                    "data_file_fsync",
                    "data_file_replace",
                    "manifest_file_parent_prepare",
                    "manifest_file_temporary_open",
                    "manifest_file_write",
                    "manifest_file_flush",
                    "manifest_file_fsync",
                    "manifest_file_replace",
                )
            ),
        }
        timing_diagnostics["derived_accounting"] = {
            "categories_are_non_additive": True,
            "note": (
                "derived duration categories can overlap and must not be summed"
            ),
        }
        timing_diagnostics["counters"] = {
            "planned_world_count": len(world_uids),
            "captured_world_count": len(captured_worlds),
            "active_agent_count": active_agent_count,
            "row_count": len(rows),
            "state_input_bytes": state_input_bytes,
            "state_host_bytes": len(raw_state),
            "metrics_input_bytes": metrics_input_bytes,
            "uncompressed_bytes": len(uncompressed),
            "compressed_bytes": len(compressed),
            "compression_ratio": (
                len(compressed) / len(uncompressed) if uncompressed else 0.0
            ),
            "records_per_second_writer": (
                len(rows)
                * 1_000_000_000
                / int(phases["writer_total"]["wall_time_ns"])
            ),
            "uncompressed_megabytes_per_second_writer": (
                len(uncompressed)
                * 1_000_000_000
                / (1024 * 1024)
                / int(phases["writer_total"]["wall_time_ns"])
            ),
            "snapshot_queue_depth": 0,
            "snapshot_queue_depth_high_water": 0,
            "resident_buffer_count": 0,
            "buffer_reuse_count": 0,
        }
        timing_diagnostics["success"] = True
    return {
        **manifest,
        "path": str(root),
        **(
            {"timing_diagnostics": dict(timing_diagnostics)}
            if timing_diagnostics is not None
            else {}
        ),
    }


def write_compact_science_checkpoint_frontier_v1(
    destination: str | Path,
    *,
    active_metrics_projection: Mapping[str, Any],
    absolute_tick: int,
    native_world_control_set_sha256: str,
) -> dict[str, Any]:
    """Seal the compact lifetime-counter frontier required for exact restore."""

    _validate_control_set_sha256(native_world_control_set_sha256)
    if (
        type(absolute_tick) is not int
        or absolute_tick < 0
        or active_metrics_projection.get("next_absolute_tick") != absolute_tick
    ):
        raise CompactScienceCaptureError(
            "compact checkpoint frontier tick is invalid"
        )
    metric_fields = tuple(active_metrics_projection.get("metric_fields", ()))
    if metric_fields != COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1:
        raise CompactScienceCaptureError(
            "compact checkpoint metric registry differs"
        )
    world_uids = tuple(active_metrics_projection.get("world_uids", ()))
    if (
        not world_uids
        or any(type(value) is not int or value <= 0 for value in world_uids)
        or len(set(world_uids)) != len(world_uids)
    ):
        raise CompactScienceCaptureError(
            "compact checkpoint world identities are invalid"
        )
    tensor = active_metrics_projection.get("tensor")
    expected_shape = (
        len(world_uids),
        512,
        len(COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1),
    )
    if tensor is None or tuple(tensor.shape) != expected_shape:
        raise CompactScienceCaptureError(
            "compact checkpoint metric tensor shape differs"
        )
    contiguous = tensor.detach().cpu().contiguous()
    raw = contiguous.numpy().astype("<i8", copy=False).tobytes(order="C")
    raw_sha256 = _sha256_bytes(raw)
    if active_metrics_projection.get("content_sha256") != raw_sha256:
        raise CompactScienceCaptureError(
            "compact checkpoint metric projection digest differs"
        )
    compressed = gzip.compress(raw, compresslevel=1, mtime=0)
    root = Path(destination).resolve()
    data_path = root / "active_agent_metrics_i64le.gz"
    _atomic_bytes(data_path, compressed)
    body: dict[str, Any] = {
        "schema_version": COMPACT_SCIENCE_CHECKPOINT_FRONTIER_SCHEMA_V1,
        "next_absolute_tick": absolute_tick,
        "world_uids": list(world_uids),
        "metric_fields": list(metric_fields),
        "shape": list(expected_shape),
        "uncompressed_bytes": len(raw),
        "uncompressed_content_sha256": raw_sha256,
        "compressed_bytes": len(compressed),
        "compressed_file_sha256": _sha256_bytes(compressed),
        "data_file": data_path.name,
        "native_world_control_set_sha256": native_world_control_set_sha256,
    }
    manifest = {**body, "artifact_sha256": canonical_sha256(body)}
    _atomic_json(root / "manifest.json", manifest)
    return {**manifest, "path": str(root)}


def load_compact_science_checkpoint_frontier_v1(
    frontier: Mapping[str, Any],
    *,
    expected_root: str | Path,
    expected_absolute_tick: int,
    expected_world_uids: Sequence[int],
    expected_native_world_control_set_sha256: str,
) -> Mapping[str, Any]:
    """Validate and load an exact compact lifetime-counter restore frontier."""

    import torch

    _validate_control_set_sha256(expected_native_world_control_set_sha256)
    expected_uids = tuple(int(value) for value in expected_world_uids)
    required = {
        "schema_version",
        "next_absolute_tick",
        "world_uids",
        "metric_fields",
        "shape",
        "uncompressed_bytes",
        "uncompressed_content_sha256",
        "compressed_bytes",
        "compressed_file_sha256",
        "data_file",
        "native_world_control_set_sha256",
        "artifact_sha256",
        "path",
    }
    if not isinstance(frontier, Mapping) or set(frontier) != required:
        raise CompactScienceCaptureError(
            "compact checkpoint frontier contract is invalid"
        )
    body = {key: frontier[key] for key in frontier if key not in {"artifact_sha256", "path"}}
    expected_shape = [
        len(expected_uids),
        512,
        len(COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1),
    ]
    if (
        frontier.get("schema_version")
        != COMPACT_SCIENCE_CHECKPOINT_FRONTIER_SCHEMA_V1
        or frontier.get("next_absolute_tick") != expected_absolute_tick
        or frontier.get("world_uids") != list(expected_uids)
        or frontier.get("metric_fields")
        != list(COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1)
        or frontier.get("shape") != expected_shape
        or frontier.get("native_world_control_set_sha256")
        != expected_native_world_control_set_sha256
        or frontier.get("artifact_sha256") != canonical_sha256(body)
    ):
        raise CompactScienceCaptureError(
            "compact checkpoint frontier identity differs"
        )
    root = Path(str(frontier["path"])).expanduser()
    if root.is_symlink():
        raise CompactScienceCaptureError(
            "compact checkpoint frontier path is a symbolic link"
        )
    root = root.resolve()
    try:
        root.relative_to(Path(expected_root).resolve())
    except ValueError as exc:
        raise CompactScienceCaptureError(
            "compact checkpoint frontier escapes its bound run"
        ) from exc
    data_file = frontier.get("data_file")
    if not isinstance(data_file, str) or Path(data_file).name != data_file:
        raise CompactScienceCaptureError(
            "compact checkpoint data file is invalid"
        )
    data_path = root / data_file
    manifest_path = root / "manifest.json"
    if (
        data_path.is_symlink()
        or manifest_path.is_symlink()
        or not data_path.is_file()
        or not manifest_path.is_file()
    ):
        raise CompactScienceCaptureError(
            "compact checkpoint frontier files are unavailable"
        )
    compressed = data_path.read_bytes()
    if (
        len(compressed) != frontier.get("compressed_bytes")
        or _sha256_bytes(compressed) != frontier.get("compressed_file_sha256")
    ):
        raise CompactScienceCaptureError(
            "compact checkpoint compressed bytes differ"
        )
    try:
        raw = gzip.decompress(compressed)
    except (OSError, EOFError) as exc:
        raise CompactScienceCaptureError(
            "compact checkpoint data is not valid gzip"
        ) from exc
    if (
        len(raw) != frontier.get("uncompressed_bytes")
        or _sha256_bytes(raw) != frontier.get("uncompressed_content_sha256")
    ):
        raise CompactScienceCaptureError(
            "compact checkpoint metric bytes differ"
        )
    tensor = torch.frombuffer(bytearray(raw), dtype=torch.int64).clone().reshape(
        tuple(expected_shape)
    )
    return {
        "schema_version": frontier["schema_version"],
        "next_absolute_tick": expected_absolute_tick,
        "world_uids": list(expected_uids),
        "metric_fields": list(COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1),
        "content_sha256": frontier["uncompressed_content_sha256"],
        "tensor": tensor,
    }


__all__ = (
    "COMPACT_SCIENCE_ASYNC_PENDING_SLOTS_DEFAULT_V1",
    "COMPACT_SCIENCE_ASYNC_PUBLICATION_SCHEMA_V1",
    "COMPACT_SCIENCE_ASYNC_WORKERS_DEFAULT_V1",
    "COMPACT_SCIENCE_DEATH_METRIC_ENCODING_V1",
    "COMPACT_SCIENCE_DEATH_METRIC_ROW_WIDTH_V1",
    "COMPACT_SCIENCE_DISPATCH_SCHEMA_V1",
    "COMPACT_SCIENCE_CHECKPOINT_FRONTIER_SCHEMA_V1",
    "COMPACT_SCIENCE_LOGICAL_WORLD_BYTES_PER_WORLD_UPPER_BOUND_V1",
    "COMPACT_SCIENCE_LOGICAL_WORLD_FIXED_BYTES_UPPER_BOUND_V1",
    "COMPACT_SCIENCE_SOURCE_SINK_RECEIPT_SHA256_BYTES_V1",
    "COMPACT_SCIENCE_SNAPSHOT_SCHEMA_V1",
    "COMPACT_SCIENCE_PACKED_BIRTH_PROVENANCE_ENCODING_V1",
    "COMPACT_SCIENCE_PACKED_BIRTH_PROVENANCE_WIDTH_V1",
    "CompactScienceCaptureError",
    "CompactScienceAsyncPublicationError",
    "CompactScienceAsyncPublicationV1",
    "CompactSciencePublicationTokenV1",
    "FROZEN_COMPACT_SCIENCE_DISPATCH_HOST_SCHEMA_V1",
    "FrozenCompactScienceDispatchHostV1",
    "canonical_compact_science_publication_path_v1",
    "ensure_compact_science_publication_parent_v1",
    "freeze_compact_science_dispatch_host_v1",
    "materialize_frozen_compact_science_dispatch_v1",
    "publish_frozen_compact_science_dispatch_v1",
    "remove_compact_science_stage_v1",
    "reconstruct_compact_birth_genome_v1",
    "write_compact_science_dispatch_v1",
    "write_compact_science_checkpoint_frontier_v1",
    "write_compact_science_snapshot_v1",
    "load_compact_science_checkpoint_frontier_v1",
)
