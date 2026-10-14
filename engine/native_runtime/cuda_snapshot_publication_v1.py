"""Plan-bound, bounded publication adapter for native V1 snapshots.

The reusable pinned buffers and worker state machine live in
``cuda_snapshot_service_v1``.  This module binds that service to one validated
snapshot plan, materializes the exact legacy V1 manifest beside the native gzip
stage, and exposes a FIFO stage/publication API to the Python lifecycle owner.

The worker never registers evidence or mutates a campaign manifest.  A stage is
made visible only by the provider thread through an atomic no-replace rename;
the provider then retains lifecycle, continuation, and terminal-index authority.
"""

from __future__ import annotations

from collections import deque
import ctypes
from dataclasses import dataclass
import errno
import gzip
import hashlib
import json
import os
from pathlib import Path
import stat
from time import perf_counter_ns
from typing import Any, Callable, Final, Mapping

from engine.campaigns.canonical import canonical_bytes, canonical_sha256

from .cuda_snapshot_encoder_v1 import (
    NATIVE_SNAPSHOT_STAGE_ENCODING_POLICY_V1,
    NATIVE_SNAPSHOT_STAGE_FILE_RECEIPT_SCHEMA_V1,
    compile_native_snapshot_encoder_plan_v1,
    encode_owned_native_snapshot_to_stage_file_v1,
)
from .cuda_snapshot_plan_v1 import (
    CompiledCudaSnapshotPlanV1,
    validate_cuda_snapshot_plan_v1,
)
from .cuda_snapshot_service_v1 import (
    NativeSnapshotCaptureRequestV1,
    NativeSnapshotEncodeRequestV1,
    NativeSnapshotServiceV1,
    NativeSnapshotSubmissionV1,
)
from .cuda_v2_compact_science_capture_v1 import (
    canonical_compact_science_publication_path_v1,
    ensure_compact_science_publication_parent_v1,
)


NATIVE_SNAPSHOT_IMPLEMENTATION_ENV_V1: Final[str] = (
    "EIEL_COMPACT_SCIENCE_SNAPSHOT_IMPLEMENTATION_V1"
)
NATIVE_SNAPSHOT_IMPLEMENTATIONS_V1: Final[tuple[str, ...]] = (
    "python_v1",
    "native_v1",
    "native_capsule_v1",
)
NATIVE_SNAPSHOT_DEFAULT_IMPLEMENTATION_V1: Final[str] = "native_capsule_v1"
NATIVE_SNAPSHOT_NATIVE_IMPLEMENTATIONS_V1: Final[tuple[str, ...]] = (
    "native_v1",
    "native_capsule_v1",
)
NATIVE_SNAPSHOT_PUBLICATION_SCHEMA_V1: Final[str] = (
    "eiel_native_snapshot_publication_service_v1"
)
NATIVE_SNAPSHOT_STAGE_SCHEMA_V1: Final[str] = "eiel_native_snapshot_v1_sealed_stage_v1"
_STAGE_SEAL_KEY_V1: Final[str] = "_native_snapshot_stage_seal_v1"
_STAGE_DIRECTORY_SEAL_KEY_V1: Final[str] = "."
_LEGACY_DATA_FILE_V1: Final[str] = "snapshot.jsonl.gz"
_LEGACY_MANIFEST_FILE_V1: Final[str] = "manifest.json"


class NativeSnapshotPublicationError(RuntimeError):
    """A native snapshot stage or publication invariant failed."""


class NativeSnapshotVisibleRootError(NativeSnapshotPublicationError):
    """Publication failed after a canonical root may have become visible."""

    def __init__(self, visible_root: Path, message: str) -> None:
        self.visible_root = str(visible_root)
        super().__init__(message)


def native_snapshot_implementation_v1() -> str:
    """Return the snapshot implementation, defaulting to native capsule V1."""

    value = os.environ.get(NATIVE_SNAPSHOT_IMPLEMENTATION_ENV_V1)
    if value is None:
        return NATIVE_SNAPSHOT_DEFAULT_IMPLEMENTATION_V1
    if value in NATIVE_SNAPSHOT_IMPLEMENTATIONS_V1:
        return value
    raise NativeSnapshotPublicationError(
        f"{NATIVE_SNAPSHOT_IMPLEMENTATION_ENV_V1} must be exactly "
        "python_v1, native_v1, or native_capsule_v1"
    )


def native_snapshot_implementation_is_native_v1(value: object) -> bool:
    """Return whether a selector value uses the bounded native service."""

    return value in NATIVE_SNAPSHOT_NATIVE_IMPLEMENTATIONS_V1


def _publication_path(value: str | os.PathLike[str], *, label: str) -> Path:
    raw = os.fspath(value)
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise NativeSnapshotPublicationError(f"{label} path is invalid")
    path = Path(os.path.abspath(raw))
    if not path.is_absolute() or path.name in {"", ".", ".."}:
        raise NativeSnapshotPublicationError(f"{label} path is invalid")
    try:
        return canonical_compact_science_publication_path_v1(path, label=label)
    except Exception as exc:
        raise NativeSnapshotPublicationError(
            f"{label} parent chain is unavailable or symlinked: {exc}"
        ) from exc


def _ensure_publication_parent(value: str | os.PathLike[str], *, label: str) -> Path:
    raw = os.fspath(value)
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise NativeSnapshotPublicationError(f"{label} path is invalid")
    try:
        return ensure_compact_science_publication_parent_v1(raw, label=label)
    except Exception as exc:
        raise NativeSnapshotPublicationError(
            f"{label} parent chain cannot be created without symlinks: {exc}"
        ) from exc


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _regular_file_identity(path: Path) -> dict[str, int]:
    value = os.lstat(path)
    if not stat.S_ISREG(value.st_mode) or path.is_symlink() or value.st_nlink != 1:
        raise NativeSnapshotPublicationError(
            f"native snapshot stage member is not one unlinked regular file: {path}"
        )
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "bytes": int(value.st_size),
        "mode": int(stat.S_IMODE(value.st_mode)),
        "link_count": int(value.st_nlink),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
    }


def _directory_identity(path: Path) -> dict[str, int]:
    value = os.lstat(path)
    if not stat.S_ISDIR(value.st_mode) or path.is_symlink():
        raise NativeSnapshotPublicationError(
            "native snapshot stage is not a regular directory"
        )
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "link_count": int(value.st_nlink),
        "mode": int(stat.S_IMODE(value.st_mode)),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
    }


def _remove_exact_stage(
    path: Path,
    *,
    expected_directory_identity: tuple[int, int] | None = None,
) -> None:
    """Remove one owned two-file stage without following links or recursion."""

    path = _publication_path(path, label="native snapshot cleanup target")
    if not os.path.lexists(path):
        return
    root_state = os.lstat(path)
    if not stat.S_ISDIR(root_state.st_mode) or path.is_symlink():
        raise NativeSnapshotPublicationError(
            "native snapshot cleanup target is not an owned stage directory"
        )
    if (
        expected_directory_identity is not None
        and (
            int(root_state.st_dev),
            int(root_state.st_ino),
        )
        != expected_directory_identity
    ):
        raise NativeSnapshotPublicationError(
            "native snapshot cleanup target no longer has its owned directory identity"
        )
    members = {entry.name: entry for entry in os.scandir(path)}
    unexpected = set(members) - {_LEGACY_DATA_FILE_V1, _LEGACY_MANIFEST_FILE_V1}
    if unexpected:
        raise NativeSnapshotPublicationError(
            "native snapshot stage cleanup found an unexpected member"
        )
    path.chmod(stat.S_IRWXU)
    for name in (_LEGACY_MANIFEST_FILE_V1, _LEGACY_DATA_FILE_V1):
        member = path / name
        if not os.path.lexists(member):
            continue
        member_state = os.lstat(member)
        if not stat.S_ISREG(member_state.st_mode) or member.is_symlink():
            raise NativeSnapshotPublicationError(
                "native snapshot stage cleanup member is not a regular file"
            )
        member.chmod(stat.S_IRUSR | stat.S_IWUSR)
        member.unlink()
    path.rmdir()
    _fsync_directory(path.parent)


def remove_native_snapshot_root_v1(value: str | os.PathLike[str]) -> None:
    """Remove one exact owned native snapshot stage/final root safely."""

    _remove_exact_stage(Path(os.fspath(value)))


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise NativeSnapshotPublicationError(
            "native snapshot publication requires Linux renameat2"
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
        raise NativeSnapshotPublicationError(
            "native snapshot final publication root already exists"
        )
    raise NativeSnapshotPublicationError(
        f"native snapshot strict publication failed: {os.strerror(failure)}"
    )


def _write_manifest_file(path: Path, manifest: Mapping[str, Any]) -> None:
    payload = canonical_bytes(dict(manifest)) + b"\n"
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
    )
    try:
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:  # pragma: no cover - POSIX write invariant
                raise OSError("native snapshot manifest write made no progress")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _parse_plan_fragment(value: object, *, label: str) -> object:
    if type(value) is not str:
        raise NativeSnapshotPublicationError(f"native snapshot {label} is absent")
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise NativeSnapshotPublicationError(
            f"native snapshot {label} is invalid"
        ) from exc
    if canonical_bytes(decoded).decode("ascii") != value:
        raise NativeSnapshotPublicationError(
            f"native snapshot {label} is not canonical"
        )
    return decoded


@dataclass(frozen=True, slots=True)
class NativeSnapshotPublicationTokenV1:
    submission: NativeSnapshotSubmissionV1
    absolute_tick: int
    staging_root: str
    destination: str
    producer_boundary_started_ns: int
    producer_handoff_completed_ns: int


@dataclass(slots=True)
class _NativeSnapshotPublicationJobV1:
    token: NativeSnapshotPublicationTokenV1
    result_consumed: bool = False


def _exact_manifest_from_native_descriptor(
    descriptor: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    absolute_tick: int,
) -> dict[str, Any]:
    if (
        descriptor.get("schema_version") != NATIVE_SNAPSHOT_STAGE_FILE_RECEIPT_SCHEMA_V1
        or descriptor.get("absolute_tick") != absolute_tick
        or descriptor.get("plan_sha256") != plan.get("plan_sha256")
        or descriptor.get("native_world_control_set_sha256")
        != plan.get("native_world_control_set_sha256")
        or descriptor.get("execution_publication_context_sha256")
        != plan.get("execution_publication_context_sha256")
        or descriptor.get("stage_encoding_policy")
        != NATIVE_SNAPSHOT_STAGE_ENCODING_POLICY_V1
        or descriptor.get("stage_file_state") != "FSYNCED_STAGE_ONLY_NOT_PUBLISHED"
        or descriptor.get("stage_file_fsync_completed") is not True
        or descriptor.get("stage_file_close_completed") is not True
        or descriptor.get("atomic_publication_performed") is not False
        or descriptor.get("directory_fsync_performed") is not False
    ):
        raise NativeSnapshotPublicationError(
            "native snapshot encoder descriptor differs from its bound plan"
        )

    boundaries = plan.get("snapshot_boundaries")
    boundary = (
        next(
            (
                row
                for row in boundaries
                if isinstance(row, Mapping)
                and row.get("absolute_tick") == absolute_tick
            ),
            None,
        )
        if isinstance(boundaries, list)
        else None
    )
    if not isinstance(boundary, Mapping):
        raise NativeSnapshotPublicationError(
            "native snapshot tick is absent from its bound plan"
        )
    selected_indices = list(boundary.get("world_indices", ()))
    selected_uids = list(boundary.get("world_uids", ()))
    if (
        descriptor.get("selected_world_indices") != selected_indices
        or descriptor.get("selected_world_uids") != selected_uids
        or descriptor.get("captured_world_count") != len(selected_indices)
        or descriptor.get("genome_catalog_included")
        is not boundary.get("include_genome_catalog")
    ):
        raise NativeSnapshotPublicationError(
            "native snapshot selection differs from its bound boundary"
        )

    worlds = plan.get("worlds")
    captured = descriptor.get("captured_worlds")
    if not isinstance(worlds, list) or not isinstance(captured, list):
        raise NativeSnapshotPublicationError(
            "native snapshot captured-world descriptors are absent"
        )
    captured_worlds: list[dict[str, Any]] = []
    active_agent_count_total = 0
    layout = plan.get("layout")
    agent_capacity = (
        layout.get("agent_capacity") if isinstance(layout, Mapping) else None
    )
    if type(agent_capacity) is not int or agent_capacity <= 0:
        raise NativeSnapshotPublicationError(
            "native snapshot plan agent capacity is invalid"
        )
    for ordinal, (world_index, world_uid) in enumerate(
        zip(selected_indices, selected_uids, strict=True)
    ):
        if (
            type(world_index) is not int
            or world_index < 0
            or world_index >= len(worlds)
            or ordinal >= len(captured)
            or not isinstance(worlds[world_index], Mapping)
            or not isinstance(captured[ordinal], Mapping)
        ):
            raise NativeSnapshotPublicationError(
                "native snapshot captured-world shape differs"
            )
        invariant = worlds[world_index]
        native_world = captured[ordinal]
        if (
            invariant.get("world_index") != world_index
            or invariant.get("world_uid") != world_uid
            or native_world.get("world_index") != world_index
            or native_world.get("world_uid") != world_uid
        ):
            raise NativeSnapshotPublicationError(
                "native snapshot captured-world identity differs"
            )
        active_agent_count = native_world.get("active_agent_count")
        canonical_state_sha256 = native_world.get("canonical_state_sha256")
        if (
            type(active_agent_count) is not int
            or active_agent_count < 0
            or active_agent_count > agent_capacity
            or not isinstance(canonical_state_sha256, str)
            or len(canonical_state_sha256) != 64
            or any(child not in "0123456789abcdef" for child in canonical_state_sha256)
        ):
            raise NativeSnapshotPublicationError(
                "native snapshot captured-world counters are invalid"
            )
        captured_worlds.append(
            {
                "world_index": world_index,
                "world_uid": world_uid,
                "cell_id": _parse_plan_fragment(
                    invariant.get("cell_id_json"), label="cell identity fragment"
                ),
                "science_ownership": _parse_plan_fragment(
                    invariant.get("science_ownership_json"),
                    label="science ownership fragment",
                ),
                "science_world_sha256": _parse_plan_fragment(
                    invariant.get("science_world_sha256_json"),
                    label="science world digest fragment",
                ),
                "science_acquisition_sha256": _parse_plan_fragment(
                    invariant.get("science_acquisition_sha256_json"),
                    label="science acquisition digest fragment",
                ),
                "active_agent_count": active_agent_count,
                "canonical_state_sha256": canonical_state_sha256,
            }
        )
        active_agent_count_total += active_agent_count
    if len(captured_worlds) != len(captured):
        raise NativeSnapshotPublicationError(
            "native snapshot captured-world count differs"
        )

    metric_fields = descriptor.get("metric_fields")
    if metric_fields != plan.get("metric_fields"):
        raise NativeSnapshotPublicationError(
            "native snapshot metric registry differs from its bound plan"
        )
    integer_fields = ("row_count", "compressed_bytes")
    if any(
        type(descriptor.get(name)) is not int or int(descriptor[name]) < 0
        for name in integer_fields
    ):
        raise NativeSnapshotPublicationError(
            "native snapshot descriptor byte/row accounting is invalid"
        )
    if (
        descriptor.get("active_agent_count") != active_agent_count_total
        or descriptor["row_count"] < len(captured_worlds) + active_agent_count_total
        or (
            descriptor.get("genome_catalog_included") is False
            and descriptor["row_count"]
            != len(captured_worlds) + active_agent_count_total
        )
    ):
        raise NativeSnapshotPublicationError(
            "native snapshot descriptor row accounting differs"
        )
    digest_fields = ("uncompressed_content_sha256", "compressed_file_sha256")
    if any(
        not isinstance(descriptor.get(name), str)
        or len(str(descriptor[name])) != 64
        or any(child not in "0123456789abcdef" for child in str(descriptor[name]))
        for name in digest_fields
    ):
        raise NativeSnapshotPublicationError(
            "native snapshot descriptor digest is invalid"
        )
    body: dict[str, Any] = {
        "schema_version": plan["snapshot_schema_id"],
        "absolute_tick": absolute_tick,
        "row_count": descriptor["row_count"],
        "captured_world_count": len(captured_worlds),
        "captured_worlds": captured_worlds,
        "metric_fields": list(metric_fields),
        "genome_catalog_included": descriptor["genome_catalog_included"],
        "uncompressed_content_sha256": descriptor["uncompressed_content_sha256"],
        "compressed_file_sha256": descriptor["compressed_file_sha256"],
        "compressed_bytes": descriptor["compressed_bytes"],
        "data_file": _LEGACY_DATA_FILE_V1,
        "native_world_control_set_sha256": plan["native_world_control_set_sha256"],
        "dropped_record_count": 0,
    }
    return {**body, "artifact_sha256": canonical_sha256(body)}


def publish_native_snapshot_stage_v1(
    staged_receipt: Mapping[str, Any],
    *,
    destination: str | os.PathLike[str],
) -> dict[str, Any]:
    """Atomically publish one sealed exact-V1 stage without replacement."""

    if staged_receipt.get("stage_schema_version") != NATIVE_SNAPSHOT_STAGE_SCHEMA_V1:
        raise NativeSnapshotPublicationError("native snapshot stage receipt differs")
    staged = _publication_path(
        str(staged_receipt.get("path") or ""), label="native snapshot stage"
    )
    root = _publication_path(destination, label="native snapshot destination")
    if (
        staged == root
        or staged.parent != root.parent
        or staged_receipt.get("destination") != str(root)
        or staged_receipt.get("stage_file_state") != "SEALED_STAGE_NOT_PUBLISHED"
    ):
        raise NativeSnapshotPublicationError(
            "native snapshot stage is outside its bound publication destination"
        )
    if os.path.lexists(root):
        raise NativeSnapshotPublicationError(
            "native snapshot final publication root already exists"
        )
    expected_members = {_LEGACY_DATA_FILE_V1, _LEGACY_MANIFEST_FILE_V1}
    if {entry.name for entry in os.scandir(staged)} != expected_members:
        raise NativeSnapshotPublicationError("native snapshot stage member set differs")
    seal = staged_receipt.get(_STAGE_SEAL_KEY_V1)
    if not isinstance(seal, Mapping):
        raise NativeSnapshotPublicationError("native snapshot stage seal is absent")
    for name in sorted(expected_members):
        expected = seal.get(name)
        actual = _regular_file_identity(staged / name)
        if not isinstance(expected, Mapping) or dict(expected) != actual:
            raise NativeSnapshotPublicationError(
                f"native snapshot staged {name} identity changed"
            )
    directory_identity = seal.get(_STAGE_DIRECTORY_SEAL_KEY_V1)
    if not isinstance(directory_identity, Mapping) or dict(
        directory_identity
    ) != _directory_identity(staged):
        raise NativeSnapshotPublicationError(
            "native snapshot stage directory identity changed"
        )
    _rename_directory_noreplace(staged, root)
    try:
        _fsync_directory(root.parent)
    except BaseException as publish_error:
        rollback_error: BaseException | None = None
        if os.path.lexists(root):
            try:
                _rename_directory_noreplace(root, staged)
                _fsync_directory(root.parent)
            except BaseException as exc:
                rollback_error = exc
        if os.path.lexists(root):
            raise NativeSnapshotVisibleRootError(
                root,
                "native snapshot publication durability failed after rename and "
                "the canonical root could not be rolled back",
            ) from (rollback_error or publish_error)
        raise NativeSnapshotPublicationError(
            "native snapshot publication durability failed and was rolled back"
        ) from publish_error
    public = {
        key: value
        for key, value in staged_receipt.items()
        if key
        not in {
            "stage_schema_version",
            "destination",
            "stage_file_state",
            "capture_metrics",
            "encoder_timing",
            "native_counters",
            "publication_timing",
            _STAGE_SEAL_KEY_V1,
        }
    }
    public["path"] = str(root)
    return public


def inspect_existing_native_snapshot_root_v1(
    value: str | os.PathLike[str],
    *,
    compiled_plan: CompiledCudaSnapshotPlanV1 | bytes,
    absolute_tick: int,
) -> dict[str, Any]:
    """Authenticate one existing exact-V1 root for crash-safe reconciliation."""

    validated = validate_cuda_snapshot_plan_v1(compiled_plan)
    plan = validated.to_mapping()
    plan["plan_sha256"] = validated.plan_sha256
    root = _publication_path(value, label="existing native snapshot root")
    _directory_identity(root)
    expected_members = {_LEGACY_DATA_FILE_V1, _LEGACY_MANIFEST_FILE_V1}
    if {entry.name for entry in os.scandir(root)} != expected_members:
        raise NativeSnapshotPublicationError(
            "existing native snapshot root member set differs"
        )
    data_path = root / _LEGACY_DATA_FILE_V1
    manifest_path = root / _LEGACY_MANIFEST_FILE_V1
    data_identity = _regular_file_identity(data_path)
    manifest_identity = _regular_file_identity(manifest_path)
    maximum_manifest_bytes = 1_048_576 + 8_192 * len(plan["worlds"])
    if manifest_identity["bytes"] > maximum_manifest_bytes:
        raise NativeSnapshotPublicationError(
            "existing native snapshot manifest exceeds its plan-derived bound"
        )
    try:
        manifest_raw = manifest_path.read_bytes()
        manifest = json.loads(manifest_raw)
        if (
            not isinstance(manifest, dict)
            or canonical_bytes(manifest) + b"\n" != manifest_raw
        ):
            raise ValueError("manifest is not exact canonical JSON")
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise NativeSnapshotPublicationError(
            "existing native snapshot manifest is invalid"
        ) from exc

    boundaries = plan["snapshot_boundaries"]
    boundary = next(
        (row for row in boundaries if row.get("absolute_tick") == absolute_tick),
        None,
    )
    captured = manifest.get("captured_worlds")
    if not isinstance(boundary, Mapping) or not isinstance(captured, list):
        raise NativeSnapshotPublicationError(
            "existing native snapshot boundary descriptor is invalid"
        )
    try:
        active_agent_count = sum(
            int(row.get("active_agent_count", -1))
            for row in captured
            if isinstance(row, Mapping)
        )
    except (TypeError, ValueError) as exc:
        raise NativeSnapshotPublicationError(
            "existing native snapshot active-agent accounting is invalid"
        ) from exc
    descriptor = {
        "schema_version": NATIVE_SNAPSHOT_STAGE_FILE_RECEIPT_SCHEMA_V1,
        "plan_sha256": validated.plan_sha256,
        "native_world_control_set_sha256": plan["native_world_control_set_sha256"],
        "execution_publication_context_sha256": plan[
            "execution_publication_context_sha256"
        ],
        "absolute_tick": absolute_tick,
        "row_count": manifest.get("row_count"),
        "captured_world_count": manifest.get("captured_world_count"),
        "active_agent_count": active_agent_count,
        "selected_world_indices": list(boundary["world_indices"]),
        "selected_world_uids": list(boundary["world_uids"]),
        "captured_worlds": captured,
        "metric_fields": manifest.get("metric_fields"),
        "genome_catalog_included": manifest.get("genome_catalog_included"),
        "uncompressed_content_sha256": manifest.get("uncompressed_content_sha256"),
        "compressed_file_sha256": manifest.get("compressed_file_sha256"),
        "compressed_bytes": manifest.get("compressed_bytes"),
        "stage_encoding_policy": NATIVE_SNAPSHOT_STAGE_ENCODING_POLICY_V1,
        "stage_file_state": "FSYNCED_STAGE_ONLY_NOT_PUBLISHED",
        "stage_file_fsync_completed": True,
        "stage_file_close_completed": True,
        "atomic_publication_performed": False,
        "directory_fsync_performed": False,
    }
    expected_manifest = _exact_manifest_from_native_descriptor(
        descriptor,
        plan=plan,
        absolute_tick=absolute_tick,
    )
    if manifest != expected_manifest or data_identity["bytes"] != manifest.get(
        "compressed_bytes"
    ):
        raise NativeSnapshotPublicationError(
            "existing native snapshot manifest differs from its bound plan or bytes"
        )

    compressed_digest = hashlib.sha256()
    uncompressed_digest = hashlib.sha256()
    try:
        with data_path.open("rb") as compressed_stream:
            for chunk in iter(lambda: compressed_stream.read(1024 * 1024), b""):
                compressed_digest.update(chunk)
        with gzip.open(data_path, "rb") as uncompressed_stream:
            for chunk in iter(lambda: uncompressed_stream.read(1024 * 1024), b""):
                uncompressed_digest.update(chunk)
    except (OSError, EOFError, gzip.BadGzipFile) as exc:
        raise NativeSnapshotPublicationError(
            "existing native snapshot gzip payload is invalid"
        ) from exc
    if (
        compressed_digest.hexdigest() != manifest["compressed_file_sha256"]
        or uncompressed_digest.hexdigest() != manifest["uncompressed_content_sha256"]
    ):
        raise NativeSnapshotPublicationError(
            "existing native snapshot content digest differs"
        )
    return {**manifest, "path": str(root)}


class NativeSnapshotPublicationServiceV1:
    """One plan-bound FIFO around the fixed-slot native snapshot service."""

    def __init__(
        self,
        *,
        compiled_plan: CompiledCudaSnapshotPlanV1 | bytes,
        runtime_owner: object,
        slot_count: int,
        loaded_extension: Any | None = None,
        native_plan_compiler: Callable[..., Any] = (
            compile_native_snapshot_encoder_plan_v1
        ),
        owned_encoder: Callable[..., dict[str, Any]] = (
            encode_owned_native_snapshot_to_stage_file_v1
        ),
        core_service_factory: Callable[..., Any] = NativeSnapshotServiceV1,
        slot_allocator: Callable[..., Any] | None = None,
        allow_unpinned_test_slots: bool = False,
        implementation_label: str = "native_v1",
    ) -> None:
        if implementation_label not in NATIVE_SNAPSHOT_NATIVE_IMPLEMENTATIONS_V1:
            raise NativeSnapshotPublicationError(
                "native snapshot service implementation label is unsupported"
            )
        validated = validate_cuda_snapshot_plan_v1(compiled_plan)
        plan = validated.to_mapping()
        plan["plan_sha256"] = validated.plan_sha256
        self._compiled_plan = validated
        self._plan = plan
        self._runtime_owner = runtime_owner
        self._native_plan = native_plan_compiler(
            validated,
            loaded_extension=loaded_extension,
        )
        self._owned_encoder = owned_encoder
        self._implementation_label = implementation_label
        self._jobs: deque[_NativeSnapshotPublicationJobV1] = deque()
        self._owned_stages: dict[Path, tuple[int, int]] = {}
        self._stage_namespace = canonical_sha256(
            {
                "plan_sha256": validated.plan_sha256,
                "pid": os.getpid(),
                "service_object_identity": id(self),
                "bound_at_monotonic_ns": perf_counter_ns(),
            }
        )[:20]
        self._closed = False
        self._aborted = False
        self._publication_completed = 0
        self._publication_failed = 0
        self._publication_elapsed_ns_total = 0
        self._publication_elapsed_ns_max = 0
        self._worker_manifest_elapsed_ns_total = 0
        self._worker_manifest_elapsed_ns_max = 0

        state_transport = plan["state_capture_transport"]
        preferred_mode = state_transport["preferred_mode"]
        state_input = next(
            (
                mode
                for mode in state_transport["accepted_modes"]
                if mode.get("mode") == preferred_mode
            ),
            None,
        )
        if not isinstance(state_input, Mapping):  # validated-plan invariant
            raise NativeSnapshotPublicationError(
                "native snapshot preferred state capture transport is absent"
            )
        self._state_capture_mode = str(preferred_mode)
        metrics_input = plan["metrics_input"]
        options: dict[str, Any] = {
            "state_shape": tuple(state_input["shape"]),
            "active_science_metrics_shape": tuple(metrics_input["shape"]),
            "slot_count": slot_count,
            "capture": self._capture,
            "encoder": self._encode,
        }
        if slot_allocator is not None:
            options["slot_allocator"] = slot_allocator
        if allow_unpinned_test_slots:
            options["allow_unpinned_test_slots"] = True
        self._service = core_service_factory(**options)

    @property
    def plan_sha256(self) -> str:
        return self._compiled_plan.plan_sha256

    @property
    def has_capacity(self) -> bool:
        return bool(self._service.has_capacity)

    @property
    def has_pending(self) -> bool:
        return bool(self._jobs)

    @property
    def head_token(self) -> NativeSnapshotPublicationTokenV1:
        if not self._jobs:
            raise NativeSnapshotPublicationError(
                "native snapshot publication FIFO is empty"
            )
        return self._jobs[0].token

    def _capture(self, request: NativeSnapshotCaptureRequestV1) -> object | None:
        context = request.context
        if (
            not isinstance(context, Mapping)
            or context.get("absolute_tick") != request.absolute_tick
        ):
            raise NativeSnapshotPublicationError(
                "native snapshot capture context is invalid"
            )
        capture = getattr(
            self._runtime_owner, "capture_native_snapshot_to_pinned_v1", None
        )
        if not callable(capture):
            raise NativeSnapshotPublicationError(
                "native runtime owner lacks pinned snapshot capture"
            )
        return capture(
            absolute_tick=request.absolute_tick,
            state_destination=request.state_destination,
            active_science_metrics_destination=(
                request.active_science_metrics_destination
            ),
        )

    def _encode(self, request: NativeSnapshotEncodeRequestV1) -> Mapping[str, Any]:
        context = request.context
        if (
            not isinstance(context, Mapping)
            or context.get("absolute_tick") != request.absolute_tick
        ):
            raise NativeSnapshotPublicationError(
                "native snapshot encoder context is invalid"
            )
        staging_root = _publication_path(
            str(context.get("staging_root") or ""),
            label="native snapshot stage",
        )
        stage_data_path = _publication_path(
            str(context.get("stage_data_path") or ""),
            label="native snapshot stage data",
        )
        destination = _publication_path(
            str(context.get("destination") or ""),
            label="native snapshot destination",
        )
        if (
            stage_data_path != staging_root / _LEGACY_DATA_FILE_V1
            or staging_root.parent != destination.parent
        ):
            raise NativeSnapshotPublicationError(
                "native snapshot encoder context paths differ"
            )
        try:
            descriptor = self._owned_encoder(
                self._native_plan,
                state=request.state,
                metrics=request.active_science_metrics,
                absolute_tick=request.absolute_tick,
                stage_path=str(stage_data_path),
                encoding_policy=NATIVE_SNAPSHOT_STAGE_ENCODING_POLICY_V1,
            )
            if descriptor.get("stage_path") != str(stage_data_path):
                raise NativeSnapshotPublicationError(
                    "native snapshot encoder wrote another stage path"
                )
            manifest_started_ns = perf_counter_ns()
            manifest = _exact_manifest_from_native_descriptor(
                descriptor,
                plan=self._plan,
                absolute_tick=request.absolute_tick,
            )
            data_identity = _regular_file_identity(stage_data_path)
            if (
                data_identity["bytes"] != manifest["compressed_bytes"]
                or descriptor.get("stage_bytes_written") != manifest["compressed_bytes"]
            ):
                raise NativeSnapshotPublicationError(
                    "native snapshot stage byte count differs"
                )
            manifest_path = staging_root / _LEGACY_MANIFEST_FILE_V1
            _write_manifest_file(manifest_path, manifest)
            for member in (stage_data_path, manifest_path):
                member.chmod(stat.S_IRUSR)
                descriptor_fd = os.open(
                    member, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
                )
                try:
                    os.fsync(descriptor_fd)
                finally:
                    os.close(descriptor_fd)
            members = {entry.name for entry in os.scandir(staging_root)}
            if members != {_LEGACY_DATA_FILE_V1, _LEGACY_MANIFEST_FILE_V1}:
                raise NativeSnapshotPublicationError(
                    "native snapshot stage member set differs before sealing"
                )
            seal: dict[str, Mapping[str, int]] = {
                _LEGACY_DATA_FILE_V1: _regular_file_identity(stage_data_path),
                _LEGACY_MANIFEST_FILE_V1: _regular_file_identity(manifest_path),
            }
            staging_root.chmod(stat.S_IRUSR | stat.S_IXUSR)
            _fsync_directory(staging_root)
            seal[_STAGE_DIRECTORY_SEAL_KEY_V1] = _directory_identity(staging_root)
            manifest_elapsed_ns = perf_counter_ns() - manifest_started_ns
            self._worker_manifest_elapsed_ns_total += manifest_elapsed_ns
            self._worker_manifest_elapsed_ns_max = max(
                self._worker_manifest_elapsed_ns_max, manifest_elapsed_ns
            )
            return {
                **manifest,
                "path": str(staging_root),
                "destination": str(destination),
                "stage_schema_version": NATIVE_SNAPSHOT_STAGE_SCHEMA_V1,
                "stage_file_state": "SEALED_STAGE_NOT_PUBLISHED",
                "capture_metrics": dict(request.capture_metrics),
                "encoder_timing": dict(descriptor.get("timing", {})),
                "native_counters": {
                    "active_agent_count": descriptor.get("active_agent_count"),
                    "d2h_bytes": request.capture_metrics.get("d2h_bytes"),
                    "uncompressed_bytes": descriptor.get("uncompressed_bytes"),
                    "compressed_bytes": descriptor.get("compressed_bytes"),
                    "row_count": descriptor.get("row_count"),
                    "captured_world_count": descriptor.get("captured_world_count"),
                },
                "publication_timing": {
                    "worker_manifest_and_stage_seal_ns": manifest_elapsed_ns,
                },
                _STAGE_SEAL_KEY_V1: seal,
            }
        except BaseException:
            owned_identity = self._owned_stages.get(staging_root)
            if os.path.lexists(staging_root) and owned_identity is not None:
                _remove_exact_stage(
                    staging_root,
                    expected_directory_identity=owned_identity,
                )
            self._owned_stages.pop(staging_root, None)
            raise

    def submit(
        self,
        runtime_owner: object,
        absolute_tick: int,
        destination: str | os.PathLike[str],
    ) -> NativeSnapshotPublicationTokenV1:
        if self._closed:
            raise NativeSnapshotPublicationError(
                "native snapshot publication service is closed"
            )
        boundary_started_ns = perf_counter_ns()
        root = _ensure_publication_parent(
            destination, label="native snapshot destination"
        )
        if os.path.lexists(root):
            raise NativeSnapshotPublicationError(
                "native snapshot final publication root already exists"
            )
        staging_root = root.parent / (
            f".{root.name}.{self._stage_namespace}.native-snapshot-v1.stage"
        )
        if os.path.lexists(staging_root):
            raise NativeSnapshotPublicationError(
                "native snapshot staging root already exists"
            )
        try:
            parent_descriptor = os.open(
                staging_root.parent,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
            )
            try:
                os.mkdir(staging_root.name, mode=0o700, dir_fd=parent_descriptor)
                stage_descriptor = os.open(
                    staging_root.name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                    dir_fd=parent_descriptor,
                )
                try:
                    stage_state = os.fstat(stage_descriptor)
                    self._owned_stages[staging_root] = (
                        int(stage_state.st_dev),
                        int(stage_state.st_ino),
                    )
                finally:
                    os.close(stage_descriptor)
                os.fsync(parent_descriptor)
            finally:
                os.close(parent_descriptor)
            if runtime_owner is not self._runtime_owner:
                raise NativeSnapshotPublicationError(
                    "native snapshot submission belongs to another runtime owner"
                )
            context = {
                "absolute_tick": absolute_tick,
                "staging_root": str(staging_root),
                "stage_data_path": str(staging_root / _LEGACY_DATA_FILE_V1),
                "destination": str(root),
            }
            submission = self._service.submit(
                absolute_tick=absolute_tick,
                context=context,
                block=False,
            )
        except BaseException:
            owned_identity = self._owned_stages.get(staging_root)
            if os.path.lexists(staging_root) and owned_identity is not None:
                _remove_exact_stage(
                    staging_root,
                    expected_directory_identity=owned_identity,
                )
            self._owned_stages.pop(staging_root, None)
            raise
        token = NativeSnapshotPublicationTokenV1(
            submission=submission,
            absolute_tick=absolute_tick,
            staging_root=str(staging_root),
            destination=str(root),
            producer_boundary_started_ns=boundary_started_ns,
            producer_handoff_completed_ns=perf_counter_ns(),
        )
        self._jobs.append(_NativeSnapshotPublicationJobV1(token=token))
        return token

    def _job_for_token(
        self, token: NativeSnapshotPublicationTokenV1
    ) -> _NativeSnapshotPublicationJobV1:
        if not isinstance(token, NativeSnapshotPublicationTokenV1):
            raise NativeSnapshotPublicationError(
                "native snapshot publication token is invalid"
            )
        job = next(
            (
                candidate
                for candidate in self._jobs
                if candidate.token.submission.submission_id
                == token.submission.submission_id
            ),
            None,
        )
        if job is None or job.token != token:
            raise NativeSnapshotPublicationError(
                "native snapshot publication token is not pending"
            )
        return job

    def ready(self, token: NativeSnapshotPublicationTokenV1) -> bool:
        job = self._job_for_token(token)
        return bool(self._jobs and self._jobs[0] is job and self._service.ready)

    def result(
        self,
        token: NativeSnapshotPublicationTokenV1,
        *,
        producer_backpressure: bool = False,
    ) -> dict[str, Any]:
        job = self._job_for_token(token)
        if not self._jobs or self._jobs[0] is not job:
            raise NativeSnapshotPublicationError(
                "native snapshot results must be consumed in FIFO order"
            )
        if job.result_consumed:
            raise NativeSnapshotPublicationError(
                "native snapshot result was already consumed"
            )
        completion = self._service.settle_oldest(
            block=True if producer_backpressure else self._service.ready,
        )
        if (
            completion.submission_id != token.submission.submission_id
            or completion.absolute_tick != token.absolute_tick
            or not isinstance(completion.result, Mapping)
        ):
            raise NativeSnapshotPublicationError(
                "native snapshot completion differs from its admitted token"
            )
        result = dict(completion.result)
        if (
            result.get("path") != token.staging_root
            or result.get("destination") != token.destination
            or result.get("absolute_tick") != token.absolute_tick
        ):
            raise NativeSnapshotPublicationError(
                "native snapshot staged receipt differs from its admitted token"
            )
        job.result_consumed = True
        return result

    def publish(
        self,
        token: NativeSnapshotPublicationTokenV1,
        staged_receipt: Mapping[str, Any],
    ) -> dict[str, Any]:
        job = self._job_for_token(token)
        if not self._jobs or self._jobs[0] is not job or not job.result_consumed:
            raise NativeSnapshotPublicationError(
                "native snapshot publication must follow FIFO result consumption"
            )
        started_ns = perf_counter_ns()
        try:
            published = publish_native_snapshot_stage_v1(
                staged_receipt,
                destination=token.destination,
            )
        except BaseException:
            self._publication_failed += 1
            raise
        elapsed_ns = perf_counter_ns() - started_ns
        self._publication_elapsed_ns_total += elapsed_ns
        self._publication_elapsed_ns_max = max(
            self._publication_elapsed_ns_max, elapsed_ns
        )
        self._owned_stages.pop(Path(token.staging_root), None)
        return published

    def acknowledge_published(
        self,
        token: NativeSnapshotPublicationTokenV1,
        *,
        published_receipt: Mapping[str, Any],
    ) -> None:
        job = self._job_for_token(token)
        if not self._jobs or self._jobs[0] is not job or not job.result_consumed:
            raise NativeSnapshotPublicationError(
                "native snapshot acknowledgement must follow FIFO publication"
            )
        if (
            published_receipt.get("path") != token.destination
            or published_receipt.get("absolute_tick") != token.absolute_tick
        ):
            raise NativeSnapshotPublicationError(
                "native snapshot published receipt differs from its token"
            )
        self._jobs.popleft()
        self._publication_completed += 1

    def poll_failure(self) -> None:
        if self._jobs and self.ready(self._jobs[0].token):
            # Failure is raised by result(), where provider settlement can apply
            # exact stage/final-root reconciliation.
            return
        telemetry = self._service.telemetry()
        if telemetry.get("service_failed") is True and not self._jobs:
            raise NativeSnapshotPublicationError(
                "native snapshot worker failed without a pending publication"
            )

    def close(self) -> None:
        if self._closed:
            return
        if self._jobs:
            raise NativeSnapshotPublicationError(
                "native snapshot publication cannot close with pending jobs"
            )
        self._service.shutdown(drain=True, raise_on_failure=True)
        self._closed = True

    def abort(self) -> None:
        if self._closed:
            return
        errors: list[BaseException] = []
        try:
            self._service.shutdown(
                cancel_pending=True,
                drain=True,
                raise_on_failure=False,
            )
        except BaseException as exc:
            errors.append(exc)
        for stage, identity in tuple(self._owned_stages.items()):
            try:
                _remove_exact_stage(
                    stage,
                    expected_directory_identity=identity,
                )
                self._owned_stages.pop(stage, None)
            except BaseException as exc:
                errors.append(exc)
        self._jobs.clear()
        self._aborted = True
        self._closed = True
        if errors:
            raise NativeSnapshotPublicationError(
                f"native snapshot abort did not clean every owned stage: {errors[0]}"
            ) from errors[0]

    def telemetry(self) -> dict[str, Any]:
        core = dict(self._service.telemetry())
        return {
            "schema_version": NATIVE_SNAPSHOT_PUBLICATION_SCHEMA_V1,
            "implementation": self._implementation_label,
            "publication_artifact_mode": "legacy_jsonl_gzip_compatibility_v1",
            "native_capsule_published": False,
            "plan_sha256": self._compiled_plan.plan_sha256,
            "snapshot_schema_id": self._plan["snapshot_schema_id"],
            "state_capture_mode": self._state_capture_mode,
            "execution_publication_context_sha256": self._plan[
                "execution_publication_context_sha256"
            ],
            "pending_jobs": len(self._jobs),
            "owned_partial_stage_count": sum(
                os.path.lexists(path) for path in self._owned_stages
            ),
            "publication_completed_count": self._publication_completed,
            "publication_failed_count": self._publication_failed,
            "publication_elapsed_ns_total": self._publication_elapsed_ns_total,
            "publication_elapsed_ns_max": self._publication_elapsed_ns_max,
            "worker_manifest_elapsed_ns_total": (
                self._worker_manifest_elapsed_ns_total
            ),
            "worker_manifest_elapsed_ns_max": self._worker_manifest_elapsed_ns_max,
            "closed": self._closed,
            "aborted": self._aborted,
            "core_service": core,
            "resident_pinned_bytes": core.get("resident_pinned_bytes"),
            "horizon_dependent_buffer_bytes": 0,
        }


def bind_native_snapshot_publication_service_v1(
    *,
    compiled_plan: CompiledCudaSnapshotPlanV1 | bytes,
    runtime_owner: object,
    slot_count: int,
    loaded_extension: Any | None = None,
    **testing_options: Any,
) -> NativeSnapshotPublicationServiceV1:
    """Bind one native plan and fixed-size slot pool before any boundary."""

    return NativeSnapshotPublicationServiceV1(
        compiled_plan=compiled_plan,
        runtime_owner=runtime_owner,
        slot_count=slot_count,
        loaded_extension=loaded_extension,
        **testing_options,
    )


__all__ = [
    "NATIVE_SNAPSHOT_DEFAULT_IMPLEMENTATION_V1",
    "NATIVE_SNAPSHOT_IMPLEMENTATION_ENV_V1",
    "NATIVE_SNAPSHOT_IMPLEMENTATIONS_V1",
    "NATIVE_SNAPSHOT_NATIVE_IMPLEMENTATIONS_V1",
    "NATIVE_SNAPSHOT_PUBLICATION_SCHEMA_V1",
    "NATIVE_SNAPSHOT_STAGE_SCHEMA_V1",
    "NativeSnapshotPublicationError",
    "NativeSnapshotPublicationServiceV1",
    "NativeSnapshotPublicationTokenV1",
    "NativeSnapshotVisibleRootError",
    "bind_native_snapshot_publication_service_v1",
    "inspect_existing_native_snapshot_root_v1",
    "native_snapshot_implementation_v1",
    "native_snapshot_implementation_is_native_v1",
    "publish_native_snapshot_stage_v1",
    "remove_native_snapshot_root_v1",
]
