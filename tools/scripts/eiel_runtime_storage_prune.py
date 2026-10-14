#!/usr/bin/env python3
"""Dry-run-first EIEL runtime storage pruning helper."""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager, nullcontext
import ctypes
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import json
import os
import shutil
import stat
import tempfile
from pathlib import Path
from collections.abc import Callable
from typing import Any, Mapping

from engine.campaigns.canonical import canonical_sha256, pretty_json
from engine.campaigns.evidence_certificate_api import (
    ArtifactLifecycleError,
    DATA2_FILESYSTEM_ROOT,
    DATA2_LIFECYCLE_MANAGEMENT_ROOT,
    RunStorageGuard,
    build_storage_disposition_receipt_v1,
    disposition_manifest_authorizes_prune_v1,
    evidence_root_requires_permanent_capsule_v1,
    legacy_reconstruction_is_accepted_v1,
    load_data2_storage_availability_policy_v1,
    require_diagnostic_local_live_state_clear_v1,
    run_artifact_transaction_fence_v1,
    storage_availability_policy_binding_v1,
    validate_artifact_disposition_manifest_v1,
    validate_certificate_request_binding_v1,
    validate_certificate_source_manifest_bindings_v1,
    validate_evidence_certificate_request_v1,
    validate_permanent_evidence_publication_v1,
    validate_reclamation_intent_v1,
    write_storage_disposition_receipt_v1,
)
from engine.campaigns.decision_evidence_retention import (
    DecisionEvidenceRetentionError,
    guard_prune_plan,
    load_manifest,
)

CATEGORIES = (
    "event_jsonl",
    "developmental_traces",
    "intermediate_checkpoints",
    "debug_traces",
    "scratch",
    "reproducible_derived",
)


def _reservation_root_for(run_root: Path) -> Path:
    lexical_root = Path(os.path.abspath(os.fspath(run_root.expanduser())))
    resolved_root = lexical_root.resolve(strict=False)
    lexical_data2 = Path(os.path.abspath(os.fspath(DATA2_FILESYSTEM_ROOT.expanduser())))
    resolved_data2 = lexical_data2.resolve(strict=False)
    lexical_is_data2 = _is_beneath(lexical_root, lexical_data2)
    resolved_is_data2 = _is_beneath(resolved_root, resolved_data2)
    if lexical_is_data2 != resolved_is_data2 or (
        lexical_is_data2 and lexical_root != resolved_root
    ):
        raise ArtifactLifecycleError(
            "release fence configured runtime root alias or symlink resolution is ambiguous"
        )
    if resolved_is_data2:
        lexical_lifecycle = Path(
            os.path.abspath(os.fspath(DATA2_LIFECYCLE_MANAGEMENT_ROOT.expanduser()))
        )
        resolved_lifecycle = lexical_lifecycle.resolve(strict=False)
        if (
            not _is_beneath(lexical_lifecycle, lexical_data2)
            or not _is_beneath(resolved_lifecycle, resolved_data2)
            or lexical_lifecycle != resolved_lifecycle
        ):
            raise ArtifactLifecycleError(
                "canonical configured runtime root lifecycle reservation root is aliased or escapes"
            )
        return resolved_lifecycle / "reservations"
    for candidate in (resolved_root, *resolved_root.parents):
        if candidate.name == "causyl-runtime":
            return candidate / "lifecycle" / "reservations"
    return resolved_root.parent / ".eiel-storage-reservations"


def _acquire_release_fence(run_root: Path) -> Any:
    """Exclude canonical reservations throughout validation and release."""

    reservation_root = _reservation_root_for(run_root)
    reservation_root.mkdir(parents=True, exist_ok=True)
    handle = (reservation_root / ".lock").open("a+b")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    return handle


def _is_beneath(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _archive_budget_scope_root(
    *,
    runtime_root: Path,
    archive_root: Path,
    requested_budget_root: Path | None,
) -> Path:
    """Bind every canonical configured runtime root archive admission to one lifecycle ledger."""

    canonical_data2 = DATA2_FILESYSTEM_ROOT.expanduser().resolve()
    canonical_lifecycle = DATA2_LIFECYCLE_MANAGEMENT_ROOT.expanduser().resolve()
    if _is_beneath(runtime_root, canonical_data2) or _is_beneath(
        archive_root, canonical_data2
    ):
        if (
            requested_budget_root is not None
            and requested_budget_root.expanduser().resolve() != canonical_lifecycle
        ):
            raise ArtifactLifecycleError(
                "canonical configured runtime root archive budget scope cannot be overridden"
            )
        return canonical_lifecycle
    return (
        archive_root
        if requested_budget_root is None
        else requested_budget_root.expanduser().resolve()
    )


def _validate_cold_archive_separation(
    *, runtime_root: Path, requested_archive_root: Path
) -> Path:
    """Require canonical configured runtime root archives to land on an observed other device."""

    runtime = runtime_root.expanduser().resolve()
    lexical_data2 = Path(os.path.abspath(os.fspath(DATA2_FILESYSTEM_ROOT.expanduser())))
    resolved_data2 = lexical_data2.resolve(strict=False)
    if not _is_beneath(runtime, resolved_data2):
        return requested_archive_root.expanduser().resolve()
    lexical_archive = Path(
        os.path.abspath(os.fspath(requested_archive_root.expanduser()))
    )
    try:
        resolved_archive = lexical_archive.resolve(strict=True)
    except OSError as exc:
        raise ArtifactLifecycleError(
            "canonical configured runtime root cold archive root must exist on an observed filesystem"
        ) from exc
    if lexical_archive != resolved_archive:
        raise ArtifactLifecycleError(
            "canonical configured runtime root cold archive root cannot use a filesystem alias"
        )
    if not resolved_archive.is_dir() or _is_beneath(resolved_archive, resolved_data2):
        raise ArtifactLifecycleError(
            "canonical configured runtime root cold archive root must be a disjoint directory"
        )
    try:
        runtime_device = _observed_filesystem_device(runtime)
        archive_device = _observed_filesystem_device(resolved_archive)
    except OSError as exc:
        raise ArtifactLifecycleError(
            "canonical archive filesystem identity is unavailable"
        ) from exc
    if archive_device == runtime_device:
        raise ArtifactLifecycleError(
            "canonical configured runtime root cold archive root must use a distinct filesystem device"
        )
    return resolved_archive


def _observed_filesystem_device(path: Path) -> int:
    return int(os.stat(path).st_dev)


def _open_directory_beneath(
    root_fd: int, parts: tuple[str, ...], *, create: bool = False
) -> int:
    """Open a directory chain relative to a pinned root without following links."""

    current = os.dup(root_fd)
    try:
        for part in parts:
            if part in {"", ".", ".."}:
                raise RuntimeError("unsafe descriptor-relative directory component")
            if create:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=current)
                except FileExistsError:
                    pass
            next_fd = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=current,
            )
            os.close(current)
            current = next_fd
        return current
    except BaseException:
        os.close(current)
        raise


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
    )


def _revalidate_current_disposition_manifest(
    manifest_path: Path,
    *,
    expected_manifest: Mapping[str, Any],
    runtime_root: Path,
) -> dict[str, Any]:
    """Re-open exact current authority without following a replaced pathname."""

    expected = validate_artifact_disposition_manifest_v1(expected_manifest)
    lexical_path = Path(os.path.abspath(os.fspath(manifest_path.expanduser())))
    resolved_root = runtime_root.expanduser().resolve()
    try:
        lexical_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ArtifactLifecycleError(
            "current disposition manifest path escapes the runtime root"
        ) from exc
    try:
        before = os.lstat(lexical_path)
    except FileNotFoundError as exc:
        raise ArtifactLifecycleError(
            "current disposition manifest disappeared before release"
        ) from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or stat.S_ISLNK(before.st_mode)
    ):
        raise ArtifactLifecycleError(
            "current disposition manifest is not a single-link regular file"
        )
    descriptor = os.open(lexical_path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(descriptor)
        if _stat_identity(opened) != _stat_identity(before) or opened.st_nlink != 1:
            raise ArtifactLifecycleError(
                "current disposition manifest changed while being opened"
            )
        with os.fdopen(os.dup(descriptor), "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        after = os.fstat(descriptor)
        after_path = os.lstat(lexical_path)
        if (
            _stat_identity(after) != _stat_identity(opened)
            or _stat_identity(after_path) != _stat_identity(opened)
            or after.st_nlink != 1
            or after_path.st_nlink != 1
        ):
            raise ArtifactLifecycleError(
                "current disposition manifest changed while being revalidated"
            )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactLifecycleError(
            "current disposition manifest is unreadable before release"
        ) from exc
    finally:
        os.close(descriptor)
    if not isinstance(raw, Mapping):
        raise ArtifactLifecycleError(
            "current disposition manifest is malformed before release"
        )
    current = validate_artifact_disposition_manifest_v1(raw)
    if current.get("manifest_sha256") != expected.get("manifest_sha256") or current.get(
        "runtime_root"
    ) != str(resolved_root):
        raise ArtifactLifecycleError(
            "current disposition manifest was replaced or reclassified before release"
        )
    return current


def _remove_sealed_tree_exact(
    path: Path, *, expected_bytes: int, expected_tree_sha256: str
) -> None:
    """Remove only the exact sealed members; unexpected late members block rmdir."""

    files: list[tuple[Path, tuple[int, int, int, int]]] = []
    directories: list[Path] = []
    entries: list[dict[str, Any]] = []
    total_bytes = 0
    for member in sorted(path.rglob("*"), key=lambda item: str(item)):
        observed = os.lstat(member)
        if stat.S_ISLNK(observed.st_mode):
            raise RuntimeError(f"sealed tree contains a symlink: {member}")
        if stat.S_ISDIR(observed.st_mode):
            directories.append(member)
            continue
        if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
            raise RuntimeError(
                f"sealed tree contains an unsupported/link member: {member}"
            )
        digest = hashlib.sha256()
        with member.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        after = os.lstat(member)
        identity = _stat_identity(observed)
        if _stat_identity(after) != identity:
            raise RuntimeError(f"sealed tree changed while hashing: {member}")
        relative = str(member.relative_to(path)).replace("\\", "/")
        total_bytes += int(observed.st_size)
        entries.append(
            {
                "path": relative,
                "bytes": int(observed.st_size),
                "sha256": digest.hexdigest(),
            }
        )
        files.append((member, identity))
    if (
        total_bytes != expected_bytes
        or canonical_sha256(entries) != expected_tree_sha256
    ):
        raise RuntimeError(
            "staged sealed tree differs from manifest before exact release"
        )
    _make_sealed_tree_directories_removable(path, directories)
    for member, identity in files:
        if _stat_identity(os.lstat(member)) != identity:
            raise RuntimeError(f"sealed tree member changed before release: {member}")
        member.unlink()
    for directory in sorted(
        {path, *(member.parent for member, _identity in files)},
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        _fsync_directory(directory)
    for directory in sorted(
        directories, key=lambda item: len(item.parts), reverse=True
    ):
        directory.rmdir()
        _fsync_directory(directory.parent)


def _sealed_tree_member_inventory(path: Path) -> list[dict[str, Any]]:
    members: list[dict[str, Any]] = []
    for member in sorted(path.rglob("*"), key=lambda item: str(item)):
        observed = os.lstat(member)
        relative = str(member.relative_to(path)).replace("\\", "/")
        if stat.S_ISLNK(observed.st_mode):
            raise RuntimeError(f"sealed tree contains a symlink: {member}")
        if stat.S_ISDIR(observed.st_mode):
            members.append(
                {
                    "path": relative,
                    "type": "directory",
                    "device": int(observed.st_dev),
                    "inode": int(observed.st_ino),
                }
            )
            continue
        if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
            raise RuntimeError(
                f"sealed tree contains an unsupported/link member: {member}"
            )
        digest = _file_digest(member)
        after = os.lstat(member)
        if _stat_identity(after) != _stat_identity(observed):
            raise RuntimeError(f"sealed tree member changed while sealing: {member}")
        members.append(
            {
                "path": relative,
                "type": "file",
                "device": int(observed.st_dev),
                "inode": int(observed.st_ino),
                "bytes": int(observed.st_size),
                "sha256": digest,
            }
        )
    return members


def _validate_sealed_tree_member_authority(
    members: list[dict[str, Any]],
    row: Mapping[str, Any],
) -> None:
    files = [
        {
            "path": member["path"],
            "bytes": member["bytes"],
            "sha256": member["sha256"],
        }
        for member in members
        if member.get("type") == "file"
    ]
    if sum(int(member["bytes"]) for member in files) != int(
        row.get("bytes", -1)
    ) or canonical_sha256(files) != row.get("tree_sha256"):
        raise RuntimeError("sealed tree member inventory differs from manifest")


def _make_sealed_tree_directories_removable(
    path: Path,
    directories: list[Path],
) -> None:
    """Grant only owner-write permission needed by an authorized exact removal.

    Checkpoint publishers may seal a tree read-only after publication.  The
    exact manifest and member identities have already been validated before
    this helper is called; adding the owner write bit is therefore limited to
    the directories that the same authorized removal is about to delete.  The
    helper deliberately does not broaden group/other permissions or touch
    regular-file modes.
    """

    for directory in sorted(
        {path, *directories},
        key=lambda candidate: (len(candidate.parts), str(candidate)),
    ):
        observed = os.lstat(directory)
        if not stat.S_ISDIR(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
            raise RuntimeError(
                f"sealed tree directory changed before authorized removal: {directory}"
            )
        mode = stat.S_IMODE(observed.st_mode)
        if mode & stat.S_IWUSR:
            continue
        os.chmod(directory, mode | stat.S_IWUSR, follow_symlinks=False)
        after = os.lstat(directory)
        if _stat_identity(after) != _stat_identity(observed):
            raise RuntimeError(
                f"sealed tree directory changed while enabling removal: {directory}"
            )


def _remove_sealed_tree_remainder(
    path: Path,
    *,
    expected_members: list[dict[str, Any]],
) -> None:
    expected = {str(member["path"]): dict(member) for member in expected_members}
    if len(expected) != len(expected_members):
        raise RuntimeError("sealed tree staged member authority is duplicated")
    actual = _sealed_tree_member_inventory(path)
    for member in actual:
        authority = expected.get(str(member["path"]))
        if authority != member:
            raise RuntimeError(
                f"staged sealed tree remainder differs: {member['path']}"
            )
    # Exact checkpoint packages intentionally contain read-only directories.
    # The release actuator has already atomically moved this fully validated
    # inode tree into its journal-bound staging root, so make only those staged
    # directories owner-removable before unlinking their authenticated members.
    # No source/runtime path or file payload is made writable.
    directories = [member for member in actual if member["type"] == "directory"]
    writable_directories = [
        (path, os.lstat(path)),
        *[
            (path / str(member["path"]), os.lstat(path / str(member["path"])))
            for member in directories
        ],
    ]
    for target, observed in writable_directories:
        if not stat.S_ISDIR(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
            raise RuntimeError(f"staged sealed tree directory changed: {target}")
        if observed.st_mode & (stat.S_IWUSR | stat.S_IXUSR) != (
            stat.S_IWUSR | stat.S_IXUSR
        ):
            os.chmod(
                target,
                stat.S_IMODE(observed.st_mode) | stat.S_IWUSR | stat.S_IXUSR,
                follow_symlinks=False,
            )
            after = os.lstat(target)
            if (
                int(after.st_dev) != int(observed.st_dev)
                or int(after.st_ino) != int(observed.st_ino)
                or not stat.S_ISDIR(after.st_mode)
                or stat.S_ISLNK(after.st_mode)
            ):
                raise RuntimeError(
                    f"staged sealed tree directory changed during release: {target}"
                )
            _fsync_directory(target)
    files = [member for member in actual if member["type"] == "file"]
    directories = [
        path / str(member["path"])
        for member in actual
        if member["type"] == "directory"
    ]
    _make_sealed_tree_directories_removable(path, directories)
    for member in files:
        target = path / str(member["path"])
        observed = os.lstat(target)
        if (
            int(observed.st_dev) != int(member["device"])
            or int(observed.st_ino) != int(member["inode"])
            or int(observed.st_size) != int(member["bytes"])
            or observed.st_nlink != 1
            or _file_digest(target) != member["sha256"]
        ):
            raise RuntimeError(
                f"staged sealed tree member changed before release: {member['path']}"
            )
        target.unlink()
        _fsync_directory(target.parent)
    for member in sorted(
        [member for member in actual if member["type"] == "directory"],
        key=lambda value: len(Path(str(value["path"])).parts),
        reverse=True,
    ):
        target = path / str(member["path"])
        observed = os.lstat(target)
        if (
            int(observed.st_dev) != int(member["device"])
            or int(observed.st_ino) != int(member["inode"])
            or not stat.S_ISDIR(observed.st_mode)
        ):
            raise RuntimeError(
                f"staged sealed tree directory changed: {member['path']}"
            )
        target.rmdir()
        _fsync_directory(target.parent)


def _tree_digest(path: Path) -> str:
    """Match the sealed-tree digest used by ArtifactDispositionManifestV1."""

    entries: list[dict[str, Any]] = []
    for member in sorted(path.rglob("*"), key=lambda item: str(item)):
        observed = os.lstat(member)
        if stat.S_ISLNK(observed.st_mode):
            raise RuntimeError(f"sealed tree contains a symlink: {member}")
        if stat.S_ISDIR(observed.st_mode):
            continue
        if not stat.S_ISREG(observed.st_mode):
            raise RuntimeError(f"sealed tree contains a non-regular member: {member}")
        if observed.st_nlink != 1:
            raise RuntimeError(f"sealed tree contains a hard link: {member}")
        digest = hashlib.sha256()
        with member.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        after = os.lstat(member)
        if (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) != (
            observed.st_dev,
            observed.st_ino,
            observed.st_size,
            observed.st_mtime_ns,
        ):
            raise RuntimeError(f"sealed tree changed while hashing: {member}")
        entries.append(
            {
                "path": str(member.relative_to(path)).replace("\\", "/"),
                "bytes": int(observed.st_size),
                "sha256": digest.hexdigest(),
            }
        )
    return canonical_sha256(entries)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_destination(
    root: Path, archive_root: Path, relative: str, run_id: str
) -> Path:
    destination = archive_root / run_id / relative
    try:
        destination.parent.resolve(strict=False).relative_to(archive_root.resolve())
    except ValueError as exc:
        raise RuntimeError(
            f"archive destination escapes archive root: {relative}"
        ) from exc
    if (
        destination.resolve(strict=False) == root
        or root in destination.resolve(strict=False).parents
    ):
        raise RuntimeError("archive destination must not be inside the runtime root")
    return destination


def _archive_version_destination(
    root: Path,
    archive_root: Path,
    relative: str,
    run_id: str,
    row: Mapping[str, Any],
) -> Path:
    """Return the deterministic destination for a changed logical object.

    The ordinary run-id/path destination remains the readable first version.
    If a later sealed manifest binds different bytes to that same logical
    identity, the digest-addressed namespace preserves both versions without
    overwriting either one.
    """

    digest_name = (
        "tree_sha256"
        if str(row.get("artifact_type") or "file") == "sealed_tree"
        else "sha256"
    )
    digest = str(row.get(digest_name) or "")
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise RuntimeError(
            f"durable row lacks a valid {digest_name} for versioned archival: {relative}"
        )
    versioned_relative = str(
        Path(".eiel-object-versions-v1") / run_id / digest_name / digest / relative
    ).replace("\\", "/")
    return _archive_destination(root, archive_root, versioned_relative, "")


_COLD_PROFILE_BUDGET_SCHEMA = "eiel_cold_profile_budget_receipt_v1"
_COLD_PROFILE_BUDGET_NAME = ".eiel-cold-profile-budget-v1.json"
_COLD_PROFILE_BUDGET_LOCK_NAME = ".eiel-cold-profile-budget-v1.lock"
_COLD_PROFILE_BUDGET_MAX_OBJECTS = 65_536
_COLD_PROFILE_BUDGET_SCOPE = "ALL_RETAIN_DURABLE_R3_R5_CONSERVATIVE"


def _cold_profile_budget_paths(archive_root: Path) -> tuple[Path, Path]:
    return (
        archive_root / _COLD_PROFILE_BUDGET_NAME,
        archive_root / _COLD_PROFILE_BUDGET_LOCK_NAME,
    )


def _cold_profile_budget_row(
    *,
    manifest_sha256: str,
    run_id: str,
    row: Mapping[str, Any],
    destination: Path,
    status: str,
) -> dict[str, Any]:
    digest_name = (
        "tree_sha256"
        if str(row.get("artifact_type") or "file") == "sealed_tree"
        else "sha256"
    )
    identity = {
        "run_id": run_id,
        "path": str(row["path"]),
        "bytes": int(row["bytes"]),
        "artifact_family": str(row.get("artifact_family") or ""),
        "artifact_type": str(row.get("artifact_type") or "file"),
        "digest_name": digest_name,
        "digest": str(row.get(digest_name) or ""),
        "destination": str(destination),
    }
    object_identity_sha256 = canonical_sha256(identity)
    return {
        **identity,
        "object_identity_sha256": object_identity_sha256,
        "staging": str(_archive_staging_path(destination, object_identity_sha256)),
        "manifest_sha256": manifest_sha256,
        "status": status,
    }


def _empty_cold_profile_budget_receipt(
    *,
    budget_scope_root: Path,
    policy_binding: Mapping[str, Any],
    max_object_bytes: int,
    max_aggregate_bytes: int,
) -> dict[str, Any]:
    return {
        "schema_version": _COLD_PROFILE_BUDGET_SCHEMA,
        "budget_scope_root": str(budget_scope_root),
        "budget_scope": _COLD_PROFILE_BUDGET_SCOPE,
        "storage_availability_policy": dict(policy_binding),
        "max_object_bytes": int(max_object_bytes),
        "max_aggregate_bytes": int(max_aggregate_bytes),
        "objects": [],
        "reserved_bytes": 0,
        "verified_bytes": 0,
        "updated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def _validate_cold_profile_budget_receipt(
    value: Mapping[str, Any],
    *,
    budget_scope_root: Path,
    policy_binding: Mapping[str, Any],
    max_object_bytes: int,
    max_aggregate_bytes: int,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactLifecycleError("cold profile budget receipt must be an object")
    core = dict(value)
    supplied = core.pop("receipt_sha256", None)
    if supplied != canonical_sha256(core):
        raise ArtifactLifecycleError("cold profile budget receipt digest differs")
    expected_fields = {
        "schema_version",
        "budget_scope_root",
        "budget_scope",
        "storage_availability_policy",
        "max_object_bytes",
        "max_aggregate_bytes",
        "objects",
        "reserved_bytes",
        "verified_bytes",
        "updated_at_utc",
    }
    if (
        set(core) != expected_fields
        or core.get("schema_version") != _COLD_PROFILE_BUDGET_SCHEMA
    ):
        raise ArtifactLifecycleError("cold profile budget receipt schema differs")
    if (
        core.get("budget_scope_root") != str(budget_scope_root)
        or core.get("budget_scope") != _COLD_PROFILE_BUDGET_SCOPE
        or core.get("storage_availability_policy") != dict(policy_binding)
        or core.get("max_object_bytes") != int(max_object_bytes)
        or core.get("max_aggregate_bytes") != int(max_aggregate_bytes)
    ):
        raise ArtifactLifecycleError(
            "cold profile budget receipt policy binding differs"
        )
    objects = core.get("objects")
    if not isinstance(objects, list) or len(objects) > _COLD_PROFILE_BUDGET_MAX_OBJECTS:
        raise ArtifactLifecycleError(
            "cold profile budget receipt object inventory is invalid"
        )
    normalized: list[dict[str, Any]] = []
    identities: set[str] = set()
    destinations: set[str] = set()
    row_fields = {
        "run_id",
        "path",
        "bytes",
        "artifact_family",
        "artifact_type",
        "digest_name",
        "digest",
        "destination",
        "object_identity_sha256",
        "staging",
        "manifest_sha256",
        "status",
    }
    for raw in objects:
        if not isinstance(raw, Mapping) or set(raw) != row_fields:
            raise ArtifactLifecycleError("cold profile budget object row is malformed")
        row = dict(raw)
        if (
            type(row.get("bytes")) is not int
            or row["bytes"] < 0
            or row["bytes"] > max_object_bytes
            or row.get("status") not in {"RESERVED", "VERIFIED"}
            or row.get("digest_name") not in {"sha256", "tree_sha256"}
            or not isinstance(row.get("digest"), str)
            or len(row["digest"]) != 64
        ):
            raise ArtifactLifecycleError(
                "cold profile budget object identity is invalid"
            )
        destination = Path(str(row["destination"])).expanduser()
        staging = Path(str(row["staging"])).expanduser()
        if not destination.is_absolute() or ".." in destination.parts:
            raise ArtifactLifecycleError("cold profile budget destination is invalid")
        identity_core = {
            name: row[name]
            for name in (
                "run_id",
                "path",
                "bytes",
                "artifact_family",
                "artifact_type",
                "digest_name",
                "digest",
                "destination",
            )
        }
        identity = str(row["object_identity_sha256"])
        if identity != canonical_sha256(identity_core):
            raise ArtifactLifecycleError("cold profile budget object digest differs")
        if staging != _archive_staging_path(destination, identity):
            raise ArtifactLifecycleError("cold profile staging identity differs")
        if identity in identities or str(destination) in destinations:
            raise ArtifactLifecycleError(
                "cold profile budget object identity is duplicated"
            )
        identities.add(identity)
        destinations.add(str(destination))
        normalized.append(row)
    reserved_bytes = sum(int(row["bytes"]) for row in normalized)
    verified_bytes = sum(
        int(row["bytes"]) for row in normalized if row["status"] == "VERIFIED"
    )
    if (
        core.get("reserved_bytes") != reserved_bytes
        or core.get("verified_bytes") != verified_bytes
        or reserved_bytes > max_aggregate_bytes
    ):
        raise ArtifactLifecycleError("cold profile budget byte accounting differs")
    return {**core, "objects": normalized, "receipt_sha256": supplied}


def _load_cold_profile_budget_receipt(
    path: Path,
    *,
    budget_scope_root: Path,
    policy_binding: Mapping[str, Any],
    max_object_bytes: int,
    max_aggregate_bytes: int,
) -> dict[str, Any]:
    if not path.exists() and not path.is_symlink():
        return _empty_cold_profile_budget_receipt(
            budget_scope_root=budget_scope_root,
            policy_binding=policy_binding,
            max_object_bytes=max_object_bytes,
            max_aggregate_bytes=max_aggregate_bytes,
        )
    if path.is_symlink() or not path.is_file():
        raise ArtifactLifecycleError(
            "cold profile budget receipt is not a regular file"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactLifecycleError(
            "cold profile budget receipt is unreadable"
        ) from exc
    return _validate_cold_profile_budget_receipt(
        value,
        budget_scope_root=budget_scope_root,
        policy_binding=policy_binding,
        max_object_bytes=max_object_bytes,
        max_aggregate_bytes=max_aggregate_bytes,
    )


def _write_cold_profile_budget_receipt(path: Path, receipt: Mapping[str, Any]) -> Path:
    core = dict(receipt)
    core.pop("receipt_sha256", None)
    objects = sorted(
        [dict(row) for row in core["objects"]],
        key=lambda row: str(row["object_identity_sha256"]),
    )
    core.update(
        {
            "objects": objects,
            "reserved_bytes": sum(int(row["bytes"]) for row in objects),
            "verified_bytes": sum(
                int(row["bytes"]) for row in objects if row["status"] == "VERIFIED"
            ),
            "updated_at_utc": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
        }
    )
    payload = {**core, "receipt_sha256": canonical_sha256(core)}
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(pretty_json(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        parent_fd = os.open(
            path.parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _acquire_cold_profile_budget_lock(budget_scope_root: Path) -> tuple[Any, Path]:
    budget_scope_root.mkdir(parents=True, exist_ok=True)
    receipt_path, lock_path = _cold_profile_budget_paths(budget_scope_root)
    descriptor = os.open(
        lock_path,
        os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
        0o600,
    )
    handle = os.fdopen(descriptor, "a+b")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    return handle, receipt_path


def _reserve_cold_profile_budget_row(
    receipt: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any],
    max_aggregate_bytes: int,
) -> tuple[dict[str, Any], str]:
    objects = [dict(row) for row in receipt.get("objects", [])]
    identity = str(candidate["object_identity_sha256"])
    existing = next(
        (row for row in objects if row["object_identity_sha256"] == identity),
        None,
    )
    if existing is not None:
        candidate_identity = dict(candidate)
        candidate_identity.pop("manifest_sha256", None)
        candidate_identity.pop("status", None)
        existing_identity = dict(existing)
        existing_identity.pop("manifest_sha256", None)
        existing_identity.pop("status", None)
        if existing_identity != candidate_identity:
            raise ArtifactLifecycleError("cold profile reservation identity differs")
        existing["manifest_sha256"] = candidate["manifest_sha256"]
        prior_status = str(existing["status"])
        existing["status"] = "RESERVED"
        return {**dict(receipt), "objects": objects}, prior_status
    if any(row["destination"] == candidate["destination"] for row in objects):
        raise ArtifactLifecycleError(
            "cold profile destination is reserved for a different exact object"
        )
    if len(objects) >= _COLD_PROFILE_BUDGET_MAX_OBJECTS:
        raise ArtifactLifecycleError("cold profile budget receipt object limit reached")
    if int(receipt.get("reserved_bytes", 0)) + int(candidate["bytes"]) > int(
        max_aggregate_bytes
    ):
        raise ArtifactLifecycleError("cold profile aggregate budget is exhausted")
    objects.append(dict(candidate))
    return {**dict(receipt), "objects": objects}, "NEW"


def _mark_cold_profile_budget_verified(
    receipt: Mapping[str, Any], *, object_identity_sha256: str
) -> dict[str, Any]:
    objects = [dict(row) for row in receipt.get("objects", [])]
    for row in objects:
        if row["object_identity_sha256"] == object_identity_sha256:
            row["status"] = "VERIFIED"
            return {**dict(receipt), "objects": objects}
    raise ArtifactLifecycleError(
        "cold profile reservation disappeared before verification"
    )


def build_archive_plan(
    root: Path,
    disposition_manifest: Mapping[str, Any],
    *,
    archive_root: Path,
    budget_root: Path | None = None,
) -> dict[str, Any]:
    """Select only exact durable rows for asynchronous cold-storage admission."""

    validated = validate_artifact_disposition_manifest_v1(disposition_manifest)
    if not legacy_reconstruction_is_accepted_v1(validated):
        raise ArtifactLifecycleError(
            "legacy reconstructed durable material requires an accepted evidence review"
        )
    runtime_root = root.expanduser().resolve()
    manifest_root = validated.get("runtime_root")
    if (
        not isinstance(manifest_root, str)
        or Path(manifest_root).expanduser().resolve() != runtime_root
    ):
        raise ArtifactLifecycleError(
            "disposition authority is bound to a different runtime root"
        )
    archive_base = _validate_cold_archive_separation(
        runtime_root=runtime_root,
        requested_archive_root=archive_root,
    )
    budget_base = _archive_budget_scope_root(
        runtime_root=runtime_root,
        archive_root=archive_base,
        requested_budget_root=budget_root,
    )
    availability_policy = load_data2_storage_availability_policy_v1()
    cold_profile_policy = availability_policy["cold_diagnostic_profile_assets"]
    max_profile_object_bytes = int(cold_profile_policy["max_object_bytes"])
    max_profile_aggregate_bytes = int(cold_profile_policy["max_aggregate_bytes"])
    rows: list[dict[str, Any]] = []
    for value in validated["artifacts"]:
        if not isinstance(value, Mapping):
            continue
        retention_class = str(value.get("retention_class") or "RX")
        if retention_class not in {"R3", "R4", "R5"}:
            continue
        if value.get("disposition") != "RETAIN_DURABLE":
            continue
        if value.get("exact_object_identity") is not True:
            continue
        if int(value.get("bytes", 0)) > max_profile_object_bytes:
            raise ArtifactLifecycleError(
                "explicit durable archive object exceeds conservative canonical "
                f"per-object cold cap: {value.get('path')}"
            )
        relative = str(value.get("path") or "").replace("\\", "/").strip("/")
        if not relative or Path(relative).is_absolute():
            raise RuntimeError("disposition manifest contains an invalid durable path")
        source = runtime_root / relative
        try:
            source.resolve(strict=False).relative_to(runtime_root)
        except ValueError as exc:
            raise RuntimeError(
                f"durable path escapes runtime root: {relative}"
            ) from exc
        rows.append(
            {
                "path": relative,
                "bytes": int(value.get("bytes", 0)),
                "retention_class": retention_class,
                "disposition": "RETAIN_DURABLE",
                "artifact_family": str(value.get("artifact_family") or ""),
                "artifact_type": str(value.get("artifact_type") or "file"),
                "release_eligible": value.get("release_eligible") is True,
                "destination": str(
                    _archive_destination(
                        runtime_root,
                        archive_base,
                        relative,
                        str(validated.get("run_id") or runtime_root.name),
                    )
                ),
                "versioned_destination": str(
                    _archive_version_destination(
                        runtime_root,
                        archive_base,
                        relative,
                        str(validated.get("run_id") or runtime_root.name),
                        value,
                    )
                ),
                **(
                    {"sha256": value["sha256"]}
                    if isinstance(value.get("sha256"), str)
                    else {}
                ),
                **(
                    {"tree_sha256": value["tree_sha256"]}
                    if isinstance(value.get("tree_sha256"), str)
                    else {}
                ),
            }
        )
    rows.sort(key=lambda row: str(row["path"]))
    selected_profile_bytes = sum(int(row["bytes"]) for row in rows)
    if selected_profile_bytes > max_profile_aggregate_bytes:
        raise ArtifactLifecycleError(
            "current manifest diagnostic/profile archive selection exceeds canonical "
            "aggregate cold cap"
        )
    return {
        "schema_version": "eiel_runtime_storage_archive_plan_v1",
        "runtime_root": str(runtime_root),
        "archive_root": str(archive_base),
        "run_id": str(validated.get("run_id") or runtime_root.name),
        "disposition_manifest_sha256": validated["manifest_sha256"],
        "selected": rows,
        "selected_count": len(rows),
        "selected_bytes": sum(int(row["bytes"]) for row in rows),
        "release_after_archive_bytes": sum(
            int(row["bytes"])
            for row in rows
            if row["retention_class"] == "R3" and row["release_eligible"]
        ),
        "storage_availability_policy": storage_availability_policy_binding_v1(
            availability_policy
        ),
        "cold_diagnostic_profile_guard": {
            "max_object_bytes": max_profile_object_bytes,
            "max_aggregate_bytes": max_profile_aggregate_bytes,
            "selected_bytes": selected_profile_bytes,
            "budget_scope": _COLD_PROFILE_BUDGET_SCOPE,
            "aggregate_scope": "cross_manifest_actuator_budget_receipt_v1",
            "cross_manifest_aggregate_status": (
                "ENFORCED_AT_APPLY_BY_SHARED_BUDGET_RECEIPT_V1"
            ),
            "budget_receipt": str(_cold_profile_budget_paths(budget_base)[0]),
        },
        "authority_status": "lifecycle_disposition_manifest",
    }


def _copy_regular_member_exact(source: str, destination: str) -> str:
    """Copy one regular member without following links or accepting mutation."""

    observed = os.lstat(source)
    if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
        raise RuntimeError(f"archive source member is linked or non-regular: {source}")
    source_descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(source_descriptor)
        if _stat_identity(opened) != _stat_identity(observed) or opened.st_nlink != 1:
            raise RuntimeError(f"archive source member changed before copy: {source}")
        destination_descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            stat.S_IMODE(observed.st_mode) or 0o600,
        )
        try:
            with (
                os.fdopen(os.dup(source_descriptor), "rb") as source_handle,
                os.fdopen(destination_descriptor, "wb") as destination_handle,
            ):
                destination_descriptor = -1
                shutil.copyfileobj(
                    source_handle, destination_handle, length=1024 * 1024
                )
                destination_handle.flush()
                os.fsync(destination_handle.fileno())
        finally:
            if destination_descriptor >= 0:
                os.close(destination_descriptor)
        after_open = os.fstat(source_descriptor)
        after_path = os.lstat(source)
        if (
            _stat_identity(after_open) != _stat_identity(observed)
            or _stat_identity(after_path) != _stat_identity(observed)
            or after_open.st_nlink != 1
            or after_path.st_nlink != 1
        ):
            raise RuntimeError(f"archive source member changed while copying: {source}")
    finally:
        os.close(source_descriptor)
    return destination


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_sealed_tree_directories(path: Path) -> None:
    directories = [path]
    for member in path.rglob("*"):
        observed = os.lstat(member)
        if stat.S_ISLNK(observed.st_mode):
            raise RuntimeError(f"archive staging tree contains a symlink: {member}")
        if stat.S_ISDIR(observed.st_mode):
            directories.append(member)
    for directory in sorted(
        directories,
        key=lambda candidate: len(candidate.parts),
        reverse=True,
    ):
        _fsync_directory(directory)


_ARCHIVE_PUBLICATION_MARKER_SCHEMA = "eiel_archive_publication_marker_v1"


def _archive_publication_marker_path(destination: Path) -> Path:
    """Return the sidecar used to recover a FUSE-compatible tree publish."""

    return destination.parent / f".{destination.name}.eiel-publish-v1"


def _archive_publication_marker_payload(
    *, staged: Path, destination: Path
) -> dict[str, Any]:
    core = {
        "schema_version": _ARCHIVE_PUBLICATION_MARKER_SCHEMA,
        "staged": str(staged),
        "destination": str(destination),
    }
    return {**core, "marker_sha256": canonical_sha256(core)}


def _read_archive_publication_marker(
    *, staged: Path | None, destination: Path
) -> dict[str, Any] | None:
    """Read an exact publication marker without treating a pathname as authority."""

    marker = _archive_publication_marker_path(destination)
    if not marker.exists() and not marker.is_symlink():
        return None
    if marker.is_symlink() or not marker.is_file():
        raise ArtifactLifecycleError(
            f"cold archive publication marker is not a regular file: {marker}"
        )
    try:
        raw = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactLifecycleError(
            f"cold archive publication marker is unreadable: {marker}"
        ) from exc
    if not isinstance(raw, Mapping):
        raise ArtifactLifecycleError(
            f"cold archive publication marker is malformed: {marker}"
        )
    value = dict(raw)
    supplied = value.pop("marker_sha256", None)
    if (
        set(value) != {"schema_version", "staged", "destination"}
        or value.get("schema_version") != _ARCHIVE_PUBLICATION_MARKER_SCHEMA
        or supplied != canonical_sha256(value)
        or value.get("destination") != str(destination)
    ):
        raise ArtifactLifecycleError(
            f"cold archive publication marker authority differs: {marker}"
        )
    if staged is not None and value.get("staged") != str(staged):
        raise ArtifactLifecycleError(
            f"cold archive publication staging authority differs: {marker}"
        )
    return {**value, "marker_sha256": supplied}


def _write_archive_publication_marker(staged: Path, destination: Path) -> Path:
    marker = _archive_publication_marker_path(destination)
    payload = _archive_publication_marker_payload(
        staged=staged,
        destination=destination,
    )
    descriptor = os.open(
        marker,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    _fsync_directory(marker.parent)
    return marker


def _remove_archive_publication_marker(staged: Path, destination: Path) -> None:
    marker = _archive_publication_marker_path(destination)
    _read_archive_publication_marker(staged=staged, destination=destination)
    marker.unlink()
    _fsync_directory(marker.parent)


def _tree_file_rows(path: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": str(member["path"]),
            "bytes": int(member["bytes"]),
            "sha256": str(member["sha256"]),
        }
        for member in _sealed_tree_member_inventory(path)
        if member.get("type") == "file"
    ]


def _recover_incomplete_archive_tree_publication(
    *, staged: Path, destination: Path, row: Mapping[str, Any]
) -> None:
    """Discard only a marker-bound partial tree after proving its exact contents."""

    _read_archive_publication_marker(staged=staged, destination=destination)
    if not destination.is_dir() or destination.is_symlink():
        raise RuntimeError(
            f"cold archive publication target is not a recoverable directory: {destination}"
        )
    if not staged.is_dir() or staged.is_symlink():
        raise RuntimeError(
            f"cold archive publication staging is not a recoverable directory: {staged}"
        )
    destination_rows = _tree_file_rows(destination)
    staged_rows = _tree_file_rows(staged)
    combined = destination_rows + staged_rows
    paths = [str(value["path"]) for value in combined]
    if len(paths) != len(set(paths)):
        raise RuntimeError("cold archive publication recovery has duplicate members")
    combined.sort(key=lambda value: str(value["path"]))
    if (
        sum(int(value["bytes"]) for value in combined)
        != int(row.get("bytes", -1))
        or canonical_sha256(combined) != row.get("tree_sha256")
    ):
        raise RuntimeError(
            "cold archive publication recovery does not match sealed-tree authority"
        )
    shutil.rmtree(destination)
    _remove_archive_publication_marker(staged, destination)


def _publish_archive_tree_without_rename(staged: Path, destination: Path) -> None:
    """Publish a sealed tree on filesystems lacking renameat2(RENAME_NOREPLACE)."""

    observed_root = os.lstat(staged)
    if not stat.S_ISDIR(observed_root.st_mode) or stat.S_ISLNK(observed_root.st_mode):
        raise RuntimeError(f"cold archive staged tree is not a directory: {staged}")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(str(destination))
    os.mkdir(destination, mode=stat.S_IMODE(observed_root.st_mode) or 0o700)
    try:
        _write_archive_publication_marker(staged, destination)
        members = sorted(
            staged.rglob("*"),
            key=lambda member: (len(member.relative_to(staged).parts), str(member)),
        )
        for member in members:
            relative = member.relative_to(staged)
            target = destination / relative
            observed = os.lstat(member)
            if stat.S_ISLNK(observed.st_mode):
                raise RuntimeError(f"cold archive staged tree contains a symlink: {member}")
            if stat.S_ISDIR(observed.st_mode):
                os.mkdir(target, mode=stat.S_IMODE(observed.st_mode) or 0o700)
                continue
            if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
                raise RuntimeError(
                    f"cold archive staged tree contains an unsupported member: {member}"
                )
            os.link(member, target, follow_symlinks=False)
            linked = os.lstat(target)
            after = os.lstat(member)
            if (
                _stat_identity(linked)[:4] != _stat_identity(observed)[:4]
                or _stat_identity(after) != _stat_identity(observed)
            ):
                raise RuntimeError(
                    f"cold archive staged tree member changed while publishing: {member}"
                )
            member.unlink()
        for directory in sorted(
            [member for member in staged.rglob("*") if member.is_dir()] + [staged],
            key=lambda member: len(member.parts),
            reverse=True,
        ):
            directory.rmdir()
        _remove_archive_publication_marker(staged, destination)
    except BaseException:
        # Leave the marker and staged/destination objects for exact recovery.
        raise


def _archive_staging_path(destination: Path, object_identity_sha256: str) -> Path:
    if len(object_identity_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in object_identity_sha256
    ):
        raise ArtifactLifecycleError("cold archive object identity is invalid")
    return destination.parent / (
        f".{destination.name}.eiel-stage-{object_identity_sha256[:32]}"
    )


def _copy_one(
    source: Path,
    destination: Path,
    *,
    row: Mapping[str, Any],
    object_identity_sha256: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = _archive_staging_path(destination, object_identity_sha256)
    artifact_type = str(row.get("artifact_type") or "file")
    if destination.exists() or destination.is_symlink():
        if (
            artifact_type == "sealed_tree"
            and _read_archive_publication_marker(
                staged=staged,
                destination=destination,
            )
            is not None
        ):
            _recover_incomplete_archive_tree_publication(
                staged=staged,
                destination=destination,
                row=row,
            )
        else:
            raise FileExistsError(str(destination))
    legacy_partials = sorted(destination.parent.glob(f".{destination.name}.partial-*"))
    if legacy_partials:
        raise ArtifactLifecycleError(
            "unaccounted legacy cold archive partial blocks exact recovery: "
            + ", ".join(str(path) for path in legacy_partials[:4])
        )
    if staged.exists() or staged.is_symlink():
        if staged.is_symlink():
            raise ArtifactLifecycleError(
                f"cold archive staging path is a symlink: {staged}"
            )
        _verify_archived_row(staged, row)
    elif artifact_type == "sealed_tree":
        if source.is_symlink() or not source.is_dir():
            raise RuntimeError(f"sealed tree source is not a directory: {source}")
        _verify_archived_row(source, row)
        shutil.copytree(
            source,
            staged,
            symlinks=True,
            copy_function=_copy_regular_member_exact,
        )
        # Verify both sides again before publishing.  A link injected while
        # walking is copied as a link and therefore fails exact-tree
        # verification instead of being dereferenced into cold storage.
        _verify_archived_row(source, row)
        _verify_archived_row(staged, row)
    else:
        observed = os.lstat(source)
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
            or source.is_symlink()
        ):
            raise RuntimeError(f"archive source is not a regular file: {source}")
        _verify_archived_row(source, row)
        _copy_regular_member_exact(str(source), str(staged))
        _verify_archived_row(source, row)
        _verify_archived_row(staged, row)
    if artifact_type == "sealed_tree":
        _fsync_sealed_tree_directories(staged)
    else:
        with staged.open("rb") as handle:
            os.fsync(handle.fileno())
    _fsync_directory(destination.parent)
    _publish_archive_no_replace(staged, destination)
    _fsync_directory(destination.parent)


def _rename_noreplace(
    source: str | Path,
    destination: str | Path,
    *,
    source_dir_fd: int = -100,
    destination_dir_fd: int = -100,
) -> None:
    """Atomically rename one entry without replacing a raced-in target."""

    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic no-replace rename is unavailable on this platform")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        source_dir_fd,
        os.fsencode(source),
        destination_dir_fd,
        os.fsencode(destination),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, os.strerror(error_number), destination)
    raise OSError(error_number, os.strerror(error_number), destination)


def _publish_archive_no_replace(staged: Path, destination: Path) -> None:
    """Atomically publish a cold object without replacing a raced-in object."""

    try:
        _rename_noreplace(staged, destination)
        return
    except OSError as exc:
        if exc.errno not in {errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP}:
            raise
    observed = os.lstat(staged)
    if stat.S_ISLNK(observed.st_mode):
        raise RuntimeError(f"cold archive staging path is a symlink: {staged}")
    if stat.S_ISREG(observed.st_mode):
        if observed.st_nlink != 1:
            raise RuntimeError(f"cold archive staging file is linked: {staged}")
        os.link(staged, destination, follow_symlinks=False)
        linked = os.lstat(destination)
        if _stat_identity(linked)[:4] != _stat_identity(observed)[:4]:
            raise RuntimeError(
                f"cold archive no-replace link identity differs: {destination}"
            )
        staged.unlink()
        if os.lstat(destination).st_nlink != 1:
            raise RuntimeError(
                f"cold archive no-replace link remains multiply linked: {destination}"
            )
        return
    if stat.S_ISDIR(observed.st_mode):
        _publish_archive_tree_without_rename(staged, destination)
        return
    raise RuntimeError(f"cold archive staging type is unsupported: {staged}")


def _verify_archived_row(path: Path, row: Mapping[str, Any]) -> None:
    artifact_type = str(row.get("artifact_type") or "file")
    if artifact_type == "sealed_tree":
        if not path.is_dir() or path.is_symlink():
            raise RuntimeError(f"archived sealed tree is missing: {path}")
        if _tree_digest(path) != row.get("tree_sha256"):
            raise RuntimeError(f"archived sealed tree digest differs: {path}")
        observed_bytes = sum(
            int(member.stat().st_size)
            for member in path.rglob("*")
            if member.is_file() and not member.is_symlink()
        )
        if observed_bytes != int(row.get("bytes", -1)):
            raise RuntimeError(f"archived sealed tree byte count differs: {path}")
        return
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"archived file is missing: {path}")
    if _file_digest(path) != row.get("sha256"):
        raise RuntimeError(f"archived file digest differs: {path}")
    if int(path.stat().st_size) != int(row.get("bytes", -1)):
        raise RuntimeError(f"archived file byte count differs: {path}")


def _verify_archive_destination_exact(
    path: Path,
    row: Mapping[str, Any],
    *,
    expected_identity: tuple[int, int, int, int] | None = None,
) -> tuple[int, int, int, int]:
    """Verify cold content and stable pathname identity without following links."""

    before = os.lstat(path)
    artifact_type = str(row.get("artifact_type") or "file")
    if stat.S_ISLNK(before.st_mode):
        raise RuntimeError(f"archive destination is a symlink: {path}")
    if artifact_type == "sealed_tree":
        if not stat.S_ISDIR(before.st_mode):
            raise RuntimeError(f"archive destination tree type differs: {path}")
    elif not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise RuntimeError(f"archive destination file type differs: {path}")
    identity = _stat_identity(before)
    if expected_identity is not None and identity != expected_identity:
        raise RuntimeError(f"archive destination pathname identity changed: {path}")
    _verify_archived_row(path, row)
    after = os.lstat(path)
    if _stat_identity(after) != identity:
        raise RuntimeError(f"archive destination changed while verifying: {path}")
    return identity


def _capture_archive_source_identity(
    source: Path, row: Mapping[str, Any]
) -> tuple[int, int, int, int]:
    observed = os.lstat(source)
    artifact_type = str(row.get("artifact_type") or "file")
    if artifact_type == "sealed_tree":
        if not stat.S_ISDIR(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
            raise RuntimeError(f"sealed tree source is not a directory: {source}")
    elif not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
        raise RuntimeError(f"archive source is linked or non-regular: {source}")
    return _stat_identity(observed)


_ARCHIVE_RELEASE_JOURNAL_SCHEMA = "eiel_runtime_archive_release_journal_v1"
_ARCHIVE_RELEASE_JOURNAL_TOKEN = object()


def _archive_release_selected_rows(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        [
            dict(row)
            for row in plan.get("selected", [])
            if isinstance(row, Mapping)
            and row.get("retention_class") == "R3"
            and row.get("release_eligible") is True
        ],
        key=lambda row: str(row.get("path") or ""),
    )


def _prune_release_journal_plan(
    *,
    runtime_root: Path,
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Project an exact prune selection into the shared release-journal shape."""

    return {
        "schema_version": "eiel_internal_prune_release_plan_v1",
        "runtime_root": str(runtime_root),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "source_prune_plan_sha256": canonical_sha256(dict(plan)),
        "archive_root": "DIRECT_RELEASE_NO_ARCHIVE",
        "selected": [
            {
                **dict(row),
                "retention_class": "R3",
                "release_eligible": True,
            }
            for row in plan.get("selected", [])
            if isinstance(row, Mapping)
        ],
    }


def _archive_release_progress_rows(
    journal: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    raw_progress = journal.get("progress", [])
    if not isinstance(raw_progress, list):
        raise ArtifactLifecycleError("archive release journal progress is malformed")
    progress: dict[str, dict[str, Any]] = {}
    for raw in raw_progress:
        if not isinstance(raw, Mapping):
            raise ArtifactLifecycleError(
                "archive release journal progress is malformed"
            )
        row = dict(raw)
        path = str(row.get("path") or "")
        if not path or path in progress:
            raise ArtifactLifecycleError(
                "archive release journal progress path differs"
            )
        progress[path] = row
    return progress


def _validate_archive_release_members(
    authority: Mapping[str, Any], members: Any
) -> list[dict[str, Any]]:
    if not isinstance(members, list):
        raise ArtifactLifecycleError(
            "sealed-tree archive release progress lacks member authority"
        )
    normalized: list[dict[str, Any]] = []
    paths: set[str] = set()
    for raw in members:
        if not isinstance(raw, Mapping):
            raise ArtifactLifecycleError(
                "sealed-tree archive release member authority is malformed"
            )
        member = dict(raw)
        member_type = member.get("type")
        expected_fields = (
            {"path", "type", "device", "inode"}
            if member_type == "directory"
            else {"path", "type", "device", "inode", "bytes", "sha256"}
            if member_type == "file"
            else set()
        )
        relative = str(member.get("path") or "").replace("\\", "/")
        if (
            not expected_fields
            or set(member) != expected_fields
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or relative in paths
            or type(member.get("device")) is not int
            or type(member.get("inode")) is not int
        ):
            raise ArtifactLifecycleError(
                "sealed-tree archive release member authority is malformed"
            )
        if member_type == "file" and (
            type(member.get("bytes")) is not int
            or int(member["bytes"]) < 0
            or not isinstance(member.get("sha256"), str)
            or len(str(member["sha256"])) != 64
        ):
            raise ArtifactLifecycleError(
                "sealed-tree archive release file authority is malformed"
            )
        paths.add(relative)
        normalized.append({**member, "path": relative})
    try:
        _validate_sealed_tree_member_authority(normalized, authority)
    except RuntimeError as exc:
        raise ArtifactLifecycleError(
            "sealed-tree archive release member authority differs from selection"
        ) from exc
    return normalized


def _archive_release_transaction_paths(
    *,
    runtime_root: Path,
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Path]:
    identity = canonical_sha256(
        {
            "schema_version": _ARCHIVE_RELEASE_JOURNAL_SCHEMA,
            "runtime_root": str(runtime_root),
            "manifest_sha256": manifest.get("manifest_sha256"),
            "archive_plan_sha256": canonical_sha256(dict(plan)),
            "selected": _archive_release_selected_rows(plan),
        }
    )
    transaction = runtime_root / ".eiel-runtime-archive-release-v1" / identity
    return {
        "transaction": transaction,
        "journal": transaction / "journal.json",
        "lock": transaction / "journal.lock",
        "staging": transaction / "staged",
    }


def _write_archive_release_journal(
    path: Path, journal: Mapping[str, Any]
) -> dict[str, Any]:
    core = dict(journal)
    core.pop("journal_sha256", None)
    payload = {**core, "journal_sha256": canonical_sha256(core)}
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(pretty_json(payload))
            handle.flush()
            os.fsync(handle.fileno())
        try:
            observed = os.lstat(path)
        except FileNotFoundError:
            observed = None
        if observed is not None and (
            not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1
        ):
            raise ArtifactLifecycleError(
                "archive release journal target is not a single-link regular file"
            )
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def _validate_archive_release_journal(
    value: Mapping[str, Any],
    *,
    runtime_root: Path,
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
    staging_root: Path,
) -> dict[str, Any]:
    journal = dict(value)
    supplied = journal.pop("journal_sha256", None)
    if supplied != canonical_sha256(journal):
        raise ArtifactLifecycleError("archive release journal SHA-256 differs")
    selected_rows = _archive_release_selected_rows(plan)
    expected = {
        "schema_version": _ARCHIVE_RELEASE_JOURNAL_SCHEMA,
        "runtime_root": str(runtime_root),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "archive_plan_sha256": canonical_sha256(dict(plan)),
        "selected": selected_rows,
        "staging_root": str(staging_root),
    }
    for name, expected_value in expected.items():
        if journal.get(name) != expected_value:
            raise ArtifactLifecycleError(
                f"archive release journal {name} differs from archive authority"
            )
    if set(journal) != {
        *expected,
        "status",
        "created_at_utc",
        "updated_at_utc",
        "progress",
    } or journal.get("status") not in {"APPLYING", "COMPLETE"}:
        raise ArtifactLifecycleError("archive release journal fields differ")
    authority = {str(row.get("path")): row for row in selected_rows}
    for path, row in _archive_release_progress_rows(journal).items():
        selected = authority.get(path)
        expected_staged = staging_root / (
            canonical_sha256({"path": path})[:24] + "-" + Path(path).name
        )
        identity = row.get("identity")
        if (
            selected is None
            or row.get("bytes") != selected.get("bytes")
            or row.get("state") not in {"PREPARED", "STAGED", "RELEASED"}
            or row.get("staging_path") != str(expected_staged)
            or not isinstance(identity, list)
            or len(identity) != 4
            or any(type(item) is not int for item in identity)
        ):
            raise ArtifactLifecycleError(
                "archive release journal progress differs from selected authority"
            )
        if selected.get("artifact_type") == "sealed_tree":
            _validate_archive_release_members(selected, row.get("members"))
        elif "members" in row:
            raise ArtifactLifecycleError(
                "file archive release progress cannot carry tree members"
            )
    if journal.get("status") == "COMPLETE" and set(
        _archive_release_progress_rows(journal)
    ) != set(authority):
        raise ArtifactLifecycleError(
            "complete archive release journal lacks exact selected progress"
        )
    if journal.get("status") == "COMPLETE" and any(
        row.get("state") != "RELEASED"
        for row in _archive_release_progress_rows(journal).values()
    ):
        raise ArtifactLifecycleError(
            "complete archive release journal contains pending progress"
        )
    return {**journal, "journal_sha256": supplied}


@contextmanager
def _direct_archive_release_journal(
    *,
    runtime_root: Path,
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> Any:
    paths = _archive_release_transaction_paths(
        runtime_root=runtime_root,
        plan=plan,
        manifest=manifest,
    )
    for directory in (
        paths["transaction"].parent,
        paths["transaction"],
        paths["staging"],
    ):
        try:
            observed = os.lstat(directory)
        except FileNotFoundError:
            directory.mkdir(mode=0o700)
            _fsync_directory(directory.parent)
            observed = os.lstat(directory)
        if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
            raise ArtifactLifecycleError(
                "archive release control path is not an exact directory"
            )
        try:
            directory.resolve(strict=True).relative_to(runtime_root)
        except ValueError as exc:
            raise ArtifactLifecycleError(
                "archive release control path escapes the runtime root"
            ) from exc
    lock_descriptor = os.open(
        paths["lock"],
        os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
        0o600,
    )
    lock_handle = os.fdopen(lock_descriptor, "a+b")
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
    try:
        try:
            observed_journal = os.lstat(paths["journal"])
        except FileNotFoundError:
            observed_journal = None
        if observed_journal is None:
            if any(paths["staging"].iterdir()):
                raise ArtifactLifecycleError(
                    "archive release staging exists without an authenticated journal"
                )
            now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            journal = _write_archive_release_journal(
                paths["journal"],
                {
                    "schema_version": _ARCHIVE_RELEASE_JOURNAL_SCHEMA,
                    "status": "APPLYING",
                    "runtime_root": str(runtime_root),
                    "manifest_sha256": manifest.get("manifest_sha256"),
                    "archive_plan_sha256": canonical_sha256(dict(plan)),
                    "selected": _archive_release_selected_rows(plan),
                    "staging_root": str(paths["staging"]),
                    "created_at_utc": now,
                    "updated_at_utc": now,
                    "progress": [],
                },
            )
        else:
            if (
                not stat.S_ISREG(observed_journal.st_mode)
                or observed_journal.st_nlink != 1
            ):
                raise ArtifactLifecycleError(
                    "archive release journal is not a single-link regular file"
                )
            descriptor = os.open(paths["journal"], os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
                raw_journal = json.load(handle)
            if not isinstance(raw_journal, Mapping):
                raise ArtifactLifecycleError("archive release journal is malformed")
            journal = _validate_archive_release_journal(
                raw_journal,
                runtime_root=runtime_root,
                plan=plan,
                manifest=manifest,
                staging_root=paths["staging"],
            )
        state: dict[str, Any] = {"journal": journal}

        def record(event: Mapping[str, Any]) -> None:
            current = state["journal"]
            progress = _archive_release_progress_rows(current)
            path = str(event.get("path") or "")
            authority = {
                str(row.get("path")): row for row in current.get("selected", [])
            }.get(path)
            if authority is None:
                raise ArtifactLifecycleError(
                    "archive release progress exceeds selected authority"
                )
            event_name = str(event.get("event") or "")
            previous = progress.get(path)
            if event_name in {"PREPARED", "STAGED", "RECOVERED_STAGED"}:
                identity = event.get("identity")
                if (
                    not isinstance(identity, list)
                    or len(identity) != 4
                    or any(type(item) is not int for item in identity)
                    or not isinstance(event.get("staging_path"), str)
                ):
                    raise ArtifactLifecycleError(
                        "archive release staging event is malformed"
                    )
                target_state = "PREPARED" if event_name == "PREPARED" else "STAGED"
                prior_identity = (
                    previous.get("identity") if previous is not None else None
                )
                identity_differs = (
                    prior_identity[:2] != list(identity)[:2]
                    if authority.get("artifact_type") == "sealed_tree"
                    and isinstance(prior_identity, list)
                    else prior_identity != list(identity)
                )
                if previous is not None and (
                    previous.get("state") == "RELEASED"
                    or identity_differs
                    or previous.get("staging_path") != event.get("staging_path")
                    or (
                        previous.get("state") == "STAGED" and target_state == "PREPARED"
                    )
                ):
                    raise ArtifactLifecycleError(
                        "archive release progress cannot regress or change identity"
                    )
                updated_row: dict[str, Any] = {
                    "path": path,
                    "bytes": int(authority.get("bytes", 0)),
                    "state": target_state,
                    "identity": (
                        list(prior_identity)
                        if authority.get("artifact_type") == "sealed_tree"
                        and isinstance(prior_identity, list)
                        else list(identity)
                    ),
                    "staging_path": str(event["staging_path"]),
                    "updated_at_utc": datetime.now(timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z"),
                }
                if authority.get("artifact_type") == "sealed_tree":
                    updated_row["members"] = _validate_archive_release_members(
                        authority, event.get("members")
                    )
                progress[path] = updated_row
            elif event_name == "RELEASED":
                if previous is None or previous.get("state") not in {
                    "STAGED",
                    "RELEASED",
                }:
                    raise ArtifactLifecycleError(
                        "archive release requires durable staging before release"
                    )
                if int(event.get("bytes", -1)) != int(authority.get("bytes", -2)):
                    raise ArtifactLifecycleError(
                        "archive release byte identity differs"
                    )
                progress[path] = {
                    **previous,
                    "state": "RELEASED",
                    "updated_at_utc": datetime.now(timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z"),
                }
            else:
                raise ArtifactLifecycleError("unknown archive release progress event")
            core = {
                name: value
                for name, value in current.items()
                if name not in {"journal_sha256", "progress", "updated_at_utc"}
            }
            core.update(
                {
                    "status": "APPLYING",
                    "updated_at_utc": datetime.now(timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z"),
                    "progress": [progress[name] for name in sorted(progress)],
                }
            )
            state["journal"] = _write_archive_release_journal(paths["journal"], core)

        def complete() -> None:
            current = state["journal"]
            progress = _archive_release_progress_rows(current)
            selected = {str(row.get("path")) for row in current.get("selected", [])}
            if set(progress) != selected or any(
                row.get("state") != "RELEASED" for row in progress.values()
            ):
                raise ArtifactLifecycleError(
                    "archive release journal cannot complete with pending rows"
                )
            if current.get("status") == "COMPLETE":
                return
            core = {
                name: value
                for name, value in current.items()
                if name not in {"journal_sha256", "status", "updated_at_utc"}
            }
            core.update(
                {
                    "status": "COMPLETE",
                    "updated_at_utc": datetime.now(timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z"),
                }
            )
            state["journal"] = _write_archive_release_journal(paths["journal"], core)

        yield {
            "token": _ARCHIVE_RELEASE_JOURNAL_TOKEN,
            "paths": paths,
            "progress": lambda: _archive_release_progress_rows(state["journal"]),
            "record": record,
            "complete": complete,
            "status": lambda: state["journal"]["status"],
        }
    finally:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()


def _sha256_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    with os.fdopen(os.dup(descriptor), "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _release_archived_source_exact(
    *,
    runtime_root: Path,
    source: Path,
    archive_destination: Path,
    row: Mapping[str, Any],
    expected_identity: tuple[int, int, int, int],
    expected_archive_identity: tuple[int, int, int, int],
    current_disposition_manifest_path: Path | None = None,
    expected_disposition_manifest: Mapping[str, Any] | None = None,
    current_authority_validator: Callable[[], None] | None = None,
    release_staging_root: Path | None = None,
    prior_release_progress: Mapping[str, Any] | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    _transaction_fence_held: bool = False,
    _archive_preverified: bool = False,
) -> None:
    """Release only the inode and bytes that were selected before archival.

    The canonical reservation lock is acquired only after slow-tier work is
    complete, so foreground admission never waits for cold I/O.  Holding it
    across the final exact verification and descriptor-relative deletion
    closes the reservation and pathname replacement races.
    """

    # The caller holds the canonical cold-budget ledger lock.  Perform the
    # final slow-tier verification before acquiring the hot reservation fence,
    # so foreground admission never waits for cold I/O.
    if not _archive_preverified:
        _verify_archive_destination_exact(
            archive_destination,
            row,
            expected_identity=expected_archive_identity,
        )
    elif _stat_identity(os.lstat(archive_destination)) != expected_archive_identity:
        raise RuntimeError(
            "archive destination identity changed after exact verification"
        )
    relative = source.relative_to(runtime_root)
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise RuntimeError("archive release path is not a bounded runtime object")
    if type(_transaction_fence_held) is not bool:
        raise ArtifactLifecycleError(
            "run artifact transaction fence ownership flag must be boolean"
        )
    staging_root = (
        (
            release_staging_root
            if release_staging_root is not None
            else runtime_root
            / ".eiel-runtime-archive-release-v1"
            / str(row.get("tree_sha256") or row.get("sha256") or "unknown")
        )
        .expanduser()
        .resolve(strict=False)
    )
    try:
        staging_root.relative_to(runtime_root)
    except ValueError as exc:
        raise ArtifactLifecycleError(
            "archived-source release staging must be inside the runtime root"
        ) from exc
    if source == staging_root or source in staging_root.parents:
        raise ArtifactLifecycleError(
            "archived-source release staging overlaps the source object"
        )
    if progress_callback is None:
        raise ArtifactLifecycleError(
            "archived-source release requires a durable progress journal"
        )
    staging_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if staging_root.is_symlink() or not staging_root.is_dir():
        raise ArtifactLifecycleError(
            "archived-source release staging is not an exact directory"
        )
    _fsync_directory(staging_root)
    _fsync_directory(staging_root.parent)
    transaction = (
        nullcontext()
        if _transaction_fence_held
        else run_artifact_transaction_fence_v1(runtime_root)
    )
    release_fence: Any | None = None
    root_fd: int | None = None
    parent_fd: int | None = None
    staging_fd: int | None = None
    with transaction:
        try:
            if current_authority_validator is not None:
                current_authority_validator()
            if current_disposition_manifest_path is not None:
                if expected_disposition_manifest is None:
                    raise ArtifactLifecycleError(
                        "current manifest recheck lacks expected disposition authority"
                    )
                _revalidate_current_disposition_manifest(
                    current_disposition_manifest_path,
                    expected_manifest=expected_disposition_manifest,
                    runtime_root=runtime_root,
                )
            release_fence = _acquire_release_fence(runtime_root)
            if RunStorageGuard.run_root_is_reserved(runtime_root):
                raise ArtifactLifecycleError(
                    "run became reserved before archived source release"
                )
            root_fd = os.open(
                runtime_root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            )
            parent_fd = _open_directory_beneath(root_fd, relative.parts[:-1])
            staging_relative = staging_root.relative_to(runtime_root)
            staging_fd = _open_directory_beneath(root_fd, tuple(staging_relative.parts))
            name = relative.parts[-1]
            staged_name = (
                canonical_sha256({"path": str(relative).replace("\\", "/")})[:24]
                + "-"
                + name
            )
            try:
                observed = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                observed = None
            try:
                staged_observed = os.stat(
                    staged_name, dir_fd=staging_fd, follow_symlinks=False
                )
            except FileNotFoundError:
                staged_observed = None
            if observed is not None and staged_observed is not None:
                raise RuntimeError(
                    "both archived source and exact release staging exist"
                )
            staged_path = staging_root / staged_name
            artifact_type = str(row.get("artifact_type") or "file")
            members: list[dict[str, Any]] | None = None
            if observed is not None:
                if _stat_identity(observed) != expected_identity:
                    raise RuntimeError(
                        "archive source pathname identity changed before exact release"
                    )
                if artifact_type == "sealed_tree":
                    if not stat.S_ISDIR(observed.st_mode):
                        raise RuntimeError(
                            "archive source tree type changed before release"
                        )
                    members = _sealed_tree_member_inventory(source)
                    _validate_sealed_tree_member_authority(members, row)
                else:
                    if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
                        raise RuntimeError(
                            "archive source file type changed before release"
                        )
                    source_fd = os.open(
                        name,
                        os.O_RDONLY | os.O_NOFOLLOW,
                        dir_fd=parent_fd,
                    )
                    try:
                        opened = os.fstat(source_fd)
                        if (
                            _stat_identity(opened) != expected_identity
                            or opened.st_nlink != 1
                            or int(opened.st_size) != int(row.get("bytes", -1))
                            or _sha256_descriptor(source_fd) != row.get("sha256")
                        ):
                            raise RuntimeError(
                                "archive source inode or content changed before exact release"
                            )
                    finally:
                        os.close(source_fd)
                prepared_event: dict[str, Any] = {
                    "event": "PREPARED",
                    "path": str(relative).replace("\\", "/"),
                    "staging_path": str(staged_path),
                    "identity": list(expected_identity),
                }
                if members is not None:
                    prepared_event["members"] = members
                progress_callback(prepared_event)
                _rename_noreplace(
                    name,
                    staged_name,
                    source_dir_fd=parent_fd,
                    destination_dir_fd=staging_fd,
                )
                os.fsync(parent_fd)
                os.fsync(staging_fd)
                staged_observed = os.stat(
                    staged_name, dir_fd=staging_fd, follow_symlinks=False
                )
                if _stat_identity(staged_observed) != expected_identity:
                    raise RuntimeError(
                        "archive source identity changed during exact staging"
                    )
                event: dict[str, Any] = {
                    "event": "STAGED",
                    "path": str(relative).replace("\\", "/"),
                    "staging_path": str(staged_path),
                    "identity": list(expected_identity),
                }
                if members is not None:
                    event["members"] = members
                progress_callback(event)
            elif staged_observed is not None:
                if not isinstance(prior_release_progress, Mapping) or (
                    prior_release_progress.get("state") not in {"PREPARED", "STAGED"}
                ):
                    raise RuntimeError(
                        "archived source staging lacks durable prepared authority"
                    )
                if artifact_type == "sealed_tree":
                    if (
                        not stat.S_ISDIR(staged_observed.st_mode)
                        or stat.S_ISLNK(staged_observed.st_mode)
                        or _stat_identity(staged_observed)[:2] != expected_identity[:2]
                    ):
                        raise RuntimeError(
                            "archived source staging identity differs on retry"
                        )
                    actual_members = _sealed_tree_member_inventory(staged_path)
                    try:
                        _validate_sealed_tree_member_authority(actual_members, row)
                    except RuntimeError:
                        prior_members = (
                            prior_release_progress.get("members")
                            if isinstance(prior_release_progress, Mapping)
                            else None
                        )
                        if not isinstance(prior_members, list) or any(
                            not isinstance(member, Mapping) for member in prior_members
                        ):
                            raise RuntimeError(
                                "partial archived tree staging lacks durable member authority"
                            )
                        members = [dict(member) for member in prior_members]
                        _validate_sealed_tree_member_authority(members, row)
                    else:
                        members = actual_members
                elif _stat_identity(staged_observed) != expected_identity:
                    raise RuntimeError(
                        "archived source staging identity differs on retry"
                    )
                event = {
                    "event": "RECOVERED_STAGED",
                    "path": str(relative).replace("\\", "/"),
                    "staging_path": str(staged_path),
                    "identity": list(expected_identity),
                }
                if members is not None:
                    event["members"] = members
                progress_callback(event)
            elif (
                isinstance(prior_release_progress, Mapping)
                and prior_release_progress.get("state") == "RELEASED"
            ):
                return
            elif (
                isinstance(prior_release_progress, Mapping)
                and prior_release_progress.get("state") == "STAGED"
            ):
                progress_callback(
                    {
                        "event": "RELEASED",
                        "path": str(relative).replace("\\", "/"),
                        "bytes": int(row.get("bytes", 0)),
                    }
                )
                return
            else:
                raise RuntimeError(
                    "archived source disappeared without durable release progress"
                )
            if RunStorageGuard.run_root_is_reserved(runtime_root):
                raise ArtifactLifecycleError(
                    "run became reserved during archived source release"
                )
            assert staged_observed is not None
            if artifact_type == "sealed_tree":
                assert members is not None
                _remove_sealed_tree_remainder(
                    staged_path,
                    expected_members=members,
                )
                os.rmdir(staged_name, dir_fd=staging_fd)
            else:
                staged_fd = os.open(
                    staged_name,
                    os.O_RDONLY | os.O_NOFOLLOW,
                    dir_fd=staging_fd,
                )
                try:
                    opened = os.fstat(staged_fd)
                    if (
                        _stat_identity(opened) != expected_identity
                        or opened.st_nlink != 1
                        or _sha256_descriptor(staged_fd) != row.get("sha256")
                    ):
                        raise RuntimeError(
                            "staged archive source changed before release"
                        )
                finally:
                    os.close(staged_fd)
                os.unlink(staged_name, dir_fd=staging_fd)
            os.fsync(staging_fd)
            progress_callback(
                {
                    "event": "RELEASED",
                    "path": str(relative).replace("\\", "/"),
                    "bytes": int(row.get("bytes", 0)),
                }
            )
        finally:
            if staging_fd is not None:
                os.close(staging_fd)
            if parent_fd is not None:
                os.close(parent_fd)
            if root_fd is not None:
                os.close(root_fd)
            if release_fence is not None:
                fcntl.flock(release_fence.fileno(), fcntl.LOCK_UN)
                release_fence.close()


def _select_archive_destination(
    *,
    canonical_destination: Path,
    versioned_destination: Path,
    row: Mapping[str, Any],
) -> tuple[Path, str, bool]:
    if canonical_destination.exists() or canonical_destination.is_symlink():
        try:
            _verify_archived_row(canonical_destination, row)
        except RuntimeError:
            if canonical_destination.is_symlink():
                raise
            if (
                _read_archive_publication_marker(
                    staged=None,
                    destination=canonical_destination,
                )
                is not None
            ):
                return canonical_destination, "canonical", False
            if versioned_destination.exists() or versioned_destination.is_symlink():
                _verify_archived_row(versioned_destination, row)
                return versioned_destination, "content_addressed_version", True
            return versioned_destination, "content_addressed_version", False
        if (
            _read_archive_publication_marker(
                staged=None,
                destination=canonical_destination,
            )
            is not None
        ):
            marker = _archive_publication_marker_path(canonical_destination)
            marker.unlink()
            _fsync_directory(marker.parent)
        return canonical_destination, "canonical", True
    return canonical_destination, "canonical", False


def apply_archive_plan(
    root: Path,
    plan: Mapping[str, Any],
    *,
    disposition_manifest: Mapping[str, Any],
    release_after_archive: bool = True,
    budget_root: Path | None = None,
    current_disposition_manifest_path: Path | None = None,
    current_authority_validator: Callable[[], None] | None = None,
    _transaction_fence_held: bool = False,
) -> dict[str, Any]:
    """Apply one archive plan using only the actuator-owned release journal."""

    return _apply_archive_plan_authorized(
        root,
        plan,
        disposition_manifest=disposition_manifest,
        release_after_archive=release_after_archive,
        budget_root=budget_root,
        current_disposition_manifest_path=current_disposition_manifest_path,
        current_authority_validator=current_authority_validator,
        _transaction_fence_held=_transaction_fence_held,
        _release_journal_capability=None,
    )


def _apply_archive_plan_authorized(
    root: Path,
    plan: Mapping[str, Any],
    *,
    disposition_manifest: Mapping[str, Any],
    release_after_archive: bool,
    budget_root: Path | None,
    current_disposition_manifest_path: Path | None,
    current_authority_validator: Callable[[], None] | None,
    _transaction_fence_held: bool,
    _release_journal_capability: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Copy durable rows atomically, verify them, and release eligible R3 rows."""

    validated_manifest = validate_artifact_disposition_manifest_v1(disposition_manifest)
    if not legacy_reconstruction_is_accepted_v1(validated_manifest):
        raise ArtifactLifecycleError(
            "legacy reconstructed durable material requires an accepted evidence review"
        )
    runtime_root = root.expanduser().resolve()
    if (
        plan.get("schema_version") != "eiel_runtime_storage_archive_plan_v1"
        or plan.get("runtime_root") != str(runtime_root)
        or plan.get("disposition_manifest_sha256")
        != validated_manifest.get("manifest_sha256")
        or plan.get("run_id")
        != str(validated_manifest.get("run_id") or runtime_root.name)
    ):
        raise ArtifactLifecycleError(
            "archive plan is not bound to the validated disposition authority"
        )
    archive_root = _validate_cold_archive_separation(
        runtime_root=runtime_root,
        requested_archive_root=Path(str(plan.get("archive_root") or "")),
    )
    budget_scope_root = _archive_budget_scope_root(
        runtime_root=runtime_root,
        archive_root=archive_root,
        requested_budget_root=budget_root,
    )
    canonical_data2 = DATA2_FILESYSTEM_ROOT.expanduser().resolve()
    if (
        _is_beneath(runtime_root, canonical_data2)
        and release_after_archive
        and current_disposition_manifest_path is None
        and any(
            isinstance(row, Mapping)
            and row.get("retention_class") == "R3"
            and row.get("release_eligible") is True
            for row in plan.get("selected", [])
        )
    ):
        raise ArtifactLifecycleError(
            "canonical configured runtime root archived-source release requires the current "
            "disposition manifest path"
        )
    if RunStorageGuard.run_root_is_reserved(runtime_root):
        raise ArtifactLifecycleError("active run storage reservation blocks archival")
    availability_policy = load_data2_storage_availability_policy_v1()
    if plan.get(
        "storage_availability_policy"
    ) != storage_availability_policy_binding_v1(availability_policy):
        raise ArtifactLifecycleError(
            "archive plan is not bound to the canonical storage availability policy"
        )
    cold_profile_policy = availability_policy["cold_diagnostic_profile_assets"]
    max_profile_object_bytes = int(cold_profile_policy["max_object_bytes"])
    max_profile_aggregate_bytes = int(cold_profile_policy["max_aggregate_bytes"])
    durable_authority = {
        str(row.get("path")): row
        for row in validated_manifest.get("artifacts", [])
        if isinstance(row, Mapping)
        and row.get("retention_class") in {"R3", "R4", "R5"}
        and row.get("disposition") == "RETAIN_DURABLE"
        and row.get("exact_object_identity") is True
    }
    selected_durable_bytes = sum(
        int(durable_authority[path].get("bytes", 0))
        for path in {
            str(row.get("path") or "")
            for row in plan.get("selected", [])
            if isinstance(row, Mapping)
        }
        if path in durable_authority
    )
    if selected_durable_bytes > max_profile_aggregate_bytes:
        raise ArtifactLifecycleError(
            "current plan durable archive selection exceeds conservative canonical "
            "aggregate cold cap"
        )
    expected_budget_receipt = _cold_profile_budget_paths(budget_scope_root)[0]
    expected_guard = {
        "max_object_bytes": max_profile_object_bytes,
        "max_aggregate_bytes": max_profile_aggregate_bytes,
        "selected_bytes": selected_durable_bytes,
        "budget_scope": _COLD_PROFILE_BUDGET_SCOPE,
        "aggregate_scope": "cross_manifest_actuator_budget_receipt_v1",
        "cross_manifest_aggregate_status": (
            "ENFORCED_AT_APPLY_BY_SHARED_BUDGET_RECEIPT_V1"
        ),
        "budget_receipt": str(expected_budget_receipt),
    }
    if plan.get("cold_diagnostic_profile_guard") != expected_guard:
        raise ArtifactLifecycleError(
            "archive plan cold budget scope differs from the expected actuator scope"
        )
    release_rows = _archive_release_selected_rows(plan) if release_after_archive else []
    if release_rows and _release_journal_capability is None:
        with _direct_archive_release_journal(
            runtime_root=runtime_root,
            plan=plan,
            manifest=validated_manifest,
        ) as direct_journal:
            result = _apply_archive_plan_authorized(
                runtime_root,
                plan,
                disposition_manifest=validated_manifest,
                release_after_archive=True,
                budget_root=budget_root,
                current_disposition_manifest_path=current_disposition_manifest_path,
                current_authority_validator=current_authority_validator,
                _transaction_fence_held=_transaction_fence_held,
                _release_journal_capability=direct_journal,
            )
            current_progress = direct_journal["progress"]()
            selected_paths = {str(row["path"]) for row in release_rows}
            if set(current_progress) == selected_paths and all(
                row.get("state") == "RELEASED" for row in current_progress.values()
            ):
                direct_journal["complete"]()
            return {
                **result,
                "release_journal": str(direct_journal["paths"]["journal"]),
                "release_journal_status": direct_journal["status"](),
            }
    if _release_journal_capability is not None:
        expected_paths = _archive_release_transaction_paths(
            runtime_root=runtime_root,
            plan=plan,
            manifest=validated_manifest,
        )
        if (
            not release_rows
            or _release_journal_capability.get("token")
            is not _ARCHIVE_RELEASE_JOURNAL_TOKEN
            or _release_journal_capability.get("paths") != expected_paths
            or not callable(_release_journal_capability.get("progress"))
            or not callable(_release_journal_capability.get("record"))
        ):
            raise ArtifactLifecycleError(
                "archived-source release lacks the actuator-owned journal capability"
            )
        release_staging_root = expected_paths["staging"]
        release_progress = _release_journal_capability["progress"]()
        release_progress_callback = _release_journal_capability["record"]
    else:
        release_staging_root = None
        release_progress = None
        release_progress_callback = None
    archived: list[dict[str, Any]] = []
    already_archived: list[str] = []
    already_released: list[str] = []
    failures: list[dict[str, str]] = []
    released: list[str] = []
    budget_receipt_path: Path | None = None
    for raw_row in plan.get("selected", []):
        row = dict(raw_row)
        relative = str(row.get("path") or "")
        authority = durable_authority.get(relative)
        if authority is None:
            failures.append(
                {
                    "path": relative,
                    "type": "ArtifactLifecycleError",
                    "message": "archive row lacks explicit RETAIN_DURABLE authority",
                }
            )
            continue
        if int(authority.get("bytes", 0)) > max_profile_object_bytes:
            failures.append(
                {
                    "path": relative,
                    "type": "ArtifactLifecycleError",
                    "message": (
                        "explicit durable object exceeds conservative canonical "
                        "per-object cold cap"
                    ),
                }
            )
            continue
        expected_canonical_destination = _archive_destination(
            runtime_root,
            archive_root,
            relative,
            str(validated_manifest.get("run_id") or runtime_root.name),
        )
        expected_versioned_destination = _archive_version_destination(
            runtime_root,
            archive_root,
            relative,
            str(validated_manifest.get("run_id") or runtime_root.name),
            authority,
        )
        identity_fields = (
            "bytes",
            "retention_class",
            "disposition",
            "artifact_family",
            "artifact_type",
            "release_eligible",
            "sha256",
            "tree_sha256",
        )
        if any(row.get(name) != authority.get(name) for name in identity_fields) or (
            Path(str(row.get("destination") or "")).expanduser()
            != expected_canonical_destination
            or Path(str(row.get("versioned_destination") or "")).expanduser()
            != expected_versioned_destination
        ):
            failures.append(
                {
                    "path": relative,
                    "type": "ArtifactLifecycleError",
                    "message": "archive row differs from exact disposition authority",
                }
            )
            continue
        source = runtime_root / relative
        canonical_destination = expected_canonical_destination
        budget_lock = None
        try:
            budget_lock, budget_receipt_path = _acquire_cold_profile_budget_lock(
                budget_scope_root
            )
            destination, destination_mode, already_present = (
                _select_archive_destination(
                    canonical_destination=canonical_destination,
                    versioned_destination=expected_versioned_destination,
                    row=row,
                )
            )
            prior_release_row = (
                release_progress.get(relative)
                if isinstance(release_progress, Mapping)
                else None
            )
            source_identity = (
                _capture_archive_source_identity(source, row)
                if source.exists() or source.is_symlink()
                else tuple(prior_release_row["identity"])
                if isinstance(prior_release_row, Mapping)
                and prior_release_row.get("state") in {"PREPARED", "STAGED", "RELEASED"}
                and isinstance(prior_release_row.get("identity"), list)
                and len(prior_release_row["identity"]) == 4
                else None
            )
            policy_binding = storage_availability_policy_binding_v1(availability_policy)
            budget_receipt = _load_cold_profile_budget_receipt(
                budget_receipt_path,
                budget_scope_root=budget_scope_root,
                policy_binding=policy_binding,
                max_object_bytes=max_profile_object_bytes,
                max_aggregate_bytes=max_profile_aggregate_bytes,
            )
            budget_candidate = _cold_profile_budget_row(
                manifest_sha256=str(validated_manifest["manifest_sha256"]),
                run_id=str(validated_manifest.get("run_id") or runtime_root.name),
                row=row,
                destination=destination,
                status="RESERVED",
            )
            budget_receipt, _reservation_status = _reserve_cold_profile_budget_row(
                budget_receipt,
                candidate=budget_candidate,
                max_aggregate_bytes=max_profile_aggregate_bytes,
            )
            _write_cold_profile_budget_receipt(budget_receipt_path, budget_receipt)
            if not already_present:
                if source_identity is None:
                    raise RuntimeError(f"archive source is missing or linked: {source}")
                _copy_one(
                    source,
                    destination,
                    row=row,
                    object_identity_sha256=str(
                        budget_candidate["object_identity_sha256"]
                    ),
                )
            else:
                already_archived.append(str(row["path"]))
            archive_destination_identity = _verify_archive_destination_exact(
                destination, row
            )
            release_candidate = (
                release_after_archive
                and row.get("retention_class") == "R3"
                and row.get("release_eligible") is True
            )
            if release_candidate:
                if source_identity is None:
                    raise RuntimeError(
                        "archived release source is absent without durable journal progress"
                    )
                else:
                    release_transaction = (
                        nullcontext()
                        if _transaction_fence_held
                        else run_artifact_transaction_fence_v1(runtime_root)
                    )
                    with release_transaction:
                        _release_archived_source_exact(
                            runtime_root=runtime_root,
                            source=source,
                            archive_destination=destination,
                            row=row,
                            expected_identity=source_identity,
                            expected_archive_identity=archive_destination_identity,
                            current_disposition_manifest_path=(
                                current_disposition_manifest_path
                            ),
                            expected_disposition_manifest=validated_manifest,
                            current_authority_validator=current_authority_validator,
                            release_staging_root=release_staging_root,
                            prior_release_progress=prior_release_row,
                            progress_callback=release_progress_callback,
                            _transaction_fence_held=True,
                            _archive_preverified=True,
                        )
                        budget_receipt = _mark_cold_profile_budget_verified(
                            budget_receipt,
                            object_identity_sha256=str(
                                budget_candidate["object_identity_sha256"]
                            ),
                        )
                        _write_cold_profile_budget_receipt(
                            budget_receipt_path, budget_receipt
                        )
                    released.append(str(row["path"]))
            if not release_candidate or source_identity is None:
                budget_receipt = _mark_cold_profile_budget_verified(
                    budget_receipt,
                    object_identity_sha256=str(
                        budget_candidate["object_identity_sha256"]
                    ),
                )
                _write_cold_profile_budget_receipt(budget_receipt_path, budget_receipt)
            archived.append(
                {
                    **row,
                    "destination": str(destination),
                    "canonical_destination": str(canonical_destination),
                    "destination_mode": destination_mode,
                    "verified": True,
                    "already_present": str(row["path"]) in already_archived,
                }
            )
        except (ArtifactLifecycleError, OSError, RuntimeError) as exc:
            failures.append(
                {
                    "path": str(row.get("path")),
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            )
        finally:
            if budget_lock is not None:
                fcntl.flock(budget_lock.fileno(), fcntl.LOCK_UN)
                budget_lock.close()
    return {
        "archived": archived,
        "already_archived": already_archived,
        "already_released": already_released,
        "released": released,
        "failures": failures,
        "archived_bytes": sum(int(row.get("bytes", 0)) for row in archived),
        "released_bytes": sum(
            int(row.get("bytes", 0))
            for row in archived
            if str(row.get("path")) in released
        ),
        "cold_profile_budget_receipt": (
            str(budget_receipt_path) if budget_receipt_path is not None else None
        ),
    }


def _bundle_root(path: Path, root: Path) -> Path | None:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return None
    parts = rel.parts
    if "checkpoint_bundles_v2" not in parts:
        return None
    idx = parts.index("checkpoint_bundles_v2")
    if len(parts) <= idx + 1:
        return None
    if str(parts[idx + 1]).startswith("bundle_gen"):
        return root.joinpath(*parts[: idx + 2])
    if len(parts) > idx + 2 and str(parts[idx + 2]).startswith("bundle_gen"):
        return root.joinpath(*parts[: idx + 3])
    return None


def _checkpoint_bundle_dirs(root: Path) -> list[Path]:
    bundles: set[Path] = set()
    if not root.exists():
        return []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        bundle = _bundle_root(path, root)
        if bundle is not None:
            bundles.add(bundle)
    return sorted(bundles, key=lambda p: str(p.relative_to(root)))


def _preserved_final_bundles(root: Path) -> set[Path]:
    by_parent: dict[Path, list[Path]] = {}
    for bundle in _checkpoint_bundle_dirs(root):
        by_parent.setdefault(bundle.parent, []).append(bundle)
    keep: set[Path] = set()
    for bundles in by_parent.values():
        keep.add(sorted(bundles, key=lambda p: p.name)[-1])
    return keep


def _classify(path: Path, root: Path, final_bundles: set[Path]) -> str:
    rel = str(path.relative_to(root)).replace("\\", "/")
    name = path.name
    if rel.endswith(".jsonl") and (
        "lineage_reproduction_funnel" in rel or "funnel" in rel
    ):
        return "event_jsonl"
    if name in {"developmental_stage_trace.jsonl", "developmental_lock_trace.jsonl"}:
        return "developmental_traces"
    if rel.endswith(".jsonl") and ("trace" in rel or "debug" in rel):
        return "debug_traces"
    bundle = _bundle_root(path, root)
    if bundle is not None and bundle not in final_bundles:
        return "intermediate_checkpoints"
    return ""


def build_prune_plan(
    root: Path,
    categories: set[str],
    disposition_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    if disposition_manifest is not None:
        validated = validate_artifact_disposition_manifest_v1(disposition_manifest)
        selected: list[dict[str, Any]] = []
        for row in validated["artifacts"]:
            if (
                not isinstance(row, Mapping)
                or row.get("release_eligible") is not True
                or row.get("exact_object_identity") is not True
            ):
                continue
            retention_class = str(row.get("retention_class") or "RX")
            if retention_class not in {"R0", "R1", "R2"}:
                continue
            relative = str(row.get("path") or "").replace("\\", "/").strip("/")
            if not relative or Path(relative).is_absolute():
                raise RuntimeError(
                    "disposition manifest contains an invalid relative path"
                )
            candidate = root / relative
            try:
                candidate.resolve(strict=False).relative_to(root)
            except ValueError as exc:
                raise RuntimeError(
                    f"disposition path escapes root: {relative}"
                ) from exc
            checkpoint_role = row.get("checkpoint_role")
            family = str(row.get("artifact_family") or "").lower()
            if checkpoint_role == "intermediate":
                category = "intermediate_checkpoints"
            elif "developmental" in family and "trace" in family:
                category = "developmental_traces"
            elif "event" in family:
                category = "event_jsonl"
            elif "trace" in family or "debug" in family:
                category = "debug_traces"
            elif retention_class == "R2":
                category = "reproducible_derived"
            else:
                category = "scratch"
            if category not in categories:
                continue
            selected.append(
                {
                    "path": relative,
                    "bytes": int(row.get("bytes", 0)),
                    "category": category,
                    "retention_class": retention_class,
                    "checkpoint_role": checkpoint_role,
                    "release_predicate": row.get("release_predicate"),
                    "artifact_type": row.get("artifact_type", "file"),
                    **(
                        {"sha256": row["sha256"]}
                        if isinstance(row.get("sha256"), str)
                        else {}
                    ),
                    **(
                        {"tree_sha256": row["tree_sha256"]}
                        if isinstance(row.get("tree_sha256"), str)
                        else {}
                    ),
                }
            )
        selected.sort(key=lambda row: (str(row["category"]), str(row["path"])))
        return {
            "schema_version": "eiel_runtime_storage_prune_plan_v1",
            "root": str(root),
            "categories": sorted(categories),
            "apply_required": True,
            "selected_count": len(selected),
            "selected_bytes": sum(int(row["bytes"]) for row in selected),
            "selected": selected,
            "preserved_examples": [],
            "preserved_final_checkpoint_bundles": [],
            "authority_status": "lifecycle_disposition_manifest",
            "disposition_manifest_sha256": validated["manifest_sha256"],
            "automatic_prune_authorized": bool(
                validated.get("prune", {}).get("automatic_prune_eligible") is True
            ),
            "unknown_artifact_policy": validated.get("retention", {}).get(
                "unknown_artifact_policy", "RX_BLOCKS_AUTO_DELETE"
            ),
            "unknown_artifact_scope": validated.get("retention", {}).get(
                "unknown_artifact_scope", "global"
            ),
        }

    final_bundles = _preserved_final_bundles(root)
    selected_files: list[dict[str, Any]] = []
    selected_bundle_dirs: set[Path] = set()
    preserved: list[str] = []
    total_bytes = 0
    for path in root.rglob("*") if root.exists() else []:
        if not path.is_file():
            continue
        rel = str(path.relative_to(root)).replace("\\", "/")
        category = _classify(path, root, final_bundles)
        if category and category in categories:
            size = int(path.stat().st_size)
            bundle = _bundle_root(path, root)
            if category == "intermediate_checkpoints" and bundle is not None:
                selected_bundle_dirs.add(bundle)
            else:
                selected_files.append(
                    {"path": rel, "bytes": size, "category": category}
                )
                total_bytes += size
        elif path.name in {
            "config.json",
            "manifest.json",
            "run_config.csv",
            "metrics.csv",
        }:
            preserved.append(rel)
    for bundle in sorted(selected_bundle_dirs, key=lambda p: str(p.relative_to(root))):
        size = sum(int(p.stat().st_size) for p in bundle.rglob("*") if p.is_file())
        selected_files.append(
            {
                "path": str(bundle.relative_to(root)).replace("\\", "/"),
                "bytes": size,
                "category": "intermediate_checkpoints",
                "directory": True,
            }
        )
        total_bytes += size
    plan = {
        "schema_version": "eiel_runtime_storage_prune_plan_v1",
        "root": str(root),
        "categories": sorted(categories),
        "apply_required": True,
        "selected_count": len(selected_files),
        "selected_bytes": total_bytes,
        "selected": sorted(
            selected_files, key=lambda row: (str(row["category"]), str(row["path"]))
        ),
        "preserved_examples": sorted(preserved)[:40],
        "preserved_final_checkpoint_bundles": [
            str(path.relative_to(root)).replace("\\", "/")
            for path in sorted(final_bundles)
        ],
    }
    plan.update(
        {
            "authority_status": "legacy_heuristic_dry_run_only",
            "disposition_manifest_sha256": None,
            "automatic_prune_authorized": False,
            "unknown_artifact_policy": "RX_BLOCKS_AUTO_DELETE",
        }
    )
    return plan


def apply_prune_plan(
    root: Path,
    plan: dict[str, Any],
    *,
    disposition_manifest: Mapping[str, Any] | None = None,
    current_disposition_manifest_path: Path | None = None,
    certificate_release_context: Mapping[str, Any] | None = None,
    _transaction_fence_held: bool = False,
    _active_run_guard: RunStorageGuard | None = None,
) -> dict[str, Any]:
    """Apply exact release authority with an actuator-owned durable journal."""

    if disposition_manifest is None or not plan.get("selected"):
        return _apply_prune_plan_authorized(
            root,
            plan,
            disposition_manifest=disposition_manifest,
            current_disposition_manifest_path=current_disposition_manifest_path,
            certificate_release_context=certificate_release_context,
            _transaction_fence_held=_transaction_fence_held,
            _release_journal_capability=None,
            _active_run_guard=_active_run_guard,
        )
    validated_manifest = validate_artifact_disposition_manifest_v1(disposition_manifest)
    runtime_root = root.expanduser().resolve()
    active_boundary = validated_manifest.get("active_release_boundary") is True
    if active_boundary:
        if _active_run_guard is None:
            raise ArtifactLifecycleError(
                "active release requires the owning live storage guard"
            )
        _active_run_guard.validate_active_release_owner_v1(
            run_root=runtime_root,
            lifecycle_contract_sha256=str(
                validated_manifest.get("lifecycle_contract_sha256")
            ),
        )
    elif _active_run_guard is not None:
        raise ArtifactLifecycleError(
            "an active storage guard cannot authorize terminal cleanup"
        )
    disposition_manifest_authorizes_prune_v1(
        validated_manifest,
        selected_paths=[str(row.get("path")) for row in plan.get("selected", [])],
        allow_active_release=active_boundary,
    )
    if Path(str(validated_manifest.get("runtime_root"))).expanduser().resolve() != (
        runtime_root
    ):
        raise ArtifactLifecycleError(
            "disposition authority is bound to a different runtime root"
        )
    journal_plan = _prune_release_journal_plan(
        runtime_root=runtime_root,
        plan=plan,
        manifest=validated_manifest,
    )
    with _direct_archive_release_journal(
        runtime_root=runtime_root,
        plan=journal_plan,
        manifest=validated_manifest,
    ) as direct_journal:
        result = _apply_prune_plan_authorized(
            runtime_root,
            plan,
            disposition_manifest=validated_manifest,
            current_disposition_manifest_path=current_disposition_manifest_path,
            certificate_release_context=certificate_release_context,
            _transaction_fence_held=_transaction_fence_held,
            _release_journal_capability=direct_journal,
            _active_run_guard=_active_run_guard,
        )
        current_progress = direct_journal["progress"]()
        selected_paths = {
            str(row.get("path"))
            for row in plan.get("selected", [])
            if isinstance(row, Mapping)
        }
        if set(current_progress) == selected_paths and all(
            row.get("state") == "RELEASED" for row in current_progress.values()
        ):
            direct_journal["complete"]()
        return {
            **result,
            "release_journal": str(direct_journal["paths"]["journal"]),
            "release_journal_status": direct_journal["status"](),
        }


def _apply_prune_plan_authorized(
    root: Path,
    plan: dict[str, Any],
    *,
    disposition_manifest: Mapping[str, Any] | None,
    current_disposition_manifest_path: Path | None,
    certificate_release_context: Mapping[str, Any] | None,
    _transaction_fence_held: bool,
    _release_journal_capability: Mapping[str, Any] | None,
    _active_run_guard: RunStorageGuard | None,
) -> dict[str, Any]:
    if type(_transaction_fence_held) is not bool:
        raise ArtifactLifecycleError(
            "run artifact transaction fence ownership flag must be boolean"
        )
    if not _transaction_fence_held:
        transaction_roots = {root.expanduser().resolve()}
        if certificate_release_context is not None:
            transaction_request = validate_evidence_certificate_request_v1(
                certificate_release_context.get("request", {})
            )
            transaction_certificate = validate_certificate_request_binding_v1(
                transaction_request,
                certificate_release_context.get("certificate", {}),
            )
            evidence_root = root.expanduser().resolve()
            manifest_paths = {
                str(binding["manifest_path"])
                for row in transaction_certificate.get("artifacts", [])
                if isinstance(row, Mapping)
                and isinstance(row.get("source_manifest_binding"), Mapping)
                for binding in [row["source_manifest_binding"]]
            }
            for relative_manifest in sorted(manifest_paths):
                candidate = (evidence_root / relative_manifest).resolve(strict=True)
                try:
                    candidate.relative_to(evidence_root)
                except ValueError as exc:
                    raise ArtifactLifecycleError(
                        "certificate source manifest escapes the evidence root"
                    ) from exc
                try:
                    source_manifest = validate_artifact_disposition_manifest_v1(
                        json.loads(candidate.read_text(encoding="utf-8"))
                    )
                except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                    raise ArtifactLifecycleError(
                        "certificate source manifest cannot establish a transaction fence"
                    ) from exc
                transaction_roots.add(
                    Path(str(source_manifest["runtime_root"])).expanduser().resolve()
                )
        with ExitStack() as transaction_stack:
            for transaction_root in sorted(transaction_roots, key=str):
                transaction_stack.enter_context(
                    run_artifact_transaction_fence_v1(transaction_root)
                )
            return _apply_prune_plan_authorized(
                root,
                plan,
                disposition_manifest=disposition_manifest,
                current_disposition_manifest_path=(current_disposition_manifest_path),
                certificate_release_context=certificate_release_context,
                _transaction_fence_held=True,
                _release_journal_capability=_release_journal_capability,
                _active_run_guard=_active_run_guard,
            )
    certificate_boundary_context: (
        tuple[
            Mapping[str, Any],
            Mapping[str, Any],
            Mapping[str, Any] | None,
            bool,
        ]
        | None
    ) = None
    diagnostic_release_requires_live_recheck = False
    if disposition_manifest is None:
        raise RuntimeError(
            "destructive cleanup requires a valid lifecycle disposition manifest"
        )
    validated_manifest = validate_artifact_disposition_manifest_v1(disposition_manifest)
    journal_plan = _prune_release_journal_plan(
        runtime_root=root.resolve(),
        plan=plan,
        manifest=validated_manifest,
    )
    selected_paths = {
        str(row.get("path"))
        for row in plan.get("selected", [])
        if isinstance(row, Mapping)
    }
    if selected_paths and _release_journal_capability is None:
        raise ArtifactLifecycleError(
            "prune release requires the actuator-owned journal capability"
        )
    if _release_journal_capability is not None:
        expected_paths = _archive_release_transaction_paths(
            runtime_root=root.resolve(),
            plan=journal_plan,
            manifest=validated_manifest,
        )
        if (
            not selected_paths
            or _release_journal_capability.get("token")
            is not _ARCHIVE_RELEASE_JOURNAL_TOKEN
            or _release_journal_capability.get("paths") != expected_paths
            or not callable(_release_journal_capability.get("progress"))
            or not callable(_release_journal_capability.get("record"))
        ):
            raise ArtifactLifecycleError(
                "prune release lacks the actuator-owned journal capability"
            )
        staging_root = expected_paths["staging"]
        release_progress = _release_journal_capability["progress"]()
        progress_callback = _release_journal_capability["record"]
        authenticated_absent_paths = frozenset(
            path
            for path, row in release_progress.items()
            if row.get("state") in {"PREPARED", "STAGED", "RELEASED"}
            and not (root.resolve() / path).exists()
        )
    else:
        staging_root = None
        release_progress = None
        progress_callback = None
        authenticated_absent_paths = frozenset()
    fully_journal_released = (
        bool(selected_paths)
        and set(release_progress or {}) == (selected_paths)
        and all(
            row.get("state") == "RELEASED" for row in (release_progress or {}).values()
        )
    )
    all_sources_crossed_destructive_boundary = (
        bool(selected_paths)
        and set(release_progress or {}) == selected_paths
        and all(
            row.get("state") in {"STAGED", "RELEASED"}
            for row in (release_progress or {}).values()
        )
        and all(
            not (root.resolve() / path).exists()
            and not (root.resolve() / path).is_symlink()
            for path in selected_paths
        )
    )
    release_authority = validated_manifest.get("release_authority", {})
    certificate_authority_kind = release_authority.get("kind")
    if certificate_authority_kind in {
        "EVIDENCE_CERTIFICATE_GATE_CLOSURE",
        "EVIDENCE_CERTIFICATE_DIAGNOSTIC_CLOSEOUT",
    }:
        if certificate_release_context is None:
            raise ArtifactLifecycleError(
                "certificate-backed release requires exact close-gate context"
            )
        bound_request = validate_evidence_certificate_request_v1(
            certificate_release_context.get("request", {})
        )
        bound_certificate = validate_certificate_request_binding_v1(
            bound_request, certificate_release_context.get("certificate", {})
        )
        bound_intent = validate_reclamation_intent_v1(
            certificate_release_context.get("intent", {})
        )
        if bound_intent.get("status") != "APPLYING":
            raise ArtifactLifecycleError(
                "certificate-backed release requires a durable APPLYING intent"
            )
        expected_selected = sorted(
            [
                {
                    "path": str(row.get("path")),
                    "bytes": int(row.get("bytes", 0)),
                    "sha256": row.get("sha256"),
                    "tree_sha256": row.get("tree_sha256"),
                }
                for row in plan.get("selected", [])
                if isinstance(row, Mapping)
            ],
            key=lambda row: row["path"],
        )
        observed_selected = sorted(
            [dict(row) for row in bound_intent.get("selected", [])],
            key=lambda row: str(row.get("path")),
        )
        if (
            bound_certificate.get("phase") != "SEALED_PRE_RELEASE"
            or bound_intent.get("request_sha256") != canonical_sha256(bound_request)
            or bound_intent.get("certificate_id")
            != bound_certificate.get("certificate_id")
            or bound_intent.get("certificate_sha256")
            != bound_certificate.get("certificate_sha256")
            or bound_intent.get("manifest_sha256")
            != validated_manifest.get("manifest_sha256")
            or bound_intent.get("programme_plan_sha256")
            != bound_request["programme_plan"]["plan_sha256"]
            or observed_selected != expected_selected
            or bound_intent.get("selected_bytes")
            != sum(int(row["bytes"]) for row in expected_selected)
        ):
            raise ArtifactLifecycleError(
                "certificate release context differs from manifest authority"
            )
        expected_bindings = {
            "kind": certificate_authority_kind,
            "certificate_id": bound_certificate["certificate_id"],
            "certificate_sha256": bound_certificate["certificate_sha256"],
            "request_sha256": bound_certificate["request_sha256"],
            "programme_plan_sha256": bound_request["programme_plan"]["plan_sha256"],
            "gate_id": bound_request["gate_id"],
        }
        diagnostic = bound_request.get("diagnostic_closeout")
        diagnostic_authority = (
            certificate_authority_kind == "EVIDENCE_CERTIFICATE_DIAGNOSTIC_CLOSEOUT"
        )
        diagnostic_release_requires_live_recheck = diagnostic_authority
        if diagnostic_authority != isinstance(diagnostic, Mapping):
            raise ArtifactLifecycleError(
                "certificate authority kind differs from its request lifecycle"
            )
        diagnostic_source_rows_by_path: dict[str, dict[str, Any]] = {}
        if diagnostic_authority:
            assert isinstance(diagnostic, Mapping)
            source_identity = diagnostic["source_manifest"]
            unresolved_source = root / str(source_identity["path"])
            if unresolved_source.is_symlink():
                raise ArtifactLifecycleError(
                    "diagnostic certificate source manifest cannot be a symlink"
                )
            source_path = unresolved_source.resolve(strict=True)
            try:
                source_path.relative_to(root.resolve())
            except ValueError as exc:
                raise ArtifactLifecycleError(
                    "diagnostic certificate source manifest escapes the evidence root"
                ) from exc
            source_manifest = validate_artifact_disposition_manifest_v1(
                json.loads(source_path.read_text(encoding="utf-8"))
            )
            expected_source_identity = {
                "manifest_sha256": source_identity["manifest_sha256"],
                "campaign_id": source_identity["campaign_id"],
                "run_id": source_identity["run_id"],
                "runtime_root": str(root.resolve()),
                "plan_sha256": source_identity["plan_sha256"],
                "lifecycle_contract_sha256": source_identity[
                    "lifecycle_contract_sha256"
                ],
            }
            if any(
                source_manifest.get(name) != expected
                for name, expected in expected_source_identity.items()
            ):
                raise ArtifactLifecycleError(
                    "diagnostic certificate source manifest identity changed"
                )
            diagnostic_source_rows_by_path = {
                str(row.get("path")): dict(row)
                for row in source_manifest.get("artifacts", [])
                if isinstance(row, Mapping)
            }
        root_requires_permanent = evidence_root_requires_permanent_capsule_v1(root)
        source_summary = bound_certificate.get("source_manifest_binding_summary")
        if source_summary is not None:
            expected_bindings.update(
                {
                    "source_manifest_binding_count": source_summary["binding_count"],
                    "source_manifest_bindings_sha256": source_summary[
                        "bindings_sha256"
                    ],
                }
            )
        certificate_rows = {
            str(row["artifact_id"]): dict(row)
            for row in bound_certificate.get("artifacts", [])
            if isinstance(row, Mapping)
        }
        manifested_certificate_rows = {
            str(row.get("artifact_id")): dict(row)
            for row in validated_manifest.get("artifacts", [])
            if isinstance(row, Mapping)
            and str(row.get("artifact_id")) in certificate_rows
        }
        if set(manifested_certificate_rows) != set(certificate_rows):
            raise ArtifactLifecycleError(
                "certificate release manifest does not exactly account for certificate rows"
            )
        for artifact_id, certificate_row in certificate_rows.items():
            manifest_row = manifested_certificate_rows[artifact_id]
            digest_name = "sha256" if "sha256" in certificate_row else "tree_sha256"
            expected_disposition = (
                "RELEASE"
                if certificate_row["release_eligible"] is True
                else "RETAIN_DURABLE"
                if certificate_row["retention_class"] in {"R3", "R4", "R5"}
                else "RX"
                if certificate_row["retention_class"] == "RX"
                else "KEEP_ACTIVE"
            )
            expected_row = {
                "artifact_id": artifact_id,
                "path": certificate_row["path"],
                "bytes": certificate_row["bytes"],
                digest_name: certificate_row[digest_name],
                "retention_class": certificate_row["retention_class"],
                "release_eligible": certificate_row["release_eligible"],
                "release_predicate": certificate_row["disposition_reason"],
                "reason": certificate_row["disposition_reason"],
                "disposition": expected_disposition,
                "certificate_disposition": certificate_row["disposition"],
                "evidence_certificate_id": bound_certificate["certificate_id"],
                "evidence_certificate_sha256": bound_certificate["certificate_sha256"],
            }
            for name in ("source_manifest_binding", "source_retention_class"):
                if name in certificate_row:
                    expected_row[name] = certificate_row[name]
            if diagnostic_authority:
                source_row = diagnostic_source_rows_by_path.get(
                    str(certificate_row["path"])
                )
                if source_row is None:
                    raise ArtifactLifecycleError(
                        f"diagnostic source row is absent for {artifact_id}"
                    )
                superseded = (
                    certificate_row.get("diagnostic_closeout_disposition")
                    == "SUPERSEDED_BY_CERTIFICATE"
                )
                expected_row.update(
                    {
                        "live_reference_status": (
                            "none"
                            if superseded
                            else str(
                                source_row.get("live_reference_status") or "unknown"
                            )
                        ),
                        "audit_hold_status": str(
                            source_row.get("audit_hold_status") or "unknown"
                        ),
                        "qualification_hold_status": str(
                            source_row.get("qualification_hold_status") or "unknown"
                        ),
                    }
                )
            if any(
                manifest_row.get(name) != expected
                for name, expected in expected_row.items()
            ):
                raise ArtifactLifecycleError(
                    f"certificate release manifest row differs for {artifact_id}"
                )
        if diagnostic_authority:

            def aggregate_source_hold_status(name: str) -> str:
                statuses = {
                    str(row.get(name) or "unknown")
                    for row in diagnostic_source_rows_by_path.values()
                }
                if "active" in statuses:
                    return "active"
                if statuses - {"released", "not_required"}:
                    return "unknown"
                return "released" if "released" in statuses else "not_required"

            retention = validated_manifest.get("retention")
            if not isinstance(retention, Mapping) or any(
                retention.get(name) != aggregate_source_hold_status(name)
                for name in ("audit_hold_status", "qualification_hold_status")
            ):
                raise ArtifactLifecycleError(
                    "diagnostic release manifest erases protected source hold authority"
                )
        expected_release_paths = {
            str(row["path"])
            for row in certificate_rows.values()
            if row.get("release_eligible") is True
        }
        if {str(row["path"]) for row in expected_selected} != expected_release_paths:
            raise ArtifactLifecycleError(
                "certificate prune selection differs from exact release-eligible rows"
            )
        requires_publication = (
            root_requires_permanent
            or diagnostic_authority
            or release_authority.get("permanent_capsule_publication_sha256") is not None
        )
        permanent_publication: dict[str, Any] | None = None
        if requires_publication:
            permanent_publication = validate_permanent_evidence_publication_v1(
                certificate_release_context.get("permanent_publication", {}),
                certificate=bound_certificate,
                source_root=root,
                require_outstanding_final_capacity=not fully_journal_released,
            )
            if permanent_publication.get("phase") != "SEALED_PRE_RELEASE":
                raise ArtifactLifecycleError(
                    "certificate release requires a SEALED_PRE_RELEASE permanent capsule"
                )
            expected_bindings.update(
                {
                    "permanent_capsule_publication_sha256": permanent_publication[
                        "publication_sha256"
                    ],
                    "permanent_capsule_sha256": permanent_publication["capsule_sha256"],
                }
            )
        if diagnostic_authority:
            assert isinstance(diagnostic, Mapping)
            certificate_ids = {
                str(row.get("artifact_id"))
                for row in bound_certificate.get("artifacts", [])
                if isinstance(row, Mapping)
            }
            manifested_ids = {
                str(row.get("artifact_id"))
                for row in validated_manifest.get("artifacts", [])
                if isinstance(row, Mapping)
                and str(row.get("artifact_id")) in certificate_ids
            }
            superseded_ids = list(diagnostic["superseded_artifact_ids"])
            preserved_ids = sorted(certificate_ids - set(superseded_ids))
            selected_ids = {
                str(manifest_row.get("artifact_id"))
                for selected_row in plan.get("selected", [])
                for manifest_row in validated_manifest.get("artifacts", [])
                if isinstance(selected_row, Mapping)
                and isinstance(manifest_row, Mapping)
                and selected_row.get("path") == manifest_row.get("path")
            }
            expected_bindings.update(
                {
                    "diagnostic_closeout_sha256": canonical_sha256(diagnostic),
                    "source_manifest": diagnostic["source_manifest"],
                    "substantive_review": diagnostic["substantive_review"],
                    "closure_receipt_sha256": diagnostic["closure_receipt_sha256"],
                    "closed_live_state": diagnostic["live_state"],
                    "closed_exact_byte_holds": diagnostic["active_exact_byte_holds"],
                    **(
                        {
                            "source_live_marker_closeout": diagnostic[
                                "source_live_marker_closeout"
                            ]
                        }
                        if "source_live_marker_closeout" in diagnostic
                        else {}
                    ),
                    "source_row_count": len(certificate_ids),
                    "accounted_source_row_count": len(manifested_ids),
                    "superseded_row_count": len(superseded_ids),
                    "preserved_row_count": len(preserved_ids),
                    "superseded_artifact_ids": superseded_ids,
                    "preserved_artifact_ids": preserved_ids,
                    "ordinary_releasable_archive_policy": ("DIRECT_RELEASE_NO_ARCHIVE"),
                }
            )
            if certificate_ids != manifested_ids or not selected_ids.issubset(
                set(superseded_ids)
            ):
                raise ArtifactLifecycleError(
                    "diagnostic closeout row accounting differs from release authority"
                )
        else:
            expected_bindings["ordinary_releasable_archive_policy"] = (
                "DIRECT_RELEASE_NO_ARCHIVE"
            )
        if dict(release_authority) != expected_bindings:
            raise ArtifactLifecycleError(
                "certificate release context differs from manifest authority"
            )
        certificate_boundary_context = (
            bound_request,
            bound_certificate,
            permanent_publication,
            root_requires_permanent or source_summary is not None,
        )
    authorization = disposition_manifest_authorizes_prune_v1(
        disposition_manifest,
        selected_paths=[str(row.get("path")) for row in plan.get("selected", [])],
        allow_active_release=(
            disposition_manifest.get("active_release_boundary") is True
        ),
    )
    manifest_root = disposition_manifest.get("runtime_root")
    if (
        not isinstance(manifest_root, str)
        or Path(manifest_root).expanduser().resolve() != root.resolve()
    ):
        raise ArtifactLifecycleError(
            "disposition authority is bound to a different runtime root"
        )
    release_fence = _acquire_release_fence(root.resolve())
    try:
        if current_disposition_manifest_path is not None:
            _revalidate_current_disposition_manifest(
                current_disposition_manifest_path,
                expected_manifest=validated_manifest,
                runtime_root=root,
            )
    except BaseException:
        fcntl.flock(release_fence.fileno(), fcntl.LOCK_UN)
        release_fence.close()
        raise
    if RunStorageGuard.run_root_is_reserved(root):
        if _active_run_guard is None:
            fcntl.flock(release_fence.fileno(), fcntl.LOCK_UN)
            release_fence.close()
            raise ArtifactLifecycleError(
                "active run storage reservation blocks destructive cleanup"
            )
        try:
            _active_run_guard.validate_active_release_owner_v1(
                run_root=root,
                lifecycle_contract_sha256=str(
                    validated_manifest.get("lifecycle_contract_sha256")
                ),
                reservation_lock_held=True,
            )
        except BaseException:
            fcntl.flock(release_fence.fileno(), fcntl.LOCK_UN)
            release_fence.close()
            raise
    if certificate_boundary_context is not None:
        (
            boundary_request,
            boundary_certificate,
            boundary_publication,
            recheck_source,
        ) = certificate_boundary_context
        try:
            if recheck_source:
                validate_certificate_source_manifest_bindings_v1(
                    boundary_request,
                    boundary_certificate,
                    evidence_root=root,
                    authenticated_absent_paths=authenticated_absent_paths,
                    require_fresh_diagnostic_review_audit=False,
                    recheck_diagnostic_local_live_state=(
                        not all_sources_crossed_destructive_boundary
                    ),
                )
            if boundary_publication is not None:
                observed_publication = validate_permanent_evidence_publication_v1(
                    boundary_publication,
                    certificate=boundary_certificate,
                    source_root=root,
                    require_outstanding_final_capacity=not fully_journal_released,
                )
                if observed_publication != boundary_publication:
                    raise ArtifactLifecycleError(
                        "permanent capsule changed at the destructive boundary"
                    )
        except (
            ArtifactLifecycleError,
            OSError,
            RuntimeError,
            UnicodeError,
            json.JSONDecodeError,
        ):
            fcntl.flock(release_fence.fileno(), fcntl.LOCK_UN)
            release_fence.close()
            raise
    manifest_rows = {
        str(row.get("path")): dict(row)
        for row in disposition_manifest.get("artifacts", [])
        if isinstance(row, Mapping)
    }
    staging_base: Path | None = None
    root_fd: int | None = None
    staging_fd: int | None = None
    staging_identity: tuple[int, int, int, int] | None = None
    if staging_root is not None:
        unresolved_staging = staging_root.expanduser()
        if unresolved_staging.is_symlink():
            raise RuntimeError("release staging root cannot be a symlink")
        staging_base = Path(os.path.abspath(unresolved_staging))
        try:
            staging_base.relative_to(root.resolve())
        except ValueError as exc:
            raise RuntimeError(
                "release staging root must be inside the bounded runtime root"
            ) from exc
        root_fd = os.open(root.resolve(), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        relative_staging = staging_base.relative_to(root.resolve())
        staging_fd = _open_directory_beneath(
            root_fd, tuple(relative_staging.parts), create=True
        )
        staging_identity = _stat_identity(os.fstat(staging_fd))
    validated_targets: list[
        tuple[
            Path,
            dict[str, Any],
            tuple[int, int, int, int],
            Path | None,
            bool,
            int | None,
            str | None,
            str | None,
            dict[str, Any],
            list[dict[str, Any]] | None,
        ]
    ] = []
    already_absent: list[str] = []
    recovered_released_rows: list[dict[str, Any]] = []
    for row in plan.get("selected", []):
        rel = str(row.get("path") or "").strip()
        if not rel:
            continue
        unresolved = root / rel
        staged_path = (
            staging_base / f"{canonical_sha256({'path': rel})[:24]}-{Path(rel).name}"
            if staging_base is not None
            else None
        )
        if staged_path is not None and (
            staged_path == unresolved or unresolved in staged_path.parents
        ):
            raise RuntimeError(
                f"release staging root overlaps selected artifact: {rel}"
            )
        candidate_parent_fd: int | None = None
        candidate_name: str | None = None
        staged_name = staged_path.name if staged_path is not None else None
        prior_release_row = (
            release_progress.get(rel) if isinstance(release_progress, Mapping) else None
        )
        if staging_fd is not None and root_fd is not None and staged_name is not None:
            source_relative = Path(rel)
            source_parent_fd = _open_directory_beneath(
                root_fd,
                (
                    tuple(source_relative.parent.parts)
                    if source_relative.parent != Path(".")
                    else ()
                ),
            )
            try:
                os.stat(
                    source_relative.name, dir_fd=source_parent_fd, follow_symlinks=False
                )
                source_exists = True
            except FileNotFoundError:
                source_exists = False
            try:
                os.stat(staged_name, dir_fd=staging_fd, follow_symlinks=False)
                staged_exists = True
            except FileNotFoundError:
                staged_exists = False
            if source_exists and staged_exists:
                os.close(source_parent_fd)
                raise RuntimeError(
                    f"both source and interrupted staging artifact exist: {rel}"
                )
            already_staged = bool(staged_exists and not source_exists)
            if already_staged:
                os.close(source_parent_fd)
                candidate_parent_fd = os.dup(staging_fd)
                candidate_name = staged_name
            elif source_exists:
                candidate_parent_fd = source_parent_fd
                candidate_name = source_relative.name
            else:
                os.close(source_parent_fd)
                if isinstance(prior_release_row, Mapping) and (
                    prior_release_row.get("state") in {"STAGED", "RELEASED"}
                ):
                    if prior_release_row.get("state") == "STAGED":
                        assert progress_callback is not None
                        progress_callback(
                            {
                                "event": "RELEASED",
                                "path": rel,
                                "bytes": int(row.get("bytes", 0)),
                            }
                        )
                    recovered_released_rows.append(dict(row))
                else:
                    already_absent.append(rel)
                continue
            path = Path(f"/proc/self/fd/{candidate_parent_fd}") / candidate_name
        else:
            if unresolved.is_symlink():
                raise RuntimeError(f"refusing to delete symlink: {unresolved}")
            if staged_path is not None and staged_path.is_symlink():
                raise RuntimeError(f"refusing staged symlink: {staged_path}")
            source_exists = unresolved.exists()
            staged_exists = staged_path is not None and staged_path.exists()
            if source_exists and staged_exists:
                raise RuntimeError(
                    f"both source and interrupted staging artifact exist: {rel}"
                )
            already_staged = bool(staged_exists and not source_exists)
            candidate = staged_path if already_staged else unresolved
            try:
                path = candidate.resolve(strict=True)  # type: ignore[union-attr]
            except FileNotFoundError:
                already_absent.append(rel)
                continue
        if source_exists and staged_exists:
            raise RuntimeError(
                f"both source and interrupted staging artifact exist: {rel}"
            )
        if candidate_parent_fd is None:
            try:
                path.relative_to(root.resolve())
            except ValueError:
                raise RuntimeError(f"refusing to delete outside root: {path}")
        authority_row = manifest_rows.get(rel)
        if authority_row is None:
            raise ArtifactLifecycleError(
                f"selected path is absent from disposition authority: {rel}"
            )
        observed = os.lstat(path)
        if stat.S_ISLNK(observed.st_mode):
            raise RuntimeError(f"refusing to delete symlink: {path}")
        identity = (
            int(observed.st_dev),
            int(observed.st_ino),
            int(observed.st_size),
            int(observed.st_mtime_ns),
        )
        if already_staged:
            prior_identity = (
                prior_release_row.get("identity")
                if isinstance(prior_release_row, Mapping)
                else None
            )
            stable_identity_matches = (
                isinstance(prior_identity, list)
                and len(prior_identity) == 4
                and (
                    prior_identity[:2] == list(identity)[:2]
                    if str(manifest_rows.get(rel, {}).get("artifact_type") or "file")
                    == "sealed_tree"
                    else prior_identity == list(identity)
                )
            )
            if (
                not isinstance(prior_release_row, Mapping)
                or prior_release_row.get("state") not in {"PREPARED", "STAGED"}
                or prior_release_row.get("staging_path") != str(staged_path)
                or not stable_identity_matches
            ):
                raise ArtifactLifecycleError(
                    f"staged release lacks authenticated journal authority: {rel}"
                )
        expected_bytes = int(authority_row.get("bytes", -1))
        artifact_type = str(authority_row.get("artifact_type") or "file")
        sealed_tree_members: list[dict[str, Any]] | None = None
        if stat.S_ISREG(observed.st_mode):
            if observed.st_nlink != 1:
                raise RuntimeError(f"refusing to delete hard-linked artifact: {path}")
            if artifact_type == "sealed_tree" or expected_bytes != observed.st_size:
                raise RuntimeError(f"artifact size/type differs from manifest: {rel}")
            expected_sha256 = authority_row.get("sha256")
            if not isinstance(expected_sha256, str):
                raise RuntimeError(f"artifact lacks file SHA-256 authority: {rel}")
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected_sha256:
                raise RuntimeError(f"artifact SHA-256 differs from manifest: {rel}")
        elif stat.S_ISDIR(observed.st_mode):
            if artifact_type != "sealed_tree":
                raise RuntimeError(f"directory lacks sealed-tree authority: {rel}")
            sealed_tree_members = _sealed_tree_member_inventory(path)
            try:
                _validate_sealed_tree_member_authority(
                    sealed_tree_members, authority_row
                )
            except RuntimeError:
                prior_members = (
                    prior_release_row.get("members")
                    if isinstance(prior_release_row, Mapping)
                    else None
                )
                if (
                    not already_staged
                    or not isinstance(prior_members, list)
                    or any(not isinstance(member, Mapping) for member in prior_members)
                ):
                    raise RuntimeError(
                        f"sealed tree SHA-256 differs from manifest: {rel}"
                    )
                sealed_tree_members = [dict(member) for member in prior_members]
                _validate_sealed_tree_member_authority(
                    sealed_tree_members, authority_row
                )
        else:
            raise RuntimeError(f"unsupported artifact type for deletion: {rel}")
        after_validation = os.lstat(path)
        if (
            after_validation.st_dev,
            after_validation.st_ino,
            after_validation.st_size,
            after_validation.st_mtime_ns,
        ) != identity:
            raise RuntimeError(f"artifact changed while validating deletion: {rel}")
        validated_targets.append(
            (
                path,
                dict(row),
                identity,
                staged_path,
                already_staged,
                candidate_parent_fd,
                candidate_name,
                staged_name,
                authority_row,
                sealed_tree_members,
            )
        )

    deleted: list[str] = [str(row["path"]) for row in recovered_released_rows]
    deleted_rows: list[dict[str, Any]] = list(recovered_released_rows)
    failures: list[dict[str, str]] = []
    recovered_staged: list[str] = []
    for (
        path,
        row,
        identity,
        staged_path,
        already_staged,
        candidate_parent_fd,
        candidate_name,
        staged_name,
        authority_row,
        sealed_tree_members,
    ) in validated_targets:
        rel = str(row["path"])
        prior_release_row = (
            release_progress.get(rel) if isinstance(release_progress, Mapping) else None
        )
        try:
            current = (
                os.stat(
                    candidate_name,
                    dir_fd=candidate_parent_fd,
                    follow_symlinks=False,
                )
                if candidate_parent_fd is not None and candidate_name is not None
                else os.lstat(path)
            )
            if _stat_identity(current) != identity:
                raise RuntimeError("artifact identity changed after validation")
            release_target = path
            if (
                staging_fd is not None
                and staging_base is not None
                and staging_identity is not None
                and candidate_parent_fd is not None
                and candidate_name is not None
                and staged_name is not None
            ):
                if current_disposition_manifest_path is not None:
                    _revalidate_current_disposition_manifest(
                        current_disposition_manifest_path,
                        expected_manifest=validated_manifest,
                        runtime_root=root,
                    )
                if RunStorageGuard.run_root_is_reserved(root) and _active_run_guard is None:
                    raise ArtifactLifecycleError(
                        "run became reserved during the release transaction"
                    )
                if RunStorageGuard.run_root_is_reserved(root):
                    _active_run_guard.validate_active_release_owner_v1(
                        run_root=root,
                        lifecycle_contract_sha256=str(
                            validated_manifest.get("lifecycle_contract_sha256")
                        ),
                        reservation_lock_held=True,
                    )
                current_staging_path = os.lstat(staging_base)
                if (
                    stat.S_ISLNK(current_staging_path.st_mode)
                    or (
                        int(current_staging_path.st_dev),
                        int(current_staging_path.st_ino),
                    )
                    != staging_identity[:2]
                ):
                    raise RuntimeError(
                        "release staging parent changed before destructive staging"
                    )
                if not already_staged:
                    assert progress_callback is not None
                    if (
                        diagnostic_release_requires_live_recheck
                        and isinstance(prior_release_row, Mapping)
                        and prior_release_row.get("state") in {"STAGED", "RELEASED"}
                    ):
                        # A source that reappears after an authenticated destructive
                        # journal state is not journal-only recovery.  Audit it before
                        # the monotonic journal rejects any attempted PREPARED replay.
                        require_diagnostic_local_live_state_clear_v1(root)
                    if sealed_tree_members is not None:
                        _make_sealed_tree_directories_removable(
                            path,
                            [
                                path / str(member["path"])
                                for member in sealed_tree_members
                                if member.get("type") == "directory"
                            ],
                        )
                    prepared_event: dict[str, Any] = {
                        "event": "PREPARED",
                        "path": rel,
                        "staging_path": str(staged_path),
                        "identity": list(identity),
                    }
                    if sealed_tree_members is not None:
                        prepared_event["members"] = sealed_tree_members
                    progress_callback(prepared_event)
                    if diagnostic_release_requires_live_recheck:
                        require_diagnostic_local_live_state_clear_v1(root)
                    _rename_noreplace(
                        candidate_name,
                        staged_name,
                        source_dir_fd=candidate_parent_fd,
                        destination_dir_fd=staging_fd,
                    )
                staged_observed = os.stat(
                    staged_name, dir_fd=staging_fd, follow_symlinks=False
                )
                if _stat_identity(staged_observed) != identity:
                    raise RuntimeError(
                        "artifact identity changed during atomic release staging"
                    )
                os.fsync(candidate_parent_fd)
                if staging_fd != candidate_parent_fd:
                    os.fsync(staging_fd)
                release_target = Path(f"/proc/self/fd/{staging_fd}") / staged_name
                if not already_staged:
                    assert progress_callback is not None
                    staged_event: dict[str, Any] = {
                        "event": "STAGED",
                        "path": rel,
                        "staging_path": str(staged_path),
                        "identity": list(identity),
                    }
                    if sealed_tree_members is not None:
                        staged_event["members"] = sealed_tree_members
                    progress_callback(staged_event)
                else:
                    recovered_staged.append(rel)
                    assert progress_callback is not None
                    staged_event = {
                        "event": "RECOVERED_STAGED",
                        "path": rel,
                        "staging_path": str(staged_path),
                        "identity": list(identity),
                    }
                    if sealed_tree_members is not None:
                        staged_event["members"] = sealed_tree_members
                    progress_callback(staged_event)
            elif staged_path is not None:
                release_target = staged_path
                recovered_staged.append(rel)
                assert progress_callback is not None
                staged_event = {
                    "event": "RECOVERED_STAGED",
                    "path": rel,
                    "staging_path": str(staged_path),
                    "identity": list(identity),
                }
                if sealed_tree_members is not None:
                    staged_event["members"] = sealed_tree_members
                progress_callback(staged_event)
            if RunStorageGuard.run_root_is_reserved(root) and _active_run_guard is None:
                raise ArtifactLifecycleError(
                    "run became reserved before physical release"
                )
            if RunStorageGuard.run_root_is_reserved(root):
                _active_run_guard.validate_active_release_owner_v1(
                    run_root=root,
                    lifecycle_contract_sha256=str(
                        validated_manifest.get("lifecycle_contract_sha256")
                    ),
                    reservation_lock_held=True,
                )
            if current_disposition_manifest_path is not None:
                _revalidate_current_disposition_manifest(
                    current_disposition_manifest_path,
                    expected_manifest=validated_manifest,
                    runtime_root=root,
                )
            if (
                diagnostic_release_requires_live_recheck
                and not already_staged
                and staging_fd is None
            ):
                require_diagnostic_local_live_state_clear_v1(root)
            release_observed = os.lstat(release_target)
            if _stat_identity(release_observed) != identity:
                raise RuntimeError("staged artifact identity changed before release")
            if stat.S_ISDIR(release_observed.st_mode):
                if sealed_tree_members is None:
                    raise RuntimeError(
                        "sealed tree release lacks exact member authority"
                    )
                _remove_sealed_tree_remainder(
                    release_target,
                    expected_members=sealed_tree_members,
                )
                if staging_fd is not None and staged_name is not None:
                    os.rmdir(staged_name, dir_fd=staging_fd)
                    os.fsync(staging_fd)
                else:
                    release_target.rmdir()
                    _fsync_directory(release_target.parent)
            else:
                if staging_fd is not None and staged_name is not None:
                    os.unlink(staged_name, dir_fd=staging_fd)
                    os.fsync(staging_fd)
                else:
                    release_target.unlink()
                    _fsync_directory(release_target.parent)
        except (ArtifactLifecycleError, OSError, RuntimeError) as exc:
            failures.append(
                {"path": rel, "type": type(exc).__name__, "message": str(exc)}
            )
            continue
        deleted.append(rel)
        deleted_rows.append(dict(row))
        if progress_callback is not None:
            try:
                progress_callback(
                    {
                        "event": "RELEASED",
                        "path": rel,
                        "bytes": int(row.get("bytes", 0)),
                    }
                )
            except (OSError, RuntimeError) as exc:
                failures.append(
                    {
                        "path": rel,
                        "type": "ProgressPersistenceError",
                        "message": f"{type(exc).__name__}: {exc}",
                    }
                )
    for target in validated_targets:
        if target[5] is not None:
            os.close(target[5])
    if staging_fd is not None:
        os.close(staging_fd)
    if root_fd is not None:
        os.close(root_fd)
    fcntl.flock(release_fence.fileno(), fcntl.LOCK_UN)
    release_fence.close()
    return {
        "deleted_count": len(deleted),
        "deleted": deleted,
        "deleted_rows": deleted_rows,
        "already_absent": already_absent,
        "recovered_staged": recovered_staged,
        "recovered_released": [str(row["path"]) for row in recovered_released_rows],
        "failures": failures,
        "authorization": authorization,
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pretty_json(value), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dry-run-first EIEL runtime storage pruning helper"
    )
    parser.add_argument("root", help="Runtime campaign/run root to inspect")
    parser.add_argument(
        "--category",
        action="append",
        choices=CATEGORIES,
        default=[],
        help="Prune category to select. Repeatable. Defaults to all supported categories.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete selected files/directories",
    )
    parser.add_argument(
        "--archive-to",
        type=Path,
        default=Path(
            os.environ.get(
                "EIEL_COLD_STORAGE_ROOT",
                "/mnt/Projects/_Emergence-cold-storage/eiel-artifacts",
            )
        ),
        help="Cold-storage root for durable R3-R5 material",
    )
    parser.add_argument(
        "--archive-apply",
        action="store_true",
        help="Atomically copy and verify exact durable R3-R5 rows; release eligible R3 sources",
    )
    parser.add_argument(
        "--retain-source-after-archive",
        action="store_true",
        help="Keep eligible R3 source bytes after a verified archive copy",
    )
    parser.add_argument(
        "--out", default="", help="Optional JSON plan/result output path"
    )
    parser.add_argument(
        "--retention-manifest",
        type=Path,
        help="Decision-bearing evidence retention manifest used to block protected deletion",
    )
    parser.add_argument(
        "--disposition-manifest",
        type=Path,
        help="Lifecycle ArtifactDispositionManifestV1 authorizing release eligibility",
    )
    parser.add_argument(
        "--protected-attempt-out",
        type=Path,
        help="Optional protected prune attempt receipt path",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.apply and args.archive_apply:
        print(
            json.dumps(
                {
                    "ok": False,
                    "mode": "blocked",
                    "error": "--apply and --archive-apply are mutually exclusive",
                },
                indent=2,
            )
            + "\n",
            end="",
        )
        return 1
    root = Path(args.root).expanduser().resolve()
    categories = set(args.category or CATEGORIES)
    disposition_manifest: dict[str, Any] | None = None
    if args.disposition_manifest is not None:
        try:
            disposition_manifest = validate_artifact_disposition_manifest_v1(
                json.loads(
                    args.disposition_manifest.expanduser()
                    .resolve()
                    .read_text(encoding="utf-8")
                )
            )
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            ArtifactLifecycleError,
        ) as exc:
            result = {
                "ok": False,
                "mode": "blocked",
                "error": f"invalid disposition manifest: {type(exc).__name__}: {exc}",
            }
            print(json.dumps(result, indent=2, sort_keys=True) + "\n", end="")
            return 1
    plan = build_prune_plan(root, categories, disposition_manifest)
    protection_receipt: dict[str, Any] | None = None
    if args.retention_manifest is not None:
        try:
            protection_receipt = guard_prune_plan(
                root=root,
                plan=plan,
                manifest=load_manifest(args.retention_manifest),
            )
        except DecisionEvidenceRetentionError as exc:
            protection_receipt = dict(getattr(exc, "receipt", {}) or {})
            if not protection_receipt:
                protection_receipt = {
                    "schema_version": "protected_prune_attempt_receipt_v1",
                    "runtime_root": str(root),
                    "decision": "blocked",
                }
            protection_receipt["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            if args.protected_attempt_out is not None:
                _write_json(
                    args.protected_attempt_out.expanduser().resolve(),
                    protection_receipt,
                )
            result = {
                "ok": False,
                "mode": "blocked",
                "plan": plan,
                "protection": protection_receipt,
            }
            text = json.dumps(result, indent=2, sort_keys=True) + "\n"
            if str(args.out or "").strip():
                Path(args.out).expanduser().resolve().write_text(text, encoding="utf-8")
            print(text, end="")
            return 1
        if args.protected_attempt_out is not None:
            _write_json(
                args.protected_attempt_out.expanduser().resolve(), protection_receipt
            )
    result: dict[str, Any] = {
        "mode": "dry_run",
        "plan": plan,
        "summary": {
            "produced_bytes": sum(
                int(row.get("bytes", 0)) for row in plan.get("selected", [])
            ),
            "release_eligible_bytes": int(plan.get("selected_bytes", 0)),
            "actually_deleted": 0,
        },
    }
    if args.archive_to is not None:
        try:
            archive_plan = (
                build_archive_plan(
                    root,
                    disposition_manifest,
                    archive_root=args.archive_to.expanduser().resolve(),
                )
                if disposition_manifest is not None
                else None
            )
        except (ArtifactLifecycleError, RuntimeError) as exc:
            result = {
                "ok": False,
                "mode": "blocked",
                "plan": plan,
                "error": f"archive plan blocked: {type(exc).__name__}: {exc}",
            }
            text = json.dumps(result, indent=2, sort_keys=True) + "\n"
            if str(args.out or "").strip():
                Path(args.out).expanduser().resolve().write_text(text, encoding="utf-8")
            print(text, end="")
            return 1
        if archive_plan is not None:
            result["archive_plan"] = archive_plan
    if args.archive_apply:
        if disposition_manifest is None or result.get("archive_plan") is None:
            result = {
                "ok": False,
                "mode": "blocked",
                "plan": plan,
                "error": "durable archival requires an explicit lifecycle disposition manifest",
            }
            text = json.dumps(result, indent=2, sort_keys=True) + "\n"
            if str(args.out or "").strip():
                Path(args.out).expanduser().resolve().write_text(text, encoding="utf-8")
            print(text, end="")
            return 1
        try:
            archive_result = apply_archive_plan(
                root,
                result["archive_plan"],
                disposition_manifest=disposition_manifest,
                release_after_archive=not args.retain_source_after_archive,
                current_disposition_manifest_path=(
                    args.disposition_manifest.expanduser().resolve()
                    if args.disposition_manifest is not None
                    else None
                ),
            )
        except (ArtifactLifecycleError, RuntimeError) as exc:
            result = {
                "ok": False,
                "mode": "blocked",
                "plan": plan,
                "archive_plan": result.get("archive_plan"),
                "error": f"durable archival blocked: {type(exc).__name__}: {exc}",
            }
            text = json.dumps(result, indent=2, sort_keys=True) + "\n"
            if str(args.out or "").strip():
                Path(args.out).expanduser().resolve().write_text(text, encoding="utf-8")
            print(text, end="")
            return 1
        archived_rows = list(archive_result.get("archived", []))
        released_bytes = int(archive_result.get("released_bytes", 0))
        receipt = build_storage_disposition_receipt_v1(
            manifest=disposition_manifest,
            bytes_before=int(disposition_manifest["summary"]["produced_bytes"]),
            bytes_retained=int(disposition_manifest["summary"]["produced_bytes"])
            - released_bytes,
            bytes_released=released_bytes,
            bytes_unresolved_rx=int(
                disposition_manifest["summary"]["unresolved_rx_bytes"]
            ),
            archived=archived_rows,
            action_failures=archive_result.get("failures", []),
            skipped_paths=archive_result.get("already_archived", []),
            authorization_reason="verified durable archive admission from lifecycle disposition",
            replacement_references=disposition_manifest.get(
                "replacement_artifacts", []
            ),
            retention_references=[str(args.disposition_manifest)],
            plan_sha256=disposition_manifest.get("plan_sha256"),
            timestamp_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            result=(
                "archive_partial_failure"
                if archive_result.get("failures")
                else "archive_applied"
            ),
        )
        receipt_path = root / "storage_disposition_receipt.json"
        write_storage_disposition_receipt_v1(receipt_path, receipt)
        result = {
            **result,
            "ok": not bool(archive_result.get("failures")),
            "mode": "archive_apply",
            "archive_result": archive_result,
            "storage_disposition_receipt_path": str(receipt_path),
        }
    if bool(args.apply):
        if (
            disposition_manifest is None
            or plan.get("automatic_prune_authorized") is not True
        ):
            result = {
                "ok": False,
                "mode": "blocked",
                "plan": plan,
                "error": "destructive cleanup requires explicit lifecycle disposition authority",
            }
            text = json.dumps(result, indent=2, sort_keys=True) + "\n"
            if str(args.out or "").strip():
                Path(args.out).expanduser().resolve().write_text(text, encoding="utf-8")
            print(text, end="")
            return 1
        try:
            delete_result = apply_prune_plan(
                root,
                plan,
                disposition_manifest=disposition_manifest,
            )
        except (ArtifactLifecycleError, RuntimeError) as exc:
            result = {
                "ok": False,
                "mode": "blocked",
                "plan": plan,
                "error": f"destructive cleanup blocked: {type(exc).__name__}: {exc}",
            }
            text = json.dumps(result, indent=2, sort_keys=True) + "\n"
            if str(args.out or "").strip():
                Path(args.out).expanduser().resolve().write_text(text, encoding="utf-8")
            print(text, end="")
            return 1
        deleted_rows = list(delete_result.get("deleted_rows", []))
        deleted_bytes = sum(int(row.get("bytes", 0)) for row in deleted_rows)
        receipt = build_storage_disposition_receipt_v1(
            manifest=disposition_manifest,
            bytes_before=int(disposition_manifest["summary"]["produced_bytes"]),
            bytes_retained=int(disposition_manifest["summary"]["produced_bytes"])
            - deleted_bytes,
            bytes_released=deleted_bytes,
            bytes_unresolved_rx=int(
                disposition_manifest["summary"]["unresolved_rx_bytes"]
            ),
            removed=deleted_rows,
            action_failures=delete_result.get("failures", []),
            skipped_paths=delete_result.get("already_absent", []),
            authorization_reason="explicit lifecycle disposition release authority",
            replacement_references=disposition_manifest.get(
                "replacement_artifacts", []
            ),
            retention_references=[str(args.disposition_manifest)],
            plan_sha256=disposition_manifest.get("plan_sha256"),
            timestamp_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            result=("partial_failure" if delete_result.get("failures") else "applied"),
        )
        receipt_path = root / "storage_disposition_receipt.json"
        write_storage_disposition_receipt_v1(receipt_path, receipt)
        result = {
            "ok": not bool(delete_result.get("failures")),
            "mode": "apply",
            "plan": plan,
            "delete_result": delete_result,
            "storage_disposition_receipt_path": str(receipt_path),
        }
    if protection_receipt is not None:
        result["protection"] = protection_receipt
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if str(args.out or "").strip():
        Path(args.out).expanduser().resolve().write_text(text, encoding="utf-8")
    print(text, end="")
    return 1 if result.get("ok") is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
