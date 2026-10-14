"""Fail-closed terminal retirement for exact lifecycle-controlled roots.

This is the final actuator in the existing run-artifact lifecycle.  It accepts
only explicitly enumerated roots, seals every member during a dry run, and
writes all authority outside the roots that may be removed.  It does not
discover storage, infer ownership from names or ages, or archive material.
"""

from __future__ import annotations

from engine.runtime_roots import public_runtime_path

from collections.abc import Mapping, Sequence
from contextlib import ExitStack
import ctypes
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import os
from pathlib import Path
import stat
from typing import Any, Final

from .artifact_lifecycle import (
    ArtifactLifecycleError,
    RunStorageGuard,
    run_artifact_transaction_fence_v1,
)
from .canonical import canonical_sha256, pretty_json


REQUEST_SCHEMA_VERSION: Final[str] = "eiel_runtime_root_retirement_request_v1"
INVENTORY_SCHEMA_VERSION: Final[str] = "eiel_runtime_root_retirement_inventory_v1"
INTENT_SCHEMA_VERSION: Final[str] = "eiel_runtime_root_retirement_intent_v1"
RECEIPT_SCHEMA_VERSION: Final[str] = "eiel_runtime_root_retirement_receipt_v1"
REVIEW_SCHEMA_VERSION: Final[str] = "eiel_runtime_root_retirement_review_v1"

ROOT_KINDS: Final[frozenset[str]] = frozenset(
    {"run_artifact", "canopy_relay", "build_artifact", "handoff", "aggregate"}
)
CANONICAL_RUNTIME_ROOTS: Final[frozenset[Path]] = frozenset(
    {
        Path(public_runtime_path('')).resolve(strict=False),
        (Path.home() / ".local" / "share" / "emergence").resolve(strict=False),
    }
)
CANONICAL_MANAGEMENT_ROOTS: Final[frozenset[Path]] = frozenset(
    {
        Path(public_runtime_path('lifecycle')).resolve(strict=False),
        Path(public_runtime_path(".permanent-evidence")).resolve(strict=False),
    }
)
MANAGEMENT_PATH_COMPONENTS: Final[frozenset[str]] = frozenset(
    {
        ".eiel-lifecycle-pending",
        ".eiel-storage-budgets",
        ".eiel-storage-reservations",
        ".eiel-storage-status",
    }
)
SANDBOXED_NON_RUNTIME_PROCESS_NAMES: Final[frozenset[str]] = frozenset(
    {
        "systemd",
        "(sd-pam)",
        "ssh-agent",
        "sd_espeak-ng-mb",
        "chrome-sandbox",
        "Proton Pass",
    }
)
PROC_ROOT: Final[Path] = Path("/proc")
REQUIRED_PREDICATES: Final[tuple[str, ...]] = (
    "terminal_cleanup_authorized",
    "no_future_consumers",
    "audit_hold_released_or_not_required",
    "qualification_hold_released_or_not_required",
    "unknown_artifacts_resolved",
)


class RootRetirementError(ArtifactLifecycleError):
    """An exact runtime root cannot be safely retired."""


def _rename_noreplace(
    source: str,
    target: str,
    *,
    src_dir_fd: int,
    dst_dir_fd: int,
) -> None:
    """Atomically rename without ever replacing an unsealed target name."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RootRetirementError("renameat2(RENAME_NOREPLACE) is unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        src_dir_fd,
        os.fsencode(source),
        dst_dir_fd,
        os.fsencode(target),
        1,
    )
    if result != 0:
        observed_errno = ctypes.get_errno()
        if observed_errno == errno.EEXIST:
            raise RootRetirementError(
                f"atomic retirement stage already exists: {target}"
            )
        raise OSError(observed_errno, os.strerror(observed_errno), source, target)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(pretty_json(value), encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return path


def _mapping(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RootRetirementError(f"{path} must be an object")
    return dict(value)


def _text(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RootRetirementError(f"{path} must be a non-empty string")
    return value.strip()


def _sha256(value: Any, *, path: str) -> str:
    result = _text(value, path=path)
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise RootRetirementError(f"{path} must be a lowercase SHA-256")
    return result


def _absolute_path(value: Any, *, path: str) -> Path:
    result = Path(_text(value, path=path)).expanduser()
    if not result.is_absolute():
        raise RootRetirementError(f"{path} must be absolute")
    if result.is_symlink():
        raise RootRetirementError(f"{path} cannot be a symlink")
    return result.resolve(strict=False)


def _validate_runtime_parent(parent: Path, *, path: str) -> None:
    """Confine terminal retirement to an exact causyl-runtime namespace."""

    if parent not in CANONICAL_RUNTIME_ROOTS:
        raise RootRetirementError(
            f"{path} must identify a canonical causyl-runtime namespace root"
        )


def _validate_not_management_root(root: Path, *, path: str) -> None:
    for management_root in CANONICAL_MANAGEMENT_ROOTS:
        if (
            root == management_root
            or management_root in root.parents
            or root in management_root.parents
        ):
            raise RootRetirementError(
                f"{path} cannot target or contain a storage management root"
            )
    if any(component in MANAGEMENT_PATH_COMPONENTS for component in root.parts):
        raise RootRetirementError(f"{path} cannot target a storage management root")


def _validate_hash_reference(value: Any, *, path: str) -> dict[str, str]:
    row = _mapping(value, path=path)
    if set(row) != {"path", "sha256"}:
        raise RootRetirementError(f"{path} must contain only path and sha256")
    return {
        "path": str(_absolute_path(row["path"], path=f"{path}.path")),
        "sha256": _sha256(row["sha256"], path=f"{path}.sha256"),
    }


def validate_root_retirement_request_v1(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one explicit terminal root-retirement request."""

    request = _mapping(value, path="request")
    allowed_fields = {
        "schema_version",
        "retirement_id",
        "programme_id",
        "work_id",
        "sealed_at_utc",
        "source",
        "allowed_runtime_roots",
        "compact_authority",
        "terminal_authorities",
        "validation",
        "roots",
        "claim_limits",
    }
    unknown = sorted(set(request) - allowed_fields)
    if unknown:
        raise RootRetirementError(f"request contains unsupported fields: {unknown}")
    if request.get("schema_version") != REQUEST_SCHEMA_VERSION:
        raise RootRetirementError("root retirement request schema mismatch")
    source = _mapping(request.get("source"), path="request.source")
    commit_sha = _text(source.get("commit_sha"), path="request.source.commit_sha")
    if len(commit_sha) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in commit_sha
    ):
        raise RootRetirementError(
            "request.source.commit_sha is not a Git/content identity"
        )
    parents_raw = request.get("allowed_runtime_roots")
    if not isinstance(parents_raw, list) or not parents_raw:
        raise RootRetirementError("allowed_runtime_roots must be a non-empty array")
    parents = [
        _absolute_path(item, path=f"request.allowed_runtime_roots[{index}]")
        for index, item in enumerate(parents_raw)
    ]
    for index, parent in enumerate(parents):
        _validate_runtime_parent(parent, path=f"request.allowed_runtime_roots[{index}]")
    if len({str(path) for path in parents}) != len(parents):
        raise RootRetirementError("allowed_runtime_roots must not contain duplicates")
    authority = _validate_hash_reference(
        request.get("compact_authority"), path="request.compact_authority"
    )
    terminal_raw = request.get("terminal_authorities")
    if not isinstance(terminal_raw, list) or not terminal_raw:
        raise RootRetirementError("terminal_authorities must be a non-empty array")
    terminal = [
        _validate_hash_reference(row, path=f"request.terminal_authorities[{index}]")
        for index, row in enumerate(terminal_raw)
    ]
    validation = _mapping(request.get("validation"), path="request.validation")
    predicates = _mapping(
        validation.get("predicates"), path="request.validation.predicates"
    )
    if set(predicates) != set(REQUIRED_PREDICATES) or any(
        predicates[name] is not True for name in REQUIRED_PREDICATES
    ):
        raise RootRetirementError(
            "all exact terminal retirement predicates must be true"
        )
    review = _mapping(
        validation.get("semantic_review"), path="request.validation.semantic_review"
    )
    if review.get("status") not in {"accepted", "accepted_with_limitations"}:
        raise RootRetirementError("semantic root-retirement review is not accepted")
    reviewer = _text(
        review.get("reviewer"), path="request.validation.semantic_review.reviewer"
    )
    roots_raw = request.get("roots")
    if not isinstance(roots_raw, list) or not roots_raw:
        raise RootRetirementError("roots must be a non-empty array")
    roots: list[dict[str, Any]] = []
    for index, raw_value in enumerate(roots_raw):
        raw = _mapping(raw_value, path=f"request.roots[{index}]")
        required = {
            "root_id",
            "path",
            "kind",
            "terminal_status",
            "unresolved_rx_bytes",
            "live_reference_count",
            "authority_receipts",
        }
        if set(raw) != required:
            raise RootRetirementError(
                f"request.roots[{index}] fields differ from the exact contract"
            )
        root = _absolute_path(raw["path"], path=f"request.roots[{index}].path")
        _validate_not_management_root(root, path=f"request.roots[{index}].path")
        if not any(root != parent and parent in root.parents for parent in parents):
            raise RootRetirementError(
                f"request.roots[{index}].path is not a strict child of an allowed runtime root"
            )
        kind = _text(raw["kind"], path=f"request.roots[{index}].kind")
        if kind not in ROOT_KINDS:
            raise RootRetirementError(f"request.roots[{index}].kind is unsupported")
        if raw["terminal_status"] != "completed":
            raise RootRetirementError(
                "root retirement requires completed terminal status"
            )
        if raw["unresolved_rx_bytes"] != 0:
            raise RootRetirementError("unresolved RX blocks root retirement")
        if raw["live_reference_count"] != 0:
            raise RootRetirementError("live references block root retirement")
        receipts_raw = raw["authority_receipts"]
        if not isinstance(receipts_raw, list) or not receipts_raw:
            raise RootRetirementError("every root requires exact authority receipts")
        receipts = [
            _validate_hash_reference(
                row, path=f"request.roots[{index}].authority_receipts[{receipt_index}]"
            )
            for receipt_index, row in enumerate(receipts_raw)
        ]
        roots.append(
            {
                "root_id": _text(
                    raw["root_id"], path=f"request.roots[{index}].root_id"
                ),
                "path": str(root),
                "kind": kind,
                "terminal_status": "completed",
                "unresolved_rx_bytes": 0,
                "live_reference_count": 0,
                "authority_receipts": receipts,
            }
        )
    root_paths = [Path(row["path"]) for row in roots]
    if len({str(path) for path in root_paths}) != len(root_paths):
        raise RootRetirementError("retirement roots must not contain duplicates")
    root_ids = [str(row["root_id"]) for row in roots]
    if len(set(root_ids)) != len(root_ids):
        raise RootRetirementError("retirement root IDs must not contain duplicates")
    for index, root in enumerate(root_paths):
        if any(
            other in root.parents or root in other.parents
            for other_index, other in enumerate(root_paths)
            if other_index != index
        ):
            raise RootRetirementError(
                "one request cannot contain overlapping retirement roots"
            )
    authority_paths = [
        Path(authority["path"]),
        *(Path(row["path"]) for row in terminal),
        *(
            Path(reference["path"])
            for root_row in roots
            for reference in root_row["authority_receipts"]
        ),
    ]
    for authority_path in authority_paths:
        if any(
            authority_path == root or root in authority_path.parents
            for root in root_paths
        ):
            raise RootRetirementError(
                "root-retirement authority must be external to every target root"
            )
    limits = request.get("claim_limits")
    if (
        not isinstance(limits, list)
        or not limits
        or not all(isinstance(item, str) and item for item in limits)
    ):
        raise RootRetirementError("claim_limits must be a non-empty array of strings")
    return {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "retirement_id": _text(
            request.get("retirement_id"), path="request.retirement_id"
        ),
        "programme_id": _text(request.get("programme_id"), path="request.programme_id"),
        "work_id": _text(request.get("work_id"), path="request.work_id"),
        "sealed_at_utc": _text(
            request.get("sealed_at_utc"), path="request.sealed_at_utc"
        ),
        "source": {
            "repository": _text(
                source.get("repository"), path="request.source.repository"
            ),
            "commit_sha": commit_sha,
        },
        "allowed_runtime_roots": [str(path) for path in parents],
        "compact_authority": authority,
        "terminal_authorities": terminal,
        "validation": {
            "predicates": {name: True for name in REQUIRED_PREDICATES},
            "semantic_review": {"status": review["status"], "reviewer": reviewer},
        },
        "roots": roots,
        "claim_limits": list(limits),
    }


def _file_sha256_fd(fd: int) -> str:
    digest = hashlib.sha256()
    os.lseek(fd, 0, os.SEEK_SET)
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def _inventory_directory_fd(fd: int, *, prefix: str = "") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with os.scandir(fd) as entries:
        names = sorted((entry.name for entry in entries), key=os.fsencode)
    for name in names:
        if name in {".", ".."} or "/" in name:
            raise RootRetirementError("unsafe directory member encountered")
        relative = f"{prefix}/{name}" if prefix else name
        observed = os.stat(name, dir_fd=fd, follow_symlinks=False)
        if stat.S_ISLNK(observed.st_mode):
            raise RootRetirementError(
                f"retirement inventory contains a symlink: {relative}"
            )
        if stat.S_ISDIR(observed.st_mode):
            child_fd = os.open(
                name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
            )
            try:
                rows.append(
                    {
                        "path": relative,
                        "type": "directory",
                        "device": int(observed.st_dev),
                        "inode": int(observed.st_ino),
                    }
                )
                rows.extend(_inventory_directory_fd(child_fd, prefix=relative))
            finally:
                os.close(child_fd)
            continue
        if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
            raise RootRetirementError(
                f"retirement inventory contains a non-regular or hard-linked member: {relative}"
            )
        file_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
        try:
            before = os.fstat(file_fd)
            if before.st_nlink != 1 or not stat.S_ISREG(before.st_mode):
                raise RootRetirementError(
                    f"member identity changed while opening: {relative}"
                )
            digest = _file_sha256_fd(file_fd)
            after = os.fstat(file_fd)
            identity_before = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            )
            identity_after = (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            )
            if identity_before != identity_after:
                raise RootRetirementError(f"member changed while hashing: {relative}")
            rows.append(
                {
                    "path": relative,
                    "type": "file",
                    "bytes": int(before.st_size),
                    "sha256": digest,
                    "device": int(before.st_dev),
                    "inode": int(before.st_ino),
                }
            )
        finally:
            os.close(file_fd)
    return rows


def inventory_retirement_root_v1(root: Path) -> dict[str, Any]:
    unresolved = root.expanduser()
    if unresolved.is_symlink():
        raise RootRetirementError(f"retirement root cannot be a symlink: {root}")
    absolute = Path(os.path.abspath(os.fspath(unresolved)))
    root_fd = os.open(absolute, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        return _inventory_retirement_root_fd_v1(root_fd, absolute=absolute)
    finally:
        os.close(root_fd)


def _inventory_retirement_root_fd_v1(fd: int, *, absolute: Path) -> dict[str, Any]:
    observed = os.fstat(fd)
    rows = _inventory_directory_fd(fd)
    after = os.fstat(fd)
    if (observed.st_dev, observed.st_ino) != (after.st_dev, after.st_ino):
        raise RootRetirementError(f"retirement root identity changed: {absolute}")
    core = {
        "path": str(absolute),
        "device": int(observed.st_dev),
        "inode": int(observed.st_ino),
        "members": rows,
        "member_count": len(rows),
        "file_count": sum(row["type"] == "file" for row in rows),
        "bytes": sum(int(row.get("bytes", 0)) for row in rows),
    }
    return {**core, "inventory_sha256": canonical_sha256(core)}


def _verify_file_reference(reference: Mapping[str, str], *, label: str) -> None:
    path = Path(reference["path"])
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise RootRetirementError(
            f"{label} is missing or cannot be opened without following links: {path}"
        ) from exc
    try:
        observed = os.fstat(fd)
        if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
            raise RootRetirementError(
                f"{label} must be one non-linked regular file: {path}"
            )
        digest = _file_sha256_fd(fd)
        after = os.fstat(fd)
        named = os.stat(path, follow_symlinks=False)
        identity = (
            observed.st_dev,
            observed.st_ino,
            observed.st_size,
            observed.st_mtime_ns,
        )
        if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise RootRetirementError(f"{label} changed while hashing: {path}")
        if (named.st_dev, named.st_ino) != (observed.st_dev, observed.st_ino):
            raise RootRetirementError(
                f"{label} path identity changed while hashing: {path}"
            )
        if digest != reference["sha256"]:
            raise RootRetirementError(f"{label} SHA-256 mismatch: {path}")
    finally:
        os.close(fd)


def _reservation_root_for(root: Path) -> Path:
    for candidate in (root, *root.parents):
        if candidate.name == "causyl-runtime":
            return candidate / "lifecycle" / "reservations"
    return root.parent / ".eiel-storage-reservations"


def _acquire_release_fence(reservation_root: Path) -> Any:
    reservation_root.mkdir(parents=True, exist_ok=True)
    handle = (reservation_root / ".lock").open("a+b")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    return handle


def _process_references(
    root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    references: list[dict[str, Any]] = []
    inspection_errors: list[dict[str, Any]] = []
    root_text = str(root)
    own_pid = os.getpid()
    for pid_path in sorted(
        (path for path in PROC_ROOT.iterdir() if path.name.isdigit()),
        key=lambda path: int(path.name),
    ):
        pid = int(pid_path.name)
        try:
            owner_uid = pid_path.stat().st_uid
        except FileNotFoundError:
            continue
        if owner_uid != os.geteuid():
            continue
        reasons: list[str] = []
        pid_errors: list[dict[str, Any]] = []
        try:
            process_name = (pid_path / "comm").read_text(encoding="utf-8").strip()
        except OSError:
            process_name = ""
        try:
            cwd = os.readlink(pid_path / "cwd")
            if cwd == root_text or cwd.startswith(root_text + os.sep):
                reasons.append("cwd")
        except FileNotFoundError:
            continue
        except PermissionError:
            pid_errors.append({"pid": pid, "probe": "cwd"})
        # The retirement CLI names its target in argv; that is not a live
        # consumer. Keep checking our cwd/fds and every other process's argv.
        if pid != own_pid:
            try:
                command = (pid_path / "cmdline").read_bytes().replace(b"\0", b" ")
                if root_text.encode() in command:
                    reasons.append("cmdline")
            except FileNotFoundError:
                continue
            except PermissionError:
                pid_errors.append({"pid": pid, "probe": "cmdline"})
        fd_root = pid_path / "fd"
        try:
            fd_paths = list(fd_root.iterdir())
        except FileNotFoundError:
            fd_paths = []
        except PermissionError:
            pid_errors.append({"pid": pid, "probe": "fd_directory"})
            fd_paths = []
        for fd_path in fd_paths:
            try:
                target = os.readlink(fd_path)
            except FileNotFoundError:
                continue
            except PermissionError:
                pid_errors.append({"pid": pid, "probe": "fd", "fd": fd_path.name})
                continue
            if target == root_text or target.startswith(root_text + os.sep):
                reasons.append("fd")
                break
        if reasons:
            references.append({"pid": pid, "reasons": sorted(set(reasons))})
        if pid_errors and process_name not in SANDBOXED_NON_RUNTIME_PROCESS_NAMES:
            inspection_errors.extend(pid_errors)
    return references, inspection_errors


def _assert_live_state_clear(root: Path) -> None:
    if RunStorageGuard.run_root_is_reserved(root):
        raise RootRetirementError(
            f"active storage reservation blocks retirement: {root}"
        )
    references, inspection_errors = _process_references(root)
    if inspection_errors:
        raise RootRetirementError(
            f"process reference inspection was not complete: {inspection_errors[:8]}"
        )
    if references:
        raise RootRetirementError(
            f"live process references block retirement: {root}: {references[:8]}"
        )


def build_root_retirement_inventory_v1(request: Mapping[str, Any]) -> dict[str, Any]:
    normalized = validate_root_retirement_request_v1(request)
    _verify_file_reference(normalized["compact_authority"], label="compact authority")
    for index, reference in enumerate(normalized["terminal_authorities"]):
        _verify_file_reference(reference, label=f"terminal authority {index}")
    roots: list[dict[str, Any]] = []
    for row in normalized["roots"]:
        root = Path(row["path"])
        for receipt_index, reference in enumerate(row["authority_receipts"]):
            _verify_file_reference(
                reference, label=f"{row['root_id']} authority receipt {receipt_index}"
            )
        _assert_live_state_clear(root)
        roots.append({"root_id": row["root_id"], **inventory_retirement_root_v1(root)})
    core = {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "retirement_id": normalized["retirement_id"],
        "programme_id": normalized["programme_id"],
        "request_sha256": canonical_sha256(normalized),
        "roots": roots,
        "root_count": len(roots),
        "member_count": sum(int(row["member_count"]) for row in roots),
        "bytes": sum(int(row["bytes"]) for row in roots),
    }
    return {**core, "inventory_sha256": canonical_sha256(core)}


def build_root_retirement_intent_v1(
    *,
    request: Mapping[str, Any],
    inventory: Mapping[str, Any],
    status: str,
    previous_intent_sha256: str | None = None,
    prepared_intent_sha256: str | None = None,
    events: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    normalized = validate_root_retirement_request_v1(request)
    if status not in {"PREPARED", "APPLYING", "APPLIED", "PARTIAL_FAILURE"}:
        raise RootRetirementError("root retirement intent status is unsupported")
    if status == "PREPARED":
        if prepared_intent_sha256 is not None:
            raise RootRetirementError("PREPARED intent cannot bind itself recursively")
    else:
        _sha256(prepared_intent_sha256, path="prepared_intent_sha256")
    core = {
        "schema_version": INTENT_SCHEMA_VERSION,
        "retirement_id": normalized["retirement_id"],
        "request_sha256": canonical_sha256(normalized),
        "inventory_sha256": inventory.get("inventory_sha256"),
        "status": status,
        "previous_intent_sha256": previous_intent_sha256,
        "prepared_intent_sha256": prepared_intent_sha256,
        "events": [dict(row) for row in events],
    }
    return {**core, "intent_sha256": canonical_sha256(core)}


def validate_root_retirement_inventory_v1(value: Mapping[str, Any]) -> dict[str, Any]:
    raw = dict(_mapping(value, path="inventory"))
    supplied = raw.pop("inventory_sha256", None)
    required = {
        "schema_version",
        "retirement_id",
        "programme_id",
        "request_sha256",
        "roots",
        "root_count",
        "member_count",
        "bytes",
    }
    if set(raw) != required:
        raise RootRetirementError(
            "root retirement inventory fields differ from the contract"
        )
    if raw.get("schema_version") != INVENTORY_SCHEMA_VERSION:
        raise RootRetirementError("root retirement inventory schema mismatch")
    if supplied != canonical_sha256(raw):
        raise RootRetirementError("root retirement inventory SHA-256 mismatch")
    roots = raw.get("roots")
    if not isinstance(roots, list) or not roots:
        raise RootRetirementError("root retirement inventory roots must be non-empty")
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for index, row_value in enumerate(roots):
        row = _mapping(row_value, path=f"inventory.roots[{index}]")
        fields = {
            "root_id",
            "path",
            "device",
            "inode",
            "members",
            "member_count",
            "file_count",
            "bytes",
            "inventory_sha256",
        }
        if set(row) != fields:
            raise RootRetirementError(
                "root inventory row fields differ from the contract"
            )
        root_id = _text(row["root_id"], path=f"inventory.roots[{index}].root_id")
        root_path = str(
            _absolute_path(row["path"], path=f"inventory.roots[{index}].path")
        )
        if root_id in seen_ids or root_path in seen_paths:
            raise RootRetirementError("root inventory IDs and paths must be unique")
        seen_ids.add(root_id)
        seen_paths.add(root_path)
        members = row.get("members")
        if not isinstance(members, list):
            raise RootRetirementError("root inventory members must be an array")
        member_paths: set[str] = set()
        for member_value in members:
            member = _mapping(member_value, path="inventory member")
            relative = _text(member.get("path"), path="inventory member.path")
            if Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise RootRetirementError("inventory member path is unsafe")
            if relative in member_paths:
                raise RootRetirementError("inventory member paths must be unique")
            member_paths.add(relative)
            if member.get("type") == "directory":
                if set(member) != {"path", "type", "device", "inode"}:
                    raise RootRetirementError(
                        "directory inventory row fields are invalid"
                    )
            elif member.get("type") == "file":
                if set(member) != {
                    "path",
                    "type",
                    "bytes",
                    "sha256",
                    "device",
                    "inode",
                }:
                    raise RootRetirementError("file inventory row fields are invalid")
                if type(member.get("bytes")) is not int or member["bytes"] < 0:
                    raise RootRetirementError("file inventory bytes are invalid")
                _sha256(member.get("sha256"), path="inventory member.sha256")
            else:
                raise RootRetirementError("inventory member type is unsupported")
            if (
                type(member.get("device")) is not int
                or member["device"] < 0
                or type(member.get("inode")) is not int
                or member["inode"] <= 0
            ):
                raise RootRetirementError(
                    "inventory member physical identity is invalid"
                )
        root_core = {key: row[key] for key in fields - {"root_id", "inventory_sha256"}}
        if row["inventory_sha256"] != canonical_sha256(root_core):
            raise RootRetirementError("root inventory row SHA-256 mismatch")
        if (
            row["member_count"] != len(members)
            or row["file_count"] != sum(member["type"] == "file" for member in members)
            or row["bytes"] != sum(int(member.get("bytes", 0)) for member in members)
        ):
            raise RootRetirementError("root inventory row summary mismatch")
    if (
        raw["root_count"] != len(roots)
        or raw["member_count"] != sum(int(row["member_count"]) for row in roots)
        or raw["bytes"] != sum(int(row["bytes"]) for row in roots)
    ):
        raise RootRetirementError("root retirement inventory summary mismatch")
    return {**raw, "inventory_sha256": supplied}


def validate_root_retirement_intent_v1(value: Mapping[str, Any]) -> dict[str, Any]:
    raw = dict(_mapping(value, path="intent"))
    supplied = raw.pop("intent_sha256", None)
    required = {
        "schema_version",
        "retirement_id",
        "request_sha256",
        "inventory_sha256",
        "status",
        "previous_intent_sha256",
        "prepared_intent_sha256",
        "events",
    }
    if set(raw) != required:
        raise RootRetirementError(
            "root retirement intent fields differ from the contract"
        )
    if raw.get("schema_version") != INTENT_SCHEMA_VERSION:
        raise RootRetirementError("root retirement intent schema mismatch")
    if supplied != canonical_sha256(raw):
        raise RootRetirementError("root retirement intent SHA-256 mismatch")
    if raw.get("status") not in {"PREPARED", "APPLYING", "APPLIED", "PARTIAL_FAILURE"}:
        raise RootRetirementError("root retirement intent status is unsupported")
    _sha256(raw.get("request_sha256"), path="intent.request_sha256")
    _sha256(raw.get("inventory_sha256"), path="intent.inventory_sha256")
    previous = raw.get("previous_intent_sha256")
    if previous is not None:
        _sha256(previous, path="intent.previous_intent_sha256")
    prepared = raw.get("prepared_intent_sha256")
    if raw["status"] == "PREPARED":
        if prepared is not None:
            raise RootRetirementError("PREPARED intent has an invalid prepared binding")
    else:
        _sha256(prepared, path="intent.prepared_intent_sha256")
    events = raw.get("events")
    if not isinstance(events, list):
        raise RootRetirementError("root retirement intent events must be an array")
    if raw["status"] == "PREPARED" and (previous is not None or events):
        raise RootRetirementError("PREPARED intent must have no predecessor or events")
    if raw["status"] != "PREPARED" and previous is None:
        raise RootRetirementError("non-PREPARED intent requires a predecessor")
    for event_value in events:
        event = _mapping(event_value, path="intent event")
        if set(event) != {"event", "root_id", "inventory_sha256", "at_utc"}:
            raise RootRetirementError("root retirement intent event fields are invalid")
        if event.get("event") not in {"STAGED", "RETIRED"}:
            raise RootRetirementError("root retirement intent event is unsupported")
        _text(event.get("root_id"), path="intent event.root_id")
        _sha256(event.get("inventory_sha256"), path="intent event.inventory_sha256")
        _text(event.get("at_utc"), path="intent event.at_utc")
    return {**raw, "intent_sha256": supplied}


def validate_root_retirement_review_v1(
    value: Mapping[str, Any],
    *,
    request_sha256: str,
    inventory_sha256: str,
) -> dict[str, Any]:
    raw = dict(_mapping(value, path="review"))
    supplied = raw.pop("review_sha256", None)
    required = {
        "schema_version",
        "retirement_id",
        "request_sha256",
        "inventory_sha256",
        "status",
        "reviewer",
        "reviewed_at_utc",
        "findings",
        "limitations",
    }
    if set(raw) != required:
        raise RootRetirementError(
            "root retirement review fields differ from the contract"
        )
    if raw.get("schema_version") != REVIEW_SCHEMA_VERSION:
        raise RootRetirementError("root retirement review schema mismatch")
    if supplied != canonical_sha256(raw):
        raise RootRetirementError("root retirement review SHA-256 mismatch")
    if raw.get("request_sha256") != request_sha256:
        raise RootRetirementError("root retirement review request binding mismatch")
    if raw.get("inventory_sha256") != inventory_sha256:
        raise RootRetirementError("root retirement review inventory binding mismatch")
    if raw.get("status") not in {"accepted", "accepted_with_limitations"}:
        raise RootRetirementError("root retirement inventory review is not accepted")
    _text(raw.get("retirement_id"), path="review.retirement_id")
    _text(raw.get("reviewer"), path="review.reviewer")
    _text(raw.get("reviewed_at_utc"), path="review.reviewed_at_utc")
    for name in ("findings", "limitations"):
        rows = raw.get(name)
        if not isinstance(rows, list) or not all(isinstance(row, str) for row in rows):
            raise RootRetirementError(f"review.{name} must be an array of strings")
    return {**raw, "review_sha256": supplied}


def _stage_name(retirement_id: str, root_id: str) -> str:
    return (
        ".eiel-retirement-"
        + canonical_sha256({"retirement_id": retirement_id, "root_id": root_id})[:24]
    )


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _validate_progress_path(
    progress_path: Path,
    *,
    request: Mapping[str, Any],
) -> Path:
    unresolved = progress_path.expanduser()
    if unresolved.is_symlink():
        raise RootRetirementError("root-retirement progress path cannot be a symlink")
    resolved = unresolved.resolve(strict=False)
    if resolved.exists():
        observed = resolved.stat(follow_symlinks=False)
        if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
            raise RootRetirementError(
                "root-retirement progress path must be a single-link regular file"
            )
    for row in request["roots"]:
        root = Path(str(row["path"]))
        stage = root.parent / _stage_name(
            str(request["retirement_id"]), str(row["root_id"])
        )
        if _paths_overlap(resolved, root) or _paths_overlap(resolved, stage):
            raise RootRetirementError(
                "root-retirement progress path must be disjoint from every target and stage"
            )
    return resolved


def _actual_rows_by_path(fd: int) -> dict[str, dict[str, Any]]:
    return {row["path"]: row for row in _inventory_directory_fd(fd)}


def _member_stage_name(relative: str) -> str:
    return ".eiel-delete-" + canonical_sha256({"path": relative})[:24]


def _open_relative_directory_fd(root_fd: int, parts: Sequence[str]) -> int:
    current = os.dup(root_fd)
    try:
        for part in parts:
            child = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current
            )
            os.close(current)
            current = child
        return current
    except Exception:
        os.close(current)
        raise


def _restore_interrupted_member_stages_fd(
    root_fd: int, *, expected_rows: Sequence[Mapping[str, Any]]
) -> None:
    for row in sorted(expected_rows, key=lambda item: str(item["path"]).count("/")):
        relative = str(row["path"])
        parts = relative.split("/")
        parent_fd = _open_relative_directory_fd(root_fd, parts[:-1])
        try:
            original = parts[-1]
            staged = _member_stage_name(relative)
            try:
                os.stat(staged, dir_fd=parent_fd, follow_symlinks=False)
                staged_exists = True
            except FileNotFoundError:
                staged_exists = False
            try:
                os.stat(original, dir_fd=parent_fd, follow_symlinks=False)
                original_exists = True
            except FileNotFoundError:
                original_exists = False
            if staged_exists and original_exists:
                raise RootRetirementError(
                    f"both original and authenticated member stage exist: {relative}"
                )
            if staged_exists:
                _rename_noreplace(
                    staged,
                    original,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                )
                os.fsync(parent_fd)
        finally:
            os.close(parent_fd)


def _delete_inventory_subset_fd(
    fd: int,
    *,
    expected_rows: Sequence[Mapping[str, Any]],
    root_device: int,
    root_inode: int,
    allow_missing_members: bool = False,
    prefix: str = "",
) -> None:
    expected = {str(row["path"]): dict(row) for row in expected_rows}
    _restore_interrupted_member_stages_fd(fd, expected_rows=expected_rows)
    actual = _actual_rows_by_path(fd)
    extras = sorted(set(actual) - set(expected))
    if extras:
        raise RootRetirementError(
            f"staged root contains unsealed members: {extras[:8]}"
        )
    missing = sorted(set(expected) - set(actual))
    if missing and not allow_missing_members:
        raise RootRetirementError(
            f"staged root differs from sealed inventory; missing members: {missing[:8]}"
        )
    for path, row in actual.items():
        if row != expected[path]:
            raise RootRetirementError(
                f"staged member differs from sealed inventory: {path}"
            )

    def ensure_owner_write(
        directory_fd: int,
        *,
        expected_device: int,
        expected_inode: int,
        relative_path: str,
    ) -> None:
        observed = os.fstat(directory_fd)
        if (
            not stat.S_ISDIR(observed.st_mode)
            or int(observed.st_dev) != expected_device
            or int(observed.st_ino) != expected_inode
        ):
            raise RootRetirementError(
                f"staged directory identity changed before permission update: {relative_path}"
            )
        if observed.st_mode & stat.S_IWUSR:
            return
        try:
            os.fchmod(directory_fd, stat.S_IMODE(observed.st_mode) | stat.S_IWUSR)
            os.fsync(directory_fd)
        except OSError as exc:
            raise RootRetirementError(
                f"cannot grant owner-write on approved staged directory: {relative_path}"
            ) from exc
        updated = os.fstat(directory_fd)
        if (
            not stat.S_ISDIR(updated.st_mode)
            or int(updated.st_dev) != expected_device
            or int(updated.st_ino) != expected_inode
            or not updated.st_mode & stat.S_IWUSR
        ):
            raise RootRetirementError(
                f"staged directory identity or permissions changed during update: {relative_path}"
            )

    def remove_dir(
        directory_fd: int,
        relative_prefix: str,
        *,
        expected_device: int,
        expected_inode: int,
    ) -> None:
        ensure_owner_write(
            directory_fd,
            expected_device=expected_device,
            expected_inode=expected_inode,
            relative_path=relative_prefix or ".",
        )
        with os.scandir(directory_fd) as entries:
            names = sorted((entry.name for entry in entries), key=os.fsencode)
        for name in names:
            relative = f"{relative_prefix}/{name}" if relative_prefix else name
            observed = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if stat.S_ISDIR(observed.st_mode):
                expected_row = expected[relative]
                child_fd = os.open(
                    name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=directory_fd,
                )
                try:
                    opened_directory = os.fstat(child_fd)
                    if int(opened_directory.st_dev) != int(
                        expected_row["device"]
                    ) or int(opened_directory.st_ino) != int(expected_row["inode"]):
                        raise RootRetirementError(
                            f"directory identity changed before deletion: {relative}"
                        )
                    remove_dir(
                        child_fd,
                        relative,
                        expected_device=int(expected_row["device"]),
                        expected_inode=int(expected_row["inode"]),
                    )
                    os.fsync(child_fd)
                    staged_name = _member_stage_name(relative)
                    try:
                        os.stat(staged_name, dir_fd=directory_fd, follow_symlinks=False)
                    except FileNotFoundError:
                        pass
                    else:
                        raise RootRetirementError(
                            f"member deletion stage already exists: {relative}"
                        )
                    _rename_noreplace(
                        name,
                        staged_name,
                        src_dir_fd=directory_fd,
                        dst_dir_fd=directory_fd,
                    )
                    os.fsync(directory_fd)
                    staged = os.stat(
                        staged_name, dir_fd=directory_fd, follow_symlinks=False
                    )
                    opened = os.fstat(child_fd)
                    if (staged.st_dev, staged.st_ino) != (opened.st_dev, opened.st_ino):
                        raise RootRetirementError(
                            f"directory identity changed during atomic staging: {relative}"
                        )
                    with os.scandir(child_fd) as remaining:
                        if next(remaining, None) is not None:
                            raise RootRetirementError(
                                f"directory gained an unsealed member during deletion: {relative}"
                            )
                    os.rmdir(staged_name, dir_fd=directory_fd)
                    os.fsync(directory_fd)
                finally:
                    os.close(child_fd)
            else:
                expected_row = expected[relative]
                member_fd = os.open(
                    name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd
                )
                try:
                    opened = os.fstat(member_fd)
                    if (
                        not stat.S_ISREG(opened.st_mode)
                        or opened.st_nlink != 1
                        or int(opened.st_dev) != int(expected_row["device"])
                        or int(opened.st_ino) != int(expected_row["inode"])
                        or int(opened.st_size) != int(expected_row["bytes"])
                        or _file_sha256_fd(member_fd) != expected_row["sha256"]
                    ):
                        raise RootRetirementError(
                            f"member changed immediately before deletion: {relative}"
                        )
                    staged_name = _member_stage_name(relative)
                    try:
                        os.stat(staged_name, dir_fd=directory_fd, follow_symlinks=False)
                    except FileNotFoundError:
                        pass
                    else:
                        raise RootRetirementError(
                            f"member deletion stage already exists: {relative}"
                        )
                    _rename_noreplace(
                        name,
                        staged_name,
                        src_dir_fd=directory_fd,
                        dst_dir_fd=directory_fd,
                    )
                    os.fsync(directory_fd)
                    staged = os.stat(
                        staged_name, dir_fd=directory_fd, follow_symlinks=False
                    )
                    if (staged.st_dev, staged.st_ino) != (opened.st_dev, opened.st_ino):
                        raise RootRetirementError(
                            f"file identity changed during atomic staging: {relative}"
                        )
                    os.unlink(staged_name, dir_fd=directory_fd)
                    os.fsync(directory_fd)
                finally:
                    os.close(member_fd)

    remove_dir(
        fd,
        prefix,
        expected_device=root_device,
        expected_inode=root_inode,
    )
    os.fsync(fd)


def apply_root_retirement_v1(
    *,
    request: Mapping[str, Any],
    inventory: Mapping[str, Any],
    intent: Mapping[str, Any],
    review: Mapping[str, Any],
    progress_path: Path,
    _transaction_fences_held: bool = False,
) -> dict[str, Any]:
    """Apply one prior exact dry run and return its external durable receipt."""

    normalized = validate_root_retirement_request_v1(request)
    progress_path = _validate_progress_path(progress_path, request=normalized)
    if not _transaction_fences_held:
        roots = sorted({Path(row["path"]) for row in normalized["roots"]}, key=str)
        with ExitStack() as stack:
            for root in roots:
                stack.enter_context(run_artifact_transaction_fence_v1(root))
            return apply_root_retirement_v1(
                request=normalized,
                inventory=inventory,
                intent=intent,
                review=review,
                progress_path=progress_path,
                _transaction_fences_held=True,
            )
    sealed = validate_root_retirement_inventory_v1(inventory)
    current_intent = validate_root_retirement_intent_v1(intent)
    request_sha256 = canonical_sha256(normalized)
    accepted_review = validate_root_retirement_review_v1(
        review,
        request_sha256=request_sha256,
        inventory_sha256=str(sealed.get("inventory_sha256")),
    )
    if accepted_review["retirement_id"] != normalized["retirement_id"]:
        raise RootRetirementError("root retirement review retirement ID mismatch")
    if (
        sealed.get("request_sha256") != request_sha256
        or current_intent.get("request_sha256") != request_sha256
        or current_intent.get("inventory_sha256") != sealed.get("inventory_sha256")
    ):
        raise RootRetirementError("request/inventory/intent binding mismatch")
    sealed_by_id = {str(row["root_id"]): dict(row) for row in sealed["roots"]}
    request_by_id = {str(row["root_id"]): dict(row) for row in normalized["roots"]}
    if set(sealed_by_id) != set(request_by_id) or any(
        sealed_by_id[root_id]["path"] != request_by_id[root_id]["path"]
        for root_id in request_by_id
    ):
        raise RootRetirementError("request and sealed inventory root sets differ")
    prepared_intent_sha256 = (
        current_intent["intent_sha256"]
        if current_intent["status"] == "PREPARED"
        else current_intent["prepared_intent_sha256"]
    )
    if current_intent["status"] not in {"PREPARED", "APPLYING", "APPLIED"}:
        raise RootRetirementError(
            "root retirement apply requires PREPARED, APPLYING, or APPLIED intent"
        )
    _verify_file_reference(normalized["compact_authority"], label="compact authority")
    for index, reference in enumerate(normalized["terminal_authorities"]):
        _verify_file_reference(reference, label=f"terminal authority {index}")
    for row in normalized["roots"]:
        for receipt_index, reference in enumerate(row["authority_receipts"]):
            _verify_file_reference(
                reference, label=f"{row['root_id']} authority receipt {receipt_index}"
            )
    inventory_by_id = sealed_by_id
    events = list(current_intent.get("events", []))
    event_states = {
        (str(row.get("root_id")), str(row.get("event")))
        for row in events
        if isinstance(row, Mapping)
    }
    state_by_root: dict[str, str] = {}
    for event in events:
        root_id = str(event["root_id"])
        if root_id not in sealed_by_id:
            raise RootRetirementError("intent event refers to an unsealed root")
        if event["inventory_sha256"] != sealed_by_id[root_id]["inventory_sha256"]:
            raise RootRetirementError("intent event root inventory binding mismatch")
        prior = state_by_root.get(root_id)
        if event["event"] == "STAGED" and prior is None:
            state_by_root[root_id] = "STAGED"
        elif event["event"] == "RETIRED" and prior == "STAGED":
            state_by_root[root_id] = "RETIRED"
        else:
            raise RootRetirementError("intent event sequence is invalid")

    def record(event: Mapping[str, Any]) -> None:
        nonlocal current_intent, events, event_states, state_by_root
        root_id = str(event["root_id"])
        event_name = str(event["event"])
        prior = state_by_root.get(root_id)
        if not (
            (event_name == "STAGED" and prior is None)
            or (event_name == "RETIRED" and prior == "STAGED")
        ):
            raise RootRetirementError(
                "root retirement progress event sequence is invalid"
            )
        events.append(dict(event))
        current_intent = build_root_retirement_intent_v1(
            request=normalized,
            inventory=sealed,
            status="APPLYING",
            previous_intent_sha256=current_intent["intent_sha256"],
            prepared_intent_sha256=prepared_intent_sha256,
            events=events,
        )
        _atomic_json(progress_path, current_intent)
        event_states.add((root_id, event_name))
        state_by_root[root_id] = event_name

    fences: list[Any] = []
    try:
        reservation_roots = sorted(
            {_reservation_root_for(Path(row["path"])) for row in normalized["roots"]},
            key=str,
        )
        for reservation_root in reservation_roots:
            fences.append(_acquire_release_fence(reservation_root))
        for row in normalized["roots"]:
            root = Path(row["path"])
            expected = inventory_by_id[row["root_id"]]
            resuming_staged_root = state_by_root.get(row["root_id"]) == "STAGED"
            parent = root.parent
            stage = parent / _stage_name(normalized["retirement_id"], row["root_id"])
            root_exists = root.exists()
            stage_exists = stage.exists()
            if root_exists and stage_exists:
                raise RootRetirementError(
                    f"both source and retirement stage exist: {root}"
                )
            if root_exists:
                if state_by_root.get(row["root_id"]) is not None:
                    raise RootRetirementError(
                        f"durable retirement progress conflicts with source root: {root}"
                    )
                _assert_live_state_clear(root)
                root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                parent_fd = os.open(
                    parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                )
                try:
                    observed = _inventory_retirement_root_fd_v1(root_fd, absolute=root)
                    if observed != {
                        key: value
                        for key, value in expected.items()
                        if key != "root_id"
                    }:
                        raise RootRetirementError(f"root changed after dry-run: {root}")
                    _rename_noreplace(
                        root.name,
                        stage.name,
                        src_dir_fd=parent_fd,
                        dst_dir_fd=parent_fd,
                    )
                    staged_identity = os.stat(
                        stage.name, dir_fd=parent_fd, follow_symlinks=False
                    )
                    opened_identity = os.fstat(root_fd)
                    if (staged_identity.st_dev, staged_identity.st_ino) != (
                        opened_identity.st_dev,
                        opened_identity.st_ino,
                    ):
                        raise RootRetirementError(
                            f"root identity changed during atomic staging: {root}"
                        )
                    os.fsync(parent_fd)
                finally:
                    os.close(parent_fd)
                    os.close(root_fd)
                record(
                    {
                        "event": "STAGED",
                        "root_id": row["root_id"],
                        "inventory_sha256": expected["inventory_sha256"],
                        "at_utc": _now(),
                    }
                )
                stage_exists = True
            if stage_exists:
                if stage.is_symlink():
                    raise RootRetirementError(
                        f"retirement stage cannot be a symlink: {stage}"
                    )
                if state_by_root.get(row["root_id"]) is None:
                    stage_fd = os.open(
                        stage, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                    )
                    parent_fd = os.open(
                        parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                    )
                    try:
                        recovered = _inventory_retirement_root_fd_v1(
                            stage_fd, absolute=root
                        )
                        if recovered != {
                            key: value
                            for key, value in expected.items()
                            if key != "root_id"
                        }:
                            raise RootRetirementError(
                                f"unauthenticated retirement stage differs from inventory: {stage}"
                            )
                        os.fsync(parent_fd)
                    finally:
                        os.close(parent_fd)
                        os.close(stage_fd)
                    record(
                        {
                            "event": "STAGED",
                            "root_id": row["root_id"],
                            "inventory_sha256": expected["inventory_sha256"],
                            "at_utc": _now(),
                        }
                    )
                elif state_by_root.get(row["root_id"]) != "STAGED":
                    raise RootRetirementError(
                        f"retirement stage conflicts with durable progress: {stage}"
                    )
                stage_fd = os.open(stage, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                try:
                    observed = os.fstat(stage_fd)
                    if int(observed.st_dev) != int(expected["device"]) or int(
                        observed.st_ino
                    ) != int(expected["inode"]):
                        raise RootRetirementError(
                            f"retirement stage identity mismatch: {stage}"
                        )
                    _delete_inventory_subset_fd(
                        stage_fd,
                        expected_rows=expected["members"],
                        root_device=int(expected["device"]),
                        root_inode=int(expected["inode"]),
                        allow_missing_members=resuming_staged_root,
                    )
                    os.fsync(stage_fd)
                finally:
                    os.close(stage_fd)
                parent_fd = os.open(
                    parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                )
                try:
                    os.rmdir(stage.name, dir_fd=parent_fd)
                    os.fsync(parent_fd)
                finally:
                    os.close(parent_fd)
                record(
                    {
                        "event": "RETIRED",
                        "root_id": row["root_id"],
                        "inventory_sha256": expected["inventory_sha256"],
                        "at_utc": _now(),
                    }
                )
            elif state_by_root.get(row["root_id"]) == "RETIRED":
                continue
            elif state_by_root.get(row["root_id"]) == "STAGED":
                parent_fd = os.open(
                    parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                )
                try:
                    os.fsync(parent_fd)
                finally:
                    os.close(parent_fd)
                record(
                    {
                        "event": "RETIRED",
                        "root_id": row["root_id"],
                        "inventory_sha256": expected["inventory_sha256"],
                        "at_utc": _now(),
                    }
                )
            else:
                raise RootRetirementError(
                    f"root disappeared without authenticated staging: {root}"
                )
        if current_intent["status"] == "APPLIED":
            if any(state_by_root.get(root_id) != "RETIRED" for root_id in sealed_by_id):
                raise RootRetirementError(
                    "APPLIED intent does not retire every sealed root"
                )
            final_intent = current_intent
        else:
            final_intent = build_root_retirement_intent_v1(
                request=normalized,
                inventory=sealed,
                status="APPLIED",
                previous_intent_sha256=current_intent["intent_sha256"],
                prepared_intent_sha256=prepared_intent_sha256,
                events=events,
            )
            _atomic_json(progress_path, final_intent)
        retired_events = [
            event for event in final_intent["events"] if event.get("event") == "RETIRED"
        ]
        if (
            len(retired_events) != len(normalized["roots"])
            or final_intent["events"][-1].get("event") != "RETIRED"
        ):
            raise RootRetirementError(
                "final retirement intent lacks its exact terminal completion event"
            )
        core = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "retirement_id": normalized["retirement_id"],
            "programme_id": normalized["programme_id"],
            "work_id": normalized["work_id"],
            "request_sha256": request_sha256,
            "inventory_sha256": sealed["inventory_sha256"],
            "final_intent_sha256": final_intent["intent_sha256"],
            "result": "applied",
            "retired_roots": [
                {
                    "root_id": row["root_id"],
                    "path": row["path"],
                    "inventory_sha256": inventory_by_id[row["root_id"]][
                        "inventory_sha256"
                    ],
                    "bytes_released": inventory_by_id[row["root_id"]]["bytes"],
                }
                for row in normalized["roots"]
            ],
            "root_count": len(normalized["roots"]),
            "bytes_released": int(sealed["bytes"]),
            "compact_authority": normalized["compact_authority"],
            "review_sha256": accepted_review["review_sha256"],
            "archive_invoked": False,
            "slow_storage_waited": False,
            "completed_at_utc": retired_events[-1]["at_utc"],
        }
        return {**core, "receipt_sha256": canonical_sha256(core)}
    finally:
        for handle in reversed(fences):
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()


__all__ = [
    "INTENT_SCHEMA_VERSION",
    "INVENTORY_SCHEMA_VERSION",
    "RECEIPT_SCHEMA_VERSION",
    "REVIEW_SCHEMA_VERSION",
    "REQUEST_SCHEMA_VERSION",
    "RootRetirementError",
    "apply_root_retirement_v1",
    "build_root_retirement_intent_v1",
    "build_root_retirement_inventory_v1",
    "inventory_retirement_root_v1",
    "validate_root_retirement_intent_v1",
    "validate_root_retirement_inventory_v1",
    "validate_root_retirement_review_v1",
    "validate_root_retirement_request_v1",
]
