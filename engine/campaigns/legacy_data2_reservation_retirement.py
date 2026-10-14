"""Bounded retirement of obsolete canonical ``configured runtime root`` reservation records.

This is a metadata-only repair lane for reservations written by the original
``eiel_run_storage_guard_v1`` contract.  It does not inspect, classify, move,
or delete a run root.  Production callers cannot redirect either the
reservation root or the receipt root.

The operation is dry-run-first and consumes an explicit list of reservation
basenames.  Apply requires the stable plan hash emitted by that preview,
validates every row, and observes every recorded PID absent before taking the
canonical reservation lock.  It then repeats the content and liveness checks
under that lock on the same boot before staging the first file.  A durable
intent journal makes partial staging or deletion replayable without weakening
the validation boundary.
"""

from __future__ import annotations

from engine.runtime_roots import public_runtime_path

import argparse
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import secrets
import stat
import sys
from typing import Any, Final, Iterator

from .canonical import canonical_sha256, pretty_json
from .storage_availability import DATA2_LIFECYCLE_MANAGEMENT_ROOT


LEGACY_STORAGE_GUARD_SCHEMA_VERSION: Final[str] = "eiel_run_storage_guard_v1"
RETIREMENT_REQUEST_SCHEMA_VERSION: Final[str] = (
    "eiel_legacy_data2_reservation_retirement_request_v1"
)
RETIREMENT_REPORT_SCHEMA_VERSION: Final[str] = (
    "eiel_legacy_data2_reservation_retirement_report_v1"
)
RETIREMENT_PLAN_SCHEMA_VERSION: Final[str] = (
    "eiel_legacy_data2_reservation_retirement_plan_v1"
)
RETIREMENT_JOURNAL_SCHEMA_VERSION: Final[str] = (
    "eiel_legacy_data2_reservation_retirement_journal_v1"
)
RETIREMENT_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "eiel_legacy_data2_reservation_retirement_receipt_v1"
)

CANONICAL_DATA2_RESERVATION_ROOT: Final[Path] = (
    DATA2_LIFECYCLE_MANAGEMENT_ROOT / "reservations"
)
CANONICAL_DATA2_RESERVATION_LOCK: Final[Path] = (
    CANONICAL_DATA2_RESERVATION_ROOT / ".lock"
)
RETIREMENT_CONTROL_DIRECTORY: Final[str] = ".legacy-reservation-retirement-v1"
MAX_RETIREMENT_TARGETS: Final[int] = 64
MAX_RESERVATION_BYTES: Final[int] = 64 * 1024
MAX_CONTROL_FILE_BYTES: Final[int] = 256 * 1024

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_RESERVATION_ID_RE = re.compile(r"[0-9a-f]{24}\Z")
_RESERVATION_FILENAME_RE = re.compile(r"[0-9a-f]{24}\.json\Z")
_BOOT_ID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z"
)

_LEGACY_CORE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "reservation_id",
        "run_root",
        "lifecycle_contract_sha256",
        "planned_bytes",
        "remaining_bytes",
        "pid",
        "status",
        "updated_at_utc",
    }
)
_TARGET_SUMMARY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "filename",
        "reservation_id",
        "reservation_sha256",
        "run_root",
        "lifecycle_contract_sha256",
        "planned_bytes",
        "remaining_bytes",
        "pid",
        "updated_at_utc",
    }
)


class LegacyData2ReservationRetirementError(RuntimeError):
    """The requested metadata retirement is not safely authorized."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _fault_point(_name: str) -> None:
    """No-op seam used only by crash-recovery tests."""


def _read_boot_id() -> str:
    try:
        value = (
            Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
        )
    except (OSError, UnicodeError) as exc:
        raise LegacyData2ReservationRetirementError(
            "cannot read the current Linux boot identity"
        ) from exc
    if _BOOT_ID_RE.fullmatch(value) is None:
        raise LegacyData2ReservationRetirementError(
            "current Linux boot identity is malformed"
        )
    return value


def _probe_pid(pid: int) -> str:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return "absent"
    except PermissionError:
        return "live"
    except OSError as exc:
        raise LegacyData2ReservationRetirementError(
            f"cannot determine liveness for PID {pid}: {exc}"
        ) from exc
    return "live"


def _validate_timestamp(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise LegacyData2ReservationRetirementError(
            f"{field} must be a normalized UTC timestamp"
        )
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LegacyData2ReservationRetirementError(
            f"{field} must be an ISO-8601 timestamp"
        ) from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise LegacyData2ReservationRetirementError(
            f"{field} must be a normalized UTC timestamp"
        )
    return value


def _validate_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise LegacyData2ReservationRetirementError(
            f"{field} must be a lowercase SHA-256"
        )
    return value


def _validate_legacy_run_root(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise LegacyData2ReservationRetirementError(
            "legacy reservation run_root must be a non-empty string"
        )
    root = Path(value)
    if not root.is_absolute() or str(root) != value or ".." in root.parts:
        raise LegacyData2ReservationRetirementError(
            "legacy reservation run_root must be a normalized absolute path"
        )
    data2 = Path(public_runtime_path())
    if root == data2 or data2 not in root.parents:
        raise LegacyData2ReservationRetirementError(
            "legacy reservation run_root must be a strict configured runtime root child"
        )
    for protected in (
        Path(public_runtime_path(".permanent-evidence")),
        DATA2_LIFECYCLE_MANAGEMENT_ROOT,
    ):
        if root == protected or protected in root.parents:
            raise LegacyData2ReservationRetirementError(
                "legacy reservation run_root cannot be a lifecycle management root"
            )
    return value


def legacy_data2_reservation_id_v1(
    *, run_root: str, lifecycle_contract_sha256: str
) -> str:
    """Reproduce the original V1 reservation filename identity."""

    normalized_root = _validate_legacy_run_root(run_root)
    contract_sha256 = _validate_sha256(
        lifecycle_contract_sha256,
        field="legacy reservation lifecycle_contract_sha256",
    )
    return canonical_sha256(
        {
            "run_root": normalized_root,
            "lifecycle_contract_sha256": contract_sha256,
        }
    )[:24]


def validate_legacy_data2_reservation_v1(
    value: Mapping[str, Any], *, filename: str
) -> dict[str, Any]:
    """Validate the exact pre-hardening storage-guard reservation contract."""

    if not isinstance(value, Mapping):
        raise LegacyData2ReservationRetirementError(
            "legacy reservation must be a JSON object"
        )
    raw = dict(value)
    supplied_sha256 = raw.pop("reservation_sha256", None)
    if set(raw) != _LEGACY_CORE_FIELDS:
        raise LegacyData2ReservationRetirementError(
            "legacy reservation fields differ from the exact "
            "eiel_run_storage_guard_v1 contract"
        )
    if raw.get("schema_version") != LEGACY_STORAGE_GUARD_SCHEMA_VERSION:
        raise LegacyData2ReservationRetirementError(
            "legacy reservation schema_version is unsupported"
        )
    reservation_id = raw.get("reservation_id")
    if (
        not isinstance(reservation_id, str)
        or _RESERVATION_ID_RE.fullmatch(reservation_id) is None
    ):
        raise LegacyData2ReservationRetirementError(
            "legacy reservation reservation_id is malformed"
        )
    run_root = _validate_legacy_run_root(raw.get("run_root"))
    contract_sha256 = _validate_sha256(
        raw.get("lifecycle_contract_sha256"),
        field="legacy reservation lifecycle_contract_sha256",
    )
    expected_id = legacy_data2_reservation_id_v1(
        run_root=run_root,
        lifecycle_contract_sha256=contract_sha256,
    )
    if reservation_id != expected_id:
        raise LegacyData2ReservationRetirementError(
            "legacy reservation identity binding mismatch"
        )
    if filename != f"{expected_id}.json":
        raise LegacyData2ReservationRetirementError(
            "legacy reservation filename binding mismatch"
        )
    planned_bytes = raw.get("planned_bytes")
    remaining_bytes = raw.get("remaining_bytes")
    if planned_bytes is not None and (
        type(planned_bytes) is not int or planned_bytes < 0
    ):
        raise LegacyData2ReservationRetirementError(
            "legacy reservation planned_bytes must be a non-negative integer or null"
        )
    if (
        type(remaining_bytes) is not int
        or remaining_bytes < 0
        or (planned_bytes is None and remaining_bytes != 0)
        or (planned_bytes is not None and remaining_bytes > planned_bytes)
    ):
        raise LegacyData2ReservationRetirementError(
            "legacy reservation remaining_bytes is invalid"
        )
    pid = raw.get("pid")
    if type(pid) is not int or pid <= 0:
        raise LegacyData2ReservationRetirementError(
            "legacy reservation pid must be a positive integer"
        )
    if raw.get("status") != "active":
        raise LegacyData2ReservationRetirementError(
            "legacy reservation status must be active"
        )
    _validate_timestamp(
        raw.get("updated_at_utc"), field="legacy reservation updated_at_utc"
    )
    expected_sha256 = canonical_sha256(raw)
    if supplied_sha256 != expected_sha256:
        raise LegacyData2ReservationRetirementError(
            "legacy reservation canonical SHA-256 mismatch"
        )
    return {**raw, "reservation_sha256": expected_sha256}


def _validate_filenames(values: Sequence[str]) -> list[str]:
    if isinstance(values, (str, bytes)):
        raise LegacyData2ReservationRetirementError(
            "reservation filenames must be a sequence"
        )
    filenames = list(values)
    if not filenames or len(filenames) > MAX_RETIREMENT_TARGETS:
        raise LegacyData2ReservationRetirementError(
            f"select between 1 and {MAX_RETIREMENT_TARGETS} reservation files"
        )
    if any(
        not isinstance(name, str)
        or _RESERVATION_FILENAME_RE.fullmatch(name) is None
        or Path(name).name != name
        for name in filenames
    ):
        raise LegacyData2ReservationRetirementError(
            "each reservation file must be an exact 24-hex .json basename"
        )
    if len(set(filenames)) != len(filenames):
        raise LegacyData2ReservationRetirementError(
            "reservation filenames must not contain duplicates"
        )
    return sorted(filenames)


def _transaction_id(filenames: Sequence[str]) -> str:
    return canonical_sha256(
        {
            "schema_version": RETIREMENT_REQUEST_SCHEMA_VERSION,
            "reservation_filenames": list(filenames),
        }
    )[:32]


def _validate_directory(path: Path, *, expected: Path | None = None) -> os.stat_result:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise LegacyData2ReservationRetirementError(
            f"required directory is unavailable: {path}"
        ) from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise LegacyData2ReservationRetirementError(
            f"path must be a real directory, not an alias: {path}"
        )
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise LegacyData2ReservationRetirementError(
            f"cannot resolve required directory: {path}"
        ) from exc
    if expected is not None and resolved != expected:
        raise LegacyData2ReservationRetirementError(
            f"reservation directory is not canonical: {path}"
        )
    if resolved != path:
        raise LegacyData2ReservationRetirementError(
            f"directory path contains an alias: {path}"
        )
    return info


def _path_exists(path: Path) -> bool:
    try:
        os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise LegacyData2ReservationRetirementError(
            f"cannot inspect path: {path}"
        ) from exc
    return True


def _read_regular_json(path: Path, *, max_bytes: int) -> dict[str, Any]:
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise LegacyData2ReservationRetirementError(
            f"cannot inspect JSON file: {path}"
        ) from exc
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
    ):
        raise LegacyData2ReservationRetirementError(
            f"JSON file must be regular, non-symlink, and single-link: {path}"
        )
    if before.st_size < 2 or before.st_size > max_bytes:
        raise LegacyData2ReservationRetirementError(
            f"JSON file size is outside the bounded contract: {path}"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise LegacyData2ReservationRetirementError(
            f"cannot safely open JSON file: {path}"
        ) from exc
    try:
        after = os.fstat(descriptor)
        if (
            not stat.S_ISREG(after.st_mode)
            or after.st_nlink != 1
            or (after.st_dev, after.st_ino, after.st_size)
            != (before.st_dev, before.st_ino, before.st_size)
        ):
            raise LegacyData2ReservationRetirementError(
                f"JSON file identity changed during read: {path}"
            )
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > max_bytes:
            raise LegacyData2ReservationRetirementError(
                f"JSON file exceeded its bounded size during read: {path}"
            )
    finally:
        os.close(descriptor)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise LegacyData2ReservationRetirementError(
            f"JSON file is malformed: {path}"
        ) from exc
    if not isinstance(value, Mapping):
        raise LegacyData2ReservationRetirementError(
            f"JSON file must contain an object: {path}"
        )
    return dict(value)


def _target_summary(filename: str, row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "filename": filename,
        "reservation_id": row["reservation_id"],
        "reservation_sha256": row["reservation_sha256"],
        "run_root": row["run_root"],
        "lifecycle_contract_sha256": row["lifecycle_contract_sha256"],
        "planned_bytes": row["planned_bytes"],
        "remaining_bytes": row["remaining_bytes"],
        "pid": row["pid"],
        "updated_at_utc": row["updated_at_utc"],
    }


def _retirement_plan_sha256(
    *,
    root: Path,
    transaction_id: str,
    filenames: Sequence[str],
    targets: Sequence[Mapping[str, Any]],
) -> str:
    """Bind the stable destructive plan, excluding fresh PID observations."""

    core = {
        "schema_version": RETIREMENT_PLAN_SCHEMA_VERSION,
        "transaction_id": transaction_id,
        "reservation_root": str(root),
        "lock_path": str(root / ".lock"),
        "reservation_filenames": list(filenames),
        "target_count": len(targets),
        "targets": [dict(row) for row in targets],
        "all_or_nothing_validated_before_first_rename": True,
        "retirement_scope": "legacy_reservation_metadata_files_only",
        "relay_absence_used_as_authority": False,
        "independent_root_protection_not_evaluated": True,
    }
    return canonical_sha256(core)


def _validate_target_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _TARGET_SUMMARY_FIELDS:
        raise LegacyData2ReservationRetirementError(
            "retirement target summary fields differ"
        )
    summary = dict(value)
    filename = summary.get("filename")
    if (
        not isinstance(filename, str)
        or _RESERVATION_FILENAME_RE.fullmatch(filename) is None
    ):
        raise LegacyData2ReservationRetirementError(
            "retirement target filename is malformed"
        )
    reconstructed = {
        "schema_version": LEGACY_STORAGE_GUARD_SCHEMA_VERSION,
        "reservation_id": summary.get("reservation_id"),
        "run_root": summary.get("run_root"),
        "lifecycle_contract_sha256": summary.get("lifecycle_contract_sha256"),
        "planned_bytes": summary.get("planned_bytes"),
        "remaining_bytes": summary.get("remaining_bytes"),
        "pid": summary.get("pid"),
        "status": "active",
        "updated_at_utc": summary.get("updated_at_utc"),
        "reservation_sha256": summary.get("reservation_sha256"),
    }
    validated = validate_legacy_data2_reservation_v1(
        reconstructed,
        filename=filename,
    )
    expected = _target_summary(filename, validated)
    if summary != expected:
        raise LegacyData2ReservationRetirementError(
            "retirement target summary is not canonical"
        )
    return expected


def _validate_authorization(
    value: Mapping[str, Any], *, targets: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "boot_id",
        "all_recorded_pids_absent",
        "pre_lock_liveness",
        "under_lock_liveness",
    }:
        raise LegacyData2ReservationRetirementError(
            "retirement PID authorization fields differ"
        )
    boot_id = value.get("boot_id")
    if not isinstance(boot_id, str) or _BOOT_ID_RE.fullmatch(boot_id) is None:
        raise LegacyData2ReservationRetirementError(
            "retirement PID authorization boot identity is malformed"
        )
    if value.get("all_recorded_pids_absent") is not True:
        raise LegacyData2ReservationRetirementError(
            "retirement PID authorization is not fail-closed"
        )
    expected_by_filename = {
        str(target["filename"]): int(target["pid"]) for target in targets
    }
    normalized: dict[str, Any] = {
        "boot_id": boot_id,
        "all_recorded_pids_absent": True,
    }
    for field, phase in (
        ("pre_lock_liveness", "before_lock"),
        ("under_lock_liveness", "under_lock"),
    ):
        rows = value.get(field)
        if not isinstance(rows, list) or len(rows) != len(targets):
            raise LegacyData2ReservationRetirementError(
                "retirement PID authorization row count mismatch"
            )
        normalized_rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in rows:
            if not isinstance(row, Mapping) or set(row) != {
                "filename",
                "pid",
                "state",
                "phase",
                "boot_id",
                "observed_at_utc",
            }:
                raise LegacyData2ReservationRetirementError(
                    "retirement PID liveness observation fields differ"
                )
            observation = dict(row)
            filename = observation.get("filename")
            if (
                not isinstance(filename, str)
                or filename in seen
                or filename not in expected_by_filename
                or observation.get("pid") != expected_by_filename[filename]
                or observation.get("state") != "absent"
                or observation.get("phase") != phase
                or observation.get("boot_id") != boot_id
            ):
                raise LegacyData2ReservationRetirementError(
                    "retirement PID liveness observation binding mismatch"
                )
            _validate_timestamp(
                observation.get("observed_at_utc"),
                field="retirement PID liveness observed_at_utc",
            )
            seen.add(filename)
            normalized_rows.append(observation)
        if seen != set(expected_by_filename):
            raise LegacyData2ReservationRetirementError(
                "retirement PID liveness target coverage mismatch"
            )
        normalized[field] = normalized_rows
    return normalized


def _load_reservation(path: Path, filename: str) -> dict[str, Any]:
    return validate_legacy_data2_reservation_v1(
        _read_regular_json(path, max_bytes=MAX_RESERVATION_BYTES),
        filename=filename,
    )


def _control_paths(root: Path, transaction_id: str) -> dict[str, Path]:
    control = root / RETIREMENT_CONTROL_DIRECTORY
    transactions = control / "transactions"
    transaction = transactions / transaction_id
    return {
        "control": control,
        "transactions": transactions,
        "transaction": transaction,
        "stage": transaction / "stage",
        "journal": transaction / "journal.json",
        "receipts": control / "receipts",
        "receipt": control / "receipts" / f"{transaction_id}.json",
    }


def _validate_existing_control_directories(paths: Mapping[str, Path]) -> None:
    ordered = ("control", "transactions", "transaction", "stage", "receipts")
    for name in ordered:
        path = paths[name]
        if _path_exists(path):
            _validate_directory(path)


def _journal_core(value: Mapping[str, Any]) -> dict[str, Any]:
    raw = dict(value)
    supplied = raw.pop("journal_sha256", None)
    expected_fields = {
        "schema_version",
        "transaction_id",
        "plan_sha256",
        "reservation_root",
        "reservation_filenames",
        "targets",
        "authorization",
        "state",
        "created_at_utc",
        "updated_at_utc",
        "relay_absence_used_as_authority",
        "independent_root_protection_not_evaluated",
    }
    if set(raw) != expected_fields:
        raise LegacyData2ReservationRetirementError(
            "legacy reservation retirement journal fields differ"
        )
    if raw.get("schema_version") != RETIREMENT_JOURNAL_SCHEMA_VERSION:
        raise LegacyData2ReservationRetirementError(
            "legacy reservation retirement journal schema is unsupported"
        )
    transaction_id = raw.get("transaction_id")
    if (
        not isinstance(transaction_id, str)
        or re.fullmatch(r"[0-9a-f]{32}", transaction_id) is None
    ):
        raise LegacyData2ReservationRetirementError(
            "legacy reservation retirement journal identity is malformed"
        )
    if raw.get("state") not in {"PREPARED", "STAGED"}:
        raise LegacyData2ReservationRetirementError(
            "legacy reservation retirement journal state is unsupported"
        )
    _validate_sha256(raw.get("plan_sha256"), field="retirement journal plan_sha256")
    _validate_timestamp(
        raw.get("created_at_utc"), field="retirement journal created_at_utc"
    )
    _validate_timestamp(
        raw.get("updated_at_utc"), field="retirement journal updated_at_utc"
    )
    if raw.get("relay_absence_used_as_authority") is not False:
        raise LegacyData2ReservationRetirementError(
            "retirement journal cannot use relay absence as authority"
        )
    if raw.get("independent_root_protection_not_evaluated") is not True:
        raise LegacyData2ReservationRetirementError(
            "retirement journal must preserve independent root protection"
        )
    filenames = raw.get("reservation_filenames")
    if not isinstance(filenames, list) or _validate_filenames(filenames) != filenames:
        raise LegacyData2ReservationRetirementError(
            "retirement journal filenames are not canonical"
        )
    targets = raw.get("targets")
    if not isinstance(targets, list):
        raise LegacyData2ReservationRetirementError(
            "retirement journal targets are malformed"
        )
    normalized_targets = [_validate_target_summary(row) for row in targets]
    if [row["filename"] for row in normalized_targets] != filenames:
        raise LegacyData2ReservationRetirementError(
            "retirement journal target order does not match its filenames"
        )
    _validate_authorization(raw.get("authorization"), targets=normalized_targets)
    if supplied != canonical_sha256(raw):
        raise LegacyData2ReservationRetirementError(
            "legacy reservation retirement journal SHA-256 mismatch"
        )
    return {**raw, "journal_sha256": supplied}


def _load_journal(path: Path) -> dict[str, Any] | None:
    if not _path_exists(path):
        return None
    return _journal_core(_read_regular_json(path, max_bytes=MAX_CONTROL_FILE_BYTES))


def _receipt_core(value: Mapping[str, Any]) -> dict[str, Any]:
    raw = dict(value)
    supplied = raw.pop("receipt_sha256", None)
    expected_fields = {
        "schema_version",
        "transaction_id",
        "plan_sha256",
        "status",
        "reservation_root",
        "lock_path",
        "reservation_filenames",
        "target_count",
        "targets",
        "authorization",
        "retired_at_utc",
        "retirement_scope",
        "relay_absence_used_as_authority",
        "independent_root_protection_not_evaluated",
    }
    if set(raw) != expected_fields:
        raise LegacyData2ReservationRetirementError(
            "legacy reservation retirement receipt fields differ"
        )
    if raw.get("schema_version") != RETIREMENT_RECEIPT_SCHEMA_VERSION:
        raise LegacyData2ReservationRetirementError(
            "legacy reservation retirement receipt schema is unsupported"
        )
    if raw.get("status") != "APPLIED":
        raise LegacyData2ReservationRetirementError(
            "legacy reservation retirement receipt status is unsupported"
        )
    _validate_sha256(raw.get("plan_sha256"), field="retirement receipt plan_sha256")
    transaction_id = raw.get("transaction_id")
    if (
        not isinstance(transaction_id, str)
        or re.fullmatch(r"[0-9a-f]{32}", transaction_id) is None
    ):
        raise LegacyData2ReservationRetirementError(
            "legacy reservation retirement receipt identity is malformed"
        )
    filenames = raw.get("reservation_filenames")
    if not isinstance(filenames, list) or _validate_filenames(filenames) != filenames:
        raise LegacyData2ReservationRetirementError(
            "retirement receipt filenames are not canonical"
        )
    targets = raw.get("targets")
    if not isinstance(targets, list):
        raise LegacyData2ReservationRetirementError(
            "retirement receipt targets are malformed"
        )
    normalized_targets = [_validate_target_summary(row) for row in targets]
    if [row["filename"] for row in normalized_targets] != filenames or raw.get(
        "target_count"
    ) != len(normalized_targets):
        raise LegacyData2ReservationRetirementError(
            "retirement receipt target accounting mismatch"
        )
    _validate_authorization(raw.get("authorization"), targets=normalized_targets)
    _validate_timestamp(
        raw.get("retired_at_utc"), field="retirement receipt retired_at_utc"
    )
    if raw.get("retirement_scope") != "legacy_reservation_metadata_files_only":
        raise LegacyData2ReservationRetirementError(
            "retirement receipt scope is unsupported"
        )
    if raw.get("relay_absence_used_as_authority") is not False:
        raise LegacyData2ReservationRetirementError(
            "retirement receipt cannot use relay absence as authority"
        )
    if raw.get("independent_root_protection_not_evaluated") is not True:
        raise LegacyData2ReservationRetirementError(
            "retirement receipt must preserve independent root protection"
        )
    if supplied != canonical_sha256(raw):
        raise LegacyData2ReservationRetirementError(
            "legacy reservation retirement receipt SHA-256 mismatch"
        )
    return {**raw, "receipt_sha256": supplied}


def _load_receipt(path: Path) -> dict[str, Any] | None:
    if not _path_exists(path):
        return None
    return _receipt_core(_read_regular_json(path, max_bytes=MAX_CONTROL_FILE_BYTES))


def _validate_transaction_binding(
    value: Mapping[str, Any],
    *,
    root: Path,
    transaction_id: str,
    filenames: Sequence[str],
    expected_plan_sha256: str | None = None,
) -> None:
    if value.get("transaction_id") != transaction_id:
        raise LegacyData2ReservationRetirementError(
            "retirement transaction identity mismatch"
        )
    if value.get("reservation_root") != str(root):
        raise LegacyData2ReservationRetirementError(
            "retirement transaction reservation root mismatch"
        )
    if value.get("reservation_filenames") != list(filenames):
        raise LegacyData2ReservationRetirementError(
            "retirement transaction filename binding mismatch"
        )
    targets = value.get("targets")
    if not isinstance(targets, list) or not all(
        isinstance(row, Mapping) for row in targets
    ):
        raise LegacyData2ReservationRetirementError(
            "retirement transaction targets are malformed"
        )
    bound_plan_sha256 = _retirement_plan_sha256(
        root=root,
        transaction_id=transaction_id,
        filenames=filenames,
        targets=targets,
    )
    if value.get("plan_sha256") != bound_plan_sha256:
        raise LegacyData2ReservationRetirementError(
            "retirement transaction plan binding mismatch"
        )
    if expected_plan_sha256 is not None and expected_plan_sha256 != bound_plan_sha256:
        raise LegacyData2ReservationRetirementError(
            "--apply requires the exact plan SHA-256 from the prior dry run"
        )
    if value.get("relay_absence_used_as_authority") is not False:
        raise LegacyData2ReservationRetirementError(
            "retirement transaction must not use relay absence as authority"
        )
    if value.get("independent_root_protection_not_evaluated") is not True:
        raise LegacyData2ReservationRetirementError(
            "retirement transaction must preserve independent root protection"
        )


def _journal_targets(journal: Mapping[str, Any]) -> list[dict[str, Any]]:
    targets = journal.get("targets")
    if not isinstance(targets, list) or not all(
        isinstance(row, Mapping) for row in targets
    ):
        raise LegacyData2ReservationRetirementError(
            "retirement journal targets are malformed"
        )
    return [dict(row) for row in targets]


def _collect_targets(
    root: Path,
    filenames: Sequence[str],
    *,
    paths: Mapping[str, Path],
    journal: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    expected_by_name: dict[str, dict[str, Any]] = {}
    if journal is not None:
        expected_targets = _journal_targets(journal)
        expected_by_name = {str(row.get("filename")): row for row in expected_targets}
        if set(expected_by_name) != set(filenames) or len(expected_by_name) != len(
            expected_targets
        ):
            raise LegacyData2ReservationRetirementError(
                "retirement journal target set is malformed"
            )
    collected: list[dict[str, Any]] = []
    for filename in filenames:
        source = root / filename
        staged = paths["stage"] / filename
        source_exists = _path_exists(source)
        staged_exists = _path_exists(staged) if _path_exists(paths["stage"]) else False
        if source_exists and staged_exists:
            raise LegacyData2ReservationRetirementError(
                f"reservation exists in both source and stage: {filename}"
            )
        if source_exists:
            row = _load_reservation(source, filename)
            summary = _target_summary(filename, row)
        elif staged_exists:
            row = _load_reservation(staged, filename)
            summary = _target_summary(filename, row)
        elif journal is not None:
            summary = expected_by_name[filename]
        else:
            raise LegacyData2ReservationRetirementError(
                f"explicit reservation file is missing: {filename}"
            )
        if journal is not None and summary != expected_by_name[filename]:
            raise LegacyData2ReservationRetirementError(
                f"reservation content differs from its durable journal: {filename}"
            )
        collected.append(summary)
    return collected


def _all_transaction_targets_absent(
    root: Path,
    filenames: Sequence[str],
    *,
    paths: Mapping[str, Path],
) -> bool:
    """Return whether an authenticated transaction has no source/stage bytes.

    This predicate is useful only after the caller has validated the durable
    journal binding.  It does not authorize deletion; it identifies the narrow
    post-deletion crash window where the journal's already-recorded two-phase
    PID authorization remains the relevant authority.
    """

    stage_exists = _path_exists(paths["stage"])
    return all(
        not _path_exists(root / filename)
        and not (stage_exists and _path_exists(paths["stage"] / filename))
        for filename in filenames
    )


def _observe_absent(
    targets: Sequence[Mapping[str, Any]], *, phase: str
) -> tuple[str, list[dict[str, Any]]]:
    boot_before = _read_boot_id()
    observations: list[dict[str, Any]] = []
    for row in targets:
        pid = int(row["pid"])
        state = _probe_pid(pid)
        observed_at = _now()
        observations.append(
            {
                "filename": row["filename"],
                "pid": pid,
                "state": state,
                "phase": phase,
                "boot_id": boot_before,
                "observed_at_utc": observed_at,
            }
        )
        if state != "absent":
            raise LegacyData2ReservationRetirementError(
                f"recorded reservation PID is not absent: {pid}"
            )
    boot_after = _read_boot_id()
    if boot_after != boot_before:
        raise LegacyData2ReservationRetirementError(
            "Linux boot identity changed during PID absence observation"
        )
    return boot_before, observations


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_directory(path: Path) -> None:
    if _path_exists(path):
        _validate_directory(path)
        return
    parent = path.parent
    _validate_directory(parent)
    try:
        path.mkdir(mode=0o700)
    except OSError as exc:
        raise LegacyData2ReservationRetirementError(
            f"cannot create retirement control directory: {path}"
        ) from exc
    _validate_directory(path)
    _fsync_directory(parent)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    parent = path.parent
    _validate_directory(parent)
    if _path_exists(path):
        existing = os.lstat(path)
        if (
            stat.S_ISLNK(existing.st_mode)
            or not stat.S_ISREG(existing.st_mode)
            or existing.st_nlink != 1
        ):
            raise LegacyData2ReservationRetirementError(
                f"refusing to replace aliased control file: {path}"
            )
    encoded = pretty_json(dict(value)).encode("ascii")
    if len(encoded) > MAX_CONTROL_FILE_BYTES:
        raise LegacyData2ReservationRetirementError(
            "retirement control record exceeds its compact size limit"
        )
    temporary = parent / f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, flags, 0o600)
        written = 0
        while written < len(encoded):
            written += os.write(descriptor, encoded[written:])
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, path)
        _fsync_directory(parent)
    except OSError as exc:
        raise LegacyData2ReservationRetirementError(
            f"cannot atomically write retirement control record: {path}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


@contextmanager
def _reservation_lock(root: Path) -> Iterator[None]:
    root_info = _validate_directory(root)
    root_flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    )
    root_descriptor = os.open(root, root_flags)
    lock_descriptor: int | None = None
    try:
        opened_root = os.fstat(root_descriptor)
        if (opened_root.st_dev, opened_root.st_ino) != (
            root_info.st_dev,
            root_info.st_ino,
        ):
            raise LegacyData2ReservationRetirementError(
                "reservation root identity changed before locking"
            )
        lock_flags = (
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        lock_descriptor = os.open(".lock", lock_flags, 0o600, dir_fd=root_descriptor)
        lock_info = os.fstat(lock_descriptor)
        if not stat.S_ISREG(lock_info.st_mode) or lock_info.st_nlink != 1:
            raise LegacyData2ReservationRetirementError(
                "canonical reservation lock must be regular and single-link"
            )
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        canonical_lock_info = os.lstat(root / ".lock")
        if (
            stat.S_ISLNK(canonical_lock_info.st_mode)
            or not stat.S_ISREG(canonical_lock_info.st_mode)
            or canonical_lock_info.st_nlink != 1
            or (canonical_lock_info.st_dev, canonical_lock_info.st_ino)
            != (lock_info.st_dev, lock_info.st_ino)
        ):
            raise LegacyData2ReservationRetirementError(
                "canonical reservation lock identity changed during acquisition"
            )
        _fsync_directory(root)
        yield
    except OSError as exc:
        raise LegacyData2ReservationRetirementError(
            "cannot acquire the canonical reservation lock"
        ) from exc
    finally:
        if lock_descriptor is not None:
            try:
                fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
            finally:
                os.close(lock_descriptor)
        os.close(root_descriptor)


def _prepare_control(root: Path, paths: Mapping[str, Path]) -> None:
    root_device = os.lstat(root).st_dev
    for name in ("control", "transactions", "receipts", "transaction", "stage"):
        _ensure_directory(paths[name])
        if os.lstat(paths[name]).st_dev != root_device:
            raise LegacyData2ReservationRetirementError(
                "retirement staging and receipt directories must share the "
                "reservation filesystem"
            )


def _build_authorization(
    *,
    boot_id: str,
    pre_lock: Sequence[Mapping[str, Any]],
    under_lock: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "boot_id": boot_id,
        "all_recorded_pids_absent": True,
        "pre_lock_liveness": [dict(row) for row in pre_lock],
        "under_lock_liveness": [dict(row) for row in under_lock],
    }


def _build_journal(
    *,
    root: Path,
    transaction_id: str,
    filenames: Sequence[str],
    targets: Sequence[Mapping[str, Any]],
    authorization: Mapping[str, Any],
    state: str,
    created_at_utc: str,
) -> dict[str, Any]:
    updated_at = _now()
    plan_sha256 = _retirement_plan_sha256(
        root=root,
        transaction_id=transaction_id,
        filenames=filenames,
        targets=targets,
    )
    core = {
        "schema_version": RETIREMENT_JOURNAL_SCHEMA_VERSION,
        "transaction_id": transaction_id,
        "plan_sha256": plan_sha256,
        "reservation_root": str(root),
        "reservation_filenames": list(filenames),
        "targets": [dict(row) for row in targets],
        "authorization": dict(authorization),
        "state": state,
        "created_at_utc": created_at_utc,
        "updated_at_utc": updated_at,
        "relay_absence_used_as_authority": False,
        "independent_root_protection_not_evaluated": True,
    }
    return {**core, "journal_sha256": canonical_sha256(core)}


def _build_receipt(
    *,
    root: Path,
    transaction_id: str,
    filenames: Sequence[str],
    targets: Sequence[Mapping[str, Any]],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    plan_sha256 = _retirement_plan_sha256(
        root=root,
        transaction_id=transaction_id,
        filenames=filenames,
        targets=targets,
    )
    core = {
        "schema_version": RETIREMENT_RECEIPT_SCHEMA_VERSION,
        "transaction_id": transaction_id,
        "plan_sha256": plan_sha256,
        "status": "APPLIED",
        "reservation_root": str(root),
        "lock_path": str(root / ".lock"),
        "reservation_filenames": list(filenames),
        "target_count": len(targets),
        "targets": [dict(row) for row in targets],
        "authorization": dict(authorization),
        "retired_at_utc": _now(),
        "retirement_scope": "legacy_reservation_metadata_files_only",
        "relay_absence_used_as_authority": False,
        "independent_root_protection_not_evaluated": True,
    }
    return {**core, "receipt_sha256": canonical_sha256(core)}


def _build_dry_run_report(
    *,
    root: Path,
    transaction_id: str,
    filenames: Sequence[str],
    targets: Sequence[Mapping[str, Any]],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    plan_sha256 = _retirement_plan_sha256(
        root=root,
        transaction_id=transaction_id,
        filenames=filenames,
        targets=targets,
    )
    core = {
        "schema_version": RETIREMENT_REPORT_SCHEMA_VERSION,
        "transaction_id": transaction_id,
        "plan_sha256": plan_sha256,
        "status": "DRY_RUN_READY",
        "apply": False,
        "reservation_root": str(root),
        "lock_path": str(root / ".lock"),
        "reservation_filenames": list(filenames),
        "target_count": len(targets),
        "targets": [dict(row) for row in targets],
        "authorization": dict(authorization),
        "all_or_nothing_validated_before_first_rename": True,
        "retirement_scope": "legacy_reservation_metadata_files_only",
        "relay_absence_used_as_authority": False,
        "independent_root_protection_not_evaluated": True,
    }
    return {**core, "report_sha256": canonical_sha256(core)}


def _validate_existing_receipt(
    receipt: Mapping[str, Any],
    *,
    root: Path,
    transaction_id: str,
    filenames: Sequence[str],
    paths: Mapping[str, Path],
    expected_plan_sha256: str | None,
) -> dict[str, Any]:
    _validate_transaction_binding(
        receipt,
        root=root,
        transaction_id=transaction_id,
        filenames=filenames,
        expected_plan_sha256=expected_plan_sha256,
    )
    if receipt.get("lock_path") != str(root / ".lock"):
        raise LegacyData2ReservationRetirementError(
            "retirement receipt lock binding mismatch"
        )
    targets = receipt.get("targets")
    if (
        not isinstance(targets, list)
        or receipt.get("target_count") != len(filenames)
        or len(targets) != len(filenames)
    ):
        raise LegacyData2ReservationRetirementError(
            "retirement receipt target accounting mismatch"
        )
    for filename in filenames:
        if _path_exists(root / filename) or (
            _path_exists(paths["stage"]) and _path_exists(paths["stage"] / filename)
        ):
            raise LegacyData2ReservationRetirementError(
                "applied retirement receipt conflicts with a surviving target"
            )
    return dict(receipt)


def _cleanup_transaction(paths: Mapping[str, Path]) -> None:
    transaction = paths["transaction"]
    if not _path_exists(transaction):
        return
    _validate_existing_control_directories(paths)
    stage = paths["stage"]
    if _path_exists(stage):
        if any(stage.iterdir()):
            raise LegacyData2ReservationRetirementError(
                "cannot clean a non-empty retirement stage"
            )
    allowed = {"journal.json", "stage"}
    unexpected = {entry.name for entry in transaction.iterdir()} - allowed
    if unexpected:
        raise LegacyData2ReservationRetirementError(
            "retirement transaction contains unexpected control files"
        )
    if _path_exists(paths["journal"]):
        paths["journal"].unlink()
        _fsync_directory(transaction)
    if _path_exists(stage):
        stage.rmdir()
        _fsync_directory(transaction)
    transaction.rmdir()
    _fsync_directory(paths["transactions"])


def retire_legacy_data2_reservations_v1(
    reservation_filenames: Sequence[str],
    *,
    apply: bool = False,
    expected_plan_sha256: str | None = None,
    unsafe_test_only_reservation_root: Path | None = None,
) -> dict[str, Any]:
    """Dry-run or retire an exact batch of obsolete legacy reservations.

    ``expected_plan_sha256`` is mandatory for apply and comes from the exact
    prior dry-run report.  Fresh PID observations and their timestamps are not
    part of that stable plan; they are re-observed at apply.

    ``unsafe_test_only_reservation_root`` deliberately has no CLI exposure and
    exists solely so focused tests never access ``configured runtime root``.
    """

    if type(apply) is not bool:
        raise LegacyData2ReservationRetirementError("apply must be boolean")
    if apply and expected_plan_sha256 is None:
        raise LegacyData2ReservationRetirementError(
            "--apply requires --expected-plan-sha256 from a prior dry run"
        )
    if not apply and expected_plan_sha256 is not None:
        raise LegacyData2ReservationRetirementError(
            "expected_plan_sha256 is accepted only with apply=True"
        )
    if expected_plan_sha256 is not None:
        expected_plan_sha256 = _validate_sha256(
            expected_plan_sha256,
            field="expected_plan_sha256",
        )
    filenames = _validate_filenames(reservation_filenames)
    root = (
        CANONICAL_DATA2_RESERVATION_ROOT
        if unsafe_test_only_reservation_root is None
        else Path(unsafe_test_only_reservation_root)
    )
    expected = (
        CANONICAL_DATA2_RESERVATION_ROOT
        if unsafe_test_only_reservation_root is None
        else root
    )
    _validate_directory(root, expected=expected)
    transaction_id = _transaction_id(filenames)
    paths = _control_paths(root, transaction_id)
    _validate_existing_control_directories(paths)

    pre_receipt = _load_receipt(paths["receipt"])
    if pre_receipt is None:
        pre_journal = _load_journal(paths["journal"])
        if pre_journal is not None:
            _validate_transaction_binding(
                pre_journal,
                root=root,
                transaction_id=transaction_id,
                filenames=filenames,
                expected_plan_sha256=expected_plan_sha256,
            )
        pre_targets = _collect_targets(
            root,
            filenames,
            paths=paths,
            journal=pre_journal,
        )
        pre_plan_sha256 = _retirement_plan_sha256(
            root=root,
            transaction_id=transaction_id,
            filenames=filenames,
            targets=pre_targets,
        )
        if apply and pre_plan_sha256 != expected_plan_sha256:
            raise LegacyData2ReservationRetirementError(
                "--apply requires the exact plan SHA-256 from the prior dry run"
            )
        pre_deleted_recovery = (
            pre_journal is not None
            and pre_journal.get("state") == "STAGED"
            and _all_transaction_targets_absent(
                root,
                filenames,
                paths=paths,
            )
        )
        if pre_deleted_recovery:
            pre_boot = ""
            pre_observations = []
        else:
            pre_boot, pre_observations = _observe_absent(
                pre_targets, phase="before_lock"
            )
    else:
        pre_journal = None
        pre_targets = []
        pre_boot = ""
        pre_observations = []
        pre_deleted_recovery = False

    with _reservation_lock(root):
        _validate_existing_control_directories(paths)
        receipt = _load_receipt(paths["receipt"])
        if receipt is not None:
            validated_receipt = _validate_existing_receipt(
                receipt,
                root=root,
                transaction_id=transaction_id,
                filenames=filenames,
                paths=paths,
                expected_plan_sha256=expected_plan_sha256,
            )
            if apply:
                _cleanup_transaction(paths)
            return validated_receipt

        journal = _load_journal(paths["journal"])
        if journal is not None:
            _validate_transaction_binding(
                journal,
                root=root,
                transaction_id=transaction_id,
                filenames=filenames,
                expected_plan_sha256=expected_plan_sha256,
            )
        targets = _collect_targets(
            root,
            filenames,
            paths=paths,
            journal=journal,
        )
        plan_sha256 = _retirement_plan_sha256(
            root=root,
            transaction_id=transaction_id,
            filenames=filenames,
            targets=targets,
        )
        if apply and plan_sha256 != expected_plan_sha256:
            raise LegacyData2ReservationRetirementError(
                "--apply requires the exact plan SHA-256 from the prior dry run"
            )
        if pre_targets != targets:
            raise LegacyData2ReservationRetirementError(
                "reservation target set changed before lock acquisition"
            )
        deleted_recovery = (
            journal is not None
            and journal.get("state") == "STAGED"
            and _all_transaction_targets_absent(
                root,
                filenames,
                paths=paths,
            )
        )
        if deleted_recovery != pre_deleted_recovery:
            raise LegacyData2ReservationRetirementError(
                "post-deletion retirement recovery state changed before lock acquisition"
            )
        if deleted_recovery:
            authorization = dict(journal["authorization"])
            if not apply:
                return _build_dry_run_report(
                    root=root,
                    transaction_id=transaction_id,
                    filenames=filenames,
                    targets=targets,
                    authorization=authorization,
                )
            receipt = _build_receipt(
                root=root,
                transaction_id=transaction_id,
                filenames=filenames,
                targets=targets,
                authorization=authorization,
            )
            _atomic_json(paths["receipt"], receipt)
            _fault_point("after_receipt_before_cleanup")
            _cleanup_transaction(paths)
            return receipt
        under_boot, under_observations = _observe_absent(targets, phase="under_lock")
        if pre_boot != under_boot:
            raise LegacyData2ReservationRetirementError(
                "Linux boot identity changed between PID absence checks"
            )
        authorization = _build_authorization(
            boot_id=under_boot,
            pre_lock=pre_observations,
            under_lock=under_observations,
        )
        if not apply:
            return _build_dry_run_report(
                root=root,
                transaction_id=transaction_id,
                filenames=filenames,
                targets=targets,
                authorization=authorization,
            )

        recovery_allows_already_deleted = (
            journal is not None and journal.get("state") == "STAGED"
        )
        _prepare_control(root, paths)
        created_at = str(journal["created_at_utc"]) if journal is not None else _now()
        journal = _build_journal(
            root=root,
            transaction_id=transaction_id,
            filenames=filenames,
            targets=targets,
            authorization=authorization,
            state=("STAGED" if recovery_allows_already_deleted else "PREPARED"),
            created_at_utc=created_at,
        )
        _atomic_json(paths["journal"], journal)
        _fault_point("after_prepared_journal")

        for filename in filenames:
            source = root / filename
            staged = paths["stage"] / filename
            source_exists = _path_exists(source)
            staged_exists = _path_exists(staged)
            if source_exists and staged_exists:
                raise LegacyData2ReservationRetirementError(
                    f"reservation exists in both source and stage: {filename}"
                )
            if source_exists:
                row = _load_reservation(source, filename)
                expected_target = next(
                    target for target in targets if target["filename"] == filename
                )
                if _target_summary(filename, row) != expected_target:
                    raise LegacyData2ReservationRetirementError(
                        f"reservation changed immediately before staging: {filename}"
                    )
                os.replace(source, staged)
                _fsync_directory(root)
                _fsync_directory(paths["stage"])
            elif not staged_exists and not recovery_allows_already_deleted:
                raise LegacyData2ReservationRetirementError(
                    f"reservation disappeared before staging: {filename}"
                )
            if not _path_exists(staged):
                continue
            staged_row = _load_reservation(staged, filename)
            expected_target = next(
                target for target in targets if target["filename"] == filename
            )
            if _target_summary(filename, staged_row) != expected_target:
                raise LegacyData2ReservationRetirementError(
                    f"staged reservation identity mismatch: {filename}"
                )
            _fault_point(f"after_stage:{filename}")

        journal = _build_journal(
            root=root,
            transaction_id=transaction_id,
            filenames=filenames,
            targets=targets,
            authorization=authorization,
            state="STAGED",
            created_at_utc=created_at,
        )
        _atomic_json(paths["journal"], journal)
        _fault_point("after_staged_journal")

        for filename in filenames:
            staged = paths["stage"] / filename
            if _path_exists(staged):
                staged_row = _load_reservation(staged, filename)
                expected_target = next(
                    target for target in targets if target["filename"] == filename
                )
                if _target_summary(filename, staged_row) != expected_target:
                    raise LegacyData2ReservationRetirementError(
                        f"staged reservation changed before deletion: {filename}"
                    )
                staged.unlink()
                _fsync_directory(paths["stage"])
            _fault_point(f"after_delete:{filename}")
        _fault_point("after_delete_all_before_receipt")

        receipt = _build_receipt(
            root=root,
            transaction_id=transaction_id,
            filenames=filenames,
            targets=targets,
            authorization=authorization,
        )
        _atomic_json(paths["receipt"], receipt)
        _fault_point("after_receipt_before_cleanup")
        _cleanup_transaction(paths)
        return receipt


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Dry-run or retire explicitly named obsolete legacy reservations "
            "from the canonical configured runtime root reservation root."
        )
    )
    parser.add_argument(
        "--reservation-file",
        action="append",
        required=True,
        dest="reservation_filenames",
        metavar="24HEX.json",
        help="exact reservation basename; repeat for an all-or-nothing batch",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="apply the validated metadata retirement (default: dry run)",
    )
    parser.add_argument(
        "--expected-plan-sha256",
        help="exact plan SHA-256 emitted by the prior dry run; required with --apply",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = retire_legacy_data2_reservations_v1(
            args.reservation_filenames,
            apply=args.apply,
            expected_plan_sha256=args.expected_plan_sha256,
        )
    except LegacyData2ReservationRetirementError as exc:
        print(f"legacy reservation retirement refused: {exc}", file=sys.stderr)
        return 2
    print(pretty_json(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANONICAL_DATA2_RESERVATION_LOCK",
    "CANONICAL_DATA2_RESERVATION_ROOT",
    "LegacyData2ReservationRetirementError",
    "legacy_data2_reservation_id_v1",
    "main",
    "retire_legacy_data2_reservations_v1",
    "validate_legacy_data2_reservation_v1",
]
