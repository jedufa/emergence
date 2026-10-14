"""Work Item Resource Accounting V1 (WRA-1).

This module binds commodity ``ccusage`` telemetry to EIEL Work and Session
records.  It deliberately does not parse Codex log files, calculate money, or
learn estimator rules.  ``ccusage`` owns token extraction; EIEL owns durable
attribution and immutable evidence.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tools.actors.canonical import canonical_sha256, pretty_json
from tools.actors.common import (
    ActorError,
    actor_state_root,
    advisory_lock,
    atomic_write_json,
    file_sha256,
    require_explicit_schema_version,
    require_known_fields,
    stable_id,
    utc_now,
    validate_timestamp,
)

RESOURCE_ESTIMATE_SCHEMA_VERSION = "eiel_resource_estimate_v1"
WORK_ACCOUNTING_SCHEMA_VERSION = "eiel_work_resource_accounting_v1"
SESSION_ACCOUNTING_SCHEMA_VERSION = "eiel_session_resource_accounting_v1"
TOKEN_SNAPSHOT_SCHEMA_VERSION = "eiel_token_snapshot_v1"
SESSION_USAGE_SCHEMA_VERSION = "eiel_session_token_usage_v1"
WORK_USAGE_SCHEMA_VERSION = "eiel_token_usage_v1"
TOKEN_REPORT_SCHEMA_VERSION = "eiel_token_report_index_v1"
CAPTURE_CLOSE_INTENT_SCHEMA_VERSION = "eiel_wra_capture_close_intent_v1"
CAPTURE_CLOSE_RESULT_SCHEMA_VERSION = "eiel_wra_capture_close_result_v1"
ESTIMATOR_REVIEW_RECEIPT_SCHEMA_VERSION = "eiel_estimator_review_receipt_v1"
ESTIMATOR_APPROVAL_RECEIPT_SCHEMA_VERSION = "eiel_estimator_approval_receipt_v1"
RESOURCE_DIMENSION = "tokens"

CCUSAGE_COMMAND_ENV = "EIEL_CCUSAGE_COMMAND"
DEFAULT_CCUSAGE_COMMAND = "ccusage"
CCUSAGE_TIMEOUT_SECONDS = 20.0

MEASUREMENT_STATUSES = ("MEASURED", "PARTIAL", "UNAVAILABLE")
ESTIMATE_CONFIDENCE = ("none", "low", "medium", "high")

# Immutable WRA artifacts are closed ABIs.  A future producer must increment
# schema_version instead of smuggling a new resource dimension into a V1
# artifact that older code would partially interpret.
RESOURCE_ARTIFACT_FIELDS: dict[str, frozenset[str]] = {
    TOKEN_SNAPSHOT_SCHEMA_VERSION: frozenset({
        "schema_version", "snapshot_id", "snapshot_sha256", "capture_kind",
        "status", "work_id", "session_id", "context_epoch_id",
        "codex_thread_id", "captured_at_utc", "model", "reasoning_level",
        "cumulative", "source", "problems", "correction_note",
        "close_intent_id", "resource_dimension", "measurement_scope",
        "attribution_assurance", "correction_capture_id", "correction_assurance",
    }),
    SESSION_USAGE_SCHEMA_VERSION: frozenset({
        "schema_version", "artifact_id", "artifact_sha256", "capture_id",
        "boundary_record", "status",
        "work_id", "session_id", "context_epoch_id", "codex_thread_id",
        "model", "reasoning_level", "duration_minutes", "tool_calls",
        "compactions", "input_tokens", "cached_input_tokens",
        "cache_creation_tokens", "output_tokens", "total_tokens",
        "start_snapshot", "end_snapshot", "measured_at_utc", "source",
        "problems", "correction_note", "close_intent_id", "resource_dimension",
        "measurement_scope", "attribution_assurance", "correction_capture_id",
        "correction_assurance",
    }),
    WORK_USAGE_SCHEMA_VERSION: frozenset({
        "schema_version", "artifact_id", "artifact_sha256", "status",
        "work_id", "model", "reasoning_level", "work_type", "subsystem",
        "duration_minutes", "tool_calls", "compactions", "input_tokens",
        "cached_input_tokens", "cache_creation_tokens", "output_tokens",
        "total_tokens", "estimated_tokens", "actual_tokens",
        "estimation_error", "source", "session_count",
        "measured_segment_count", "session_breakdown", "problems",
        "generated_at_utc", "correction_note", "resource_dimension",
        "measurement_scope", "attribution_assurance", "correction_assurance",
    }),
    TOKEN_REPORT_SCHEMA_VERSION: frozenset({
        "schema_version", "generated_at_utc", "work_item_count",
        "work_items", "report_sha256",
    }),
}
TOKEN_CUMULATIVE_FIELDS = frozenset({
    "input_tokens", "cached_input_tokens", "cache_creation_tokens",
    "output_tokens", "total_tokens", "models", "first_activity",
    "last_activity",
})


def _optional_nonnegative_int(value: Any, *, label: str) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        raise ActorError(f"{label} must be a non-negative integer or null", code="resource_estimate_invalid")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ActorError(f"{label} must be a non-negative integer or null", code="resource_estimate_invalid") from exc
    if parsed < 0:
        raise ActorError(f"{label} must be non-negative", code="resource_estimate_invalid")
    return parsed


def normalize_resource_estimate(raw: Mapping[str, Any] | None, *, generated_at: str | None = None) -> dict[str, Any]:
    """Return the immutable prediction surface created with every Work Item."""

    value = dict(raw or {})
    if raw is not None and "schema_version" in value:
        require_explicit_schema_version(
            value,
            expected=RESOURCE_ESTIMATE_SCHEMA_VERSION,
            label="resource estimate",
        )
    tokens = value.get("estimated_tokens") if isinstance(value.get("estimated_tokens"), Mapping) else {}
    low = _optional_nonnegative_int(dict(tokens).get("low"), label="estimated_tokens.low")
    expected = _optional_nonnegative_int(dict(tokens).get("expected"), label="estimated_tokens.expected")
    high = _optional_nonnegative_int(dict(tokens).get("high"), label="estimated_tokens.high")
    populated = [number for number in (low, expected, high) if number is not None]
    if low is not None and expected is not None and low > expected:
        raise ActorError("estimated_tokens.low may not exceed expected", code="resource_estimate_invalid")
    if expected is not None and high is not None and expected > high:
        raise ActorError("estimated_tokens.expected may not exceed high", code="resource_estimate_invalid")
    if low is not None and high is not None and low > high:
        raise ActorError("estimated_tokens.low may not exceed high", code="resource_estimate_invalid")
    confidence = str(value.get("confidence") or "none").strip().lower()
    if confidence not in ESTIMATE_CONFIDENCE:
        raise ActorError(
            "resource estimate confidence must be one of " + ", ".join(ESTIMATE_CONFIDENCE),
            code="resource_estimate_invalid",
        )
    method = str(value.get("method") or "static").strip().lower()
    if method != "static":
        raise ActorError("WRA-1 supports only the static estimate method", code="resource_estimate_invalid")
    recorded = value.get("recorded_at_utc")
    if recorded:
        recorded = validate_timestamp(str(recorded))
    else:
        recorded = utc_now(generated_at)
    return {
        "schema_version": RESOURCE_ESTIMATE_SCHEMA_VERSION,
        "estimated_tokens": {"low": low, "expected": expected, "high": high},
        "confidence": confidence,
        "method": method,
        "recorded_at_utc": recorded,
        "prediction_present": bool(populated),
    }


def validate_resource_estimate(raw: Any) -> list[str]:
    if not isinstance(raw, Mapping):
        return ["resource estimate must be an object"]
    try:
        require_explicit_schema_version(
            raw,
            expected=RESOURCE_ESTIMATE_SCHEMA_VERSION,
            label="resource estimate",
        )
        normalize_resource_estimate(raw)
    except ActorError as exc:
        return [str(exc)]
    return []


def load_resource_estimate(raw: Any) -> dict[str, Any]:
    """Strict persisted-record loader for ``eiel_resource_estimate_v1``."""

    if not isinstance(raw, Mapping):
        raise ActorError("resource estimate must be an object", code="schema_record_malformed")
    require_explicit_schema_version(
        raw,
        expected=RESOURCE_ESTIMATE_SCHEMA_VERSION,
        label="resource estimate",
    )
    return normalize_resource_estimate(raw)


def normalize_work_resource_accounting(
    raw: Mapping[str, Any] | None,
    *,
    estimate: Mapping[str, Any],
) -> dict[str, Any]:
    value = dict(raw or {})
    if raw is not None and "schema_version" in value:
        require_explicit_schema_version(
            value,
            expected=WORK_ACCOUNTING_SCHEMA_VERSION,
            label="Work resource accounting projection",
        )
    artifact = dict(value.get("artifact") or {}) if isinstance(value.get("artifact"), Mapping) else None
    error = dict(value.get("estimation_error") or {}) if isinstance(value.get("estimation_error"), Mapping) else None
    history = [dict(row) for row in (value.get("artifact_history") or []) if isinstance(row, Mapping)]
    return {
        "schema_version": WORK_ACCOUNTING_SCHEMA_VERSION,
        "estimate": dict(estimate),
        "artifact": artifact,
        "estimation_error": error,
        "artifact_history": history,
    }


def normalize_session_resource_accounting(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    value = dict(raw or {})
    if raw is not None and "schema_version" in value:
        require_explicit_schema_version(
            value,
            expected=SESSION_ACCOUNTING_SCHEMA_VERSION,
            label="Session resource accounting projection",
        )
    captures = [dict(row) for row in (value.get("captures") or []) if isinstance(row, Mapping)]
    artifacts = [dict(row) for row in (value.get("usage_artifacts") or []) if isinstance(row, Mapping)]
    active = str(value.get("active_capture_id") or "").strip().upper() or None
    return {
        "schema_version": SESSION_ACCOUNTING_SCHEMA_VERSION,
        "captures": captures,
        "active_capture_id": active,
        "usage_artifacts": artifacts,
        "latest_usage_artifact": (
            dict(value.get("latest_usage_artifact") or {})
            if isinstance(value.get("latest_usage_artifact"), Mapping)
            else None
        ),
    }


def _parse_accounting_time(value: Any, *, label: str, errors: list[str]) -> dt.datetime | None:
    if not value:
        errors.append(f"resource accounting {label} is missing")
        return None
    try:
        return dt.datetime.fromisoformat(validate_timestamp(str(value)).replace("Z", "+00:00"))
    except ActorError as exc:
        errors.append(f"resource accounting {label} is invalid: {exc}")
        return None


def validate_session_resource_accounting(raw: Any) -> list[str]:
    """Validate the durable WRA state machine, including closed intervals.

    Capture commands already prevented a second *active* interval, but v1.17 did
    not validate retained closed intervals.  That made corrupted imported state
    look authoritative after restart.  Validation therefore treats START/END
    evidence and interval non-overlap as durable invariants, not command-time
    conveniences.
    """

    if raw is None:
        return []
    if not isinstance(raw, Mapping):
        return ["resource_accounting must be an object"]
    try:
        require_explicit_schema_version(
            raw,
            expected=SESSION_ACCOUNTING_SCHEMA_VERSION,
            label="Session resource accounting projection",
        )
        value = normalize_session_resource_accounting(raw)
    except ActorError as exc:
        return [str(exc)]
    seen: set[str] = set()
    errors: list[str] = []
    closed: list[tuple[str, dt.datetime, dt.datetime]] = []
    started_ids: set[str] = set()
    for row in value["captures"]:
        capture_id = str(row.get("capture_id") or "")
        if not capture_id:
            errors.append("resource accounting capture has no capture_id")
        elif capture_id in seen:
            errors.append(f"duplicate resource accounting capture_id: {capture_id}")
        seen.add(capture_id)
        status = str(row.get("status") or "")
        if status not in {"STARTED", "COMPLETED", "UNAVAILABLE"}:
            errors.append(f"invalid resource accounting capture status: {row.get('status')}")
            continue
        if status in {"STARTED", "COMPLETED"} and not isinstance(row.get("start_snapshot"), Mapping):
            errors.append(f"resource accounting capture {capture_id or '<missing>'} has no START evidence")
        if status == "STARTED":
            started_ids.add(capture_id)
            if row.get("end_snapshot") is not None or row.get("ended_at_utc") is not None:
                errors.append(f"STARTED resource accounting capture {capture_id} already has END evidence")
        if status == "COMPLETED":
            if not isinstance(row.get("end_snapshot"), Mapping):
                errors.append(f"resource accounting capture {capture_id or '<missing>'} has no END evidence")
            start = _parse_accounting_time(row.get("started_at_utc"), label=f"capture {capture_id} start", errors=errors)
            end = _parse_accounting_time(row.get("ended_at_utc"), label=f"capture {capture_id} end", errors=errors)
            if start is not None and end is not None:
                if end < start:
                    errors.append(f"resource accounting capture {capture_id} ends before it starts")
                else:
                    closed.append((capture_id, start, end))
    active = value.get("active_capture_id")
    if active and active not in seen:
        errors.append("active_capture_id does not identify a retained capture")
    if active and active not in started_ids:
        errors.append("active_capture_id must identify a STARTED capture")
    if len(started_ids) > 1:
        errors.append("resource accounting retains more than one STARTED capture")
    for index, (left_id, left_start, left_end) in enumerate(closed):
        for right_id, right_start, right_end in closed[index + 1 :]:
            # Adjacent boundaries are allowed. A positive-width intersection is not.
            if max(left_start, right_start) < min(left_end, right_end):
                errors.append(
                    f"resource accounting closed intervals overlap: {left_id} and {right_id}"
                )
    return list(dict.fromkeys(errors))


def validate_session_resource_accounting_binding(
    raw: Any,
    *,
    session_record: Mapping[str, Any],
    work_record: Mapping[str, Any],
    state_root: str | Path | None = None,
) -> list[str]:
    """Bind WRA evidence to its authoritative Work, Session, epoch and interval."""

    errors = validate_session_resource_accounting(raw)
    if not isinstance(raw, Mapping):
        return errors
    try:
        accounting = normalize_session_resource_accounting(raw)
    except ActorError as exc:
        return [*errors, str(exc)]
    payload = dict(session_record.get("payload") or {})
    epoch = dict(payload.get("epoch") or {})
    expected = {
        "work_id": str(work_record.get("record_id") or ""),
        "session_id": str(session_record.get("record_id") or ""),
        "context_epoch_id": str(epoch.get("epoch_id") or ""),
        "codex_thread_id": str(payload.get("current_codex_id") or ""),
    }
    epoch_start: dt.datetime | None = None
    epoch_end: dt.datetime | None = None
    if epoch.get("started_at_utc"):
        try:
            epoch_start = dt.datetime.fromisoformat(validate_timestamp(str(epoch["started_at_utc"])).replace("Z", "+00:00"))
        except ActorError:
            errors.append("resource accounting Session epoch start is invalid")
    if epoch.get("ended_at_utc"):
        try:
            epoch_end = dt.datetime.fromisoformat(validate_timestamp(str(epoch["ended_at_utc"])).replace("Z", "+00:00"))
        except ActorError:
            errors.append("resource accounting Session epoch end is invalid")

    for capture in accounting.get("captures") or []:
        capture_id = str(capture.get("capture_id") or "<missing>")
        for field, expected_value in expected.items():
            actual = str(capture.get(field) or "")
            if not actual:
                errors.append(f"resource accounting capture {capture_id} is missing {field}")
            elif expected_value and actual != expected_value:
                errors.append(
                    f"resource accounting capture {capture_id} {field} attribution mismatch: {actual} != {expected_value}"
                )
        start = None
        end = None
        if capture.get("started_at_utc"):
            try:
                start = dt.datetime.fromisoformat(validate_timestamp(str(capture["started_at_utc"])).replace("Z", "+00:00"))
            except ActorError:
                pass
        if capture.get("ended_at_utc"):
            try:
                end = dt.datetime.fromisoformat(validate_timestamp(str(capture["ended_at_utc"])).replace("Z", "+00:00"))
            except ActorError:
                pass
        if epoch_start is not None and start is not None and start < epoch_start:
            errors.append(f"resource accounting capture {capture_id} starts outside the Session epoch boundary")
        if epoch_end is not None and end is not None and end > epoch_end:
            errors.append(f"resource accounting capture {capture_id} ends outside the Session epoch boundary")

        ref = capture.get("usage_artifact")
        if isinstance(ref, Mapping):
            try:
                artifact = _read_artifact(
                    ref,
                    expected_schema_version=SESSION_USAGE_SCHEMA_VERSION,
                    state_root=state_root,
                    expected_artifact_id=str(ref.get("artifact_id") or ""),
                    expected_work_id=str(capture.get("work_id") or ""),
                    expected_session_id=str(capture.get("session_id") or ""),
                    expected_context_epoch_id=str(capture.get("context_epoch_id") or ""),
                    expected_codex_thread_id=str(capture.get("codex_thread_id") or ""),
                    expected_status=str(ref.get("status") or "") or None,
                )
            except ActorError as exc:
                errors.append(str(exc))
                continue
            for field in ("work_id", "session_id", "context_epoch_id", "codex_thread_id"):
                artifact_value = str(artifact.get(field) or "")
                capture_value = str(capture.get(field) or "")
                if artifact_value != capture_value:
                    errors.append(
                        f"resource accounting artifact {artifact.get('artifact_id')} {field} attribution mismatch"
                    )
            if artifact.get("started_at_utc") and start is not None:
                try:
                    artifact_start = dt.datetime.fromisoformat(validate_timestamp(str(artifact["started_at_utc"])).replace("Z", "+00:00"))
                    if artifact_start != start:
                        errors.append(f"resource accounting artifact {artifact.get('artifact_id')} start boundary mismatch")
                except ActorError:
                    errors.append(f"resource accounting artifact {artifact.get('artifact_id')} start boundary invalid")
            if artifact.get("ended_at_utc") and end is not None:
                try:
                    artifact_end = dt.datetime.fromisoformat(validate_timestamp(str(artifact["ended_at_utc"])).replace("Z", "+00:00"))
                    if artifact_end != end:
                        errors.append(f"resource accounting artifact {artifact.get('artifact_id')} end boundary mismatch")
                except ActorError:
                    errors.append(f"resource accounting artifact {artifact.get('artifact_id')} end boundary invalid")
    return list(dict.fromkeys(errors))


def load_work_resource_accounting(
    raw: Any,
    *,
    estimate: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ActorError("Work resource accounting projection must be an object", code="schema_record_malformed")
    require_explicit_schema_version(
        raw,
        expected=WORK_ACCOUNTING_SCHEMA_VERSION,
        label="Work resource accounting projection",
    )
    return normalize_work_resource_accounting(raw, estimate=estimate)


def load_session_resource_accounting(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ActorError("Session resource accounting projection must be an object", code="schema_record_malformed")
    require_explicit_schema_version(
        raw,
        expected=SESSION_ACCOUNTING_SCHEMA_VERSION,
        label="Session resource accounting projection",
    )
    value = normalize_session_resource_accounting(raw)
    errors = validate_session_resource_accounting(value)
    if errors:
        raise ActorError(
            "invalid Session resource accounting projection",
            code="schema_record_malformed",
            problems=errors,
        )
    return value


def _session_accounting_or_legacy_default(raw: Any) -> dict[str, Any]:
    return normalize_session_resource_accounting(None) if raw is None else load_session_resource_accounting(raw)


def _resource_estimate_or_legacy_default(
    raw: Any,
    *,
    generated_at: str | None = None,
) -> dict[str, Any]:
    return (
        normalize_resource_estimate(None, generated_at=generated_at)
        if raw is None
        else load_resource_estimate(raw)
    )


def _work_accounting_or_legacy_default(
    raw: Any,
    *,
    estimate: Mapping[str, Any],
) -> dict[str, Any]:
    return (
        normalize_work_resource_accounting(None, estimate=estimate)
        if raw is None
        else load_work_resource_accounting(raw, estimate=estimate)
    )


def _resource_root(state_root: str | Path | None) -> Path:
    return actor_state_root(state_root) / "resource-accounting"


def resource_thread_lock(state_root: str | Path | None, thread_id: str):
    """Serialize attribution decisions for one Codex thread globally.

    The file name is a digest so external thread identifiers never become
    uncontrolled path components. Work lifecycle locks still protect Work and
    Session authority; this lock protects the independent attribution axis.
    """

    normalized = str(thread_id or "").strip()
    if not normalized:
        raise ActorError("Codex thread ID is required for resource attribution", code="resource_thread_missing")
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return advisory_lock(_resource_root(state_root) / "thread-locks" / f"{digest}.lock")


def resource_thread_conflicts(
    state_root: str | Path | None,
    thread_id: str,
    *,
    exclude_session_id: str | None = None,
) -> list[str]:
    """Return Sessions that still own the thread's accounting interval."""

    # Local import avoids making the portfolio layer depend on WRA-1.
    from tools.actors.portfolio import SESSION_ACTIVE_STATUSES, session_store

    wanted = str(thread_id or "").strip()
    if not wanted:
        return []
    conflicts: list[str] = []
    store_obj = session_store(state_root)
    for row in store_obj.list_read_only():
        record_id = str(row.get("record_id") or "")
        if (
            not record_id
            or record_id == str(exclude_session_id or "")
            or row.get("status") == "INVALID"
        ):
            continue
        try:
            record = store_obj.get_read_only(record_id)
        except ActorError:
            continue
        payload = dict(record.get("payload") or {})
        accounting = _session_accounting_or_legacy_default(payload.get("resource_accounting"))
        active_id = str(accounting.get("active_capture_id") or "")
        active_capture = next(
            (
                dict(capture)
                for capture in accounting.get("captures") or []
                if str(dict(capture).get("capture_id") or "") == active_id
            ),
            None,
        )
        owns_interval = bool(
            active_capture and str(active_capture.get("codex_thread_id") or "") == wanted
        )
        active_identity = bool(
            record.get("status") in SESSION_ACTIVE_STATUSES
            and not dict(payload.get("epoch") or {}).get("ended_at_utc")
            and str(payload.get("current_codex_id") or "") == wanted
        )
        if owns_interval or active_identity:
            conflicts.append(record_id)
    return sorted(set(conflicts))


def require_resource_thread_available(
    state_root: str | Path | None,
    thread_id: str,
    *,
    exclude_session_id: str | None = None,
) -> None:
    conflicts = resource_thread_conflicts(
        state_root,
        thread_id,
        exclude_session_id=exclude_session_id,
    )
    if conflicts:
        raise ActorError(
            "Codex thread already owns another open resource-accounting interval",
            code="resource_thread_attribution_conflict",
            problems=["Conflicting Sessions: " + ", ".join(conflicts)],
            related_records=conflicts,
            next_actions=[
                {
                    "action": "close_previous_epoch",
                    "description": (
                        "Close the prior Session Context Epoch or terminal Session before reusing this Codex thread."
                    ),
                }
            ],
        )


def _artifact_reference(
    path: Path,
    *,
    artifact_id: str,
    schema_version: str,
    status: str,
    state_root: str | Path | None = None,
) -> dict[str, Any]:
    reference = {
        "artifact_id": artifact_id,
        "schema_version": schema_version,
        "path": str(path),  # advisory historical path
        "sha256": file_sha256(path),
        "status": status,
    }
    try:
        reference["state_root_locator"] = path.resolve().relative_to(actor_state_root(state_root).resolve()).as_posix()
    except ValueError:
        reference["state_root_locator"] = None
    return reference


def _write_immutable(path: Path, value: Mapping[str, Any]) -> Path:
    """Write one immutable artifact, allowing only byte-identical retries."""

    expected_bytes = pretty_json(dict(value)).encode("ascii")
    if path.exists():
        try:
            existing_bytes = path.read_bytes()
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ActorError(
                f"cannot validate existing resource artifact: {path}: {exc}",
                code="resource_artifact_invalid",
            ) from exc
        if existing_bytes == expected_bytes:
            return path
        try:
            existing = json.loads(existing_bytes.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ActorError(
                f"cannot validate existing resource artifact: {path}: {exc}",
                code="resource_artifact_invalid",
            ) from exc
        if existing != dict(value) or existing_bytes != expected_bytes:
            raise ActorError(
                f"immutable resource artifact conflict: {path}",
                code="resource_artifact_conflict",
                problems=[str(path), "existing artifact is not byte-identical to canonical output"],
            )
    return atomic_write_json(path, value)


def _command_tokens(override: str | Sequence[str] | None = None) -> list[str]:
    if isinstance(override, str):
        parts = shlex.split(override)
    elif override is not None:
        parts = [str(item) for item in override]
    else:
        parts = shlex.split(os.environ.get(CCUSAGE_COMMAND_ENV, DEFAULT_CCUSAGE_COMMAND))
    if not parts:
        raise ActorError("ccusage command is empty", code="ccusage_command_invalid")
    return parts


def _run_command(command: Sequence[str], *, timeout_seconds: float = CCUSAGE_TIMEOUT_SECONDS) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout_seconds,
    )


def _ccusage_version(base: Sequence[str]) -> tuple[str | None, list[str]]:
    try:
        result = _run_command([*base, "--version"], timeout_seconds=5.0)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, [f"ccusage version unavailable: {exc}"]
    if result.returncode != 0:
        return None, [result.stderr.strip() or f"ccusage --version exited {result.returncode}"]
    return result.stdout.strip() or None, []


def _counter(row: Mapping[str, Any], *names: str) -> tuple[int | None, str | None, bool]:
    """Return one non-negative counter plus validation metadata.

    ``ccusage`` owns telemetry parsing, but EIEL must still reject malformed
    structured output instead of silently converting an invalid present value
    into zero.  The final boolean distinguishes "not emitted" from "emitted
    but invalid".
    """

    for name in names:
        if name not in row or row[name] is None:
            continue
        value = row[name]
        if isinstance(value, bool):
            return None, f"ccusage counter {name} must be a non-negative integer", True
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None, f"ccusage counter {name} must be a non-negative integer", True
        if parsed < 0:
            return None, f"ccusage counter {name} must be non-negative", True
        return parsed, None, True
    return None, None, False


def _session_candidates(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, list):
        return [row for row in value if isinstance(row, Mapping)]
    if not isinstance(value, Mapping):
        return []
    for key in ("sessions", "data", "items", "results"):
        rows = value.get(key)
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, Mapping)]
        if isinstance(rows, Mapping):
            return [dict(rows)]
    return [dict(value)]


def _select_session_row(value: Any, *, thread_id: str) -> Mapping[str, Any] | None:
    rows = _session_candidates(value)
    exact: list[Mapping[str, Any]] = []
    for row in rows:
        observed = str(
            row.get("session")
            or row.get("sessionId")
            or row.get("session_id")
            or row.get("threadId")
            or row.get("thread_id")
            or row.get("id")
            or ""
        ).strip()
        if observed == thread_id:
            exact.append(row)
    if exact:
        return exact[0]
    if len(rows) == 1:
        row = rows[0]
        if any(name in row for name in ("inputTokens", "input_tokens", "totalTokens", "total_tokens")):
            return row
    return None


def _normalize_ccusage_row(row: Mapping[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    input_tokens, input_problem, input_present = _counter(row, "inputTokens", "input_tokens")
    cached_input, cached_problem, cached_present = _counter(
        row,
        "cacheReadTokens",
        "cache_read_tokens",
        "cachedInputTokens",
        "cached_input_tokens",
    )
    cache_creation, creation_problem, creation_present = _counter(
        row, "cacheCreationTokens", "cache_creation_tokens"
    )
    output_tokens, output_problem, output_present = _counter(row, "outputTokens", "output_tokens")
    total_tokens, total_problem, total_present = _counter(row, "totalTokens", "total_tokens")
    problems = [
        problem
        for problem in (input_problem, cached_problem, creation_problem, output_problem, total_problem)
        if problem
    ]
    if problems:
        return None, problems
    if not any((input_present, cached_present, creation_present, output_present, total_present)):
        return None, ["ccusage session JSON contains no recognized token counters"]
    input_tokens = input_tokens or 0
    cached_input = cached_input or 0
    cache_creation = cache_creation or 0
    output_tokens = output_tokens or 0
    if total_tokens is None:
        total_tokens = input_tokens + cached_input + cache_creation + output_tokens
    models_value = row.get("modelsUsed") or row.get("models_used") or row.get("models") or []
    if isinstance(models_value, str):
        models = [models_value]
    elif isinstance(models_value, list):
        models = [str(item) for item in models_value if str(item or "").strip()]
    else:
        models = []
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input,
        "cache_creation_tokens": cache_creation,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "models": sorted(set(models)),
        "first_activity": row.get("firstActivity") or row.get("first_activity"),
        "last_activity": row.get("lastActivity") or row.get("last_activity"),
    }, []


def query_ccusage_snapshot(
    *,
    thread_id: str,
    ccusage_command: str | Sequence[str] | None = None,
) -> dict[str, Any]:
    """Query one Codex session through ccusage and return normalized evidence.

    Failures are represented explicitly rather than raised because accounting
    must never invent counters or block otherwise valid engineering lifecycle
    completion.
    """

    actual_thread = str(thread_id or "").strip()
    if not actual_thread:
        return {"status": "UNAVAILABLE", "problems": ["Codex thread ID is missing"], "source": {"tool": "ccusage"}}
    try:
        base = _command_tokens(ccusage_command)
    except ActorError as exc:
        return {"status": "UNAVAILABLE", "problems": [str(exc)], "source": {"tool": "ccusage"}}
    executable = base[0]
    if os.path.sep not in executable and shutil.which(executable) is None:
        return {
            "status": "UNAVAILABLE",
            "problems": [f"ccusage executable was not found: {executable}"],
            "source": {"tool": "ccusage", "command": base},
        }
    version, version_problems = _ccusage_version(base)
    command = [*base, "codex", "session", "--id", actual_thread, "--json", "--no-cost", "--offline"]
    try:
        result = _run_command(command)
    except subprocess.TimeoutExpired:
        return {
            "status": "UNAVAILABLE",
            "problems": [f"ccusage session query timed out after {CCUSAGE_TIMEOUT_SECONDS:g} seconds"],
            "source": {"tool": "ccusage", "version": version, "command": command},
        }
    except OSError as exc:
        return {
            "status": "UNAVAILABLE",
            "problems": [f"ccusage session query failed: {exc}"],
            "source": {"tool": "ccusage", "version": version, "command": command},
        }
    source = {
        "tool": "ccusage",
        "version": version,
        "version_warnings": version_problems,
        "command": command,
        "mode": "codex_session_json_no_cost_offline",
        "response_sha256": hashlib.sha256(result.stdout.encode("utf-8")).hexdigest(),
    }
    problems: list[str] = []
    if result.returncode != 0:
        problems.extend(version_problems)
        problems.append(result.stderr.strip() or f"ccusage exited {result.returncode}")
        return {"status": "UNAVAILABLE", "problems": problems, "source": source}
    try:
        decoded = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        problems.append(f"ccusage returned invalid JSON: {exc}")
        return {"status": "UNAVAILABLE", "problems": problems, "source": source}
    row = _select_session_row(decoded, thread_id=actual_thread)
    if row is None:
        problems.append(f"ccusage did not return session {actual_thread}")
        return {"status": "UNAVAILABLE", "problems": problems, "source": source}
    normalized, row_problems = _normalize_ccusage_row(row)
    problems.extend(row_problems)
    if normalized is None:
        return {"status": "UNAVAILABLE", "problems": problems, "source": source}
    return {"status": "MEASURED", "cumulative": normalized, "problems": problems, "source": source}


def _snapshot(
    *,
    kind: str,
    record: Mapping[str, Any],
    query: Mapping[str, Any],
    captured_at: str,
    correction_note: str | None,
) -> dict[str, Any]:
    payload = dict(record.get("payload") or {})
    epoch = dict(payload.get("epoch") or {})
    model_assignment = dict(payload.get("model_assignment") or {})
    body = {
        "schema_version": TOKEN_SNAPSHOT_SCHEMA_VERSION,
        "resource_dimension": RESOURCE_DIMENSION,
        "measurement_scope": "CUMULATIVE_CODEX_THREAD_COUNTER",
        "attribution_assurance": "EXACT_SESSION_EPOCH_THREAD",
        "capture_kind": kind,
        "status": str(query.get("status") or "UNAVAILABLE"),
        "work_id": payload.get("work_id"),
        "session_id": record.get("record_id"),
        "context_epoch_id": epoch.get("epoch_id"),
        "codex_thread_id": payload.get("current_codex_id"),
        "captured_at_utc": captured_at,
        "model": model_assignment.get("model"),
        "reasoning_level": model_assignment.get("reasoning_level"),
        "cumulative": dict(query.get("cumulative") or {}) if isinstance(query.get("cumulative"), Mapping) else None,
        "source": dict(query.get("source") or {}),
        "problems": [str(item) for item in (query.get("problems") or [])],
        "correction_note": str(correction_note or "").strip() or None,
        "correction_capture_id": None,
        "correction_assurance": "OPERATOR_ATTESTED_CORRECTION" if correction_note else None,
    }
    snapshot_id = stable_id("TOKSNAP", body)
    body["snapshot_id"] = snapshot_id
    body["snapshot_sha256"] = canonical_sha256(body)
    return body


def _snapshot_path(state_root: str | Path | None, snapshot: Mapping[str, Any]) -> Path:
    return (
        _resource_root(state_root)
        / "sessions"
        / str(snapshot.get("session_id"))
        / str(snapshot.get("context_epoch_id") or "NO-EPOCH")
        / f"{snapshot.get('snapshot_id')}.{str(snapshot.get('capture_kind')).lower()}_snapshot.json"
    )


def _write_snapshot(state_root: str | Path | None, snapshot: Mapping[str, Any]) -> dict[str, Any]:
    path = _write_immutable(_snapshot_path(state_root, snapshot), snapshot)
    return _artifact_reference(
        path,
        artifact_id=str(snapshot["snapshot_id"]),
        schema_version=TOKEN_SNAPSHOT_SCHEMA_VERSION,
        status=str(snapshot.get("status") or "UNAVAILABLE"),
        state_root=state_root,
    )


def _capture_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(record.get("payload") or {})
    epoch = dict(payload.get("epoch") or {})
    return {
        "work_id": payload.get("work_id"),
        "session_id": record.get("record_id"),
        "context_epoch_id": epoch.get("epoch_id"),
        "codex_thread_id": payload.get("current_codex_id"),
    }


def capture_start(
    record: Mapping[str, Any],
    *,
    state_root: str | Path | None,
    generated_at: str | None = None,
    ccusage_command: str | Sequence[str] | None = None,
    correction_note: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Capture one immutable cumulative START snapshot for an epoch/thread."""

    timestamp = utc_now(generated_at)
    payload = dict(record.get("payload") or {})
    accounting = _session_accounting_or_legacy_default(payload.get("resource_accounting"))
    identity = _capture_identity(record)
    matching = [
        row
        for row in accounting["captures"]
        if all(str(row.get(key) or "") == str(value or "") for key, value in identity.items())
    ]
    active_id = str(accounting.get("active_capture_id") or "")
    if active_id and not any(str(row.get("capture_id") or "") == active_id for row in matching):
        raise ActorError(
            "an active resource segment must be closed before a different Context Epoch or Codex thread starts",
            code="resource_capture_overlap",
            problems=[active_id],
        )
    if matching and not correction_note:
        return accounting, dict(matching[-1])
    retained_captures = [row for row in accounting["captures"] if row not in matching]
    query = query_ccusage_snapshot(
        thread_id=str(identity.get("codex_thread_id") or ""),
        ccusage_command=ccusage_command,
    )
    snapshot = _snapshot(
        kind="START",
        record=record,
        query=query,
        captured_at=timestamp,
        correction_note=correction_note,
    )
    reference = _write_snapshot(state_root, snapshot)
    capture_body = {
        **identity,
        "start_snapshot": reference,
        "started_at_utc": timestamp,
        "status": "STARTED" if snapshot["status"] == "MEASURED" else "UNAVAILABLE",
        "end_snapshot": None,
        "usage_artifact": None,
        "ended_at_utc": None,
        "problems": list(snapshot.get("problems") or []),
        "correction_note": str(correction_note or "").strip() or None,
        "supersedes_capture_ids": [str(row.get("capture_id")) for row in matching if row.get("capture_id")],
    }
    capture_id = stable_id("TOKCAP", capture_body)
    capture_body["capture_id"] = capture_id
    accounting["captures"] = [*retained_captures, capture_body]
    accounting["active_capture_id"] = capture_id if capture_body["status"] == "STARTED" else None
    return accounting, capture_body


def load_resource_artifact_value(
    raw: Any,
    *,
    expected_schema_version: str,
) -> dict[str, Any]:
    """Strictly load one immutable WRA artifact already decoded from JSON."""

    if not isinstance(raw, Mapping):
        raise ActorError("resource artifact must be an object", code="schema_record_malformed")
    require_explicit_schema_version(
        raw, expected=expected_schema_version, label="resource artifact"
    )
    allowed = RESOURCE_ARTIFACT_FIELDS.get(expected_schema_version)
    if allowed is None:
        raise ActorError(
            "resource artifact loader has no closed field contract for this version",
            code="resource_artifact_schema_contract_missing",
            problems=[expected_schema_version],
        )
    require_known_fields(
        raw,
        allowed=allowed,
        label=f"{expected_schema_version} resource artifact",
    )
    value = dict(raw)
    if expected_schema_version == TOKEN_SNAPSHOT_SCHEMA_VERSION and value.get("cumulative") is not None:
        cumulative = value.get("cumulative")
        if not isinstance(cumulative, Mapping):
            raise ActorError(
                "token snapshot cumulative counters must be an object or null",
                code="schema_record_malformed",
            )
        require_known_fields(
            cumulative,
            allowed=TOKEN_CUMULATIVE_FIELDS,
            label="token snapshot cumulative counters",
        )
    digest_field = {
        TOKEN_SNAPSHOT_SCHEMA_VERSION: "snapshot_sha256",
        SESSION_USAGE_SCHEMA_VERSION: "artifact_sha256",
        WORK_USAGE_SCHEMA_VERSION: "artifact_sha256",
        TOKEN_REPORT_SCHEMA_VERSION: "report_sha256",
    }.get(expected_schema_version)
    if digest_field:
        embedded = value.get(digest_field)
        material = dict(value)
        material.pop(digest_field, None)
        actual = canonical_sha256(material)
        if not isinstance(embedded, str) or embedded != actual:
            raise ActorError(
                "resource artifact canonical digest mismatch",
                code="resource_artifact_canonical_digest_mismatch",
            )
    return value


def load_token_snapshot(raw: Any) -> dict[str, Any]:
    return load_resource_artifact_value(raw, expected_schema_version=TOKEN_SNAPSHOT_SCHEMA_VERSION)


def load_session_token_usage(raw: Any) -> dict[str, Any]:
    return load_resource_artifact_value(raw, expected_schema_version=SESSION_USAGE_SCHEMA_VERSION)


def load_work_token_usage(raw: Any) -> dict[str, Any]:
    return load_resource_artifact_value(raw, expected_schema_version=WORK_USAGE_SCHEMA_VERSION)


def load_token_report_index(raw: Any) -> dict[str, Any]:
    return load_resource_artifact_value(raw, expected_schema_version=TOKEN_REPORT_SCHEMA_VERSION)


def _read_artifact(
    reference: Mapping[str, Any],
    *,
    expected_schema_version: str,
    state_root: str | Path | None = None,
    expected_sha: bool = True,
    expected_artifact_id: str | None = None,
    expected_work_id: str | None = None,
    expected_session_id: str | None = None,
    expected_context_epoch_id: str | None = None,
    expected_codex_thread_id: str | None = None,
    expected_status: str | None = None,
) -> dict[str, Any]:
    """Load and identity-bind one immutable WRA artifact.

    A matching file digest is not sufficient accounting truth: the retained
    reference, the artifact's own sealed identity, and the authoritative
    Work/Session/Epoch/thread owner must all agree.
    """

    require_explicit_schema_version(
        reference,
        expected=expected_schema_version,
        label="resource artifact reference",
    )
    candidates: list[Path] = []
    locator = str(reference.get("state_root_locator") or "").strip()
    if locator:
        candidate = (actor_state_root(state_root) / locator).resolve()
        try:
            candidate.relative_to(actor_state_root(state_root).resolve())
        except ValueError as exc:
            raise ActorError("resource artifact locator escapes actor state root", code="resource_artifact_reference_invalid") from exc
        candidates.append(candidate)
    advisory = Path(str(reference.get("path") or "")).expanduser()
    if str(advisory):
        candidates.append(advisory)
    path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0] if candidates else Path(""))
    if not path.is_file():
        raise ActorError(f"resource artifact is missing: {path}", code="resource_artifact_missing")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ActorError(f"cannot read resource artifact: {path}: {exc}", code="resource_artifact_invalid") from exc
    if not isinstance(value, dict):
        raise ActorError(f"resource artifact must be an object: {path}", code="resource_artifact_invalid")
    loaded = load_resource_artifact_value(value, expected_schema_version=expected_schema_version)

    expected_file_sha = str(reference.get("sha256") or "")
    if expected_sha:
        if not expected_file_sha:
            raise ActorError(
                f"resource artifact reference has no file digest: {path}",
                code="resource_artifact_reference_invalid",
            )
        if file_sha256(path) != expected_file_sha:
            raise ActorError(f"resource artifact digest mismatch: {path}", code="resource_artifact_digest_mismatch")

    id_field = {
        TOKEN_SNAPSHOT_SCHEMA_VERSION: "snapshot_id",
        SESSION_USAGE_SCHEMA_VERSION: "artifact_id",
        WORK_USAGE_SCHEMA_VERSION: "artifact_id",
        TOKEN_REPORT_SCHEMA_VERSION: "report_id",
    }.get(expected_schema_version, "artifact_id")
    artifact_id = str(loaded.get(id_field) or "")
    reference_id = str(reference.get("artifact_id") or "")
    if not reference_id:
        raise ActorError(
            f"resource artifact reference has no artifact_id: {path}",
            code="resource_artifact_reference_invalid",
        )
    if artifact_id != reference_id:
        raise ActorError(
            f"resource artifact reference identity mismatch: {reference_id} != {artifact_id}",
            code="resource_artifact_identity_mismatch",
        )
    if expected_artifact_id is not None and artifact_id != str(expected_artifact_id):
        raise ActorError(
            f"resource artifact ID mismatch: {artifact_id} != {expected_artifact_id}",
            code="resource_artifact_identity_mismatch",
        )

    reference_status = str(reference.get("status") or "")
    artifact_status = str(loaded.get("status") or "")
    if reference_status and reference_status != artifact_status:
        raise ActorError(
            f"resource artifact status mismatch: {reference_status} != {artifact_status}",
            code="resource_artifact_identity_mismatch",
        )
    if expected_status is not None and artifact_status != str(expected_status):
        raise ActorError(
            f"resource artifact status mismatch: {artifact_status} != {expected_status}",
            code="resource_artifact_identity_mismatch",
        )

    expected_owners = {
        "work_id": expected_work_id,
        "session_id": expected_session_id,
        "context_epoch_id": expected_context_epoch_id,
        "codex_thread_id": expected_codex_thread_id,
    }
    for field, expected_value in expected_owners.items():
        if expected_value is None:
            continue
        actual = str(loaded.get(field) or "")
        if actual != str(expected_value):
            raise ActorError(
                f"resource artifact {artifact_id} {field} attribution mismatch: {actual} != {expected_value}",
                code="resource_artifact_attribution_mismatch",
            )
    return loaded


def _elapsed_minutes(start: str | None, end: str | None) -> float | None:
    if not start or not end:
        return None
    try:
        left = dt.datetime.fromisoformat(validate_timestamp(start).replace("Z", "+00:00"))
        right = dt.datetime.fromisoformat(validate_timestamp(end).replace("Z", "+00:00"))
    except ActorError:
        return None
    return max(0.0, round((right - left).total_seconds() / 60.0, 3))


def _delta(start: Mapping[str, Any], end: Mapping[str, Any]) -> tuple[dict[str, int] | None, list[str]]:
    left = dict(start.get("cumulative") or {}) if isinstance(start.get("cumulative"), Mapping) else {}
    right = dict(end.get("cumulative") or {}) if isinstance(end.get("cumulative"), Mapping) else {}
    fields = (
        "input_tokens",
        "cached_input_tokens",
        "cache_creation_tokens",
        "output_tokens",
        "total_tokens",
    )
    problems: list[str] = []
    result: dict[str, int] = {}
    for field in fields:
        if field not in left or field not in right:
            problems.append(f"token counter unavailable for delta: {field}")
            continue
        difference = int(right[field]) - int(left[field])
        if difference < 0:
            problems.append(f"cumulative token counter decreased: {field}")
        result[field] = difference
    if problems:
        return None, problems
    return result, []


def _session_usage_artifact(
    *,
    record: Mapping[str, Any],
    capture: Mapping[str, Any],
    start: Mapping[str, Any],
    end: Mapping[str, Any],
    delta: Mapping[str, int] | None,
    problems: Iterable[str],
    generated_at: str,
    correction_note: str | None,
) -> dict[str, Any]:
    payload = dict(record.get("payload") or {})
    model_assignment = dict(payload.get("model_assignment") or {})
    context_health = dict(payload.get("context_health") or {})
    query_models = sorted(set([*(dict(start.get("cumulative") or {}).get("models") or []), *(dict(end.get("cumulative") or {}).get("models") or [])]))
    problem_list = list(dict.fromkeys(str(item) for item in problems if str(item).strip()))
    measured = delta is not None and not problem_list
    body = {
        "schema_version": SESSION_USAGE_SCHEMA_VERSION,
        "resource_dimension": RESOURCE_DIMENSION,
        "measurement_scope": "THREAD_INTERVAL_MEASURED",
        "attribution_assurance": "EXACT_SESSION_EPOCH_THREAD",
        "status": "MEASURED" if measured else "UNAVAILABLE",
        "work_id": payload.get("work_id"),
        "session_id": record.get("record_id"),
        "context_epoch_id": capture.get("context_epoch_id"),
        "codex_thread_id": capture.get("codex_thread_id"),
        "model": model_assignment.get("model") or (query_models[0] if len(query_models) == 1 else ("MIXED" if query_models else None)),
        "reasoning_level": model_assignment.get("reasoning_level"),
        "duration_minutes": _elapsed_minutes(capture.get("started_at_utc"), generated_at),
        "tool_calls": int(context_health.get("tool_call_count") or 0),
        "compactions": int(context_health.get("compaction_count") or 0),
        "input_tokens": dict(delta or {}).get("input_tokens"),
        "cached_input_tokens": dict(delta or {}).get("cached_input_tokens"),
        "cache_creation_tokens": dict(delta or {}).get("cache_creation_tokens"),
        "output_tokens": dict(delta or {}).get("output_tokens"),
        "total_tokens": dict(delta or {}).get("total_tokens"),
        "start_snapshot": dict(capture.get("start_snapshot") or {}),
        "end_snapshot": {
            "artifact_id": end.get("snapshot_id"),
            "schema_version": TOKEN_SNAPSHOT_SCHEMA_VERSION,
            "status": end.get("status"),
        },
        "measured_at_utc": generated_at,
        "source": dict(end.get("source") or start.get("source") or {"tool": "ccusage"}),
        "problems": problem_list,
        "correction_note": str(correction_note or "").strip() or None,
        "correction_capture_id": str(capture.get("capture_id") or "") if correction_note else None,
        "correction_assurance": "OPERATOR_ATTESTED_CORRECTION" if correction_note else None,
    }
    artifact_id = stable_id("SESTOK", body)
    body["artifact_id"] = artifact_id
    body["artifact_sha256"] = canonical_sha256(body)
    return body


def _session_usage_path(state_root: str | Path | None, artifact: Mapping[str, Any]) -> Path:
    return (
        _resource_root(state_root)
        / "sessions"
        / str(artifact.get("session_id"))
        / str(artifact.get("context_epoch_id") or "NO-EPOCH")
        / f"{artifact.get('artifact_id')}.session_token_usage.json"
    )


def _unavailable_query(reason: str, *, mode: str) -> dict[str, Any]:
    return {
        "status": "UNAVAILABLE",
        "problems": [str(reason)],
        "source": {"tool": "ccusage", "mode": mode},
    }


def _capture_close_intent(
    record: Mapping[str, Any],
    capture: Mapping[str, Any],
    *,
    state_root: str | Path | None,
    boundary_action: str,
) -> tuple[dict[str, Any], Path, bool]:
    body = {
        "schema_version": CAPTURE_CLOSE_INTENT_SCHEMA_VERSION,
        "intent_id": None,
        "work_id": dict(record.get("payload") or {}).get("work_id"),
        "session_id": record.get("record_id"),
        "capture_id": capture.get("capture_id"),
        "context_epoch_id": capture.get("context_epoch_id"),
        "codex_thread_id": capture.get("codex_thread_id"),
        "boundary_action": str(boundary_action),
        "boundary_record": {
            "record_id": record.get("record_id"),
            "revision": int(record.get("revision") or 0),
            "record_sha256": record.get("record_sha256"),
            "status": record.get("status"),
        },
    }
    body["intent_id"] = stable_id("WRAEND", {key: value for key, value in body.items() if key != "intent_id"})
    material = dict(body)
    body["intent_sha256"] = canonical_sha256(material)
    path = _resource_root(state_root) / "close-intents" / f"{body['intent_id']}.json"
    created = not path.exists()
    _write_immutable(path, body)
    return body, path, created


def _close_result_path(state_root: str | Path | None, intent_id: str) -> Path:
    return _resource_root(state_root) / "close-results" / f"{intent_id}.json"


def _write_close_result(
    *,
    state_root: str | Path | None,
    intent: Mapping[str, Any],
    end_ref: Mapping[str, Any],
    usage_ref: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "schema_version": CAPTURE_CLOSE_RESULT_SCHEMA_VERSION,
        "intent_id": intent.get("intent_id"),
        "capture_id": intent.get("capture_id"),
        "boundary_record": dict(intent.get("boundary_record") or {}),
        "end_snapshot": dict(end_ref),
        "usage_artifact": dict(usage_ref),
    }
    body["result_sha256"] = canonical_sha256(body)
    _write_immutable(_close_result_path(state_root, str(intent.get("intent_id") or "")), body)
    return body


def _load_close_result(
    *,
    state_root: str | Path | None,
    intent: Mapping[str, Any],
) -> dict[str, Any] | None:
    path = _close_result_path(state_root, str(intent.get("intent_id") or ""))
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    require_explicit_schema_version(raw, expected=CAPTURE_CLOSE_RESULT_SCHEMA_VERSION, label="WRA capture close result")
    material = dict(raw)
    observed = str(material.pop("result_sha256", "") or "")
    if observed != canonical_sha256(material):
        raise ActorError("WRA capture close result digest mismatch", code="resource_close_result_invalid")
    if str(raw.get("intent_id") or "") != str(intent.get("intent_id") or ""):
        raise ActorError("WRA capture close result identity mismatch", code="resource_close_result_invalid")
    return dict(raw)


def _adopt_close_result(
    record: Mapping[str, Any],
    capture: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    state_root: str | Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    usage_ref = dict(result.get("usage_artifact") or {})
    end_ref = dict(result.get("end_snapshot") or {})
    usage = _read_artifact(
        usage_ref,
        expected_schema_version=SESSION_USAGE_SCHEMA_VERSION,
        state_root=state_root,
        expected_artifact_id=str(usage_ref.get("artifact_id") or ""),
        expected_work_id=str(dict(record.get("payload") or {}).get("work_id") or ""),
        expected_session_id=str(record.get("record_id") or ""),
        expected_context_epoch_id=str(capture.get("context_epoch_id") or ""),
        expected_codex_thread_id=str(capture.get("codex_thread_id") or ""),
        expected_status=str(usage_ref.get("status") or "") or None,
    )
    closed = dict(capture)
    closed.update({
        "close_intent_id": result.get("intent_id"),
        "end_snapshot": end_ref,
        "usage_artifact": usage_ref,
        "ended_at_utc": usage.get("measured_at_utc"),
        "status": "COMPLETED" if usage.get("status") == "MEASURED" else "UNAVAILABLE",
        "problems": list(usage.get("problems") or []),
    })
    return closed, usage_ref


def _find_orphan_close_result(
    *,
    state_root: str | Path | None,
    intent: Mapping[str, Any],
) -> dict[str, Any] | None:
    directory = (
        _resource_root(state_root)
        / "sessions"
        / str(intent.get("session_id"))
        / str(intent.get("context_epoch_id") or "NO-EPOCH")
    )
    if not directory.is_dir():
        return None
    ends: list[tuple[Path, dict[str, Any]]] = []
    usages: list[tuple[Path, dict[str, Any]]] = []
    for path in directory.glob("*.json"):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if str(raw.get("close_intent_id") or "") != str(intent.get("intent_id") or ""):
            continue
        if raw.get("schema_version") == TOKEN_SNAPSHOT_SCHEMA_VERSION and raw.get("capture_kind") == "END":
            ends.append((path, load_token_snapshot(raw)))
        elif raw.get("schema_version") == SESSION_USAGE_SCHEMA_VERSION:
            usages.append((path, load_session_token_usage(raw)))
    if len(ends) != 1 or len(usages) != 1:
        if ends or usages:
            raise ActorError(
                "ambiguous orphan WRA close artifacts",
                code="resource_close_orphan_ambiguous",
                problems=[f"END snapshots: {len(ends)}", f"usage artifacts: {len(usages)}"],
            )
        return None
    end_path, end = ends[0]
    usage_path, usage = usages[0]
    end_ref = _artifact_reference(
        end_path,
        artifact_id=str(end.get("snapshot_id") or ""),
        schema_version=TOKEN_SNAPSHOT_SCHEMA_VERSION,
        status=str(end.get("status") or "UNAVAILABLE"),
    )
    usage_ref = _artifact_reference(
        usage_path,
        artifact_id=str(usage.get("artifact_id") or ""),
        schema_version=SESSION_USAGE_SCHEMA_VERSION,
        status=str(usage.get("status") or "UNAVAILABLE"),
    )
    return _write_close_result(state_root=state_root, intent=intent, end_ref=end_ref, usage_ref=usage_ref)


def _close_capture(
    record: Mapping[str, Any],
    capture: Mapping[str, Any],
    *,
    state_root: str | Path | None,
    timestamp: str,
    ccusage_command: str | Sequence[str] | None,
    correction_note: str | None,
    unavailable_reason: str | None = None,
    boundary_action: str = "RESOURCE_CAPTURE_END",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Close one capture at one explicit accounting boundary.

    A cumulative counter is useful only at the boundary where it was sampled.
    Callers that discover an already-ended historical epoch without an END
    snapshot must pass ``unavailable_reason`` rather than querying the current
    counter and retroactively attributing later work to the old epoch.
    """

    closed = dict(capture)
    intent, _intent_path, intent_created = _capture_close_intent(
        record,
        closed,
        state_root=state_root,
        boundary_action=boundary_action,
    )
    if not intent_created:
        retained = _load_close_result(state_root=state_root, intent=intent)
        if retained is None:
            retained = _find_orphan_close_result(state_root=state_root, intent=intent)
        if retained is not None:
            return _adopt_close_result(
                record,
                closed,
                retained,
                state_root=state_root,
            )
        unavailable_reason = unavailable_reason or (
            "durable WRA close intent exists but no exact boundary result survived; "
            "the current cumulative counter cannot reconstruct the historical boundary"
        )
    query = (
        _unavailable_query(unavailable_reason, mode="missing_epoch_boundary_end")
        if unavailable_reason
        else query_ccusage_snapshot(
            thread_id=str(closed.get("codex_thread_id") or ""),
            ccusage_command=ccusage_command,
        )
    )
    end_snapshot = _snapshot(
        kind="END",
        record={
            **dict(record),
            "payload": {
                **dict(record.get("payload") or {}),
                "current_codex_id": closed.get("codex_thread_id"),
                "epoch": {
                    **dict(dict(record.get("payload") or {}).get("epoch") or {}),
                    "epoch_id": closed.get("context_epoch_id"),
                },
            },
        },
        query=query,
        captured_at=timestamp,
        correction_note=correction_note,
    )
    end_snapshot.pop("snapshot_id", None)
    end_snapshot.pop("snapshot_sha256", None)
    end_snapshot["close_intent_id"] = intent.get("intent_id")
    end_snapshot["boundary_record"] = dict(intent.get("boundary_record") or {})
    end_snapshot["snapshot_id"] = stable_id("TOKSNAP", end_snapshot)
    end_snapshot["snapshot_sha256"] = canonical_sha256(end_snapshot)
    end_ref = _write_snapshot(state_root, end_snapshot)
    problems = [*list(closed.get("problems") or []), *list(end_snapshot.get("problems") or [])]
    try:
        start_ref = dict(closed.get("start_snapshot") or {})
        start_snapshot = _read_artifact(
            start_ref,
            expected_schema_version=TOKEN_SNAPSHOT_SCHEMA_VERSION,
            state_root=state_root,
            expected_artifact_id=str(start_ref.get("artifact_id") or ""),
            expected_work_id=str(closed.get("work_id") or ""),
            expected_session_id=str(closed.get("session_id") or ""),
            expected_context_epoch_id=str(closed.get("context_epoch_id") or ""),
            expected_codex_thread_id=str(closed.get("codex_thread_id") or ""),
            expected_status=str(start_ref.get("status") or "") or None,
        )
    except ActorError as exc:
        start_snapshot = {
            "status": "UNAVAILABLE",
            "cumulative": None,
            "source": {"tool": "ccusage", "mode": "invalid_start_artifact"},
        }
        problems.append(str(exc))
    delta, delta_problems = _delta(start_snapshot, end_snapshot)
    problems.extend(delta_problems)
    usage = _session_usage_artifact(
        record=record,
        capture=closed,
        start=start_snapshot,
        end=end_snapshot,
        delta=delta,
        problems=problems,
        generated_at=timestamp,
        correction_note=correction_note,
    )
    usage.pop("artifact_id", None)
    usage.pop("artifact_sha256", None)
    usage["capture_id"] = closed.get("capture_id")
    usage["close_intent_id"] = intent.get("intent_id")
    usage["boundary_record"] = dict(intent.get("boundary_record") or {})
    usage["artifact_id"] = stable_id("SESTOK", usage)
    usage["artifact_sha256"] = canonical_sha256(usage)
    usage_path = _write_immutable(_session_usage_path(state_root, usage), usage)
    usage_ref = _artifact_reference(
        usage_path,
        artifact_id=str(usage["artifact_id"]),
        schema_version=SESSION_USAGE_SCHEMA_VERSION,
        status=str(usage["status"]),
        state_root=state_root,
    )
    _write_close_result(state_root=state_root, intent=intent, end_ref=end_ref, usage_ref=usage_ref)
    closed.update(
        {
            "close_intent_id": intent.get("intent_id"),
            "end_snapshot": end_ref,
            "usage_artifact": usage_ref,
            "ended_at_utc": timestamp,
            "status": "COMPLETED" if usage["status"] == "MEASURED" else "UNAVAILABLE",
            "problems": list(usage.get("problems") or []),
        }
    )
    return closed, usage_ref


def finalize_active_capture(
    record: Mapping[str, Any],
    *,
    state_root: str | Path | None,
    generated_at: str | None = None,
    ccusage_command: str | Sequence[str] | None = None,
    correction_note: str | None = None,
    unavailable_reason: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Close only the currently active resource segment.

    This is the primitive used at Context Epoch, authority-phase, and Session
    terminal boundaries.  It deliberately does not close older captures with
    the same current cumulative counter, because doing so would overlap token
    intervals and double-count Work usage.
    """

    timestamp = utc_now(generated_at)
    payload = dict(record.get("payload") or {})
    accounting = _session_accounting_or_legacy_default(payload.get("resource_accounting"))
    active_id = str(accounting.get("active_capture_id") or "")
    if not active_id:
        return accounting, []
    updated: list[dict[str, Any]] = []
    completed_refs: list[dict[str, Any]] = []
    found = False
    for retained in accounting["captures"]:
        capture = dict(retained)
        if str(capture.get("capture_id") or "") != active_id:
            updated.append(capture)
            continue
        found = True
        if capture.get("usage_artifact"):
            existing_ref = dict(capture["usage_artifact"])
            updated.append(capture)
            completed_refs.append(existing_ref)
            continue
        closed, usage_ref = _close_capture(
            record,
            capture,
            state_root=state_root,
            timestamp=timestamp,
            ccusage_command=ccusage_command,
            correction_note=correction_note,
            unavailable_reason=unavailable_reason,
        )
        updated.append(closed)
        completed_refs.append(usage_ref)
    if not found:
        raise ActorError(
            "active resource capture is missing from retained captures",
            code="resource_active_capture_missing",
            problems=[active_id],
        )
    accounting["captures"] = updated
    accounting["active_capture_id"] = None
    accounting["usage_artifacts"] = [
        *accounting["usage_artifacts"],
        *[ref for ref in completed_refs if ref not in accounting["usage_artifacts"]],
    ]
    accounting["latest_usage_artifact"] = (
        completed_refs[-1] if completed_refs else accounting.get("latest_usage_artifact")
    )
    return accounting, completed_refs


def finalize_session(
    record: Mapping[str, Any],
    *,
    state_root: str | Path | None,
    generated_at: str | None = None,
    ccusage_command: str | Sequence[str] | None = None,
    correction_note: str | None = None,
    unavailable_reason: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Close the active Session segment without overlapping old epochs.

    Old open captures that no longer own ``active_capture_id`` cannot be
    reconstructed from today's cumulative counter.  They are sealed as
    explicit UNAVAILABLE evidence instead of being measured late and counted
    again.  A legacy Session with no START snapshot follows the same rule.
    """

    timestamp = utc_now(generated_at)
    payload = dict(record.get("payload") or {})
    accounting = _session_accounting_or_legacy_default(payload.get("resource_accounting"))
    if not accounting["captures"]:
        created_at = str(record.get("created_at_utc") or timestamp)
        missing_query = _unavailable_query(
            "WRA-1 START snapshot is missing; historical Session usage cannot be reconstructed from one cumulative END snapshot",
            mode="missing_start_evidence",
        )
        start_snapshot = _snapshot(
            kind="START",
            record=record,
            query=missing_query,
            captured_at=created_at,
            correction_note=correction_note,
        )
        start_reference = _write_snapshot(state_root, start_snapshot)
        capture_body = {
            **_capture_identity(record),
            "start_snapshot": start_reference,
            "started_at_utc": created_at,
            "status": "UNAVAILABLE",
            "end_snapshot": None,
            "usage_artifact": None,
            "ended_at_utc": None,
            "problems": list(start_snapshot.get("problems") or []),
            "correction_note": str(correction_note or "").strip() or None,
        }
        capture_body["capture_id"] = stable_id("TOKCAP", capture_body)
        accounting["captures"] = [capture_body]
        accounting["active_capture_id"] = None

    # A later cumulative counter is not an historical boundary.  Corrections
    # may close the capture that is active *now*, but they may never re-sample
    # an already closed or orphaned historical segment from current ccusage
    # totals.  Doing so would manufacture a later boundary and silently include
    # unrelated work from subsequent epochs.
    if correction_note and not accounting.get("active_capture_id"):
        raise ActorError(
            "resource correction requires an explicitly active capture boundary",
            code="resource_correction_boundary_missing",
            problems=[
                "Historical captures remain immutable.",
                "Start a new capture or attach an externally measured correction artifact with exact START/END snapshots.",
            ],
        )

    # Close the one interval that is known to be active at this boundary.
    candidate = dict(record)
    candidate["payload"] = {**payload, "resource_accounting": accounting}
    accounting, completed_refs = finalize_active_capture(
        candidate,
        state_root=state_root,
        generated_at=timestamp,
        ccusage_command=ccusage_command,
        correction_note=correction_note,
        unavailable_reason=unavailable_reason,
    )

    # Seal any orphaned historical open capture without querying the current
    # cumulative total.  Such a query would include work from later epochs.
    updated: list[dict[str, Any]] = []
    for retained in accounting["captures"]:
        capture = dict(retained)
        if capture.get("usage_artifact"):
            updated.append(capture)
            continue
        closed, usage_ref = _close_capture(
            {**dict(record), "payload": {**payload, "resource_accounting": accounting}},
            capture,
            state_root=state_root,
            timestamp=timestamp,
            ccusage_command=ccusage_command,
            correction_note=correction_note,
            unavailable_reason=(
                "resource segment ended without an END snapshot at its Context Epoch boundary; "
                "current cumulative counters cannot safely reconstruct the historical interval"
            ),
        )
        updated.append(closed)
        completed_refs.append(usage_ref)
    accounting["captures"] = updated
    accounting["active_capture_id"] = None
    accounting["usage_artifacts"] = [
        *accounting["usage_artifacts"],
        *[ref for ref in completed_refs if ref not in accounting["usage_artifacts"]],
    ]
    accounting["latest_usage_artifact"] = (
        completed_refs[-1] if completed_refs else accounting.get("latest_usage_artifact")
    )
    if not completed_refs and isinstance(accounting.get("latest_usage_artifact"), Mapping):
        # Idempotent retries return the retained current artifact rather than
        # reporting an empty, misleading UNAVAILABLE result.
        completed_refs = [dict(accounting["latest_usage_artifact"])]
    return accounting, completed_refs


def _uniform(values: Iterable[str | None]) -> str | None:
    rows = sorted(set(str(value).strip() for value in values if str(value or "").strip()))
    if not rows:
        return None
    return rows[0] if len(rows) == 1 else "MIXED"


def estimation_error(estimate: Mapping[str, Any], actual_tokens: int | None) -> dict[str, Any] | None:
    if actual_tokens is None:
        return None
    tokens = dict(estimate.get("estimated_tokens") or {})
    expected = tokens.get("expected")
    low = tokens.get("low")
    high = tokens.get("high")
    token_delta = actual_tokens - int(expected) if expected is not None else None
    percent = round(token_delta / int(expected) * 100.0, 3) if expected not in (None, 0) and token_delta is not None else None
    within = bool(int(low) <= actual_tokens <= int(high)) if low is not None and high is not None else None
    return {
        "expected_tokens": expected,
        "actual_tokens": actual_tokens,
        "token_delta": token_delta,
        "percent_error": percent,
        "within_low_high": within,
        "confidence_was_correct": within,
    }


def aggregate_work_usage(
    work_record: Mapping[str, Any],
    session_records: Iterable[Mapping[str, Any]],
    *,
    state_root: str | Path | None,
    generated_at: str | None = None,
    correction_note: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create one immutable Work-level token_usage.json from linked Session evidence."""

    timestamp = utc_now(generated_at)
    work_payload = dict(work_record.get("payload") or {})
    estimate = _resource_estimate_or_legacy_default(
        work_payload.get("resource_estimate"),
        generated_at=str(work_record.get("created_at_utc") or timestamp),
    )
    session_rows: list[dict[str, Any]] = []
    problems: list[str] = []
    measured: list[dict[str, Any]] = []
    tool_calls = 0
    compactions = 0
    for record in sorted(session_records, key=lambda row: str(row.get("record_id") or "")):
        payload = dict(record.get("payload") or {})
        accounting = _session_accounting_or_legacy_default(payload.get("resource_accounting"))
        binding_errors = validate_session_resource_accounting_binding(
            accounting,
            session_record=record,
            work_record=work_record,
            state_root=state_root,
        )
        if binding_errors:
            problems.extend(f"Session {record.get('record_id')} WRA attribution/boundary: {item}" for item in binding_errors)
            session_rows.append({"session_id": record.get("record_id"), "status": "UNAVAILABLE", "artifacts": []})
            continue
        # ``usage_artifacts`` is immutable history. Attribution uses only the
        # current artifact bound by each capture so corrections never double
        # count an old measurement.
        refs = [
            dict(capture.get("usage_artifact") or {})
            for capture in accounting.get("captures") or []
            if isinstance(capture, Mapping) and isinstance(capture.get("usage_artifact"), Mapping)
        ]
        if not refs:
            problems.append(f"Session {record.get('record_id')} has no token usage artifact")
            session_rows.append({"session_id": record.get("record_id"), "status": "UNAVAILABLE", "artifacts": []})
            continue
        session_artifacts: list[dict[str, Any]] = []
        for ref in refs:
            try:
                artifact = _read_artifact(
                    ref,
                    expected_schema_version=SESSION_USAGE_SCHEMA_VERSION,
                    state_root=state_root,
                    expected_artifact_id=str(ref.get("artifact_id") or ""),
                    expected_work_id=str(work_record.get("record_id") or ""),
                    expected_session_id=str(record.get("record_id") or ""),
                    expected_context_epoch_id=str(dict(payload.get("epoch") or {}).get("epoch_id") or ""),
                    expected_codex_thread_id=str(payload.get("current_codex_id") or ""),
                    expected_status=str(ref.get("status") or "") or None,
                )
            except ActorError as exc:
                problems.append(str(exc))
                continue
            session_artifacts.append(artifact)
            tool_calls += int(artifact.get("tool_calls") or 0)
            compactions += int(artifact.get("compactions") or 0)
            if artifact.get("status") == "MEASURED":
                measured.append(artifact)
            else:
                problems.extend(str(item) for item in artifact.get("problems") or [])
        session_rows.append(
            {
                "session_id": record.get("record_id"),
                "status": (
                    "MEASURED"
                    if session_artifacts and all(row.get("status") == "MEASURED" for row in session_artifacts)
                    else ("PARTIAL" if any(row.get("status") == "MEASURED" for row in session_artifacts) else "UNAVAILABLE")
                ),
                "artifacts": [
                    {
                        "artifact_id": row.get("artifact_id"),
                        "status": row.get("status"),
                        "total_tokens": row.get("total_tokens"),
                        "model": row.get("model"),
                        "reasoning_level": row.get("reasoning_level"),
                    }
                    for row in session_artifacts
                ],
            }
        )
    all_session_count = len(session_rows)
    measured_count = len(measured)
    if measured_count == 0:
        status = "UNAVAILABLE"
    elif any(row.get("status") != "MEASURED" for row in session_rows):
        status = "PARTIAL"
    else:
        status = "MEASURED"
    sums: dict[str, int | None] = {}
    for field in ("input_tokens", "cached_input_tokens", "cache_creation_tokens", "output_tokens", "total_tokens"):
        sums[field] = sum(int(row.get(field) or 0) for row in measured) if measured else None
    actual = sums["total_tokens"]
    error = estimation_error(estimate, int(actual) if actual is not None else None)
    durations = [float(row.get("duration_minutes") or 0.0) for row in measured if row.get("duration_minutes") is not None]
    body = {
        "schema_version": WORK_USAGE_SCHEMA_VERSION,
        "resource_dimension": RESOURCE_DIMENSION,
        "measurement_scope": "AGGREGATED_SESSION_THREAD_INTERVALS",
        "attribution_assurance": "EXACT_WHERE_MEASURED_PARTIAL_WHERE_MISSING",
        "status": status,
        "work_id": work_record.get("record_id"),
        "model": _uniform(row.get("model") for row in measured),
        "reasoning_level": _uniform(row.get("reasoning_level") for row in measured),
        "work_type": work_payload.get("work_type"),
        "subsystem": work_payload.get("subsystem"),
        "duration_minutes": round(sum(durations), 3) if durations else None,
        "tool_calls": tool_calls,
        "compactions": compactions,
        "input_tokens": sums["input_tokens"],
        "cached_input_tokens": sums["cached_input_tokens"],
        "cache_creation_tokens": sums["cache_creation_tokens"],
        "output_tokens": sums["output_tokens"],
        "total_tokens": actual,
        "estimated_tokens": dict(estimate.get("estimated_tokens") or {}),
        "actual_tokens": actual,
        "estimation_error": error,
        "source": {
            "tool": "ccusage",
            "versions": sorted(set(str(dict(row.get("source") or {}).get("version") or "unknown") for row in measured)),
            "integration": "EIEL WRA-1",
        },
        "session_count": all_session_count,
        "measured_segment_count": measured_count,
        "session_breakdown": session_rows,
        "problems": list(dict.fromkeys(problems)),
        "generated_at_utc": timestamp,
        "correction_note": str(correction_note or "").strip() or None,
        "correction_assurance": "OPERATOR_ATTESTED_CORRECTION" if correction_note else None,
    }
    artifact_id = stable_id("WRA", body)
    body["artifact_id"] = artifact_id
    body["artifact_sha256"] = canonical_sha256(body)
    path = _write_immutable(
        _resource_root(state_root) / "work" / str(work_record.get("record_id")) / f"{artifact_id}.token_usage.json",
        body,
    )
    reference = _artifact_reference(
        path,
        artifact_id=artifact_id,
        schema_version=WORK_USAGE_SCHEMA_VERSION,
        status=status,
        state_root=state_root,
    )
    return body, reference


def work_accounting_projection(
    work_payload: Mapping[str, Any],
    *,
    artifact_reference: Mapping[str, Any],
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    estimate = _resource_estimate_or_legacy_default(work_payload.get("resource_estimate"))
    current = _work_accounting_or_legacy_default(
        work_payload.get("resource_accounting"),
        estimate=estimate,
    )
    previous = current.get("artifact")
    history = list(current.get("artifact_history") or [])
    if isinstance(previous, Mapping) and previous and dict(previous) not in history:
        history.append(dict(previous))
    current.update(
        {
            "artifact": dict(artifact_reference),
            "estimation_error": dict(artifact.get("estimation_error") or {}) if isinstance(artifact.get("estimation_error"), Mapping) else None,
            "artifact_history": history,
        }
    )
    return current


def export_token_report(
    work_records: Iterable[Mapping[str, Any]],
    *,
    output_dir: str | Path,
    state_root: str | Path | None = None,
    generated_at: str | None = None,
) -> tuple[dict[str, Any], Path]:
    """Export attached Work token artifacts and a deterministic index."""

    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for record in sorted(work_records, key=lambda row: str(row.get("record_id") or "")):
        payload = dict(record.get("payload") or {})
        estimate = _resource_estimate_or_legacy_default(
            payload.get("resource_estimate"),
            generated_at=str(record.get("created_at_utc") or utc_now(generated_at)),
        )
        accounting = _work_accounting_or_legacy_default(
            payload.get("resource_accounting"),
            estimate=estimate,
        )
        reference = accounting.get("artifact")
        if not isinstance(reference, Mapping) or not reference:
            continue
        artifact = _read_artifact(
            reference,
            expected_schema_version=WORK_USAGE_SCHEMA_VERSION,
            state_root=state_root,
            expected_artifact_id=str(reference.get("artifact_id") or ""),
            expected_work_id=str(record.get("record_id") or ""),
            expected_status=str(reference.get("status") or "") or None,
        )
        work_dir = root / str(record.get("record_id"))
        target = work_dir / "token_usage.json"
        atomic_write_json(target, artifact)
        rows.append(
            {
                "work_id": record.get("record_id"),
                "status": artifact.get("status"),
                "artifact_id": artifact.get("artifact_id"),
                "source_sha256": reference.get("sha256"),
                "export_path": str(target),
                "total_tokens": artifact.get("total_tokens"),
                "estimation_error": artifact.get("estimation_error"),
            }
        )
    index_body = {
        "schema_version": TOKEN_REPORT_SCHEMA_VERSION,
        "generated_at_utc": utc_now(generated_at),
        "work_item_count": len(rows),
        "work_items": rows,
    }
    index_body["report_sha256"] = canonical_sha256(index_body)
    index_path = atomic_write_json(root / "token_report_index.json", index_body)
    return index_body, index_path


def resource_accounting_coverage(
    work_records: Iterable[Mapping[str, Any]],
    session_records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Report WRA coverage without turning accounting into a lifecycle gate."""

    works = [dict(row) for row in work_records]
    sessions = [dict(row) for row in session_records]
    terminal_works = [row for row in works if str(row.get("status") or "") in {"COMPLETED", "CANCELLED", "SUPERSEDED"}]
    measured_segments = 0
    total_segments = 0
    unmanaged_sessions: list[str] = []
    for record in sessions:
        accounting = _session_accounting_or_legacy_default(dict(record.get("payload") or {}).get("resource_accounting"))
        captures = [dict(row) for row in accounting.get("captures") or [] if isinstance(row, Mapping)]
        total_segments += len(captures)
        measured_segments += sum(1 for row in captures if str(row.get("status") or "") == "COMPLETED" and isinstance(row.get("usage_artifact"), Mapping))
        if not captures:
            unmanaged_sessions.append(str(record.get("record_id") or ""))
    coverage = round(measured_segments / total_segments, 6) if total_segments else None
    body = {
        "schema_version": "eiel_wra_coverage_report_v1",
        "resource_dimension": RESOURCE_DIMENSION,
        "lifecycle_gate": False,
        "terminal_work_count": len(terminal_works),
        "session_count": len(sessions),
        "resource_segment_count": total_segments,
        "measured_segment_count": measured_segments,
        "unmanaged_session_ids": sorted(value for value in unmanaged_sessions if value),
        "coverage_fraction": coverage,
    }
    body["report_sha256"] = canonical_sha256(body)
    return body


def retain_estimator_review_receipt(
    *,
    state_root: str | Path | None,
    report_path: str | Path,
    rules_path: str | Path,
    included_artifacts: Sequence[Mapping[str, Any]],
    generated_at: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    report = Path(report_path).expanduser().resolve()
    rules = Path(rules_path).expanduser().resolve()
    if not report.is_file() or not rules.is_file():
        raise ActorError("estimator review requires report and rules files", code="estimator_review_input_missing")
    body = {
        "schema_version": ESTIMATOR_REVIEW_RECEIPT_SCHEMA_VERSION,
        "report_sha256": file_sha256(report),
        "rules_before_sha256": file_sha256(rules),
        "included_artifacts": [dict(row) for row in included_artifacts],
        "reviewed_at_utc": utc_now(generated_at),
        "status": "PENDING_HUMAN_APPROVAL",
    }
    body["receipt_id"] = stable_id("ESTREV", body)
    body["receipt_sha256"] = canonical_sha256(body)
    path = _write_immutable(_resource_root(state_root) / "estimator-reviews" / f"{body['receipt_id']}.json", body)
    return body, _artifact_reference(path, artifact_id=body["receipt_id"], schema_version=ESTIMATOR_REVIEW_RECEIPT_SCHEMA_VERSION, status=body["status"], state_root=state_root)


def retain_estimator_approval_receipt(
    *,
    state_root: str | Path | None,
    review_reference: Mapping[str, Any],
    rules_after_path: str | Path,
    authority: str,
    decision: str,
    generated_at: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rules = Path(rules_after_path).expanduser().resolve()
    if not rules.is_file():
        raise ActorError("approved estimator rules file is missing", code="estimator_rules_missing")
    decision_value = str(decision or "").upper()
    if decision_value not in {"APPROVED", "REJECTED"}:
        raise ActorError("estimator approval decision is invalid", code="estimator_approval_invalid")
    if not str(authority or "").strip():
        raise ActorError("estimator approval requires human authority", code="estimator_approval_authority_missing")
    body = {
        "schema_version": ESTIMATOR_APPROVAL_RECEIPT_SCHEMA_VERSION,
        "review_receipt": dict(review_reference),
        "decision": decision_value,
        "human_authority": str(authority).strip(),
        "rules_after_sha256": file_sha256(rules),
        "decided_at_utc": utc_now(generated_at),
    }
    body["receipt_id"] = stable_id("ESTAPP", body)
    body["receipt_sha256"] = canonical_sha256(body)
    path = _write_immutable(_resource_root(state_root) / "estimator-approvals" / f"{body['receipt_id']}.json", body)
    return body, _artifact_reference(path, artifact_id=body["receipt_id"], schema_version=ESTIMATOR_APPROVAL_RECEIPT_SCHEMA_VERSION, status=decision_value, state_root=state_root)


__all__ = [
    "RESOURCE_ESTIMATE_SCHEMA_VERSION",
    "SESSION_ACCOUNTING_SCHEMA_VERSION",
    "SESSION_USAGE_SCHEMA_VERSION",
    "TOKEN_REPORT_SCHEMA_VERSION",
    "TOKEN_SNAPSHOT_SCHEMA_VERSION",
    "WORK_ACCOUNTING_SCHEMA_VERSION",
    "WORK_USAGE_SCHEMA_VERSION",
    "aggregate_work_usage",
    "capture_start",
    "estimation_error",
    "export_token_report",
    "finalize_active_capture",
    "finalize_session",
    "load_resource_artifact_value",
    "load_resource_estimate",
    "load_session_resource_accounting",
    "load_session_token_usage",
    "load_token_report_index",
    "load_token_snapshot",
    "load_work_resource_accounting",
    "load_work_token_usage",
    "normalize_resource_estimate",
    "normalize_session_resource_accounting",
    "normalize_work_resource_accounting",
    "query_ccusage_snapshot",
    "require_resource_thread_available",
    "resource_thread_conflicts",
    "resource_thread_lock",
    "validate_resource_estimate",
    "validate_session_resource_accounting",
    "validate_session_resource_accounting_binding",
    "work_accounting_projection",
]
