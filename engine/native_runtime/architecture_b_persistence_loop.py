"""Pull-only persistence adapter for Architecture B boundary snapshots."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypedDict

from .architecture_b_reporting import (
    SNAPSHOT_ARTIFACT_MANIFEST_NAME,
    SNAPSHOT_FAILURE_RECORD_NAME,
    SNAPSHOT_STAGING_DIRECTORY,
    persist_architecture_b_snapshot,
)
from .live_chronicle_stream import enqueue_event, event_from_snapshot


SCHEMA_VERSION = "eiel_architecture_b_persistence_loop_result_v1"
SNAPSHOT_BATCH_SCHEMA_VERSION = "eiel_native_architecture_b_snapshot_batch_v1"
ACKNOWLEDGEMENT_RECEIPT_SCHEMA_VERSION = (
    "eiel_native_architecture_b_persistence_acknowledgement_receipt_v1"
)

_BATCH_FIELDS = frozenset(
    {
        "controller_id",
        "owner_epoch",
        "after_snapshot_sequence",
        "snapshots",
        "highest_snapshot_sequence",
        "terminal_state_seen",
        "batch_payload_bytes",
    }
)
_ACKNOWLEDGEMENT_FIELDS = frozenset(
    {
        "api_version",
        "request_id",
        "expected_controller_id",
        "expected_owner_epoch",
        "snapshot_id",
        "snapshot_digest_sha256",
        "outcome",
        "artifact_manifest_digest_sha256",
        "acknowledged_channel_cursor_ranges",
        "failure_code",
    }
)
_RECEIPT_FIELDS = frozenset(
    {
        "request_id",
        "controller_id",
        "owner_epoch",
        "snapshot_id",
        "disposition",
        "persistence_barrier_released",
        "terminal_status_changed_to_failed",
    }
)
_TERMINAL_STATES = frozenset({"run_complete", "cancelled", "failed"})
_EXACT_REPORTING_CURSOR_CHANNELS = frozenset(
    {
        "exact_reporting",
        "behaviors",
        "metrics",
        "lineage",
        "typed_lifecycle",
    }
)
_SCHEDULE_FIELDS = frozenset(
    {
        "schema_version",
        "schedule_origin_generation",
        "imported_boundary_ordinal",
        "progress_every_generations",
        "report_every_generations",
        "checkpoint_every_generations",
        "explicit_next_generations",
    }
)


class ArchitectureBPersistenceLoopError(RuntimeError):
    """Raised after the pull adapter fails closed and notifies the native owner."""


class ArchitectureBPersistenceProtocolError(ValueError):
    """The native mailbox returned a value outside its closed protocol."""


class _ArchitectureBPersistenceReportedFailure(RuntimeError):
    """The production reporter returned a validated, durable failure result."""


class ArchitectureBPersistenceLoopResult(TypedDict):
    schema_version: str
    stop_reason: str
    terminal_state_seen: str
    snapshots_seen: int
    progress_only_snapshots: int
    skipped_snapshot_sequences: int
    persistence_acknowledgements: int
    identical_acknowledgement_retries: int
    checkpoints_persisted: int
    architecture_b_checkpoint_bundle_paths: list[str]
    last_snapshot_sequence: int
    last_progress: dict[str, Any] | None


def _protocol_error(message: str) -> ArchitectureBPersistenceProtocolError:
    return ArchitectureBPersistenceProtocolError(
        f"EIEL-ARCHITECTURE-B-PERSISTENCE-PROTOCOL-FAILED: {message}"
    )


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _protocol_error(f"{field} must be a mapping")
    return dict(value)


def _closed_mapping(
    value: Any,
    *,
    field: str,
    required_fields: frozenset[str],
    optional_fields: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    resolved = _mapping(value, field=field)
    actual = frozenset(resolved)
    missing = sorted(required_fields - actual)
    unexpected = sorted(actual - required_fields - optional_fields)
    if missing or unexpected:
        raise _protocol_error(
            f"{field} fields mismatch: missing={missing} unexpected={unexpected}"
        )
    return resolved


def _strict_int(value: Any, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise _protocol_error(f"{field} must be an integer at least {minimum}")
    return value


def _identifier(value: Any, *, field: str) -> str:
    if type(value) is not str or not value.strip():
        raise _protocol_error(f"{field} must be a non-empty string")
    return value


def _sha256(value: Any, *, field: str) -> str:
    resolved = _identifier(value, field=field)
    if len(resolved) != 64 or any(char not in "0123456789abcdef" for char in resolved):
        raise _protocol_error(f"{field} must be lowercase SHA-256")
    return resolved


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise _protocol_error(f"value is not canonical JSON: {exc}") from exc


def _due_flag(due_mask: Mapping[str, Any], *names: str) -> bool:
    values: list[bool] = []
    for name in names:
        if name not in due_mask:
            continue
        value = due_mask[name]
        if type(value) is not bool:
            raise _protocol_error(f"snapshot.due_mask.{name} must be boolean")
        values.append(value)
    if len(set(values)) > 1:
        raise _protocol_error(f"snapshot.due_mask aliases disagree: {list(names)}")
    return values[0] if values else False


def _cancellation_checkpoint_observed(snapshot: Mapping[str, Any]) -> bool:
    """Validate the native cancellation checkpoint override without owner reads."""

    observed_present = "cancellation_observed_at_snapshot_cutoff" in snapshot
    requested_present = "cancellation_checkpoint_requested" in snapshot
    if observed_present != requested_present:
        raise _protocol_error(
            "snapshot cancellation observation and checkpoint request must be "
            "declared together"
        )
    observed = snapshot.get("cancellation_observed_at_snapshot_cutoff", False)
    if type(observed) is not bool:
        raise _protocol_error(
            "snapshot.cancellation_observed_at_snapshot_cutoff must be boolean"
        )
    requested = snapshot.get("cancellation_checkpoint_requested", False)
    if type(requested) is not bool:
        raise _protocol_error(
            "snapshot.cancellation_checkpoint_requested must be boolean"
        )
    if requested is not observed:
        raise _protocol_error(
            "snapshot cancellation checkpoint request contradicts its observation"
        )
    for field in ("success_marker_allowed", "promotion_result_allowed"):
        if observed and field not in snapshot:
            raise _protocol_error(f"snapshot.{field} is required for cancellation")
        if field in snapshot and snapshot[field] is not False:
            raise _protocol_error(f"snapshot.{field} must be false")
    return observed


def _cursor_ranges(value: Any, *, field: str) -> dict[str, list[int]]:
    raw = _mapping(value, field=field)
    ranges: dict[str, list[int]] = {}
    for channel, interval in raw.items():
        if type(channel) is not str or not channel:
            raise _protocol_error(f"{field} contains an invalid channel name")
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise _protocol_error(f"{field}.{channel} must be a two-value range")
        start = _strict_int(interval[0], field=f"{field}.{channel}[0]")
        stop = _strict_int(interval[1], field=f"{field}.{channel}[1]")
        if stop < start:
            raise _protocol_error(f"{field}.{channel} is reversed")
        ranges[channel] = [start, stop]
    return ranges


def _validated_snapshot_schedule(
    snapshot: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    int,
    int,
    int,
    int,
    frozenset[int],
    bool | None,
]:
    schedule = _closed_mapping(
        snapshot.get("schedule_state"),
        field="snapshot.schedule_state",
        required_fields=_SCHEDULE_FIELDS,
        optional_fields=frozenset({"checkpoint_on_cancel"}),
    )
    if schedule["schema_version"] != "eiel_native_snapshot_schedule_v1":
        raise _protocol_error("snapshot schedule_state schema_version mismatch")
    origin = _strict_int(
        schedule["schedule_origin_generation"],
        field="snapshot.schedule_state.schedule_origin_generation",
    )
    imported_ordinal = _strict_int(
        schedule["imported_boundary_ordinal"],
        field="snapshot.schedule_state.imported_boundary_ordinal",
    )
    _strict_int(
        schedule["progress_every_generations"],
        field="snapshot.schedule_state.progress_every_generations",
    )
    report_cadence = _strict_int(
        schedule["report_every_generations"],
        field="snapshot.schedule_state.report_every_generations",
    )
    checkpoint_cadence = _strict_int(
        schedule["checkpoint_every_generations"],
        field="snapshot.schedule_state.checkpoint_every_generations",
    )
    raw_explicit = schedule["explicit_next_generations"]
    if not isinstance(raw_explicit, list):
        raise _protocol_error(
            "snapshot.schedule_state.explicit_next_generations must be a list"
        )
    explicit = [
        _strict_int(
            value,
            field=(
                "snapshot.schedule_state.explicit_next_generations"
                f"[{index}]"
            ),
        )
        for index, value in enumerate(raw_explicit)
    ]
    if explicit != sorted(set(explicit)):
        raise _protocol_error(
            "snapshot.schedule_state.explicit_next_generations must be sorted "
            "and unique"
        )
    if any(generation <= origin for generation in explicit):
        raise _protocol_error(
            "snapshot.schedule_state.explicit_next_generations must contain "
            "only future coordinates after schedule_origin_generation"
        )
    checkpoint_on_cancel: bool | None = None
    if "checkpoint_on_cancel" in schedule:
        checkpoint_on_cancel = schedule["checkpoint_on_cancel"]
        if type(checkpoint_on_cancel) is not bool:
            raise _protocol_error(
                "snapshot.schedule_state.checkpoint_on_cancel must be boolean"
            )
    return (
        schedule,
        origin,
        imported_ordinal,
        report_cadence,
        checkpoint_cadence,
        frozenset(explicit),
        checkpoint_on_cancel,
    )


def _validate_droppable_sequence_gap(
    snapshot: Mapping[str, Any],
    *,
    requested_after: int,
    sequence: int,
) -> int:
    gap = sequence - requested_after - 1
    if gap <= 0:
        return 0

    (
        _schedule,
        origin,
        imported_ordinal,
        report_cadence,
        checkpoint_cadence,
        explicit_generations,
        _checkpoint_on_cancel,
    ) = _validated_snapshot_schedule(snapshot)
    next_generation = _strict_int(
        snapshot.get("next_generation"), field="snapshot.next_generation"
    )
    schedule_coordinate = next_generation - origin
    if schedule_coordinate <= 0 or schedule_coordinate != sequence:
        raise _protocol_error(
            "snapshot schedule coordinates cannot prove its sequence gap"
        )
    if requested_after < imported_ordinal:
        raise _protocol_error(
            "snapshot read cursor precedes the imported boundary ordinal"
        )

    for skipped_sequence in range(requested_after + 1, sequence):
        coordinate = skipped_sequence
        exact_due = report_cadence > 0 and coordinate % report_cadence == 0
        checkpoint_due = (
            checkpoint_cadence > 0 and coordinate % checkpoint_cadence == 0
        ) or origin + coordinate in explicit_generations
        if exact_due or checkpoint_due:
            due = "exact reporting" if exact_due else "checkpoint"
            raise _protocol_error(
                "snapshot sequence gap crosses a persistence-required boundary: "
                f"sequence={skipped_sequence} due={due}"
            )
    return gap


def _snapshot_identity(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    sequence = _strict_int(
        snapshot.get("snapshot_sequence"),
        field="snapshot.snapshot_sequence",
        minimum=1,
    )
    ordinal = _strict_int(
        snapshot.get("boundary_ordinal"),
        field="snapshot.boundary_ordinal",
        minimum=1,
    )
    if ordinal != sequence:
        raise _protocol_error("snapshot boundary ordinal and sequence disagree")
    completed = _strict_int(
        snapshot.get("completed_generation"), field="snapshot.completed_generation"
    )
    next_generation = _strict_int(
        snapshot.get("next_generation"), field="snapshot.next_generation"
    )
    if next_generation != completed + 1:
        raise _protocol_error("snapshot next_generation is not post-turnover")
    if _strict_int(snapshot.get("tick"), field="snapshot.tick") != 0:
        raise _protocol_error("snapshot tick must be zero")
    if snapshot.get("phase") != "generation_complete":
        raise _protocol_error("snapshot phase must be generation_complete")

    identity = {
        "run_id": _identifier(snapshot.get("run_id"), field="snapshot.run_id"),
        "controller_id": _identifier(
            snapshot.get("controller_id"), field="snapshot.controller_id"
        ),
        "owner_epoch": _strict_int(
            snapshot.get("owner_epoch"), field="snapshot.owner_epoch", minimum=1
        ),
        "snapshot_id": _identifier(
            snapshot.get("snapshot_id"), field="snapshot.snapshot_id"
        ),
        "snapshot_sequence": sequence,
        "boundary_ordinal": ordinal,
        "completed_generation": completed,
        "next_generation": next_generation,
    }
    expected_snapshot_id = (
        f"{identity['controller_id']}:owner:{identity['owner_epoch']}:"
        f"boundary:{sequence}"
    )
    if identity["snapshot_id"] != expected_snapshot_id:
        raise _protocol_error("snapshot identity is not bound to its controller boundary")
    return identity


def _request_id(identity: Mapping[str, Any]) -> str:
    identity_digest = hashlib.sha256(_canonical_json_bytes(dict(identity))).hexdigest()
    return f"persist-{identity['snapshot_sequence']}-{identity_digest[:16]}"


def _validate_snapshot(
    raw: Any,
    *,
    controller_id: str,
    owner_epoch: int,
    expected_imported_boundary_ordinal: int,
) -> tuple[dict[str, Any], dict[str, Any], bool, bool, bool, dict[str, list[int]]]:
    snapshot = _mapping(raw, field="snapshot")
    if snapshot.get("schema_version") != "eiel_native_architecture_b_boundary_v1":
        raise _protocol_error("snapshot schema_version mismatch")
    identity = _snapshot_identity(snapshot)
    if identity["controller_id"] != controller_id or identity["owner_epoch"] != owner_epoch:
        raise _protocol_error("snapshot identity does not match its mailbox batch")

    snapshot_digest = _sha256(
        snapshot.get("snapshot_digest_sha256"),
        field="snapshot.snapshot_digest_sha256",
    )
    if _sha256(snapshot.get("snapshot_digest"), field="snapshot.snapshot_digest") != snapshot_digest:
        raise _protocol_error("snapshot digest aliases disagree")
    if snapshot.get("snapshot_digest_pending_adapter") is not False:
        raise _protocol_error("snapshot digest must be finalized by the native owner")
    expected_controller_status = (
        "generation_complete_waiting_persistence"
        if snapshot.get("persistence_ack_required") is True
        else "generation_complete_ready_to_resolve_terminal"
    )
    if snapshot.get("controller_status_at_clone") != expected_controller_status:
        raise _protocol_error("snapshot controller status contradicts its persistence gate")

    due_mask = _mapping(snapshot.get("due_mask"), field="snapshot.due_mask")
    allowed_due_fields = {
        "progress",
        "compact_progress",
        "exact_reporting",
        "behaviors",
        "metrics",
        "lineage",
        "typed_lifecycle",
        "checkpoint",
        "checkpoint_when_due",
        "chronicle",
        "scientific_chronicle",
        "scientific_chronicle_when_enabled",
        "final",
    }
    unexpected_due_fields = sorted(set(due_mask) - allowed_due_fields)
    if unexpected_due_fields:
        raise _protocol_error(
            f"snapshot due mask has unsupported fields: {unexpected_due_fields}"
        )
    due_progress = _due_flag(due_mask, "progress", "compact_progress")
    due_exact = _due_flag(due_mask, "exact_reporting")
    due_checkpoint = _due_flag(due_mask, "checkpoint", "checkpoint_when_due")
    cancellation_checkpoint_observed = _cancellation_checkpoint_observed(snapshot)
    due_chronicle = _due_flag(
        due_mask,
        "chronicle",
        "scientific_chronicle",
        "scientific_chronicle_when_enabled",
    )
    if not any((due_progress, due_exact, due_checkpoint)):
        raise _protocol_error("snapshot due mask schedules no observer")
    if due_chronicle and not due_exact:
        raise _protocol_error("Chronicle cannot be scheduled without exact reporting")
    for facet in ("behaviors", "metrics", "lineage", "typed_lifecycle"):
        if facet in due_mask and _due_flag(due_mask, facet) != due_exact:
            raise _protocol_error(
                f"snapshot.due_mask.{facet} must match exact_reporting"
            )
    (
        _schedule,
        schedule_origin,
        imported_ordinal,
        _report_cadence,
        checkpoint_cadence,
        explicit_generations,
        checkpoint_on_cancel,
    ) = _validated_snapshot_schedule(snapshot)
    if imported_ordinal != expected_imported_boundary_ordinal:
        raise _protocol_error(
            "snapshot imported boundary ordinal contradicts the persistence cursor"
        )
    schedule_coordinate = identity["next_generation"] - schedule_origin
    if (
        schedule_coordinate <= 0
        or schedule_coordinate != identity["snapshot_sequence"]
    ):
        raise _protocol_error(
            "snapshot schedule coordinates contradict its sequence"
        )
    expected_checkpoint_due = (
        checkpoint_cadence > 0
        and schedule_coordinate % checkpoint_cadence == 0
    ) or identity["next_generation"] in explicit_generations
    expected_checkpoint_due = (
        expected_checkpoint_due or cancellation_checkpoint_observed
    )
    if due_checkpoint != expected_checkpoint_due:
        raise _protocol_error(
            "snapshot checkpoint due mask contradicts schedule_state"
        )
    if cancellation_checkpoint_observed and not due_checkpoint:
        raise _protocol_error(
            "cancellation-observed snapshot must force a checkpoint-capable boundary"
        )
    if cancellation_checkpoint_observed and checkpoint_on_cancel is False:
        raise _protocol_error(
            "cancellation-observed snapshot contradicts disabled checkpoint_on_cancel"
        )
    acknowledgement_required = due_exact or due_checkpoint or due_chronicle
    if snapshot.get("persistence_ack_required") is not acknowledgement_required:
        raise _protocol_error("snapshot persistence requirement contradicts its due mask")
    expected_profile = (
        "canonical_checkpoint"
        if due_checkpoint
        else "exact_report" if due_exact else "progress_only"
    )
    if snapshot.get("payload_profile") != expected_profile:
        raise _protocol_error("snapshot payload profile contradicts its due mask")
    compact_progress = _mapping(
        snapshot.get("compact_progress"), field="snapshot.compact_progress"
    )
    for field, expected in (
        ("completed_generation", identity["completed_generation"]),
        ("next_generation", identity["next_generation"]),
        ("tick", 0),
    ):
        if _strict_int(
            compact_progress.get(field), field=f"snapshot.compact_progress.{field}"
        ) != expected:
            raise _protocol_error(
                f"snapshot compact progress {field} contradicts its boundary"
            )
    _strict_int(
        compact_progress.get("alive_count"),
        field="snapshot.compact_progress.alive_count",
    )
    expected_ranges = _cursor_ranges(
        snapshot.get("expected_acknowledged_channel_cursor_ranges"),
        field="snapshot.expected_acknowledged_channel_cursor_ranges",
    )
    if not acknowledgement_required and expected_ranges:
        raise _protocol_error("progress-only snapshot has acknowledgement cursor ranges")
    expected_range_names = set(_EXACT_REPORTING_CURSOR_CHANNELS if due_exact else ())
    if due_chronicle:
        expected_range_names.add("chronicle")
    if due_checkpoint:
        expected_range_names.add("checkpoint_generation")
    if set(expected_ranges) != expected_range_names:
        raise _protocol_error(
            "snapshot acknowledgement cursor fields do not exactly match due channels: "
            f"expected={sorted(expected_range_names)} "
            f"actual={sorted(expected_ranges)}"
        )
    return (
        snapshot,
        identity,
        acknowledgement_required,
        due_checkpoint,
        not acknowledgement_required,
        expected_ranges,
    )


def _validate_persistence_acknowledgement(
    raw: Any,
    *,
    identity: Mapping[str, Any],
    snapshot_digest: str,
    request_id: str,
    expected_ranges: Mapping[str, list[int]],
) -> tuple[dict[str, Any], dict[str, list[int]]]:
    acknowledgement = _closed_mapping(
        raw,
        field="persistence acknowledgement",
        required_fields=_ACKNOWLEDGEMENT_FIELDS,
    )
    expected_values = {
        "api_version": 1,
        "request_id": request_id,
        "expected_controller_id": identity["controller_id"],
        "expected_owner_epoch": identity["owner_epoch"],
        "snapshot_id": identity["snapshot_id"],
        "snapshot_digest_sha256": snapshot_digest,
    }
    for field, expected in expected_values.items():
        if acknowledgement.get(field) != expected:
            raise _protocol_error(
                f"persistence acknowledgement {field} mismatch: "
                f"expected={expected!r} actual={acknowledgement.get(field)!r}"
            )
    outcome = acknowledgement.get("outcome")
    if outcome not in {"success", "failed"}:
        raise _protocol_error(
            "persistence acknowledgement outcome must be success or failed"
        )
    failure_code = acknowledgement.get("failure_code")
    if outcome == "success":
        if failure_code is not None:
            raise _protocol_error(
                "successful persistence acknowledgement has a failure_code"
            )
    else:
        _identifier(
            failure_code,
            field="persistence acknowledgement failure_code",
        )
    _sha256(
        acknowledgement.get("artifact_manifest_digest_sha256"),
        field="persistence acknowledgement artifact_manifest_digest_sha256",
    )
    persisted_ranges = _cursor_ranges(
        acknowledgement.get("acknowledged_channel_cursor_ranges"),
        field="persistence acknowledgement acknowledged_channel_cursor_ranges",
    )
    if persisted_ranges != dict(expected_ranges):
        missing = sorted(set(expected_ranges) - set(persisted_ranges))
        unexpected = sorted(set(persisted_ranges) - set(expected_ranges))
        mismatched = sorted(
            channel
            for channel in set(expected_ranges) & set(persisted_ranges)
            if persisted_ranges[channel] != expected_ranges[channel]
        )
        raise _protocol_error(
            "persistence acknowledgement cursor map does not exactly match the "
            "native expected map: "
            f"missing={missing} unexpected={unexpected} mismatched={mismatched}"
        )
    return acknowledgement, persisted_ranges


class ArchitectureBPersistenceLoop:
    """Consume detached snapshots without entering or inspecting trajectory state."""

    def __init__(
        self,
        owner: Any,
        runner: Any,
        *,
        staging_root: str | Path | None = None,
        poll_timeout_ms: int = 100,
        idle_timeout_seconds: float = 300.0,
        run_thread: threading.Thread | None = None,
        initial_after_snapshot_sequence: int = 0,
        snapshot_observer: Callable[[Mapping[str, Any]], None] | None = None,
        clock: Any = time.monotonic,
    ) -> None:
        self._owner = owner
        self._runner = runner
        self._staging_root = staging_root
        self._poll_timeout_ms = _strict_int(
            poll_timeout_ms, field="poll_timeout_ms", minimum=1
        )
        if self._poll_timeout_ms > 60_000:
            raise _protocol_error("poll_timeout_ms exceeds the native 60000 ms bound")
        if (
            isinstance(idle_timeout_seconds, bool)
            or not isinstance(idle_timeout_seconds, (int, float))
            or not math.isfinite(float(idle_timeout_seconds))
            or float(idle_timeout_seconds) <= 0
        ):
            raise _protocol_error("idle_timeout_seconds must be finite and positive")
        self._idle_timeout_seconds = float(idle_timeout_seconds)
        self._run_thread = run_thread
        self._after_sequence = _strict_int(
            initial_after_snapshot_sequence,
            field="initial_after_snapshot_sequence",
        )
        self._initial_after_sequence = self._after_sequence
        if snapshot_observer is not None and not callable(snapshot_observer):
            raise _protocol_error("snapshot_observer must be callable")
        self._snapshot_observer = snapshot_observer
        self._clock = clock
        self._entry_lock = threading.Lock()
        self._entered = False
        self._failure_notified = False

    def run(self) -> ArchitectureBPersistenceLoopResult:
        with self._entry_lock:
            if self._entered:
                raise ArchitectureBPersistenceLoopError(
                    "EIEL-ARCHITECTURE-B-PERSISTENCE-LOOP-FAILED: "
                    "persistence loop cannot be re-entered"
                )
            self._entered = True

        try:
            return self._run_once()
        except BaseException as cause:
            self._notify_failure_once(cause)
            error = ArchitectureBPersistenceLoopError(
                "EIEL-ARCHITECTURE-B-PERSISTENCE-LOOP-FAILED: "
                f"{type(cause).__name__}: {cause}"
            )
            notification_error = getattr(cause, "_persistence_notification_error", None)
            if notification_error is not None:
                error.add_note(
                    "native failure notification also failed: "
                    f"{type(notification_error).__name__}: {notification_error}"
                )
            raise error from cause

    def _run_once(self) -> ArchitectureBPersistenceLoopResult:
        read_snapshots = getattr(self._owner, "read_snapshots", None)
        acknowledge = getattr(self._owner, "acknowledge_snapshot_persistence", None)
        if not callable(read_snapshots) or not callable(acknowledge):
            raise _protocol_error("owner does not expose the Architecture B mailbox API")

        controller_id: str | None = None
        owner_epoch: int | None = None
        deadline = float(self._clock()) + self._idle_timeout_seconds
        terminal_state = "none"
        snapshots_seen = 0
        progress_only = 0
        skipped_sequences = 0
        acknowledgements = 0
        identical_retries = 0
        checkpoints = 0
        checkpoint_paths: list[str] = []
        last_progress: dict[str, Any] | None = None
        cancellation_checkpoint_acknowledged_sequence: int | None = None

        while True:
            remaining = deadline - float(self._clock())
            if remaining <= 0:
                raise TimeoutError(
                    "Architecture B snapshot persistence idle timeout expired"
                )
            wait_timeout_ms = min(
                self._poll_timeout_ms,
                max(1, int(math.ceil(remaining * 1000.0))),
            )
            requested_after = self._after_sequence
            raw_batch = read_snapshots(
                {
                    "after_snapshot_sequence": requested_after,
                    "max_snapshots": 1,
                    "wait_timeout_ms": wait_timeout_ms,
                }
            )
            batch = _closed_mapping(
                raw_batch,
                field="snapshot batch",
                required_fields=_BATCH_FIELDS,
                optional_fields=frozenset({"schema_version"}),
            )
            if "schema_version" in batch and batch["schema_version"] != SNAPSHOT_BATCH_SCHEMA_VERSION:
                raise _protocol_error("snapshot batch schema_version mismatch")
            batch_controller_id = _identifier(
                batch.get("controller_id"), field="snapshot batch.controller_id"
            )
            batch_owner_epoch = _strict_int(
                batch.get("owner_epoch"),
                field="snapshot batch.owner_epoch",
                minimum=1,
            )
            if controller_id is None:
                controller_id = batch_controller_id
                owner_epoch = batch_owner_epoch
            elif controller_id != batch_controller_id or owner_epoch != batch_owner_epoch:
                raise _protocol_error("snapshot mailbox identity changed during persistence")
            if batch.get("after_snapshot_sequence") != requested_after:
                raise _protocol_error("snapshot batch read cursor mismatch")
            _strict_int(
                batch.get("batch_payload_bytes"),
                field="snapshot batch.batch_payload_bytes",
            )
            highest = _strict_int(
                batch.get("highest_snapshot_sequence"),
                field="snapshot batch.highest_snapshot_sequence",
            )
            raw_snapshots = batch.get("snapshots")
            if not isinstance(raw_snapshots, list) or len(raw_snapshots) > 1:
                raise _protocol_error("snapshot batch snapshots must contain at most one item")
            terminal_state = batch.get("terminal_state_seen")
            if terminal_state not in _TERMINAL_STATES | {"none"}:
                raise _protocol_error("snapshot batch terminal_state_seen is invalid")

            if raw_snapshots:
                if cancellation_checkpoint_acknowledged_sequence is not None:
                    raise _protocol_error(
                        "a snapshot followed the durably acknowledged "
                        "cancellation-observed boundary"
                    )
                assert controller_id is not None and owner_epoch is not None
                (
                    snapshot,
                    identity,
                    acknowledgement_required,
                    checkpoint_due,
                    progress_only_snapshot,
                    expected_ranges,
                ) = _validate_snapshot(
                    raw_snapshots[0],
                    controller_id=controller_id,
                    owner_epoch=owner_epoch,
                    expected_imported_boundary_ordinal=self._initial_after_sequence,
                )
                sequence = int(identity["snapshot_sequence"])
                if sequence <= requested_after or highest != sequence:
                    raise _protocol_error("snapshot batch sequence frontier is inconsistent")
                if terminal_state in _TERMINAL_STATES and acknowledgement_required:
                    raise _protocol_error(
                        "terminal state was observed before the last required "
                        "persistence acknowledgement"
                    )
                skipped_sequences += _validate_droppable_sequence_gap(
                    snapshot,
                    requested_after=requested_after,
                    sequence=sequence,
                )
                snapshots_seen += 1

                live_event = event_from_snapshot(snapshot)
                if live_event is not None:
                    try:
                        enqueue_event(live_event)
                    except Exception:
                        pass

                if self._snapshot_observer is not None:
                    observed = self._snapshot_observer(snapshot)
                    if observed is not None:
                        raise _protocol_error("snapshot_observer must return None")

                compact_progress = snapshot.get("compact_progress")
                if compact_progress is not None:
                    last_progress = copy.deepcopy(
                        _mapping(compact_progress, field="snapshot.compact_progress")
                    )

                if progress_only_snapshot:
                    progress_only += 1
                elif acknowledgement_required:
                    request_id = _request_id(identity)
                    raw_acknowledgement = persist_architecture_b_snapshot(
                        self._runner,
                        snapshot,
                        staging_root=self._staging_root,
                        request_id=request_id,
                    )
                    snapshot_digest = str(snapshot["snapshot_digest_sha256"])
                    persisted, _persisted_ranges = _validate_persistence_acknowledgement(
                        raw_acknowledgement,
                        identity=identity,
                        snapshot_digest=snapshot_digest,
                        request_id=request_id,
                        expected_ranges=expected_ranges,
                    )
                    exact_acknowledgement = copy.deepcopy(persisted)
                    exact_acknowledgement[
                        "acknowledged_channel_cursor_ranges"
                    ] = copy.deepcopy(expected_ranges)
                    if persisted["outcome"] == "failed":
                        reported_failure = self._reported_persistence_failure(
                            persisted,
                            identity=identity,
                            snapshot_digest=snapshot_digest,
                        )
                        self._failure_notified = True
                        raw_receipt = acknowledge(exact_acknowledgement)
                        self._validate_receipt(
                            raw_receipt,
                            acknowledgement=exact_acknowledgement,
                        )
                        raise reported_failure
                    if checkpoint_due:
                        checkpoint_path = self._published_checkpoint_bundle_path(
                            identity=identity,
                            snapshot_digest=snapshot_digest,
                            artifact_manifest_digest=str(
                                persisted["artifact_manifest_digest_sha256"]
                            ),
                        )
                        checkpoints += 1
                        checkpoint_paths.append(checkpoint_path)
                    receipt = self._validate_receipt(
                        acknowledge(exact_acknowledgement),
                        acknowledgement=exact_acknowledgement,
                    )
                    acknowledgements += 1
                    identical_retries += receipt["disposition"] == "identical_retry"
                    if _cancellation_checkpoint_observed(snapshot):
                        cancellation_checkpoint_acknowledged_sequence = sequence

                self._after_sequence = sequence
                deadline = float(self._clock()) + self._idle_timeout_seconds
            elif highest != requested_after:
                raise _protocol_error("empty snapshot batch advanced its sequence frontier")

            if terminal_state in _TERMINAL_STATES:
                if (
                    terminal_state == "run_complete"
                    and cancellation_checkpoint_acknowledged_sequence is not None
                ):
                    raise _protocol_error(
                        "run_complete cannot follow a cancellation-observed boundary"
                    )
                if (
                    terminal_state == "cancelled"
                    and cancellation_checkpoint_acknowledged_sequence is None
                ):
                    raise _protocol_error(
                        "cancelled terminal visibility requires a durable "
                        "cancellation checkpoint acknowledgement"
                    )
                if (
                    terminal_state == "cancelled"
                    and cancellation_checkpoint_acknowledged_sequence
                    != self._after_sequence
                ):
                    raise _protocol_error(
                        "cancelled terminal visibility preceded its durable cancellation checkpoint acknowledgement"
                    )
                return ArchitectureBPersistenceLoopResult(
                    schema_version=SCHEMA_VERSION,
                    stop_reason="terminal_state_seen",
                    terminal_state_seen=terminal_state,
                    snapshots_seen=snapshots_seen,
                    progress_only_snapshots=progress_only,
                    skipped_snapshot_sequences=skipped_sequences,
                    persistence_acknowledgements=acknowledgements,
                    identical_acknowledgement_retries=identical_retries,
                    checkpoints_persisted=checkpoints,
                    architecture_b_checkpoint_bundle_paths=list(checkpoint_paths),
                    last_snapshot_sequence=self._after_sequence,
                    last_progress=last_progress,
                )
            if not raw_snapshots and self._run_thread is not None and not self._run_thread.is_alive():
                return ArchitectureBPersistenceLoopResult(
                    schema_version=SCHEMA_VERSION,
                    stop_reason="run_thread_finished",
                    terminal_state_seen="none",
                    snapshots_seen=snapshots_seen,
                    progress_only_snapshots=progress_only,
                    skipped_snapshot_sequences=skipped_sequences,
                    persistence_acknowledgements=acknowledgements,
                    identical_acknowledgement_retries=identical_retries,
                    checkpoints_persisted=checkpoints,
                    architecture_b_checkpoint_bundle_paths=list(checkpoint_paths),
                    last_snapshot_sequence=self._after_sequence,
                    last_progress=last_progress,
                )
            if float(self._clock()) >= deadline:
                raise TimeoutError(
                    "Architecture B snapshot persistence idle timeout expired"
                )

    def _snapshot_staging_root(self) -> Path:
        if self._staging_root is not None:
            return Path(self._staging_root)
        run_dir = str(getattr(self._runner, "run_dir", "") or "")
        if not run_dir:
            raise _protocol_error(
                "runner.run_dir is required when staging_root is omitted"
            )
        return Path(run_dir) / SNAPSHOT_STAGING_DIRECTORY

    def _reported_persistence_failure(
        self,
        acknowledgement: Mapping[str, Any],
        *,
        identity: Mapping[str, Any],
        snapshot_digest: str,
    ) -> _ArchitectureBPersistenceReportedFailure:
        failure_code = str(acknowledgement["failure_code"])
        failure_digest = str(acknowledgement["artifact_manifest_digest_sha256"])
        durable_cause: str | None = None
        try:
            root = self._snapshot_staging_root()
            for failure_path in sorted(
                root.glob(f".quarantine/snapshot-*/{SNAPSHOT_FAILURE_RECORD_NAME}")
            ):
                payload = failure_path.read_bytes()
                if hashlib.sha256(payload).hexdigest() != failure_digest:
                    continue
                record = _mapping(
                    json.loads(payload), field="snapshot persistence failure record"
                )
                if (
                    record.get("error_code") == failure_code
                    and record.get("snapshot_id") == identity["snapshot_id"]
                    and record.get("snapshot_digest_sha256") == snapshot_digest
                    and type(record.get("message")) is str
                    and record["message"]
                ):
                    durable_cause = str(record["message"])
                break
        except BaseException:
            durable_cause = None

        detail = (
            f"cause={durable_cause}"
            if durable_cause is not None
            else f"failure_record_digest_sha256={failure_digest}"
        )
        return _ArchitectureBPersistenceReportedFailure(
            "EIEL-ARCHITECTURE-B-PERSISTENCE-REPORTED-FAILED: "
            f"failure_code={failure_code} {detail}"
        )

    def _published_checkpoint_bundle_path(
        self,
        *,
        identity: Mapping[str, Any],
        snapshot_digest: str,
        artifact_manifest_digest: str,
    ) -> str:
        root = self._snapshot_staging_root()

        matches: list[Path] = []
        for manifest_path in sorted(
            root.glob(f"snapshot-*/{SNAPSHOT_ARTIFACT_MANIFEST_NAME}")
        ):
            try:
                payload = manifest_path.read_bytes()
            except OSError as exc:
                raise _protocol_error(
                    f"published snapshot manifest cannot be read: {exc}"
                ) from exc
            if hashlib.sha256(payload).hexdigest() != artifact_manifest_digest:
                continue
            try:
                manifest = _mapping(
                    json.loads(payload), field="published snapshot artifact manifest"
                )
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise _protocol_error(
                    f"published snapshot manifest is invalid JSON: {exc}"
                ) from exc
            manifest_identity = _mapping(
                manifest.get("snapshot_identity"),
                field="published snapshot artifact manifest.snapshot_identity",
            )
            if manifest_identity != dict(identity):
                raise _protocol_error(
                    "published snapshot manifest identity contradicts its acknowledgement"
                )
            if manifest.get("snapshot_digest_sha256") != snapshot_digest:
                raise _protocol_error(
                    "published snapshot manifest digest identity contradicts its snapshot"
                )
            artifacts = manifest.get("artifacts")
            if not isinstance(artifacts, list):
                raise _protocol_error(
                    "published snapshot artifact manifest.artifacts must be an array"
                )
            checkpoint_records = [
                record
                for record in artifacts
                if isinstance(record, Mapping) and record.get("facet") == "checkpoint"
            ]
            if not checkpoint_records or any(
                type(record.get("path")) is not str
                or not str(record["path"]).startswith("checkpoint/")
                for record in checkpoint_records
            ):
                raise _protocol_error(
                    "checkpoint-due snapshot manifest has no closed checkpoint bundle"
                )
            checkpoint_path = manifest_path.parent / "checkpoint"
            if not checkpoint_path.is_dir() or not (
                checkpoint_path / "bundle_complete.json"
            ).is_file():
                raise _protocol_error(
                    "checkpoint-due snapshot publication has no complete checkpoint bundle"
                )
            matches.append(checkpoint_path.resolve())
        if len(matches) != 1:
            raise _protocol_error(
                "checkpoint-due acknowledgement does not resolve to exactly one "
                f"published bundle: matches={len(matches)}"
            )
        return str(matches[0])

    def _validate_receipt(
        self,
        raw: Any,
        *,
        acknowledgement: Mapping[str, Any],
    ) -> dict[str, Any]:
        receipt = _closed_mapping(
            raw,
            field="persistence acknowledgement receipt",
            required_fields=_RECEIPT_FIELDS,
            optional_fields=frozenset({"schema_version"}),
        )
        if (
            "schema_version" in receipt
            and receipt["schema_version"] != ACKNOWLEDGEMENT_RECEIPT_SCHEMA_VERSION
        ):
            raise _protocol_error("persistence acknowledgement receipt schema mismatch")
        expected = {
            "request_id": acknowledgement["request_id"],
            "controller_id": acknowledgement["expected_controller_id"],
            "owner_epoch": acknowledgement["expected_owner_epoch"],
            "snapshot_id": acknowledgement["snapshot_id"],
        }
        for field, value in expected.items():
            if receipt.get(field) != value:
                raise _protocol_error(
                    f"persistence acknowledgement receipt {field} mismatch"
                )
        if acknowledgement["outcome"] == "failed":
            if receipt.get("disposition") != "accepted":
                raise _protocol_error(
                    "native owner rejected the failed persistence acknowledgement"
                )
            if receipt.get("persistence_barrier_released") is not False:
                raise _protocol_error(
                    "failed persistence acknowledgement released its barrier"
                )
            if receipt.get("terminal_status_changed_to_failed") is not True:
                raise _protocol_error(
                    "failed persistence acknowledgement did not fail the owner"
                )
        else:
            if receipt.get("disposition") not in {"accepted", "identical_retry"}:
                raise _protocol_error(
                    "native owner rejected the exact persistence acknowledgement"
                )
            if receipt.get("persistence_barrier_released") is not True:
                raise _protocol_error(
                    "persistence acknowledgement did not release its barrier"
                )
            if receipt.get("terminal_status_changed_to_failed") is not False:
                raise _protocol_error(
                    "successful persistence acknowledgement changed terminal state"
                )
        return receipt

    def _notify_failure_once(self, cause: BaseException) -> None:
        if self._failure_notified:
            return
        self._failure_notified = True
        failure_operation = getattr(self._owner, "fail_snapshot_persistence", None)
        if not callable(failure_operation):
            notification_error: BaseException = _protocol_error(
                "owner does not expose fail_snapshot_persistence"
            )
            setattr(cause, "_persistence_notification_error", notification_error)
            return
        message = f"{type(cause).__name__}: {cause}"[:2048]
        try:
            failure_operation(message)
        except BaseException as notification_error:
            setattr(cause, "_persistence_notification_error", notification_error)


def run_architecture_b_persistence_loop(
    owner: Any,
    runner: Any,
    **kwargs: Any,
) -> ArchitectureBPersistenceLoopResult:
    """Run one pull-only adapter instance; trajectory dispatch belongs to the caller."""

    return ArchitectureBPersistenceLoop(owner, runner, **kwargs).run()


__all__ = [
    "ArchitectureBPersistenceLoop",
    "ArchitectureBPersistenceLoopError",
    "ArchitectureBPersistenceLoopResult",
    "ArchitectureBPersistenceProtocolError",
    "SCHEMA_VERSION",
    "run_architecture_b_persistence_loop",
]
