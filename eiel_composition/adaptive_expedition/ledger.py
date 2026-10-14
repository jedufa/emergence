"""Tamper-evident, append-only recovery ledger for adaptive expeditions.

The ledger is deliberately not an execution authority.  It records decisions and
the exact child bindings produced by the canonical science front door.  Every
operation consumes and returns plain canonical mappings so the public contract
package can validate the same values without depending on this implementation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Final

from emergence_contracts import canonical_json_bytes

LEDGER_SCHEMA_V1: Final[str] = "eiel_adaptive_expedition_ledger_v1"
LEDGER_EVENT_SCHEMA_V1: Final[str] = "eiel_adaptive_expedition_ledger_event_v1"
LEDGER_HEADER_FILE_V1: Final[str] = "ledger-header.json"
LEDGER_EVENTS_DIRECTORY_V1: Final[str] = "events"

FRONTIER_OBSERVED: Final[str] = "FRONTIER_OBSERVED"
DECISION_PREPARED: Final[str] = "DECISION_PREPARED"
ACTION_DISPATCH_BOUND: Final[str] = "ACTION_DISPATCH_BOUND"
ACTION_WAIT: Final[str] = "ACTION_WAIT"
ACTION_SETTLED: Final[str] = "ACTION_SETTLED"
STOP_REQUESTED: Final[str] = "STOP_REQUESTED"
EXPEDITION_TERMINAL: Final[str] = "EXPEDITION_TERMINAL"

LEDGER_EVENT_TYPES_V1: Final[frozenset[str]] = frozenset(
    {
        FRONTIER_OBSERVED,
        DECISION_PREPARED,
        ACTION_DISPATCH_BOUND,
        ACTION_WAIT,
        ACTION_SETTLED,
        STOP_REQUESTED,
        EXPEDITION_TERMINAL,
    }
)
WAIT_KINDS_V1: Final[frozenset[str]] = frozenset({"WAIT_RESOURCE", "WAIT_STORAGE"})
TERMINAL_STATUSES_V1: Final[frozenset[str]] = frozenset(
    {"COMPLETED", "STOPPED", "FAILED", "QUARANTINED"}
)
_EVENT_FILE_RE_V1: Final[re.Pattern[str]] = re.compile(r"([0-9]{8})\.json")

# This is an adapter boundary, not another public journal schema.  The public
# JournalRecordKindV1 vocabulary is deliberately coarser than the private
# crash-recovery transitions.  Child waits and partial settlements remain
# CHILD_REQUEST_GENERATED-family records; only the last declared settlement
# folds to FRONTIER_SETTLED.  A stop request is a committed control decision
# until the separate terminal record is appended.
_PUBLIC_JOURNAL_KIND_BY_EVENT_V1: Final[Mapping[str, str]] = {
    FRONTIER_OBSERVED: "OBSERVATION_ACCEPTED",
    DECISION_PREPARED: "DECISION_COMMITTED",
    ACTION_DISPATCH_BOUND: "CHILD_REQUEST_GENERATED",
    ACTION_WAIT: "CHILD_REQUEST_GENERATED",
    ACTION_SETTLED: "CHILD_REQUEST_GENERATED",
    STOP_REQUESTED: "DECISION_COMMITTED",
    EXPEDITION_TERMINAL: "EXPEDITION_TERMINATED",
}


class AdaptiveExpeditionLedgerError(ValueError):
    """Base failure for an invalid or unsafe expedition ledger."""


class LedgerIntegrityError(AdaptiveExpeditionLedgerError):
    """The ledger's canonical representation or hash chain is invalid."""


class LedgerTransitionError(AdaptiveExpeditionLedgerError):
    """A proposed event violates the expedition recovery state machine."""


class LedgerConflictError(AdaptiveExpeditionLedgerError):
    """Another append already owns the immutable event position."""


def _canonical_clone(value: Any, *, label: str) -> Any:
    try:
        return json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise LedgerIntegrityError(f"{label} is not canonical JSON data") from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise LedgerIntegrityError(f"{field} must be a lowercase SHA-256")
    return value


def _require_identifier(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 512:
        raise LedgerIntegrityError(f"{field} must be a bounded non-empty string")
    return value


def _required_payload_fields(
    payload: Mapping[str, Any], event_type: str, fields: Sequence[str]
) -> None:
    missing = [field for field in fields if field not in payload]
    if missing:
        raise LedgerTransitionError(
            f"{event_type} payload is missing required fields: {missing}"
        )


def _ledger_genesis_body_v1(
    *, expedition_id: str, manifest_sha256: str, policy_sha256: str
) -> dict[str, Any]:
    return {
        "schema_version": LEDGER_SCHEMA_V1,
        "expedition_id": expedition_id,
        "manifest_sha256": manifest_sha256,
        "policy_sha256": policy_sha256,
        "authority_effect": "none",
    }


def new_expedition_ledger_v1(
    *, expedition_id: str, manifest_sha256: str, policy_sha256: str
) -> dict[str, Any]:
    """Create an empty immutable-value ledger projection."""

    identifier = _require_identifier(expedition_id, field="expedition_id")
    manifest = _require_sha256(manifest_sha256, field="manifest_sha256")
    policy = _require_sha256(policy_sha256, field="policy_sha256")
    genesis_body = _ledger_genesis_body_v1(
        expedition_id=identifier,
        manifest_sha256=manifest,
        policy_sha256=policy,
    )
    genesis_sha256 = _sha256(genesis_body)
    return {
        **genesis_body,
        "genesis_sha256": genesis_sha256,
        "events": [],
        "head_event_sha256": genesis_sha256,
    }


def _frontier_is_complete(frontier: Mapping[str, Any]) -> bool:
    decision = frontier.get("decision")
    if not isinstance(decision, Mapping):
        return False
    actions = frontier.get("actions")
    if not isinstance(actions, Mapping):
        return False
    action_ids = decision.get("action_ids")
    return isinstance(action_ids, list) and all(
        action_id in actions
        and isinstance(actions[action_id], Mapping)
        and actions[action_id].get("settlement") is not None
        for action_id in action_ids
    )


def _validate_event_shape_v1(
    raw: object,
    *,
    expected_index: int,
    expected_previous_sha256: str,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise LedgerIntegrityError(f"ledger event {expected_index} is not an object")
    event = _canonical_clone(dict(raw), label=f"ledger event {expected_index}")
    expected_fields = {
        "schema_version",
        "event_index",
        "event_type",
        "frontier_ordinal",
        "previous_event_sha256",
        "payload",
        "event_sha256",
    }
    if set(event) != expected_fields:
        raise LedgerIntegrityError(f"ledger event {expected_index} fields differ")
    event_type = event.get("event_type")
    frontier_ordinal = event.get("frontier_ordinal")
    if (
        event.get("schema_version") != LEDGER_EVENT_SCHEMA_V1
        or event.get("event_index") != expected_index
        or event_type not in LEDGER_EVENT_TYPES_V1
        or type(frontier_ordinal) is not int
        or frontier_ordinal < 0
        or event.get("previous_event_sha256") != expected_previous_sha256
        or not isinstance(event.get("payload"), Mapping)
    ):
        raise LedgerIntegrityError(f"ledger event {expected_index} envelope is invalid")
    body = {key: value for key, value in event.items() if key != "event_sha256"}
    supplied_sha256 = _require_sha256(
        event.get("event_sha256"), field=f"events[{expected_index}].event_sha256"
    )
    if supplied_sha256 != _sha256(body):
        raise LedgerIntegrityError(f"ledger event {expected_index} digest changed")
    return event


def _replay_ledger_v1(ledger: Mapping[str, Any]) -> dict[str, Any]:
    expected_fields = {
        "schema_version",
        "expedition_id",
        "manifest_sha256",
        "policy_sha256",
        "authority_effect",
        "genesis_sha256",
        "events",
        "head_event_sha256",
    }
    if set(ledger) != expected_fields:
        raise LedgerIntegrityError("ledger fields differ")
    if ledger.get("schema_version") != LEDGER_SCHEMA_V1:
        raise LedgerIntegrityError("ledger schema version is invalid")
    if ledger.get("authority_effect") != "none":
        raise LedgerIntegrityError("ledger must not claim execution authority")
    expedition_id = _require_identifier(
        ledger.get("expedition_id"), field="expedition_id"
    )
    manifest_sha256 = _require_sha256(
        ledger.get("manifest_sha256"), field="manifest_sha256"
    )
    policy_sha256 = _require_sha256(ledger.get("policy_sha256"), field="policy_sha256")
    genesis_body = _ledger_genesis_body_v1(
        expedition_id=expedition_id,
        manifest_sha256=manifest_sha256,
        policy_sha256=policy_sha256,
    )
    genesis_sha256 = _require_sha256(
        ledger.get("genesis_sha256"), field="genesis_sha256"
    )
    if genesis_sha256 != _sha256(genesis_body):
        raise LedgerIntegrityError("ledger genesis digest changed")
    raw_events = ledger.get("events")
    if not isinstance(raw_events, list):
        raise LedgerIntegrityError("ledger events must be an array")

    frontiers: dict[int, dict[str, Any]] = {}
    current_frontier_ordinal = -1
    decision_ids: set[str] = set()
    action_ids_seen: set[str] = set()
    child_binding_sha256s: set[str] = set()
    child_request_sha256s: set[str] = set()
    wait_receipt_sha256s: set[str] = set()
    stop_event: dict[str, Any] | None = None
    terminal_event: dict[str, Any] | None = None
    previous_sha256 = genesis_sha256
    normalized_events: list[dict[str, Any]] = []

    for event_index, raw_event in enumerate(raw_events):
        if terminal_event is not None:
            raise LedgerTransitionError(
                "ledger contains an event after terminal closeout"
            )
        event = _validate_event_shape_v1(
            raw_event,
            expected_index=event_index,
            expected_previous_sha256=previous_sha256,
        )
        normalized_events.append(event)
        event_type = str(event["event_type"])
        frontier_ordinal = int(event["frontier_ordinal"])
        payload = dict(event["payload"])

        if event_type == FRONTIER_OBSERVED:
            _required_payload_fields(
                payload, event_type, ("frontier_id", "observation_sha256")
            )
            _require_identifier(payload["frontier_id"], field="frontier_id")
            _require_sha256(payload["observation_sha256"], field="observation_sha256")
            if stop_event is not None:
                raise LedgerTransitionError(
                    "a stopped expedition cannot observe a frontier"
                )
            if frontier_ordinal != current_frontier_ordinal + 1:
                raise LedgerTransitionError(
                    "frontier ordinals must be contiguous from zero"
                )
            if current_frontier_ordinal >= 0 and not _frontier_is_complete(
                frontiers[current_frontier_ordinal]
            ):
                raise LedgerTransitionError(
                    "a new frontier requires every prior action to settle"
                )
            current_frontier_ordinal = frontier_ordinal
            frontiers[frontier_ordinal] = {
                "frontier_ordinal": frontier_ordinal,
                "observation": payload,
                "observation_event_sha256": event["event_sha256"],
                "decision": None,
                "decision_event_sha256": None,
                "actions": {},
            }

        elif event_type == DECISION_PREPARED:
            _required_payload_fields(
                payload,
                event_type,
                (
                    "decision_id",
                    "decision_sha256",
                    "action_ids",
                    "rng_cursor_before_sha256",
                    "rng_cursor_after_sha256",
                ),
            )
            decision_id = _require_identifier(
                payload["decision_id"], field="decision_id"
            )
            _require_sha256(payload["decision_sha256"], field="decision_sha256")
            _require_sha256(
                payload["rng_cursor_before_sha256"],
                field="rng_cursor_before_sha256",
            )
            _require_sha256(
                payload["rng_cursor_after_sha256"],
                field="rng_cursor_after_sha256",
            )
            raw_action_ids = payload["action_ids"]
            if isinstance(raw_action_ids, (str, bytes, bytearray)) or not isinstance(
                raw_action_ids, Sequence
            ):
                raise LedgerTransitionError("decision action_ids must be an array")
            action_ids = [
                _require_identifier(value, field="action_ids[]")
                for value in raw_action_ids
            ]
            if len(set(action_ids)) != len(action_ids):
                raise LedgerTransitionError("decision action_ids must be unique")
            if stop_event is not None:
                raise LedgerTransitionError(
                    "a stopped expedition cannot prepare a decision"
                )
            if (
                frontier_ordinal != current_frontier_ordinal
                or frontier_ordinal not in frontiers
            ):
                raise LedgerTransitionError(
                    "decision does not bind the current frontier"
                )
            frontier = frontiers[frontier_ordinal]
            if frontier["decision"] is not None or decision_id in decision_ids:
                raise LedgerTransitionError("a decision can be prepared exactly once")
            if any(action_id in action_ids_seen for action_id in action_ids):
                raise LedgerTransitionError(
                    "action identity was reused across decisions"
                )
            normalized_payload = {**payload, "action_ids": action_ids}
            frontier["decision"] = normalized_payload
            frontier["decision_event_sha256"] = event["event_sha256"]
            decision_ids.add(decision_id)
            action_ids_seen.update(action_ids)

        elif event_type == ACTION_DISPATCH_BOUND:
            _required_payload_fields(
                payload,
                event_type,
                (
                    "decision_id",
                    "action_id",
                    "child_request_sha256",
                    "child_binding_sha256",
                ),
            )
            decision_id = _require_identifier(
                payload["decision_id"], field="decision_id"
            )
            action_id = _require_identifier(payload["action_id"], field="action_id")
            child_request_sha256 = _require_sha256(
                payload["child_request_sha256"], field="child_request_sha256"
            )
            child_binding_sha256 = _require_sha256(
                payload["child_binding_sha256"], field="child_binding_sha256"
            )
            if stop_event is not None:
                raise LedgerTransitionError(
                    "a stopped expedition cannot dispatch an action"
                )
            if (
                frontier_ordinal != current_frontier_ordinal
                or frontier_ordinal not in frontiers
            ):
                raise LedgerTransitionError(
                    "action dispatch does not bind the current frontier"
                )
            frontier = frontiers[frontier_ordinal]
            decision = frontier["decision"]
            if not isinstance(decision, Mapping):
                raise LedgerTransitionError(
                    "action dispatch requires a prepared decision"
                )
            if decision.get(
                "decision_id"
            ) != decision_id or action_id not in decision.get("action_ids", []):
                raise LedgerTransitionError("action dispatch differs from its decision")
            actions = frontier["actions"]
            if action_id in actions:
                raise LedgerTransitionError("an action can bind a child exactly once")
            if child_binding_sha256 in child_binding_sha256s:
                raise LedgerTransitionError("child binding identity was reused")
            if child_request_sha256 in child_request_sha256s:
                raise LedgerTransitionError("child request identity was reused")
            actions[action_id] = {
                "binding": payload,
                "binding_event_sha256": event["event_sha256"],
                "waits": [],
                "settlement": None,
                "settlement_event_sha256": None,
            }
            child_binding_sha256s.add(child_binding_sha256)
            child_request_sha256s.add(child_request_sha256)

        elif event_type in {ACTION_WAIT, ACTION_SETTLED}:
            common_fields = ("action_id", "child_binding_sha256")
            _required_payload_fields(payload, event_type, common_fields)
            action_id = _require_identifier(payload["action_id"], field="action_id")
            child_binding_sha256 = _require_sha256(
                payload["child_binding_sha256"], field="child_binding_sha256"
            )
            if frontier_ordinal not in frontiers:
                raise LedgerTransitionError(f"{event_type} names an unknown frontier")
            action = frontiers[frontier_ordinal]["actions"].get(action_id)
            if not isinstance(action, Mapping):
                raise LedgerTransitionError(f"{event_type} requires a bound child")
            binding = action.get("binding")
            if (
                not isinstance(binding, Mapping)
                or binding.get("child_binding_sha256") != child_binding_sha256
            ):
                raise LedgerTransitionError(f"{event_type} child binding changed")
            if action.get("settlement") is not None:
                raise LedgerTransitionError(
                    "a settled action cannot receive another event"
                )
            if event_type == ACTION_WAIT:
                _required_payload_fields(
                    payload, event_type, ("wait_kind", "wait_receipt_sha256")
                )
                if payload["wait_kind"] not in WAIT_KINDS_V1:
                    raise LedgerTransitionError("action wait kind is unsupported")
                wait_sha256 = _require_sha256(
                    payload["wait_receipt_sha256"], field="wait_receipt_sha256"
                )
                if wait_sha256 in wait_receipt_sha256s:
                    raise LedgerTransitionError("wait receipt identity was duplicated")
                action["waits"].append(
                    {**payload, "event_sha256": event["event_sha256"]}
                )
                wait_receipt_sha256s.add(wait_sha256)
            else:
                _required_payload_fields(
                    payload,
                    event_type,
                    (
                        "outcome",
                        "child_result_sha256",
                        "authenticated_child_request_sha256",
                    ),
                )
                _require_identifier(payload["outcome"], field="outcome")
                _require_sha256(
                    payload["child_result_sha256"], field="child_result_sha256"
                )
                authenticated_child_request_sha256 = _require_sha256(
                    payload["authenticated_child_request_sha256"],
                    field="authenticated_child_request_sha256",
                )
                if authenticated_child_request_sha256 != binding.get(
                    "child_request_sha256"
                ):
                    raise LedgerTransitionError(
                        "settled child request differs from its dispatch binding"
                    )
                action["settlement"] = payload
                action["settlement_event_sha256"] = event["event_sha256"]

        elif event_type == STOP_REQUESTED:
            _required_payload_fields(payload, event_type, ("reason",))
            _require_identifier(payload["reason"], field="reason")
            if stop_event is not None:
                raise LedgerTransitionError("stop can be requested exactly once")
            if frontier_ordinal != max(current_frontier_ordinal, 0):
                raise LedgerTransitionError(
                    "stop request does not bind the current frontier ordinal"
                )
            stop_event = {**payload, "event_sha256": event["event_sha256"]}

        elif event_type == EXPEDITION_TERMINAL:
            _required_payload_fields(payload, event_type, ("terminal_status",))
            if frontier_ordinal != max(current_frontier_ordinal, 0):
                raise LedgerTransitionError(
                    "terminal event does not bind the current frontier ordinal"
                )
            terminal_status = payload["terminal_status"]
            if terminal_status not in TERMINAL_STATUSES_V1:
                raise LedgerTransitionError("terminal status is unsupported")
            active_actions = [
                action_id
                for frontier in frontiers.values()
                for action_id, action in frontier["actions"].items()
                if action.get("settlement") is None
            ]
            if active_actions:
                raise LedgerTransitionError(
                    "terminal closeout has unsettled bound actions: "
                    f"{sorted(active_actions)}"
                )
            if terminal_status == "STOPPED" and stop_event is None:
                raise LedgerTransitionError("STOPPED terminal requires a stop request")
            if terminal_status == "COMPLETED" and (
                current_frontier_ordinal < 0
                or not _frontier_is_complete(frontiers[current_frontier_ordinal])
            ):
                raise LedgerTransitionError(
                    "COMPLETED terminal requires a fully settled frontier"
                )
            terminal_event = {**payload, "event_sha256": event["event_sha256"]}

        previous_sha256 = event["event_sha256"]

    head_sha256 = _require_sha256(
        ledger.get("head_event_sha256"), field="head_event_sha256"
    )
    if head_sha256 != previous_sha256:
        raise LedgerIntegrityError("ledger head does not bind the event chain")

    frontier_projection = []
    for index in sorted(frontiers):
        frontier = frontiers[index]
        actions = {
            action_id: dict(action)
            for action_id, action in sorted(frontier["actions"].items())
        }
        frontier_projection.append({**frontier, "actions": actions})
    current = (
        None
        if current_frontier_ordinal < 0
        else frontier_projection[current_frontier_ordinal]
    )
    pending = []
    if current is not None:
        pending = [
            action_id
            for action_id, action in current["actions"].items()
            if action.get("settlement") is None
        ]
    return {
        "schema_version": "eiel_adaptive_expedition_reconstructed_frontier_v1",
        "expedition_id": expedition_id,
        "manifest_sha256": manifest_sha256,
        "policy_sha256": policy_sha256,
        "event_count": len(normalized_events),
        "head_event_sha256": previous_sha256,
        "next_event_index": len(normalized_events),
        "frontiers": frontier_projection,
        "current_frontier_ordinal": (
            None if current_frontier_ordinal < 0 else current_frontier_ordinal
        ),
        "current_frontier": current,
        "pending_action_ids": sorted(pending),
        "stop": stop_event,
        "terminal": terminal_event,
        "authority_effect": "none",
    }


def validate_expedition_ledger_v1(value: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute the complete hash chain and transition history."""

    if not isinstance(value, Mapping):
        raise LedgerIntegrityError("ledger must be an object")
    ledger = _canonical_clone(dict(value), label="ledger")
    _replay_ledger_v1(ledger)
    return ledger


def reconstruct_expedition_frontier_v1(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the complete reconstructible projection of a valid ledger."""

    ledger = validate_expedition_ledger_v1(value)
    return _replay_ledger_v1(ledger)


def _content_digest_mapping_v1(value: str, *, field: str) -> dict[str, str]:
    return {"algorithm": "sha256", "hex": _require_sha256(value, field=field)}


def project_ledger_to_public_journal_v1(
    value: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Fold private recovery events into public ``JournalRecordV1`` mappings.

    The returned mappings contain canonical ``ContentDigest`` envelopes and can
    be validated directly by the public contract.  They retain the complete
    private event as a canonical payload, so the coarse public record family
    never discards wait, stop-intent, or exactly-once binding evidence.
    """

    ledger = validate_expedition_ledger_v1(value)
    declared_actions: dict[int, set[str]] = {}
    settled_actions: dict[int, set[str]] = {}
    records: list[dict[str, Any]] = []
    previous_record_sha256: str | None = None

    for event in ledger["events"]:
        event_type = event["event_type"]
        frontier_ordinal = event["frontier_ordinal"]
        if event_type == DECISION_PREPARED:
            declared_actions[frontier_ordinal] = set(event["payload"]["action_ids"])
            settled_actions[frontier_ordinal] = set()
        elif event_type == ACTION_SETTLED:
            settled_actions.setdefault(frontier_ordinal, set()).add(
                event["payload"]["action_id"]
            )

        record_kind = _PUBLIC_JOURNAL_KIND_BY_EVENT_V1[event_type]
        if event_type == ACTION_SETTLED and declared_actions.get(
            frontier_ordinal
        ) == settled_actions.get(frontier_ordinal):
            record_kind = "FRONTIER_SETTLED"

        payload = {
            "schema_version": LEDGER_EVENT_SCHEMA_V1,
            "payload": {
                "authority_effect": "none",
                "private_event": event,
                "projection_record_kind": record_kind,
            },
        }
        record = {
            "schema_version": "eiel_adaptive_science_journal_record_v1",
            "expedition_id": ledger["expedition_id"],
            "record_sequence": event["event_index"],
            "frontier_ordinal": frontier_ordinal,
            "record_kind": record_kind,
            "manifest_sha256": _content_digest_mapping_v1(
                ledger["manifest_sha256"], field="manifest_sha256"
            ),
            "subject_sha256": _content_digest_mapping_v1(
                event["event_sha256"], field="event_sha256"
            ),
            "payload": payload,
            "payload_sha256": _content_digest_mapping_v1(
                _sha256(payload), field="payload_sha256"
            ),
            "previous_record_sha256": (
                None
                if previous_record_sha256 is None
                else _content_digest_mapping_v1(
                    previous_record_sha256, field="previous_record_sha256"
                )
            ),
        }
        previous_record_sha256 = _sha256(record)
        records.append(record)

    return records


def append_ledger_event_v1(
    value: Mapping[str, Any],
    *,
    event_type: str,
    frontier_ordinal: int,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a new ledger with one semantically valid event appended."""

    ledger = validate_expedition_ledger_v1(value)
    if event_type not in LEDGER_EVENT_TYPES_V1:
        raise LedgerTransitionError(f"unsupported ledger event type: {event_type}")
    if type(frontier_ordinal) is not int or frontier_ordinal < 0:
        raise LedgerTransitionError("frontier_ordinal must be a non-negative integer")
    if not isinstance(payload, Mapping):
        raise LedgerTransitionError("event payload must be an object")
    canonical_payload = _canonical_clone(dict(payload), label="event payload")
    body = {
        "schema_version": LEDGER_EVENT_SCHEMA_V1,
        "event_index": len(ledger["events"]),
        "event_type": event_type,
        "frontier_ordinal": frontier_ordinal,
        "previous_event_sha256": ledger["head_event_sha256"],
        "payload": canonical_payload,
    }
    event = {**body, "event_sha256": _sha256(body)}
    candidate = {
        **ledger,
        "events": [*ledger["events"], event],
        "head_event_sha256": event["event_sha256"],
    }
    validate_expedition_ledger_v1(candidate)
    return candidate


def _immutable_canonical_write_v1(path: Path, value: Mapping[str, Any]) -> None:
    encoded = canonical_json_bytes(dict(value))
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".append-stage-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise LedgerConflictError(
                f"immutable ledger path already exists: {path}"
            ) from exc
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _read_canonical_object_v1(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise LedgerIntegrityError(f"{label} is not a regular file: {path}")
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LedgerIntegrityError(f"{label} is unreadable: {path}") from exc
    if not isinstance(value, Mapping) or raw != canonical_json_bytes(value):
        raise LedgerIntegrityError(f"{label} is not canonical JSON: {path}")
    return dict(value)


def initialize_ledger_store_v1(root: str | Path, ledger: Mapping[str, Any]) -> Path:
    """Create an immutable header and empty append-only event directory."""

    value = validate_expedition_ledger_v1(ledger)
    if value["events"]:
        raise LedgerTransitionError(
            "a ledger store must initialize from an empty ledger"
        )
    target = Path(root).expanduser().absolute()
    if target.is_symlink():
        raise LedgerIntegrityError("ledger store root must not be a symbolic link")
    target.mkdir(parents=True, exist_ok=True)
    events = target / LEDGER_EVENTS_DIRECTORY_V1
    if events.is_symlink():
        raise LedgerIntegrityError(
            "ledger events directory must not be a symbolic link"
        )
    events.mkdir(exist_ok=True)
    _immutable_canonical_write_v1(target / LEDGER_HEADER_FILE_V1, value)
    return target


def read_ledger_store_v1(root: str | Path) -> dict[str, Any]:
    """Reconstruct a ledger exclusively from its immutable header and events."""

    target = Path(root).expanduser().absolute()
    if target.is_symlink() or not target.is_dir():
        raise LedgerIntegrityError("ledger store root is not a regular directory")
    header = _read_canonical_object_v1(
        target / LEDGER_HEADER_FILE_V1, label="ledger header"
    )
    validated_header = validate_expedition_ledger_v1(header)
    if validated_header["events"]:
        raise LedgerIntegrityError("ledger store header must remain event-free")
    events_root = target / LEDGER_EVENTS_DIRECTORY_V1
    if events_root.is_symlink() or not events_root.is_dir():
        raise LedgerIntegrityError("ledger events directory is invalid")
    indexed_paths: list[tuple[int, Path]] = []
    for child in events_root.iterdir():
        if child.name.startswith(".append-stage-"):
            continue
        match = _EVENT_FILE_RE_V1.fullmatch(child.name)
        if match is None:
            raise LedgerIntegrityError(f"unexpected ledger event path: {child}")
        indexed_paths.append((int(match.group(1)), child))
    indexed_paths.sort()
    if [index for index, _path in indexed_paths] != list(range(len(indexed_paths))):
        raise LedgerIntegrityError("ledger event files are not contiguous from zero")
    events = [
        _read_canonical_object_v1(path, label=f"ledger event {index}")
        for index, path in indexed_paths
    ]
    head = validated_header["genesis_sha256"]
    if events:
        head = events[-1].get("event_sha256")
    candidate = {
        **validated_header,
        "events": events,
        "head_event_sha256": head,
    }
    return validate_expedition_ledger_v1(candidate)


def append_ledger_store_event_v1(
    root: str | Path,
    *,
    event_type: str,
    frontier_ordinal: int,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Append exactly one no-overwrite event and return the reconstructed ledger."""

    target = Path(root).expanduser().absolute()
    ledger = read_ledger_store_v1(target)
    candidate = append_ledger_event_v1(
        ledger,
        event_type=event_type,
        frontier_ordinal=frontier_ordinal,
        payload=payload,
    )
    event = candidate["events"][-1]
    event_path = (
        target / LEDGER_EVENTS_DIRECTORY_V1 / f"{int(event['event_index']):08d}.json"
    )
    _immutable_canonical_write_v1(event_path, event)
    return read_ledger_store_v1(target)


__all__ = [
    "ACTION_DISPATCH_BOUND",
    "ACTION_SETTLED",
    "ACTION_WAIT",
    "AdaptiveExpeditionLedgerError",
    "DECISION_PREPARED",
    "EXPEDITION_TERMINAL",
    "FRONTIER_OBSERVED",
    "LedgerConflictError",
    "LedgerIntegrityError",
    "LedgerTransitionError",
    "STOP_REQUESTED",
    "append_ledger_event_v1",
    "append_ledger_store_event_v1",
    "initialize_ledger_store_v1",
    "new_expedition_ledger_v1",
    "project_ledger_to_public_journal_v1",
    "read_ledger_store_v1",
    "reconstruct_expedition_frontier_v1",
    "validate_expedition_ledger_v1",
]
