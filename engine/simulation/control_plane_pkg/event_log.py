from __future__ import annotations
"""simulation/control_plane_pkg/event_log.py

Append-only executed event log for ControlPlane.

Phase 2 requirements:
- Immutable JSONL under run_dir/events/
- Each applied command produces an event record with:
  - event_id UUID
  - generation_applied
  - command_uuid + payload + payload_hash
  - deterministic ordering fields (target_generation, submission_sequence_number, command_uuid)

Governance/Determinism notes
---------------------------
- Event log corruption MUST be detected deterministically. Malformed lines are NOT skipped.
  On corruption, the simulation must fail safe (raise EventLogCorruptionError).
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterator
import os
import json
import uuid as _uuid

EVENT_SCHEMA_VERSION = "command_event_v1"


class EventLogCorruptionError(RuntimeError):
    """Raised when event log JSONL is corrupted or schema-invalid."""


@dataclass(frozen=True)
class EventRecord:
    schema_version: str
    event_id: str
    generation_applied: int
    target_generation: int
    submission_sequence_number: int
    command_uuid: str
    command_payload_hash: str
    command_payload: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "generation_applied": int(self.generation_applied),
            "target_generation": int(self.target_generation),
            "submission_sequence_number": int(self.submission_sequence_number),
            "command_uuid": self.command_uuid,
            "command_payload_hash": self.command_payload_hash,
            "command_payload": self.command_payload,
        }


def events_dir(run_dir: str) -> str:
    """Return run_dir/events directory path."""
    return os.path.join(str(run_dir), "events")


def events_path(run_dir: str) -> str:
    """Return run_dir/events/command_events.jsonl path."""
    return os.path.join(events_dir(run_dir), "command_events.jsonl")


def ensure_events_dir(run_dir: str) -> None:
    """Create events directory if missing."""
    os.makedirs(events_dir(run_dir), exist_ok=True)


def append_event(run_dir: str, rec: EventRecord) -> None:
    """Append one immutable event record and best-effort flush+fsync."""
    ensure_events_dir(run_dir)
    p = events_path(run_dir)
    line = json.dumps(rec.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    with open(p, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass


def new_event_id() -> str:
    """Generate a new UUID event id."""
    return str(_uuid.uuid4())


def _validate_event_dict(d: Dict[str, Any], line_no: int) -> Dict[str, Any]:
    required = [
        "schema_version",
        "event_id",
        "generation_applied",
        "target_generation",
        "submission_sequence_number",
        "command_uuid",
        "command_payload_hash",
        "command_payload",
    ]
    for k in required:
        if k not in d:
            raise EventLogCorruptionError(f"Event log corruption: missing field '{k}' at line {line_no}")
    if str(d.get("schema_version")) != EVENT_SCHEMA_VERSION:
        raise EventLogCorruptionError(f"Event log corruption: unexpected schema_version at line {line_no}")
    if not str(d.get("event_id") or ""):
        raise EventLogCorruptionError(f"Event log corruption: empty event_id at line {line_no}")
    if not str(d.get("command_uuid") or ""):
        raise EventLogCorruptionError(f"Event log corruption: empty command_uuid at line {line_no}")
    ph = str(d.get("command_payload_hash") or "")
    if not ph.startswith("sha256:"):
        raise EventLogCorruptionError(f"Event log corruption: invalid command_payload_hash at line {line_no}")
    try:
        int(d.get("generation_applied"))
        int(d.get("target_generation"))
        int(d.get("submission_sequence_number"))
    except Exception as e:
        raise EventLogCorruptionError(f"Event log corruption: non-integer fields at line {line_no}: {e}")
    if not isinstance(d.get("command_payload"), dict):
        raise EventLogCorruptionError(f"Event log corruption: command_payload not a dict at line {line_no}")
    return d


def iter_events(run_dir: str) -> Iterator[Dict[str, Any]]:
    """Iterate event records strictly.

    Raises EventLogCorruptionError on any malformed JSON line or schema violation.
    """
    p = events_path(run_dir)
    if not os.path.exists(p):
        return iter(())

    def _gen():
        with open(p, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                raw = line.strip()
                if not raw:
                    continue
                try:
                    d = json.loads(raw)
                except Exception as e:
                    raise EventLogCorruptionError(f"Event log corruption: JSON decode error at line {i}: {e}")
                if not isinstance(d, dict):
                    raise EventLogCorruptionError(f"Event log corruption: non-dict JSON at line {i}")
                yield _validate_event_dict(d, i)

    return _gen()


def event_log_length(run_dir: str) -> int:
    """Return count of event log records."""
    return sum(1 for _ in iter_events(run_dir))
