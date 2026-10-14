from __future__ import annotations
"""simulation/control_plane_pkg/wal.py

Write-Ahead Log (WAL) for ControlPlane command lifecycle.

Phase 2 requirements:
- Append-only JSONL
- Durability rule (CMD-WAL-01): transition to APPLYING must be preceded by WAL append+flush
- Includes command_payload_hash (CMD-WAL-02): sha256 over canonical JSON payload
- Recovery: rebuild known/pending commands from WAL + event log

Governance/Determinism notes
---------------------------
- WAL corruption MUST be detected deterministically. Malformed lines are NOT skipped.
  On corruption, the simulation must fail safe (raise WalCorruptionError).
- The WAL is the durability authority for terminal states like REJECTED/CANCELLED/SUPERSEDED.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional
import os
import json

from simulation.control_plane_pkg.util_json import canonical_json_bytes


WAL_SCHEMA_VERSION = "command_wal_v1"


class WalCorruptionError(RuntimeError):
    """Raised when WAL JSONL is corrupted or schema-invalid.

    Determinism requirement: do not silently skip malformed WAL lines.
    """


@dataclass(frozen=True)
class WalEntry:
    schema_version: str
    command_uuid: str
    transition: str
    # generation_index: generation when the transition was recorded (time-of-transition)
    generation_index: int
    submission_sequence_number: int
    command_payload_hash: str
    # target_generation: scheduled generation the command is intended to apply at (time-of-effect)
    target_generation: Optional[int] = None
    command_payload: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-safe dict.

        Note on semantics:
        - generation_index: the generation when the transition was recorded (time-of-transition).
        - target_generation: the scheduled generation the command is intended to apply at (time-of-effect).
        """
        d: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "command_uuid": self.command_uuid,
            "transition": self.transition,
            "generation_index": int(self.generation_index),
            "submission_sequence_number": int(self.submission_sequence_number),
            "command_payload_hash": self.command_payload_hash,
        }
        if self.target_generation is not None:
            d["target_generation"] = int(self.target_generation)
        if self.command_payload is not None:
            d["command_payload"] = self.command_payload
        return d


def wal_dir(run_dir: str) -> str:
    """Return run_dir/wal directory path."""
    return os.path.join(str(run_dir), "wal")


def wal_path(run_dir: str) -> str:
    """Return run_dir/wal/command_wal.jsonl path."""
    return os.path.join(wal_dir(run_dir), "command_wal.jsonl")


def ensure_wal_dir(run_dir: str) -> None:
    """Create WAL directory if missing."""
    os.makedirs(wal_dir(run_dir), exist_ok=True)


def append_wal_entry(run_dir: str, entry: WalEntry) -> None:
    """Append one WAL entry and best-effort flush+fsync."""
    ensure_wal_dir(run_dir)
    p = wal_path(run_dir)
    line = json.dumps(entry.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    with open(p, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            # Best-effort; filesystems / platforms may not support fsync.
            pass


def _validate_wal_dict(d: Dict[str, Any], line_no: int) -> Dict[str, Any]:
    """Validate required WAL fields. Raises WalCorruptionError on invalid."""
    required = [
        "schema_version",
        "command_uuid",
        "transition",
        "generation_index",
        "submission_sequence_number",
        "command_payload_hash",
    ]
    for k in required:
        if k not in d:
            raise WalCorruptionError(f"WAL corruption: missing field '{k}' at line {line_no}")
    if str(d.get("schema_version")) != WAL_SCHEMA_VERSION:
        raise WalCorruptionError(f"WAL corruption: unexpected schema_version at line {line_no}")
    cu = str(d.get("command_uuid") or "")
    if not cu:
        raise WalCorruptionError(f"WAL corruption: empty command_uuid at line {line_no}")
    tr = str(d.get("transition") or "")
    if "->" not in tr:
        raise WalCorruptionError(f"WAL corruption: invalid transition '{tr}' at line {line_no}")
    # ints coercible
    try:
        int(d.get("generation_index"))
        int(d.get("submission_sequence_number"))
    except Exception as e:
        raise WalCorruptionError(f"WAL corruption: non-integer fields at line {line_no}: {e}")
    ph = str(d.get("command_payload_hash") or "")
    if not ph.startswith("sha256:"):
        raise WalCorruptionError(f"WAL corruption: invalid command_payload_hash at line {line_no}")
    return d


def iter_wal_entries(run_dir: str) -> Iterator[Dict[str, Any]]:
    """Iterate WAL entries strictly.

    Raises WalCorruptionError on any malformed JSON line or schema violation.
    """
    p = wal_path(run_dir)
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
                    raise WalCorruptionError(f"WAL corruption: JSON decode error at line {i}: {e}")
                if not isinstance(d, dict):
                    raise WalCorruptionError(f"WAL corruption: non-dict JSON at line {i}")
                yield _validate_wal_dict(d, i)

    return _gen()


def compute_command_payload_hash(payload_obj: Any) -> str:
    """Compute sha256 hash over canonical JSON encoding of payload."""
    import hashlib
    b = canonical_json_bytes(payload_obj)
    return "sha256:" + hashlib.sha256(b).hexdigest()
