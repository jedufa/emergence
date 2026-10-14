#!/usr/bin/env python3
"""Bounded Codex local-state retention using supported CLI deletion.

This adapter never edits Codex SQLite state and never guesses which session is
active.  It copies old, unlocked rollout files to HDD, verifies the copy, and
then asks the installed `codex delete --force <session-id>` command to perform
the supported deletion.  If the CLI rejects an identifier, the source stays
untouched and the report remains fail-closed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any


# Current Codex sessions use UUID identifiers. Keep the compact identifier
# accepted by older local rollouts so historical state remains inspectable,
# while the supported delete path remains fail-closed for anything else.
SESSION_ID_RE = re.compile(
    r"(?P<id>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|[0-9a-f]{26})",
    re.IGNORECASE,
)
DEFAULT_MAX_HISTORY_BYTES = 1024 * 1024 * 1024
DEFAULT_AGE_DAYS = 30


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _codex_home() -> Path:
    return Path(os.environ.get("CODEX_HOME", "~/.codex")).expanduser().resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_bytes(root: Path) -> int:
    total = 0
    if not root.exists():
        return 0
    for path in root.rglob("*"):
        try:
            if path.is_file() and not path.is_symlink():
                total += int(path.stat().st_size)
        except OSError:
            continue
    return total


def _root_usage(path: Path) -> dict[str, Any]:
    stats = os.statvfs(path)
    total = int(stats.f_blocks * stats.f_frsize)
    free = int(stats.f_bavail * stats.f_frsize)
    return {
        "path": str(path),
        "total_bytes": total,
        "free_bytes": free,
        "used_fraction": (0.0 if total == 0 else (total - free) / total),
    }


def _session_id(path: Path) -> str | None:
    match = SESSION_ID_RE.search(path.name)
    return None if match is None else match.group("id")


def _session_rows(home: Path) -> list[dict[str, Any]]:
    root = home / "sessions"
    rows: list[dict[str, Any]] = []
    if not root.is_dir():
        return rows
    for path in root.rglob("rollout-*.jsonl"):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            stat = path.stat()
            with path.open("rb") as handle:
                first_line = handle.readline(1024 * 1024)
            metadata = json.loads(first_line.decode("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            rows.append({"path": str(path), "status": "unreadable"})
            continue
        payload = metadata.get("payload") if isinstance(metadata, dict) else None
        session_id = _session_id(path)
        if isinstance(payload, dict) and isinstance(payload.get("session_id"), str):
            session_id = payload["session_id"]
        rows.append(
            {
                "path": str(path),
                "session_id": session_id,
                "bytes": int(stat.st_size),
                "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                "lock_path": str(home / "thread-writer-locks" / f"{session_id}.lock") if session_id else None,
                "status": "candidate",
            }
        )
    return rows


def _history_config(home: Path) -> dict[str, Any]:
    config = home / "config.toml"
    try:
        import tomllib

        payload = tomllib.loads(config.read_text(encoding="utf-8")) if config.is_file() else {}
    except (OSError, UnicodeError, ValueError):
        payload = {}
    history = payload.get("history") if isinstance(payload, dict) else None
    return {
        "config_path": str(config),
        "persistence": history.get("persistence") if isinstance(history, dict) else None,
        "max_bytes": history.get("max_bytes") if isinstance(history, dict) else None,
    }


def _ensure_history_cap(home: Path, *, max_bytes: int, apply: bool) -> dict[str, Any]:
    config = home / "config.toml"
    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    current = _history_config(home)
    observed = current.get("max_bytes")
    if isinstance(observed, int) and observed <= max_bytes:
        return {**current, "changed": False, "reason": "existing cap is already bounded"}
    if not apply:
        return {**current, "changed": False, "would_set_max_bytes": max_bytes}
    home.mkdir(parents=True, exist_ok=True)
    lock_path = config.with_name("config.toml.eiel-retention.lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        original = config.read_text(encoding="utf-8") if config.exists() else ""
        backup = config.with_name(f"config.toml.eiel-backup-{_now().strftime('%Y%m%dT%H%M%SZ')}")
        if config.exists():
            backup.write_text(original, encoding="utf-8")
        lines = original.splitlines()
        history_start = next((i for i, line in enumerate(lines) if line.strip() == "[history]"), None)
        if history_start is None:
            lines.extend(["", "[history]", f"max_bytes = {max_bytes}"])
        else:
            next_table = next(
                (i for i in range(history_start + 1, len(lines)) if lines[i].startswith("[") and lines[i].endswith("]")),
                len(lines),
            )
            max_line = next(
                (i for i in range(history_start + 1, next_table) if lines[i].lstrip().startswith("max_bytes")),
                None,
            )
            if max_line is None:
                lines.insert(next_table, f"max_bytes = {max_bytes}")
            else:
                lines[max_line] = f"max_bytes = {max_bytes}"
        temporary = config.with_name(f".{config.name}.{os.getpid()}.tmp")
        temporary.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        os.replace(temporary, config)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return {
        **_history_config(home),
        "changed": True,
        "backup_path": str(backup) if config.exists() else None,
        "max_bytes": max_bytes,
    }


def _archive_and_delete_session(row: dict[str, Any], *, archive_root: Path, apply: bool) -> dict[str, Any]:
    source = Path(str(row["path"])).expanduser().resolve()
    session_id = row.get("session_id")
    if not isinstance(session_id, str) or not SESSION_ID_RE.fullmatch(session_id):
        return {**row, "status": "blocked", "reason": "session id is not a supported local identifier"}
    lock_path = Path(str(row.get("lock_path") or ""))
    if lock_path.exists():
        return {**row, "status": "held", "reason": "active thread-writer lock exists"}
    destination = archive_root / source.parent.name / source.name
    if not apply:
        return {**row, "status": "eligible", "archive_path": str(destination)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    created_destination = False
    if destination.exists():
        try:
            destination_digest = _sha256(destination)
            source_digest = _sha256(source)
        except FileNotFoundError:
            return {
                **row,
                "status": "skipped_source_missing",
                "reason": "rollout disappeared before archive verification",
            }
        if destination_digest != source_digest:
            return {**row, "status": "blocked", "reason": "existing archive digest differs"}
    else:
        temporary_fd, temporary_name = tempfile.mkstemp(
            prefix=f".{source.name}.", suffix=".partial", dir=str(destination.parent)
        )
        os.close(temporary_fd)
        temporary = Path(temporary_name)
        try:
            try:
                with source.open("rb") as source_handle, temporary.open("wb") as destination_handle:
                    shutil.copyfileobj(source_handle, destination_handle, length=1024 * 1024)
                    destination_handle.flush()
                    os.fsync(destination_handle.fileno())
            except FileNotFoundError:
                # Discovery and archival are deliberately separate.  A
                # concurrent supported cleanup may consume a rollout after
                # discovery; preserve fail-closed semantics for that row but
                # let the bounded pass continue to later candidates.
                return {
                    **row,
                    "status": "skipped_source_missing",
                    "reason": "rollout disappeared before archival",
                }
            os.replace(temporary, destination)
            created_destination = True
        finally:
            temporary.unlink(missing_ok=True)
    try:
        destination_digest = _sha256(destination)
        source_digest = _sha256(source)
    except FileNotFoundError:
        if created_destination:
            destination.unlink(missing_ok=True)
        return {
            **row,
            "status": "skipped_source_missing",
            "reason": "rollout disappeared before archive verification",
        }
    if destination_digest != source_digest:
        return {**row, "status": "blocked", "reason": "archive verification failed"}
    command = ["codex", "delete", "--force", session_id]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return {
            **row,
            "status": "blocked",
            "archive_path": str(destination),
            "reason": "supported codex delete rejected session",
            "command_stderr": completed.stderr[-1000:],
        }
    return {
        **row,
        "status": "deleted_via_supported_cli",
        "archive_path": str(destination),
        "archive_sha256": _sha256(destination),
        "delete_command": command,
    }


def build_report(
    *,
    home: Path,
    age_days: int,
    archive_root: Path,
    apply: bool,
    max_history_bytes: int,
    ensure_config: bool,
) -> dict[str, Any]:
    cutoff = _now() - timedelta(days=age_days)
    rows = _session_rows(home)
    candidates: list[dict[str, Any]] = []
    for row in rows:
        if row.get("status") != "candidate":
            continue
        try:
            modified = datetime.fromisoformat(str(row["mtime_utc"]))
        except ValueError:
            continue
        if modified < cutoff:
            candidates.append(row)
    actions = [
        _archive_and_delete_session(row, archive_root=archive_root, apply=apply)
        for row in candidates
    ]
    root_usage = _root_usage(Path("/"))
    return {
        "schema_version": "eiel_codex_storage_retention_report_v1",
        "codex_home": str(home),
        "root_usage": root_usage,
        "root_floor": {
            "minimum_free_fraction": 0.10,
            "status": "below_floor" if root_usage["used_fraction"] > 0.90 else "healthy",
            "action": (
                "retain_ambiguous_root_data_and_report"
                if root_usage["used_fraction"] > 0.90
                else "none"
            ),
        },
        "codex_home_bytes": _directory_bytes(home),
        "sessions_bytes": _directory_bytes(home / "sessions"),
        "history_config": (
            _ensure_history_cap(home, max_bytes=max_history_bytes, apply=apply)
            if ensure_config
            else _history_config(home)
        ),
        "policy": {
            "resumable_window_days": age_days,
            "archive_root": str(archive_root),
            "supported_delete": "codex delete --force <session-id>",
            "sqlite_policy": "never edited by this adapter",
        },
        "session_count": len(rows),
        "candidate_count": len(candidates),
        "actions": actions,
        "blocked_count": sum(1 for row in actions if row.get("status") == "blocked"),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bounded Codex local-state retention")
    parser.add_argument("--codex-home", type=Path, default=_codex_home())
    parser.add_argument("--age-days", type=int, default=DEFAULT_AGE_DAYS)
    parser.add_argument("--max-history-bytes", type=int, default=DEFAULT_MAX_HISTORY_BYTES)
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=Path("/mnt/Projects/_Emergence-cold-storage/codex-sessions"),
    )
    parser.add_argument("--ensure-config", action="store_true")
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.age_days < 1:
        raise SystemExit("--age-days must be positive")
    report = build_report(
        home=args.codex_home.expanduser().resolve(),
        age_days=args.age_days,
        archive_root=args.archive_root.expanduser().resolve(),
        apply=args.apply,
        max_history_bytes=args.max_history_bytes,
        ensure_config=args.ensure_config,
    )
    print(json.dumps(report, indent=2, sort_keys=True) + "\n", end="")
    return 1 if report.get("blocked_count") else 0


if __name__ == "__main__":
    raise SystemExit(main())
