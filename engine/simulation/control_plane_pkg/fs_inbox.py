from __future__ import annotations
"""simulation/control_plane_pkg/fs_inbox.py

Filesystem inbox adapter for ControlPlane command submission (Phase 2).

Contract
--------
- Operator (or tooling) drops JSON files into: <run_dir>/inbox/
- ControlPlane polls the inbox (safe boundary or outside) and submits commands.
- Processed files are moved to: <run_dir>/inbox_processed/
- Failed parses are moved to: <run_dir>/inbox_rejected/

Determinism
-----------
- Inbox processing order is deterministic:
  sort by filename (lexicographic), then by full path.
- File mtimes are ignored.
"""

from typing import Any, Dict, List, Tuple
import os
import json
import shutil


def inbox_dir(run_dir: str) -> str:
    return os.path.join(str(run_dir), "inbox")


def processed_dir(run_dir: str) -> str:
    return os.path.join(str(run_dir), "inbox_processed")


def rejected_dir(run_dir: str) -> str:
    return os.path.join(str(run_dir), "inbox_rejected")


def ensure_dirs(run_dir: str) -> None:
    os.makedirs(inbox_dir(run_dir), exist_ok=True)
    os.makedirs(processed_dir(run_dir), exist_ok=True)
    os.makedirs(rejected_dir(run_dir), exist_ok=True)


def poll_inbox(run_dir: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Return list of (filepath, decoded_json) for new inbox json files."""
    ensure_dirs(run_dir)
    inbox = inbox_dir(run_dir)
    files = []
    for fn in os.listdir(inbox):
        if not fn.endswith(".json"):
            continue
        files.append(os.path.join(inbox, fn))
    files.sort()
    out: List[Tuple[str, Dict[str, Any]]] = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if not isinstance(obj, dict):
                raise TypeError("command JSON must be an object")
            out.append((fp, obj))
        except Exception:
            # Treat as rejected; move for visibility.
            try:
                shutil.move(fp, os.path.join(rejected_dir(run_dir), os.path.basename(fp)))
            except Exception:
                pass
    return out


def mark_processed(run_dir: str, filepath: str) -> None:
    ensure_dirs(run_dir)
    try:
        shutil.move(filepath, os.path.join(processed_dir(run_dir), os.path.basename(filepath)))
    except Exception:
        pass


def mark_rejected(run_dir: str, filepath: str) -> None:
    """Move an inbox file into inbox_rejected/ for operator visibility."""
    ensure_dirs(run_dir)
    try:
        shutil.move(filepath, os.path.join(rejected_dir(run_dir), os.path.basename(filepath)))
    except Exception:
        pass
