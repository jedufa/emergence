"""Fresh-process verification runner for the V1.10.3 Scheduler gate.

Process-owning tests deliberately do not share one pytest interpreter.  This
module provides one conventional command while retaining that isolation.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

SCHEMA_VERSION = "eiel_scheduler_verification_ledger_v1"
DEFAULT_NODEIDS = Path(__file__).with_name("verification") / "v1_10_3_frozen_nodeids.txt"


def _tail(text: str, lines: int = 30) -> str:
    rows = text.splitlines()
    return "\n".join(rows[-lines:])


def _run_node(repo: Path, nodeid: str, timeout_seconds: float) -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", "-q", nodeid]
    started = time.monotonic()
    proc = subprocess.Popen(
        command,
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=max(1.0, timeout_seconds))
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            stdout, stderr = proc.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            stdout, stderr = proc.communicate()
    elapsed = time.monotonic() - started
    status = "TIMEOUT" if timed_out else ("PASSED" if proc.returncode == 0 else "FAILED")
    return {
        "nodeid": nodeid,
        "status": status,
        "returncode": proc.returncode,
        "duration_seconds": round(elapsed, 3),
        "command": command,
        "stdout_tail": _tail(stdout),
        "stderr_tail": _tail(stderr),
    }


def _read_nodeids(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"frozen node-id file is missing: {path}")
    values = [row.strip() for row in path.read_text(encoding="utf-8").splitlines()]
    nodeids = [row for row in values if row and not row.startswith("#")]
    if not nodeids:
        raise ValueError(f"frozen node-id file is empty: {path}")
    if len(nodeids) != len(set(nodeids)):
        raise ValueError(f"frozen node-id file contains duplicates: {path}")
    return nodeids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodeids-file", type=Path, default=DEFAULT_NODEIDS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo = args.repo_root.expanduser().resolve()
    nodeids = _read_nodeids(args.nodeids_file.expanduser().resolve())
    results: list[dict[str, Any]] = []
    for index, nodeid in enumerate(nodeids, start=1):
        print(f"[{index}/{len(nodeids)}] {nodeid}", flush=True)
        result = _run_node(repo, nodeid, args.timeout_seconds)
        results.append(result)
        print(f"  {result['status']} ({result['duration_seconds']}s)", flush=True)
        if result["status"] != "PASSED":
            if result.get("stdout_tail"):
                print("  stdout tail:", flush=True)
                print(result["stdout_tail"], flush=True)
            if result.get("stderr_tail"):
                print("  stderr tail:", flush=True)
                print(result["stderr_tail"], flush=True)
    counts = {
        status: sum(1 for row in results if row["status"] == status)
        for status in ("PASSED", "FAILED", "TIMEOUT")
    }
    ledger = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "repo_root": str(repo),
        "nodeids_file": str(args.nodeids_file),
        "target_count": len(nodeids),
        "counts": counts,
        "status": "PASSED" if counts["FAILED"] == 0 and counts["TIMEOUT"] == 0 else "FAILED",
        "results": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: ledger[key] for key in ("status", "target_count", "counts")}, sort_keys=True))
    return 0 if ledger["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
