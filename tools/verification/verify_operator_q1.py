"""Run the bounded OPERATOR-Q1 qualification in fresh subprocesses."""

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

DEFAULT_TARGETS = Path(__file__).parents[1] / "actors" / "verification" / "v1_11_operator_q1_targets.txt"


def _tail(value: str, lines: int = 30) -> str:
    return "\n".join(value.splitlines()[-lines:])


def _targets(path: Path) -> list[str]:
    values = [row.strip() for row in path.read_text(encoding="utf-8").splitlines()]
    rows = [row for row in values if row and not row.startswith("#")]
    if not rows or len(rows) != len(set(rows)):
        raise ValueError(f"invalid OPERATOR-Q1 target manifest: {path}")
    return rows


def _run(repo: Path, target: str, timeout_seconds: float) -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", "-q", target]
    started = time.monotonic()
    proc = subprocess.Popen(
        command, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True,
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
        "target": target,
        "status": status,
        "returncode": proc.returncode,
        "duration_seconds": round(elapsed, 3),
        "stdout_tail": _tail(stdout),
        "stderr_tail": _tail(stderr),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets-file", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo = args.repo_root.expanduser().resolve()
    targets = _targets(args.targets_file.expanduser().resolve())
    results: list[dict[str, Any]] = []
    for index, target in enumerate(targets, start=1):
        print(f"[{index}/{len(targets)}] {target}", flush=True)
        result = _run(repo, target, args.timeout_seconds)
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
        status: sum(row["status"] == status for row in results)
        for status in ("PASSED", "FAILED", "TIMEOUT")
    }
    ledger = {
        "schema_version": "eiel_operator_q1_ledger_v1",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "repo_root": str(repo),
        "targets_file": str(args.targets_file),
        "target_count": len(targets),
        "counts": counts,
        "status": "PASSED" if not counts["FAILED"] and not counts["TIMEOUT"] else "FAILED",
        "results": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: ledger[k] for k in ("status", "target_count", "counts")}, sort_keys=True))
    return 0 if ledger["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
