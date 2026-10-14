#!/usr/bin/env python3
"""Run the one allowed Genome V2 build under a memory/process guard."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

from engine.evolution.genome_upgrade_v2_build_admission_v1 import (
    build_admission_policy_v1,
    evaluate_build_snapshot_v1,
)


LOCK_PATH = Path("/tmp/eiel-genome-v2-build-v1.lock")


def _memory() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        name, raw = line.split(":", 1)
        amount = int(raw.strip().split()[0]) * 1024
        values[name] = amount
    return {
        "available_bytes": values["MemAvailable"],
        "total_bytes": values["MemTotal"],
        "swap_free_bytes": values["SwapFree"],
        "swap_total_bytes": values["SwapTotal"],
    }


def _processes() -> list[dict[str, Any]]:
    rows = []
    for child in Path("/proc").iterdir():
        if not child.name.isdigit():
            continue
        try:
            status = {
                line.split(":", 1)[0]: line.split(":", 1)[1].strip()
                for line in (child / "status").read_text(
                    encoding="ascii"
                ).splitlines()
                if ":" in line
            }
            command = (child / "cmdline").read_bytes().replace(
                b"\0", b" "
            ).decode("utf-8", errors="replace").strip()
            rows.append(
                {
                    "pid": int(child.name),
                    "ppid": int(status["PPid"]),
                    "comm": status["Name"],
                    "command": command,
                }
            )
        except (FileNotFoundError, KeyError, OSError, ValueError):
            continue
    return rows


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    temporary.replace(path)


def _snapshot(*, phase: str, guarded_root_pid: int | None = None) -> dict[str, Any]:
    memory = _memory()
    result = evaluate_build_snapshot_v1(
        memory=memory,
        processes=_processes(),
        phase=phase,
        guarded_root_pid=guarded_root_pid,
    )
    result["memory"] = memory
    result["observed_unix_ns"] = time.time_ns()
    return result


def _terminate_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait(timeout=10)


def _run(args: argparse.Namespace) -> int:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write(output / "genome_v2_build_admission_policy.json", build_admission_policy_v1())
    with LOCK_PATH.open("w", encoding="ascii") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("another guarded Genome V2 build owns the lock")
        before = _snapshot(phase="preflight")
        _write(output / "genome_v2_build_resource_preflight.json", before)
        if not before["admitted"]:
            return 2
        log_path = output / "genome_v2_build.log"
        snapshots = []
        started_ns = time.time_ns()
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                args.command,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            guard_failure = None
            while process.poll() is None:
                snapshot = _snapshot(
                    phase="monitor", guarded_root_pid=process.pid
                )
                snapshots.append(snapshot)
                if not snapshot["admitted"]:
                    guard_failure = list(snapshot["reasons"])
                    _terminate_group(process)
                    break
                time.sleep(args.interval_seconds)
            returncode = process.wait()
        after = _snapshot(phase="preflight")
        monitor = {
            "schema_version": "genome_v2_build_process_monitor_v1",
            "snapshot_count": len(snapshots),
            "snapshots": snapshots,
            "guard_failure": guard_failure,
        }
        summary = {
            "schema_version": "genome_v2_build_resource_summary_v1",
            "started_unix_ns": started_ns,
            "finished_unix_ns": time.time_ns(),
            "command": list(args.command),
            "returncode": returncode,
            "guard_failure": guard_failure,
            "preflight": before,
            "postflight": after,
            "log_path": str(log_path),
            "ok": returncode == 0 and guard_failure is None,
        }
        _write(output / "genome_v2_build_process_monitor.json", monitor)
        _write(output / "genome_v2_build_resource_summary.json", summary)
        return 0 if summary["ok"] else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        raise SystemExit("a guarded build command is required after --")
    if args.interval_seconds <= 0:
        raise SystemExit("--interval-seconds must be positive")
    return _run(args)


if __name__ == "__main__":
    sys.exit(main())
