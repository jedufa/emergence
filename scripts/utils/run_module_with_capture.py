#!/usr/bin/env python3
"""
File: scripts/utils/run_module_with_capture.py

Purpose
-------
Run a Python module (python -m <module>) while:
- preserving colored output (child sees a real TTY via a PTY),
- streaming output live to the console,
- and teeing the same output into a timestamped log file.

Why this exists
---------------
Piping stdout/stderr (stdout=PIPE, stderr=PIPE) breaks color because most libs disable ANSI
when stdout isn't a TTY. It can also cause "looks stuck" behavior.

This PTY-backed runner fixes that:
- Child process stdout/stderr are attached to a pseudo-terminal (PTY) => isatty() == True.
- We read from the PTY master and write to both console and a log file.

Trade-offs
----------
- stdout and stderr are merged into one stream (typical of PTY approaches).
- Works on Unix-like systems (Linux/macOS). On Windows native, PTY behavior differs.
  (You're on WSL/Linux, so you're fine.)

Usage
-----
python -m scripts.utils.run_module_with_capture --module experiments.runs.validation_knob_foo_nightly --label validation_knob_foo_nightly

Exit codes
----------
Returns the child process exit code.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import pathlib
import subprocess
import sys
import time
from typing import Optional


def _default_log_dir() -> pathlib.Path:
    """Pick a stable log root that matches harness outputs if EMERGENCE_OUTPUTS_DIR is set."""
    root = os.environ.get("EMERGENCE_OUTPUTS_DIR", "runs")
    return pathlib.Path(root) / "captures"


def run_module_with_capture(module: str, label: str, python_exe: Optional[str] = None) -> int:
    """Run a module inside a PTY so colored output is preserved, and tee output to a log.

    Args:
        module: Python module path (e.g. "experiments.runs.validation_knob_x_nightly").
        label: Label used in the log filename.
        python_exe: Optional explicit python executable (defaults to current interpreter).

    Returns:
        int: child exit code.
    """
    import pty
    import select
    import tty

    log_dir = _default_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{ts}__{label}.log"

    exe = python_exe or sys.executable
    cmd = [exe, "-u", "-m", module]

    header = (
        f"\n=== run_module_with_capture (PTY) ===\n"
        f"module={module}\n"
        f"label={label}\n"
        f"log={log_path}\n"
        f"cmd={' '.join(cmd)}\n"
        f"====================================\n"
    )
    sys.stdout.write(header)
    sys.stdout.flush()

    env = os.environ.copy()
    # Encourage color for libs that respect it (child still has a TTY anyway).
    env.setdefault("TERM", "xterm-256color")
    env.setdefault("FORCE_COLOR", "1")
    env["PYTHONUNBUFFERED"] = "1"

    # Create a PTY pair. Child uses the slave as its controlling terminal.
    master_fd, slave_fd = pty.openpty()

    # Make master "raw-ish" so we don't mangle escape sequences.
    try:
        tty.setraw(master_fd)
    except Exception:
        pass

    with open(log_path, "w", encoding="utf-8", errors="replace") as log_f:
        log_f.write(header)
        log_f.flush()

        proc = subprocess.Popen(
            cmd,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,     # merged stream
            close_fds=True,
            env=env,
            text=False,          # bytes
        )

        # Parent doesn't need the slave.
        try:
            os.close(slave_fd)
        except Exception:
            pass

        # Stream loop: read from PTY master, write to console + log.
        try:
            while True:
                # Use select to avoid blocking forever if the child exits.
                r, _, _ = select.select([master_fd], [], [], 0.25)
                if r:
                    try:
                        data = os.read(master_fd, 4096)
                    except OSError:
                        data = b""
                    if data:
                        try:
                            txt = data.decode("utf-8", errors="replace")
                        except Exception:
                            txt = str(data)
                        sys.stdout.write(txt)
                        sys.stdout.flush()
                        log_f.write(txt)
                        log_f.flush()
                # Exit condition: process ended and no more output pending.
                rc = proc.poll()
                if rc is not None:
                    # Drain whatever remains quickly.
                    while True:
                        r, _, _ = select.select([master_fd], [], [], 0.05)
                        if not r:
                            break
                        try:
                            data = os.read(master_fd, 4096)
                        except OSError:
                            data = b""
                        if not data:
                            break
                        try:
                            txt = data.decode("utf-8", errors="replace")
                        except Exception:
                            txt = str(data)
                        sys.stdout.write(txt)
                        sys.stdout.flush()
                        log_f.write(txt)
                        log_f.flush()
                    footer = f"\n=== run_module_with_capture DONE === exit_code={rc}\n"
                    sys.stdout.write(footer)
                    sys.stdout.flush()
                    log_f.write(footer)
                    log_f.flush()
                    return int(rc)
        finally:
            try:
                os.close(master_fd)
            except Exception:
                pass


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser()
    p.add_argument("--module", required=True, help="Module to run, e.g. experiments.runs.validation_knob_x_nightly")
    p.add_argument("--label", required=True, help="Label for log naming")
    p.add_argument("--python", default=None, help="Optional python executable override")
    args = p.parse_args(argv)

    return run_module_with_capture(module=args.module, label=args.label, python_exe=args.python)


if __name__ == "__main__":
    raise SystemExit(main())
