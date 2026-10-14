"""Tiny durable launch wrapper for one Scheduler-owned worker process.

The wrapper closes the hard-crash gap between ``Popen`` and durable PID
recording.  It writes its identity, waits for an explicit Scheduler ACK, then
starts the actual worker in a separate process group.  It also owns bounded
stdout/stderr pumping and removes worker descendants before exiting.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import BinaryIO


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _process_start_identity(pid: int) -> str | None:
    try:
        text = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8")
        remainder = text[text.rfind(")") + 2 :].split()
        start_ticks = remainder[19]
        boot = Path("/proc/sys/kernel/random/boot_id")
        boot_id = boot.read_text(encoding="utf-8").strip() if boot.is_file() else "unknown-boot"
        return f"linux-proc:{boot_id}:{start_ticks}"
    except (OSError, ValueError, IndexError):
        return None


def _group_members(process_group_id: int) -> list[int]:
    members: list[int] = []
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return members
    for path in proc_root.iterdir():
        if not path.name.isdigit():
            continue
        try:
            text = (path / "stat").read_text(encoding="utf-8")
            remainder = text[text.rfind(")") + 2 :].split()
            process_group = int(remainder[2])  # field 5
            session_id = int(remainder[3])     # field 6
            state = remainder[0]
        except (OSError, ValueError, IndexError):
            continue
        if process_group == process_group_id and session_id == process_group_id and state != "Z":
            members.append(int(path.name))
    return members


def _terminate_group(process_group_id: int, *, grace_seconds: float = 1.0) -> None:
    if process_group_id <= 0 or not _group_members(process_group_id):
        return
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    deadline = time.monotonic() + max(0.1, grace_seconds)
    while time.monotonic() < deadline:
        if not _group_members(process_group_id):
            return
        time.sleep(0.05)
    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and _group_members(process_group_id):
        time.sleep(0.05)


class _BoundedLog:
    def __init__(self, path: Path, maximum_bytes: int) -> None:
        self.path = path
        self.maximum_bytes = max(4096, int(maximum_bytes))
        self.marker = b"[EIEL scheduler truncated older worker output]\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("wb")
        self.size = 0
        self.lock = threading.Lock()

    def write(self, data: bytes) -> None:
        if not data:
            return
        with self.lock:
            if self.size + len(data) <= self.maximum_bytes:
                self.handle.write(data)
                self.handle.flush()
                self.size += len(data)
                return
            self.handle.flush()
            keep = max(0, self.maximum_bytes - len(self.marker) - min(len(data), self.maximum_bytes // 2))
            tail = b""
            if keep and self.path.is_file():
                with self.path.open("rb") as source:
                    source.seek(max(0, self.size - keep))
                    tail = source.read(keep)
            combined = self.marker + tail + data
            combined = combined[-self.maximum_bytes :]
            self.handle.close()
            self.handle = self.path.open("wb")
            self.handle.write(combined)
            self.handle.flush()
            self.size = len(combined)

    def close(self) -> None:
        with self.lock:
            if not self.handle.closed:
                self.handle.close()


class _ThreadIdentityCapture:
    """Persist the first authoritative Codex thread identity outside logs."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.buffer = b""
        self.thread_id: str | None = None
        self.conflict: str | None = None
        self.lock = threading.Lock()

    def feed(self, data: bytes) -> None:
        if self.path is None or not data:
            return
        with self.lock:
            self.buffer += data
            while b"\n" in self.buffer:
                raw, self.buffer = self.buffer.split(b"\n", 1)
                try:
                    event = json.loads(raw.decode("utf-8", errors="replace"))
                except json.JSONDecodeError:
                    continue
                if event.get("type") != "thread.started":
                    continue
                value = str(event.get("thread_id") or dict(event.get("thread") or {}).get("id") or "")
                if not value:
                    continue
                if self.thread_id is None:
                    self.thread_id = value
                    _atomic_json(
                        self.path,
                        {
                            "schema_version": "eiel_codex_thread_identity_v1",
                            "status": "BOUND",
                            "thread_id": value,
                            "captured_at_unix": time.time(),
                        },
                    )
                elif self.thread_id != value:
                    self.conflict = value
                    _atomic_json(
                        self.path,
                        {
                            "schema_version": "eiel_codex_thread_identity_v1",
                            "status": "CONFLICT",
                            "thread_id": self.thread_id,
                            "conflicting_thread_id": value,
                            "captured_at_unix": time.time(),
                        },
                    )


def _pump(stream: BinaryIO, output: _BoundedLog, capture: _ThreadIdentityCapture | None = None) -> None:
    try:
        while True:
            chunk = stream.read(65536)
            if not chunk:
                return
            output.write(chunk)
            if capture is not None:
                capture.feed(chunk)
    finally:
        try:
            stream.close()
        except OSError:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--handshake-path", required=True)
    parser.add_argument("--ack-path", required=True)
    parser.add_argument("--stdout-path", required=True)
    parser.add_argument("--stderr-path", required=True)
    parser.add_argument("--thread-identity-path")
    parser.add_argument("--assignment-id", required=True)
    parser.add_argument("--lease-id", required=True)
    parser.add_argument("--launch-nonce", required=True)
    parser.add_argument("--maximum-log-bytes", type=int, default=2 * 1024 * 1024)
    parser.add_argument("--ack-timeout-seconds", type=float, default=30.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("worker command is required after --")

    handshake_path = Path(args.handshake_path).resolve()
    ack_path = Path(args.ack_path).resolve()
    stdout_path = Path(args.stdout_path).resolve()
    stderr_path = Path(args.stderr_path).resolve()
    thread_identity_path = Path(args.thread_identity_path).resolve() if args.thread_identity_path else None
    handshake = {
        "schema_version": "eiel_scheduler_launch_handshake_v1",
        "status": "WAITING_ACK",
        "supervisor_pid": os.getpid(),
        "supervisor_process_start_identity": _process_start_identity(os.getpid()),
        "created_at_unix": time.time(),
        "target_pid": None,
        "target_process_start_identity": None,
        "target_process_group_id": None,
        "returncode": None,
    }
    _atomic_json(handshake_path, handshake)

    deadline = time.monotonic() + max(1.0, float(args.ack_timeout_seconds))
    ack: dict | None = None
    while ack is None:
        if ack_path.is_file():
            try:
                candidate = json.loads(ack_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                candidate = None
            if candidate is not None:
                if (
                    candidate.get("schema_version") != "eiel_scheduler_launch_ack_v1"
                    or candidate.get("assignment_id") != args.assignment_id
                    or candidate.get("lease_id") != args.lease_id
                    or candidate.get("launch_nonce") != args.launch_nonce
                ):
                    handshake.update({"status": "ACK_INVALID", "returncode": 126})
                    _atomic_json(handshake_path, handshake)
                    return 126
                ack = candidate
                break
        if time.monotonic() >= deadline:
            handshake.update({"status": "ACK_TIMEOUT", "returncode": 124})
            _atomic_json(handshake_path, handshake)
            return 124
        time.sleep(0.02)

    child: subprocess.Popen[bytes] | None = None
    target_group = 0

    def stop(_signum, _frame) -> None:
        if target_group:
            _terminate_group(target_group, grace_seconds=0.5)
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    stdout_log = _BoundedLog(stdout_path, args.maximum_log_bytes)
    stderr_log = _BoundedLog(stderr_path, args.maximum_log_bytes)
    identity_capture = _ThreadIdentityCapture(thread_identity_path)
    threads: list[threading.Thread] = []
    try:
        child = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        target_group = child.pid
        handshake.update({
            "status": "RUNNING",
            "target_pid": child.pid,
            "target_process_start_identity": _process_start_identity(child.pid),
            "target_process_group_id": child.pid,
            "started_at_unix": time.time(),
        })
        _atomic_json(handshake_path, handshake)
        assert child.stdout is not None and child.stderr is not None
        threads = [
            threading.Thread(target=_pump, args=(child.stdout, stdout_log, identity_capture), name="eiel-worker-stdout", daemon=True),
            threading.Thread(target=_pump, args=(child.stderr, stderr_log), name="eiel-worker-stderr", daemon=True),
        ]
        for thread in threads:
            thread.start()
        returncode = child.wait()
        # Descendants may retain inherited stdout/stderr pipe descriptors after
        # the direct worker exits.  Kill the owned target group before joining
        # pump threads, otherwise two sequential five-second joins can delay
        # cleanup while ghost children keep the pipes open.
        _terminate_group(target_group, grace_seconds=0.5)
        for thread in threads:
            thread.join(timeout=2)
        if identity_capture.conflict:
            returncode = 125
            handshake.update({"status": "THREAD_IDENTITY_CONFLICT", "returncode": returncode, "ended_at_unix": time.time()})
        else:
            handshake.update({"status": "EXITED", "returncode": returncode, "ended_at_unix": time.time()})
        _atomic_json(handshake_path, handshake)
        return int(returncode)
    except KeyboardInterrupt:
        if target_group:
            _terminate_group(target_group, grace_seconds=0.5)
        handshake.update({"status": "TERMINATED", "returncode": 130, "ended_at_unix": time.time()})
        _atomic_json(handshake_path, handshake)
        return 130
    finally:
        # Group cleanup is required even when the direct child has already
        # exited; descendants can outlive the leader and retain repository or
        # log-file access.
        if target_group:
            _terminate_group(target_group, grace_seconds=0.5)
        for thread in threads:
            thread.join(timeout=1)
        stdout_log.close()
        stderr_log.close()


if __name__ == "__main__":
    raise SystemExit(main())
