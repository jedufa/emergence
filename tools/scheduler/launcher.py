"""Composable worker launchers for the optional EIEL scheduler."""

from __future__ import annotations

import json
import hashlib
import os
import shutil
import signal
import subprocess
import threading
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from tools.actors.common import ActorError, atomic_write_text


@dataclass(frozen=True)
class LaunchResult:
    returncode: int
    pid: int
    command: list[str]
    stdout_path: str
    stderr_path: str
    result_path: str
    worker_identity: dict[str, Any]


HeartbeatCallback = Callable[[int], None]
StartedCallback = Callable[[dict[str, Any]], None]
ThreadStartedCallback = Callable[[str], None]



def launcher_host_capability(
    *, platform_name: str | None = None, proc_root: str | Path = "/proc"
) -> dict[str, Any]:
    """Report the explicit Linux process-ownership requirement."""

    platform_value = str(platform_name or sys.platform)
    proc = Path(proc_root)
    ok = platform_value.startswith("linux") and proc.is_dir()
    return {
        "ok": ok,
        "platform": platform_value,
        "proc_root": str(proc),
        "reason": (
            "Linux /proc process identity and process-group ownership are available"
            if ok
            else "EIEL Scheduler launchers require Linux with a mounted /proc filesystem"
        ),
    }


def _require_launcher_host() -> None:
    capability = launcher_host_capability()
    if not capability["ok"]:
        raise ActorError(
            capability["reason"],
            code="worker_launcher_host_unsupported",
            problems=[f"platform={capability['platform']}", f"proc_root={capability['proc_root']}"],
        )

def process_start_identity(pid: int) -> str | None:
    """Return a best-effort identity that distinguishes PID reuse."""

    try:
        text = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8")
        remainder = text[text.rfind(")") + 2 :].split()
        start_ticks = remainder[19]
        boot = Path("/proc/sys/kernel/random/boot_id")
        boot_id = boot.read_text(encoding="utf-8").strip() if boot.is_file() else "unknown-boot"
        return f"linux-proc:{boot_id}:{start_ticks}"
    except (OSError, ValueError, IndexError):
        return None


def process_identity_is_live(pid: int | None, identity: str | None) -> bool:
    if pid is None or int(pid) <= 0:
        return False
    observed = process_start_identity(int(pid))
    return observed is not None and bool(identity) and observed == identity



def process_group_members(process_group_id: int) -> list[int]:
    """Return live members of one Linux session/process group."""

    members: list[int] = []
    if int(process_group_id) <= 0 or not Path("/proc").is_dir():
        return members
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            text = (path / "stat").read_text(encoding="utf-8")
            remainder = text[text.rfind(")") + 2 :].split()
            state = remainder[0]
            process_group = int(remainder[2])
            session_id = int(remainder[3])
        except (OSError, ValueError, IndexError):
            continue
        if process_group == int(process_group_id) and session_id == int(process_group_id) and state != "Z":
            members.append(int(path.name))
    return members


def process_group_is_live(process_group_id: int | None) -> bool:
    return bool(process_group_id and process_group_members(int(process_group_id)))


def terminate_process_group_id(process_group_id: int | None, *, grace_seconds: float = 2.0) -> None:
    """Terminate a Scheduler-owned session even when its leader already exited."""

    if process_group_id is None or int(process_group_id) <= 0:
        return
    pgid = int(process_group_id)
    if not process_group_is_live(pgid):
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    deadline = time.monotonic() + max(0.1, float(grace_seconds))
    while time.monotonic() < deadline:
        if not process_group_is_live(pgid):
            return
        time.sleep(0.05)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and process_group_is_live(pgid):
        time.sleep(0.05)


def terminate_process_group(proc: subprocess.Popen[str] | subprocess.Popen[bytes], *, grace_seconds: float = 2.0) -> None:
    """Terminate the complete wrapper group, including after leader exit."""

    terminate_process_group_id(proc.pid, grace_seconds=grace_seconds)
    if proc.poll() is None:
        try:
            proc.wait(timeout=max(0.1, float(grace_seconds)))
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except OSError:
                pass
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass


def _install_worker_signal_guard(proc: subprocess.Popen[str]):
    previous: dict[int, Any] = {}
    if threading.current_thread() is not threading.main_thread():
        return lambda: None
    def handler(signum, _frame):
        terminate_process_group(proc)
        raise KeyboardInterrupt(f"scheduler received signal {signum}")
    for sig in (signal.SIGTERM, signal.SIGINT):
        previous[sig] = signal.getsignal(sig)
        signal.signal(sig, handler)
    def restore() -> None:
        for sig, value in previous.items():
            signal.signal(sig, value)
    return restore

def _wait_with_heartbeat(
    proc: subprocess.Popen[str] | subprocess.Popen[bytes],
    *,
    timeout_seconds: int | None,
    heartbeat: HeartbeatCallback | None,
    heartbeat_interval_seconds: float,
    progress: Callable[[], None] | None = None,
) -> int:
    started = time.monotonic()
    heartbeat_interval = max(0.25, float(heartbeat_interval_seconds))
    poll_interval = min(0.25, heartbeat_interval)
    next_heartbeat = started + heartbeat_interval
    while True:
        if progress is not None:
            progress()
        remaining = None if timeout_seconds is None else max(0.0, float(timeout_seconds) - (time.monotonic() - started))
        if remaining is not None and remaining <= 0:
            terminate_process_group(proc)
            raise ActorError(f"worker exceeded {timeout_seconds}s timeout", code="worker_timeout")
        try:
            until_heartbeat = max(0.0, next_heartbeat - time.monotonic())
            wait_for = min(poll_interval, until_heartbeat or poll_interval)
            if remaining is not None:
                wait_for = min(wait_for, remaining)
            returncode = proc.wait(timeout=max(0.01, wait_for))
            if progress is not None:
                progress()
            return returncode
        except subprocess.TimeoutExpired:
            if heartbeat is not None and time.monotonic() >= next_heartbeat:
                try:
                    heartbeat(proc.pid)
                    next_heartbeat = time.monotonic() + heartbeat_interval
                except Exception:
                    terminate_process_group(proc)
                    raise


def _launch_paths(logs: Path) -> tuple[Path, Path]:
    return logs / "launch_handshake.json", logs / "launch_ack.txt"


def launch_contract_for_assignment(assignment: Mapping[str, Any], lease: Mapping[str, Any]) -> dict[str, str]:
    assignment_id = str(assignment.get("record_id") or "")
    lease_id = str(lease.get("record_id") or "")
    material_digest = str(dict(assignment.get("payload") or {}).get("assignment_material_sha256") or "")
    nonce = hashlib.sha256(
        f"{assignment_id}\0{lease_id}\0{material_digest}".encode("utf-8")
    ).hexdigest()
    return {
        "assignment_id": assignment_id,
        "lease_id": lease_id,
        "launch_nonce": nonce,
    }


def _start_wrapped_process(
    *,
    target_command: list[str],
    logs: Path,
    stdout_path: Path,
    stderr_path: Path,
    identity: dict[str, Any],
    on_started: StartedCallback | None,
    maximum_log_bytes: int = 2 * 1024 * 1024,
) -> tuple[subprocess.Popen[bytes], dict[str, Any], Path, Path]:
    """Start the durable ACK-gated wrapper and bind its supervisor identity."""

    _require_launcher_host()
    logs.mkdir(parents=True, exist_ok=True)
    handshake_path, ack_path = _launch_paths(logs)
    thread_identity_path = logs / "thread_identity.json"
    for path in (handshake_path, ack_path, stdout_path, stderr_path, thread_identity_path):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    wrapper_command = [
        sys.executable, "-m", "tools.scheduler.exec_wrapper",
        "--handshake-path", str(handshake_path),
        "--ack-path", str(ack_path),
        "--stdout-path", str(stdout_path),
        "--stderr-path", str(stderr_path),
        "--thread-identity-path", str(thread_identity_path),
        "--assignment-id", str(identity.get("assignment_id") or ""),
        "--lease-id", str(identity.get("lease_id") or ""),
        "--launch-nonce", str(identity.get("launch_nonce") or ""),
        "--maximum-log-bytes", str(int(maximum_log_bytes)),
        "--", *target_command,
    ]
    proc: subprocess.Popen[bytes] = subprocess.Popen(
        wrapper_command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=Path(__file__).resolve().parents[2],
        start_new_session=True,
    )
    deadline = time.monotonic() + 10.0
    handshake: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        if handshake_path.is_file():
            try:
                handshake = json.loads(handshake_path.read_text(encoding="utf-8"))
                break
            except (json.JSONDecodeError, OSError):
                pass
        if proc.poll() is not None:
            break
        time.sleep(0.02)
    if handshake is None:
        terminate_process_group(proc)
        raise ActorError("worker launch wrapper did not publish its identity", code="worker_launch_handshake_missing")
    bound = {
        **identity,
        "pid": int(handshake.get("supervisor_pid") or proc.pid),
        "process_start_identity": handshake.get("supervisor_process_start_identity") or process_start_identity(proc.pid),
        "process_group_id": proc.pid,
        "handshake_path": str(handshake_path),
        "ack_path": str(ack_path),
        "thread_identity_path": str(thread_identity_path),
    }
    try:
        if on_started is not None:
            on_started(bound)
        elif os.environ.get("EIEL_SCHEDULER_TEST_MODE") == "1":
            ack = {
                "schema_version": "eiel_scheduler_launch_ack_v1",
                "assignment_id": str(identity.get("assignment_id") or ""),
                "lease_id": str(identity.get("lease_id") or ""),
                "launch_nonce": str(identity.get("launch_nonce") or ""),
            }
            atomic_write_text(ack_path, json.dumps(ack, sort_keys=True, separators=(",", ":")) + "\n")
        else:
            raise ActorError(
                "worker launcher requires the Scheduler activation callback",
                code="worker_launch_activation_required",
            )
    except BaseException:
        terminate_process_group(proc)
        raise
    return proc, bound, handshake_path, ack_path


class WorkerLauncher(Protocol):
    def launch(
        self,
        *,
        assignment: Mapping[str, Any],
        lease: Mapping[str, Any],
        state_root: str | Path,
        result_path: str | Path,
        log_dir: str | Path,
        heartbeat: HeartbeatCallback | None = None,
        heartbeat_interval_seconds: float = 60.0,
        on_started: StartedCallback | None = None,
        on_thread_started: ThreadStartedCallback | None = None,
    ) -> LaunchResult: ...


class FakeWorkerLauncher:
    def __init__(self, *, behavior: str = "checkpoint", fixture_content: str = "scheduler fake worker fixture\n", timeout_seconds: int = 300, sleep_seconds: float = 0.0) -> None:
        self.behavior = behavior
        self.fixture_content = fixture_content
        self.timeout_seconds = max(10, int(timeout_seconds))
        self.sleep_seconds = max(0.0, float(sleep_seconds))

    def launch(self, *, assignment, lease, state_root, result_path, log_dir, heartbeat=None, heartbeat_interval_seconds=60.0, on_started=None, on_thread_started=None) -> LaunchResult:
        assignment_id = str(assignment.get("record_id"))
        lease_id = str(lease.get("record_id"))
        worker_id = f"fake-{assignment_id.lower()}"
        logs = Path(log_dir)
        stdout_path = logs / "stdout.log"
        stderr_path = logs / "stderr.log"
        command = [
            sys.executable, "-m", "tools.scheduler.worker",
            "--assignment-id", assignment_id,
            "--lease-id", lease_id,
            "--state-root", str(Path(state_root).resolve()),
            "--result-path", str(Path(result_path).resolve()),
            "--worker-id", worker_id,
            "--behavior", self.behavior,
            "--fixture-content", self.fixture_content,
            "--sleep-seconds", str(self.sleep_seconds),
        ]
        identity = {
            "kind": "fake",
            "worker_id": worker_id,
            "thread_id": f"FAKE-THREAD-{assignment_id}",
            **launch_contract_for_assignment(assignment, lease),
        }
        proc, identity, _handshake_path, _ack_path = _start_wrapped_process(
            target_command=command, logs=logs, stdout_path=stdout_path, stderr_path=stderr_path,
            identity=identity, on_started=on_started,
        )
        restore_signals = _install_worker_signal_guard(proc)
        try:
            if on_thread_started is not None:
                on_thread_started(str(identity["thread_id"]))
            returncode = _wait_with_heartbeat(
                proc,
                timeout_seconds=self.timeout_seconds,
                heartbeat=heartbeat,
                heartbeat_interval_seconds=heartbeat_interval_seconds,
            )
        except BaseException:
            terminate_process_group(proc)
            raise
        finally:
            restore_signals()
            terminate_process_group(proc, grace_seconds=0.2)
        return LaunchResult(
            returncode=returncode,
            pid=int(identity["pid"]),
            command=command,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            result_path=str(Path(result_path).resolve()),
            worker_identity=identity,
        )



def probe_codex_cli(executable: str = "codex") -> dict[str, Any]:
    host = launcher_host_capability()
    if not host["ok"]:
        return {
            "ok": False,
            "status": "UNSUPPORTED_HOST",
            "executable": None,
            "missing_flags": [],
            "reason": host["reason"],
            "host": host,
            "optionality_audit": "NOT_MEASURED",
        }
    resolved = shutil.which(executable)
    if not resolved:
        return {
            "ok": False,
            "status": "OPTIONAL_UNAVAILABLE",
            "executable": None,
            "missing_flags": [],
            "reason": "codex executable is not available on PATH",
            "optionality_audit": "NOT_MEASURED",
        }
    launcher = CodexExecLauncher(executable=resolved)
    result = launcher.probe()
    return {
        **result,
        "status": "CODEX_EXEC_READY" if result.get("ok") else "CODEX_EXEC_UNSUPPORTED",
        "executable": resolved,
        "optionality_audit": "NOT_MEASURED",
    }


class CodexExecLauncher:
    """Launch one fresh non-interactive Codex worker through ``codex exec``."""

    def __init__(self, *, executable: str = "codex") -> None:
        resolved = shutil.which(executable)
        if not resolved:
            raise ActorError("codex executable is not available on PATH", code="codex_cli_unavailable")
        self.executable = resolved

    def probe(self) -> dict[str, Any]:
        global_help = subprocess.run([self.executable, "--help"], text=True, capture_output=True, check=False)
        exec_help = subprocess.run([self.executable, "exec", "--help"], text=True, capture_output=True, check=False)
        global_text = global_help.stdout + global_help.stderr
        exec_text = exec_help.stdout + exec_help.stderr
        global_required = ("--ask-for-approval", "--cd")
        exec_required = ("--json", "--color", "--sandbox", "--output-last-message", "--output-schema", "--ephemeral")
        missing = [flag for flag in global_required if flag not in global_text]
        missing.extend(flag for flag in exec_required if flag not in exec_text)
        return {
            "ok": global_help.returncode == 0 and exec_help.returncode == 0 and not missing,
            "missing_flags": missing,
            "global_returncode": global_help.returncode,
            "exec_returncode": exec_help.returncode,
        }

    def _prompt(self, assignment: Mapping[str, Any], lease: Mapping[str, Any], *, state_root: Path, result_path: Path) -> str:
        ap = dict(assignment.get("payload") or {})
        material = dict(ap.get("material") or {})
        assignment_id = str(assignment.get("record_id") or "")
        assignment_digest = str(ap.get("assignment_material_sha256") or "")
        lease_id = str(lease.get("record_id") or "")
        session_id = str(dict(material.get("session") or {}).get("record_id") or "")
        repository = dict(material.get("repository") or {})
        return "\n".join([
            "You are one fresh EIEL Codex worker operating under an immutable Assignment.",
            "The Scheduler mechanically activated the exact Session before launching you.",
            "Do not select other work. Do not redesign architecture. Do not repair Bootstrap.",
            f"Assignment record ID: {assignment_id}",
            f"Assignment material SHA-256: {assignment_digest}",
            f"Lease record ID: {lease_id}",
            f"Session ID: {session_id}",
            f"Context Receipt path: {dict(material.get('continuation') or {}).get('context_receipt_path')}",
            f"Result JSON path: {result_path}",
            "Verify the bounded Work Package, execute only the assigned authority phase, and obey all stop conditions.",
            "After any implementation commit, create a fresh verify-only Bootstrap for the new HEAD.",
            f"Then attach it with: python -m tools.eiel sessionManager attach-bootstrap {session_id} --bootstrap-record <BST-ID> --repo-root {repository.get('repo_root')} --state-root {state_root}",
            "For a CHECKPOINTED outcome, checkpoint and then pause or interrupt the Session so the exact next action is manually resumable. Otherwise complete or block it through Session Manager.",
            "Write one structured Worker Result JSON to the required Result path. Do not mark Work complete.",
        ])

    def launch(self, *, assignment, lease, state_root, result_path, log_dir, heartbeat=None, heartbeat_interval_seconds=60.0, on_started=None, on_thread_started=None) -> LaunchResult:
        probe = self.probe()
        if not probe["ok"]:
            raise ActorError("installed Codex CLI does not satisfy the qualified exec contract", code="codex_cli_contract_unsupported", problems=probe["missing_flags"])
        material = dict(dict(assignment.get("payload") or {}).get("material") or {})
        assignment_id = str(assignment.get("record_id") or "")
        worker_id = f"codex-{assignment_id.lower()}"
        repository = dict(material.get("repository") or {})
        profile = dict(material.get("worker_profile") or {})
        logs = Path(log_dir)
        logs.mkdir(parents=True, exist_ok=True)
        stdout_path = logs / "events.jsonl"
        stderr_path = logs / "stderr.log"
        prompt_path = logs / "worker_prompt.txt"
        result = Path(result_path).resolve()
        atomic_write_text(prompt_path, self._prompt(assignment, lease, state_root=Path(state_root).resolve(), result_path=result))
        schema_path = Path(__file__).resolve().parent / "schemas" / "eiel_worker_result_v1.schema.json"
        command = [
            self.executable,
            "--ask-for-approval", "never",
            "--cd", str(repository.get("repo_root")),
            "exec",
            "--json",
            "--color", "never",
            "--sandbox", str(profile.get("sandbox") or "workspace-write"),
            "--output-schema", str(schema_path),
            "--output-last-message", str(result),
        ]
        if profile.get("model"):
            command.extend(["--model", str(profile["model"])])
        if profile.get("profile"):
            command.extend(["--profile", str(profile["profile"])])
        if profile.get("reasoning_level"):
            command.extend(["--config", f"model_reasoning_effort={json.dumps(str(profile['reasoning_level']).lower())}"])
        if profile.get("ephemeral", True):
            command.append("--ephemeral")
        command.append(prompt_path.read_text(encoding="utf-8"))
        env_command = [sys.executable, "-c", (
            "import os,sys; os.environ['EIEL_ACTOR_STATE_ROOT']=" + repr(str(Path(state_root).resolve())) + "; "
            "os.environ['EIEL_ASSIGNMENT_ID']=" + repr(assignment_id) + "; "
            "os.environ['EIEL_LEASE_ID']=" + repr(str(lease.get('record_id'))) + "; "
            "os.execvp(sys.argv[1], sys.argv[1:])"
        ), *command]
        identity = {
            "kind": "codex_exec",
            "worker_id": worker_id,
            "thread_id": None,
            **launch_contract_for_assignment(assignment, lease),
        }
        proc, identity, _handshake_path, _ack_path = _start_wrapped_process(
            target_command=env_command, logs=logs, stdout_path=stdout_path, stderr_path=stderr_path,
            identity=identity, on_started=on_started,
        )
        restore_signals = _install_worker_signal_guard(proc)
        thread_seen: list[str] = []
        thread_identity_path = Path(str(identity.get("thread_identity_path") or logs / "thread_identity.json"))

        def progress() -> None:
            if not thread_identity_path.is_file():
                return
            try:
                captured = json.loads(thread_identity_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return
            if captured.get("status") == "CONFLICT":
                raise ActorError(
                    "Codex worker emitted conflicting thread identities",
                    code="codex_thread_identity_conflict",
                )
            if captured.get("status") != "BOUND" or thread_seen:
                return
            thread_id = str(captured.get("thread_id") or "")
            if not thread_id:
                raise ActorError("thread.started did not contain a thread identity", code="codex_thread_identity_missing")
            if on_thread_started is not None:
                on_thread_started(thread_id)
            thread_seen.append(thread_id)
            identity["thread_id"] = thread_id

        try:
            returncode = _wait_with_heartbeat(
                proc, timeout_seconds=None, heartbeat=heartbeat,
                heartbeat_interval_seconds=heartbeat_interval_seconds, progress=progress,
            )
            progress()
            if not thread_seen:
                raise ActorError("Codex worker never emitted thread.started", code="codex_thread_identity_missing")
        except BaseException:
            terminate_process_group(proc)
            raise
        finally:
            restore_signals()
            terminate_process_group(proc, grace_seconds=0.2)
        return LaunchResult(
            returncode=returncode,
            pid=int(identity["pid"]),
            command=command[:-1] + ["<bounded-worker-prompt>"],
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            result_path=str(result),
            worker_identity=identity,
        )



__all__ = ["CodexExecLauncher", "FakeWorkerLauncher", "LaunchResult", "WorkerLauncher", "probe_codex_cli", "process_start_identity", "process_identity_is_live", "process_group_members", "process_group_is_live", "terminate_process_group_id", "terminate_process_group", "launch_contract_for_assignment"]
