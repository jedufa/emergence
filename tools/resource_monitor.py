"""EIEL Resource Monitor: read-only, stateless resource visibility.

ERM answers whether launching another EIEL workload looks safe right now. It
does not reserve, launch, cancel, queue, schedule, lock, or write state.
"""

from __future__ import annotations

from engine.runtime_roots import public_runtime_path

import argparse
import datetime as dt
import json
import math
import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

SCHEMA_VERSION: Final[str] = "eiel_resource_monitor_report_v1"
DEFAULT_RUNTIME_ROOT: Final[Path] = Path(public_runtime_path(''))
DEFAULT_LIVE_SLOTS: Final[Path] = DEFAULT_RUNTIME_ROOT / "live/simulation-slots"
DEFAULT_MIN_SAFE_RAM_BYTES: Final[int] = 4 * 1024**3
DEFAULT_MIN_SAFE_VRAM_BYTES: Final[int] = 2 * 1024**3
DEFAULT_MIN_SAFE_STORAGE_BYTES: Final[int] = 10 * 1024**3
DEFAULT_POLL_INTERVAL_S: Final[float] = 30.0
RECOMMENDATION_VALUES: Final[tuple[str, ...]] = ("SAFE", "CAUTION", "WAIT", "UNKNOWN")
CONFIDENCE_VALUES: Final[tuple[str, ...]] = ("HIGH", "MEDIUM", "LOW")
BLOCKER_CLASSES: Final[tuple[str, ...]] = (
    "GPU_BLOCKER",
    "CPU_CAUTION",
    "RAM_CAUTION",
    "VRAM_CAUTION",
    "STORAGE_CAUTION",
    "UNKNOWN",
)
EIEL_PROCESS_TERMS: Final[tuple[str, ...]] = (
    "eiel",
    "run_registry_parallel.py",
    "run_registry_batch.py",
    "eiel_campaign.py",
    "tools/scripts/eiel_",
    "infra.webui.server",
    "qualification",
    "replay",
)
CPU_TERMS: Final[tuple[str, ...]] = ("--device cpu", "device=cpu", "cpu")
TERMINAL_HEARTBEAT_STATES: Final[set[str]] = {
    "complete",
    "completed",
    "failed",
    "interrupted",
    "cancelled",
    "canceled",
}


@dataclass(frozen=True)
class MonitorConfig:
    repo_root: Path
    runtime_root: Path = DEFAULT_RUNTIME_ROOT
    proc_root: Path = Path("/proc")
    live_slots_root: Path = DEFAULT_LIVE_SLOTS
    cpu_sample_interval_s: float = 0.1
    stale_heartbeat_s: float = 300.0
    min_safe_ram_bytes: int = DEFAULT_MIN_SAFE_RAM_BYTES
    min_safe_vram_bytes: int = DEFAULT_MIN_SAFE_VRAM_BYTES
    min_safe_storage_bytes: int = DEFAULT_MIN_SAFE_STORAGE_BYTES
    usable_vram_gib: float | None = None
    ste_paths: tuple[Path, ...] = ()


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _parse_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace("%", "")
        if not cleaned:
            return None
        try:
            parsed = float(cleaned)
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None
    return None


def _parse_int(value: Any) -> int | None:
    parsed = _parse_float(value)
    if parsed is None:
        return None
    return int(parsed)


def _parse_nonnegative_seconds(value: Any) -> float | None:
    parsed = _parse_float(value)
    if parsed is None or parsed < 0:
        return None
    return round(parsed, 3)


def _bytes_gib(value: int | float | None) -> float | None:
    if value is None:
        return None
    return round(float(value) / float(1024**3), 3)


def _ratio(
    numerator: int | float | None, denominator: int | float | None
) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return round(float(numerator) / float(denominator), 6)


def parse_meminfo(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in text.splitlines():
        key, sep, raw = line.partition(":")
        if not sep:
            continue
        match = re.search(r"(-?\d+)", raw)
        if match:
            out[key.strip()] = int(match.group(1)) * 1024
    return out


def parse_cpuinfo(text: str) -> dict[str, Any]:
    model = ""
    logical = 0
    physical_pairs: set[tuple[str, str]] = set()
    current: dict[str, str] = {}
    for line in [*text.splitlines(), ""]:
        if not line.strip():
            if current:
                logical += 1 if "processor" in current else 0
                if "physical id" in current and "core id" in current:
                    physical_pairs.add((current["physical id"], current["core id"]))
                current = {}
            continue
        key, sep, value = line.partition(":")
        if sep:
            current[key.strip()] = value.strip()
            if key.strip() == "model name" and not model:
                model = value.strip()
    return {
        "model": model or "unknown",
        "physical_cores": len(physical_pairs) or None,
        "logical_threads": logical or os.cpu_count(),
    }


def parse_proc_stat_cpu(text: str) -> tuple[int, int] | None:
    for line in text.splitlines():
        if not line.startswith("cpu "):
            continue
        parts = line.split()[1:]
        try:
            values = [int(part) for part in parts[:10]]
        except ValueError:
            return None
        if len(values) < 4:
            return None
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        total = sum(values)
        return total, idle
    return None


def cpu_utilization_percent(
    before: tuple[int, int] | None, after: tuple[int, int] | None
) -> float | None:
    if before is None or after is None:
        return None
    total_delta = after[0] - before[0]
    idle_delta = after[1] - before[1]
    if total_delta <= 0 or idle_delta < 0:
        return None
    busy = max(0, total_delta - idle_delta)
    return round((busy / total_delta) * 100.0, 3)


def collect_cpu(
    config: MonitorConfig, *, sleep: Callable[[float], None] = time.sleep
) -> dict[str, Any]:
    cpuinfo = parse_cpuinfo(_read_text(config.proc_root / "cpuinfo"))
    load = os.getloadavg() if hasattr(os, "getloadavg") else (None, None, None)
    before = parse_proc_stat_cpu(_read_text(config.proc_root / "stat"))
    if config.cpu_sample_interval_s > 0:
        sleep(config.cpu_sample_interval_s)
    after = parse_proc_stat_cpu(_read_text(config.proc_root / "stat"))
    logical = cpuinfo.get("logical_threads")
    load1 = load[0] if load[0] is not None else None
    return {
        "model": cpuinfo["model"],
        "physical_cores": cpuinfo["physical_cores"],
        "logical_threads": logical,
        "load_average": {
            "one_minute": load[0],
            "five_minutes": load[1],
            "fifteen_minutes": load[2],
            "one_minute_per_logical_thread": _ratio(load1, logical),
        },
        "utilization_percent": cpu_utilization_percent(before, after),
    }


def collect_memory(config: MonitorConfig) -> dict[str, Any]:
    mem = parse_meminfo(_read_text(config.proc_root / "meminfo"))
    total = mem.get("MemTotal")
    available = mem.get("MemAvailable")
    swap_total = mem.get("SwapTotal")
    swap_free = mem.get("SwapFree")
    used = total - available if total is not None and available is not None else None
    swap_used = (
        swap_total - swap_free
        if swap_total is not None and swap_free is not None
        else None
    )
    return {
        "ram_total_bytes": total,
        "ram_used_bytes": used,
        "ram_available_bytes": available,
        "ram_used_gib": _bytes_gib(used),
        "ram_available_gib": _bytes_gib(available),
        "ram_utilization": _ratio(used, total),
        "swap_total_bytes": swap_total,
        "swap_used_bytes": swap_used,
        "swap_available_bytes": swap_free,
        "swap_used_gib": _bytes_gib(swap_used),
        "swap_available_gib": _bytes_gib(swap_free),
        "swap_utilization": _ratio(swap_used, swap_total),
    }


def parse_nvidia_smi_gpu_csv(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(text.splitlines()):
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            continue
        has_uuid = len(parts) >= 7
        offset = 1 if has_uuid else 0
        used_mib = _parse_int(parts[2 + offset])
        free_mib = _parse_int(parts[3 + offset])
        total_mib = _parse_int(parts[4 + offset])
        rows.append(
            {
                "index": index,
                **({"uuid": parts[0]} if has_uuid and parts[0] else {}),
                "model": parts[offset] or "unknown",
                "utilization_percent": _parse_float(parts[1 + offset]),
                "vram_used_bytes": used_mib * 1024**2 if used_mib is not None else None,
                "vram_free_bytes": free_mib * 1024**2 if free_mib is not None else None,
                "vram_total_bytes": (
                    total_mib * 1024**2 if total_mib is not None else None
                ),
                "temperature_c": _parse_float(parts[5 + offset]),
            }
        )
    return rows


def parse_nvidia_smi_process_csv(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        has_uuid = len(parts) >= 4
        offset = 1 if has_uuid else 0
        pid = _parse_int(parts[offset])
        used_mib = _parse_int(parts[2 + offset])
        if pid is None:
            continue
        rows.append(
            {
                **({"gpu_uuid": parts[0]} if has_uuid and parts[0] else {}),
                "pid": pid,
                "process_name": parts[1 + offset],
                "vram_used_bytes": used_mib * 1024**2 if used_mib is not None else None,
            }
        )
    return rows


def _nvidia_smi_process_csv_is_complete(
    text: str, rows: Sequence[Mapping[str, Any]]
) -> bool:
    """Return whether every non-empty process row was parsed semantically.

    A successful ``nvidia-smi`` exit code alone is insufficient for an
    absence proof: silently dropping one malformed process row could turn an
    occupied GPU into an apparently idle one.
    """

    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) != len(rows):
        return False
    for line, row in zip(lines, rows, strict=True):
        parts = [part.strip() for part in line.split(",")]
        if (
            len(parts) != 4
            or not parts[0]
            or _parse_int(parts[1]) is None
            or not parts[2]
            or not row.get("gpu_uuid")
            or _parse_int(row.get("pid")) is None
        ):
            return False
    return True


def _run_command(
    command: Sequence[str],
    *,
    runner: CommandRunner,
    timeout_s: float,
) -> subprocess.CompletedProcess[str] | None:
    try:
        return runner(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def collect_gpu(
    *,
    runner: CommandRunner = subprocess.run,
    timeout_s: float = 2.0,
    usable_vram_gib: float | None = None,
) -> dict[str, Any]:
    gpu_query = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,utilization.gpu,memory.used,memory.free,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        runner=runner,
        timeout_s=timeout_s,
    )
    if gpu_query is None or gpu_query.returncode != 0:
        return {
            "available": False,
            "status": "unavailable",
            "gpus": [],
            "processes": [],
            "process_query_complete": False,
            "error": "nvidia-smi unavailable or failed",
        }
    gpus = parse_nvidia_smi_gpu_csv(gpu_query.stdout)
    if usable_vram_gib is not None:
        usable_bytes = max(0, int(float(usable_vram_gib) * 1024**3))
        for row in gpus:
            total_bytes = _parse_int(row.get("vram_total_bytes"))
            row["usable_vram_bytes"] = (
                min(usable_bytes, total_bytes)
                if total_bytes is not None
                else usable_bytes
            )
    process_query = _run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        runner=runner,
        timeout_s=timeout_s,
    )
    process_query_ok = process_query is not None and process_query.returncode == 0
    processes = (
        parse_nvidia_smi_process_csv(process_query.stdout)
        if process_query_ok
        else []
    )
    process_query_complete = bool(
        process_query_ok
        and _nvidia_smi_process_csv_is_complete(process_query.stdout, processes)
    )
    return {
        "available": bool(gpus),
        "status": "detected" if gpus else "unavailable",
        "gpus": gpus,
        "processes": processes,
        "process_query_complete": process_query_complete,
        "error": (
            ""
            if process_query_complete
            else (
                "nvidia-smi compute-process query contained malformed rows"
                if process_query_ok
                else "nvidia-smi compute-process query unavailable or failed"
            )
        ),
    }


def _disk_usage(path: Path) -> dict[str, Any]:
    try:
        usage = shutil.disk_usage(path)
    except OSError as exc:
        return {"path": str(path), "available": False, "error": str(exc)}
    return {
        "path": str(path),
        "available": True,
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": int(usage.free),
        "free_gib": _bytes_gib(usage.free),
        "utilization": _ratio(usage.used, usage.total),
    }


def collect_storage(config: MonitorConfig) -> dict[str, Any]:
    paths: list[Path] = [config.repo_root, config.runtime_root]
    seen: set[str] = set()
    roots: list[dict[str, Any]] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        roots.append(_disk_usage(path))
    return {"roots": roots}


def _proc_pid_dirs(proc_root: Path) -> Iterable[Path]:
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return []
    return [entry for entry in entries if entry.name.isdigit()]


def _cmdline_text(path: Path) -> str:
    raw = _read_text(path)
    return " ".join(part for part in raw.split("\0") if part).strip()


def _abbrev(text: str, limit: int = 500) -> str:
    value = str(text)
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _proc_cwd(pid_dir: Path) -> str:
    try:
        return str((pid_dir / "cwd").resolve())
    except OSError:
        return ""


def _boot_time_seconds(proc_root: Path) -> float | None:
    for line in _read_text(proc_root / "stat").splitlines():
        if line.startswith("btime "):
            return _parse_float(line.split()[1])
    return None


def _process_start_seconds(
    pid_dir: Path, proc_root: Path, boot_time: float | None
) -> float | None:
    stat = _read_text(pid_dir / "stat")
    if not stat or boot_time is None:
        return None
    try:
        after = stat.rsplit(") ", 1)[1]
        fields = after.split()
        start_ticks = int(fields[19])
        ticks_per_second = os.sysconf(os.sysconf_names.get("SC_CLK_TCK", "SC_CLK_TCK"))
    except (IndexError, OSError, TypeError, ValueError):
        return None
    return boot_time + (start_ticks / float(ticks_per_second))


def _process_start_ticks(pid_dir: Path) -> int | None:
    stat = _read_text(pid_dir / "stat")
    if not stat:
        return None
    try:
        after = stat.rsplit(") ", 1)[1]
        return int(after.split()[19])
    except (IndexError, TypeError, ValueError):
        return None


def observe_process_identity_v1(
    pid: int,
    expected_identity: str,
    *,
    proc_root: str | Path = "/proc",
) -> dict[str, Any]:
    """Conservatively classify one Linux PID/start identity.

    ``ABSENT`` and ``REUSED`` are conclusive dead-owner observations.  Any
    missing boot or stat evidence is ``UNKNOWN`` and must never authorize a
    resource reclaim.
    """

    root = Path(proc_root)
    observed_at_utc = dt.datetime.now(dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    base = {
        "schema_version": "eiel_process_identity_observation_v1",
        "pid": pid,
        "expected_process_start_identity": expected_identity,
        "observed_at_utc": observed_at_utc,
    }
    if (
        type(pid) is not int
        or pid <= 0
        or not isinstance(expected_identity, str)
        or not expected_identity.startswith("linux-proc:")
    ):
        return {**base, "state": "UNKNOWN", "observed_process_start_identity": None}
    pid_dir = root / str(pid)
    if not pid_dir.is_dir():
        return {**base, "state": "ABSENT", "observed_process_start_identity": None}
    start_ticks = _process_start_ticks(pid_dir)
    try:
        boot_id = (root / "sys/kernel/random/boot_id").read_text(
            encoding="utf-8"
        ).strip()
    except OSError:
        boot_id = ""
    if start_ticks is None or not boot_id:
        return {**base, "state": "UNKNOWN", "observed_process_start_identity": None}
    observed_identity = f"linux-proc:{boot_id}:{start_ticks}"
    return {
        **base,
        "state": "LIVE" if observed_identity == expected_identity else "REUSED",
        "observed_process_start_identity": observed_identity,
    }


def observe_legacy_process_start_ticks_v1(
    pid: int,
    expected_process_start_ticks: int,
    *,
    proc_root: str | Path = "/proc",
) -> dict[str, Any]:
    """Observe one migration-only PID/start-ticks identity conservatively.

    Historical sealed Workhorse contexts predate the boot-ID process identity.
    This observer preserves those exact bytes instead of inventing a V2
    identity.  It is evidence only for the digest-pinned legacy migration and
    is intentionally not used by normal admission or dead-owner reclaim.
    """

    root = Path(proc_root)
    observed_at_utc = dt.datetime.now(dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    base = {
        "schema_version": "eiel_legacy_process_start_ticks_observation_v1",
        "pid": pid,
        "expected_process_start_ticks": expected_process_start_ticks,
        "observed_at_utc": observed_at_utc,
    }
    if (
        type(pid) is not int
        or pid <= 0
        or type(expected_process_start_ticks) is not int
        or expected_process_start_ticks <= 0
    ):
        return {**base, "state": "UNKNOWN", "observed_process_start_ticks": None}
    pid_dir = root / str(pid)
    if not pid_dir.is_dir():
        return {**base, "state": "ABSENT", "observed_process_start_ticks": None}
    observed_ticks = _process_start_ticks(pid_dir)
    if observed_ticks is None:
        return {**base, "state": "UNKNOWN", "observed_process_start_ticks": None}
    return {
        **base,
        "state": (
            "LIVE" if observed_ticks == expected_process_start_ticks else "REUSED"
        ),
        "observed_process_start_ticks": observed_ticks,
    }


def _arg_value(parts: Sequence[str], names: set[str]) -> str:
    for index, part in enumerate(parts):
        if part in names and index + 1 < len(parts):
            return parts[index + 1]
        for name in names:
            prefix = f"{name}="
            if part.startswith(prefix):
                return part[len(prefix) :]
    return ""


def _has_flag(parts: Sequence[str], names: set[str]) -> bool:
    return any(part in names for part in parts)


def _classify_device(command: str, gpu_pids: set[int], pid: int) -> str:
    lowered = command.lower()
    if pid in gpu_pids:
        return "gpu"
    if any(term in lowered for term in CPU_TERMS):
        return "cpu"
    return "unknown"


def _worktree_for_cwd(cwd: str, repo_root: Path) -> str:
    if not cwd:
        return ""
    path = Path(cwd)
    candidates = [repo_root, repo_root.parent / "emergence-worktrees"]
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            path.resolve().relative_to(resolved)
        except (OSError, ValueError):
            continue
        if candidate.name == "emergence-worktrees":
            parts = path.resolve().relative_to(resolved).parts
            return str(resolved / parts[0]) if parts else str(resolved)
        return str(resolved)
    return cwd if "emergence" in cwd.lower() else ""


def discover_process_workloads(
    config: MonitorConfig,
    *,
    gpu_processes: Sequence[Mapping[str, Any]] = (),
    now_seconds: float | None = None,
) -> list[dict[str, Any]]:
    now = time.time() if now_seconds is None else float(now_seconds)
    boot = _boot_time_seconds(config.proc_root)
    gpu_pids = {int(row["pid"]) for row in gpu_processes if row.get("pid") is not None}
    workloads: list[dict[str, Any]] = []
    own_pid = os.getpid()
    for pid_dir in _proc_pid_dirs(config.proc_root):
        try:
            pid = int(pid_dir.name)
        except ValueError:
            continue
        if pid == own_pid:
            continue
        command = _cmdline_text(pid_dir / "cmdline")
        cwd = _proc_cwd(pid_dir)
        command_lower = command.lower()
        worktree = _worktree_for_cwd(cwd, config.repo_root)
        command_matches_eiel = any(term in command_lower for term in EIEL_PROCESS_TERMS)
        command_matches_codex = "codex" in command_lower and bool(worktree)
        suspicious_gpu_process = pid in gpu_pids and any(
            term in command_lower for term in ("python", ".venv", "pytest", "bash")
        )
        if not (
            command_matches_eiel or command_matches_codex or suspicious_gpu_process
        ):
            continue
        parts = command.split()
        start = _process_start_seconds(pid_dir, config.proc_root, boot)
        elapsed = max(0.0, now - start) if start is not None else None
        classification = (
            "EIEL PROCESS"
            if (command_matches_eiel or command_matches_codex)
            else "UNKNOWN EIEL PROCESS"
        )
        workload = {
            "pid": pid,
            "classification": classification,
            "command": _abbrev(command),
            "campaign": _arg_value(
                parts,
                {
                    "--batch-tag",
                    "--batch_tag",
                    "--campaign",
                    "--campaign-id",
                    "--campaign_id",
                },
            ),
            "registry": _arg_value(parts, {"--registry"}),
            "runs_root": _arg_value(parts, {"--runs-root", "--runs_root"}),
            "worktree": worktree,
            "device": _classify_device(command, gpu_pids, pid),
            "elapsed_seconds": round(elapsed, 3) if elapsed is not None else None,
        }
        gpu_admission_mode = _arg_value(
            parts, {"--gpu-admission-mode", "--gpu-mode"}
        )
        workload_class = _arg_value(parts, {"--workload-class", "--workload_class"})
        peak_vram_gib = _arg_value(parts, {"--peak-vram-gib", "--peak_vram_gib"})
        if gpu_admission_mode:
            workload["gpu_admission_mode"] = gpu_admission_mode
        if workload_class:
            workload["workload_class"] = workload_class
        if peak_vram_gib:
            workload["peak_vram_gib"] = peak_vram_gib
        if _has_flag(parts, {"--exclusive-gpu", "--exclusive_gpu"}):
            workload["exclusive_gpu"] = True
        workloads.append(workload)
    return sorted(workloads, key=lambda row: int(row["pid"]))


def _parse_utc(value: Any) -> dt.datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _heartbeat_owner_is_gone_or_reused(
    config: MonitorConfig, heartbeat: Mapping[str, Any]
) -> bool:
    pid = _parse_int(heartbeat.get("pid"))
    if pid is None or pid <= 0:
        return False
    pid_dir = config.proc_root / str(pid)
    if not pid_dir.is_dir():
        return True
    recorded_start_ticks = _parse_int(heartbeat.get("pid_start_ticks"))
    if recorded_start_ticks is None or recorded_start_ticks < 0:
        return False
    observed_start_ticks = _process_start_ticks(pid_dir)
    return (
        observed_start_ticks is not None
        and observed_start_ticks != recorded_start_ticks
    )


def _heartbeat_run_is_terminal(
    config: MonitorConfig, heartbeat: Mapping[str, Any]
) -> bool:
    raw_run_root = heartbeat.get("run_root")
    if not isinstance(raw_run_root, str) or not raw_run_root.strip():
        return False
    candidate = Path(raw_run_root)
    if not candidate.is_absolute():
        return False
    try:
        runtime_root = config.runtime_root.resolve(strict=True)
        run_root = candidate.resolve(strict=True)
    except OSError:
        return False
    if not run_root.is_relative_to(runtime_root) or run_root == runtime_root:
        return False
    state_path = run_root / "execution_state.json"
    try:
        resolved_state_path = state_path.resolve(strict=True)
    except OSError:
        return False
    if not resolved_state_path.is_relative_to(run_root):
        return False
    state = _read_json(resolved_state_path)
    if state.get("schema_version") != "eiel_execution_state_v1":
        return False
    state_status = str(state.get("status") or "").lower()
    if state_status not in TERMINAL_HEARTBEAT_STATES:
        return False
    result_path = run_root / "execution_result.json"
    try:
        resolved_result_path = result_path.resolve(strict=True)
    except OSError:
        return False
    if not resolved_result_path.is_relative_to(run_root):
        return False
    result = _read_json(resolved_result_path)
    if str(result.get("schema_version") or "") not in {
        "eiel_execution_result_v1",
        "eiel_execution_result_v2",
    }:
        return False
    return str(result.get("status") or "").lower() == state_status


def discover_slot_heartbeats(
    config: MonitorConfig, *, now: dt.datetime | None = None
) -> list[dict[str, Any]]:
    clock = now or dt.datetime.now(dt.timezone.utc)
    try:
        paths = sorted(config.live_slots_root.glob("*.json"))
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for path in paths:
        data = _read_json(path)
        if not data:
            continue
        updated = _parse_utc(data.get("updated_at"))
        age = (clock - updated).total_seconds() if updated is not None else None
        status = str(data.get("status") or data.get("stage") or "unknown")
        if status.lower() in TERMINAL_HEARTBEAT_STATES:
            continue
        stale = age is None or age > config.stale_heartbeat_s
        if (
            stale
            and _heartbeat_owner_is_gone_or_reused(config, data)
            and _heartbeat_run_is_terminal(config, data)
        ):
            continue
        rows.append(
            {
                "pid": data.get("pid"),
                "status": status,
                "stale": stale,
                "age_seconds": round(age, 3) if age is not None else None,
                "device": data.get("device") or "unknown",
                "campaign": data.get("campaign_id") or "",
                "run_id": data.get("run_id") or "",
                "run_root": data.get("run_root") or "",
                "current_generation": data.get("current_generation"),
                "target_generation": data.get("target_generation"),
                "progress": data.get("progress")
                or data.get("progress_fraction")
                or "UNKNOWN",
                "progress_fraction": data.get("progress_fraction"),
                "elapsed_runtime_seconds": _parse_nonnegative_seconds(
                    data.get("elapsed_runtime_seconds")
                    or data.get("runtime_elapsed_seconds")
                    or data.get("elapsed_seconds")
                )
                or "UNKNOWN",
                "checkpoint_frequency": data.get("checkpoint_frequency") or "UNKNOWN",
                "checkpoint_progress": data.get("checkpoint_progress")
                or data.get("checkpoint_progress_fraction")
                or "UNKNOWN",
                "eta_seconds": _parse_nonnegative_seconds(data.get("eta_seconds"))
                or "UNKNOWN",
                "stage": data.get("stage") or "",
                "gpu_admission_mode": data.get("gpu_admission_mode")
                or data.get("gpu_mode"),
                "exclusive_gpu": data.get("exclusive_gpu"),
                "peak_vram_gib": data.get("peak_vram_gib")
                or data.get("declared_peak_vram_gib"),
                "workload_class": data.get("workload_class")
                or data.get("campaign_class"),
                "logical_world_count": data.get("logical_world_count"),
                "gpu_device_id": data.get("gpu_device_id"),
                "scheduler_resource_lease_id": data.get(
                    "scheduler_resource_lease_id"
                ),
                "scheduler_resource_ids": data.get("scheduler_resource_ids"),
                "scheduler_fencing_token_digest": data.get(
                    "scheduler_fencing_token_digest"
                ),
                "path": str(path),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("campaign")),
            str(row.get("run_id")),
            str(row.get("path")),
        ),
    )[:50]


def _candidate_ste_paths(
    runtime_root: Path, explicit_paths: Sequence[Path]
) -> list[Path]:
    out = [path for path in explicit_paths if path.is_file()]
    search_roots = [
        runtime_root / "live",
        runtime_root / "campaigns",
        runtime_root / "evidence_raw",
    ]
    names = {
        "scientific_throughput_efficiency_report.json",
        "scientific_throughput_efficiency_summary.json",
        "ste_report.json",
        "parallel_campaign_summary.json",
    }
    for root in search_roots:
        if not root.is_dir():
            continue
        try:
            children = sorted(
                root.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True
            )[:20]
        except OSError:
            continue
        for child in children:
            if child.is_file() and child.name in names:
                out.append(child)
            if child.is_dir():
                for name in names:
                    path = child / name
                    if path.is_file():
                        out.append(path)
    dedup: dict[str, Path] = {}
    for path in out:
        dedup[str(path.resolve())] = path
    return sorted(
        dedup.values(),
        key=lambda item: item.stat().st_mtime if item.exists() else 0,
        reverse=True,
    )


def collect_ste(config: MonitorConfig) -> dict[str, Any]:
    for path in _candidate_ste_paths(config.runtime_root, config.ste_paths):
        data = _read_json(path)
        if not data:
            continue
        resource = data.get("resource_efficiency")
        scientific = data.get("scientific_efficiency") or data.get("science_throughput")
        discovery = data.get("scientific_discovery_efficiency")
        if not any(
            isinstance(item, Mapping) for item in (resource, scientific, discovery)
        ):
            continue
        return {
            "available": True,
            "path": str(path),
            "resource_efficiency": (
                dict(resource) if isinstance(resource, Mapping) else {}
            ),
            "scientific_efficiency": (
                dict(scientific) if isinstance(scientific, Mapping) else {}
            ),
            "scientific_discovery_efficiency": (
                dict(discovery) if isinstance(discovery, Mapping) else {}
            ),
            "primary_science_speed": (
                dict(data["primary_science_speed"])
                if isinstance(data.get("primary_science_speed"), Mapping)
                else {}
            ),
            "scientific_throughput": (
                dict(data["scientific_throughput"])
                if isinstance(data.get("scientific_throughput"), Mapping)
                else {}
            ),
            "qualification_throughput": (
                dict(data["qualification_throughput"])
                if isinstance(data.get("qualification_throughput"), Mapping)
                else {}
            ),
            "evidence_throughput": (
                dict(data["evidence_throughput"])
                if isinstance(data.get("evidence_throughput"), Mapping)
                else {}
            ),
            "throughput_taxonomy": (
                dict(data["throughput_taxonomy"])
                if isinstance(data.get("throughput_taxonomy"), Mapping)
                else {}
            ),
        }
    return {
        "available": False,
        "status": "STE unavailable",
        "resource_efficiency": {},
        "scientific_efficiency": {},
        "scientific_discovery_efficiency": {},
        "primary_science_speed": {},
        "scientific_throughput": {},
        "qualification_throughput": {},
        "evidence_throughput": {},
        "throughput_taxonomy": {},
    }


def discover_campaign_manifests(config: MonitorConfig) -> list[dict[str, Any]]:
    root = config.runtime_root / "campaigns"
    if not root.is_dir():
        return []
    try:
        paths = sorted(
            root.glob("*/campaign_manifest.json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )[:20]
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for path in paths:
        data = _read_json(path)
        if not data:
            continue
        rows.append(
            {
                "path": str(path),
                "campaign_id": data.get("campaign_id")
                or data.get("id")
                or path.parent.name,
                "status": data.get("status") or "UNKNOWN",
                "device": data.get("device")
                or data.get("execution_device")
                or "UNKNOWN",
                "created_at": data.get("created_at_utc")
                or data.get("created_at")
                or "UNKNOWN",
            }
        )
    return rows


def _blocker(
    kind: str,
    blocker_class: str,
    *,
    source: str,
    pid: Any = None,
    campaign: Any = "",
    worktree: Any = "",
    run_id: Any = "",
    eta_seconds: Any = "UNKNOWN",
    reason: str = "",
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "kind": kind,
        "class": blocker_class if blocker_class in BLOCKER_CLASSES else "UNKNOWN",
        "source": source,
        "pid": pid if pid not in ("", None) else "UNKNOWN",
        "campaign": campaign or "UNKNOWN",
        "worktree": worktree or "UNKNOWN",
        "run_id": run_id or "UNKNOWN",
        "eta_seconds": (
            eta_seconds
            if _parse_nonnegative_seconds(eta_seconds) is not None
            else "UNKNOWN"
        ),
        "reason": reason,
    }
    if details:
        row["details"] = dict(details)
    return row


def _known_text(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text.upper() == "UNKNOWN" else text


def gpu_occupancy_state(report: Mapping[str, Any]) -> str:
    """Describe observed CUDA occupancy without making a demand-specific decision."""

    gpu = dict(report.get("gpu") or {})
    if not gpu.get("available"):
        return "UNKNOWN"
    if gpu.get("processes"):
        return "CONTENDED"
    for row in (*list(report.get("running_eiel_workloads") or []), *list(report.get("slot_heartbeats") or [])):
        if _gpu_device(row) and not bool(row.get("stale")):
            return "CONTENDED"
    return "IDLE"


def _gpu_device(row: Mapping[str, Any]) -> bool:
    return str(row.get("device") or "").lower() in {"gpu", "cuda"}


def build_blockers(
    report: Mapping[str, Any], config: MonitorConfig
) -> list[dict[str, Any]]:
    gpu = dict(report.get("gpu") or {})
    memory = dict(report.get("memory") or {})
    storage = dict(report.get("storage") or {})
    workloads = list(report.get("running_eiel_workloads") or [])
    heartbeats = list(report.get("slot_heartbeats") or [])
    blockers: list[dict[str, Any]] = []

    if not gpu.get("available"):
        blockers.append(
            _blocker(
                "gpu_telemetry_unavailable",
                "UNKNOWN",
                source="nvidia_smi",
                reason="GPU telemetry is unavailable; CUDA launch safety cannot be assessed.",
            )
        )

    for row in workloads:
        device = str(row.get("device") or "unknown").lower()
        classification = str(row.get("classification") or "")
        if classification == "UNKNOWN EIEL PROCESS":
            blockers.append(
                _blocker(
                    "unattributed_eiel_process",
                    "UNKNOWN",
                    source="procfs",
                    pid=row.get("pid"),
                    campaign=row.get("campaign"),
                    worktree=row.get("worktree"),
                    reason="An EIEL-like process could not be attributed to an exact campaign owner.",
                )
            )
        elif device == "gpu":
            blockers.append(
                _blocker(
                    "active_gpu_campaign",
                    "GPU_BLOCKER",
                    source="procfs",
                    pid=row.get("pid"),
                    campaign=row.get("campaign"),
                    worktree=row.get("worktree"),
                    reason="A GPU EIEL workload is already active.",
                    details={"elapsed_seconds": row.get("elapsed_seconds")},
                )
            )
        elif device == "cpu":
            blockers.append(
                _blocker(
                    "active_cpu_campaign",
                    "CPU_CAUTION",
                    source="procfs",
                    pid=row.get("pid"),
                    campaign=row.get("campaign"),
                    worktree=row.get("worktree"),
                    reason="A CPU EIEL workload is active; CUDA may be usable but CPU/RAM contention is possible.",
                    details={"elapsed_seconds": row.get("elapsed_seconds")},
                )
            )

    for row in heartbeats:
        device = str(row.get("device") or "unknown").lower()
        stale = bool(row.get("stale"))
        if device in {"gpu", "cuda"} and stale:
            blockers.append(
                _blocker(
                    "stale_gpu_heartbeat",
                    "UNKNOWN",
                    source="heartbeat",
                    pid=row.get("pid"),
                    campaign=row.get("campaign"),
                    run_id=row.get("run_id"),
                    eta_seconds=row.get("eta_seconds"),
                    reason="A stale GPU heartbeat exists; ERM cannot prove whether it still matters.",
                    details={"age_seconds": row.get("age_seconds")},
                )
            )
        elif device in {"gpu", "cuda"}:
            blockers.append(
                _blocker(
                    "active_gpu_campaign",
                    "GPU_BLOCKER",
                    source="heartbeat",
                    pid=row.get("pid"),
                    campaign=row.get("campaign"),
                    run_id=row.get("run_id"),
                    eta_seconds=row.get("eta_seconds"),
                    reason="A live GPU campaign heartbeat is active.",
                    details={
                        "generation": row.get("current_generation", "UNKNOWN"),
                        "target_generation": row.get("target_generation", "UNKNOWN"),
                        "progress": row.get("progress", "UNKNOWN"),
                    },
                )
            )
        elif device == "cpu" and not stale:
            blockers.append(
                _blocker(
                    "active_cpu_campaign",
                    "CPU_CAUTION",
                    source="heartbeat",
                    pid=row.get("pid"),
                    campaign=row.get("campaign"),
                    run_id=row.get("run_id"),
                    eta_seconds=row.get("eta_seconds"),
                    reason="A live CPU campaign heartbeat is active.",
                    details={
                        "generation": row.get("current_generation", "UNKNOWN"),
                        "target_generation": row.get("target_generation", "UNKNOWN"),
                    },
                )
            )

    known_gpu_pids: set[int] = set()
    for source in (workloads, heartbeats):
        for row in source:
            if not _gpu_device(row):
                continue
            parsed_pid = _parse_int(row.get("pid"))
            if parsed_pid is not None:
                known_gpu_pids.add(parsed_pid)
    for process in gpu.get("processes") or []:
        if not isinstance(process, Mapping):
            continue
        try:
            pid = int(process.get("pid"))
        except (TypeError, ValueError):
            pid = None
        if pid is None or pid in known_gpu_pids:
            continue
        blockers.append(
            _blocker(
                "unattributed_gpu_process",
                "UNKNOWN",
                source="nvidia_smi",
                pid=pid,
                reason="A CUDA compute process is present without an attributable EIEL owner and sharing policy.",
                details={"process_name": process.get("process_name", "UNKNOWN")},
            )
        )

    gpus = list(gpu.get("gpus") or [])
    free_values = [_parse_int(row.get("vram_free_bytes")) for row in gpus]
    min_vram_free = min(
        [value for value in free_values if value is not None], default=None
    )
    if min_vram_free is not None and min_vram_free < config.min_safe_vram_bytes:
        blockers.append(
            _blocker(
                "vram_headroom_low",
                "VRAM_CAUTION",
                source="nvidia_smi",
                reason=f"VRAM free is below {config.min_safe_vram_bytes // 1024**3} GiB.",
                details={
                    "vram_free_bytes": min_vram_free,
                    "minimum_safe_vram_bytes": config.min_safe_vram_bytes,
                },
            )
        )

    ram_available = _parse_int(memory.get("ram_available_bytes"))
    if ram_available is not None and ram_available < config.min_safe_ram_bytes:
        blockers.append(
            _blocker(
                "ram_headroom_low",
                "RAM_CAUTION",
                source="procfs",
                reason=f"RAM available is below {config.min_safe_ram_bytes // 1024**3} GiB.",
                details={
                    "ram_available_bytes": ram_available,
                    "minimum_safe_ram_bytes": config.min_safe_ram_bytes,
                },
            )
        )
    swap_used = _parse_int(memory.get("swap_used_bytes")) or 0
    if swap_used > 0:
        blockers.append(
            _blocker(
                "swap_in_use",
                "RAM_CAUTION",
                source="procfs",
                reason="Swap is in use; another compile or CUDA campaign may increase reclaim pressure.",
                details={"swap_used_bytes": swap_used},
            )
        )

    for row in storage.get("roots") or []:
        if not row.get("available"):
            blockers.append(
                _blocker(
                    "storage_telemetry_unavailable",
                    "STORAGE_CAUTION",
                    source="filesystem",
                    reason=f"Storage telemetry is unavailable for {row.get('path', 'UNKNOWN')}.",
                )
            )
            continue
        free = _parse_int(row.get("free_bytes"))
        if free is not None and free < config.min_safe_storage_bytes:
            blockers.append(
                _blocker(
                    "storage_headroom_low",
                    "STORAGE_CAUTION",
                    source="filesystem",
                    reason=(
                        "Storage free space is below "
                        f"{config.min_safe_storage_bytes // 1024**3} GiB."
                    ),
                    details={
                        "path": row.get("path"),
                        "free_bytes": free,
                        "minimum_safe_storage_bytes": config.min_safe_storage_bytes,
                    },
                )
            )
    return blockers


def blocker_classification(blockers: Sequence[Mapping[str, Any]]) -> list[str]:
    present = {str(row.get("class") or "UNKNOWN") for row in blockers}
    return [name for name in BLOCKER_CLASSES if name in present]


def recommended_poll_interval_seconds(
    recommendation: Mapping[str, Any], polling_interval_s: float | None
) -> int | str:
    if str(recommendation.get("value") or "UNKNOWN").upper() == "SAFE":
        return "UNKNOWN"
    interval = (
        DEFAULT_POLL_INTERVAL_S if polling_interval_s is None else polling_interval_s
    )
    return max(1, int(math.ceil(interval)))


def next_check_after_seconds(
    recommendation: Mapping[str, Any], polling_interval_s: float | None
) -> int | str:
    return recommended_poll_interval_seconds(recommendation, polling_interval_s)


def recommend(report: Mapping[str, Any], config: MonitorConfig) -> dict[str, str]:
    blockers = list(report.get("blockers") or build_blockers(report, config))
    classes = set(blocker_classification(blockers))
    kinds = {str(row.get("kind") or "") for row in blockers}
    if "gpu_telemetry_unavailable" in kinds:
        return {
            "value": "UNKNOWN",
            "confidence": "LOW",
            "reason": "GPU telemetry is unavailable; CUDA launch safety cannot be assessed.",
        }
    if "unattributed_eiel_process" in kinds:
        return {
            "value": "UNKNOWN",
            "confidence": "LOW",
            "reason": "One or more GPU/EIEL processes could not be attributed to an exact campaign owner.",
        }
    if "GPU_BLOCKER" in classes:
        owner = next(
            (
                _known_text(row.get("campaign"))
                or _known_text(row.get("run_id"))
                or f"PID {row.get('pid')}"
                for row in blockers
                if row.get("class") == "GPU_BLOCKER"
            ),
            "active GPU workload",
        )
        return {
            "value": "WAIT",
            "confidence": "HIGH",
            "reason": f"CUDA launch is blocked by active GPU workload evidence ({owner}).",
        }
    if "VRAM_CAUTION" in classes:
        return {
            "value": "WAIT",
            "confidence": "HIGH",
            "reason": f"VRAM free is below {config.min_safe_vram_bytes // 1024**3} GiB.",
        }
    if "RAM_CAUTION" in classes and "ram_headroom_low" in kinds:
        return {
            "value": "WAIT",
            "confidence": "HIGH",
            "reason": f"RAM available is below {config.min_safe_ram_bytes // 1024**3} GiB.",
        }
    if "UNKNOWN" in classes:
        return {
            "value": "CAUTION",
            "confidence": "MEDIUM",
            "reason": "Unknown or stale operational evidence exists; verify before launching overlapping work.",
        }
    if "CPU_CAUTION" in classes:
        return {
            "value": "CAUTION",
            "confidence": "MEDIUM",
            "reason": "No GPU blocker was detected, but background CPU EIEL activity may contend for CPU or RAM.",
        }
    if "RAM_CAUTION" in classes:
        return {
            "value": "CAUTION",
            "confidence": "MEDIUM",
            "reason": "RAM caution evidence exists; another compile or CUDA campaign may increase reclaim pressure.",
        }
    if "STORAGE_CAUTION" in classes:
        return {
            "value": "CAUTION",
            "confidence": "MEDIUM",
            "reason": "Storage caution evidence exists; verify runtime headroom before launching more work.",
        }
    return {
        "value": "SAFE",
        "confidence": "HIGH",
        "reason": "GPU is idle, RAM headroom is sufficient, and no conflicting CUDA workload was detected.",
    }


def collect_report(
    config: MonitorConfig,
    *,
    runner: CommandRunner = subprocess.run,
    sleep: Callable[[float], None] = time.sleep,
    now: dt.datetime | None = None,
    now_seconds: float | None = None,
    polling_interval_s: float | None = None,
) -> dict[str, Any]:
    gpu = collect_gpu(runner=runner, usable_vram_gib=config.usable_vram_gib)
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": (
            _utc_now() if now is None else now.strftime("%Y-%m-%dT%H:%M:%SZ")
        ),
        "mode": "read_only_stateless_fresh_inspection",
        "non_goals": [
            "no_daemon",
            "no_scheduler",
            "no_queue",
            "no_reservations",
            "no_locks",
            "no_launch",
            "no_cancel",
            "no_execution_mutation",
        ],
        "cpu": collect_cpu(config, sleep=sleep),
        "memory": collect_memory(config),
        "gpu": gpu,
        "storage": collect_storage(config),
        "running_eiel_workloads": discover_process_workloads(
            config,
            gpu_processes=list(gpu.get("processes") or []),
            now_seconds=now_seconds,
        ),
        "slot_heartbeats": discover_slot_heartbeats(config, now=now),
        "campaign_manifests": discover_campaign_manifests(config),
        "ste": collect_ste(config),
    }
    report["gpu"]["occupancy_state"] = gpu_occupancy_state(report)
    report["gpu"]["compute_process_count"] = len(report["gpu"].get("processes") or [])
    report["gpu"]["observed_vram_used_bytes"] = sum(
        _parse_int(row.get("vram_used_bytes")) or 0
        for row in report["gpu"].get("processes") or []
        if isinstance(row, Mapping)
    )
    report["blockers"] = build_blockers(report, config)
    report["recommendation"] = recommend(report, config)
    report["confidence"] = report["recommendation"]["confidence"]
    report["blocker_classification"] = blocker_classification(report["blockers"])
    report["recommended_poll_interval_seconds"] = recommended_poll_interval_seconds(
        report["recommendation"], polling_interval_s
    )
    report["next_check_after_seconds"] = report["recommended_poll_interval_seconds"]
    return report


def _fmt_bytes(value: Any) -> str:
    parsed = _parse_int(value)
    if parsed is None:
        return "UNKNOWN"
    return f"{_bytes_gib(parsed):.3f} GiB"


def _fmt_percent(value: Any) -> str:
    parsed = _parse_float(value)
    if parsed is None:
        return "UNKNOWN"
    return f"{parsed:.1f}%"


def _fmt_gb(value: Any) -> str:
    parsed = _parse_int(value)
    if parsed is None:
        return "UNKNOWN"
    return f"{float(parsed) / float(1024**3):.1f}"


def _report_clock(report: Mapping[str, Any]) -> str:
    generated = _parse_utc(report.get("generated_at_utc"))
    if generated is None:
        return "UNKNOWN"
    return generated.strftime("%H:%M:%S")


def _campaign_count(report: Mapping[str, Any], device: str) -> int:
    wanted = device.lower()
    count = 0
    for row in report.get("running_eiel_workloads") or []:
        if str(row.get("device") or "").lower() == wanted:
            count += 1
    for row in report.get("slot_heartbeats") or []:
        if str(row.get("device") or "").lower() == wanted and not bool(
            row.get("stale")
        ):
            count += 1
    return count


def _first_generation(report: Mapping[str, Any]) -> str:
    for row in report.get("slot_heartbeats") or []:
        generation = row.get("current_generation")
        if generation not in (None, ""):
            return f"G{generation}"
    return "UNKNOWN"


def _first_eta(report: Mapping[str, Any]) -> str:
    for row in report.get("slot_heartbeats") or []:
        eta = _parse_nonnegative_seconds(row.get("eta_seconds"))
        if eta is not None:
            return f"{eta:g}s"
    return "UNKNOWN"


def render_compact(report: Mapping[str, Any]) -> str:
    gpu = dict(report.get("gpu") or {})
    rec = dict(report.get("recommendation") or {})
    blockers = list(report.get("blocker_classification") or [])
    first_gpu = next(iter(gpu.get("gpus") or []), {})
    gpu_state = str(gpu.get("occupancy_state") or "IDLE").title()
    if not gpu.get("available"):
        gpu_state = "UNKNOWN"
    elif "GPU_BLOCKER" in blockers:
        gpu_state = "Busy"
    elif "VRAM_CAUTION" in blockers:
        gpu_state = "VRAM caution"
    return "\n".join(
        [
            _report_clock(report),
            "",
            str(rec.get("value", "UNKNOWN")),
            "",
            "GPU:",
            gpu_state,
            "",
            "VRAM:",
            f"{_fmt_gb(first_gpu.get('vram_used_bytes'))} / {_fmt_gb(first_gpu.get('vram_total_bytes'))} GB",
            "",
            "GPU Campaigns:",
            str(_campaign_count(report, "gpu")),
            "",
            "CPU Campaigns:",
            str(_campaign_count(report, "cpu")),
            "",
            "Generation:",
            _first_generation(report),
            "",
            "ETA:",
            _first_eta(report),
            "",
            "Blockers:",
            ", ".join(blockers) if blockers else "none",
            "",
            "Poll Interval:",
            str(report.get("recommended_poll_interval_seconds", "UNKNOWN")),
            "",
            "Reason:",
            str(rec.get("reason", "No recommendation reason produced.")),
        ]
    )


def render_human(report: Mapping[str, Any]) -> str:
    cpu = dict(report.get("cpu") or {})
    memory = dict(report.get("memory") or {})
    gpu = dict(report.get("gpu") or {})
    storage = dict(report.get("storage") or {})
    workloads = list(report.get("running_eiel_workloads") or [])
    heartbeats = list(report.get("slot_heartbeats") or [])
    blockers = list(report.get("blockers") or [])
    rec = dict(report.get("recommendation") or {})
    ste = dict(report.get("ste") or {})
    lines = [
        "================================================",
        "EIEL Resource Monitor",
        "",
        "CPU",
        f"  utilization: {_fmt_percent(cpu.get('utilization_percent'))}",
        f"  load 1m: {dict(cpu.get('load_average') or {}).get('one_minute', 'UNKNOWN')}",
        f"  physical cores: {cpu.get('physical_cores', 'UNKNOWN')}",
        f"  logical threads: {cpu.get('logical_threads', 'UNKNOWN')}",
        "",
        "GPU",
    ]
    if gpu.get("available"):
        lines.append(
            f"  occupancy: {gpu.get('occupancy_state', 'UNKNOWN')} ({gpu.get('compute_process_count', len(gpu.get('processes') or []))} compute processes)"
        )
        for row in gpu.get("gpus") or []:
            lines.extend(
                [
                    f"  GPU {row.get('index', 0)}: {row.get('model', 'unknown')}",
                    f"    utilization: {_fmt_percent(row.get('utilization_percent'))}",
                    f"    VRAM used/free/total: {_fmt_bytes(row.get('vram_used_bytes'))} / {_fmt_bytes(row.get('vram_free_bytes'))} / {_fmt_bytes(row.get('vram_total_bytes'))}",
                    f"    usable CUDA VRAM: {_fmt_bytes(row.get('usable_vram_bytes'))}",
                    f"    temperature: {row.get('temperature_c', 'UNKNOWN')} C",
                ]
            )
        if gpu.get("processes"):
            lines.append("  compute processes:")
            for row in gpu.get("processes") or []:
                lines.append(
                    f"    PID {row.get('pid')}: {row.get('process_name', 'unknown')} ({_fmt_bytes(row.get('vram_used_bytes'))})"
                )
        else:
            lines.append("  compute processes: none detected")
    else:
        lines.append(f"  {gpu.get('status', 'unavailable')}: {gpu.get('error', '')}")
    lines.extend(
        [
            "",
            "Memory",
            f"  RAM used/available: {_fmt_bytes(memory.get('ram_used_bytes'))} / {_fmt_bytes(memory.get('ram_available_bytes'))}",
            f"  swap used/available: {_fmt_bytes(memory.get('swap_used_bytes'))} / {_fmt_bytes(memory.get('swap_available_bytes'))}",
            "",
            "Storage",
        ]
    )
    for row in storage.get("roots") or []:
        lines.append(f"  {row.get('path')}: free {row.get('free_gib', 'UNKNOWN')} GiB")
    lines.extend(["", "Running EIEL workloads"])
    if workloads:
        for row in workloads:
            campaign = row.get("campaign") or "UNKNOWN"
            lines.append(
                f"  PID {row.get('pid')}: {row.get('classification')} device={row.get('device')} campaign={campaign} elapsed={row.get('elapsed_seconds', 'UNKNOWN')}s"
            )
    else:
        lines.append("  none detected")
    if heartbeats:
        lines.append("")
        lines.append("Campaign progress")
        for row in heartbeats[:12]:
            lines.append(
                f"  {row.get('campaign') or 'UNKNOWN'} {row.get('status')} G{row.get('current_generation', 'UNKNOWN')}/{row.get('target_generation', 'UNKNOWN')} ETA={row.get('eta_seconds', 'UNKNOWN')} elapsed={row.get('elapsed_runtime_seconds', 'UNKNOWN')}s checkpoint={row.get('checkpoint_progress', 'UNKNOWN')}"
            )
        if len(heartbeats) > 12:
            lines.append(f"  ... {len(heartbeats) - 12} more heartbeat records")
    lines.extend(["", "Blockers"])
    if blockers:
        for row in blockers:
            lines.append(
                f"  {row.get('class', 'UNKNOWN')} {row.get('kind', 'unknown')} source={row.get('source', 'UNKNOWN')} campaign={row.get('campaign', 'UNKNOWN')} eta={row.get('eta_seconds', 'UNKNOWN')}"
            )
    else:
        lines.append("  none")
    lines.extend(["", "STE"])
    if ste.get("available"):
        lines.append(f"  source: {ste.get('path')}")
        primary = ste.get("primary_science_speed")
        aggregate = ste.get("scientific_throughput")
        qualification = ste.get("qualification_throughput")
        evidence = ste.get("evidence_throughput")
        if isinstance(primary, Mapping):
            lines.append(
                "  Primary Science Speed: "
                f"primary gen/h={primary.get('primary_world_generations_per_hour', 'UNKNOWN')}"
            )
        if isinstance(aggregate, Mapping):
            lines.append(
                "  Scientific Throughput: "
                f"worlds/s={aggregate.get('worlds_completed_per_second', 'UNKNOWN')} "
                f"world-gen/h={aggregate.get('world_generations_per_hour', 'UNKNOWN')}"
            )
        if isinstance(qualification, Mapping):
            lines.append(
                "  Qualification Throughput: "
                f"qualification gen/h={qualification.get('qualification_generations_per_hour', 'UNKNOWN')}"
            )
        if isinstance(evidence, Mapping):
            lines.append(
                "  Evidence Throughput: "
                f"packages/h={evidence.get('evidence_packages_per_hour', 'UNKNOWN')} "
                f"bytes/h={evidence.get('evidence_bytes_per_hour', 'UNKNOWN')}"
            )
        lines.append(
            f"  Resource Efficiency: {ste.get('resource_efficiency') or 'UNKNOWN'}"
        )
        lines.append(
            f"  Scientific Efficiency: {ste.get('scientific_efficiency') or 'UNKNOWN'}"
        )
        lines.append(
            f"  Scientific Discovery Efficiency: {ste.get('scientific_discovery_efficiency') or 'UNKNOWN'}"
        )
    else:
        lines.append("  STE unavailable")
    lines.extend(
        [
            "",
            "Recommendation",
            f"  {rec.get('value', 'UNKNOWN')} confidence={rec.get('confidence', 'LOW')}",
            f"  blocker_classification={report.get('blocker_classification') or []}",
            f"  recommended_poll_interval_seconds={report.get('recommended_poll_interval_seconds', 'UNKNOWN')}",
            f"  next_check_after_seconds={report.get('next_check_after_seconds', 'UNKNOWN')} (compatibility alias)",
            "",
            "Reason",
            f"  {rec.get('reason', 'No recommendation reason produced.')}",
            "================================================",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only EIEL Resource Monitor.")
    parser.add_argument(
        "command", nargs="?", choices=["erm"], help="Optional compatibility subcommand."
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit JSON instead of human-readable text."
    )
    parser.add_argument(
        "--compact", action="store_true", help="Emit compact operational text."
    )
    parser.add_argument(
        "--watch",
        nargs="?",
        const=DEFAULT_POLL_INTERVAL_S,
        type=float,
        metavar="SECONDS",
        help="Poll in the foreground every SECONDS until interrupted.",
    )
    parser.add_argument(
        "--wait-until",
        choices=RECOMMENDATION_VALUES,
        help="Poll until the recommendation reaches the requested state, then exit.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL_S,
        help="Polling interval for --wait-until when --watch is not supplied.",
    )
    parser.add_argument("--max-iterations", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--live-slots-root", type=Path)
    parser.add_argument("--proc-root", type=Path, default=Path("/proc"))
    parser.add_argument("--ste-path", type=Path, action="append", default=[])
    parser.add_argument("--cpu-sample-interval-s", type=float, default=0.1)
    parser.add_argument("--stale-heartbeat-s", type=float, default=300.0)
    parser.add_argument("--min-safe-ram-gib", type=float, default=4.0)
    parser.add_argument("--min-safe-vram-gib", type=float, default=2.0)
    parser.add_argument("--min-safe-storage-gib", type=float, default=10.0)
    parser.add_argument(
        "--usable-vram-gib",
        type=float,
        help="Optional host/GPU capacity profile used for peak shared-admission accounting.",
    )
    return parser


def _positive_interval(value: float) -> float:
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise SystemExit("polling interval must be a positive finite number")
    return float(value)


def _emit_report(
    report: Mapping[str, Any], *, json_output: bool, compact: bool
) -> None:
    if json_output:
        print(json.dumps(report, indent=2, sort_keys=True))
    elif compact:
        print(render_compact(report))
    else:
        print(render_human(report))


def _recommendation_value(report: Mapping[str, Any]) -> str:
    return str(
        dict(report.get("recommendation") or {}).get("value") or "UNKNOWN"
    ).upper()


def main(
    argv: Sequence[str] | None = None,
    *,
    collect_report_fn: Callable[..., dict[str, Any]] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> int:
    args = build_parser().parse_args(argv)
    runtime_root = args.runtime_root.expanduser().resolve()
    watch_interval = (
        _positive_interval(args.watch)
        if args.watch is not None
        else _positive_interval(args.interval)
    )
    config = MonitorConfig(
        repo_root=args.repo_root.expanduser().resolve(),
        runtime_root=runtime_root,
        proc_root=args.proc_root.expanduser().resolve(),
        live_slots_root=(
            args.live_slots_root.expanduser().resolve()
            if args.live_slots_root
            else runtime_root / "live/simulation-slots"
        ),
        cpu_sample_interval_s=max(0.0, float(args.cpu_sample_interval_s)),
        stale_heartbeat_s=max(1.0, float(args.stale_heartbeat_s)),
        min_safe_ram_bytes=int(float(args.min_safe_ram_gib) * 1024**3),
        min_safe_vram_bytes=int(float(args.min_safe_vram_gib) * 1024**3),
        min_safe_storage_bytes=int(float(args.min_safe_storage_gib) * 1024**3),
        usable_vram_gib=(
            None
            if args.usable_vram_gib is None
            else max(0.0, float(args.usable_vram_gib))
        ),
        ste_paths=tuple(Path(path).expanduser().resolve() for path in args.ste_path),
    )
    collector = collect_report if collect_report_fn is None else collect_report_fn
    should_loop = args.watch is not None or args.wait_until is not None
    iteration = 0
    while True:
        iteration += 1
        report = collector(config, polling_interval_s=watch_interval)
        matched = (
            args.wait_until is None
            or _recommendation_value(report) == str(args.wait_until).upper()
        )
        if args.watch is not None or matched:
            _emit_report(report, json_output=args.json, compact=args.compact)
        if not should_loop or matched and args.wait_until is not None:
            return 0
        if args.max_iterations and iteration >= args.max_iterations:
            return 1 if args.wait_until else 0
        sleep_fn(watch_interval)
        if args.watch is not None:
            print("", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
