#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any


PROC_FIELDS = [
    "timestamp_utc",
    "label",
    "sample",
    "kind",
    "pid",
    "ppid",
    "role",
    "comm",
    "state",
    "cpu_percent",
    "cpu_num",
    "rss_kib",
    "threads",
    "voluntary_ctxt_switches",
    "nonvoluntary_ctxt_switches",
    "read_bytes",
    "write_bytes",
    "loadavg_1m",
    "loadavg_5m",
    "loadavg_15m",
    "runnable_processes",
    "sleeping_processes",
    "per_core_cpu_percent",
    "cpu_total_percent",
]

GPU_FIELDS = [
    "timestamp_utc",
    "label",
    "sample",
    "gpu_timestamp",
    "utilization_gpu_percent",
    "utilization_memory_percent",
    "memory_used_mib",
    "memory_total_mib",
    "power_draw_w",
    "temperature_gpu_c",
    "clocks_sm_mhz",
    "clocks_mem_mhz",
    "compute_app_count",
    "compute_app_pids",
    "compute_app_used_memory_mib",
    "error",
]


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _cpu_times() -> tuple[int, list[tuple[int, int]]]:
    total = 0
    cores: list[tuple[int, int]] = []
    for line in _read_text(Path("/proc/stat")).splitlines():
        if not line.startswith("cpu"):
            continue
        parts = line.split()
        values = [int(float(x)) for x in parts[1:] if x.replace(".", "", 1).isdigit()]
        if not values:
            continue
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        all_time = sum(values)
        if parts[0] == "cpu":
            total = all_time
        elif parts[0][3:].isdigit():
            cores.append((all_time, idle))
    return total, cores


def _cpu_percent(prev: tuple[int, int] | None, cur: tuple[int, int]) -> float:
    if prev is None:
        return 0.0
    delta_total = max(0, cur[0] - prev[0])
    delta_idle = max(0, cur[1] - prev[1])
    if delta_total <= 0:
        return 0.0
    return max(0.0, min(100.0, 100.0 * (delta_total - delta_idle) / delta_total))


def _loadavg() -> tuple[str, str, str]:
    parts = _read_text(Path("/proc/loadavg")).split()
    if len(parts) >= 3:
        return parts[0], parts[1], parts[2]
    return "", "", ""


def _parse_status(pid: int) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in _read_text(Path(f"/proc/{pid}/status")).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        out[key.strip()] = value.strip().split()[0] if value.strip() else ""
    return out


def _parse_io(pid: int) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in _read_text(Path(f"/proc/{pid}/io")).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        out[key.strip()] = value.strip()
    return out


def _parse_stat(pid: int) -> dict[str, Any] | None:
    raw = _read_text(Path(f"/proc/{pid}/stat")).strip()
    if not raw or ")" not in raw:
        return None
    left, right = raw.rsplit(")", 1)
    comm = left.split("(", 1)[1] if "(" in left else ""
    fields = right.strip().split()
    try:
        return {
            "comm": comm,
            "state": fields[0],
            "ppid": int(fields[1]),
            "utime": int(fields[11]),
            "stime": int(fields[12]),
            "threads": int(fields[17]),
            "cpu_num": int(fields[36]) if len(fields) > 36 else -1,
        }
    except Exception:
        return None


def _cmdline(pid: int) -> str:
    raw = _read_text(Path(f"/proc/{pid}/cmdline"))
    return raw.replace("\x00", " ").strip()


def _role(cmd: str) -> str:
    if "multiworker_contention_monitor.py" in cmd:
        return "monitor"
    if "run_registry_parallel.py" in cmd:
        return "wrapper"
    if "run_registry_batch.py" in cmd:
        return "registry_batch"
    if "engine.main" in cmd or "simulation" in cmd:
        return "engine"
    if "python" in cmd:
        return "python"
    return "other"


def _matching_pids(tokens: list[str]) -> list[int]:
    pids: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        cmd = _cmdline(pid)
        if not cmd:
            continue
        if any(token and token in cmd for token in tokens):
            pids.append(pid)
    return sorted(set(pids))


def _process_rows(
    *,
    label: str,
    sample: int,
    tokens: list[str],
    prev_proc_ticks: dict[int, int],
    interval_s: float,
    prev_cpu_total: int | None,
    prev_cores: list[tuple[int, int]] | None,
) -> tuple[list[dict[str, Any]], dict[int, int], int, list[tuple[int, int]]]:
    timestamp = _utc_now()
    total_ticks, cores = _cpu_times()
    load1, load5, load15 = _loadavg()
    cpu_total_percent = ""
    if prev_cpu_total is not None:
        cur_idle = sum(idle for _, idle in cores)
        prev_idle = sum(idle for _, idle in (prev_cores or []))
        cpu_total_percent = _cpu_percent((prev_cpu_total, prev_idle), (total_ticks, cur_idle))
    per_core = ""
    if prev_cores:
        per_core = ";".join(f"{_cpu_percent(prev, cur):.1f}" for prev, cur in zip(prev_cores, cores))

    selected: list[dict[str, Any]] = []
    state_counts = {"R": 0, "S": 0, "D": 0, "I": 0}
    clk = os.sysconf(os.sysconf_names.get("SC_CLK_TCK", "SC_CLK_TCK"))
    next_proc_ticks: dict[int, int] = {}
    for pid in _matching_pids(tokens):
        stat = _parse_stat(pid)
        if stat is None:
            continue
        status = _parse_status(pid)
        io = _parse_io(pid)
        ticks = int(stat["utime"]) + int(stat["stime"])
        next_proc_ticks[pid] = ticks
        prev_ticks = prev_proc_ticks.get(pid)
        cpu_percent = ""
        if prev_ticks is not None and interval_s > 0:
            cpu_percent = 100.0 * max(0, ticks - prev_ticks) / float(clk) / interval_s
        state = str(stat.get("state", ""))
        state_counts[state] = state_counts.get(state, 0) + 1
        cmd = _cmdline(pid)
        selected.append(
            {
                "timestamp_utc": timestamp,
                "label": label,
                "sample": sample,
                "kind": "process",
                "pid": pid,
                "ppid": stat.get("ppid", ""),
                "role": _role(cmd),
                "comm": stat.get("comm", ""),
                "state": state,
                "cpu_percent": f"{cpu_percent:.3f}" if isinstance(cpu_percent, float) else "",
                "cpu_num": stat.get("cpu_num", ""),
                "rss_kib": status.get("VmRSS", ""),
                "threads": stat.get("threads", ""),
                "voluntary_ctxt_switches": status.get("voluntary_ctxt_switches", ""),
                "nonvoluntary_ctxt_switches": status.get("nonvoluntary_ctxt_switches", ""),
                "read_bytes": io.get("read_bytes", ""),
                "write_bytes": io.get("write_bytes", ""),
                "loadavg_1m": load1,
                "loadavg_5m": load5,
                "loadavg_15m": load15,
                "runnable_processes": state_counts.get("R", 0),
                "sleeping_processes": state_counts.get("S", 0) + state_counts.get("I", 0),
                "per_core_cpu_percent": "",
                "cpu_total_percent": "",
            }
        )
    system_row = {
        "timestamp_utc": timestamp,
        "label": label,
        "sample": sample,
        "kind": "system",
        "pid": "",
        "ppid": "",
        "role": "system",
        "comm": "",
        "state": "",
        "cpu_percent": "",
        "cpu_num": "",
        "rss_kib": "",
        "threads": "",
        "voluntary_ctxt_switches": "",
        "nonvoluntary_ctxt_switches": "",
        "read_bytes": "",
        "write_bytes": "",
        "loadavg_1m": load1,
        "loadavg_5m": load5,
        "loadavg_15m": load15,
        "runnable_processes": state_counts.get("R", 0),
        "sleeping_processes": state_counts.get("S", 0) + state_counts.get("I", 0),
        "per_core_cpu_percent": per_core,
        "cpu_total_percent": f"{cpu_total_percent:.3f}" if isinstance(cpu_total_percent, float) else "",
    }
    return [system_row, *selected], next_proc_ticks, total_ticks, cores


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(cmd, text=True, capture_output=True, timeout=8, check=False)
    except Exception:
        return None


def _gpu_row(label: str, sample: int) -> dict[str, Any]:
    row = {field: "" for field in GPU_FIELDS}
    row.update({"timestamp_utc": _utc_now(), "label": label, "sample": sample})
    cp = _run(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,clocks.mem",
            "--format=csv,noheader,nounits",
        ]
    )
    if cp is None or cp.returncode != 0:
        row["error"] = (cp.stderr.strip() if cp else "nvidia-smi unavailable")
        return row
    first = cp.stdout.strip().splitlines()[0] if cp.stdout.strip() else ""
    values = [part.strip() for part in first.split(",")]
    keys = [
        "gpu_timestamp",
        "utilization_gpu_percent",
        "utilization_memory_percent",
        "memory_used_mib",
        "memory_total_mib",
        "power_draw_w",
        "temperature_gpu_c",
        "clocks_sm_mhz",
        "clocks_mem_mhz",
    ]
    for key, value in zip(keys, values):
        row[key] = value

    apps = _run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"])
    if apps is not None and apps.returncode == 0:
        pids: list[str] = []
        mems: list[str] = []
        for line in apps.stdout.strip().splitlines():
            parts = [part.strip() for part in line.split(",")]
            if not parts or not parts[0]:
                continue
            pids.append(parts[0])
            if len(parts) >= 3:
                mems.append(parts[2])
        row["compute_app_count"] = len(pids)
        row["compute_app_pids"] = ";".join(pids)
        row["compute_app_used_memory_mib"] = ";".join(mems)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Lightweight external monitor for EIEL multi-worker contention runs")
    parser.add_argument("--label", required=True)
    parser.add_argument("--match-token", action="append", default=[])
    parser.add_argument("--interval-s", type=float, default=3.0)
    parser.add_argument("--process-out", required=True)
    parser.add_argument("--gpu-out", required=True)
    args = parser.parse_args()

    stop = False

    def _stop(_signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    tokens = [str(token) for token in args.match_token if str(token)]
    tokens.append(str(Path(args.process_out).resolve().parent))
    process_out = Path(args.process_out)
    gpu_out = Path(args.gpu_out)
    process_out.parent.mkdir(parents=True, exist_ok=True)
    gpu_out.parent.mkdir(parents=True, exist_ok=True)

    prev_proc_ticks: dict[int, int] = {}
    prev_cpu_total: int | None = None
    prev_cores: list[tuple[int, int]] | None = None
    last_time = time.monotonic()
    with process_out.open("w", newline="", encoding="utf-8") as pf, gpu_out.open("w", newline="", encoding="utf-8") as gf:
        pw = csv.DictWriter(pf, fieldnames=PROC_FIELDS)
        gw = csv.DictWriter(gf, fieldnames=GPU_FIELDS)
        pw.writeheader()
        gw.writeheader()
        sample = 0
        while not stop:
            now = time.monotonic()
            interval = max(0.001, now - last_time)
            rows, prev_proc_ticks, prev_cpu_total, prev_cores = _process_rows(
                label=str(args.label),
                sample=sample,
                tokens=tokens,
                prev_proc_ticks=prev_proc_ticks,
                interval_s=interval,
                prev_cpu_total=prev_cpu_total,
                prev_cores=prev_cores,
            )
            for row in rows:
                pw.writerow(row)
            gw.writerow(_gpu_row(str(args.label), sample))
            pf.flush()
            gf.flush()
            sample += 1
            last_time = now
            deadline = time.monotonic() + max(1.0, float(args.interval_s))
            while not stop and time.monotonic() < deadline:
                time.sleep(0.2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
