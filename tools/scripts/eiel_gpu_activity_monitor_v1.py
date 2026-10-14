"""Record bounded 100 ms NVIDIA GPU activity telemetry for EIEL runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import signal
import subprocess
import time
from pathlib import Path
from typing import Final


SCHEMA_VERSION: Final[str] = "eiel_gpu_activity_monitor_v1"
QUERY_FIELDS: Final[tuple[str, ...]] = (
    "timestamp",
    "index",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "power.draw",
    "clocks.sm",
)
CSV_FIELDS: Final[tuple[str, ...]] = (
    "observed_unix_ns",
    "observed_perf_ns",
    "device_timestamp",
    "device_index",
    "sm_utilization_percent",
    "memory_utilization_percent",
    "memory_used_mib",
    "memory_total_mib",
    "power_draw_w",
    "sm_clock_mhz",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_row(raw: str) -> list[str]:
    values = next(csv.reader([raw], skipinitialspace=True))
    if len(values) != len(QUERY_FIELDS):
        raise RuntimeError("nvidia-smi telemetry row has an unexpected field count")
    return [value.strip() for value in values]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--interval-ms", type=int, default=100)
    args = parser.parse_args()
    if args.device_index < 0:
        parser.error("--device-index must be nonnegative")
    if not 50 <= args.interval_ms <= 60_000:
        parser.error("--interval-ms must be between 50 and 60000")

    output = args.output_csv.expanduser().resolve()
    receipt = args.receipt.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or receipt.exists():
        raise RuntimeError("GPU telemetry outputs already exist")

    command = [
        "nvidia-smi",
        f"--id={args.device_index}",
        f"--query-gpu={','.join(QUERY_FIELDS)}",
        "--format=csv,noheader,nounits",
        f"--loop-ms={args.interval_ms}",
    ]
    stop = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    started_unix_ns = time.time_ns()
    started_perf_ns = time.perf_counter_ns()
    sample_count = 0
    process = subprocess.Popen(  # noqa: S603 - fixed executable and bounded args.
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    try:
        assert process.stdout is not None
        with output.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(CSV_FIELDS)
            handle.flush()
            while not stop:
                raw = process.stdout.readline()
                if raw == "":
                    break
                values = _parse_row(raw)
                writer.writerow(
                    [time.time_ns(), time.perf_counter_ns(), *values]
                )
                sample_count += 1
                handle.flush()
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    completed_unix_ns = time.time_ns()
    completed_perf_ns = time.perf_counter_ns()
    stderr = ""
    if process.stderr is not None:
        stderr = process.stderr.read().strip()
    status = "complete" if sample_count > 0 and process.returncode in {0, -15} else "failed"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "device_index": args.device_index,
        "interval_ms": args.interval_ms,
        "command": command,
        "started_unix_ns": started_unix_ns,
        "completed_unix_ns": completed_unix_ns,
        "started_perf_ns": started_perf_ns,
        "completed_perf_ns": completed_perf_ns,
        "wall_time_ns": completed_perf_ns - started_perf_ns,
        "sample_count": sample_count,
        "csv_path": str(output),
        "csv_sha256": _sha256(output),
        "process_returncode": process.returncode,
        "stderr": stderr,
    }
    receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))
    return 0 if status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
