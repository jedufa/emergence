"""Read-only HTTP service for Emergence GPU/CPU simulation slot telemetry."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from typing import Any, Mapping, Sequence
from urllib.parse import quote, unquote, urlsplit

try:
    from engine.monitoring.slot_telemetry import (
        CPU_CAPACITY,
        GPU_CAPACITY,
        LEGACY_SCHEMA_VERSION,
        SUPPORTED_SCHEMA_VERSIONS,
        telemetry_root,
    )
except ModuleNotFoundError:
    from experiments.projects.emergence.engine.monitoring.slot_telemetry import (
        CPU_CAPACITY,
        GPU_CAPACITY,
        LEGACY_SCHEMA_VERSION,
        SUPPORTED_SCHEMA_VERSIONS,
        telemetry_root,
    )


STATIC_ROOT = Path(__file__).resolve().parent / "static"
CANOPY_ROOT = Path(__file__).resolve().parents[1] / "visualization" / "living_lineage_canopy"
ACTIVE_STATUSES = frozenset({"starting", "running", "finishing", "stale", "untracked"})
LIVE_OWNER_STATUSES = frozenset({"starting", "running", "finishing"})
TERMINAL_STATUSES = frozenset({"complete", "failed", "interrupted", "cancelled"})
GENERATION_PROGRESS_SCHEMA = "eiel_cuda_architecture_b_generation_progress_v1"
_PROGRESS_SAMPLES: dict[str, tuple[int, float, float | None]] = {}
_PROGRESS_SAMPLES_LOCK = threading.Lock()
_CPU_SAMPLE: tuple[int, int] | None = None
_CPU_SAMPLE_LOCK = threading.Lock()
_RELAY_FILE_KEYS = frozenset(
    {
        "playback_file_name",
        "provenance_file_name",
        "status_file_name",
        "cross_check_file_name",
    }
)


class ObservatoryReadError(Exception):
    def __init__(self, status: HTTPStatus, message: str):
        self.status = status
        super().__init__(message)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return dict(value) if isinstance(value, Mapping) else None


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _mime_type(path: Path) -> str:
    return {
        ".html": "text/html; charset=utf-8",
        ".js": "text/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".json": "application/json; charset=utf-8",
        ".ico": "image/x-icon",
        ".png": "image/png",
        ".svg": "image/svg+xml",
    }.get(path.suffix.lower(), "application/octet-stream")


def living_lineage_canopy_asset(path: str) -> tuple[bytes, str]:
    asset = "index.html" if path in {"", "/", "/living-lineage-canopy", "/living-lineage-canopy/"} else path.removeprefix("/living-lineage-canopy/")
    candidate = (CANOPY_ROOT / unquote(asset)).resolve()
    if not _inside(candidate, CANOPY_ROOT) or not candidate.is_file():
        raise ObservatoryReadError(HTTPStatus.NOT_FOUND, "canopy asset not found")
    return candidate.read_bytes(), _mime_type(candidate)


def _relay_root_for_run(run_root: Any) -> Path | None:
    text = str(run_root or "").strip()
    if not text:
        return None
    try:
        root = Path(text).expanduser().resolve()
    except OSError:
        return None
    candidates = [root / "lineage-playback-relay"]
    if root.name == "lineage-playback-relay":
        candidates.insert(0, root)
    for candidate in candidates:
        manifest = candidate / "relay_manifest.json"
        if manifest.is_file():
            return candidate
    return None


def _tick_native_observability_for_run(run_root: Any) -> dict[str, Any] | None:
    """Return the narrow, non-authoritative tick observability contract.

    This is deliberately a display-only read.  In particular, it must not be
    treated as a Canopy relay: the reference route can publish committed tick
    cursors and checkpoint cursors without having a lineage projection that
    the Canopy exporter can truthfully render.
    """

    text = str(run_root or "").strip()
    if not text:
        return None
    try:
        candidate = Path(text).expanduser().resolve() / "tick_native_observability.json"
        payload = _read_json(candidate)
    except OSError:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("schema_version") != "eiel_tick_native_observability_contract_v1":
        return None
    if payload.get("route_identity") != "tick_native_execution_route_v1":
        return None
    live = payload.get("live_simulations")
    canopy = payload.get("canopy")
    if not isinstance(live, dict) or not isinstance(canopy, dict):
        return None
    return {
        "runtime_partition_ticks": live.get("runtime_partition_ticks"),
        "telemetry_precision": live.get("telemetry_precision"),
        "canopy_relay_compatible": canopy.get("relay_compatible"),
        "canopy_checkpoint_every_ticks": canopy.get("checkpoint_every_ticks"),
        "canopy_reason": canopy.get("reason"),
    }


def _observatory_prefix(prefix: str) -> str:
    cleaned = "/" + str(prefix or "/api/v1/observatory").strip("/")
    return cleaned.rstrip("/")


def _relay_route(prefix: str, device: str, position: int, suffix: str) -> str:
    return f"{_observatory_prefix(prefix)}/relay/{device}/{position}/{suffix.lstrip('/')}"


def _observatory_for_slot(slot: Mapping[str, Any], *, route_prefix: str) -> dict[str, Any]:
    relay_root = _relay_root_for_run(slot.get("run_root"))
    tick_native = _tick_native_observability_for_run(slot.get("run_root"))
    base = {
        "schema_version": "eiel_observatory_slot_projection_v1",
        "canopy_available": False,
        "canopy_url": None,
        "canopy_reason": None,
        "canopy_checkpoint_every_ticks": None,
        "tick_native": tick_native,
        "semantics": None,
        "relay_state": None,
        "relay_sequence": None,
        "relay_boundary": None,
        "relay_warning_count": 0,
    }
    if relay_root is None:
        if tick_native and tick_native.get("canopy_relay_compatible") is False:
            return {
                **base,
                "canopy_reason": tick_native.get("canopy_reason"),
                "canopy_checkpoint_every_ticks": tick_native.get("canopy_checkpoint_every_ticks"),
            }
        return base
    manifest = _read_json(relay_root / "relay_manifest.json") or {}
    status = _read_json(relay_root / str(manifest.get("status_file_name") or "")) or {}
    device = str(slot.get("device") or "")
    position = int(slot.get("position") or 0)
    manifest_url = _relay_route(route_prefix, device, position, "manifest")
    return {
        **base,
        "canopy_available": True,
        "canopy_url": f"/living-lineage-canopy/index.html?relay={quote(manifest_url, safe='')}",
        "semantics": manifest.get("semantics_mode"),
        "relay_state": status.get("state") or manifest.get("source_state"),
        "relay_sequence": manifest.get("sequence_number"),
        "relay_boundary": status.get("playback_boundary") or manifest.get("latest_exported_playback_boundary"),
        "relay_warning_count": len(status.get("incomplete_artifact_warnings") or manifest.get("warnings") or []),
    }


def _attach_observatory(
    slots: Sequence[dict[str, Any]], *, route_prefix: str
) -> None:
    for slot in slots:
        slot["observatory"] = _observatory_for_slot(slot, route_prefix=route_prefix)


def _pid_start_ticks(pid: int) -> int | None:
    try:
        tail = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1]
        return int(tail.split()[19])
    except (OSError, UnicodeError, ValueError, IndexError):
        return None


def _producer_alive(payload: Mapping[str, Any]) -> bool:
    pid = payload.get("pid")
    expected = payload.get("pid_start_ticks")
    if type(pid) is not int or pid < 1:
        return False
    observed = _pid_start_ticks(pid)
    if observed is None:
        return False
    return expected is None or observed == expected


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number == number and abs(number) != float("inf") else None


def _bounded_tick_rate(value: Any) -> float | None:
    rate = _safe_float(value)
    return rate if rate is not None and 0.0 < rate <= 1_000_000_000.0 else None


def _legacy_boundary_tick_projection(
    allocation: Mapping[str, Any], cell: Mapping[str, Any]
) -> dict[str, Any]:
    """Expose a v1 heartbeat's committed boundary without inventing live ticks."""

    if allocation.get("schema_version") != LEGACY_SCHEMA_VERSION:
        return {}
    current = cell.get("current_generation")
    target = cell.get("target_generation")
    ticks_per_generation = allocation.get("ticks_per_generation")
    if (
        type(current) is not int
        or type(target) is not int
        or type(ticks_per_generation) is not int
        or current < 0
        or target < current
        or ticks_per_generation <= 0
    ):
        return {}
    rate = _safe_float(cell.get("rate_generations_per_second"))
    return {
        "next_absolute_tick": current * ticks_per_generation,
        "target_absolute_tick": target * ticks_per_generation,
        "rate_ticks_per_second": (
            rate * ticks_per_generation if rate is not None and rate > 0 else None
        ),
        "tick_precision": "boundary-only",
    }


def _cpu_utilization_between(
    previous: tuple[int, int], current: tuple[int, int]
) -> float | None:
    previous_idle, previous_total = previous
    current_idle, current_total = current
    idle_delta = current_idle - previous_idle
    total_delta = current_total - previous_total
    if total_delta <= 0 or idle_delta < 0:
        return None
    return max(0.0, min(100.0, 100.0 * (1.0 - idle_delta / total_delta)))


def _cpu_snapshot() -> dict[str, Any]:
    base: dict[str, Any] = {
        "available": False,
        "utilization_percent": None,
        "logical_cpu_count": os.cpu_count(),
        "load_average_1m": None,
        "load_average_5m": None,
        "load_average_15m": None,
        "error": None,
    }
    try:
        fields = Path("/proc/stat").read_text(encoding="ascii").splitlines()[0].split()
        if not fields or fields[0] != "cpu":
            raise ValueError("aggregate CPU counters are unavailable")
        counters = [int(value) for value in fields[1:]]
        if len(counters) < 4:
            raise ValueError("aggregate CPU counters are incomplete")
        idle = counters[3] + (counters[4] if len(counters) > 4 else 0)
        sample = (idle, sum(counters))
        global _CPU_SAMPLE
        with _CPU_SAMPLE_LOCK:
            previous = _CPU_SAMPLE
            _CPU_SAMPLE = sample
        utilization = (
            _cpu_utilization_between(previous, sample) if previous is not None else None
        )
        load_1m, load_5m, load_15m = os.getloadavg()
        base.update(
            {
                "available": True,
                "utilization_percent": utilization,
                "load_average_1m": load_1m,
                "load_average_5m": load_5m,
                "load_average_15m": load_15m,
            }
        )
    except (OSError, UnicodeError, ValueError, IndexError) as exc:
        base["error"] = f"{type(exc).__name__}: {exc}"
    return base


def _idle_slot(device: str, position: int) -> dict[str, Any]:
    return {
        "device": device,
        "position": position,
        "label": f"{'GPU' if device == 'gpu' else 'CPU'} {position:02d}",
        "status": "idle",
        "cell_id": None,
        "run_id": None,
        "campaign_id": None,
        "batch_id": None,
        "stage": "Available",
        "next_absolute_tick": None,
        "target_absolute_tick": None,
        "tick_precision": "unavailable",
        "progress_fraction": 0.0,
        "rate_ticks_per_second": None,
        "eta_seconds": None,
        "elapsed_seconds": None,
        "updated_at": None,
        "pid": None,
        "run_root": None,
        "conflicts": [],
    }


def _allocation_rows(
    root: Path,
    *,
    now: float,
    stale_after_seconds: float,
    terminal_retention_seconds: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    allocations: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    try:
        paths = sorted(root.glob("*.json"))
    except OSError:
        paths = []
    for path in paths:
        value = _read_json(path)
        if value is None or value.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
            invalid.append({"path": str(path), "reason": "invalid heartbeat"})
            continue
        device = value.get("device")
        cells = value.get("cells")
        if device not in {"gpu", "cpu"} or not isinstance(cells, list):
            invalid.append({"path": str(path), "reason": "invalid device/cells"})
            continue
        try:
            age = max(0.0, now - path.stat().st_mtime)
        except OSError:
            age = float("inf")
        status = str(value.get("status") or "unknown").lower()
        alive = _producer_alive(value)
        if status in ACTIVE_STATUSES and not alive:
            continue
        if status in ACTIVE_STATUSES and age > stale_after_seconds:
            status = "stale"
        if status == "stale" and age > terminal_retention_seconds:
            continue
        if status in TERMINAL_STATUSES and age > terminal_retention_seconds:
            continue
        value["status"] = status
        value["heartbeat_age_seconds"] = age
        value["producer_alive"] = alive
        value["heartbeat_path"] = str(path)
        allocations.append(value)
    # Live owners win visual positions. Stale and terminal receipts remain
    # briefly visible only when no live owner currently holds the position.
    allocations.sort(
        key=lambda row: (
            (
                0
                if str(row.get("status")) in LIVE_OWNER_STATUSES
                else 1
                if str(row.get("status")) == "stale"
                else 2
            ),
            float(row.get("heartbeat_age_seconds") or 0.0),
        )
    )
    return allocations, invalid


def _merge_slots(
    allocations: Sequence[Mapping[str, Any]], device: str, capacity: int
) -> list[dict[str, Any]]:
    slots = [_idle_slot(device, index) for index in range(1, capacity + 1)]
    for allocation in allocations:
        if allocation.get("device") != device:
            continue
        for cell in allocation.get("cells", []):
            if not isinstance(cell, Mapping) or type(cell.get("position")) is not int:
                continue
            position = int(cell["position"])
            if position < 1 or position > capacity:
                continue
            tick_projection = _legacy_boundary_tick_projection(allocation, cell)
            candidate = {
                "device": device,
                "position": position,
                "label": f"{'GPU' if device == 'gpu' else 'CPU'} {position:02d}",
                "status": allocation.get("status"),
                "cell_id": cell.get("cell_id"),
                "run_id": allocation.get("run_id"),
                "campaign_id": allocation.get("campaign_id"),
                "batch_id": allocation.get("batch_id"),
                "profile_id": allocation.get("profile_id"),
                "stage": allocation.get("stage"),
                "next_absolute_tick": tick_projection.get(
                    "next_absolute_tick", cell.get("next_absolute_tick")
                ),
                "target_absolute_tick": tick_projection.get(
                    "target_absolute_tick", cell.get("target_absolute_tick")
                ),
                "tick_precision": tick_projection.get("tick_precision", "exact"),
                "progress_fraction": _safe_float(cell.get("progress_fraction")) or 0.0,
                "rate_ticks_per_second": _bounded_tick_rate(
                    tick_projection.get("rate_ticks_per_second", cell.get("rate_ticks_per_second"))
                ),
                "eta_seconds": _safe_float(cell.get("eta_seconds")),
                "elapsed_seconds": _safe_float(allocation.get("elapsed_seconds")),
                "updated_at": allocation.get("updated_at"),
                "heartbeat_age_seconds": allocation.get("heartbeat_age_seconds"),
                "producer_alive": allocation.get("producer_alive"),
                "pid": allocation.get("pid"),
                "run_root": allocation.get("run_root"),
                "error": allocation.get("error"),
                "allocation_id": allocation.get("allocation_id"),
                "conflicts": [],
            }
            existing = slots[position - 1]
            if existing["status"] == "idle":
                slots[position - 1] = candidate
            elif existing.get("status") == "conflict":
                if (
                    candidate.get("status") in LIVE_OWNER_STATUSES
                    and existing.get("allocation_id") != candidate.get("allocation_id")
                ):
                    existing["conflicts"].append(
                        {
                            "allocation_id": candidate.get("allocation_id"),
                            "run_id": candidate.get("run_id"),
                            "pid": candidate.get("pid"),
                            "cell_id": candidate.get("cell_id"),
                        }
                    )
            elif (
                existing.get("status") in LIVE_OWNER_STATUSES
                and candidate.get("status") not in LIVE_OWNER_STATUSES
            ):
                # Retained history must not conflict with a current owner.
                continue
            elif (
                existing.get("status") not in LIVE_OWNER_STATUSES
                and candidate.get("status") in LIVE_OWNER_STATUSES
            ):
                slots[position - 1] = candidate
            elif (
                existing.get("status") not in LIVE_OWNER_STATUSES
                and candidate.get("status") not in LIVE_OWNER_STATUSES
            ):
                # Multiple stale/terminal receipts are history, not competing
                # live allocations. Sorting keeps the newest relevant receipt.
                continue
            elif existing.get("allocation_id") != candidate.get("allocation_id"):
                existing["status"] = "conflict"
                existing["conflicts"].append(
                    {
                        "allocation_id": candidate.get("allocation_id"),
                        "run_id": candidate.get("run_id"),
                        "pid": candidate.get("pid"),
                        "cell_id": candidate.get("cell_id"),
                    }
                )
    return slots


def _nvidia_snapshot() -> dict[str, Any]:
    base = {
        "available": False,
        "name": None,
        "utilization_percent": None,
        "memory_used_mib": None,
        "memory_total_mib": None,
        "temperature_c": None,
        "compute_processes": [],
        "error": None,
    }
    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        parts = [item.strip() for item in query.stdout.strip().split(",")]
        if len(parts) >= 5:
            base.update(
                {
                    "available": True,
                    "name": parts[0],
                    "utilization_percent": float(parts[1]),
                    "memory_used_mib": float(parts[2]),
                    "memory_total_mib": float(parts[3]),
                    "temperature_c": float(parts[4]),
                }
            )
        processes = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        rows = []
        for line in processes.stdout.splitlines():
            columns = [item.strip() for item in line.split(",")]
            if len(columns) >= 3 and columns[0].isdigit():
                rows.append(
                    {
                        "pid": int(columns[0]),
                        "process": Path(columns[1]).name,
                        "memory_mib": _safe_float(float(columns[2])),
                    }
                )
        base["compute_processes"] = rows
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        base["error"] = f"{type(exc).__name__}: {exc}"
    return base


def _looks_like_emergence_process(pid: int, process_name: str) -> bool:
    name = Path(process_name).name.lower()
    if "python" not in name and "eiel" not in name:
        return False
    try:
        cwd = str(Path(f"/proc/{pid}/cwd").resolve())
    except OSError:
        return False
    return "/experiments/projects/emergence" in cwd


def _argument_value(argv: Sequence[str], name: str) -> str | None:
    try:
        index = argv.index(name)
    except ValueError:
        return None
    return argv[index + 1] if index + 1 < len(argv) else None


def _progress_rate(path: Path, generation: int, updated_at: float) -> float | None:
    key = str(path)
    with _PROGRESS_SAMPLES_LOCK:
        previous = _PROGRESS_SAMPLES.get(key)
        if previous is None:
            rate = None
        elif generation > previous[0] and updated_at > previous[1]:
            observed = (generation - previous[0]) / (updated_at - previous[1])
            rate = observed if previous[2] is None else 0.35 * observed + 0.65 * previous[2]
        else:
            rate = previous[2]
        _PROGRESS_SAMPLES[key] = (generation, updated_at, rate)
    return rate


def _generation_progress_metadata(
    run_root: Path,
    *,
    target_generation: int | None,
    ticks_per_generation: int | None,
) -> dict[str, Any]:
    try:
        candidates = list(run_root.glob("attempt-*/*/generation_progress.json"))
        path = max(candidates, key=lambda item: item.stat().st_mtime)
        updated_at = path.stat().st_mtime
    except (OSError, ValueError):
        return {}
    payload = _read_json(path)
    if payload is None or payload.get("schema_version") != GENERATION_PROGRESS_SCHEMA:
        return {}
    generation = payload.get("completed_generation_boundary")
    start_generation = payload.get("start_generation")
    if type(generation) is not int or generation < 0:
        return {}
    target = target_generation
    if type(target) is not int or target < generation:
        candidate_target = payload.get("target_next_generation")
        target = candidate_target if type(candidate_target) is int else generation
    rate = _progress_rate(path, generation, updated_at)
    if rate is None and start_generation == 0 and generation > 0:
        baseline = path.parent / "owner" / "metrics.csv"
        try:
            elapsed = updated_at - baseline.stat().st_mtime
        except OSError:
            elapsed = 0.0
        if elapsed > 0:
            rate = generation / elapsed
    if type(ticks_per_generation) is not int or ticks_per_generation <= 0:
        return {}
    next_absolute_tick = generation * ticks_per_generation
    target_absolute_tick = target * ticks_per_generation
    tick_rate = rate * ticks_per_generation if rate is not None else None
    eta = (
        (target_absolute_tick - next_absolute_tick) / tick_rate
        if tick_rate and target_absolute_tick > next_absolute_tick
        else 0.0 if target_absolute_tick == next_absolute_tick else None
    )
    phase = str(payload.get("phase") or "running")
    failure = payload.get("failure")
    status = "failed" if phase == "failure" or failure else "finishing" if phase == "final" else "running"
    route = str(payload.get("route_id") or "native route")
    control = str(payload.get("run_control_status") or phase)
    return {
        "tick_progress_available": True,
        "next_absolute_tick": next_absolute_tick,
        "target_absolute_tick": target_absolute_tick,
        "progress_fraction": next_absolute_tick / target_absolute_tick if target_absolute_tick > 0 else 0.0,
        "rate_ticks_per_second": tick_rate,
        "eta_seconds": eta,
        "stage": f"{route} · Tick {next_absolute_tick} · {control}",
        "status": status,
        "updated_at": datetime.fromtimestamp(updated_at, timezone.utc).isoformat().replace("+00:00", "Z"),
        "error": json.dumps(failure, sort_keys=True) if failure else None,
        "progress_path": str(path),
    }


def _untracked_process_metadata(pid: int) -> dict[str, Any]:
    try:
        argv = [
            item.decode("utf-8", "replace")
            for item in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            if item
        ]
    except OSError:
        argv = []
    run_id = _argument_value(argv, "--run-id")
    run_root = _argument_value(argv, "--run-root")
    searchable = run_id or " ".join(argv)
    shape = re.search(r"(?:^|[-_])b([1-9][0-9]*)g([1-9][0-9]*)", searchable)
    slots = min(GPU_CAPACITY, int(shape.group(1))) if shape else 1
    generations = int(shape.group(2)) if shape else None
    metadata: dict[str, Any] = {
        "run_id": run_id,
        "run_root": run_root,
        "inferred_slot_count": slots,
    }
    if run_root:
        root = Path(run_root)
        launch = _read_json(root / "launch.json") or {}
        identity = launch.get("identity")
        if isinstance(identity, Mapping):
            cell_ids = identity.get("cell_ids")
            if isinstance(cell_ids, list) and all(type(item) is str for item in cell_ids):
                metadata["cell_ids"] = cell_ids
            ticks = identity.get("ticks_per_generation")
            if type(ticks) is int:
                metadata["ticks_per_generation"] = ticks
        metadata.update(_generation_progress_metadata(
            root,
            target_generation=generations,
            ticks_per_generation=metadata.get("ticks_per_generation"),
        ))
    return metadata


def _overlay_untracked_gpu_slots(
    slots: list[dict[str, Any]], processes: Sequence[Mapping[str, Any]]
) -> None:
    available = [index for index, slot in enumerate(slots) if slot["status"] == "idle"]
    for process in processes:
        count = max(1, int(process.get("inferred_slot_count") or 1))
        for ordinal in range(1, count + 1):
            if not available:
                return
            index = available.pop(0)
            position = index + 1
            run_id = str(process.get("run_id") or process.get("process") or "GPU process")
            cell_ids = process.get("cell_ids")
            cell_id = (
                cell_ids[ordinal - 1]
                if isinstance(cell_ids, list) and ordinal <= len(cell_ids)
                else f"{run_id} [{ordinal}/{count}]" if count > 1 else run_id
            )
            progress_available = process.get("tick_progress_available") is True
            slots[index] = {
                **_idle_slot("gpu", position),
                "status": process.get("status") if progress_available else "untracked",
                "cell_id": cell_id,
                "run_id": process.get("run_id"),
                "stage": process.get("stage") or "Running without detailed telemetry",
                "next_absolute_tick": process.get("next_absolute_tick"),
                "target_absolute_tick": process.get("target_absolute_tick"),
                "tick_precision": "boundary-only" if progress_available else "unavailable",
                "progress_fraction": process.get("progress_fraction") or 0.0,
                "rate_ticks_per_second": process.get("rate_ticks_per_second"),
                "eta_seconds": process.get("eta_seconds"),
                "updated_at": process.get("updated_at"),
                "pid": process.get("pid"),
                "run_root": process.get("run_root"),
                "error": process.get("error") if progress_available else "Progress, speed, and ETA unavailable for this runner",
                "telemetry_source": "tick_progress" if progress_available else "process inference",
            }


def _slot_for(root: Path, device: str, position: int) -> dict[str, Any]:
    selected_device = str(device or "").strip().lower()
    if selected_device not in {"gpu", "cpu"}:
        raise ObservatoryReadError(HTTPStatus.BAD_REQUEST, "device must be gpu or cpu")
    selected_position = int(position)
    snapshot = build_snapshot(root, include_system=False)
    slots = snapshot[f"{selected_device}_slots"]
    if selected_position < 1 or selected_position > len(slots):
        raise ObservatoryReadError(HTTPStatus.NOT_FOUND, "slot position not found")
    return slots[selected_position - 1]


def _relay_root_for_slot(root: Path, device: str, position: int) -> Path:
    slot = _slot_for(root, device, position)
    relay_root = _relay_root_for_run(slot.get("run_root"))
    if relay_root is None:
        raise ObservatoryReadError(HTTPStatus.NOT_FOUND, "relay manifest not available for slot")
    return relay_root


def _safe_relay_file(relay_root: Path, file_name: str) -> Path:
    if "/" in file_name or "\\" in file_name or file_name in {"", ".", ".."}:
        raise ObservatoryReadError(HTTPStatus.BAD_REQUEST, "invalid relay file name")
    candidate = (relay_root / file_name).resolve()
    if not _inside(candidate, relay_root) or not candidate.is_file():
        raise ObservatoryReadError(HTTPStatus.NOT_FOUND, "relay file not found")
    return candidate


def observatory_relay_manifest(
    root: Path,
    *,
    device: str,
    position: int,
    route_prefix: str = "/api/v1/observatory",
) -> dict[str, Any]:
    relay_root = _relay_root_for_slot(root, device, position)
    manifest = _read_json(relay_root / "relay_manifest.json")
    if manifest is None:
        raise ObservatoryReadError(HTTPStatus.NOT_FOUND, "relay manifest not readable")
    rewritten = dict(manifest)
    for key in _RELAY_FILE_KEYS:
        value = rewritten.get(key)
        if isinstance(value, str) and value:
            file_path = _safe_relay_file(relay_root, value)
            rewritten[key] = _relay_route(
                route_prefix,
                str(device).strip().lower(),
                int(position),
                f"file/{quote(file_path.name, safe='')}",
            )
    rewritten["observatory_proxy"] = {
        "schema_version": "eiel_observatory_relay_proxy_v1",
        "slot": {"device": str(device).strip().lower(), "position": int(position)},
        "hash_policy": "artifact bytes are served unchanged; manifest URLs are browser proxy paths",
    }
    return rewritten


def observatory_relay_file(
    root: Path,
    *,
    device: str,
    position: int,
    file_name: str,
) -> tuple[bytes, str]:
    relay_root = _relay_root_for_slot(root, device, position)
    manifest = _read_json(relay_root / "relay_manifest.json") or {}
    allowed = {
        str(manifest[key])
        for key in _RELAY_FILE_KEYS
        if isinstance(manifest.get(key), str) and manifest.get(key)
    }
    decoded = unquote(str(file_name or ""))
    if decoded not in allowed:
        raise ObservatoryReadError(HTTPStatus.NOT_FOUND, "relay file is not listed by manifest")
    path = _safe_relay_file(relay_root, decoded)
    return path.read_bytes(), _mime_type(path)


def observatory_route_response(
    root: Path,
    *,
    path: str,
    route_prefix: str = "/api/v1/observatory",
) -> tuple[HTTPStatus, dict[str, Any] | bytes, str]:
    prefix = _observatory_prefix(route_prefix)
    rel = str(path or "")
    if not rel.startswith(f"{prefix}/relay/"):
        raise ObservatoryReadError(HTTPStatus.NOT_FOUND, "observatory route not found")
    parts = rel[len(f"{prefix}/relay/") :].strip("/").split("/")
    if len(parts) < 3:
        raise ObservatoryReadError(HTTPStatus.NOT_FOUND, "observatory route not found")
    device, position_text, kind = parts[:3]
    try:
        position = int(position_text)
    except ValueError as exc:
        raise ObservatoryReadError(HTTPStatus.BAD_REQUEST, "invalid slot position") from exc
    if kind == "manifest" and len(parts) == 3:
        return (
            HTTPStatus.OK,
            observatory_relay_manifest(
                root,
                device=device,
                position=position,
                route_prefix=route_prefix,
            ),
            "application/json; charset=utf-8",
        )
    if kind == "file" and len(parts) == 4:
        body, content_type = observatory_relay_file(
            root,
            device=device,
            position=position,
            file_name=parts[3],
        )
        return HTTPStatus.OK, body, content_type
    raise ObservatoryReadError(HTTPStatus.NOT_FOUND, "observatory route not found")


def _untracked_emergence_processes(tracked_pids: set[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        pid = int(proc.name)
        if pid == os.getpid() or pid in tracked_pids:
            continue
        try:
            argv = (proc / "cmdline").read_bytes().split(b"\0")
            decoded = [item.decode("utf-8", "replace") for item in argv if item]
        except OSError:
            continue
        script = next(
            (
                Path(item).name
                for item in decoded[1:]
                if Path(item).name.startswith("eiel_") and Path(item).suffix == ".py"
            ),
            "",
        )
        if script == "eiel_slot_monitor.py":
            continue
        if not script:
            continue
        rows.append({"pid": pid, "process": script})
    return sorted(rows, key=lambda row: int(row["pid"]))


def build_snapshot(
    root: Path,
    *,
    stale_after_seconds: float = 15.0,
    terminal_retention_seconds: float = 300.0,
    include_system: bool = True,
    observatory_route_prefix: str = "/api/v1/observatory",
    now: float | None = None,
) -> dict[str, Any]:
    current = time.time() if now is None else float(now)
    allocations, invalid = _allocation_rows(
        root,
        now=current,
        stale_after_seconds=stale_after_seconds,
        terminal_retention_seconds=terminal_retention_seconds,
    )
    gpu_slots = _merge_slots(allocations, "gpu", GPU_CAPACITY)
    cpu_slots = _merge_slots(allocations, "cpu", CPU_CAPACITY)
    cpu_system = _cpu_snapshot() if include_system else {"available": False}
    nvidia = (
        _nvidia_snapshot()
        if include_system
        else {"available": False, "compute_processes": []}
    )
    tracked_gpu_pids = {
        int(row["pid"])
        for row in allocations
        if row.get("device") == "gpu" and type(row.get("pid")) is int
    }
    untracked = [
        {**row, **_untracked_process_metadata(int(row["pid"]))}
        for row in nvidia.get("compute_processes", [])
        if row.get("pid") not in tracked_gpu_pids
        and type(row.get("pid")) is int
        and _looks_like_emergence_process(int(row["pid"]), str(row.get("process") or ""))
    ]
    _overlay_untracked_gpu_slots(gpu_slots, untracked)
    _attach_observatory(gpu_slots, route_prefix=observatory_route_prefix)
    _attach_observatory(cpu_slots, route_prefix=observatory_route_prefix)
    tracked_pids = {
        int(row["pid"]) for row in allocations if type(row.get("pid")) is int
    }
    untracked_emergence = _untracked_emergence_processes(tracked_pids)
    return {
        "schema_version": "eiel_simulation_slot_snapshot_v1",
        "utc": _utc_now(),
        "telemetry_root": str(root),
        "gpu_capacity": GPU_CAPACITY,
        "cpu_capacity": CPU_CAPACITY,
        "gpu_active": sum(
            row["status"] in ACTIVE_STATUSES or row["status"] == "conflict"
            for row in gpu_slots
        ),
        "cpu_active": sum(
            row["status"] in ACTIVE_STATUSES or row["status"] == "conflict"
            for row in cpu_slots
        ),
        "gpu_slots": gpu_slots,
        "cpu_slots": cpu_slots,
        "cpu_system": cpu_system,
        "nvidia": nvidia,
        "untracked_gpu_processes": untracked,
        "untracked_emergence_processes": untracked_emergence,
        "invalid_heartbeats": invalid,
    }


class SlotMonitorServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], root: Path):
        self.telemetry_root = root
        super().__init__(address, SlotMonitorHandler)


class SlotMonitorHandler(BaseHTTPRequestHandler):
    server_version = "EmergenceSlotMonitor/1"

    def log_message(self, format: str, *args: Any) -> None:
        return None

    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: HTTPStatus, payload: Mapping[str, Any]) -> None:
        self._send(
            status,
            (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"),
            "application/json; charset=utf-8",
        )

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        server = self.server
        if not isinstance(server, SlotMonitorServer):
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "invalid server"})
            return
        if path == "/api/v1/health":
            self._json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "schema_version": "eiel_simulation_slot_health_v1",
                    "utc": _utc_now(),
                    "telemetry_root": str(server.telemetry_root),
                },
            )
            return
        if path == "/api/v1/snapshot":
            self._json(HTTPStatus.OK, build_snapshot(server.telemetry_root))
            return
        if path.startswith("/api/v1/observatory/"):
            try:
                status, payload, content_type = observatory_route_response(
                    server.telemetry_root,
                    path=path,
                    route_prefix="/api/v1/observatory",
                )
            except ObservatoryReadError as exc:
                self._json(exc.status, {"error": str(exc)})
                return
            except OSError as exc:
                self._json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
                return
            if isinstance(payload, bytes):
                self._send(status, payload, content_type)
            else:
                self._json(status, payload)
            return
        if path.startswith("/living-lineage-canopy"):
            try:
                body, content_type = living_lineage_canopy_asset(path)
            except ObservatoryReadError as exc:
                self._json(exc.status, {"error": str(exc)})
                return
            except OSError as exc:
                self._json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
                return
            self._send(HTTPStatus.OK, body, content_type)
            return
        asset = "index.html" if path in {"", "/"} else path.removeprefix("/")
        if asset not in {"index.html", "app.js", "style.css"}:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        file_path = STATIC_ROOT / asset
        try:
            body = file_path.read_bytes()
        except OSError:
            self._json(HTTPStatus.NOT_FOUND, {"error": "asset not found"})
            return
        self._send(HTTPStatus.OK, body, _mime_type(file_path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument("--telemetry-root", type=Path, default=telemetry_root())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.telemetry_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    server = SlotMonitorServer((str(args.host), int(args.port)), root)
    print(
        json.dumps(
            {
                "ok": True,
                "url": f"http://{args.host}:{args.port}/",
                "telemetry_root": str(root),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ObservatoryReadError",
    "SlotMonitorServer",
    "build_snapshot",
    "living_lineage_canopy_asset",
    "main",
    "observatory_relay_file",
    "observatory_relay_manifest",
    "observatory_route_response",
]
