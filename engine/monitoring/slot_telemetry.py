"""Fail-open telemetry for the standalone 72-GPU + 1-CPU slot monitor.

The files emitted here are operational metadata.  They are deliberately kept
outside run/checkpoint trees and must never participate in deterministic state,
receipts, or scientific result digests.
"""

from __future__ import annotations

from engine.runtime_roots import public_runtime_path

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Mapping, Sequence
from uuid import uuid4


SCHEMA_VERSION = "eiel_simulation_slot_heartbeat_v2"
LEGACY_SCHEMA_VERSION = "eiel_simulation_slot_heartbeat_v1"
SUPPORTED_SCHEMA_VERSIONS = frozenset({LEGACY_SCHEMA_VERSION, SCHEMA_VERSION})
DEFAULT_TELEMETRY_ROOT = Path(public_runtime_path('live/simulation-slots'))
GPU_CAPACITY = 72
CPU_CAPACITY = 1
TERMINAL_STATUSES = frozenset({"complete", "failed", "interrupted", "cancelled"})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def telemetry_root() -> Path:
    raw = os.environ.get("EIEL_SLOT_TELEMETRY_ROOT", "").strip()
    return Path(raw).expanduser().resolve() if raw else DEFAULT_TELEMETRY_ROOT


def _safe_text(value: Any, *, fallback: str = "") -> str:
    text = str(value or fallback).strip()
    return text[:512]


def _pid_start_ticks(pid: int) -> int | None:
    try:
        # Field 22 follows a parenthesized process name that may contain spaces.
        tail = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1]
        return int(tail.split()[19])
    except (OSError, UnicodeError, ValueError, IndexError):
        return None


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


class NullSlotLease:
    """No-op lease returned when operational telemetry cannot be initialized."""

    path: Path | None = None

    def update(self, **_: Any) -> None:
        return None

    def close(self, **_: Any) -> None:
        return None


@dataclass(frozen=True, slots=True)
class SlotCell:
    position: int
    cell_id: str


class SlotLease:
    """One process-owned allocation heartbeat with a low-frequency liveness writer."""

    def __init__(
        self,
        *,
        device: str,
        run_id: str,
        cell_ids: Sequence[str],
        current_generation: int | None = None,
        target_generation: int | None = None,
        ticks_per_generation: int | None = None,
        start_absolute_tick: int | None = None,
        target_absolute_tick: int | None = None,
        run_root: str | Path,
        campaign_id: str = "",
        batch_id: str = "",
        profile_id: str = "",
        logical_world_count: int | None = None,
        gpu_admission_mode: str | None = None,
        exclusive_gpu: bool | None = None,
        peak_vram_gib: float | None = None,
        workload_class: str = "",
        gpu_device_id: str = "",
        scheduler_resource_lease_id: str = "",
        scheduler_resource_ids: Sequence[str] = (),
        scheduler_fencing_token_digest: str = "",
        estimated_total_seconds: float | None = None,
        root: Path | None = None,
        heartbeat_interval_seconds: float = 2.0,
        start_writer: bool = True,
        clock: Any = time.monotonic,
    ) -> None:
        normalized_device = _safe_text(device).lower()
        if normalized_device in {"cuda", "gpu"}:
            normalized_device = "gpu"
        if normalized_device != "gpu" and normalized_device != "cpu":
            raise ValueError("slot device must be gpu/cuda or cpu")
        ids = tuple(_safe_text(value) for value in cell_ids)
        if not ids or any(not value for value in ids) or len(set(ids)) != len(ids):
            raise ValueError("slot cell ids must be non-empty and unique")
        capacity = GPU_CAPACITY if normalized_device == "gpu" else CPU_CAPACITY
        if normalized_device == "cpu" and len(ids) != 1:
            raise ValueError("the CPU monitor exposes exactly one simulation position")
        if len(ids) > capacity:
            raise ValueError(f"{normalized_device} allocation exceeds capacity {capacity}")
        generation_bounds = (current_generation, target_generation, ticks_per_generation)
        tick_bounds = (start_absolute_tick, target_absolute_tick)
        uses_generation_bounds = any(value is not None for value in generation_bounds)
        uses_tick_bounds = any(value is not None for value in tick_bounds)
        if uses_generation_bounds == uses_tick_bounds:
            raise ValueError("provide exactly one generation or absolute-tick bound set")

        self.device = normalized_device
        self.capacity = capacity
        self.run_id = _safe_text(run_id, fallback=f"pid-{os.getpid()}")
        self.cells = tuple(SlotCell(index, value) for index, value in enumerate(ids, 1))
        if uses_generation_bounds:
            if any(value is None for value in generation_bounds):
                raise ValueError("generation bounds must be supplied together")
            if (
                isinstance(current_generation, bool)
                or isinstance(target_generation, bool)
                or isinstance(ticks_per_generation, bool)
                or int(current_generation) < 0
                or int(target_generation) <= int(current_generation)
            ):
                raise ValueError("generation bounds are invalid")
            if int(ticks_per_generation) < 1:
                raise ValueError("ticks_per_generation must be positive")
            self.ticks_per_generation = int(ticks_per_generation)
            # The public heartbeat is tick-native. Generations remain accepted
            # only at this compatibility boundary for retained runners.
            self.start_absolute_tick = int(current_generation) * self.ticks_per_generation
            self.target_absolute_tick = int(target_generation) * self.ticks_per_generation
        else:
            if any(value is None for value in tick_bounds):
                raise ValueError("absolute-tick bounds must be supplied together")
            if (
                isinstance(start_absolute_tick, bool)
                or isinstance(target_absolute_tick, bool)
                or not isinstance(start_absolute_tick, int)
                or not isinstance(target_absolute_tick, int)
                or start_absolute_tick < 0
                or target_absolute_tick <= start_absolute_tick
            ):
                raise ValueError("absolute-tick bounds are invalid")
            self.ticks_per_generation = 1
            self.start_absolute_tick = start_absolute_tick
            self.target_absolute_tick = target_absolute_tick
        self.next_absolute_tick = self.start_absolute_tick
        self.run_root = str(Path(run_root).expanduser().resolve())
        self.campaign_id = _safe_text(campaign_id)
        self.batch_id = _safe_text(batch_id)
        self.profile_id = _safe_text(profile_id)
        self.logical_world_count = (
            int(logical_world_count)
            if type(logical_world_count) is int and logical_world_count > 0
            else len(ids)
        )
        mode = _safe_text(gpu_admission_mode).lower().replace("-", "_")
        if mode and mode not in {"shared_functional", "exclusive"}:
            raise ValueError("GPU admission mode is invalid")
        if exclusive_gpu is not None and type(exclusive_gpu) is not bool:
            raise ValueError("exclusive_gpu must be an exact boolean")
        if mode == "shared_functional" and exclusive_gpu is not False:
            raise ValueError("shared GPU telemetry must explicitly be non-exclusive")
        if mode == "exclusive" and exclusive_gpu is not True:
            raise ValueError("exclusive GPU telemetry must explicitly bind exclusivity")
        if peak_vram_gib is not None and (
            isinstance(peak_vram_gib, bool)
            or not isinstance(peak_vram_gib, (int, float))
            or not math.isfinite(float(peak_vram_gib))
            or float(peak_vram_gib) <= 0
        ):
            raise ValueError("peak_vram_gib must be a positive finite number")
        if mode == "shared_functional" and peak_vram_gib is None:
            raise ValueError("shared GPU telemetry requires peak_vram_gib")
        self.gpu_admission_mode = mode or None
        self.exclusive_gpu = exclusive_gpu
        self.peak_vram_gib = (
            None if peak_vram_gib is None else float(peak_vram_gib)
        )
        self.workload_class = _safe_text(workload_class)
        self.gpu_device_id = _safe_text(gpu_device_id)
        self.scheduler_resource_lease_id = _safe_text(
            scheduler_resource_lease_id
        )
        self.scheduler_resource_ids = tuple(
            _safe_text(value) for value in scheduler_resource_ids
        )
        if any(not value for value in self.scheduler_resource_ids):
            raise ValueError("Scheduler resource IDs must be non-empty strings")
        self.scheduler_fencing_token_digest = _safe_text(
            scheduler_fencing_token_digest
        )
        self.estimated_total_seconds = (
            float(estimated_total_seconds)
            if isinstance(estimated_total_seconds, (int, float))
            and not isinstance(estimated_total_seconds, bool)
            and math.isfinite(float(estimated_total_seconds))
            and float(estimated_total_seconds) > 0
            else None
        )
        self.pid = os.getpid()
        self.pid_start_ticks = _pid_start_ticks(self.pid)
        digest = hashlib.sha256(
            f"{self.device}\0{self.pid}\0{self.run_root}\0{self.run_id}".encode("utf-8")
        ).hexdigest()[:20]
        self.allocation_id = f"{self.device}-{self.pid}-{digest}"
        self.path = (root or telemetry_root()) / f"{self.allocation_id}.json"
        self.started_at = _utc_now()
        self._clock = clock
        self._started_monotonic = float(clock())
        self._last_sample_monotonic = self._started_monotonic
        self._last_sample_absolute_tick = self.next_absolute_tick
        self._ema_seconds_per_tick: float | None = None
        self._status = "starting"
        self._stage = "starting"
        self._error = ""
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._heartbeat_interval_seconds = max(0.25, float(heartbeat_interval_seconds))
        self._thread: threading.Thread | None = None
        self._write()
        if start_writer:
            self._thread = threading.Thread(
                target=self._writer_loop,
                name=f"slot-heartbeat-{self.device}",
                daemon=True,
            )
            self._thread.start()

    def _writer_loop(self) -> None:
        while not self._stop.wait(self._heartbeat_interval_seconds):
            try:
                self._write()
            except Exception:
                # Monitoring cannot become a simulation failure path.
                continue

    def _rate_eta(self, now: float) -> tuple[float | None, float | None]:
        rate = None
        if self._ema_seconds_per_tick and self._ema_seconds_per_tick > 0:
            rate = 1.0 / self._ema_seconds_per_tick
        elif self.estimated_total_seconds:
            total = self.target_absolute_tick - self.start_absolute_tick
            rate = total / self.estimated_total_seconds
        remaining = max(0, self.target_absolute_tick - self.next_absolute_tick)
        eta = (remaining / rate) if rate and rate > 0 else None
        if eta is not None:
            eta = max(0.0, eta)
        return rate, eta

    def _payload(self) -> dict[str, Any]:
        now = float(self._clock())
        rate, eta = self._rate_eta(now)
        elapsed = max(0.0, now - self._started_monotonic)
        denominator = self.target_absolute_tick - self.start_absolute_tick
        progress = (self.next_absolute_tick - self.start_absolute_tick) / denominator
        cells = [
            {
                "position": cell.position,
                "cell_id": cell.cell_id,
                "status": self._status,
                "next_absolute_tick": self.next_absolute_tick,
                "target_absolute_tick": self.target_absolute_tick,
                "progress_fraction": min(1.0, max(0.0, progress)),
                "rate_ticks_per_second": rate,
                "eta_seconds": eta,
            }
            for cell in self.cells
        ]
        return {
            "schema_version": SCHEMA_VERSION,
            "allocation_id": self.allocation_id,
            "device": self.device,
            "capacity": self.capacity,
            "pid": self.pid,
            "pid_start_ticks": self.pid_start_ticks,
            "producer": "emergence_slot_telemetry",
            "run_id": self.run_id,
            "campaign_id": self.campaign_id,
            "batch_id": self.batch_id,
            "profile_id": self.profile_id,
            "logical_world_count": self.logical_world_count,
            "gpu_admission_mode": self.gpu_admission_mode,
            "exclusive_gpu": self.exclusive_gpu,
            "peak_vram_gib": self.peak_vram_gib,
            "workload_class": self.workload_class or None,
            "gpu_device_id": self.gpu_device_id or None,
            "scheduler_resource_lease_id": (
                self.scheduler_resource_lease_id or None
            ),
            "scheduler_resource_ids": list(self.scheduler_resource_ids),
            "scheduler_fencing_token_digest": (
                self.scheduler_fencing_token_digest or None
            ),
            "run_root": self.run_root,
            "status": self._status,
            "stage": self._stage,
            "error": self._error or None,
            "start_absolute_tick": self.start_absolute_tick,
            "next_absolute_tick": self.next_absolute_tick,
            "target_absolute_tick": self.target_absolute_tick,
            "elapsed_seconds": elapsed,
            "rate_ticks_per_second": rate,
            "eta_seconds": eta,
            "started_at": self.started_at,
            "updated_at": _utc_now(),
            "cells": cells,
        }

    def _write(self) -> None:
        with self._lock:
            _atomic_json(self.path, self._payload())

    def update(
        self,
        *,
        current_generation: int | None = None,
        next_absolute_tick: int | None = None,
        status: str = "running",
        stage: str = "running",
        error: str = "",
    ) -> None:
        with self._lock:
            now = float(self._clock())
            if current_generation is not None:
                if isinstance(current_generation, bool):
                    raise ValueError("current_generation must be an integer")
                generation = int(current_generation)
                next_from_generation = generation * self.ticks_per_generation
                if next_absolute_tick is not None and next_absolute_tick != next_from_generation:
                    raise ValueError("generation and tick progress disagree")
                next_absolute_tick = next_from_generation
            if next_absolute_tick is not None:
                if isinstance(next_absolute_tick, bool) or not isinstance(next_absolute_tick, int):
                    raise ValueError("next_absolute_tick must be an integer")
                if (
                    next_absolute_tick < self.next_absolute_tick
                    or next_absolute_tick > self.target_absolute_tick
                ):
                    raise ValueError("slot progress must be monotonic and within target")
                delta_ticks = next_absolute_tick - self._last_sample_absolute_tick
                delta_seconds = now - self._last_sample_monotonic
                if delta_ticks > 0 and delta_seconds > 0:
                    sample = delta_seconds / delta_ticks
                    self._ema_seconds_per_tick = (
                        sample
                        if self._ema_seconds_per_tick is None
                        else 0.25 * sample + 0.75 * self._ema_seconds_per_tick
                    )
                    self._last_sample_absolute_tick = next_absolute_tick
                    self._last_sample_monotonic = now
                self.next_absolute_tick = next_absolute_tick
            self._status = _safe_text(status, fallback="running").lower()
            self._stage = _safe_text(stage, fallback=self._status)
            self._error = _safe_text(error)
            _atomic_json(self.path, self._payload())

    def close(self, *, status: str = "complete", error: str = "") -> None:
        normalized = _safe_text(status, fallback="complete").lower()
        if normalized not in TERMINAL_STATUSES:
            normalized = "failed"
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=max(0.5, self._heartbeat_interval_seconds * 2))
        target = self.target_absolute_tick if normalized == "complete" else self.next_absolute_tick
        self.update(
            next_absolute_tick=target,
            status=normalized,
            stage=normalized,
            error=error,
        )


def start_slot_lease(**kwargs: Any) -> SlotLease | NullSlotLease:
    """Create a lease without allowing telemetry setup to fail a simulation."""

    if os.environ.get("EIEL_SLOT_MONITOR", "auto").strip().lower() in {"0", "off", "false", "no"}:
        return NullSlotLease()
    try:
        return SlotLease(**kwargs)
    except Exception:
        return NullSlotLease()


def start_runner_slot_lease(
    runner: Any, *, start_generation: int, target_generation: int
) -> SlotLease | NullSlotLease:
    cfg = getattr(runner, "sim_cfg", None)
    configured = _safe_text(getattr(cfg, "device", "cpu"), fallback="cpu").lower()
    device = "gpu" if configured.startswith("cuda") else "cpu"
    run_root = Path(getattr(runner, "run_dir", Path.cwd())).resolve()
    run_id = _safe_text(getattr(cfg, "run_id", "")) or run_root.name
    ticks = int(getattr(cfg, "steps_per_generation", 0) or getattr(cfg, "ticks_per_generation", 0) or 1)
    return start_slot_lease(
        device=device,
        run_id=run_id,
        cell_ids=(run_id,),
        current_generation=int(start_generation),
        target_generation=int(target_generation),
        ticks_per_generation=ticks,
        run_root=run_root,
        campaign_id=_safe_text(getattr(cfg, "campaign_id", "")),
        profile_id=_safe_text(getattr(cfg, "profile_id", "")),
    )


def start_tick_runner_slot_lease(
    runner: Any, *, start_absolute_tick: int, target_absolute_tick: int
) -> SlotLease | NullSlotLease:
    """Create a tick-native heartbeat without generation compatibility inputs."""

    cfg = getattr(runner, "sim_cfg", None)
    configured = _safe_text(getattr(cfg, "device", "cpu"), fallback="cpu").lower()
    device = "gpu" if configured.startswith("cuda") else "cpu"
    run_root = Path(getattr(runner, "run_dir", Path.cwd())).resolve()
    run_id = _safe_text(getattr(cfg, "run_id", "")) or run_root.name
    return start_slot_lease(
        device=device,
        run_id=run_id,
        cell_ids=(run_id,),
        start_absolute_tick=int(start_absolute_tick),
        target_absolute_tick=int(target_absolute_tick),
        run_root=run_root,
        campaign_id=_safe_text(getattr(cfg, "campaign_id", "")),
        profile_id=_safe_text(getattr(cfg, "profile_id", "")),
    )


__all__ = [
    "CPU_CAPACITY",
    "DEFAULT_TELEMETRY_ROOT",
    "GPU_CAPACITY",
    "LEGACY_SCHEMA_VERSION",
    "NullSlotLease",
    "SCHEMA_VERSION",
    "SUPPORTED_SCHEMA_VERSIONS",
    "SlotLease",
    "start_runner_slot_lease",
    "start_slot_lease",
    "start_tick_runner_slot_lease",
    "telemetry_root",
]
