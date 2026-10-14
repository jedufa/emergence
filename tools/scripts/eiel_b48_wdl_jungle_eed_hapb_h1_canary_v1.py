#!/usr/bin/env python3
"""Bounded H1 traffic canary for the host-authoritative policy-bank design.

This is intentionally not an execution implementation.  It replays an actual
resident B1 all-active Genome V2 refill observation as stable row identifiers,
scales that observation across the fixed B48 envelope, and measures only the
required D2H discovery, CPU gather, and pinned H2D path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time
from typing import Any, Final, Mapping, Sequence


SCHEMA_VERSION: Final[str] = "eiel_b48_wdl_jungle_eed_hapb_h1_canary_v1"
POLICY_WIDTH: Final[int] = 9_991
POLICY_ROW_BYTES: Final[int] = POLICY_WIDTH * 4
CELL_COUNT: Final[int] = 48
CAPACITY: Final[int] = 512
P256: Final[int] = 256
TICKS: Final[int] = 160
GENERATIONS: Final[int] = 200
MAX_G200_SECONDS: Final[float] = 544_320.0
MAX_HOST_RSS_BYTES: Final[int] = 4 * 1024**3
MAX_PSI_AVG10_DELTA: Final[float] = 0.05


class H1CanaryError(RuntimeError):
    """The bounded H1 canary cannot make a faithful measurement."""


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _read_meminfo() -> dict[str, int]:
    result: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        name, raw = line.split(":", 1)
        fields = raw.split()
        if fields:
            result[name] = int(fields[0]) * 1024
    return result


def _psi_memory_avg10() -> float:
    path = Path("/proc/pressure/memory")
    if not path.is_file():
        return 0.0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("some "):
            for field in line.split():
                if field.startswith("avg10="):
                    return float(field.split("=", 1)[1])
    return 0.0


def _major_faults() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_majflt)


def _host_snapshot() -> dict[str, object]:
    status = Path("/proc/self/status").read_text(encoding="utf-8")
    fields = {
        line.split(":", 1)[0]: line.split(":", 1)[1].split()[0]
        for line in status.splitlines()
        if ":" in line and line.split(":", 1)[0] in {"VmRSS", "VmHWM"}
    }
    mem = _read_meminfo()
    return {
        "rss_bytes": int(fields.get("VmRSS", "0")) * 1024,
        "hwm_bytes": int(fields.get("VmHWM", "0")) * 1024,
        "major_faults": _major_faults(),
        "swap_free_bytes": int(mem.get("SwapFree", 0)),
        "psi_memory_avg10": _psi_memory_avg10(),
    }


def _delta(before: Mapping[str, object], after: Mapping[str, object]) -> dict[str, object]:
    return {
        "rss_bytes": int(after["rss_bytes"]),
        "hwm_bytes": int(after["hwm_bytes"]),
        "major_fault_delta": int(after["major_faults"]) - int(before["major_faults"]),
        "swap_free_delta_bytes": int(after["swap_free_bytes"]) - int(before["swap_free_bytes"]),
        "psi_memory_avg10_delta": float(after["psi_memory_avg10"]) - float(before["psi_memory_avg10"]),
    }


def _load_observed_trace(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != "eiel_hapb_policy_refill_trace_v1":
        raise H1CanaryError("H1 requires the resident policy-refill trace schema")
    rows = raw.get("rows_by_tick")
    if not isinstance(rows, list) or len(rows) != TICKS:
        raise H1CanaryError("resident trace is not one T160 observation")
    observed: list[list[int]] = []
    for tick, values in enumerate(rows):
        if not isinstance(values, list) or not values:
            raise H1CanaryError(f"resident trace tick {tick} is empty")
        slots: list[int] = []
        for value in values:
            if not isinstance(value, list) or len(value) != 2 or value[0] != 0:
                raise H1CanaryError("resident B1 trace has a malformed row")
            slot = value[1]
            if type(slot) is not int or not 0 <= slot < CAPACITY:
                raise H1CanaryError("resident B1 trace slot is outside C512")
            slots.append(slot)
        if slots != list(range(P256)):
            raise H1CanaryError(
                "H1 requires observed stable low-slot P256 occupancy on every tick"
            )
        observed.append(slots)
    if raw.get("source_commit") != "03d1a9220c79ab81fa7b56a298aed21aafaecf96":
        raise H1CanaryError("resident trace is not bound to the 03d1a922 oracle artifact")
    return {"trace": raw, "observed_slots_by_tick": observed}


def _scaled_indices(observed: Sequence[Sequence[int]], *, active_rows: int, representation: str) -> tuple[tuple[int, ...], ...]:
    if active_rows not in {P256, CAPACITY}:
        raise H1CanaryError("H1 active row scale must be P256 or C512")
    if representation not in {"pure", "p256_hot_prefix"}:
        raise H1CanaryError("H1 representation is unsupported")
    result: list[tuple[int, ...]] = []
    expected = tuple(range(P256))
    for slots in observed:
        if tuple(slots) != expected:
            raise H1CanaryError("resident observed slot ordering changed during scale")
        per_cell = range(active_rows) if representation == "pure" else range(P256, active_rows)
        result.append(tuple(cell * CAPACITY + slot for cell in range(CELL_COUNT) for slot in per_cell))
    return tuple(result)


def _pressure_passed(delta: Mapping[str, object]) -> bool:
    return (
        int(delta["rss_bytes"]) <= MAX_HOST_RSS_BYTES
        and int(delta["hwm_bytes"]) <= MAX_HOST_RSS_BYTES
        and int(delta["major_fault_delta"]) == 0
        and int(delta["swap_free_delta_bytes"]) >= 0
        and float(delta["psi_memory_avg10_delta"]) <= MAX_PSI_AVG10_DELTA
    )


def _run_variant(
    *,
    indices_by_tick: Sequence[Sequence[int]],
    representation: str,
    active_rows: int,
    tile_rows: int,
    buffer_count: int,
    canonical_pinned: bool,
) -> dict[str, object]:
    import torch

    canonical = torch.empty(
        (CELL_COUNT, CAPACITY, POLICY_WIDTH), dtype=torch.float32, pin_memory=canonical_pinned
    )
    canonical.zero_()  # Touch the single canonical host authority once.
    staging = [
        torch.empty((tile_rows, POLICY_WIDTH), dtype=torch.float32, pin_memory=True)
        for _ in range(buffer_count)
    ]
    device_tiles = [
        torch.empty((tile_rows, POLICY_WIDTH), dtype=torch.float32, device="cuda")
        for _ in range(buffer_count)
    ]
    before = _host_snapshot()
    d2h_seconds = 0.0
    gather_seconds = 0.0
    h2d_seconds = 0.0
    sync_count = 0
    tile_count = 0
    total_bytes = 0
    canonical_rows = canonical.reshape(-1, POLICY_WIDTH)
    started = time.monotonic()
    for tick_rows in indices_by_tick:
        cpu_indices = torch.tensor(tick_rows, dtype=torch.int64)
        device_indices = cpu_indices.to(device="cuda", non_blocking=False)
        discovery = torch.empty_like(cpu_indices, pin_memory=True)
        t0 = time.monotonic()
        discovery.copy_(device_indices, non_blocking=True)
        torch.cuda.synchronize()
        d2h_seconds += time.monotonic() - t0
        sync_count += 1
        if not torch.equal(discovery, cpu_indices):
            raise H1CanaryError("D2H policy-row discovery changed stable row order")
        for tile_index, start in enumerate(range(0, len(cpu_indices), tile_rows)):
            stop = min(start + tile_rows, len(cpu_indices))
            count = stop - start
            buffer_index = tile_index % buffer_count
            t0 = time.monotonic()
            torch.index_select(canonical_rows, 0, cpu_indices[start:stop], out=staging[buffer_index][:count])
            gather_seconds += time.monotonic() - t0
            t0 = time.monotonic()
            device_tiles[buffer_index][:count].copy_(staging[buffer_index][:count], non_blocking=True)
            torch.cuda.synchronize()
            h2d_seconds += time.monotonic() - t0
            sync_count += 1
            tile_count += 1
            total_bytes += count * POLICY_ROW_BYTES
    wall_seconds = time.monotonic() - started
    after = _host_snapshot()
    pressure = _delta(before, after)
    return {
        "representation": representation,
        "active_rows_per_cell": active_rows,
        "tile_rows": tile_rows,
        "buffer_count": buffer_count,
        "canonical_host_storage": "pinned" if canonical_pinned else "pageable",
        "canonical_host_bytes": canonical.numel() * canonical.element_size(),
        "bounded_pinned_staging_bytes": sum(value.numel() * value.element_size() for value in staging),
        "d2h_discovery_seconds": d2h_seconds,
        "cpu_gather_seconds": gather_seconds,
        "h2d_seconds": h2d_seconds,
        "h2d_effective_gib_per_second": (
            total_bytes / h2d_seconds / 1024**3 if h2d_seconds else None
        ),
        "gpu_idle_seconds": h2d_seconds,
        "synchronization_count": sync_count,
        "tile_count": tile_count,
        "policy_bytes_per_generation": total_bytes,
        "wall_seconds_per_generation": wall_seconds,
        "projected_g200_seconds": wall_seconds * GENERATIONS,
        "host_pressure": pressure,
        "host_pressure_passed": _pressure_passed(pressure),
    }


def run_h1_canary_v1(*, trace_path: Path, output_root: Path, try_fully_pinned: bool) -> dict[str, object]:
    """Freeze the H1 gate, then measure every required pageable variant."""

    import torch

    if output_root.exists():
        raise H1CanaryError("H1 evidence root already exists")
    if not torch.cuda.is_available():
        raise H1CanaryError("H1 requires an available CUDA device")
    observed = _load_observed_trace(trace_path.expanduser().resolve())
    output_root.mkdir(parents=True)
    frozen = {
        "schema_version": SCHEMA_VERSION,
        "authority": "h1_frozen_throughput_host_pressure_gate_v1",
        "trace_sha256": _sha256(observed["trace"]),
        "maximum_g200_seconds": MAX_G200_SECONDS,
        "max_host_rss_bytes": MAX_HOST_RSS_BYTES,
        "max_major_fault_delta": 0,
        "min_swap_free_delta_bytes": 0,
        "max_psi_memory_avg10_delta": MAX_PSI_AVG10_DELTA,
        "fully_pinned_only_if_safely_admitted": True,
    }
    frozen["receipt_sha256"] = _sha256(frozen)
    (output_root / "h1_frozen_gate.json").write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    memlock = resource.getrlimit(resource.RLIMIT_MEMLOCK)[0]
    canonical_bytes = CELL_COUNT * CAPACITY * POLICY_ROW_BYTES
    fully_pinned_admitted = bool(try_fully_pinned and (memlock < 0 or memlock >= canonical_bytes))
    variants: list[dict[str, object]] = []
    for active_rows in (P256, CAPACITY):
        for representation in ("pure", "p256_hot_prefix"):
            indices = _scaled_indices(observed["observed_slots_by_tick"], active_rows=active_rows, representation=representation)
            for tile_rows in (512, 4096):
                for buffer_count in (1, 2):
                    variants.append(_run_variant(indices_by_tick=indices, representation=representation, active_rows=active_rows, tile_rows=tile_rows, buffer_count=buffer_count, canonical_pinned=False))
                if fully_pinned_admitted:
                    variants.append(_run_variant(indices_by_tick=indices, representation=representation, active_rows=active_rows, tile_rows=tile_rows, buffer_count=1, canonical_pinned=True))
    accepted = [
        row for row in variants
        if bool(row["host_pressure_passed"])
        and float(row["projected_g200_seconds"]) <= MAX_G200_SECONDS
    ]
    result = {
        "schema_version": SCHEMA_VERSION,
        "authority": "h1_preimplementation_bandwidth_host_pressure_canary_v1",
        "trace_path": str(trace_path.expanduser().resolve()),
        "trace_sha256": _sha256(observed["trace"]),
        "frozen_gate": frozen,
        "canonical_host_shape": [CELL_COUNT, CAPACITY, POLICY_WIDTH],
        "fully_pinned_requested": try_fully_pinned,
        "fully_pinned_admitted": fully_pinned_admitted,
        "memlock_limit_bytes": memlock,
        "variants": variants,
        "accepted_variant_count": len(accepted),
        "h1_gate_passed": bool(accepted),
        "qualification_claim": False,
        "execution_admission": False,
    }
    result["receipt_sha256"] = _sha256(result)
    (output_root / "h1_canary_receipt.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--try-fully-pinned", action="store_true")
    args = parser.parse_args(argv)
    result = run_h1_canary_v1(trace_path=args.trace, output_root=args.output_root, try_fully_pinned=args.try_fully_pinned)
    print(json.dumps({"accepted_variant_count": result["accepted_variant_count"], "h1_gate_passed": result["h1_gate_passed"], "receipt_sha256": result["receipt_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
