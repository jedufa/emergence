#!/usr/bin/env python3
"""Fail-closed HAPB traffic accounting before any tiled policy implementation.

This tool deliberately models only a resident-observed policy-refill trace and
predeclared timing samples.  It cannot qualify a route, select a result after
the fact, or stand in for the later H2D/D2H canary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Sequence


SCHEMA_VERSION: Final[str] = "eiel_b48_wdl_jungle_eed_hapb_feasibility_v1"
POLICY_WIDTH: Final[int] = 9_991
POLICY_ROW_BYTES: Final[int] = POLICY_WIDTH * 4
CELL_COUNT: Final[int] = 48
HOT_PREFIX_ROWS: Final[int] = 256
TICKS_PER_GENERATION: Final[int] = 160
GENERATIONS: Final[int] = 200
MAX_G200_SECONDS: Final[float] = 544_320.0  # 90% of the frozen seven-day budget.


class HapbFeasibilityError(ValueError):
    """The trace, timing sample, or acceptance rule is not faithful enough."""


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _exact_int(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise HapbFeasibilityError(f"{name} must be an exact integer >= {minimum}")
    return value


@dataclass(frozen=True)
class PolicyRefillTraceV1:
    """A bounded resident-oracle trace of stable active policy rows."""

    route_identity: str
    source_commit: str
    ticks_per_generation: int
    rows_by_tick: tuple[tuple[tuple[int, int], ...], ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PolicyRefillTraceV1":
        if value.get("schema_version") != "eiel_hapb_policy_refill_trace_v1":
            raise HapbFeasibilityError("trace schema is not eiel_hapb_policy_refill_trace_v1")
        route = value.get("route_identity")
        commit = value.get("source_commit")
        if not isinstance(route, str) or not route:
            raise HapbFeasibilityError("trace route identity is missing")
        if not isinstance(commit, str) or len(commit) < 12:
            raise HapbFeasibilityError("trace source commit is missing")
        ticks = _exact_int(value.get("ticks_per_generation"), name="trace ticks")
        rows_raw = value.get("rows_by_tick")
        if not isinstance(rows_raw, list) or len(rows_raw) != ticks:
            raise HapbFeasibilityError("trace must retain one stable-row list per tick")
        rows: list[tuple[tuple[int, int], ...]] = []
        for tick, tick_rows in enumerate(rows_raw):
            if not isinstance(tick_rows, list):
                raise HapbFeasibilityError(f"trace tick {tick} is not a row list")
            parsed: list[tuple[int, int]] = []
            previous = (-1, -1)
            for raw in tick_rows:
                if not isinstance(raw, list) or len(raw) != 2:
                    raise HapbFeasibilityError(f"trace tick {tick} row is malformed")
                row = (_exact_int(raw[0], name="cell"), _exact_int(raw[1], name="slot"))
                if row[0] >= CELL_COUNT or row[1] >= 512 or row <= previous:
                    raise HapbFeasibilityError("trace rows must be unique stable cell/slot pairs")
                previous = row
                parsed.append(row)
            rows.append(tuple(parsed))
        return cls(route, commit, ticks, tuple(rows))


def traffic_for_trace_v1(
    trace: PolicyRefillTraceV1, *, representation: str, tile_rows: int
) -> dict[str, int]:
    """Return exact policy payload and tile counts for one representation."""

    if representation not in {"pure", "p256_hot_prefix"}:
        raise HapbFeasibilityError("representation must be pure or p256_hot_prefix")
    if tile_rows not in {512, 4096}:
        raise HapbFeasibilityError("tile rows must be exactly 512 or 4096")
    per_tick: list[int] = []
    tiles = 0
    for active in trace.rows_by_tick:
        tiled = (
            active
            if representation == "pure"
            else tuple(row for row in active if row[1] >= HOT_PREFIX_ROWS)
        )
        per_tick.append(len(tiled) * POLICY_ROW_BYTES)
        tiles += math.ceil(len(tiled) / tile_rows)
    return {
        "policy_row_bytes": POLICY_ROW_BYTES,
        "tiled_rows_per_generation": sum(value // POLICY_ROW_BYTES for value in per_tick),
        "policy_bytes_per_generation": sum(per_tick),
        "maximum_policy_bytes_per_tick": max(per_tick, default=0),
        "tile_count_per_generation": tiles,
    }


def decide_feasibility_v1(
    trace: PolicyRefillTraceV1,
    *,
    representation: str,
    tile_rows: int,
    measured_seconds_per_generation: float,
    host_pressure_passed: bool,
) -> dict[str, object]:
    """Apply the frozen H1 gate to a measurement made before result selection."""

    if type(measured_seconds_per_generation) not in {int, float} or measured_seconds_per_generation <= 0:
        raise HapbFeasibilityError("measured seconds per generation must be positive")
    if type(host_pressure_passed) is not bool:
        raise HapbFeasibilityError("host pressure outcome must be an exact boolean")
    traffic = traffic_for_trace_v1(trace, representation=representation, tile_rows=tile_rows)
    projected = float(measured_seconds_per_generation) * GENERATIONS
    return {
        "schema_version": SCHEMA_VERSION,
        "authority": "h1_preimplementation_traffic_and_host_pressure_gate_v1",
        "trace": {
            "route_identity": trace.route_identity,
            "source_commit": trace.source_commit,
            "ticks_per_generation": trace.ticks_per_generation,
            "trace_sha256": _canonical_sha256({
                "route_identity": trace.route_identity,
                "source_commit": trace.source_commit,
                "ticks_per_generation": trace.ticks_per_generation,
                "rows_by_tick": trace.rows_by_tick,
            }),
        },
        "representation": representation,
        "tile_rows": tile_rows,
        "traffic": traffic,
        "frozen_acceptance": {
            "generations": GENERATIONS,
            "maximum_g200_seconds": MAX_G200_SECONDS,
            "host_pressure_must_pass": True,
        },
        "measurement": {"seconds_per_generation": float(measured_seconds_per_generation), "host_pressure_passed": host_pressure_passed},
        "projected_g200_seconds": projected,
        "accepted": projected <= MAX_G200_SECONDS and host_pressure_passed,
        "qualification_claim": False,
        "execution_admission": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--representation", choices=("pure", "p256_hot_prefix"), required=True)
    parser.add_argument("--tile-rows", type=int, choices=(512, 4096), required=True)
    parser.add_argument("--seconds-per-generation", type=float, required=True)
    parser.add_argument("--host-pressure-passed", choices=("true", "false"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise HapbFeasibilityError("output already exists; H1 evidence is immutable")
    trace_raw = json.loads(args.trace.read_text(encoding="utf-8"))
    if not isinstance(trace_raw, Mapping):
        raise HapbFeasibilityError("trace root must be an object")
    result = decide_feasibility_v1(
        PolicyRefillTraceV1.from_mapping(trace_raw),
        representation=args.representation,
        tile_rows=args.tile_rows,
        measured_seconds_per_generation=args.seconds_per_generation,
        host_pressure_passed=args.host_pressure_passed == "true",
    )
    result["receipt_sha256"] = _canonical_sha256(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"accepted": result["accepted"], "receipt_sha256": result["receipt_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
