#!/usr/bin/env python3
"""Measure authoritative-state admission and prove it is outside native ticks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Sequence

from engine.authoritative_state.identity import canonical_sha256
from engine.authoritative_state.production_identity import (
    validate_admission_identity_v1,
)


SCHEMA_VERSION = "eiel_authoritative_state_admission_performance_receipt_v1"
NATIVE_SUFFIXES = frozenset({".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp"})


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _percentile(values: list[int], fraction: float) -> int:
    index = max(0, math.ceil(len(values) * fraction) - 1)
    return values[index]


def _native_call_sites(source_root: Path) -> list[str]:
    token = "validate_admission_identity_v1"
    matches: list[str] = []
    for path in sorted(
        path
        for path in (source_root / "engine").rglob("*")
        if path.is_file() and path.suffix in NATIVE_SUFFIXES
    ):
        for line, text in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(),
            start=1,
        ):
            if token in text:
                matches.append(f"{path.relative_to(source_root)}:{line}")
    return matches


def build_receipt(
    *,
    manifest_path: Path,
    source_root: Path,
    iterations: int,
    batch_cells: int,
    generations: int,
    ticks_per_generation: int,
) -> dict[str, Any]:
    if iterations < 10:
        raise ValueError("iterations must be at least 10")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = manifest["artifact_identity"]["runtime_resource_receipt"][
        "authoritative_state_identity_v1"
    ]
    expected = canonical_sha256(identity)
    for _ in range(10):
        observed = validate_admission_identity_v1(
            request=identity,
            provider=identity,
            manifest=identity,
            binary=identity,
        )
        if observed != expected:
            raise RuntimeError("warmup admission identity differs")
    timings: list[int] = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        observed = validate_admission_identity_v1(
            request=identity,
            provider=identity,
            manifest=identity,
            binary=identity,
        )
        elapsed = time.perf_counter_ns() - started
        if observed != expected:
            raise RuntimeError("measured admission identity differs")
        timings.append(elapsed)
    timings.sort()
    native_matches = _native_call_sites(source_root)
    if native_matches:
        raise RuntimeError(
            "admission validation is referenced by native tick sources: "
            + ", ".join(native_matches)
        )
    physical_cell_ticks = batch_cells * generations * ticks_per_generation
    median = int(statistics.median(timings))
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "identity_schema_sha256": identity["state_schema_sha256"],
        "identity_sha256": expected,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _file_sha256(manifest_path),
        "iterations": iterations,
        "warmup_iterations": 10,
        "admission_validation_wall_ns": {
            "minimum": timings[0],
            "median": median,
            "p95": _percentile(timings, 0.95),
            "maximum": timings[-1],
        },
        "execution_envelope": {
            "batch_cells": batch_cells,
            "generations": generations,
            "ticks_per_generation": ticks_per_generation,
            "physical_cell_ticks": physical_cell_ticks,
        },
        "amortized_admission_ns_per_physical_cell_tick": (
            median / physical_cell_ticks
        ),
        "native_tick_admission_call_site_count": 0,
        "native_tick_admission_call_sites": [],
        "per_tick_validation_added": False,
        "incremental_post_dispatch_validation_overhead_ns_per_tick": 0,
        "post_dispatch_zero_basis": (
            "No native C, C++, CUDA, or header source references the admission "
            "validator; admission runs in the artifact loader before dispatch."
        ),
    }
    unsigned = json.dumps(
        receipt,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    receipt["receipt_sha256"] = hashlib.sha256(unsigned).hexdigest()
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--batch-cells", type=int, default=72)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--ticks-per-generation", type=int, default=80)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    receipt = build_receipt(
        manifest_path=args.manifest.resolve(strict=True),
        source_root=args.source_root.resolve(strict=True),
        iterations=args.iterations,
        batch_cells=args.batch_cells,
        generations=args.generations,
        ticks_per_generation=args.ticks_per_generation,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(args.output),
                "receipt_sha256": receipt["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
