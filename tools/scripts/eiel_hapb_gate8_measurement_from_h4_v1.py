#!/usr/bin/env python3
"""Build a Gate 8 measurement row from an H4 continuation receipt.

This is a projection of already-executed exclusive H4 evidence.  It never
launches CUDA, never treats WAIT-RESOURCE as PERF-B, and never marks an
ACS=0 correctness canary as a sustained production rehearsal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from engine.campaigns.canonical import canonical_sha256
from engine.campaigns.orchestration import atomic_write_json


SCHEMA_VERSION = "eiel_hapb_gate8_measurement_from_h4_v1"
TICKS_PER_GENERATION = 160


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path} is not a JSON object")
    return value


def _g0_to_g6_native_execute_ns(receipt: Mapping[str, Any]) -> int:
    timing = receipt.get("secondary_boundary_timing")
    _require(isinstance(timing, Mapping), "H4 secondary_boundary_timing is missing")
    samples = timing.get("samples")
    _require(isinstance(samples, list), "H4 timing samples are missing")
    for sample in samples:
        if isinstance(sample, Mapping) and sample.get("label") == "g0_to_g6":
            wall = sample.get("native_execute_wall_ns")
            _require(type(wall) is int and wall > 0, "g0_to_g6 native_execute_wall_ns is invalid")
            return wall
    raise RuntimeError("g0_to_g6 timing sample is absent")


def measurement_row_from_h4_receipt_v1(
    receipt: Mapping[str, Any],
    *,
    candidate_id: str,
    route: str,
    receipt_path: Path,
    sustained_rehearsal_pass: bool = False,
) -> dict[str, Any]:
    cells = receipt.get("physical_cells")
    generations = receipt.get("generations")
    _require(type(cells) is int and cells > 0, "physical_cells is invalid")
    _require(type(generations) is int and generations > 0, "generations is invalid")
    wall_ns = _g0_to_g6_native_execute_ns(receipt)
    seconds = wall_ns / 1_000_000_000.0 / float(generations)
    _require(seconds > 0.0, "amortized generation seconds must be positive")
    exact = receipt.get("continuation_exact") is True
    restore_ok = receipt.get("checkpoint_restore_boundary_mismatch_count") == 0
    memory = receipt.get("memory")
    if not isinstance(memory, Mapping):
        memory = {
            "device_total_bytes": None,
            "minimum_free_at_peak_bytes": None,
            "rule_pass": False,
        }
    return {
        "candidate_id": candidate_id,
        "route": route,
        "batch_cells": cells,
        "generations": generations,
        "ticks_per_generation": TICKS_PER_GENERATION,
        "semantically_exact": exact and restore_ok,
        "sustained_rehearsal_pass": bool(sustained_rehearsal_pass),
        "production_execution_admission": False,
        "qualification_claim": False,
        "amortized_production_seconds_per_generation": seconds,
        "production_equivalent_cell_generations_per_hour": cells * 3600.0 / seconds,
        "ms_per_tick": (seconds / TICKS_PER_GENERATION) * 1000.0,
        "native_execute_wall_ns": wall_ns,
        "amortized_generation_wall_seconds": seconds,
        "checkpoint_bytes": receipt.get("midpoint_checkpoint_bytes"),
        "checkpoint_persist_wall_ns": receipt.get("checkpoint_persist_wall_ns"),
        "gpu_margin_bytes": memory.get("physical_margin_bytes"),
        "host_margin_bytes": (
            receipt.get("host") or {}
        ).get("physical_margin_bytes")
        if isinstance(receipt.get("host"), Mapping)
        else None,
        "semantic_status": (
            "exact"
            if exact and restore_ok
            else "not_exact"
        ),
        "admission_status": (
            "exclusive_engineering"
            if (receipt.get("exclusive_cuda_admission") or {}).get("admitted") is True
            else "not_exclusive"
        ),
        "memory": {
            "device_total_bytes": memory.get("device_total_bytes"),
            "minimum_free_at_peak_bytes": memory.get("minimum_free_at_peak_bytes"),
            "peak_used_bytes": memory.get("peak_used_bytes"),
            "physical_margin_bytes": memory.get("physical_margin_bytes"),
            "rule_pass": memory.get("rule_pass") is True,
        },
        "host": dict(receipt.get("host") or {})
        if isinstance(receipt.get("host"), Mapping)
        else {"status": "UNAVAILABLE", "physical_margin_bytes": None, "rule_pass": False},
        "timing": {
            "source": "h4_secondary_boundary_g0_to_g6",
            "acs_checkpoint_slots": receipt.get("acs_checkpoint_slots"),
            "unavailable": [] if wall_ns else ["native_execute_wall_ns"],
        },
        "source_receipt": str(receipt_path),
        "source_receipt_sha256": receipt.get("receipt_sha256"),
        "continuation_exact": exact,
        "checkpoint_restore_boundary_mismatch_count": receipt.get(
            "checkpoint_restore_boundary_mismatch_count"
        ),
        "finalist_clean_process_count": 1,
    }


def compare_h4_canaries_v1(
    run1: Mapping[str, Any], run2: Mapping[str, Any]
) -> dict[str, Any]:
    fields = (
        "continuation_exact",
        "fresh_owner_restore",
        "checkpoint_restore_boundary_mismatch_count",
        "physical_cells",
        "generations",
        "artifact_sha256",
        "midpoint_checkpoint_sha256",
    )
    mismatches = [
        field
        for field in fields
        if run1.get(field) != run2.get(field)
    ]
    return {
        "schema_version": f"{SCHEMA_VERSION}_canary_compare",
        "agree": not mismatches,
        "mismatched_fields": mismatches,
        "run1_receipt_sha256": run1.get("receipt_sha256"),
        "run2_receipt_sha256": run2.get("receipt_sha256"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h4-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--route", default="hapb")
    parser.add_argument("--sustained-rehearsal-pass", action="store_true")
    parser.add_argument("--compare-h4-receipt", type=Path)
    args = parser.parse_args()
    receipt_path = args.h4_receipt.expanduser().resolve()
    output = args.output.expanduser().resolve()
    _require(receipt_path.is_file(), f"H4 receipt is absent: {receipt_path}")
    _require(not output.exists(), f"refusing to overwrite {output}")
    receipt = _load_json(receipt_path)
    row = measurement_row_from_h4_receipt_v1(
        receipt,
        candidate_id=args.candidate_id,
        route=args.route,
        receipt_path=receipt_path,
        sustained_rehearsal_pass=args.sustained_rehearsal_pass,
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "measurement": row,
        "gpu_execution_performed": False,
    }
    if args.compare_h4_receipt is not None:
        other_path = args.compare_h4_receipt.expanduser().resolve()
        _require(other_path.is_file(), f"compare H4 receipt is absent: {other_path}")
        payload["canary_compare"] = compare_h4_canaries_v1(
            _load_json(other_path), receipt
        )
    payload["receipt_sha256"] = canonical_sha256(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, payload)
    print(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "output": str(output),
        "candidate_id": args.candidate_id,
        "batch_cells": row["batch_cells"],
        "semantically_exact": row["semantically_exact"],
        "sustained_rehearsal_pass": row["sustained_rehearsal_pass"],
        "canary_agree": None if "canary_compare" not in payload else payload["canary_compare"]["agree"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
