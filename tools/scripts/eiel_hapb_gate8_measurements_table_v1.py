#!/usr/bin/env python3
"""Assemble a Gate 8 measurement table from executed receipts. CPU only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from engine.campaigns.canonical import canonical_sha256
from engine.campaigns.orchestration import atomic_write_json
from tools.scripts.eiel_hapb_gate8_benchmark_analysis_v1 import INPUT_SCHEMA, PROMOTION_THRESHOLD
from tools.scripts.eiel_hapb_gate8_measurement_from_h4_v1 import (
    compare_h4_canaries_v1,
    measurement_row_from_h4_receipt_v1,
)


SCHEMA_VERSION = "eiel_hapb_gate8_measurements_table_v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path} is not a JSON object")
    return value


def resident_row_from_receipt_v1(receipt: Mapping[str, Any], *, receipt_path: Path) -> dict[str, Any]:
    _require(receipt.get("candidate_id") == "resident_b48", "resident receipt candidate_id is not resident_b48")
    _require(receipt.get("batch_cells") == 48, "resident receipt is not B48")
    seconds = receipt.get("amortized_production_seconds_per_generation")
    _require(type(seconds) in {int, float} and float(seconds) > 0.0, "resident amortized seconds invalid")
    memory = receipt.get("memory") if isinstance(receipt.get("memory"), Mapping) else {}
    return {
        "candidate_id": "resident_b48",
        "route": "resident",
        "batch_cells": 48,
        "semantically_exact": True,
        "sustained_rehearsal_pass": True,
        "production_execution_admission": False,
        "qualification_claim": False,
        "amortized_production_seconds_per_generation": float(seconds),
        "memory": {
            "device_total_bytes": memory.get("device_total_bytes"),
            "minimum_free_at_peak_bytes": memory.get("minimum_free_at_peak_bytes"),
            "peak_used_bytes": memory.get("peak_used_bytes"),
            "rule_pass": memory.get("rule_pass") is True,
        },
        "timing": {"source": "gate8_resident_b48_receipt", "unavailable": []},
        "finalist_clean_process_count": 1,
        "source_receipt": str(receipt_path),
        "source_receipt_sha256": receipt.get("receipt_sha256"),
    }


def assemble_measurements_table_v1(
    *,
    resident: Mapping[str, Any],
    hapb_rows: list[Mapping[str, Any]],
    high_b_abi: str,
    ladder_complete: bool,
    ladder_stop_reason: str | None,
) -> dict[str, Any]:
    _require(high_b_abi in {"not_implemented", "frozen"}, "high_b_abi must be not_implemented or frozen")
    measurements = [dict(resident), *[dict(row) for row in hapb_rows]]
    return {
        "schema_version": INPUT_SCHEMA,
        "high_b_abi": high_b_abi,
        "implemented_batch_cells": sorted({int(row["batch_cells"]) for row in hapb_rows}),
        "ladder_complete": ladder_complete,
        "ladder_stop_reason": ladder_stop_reason,
        "promotion_threshold": PROMOTION_THRESHOLD,
        "measurements": measurements,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resident-receipt", type=Path, required=True)
    parser.add_argument("--h4-receipt", action="append", default=[], type=Path)
    parser.add_argument("--candidate-id", action="append", default=[])
    parser.add_argument(
        "--sustained-candidate-id",
        action="append",
        default=[],
        help="candidate ids whose H4 G6 horizon matches the Resident control",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--high-b-abi", default="frozen")
    parser.add_argument("--ladder-complete", action="store_true")
    parser.add_argument("--ladder-stop-reason", default=None)
    parser.add_argument("--compare-h4-a", type=Path)
    parser.add_argument("--compare-h4-b", type=Path)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    _require(not output.exists(), f"refusing to overwrite {output}")
    resident_path = args.resident_receipt.expanduser().resolve()
    resident = resident_row_from_receipt_v1(_load_json(resident_path), receipt_path=resident_path)
    _require(len(args.h4_receipt) == len(args.candidate_id), "--h4-receipt and --candidate-id counts must match")
    sustained = set(args.sustained_candidate_id)
    hapb_rows: list[dict[str, Any]] = []
    for path, candidate_id in zip(args.h4_receipt, args.candidate_id, strict=True):
        receipt_path = path.expanduser().resolve()
        hapb_rows.append(
            measurement_row_from_h4_receipt_v1(
                _load_json(receipt_path),
                candidate_id=candidate_id,
                route="hapb",
                receipt_path=receipt_path,
                sustained_rehearsal_pass=candidate_id in sustained,
            )
        )
    table = assemble_measurements_table_v1(
        resident=resident,
        hapb_rows=hapb_rows,
        high_b_abi=args.high_b_abi,
        ladder_complete=bool(args.ladder_complete),
        ladder_stop_reason=args.ladder_stop_reason,
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "table": table,
        "gpu_execution_performed": False,
    }
    if args.compare_h4_a is not None or args.compare_h4_b is not None:
        _require(args.compare_h4_a is not None and args.compare_h4_b is not None, "compare receipts must be paired")
        payload["canary_compare"] = compare_h4_canaries_v1(
            _load_json(args.compare_h4_a.expanduser().resolve()),
            _load_json(args.compare_h4_b.expanduser().resolve()),
        )
    payload["receipt_sha256"] = canonical_sha256(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, payload)
    print(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "output": str(output),
        "hapb_candidates": [row["candidate_id"] for row in hapb_rows],
        "ladder_complete": table["ladder_complete"],
        "canary_agree": None if "canary_compare" not in payload else payload["canary_compare"]["agree"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
