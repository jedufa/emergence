#!/usr/bin/env python3
"""Freeze one evidence-backed H1 HAPB representation decision."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


class H1VerdictError(RuntimeError):
    pass


def _sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def freeze_h1_verdict_v1(receipt_path: Path, *, output: Path) -> dict[str, Any]:
    if output.exists():
        raise H1VerdictError("H1 verdict output already exists")
    raw = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("schema_version") != "eiel_b48_wdl_jungle_eed_hapb_h1_canary_v1":
        raise H1VerdictError("H1 canary receipt schema is not recognized")
    rows = raw.get("variants")
    if not isinstance(rows, list):
        raise H1VerdictError("H1 canary receipt lacks variants")
    admitted = [
        row for row in rows
        if isinstance(row, Mapping)
        and row.get("representation") == "p256_hot_prefix"
        and row.get("active_rows_per_cell") == 512
        and row.get("tile_rows") == 512
        and row.get("buffer_count") == 1
        and row.get("canonical_host_storage") == "pageable"
        and row.get("host_pressure_passed") is True
        and float(row.get("projected_g200_seconds", float("inf"))) <= float(raw["frozen_gate"]["maximum_g200_seconds"])
    ]
    if len(admitted) != 1:
        raise H1VerdictError("required worst-case P256-hot-prefix policy is not uniquely admitted")
    selected = dict(admitted[0])
    verdict = {
        "schema_version": "eiel_b48_wdl_jungle_eed_hapb_h1_verdict_v1",
        "authority": "frozen_h1_preimplementation_representation_decision_v1",
        "h1_canary_path": str(receipt_path.resolve()),
        "h1_canary_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        "h1_receipt_sha256": raw.get("receipt_sha256"),
        "verdict": "PASS_START_H2",
        "selected_representation": {
            "representation": "p256_hot_prefix",
            "canonical_host_storage": "pageable",
            "bounded_pinned_staging": True,
            "tile_rows": 512,
            "buffer_count": 1,
            "active_rows_per_cell_canary": 512,
        },
        "selected_measurement": selected,
        "fully_pinned_canonical_admitted": raw.get("fully_pinned_admitted"),
        "qualification_claim": False,
        "execution_admission": False,
    }
    verdict["receipt_sha256"] = _sha(verdict)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return verdict


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h1-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = freeze_h1_verdict_v1(args.h1_receipt, output=args.output)
    print(json.dumps({"verdict": result["verdict"], "receipt_sha256": result["receipt_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
