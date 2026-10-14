#!/usr/bin/env python3
"""Replay GMNW boundary receipts under the frozen authority contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Final


ROOT: Final[Path] = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.perf.gmnw_evidence_receipt_v1 import (  # noqa: E402
    consistency_receipt,
    file_sha256,
    load_contract,
    replay_boundary_receipt,
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON receipt must be an object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def execute(
    *,
    contract_path: Path,
    g4_to_g5_receipt: Path,
    g8_to_g9_receipt: Path,
    output_directory: Path,
) -> dict[str, Any]:
    expected_names = {
        "g4_to_g5_replayed.json",
        "g8_to_g9_replayed.json",
        "decision_consistency_receipt.json",
    }
    if output_directory.exists():
        unexpected = sorted(
            path.name for path in output_directory.iterdir() if path.name not in expected_names
        )
        if unexpected:
            raise FileExistsError(
                f"output directory contains unrelated files: {unexpected}"
            )
    contract = load_contract(contract_path)
    contract_sha256 = file_sha256(contract_path)
    rows = []
    for source_path, output_name in (
        (g4_to_g5_receipt, "g4_to_g5_replayed.json"),
        (g8_to_g9_receipt, "g8_to_g9_replayed.json"),
    ):
        row = replay_boundary_receipt(
            _load_json(source_path),
            source_path=source_path,
            source_sha256=file_sha256(source_path),
            contract=contract,
            contract_path=contract_path,
            contract_sha256=contract_sha256,
        )
        _write_json(output_directory / output_name, row)
        rows.append(row)
    decision = consistency_receipt(
        rows,
        contract_path=contract_path,
        contract_sha256=contract_sha256,
    )
    _write_json(output_directory / "decision_consistency_receipt.json", decision)
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--g4-to-g5-receipt", type=Path, required=True)
    parser.add_argument("--g8-to-g9-receipt", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    result = execute(
        contract_path=args.contract.expanduser().resolve(),
        g4_to_g5_receipt=args.g4_to_g5_receipt.expanduser().resolve(),
        g8_to_g9_receipt=args.g8_to_g9_receipt.expanduser().resolve(),
        output_directory=args.output_directory.expanduser().resolve(),
    )
    print(json.dumps({"ok": result["ok"], "decision": result["decision"]}))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
