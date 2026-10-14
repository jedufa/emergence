#!/usr/bin/env python3
"""Reconcile and publish an explicit supported-route state-context inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.authoritative_state.route_inventory import (  # noqa: E402
    derive_production_route_inventory_v1,
    publish_production_inventory_atomic_v1,
    publish_reconciliation_receipt_atomic_v1,
    reconcile_supported_route_inventory_v1,
)
from engine.authoritative_state.schema import content_sha256  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    derive = commands.add_parser(
        "derive",
        help="derive and atomically emit the canonical production inventory",
    )
    derive.add_argument("--source-root", type=Path, required=True)
    derive.add_argument("--output-inventory", type=Path, required=True)
    reconcile = commands.add_parser(
        "reconcile",
        help="validate an explicit inventory and publish its receipt",
    )
    reconcile.add_argument("--inventory", type=Path, required=True)
    reconcile.add_argument("--source-root", type=Path, required=True)
    reconcile.add_argument("--output-receipt", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "derive":
        inventory = derive_production_route_inventory_v1(
            source_root=args.source_root,
        )
        path = publish_production_inventory_atomic_v1(
            args.output_inventory,
            inventory,
        )
        print(
            json.dumps(
                {
                    "ok": True,
                    "inventory_path": str(path),
                    "inventory_content_sha256": content_sha256(inventory),
                    "context_count": len(inventory["contexts"]),
                },
                sort_keys=True,
            )
        )
        return 0

    receipt = reconcile_supported_route_inventory_v1(
        args.inventory,
        source_root=args.source_root,
    )
    path = publish_reconciliation_receipt_atomic_v1(
        args.output_receipt,
        receipt,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "receipt_path": str(path),
                "receipt_sha256": receipt["receipt_sha256"],
                "context_count": receipt["context_count"],
                "distinct_canonical_field_count": receipt[
                    "distinct_canonical_field_count"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
