#!/usr/bin/env python3
"""Validate or execute a versioned CPU-only EIEL validation plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.validation_plans import (  # noqa: E402
    ValidationPlanError,
    execute_plan,
    load_plan,
    write_receipt,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="validate without executing")
    validate.add_argument("plan", type=Path)
    execute = subparsers.add_parser("execute", help="validate and execute CPU-only steps")
    execute.add_argument("plan", type=Path)
    execute.add_argument("--root", type=Path, default=ROOT)
    execute.add_argument("--receipt", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        plan = load_plan(args.plan)
        if args.command == "validate":
            print(json.dumps({"ok": True, "plan_id": plan["plan_id"], "schema_version": plan["schema_version"]}, sort_keys=True))
            return 0
        receipt = execute_plan(plan, root=args.root.resolve())
        write_receipt(args.receipt, receipt)
        print(json.dumps({"ok": receipt["status"] == "passed", "receipt": str(args.receipt), "status": receipt["status"]}, sort_keys=True))
        return 0 if receipt["status"] == "passed" else 1
    except ValidationPlanError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
