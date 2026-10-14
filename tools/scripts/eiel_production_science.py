#!/usr/bin/env python3
"""Canonical science-shaped front door for active Qualified Runtime Releases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for _path in (_REPOSITORY_ROOT, _REPOSITORY_ROOT / "engine"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from engine.qrr.production_front_door_v1 import (  # noqa: E402
    PRODUCTION_SCIENCE_REQUEST_V2_SCHEMA_VERSION,
    ProductionScienceFrontDoorError,
    activate_qrr_v1,
    execute_science_request_v1,
    execute_science_request_v2,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eiel-production-science")
    subparsers = parser.add_subparsers(dest="command", required=True)
    activate = subparsers.add_parser("activate-qrr")
    activate.add_argument("--release-root", type=Path, required=True)
    activate.add_argument("--qrr-root", type=Path)
    for name in ("materialize", "check-representability", "preflight", "run"):
        command = subparsers.add_parser(name)
        command.add_argument("--request", type=Path, required=True)
        command.add_argument("--qrr-root", type=Path)
        command.add_argument("--run-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "activate-qrr":
            kwargs = {} if args.qrr_root is None else {"qrr_root": args.qrr_root}
            result = activate_qrr_v1(args.release_root, **kwargs)
        else:
            kwargs = {}
            if args.qrr_root is not None:
                kwargs["qrr_root"] = args.qrr_root
            if args.run_root is not None:
                kwargs["run_root"] = args.run_root
            raw_request = json.loads(args.request.read_text(encoding="utf-8"))
            execute = (
                execute_science_request_v2
                if isinstance(raw_request, dict)
                and raw_request.get("schema_version")
                == PRODUCTION_SCIENCE_REQUEST_V2_SCHEMA_VERSION
                else execute_science_request_v1
            )
            result = execute(args.command, args.request, **kwargs)
    except (ProductionScienceFrontDoorError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
