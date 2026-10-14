"""Small operational surface for the fail-closed WRC Atlas resume ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from engine.campaigns.wrc_atlas_resume import AtlasResumeError, scan_campaign
from engine.wrc.reducer.canonical import pretty_json


def _object(path: str, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AtlasResumeError(f"cannot read {label}") from exc
    if not isinstance(value, Mapping):
        raise AtlasResumeError(f"{label} must be an object")
    return value


def _scan(args: argparse.Namespace) -> int:
    try:
        result = scan_campaign(args.root, write_status=not args.no_write_status)
    except AtlasResumeError as exc:
        print(pretty_json({"ok": False, "error": str(exc)}), end="")
        return 2
    print(pretty_json({"ok": result["counts"]["invalid"] == 0, **result}), end="")
    return 0 if result["counts"]["invalid"] == 0 else 1


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Inspect a WRC Atlas resume ledger; this command never launches worlds.")
    sub = result.add_subparsers(dest="command", required=True)
    scan = sub.add_parser("scan", help="validate planned worlds and emit deterministic status")
    scan.add_argument("root")
    scan.add_argument("--no-write-status", action="store_true")
    scan.set_defaults(func=_scan)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
