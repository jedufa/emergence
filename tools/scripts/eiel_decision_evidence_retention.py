#!/usr/bin/env python3
"""Generate and release decision-bearing evidence audit holds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.campaigns.canonical import pretty_json  # noqa: E402
from engine.campaigns.decision_evidence_retention import (  # noqa: E402
    build_audit_hold_manifest,
    release_audit_hold,
    retention_schema,
    write_audit_hold_manifest,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON document must be an object: {path}")
    return value


def _parse_generations(values: Sequence[str]) -> list[int]:
    generations: list[int] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                generations.append(int(part))
    return generations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("schema")
    generate = subparsers.add_parser("generate")
    generate.add_argument("--execution-plan", type=Path, required=True)
    generate.add_argument("--runtime-root", type=Path)
    generate.add_argument("--decision-predicate", required=True)
    generate.add_argument("--required-generation", action="append", required=True)
    generate.add_argument("--out", type=Path, required=True)
    release = subparsers.add_parser("release")
    release.add_argument("--manifest", type=Path, required=True)
    release.add_argument("--audit-decision", type=Path, required=True)
    release.add_argument("--compact-archive-sha256", required=True)
    release.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "schema":
        result = retention_schema()
    elif args.command == "generate":
        plan = _read_json(args.execution_plan)
        runtime_root = (
            args.runtime_root
            if args.runtime_root is not None
            else Path(str(plan["output"]["runtime_root"]))
        )
        route_bindings = {
            "resolved_profile": plan.get("resolved_profile"),
            "qualification_key_sha256": plan.get("qualification_key_sha256"),
            "execution_route_sha256": (
                plan.get("execution_route_contract", {}) or {}
            ).get("execution_route_sha256"),
        }
        result = build_audit_hold_manifest(
            campaign_id=str(plan["campaign_id"]),
            runtime_root=runtime_root,
            decision_predicate=str(args.decision_predicate),
            required_generations=_parse_generations(args.required_generation),
            plan_sha256=str(plan.get("plan_sha256") or ""),
            route_bindings=route_bindings,
        )
        write_audit_hold_manifest(args.out, result)
        result = {**result, "manifest_path": str(args.out.expanduser().resolve())}
    else:
        manifest = _read_json(args.manifest)
        audit_decision = _read_json(args.audit_decision)
        result = release_audit_hold(
            manifest=manifest,
            audit_decision=audit_decision,
            compact_archive_sha256=str(args.compact_archive_sha256),
        )
        args.out.expanduser().resolve().write_text(pretty_json(result), encoding="utf-8")
        result = {**result, "release_receipt_path": str(args.out.expanduser().resolve())}
    print(pretty_json(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
