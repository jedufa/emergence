"""Inspect current execution identity through a non-authoritative CRCA shadow."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .execution_identity_projection import (
    ShadowProjectionError,
    diff_shadow_projections_v1,
    explain_shadow_projection_diff_v1,
    inspect_shadow_projection_v1,
    project_from_input_v1,
    write_shadow_projection_v1,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eiel plan",
        description=(
            "Project, inspect, or compare layered CRCA shadow identity. "
            "These commands grant no execution or Qualification authority."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    project = commands.add_parser("project", help="write one validated shadow receipt")
    project.add_argument("input", type=Path, help="portable current-lock input JSON")
    project.add_argument("output", type=Path, help="new shadow receipt JSON")
    project.add_argument("--human", action="store_true")
    inspect = commands.add_parser(
        "inspect", help="validate and summarize a shadow receipt"
    )
    inspect.add_argument("receipt", type=Path)
    inspect.add_argument("--human", action="store_true")
    diff = commands.add_parser("diff", help="compare two validated shadow receipts")
    diff.add_argument("left", type=Path)
    diff.add_argument("right", type=Path)
    diff.add_argument("--human", action="store_true")
    explain = commands.add_parser("explain", help="explain a shadow identity decision")
    explain.add_argument("left", type=Path)
    explain.add_argument("right", type=Path)
    explain.add_argument("--human", action="store_true")
    return parser


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ShadowProjectionError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ShadowProjectionError(f"{path} must contain a JSON object")
    return dict(value)


def _emit(value: Mapping[str, Any], *, human: bool) -> None:
    if not human:
        print(json.dumps(value, indent=2, sort_keys=True))
        return
    schema = value.get("schema_version")
    if schema == "crca_shadow_projection_inspection_v1":
        print(f"{value['status']}: {value['projection_sha256']}")
        print(f"Run candidate: {value['run_id_candidate']}")
        print(f"Attempt candidate: {value['attempt_id_candidate']}")
        print("Authority: shadow only; no Run or Attempt was created.")
    elif schema == "crca_shadow_projection_diff_v1":
        print(f"Decision: {value['decision']}")
        layers = value.get("changed_layers", [])
        print("Changed layers: " + (", ".join(layers) if layers else "none"))
        print(
            "Authority: shadow only; execution and Qualification authority are false."
        )
    elif schema == "crca_shadow_projection_explanation_v1":
        for line in value["explanation"]:
            print(line)
    else:
        print(json.dumps(value, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "project":
            receipt = project_from_input_v1(_read_object(args.input))
            target = write_shadow_projection_v1(args.output, receipt)
            result = inspect_shadow_projection_v1(receipt)
            result["written_path"] = str(target)
        elif args.command == "inspect":
            result = inspect_shadow_projection_v1(_read_object(args.receipt))
        elif args.command == "diff":
            result = diff_shadow_projections_v1(
                _read_object(args.left), _read_object(args.right)
            )
        else:
            result = explain_shadow_projection_diff_v1(
                _read_object(args.left), _read_object(args.right)
            )
    except (ShadowProjectionError, OSError) as exc:
        print(f"eiel plan: {exc}", file=sys.stderr)
        return 2
    _emit(result, human=args.human)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
