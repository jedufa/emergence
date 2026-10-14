#!/usr/bin/env python3
"""Compile or validate the planning-only Phase F v2 case registry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = ROOT / "engine"
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from evolution.phase_f_v2.canonical import canonical_bytes, canonical_sha256  # noqa: E402
from evolution.phase_f_v2.matrix import (  # noqa: E402
    CASE_COUNT,
    CHECKPOINT_CASE_COUNT,
    EXPECTED_ACCELERATED_INVOCATIONS,
    EXPECTED_CPU_ORACLE_EVALUATIONS,
    EXPECTED_TOTAL_REQUIRED_EVALUATIONS,
    PhaseFV2MatrixError,
    build_phase_f_v2_matrix,
    load_phase_f_v2_matrix,
)
from evolution.phase_f_v2.storage import atomic_write_canonical_json  # noqa: E402


CANONICAL_REGISTRY = (
    ROOT
    / "_spec"
    / "evolutionary_capacity"
    / "phase_f"
    / "v2"
    / "forced_combination_matrix.json"
)


class PhaseFV2RegistryCliError(ValueError):
    """The planning registry cannot be compiled or independently verified."""


def _expected_case_ids() -> list[str]:
    return [
        f"pf2_f{family:02d}_{profile}_r{role}"
        for family in range(1, 13)
        for profile in ("b12", "b16")
        for role in range(5)
    ]


def _expected_checkpoint_ids() -> set[str]:
    return {
        *(
            f"pf2_f12_{profile}_r{role}"
            for profile in ("b12", "b16")
            for role in range(5)
        ),
        *(f"pf2_f09_{profile}_r4" for profile in ("b12", "b16")),
        *(f"pf2_f11_{profile}_r4" for profile in ("b12", "b16")),
    }


def _verified_summary(
    registry: Mapping[str, Any], *, command: str, path: Path
) -> dict[str, Any]:
    """Verify core identity and accounting without relying on a receipt."""

    cases = registry.get("cases")
    if not isinstance(cases, list):
        raise PhaseFV2RegistryCliError("registry cases must be a list")
    case_ids = [
        case.get("case_id") if isinstance(case, Mapping) else None for case in cases
    ]
    expected_ids = _expected_case_ids()
    if case_ids != expected_ids:
        raise PhaseFV2RegistryCliError(
            "registry case IDs or ordering are not canonical"
        )
    if registry.get("case_count") != CASE_COUNT or len(cases) != CASE_COUNT:
        raise PhaseFV2RegistryCliError(
            "registry must contain exactly 120 logical cases"
        )

    checkpoint_ids = {
        case["case_id"]
        for case in cases
        if isinstance(case.get("checkpoint_schedule"), Mapping)
        and case["checkpoint_schedule"].get("kind") != "none"
    }
    if (
        registry.get("checkpoint_case_count") != CHECKPOINT_CASE_COUNT
        or checkpoint_ids != _expected_checkpoint_ids()
    ):
        raise PhaseFV2RegistryCliError("registry checkpoint set is not canonical")

    try:
        cpu_count = sum(
            case["execution_accounting"]["cpu_oracle_evaluations"] for case in cases
        )
        accelerated_count = sum(
            case["execution_accounting"]["accelerated_invocations"] for case in cases
        )
    except (KeyError, TypeError) as exc:
        raise PhaseFV2RegistryCliError("registry accounting is malformed") from exc
    if cpu_count != EXPECTED_CPU_ORACLE_EVALUATIONS:
        raise PhaseFV2RegistryCliError("CPU oracle evaluation count is not canonical")
    if accelerated_count != EXPECTED_ACCELERATED_INVOCATIONS:
        raise PhaseFV2RegistryCliError("accelerated invocation count is not canonical")
    if cpu_count + accelerated_count != EXPECTED_TOTAL_REQUIRED_EVALUATIONS:
        raise PhaseFV2RegistryCliError(
            "total required evaluation count is not canonical"
        )

    matrix_sha256 = registry.get("matrix_sha256")
    unsigned = dict(registry)
    unsigned.pop("matrix_sha256", None)
    if (
        not isinstance(matrix_sha256, str)
        or canonical_sha256(unsigned) != matrix_sha256
    ):
        raise PhaseFV2RegistryCliError("registry matrix digest is not canonical")

    return {
        "ok": True,
        "command": command,
        "path": str(path),
        "matrix_sha256": matrix_sha256,
        "case_count": len(cases),
        "checkpoint_case_count": len(checkpoint_ids),
        "cpu_oracle_evaluations": cpu_count,
        "accelerated_invocations": accelerated_count,
        "total_required_evaluations": cpu_count + accelerated_count,
        "first_case_id": case_ids[0],
        "last_case_id": case_ids[-1],
        "planning_only": True,
    }


def _compile(output: Path) -> dict[str, Any]:
    generated = build_phase_f_v2_matrix()
    tracked = load_phase_f_v2_matrix(CANONICAL_REGISTRY)
    canonical_payload = canonical_bytes(tracked) + b"\n"
    try:
        tracked_payload = CANONICAL_REGISTRY.read_bytes()
    except OSError as exc:
        raise PhaseFV2RegistryCliError(
            f"cannot read tracked canonical registry: {CANONICAL_REGISTRY}"
        ) from exc
    if tracked_payload != canonical_payload or generated != tracked:
        raise PhaseFV2RegistryCliError(
            "generated registry does not exactly reproduce the tracked canonical registry"
        )

    atomic_write_canonical_json(output, generated)
    try:
        written_payload = output.read_bytes()
    except OSError as exc:
        raise PhaseFV2RegistryCliError(
            f"cannot read compiled registry: {output}"
        ) from exc
    if written_payload != canonical_payload:
        raise PhaseFV2RegistryCliError("compiled registry bytes are not canonical")
    written = load_phase_f_v2_matrix(output)
    return _verified_summary(written, command="compile", path=output)


def _validate(path: Path) -> dict[str, Any]:
    registry = load_phase_f_v2_matrix(path)
    return _verified_summary(registry, command="validate", path=path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate a planning registry")
    validate.add_argument("registry", nargs="?", type=Path, default=CANONICAL_REGISTRY)

    compile_parser = subparsers.add_parser(
        "compile", help="compile the tracked planning registry"
    )
    compile_parser.add_argument("--output", type=Path, default=CANONICAL_REGISTRY)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "compile":
            summary = _compile(args.output)
        else:
            summary = _validate(args.registry)
        print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
        return 0
    except (OSError, PhaseFV2MatrixError, PhaseFV2RegistryCliError, ValueError) as exc:
        print(
            json.dumps(
                {"ok": False, "command": args.command, "error": str(exc)},
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
