"""CLI for the canonical EIEL execution request front door."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import importlib
import json
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

from engine.campaigns.execution_front_door_v1 import (
    ExecutionFrontDoorError,
    ExecutionRequestV2,
    execute_request_v1,
    execute_request_v2,
    load_execution_request_v1,
)


LaunchpadAdapterFactoryV1 = Callable[..., object]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.eiel",
        description=(
            "Run or preflight one machine-readable request through the canonical "
            "campaign compiler, preflight, provider, and lifecycle path."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, mode, help_text in (
        ("execute", "execute", "preflight and execute one request"),
        ("validate", "validate", "preflight or mechanically validate one request"),
    ):
        subparser = subparsers.add_parser(command, help=help_text)
        subparser.add_argument("request", type=Path, help="JSON execution request")
        subparser.add_argument(
            "--repo-root",
            type=Path,
            default=Path.cwd(),
            help="repository root used for source identity (default: cwd)",
        )
        subparser.add_argument(
            "--human",
            action="store_true",
            help="print a concise human summary instead of JSON",
        )
        subparser.add_argument(
            "--science-context",
            type=Path,
            help="content-addressed route context for a rich science V2 request",
        )
        science_authority = subparser.add_mutually_exclusive_group()
        science_authority.add_argument(
            "--qrr-root",
            type=Path,
            help="active-QRR registry root for rich science validation",
        )
        science_authority.add_argument(
            "--active-qrr-receipt",
            type=Path,
            help="exact read-only QRR receipt for rich science validation",
        )
        subparser.set_defaults(mode=mode)
    return parser


def _read_mapping(path: Path, *, name: str) -> dict[str, Any]:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ExecutionFrontDoorError(f"{name} must not be a symbolic link")
    resolved = expanded.resolve()
    if not resolved.is_file():
        raise ExecutionFrontDoorError(f"{name} is not a regular file: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExecutionFrontDoorError(f"{name} is unreadable: {resolved}") from error
    if not isinstance(value, Mapping):
        raise ExecutionFrontDoorError(f"{name} must be an object")
    return dict(value)


def _is_rich_science_request(value: Mapping[str, Any]) -> bool:
    """Recognize only the public rich discriminants at the CLI dispatch seam."""

    return bool({"science_campaign", "science_campaign_ref"} & set(value))


def _print_result(result: dict[str, object], *, human: bool) -> None:
    if not human:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    print(f"status: {result.get('status')}")
    print(f"result: {result.get('result_class')}")
    print(f"campaign: {result.get('campaign_id')}")
    print(f"run_root: {result.get('run_root')}")
    mechanical = result.get("mechanical_verification")
    if isinstance(mechanical, dict):
        print(f"mechanical: {mechanical.get('status')}")
    print(f"qualification: {result.get('qualification_status')}")
    baseline = result.get("baseline")
    if isinstance(baseline, dict):
        print(f"baseline: {baseline.get('overall_status')}")
        if baseline.get("overall_status") != "LEGACY_UNPROFILED":
            print(
                "baseline_dimensions: "
                f"configuration={baseline.get('configuration_status')}, "
                f"qualification={baseline.get('qualification_status')}, "
                f"operations={baseline.get('operational_status')}, "
                f"execution_admission={baseline.get('execution_admission_status')}"
            )
            deviations = baseline.get("deviations")
            if isinstance(deviations, list):
                for row in deviations:
                    if isinstance(row, dict):
                        print(
                            "baseline_deviation: "
                            f"{row.get('requirement_id')}: {row.get('reason')}"
                        )
    preflight = result.get("preflight")
    if isinstance(preflight, dict) and preflight.get("ready_for_execution") is not True:
        reasons = preflight.get("reasons")
        if isinstance(reasons, list):
            for row in reasons:
                if isinstance(row, dict) and row.get("code"):
                    print(f"preflight_reason: {row.get('code')}: {row.get('message')}")
    warnings = result.get("warnings")
    if isinstance(warnings, list):
        for warning in warnings:
            print(f"warning: {warning}")


def main(
    argv: Sequence[str] | None = None,
    *,
    cuda_full_evolution_v2_adapter_factory: LaunchpadAdapterFactoryV1 | None = None,
) -> int:
    parser = _parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        raw_request = _read_mapping(args.request, name="execution request")
        if _is_rich_science_request(raw_request):
            if args.mode != "validate":
                raise ExecutionFrontDoorError(
                    "SCIENCE_INSTRUMENT_EXECUTION_NOT_ENABLED: use eiel validate for "
                    "host preparation; route execution remains qualification-gated"
                )
            if args.science_context is None:
                raise ExecutionFrontDoorError(
                    "rich science V2 validation requires --science-context"
                )
            # The unified CLI is a legacy dispatcher.  The target composition
            # module owns rich-science dependencies, matching CORE_TOOLS routing.
            science_cli = importlib.import_module(
                "eiel_composition.science_instrument_cli"
            )
            result = science_cli.validate_rich_science_request_v2_from_files(
                raw_request,
                science_context=args.science_context,
                qrr_root=args.qrr_root,
                active_qrr_receipt=args.active_qrr_receipt,
            )
        else:
            if any(
                value is not None
                for value in (
                    args.science_context,
                    args.qrr_root,
                    args.active_qrr_receipt,
                )
            ):
                raise ExecutionFrontDoorError(
                    "science route options require a rich science V2 request"
                )
            request = load_execution_request_v1(args.request)
            if isinstance(request, ExecutionRequestV2):
                if cuda_full_evolution_v2_adapter_factory is None:
                    raise ExecutionFrontDoorError(
                        "ExecutionRequestV2 requires the canonical eiel_composition "
                        "execution CLI entrypoint"
                    )
                cuda_full_evolution_v2_adapter = (
                    cuda_full_evolution_v2_adapter_factory(repo_root=args.repo_root)
                )
                result = execute_request_v2(
                    request,
                    repo_root=args.repo_root,
                    mode=args.mode,
                    cuda_full_evolution_v2_crca_adapter=(
                        cuda_full_evolution_v2_adapter
                    ),
                    cuda_full_evolution_v2_checkpoint_occurrence_adapter=(
                        cuda_full_evolution_v2_adapter
                    ),
                )
            else:
                result = execute_request_v1(
                    request,
                    repo_root=args.repo_root,
                    mode=args.mode,
                )
    except (ExecutionFrontDoorError, OSError, TypeError, ValueError) as exc:
        error = {
            "schema_version": "eiel_execution_front_door_error_v1",
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        if args.human:
            print(f"error: {error['error']}", file=sys.stderr)
        else:
            print(json.dumps(error, indent=2, sort_keys=True))
        return 2
    _print_result(result, human=args.human)
    return 0 if result.get("ok") is True else 1


__all__ = ["main"]
