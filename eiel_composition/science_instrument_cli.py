"""Host-only operator surface for EIEL_SCIENCE_INSTRUMENT_V1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from engine.qrr.production_front_door_v1 import (
    DEFAULT_QRR_ROOT,
    resolve_active_qrr_v2,
)
from .science_instrument_artifacts_v1 import (
    ScienceInstrumentArtifactError,
    load_science_host_context_v1,
    serialize_science_host_context_v1,
)
from .science_instrument_host_front_door_v1 import (
    ScienceInstrumentHostPreparationError,
    prepare_rich_science_request_v2,
)
from .science_qrr_projection_v1 import ScienceQRRProjectionError
from .science_request_v2 import (
    ScienceRequestV2Error,
    normalize_rich_production_science_request_v2,
)


class _ActiveQRRResolver:
    def __init__(self, qrr_root: Path) -> None:
        self.qrr_root = qrr_root

    def resolve(self, flat_request: dict[str, Any]) -> dict[str, Any]:
        _, receipt = resolve_active_qrr_v2(flat_request, qrr_root=self.qrr_root)
        return receipt


class _ReceiptResolver:
    def __init__(self, receipt: dict[str, Any]) -> None:
        self.receipt = receipt

    def resolve(self, flat_request: dict[str, Any]) -> dict[str, Any]:
        return dict(self.receipt)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eiel science",
        description=(
            "Run one self-resolving EIEL Science Request V1. The run command "
            "validates and plans internally; validate and plan are optional "
            "diagnostic commands."
        ),
        epilog=(
            "Normal workflow: eiel science run REQUEST --human. Machine "
            "installation binds the current authority through "
            "EIEL_SCIENCE_EXECUTION_AUTHORITY_V1 or the canonical system/user "
            "authority path; request commands never accept context, Baseline, "
            "base-plan, materialization-root, or runtime-root arguments."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        (
            "run",
            "normal workflow: resolve, materialize, execute every batch, reduce, aggregate, and close out",
        ),
        (
            "validate",
            "optional diagnostic: validate and compile without runtime effects",
        ),
        (
            "plan",
            "optional diagnostic: resolve authorities and create an admitted executable materialization",
        ),
    ):
        public = subparsers.add_parser(command, help=help_text)
        public.add_argument("request", type=Path)
        if command == "run":
            public.add_argument(
                "--resume",
                "--resume-root",
                dest="resume_root",
                type=Path,
                help="resume a previously journaled campaign root",
            )
            public.add_argument(
                "--reducer-recovery-source-root",
                dest="reducer_recovery_source_root",
                type=Path,
                help=(
                    "recover compact reduction from one sealed completed child "
                    "without rerunning the simulation"
                ),
            )
            public.add_argument(
                "--compact-reducer-workers",
                type=int,
                default=4,
                help=(
                    "bounded native compact-reducer worker count; lower values "
                    "reduce recovery memory pressure"
                ),
            )
        public.add_argument("--human", action="store_true")

    activate = subparsers.add_parser(
        "activate-authority",
        help=(
            "machine administration: install one sealed current Science Instrument "
            "machine/base-plan authority"
        ),
    )
    activate.add_argument("base_campaign_plan", type=Path)
    activate.add_argument("--authority-id", required=True)
    activate.add_argument(
        "--append-to-set",
        action="store_true",
        help=(
            "append idempotently to the existing bounded authority-set schema "
            "instead of replacing the target"
        ),
    )
    activate.add_argument(
        "--output",
        type=Path,
        help="task-owned authority file; defaults to the canonical machine path",
    )
    activate.add_argument("--human", action="store_true")

    activate_set = subparsers.add_parser(
        "activate-authority-set",
        help=(
            "machine administration: install one bounded existing-schema "
            "Science Instrument authority set in a single write"
        ),
    )
    activate_set.add_argument(
        "--member",
        action="append",
        nargs=2,
        metavar=("AUTHORITY_ID", "BASE_CAMPAIGN_PLAN"),
        help="repeat once for each exact authority member",
    )
    activate_set.add_argument(
        "--preserve-authorities-from",
        action="append",
        type=Path,
        default=[],
        help="repeat for each sealed authority file whose members must be retained",
    )
    activate_set.add_argument("--output", type=Path, required=True)
    activate_set.add_argument("--human", action="store_true")

    normalize = subparsers.add_parser(
        "normalize",
        help="lower-level diagnostic: normalize one inline rich V2 request",
    )
    normalize.add_argument("request", type=Path)
    normalize.add_argument("--human", action="store_true")

    prepare = subparsers.add_parser(
        "prepare",
        help="lower-level diagnostic: compile, classify, lower, resolve, and admit host-side",
    )
    prepare.add_argument("request", type=Path)
    prepare.add_argument("--context", type=Path, required=True)
    authority = prepare.add_mutually_exclusive_group()
    authority.add_argument("--qrr-root", type=Path)
    authority.add_argument("--active-qrr-receipt", type=Path)
    prepare.add_argument("--human", action="store_true")

    demo = subparsers.add_parser(
        "demo",
        help=(
            "technical-only demonstration: compile/observe/checkpoint/"
            "publish/reduce pipeline"
        ),
    )
    demo.add_argument("--output", type=Path)
    demo.add_argument("--human", action="store_true")
    return parser


def _read_mapping(path: Path, *, name: str) -> dict[str, Any]:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ScienceInstrumentArtifactError(
            f"{name} must not be a symbolic link: {expanded}"
        )
    resolved = expanded.resolve()
    if not resolved.is_file():
        raise ScienceInstrumentArtifactError(
            f"{name} is not a regular file: {resolved}"
        )
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ScienceInstrumentArtifactError(
            f"{name} is unreadable: {resolved}"
        ) from error
    if not isinstance(value, dict):
        raise ScienceInstrumentArtifactError(f"{name} must be an object")
    return value


def _print(value: dict[str, object], *, human: bool) -> None:
    if not human:
        print(json.dumps(value, indent=2, sort_keys=True))
        return
    route_id = value.get("route_id")
    if route_id is None and isinstance(value.get("route_demonstration"), dict):
        routes = value["route_demonstration"]
        assert isinstance(routes, dict)
        route_a = routes.get("route_a")
        route_b = routes.get("route_b")
        if isinstance(route_a, dict) and isinstance(route_b, dict):
            route_id = f"{route_a.get('route_id')} <-> {route_b.get('route_id')}"
    effects = value.get("effects_performed")
    if effects is None:
        effects = value.get("runtime_effects_performed")
    print(f"status: {value.get('status')}")
    if value.get("campaign_id") is not None:
        print(f"campaign_id: {value.get('campaign_id')}")
    if value.get("primary_outcome") is not None:
        print(f"outcome: {value.get('primary_outcome')}")
    print(f"request_sha256: {value.get('request_sha256')}")
    if route_id is not None:
        print(f"route_id: {route_id}")
    materialization = value.get("materialization")
    if isinstance(materialization, dict):
        print(f"materialization_root: {materialization.get('root')}")
        print(f"batch_count: {materialization.get('batch_count')}")
    if value.get("runtime_root") is not None:
        print(f"runtime_root: {value.get('runtime_root')}")
    if value.get("report_path") is not None:
        print(f"report: {value.get('report_path')}")
    if value.get("cleanup_reason") is not None:
        print(f"cleanup pending: {value.get('cleanup_reason')}")
    if effects is not None:
        print(f"effects_performed: {effects}")
    if value.get("execution_started") is not None:
        print(f"execution_started: {value.get('execution_started')}")
    if value.get("error") is not None:
        print(f"error: {value.get('error')}")


def validate_rich_science_request_v2_from_files(
    request: dict[str, Any],
    *,
    science_context: Path,
    qrr_root: Path | None = None,
    active_qrr_receipt: Path | None = None,
) -> dict[str, object]:
    """Prepare one rich request for the canonical validation gateway."""

    if qrr_root is not None and active_qrr_receipt is not None:
        raise ScienceInstrumentArtifactError("supply at most one QRR authority source")
    context = load_science_host_context_v1(science_context)
    resolver = (
        _ReceiptResolver(
            _read_mapping(
                active_qrr_receipt,
                name="active QRR resolution receipt",
            )
        )
        if active_qrr_receipt is not None
        else _ActiveQRRResolver(qrr_root or DEFAULT_QRR_ROOT)
    )
    prepared = prepare_rich_science_request_v2(
        request,
        context,
        active_qrr_resolver=resolver,
    )
    admitted = prepared.admission_receipt is not None
    context_artifact = serialize_science_host_context_v1(context)
    return {
        "schema_version": "eiel_science_instrument_validation_result_v1",
        "ok": admitted,
        "status": "ADMITTED_HOST_READY" if admitted else "SEMANTIC_REJECTION",
        "result_class": (
            "SCIENCE_HOST_PREPARATION" if admitted else "SCIENCE_SEMANTIC_REJECTION"
        ),
        "campaign_id": prepared.normalized.campaign.campaign_id,
        "run_root": None,
        "qualification_status": "TECHNICAL_ONLY_HOST_PREPARATION",
        "validation_only": True,
        "effects_performed": False,
        "route_id": context.route_binding.route_id,
        "science_host_context_sha256": context_artifact["context_sha256"],
        "science_host_preparation": prepared.to_dict(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "activate-authority-set":
            from .science_request_v1_front_door import (
                activate_science_execution_authority_set_v1,
            )

            result = activate_science_execution_authority_set_v1(
                tuple(
                    (authority_id, Path(base_plan))
                    for authority_id, base_plan in (args.member or [])
                ),
                output_path=args.output,
                preserve_from=tuple(args.preserve_authorities_from),
            )
            _print(result, human=args.human)
            return 0
        if args.command == "activate-authority":
            from .science_request_v1_front_door import (
                activate_science_execution_authority_v1,
            )

            options: dict[str, Any] = {
                "authority_id": args.authority_id,
                "append_to_set": args.append_to_set,
            }
            if args.output is not None:
                options["output_path"] = args.output
            result = activate_science_execution_authority_v1(
                args.base_campaign_plan,
                **options,
            )
            _print(result, human=args.human)
            return 0
        if args.command in {"validate", "plan", "run"}:
            from .science_request_v1_front_door import (
                plan_science_request_v1,
                run_science_request_v1,
                validate_science_request_v1,
            )

            action = {
                "validate": validate_science_request_v1,
                "plan": plan_science_request_v1,
                "run": run_science_request_v1,
            }[args.command]
            if args.command == "run":
                from .science_run_report import run_science_with_report_v1

                # Reject invalid requests before allocating an owned output root.
                validation = validate_science_request_v1(args.request)
                if validation.get("ok") is not True:
                    _print(validation, human=args.human)
                    return 2
                result = run_science_with_report_v1(
                    args.request,
                    executor=action,
                    repo_root=Path(__file__).resolve().parents[1],
                    resume_root=args.resume_root,
                    reducer_recovery_source_root=args.reducer_recovery_source_root,
                    compact_reducer_workers=args.compact_reducer_workers,
                )
            else:
                result = action(args.request)
            _print(result, human=args.human)
            return 0 if result.get("ok") is True else 1
        if args.command == "demo":
            from .science_instrument_demo_v1 import (
                build_science_functional_demonstration_v1,
            )

            result = build_science_functional_demonstration_v1()
            if args.output is not None:
                expanded = args.output.expanduser()
                if expanded.exists() and expanded.is_symlink():
                    raise ScienceInstrumentArtifactError(
                        f"demonstration output must not be a symbolic link: {expanded}"
                    )
                expanded.parent.mkdir(parents=True, exist_ok=True)
                expanded.write_text(
                    json.dumps(result, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            _print(result, human=args.human)
            return 0 if result.get("ok") is True else 1
        request = _read_mapping(args.request, name="science request")
        if args.command == "normalize":
            normalized = normalize_rich_production_science_request_v2(request)
            result: dict[str, object] = {
                "schema_version": "eiel_science_normalization_result_v1",
                "ok": True,
                "status": "NORMALIZED",
                "primary_outcome": None,
                "request_sha256": normalized.normalized_request["request_sha256"],
                "route_id": None,
                "normalized_request": normalized.normalized_request,
                "campaign_resolution_receipt": (normalized.content_resolution_receipt),
                "effects_performed": False,
            }
        else:
            context = load_science_host_context_v1(args.context)
            resolver = None
            if args.active_qrr_receipt is not None:
                resolver = _ReceiptResolver(
                    _read_mapping(
                        args.active_qrr_receipt,
                        name="active QRR resolution receipt",
                    )
                )
            elif args.qrr_root is not None:
                resolver = _ActiveQRRResolver(args.qrr_root)
            else:
                resolver = _ActiveQRRResolver(DEFAULT_QRR_ROOT)
            prepared = prepare_rich_science_request_v2(
                request, context, active_qrr_resolver=resolver
            )
            result = {
                **prepared.to_dict(),
                "ok": prepared.admission_receipt is not None,
                "status": (
                    "ADMITTED_HOST_READY"
                    if prepared.admission_receipt is not None
                    else "SEMANTIC_REJECTION"
                ),
                "primary_outcome": prepared.classification.primary_outcome.value,
                "request_sha256": prepared.normalized.normalized_request[
                    "request_sha256"
                ],
                "route_id": context.route_binding.route_id,
            }
    except Exception as error:
        from .science_request_v1_front_door import ScienceRequestV1FrontDoorError

        if isinstance(error, ScienceRequestV1FrontDoorError):
            result = error.to_result()
            _print(result, human=args.human)
            return 2
        if not isinstance(
            error,
            (
                OSError,
                TypeError,
                ValueError,
                ScienceInstrumentArtifactError,
                ScienceInstrumentHostPreparationError,
                ScienceQRRProjectionError,
                ScienceRequestV2Error,
            ),
        ):
            raise
        result = {
            "schema_version": "eiel_science_operator_error_v1",
            "ok": False,
            "status": "ERROR",
            "primary_outcome": None,
            "request_sha256": None,
            "route_id": None,
            "effects_performed": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        _print(result, human=args.human)
        return 2
    _print(result, human=args.human)
    return 0 if result.get("ok") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ("main", "validate_rich_science_request_v2_from_files")
