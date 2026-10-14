#!/usr/bin/env python3
"""Materialize and run covered WRC science requests on the full-stack route."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.campaigns.canonical import pretty_json  # noqa: E402
from engine.campaigns.compiler import (  # noqa: E402
    CampaignError,
    compile_campaign,
    load_campaign_spec,
    resolve_campaign_profile,
)
from engine.campaigns.wrc_science import (  # noqa: E402
    WRCScienceRequestError,
    materialize_full_stack_wrc_science_spec_v1,
)
from engine.campaigns.wrc_atlas_v3 import (  # noqa: E402
    WRCAtlasError,
    load_wrc_atlas_rows_v3,
    load_wrc_v1_candidate_requests_v3,
)
from engine.campaigns.wrc_variability import (  # noqa: E402
    build_wrc_parameter_variability_envelope_v1,
)


DEFAULT_BASE_SPEC = (
    ROOT
    / "_spec/campaigns/v1/"
    "wdl_jungle_eed_gv2_birth_w32p256c512_b48g200t160cuda_allocation_diagnostic.yaml"
)
DEFAULT_ATLAS_REGISTRY = ROOT / "_spec/registry/v1"


def _parse_axes(rows: Sequence[str]) -> dict[str, list[Any]]:
    axes: dict[str, list[Any]] = {}
    for row in rows:
        name, separator, raw_values = row.partition("=")
        if not separator or not name:
            raise WRCScienceRequestError(
                "PARAMETER_VALUE_OUT_OF_RANGE: --axis must be NAME=JSON_ARRAY"
            )
        if name in axes:
            raise WRCScienceRequestError(
                f"PARAMETER_VALUE_OUT_OF_RANGE: duplicate axis {name}"
            )
        try:
            values = json.loads(raw_values)
        except json.JSONDecodeError as exc:
            raise WRCScienceRequestError(
                f"PARAMETER_VALUE_OUT_OF_RANGE: {name} is not JSON"
            ) from exc
        if not isinstance(values, list):
            raise WRCScienceRequestError(
                f"PARAMETER_VALUE_OUT_OF_RANGE: {name} must be a JSON array"
            )
        axes[name] = values
    return axes


def _parse_seeds(rows: Sequence[str]) -> dict[str, list[int]]:
    parsed = _parse_axes(rows)
    return {str(owner): list(values) for owner, values in parsed.items()}


def _parse_fixed_axes(rows: Sequence[str]) -> dict[str, list[Any]]:
    fixed: dict[str, list[Any]] = {}
    for row in rows:
        name, separator, raw_value = row.partition("=")
        if not separator or not name or name in fixed:
            raise WRCScienceRequestError(
                "PARAMETER_VALUE_OUT_OF_RANGE: --fixed-axis must be unique NAME=JSON_VALUE"
            )
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError as exc:
            raise WRCScienceRequestError(
                f"PARAMETER_VALUE_OUT_OF_RANGE: {name} is not JSON"
            ) from exc
        if isinstance(value, (dict, list)):
            raise WRCScienceRequestError(
                f"PARAMETER_VALUE_OUT_OF_RANGE: {name} must be a JSON scalar"
            )
        fixed[name] = [value]
    return fixed


def _write_materialized_spec(value: dict[str, Any]) -> Path:
    output = value.get("output")
    if not isinstance(output, dict) or not isinstance(output.get("runtime_root"), str):
        raise WRCScienceRequestError("ROUTE_CAPABILITY_CHANGED: materialized output root")
    root = Path(output["runtime_root"]).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "wrc_science_request.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(pretty_json(value), encoding="utf-8")
    temporary.replace(destination)
    return destination


def _allocation_diagnostic_receipt_for_plan(
    materialized_path: Path, *, plan_sha256: str, cuda_extension_sha256: str
) -> Path | None:
    """Find the one exact allocation-only receipt for this materialized plan.

    Receipt selection is deliberately narrow: this operator entry never asks
    an operator to plumb a resource receipt, and the lower-level preflight
    remains the authority that validates every receipt field before promotion.
    """

    candidates: list[Path] = []
    request_root = materialized_path.parent.parent
    for path in sorted(
        request_root.glob(
            "*/wrc_science_allocation_diagnostic_*/allocation_diagnostic_receipt.json"
        )
    ):
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise WRCScienceRequestError(
                f"RESOURCE_NOT_ADMITTED: unreadable allocation diagnostic receipt: {path}"
            ) from exc
        if not isinstance(loaded, Mapping):
            raise WRCScienceRequestError(
                f"RESOURCE_NOT_ADMITTED: malformed allocation diagnostic receipt: {path}"
            )
        artifact = loaded.get("artifact")
        exact_plan = loaded.get("base_b48_plan_sha256") == plan_sha256
        same_artifact = (
            isinstance(artifact, Mapping)
            and artifact.get("sha256") == cuda_extension_sha256
        )
        if exact_plan or same_artifact:
            candidates.append(path)
    if len(candidates) > 1:
        raise WRCScienceRequestError(
            "RESOURCE_NOT_ADMITTED: ambiguous exact allocation diagnostic receipts"
        )
    return candidates[0] if candidates else None


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(pretty_json(value), encoding="utf-8")
    temporary.replace(path)


def _run_campaign_command(command: Sequence[str]) -> tuple[int, dict[str, Any]]:
    from tools.scripts.eiel_campaign import main as campaign_main

    captured = io.StringIO()
    with redirect_stdout(captured):
        status = campaign_main(command)
    try:
        payload = json.loads(captured.getvalue())
    except json.JSONDecodeError as exc:
        raise CampaignError("campaign command did not return JSON") from exc
    if not isinstance(payload, dict):
        raise CampaignError("campaign command did not return an object receipt")
    return status, payload


def _validate_atlas_selection(
    rows: Sequence[str], *, profile: Any, axes: Mapping[str, Sequence[Any]]
) -> tuple[str, ...]:
    requests = {
        request.row_id: request
        for request in load_wrc_v1_candidate_requests_v3(
            DEFAULT_ATLAS_REGISTRY, profile=profile
        )
    }
    unknown = sorted(set(rows) - set(requests))
    if unknown:
        raise WRCScienceRequestError(
            f"PARAMETER_NOT_IN_QUALIFIED_ENVELOPE: unknown Atlas rows {unknown!r}"
        )
    expected = {requests[row].parameter_name for row in rows}
    requested = set(axes)
    if requested != expected:
        raise WRCScienceRequestError(
            "PARAMETER_NOT_IN_QUALIFIED_ENVELOPE: Atlas rows and --axis parameters differ"
        )
    return tuple(sorted(set(rows)))


def _validate_fixed_axes(base: Mapping[str, Any], fixed_axes: Mapping[str, Sequence[Any]]) -> None:
    matrix = base.get("matrix")
    bindings = matrix.get("parameter_bindings") if isinstance(matrix, Mapping) else None
    existing = set(bindings.values()) if isinstance(bindings, Mapping) else set()
    unsupported = sorted(set(fixed_axes) - {str(value) for value in existing})
    if unsupported:
        raise WRCScienceRequestError(
            "PARAMETER_NOT_IN_QUALIFIED_ENVELOPE: fixed context controls must "
            f"already be base matrix dimensions: {unsupported!r}"
        )


def _classify_wrc_preflight(payload: dict[str, Any]) -> dict[str, Any]:
    """Expose the four science gates without rewriting provider evidence."""

    reasons = [
        row
        for field in ("reasons", "execution_authority_reasons")
        for row in payload.get(field, [])
        if isinstance(row, Mapping)
    ]
    resource_codes = {
        "insufficient_free_vram",
        "provider_vram_bytes_upper_bound_unavailable",
        "resource_admission_rejected",
    }
    resource_rejected = any(row.get("code") in resource_codes for row in reasons)
    resources = payload.get("resources")
    cuda = resources.get("cuda") if isinstance(resources, Mapping) else None
    resource_admitted = (
        isinstance(cuda, Mapping)
        and cuda.get("free_covers_admission_bound_and_headroom") is True
    )
    route = payload.get("execution_route")
    route_qualified = isinstance(route, Mapping) and route.get("authority_state") == "route_qualified"
    envelope_exact = isinstance(route, Mapping) and route.get("variability_envelope_exact") is True
    payload["wrc_science_gates"] = {
        "underlying_route": "QUALIFIED" if route_qualified else "NOT_QUALIFIED",
        "variability_capability": (
            "ENVELOPE_BOUND_PENDING_CAPABILITY_DECISION"
            if envelope_exact
            else "NOT_BOUND"
        ),
        "parameter_validation": "PASSED_AT_MATERIALIZATION",
        "dependency_activation": "PASSED_FOR_ALL_CELLS_AT_MATERIALIZATION",
        "resource_admission": (
            "RESOURCE_NOT_ADMITTED"
            if resource_rejected
            else "RESOURCE_ADMITTED"
            if resource_admitted
            else "NOT_EVALUATED"
        ),
    }
    if resource_rejected:
        payload["wrc_reason_codes"] = [
            {
                "code": "RESOURCE_NOT_ADMITTED",
                "path": "execution.resource_admission",
                "message": "Provider admission rejected this request; qualification is unchanged.",
            }
        ]
    return payload


def _validate_cell_activation(spec: Mapping[str, Any], plan: Any) -> None:
    """Apply every selected cell through the production configuration seam.

    This is a host-only activation check: it validates actual per-cell control
    application and dependency contracts without claiming a CUDA execution.
    """

    from engine.campaigns.patch_runner_factory import (
        PatchRunnerFactoryError,
        build_patch_wdl_jungle_eed_fixed_mlp_runner_v1,
    )

    artifacts = spec.get("artifacts")
    execution = spec.get("execution")
    if not isinstance(artifacts, Mapping) or not isinstance(execution, Mapping):
        raise WRCScienceRequestError("ROUTE_CAPABILITY_CHANGED: missing activation inputs")
    try:
        artifact_path = str(artifacts["cuda_extension"]["path"])
        qualification_path = str(artifacts["qualification_receipt"]["path"])
        reporting_path = str(artifacts["reporting_receipt"]["path"])
        if "total_ticks" in execution:
            if (
                execution.get("execution_route_identity")
                != "tick_native_execution_route_v1"
                or type(execution.get("total_ticks")) is not int
                or execution["total_ticks"] < 1
                or {"generation", "generations"} & set(execution)
            ):
                raise WRCScienceRequestError(
                    "ROUTE_CAPABILITY_CHANGED: malformed tick-native activation inputs"
                )
            # The runner factory is used here only to construct and validate
            # cell configuration. These probe values never enter the request,
            # route identity, schedule, provider, or execution authority.
            generations = 1
            ticks = 1
        else:
            generations = int(execution["generations"])
            ticks = int(execution["ticks_per_generation"])
    except (KeyError, TypeError, ValueError) as exc:
        raise WRCScienceRequestError("ROUTE_CAPABILITY_CHANGED: malformed activation inputs") from exc
    for cell in plan.cells:
        try:
            build_patch_wdl_jungle_eed_fixed_mlp_runner_v1(
                cell,
                Path("/tmp") / "eiel-wrc-science-activation" / cell.cell_id,
                generations=generations,
                ticks_per_generation=ticks,
                artifact_path=artifact_path,
                qualification_receipt_path=qualification_path,
                reporting_receipt_path=reporting_path,
                runner_constructor=lambda **kwargs: SimpleNamespace(**kwargs),
            )
        except (PatchRunnerFactoryError, ValueError) as exc:
            message = str(exc)
            code = (
                "PARAMETER_CONTEXT_INACTIVE"
                if "activation preflight" in message
                else "ROUTE_CAPABILITY_CHANGED"
            )
            raise WRCScienceRequestError(f"{code}: {message}") from exc


def _write_coverage_register(output: Path, *, profile: Any) -> int:
    """Write a raw, reviewable Atlas-to-envelope coverage projection."""

    requests = load_wrc_atlas_rows_v3(DEFAULT_ATLAS_REGISTRY)
    envelope = build_wrc_parameter_variability_envelope_v1(
        profile,
        world_base="patch",
        world_overlays=("eed_resource_impact", "jungle", "wdl_delayed_hazard"),
        requested_parameter_names={
            request.parameter_name
            for request in requests
            if request.launcher_profile == "STANDARD_CUDA"
        },
    )
    if envelope is None:
        raise WRCScienceRequestError("PARAMETER_NOT_IN_QUALIFIED_ENVELOPE: Atlas cohort")
    covered = {str(row["name"]): row for row in envelope.parameter_contracts}
    fields = (
        "ID",
        "parameter",
        "semantic_class",
        "type",
        "approved_contract",
        "per_cell_permission",
        "route_scope",
        "activation_contract",
        "envelope_coverage",
        "exclusion_reason",
        "capability_status",
        "scientific_evidence_status",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for request in requests:
            contract = covered.get(request.parameter_name)
            runtime_contract = profile.parameter_contracts.get(request.parameter_name)
            covered_standard = (
                request.launcher_profile == "STANDARD_CUDA" and contract is not None
            )
            if covered_standard:
                semantic_class = "EXPERIMENT_VARIABLE"
                capability_status = "ENVELOPE_BOUND_PENDING_CAPABILITY_DECISION"
                exclusion_reason = ""
            elif runtime_contract is not None and runtime_contract.qualification_affecting:
                semantic_class = "QUALIFICATION_AFFECTING"
                capability_status = "EXCLUDED_QUALIFICATION_AFFECTING"
                exclusion_reason = "control changes the execution capability"
            else:
                semantic_class = "PROVENANCE_ONLY"
                capability_status = f"EXCLUDED_{request.launcher_profile}"
                exclusion_reason = (
                    f"Atlas launcher profile {request.launcher_profile} is outside "
                    "the full-stack STANDARD_CUDA instrument"
                )
            contract_payload = contract if contract is not None else {}
            writer.writerow(
                {
                    "ID": request.row_id,
                    "parameter": request.parameter_name,
                    "semantic_class": semantic_class,
                    "type": contract_payload.get("type", request.fields.get("Type", "")),
                    "approved_contract": json.dumps(
                        contract_payload.get("value_contract", request.fields.get("Legal domain", "")),
                        sort_keys=True,
                    ),
                    "per_cell_permission": "YES" if contract_payload.get("per_cell") else "NO",
                    "route_scope": json.dumps(contract_payload.get("variability_route_scopes", [])),
                    "activation_contract": json.dumps(contract_payload.get("activation_contract", {}), sort_keys=True),
                    "envelope_coverage": "YES" if covered_standard else "NO",
                    "exclusion_reason": exclusion_reason,
                    "capability_status": capability_status,
                    "scientific_evidence_status": "NO_SCIENTIFIC_RESULT",
                }
            )
    return len(requests)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("coverage", "materialize", "validate", "plan", "preflight", "run"),
        help="materialize only, or delegate the selected lifecycle step to eiel_campaign",
    )
    parser.add_argument("--base-spec", type=Path, default=DEFAULT_BASE_SPEC)
    parser.add_argument(
        "--row",
        action="append",
        default=[],
        metavar="ATLAS_ID",
        help="canonical P011-P083 Atlas row; repeat for a multi-knob request",
    )
    parser.add_argument(
        "--axis",
        action="append",
        default=[],
        metavar="NAME=JSON_ARRAY",
        help="covered Atlas parameter and one or more approved values; repeat per knob",
    )
    parser.add_argument(
        "--seed",
        action="append",
        default=[],
        metavar="OWNER=JSON_ARRAY",
        help="ecology/task seed allocation for this science request; repeat per owner",
    )
    parser.add_argument(
        "--fixed-axis",
        action="append",
        default=[],
        metavar="NAME=JSON_VALUE",
        help="fix one existing base matrix context dimension for this design",
    )
    parser.add_argument(
        "--generations",
        type=int,
        help="legacy-route horizon only; tick-native bases use their frozen absolute total_ticks",
    )
    parser.add_argument(
        "--coverage-output",
        type=Path,
        help="required raw CSV destination when command is coverage",
    )
    parser.add_argument(
        "--prepared-owner-receipt",
        type=Path,
        action="append",
        default=[],
        help="provider-owned resource receipt, only when preflight requests promotion",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        base = load_campaign_spec(args.base_spec)
        profile = resolve_campaign_profile(base)
        if args.command == "coverage":
            if args.coverage_output is None:
                raise WRCScienceRequestError("PARAMETER_VALUE_OUT_OF_RANGE: --coverage-output")
            rows = _write_coverage_register(args.coverage_output, profile=profile)
            print(
                pretty_json(
                    {
                        "ok": True,
                        "schema_version": "eiel_wrc_variability_coverage_register_v1",
                        "coverage_output": str(args.coverage_output.resolve()),
                        "atlas_row_count": rows,
                    }
                ),
                end="",
            )
            return 0
        if not args.row or not args.axis:
            raise WRCScienceRequestError(
                "PARAMETER_NOT_IN_QUALIFIED_ENVELOPE: --row and --axis are required"
            )
        axes = _parse_axes(args.axis)
        atlas_rows = _validate_atlas_selection(args.row, profile=profile, axes=axes)
        fixed_axes = _parse_fixed_axes(args.fixed_axis)
        if set(axes) & set(fixed_axes):
            raise WRCScienceRequestError(
                "PARAMETER_VALUE_OUT_OF_RANGE: --axis and --fixed-axis overlap"
            )
        _validate_fixed_axes(base, fixed_axes)
        spec, envelope = materialize_full_stack_wrc_science_spec_v1(
            base,
            profile=profile,
            axes=axes,
            seeds=_parse_seeds(args.seed),
            fixed_context_axes=fixed_axes,
            generations=args.generations,
        )
        plan = compile_campaign(spec)
        _validate_cell_activation(spec, plan)
        materialized_path = _write_materialized_spec(spec)
        if args.command == "materialize":
            print(
                pretty_json(
                    {
                        "ok": True,
                        "schema_version": "eiel_wrc_science_materialization_receipt_v1",
                        "materialized_spec_path": str(materialized_path),
                        "campaign_id": plan.campaign_id,
                        "plan_sha256": plan.plan_sha256,
                        "qualification_key_sha256": plan.qualification_key_sha256,
                        "variability_envelope": envelope.payload(),
                        "atlas_row_ids": list(atlas_rows),
                        "fixed_context_parameters": sorted(fixed_axes),
                        "qualification_receipt_source": "base_spec.artifacts.qualification_receipt",
                        "dependency_activation": "PASSED_FOR_ALL_CELLS",
                    }
                ),
                end="",
            )
            return 0
        delegated = [args.command, str(materialized_path)]
        if args.command in {"preflight", "run"}:
            for receipt in args.prepared_owner_receipt:
                delegated.extend(["--prepared-owner-receipt", str(receipt)])
            allocation_receipt = _allocation_diagnostic_receipt_for_plan(
                materialized_path,
                plan_sha256=plan.plan_sha256,
                cuda_extension_sha256=str(spec["artifacts"]["cuda_extension"]["sha256"]),
            )
            if allocation_receipt is not None:
                delegated.extend(
                    ["--allocation-diagnostic-receipt", str(allocation_receipt)]
                )
        status, delegated_payload = _run_campaign_command(delegated)
        if args.command == "preflight":
            print(pretty_json(_classify_wrc_preflight(delegated_payload)), end="")
            return status
        if args.command != "run" or status != 0:
            print(pretty_json(delegated_payload), end="")
            return status
        runtime_root = Path(str(spec["output"]["runtime_root"])).resolve()
        analysis_status, analysis = _run_campaign_command(["analyze", str(runtime_root)])
        report_status, report = _run_campaign_command(["report", str(runtime_root)])
        receipt = {
            "ok": analysis_status == 0 and report_status == 0,
            "schema_version": "eiel_wrc_science_run_receipt_v1",
            "materialized_spec_path": str(materialized_path),
            "campaign": delegated_payload,
            "analysis": analysis,
            "report": report,
            "normal_evidence_outputs": {
                "analysis_path": analysis.get("analysis_path"),
                "report_path": report.get("report_path"),
            },
        }
        _write_json(runtime_root / "wrc_science_execution_receipt.json", receipt)
        print(pretty_json(receipt), end="")
        return 0 if receipt["ok"] else 1
    except (CampaignError, OSError, WRCAtlasError, WRCScienceRequestError, ValueError) as exc:
        print(
            pretty_json(
                {
                    "ok": False,
                    "schema_version": "eiel_wrc_science_failure_v1",
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            ),
            end="",
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
