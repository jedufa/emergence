#!/usr/bin/env python3
"""Validate, plan, preflight, execute, resume, analyze, and report EIEL campaigns."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = ROOT / "engine"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if ENGINE_ROOT.is_dir() and str(ENGINE_ROOT) not in sys.path:
    # Campaign subprocesses run with a deliberately clean environment.  The
    # simulator also retains historical top-level imports (for example,
    # ``evolution``), so make the in-tree engine importable without relying on
    # an ambient PYTHONPATH.
    sys.path.insert(0, str(ENGINE_ROOT))

from engine.campaigns.canonical import canonical_sha256, pretty_json  # noqa: E402
from engine.campaigns.compiler import (  # noqa: E402
    CampaignError,
    compile_campaign,
    load_campaign_spec,
    resolve_campaign_profile,
)
from engine.campaigns.providers import default_execution_provider_registry  # noqa: E402
from engine.campaigns.qualification import (  # noqa: E402
    ACS_FOCUSED_ROUTE_OBLIGATIONS_V1,
)


def _compile_registered_campaign(spec: Mapping[str, Any]):
    from engine.campaigns.tensor_world_c1_provider import (
        is_tensor_world_c1_spec_v1,
        tensor_world_c1_profile_registry_v1,
    )

    if is_tensor_world_c1_spec_v1(spec):
        return compile_campaign(
            spec,
            registry=tensor_world_c1_profile_registry_v1(),
        )
    return compile_campaign(spec)


def _registered_providers_for_plan(plan: Mapping[str, Any]):
    from engine.campaigns.tensor_world_c1_provider import (
        TENSOR_WORLD_C1_PROFILE_ID,
        tensor_world_c1_provider_registry_v1,
    )

    if plan.get("resolved_profile") == TENSOR_WORLD_C1_PROFILE_ID:
        return tensor_world_c1_provider_registry_v1()
    return default_execution_provider_registry()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _read_object(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CampaignError(f"cannot read {name}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CampaignError(f"{name} must be a JSON object: {path}")
    return value


def _runtime_root(plan: Mapping[str, Any]) -> Path:
    raw = plan.get("output")
    if not isinstance(raw, Mapping) or not isinstance(raw.get("runtime_root"), str):
        raise CampaignError("compiled plan lacks output.runtime_root")
    root = Path(str(raw["runtime_root"])).expanduser()
    if not root.is_absolute():
        raise CampaignError("compiled runtime root must be absolute")
    return root.resolve()


def _write_plan(plan: Mapping[str, Any]) -> Path:
    root = _runtime_root(plan)
    path = root / "execution_plan.json"
    if path.exists():
        previous = _read_object(path, name="existing execution plan")
        if previous.get("plan_sha256") != plan.get("plan_sha256"):
            raise CampaignError(f"run root contains a different execution plan: {path}")
    _atomic_text(path, pretty_json(plan))
    return path


def _selected_batch_measurement_plan(
    plan: Mapping[str, Any], selected: str
) -> dict[str, Any]:
    """Return a measurement-only view bound to one exact physical batch."""

    batches = plan.get("physical_batches")
    if not isinstance(batches, list) or not batches:
        raise CampaignError("compiled plan has no physical batch")
    matches = [
        row
        for row in batches
        if isinstance(row, Mapping) and row.get("batch_id") == selected
    ]
    if not matches:
        raise CampaignError(f"unknown physical batch: {selected!r}")
    if len(matches) != 1:
        raise CampaignError(f"duplicate physical batch: {selected!r}")
    return {**dict(plan), "physical_batches": [dict(matches[0])]}


def _execution_geometry_receipt(
    plan: Mapping[str, Any], *, backend: str
) -> dict[str, Any]:
    """Expose planned geometry without overstating provider admission.

    Validation has no execution provider or inspected device.  It therefore
    reports the requested CUDA/CPU device class and preserves unavailable
    route and VRAM admission values rather than manufacturing an execution
    claim from the planning estimate.
    """

    batches = plan.get("physical_batches")
    if not isinstance(batches, list):
        raise CampaignError("compiled plan has no physical batch mapping")
    estimate = plan.get("resource_estimate")
    if not isinstance(estimate, Mapping):
        raise CampaignError("compiled plan has no resource estimate")
    execution = plan.get("execution")
    if not isinstance(execution, Mapping):
        raise CampaignError("compiled plan has no execution payload")
    realized_sizes = [
        len(row.get("cell_ids", []))
        for row in batches
        if isinstance(row, Mapping)
        and isinstance(row.get("cell_ids"), Sequence)
        and not isinstance(row.get("cell_ids"), (str, bytes))
    ]
    if len(realized_sizes) != len(batches) or not realized_sizes:
        raise CampaignError("compiled plan has an invalid physical batch mapping")
    physical_batch_size = estimate.get("physical_batch_cells")
    if type(physical_batch_size) is not int or physical_batch_size <= 0:
        raise CampaignError(
            "compiled plan has an invalid requested physical batch size"
        )
    return {
        "logical_cells": len(plan.get("cells", [])),
        "physical_batch_count": len(batches),
        "physical_batch_size": physical_batch_size,
        "realized_physical_batch_sizes": realized_sizes,
        "resident": "physical_window_generations" in execution,
        "backend": backend,
        "device": "cuda" if backend == "cuda_resident" else "cpu",
        "route": "unresolved_until_provider_preflight",
        "estimated_vram": estimate.get("vram_bytes_projection"),
        # This command has not run provider preflight against an inspected
        # device, so even an authoritative planning upper bound is not an
        # admission.  Keep the planned estimate above but fail closed on the
        # explicitly admission-named field.
        "admitted_vram": None,
    }


def _validate(spec_path: Path) -> dict[str, Any]:
    spec = load_campaign_spec(spec_path)
    plan = compile_campaign(spec)
    profile = resolve_campaign_profile(spec)
    payload = plan.payload()
    return {
        "ok": True,
        "schema_version": "eiel_campaign_validation_receipt_v1",
        "campaign_id": plan.campaign_id,
        "logical_cells": len(plan.cells),
        "physical_batches": len(plan.batches),
        "execution_geometry": _execution_geometry_receipt(
            payload, backend=profile.backend
        ),
        "resolved_profile": profile.public_receipt(),
        "qualification_key_sha256": plan.qualification_key_sha256,
        "plan_sha256": plan.plan_sha256,
        "scope": "schema, capability resolution, typed matrix, qualification identity, and planning bounds only",
    }


def _current_source_commit() -> str:
    """Resolve the checked-out source identity for an evidence receipt."""

    result = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    commit = result.stdout.strip().lower()
    if (
        result.returncode != 0
        or len(commit) != 40
        or any(char not in "0123456789abcdef" for char in commit)
    ):
        raise CampaignError("cannot resolve the current source commit for WRC audit")
    return commit


def _wrc_resident_route_audit(spec_path: Path, *, output: Path) -> dict[str, Any]:
    """Write a planning-only current-source WRC resident-route audit.

    This intentionally records unavailable CUDA artifact and admission
    identities rather than turning a batch projection into an execution claim.
    """

    base_spec = load_campaign_spec(spec_path)
    provider = default_execution_provider_registry().providers()[0]
    matrix: list[dict[str, Any]] = []
    for batch_cells in (6, 8, 24, 48):
        spec = json.loads(json.dumps(base_spec))
        spec["execution"]["batch"] = {"mode": "explicit", "cells": batch_cells}
        plan = compile_campaign(spec)
        request = provider.translate_batch(plan, "batch-0000")
        estimate = provider.estimate_resources(plan, [request])
        if estimate.get("vram_bytes_upper_bound") != "unavailable":
            raise CampaignError("WRC audit expected unaudited wider resident batch")
        matrix.append(
            {
                "requested_physical_batch_cells": batch_cells,
                "realized_physical_batch_cells": len(request.cell_ids),
                "estimated_vram_bytes": estimate.get("vram_bytes_projection"),
                "admitted_vram_bytes": None,
                "admission_status": "NOT_ADMITTED",
                "reason": estimate.get("vram_bytes_unavailable_reason"),
            }
        )
    profile = resolve_campaign_profile(base_spec)
    audit = {
        "ok": True,
        "schema_version": "eiel_wrc_resident_route_audit_v1",
        "source_commit": _current_source_commit(),
        "active_route": {
            "provider_id": provider.provider_id,
            "profile_id": profile.profile_id,
            "backend": profile.backend,
            "device": "cuda" if profile.backend == "cuda_resident" else "cpu",
            "route_status": "UNRESOLVED_UNTIL_PROVIDER_PREFLIGHT",
        },
        "artifact_identity": {
            "status": "UNAVAILABLE",
            "value": None,
            "reason": "no current-source inspected CUDA artifact is bound by this planning audit",
        },
        "receipt_identities": {
            "provider_preflight": "UNAVAILABLE",
            "allocation_audit": "UNAVAILABLE",
            "reason": "B6-B48 retain only planning projections until exact allocation audit",
        },
        "prerequisite_disposition": {
            "obsolete_work_id": "WORK-8C72BA26261B",
            "obsolete_work_status": "OBSOLETE",
            "obsolete_work_reason": "Retire stale continuation without claiming qualification.",
            "acceptance_inherited": False,
            "current_dependency_work_id": "WORK-35F15969BC7A",
            "current_dependency_status": "COMPLETED_ACCEPTED",
        },
        "batch_admission_matrix": matrix,
        "limitations": [
            "This is a planning audit, not CUDA execution or qualification evidence.",
            "B6-B48 are not admitted until an exact current-artifact allocation audit succeeds.",
        ],
    }
    _atomic_text(output, pretty_json(audit))
    return {**audit, "audit_path": str(output)}


def _plan(spec_path: Path) -> tuple[dict[str, Any], Path]:
    plan = compile_campaign(load_campaign_spec(spec_path)).payload()
    path = _write_plan(plan)
    return plan, path


def _qualify_command(spec_path: Path, route: Mapping[str, Any]) -> str:
    """Return a full qualifier unless exact partial evidence can be extended."""

    command = [
        sys.executable,
        "-m",
        "tools.scripts.eiel_campaign",
        "qualify",
        str(spec_path.expanduser().resolve()),
    ]
    required = route.get("required_obligations")
    covered = route.get("covered_obligations")
    missing = route.get("missing_obligations")
    reasons = route.get("reasons")
    binding_reason_codes = (
        {
            str(row.get("code"))
            for row in reasons
            if isinstance(row, Mapping) and row.get("code") is not None
        }
        if isinstance(reasons, list)
        else set()
    )
    binding_mismatch_codes = {
        "execution_route_qualification_schema_mismatch",
        "execution_route_hash_mismatch",
        "execution_route_provider_mismatch",
        "execution_route_profile_mismatch",
    }
    arrays_are_strings = all(
        isinstance(values, list)
        and all(isinstance(value, str) and value for value in values)
        for values in (required, covered, missing)
    )
    if arrays_are_strings:
        required_set = set(required)
        covered_set = set(covered)
        missing_set = set(missing)
        carryable_covered = covered_set - set(ACS_FOCUSED_ROUTE_OBLIGATIONS_V1)
        same_route_partial = (
            route.get("route_hash_exact") is True
            and route.get("provider_binding_exact") is True
            and route.get("profile_binding_exact") is True
            and route.get("artifact_conformance_exact") is True
            and route.get("variability_envelope_exact") is True
            and not (binding_reason_codes & binding_mismatch_codes)
            and bool(carryable_covered)
            and bool(missing_set)
            and not (carryable_covered & missing_set)
            and carryable_covered | missing_set == required_set
        )
        if same_route_partial:
            for obligation in sorted(missing_set):
                command.extend(["--missing-obligation", obligation])
    return shlex.join(command)


def _preflight(
    spec_path: Path,
    *,
    prepared_owner_receipt_path: Path | None = None,
    prepared_owner_receipt_paths: Sequence[Path] | None = None,
    allocation_diagnostic_receipt_path: Path | None = None,
    provider_registry: Any | None = None,
) -> dict[str, Any]:
    spec = load_campaign_spec(spec_path)
    compiled = compile_campaign(spec)
    plan = compiled.payload()
    plan_path = _write_plan(plan)
    from engine.campaigns.preflight import preflight_campaign

    profile = resolve_campaign_profile(spec)
    receipt_paths = list(prepared_owner_receipt_paths or ())
    if prepared_owner_receipt_path is not None:
        receipt_paths.append(prepared_owner_receipt_path)
    prepared_owner_receipts = [
        _read_object(
            path.expanduser().resolve(),
            name="prepared-owner resource audit receipt",
        )
        for path in receipt_paths
    ]
    prepared_owner_receipt = (
        prepared_owner_receipts[0] if len(prepared_owner_receipts) == 1 else None
    )
    allocation_diagnostic_receipt = (
        None
        if allocation_diagnostic_receipt_path is None
        else _read_object(
            allocation_diagnostic_receipt_path.expanduser().resolve(),
            name="allocation diagnostic resource receipt",
        )
    )
    receipt = preflight_campaign(
        plan,
        profile=profile,
        provider_registry=provider_registry,
        prepared_owner_resource_receipt=prepared_owner_receipt,
        allocation_diagnostic_resource_receipt=allocation_diagnostic_receipt,
        enforce_storage_admission=True,
    )
    if len(prepared_owner_receipts) > 1:
        _promote_multi_prepared_owner_receipts(
            receipt,
            plan=plan,
            profile=profile,
            provider_registry=provider_registry,
            prepared_owner_receipts=prepared_owner_receipts,
        )
    route = receipt.get("execution_route")
    if (
        isinstance(route, dict)
        and route.get("evaluated") is True
        and route.get("authority_state") != "route_qualified"
    ):
        route["qualify_command"] = _qualify_command(spec_path, route)
    receipt["execution_plan_path"] = str(plan_path)
    prepared_owner = receipt.get("prepared_owner_resource_audit")
    if (
        isinstance(prepared_owner, dict)
        and receipt.get("ready_for_prepared_owner_measurement") is True
    ):
        batch_ids = prepared_owner.get("batch_ids")
        if isinstance(batch_ids, list) and batch_ids:
            prepared_owner["measure_command"] = shlex.join(
                [
                    sys.executable,
                    "-m",
                    "tools.scripts.eiel_campaign",
                    "measure-prepared-owner",
                    str(spec_path.expanduser().resolve()),
                    "--batch-id",
                    str(batch_ids[0]),
                ]
            )
    path = _runtime_root(plan) / "preflight_receipt.json"
    _atomic_text(path, pretty_json(receipt))
    receipt["preflight_receipt_path"] = str(path)
    return receipt


def _promote_multi_prepared_owner_receipts(
    receipt: dict[str, Any],
    *,
    plan: Mapping[str, Any],
    profile: Any,
    provider_registry: Any | None,
    prepared_owner_receipts: Sequence[Mapping[str, Any]],
) -> None:
    from engine.campaigns.preflight import preflight_campaign

    prepared_owner = receipt.get("prepared_owner_resource_audit")
    if not isinstance(prepared_owner, dict):
        return
    batch_ids = prepared_owner.get("batch_ids")
    if not isinstance(batch_ids, list) or not batch_ids:
        return
    if len(prepared_owner_receipts) != len(batch_ids):
        prepared_owner["multi_receipt_promotion"] = {
            "schema_version": "eiel_prepared_owner_multi_receipt_promotion_v1",
            "promoted": False,
            "reason": "receipt_count_does_not_match_physical_batch_count",
            "receipt_count": len(prepared_owner_receipts),
            "batch_count": len(batch_ids),
        }
        return
    by_request_sha: dict[str, Mapping[str, Any]] = {}
    duplicate_request_shas: set[str] = set()
    for receipt_row in prepared_owner_receipts:
        request = receipt_row.get("request")
        request_sha = (
            request.get("batch_request_sha256")
            if isinstance(request, Mapping)
            else None
        )
        if not isinstance(request_sha, str) or not request_sha:
            prepared_owner["multi_receipt_promotion"] = {
                "schema_version": "eiel_prepared_owner_multi_receipt_promotion_v1",
                "promoted": False,
                "reason": "receipt_missing_batch_request_sha256",
            }
            return
        if request_sha in by_request_sha:
            duplicate_request_shas.add(request_sha)
        by_request_sha[request_sha] = receipt_row
    if duplicate_request_shas:
        prepared_owner["multi_receipt_promotion"] = {
            "schema_version": "eiel_prepared_owner_multi_receipt_promotion_v1",
            "promoted": False,
            "reason": "duplicate_receipt_batch_request_sha256",
            "duplicate_batch_request_sha256s": sorted(duplicate_request_shas),
        }
        return

    promotions: list[dict[str, Any]] = []
    promoted_vram_bound = 0
    unmatched_batch_ids: list[str] = []
    for batch_id in sorted(str(value) for value in batch_ids):
        narrowed = _selected_batch_measurement_plan(plan, batch_id)
        narrowed_receipt = preflight_campaign(
            narrowed,
            profile=profile,
            provider_registry=provider_registry,
        )
        authorization = narrowed_receipt.get("prepared_owner_resource_audit", {}).get(
            "measurement_authorization"
        )
        request_bindings = (
            authorization.get("request_bindings")
            if isinstance(authorization, Mapping)
            else None
        )
        expected_request_sha = (
            request_bindings.get("batch_request_sha256")
            if isinstance(request_bindings, Mapping)
            else None
        )
        if not isinstance(expected_request_sha, str):
            unmatched_batch_ids.append(batch_id)
            continue
        receipt_row = by_request_sha.get(expected_request_sha)
        if receipt_row is None:
            unmatched_batch_ids.append(batch_id)
            continue
        promoted = preflight_campaign(
            narrowed,
            profile=profile,
            provider_registry=provider_registry,
            prepared_owner_resource_receipt=receipt_row,
        )
        promotion = promoted.get("prepared_owner_resource_audit", {}).get(
            "receipt_promotion"
        )
        if promoted.get("ready_for_execution") is not True or not isinstance(
            promotion, Mapping
        ):
            unmatched_batch_ids.append(batch_id)
            continue
        vram_bound = promotion.get("vram_bytes_upper_bound")
        if type(vram_bound) is not int:
            unmatched_batch_ids.append(batch_id)
            continue
        promoted_vram_bound = max(promoted_vram_bound, int(vram_bound))
        promotions.append(
            {
                "batch_id": batch_id,
                "receipt_sha256": receipt_row.get("receipt_sha256"),
                "batch_request_sha256": expected_request_sha,
                "vram_bytes_upper_bound": vram_bound,
            }
        )

    if unmatched_batch_ids or len(promotions) != len(batch_ids):
        prepared_owner["multi_receipt_promotion"] = {
            "schema_version": "eiel_prepared_owner_multi_receipt_promotion_v1",
            "promoted": False,
            "reason": "not_all_physical_batches_promoted",
            "unmatched_batch_ids": unmatched_batch_ids,
            "promoted_batches": promotions,
        }
        return

    execution_reasons = [
        row
        for row in receipt.get("execution_authority_reasons", [])
        if not (
            isinstance(row, Mapping)
            and row.get("code") == "provider_vram_bytes_upper_bound_unavailable"
        )
    ]
    receipt["execution_authority_reasons"] = execution_reasons
    receipt["ready_for_execution"] = bool(
        receipt.get("ok") is True and not execution_reasons
    )
    resources = receipt.get("resources")
    if isinstance(resources, dict):
        resources["vram_bytes_upper_bound"] = promoted_vram_bound
        resources["provider_estimate_within_budget"] = True
    prepared_owner["multi_receipt_promotion"] = {
        "schema_version": "eiel_prepared_owner_multi_receipt_promotion_v1",
        "promoted": True,
        "physical_batch_count": len(batch_ids),
        "promoted_vram_bytes_upper_bound": promoted_vram_bound,
        "batch_promotions": promotions,
    }


def _measure_prepared_owner(
    spec_path: Path,
    *,
    batch_id: str | None,
    output_root: Path | None,
    provider_registry: Any | None = None,
) -> dict[str, Any]:
    """Prepare and measure one exact physical owner without executing it."""

    from engine.campaigns.preflight import preflight_campaign
    from engine.campaigns.providers import default_execution_provider_registry

    resolved = spec_path.expanduser().resolve()
    spec = load_campaign_spec(resolved)
    plan = compile_campaign(spec).payload()
    _write_plan(plan)
    batches = plan.get("physical_batches")
    if not isinstance(batches, list) or not batches:
        raise CampaignError("compiled plan has no physical batch")
    known = [
        str(row["batch_id"])
        for row in batches
        if isinstance(row, Mapping) and isinstance(row.get("batch_id"), str)
    ]
    selected = batch_id or (known[0] if len(known) == 1 else None)
    if selected is None:
        raise CampaignError(
            "--batch-id is required when the plan has multiple physical batches"
        )
    if selected not in known:
        raise CampaignError(f"unknown physical batch: {selected!r}")
    measurement_plan = _selected_batch_measurement_plan(plan, selected)
    providers = provider_registry or default_execution_provider_registry()
    profile = resolve_campaign_profile(spec)
    preflight = preflight_campaign(
        measurement_plan,
        profile=profile,
        provider_registry=providers,
    )
    if preflight.get("ready_for_prepared_owner_measurement") is not True:
        reasons = [
            *preflight.get("reasons", []),
            *preflight.get("prepared_owner_resource_audit", {}).get(
                "measurement_authority_reasons", []
            ),
        ]
        raise CampaignError(
            "prepared-owner measurement is not authorized for the exact plan: "
            + "; ".join(
                str(row.get("message", row)) if isinstance(row, Mapping) else str(row)
                for row in reasons
            )
        )
    provider = providers.get(str(measurement_plan.get("resolved_profile", "")))
    measure = getattr(provider, "measure_prepared_owner_resources_v1", None)
    if provider is None or not callable(measure):
        raise CampaignError(
            "selected provider does not support prepared-owner resource measurement"
        )
    prepared_owner = preflight.get("prepared_owner_resource_audit")
    authorization = (
        prepared_owner.get("measurement_authorization")
        if isinstance(prepared_owner, Mapping)
        else None
    )
    if not isinstance(authorization, Mapping):
        raise CampaignError(
            "prepared-owner preflight did not produce measurement authorization"
        )
    bindings = authorization.get("request_bindings")
    if (
        not isinstance(bindings, Mapping)
        or bindings.get("batch_request_sha256") is None
    ):
        raise CampaignError("prepared-owner measurement authorization is incomplete")
    request = provider.translate_batch(measurement_plan, selected)
    audit_id = f"{selected}-{str(plan['plan_sha256'])[:12]}"
    root = (
        output_root.expanduser().resolve()
        if output_root is not None
        else _runtime_root(plan) / "prepared_owner_resource_audits" / audit_id
    )
    result = measure(
        measurement_plan,
        request,
        audit_root=root,
        audit_id=audit_id,
        measurement_authorization=authorization,
    )
    if not isinstance(result, Mapping):
        raise CampaignError("prepared-owner measurement returned no typed receipt")
    return {
        **dict(result),
        "ok": bool(
            isinstance(result.get("decision"), Mapping)
            and result["decision"].get("ok") is True
        ),
        "command": "measure-prepared-owner",
        "spec_path": str(resolved),
    }


def _route_contract_for_plan(
    plan: Mapping[str, Any], *, provider_registry: Any | None = None
) -> tuple[dict[str, Any], Any]:
    providers = provider_registry or _registered_providers_for_plan(plan)
    provider = providers.get(str(plan.get("resolved_profile", "")))
    if provider is None:
        raise CampaignError(
            f"profile {plan.get('resolved_profile')!r} has no execution provider"
        )
    embedded = plan.get("execution_route_contract")
    if isinstance(embedded, Mapping):
        contract = dict(embedded)
    else:
        compiler = getattr(provider, "execution_route_contract", None)
        if not callable(compiler):
            raise CampaignError(
                f"provider {provider.provider_id!r} does not expose an execution route contract"
            )
        raw = compiler(plan)
        if isinstance(raw, Mapping):
            contract = dict(raw)
        elif callable(getattr(raw, "payload", None)):
            payload = raw.payload()
            if not isinstance(payload, Mapping):
                raise CampaignError(
                    "provider execution route contract payload must be an object"
                )
            contract = dict(payload)
            route_hash = getattr(raw, "execution_route_sha256", None)
            if isinstance(route_hash, str):
                contract["execution_route_sha256"] = route_hash
        else:
            raise CampaignError("provider execution route contract must be an object")

    # Canary shape is configuration, not route identity. Supply execution-plan
    # defaults only when the provider did not state its qualified constraints.
    requirements = contract.get("canary_requirements")
    requirements = dict(requirements) if isinstance(requirements, Mapping) else {}
    execution = plan.get("execution")
    if isinstance(execution, Mapping):
        if (
            execution.get("execution_route_identity")
            == "tick_native_execution_route_v1"
        ):
            control = execution.get("tick_native_control")
            control = dict(control) if isinstance(control, Mapping) else {}
            requirements.update(
                {
                    "time_authority": "absolute_tick_v1",
                    "total_ticks": execution.get("total_ticks"),
                    "checkpoint_ticks": list(execution.get("checkpoint_ticks", [])),
                    "resident_owner_span_ticks": control.get(
                        "resident_owner_span_ticks", 1_120
                    ),
                    "native_dispatch_window_ticks": control.get(
                        "native_dispatch_window_ticks", 1_120
                    ),
                    "event_window_capacity": control.get(
                        "event_window_capacity", 65_536
                    ),
                }
            )
        else:
            lifecycle = execution.get("internal_lifecycle_schedule")
            internal_interval = (
                lifecycle.get("interval_ticks")
                if isinstance(lifecycle, Mapping)
                else None
            )
            requirements.setdefault(
                "qualified_ticks_per_generation",
                execution.get("ticks_per_generation", internal_interval),
            )
            requirements.setdefault(
                "boundary_generations",
                max(2, int(getattr(provider, "resident_segment_generation_min", 2))),
            )
            requirements.setdefault(
                "resident_segment_generation_min",
                max(1, int(getattr(provider, "resident_segment_generation_min", 1))),
            )
    qualification_key = plan.get("qualification_key")
    if isinstance(qualification_key, Mapping):
        admitted = qualification_key.get("admitted_batch_range")
        if (
            "admitted_physical_batch_sizes" not in requirements
            and isinstance(admitted, (list, tuple))
            and len(admitted) == 2
            and all(type(item) is int and item > 0 for item in admitted)
        ):
            requirements["admitted_physical_batch_sizes"] = list(
                range(int(admitted[0]), int(admitted[1]) + 1)
            )
    contract["canary_requirements"] = requirements
    return contract, provider


def _qualify(
    spec_path: Path,
    *,
    plan_only: bool,
    missing_obligations: Sequence[str] = (),
    output: Path | None = None,
    run_root: Path | None = None,
    tick_native_execution_path: Path | None = None,
    inherit_route_qualification_path: Path | None = None,
    provider_registry: Any | None = None,
) -> dict[str, Any]:
    """Plan or explicitly execute one provider-owned route canary."""

    from engine.qualification_plans.v1 import (
        generate_execution_route_canary_plan,
    )

    resolved_spec = spec_path.expanduser().resolve()
    plan = _compile_registered_campaign(load_campaign_spec(resolved_spec)).payload()
    if tick_native_execution_path is not None:
        from engine.campaigns.tick_native import TickNativeExecutionRequestV1
        from engine.qrr.tick_native_materialization_v1 import (
            canonical_tick_native_plan_v1,
        )

        reference = _read_object(
            tick_native_execution_path.expanduser().resolve(),
            name="tick-native qualification execution",
        )
        execution = dict(plan.get("execution") or {})
        request_execution = {**execution, **reference}
        if (
            "total_ticks" not in reference
            and reference.get("absolute_ticks") is not None
        ):
            request_execution["total_ticks"] = reference.get("absolute_ticks")
        control = dict(execution.get("tick_native_control") or {})
        reference_control = reference.get("tick_native_control")
        if reference_control is not None and not isinstance(reference_control, Mapping):
            raise CampaignError("tick-native qualification control must be an object")
        if isinstance(reference_control, Mapping):
            control.update(reference_control)
        for name in (
            "resident_owner_span_ticks",
            "native_dispatch_window_ticks",
            "refill_chunk_ticks",
            "reducer_every_ticks",
            "archive_every_ticks",
            "event_window_capacity",
            "source_window_ticks",
            "legacy_turnover_interval_ticks",
        ):
            if name in reference:
                control[name] = reference[name]
        request_execution["tick_native_control"] = control
        tick_request = TickNativeExecutionRequestV1.from_execution(request_execution)
        plan = canonical_tick_native_plan_v1(plan, tick_request=tick_request)
    if inherit_route_qualification_path is not None:
        inheritance = _read_object(
            inherit_route_qualification_path.expanduser().resolve(),
            name="route qualification inheritance",
        )
        plan = {**plan, "route_qualification_inheritance": inheritance}
        plan.pop("plan_sha256", None)
        plan["plan_sha256"] = canonical_sha256(plan)
    contract, provider = _route_contract_for_plan(
        plan, provider_registry=provider_registry
    )
    canary = generate_execution_route_canary_plan(
        contract,
        missing_obligations=(
            list(missing_obligations) if missing_obligations else None
        ),
    )
    route_hash = str(canary["execution_route_sha256"])
    qualification_root = (
        run_root.expanduser().resolve()
        if run_root is not None
        else (_runtime_root(plan) / "route_qualification" / route_hash).resolve()
    )
    plan_path = (
        output.expanduser().resolve()
        if output is not None
        else qualification_root / "qualification_plan.json"
    )
    _atomic_text(plan_path, pretty_json(canary))
    command = [
        sys.executable,
        "-m",
        "tools.scripts.eiel_campaign",
        "qualify",
        str(resolved_spec),
    ]
    for obligation in sorted(set(missing_obligations)):
        command.extend(["--missing-obligation", obligation])
    if tick_native_execution_path is not None:
        command.extend(
            ["--tick-native-execution", str(tick_native_execution_path.resolve())]
        )
    if inherit_route_qualification_path is not None:
        command.extend(
            [
                "--inherit-route-qualification",
                str(inherit_route_qualification_path.resolve()),
            ]
        )
    if output is not None:
        command.extend(["--output", str(plan_path)])
    if run_root is not None:
        command.extend(["--run-root", str(qualification_root)])

    common = {
        "schema_version": "eiel_execution_route_qualify_receipt_v1",
        "campaign_id": plan["campaign_id"],
        "execution_route_sha256": route_hash,
        "missing_obligations": canary["missing_obligations"],
        "shape": canary["shape"],
        "qualification_plan_path": str(plan_path),
        "qualification_run_root": str(qualification_root),
        "qualify_command": shlex.join(command),
    }
    if plan_only:
        return {
            **common,
            "ok": True,
            "status": "plan_generated",
            "qualification_succeeded": False,
            "execution_started": False,
        }

    executor = getattr(provider, "execute_route_qualification", None)
    if not callable(executor):
        raise CampaignError(
            f"provider {provider.provider_id!r} cannot execute route qualification; "
            f"the deterministic plan is at {plan_path}"
        )
    result = executor(
        campaign_plan=plan,
        canary=canary,
        run_root=qualification_root,
    )
    if (
        not isinstance(result, Mapping)
        or result.get("qualification_succeeded") is not True
    ):
        raise CampaignError(
            "route canary did not produce successful qualification evidence"
        )
    return {
        **common,
        **dict(result),
        "ok": True,
        "status": "route_qualified",
        "qualification_succeeded": True,
        "execution_started": True,
        "missing_obligations": [],
    }


def _materialize_long_horizon(
    uninterrupted_run_root: Path,
    restored_run_root: Path,
    *,
    output: Path,
) -> dict[str, Any]:
    from engine.campaigns.long_horizon_evidence_materialization import (
        materialize_long_horizon_evidence_v1,
    )

    evidence = materialize_long_horizon_evidence_v1(
        uninterrupted_run_root=uninterrupted_run_root,
        restored_run_root=restored_run_root,
        output_path=output,
    )
    return {
        "ok": True,
        "schema_version": "eiel_long_horizon_materialization_receipt_v1",
        "evidence_path": str(output.expanduser().resolve()),
        "evidence_sha256": evidence["evidence_sha256"],
        "ticks_committed": evidence["ticks_committed"],
        "owner_boundaries_crossed": evidence["owner_boundaries_crossed"],
    }


def _materialize_physical_b136_acs_parity(
    sync_run_root: Path,
    sync_restored_run_root: Path,
    async_run_root: Path,
    async_restored_run_root: Path,
    *,
    output: Path,
    acs2_gpu_telemetry_receipt: Path,
) -> dict[str, Any]:
    from engine.campaigns.physical_b136_acs_parity_v1 import (
        materialize_physical_b136_acs_parity_v1,
    )

    evidence = materialize_physical_b136_acs_parity_v1(
        sync_run_root,
        sync_restored_run_root,
        async_run_root,
        async_restored_run_root,
        output,
        acs2_gpu_telemetry_receipt,
    )
    return {
        "ok": True,
        "schema_version": "eiel_physical_b136_acs_parity_materialization_receipt_v1",
        "evidence_path": str(output.expanduser().resolve()),
        "evidence_sha256": evidence["receipt_sha256"],
        "status": evidence["status"],
        "gate": evidence["gate"],
    }


def _frozen_long_horizon_plan(
    execution_plan_path: Path,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Load the exact Launchpad-bound plan named by paired-run evidence."""

    supplied_plan = execution_plan_path.expanduser()
    if supplied_plan.is_symlink() or not supplied_plan.is_file():
        raise CampaignError(
            f"frozen execution plan must be an existing regular file: {supplied_plan}"
        )
    resolved_plan = supplied_plan.resolve()
    plan = _read_object(resolved_plan, name="frozen execution plan")
    unsigned = dict(plan)
    supplied_plan_sha256 = unsigned.pop("plan_sha256", None)
    if supplied_plan_sha256 != canonical_sha256(unsigned):
        raise CampaignError("frozen execution plan has an invalid plan_sha256")

    bindings = evidence.get("paired_run_bindings")
    uninterrupted = (
        bindings.get("uninterrupted") if isinstance(bindings, Mapping) else None
    )
    run_root = (
        uninterrupted.get("run_root")
        if isinstance(uninterrupted, Mapping)
        else None
    )
    bound_plan_sha256 = (
        uninterrupted.get("plan_sha256")
        if isinstance(uninterrupted, Mapping)
        else None
    )
    if not isinstance(run_root, str) or not Path(run_root).is_absolute():
        raise CampaignError(
            "long-horizon evidence lacks an absolute uninterrupted run root"
        )
    expected_plan_path = Path(run_root).resolve() / "execution_plan.json"
    if resolved_plan != expected_plan_path:
        raise CampaignError(
            "frozen execution plan is not the uninterrupted plan bound by evidence"
        )
    if bound_plan_sha256 != supplied_plan_sha256:
        raise CampaignError(
            "frozen execution plan differs from the uninterrupted evidence binding"
        )
    output = plan.get("output")
    if (
        not isinstance(output, Mapping)
        or output.get("runtime_root") != str(Path(run_root).resolve())
    ):
        raise CampaignError(
            "frozen execution plan runtime root differs from the evidence binding"
        )
    return plan


def _promote_long_horizon(
    spec_path: Path,
    evidence_path: Path,
    *,
    frozen_execution_plan_path: Path | None = None,
) -> dict[str, Any]:
    evidence = _read_object(
        evidence_path.expanduser().resolve(), name="long-horizon evidence"
    )
    if frozen_execution_plan_path is None:
        resolved_spec = spec_path.expanduser().resolve()
        plan = _compile_registered_campaign(load_campaign_spec(resolved_spec)).payload()
        plan_source = {
            "mode": "compiled_spec_v1",
            "path": str(resolved_spec),
        }
    else:
        plan = _frozen_long_horizon_plan(frozen_execution_plan_path, evidence)
        plan_source = {
            "mode": "launchpad_frozen_execution_plan_v1",
            "path": str(frozen_execution_plan_path.expanduser().resolve()),
        }
    _contract, provider = _route_contract_for_plan(plan)
    promote = getattr(provider, "promote_long_horizon_qualification", None)
    if not callable(promote):
        raise CampaignError(
            f"provider {provider.provider_id!r} cannot promote long-horizon evidence"
        )
    result = promote(campaign_plan=plan, long_horizon_evidence=evidence)
    if (
        not isinstance(result, Mapping)
        or result.get("qualification_succeeded") is not True
    ):
        raise CampaignError("long-horizon evidence promotion did not succeed")
    return {
        **dict(result),
        "ok": True,
        "schema_version": "eiel_long_horizon_promotion_receipt_v1",
        "status": "long_horizon_promoted",
        "plan_source": plan_source,
    }


def _promote_science_route(
    spec_path: Path,
    evidence_path: Path,
    attestation_path: Path,
    *,
    frozen_execution_plan_path: Path | None = None,
) -> dict[str, Any]:
    evidence = _read_object(
        evidence_path.expanduser().resolve(), name="long-horizon evidence"
    )
    attestation = _read_object(
        attestation_path.expanduser().resolve(),
        name="canonical full production route attestation",
    )
    if frozen_execution_plan_path is None:
        resolved_spec = spec_path.expanduser().resolve()
        plan = _compile_registered_campaign(load_campaign_spec(resolved_spec)).payload()
        plan_source = {
            "mode": "compiled_spec_v1",
            "path": str(resolved_spec),
        }
    else:
        plan = _frozen_long_horizon_plan(frozen_execution_plan_path, evidence)
        plan_source = {
            "mode": "launchpad_frozen_execution_plan_v1",
            "path": str(frozen_execution_plan_path.expanduser().resolve()),
        }
    _contract, provider = _route_contract_for_plan(plan)
    promote = getattr(provider, "promote_science_route_qualification", None)
    if not callable(promote):
        raise CampaignError(
            f"provider {provider.provider_id!r} cannot promote science-route evidence"
        )
    result = promote(
        campaign_plan=plan,
        long_horizon_evidence=evidence,
        canonical_full_production_route_attestation=attestation,
    )
    if (
        not isinstance(result, Mapping)
        or result.get("qualification_succeeded") is not True
    ):
        raise CampaignError("science-route evidence promotion did not succeed")
    return {
        **dict(result),
        "ok": True,
        "schema_version": "eiel_canonical_full_production_route_promotion_receipt_v1",
        "status": "science_route_promoted",
        "plan_source": plan_source,
    }


def _run(
    spec_path: Path,
    *,
    provider_registry: Any | None = None,
    prepared_owner_receipt_path: Path | None = None,
    prepared_owner_receipt_paths: Sequence[Path] | None = None,
    allocation_diagnostic_receipt_path: Path | None = None,
) -> dict[str, Any]:
    receipt = (
        _preflight(spec_path)
        if (
            prepared_owner_receipt_path is None
            and not prepared_owner_receipt_paths
            and allocation_diagnostic_receipt_path is None
        )
        else _preflight(
            spec_path,
            prepared_owner_receipt_path=prepared_owner_receipt_path,
            prepared_owner_receipt_paths=prepared_owner_receipt_paths,
            allocation_diagnostic_receipt_path=allocation_diagnostic_receipt_path,
            provider_registry=provider_registry,
        )
    )
    if receipt.get("ready_for_execution") is not True:
        reasons = [
            *receipt.get("reasons", []),
            *receipt.get("execution_authority_reasons", []),
        ]
        raise CampaignError(
            "campaign is not execution-ready: "
            + "; ".join(str(item) for item in reasons)
        )
    from engine.campaigns.orchestration import execute_campaign
    from engine.campaigns.providers import default_execution_provider_registry

    root = Path(receipt["execution_plan_path"]).parent
    plan = _read_object(root / "execution_plan.json", name="execution plan")
    providers = provider_registry or default_execution_provider_registry()
    return execute_campaign(plan, run_root=root, provider_registry=providers)


def _resume(
    run_root: Path,
    *,
    provider_registry: Any | None = None,
    prepared_owner_receipt_path: Path | None = None,
    cuda_memory_detector: Any | None = None,
    cuda_hardware_detector: Any | None = None,
) -> dict[str, Any]:
    from engine.campaigns.providers import default_execution_provider_registry

    root = run_root.expanduser().resolve()
    plan = _read_object(root / "execution_plan.json", name="execution plan")
    preflight_path = root / "preflight_receipt.json"
    preflight = _read_object(preflight_path, name="preflight receipt")
    prepared_owner = preflight.get("prepared_owner_resource_audit")
    resources = preflight.get("resources")
    provider_estimate = (
        resources.get("provider_estimate") if isinstance(resources, Mapping) else None
    )
    promoted_path = (
        bool(
            isinstance(prepared_owner, Mapping)
            and isinstance(prepared_owner.get("receipt_promotion"), Mapping)
        )
        or bool(
            isinstance(resources, Mapping)
            and type(resources.get("vram_bytes_upper_bound")) is not int
        )
        or bool(
            isinstance(provider_estimate, Mapping)
            and provider_estimate.get("vram_bytes_unavailable_reason")
        )
    )
    if promoted_path:
        if prepared_owner_receipt_path is None:
            raise CampaignError(
                "resume requires an explicit prepared-owner resource receipt"
            )
        from engine.campaigns.preflight import preflight_campaign

        providers = provider_registry or default_execution_provider_registry()
        receipt = _read_object(
            prepared_owner_receipt_path.expanduser().resolve(),
            name="prepared-owner resource audit receipt",
        )
        kwargs: dict[str, Any] = {
            "provider_registry": providers,
            "prepared_owner_resource_receipt": receipt,
        }
        if cuda_memory_detector is not None:
            kwargs["cuda_memory_detector"] = cuda_memory_detector
        if cuda_hardware_detector is not None:
            kwargs["cuda_hardware_detector"] = cuda_hardware_detector
        preflight = preflight_campaign(plan, **kwargs)
        _atomic_text(preflight_path, pretty_json(preflight))
    if preflight.get("ready_for_execution") is not True:
        raise CampaignError("run root is not execution-ready")
    if preflight.get("plan_sha256") != plan.get("plan_sha256"):
        raise CampaignError("preflight receipt does not bind the execution plan")
    from engine.campaigns.orchestration import execute_campaign

    providers = provider_registry or default_execution_provider_registry()
    return execute_campaign(plan, run_root=root, provider_registry=providers)


def _measure_acs_restore(
    run_root: Path,
    *,
    output: Path,
    batch_id: str | None = None,
    provider_registry: Any | None = None,
) -> dict[str, Any]:
    """Measure one strict terminal-checkpoint reload without redispatching ticks.

    ACS4 needs an attributable B48 restore measurement, not merely a unit-test
    assertion that the provider exposes ``resume_batch``.  A completed terminal
    checkpoint takes the provider's strict reload/verification path and returns
    without a new native dispatch, so this is safe to run as a CPU-side
    continuation check after a guarded execution window has closed.
    """

    root = run_root.expanduser().resolve()
    receipt_path = output.expanduser().resolve()
    if receipt_path.exists():
        raise CampaignError(f"ACS restore receipt already exists: {receipt_path}")
    plan = _read_object(root / "execution_plan.json", name="execution plan")
    batches = plan.get("physical_batches")
    if not isinstance(batches, list):
        raise CampaignError("execution plan has no physical batches")
    candidates = [
        dict(row)
        for row in batches
        if isinstance(row, Mapping)
        and (batch_id is None or row.get("batch_id") == batch_id)
    ]
    if len(candidates) != 1:
        selector = batch_id if batch_id is not None else "the sole batch"
        raise CampaignError(
            f"ACS restore requires exactly one selected batch: {selector}"
        )
    batch = candidates[0]
    selected_batch_id = batch.get("batch_id")
    if not isinstance(selected_batch_id, str) or not selected_batch_id:
        raise CampaignError("selected batch lacks a batch_id")
    manifest_path = root / "batches" / selected_batch_id / "manifest.json"
    manifest = _read_object(manifest_path, name="completed source batch manifest")
    if manifest.get("status") != "completed":
        raise CampaignError("ACS restore source batch must be completed")
    checkpoint = manifest.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise CampaignError("ACS restore source batch lacks a durable checkpoint")
    provider_registry = provider_registry or default_execution_provider_registry()
    provider = provider_registry.get(plan.get("resolved_profile"))
    if provider is None:
        raise CampaignError("ACS restore source profile has no execution provider")
    resume = getattr(provider, "resume_batch", None)
    if not callable(resume):
        raise CampaignError("ACS restore provider lacks resume_batch")
    from engine.campaigns.orchestration import BatchExecutionContextV1
    from engine.qrr.execution_request_v1 import build_execution_request_v1

    bound_request = build_execution_request_v1(provider, plan, selected_batch_id)
    context_root = receipt_path.parent / f".{receipt_path.stem}.restore-context"
    context = BatchExecutionContextV1(
        run_root=context_root,
        batch_root=context_root / "batch",
        manifest_path=context_root / "manifest.json",
        execution_route_sha256=bound_request.resident_request.execution_route_sha256,
    )
    started_ns = time.perf_counter_ns()
    raw = resume(plan, bound_request.resident_request, dict(checkpoint), context)
    elapsed_seconds = (time.perf_counter_ns() - started_ns) / 1_000_000_000.0
    if not isinstance(raw, Mapping) or raw.get("status") != "completed":
        raise CampaignError("ACS restore did not complete")
    restored_result = raw.get("result")
    if not isinstance(restored_result, Mapping):
        raise CampaignError("ACS restore result is missing")
    fresh_owner = restored_result.get("fresh_owner_restore_receipt")
    if not isinstance(fresh_owner, Mapping):
        raise CampaignError("ACS restore lacks a fresh-owner restore receipt")
    source_digests = checkpoint.get("terminal_cell_digests")
    restored_digests = fresh_owner.get("restored_terminal_cell_digests")
    semantic_identity = (
        fresh_owner.get("projection_bundles_reloaded") is True
        and isinstance(source_digests, Mapping)
        and isinstance(restored_digests, Mapping)
        and dict(source_digests) == dict(restored_digests)
    )
    core = {
        "schema_version": "eiel_acs_restore_measurement_v1",
        "status": "passed" if semantic_identity else "failed",
        "source_run_root": str(root),
        "source_manifest_path": str(manifest_path),
        "source_manifest_sha256": canonical_sha256(manifest),
        "source_checkpoint_sha256": canonical_sha256(dict(checkpoint)),
        "batch_id": selected_batch_id,
        "restore_wall_seconds": elapsed_seconds,
        "semantic_digest_identity": semantic_identity,
        "fresh_owner_restore_receipt": dict(fresh_owner),
        "claim_boundary": (
            "strict terminal-checkpoint reload and digest comparison only; "
            "no simulation ticks are redispatched by this measurement"
        ),
    }
    receipt = {**core, "receipt_sha256": canonical_sha256(core)}
    _atomic_text(receipt_path, pretty_json(receipt))
    return {
        "ok": semantic_identity,
        "schema_version": receipt["schema_version"],
        "status": receipt["status"],
        "semantic_digest_identity": semantic_identity,
        "restore_wall_seconds": elapsed_seconds,
        "output": str(receipt_path),
    }


def _validate_adaptive_stage_manifest(
    value: Mapping[str, Any],
    *,
    source_plan: Mapping[str, Any],
    restore_plan: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = dict(value)
    declared = manifest.pop("receipt_sha256", None)
    stages = {
        "C2": (25, 50, 4_000, 8_000),
        "C3": (50, 75, 8_000, 12_000),
        "C4": (75, 100, 12_000, 16_000),
        "C5": (100, 125, 16_000, 20_000),
    }
    coordinates = stages.get(str(manifest.get("stage")))
    source_execution = source_plan.get("execution")
    restore_execution = restore_plan.get("execution")
    source_cells = source_plan.get("cells")
    restore_cells = restore_plan.get("cells")
    if (
        manifest.get("schema_version") != "eiel_wrc_b48_adaptive_stage_manifest_v1"
        or not isinstance(declared, str)
        or canonical_sha256(manifest) != declared
        or coordinates is None
        or not isinstance(source_execution, Mapping)
        or not isinstance(restore_execution, Mapping)
        or not isinstance(source_cells, list)
        or not isinstance(restore_cells, list)
        or len(source_cells) != 48
        or len(restore_cells) != 48
        or manifest.get("source_plan_sha256") != source_plan.get("plan_sha256")
        or manifest.get("restore_plan_sha256") != restore_plan.get("plan_sha256")
        or manifest.get("qualification_key_sha256")
        != source_plan.get("qualification_key_sha256")
        or restore_plan.get("qualification_key_sha256")
        != source_plan.get("qualification_key_sha256")
        or manifest.get("qualification_receipt_action") != "unchanged"
        or manifest.get("physical_cell_count") != 48
        or manifest.get("configuration_count") != 16
        or manifest.get("seed_replicates_per_configuration") != 3
        or tuple(
            manifest.get(name)
            for name in (
                "source_generation",
                "restore_generation",
                "tick_start",
                "tick_end",
            )
        )
        != coordinates
        or source_execution.get("generations") != coordinates[0]
        or restore_execution.get("generations") != coordinates[1]
        or manifest.get("production_artifacts_sha256")
        != canonical_sha256(source_plan.get("artifacts"))
        or restore_plan.get("artifacts") != source_plan.get("artifacts")
    ):
        raise CampaignError("adaptive stage manifest binding is invalid")

    actual_overrides: dict[str, dict[str, Any]] = {}
    for source_cell, restore_cell in zip(source_cells, restore_cells, strict=True):
        if not isinstance(source_cell, Mapping) or not isinstance(
            restore_cell, Mapping
        ):
            raise CampaignError("adaptive stage cell mapping is malformed")
        source_parameters = source_cell.get("parameters")
        restore_parameters = restore_cell.get("parameters")
        if (
            {key: item for key, item in source_cell.items() if key != "parameters"}
            != {key: item for key, item in restore_cell.items() if key != "parameters"}
            or not isinstance(source_parameters, Mapping)
            or not isinstance(restore_parameters, Mapping)
            or set(source_parameters) != set(restore_parameters)
        ):
            raise CampaignError("adaptive stage changes cell identity or schema")
        changes = {
            name: restore_parameters[name]
            for name in sorted(source_parameters)
            if source_parameters[name] != restore_parameters[name]
        }
        if changes:
            actual_overrides[str(source_cell.get("cell_id"))] = changes
    declared_rows = manifest.get("cell_overrides")
    if not isinstance(declared_rows, list):
        raise CampaignError("adaptive stage cell overrides are absent")
    declared_overrides = {
        str(row.get("cell_id")): dict(row.get("overrides", {}))
        for row in declared_rows
        if isinstance(row, Mapping) and isinstance(row.get("overrides"), Mapping)
    }
    if (
        not actual_overrides
        or declared_overrides != actual_overrides
        or len(declared_rows) != len(declared_overrides)
        or manifest.get("changed_cell_count") != len(actual_overrides)
    ):
        raise CampaignError("adaptive stage overrides differ from the restore plan")
    return {**manifest, "receipt_sha256": declared}


def _adaptive_resume(
    source_run_root: Path,
    restore_plan_path: Path,
    stage_manifest_path: Path,
    restore_run_root: Path,
    *,
    provider_registry: Any | None = None,
    prepared_owner_receipt_path: Path | None = None,
    allocation_diagnostic_receipt_path: Path | None = None,
) -> dict[str, Any]:
    """Preflight, seed, and execute one exact adaptive continuation stage."""

    from engine.campaigns.orchestration import (
        execute_campaign,
        prepare_adaptive_campaign_continuation_v1,
    )
    from engine.campaigns.preflight import preflight_campaign
    from engine.campaigns.providers import default_execution_provider_registry

    source_root = source_run_root.expanduser().resolve()
    restore_root = restore_run_root.expanduser().resolve()
    if source_root == restore_root:
        raise CampaignError(
            "adaptive continuation requires distinct source and restore run roots"
        )
    source_plan = _read_object(
        source_root / "execution_plan.json", name="source execution plan"
    )
    restore_plan = _read_object(
        restore_plan_path.expanduser().resolve(), name="restore execution plan"
    )
    stage_manifest = _validate_adaptive_stage_manifest(
        _read_object(
            stage_manifest_path.expanduser().resolve(),
            name="adaptive stage manifest",
        ),
        source_plan=source_plan,
        restore_plan=restore_plan,
    )
    persisted_plan_path = restore_root / "execution_plan.json"
    if persisted_plan_path.exists():
        persisted = _read_object(
            persisted_plan_path, name="persisted restore execution plan"
        )
        if persisted != restore_plan:
            raise CampaignError("restore run root contains a different execution plan")
    else:
        _atomic_text(persisted_plan_path, pretty_json(restore_plan))

    providers = provider_registry or default_execution_provider_registry()
    prepared_owner_receipt = (
        None
        if prepared_owner_receipt_path is None
        else _read_object(
            prepared_owner_receipt_path.expanduser().resolve(),
            name="prepared-owner resource audit receipt",
        )
    )
    allocation_receipt = (
        None
        if allocation_diagnostic_receipt_path is None
        else _read_object(
            allocation_diagnostic_receipt_path.expanduser().resolve(),
            name="allocation diagnostic resource receipt",
        )
    )
    preflight = preflight_campaign(
        restore_plan,
        provider_registry=providers,
        prepared_owner_resource_receipt=prepared_owner_receipt,
        allocation_diagnostic_resource_receipt=allocation_receipt,
    )
    _atomic_text(restore_root / "preflight_receipt.json", pretty_json(preflight))
    if preflight.get("ready_for_execution") is not True:
        reasons = [
            *preflight.get("reasons", []),
            *preflight.get("execution_authority_reasons", []),
        ]
        raise CampaignError(
            "adaptive restore is not execution-ready: "
            + "; ".join(str(item) for item in reasons)
        )
    transition_path = restore_root / "adaptive_transition_receipt.json"
    if transition_path.exists():
        transition = _read_object(transition_path, name="adaptive transition receipt")
        unsigned_transition = dict(transition)
        declared_transition_sha256 = unsigned_transition.pop("receipt_sha256", None)
        source_batches = source_plan.get("physical_batches")
        restore_batches = restore_plan.get("physical_batches")
        expected_batch_id = (
            source_batches[0].get("batch_id")
            if isinstance(source_batches, list)
            and len(source_batches) == 1
            and isinstance(source_batches[0], Mapping)
            else None
        )
        expected_cell_ids = (
            list(source_batches[0].get("cell_ids", []))
            if expected_batch_id is not None
            else None
        )
        if (
            transition.get("schema_version")
            != "eiel_wrc_adaptive_campaign_transition_v1"
            or transition.get("source_campaign_id") != source_plan.get("campaign_id")
            or transition.get("restore_campaign_id") != restore_plan.get("campaign_id")
            or transition.get("source_plan_sha256") != source_plan.get("plan_sha256")
            or transition.get("restore_plan_sha256") != restore_plan.get("plan_sha256")
            or transition.get("qualification_key_sha256")
            != restore_plan.get("qualification_key_sha256")
            or restore_batches != source_batches
            or transition.get("batch_id") != expected_batch_id
            or transition.get("cell_ids") != expected_cell_ids
            or transition.get("source_batch_manifest_path")
            != str(source_root / "batches" / str(expected_batch_id) / "manifest.json")
            or transition.get("restore_batch_manifest_path")
            != str(restore_root / "batches" / str(expected_batch_id) / "manifest.json")
            or transition.get("qualification_receipt_action") != "unchanged"
            or transition.get("status") != "READY_TO_RESUME"
            or not isinstance(transition.get("adaptive_rebinding_receipt_sha256"), str)
            or not isinstance(declared_transition_sha256, str)
            or canonical_sha256(unsigned_transition) != declared_transition_sha256
        ):
            raise CampaignError(
                "persisted adaptive transition receipt differs from this stage"
            )
        transition = {**transition, "receipt_path": str(transition_path)}
    else:
        transition = prepare_adaptive_campaign_continuation_v1(
            source_plan,
            restore_plan,
            source_run_root=source_root,
            restore_run_root=restore_root,
            provider_registry=providers,
        )
    stage_binding = {
        "schema_version": "eiel_wrc_adaptive_stage_execution_binding_v1",
        "stage": stage_manifest["stage"],
        "source_plan_sha256": source_plan["plan_sha256"],
        "restore_plan_sha256": restore_plan["plan_sha256"],
        "stage_manifest_receipt_sha256": stage_manifest["receipt_sha256"],
        "transition_receipt_sha256": transition["receipt_sha256"],
        "adaptive_rebinding_receipt_sha256": transition[
            "adaptive_rebinding_receipt_sha256"
        ],
        "qualification_receipt_action": "unchanged",
    }
    stage_binding = {
        **stage_binding,
        "receipt_sha256": canonical_sha256(stage_binding),
    }
    stage_binding_path = restore_root / "adaptive_stage_execution_binding.json"
    if (
        stage_binding_path.exists()
        and _read_object(stage_binding_path, name="adaptive stage execution binding")
        != stage_binding
    ):
        raise CampaignError("persisted adaptive stage execution binding differs")
    _atomic_text(stage_binding_path, pretty_json(stage_binding))
    execution = execute_campaign(
        restore_plan,
        run_root=restore_root,
        provider_registry=providers,
    )
    return {
        **execution,
        "schema_version": "eiel_wrc_adaptive_stage_execution_receipt_v1",
        "source_run_root": str(source_root),
        "restore_run_root": str(restore_root),
        "transition_receipt_path": transition["receipt_path"],
        "stage_execution_binding_path": str(stage_binding_path),
        "stage_execution_binding_sha256": stage_binding["receipt_sha256"],
        "adaptive_rebinding_receipt_sha256": transition[
            "adaptive_rebinding_receipt_sha256"
        ],
        "qualification_receipt_action": transition["qualification_receipt_action"],
    }


def _analyze(run_root: Path) -> dict[str, Any]:
    from engine.campaigns.analysis import analyze_campaign

    root = run_root.expanduser().resolve()
    plan = _read_object(root / "execution_plan.json", name="execution plan")
    metrics = _read_object(root / "cell_metrics.json", name="cell metrics")
    rows = metrics.get("rows")
    if not isinstance(rows, list):
        raise CampaignError("cell_metrics.json must contain a rows array")
    generation_rows = None
    analysis = plan.get("analysis")
    if isinstance(analysis, Mapping) and analysis.get("extension") is not None:
        generation_series = _read_object(
            root / "generation_metrics.json", name="generation metrics"
        )
        if (
            generation_series.get("campaign_id") != plan.get("campaign_id")
            or generation_series.get("plan_sha256") != plan.get("plan_sha256")
            or not isinstance(generation_series.get("rows"), list)
        ):
            raise CampaignError(
                "generation_metrics.json does not bind the execution plan"
            )
        generation_rows = generation_series["rows"]
    result = analyze_campaign(
        plan,
        rows,
        generation_metric_rows=generation_rows,
        run_root=root,
    )
    path = root / "analysis.json"
    _atomic_text(path, pretty_json(result))
    return {**result, "ok": True, "analysis_path": str(path)}


def _report(run_root: Path) -> dict[str, Any]:
    from engine.campaigns.analysis import render_analysis_markdown

    root = run_root.expanduser().resolve()
    analysis_path = root / "analysis.json"
    analysis = (
        _read_object(analysis_path, name="analysis")
        if analysis_path.exists()
        else _analyze(root)
    )
    markdown = render_analysis_markdown(analysis)
    path = root / "report.md"
    _atomic_text(path, markdown)
    return {
        "ok": True,
        "schema_version": "eiel_campaign_report_receipt_v1",
        "analysis_path": str(analysis_path),
        "report_path": str(path),
        "decision": analysis.get("decision"),
        "decision_label": (
            analysis.get("scientific_analysis", {}).get("decision_label")
            if isinstance(analysis.get("scientific_analysis"), Mapping)
            else None
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "plan", "preflight", "run"):
        command = subparsers.add_parser(name)
        command.add_argument("spec", type=Path)
        if name in {"preflight", "run"}:
            command.add_argument(
                "--prepared-owner-receipt",
                type=Path,
                action="append",
                help=(
                    "exact prepared-owner resource receipt used to promote "
                    "the conservative VRAM projection; repeat for multi-batch "
                    "campaigns"
                ),
            )
            command.add_argument(
                "--allocation-diagnostic-receipt",
                type=Path,
                help=(
                    "exact zero-dispatch B48 allocation receipt used only for "
                    "resource admission"
                ),
            )
    qualify = subparsers.add_parser("qualify")
    qualify.add_argument("spec", type=Path)
    qualify.add_argument(
        "--missing-obligation",
        action="append",
        default=[],
        help="qualify only this missing route obligation; repeat as needed",
    )
    qualify.add_argument(
        "--plan-only",
        action="store_true",
        help="write the deterministic canary plan without launching execution",
    )
    qualify.add_argument("--output", type=Path)
    qualify.add_argument(
        "--run-root",
        type=Path,
        help=(
            "explicit fresh route-qualification run root; use only to isolate "
            "a new attempt from a preserved incompatible root"
        ),
    )
    qualify.add_argument(
        "--tick-native-execution",
        type=Path,
        help="canonical absolute-tick execution request for delta qualification",
    )
    qualify.add_argument(
        "--inherit-route-qualification",
        type=Path,
        help=(
            "exact same-artifact legacy route receipt whose unchanged obligations "
            "are inherited by a tick-native delta"
        ),
    )
    materialize_long = subparsers.add_parser(
        "materialize-long-horizon",
        help="materialize immutable evidence from uninterrupted/restored Launchpad runs",
    )
    materialize_long.add_argument("uninterrupted_run_root", type=Path)
    materialize_long.add_argument("restored_run_root", type=Path)
    materialize_long.add_argument("--output", type=Path, required=True)
    materialize_acs_parity = subparsers.add_parser(
        "materialize-physical-b136-acs-parity",
        help=(
            "bind ACS0/ACS2 uninterrupted and fresh-process-restored "
            "physical-B136 parity"
        ),
    )
    materialize_acs_parity.add_argument("sync_run_root", type=Path)
    materialize_acs_parity.add_argument("sync_restored_run_root", type=Path)
    materialize_acs_parity.add_argument("async_run_root", type=Path)
    materialize_acs_parity.add_argument(
        "async_restored_run_root",
        type=Path,
        nargs="?",
        help="required fourth root; omitted legacy three-root forms fail closed",
    )
    materialize_acs_parity.add_argument("--output", type=Path, required=True)
    materialize_acs_parity.add_argument(
        "--acs2-gpu-telemetry-receipt",
        "--acs-gpu-telemetry-receipt",
        type=Path,
        required=True,
        help=(
            "one passed, exclusive, process-attributed guarded ACS2 telemetry "
            "receipt bound to the uninterrupted ACS2 batch"
        ),
    )
    promote_long = subparsers.add_parser(
        "promote-long-horizon",
        help="extend an exact bounded route receipt without rerunning canaries",
    )
    promote_long.add_argument("spec", type=Path)
    promote_long.add_argument("evidence", type=Path)
    promote_long.add_argument(
        "--frozen-execution-plan",
        type=Path,
        help=(
            "use the exact self-hashed Launchpad plan bound by the evidence "
            "instead of recompiling the source spec"
        ),
    )
    promote_science = subparsers.add_parser(
        "promote-science-route",
        help=(
            "attach a separate digest-bound canonical full production route "
            "attestation without relabeling long-horizon evidence"
        ),
    )
    promote_science.add_argument("spec", type=Path)
    promote_science.add_argument("evidence", type=Path)
    promote_science.add_argument("attestation", type=Path)
    promote_science.add_argument(
        "--frozen-execution-plan",
        type=Path,
        help=(
            "use the exact self-hashed Launchpad plan bound by the evidence "
            "instead of recompiling the source spec"
        ),
    )
    measure = subparsers.add_parser(
        "measure-prepared-owner",
        help=(
            "prepare one full resident batch and record structural "
            "zero-dispatch counters and resources"
        ),
    )
    measure.add_argument("spec", type=Path)
    measure.add_argument(
        "--batch-id",
        help="physical batch ID; optional only when exactly one batch exists",
    )
    resident_audit = subparsers.add_parser(
        "wrc-resident-audit",
        help="write the current-source planning-only WRC resident route audit",
    )
    resident_audit.add_argument("spec", type=Path)
    resident_audit.add_argument("--output", type=Path, required=True)
    measure.add_argument(
        "--output-root",
        type=Path,
        help="new audit directory (defaults below the campaign runtime root)",
    )
    for name in ("resume", "analyze", "report"):
        command = subparsers.add_parser(name)
        command.add_argument("run_root", type=Path)
        if name == "resume":
            command.add_argument(
                "--prepared-owner-receipt",
                type=Path,
                help="exact receipt required to freshly revalidate a promoted route",
            )
    restore_measurement = subparsers.add_parser(
        "measure-acs-restore",
        help="strictly reload one completed terminal checkpoint without redispatching ticks",
    )
    restore_measurement.add_argument("run_root", type=Path)
    restore_measurement.add_argument("--output", type=Path, required=True)
    restore_measurement.add_argument("--batch-id")
    adaptive_resume = subparsers.add_parser(
        "adaptive-resume",
        help=(
            "preflight and resume one generation-boundary WRC adaptive stage "
            "from a completed source root"
        ),
    )
    adaptive_resume.add_argument("source_run_root", type=Path)
    adaptive_resume.add_argument("restore_plan", type=Path)
    adaptive_resume.add_argument("stage_manifest", type=Path)
    adaptive_resume.add_argument("restore_run_root", type=Path)
    adaptive_resume.add_argument(
        "--prepared-owner-receipt",
        type=Path,
        help="exact prepared-owner resource receipt for the restore plan",
    )
    adaptive_resume.add_argument(
        "--allocation-diagnostic-receipt",
        type=Path,
        help="exact allocation diagnostic receipt for the restore plan",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        # Native owners may emit human-readable progress. Keep stdout reserved
        # for the single machine-readable command receipt.
        with redirect_stdout(sys.stderr):
            if args.command == "validate":
                result = _validate(args.spec)
            elif args.command == "plan":
                plan, path = _plan(args.spec)
                result = {
                    "ok": True,
                    "schema_version": "eiel_campaign_plan_receipt_v1",
                    "campaign_id": plan["campaign_id"],
                    "plan_sha256": plan["plan_sha256"],
                    "execution_plan_path": str(path),
                }
            elif args.command == "preflight":
                result = _preflight(
                    args.spec,
                    prepared_owner_receipt_paths=args.prepared_owner_receipt,
                    allocation_diagnostic_receipt_path=(
                        args.allocation_diagnostic_receipt
                    ),
                )
            elif args.command == "run":
                result = _run(
                    args.spec,
                    prepared_owner_receipt_paths=args.prepared_owner_receipt,
                    allocation_diagnostic_receipt_path=(
                        args.allocation_diagnostic_receipt
                    ),
                )
            elif args.command == "qualify":
                result = _qualify(
                    args.spec,
                    plan_only=args.plan_only,
                    missing_obligations=args.missing_obligation,
                    output=args.output,
                    run_root=args.run_root,
                    tick_native_execution_path=args.tick_native_execution,
                    inherit_route_qualification_path=(args.inherit_route_qualification),
                )
            elif args.command == "materialize-long-horizon":
                result = _materialize_long_horizon(
                    args.uninterrupted_run_root,
                    args.restored_run_root,
                    output=args.output,
                )
            elif args.command == "materialize-physical-b136-acs-parity":
                if args.async_restored_run_root is None:
                    raise ValueError(
                        "physical-B136 ACS parity now requires four roots ordered "
                        "ACS0 uninterrupted/restored and ACS2 uninterrupted/restored"
                    )
                result = _materialize_physical_b136_acs_parity(
                    args.sync_run_root,
                    args.sync_restored_run_root,
                    args.async_run_root,
                    args.async_restored_run_root,
                    output=args.output,
                    acs2_gpu_telemetry_receipt=args.acs2_gpu_telemetry_receipt,
                )
            elif args.command == "promote-long-horizon":
                result = _promote_long_horizon(
                    args.spec,
                    args.evidence,
                    frozen_execution_plan_path=args.frozen_execution_plan,
                )
            elif args.command == "promote-science-route":
                result = _promote_science_route(
                    args.spec,
                    args.evidence,
                    args.attestation,
                    frozen_execution_plan_path=args.frozen_execution_plan,
                )
            elif args.command == "measure-prepared-owner":
                result = _measure_prepared_owner(
                    args.spec,
                    batch_id=args.batch_id,
                    output_root=args.output_root,
                )
            elif args.command == "wrc-resident-audit":
                result = _wrc_resident_route_audit(args.spec, output=args.output)
            elif args.command == "resume":
                result = _resume(
                    args.run_root,
                    prepared_owner_receipt_path=args.prepared_owner_receipt,
                )
            elif args.command == "measure-acs-restore":
                result = _measure_acs_restore(
                    args.run_root,
                    output=args.output,
                    batch_id=args.batch_id,
                )
            elif args.command == "adaptive-resume":
                result = _adaptive_resume(
                    args.source_run_root,
                    args.restore_plan,
                    args.stage_manifest,
                    args.restore_run_root,
                    prepared_owner_receipt_path=args.prepared_owner_receipt,
                    allocation_diagnostic_receipt_path=(
                        args.allocation_diagnostic_receipt
                    ),
                )
            elif args.command == "analyze":
                result = _analyze(args.run_root)
            else:
                result = _report(args.run_root)
    except Exception as exc:  # noqa: BLE001 - machine-readable fail-closed CLI.
        result = {
            "ok": False,
            "schema_version": "eiel_campaign_failure_v1",
            "command": args.command,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
    print(pretty_json(result), end="")
    return 0 if result.get("ok") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
