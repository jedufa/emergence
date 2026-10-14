#!/usr/bin/env python3
"""Audit, query, plan, and decide EIEL Qualification Framework V1 records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ENGINE_ROOT = ROOT / "engine"
if ENGINE_ROOT.is_dir() and str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from engine.campaigns.canonical import canonical_sha256, pretty_json  # noqa: E402
from engine.campaigns.qualification_framework import (  # noqa: E402
    EvidenceBindingV1,
    OperatingEnvelopeV1,
    QualificationCatalogV1,
    QualificationFrameworkError,
    generate_delta_registry,
    delta_status_row,
    load_catalog,
    plan_delta_campaign,
    publish_accepted_delta,
    resolve_qualification,
    validate_delta_plan,
    validate_authority_bundle,
)
from engine.campaigns.qualification_decisions import qualification_decision_payload  # noqa: E402


def _read_json_object(path: str | Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QualificationFrameworkError(f"cannot read {name}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise QualificationFrameworkError(f"{name} must be a JSON object: {path}")
    return value


def _write_or_print(payload: dict[str, Any], output: str | Path | None) -> None:
    text = pretty_json(payload)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)


def _catalog_stats(catalog: QualificationCatalogV1) -> dict[str, Any]:
    return {
        "schema_version": "eiel_qualification_audit_v1",
        "ok": True,
        "anchor_count": len(catalog.anchors),
        "envelope_count": len(catalog.envelopes),
        "resource_capacity_envelope_count": len(catalog.resource_capacity_envelopes),
        "historical_authority_count": len(catalog.historical_authority),
        "scopes": sorted(
            {record.scope for record in catalog.anchors}
            | {record.scope for record in catalog.envelopes}
            | {"RESOURCE_CAPACITY_ENVELOPE" for _ in catalog.resource_capacity_envelopes}
        ),
        "catalog_sha256": catalog.payload()["catalog_sha256"],
        "anchors": [
            {
                "anchor_id": anchor.anchor_id,
                "scope": anchor.scope,
                "decision": anchor.decision,
                "source_commit": anchor.source_commit,
                "execution_route_sha256": anchor.execution_route_sha256,
            }
            for anchor in sorted(catalog.anchors, key=lambda row: row.anchor_id)
        ],
        "envelopes": [
            {
                "envelope_id": envelope.envelope_id,
                "scope": envelope.scope,
                "decision": envelope.decision,
                "interpolation_policy": envelope.interpolation_policy,
                "lifecycle_stage": envelope.lifecycle_stage,
                "authority_state": envelope.authority_state,
                "required_resource_capacity_id": envelope.required_resource_capacity_id,
                "missing_corner_count": len(envelope.missing_corners()),
            }
            for envelope in sorted(catalog.envelopes, key=lambda row: row.envelope_id)
        ],
        "resource_capacity_envelopes": [
            {
                "resource_id": resource.resource_id,
                "decision": resource.decision,
                "backend": resource.backend,
                "device": resource.device,
                "batch_size": resource.batch_size.payload(),
            }
            for resource in sorted(catalog.resource_capacity_envelopes, key=lambda row: row.resource_id)
        ],
        "historical_authority": [
            {
                "historical_id": record.historical_id,
                "state": record.state,
                "resolver_eligible": record.resolver_eligible,
                "current_authority": record.current_authority,
                "revalidation_required": record.revalidation_required,
            }
            for record in sorted(catalog.historical_authority, key=lambda row: row.historical_id)
        ],
    }


def _find_envelope(catalog: QualificationCatalogV1, envelope_id: str) -> OperatingEnvelopeV1:
    for envelope in catalog.envelopes:
        if envelope.envelope_id == envelope_id:
            return envelope
    raise QualificationFrameworkError(f"unknown envelope_id: {envelope_id}")


def _audit(args: argparse.Namespace) -> dict[str, Any]:
    return _catalog_stats(load_catalog(args.catalog))


def _status(args: argparse.Namespace) -> dict[str, Any]:
    catalog = load_catalog(args.catalog)
    request = _read_json_object(args.request, name="qualification request")
    result = resolve_qualification(request, catalog)
    result["status_row"] = delta_status_row(result)
    return result


def _plan(args: argparse.Namespace) -> dict[str, Any]:
    catalog = load_catalog(args.catalog)
    request = _read_json_object(args.request, name="qualification request")
    envelope = _find_envelope(catalog, args.envelope_id)
    return plan_delta_campaign(request, envelope)


def _validate_plan(args: argparse.Namespace) -> dict[str, Any]:
    plan = _read_json_object(args.plan, name="qualification plan")
    return validate_delta_plan(plan)


def _generate_registry(args: argparse.Namespace) -> dict[str, Any]:
    if args.plan and not args.decision:
        plan = _read_json_object(args.plan, name="qualification plan")
        registry = generate_delta_registry(plan, campaign_id=args.campaign_id)
        return registry
    if args.catalog and args.plan and args.decision:
        catalog = load_catalog(args.catalog)
        plan = _read_json_object(args.plan, name="qualification plan")
        decision = _read_json_object(args.decision, name="qualification decision")
        evidence = [
            EvidenceBindingV1(
                evidence_id=str(row["evidence_id"]),
                kind=str(row["kind"]),
                uri=str(row["uri"]),
                sha256=str(row["sha256"]),
            )
            for row in decision.get("evidence_bindings") or ()
        ]
        accepted = publish_accepted_delta(
            catalog,
            envelope_id=args.envelope_id,
            plan=plan,
            decision=decision,
            evidence=evidence,
        )
        return accepted.payload()
    if not args.catalog:
        raise QualificationFrameworkError("--catalog is required unless --plan-only registry generation is used")
    catalog = load_catalog(args.catalog)
    return catalog.payload()


def _decide(args: argparse.Namespace) -> dict[str, Any]:
    return qualification_decision_payload(
        manifest_path=args.manifest,
        evidence_path=args.evidence,
        profile=args.profile,
        plan_sha256=args.plan_sha256,
        source_commit=args.source_commit,
        evidence_uri=args.evidence_uri,
    )


def _index(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.evidence_root)
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("qualification_decision.json")) if root.exists() else []:
        try:
            payload = _read_json_object(path, name="qualification decision")
        except QualificationFrameworkError as exc:
            rows.append({"path": str(path), "status": "invalid", "error": str(exc)})
            continue
        rows.append(
            {
                "path": str(path),
                "status": "ok",
                "decision": payload.get("decision"),
                "profile": payload.get("profile"),
                "sha256": canonical_sha256(payload),
            }
        )
    payload = {
        "schema_version": "eiel_qualification_evidence_index_v1",
        "evidence_root": str(root),
        "records": rows,
    }
    payload["index_sha256"] = canonical_sha256(payload)
    return payload


def _validate_authority(args: argparse.Namespace) -> dict[str, Any]:
    bundle = _read_json_object(args.bundle, name="QF-1 authority bundle")
    catalog = load_catalog(args.catalog) if args.catalog else None
    return validate_authority_bundle(bundle, catalog=catalog)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--catalog", required=True)
    audit.add_argument("--output")

    status = subparsers.add_parser("status")
    status.add_argument("--catalog", required=True)
    status.add_argument("--request", required=True)
    status.add_argument("--output")

    plan = subparsers.add_parser("plan")
    plan.add_argument("--catalog", required=True)
    plan.add_argument("--request", required=True)
    plan.add_argument("--envelope-id", required=True)
    plan.add_argument("--output")

    validate_plan = subparsers.add_parser("validate-plan")
    validate_plan.add_argument("--plan", required=True)
    validate_plan.add_argument("--output")

    generate_registry = subparsers.add_parser("generate-registry")
    generate_registry.add_argument("--catalog")
    generate_registry.add_argument("--plan")
    generate_registry.add_argument("--decision")
    generate_registry.add_argument("--envelope-id")
    generate_registry.add_argument("--campaign-id", default="qf1-synthetic-delta")
    generate_registry.add_argument("--output", required=True)

    decide = subparsers.add_parser("decide")
    decide.add_argument("--evidence")
    decide.add_argument("--manifest")
    decide.add_argument("--profile")
    decide.add_argument("--plan-sha256")
    decide.add_argument("--source-commit")
    decide.add_argument("--evidence-uri")
    decide.add_argument("--output")

    index = subparsers.add_parser("index")
    index.add_argument("--evidence-root", required=True)
    index.add_argument("--output")

    validate_authority = subparsers.add_parser("validate-authority")
    validate_authority.add_argument("--bundle", required=True)
    validate_authority.add_argument("--catalog")
    validate_authority.add_argument("--output")

    args = parser.parse_args(argv)
    try:
        handlers = {
            "audit": _audit,
            "status": _status,
            "plan": _plan,
            "validate-plan": _validate_plan,
            "generate-registry": _generate_registry,
            "decide": _decide,
            "index": _index,
            "validate-authority": _validate_authority,
        }
        payload = handlers[args.command](args)
    except QualificationFrameworkError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    _write_or_print(payload, getattr(args, "output", None))
    return 0 if payload.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
