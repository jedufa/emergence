"""EIEL Technical Qualification Manager over current authority records and QF contracts."""

from __future__ import annotations

import argparse
import shlex
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import unquote, urlparse

from tools.actors.canonical import canonical_sha256
from engine.campaigns.qualification_framework import (
    QualificationCatalogV1,
    QualificationFrameworkError,
    delta_status_row,
    impact_families,
    load_catalog,
    plan_delta_campaign,
    resolve_qualification,
    validate_authority_bundle,
    validate_delta_plan,
)
from engine.qualification_plans.v1 import QualificationPlanError, validate_plan as validate_declarative_plan
from tools.actors.common import (
    ActorArgumentParser,
    ActorError,
    actor_state_root,
    atomic_write_bytes,
    atomic_write_json,
    emit_actor_error,
    emit_payload,
    file_sha256,
    forbidden_operation_error,
    new_record,
    read_json,
    read_structured,
    repository_root,
    stable_id,
    utc_now,
    with_guidance,
)
from tools.actors.store import ActorRecordStore
from tools.actors.release import operator_metadata
from engine.campaigns.qualification_decisions import qualification_decision_payload

ACTOR_NAME = "qualificationManager"
STORE_NAMESPACE = "qualifications"
PLAN_RECORD_TYPE = "technical_qualification_plan"
DECISION_RECORD_TYPE = "technical_qualification_decision"
CHECK_RECORD_TYPE = "technical_qualification_check"
IMPACT_SCHEMA_VERSION = "eiel_qualification_impact_assessment_v1"
AUTHORITY_REGISTER_SCHEMA = "eiel_qualification_authority_register_v1"
DEFAULT_AUTHORITY_REGISTER = repository_root() / "docs/100_governance-authority/110_authority-registers/EIEL-DOC-000005--eiel-qualification-authority-register.json"
BLOCKING_SCOPE_TOKENS = ("blocked", "not_qualified", "failed_")
LOW_LEVEL_QUALIFICATION_CLI = "python -m tools.scripts.eiel_qualification"
CAMPAIGN_QUALIFICATION_CLI = "python -m tools.scripts.eiel_campaign qualify"
QUALIFICATION_INPUT_ARTIFACT_SCHEMA_VERSION = "eiel_qualification_input_artifact_v1"


def _retain_qualification_input(
    source: str | Path,
    *,
    kind: str,
    state_root: str | Path | None,
    generated_at: str | None,
) -> dict[str, Any]:
    source_path = Path(source).expanduser().resolve()
    try:
        data = source_path.read_bytes()
    except OSError as exc:
        raise ActorError(
            f"cannot retain qualification {kind} input: {source_path}: {exc}",
            code="qualification_input_retention_failed",
        ) from exc
    digest = __import__("hashlib").sha256(data).hexdigest()
    artifact_id = stable_id(
        "QINPUT",
        {"kind": kind, "bytes_sha256": digest},
    )
    root = actor_state_root(state_root) / "qualification-inputs" / artifact_id
    bytes_path = root / "content.bin"
    manifest_path = root / "manifest.json"
    if bytes_path.exists() and bytes_path.read_bytes() != data:
        raise ActorError(
            "immutable qualification input artifact conflicts with retained bytes",
            code="qualification_input_artifact_conflict",
            problems=[str(bytes_path)],
        )
    if not bytes_path.exists():
        atomic_write_bytes(bytes_path, data)
    manifest = {
        "schema_version": QUALIFICATION_INPUT_ARTIFACT_SCHEMA_VERSION,
        "artifact_id": artifact_id,
        "kind": str(kind),
        "source_uri": str(source_path),
        "bytes_sha256": digest,
        "size_bytes": len(data),
        "content_path": str(bytes_path),
        "state_root_locator": bytes_path.relative_to(actor_state_root(state_root)).as_posix(),
        "recorded_at_utc": utc_now(generated_at),
    }
    if manifest_path.exists():
        existing = read_structured(manifest_path, name="qualification input artifact")
        material = {key: value for key, value in manifest.items() if key != "recorded_at_utc"}
        prior = {key: value for key, value in existing.items() if key != "recorded_at_utc"}
        if material != prior:
            raise ActorError(
                "immutable qualification input manifest conflicts with retained artifact",
                code="qualification_input_artifact_conflict",
                problems=[str(manifest_path)],
            )
        manifest = dict(existing)
    else:
        atomic_write_json(manifest_path, manifest)
    return {
        "schema_version": QUALIFICATION_INPUT_ARTIFACT_SCHEMA_VERSION,
        "artifact_id": artifact_id,
        "kind": str(kind),
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "content_path": str(bytes_path),
        "state_root_locator": bytes_path.relative_to(actor_state_root(state_root)).as_posix(),
        "bytes_sha256": digest,
        "size_bytes": len(data),
    }


def integration_boundary() -> dict[str, Any]:
    return {
        "schema_version": "eiel_qualification_manager_integration_boundary_v1",
        "manager_role": "codex_friendly_authority_resolution_plan_tracking_and_handoff",
        "shared_core": "engine.campaigns.qualification_framework",
        "lower_level_qualification_cli": LOW_LEVEL_QUALIFICATION_CLI,
        "campaign_route_qualification_cli": CAMPAIGN_QUALIFICATION_CLI,
        "manager_executes_campaigns": False,
        "manager_reimplements_qualification_semantics": False,
        "scientific_qualification_effect": "none",
    }


def _mission_campaign_spec(args: argparse.Namespace) -> str | None:
    mission_id = getattr(args, "mission", None)
    if not mission_id:
        return None
    mission = ActorRecordStore(actor_state_root(args.state_root), "missions").get(mission_id)
    references = dict(dict(mission.get("payload") or {}).get("references") or {})
    paths = [str(item) for item in references.get("campaign_spec_paths") or [] if str(item).strip()]
    return paths[0] if paths else None


def _store(args: argparse.Namespace) -> ActorRecordStore:
    return ActorRecordStore(actor_state_root(args.state_root), STORE_NAMESPACE)


def _common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--state-root")
    parser.add_argument("--generated-at")
    parser.add_argument("--output")
    parser.add_argument("--human", action="store_true")


def _authority_options(parser: argparse.ArgumentParser, *, request: bool = False) -> None:
    parser.add_argument("--catalog", help="Use a QF-1 catalog instead of the current authority register.")
    parser.add_argument("--authority-register", help="Qualification authority register JSON; defaults to the repository register.")
    if request:
        parser.add_argument("--request")
        parser.add_argument("--mission")
        parser.add_argument("--track-label")


def _catalog(path: str | None) -> QualificationCatalogV1:
    if not path:
        raise ActorError("--catalog is required for this QF-1 operation", code="missing_qf_catalog")
    try:
        return load_catalog(path)
    except QualificationFrameworkError as exc:
        raise ActorError(str(exc), code="invalid_qf_catalog") from exc


def _register_path(args: argparse.Namespace) -> Path:
    path = Path(args.authority_register).expanduser().resolve() if getattr(args, "authority_register", None) else DEFAULT_AUTHORITY_REGISTER
    if not path.is_file():
        raise ActorError(
            f"qualification authority register is missing: {path}",
            code="authority_register_missing",
            next_actions=[{"description": "supply the current register", "command": "python -m tools.eiel qualificationManager status --authority-register PATH"}],
        )
    return path


def load_authority_register(path: str | Path) -> dict[str, Any]:
    register = read_json(path, name="qualification authority register")
    if register.get("schema_version") != AUTHORITY_REGISTER_SCHEMA:
        raise ActorError("unsupported qualification authority register schema", code="unsupported_authority_register_schema")
    if not isinstance(register.get("tracks"), list):
        raise ActorError("qualification authority register tracks must be a list", code="invalid_authority_register")
    return register


def authority_register_summary(register: Mapping[str, Any]) -> dict[str, Any]:
    tracks = [row for row in register.get("tracks") or [] if isinstance(row, Mapping)]
    counts = Counter(str(row.get("status") or "unknown") for row in tracks)
    return {
        "schema_version": "eiel_qualification_authority_register_summary_v1",
        "track_count": len(tracks),
        "status_counts": dict(sorted(counts.items())),
        "labels": sorted(str(row.get("label")) for row in tracks),
        "repository": dict(register.get("repository") or {}),
        "last_updated": register.get("last_updated"),
        "policy_no_claims": dict(register.get("policy") or {}).get("no_claims"),
        "technical_only": True,
        "scientific_qualification_effect": "none",
    }


def catalog_summary(catalog: QualificationCatalogV1) -> dict[str, Any]:
    return {
        "schema_version": "eiel_qualification_catalog_summary_v1",
        "anchors": {"count": len(catalog.anchors), "decision_counts": dict(sorted(Counter(str(row.decision) for row in catalog.anchors).items())), "ids": sorted(row.anchor_id for row in catalog.anchors)},
        "envelopes": {"count": len(catalog.envelopes), "decision_counts": dict(sorted(Counter(str(row.decision) for row in catalog.envelopes).items())), "ids": sorted(row.envelope_id for row in catalog.envelopes)},
        "resource_capacity_envelopes": {"count": len(catalog.resource_capacity_envelopes), "decision_counts": dict(sorted(Counter(str(row.decision) for row in catalog.resource_capacity_envelopes).items())), "ids": sorted(row.resource_id for row in catalog.resource_capacity_envelopes)},
        "historical_authority": {"count": len(catalog.historical_authority), "state_counts": dict(sorted(Counter(str(row.state) for row in catalog.historical_authority).items()))},
        "technical_only": True,
        "scientific_qualification_effect": "none",
    }


def _request_from_mission(args: argparse.Namespace) -> dict[str, Any]:
    mission = ActorRecordStore(actor_state_root(args.state_root), "missions").get(args.mission)
    payload = dict(mission.get("payload") or {})
    request = payload.get("qualification_request") or payload.get("technical_route")
    if not isinstance(request, Mapping):
        raise ActorError(
            "mission has no qualification_request or technical_route object",
            code="mission_qualification_request_missing",
            problems=["qualificationManager cannot infer an exact route safely from prose"],
            next_actions=[{"description": "add a structured request", "command": f"python -m tools.eiel missionPlanner update {args.mission} --set qualification_request.FIELD=VALUE"}],
            related_records=[args.mission],
        )
    return dict(request)


def _load_request(args: argparse.Namespace) -> dict[str, Any]:
    if getattr(args, "request", None):
        request = read_structured(args.request, name="qualification request")
    elif getattr(args, "mission", None):
        request = _request_from_mission(args)
    elif getattr(args, "track_label", None):
        request = {"track_label": args.track_label}
    else:
        raise ActorError(
            "qualification check requires --request, --mission, or --track-label",
            code="qualification_request_missing",
            next_actions=[{"description": "check a known track", "command": "python -m tools.eiel qualificationManager check --track-label TRACK"}],
        )
    if getattr(args, "track_label", None):
        request["track_label"] = args.track_label
    return request


def _normalized_device(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).lower()
    return {"cuda": "gpu", "nvidia": "gpu"}.get(text, text)


def _request_axes(request: Mapping[str, Any]) -> dict[str, Any]:
    grid = request.get("grid") if isinstance(request.get("grid"), Mapping) else {}
    return {
        "world": request.get("world", request.get("world_family")),
        "grid_width": request.get("grid_width", grid.get("width")),
        "grid_height": request.get("grid_height", grid.get("height")),
        "initial_population": request.get("initial_population", request.get("population")),
        "resident_capacity": request.get("resident_capacity"),
        "ticks_per_generation": request.get("ticks_per_generation"),
        "device": _normalized_device(request.get("device")),
        "batch_cells": request.get("batch_cells", request.get("batch_size", request.get("logical_cells"))),
        "route_hash": request.get("route_hash", request.get("execution_route_sha256")),
    }


def _track_axes(track: Mapping[str, Any]) -> dict[str, Any]:
    grid = track.get("grid") if isinstance(track.get("grid"), Mapping) else {}
    envelope = track.get("execution_envelope") if isinstance(track.get("execution_envelope"), Mapping) else {}
    return {
        "world": track.get("world"),
        "grid_width": grid.get("width"),
        "grid_height": grid.get("height"),
        "initial_population": track.get("initial_population"),
        "resident_capacity": track.get("resident_capacity"),
        "ticks_per_generation": track.get("ticks_per_generation"),
        "device": _normalized_device(envelope.get("device")),
        "batch_cells": envelope.get("batch_cells", envelope.get("logical_cells", envelope.get("physical_batch_cells"))),
        "route_hash": track.get("route_hash"),
    }


def _track_consistency(track: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    status = str(track.get("status") or "unknown").lower()
    scopes = [str(item).lower() for item in track.get("authority_scope") or []]
    if status == "qualified" and any(any(token in scope for token in BLOCKING_SCOPE_TOKENS) for scope in scopes):
        issues.append("status is qualified but authority_scope contains a blocking token")
    if status == "qualified" and not track.get("durable_report_path"):
        issues.append("qualified track has no durable_report_path")
    if status == "superseded" and not track.get("superseded_by"):
        blocker = track.get("blocker") if isinstance(track.get("blocker"), Mapping) else {}
        limitations = [str(item).lower() for item in track.get("limitations") or []]
        has_named_successor = bool(blocker.get("code") or blocker.get("reason")) or any("superseded by" in item for item in limitations)
        if not has_named_successor:
            issues.append("superseded track has no successor reference in superseded_by, blocker, or limitations")
    return issues


def resolve_authority_register(request: Mapping[str, Any], register: Mapping[str, Any]) -> dict[str, Any]:
    tracks = [dict(row) for row in register.get("tracks") or [] if isinstance(row, Mapping)]
    label = str(request.get("track_label") or request.get("label") or "").strip()
    supplied = {k: v for k, v in _request_axes(request).items() if v is not None and v != ""}
    if not label and len(supplied) < 4:
        return {
            "schema_version": "eiel_qualification_register_resolution_v1",
            "decision": "INVALID_REQUEST",
            "qualified": False,
            "record_id": None,
            "reasons": ["provide a track_label or at least four exact route axes"],
            "reason_codes": ["insufficient_route_identity"],
            "next_required_action": "add exact world, grid, population/capacity, ticks, device, batch, or route hash",
            "request": dict(request),
            "candidates": [],
        }
    exact: list[dict[str, Any]] = []
    near: list[dict[str, Any]] = []
    for track in tracks:
        if label:
            if str(track.get("label")) == label:
                axes = _track_axes(track)
                mismatches = {
                    key: {"requested": value, "track": axes.get(key)}
                    for key, value in supplied.items()
                    if axes.get(key) != value
                }
                if mismatches:
                    return {
                        "schema_version": "eiel_qualification_register_resolution_v1",
                        "decision": "REQUEST_OUTSIDE_AUTHORITY",
                        "qualified": False,
                        "record_id": track.get("label"),
                        "track": track,
                        "reasons": ["the named authority track does not cover the supplied route axes"],
                        "reason_codes": ["request_outside_authority_envelope"],
                        "mismatches": mismatches,
                        "next_required_action": "correct the request identity or create a bounded qualification plan for the changed route",
                        "request": dict(request),
                        "candidates": [{"label": track.get("label"), "status": track.get("status"), "mismatches": mismatches}],
                    }
                exact = [track]
                break
            continue
        axes = _track_axes(track)
        mismatches = {key: {"requested": value, "track": axes.get(key)} for key, value in supplied.items() if axes.get(key) != value}
        if not mismatches:
            exact.append(track)
        else:
            score = len(supplied) - len(mismatches)
            if score > 0:
                near.append({"label": track.get("label"), "status": track.get("status"), "match_score": score, "mismatches": mismatches})
    if not exact:
        return {
            "schema_version": "eiel_qualification_register_resolution_v1",
            "decision": "NO_CURRENT_AUTHORITY",
            "qualified": False,
            "record_id": None,
            "reasons": ["no authority track exactly matches the request"],
            "reason_codes": ["no_exact_authority_record"],
            "next_required_action": "create the smallest bounded technical qualification plan for the unmatched route",
            "request": dict(request),
            "candidates": sorted(near, key=lambda row: (-int(row["match_score"]), str(row["label"])))[:5],
        }
    if len(exact) > 1:
        return {
            "schema_version": "eiel_qualification_register_resolution_v1",
            "decision": "AMBIGUOUS_AUTHORITY",
            "qualified": False,
            "record_id": None,
            "reasons": ["multiple authority tracks match the supplied axes; add route_hash or track_label"],
            "reason_codes": ["ambiguous_authority_match"],
            "next_required_action": "identify the exact track label or execution route hash",
            "request": dict(request),
            "candidates": [{"label": row.get("label"), "status": row.get("status"), "route_hash": row.get("route_hash")} for row in exact],
        }
    track = exact[0]
    consistency = _track_consistency(track)
    status = str(track.get("status") or "unknown").lower()
    status_map = {
        "qualified": "QUALIFIED",
        "partial": "QUALIFIED_WITH_LIMITS",
        "in_progress": "QUALIFICATION_IN_PROGRESS",
        "blocked": "BLOCKED",
        "rejected": "REJECTED",
        "superseded": "SUPERSEDED",
        "expired": "EXPIRED",
        "unknown": "UNKNOWN",
    }
    decision = status_map.get(status, "UNKNOWN")
    qualified = decision in {"QUALIFIED", "QUALIFIED_WITH_LIMITS"} and not consistency
    if consistency:
        decision = "INCONSISTENT_AUTHORITY_RECORD"
    repository_commit = dict(register.get("repository") or {}).get("commit")
    requested_commit = request.get("source_commit")
    source_warning = None
    if requested_commit and repository_commit and requested_commit != repository_commit and not track.get("source_commit"):
        source_warning = "request source_commit differs from register repository commit and track has no per-track source binding"
        qualified = False
        decision = "REVALIDATION_REQUIRED"
    reasons = []
    if consistency:
        reasons.extend(consistency)
    if source_warning:
        reasons.append(source_warning)
    if not reasons:
        reasons.append(f"authority track status is {status}")
    next_required = list(track.get("next_required_evidence") or [])
    if not next_required and not qualified:
        next_required = ["review the durable report and create a bounded qualification plan"]
    return {
        "schema_version": "eiel_qualification_register_resolution_v1",
        "decision": decision,
        "qualified": qualified,
        "record_id": track.get("label"),
        "track": track,
        "scope": track.get("authority_scope") or [],
        "limitations": track.get("limitations") or [],
        "reasons": reasons,
        "reason_codes": [
            "current_authority" if qualified else {
                "blocked": "authority_blocked",
                "superseded": "authority_superseded",
                "expired": "authority_expired",
                "rejected": "authority_rejected",
                "in_progress": "qualification_in_progress",
            }.get(status, "authority_not_admitted")
        ],
        "next_required_action": next_required[0] if next_required else "none",
        "next_required_evidence": next_required,
        "request": dict(request),
        "candidates": [],
        "register_repository_commit": repository_commit,
    }


def _resolve(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], str]:
    request = _load_request(args)
    if getattr(args, "catalog", None):
        catalog = _catalog(args.catalog)
        result = dict(resolve_qualification(request, catalog))
        result["status_row"] = delta_status_row(result)
        return request, result, "qf_catalog"
    register_path = _register_path(args)
    result = resolve_authority_register(request, load_authority_register(register_path))
    return request, result, "authority_register"


def _positive(result: Mapping[str, Any]) -> bool:
    return str(result.get("decision")) in {"QUALIFIED", "QUALIFIED_WITH_EXCLUSIONS", "QUALIFIED_WITH_LIMITS"} and bool(result.get("qualified", True))


def _reason_codes(result: Mapping[str, Any]) -> list[str]:
    codes = [str(item) for item in result.get("reason_codes") or [] if str(item)]
    for reason in result.get("reasons") or []:
        if isinstance(reason, Mapping) and reason.get("code"):
            codes.append(str(reason["code"]))
    return list(dict.fromkeys(codes))


def _problem_texts(result: Mapping[str, Any]) -> list[str]:
    rows: list[str] = []
    for reason in result.get("reasons") or []:
        if isinstance(reason, Mapping):
            code = str(reason.get("code") or "qualification_reason")
            message = str(reason.get("message") or reason)
            rows.append(f"{code}: {message}")
        else:
            rows.append(str(reason))
    return rows


def _result_next_actions(
    result: Mapping[str, Any],
    *,
    request_path: str | None = None,
    mission_id: str | None = None,
    catalog_path: str | None = None,
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    if _positive(result):
        if mission_id:
            actions.append({
                "description": "store the source-bound qualification resolution before attaching it",
                "command": f"python -m tools.eiel qualificationManager check --mission {shlex.quote(mission_id)} --store",
            })
        return actions

    request_arg = (
        f"--request {shlex.quote(str(request_path))}"
        if request_path
        else (
            f"--mission {shlex.quote(str(mission_id))}"
            if mission_id
            else "--request REQUEST.json"
        )
    )
    command = f"python -m tools.eiel qualificationManager plan {request_arg}"
    if catalog_path:
        envelope_id = str(result.get("record_id") or "").strip()
        command += f" --catalog {shlex.quote(str(catalog_path))}"
        if envelope_id:
            command += f" --envelope-id {shlex.quote(envelope_id)}"
    actions.append({
        "description": "generate the smallest bounded qualification plan",
        "command": command,
    })
    return actions


def command_status(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    if args.request or args.mission or args.track_label:
        request, result, source = _resolve(args)
        positive = _positive(result)
        return with_guidance(
            {"ok": positive, "source": source, "request": request, "qualification": result},
            status=str(result.get("decision")), reason_codes=_reason_codes(result),
            problems=[] if positive else _problem_texts(result), warnings=result.get("limitations") or [],
            next_actions=_result_next_actions(result, request_path=args.request, mission_id=args.mission, catalog_path=args.catalog),
            related_records=[str(result.get("record_id") or "")],
        ), 0 if positive else 1
    if args.catalog:
        summary = catalog_summary(_catalog(args.catalog))
        return with_guidance({"ok": True, "catalog": str(Path(args.catalog).resolve()), "summary": summary}, status="CATALOG_READY"), 0
    path = _register_path(args)
    register = load_authority_register(path)
    summary = authority_register_summary(register)
    return with_guidance({"ok": True, "authority_register": str(path), "authority_register_sha256": file_sha256(path), "summary": summary, "integration_boundary": integration_boundary()}, status="REGISTER_READY"), 0


def command_check(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    request, result, source = _resolve(args)
    positive = _positive(result)
    record_path = None
    record_id = None
    if args.store:
        record_id = stable_id("QCHK", {"source": source, "request": request, "result": result})
        source_uri = str(Path(args.catalog).resolve()) if args.catalog else str(_register_path(args))
        record = new_record(
            actor=ACTOR_NAME,
            record_type=CHECK_RECORD_TYPE,
            record_id=record_id,
            status=str(result.get("decision") or "UNKNOWN"),
            payload={"request": request, "result": result, "source": source, "authoritative_effect": "none_read_only_resolution"},
            generated_at=utc_now(args.generated_at),
            source_refs=({"kind": source, "uri": source_uri},),
            action="check",
        )
        record_path = _store(args).put(record)
    return with_guidance(
        {
            "ok": positive,
            "source": source,
            "check_id": record_id,
            "record_path": str(record_path) if record_path else None,
            "decision": result.get("decision"),
            "status_row": result.get("status_row"),
            "record_id": result.get("record_id"),
            "scope": result.get("scope"),
            "reasons": result.get("reasons") or [],
            "limitations": result.get("limitations") or [],
            "full_result": result,
            "integration_boundary": integration_boundary(),
        },
        status=str(result.get("decision")), reason_codes=_reason_codes(result),
        problems=[] if positive else _problem_texts(result), warnings=result.get("limitations") or [],
        next_actions=(
            [{
                "description": "attach this source-bound qualification check to the mission",
                "command": f"python -m tools.eiel missionPlanner update {args.mission} --qualification-ref {record_id}",
            }]
            if args.mission and record_id
            else _result_next_actions(result, request_path=args.request, mission_id=args.mission, catalog_path=args.catalog)
        ),
        related_records=[item for item in (record_id, result.get("record_id")) if item],
    ), 0 if positive else 1


def command_explain(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    request, result, source = _resolve(args)
    positive = _positive(result)
    row = dict(result.get("status_row") or {})
    explanation = {
        "schema_version": "eiel_qualification_explanation_v1_1",
        "source": source,
        "decision": result.get("decision"),
        "qualified": positive,
        "authority_state": row.get("authority_state", result.get("decision")),
        "lifecycle_stage": row.get("lifecycle_stage"),
        "scope": row.get("qualification_scope", result.get("scope")),
        "record_id": row.get("record_id", result.get("record_id")),
        "blocking_reason": row.get("blocking_reason") or (result.get("reasons") or [None])[0],
        "next_required_action": row.get("next_required_action") or result.get("next_required_action"),
        "reasons": result.get("reasons") or [],
        "limitations": result.get("limitations") or [],
        "request": request,
    }
    return with_guidance(
        {"ok": positive, "explanation": explanation}, status=str(result.get("decision")),
        reason_codes=_reason_codes(result), problems=[] if positive else _problem_texts(result),
        warnings=result.get("limitations") or [], next_actions=_result_next_actions(result, request_path=args.request, mission_id=args.mission, catalog_path=args.catalog),
        related_records=[str(explanation.get("record_id") or "")],
    ), 0 if positive else 1


def _find_envelope(catalog: QualificationCatalogV1, envelope_id: str):
    for envelope in catalog.envelopes:
        if envelope.envelope_id == envelope_id:
            return envelope
    raise ActorError(f"unknown qualification envelope: {envelope_id}", code="unknown_qualification_envelope")


def command_plan(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    request = _load_request(args)
    generated_at = utc_now(args.generated_at)
    if args.catalog:
        if not args.envelope_id:
            raise ActorError("QF catalog plan requires --envelope-id", code="missing_envelope_id")
        catalog = _catalog(args.catalog)
        try:
            plan = plan_delta_campaign(request, _find_envelope(catalog, args.envelope_id))
            validation = validate_delta_plan(plan)
        except QualificationFrameworkError as exc:
            raise ActorError(str(exc), code="qualification_plan_failed") from exc
        source_refs = ({"kind": "qualification_catalog", "uri": str(Path(args.catalog).resolve())},)
    else:
        register_path = _register_path(args)
        resolution = resolve_authority_register(request, load_authority_register(register_path))
        if _positive(resolution):
            raise ActorError(
                "route already has current qualification authority; no new plan is justified",
                code="qualification_already_current",
                next_actions=[{
                    "description": "store the current authority resolution, then attach its QCHK record",
                    "command": f"python -m tools.eiel qualificationManager check --mission {args.mission} --store",
                }] if args.mission else [],
                related_records=[str(resolution.get("record_id"))],
            )
        plan = {
            "schema_version": "eiel_register_backed_qualification_plan_v1",
            "plan_class": "bounded_missing_authority",
            "request": request,
            "current_resolution": resolution,
            "required_checks": [
                "exact route identity and source/build binding",
                "deterministic repeat parity",
                "checkpoint and continuation integrity when stateful",
                "reference-route or reporting parity",
                "resource envelope confirmation",
            ],
            "required_evidence": resolution.get("next_required_evidence") or [resolution.get("next_required_action")],
            "execution_authority": "none_plan_only",
            "scientific_effect": "none",
            "stopping_rule": "stop after the declared technical checks; do not run a science campaign",
        }
        validation = {"ok": True, "errors": [], "plan_kind": "register_backed_bounded_plan"}
        source_refs = ({"kind": "qualification_authority_register", "uri": str(register_path), "sha256": file_sha256(register_path)},)
    plan_id = stable_id("QPLAN", plan)
    campaign_spec = _mission_campaign_spec(args)
    execution_handoff = {
        "owner": "existing_eiel_campaign_tooling",
        "campaign_spec": campaign_spec,
        "plan_only_command": (
            f"{CAMPAIGN_QUALIFICATION_CLI} {campaign_spec} --plan-only"
            if campaign_spec else None
        ),
        "execution_command": (
            f"{CAMPAIGN_QUALIFICATION_CLI} {campaign_spec}"
            if campaign_spec else None
        ),
        "manager_executes": False,
        "note": "Use the exact command emitted by campaign preflight/qualification receipts when available; never replace it with a bespoke launcher.",
    }
    payload = {
        "plan_id": plan_id,
        "plan": plan,
        "validation": validation,
        "execution_authority": "none_plan_only",
        "scientific_effect": "none",
        "integration_boundary": integration_boundary(),
        "execution_handoff": execution_handoff,
    }
    record = new_record(
        actor=ACTOR_NAME, record_type=PLAN_RECORD_TYPE, record_id=plan_id,
        status="PLAN_READY" if validation.get("ok") else "PLAN_INVALID", payload=payload,
        generated_at=generated_at, source_refs=source_refs, action="plan",
    )
    path = _store(args).put(record)
    return with_guidance(
        {"ok": bool(validation.get("ok")), "plan_id": plan_id, "record_path": str(path), **payload},
        status="PLAN_READY" if validation.get("ok") else "PLAN_INVALID",
        problems=validation.get("errors") or [],
        next_actions=([
            {
                "action": "generate_exact_campaign_qualification_plan",
                "description": "Use the existing campaign CLI to generate the provider-owned deterministic canary plan. Qualification Manager will not launch it.",
                "command": execution_handoff["plan_only_command"],
            }
        ] if execution_handoff["plan_only_command"] else [
            {
                "action": "bind_campaign_spec",
                "description": "Attach the approved campaign specification to the mission so the existing campaign CLI can emit the exact qualification command.",
                "command": f"python -m tools.eiel missionPlanner update {args.mission or '<mission-id>'} --campaign-spec-ref <campaign-spec>",
            }
        ]),
        related_records=[plan_id],
    ), 0 if validation.get("ok") else 1


def validate_qualification_plan(raw: Mapping[str, Any]) -> dict[str, Any]:
    schema = str(raw.get("schema_version") or "")
    try:
        if schema == "eiel_qualification_plan_v1":
            canonical = validate_declarative_plan(raw)
            return {"schema_version": "eiel_qualification_manager_plan_validation_v1", "ok": True, "plan_kind": "declarative_qualification_plan", "canonical_plan": canonical, "errors": []}
        if schema == "eiel_register_backed_qualification_plan_v1":
            errors = []
            if not isinstance(raw.get("request"), Mapping):
                errors.append("register-backed plan requires request")
            if not list(raw.get("required_checks") or []):
                errors.append("register-backed plan requires required_checks")
            if raw.get("execution_authority") != "none_plan_only":
                errors.append("qualification plan cannot own execution authority")
            return {"schema_version": "eiel_qualification_manager_plan_validation_v1", "ok": not errors, "plan_kind": "register_backed_bounded_plan", "canonical_plan": dict(raw), "errors": errors}
        canonical = validate_delta_plan(raw)
        return {"schema_version": "eiel_qualification_manager_plan_validation_v1", "ok": bool(canonical.get("ok")), "plan_kind": "qualification_delta_plan", "canonical_plan": canonical, "errors": list(canonical.get("errors") or [])}
    except (QualificationPlanError, QualificationFrameworkError) as exc:
        return {"schema_version": "eiel_qualification_manager_plan_validation_v1", "ok": False, "plan_kind": "unknown" if not schema else schema, "canonical_plan": None, "errors": [str(exc)]}


def command_validate(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    validation = validate_qualification_plan(read_structured(args.plan, name="qualification plan"))
    return with_guidance(
        {"ok": validation["ok"], "validation": validation}, status="VALID" if validation["ok"] else "BLOCKED",
        problems=validation["errors"], next_actions=[] if validation["ok"] else [{"description": "repair the plan before execution"}],
    ), 0 if validation["ok"] else 1


def command_decide(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    retained_inputs = []
    if args.manifest:
        retained_inputs.append(
            _retain_qualification_input(
                args.manifest,
                kind="campaign_evidence_manifest",
                state_root=args.state_root,
                generated_at=args.generated_at,
            )
        )
    if args.evidence:
        retained_inputs.append(
            _retain_qualification_input(
                args.evidence,
                kind="qualification_evidence_decision",
                state_root=args.state_root,
                generated_at=args.generated_at,
            )
        )
    try:
        decision = qualification_decision_payload(
            manifest_path=args.manifest, evidence_path=args.evidence, profile=args.profile,
            plan_sha256=args.plan_sha256, source_commit=args.source_commit, evidence_uri=args.evidence_uri,
        )
    except (QualificationFrameworkError, NotImplementedError) as exc:
        raise ActorError(str(exc), code="qualification_decision_failed") from exc
    for binding in decision.get("evidence_bindings") or []:
        if not isinstance(binding, Mapping):
            continue
        uri = str(binding.get("uri") or "").strip()
        if not uri:
            continue
        parsed = urlparse(uri)
        candidate: Path | None
        if parsed.scheme == "file":
            candidate = Path(unquote(parsed.path)).expanduser().resolve()
        elif not parsed.scheme:
            candidate = Path(uri).expanduser().resolve()
        else:
            candidate = None
        if candidate is not None and candidate.is_file():
            retained_inputs.append(
                _retain_qualification_input(
                    candidate,
                    kind=f"evidence_binding:{str(binding.get('kind') or 'unknown')}",
                    state_root=args.state_root,
                    generated_at=args.generated_at,
                )
            )
    decision_id = stable_id("QDEC", decision)
    decision_status = str(decision.get("decision_status") or (
        "QUALIFIED" if decision.get("accepted_for_envelope_expansion") else "NOT_QUALIFIED"
    ))
    record = new_record(
        actor=ACTOR_NAME, record_type=DECISION_RECORD_TYPE, record_id=decision_id,
        status=decision_status,
        payload={
            "decision_id": decision_id,
            "decision": decision,
            "retained_inputs": retained_inputs,
            "technical_only": True,
            "scientific_qualification_effect": "none",
            "campaign_execution_effect": "none",
        },
        generated_at=utc_now(args.generated_at),
        source_refs=tuple({"kind": kind, "uri": str(Path(path).resolve())} for kind, path in (("manifest", args.manifest), ("evidence", args.evidence)) if path),
        action="decide",
    )
    path = _store(args).put(record)
    positive = bool(decision.get("ok"))
    return with_guidance(
        {
            "ok": positive,
            "decision_id": decision_id,
            "record_path": str(path),
            "qualification_decision": decision,
            "retained_inputs": retained_inputs,
        },
        status=decision_status,
        reason_codes=["technical_qualification_accepted"] if positive else ["qualification_evidence_rejected"],
        problems=decision.get("errors") or [],
        next_actions=[] if positive else [{"description": "repair the technical evidence; do not widen qualification"}],
        related_records=[decision_id],
    ), 0 if positive else 1


def command_register(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    if args.catalog:
        catalog = _catalog(args.catalog)
        payload: dict[str, Any] = {"ok": True, "catalog": str(Path(args.catalog).resolve()), "catalog_sha256": canonical_sha256(catalog.payload()), "summary": catalog_summary(catalog)}
        if args.full:
            payload["catalog_payload"] = catalog.payload()
    else:
        path = _register_path(args)
        register = load_authority_register(path)
        payload = {"ok": True, "authority_register": str(path), "authority_register_sha256": file_sha256(path), "summary": authority_register_summary(register)}
        if args.full:
            payload["register_payload"] = register
    if args.include_manager_records:
        payload["manager_records"] = _store(args).list()
    return with_guidance(payload, status="REGISTER_READY"), 0


def command_audit(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    errors: list[str] = []
    warnings: list[str] = []
    if args.catalog:
        catalog = _catalog(args.catalog)
        payload: dict[str, Any] = {"source": "qf_catalog", "path": str(Path(args.catalog).resolve()), "summary": catalog_summary(catalog), "sha256": canonical_sha256(catalog.payload())}
        if args.authority_bundle:
            bundle = read_json(args.authority_bundle, name="qualification authority bundle")
            try:
                authority = validate_authority_bundle(bundle, catalog=catalog)
            except QualificationFrameworkError as exc:
                authority = {"ok": False, "errors": [str(exc)]}
            payload["authority_bundle_validation"] = authority
            if not authority.get("ok"):
                errors.extend(authority.get("errors") or ["authority bundle invalid"])
    else:
        path = _register_path(args)
        register = load_authority_register(path)
        tracks = [dict(row) for row in register.get("tracks") or [] if isinstance(row, Mapping)]
        labels = [str(row.get("label")) for row in tracks]
        duplicates = sorted(label for label, count in Counter(labels).items() if count > 1)
        if duplicates:
            errors.append(f"duplicate track labels: {duplicates}")
        semantic = []
        for track in tracks:
            issues = _track_consistency(track)
            if issues:
                semantic.append({"label": track.get("label"), "issues": issues})
        if semantic:
            warnings.append(f"{len(semantic)} track(s) have semantic consistency warnings")
        missing_reports = []
        for track in tracks:
            report = track.get("durable_report_path")
            if report and not (repository_root() / str(report)).is_file():
                missing_reports.append({"label": track.get("label"), "path": report})
        if missing_reports:
            warnings.append(f"{len(missing_reports)} durable report path(s) are absent from this snapshot")
        payload = {"source": "authority_register", "path": str(path), "summary": authority_register_summary(register), "sha256": file_sha256(path), "semantic_warnings": semantic, "missing_durable_reports": missing_reports}
    payload.update({"ok": not errors, "errors": errors, "warnings": warnings})
    return with_guidance(payload, status="AUDIT_PASS" if not errors else "AUDIT_BLOCKED", problems=errors, warnings=warnings), 0 if not errors else 1


def _path_impact(path: str) -> set[str]:
    normalized = path.replace("\\", "/").lower()
    families: set[str] = set()
    if "checkpoint" in normalized or "resume" in normalized:
        families.update({"checkpoint_resume", "continuation_compatibility", "deterministic_repeat"})
    if normalized.startswith("engine/env/"):
        families.update({"world_state_dimension_integrity", "ecology_mechanics_requalification"})
    if normalized.startswith("engine/agents/") or normalized.startswith("engine/evolution/"):
        families.update({"agent_genome_mechanics_requalification", "selection_and_reproduction_integrity"})
    if normalized.startswith("engine/perf/") or normalized.startswith("engine/native_runtime/"):
        families.update({"execution_route_identity", "cpu_cuda_parity", "resource_capacity_envelope"})
    if normalized.startswith("engine/campaigns/resources") or "resource_monitor" in normalized:
        families.update({"resource_capacity_envelope", "resource_admission_policy_review"})
    if normalized.startswith("tools/evidence/"):
        families.update({"evidence_reporting_integrity", "decision_profile_regression"})
    if "qualification" in normalized:
        families.update({"qualification_framework_authority", "qualification_register_regression"})
    if not families:
        families.add("explicit_path_impact_review")
    return families


def command_impact(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    axes = list(args.axis or [])
    paths = list(args.changed_path or [])
    if args.paths_file:
        paths.extend(line.strip() for line in Path(args.paths_file).read_text(encoding="utf-8").splitlines() if line.strip())
    if not axes and not paths:
        raise ActorError("impact requires at least one --axis, --changed-path, or --paths-file", code="impact_input_missing")
    families = set(impact_families(axes))
    per_path = []
    for path in sorted(set(paths)):
        affected = sorted(_path_impact(path))
        families.update(affected)
        per_path.append({"path": path, "impact_families": affected})
    payload = {
        "schema_version": IMPACT_SCHEMA_VERSION,
        "changed_axes": sorted(set(axes)),
        "changed_paths": per_path,
        "impact_families": sorted(families),
        "automatic_invalidation": False,
        "next_required_action": "review affected qualification records and create bounded delta plans where applicable",
        "scientific_effect": "none",
    }
    return with_guidance(
        {"ok": True, "impact": payload}, status="IMPACT_REVIEW_REQUIRED",
        next_actions=[{"description": payload["next_required_action"], "command": "python -m tools.eiel qualificationManager status"}],
    ), 0


def _stored_record_guidance(record: Mapping[str, Any]) -> tuple[list[str], list[str], list[dict[str, Any]], list[str]]:
    payload = dict(record.get("payload") or {})
    record_type = str(record.get("record_type") or "")
    status = str(record.get("status") or "UNKNOWN")
    problems: list[str] = []
    warnings: list[str] = []
    actions: list[dict[str, Any]] = []
    related: list[str] = [str(record.get("record_id"))]
    if record_type == CHECK_RECORD_TYPE:
        result = dict(payload.get("result") or {})
        positive = _positive(result)
        if not positive:
            problems.extend(str(item) for item in result.get("reasons") or [])
        warnings.extend(str(item) for item in result.get("limitations") or [])
        actions.extend(_result_next_actions(result))
        if result.get("record_id"):
            related.append(str(result["record_id"]))
    elif record_type == PLAN_RECORD_TYPE:
        validation = dict(payload.get("validation") or {})
        if not validation.get("ok", status == "PLAN_READY"):
            problems.extend(str(item) for item in validation.get("errors") or ["qualification plan is invalid"])
        else:
            actions.append({
                "description": "Execute this bounded technical plan through existing Codex campaign tooling, then import structured evidence.",
                "command": "python -m tools.eiel qualificationManager decide --evidence QUALIFICATION_EVIDENCE.json",
            })
    elif record_type == DECISION_RECORD_TYPE:
        decision = dict(payload.get("decision") or {})
        if not decision.get("accepted_for_envelope_expansion"):
            problems.extend(str(item) for item in decision.get("errors") or ["technical evidence did not grant qualification"])
            actions.append({"description": "Repair the declared evidence obligations; do not widen technical authority."})
    else:
        warnings.append(f"unrecognized qualification manager record type: {record_type}")
    return problems, warnings, actions, related


def command_show(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    record = _store(args).get(args.record_id)
    problems, warnings, actions, related = _stored_record_guidance(record)
    return with_guidance(
        {"ok": not problems, "record": record},
        status=str(record.get("status") or "UNKNOWN"),
        problems=problems, warnings=warnings, next_actions=actions, related_records=related,
    ), 0


def command_history(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    store = _store(args)
    record = store.get(args.record_id)
    rows = store.history(args.record_id)
    problems, warnings, actions, related = _stored_record_guidance(record)
    return with_guidance(
        {"ok": not problems, "record_id": args.record_id, "revision_count": len(rows), "history": rows, "record": record},
        status=str(record.get("status") or "UNKNOWN"),
        problems=problems, warnings=warnings, next_actions=actions, related_records=related,
    ), 0


def command_next(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    if args.record_id:
        return command_show(args)
    return command_explain(args)


def command_handoff(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    record = _store(args).get(args.record_id)
    payload = dict(record.get("payload") or {})
    problems, warnings, actions, related = _stored_record_guidance(record)
    handoff = {
        "schema_version": "eiel_codex_technical_qualification_handoff_v1",
        **operator_metadata(),
        "record_id": args.record_id,
        "record_type": record.get("record_type"),
        "status": record.get("status"),
        "request": payload.get("request"),
        "result": payload.get("result"),
        "plan": payload.get("plan"),
        "decision": payload.get("decision"),
        "source_refs": record.get("source_refs") or [],
        "record_sha256": record.get("record_sha256"),
        "technical_only": True,
        "scientific_qualification_effect": "none",
        "integration_boundary": integration_boundary(),
        "execution_handoff": payload.get("execution_handoff"),
    }
    return with_guidance(
        {"ok": not problems, "handoff": handoff},
        status=str(record.get("status") or "UNKNOWN"),
        problems=problems, warnings=warnings, next_actions=actions, related_records=related,
    ), 0


def _human(payload: Mapping[str, Any]) -> str:
    if isinstance(payload.get("qualification"), Mapping):
        row = payload["qualification"]
        return f"Qualification: {row.get('decision')}\nRecord: {row.get('record_id')}\nNext: {row.get('next_required_action')}"
    if isinstance(payload.get("explanation"), Mapping):
        row = payload["explanation"]
        return f"Qualification: {row.get('decision')}\nBlocking reason: {row.get('blocking_reason')}\nNext: {row.get('next_required_action')}"
    if isinstance(payload.get("qualification_decision"), Mapping):
        row = payload["qualification_decision"]
        return f"Technical decision: {row.get('decision')}\nAccepted: {row.get('accepted_for_envelope_expansion')}"
    if isinstance(payload.get("impact"), Mapping):
        return "Qualification impact families:\n" + "\n".join(f"- {item}" for item in payload["impact"].get("impact_families") or [])
    if isinstance(payload.get("summary"), Mapping):
        summary = payload["summary"]
        if "track_count" in summary:
            return f"Authority tracks: {summary['track_count']}\nStatuses: {summary['status_counts']}"
        return f"Anchors: {summary['anchors']['count']}\nEnvelopes: {summary['envelopes']['count']}\nResource envelopes: {summary['resource_capacity_envelopes']['count']}"
    return str(dict(payload))


def build_parser() -> argparse.ArgumentParser:
    parser = ActorArgumentParser(prog="python -m tools.eiel qualificationManager", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True, parser_class=ActorArgumentParser)

    status = sub.add_parser("status", help="Show the real authority register or resolve one exact request.")
    _common_options(status)
    _authority_options(status, request=True)

    check = sub.add_parser("check", help="Check whether a requested route/configuration has current technical authority.")
    _common_options(check)
    _authority_options(check, request=True)
    check.add_argument("--store", action="store_true")

    explain = sub.add_parser("explain", help="Explain a qualification result, limitations, and exact next action.")
    _common_options(explain)
    _authority_options(explain, request=True)

    plan = sub.add_parser("plan", help="Generate the smallest bounded missing-qualification plan without execution.")
    _common_options(plan)
    _authority_options(plan, request=True)
    plan.add_argument("--envelope-id")

    validate = sub.add_parser("validate", help="Validate a declarative, delta, or register-backed qualification plan.")
    _common_options(validate)
    validate.add_argument("--plan", required=True)

    decide = sub.add_parser("decide", help="Import bounded structured technical evidence into a technical decision record.")
    _common_options(decide)
    decide.add_argument("--manifest")
    decide.add_argument("--evidence")
    decide.add_argument("--profile")
    decide.add_argument("--plan-sha256")
    decide.add_argument("--source-commit")
    decide.add_argument("--evidence-uri")

    register = sub.add_parser("register", help="Inspect current qualification authority and manager records.")
    _common_options(register)
    _authority_options(register)
    register.add_argument("--full", action="store_true")
    register.add_argument("--include-manager-records", action="store_true")

    audit = sub.add_parser("audit", help="Audit the current register or a QF catalog and optional authority bundle.")
    _common_options(audit)
    _authority_options(audit)
    audit.add_argument("--authority-bundle")

    impact = sub.add_parser("impact", help="Classify technical qualification impacts without invalidating authority automatically.")
    _common_options(impact)
    impact.add_argument("--axis", action="append", default=[])
    impact.add_argument("--changed-path", action="append", default=[])
    impact.add_argument("--paths-file")

    show = sub.add_parser("show", help="Show one retained qualification check, plan, or decision record.")
    _common_options(show)
    show.add_argument("record_id")

    history = sub.add_parser("history", help="Show immutable history for one retained qualification record.")
    _common_options(history)
    history.add_argument("record_id")

    next_cmd = sub.add_parser("next", help="Show the next safe qualification action for a request or retained record.")
    _common_options(next_cmd)
    _authority_options(next_cmd, request=True)
    next_cmd.add_argument("record_id", nargs="?")

    handoff = sub.add_parser("handoff", help="Produce a compact Codex technical-qualification handoff.")
    _common_options(handoff)
    handoff.add_argument("record_id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    forbidden = {
        "run": "Qualification Manager creates and evaluates bounded plans; it never executes campaigns.",
        "execute": "Use Codex and the existing campaign tooling to execute qualification plans.",
        "launch": "Qualification Manager cannot launch processes or campaigns.",
        "qualify-prose": "Qualification requires structured, source-bound evidence rather than prose approval.",
    }
    if raw_args and raw_args[0] in forbidden:
        exc = forbidden_operation_error(f"eiel {ACTOR_NAME}", raw_args[0], forbidden[raw_args[0]])
        emit_actor_error(exc, actor=ACTOR_NAME, command=raw_args[0], raw_args=raw_args)
        return 2
    args: argparse.Namespace | None = None
    try:
        args = build_parser().parse_args(raw_args)
        handlers = {
            "status": command_status,
            "check": command_check,
            "explain": command_explain,
            "plan": command_plan,
            "validate": command_validate,
            "decide": command_decide,
            "register": command_register,
            "audit": command_audit,
            "impact": command_impact,
            "show": command_show,
            "history": command_history,
            "next": command_next,
            "handoff": command_handoff,
        }
        payload, code = handlers[args.command](args)
        emit_payload(payload, output=args.output, human_text=_human(payload), json_output=not args.human)
        return code
    except ActorError as exc:
        command = raw_args[0] if raw_args else "unknown"
        emit_actor_error(
            exc,
            actor=ACTOR_NAME,
            command=command,
            raw_args=raw_args,
            parsed_args=args,
        )
        return 2
    except (QualificationFrameworkError, QualificationPlanError, NotImplementedError, OSError) as exc:
        command = raw_args[0] if raw_args else "unknown"
        wrapped = ActorError(str(exc), code="qualification_operation_failed")
        emit_actor_error(
            wrapped,
            actor=ACTOR_NAME,
            command=command,
            raw_args=raw_args,
            parsed_args=args,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
