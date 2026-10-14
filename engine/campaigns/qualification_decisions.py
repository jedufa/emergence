"""Shared technical-qualification evidence import and decision contract.

Both the low-level ``tools.scripts.eiel_qualification`` CLI and the
Codex-facing ``qualificationManager`` use this module.  Keeping the decision
import here prevents two operator surfaces from drifting into different
meanings of technical qualification.

The contract is deliberately fail-closed:

* a bare prose or ``{"decision": "A"}`` object is never authority;
* an A/B/C/D result must retain structured plan/obligation bindings;
* campaign-manifest imports use the existing Evidence Decision Layer profile;
* accepted output is technical only and never grants scientific authority.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from engine.campaigns.canonical import canonical_sha256
from engine.campaigns.qualification_framework import QualificationFrameworkError
from tools.evidence.contracts import check_campaign_schema
from tools.evidence.decisions import analyze_profile

SCHEMA_VERSION = "eiel_qualification_decision_import_v1"


def read_json_object(path: str | Path, *, name: str) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QualificationFrameworkError(f"cannot read {name}: {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise QualificationFrameworkError(f"{name} must be a JSON object: {source}")
    return value


def stable_schema_status(schema_status: Mapping[str, Any]) -> dict[str, Any]:
    """Remove path-only fields while retaining deterministic run semantics."""

    stable = {
        key: value
        for key, value in dict(schema_status).items()
        if key not in {"checked_files", "missing_files", "run_status"}
    }
    rows: list[dict[str, Any]] = []
    for row in schema_status.get("run_status") or ():
        if not isinstance(row, Mapping):
            continue
        rows.append(
            {
                key: value
                for key, value in dict(row).items()
                if not key.endswith("_path") and not key.endswith("_dir")
            }
        )
    stable["run_status"] = sorted(
        rows,
        key=lambda item: (
            str(item.get("run_id", "")),
            str(item.get("scenario_id", "")),
            str(item.get("seed", "")),
        ),
    )
    return stable


def _structured_binding_present(evidence: Mapping[str, Any]) -> bool:
    return any(
        key in evidence
        for key in (
            "required_obligations",
            "plan_sha256",
            "required_obligations_sha256",
            "evidence_bindings",
        )
    )


def qualification_decision_payload(
    *,
    manifest_path: str | Path | None = None,
    evidence_path: str | Path | None = None,
    profile: str | None = None,
    plan_sha256: str | None = None,
    source_commit: str | None = None,
    evidence_uri: str | None = None,
) -> dict[str, Any]:
    """Import one bounded technical qualification decision.

    This function does not publish or expand an authority register.  It only
    produces a deterministic, evidence-bound decision payload for a later
    controlled authority workflow.
    """

    if manifest_path:
        manifest_source = Path(manifest_path).expanduser().resolve()
        manifest = read_json_object(manifest_source, name="campaign evidence manifest")
        schema_status = check_campaign_schema(manifest_source)
        profile_name = profile or str(schema_status.get("profile") or "")
        profile_decision = analyze_profile(schema_status, profile_name)
        required = [str(item) for item in manifest.get("qf_delta_required_obligations") or ()]
        satisfied = [str(item) for item in manifest.get("qf_delta_satisfied_obligations") or ()]
        missing = sorted(set(required) - set(satisfied))
        resolved_evidence_uri = evidence_uri or str(manifest_source)
        binding = {
            "schema_status": stable_schema_status(schema_status),
            "profile_decision": profile_decision,
            "evidence_uri": resolved_evidence_uri,
            "plan_sha256": plan_sha256,
            "source_commit": source_commit,
            "required_obligations_sha256": manifest.get("qf_delta_required_obligations_sha256"),
            "satisfied_recipe_obligations": satisfied,
        }
        decision = str(profile_decision.get("decision") or "")
        accepted = bool(schema_status.get("ok")) and decision == "A" and bool(required) and not missing
        errors = list(schema_status.get("errors") or [])
        if not required:
            errors.append("manifest missing qf_delta_required_obligations")
        if missing:
            errors.append(f"manifest missing satisfied delta obligations: {missing}")
        return {
            "schema_version": SCHEMA_VERSION,
            "ok": accepted,
            "profile": profile_name,
            "decision": decision,
            "accepted_for_envelope_expansion": accepted,
            "no_post_hoc_widening": True,
            "plan_sha256": plan_sha256,
            "source_commit": source_commit,
            "required_obligations_sha256": manifest.get("qf_delta_required_obligations_sha256"),
            "satisfied_recipe_obligations": satisfied,
            "evidence_sha256": canonical_sha256(binding),
            "evidence_bindings": [
                {
                    "evidence_id": "qual-delta-edl-decision",
                    "kind": "evidence_decision_layer",
                    "uri": resolved_evidence_uri,
                    "sha256": canonical_sha256(binding),
                }
            ],
            "schema_ok": bool(schema_status.get("ok")),
            "profile_decision": profile_decision,
            "errors": [] if accepted else errors,
            "technical_qualification_effect": "bounded_decision_only",
            "scientific_qualification_effect": "none",
        }

    if not evidence_path:
        raise QualificationFrameworkError("decide requires either --manifest or --evidence")

    evidence_source = Path(evidence_path).expanduser().resolve()
    evidence = read_json_object(evidence_source, name="qualification evidence decision")
    decision = str(evidence.get("decision") or "")
    if decision not in {"A", "B", "C", "D"}:
        return {
            "schema_version": SCHEMA_VERSION,
            "ok": False,
            "decision": decision,
            "accepted_for_envelope_expansion": False,
            "errors": ["qualification evidence decision must be A, B, C, or D"],
            "technical_qualification_effect": "none",
            "scientific_qualification_effect": "none",
        }
    if not _structured_binding_present(evidence):
        return {
            "schema_version": SCHEMA_VERSION,
            "ok": False,
            "decision": decision,
            "accepted_for_envelope_expansion": False,
            "no_post_hoc_widening": True,
            "errors": [
                "structured qualification evidence bindings are missing; "
                "prose or a bare decision is insufficient"
            ],
            "technical_qualification_effect": "none",
            "scientific_qualification_effect": "none",
        }

    required = [str(item) for item in evidence.get("required_obligations") or ()]
    satisfied = [str(item) for item in evidence.get("satisfied_recipe_obligations") or ()]
    missing = sorted(set(required) - set(satisfied))
    accepted = decision == "A" and (not required or not missing)
    return {
        "schema_version": SCHEMA_VERSION,
        "ok": accepted,
        "decision": decision,
        "accepted_for_envelope_expansion": accepted,
        "no_post_hoc_widening": True,
        "source_commit": source_commit or evidence.get("source_commit"),
        "plan_sha256": evidence.get("plan_sha256"),
        "required_obligations_sha256": evidence.get("required_obligations_sha256"),
        "satisfied_recipe_obligations": satisfied,
        "evidence_sha256": canonical_sha256(evidence),
        "evidence_bindings": [dict(row) for row in evidence.get("evidence_bindings") or [] if isinstance(row, Mapping)],
        "limitations": [str(item) for item in evidence.get("limitations") or ()],
        "errors": [] if accepted else (
            [f"missing satisfied delta obligations: {missing}"]
            if missing
            else ["non-A decision does not expand qualification"]
        ),
        "technical_qualification_effect": "bounded_decision_only" if accepted else "none",
        "scientific_qualification_effect": "none",
    }
