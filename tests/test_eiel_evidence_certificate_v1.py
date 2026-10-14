from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any, Mapping

import pytest

from engine.campaigns import evidence_certificate as certificate_module
from engine.campaigns import permanent_evidence_capsule as capsule_module
from engine.campaigns.artifact_lifecycle import (
    ArtifactLifecycleError,
    DEBUG_FORENSIC,
    SCIENCE_COMPACT,
    build_artifact_disposition_manifest_v1,
    build_sealed_artifact_inventory_row_v1,
    build_storage_disposition_receipt_v1,
    lifecycle_profile_contract_v1,
    validate_artifact_disposition_manifest_v1,
    validate_run_artifact_lifecycle_contract_v1,
)
from engine.campaigns.evidence_certificate import (
    DIAGNOSTIC_UPGRADE_SPEC_SCHEMA_VERSION,
    EvidenceCertificateError,
    build_evidence_certificate_v1,
    build_reclamation_intent_v1,
    certificate_disposition_manifest_v1,
    diagnostic_local_live_state_audit_v1,
    finalize_evidence_certificate_v1,
    upgrade_evidence_certificate_request_source_bindings_v1,
    validate_certificate_source_manifest_bindings_v1,
    validate_evidence_certificate_v1,
    validate_evidence_certificate_request_v1,
    validate_storage_receipt_certificate_binding_v1,
    utc_now,
)
from engine.campaigns.permanent_evidence_capsule import (
    DEFAULT_MAX_CAPSULE_BYTES,
    build_permanent_evidence_capsule_v1,
    publish_permanent_evidence_capsule_v1,
    validate_permanent_evidence_capsule_v1,
    validate_permanent_evidence_publication_v1,
)
from engine.campaigns.canonical import canonical_sha256
from tools.scripts import eiel_runtime_storage_prune as prune_module
from tools.scripts import eiel_evidence_certificate_reclaim as reclaim_module
from tools.scripts.eiel_evidence_certificate_reclaim import (
    _apply,
    _dry_run,
    historical_audit,
)
from tools.scripts.eiel_runtime_storage_prune import (
    apply_prune_plan,
    build_prune_plan,
)


def _contract() -> dict:
    contract = lifecycle_profile_contract_v1(
        SCIENCE_COMPACT,
        total_max_bytes=10_000_000,
        checkpoint_schedule=[1],
    )
    contract["retention"]["unknown_artifact_scope"] = "exact_object"
    contract.pop("contract_sha256", None)
    return validate_run_artifact_lifecycle_contract_v1(contract)


def _debug_contract() -> dict:
    contract = lifecycle_profile_contract_v1(
        DEBUG_FORENSIC,
        total_max_bytes=10_000_000,
        checkpoint_schedule=[1],
    )
    contract["retention"]["unknown_artifact_scope"] = "exact_object"
    contract.pop("contract_sha256", None)
    return validate_run_artifact_lifecycle_contract_v1(contract)


def _artifact(
    artifact_id: str,
    path: str,
    *,
    evidence_class: str = "RAW_TRANSIENT_EVIDENCE",
    consumers: list[str] | None = None,
    last_use: str | None = None,
    live: str | None = None,
    forensic_status: str = "not_applicable",
    authorities: list[str] | None = None,
    policy: str = "KNOWN",
    materialization_group: str | None = None,
) -> dict:
    live_state = {
        "writer": False,
        "reference": False,
        "resume": False,
        "reducer": False,
        "consumer": False,
    }
    if live is not None:
        live_state[live] = True
    row = {
        "artifact_id": artifact_id,
        "path": path,
        "artifact_family": "gate-evidence",
        "owner_writer": "fixture-writer",
        "evidence_class": evidence_class,
        "reproduction_role": "raw-proof-input",
        "produced_by": "G1",
        "consumers": list(consumers or []),
        "last_use": last_use,
        "live_state": live_state,
        "forensic_hold": {
            "status": forensic_status,
            "closure_receipt_sha256": (
                "f" * 64 if forensic_status == "closed" else None
            ),
        },
        "retention_authorities": list(authorities or []),
        "policy": policy,
        "dependency_state": "known",
    }
    if materialization_group is not None:
        row["materialization_group"] = materialization_group
    return row


def _request(
    artifacts: list[dict],
    *,
    ordered: list[str] | None = None,
    closed: list[str] | None = None,
    gate_id: str = "G1",
    qualification_status: str = "not_applicable",
    qualification_ids: list[str] | None = None,
) -> dict:
    ordered = list(ordered or ["G1", "G2"])
    closed = list(closed or [gate_id])
    plan_id = "programme-plan-fixture"
    plan_sha256 = canonical_sha256(
        {
            "programme_id": "certificate-fixture",
            "plan_id": plan_id,
            "ordered_gates": ordered,
        }
    )
    closure_receipt_core = {
        "schema_version": "eiel_programme_gate_closure_receipt_v1",
        "programme_id": "certificate-fixture",
        "plan_id": plan_id,
        "plan_sha256": plan_sha256,
        "gate_id": gate_id,
        "closed_gates": closed,
        "status": "CLOSED",
        "decision_receipt_sha256": "8" * 64,
    }
    closure_receipt = {
        **closure_receipt_core,
        "receipt_sha256": canonical_sha256(closure_receipt_core),
    }
    return {
        "schema_version": "eiel_evidence_certificate_request_v1",
        "programme_id": "certificate-fixture",
        "work_id": "work-certificate-fixture",
        "gate_id": gate_id,
        "obligation_id": "obligation-proof",
        "bounded_claim": "the bounded fixture gate satisfies its declared criterion",
        "sealed_at_utc": "2026-08-23T12:00:00Z",
        "identity": {
            "source": {
                "repository": "/repo/emergence",
                "commit_sha": "1" * 40,
            },
            "artifact_hashes": {},
            "runtime": {"route": "host-fixture-v1"},
            "config_sha256": "2" * 64,
            "request_sha256": "3" * 64,
            "plan_sha256": plan_sha256,
            "environment": {"python": "3.14.4"},
            "device": {"kind": "host"},
            "rng_bindings": [],
            "rng_not_applicable_reason": "deterministic storage fixture",
        },
        "programme_plan": {
            "plan_id": plan_id,
            "plan_sha256": plan_sha256,
            "closure_receipt_sha256": closure_receipt["receipt_sha256"],
            "closure_receipt": closure_receipt,
            "ordered_gates": ordered,
            "closed_gates": closed,
        },
        "artifact_lifecycle_contract": _contract(),
        "governing_procedures": ["EIEL-ALCM-1", "EIEL-ESAB-1"],
        "evidence_obligations": [
            {
                "obligation_id": "obligation-proof",
                "description": "mechanical and semantic proof",
                "status": "satisfied",
            }
        ],
        "validation": {
            "mechanical_receipts": [
                {"receipt_id": "validator-1", "status": "passed", "sha256": "6" * 64}
            ],
            "semantic_review": {
                "status": "accepted",
                "reviewer": "same-identity-reduced-assurance",
            },
        },
        "result": {
            "decision": "ACCEPTED",
            "criteria": [{"criterion": "fixture exactness", "status": "passed"}],
            "measurements": {"objects": len(artifacts)},
            "digests": {"result": "7" * 64},
            "limitations": ["host-side fixture only"],
            "no_claims": ["no scientific or formal qualification claim"],
        },
        "reproduction": {
            "prerequisites": ["source commit and Python 3.14"],
            "commands": ["pytest -q tests/test_eiel_evidence_certificate_v1.py"],
            "verification_rules": ["all focused certificate tests pass"],
            "expected_results": [{"name": "pytest", "status": "passed"}],
            "guarantee": "REVALIDATION_ONLY",
        },
        "qualification_context": {
            "status": qualification_status,
            "authority_reference": (
                "qualification-fixture" if qualification_status == "active" else None
            ),
            "exact_required_artifact_ids": list(qualification_ids or []),
        },
        "audit_context": {
            "status": "released",
            "authority_reference": "semantic-review-fixture",
            "exact_required_artifact_ids": [],
        },
        "inventory_complete": True,
        "artifacts": artifacts,
    }


def _write_artifacts(root: Path, rows: list[dict]) -> None:
    for index, row in enumerate(rows):
        path = root / row["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((f"artifact-{index}-" + row["artifact_id"]).encode())


def _write_source_manifest(
    root: Path,
    request: Mapping[str, Any],
    *,
    name: str = "source-disposition.json",
    retention_classes: Mapping[str, str] | None = None,
    artifact_ids: set[str] | None = None,
) -> tuple[dict[str, Any], Path]:
    rows = [
        row
        for row in request["artifacts"]
        if artifact_ids is None or row["artifact_id"] in artifact_ids
    ]
    classes = dict(retention_classes or {})
    inventory = [
        build_sealed_artifact_inventory_row_v1(
            run_root=root,
            path=root / row["path"],
            artifact_family=row["artifact_family"],
            retention_class=classes.get(row["artifact_id"], "R1"),
            owner_writer=row["owner_writer"],
        )
        for row in rows
    ]
    manifest = build_artifact_disposition_manifest_v1(
        {
            "campaign_id": "source-fixture",
            "plan_sha256": "d" * 64,
            "artifact_lifecycle": request["artifact_lifecycle_contract"],
        },
        run_root=root,
        terminal_status="completed",
        live_reference_status="none",
        audit_hold_status="not_required",
        qualification_hold_status="not_required",
        artifact_inventory=inventory,
        inventory_complete=True,
    )
    path = root / "control" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest, path


def _combined_bound_fixture(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    source_root = root / "source-run"
    request = _request([_artifact("raw-a", "source-run/raw/a.bin")])
    _write_artifacts(root, request["artifacts"])
    inventory = [
        build_sealed_artifact_inventory_row_v1(
            run_root=source_root,
            path=root / request["artifacts"][0]["path"],
            artifact_family=request["artifacts"][0]["artifact_family"],
            retention_class="R1",
            owner_writer=request["artifacts"][0]["owner_writer"],
        )
    ]
    source_manifest = build_artifact_disposition_manifest_v1(
        {
            "campaign_id": "source-fixture",
            "plan_sha256": "d" * 64,
            "artifact_lifecycle": request["artifact_lifecycle_contract"],
        },
        run_root=source_root,
        terminal_status="completed",
        live_reference_status="none",
        audit_hold_status="not_required",
        qualification_hold_status="not_required",
        artifact_inventory=inventory,
        inventory_complete=True,
    )
    source_path = source_root / "artifact_disposition_manifest.json"
    source_path.write_text(json.dumps(source_manifest, indent=2, sort_keys=True) + "\n")
    upgraded = upgrade_evidence_certificate_request_source_bindings_v1(
        request,
        evidence_root=root,
        source_manifest_paths=[source_path],
    )
    return upgraded, source_manifest, source_path, source_root


def _debug_request(
    root: Path,
    artifacts: list[dict],
    *,
    source_classes: Mapping[str, str] | None = None,
    closeout_dispositions: Mapping[str, str] | None = None,
    source_live_reference_status: str = "active",
    source_audit_hold_status: str = "not_required",
    source_qualification_hold_status: str = "not_required",
    source_row_statuses: Mapping[str, Mapping[str, str]] | None = None,
    include_source_live_marker_closeout: bool = True,
) -> tuple[dict, dict]:
    """Build a real-shape per-run DEBUG manifest and its exact closeout request."""

    _write_artifacts(root, artifacts)
    contract = _debug_contract()
    classes = dict(source_classes or {})
    row_statuses = dict(source_row_statuses or {})
    inventory = [
        build_sealed_artifact_inventory_row_v1(
            run_root=root,
            path=root / row["path"],
            artifact_family=row["artifact_family"],
            retention_class=classes.get(row["artifact_id"], "R1"),
            owner_writer=row["owner_writer"],
            **dict(row_statuses.get(row["artifact_id"], {})),
        )
        for row in artifacts
    ]
    source_plan_sha256 = "a" * 64
    source_campaign_id = "twbb-v3-baseline-r18"
    source_manifest = build_artifact_disposition_manifest_v1(
        {
            "campaign_id": source_campaign_id,
            "plan_sha256": source_plan_sha256,
            "artifact_lifecycle": contract,
        },
        run_root=root,
        terminal_status="completed",
        live_reference_status=source_live_reference_status,
        audit_hold_status=source_audit_hold_status,
        qualification_hold_status=source_qualification_hold_status,
        artifact_inventory=inventory,
        inventory_complete=True,
    )
    source_path = root / "control" / "source_artifact_disposition_manifest.json"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_by_path = {row["path"]: row for row in source_manifest["artifacts"]}
    dispositions = dict(closeout_dispositions or {})
    superseded_ids: list[str] = []
    for row in artifacts:
        source_row = source_by_path[row["path"]]
        action = dispositions.get(row["artifact_id"], "SUPERSEDED_BY_CERTIFICATE")
        row.update(
            {
                "source_artifact_id": source_row["artifact_id"],
                "source_retention_class": source_row["retention_class"],
                "diagnostic_closeout_disposition": action,
                "source_manifest_binding": {
                    "manifest_path": str(source_path.relative_to(root)),
                    "manifest_sha256": source_manifest["manifest_sha256"],
                    "artifact_id": source_row["artifact_id"],
                    "bytes": source_row["bytes"],
                    "retention_class": source_row["retention_class"],
                    **(
                        {"sha256": source_row["sha256"]}
                        if "sha256" in source_row
                        else {"tree_sha256": source_row["tree_sha256"]}
                    ),
                },
            }
        )
        if action == "SUPERSEDED_BY_CERTIFICATE":
            superseded_ids.append(row["artifact_id"])
    request = _request(artifacts)
    request["artifact_lifecycle_contract"] = contract
    request["diagnostic_closeout"] = {
        "source_manifest": {
            "path": str(source_path.relative_to(root)),
            "manifest_sha256": source_manifest["manifest_sha256"],
            "campaign_id": source_manifest["campaign_id"],
            "run_id": source_manifest["run_id"],
            "plan_sha256": source_manifest["plan_sha256"],
            "lifecycle_contract_sha256": source_manifest["lifecycle_contract_sha256"],
            **(
                {"runtime_root": source_manifest["runtime_root"]}
                if include_source_live_marker_closeout
                else {}
            ),
        },
        "substantive_review": {
            "status": "accepted_with_limitations",
            "reviewer": "same-identity-reduced-assurance",
            "receipt_sha256": "b" * 64,
        },
        "closure_receipt_sha256": "c" * 64,
        "conclusion": "the bounded diagnostic is closed and its exact raw rows are superseded",
        "limitations": ["host-side storage fixture; no new science claim"],
        "unresolved_anomalies": [],
        "live_state": {
            "writer": False,
            "reference": False,
            "resume": False,
            "reducer": False,
            "consumer": False,
        },
        "active_exact_byte_holds": {
            "forensic": [],
            "audit": [],
            "qualification": [],
            "replay": [],
        },
        "superseded_artifact_ids": sorted(superseded_ids),
        **(
            {
                "source_live_marker_closeout": {
                    "schema_version": "eiel_debug_source_live_marker_closeout_v1",
                    "status": "CLOSED_BY_CURRENT_STATE_AUDIT",
                    "audited_at_utc": utc_now(),
                    "source_manifest_sha256": source_manifest["manifest_sha256"],
                    "source_campaign_id": source_manifest["campaign_id"],
                    "source_run_id": source_manifest["run_id"],
                    "source_runtime_root": source_manifest["runtime_root"],
                    "selected_artifact_count": len(superseded_ids),
                    "selected_artifact_ids_sha256": canonical_sha256(
                        sorted(superseded_ids)
                    ),
                    "checks": {
                        "writer": False,
                        "reference": False,
                        "resume": False,
                        "reducer": False,
                        "consumer": False,
                        "canonical_reservation_overlap": False,
                        "relay_process_references": [],
                    },
                    "scope": "SOURCE_MANIFEST_LIVE_REFERENCE_MARKER_ONLY",
                    "audit_method": "exact fixture process, relay, and reservation audit",
                }
            }
            if include_source_live_marker_closeout
            else {}
        ),
    }
    return request, source_manifest


def _certificate(root: Path, request: dict) -> dict:
    _write_artifacts(root, request["artifacts"])
    return build_evidence_certificate_v1(request, evidence_root=root)


def _certificate_prune_context(
    request: dict, certificate: dict, manifest: dict, plan: dict
) -> dict:
    intent = build_reclamation_intent_v1(
        request=request,
        certificate=certificate,
        manifest=manifest,
        selected=plan["selected"],
        status="APPLYING",
        created_at_utc="2026-08-23T12:00:01Z",
    )
    return {"request": request, "certificate": certificate, "intent": intent}


def _cli_args(
    root: Path, request: dict, name: str = "gate-close"
) -> argparse.Namespace:
    request_path = root / f"{name}-request.json"
    request_path.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n")
    return argparse.Namespace(
        request=request_path,
        evidence_root=root,
        output_root=root / "control" / name,
        permanent_root=None,
        max_permanent_capsule_bytes=DEFAULT_MAX_CAPSULE_BYTES,
        max_permanent_root_bytes=512 * 1024 * 1024,
        apply=False,
    )


def _configure_near_cap_permanent_root(
    root: Path,
    request: Mapping[str, Any],
    args: argparse.Namespace,
    *,
    label: str,
) -> tuple[Path, int]:
    certificate = build_evidence_certificate_v1(request, evidence_root=root)
    measurement_root = root.parent / f"{root.name}-{label}-measurement"
    measurement = publish_permanent_evidence_capsule_v1(
        certificate,
        permanent_root=measurement_root,
        source_root=root,
    )
    root_limit = int(measurement["total_bytes"]) + DEFAULT_MAX_CAPSULE_BYTES + 4096
    _, reservation_bytes, _ = capsule_module._family_reservation(
        certificate,
        capsule_limit=DEFAULT_MAX_CAPSULE_BYTES,
        root_limit=root_limit,
    )
    filler_bytes = (
        root_limit
        - int(measurement["total_bytes"])
        - DEFAULT_MAX_CAPSULE_BYTES
        - len(reservation_bytes)
    )
    assert filler_bytes > 0
    permanent_root = root.parent / f"{root.name}-{label}-permanent"
    permanent_root.mkdir()
    (permanent_root / "existing-evidence.bin").write_bytes(b"x" * filler_bytes)
    args.permanent_root = permanent_root
    args.max_permanent_root_bytes = root_limit
    return permanent_root, root_limit


def test_01_deterministic_certificate_serialization(tmp_path: Path) -> None:
    request = _request([_artifact("raw-a", "raw/a.bin")])
    _write_artifacts(tmp_path, request["artifacts"])
    first = build_evidence_certificate_v1(request, evidence_root=tmp_path)
    second = build_evidence_certificate_v1(request, evidence_root=tmp_path)
    assert first == second


def test_02_certificate_integrity_hash_binding(tmp_path: Path) -> None:
    certificate = _certificate(tmp_path, _request([_artifact("raw-a", "raw/a.bin")]))
    validate_evidence_certificate_v1(certificate)
    tampered = copy.deepcopy(certificate)
    tampered["bounded_claim"] = "tampered"
    with pytest.raises(EvidenceCertificateError, match="SHA-256 mismatch"):
        validate_evidence_certificate_v1(tampered)


def test_03_complete_inventory_reduction_has_exact_hashes_and_bytes(
    tmp_path: Path,
) -> None:
    request = _request(
        [_artifact("raw-a", "raw/a.bin"), _artifact("raw-b", "raw/b.bin")]
    )
    certificate = _certificate(tmp_path, request)
    assert certificate["inventory_complete"] is True
    assert len(certificate["artifacts"]) == 2
    assert all(
        row.get("sha256") and row["bytes"] > 0 for row in certificate["artifacts"]
    )
    assert certificate["disposition_summary"]["produced_bytes"] == sum(
        row["bytes"] for row in certificate["artifacts"]
    )


def test_04_release_refused_before_certificate_completeness(tmp_path: Path) -> None:
    request = _request([_artifact("raw-a", "raw/a.bin")])
    _write_artifacts(tmp_path, request["artifacts"])
    request["inventory_complete"] = False
    with pytest.raises(EvidenceCertificateError, match="complete producer inventory"):
        build_evidence_certificate_v1(request, evidence_root=tmp_path)


def test_05_release_refused_with_future_consumer(tmp_path: Path) -> None:
    request = _request(
        [_artifact("dependency", "raw/dependency.bin", consumers=["G2"], last_use="G2")]
    )
    certificate = _certificate(tmp_path, request)
    row = certificate["artifacts"][0]
    assert row["disposition"] == "HOLD_UNTIL_LAST_USE"
    assert row["release_eligible"] is False


def test_06_release_after_last_consumer(tmp_path: Path) -> None:
    request = _request(
        [
            _artifact(
                "dependency", "raw/dependency.bin", consumers=["G2"], last_use="G2"
            )
        ],
        gate_id="G2",
        closed=["G1", "G2"],
    )
    certificate = _certificate(tmp_path, request)
    assert certificate["artifacts"][0]["disposition"] == "RELEASE_NOW"


def test_07_specific_dependency_retained_while_sibling_releases(tmp_path: Path) -> None:
    request = _request(
        [
            _artifact("sibling", "raw/sibling.bin"),
            _artifact(
                "dependency", "raw/dependency.bin", consumers=["G2"], last_use="G2"
            ),
        ]
    )
    certificate = _certificate(tmp_path, request)
    rows = {row["artifact_id"]: row for row in certificate["artifacts"]}
    assert rows["sibling"]["release_eligible"] is True
    assert rows["dependency"]["release_eligible"] is False


def test_08_forensic_hold_blocks_deletion(tmp_path: Path) -> None:
    request = _request(
        [
            _artifact(
                "failure-trace",
                "raw/failure.bin",
                evidence_class="FORENSIC_HOLD",
                forensic_status="active",
            )
        ]
    )
    certificate = _certificate(tmp_path, request)
    assert certificate["artifacts"][0]["disposition"] == "FORENSIC_HOLD"


def test_09_forensic_hold_releases_after_closure_receipt(tmp_path: Path) -> None:
    request = _request(
        [
            _artifact(
                "failure-trace",
                "raw/failure.bin",
                evidence_class="FORENSIC_HOLD",
                forensic_status="closed",
            )
        ]
    )
    certificate = _certificate(tmp_path, request)
    assert certificate["artifacts"][0]["release_eligible"] is True


def test_10_retain_asset_survives_cleanup(tmp_path: Path) -> None:
    request = _request(
        [
            _artifact(
                "asset",
                "raw/population.bin",
                evidence_class="SCIENTIFIC_ENGINEERING_ASSET",
            ),
            _artifact("raw-a", "raw/a.bin"),
        ]
    )
    certificate = _certificate(tmp_path, request)
    manifest = certificate_disposition_manifest_v1(
        request, certificate, evidence_root=tmp_path
    )
    plan = build_prune_plan(tmp_path, set(prune_module.CATEGORIES), manifest)
    result = apply_prune_plan(
        tmp_path,
        plan,
        disposition_manifest=manifest,
        certificate_release_context=_certificate_prune_context(
            request, certificate, manifest, plan
        ),
    )
    assert not (tmp_path / "raw/a.bin").exists()
    assert (tmp_path / "raw/population.bin").is_file()
    assert result["failures"] == []


def test_11_unknown_policy_fails_closed(tmp_path: Path) -> None:
    request = _request(
        [
            _artifact("unknown", "raw/unknown.bin", policy="UNKNOWN_POLICY"),
            _artifact("known", "raw/known.bin"),
        ]
    )
    certificate = _certificate(tmp_path, request)
    rows = {row["artifact_id"]: row for row in certificate["artifacts"]}
    assert rows["unknown"]["retention_class"] == "RX"
    assert rows["unknown"]["release_eligible"] is False
    assert rows["known"]["release_eligible"] is True


@pytest.mark.parametrize(
    "live", ["writer", "reference", "resume", "reducer", "consumer"]
)
def test_12_live_writer_or_reference_blocks_deletion(tmp_path: Path, live: str) -> None:
    request = _request([_artifact("live", "raw/live.bin", live=live)])
    certificate = _certificate(tmp_path, request)
    assert certificate["artifacts"][0]["disposition"] == "HOLD_UNTIL_LAST_USE"


def test_13_interrupted_staged_disposition_is_recoverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request([_artifact("raw-a", "raw/a.bin")])
    certificate = _certificate(tmp_path, request)
    manifest = certificate_disposition_manifest_v1(
        request, certificate, evidence_root=tmp_path
    )
    plan = build_prune_plan(tmp_path, set(prune_module.CATEGORIES), manifest)
    real_unlink = prune_module.os.unlink
    failed = False

    def interrupt_once(path, *args, **kwargs):
        nonlocal failed
        if kwargs.get("dir_fd") is not None and not failed:
            failed = True
            raise OSError("simulated interruption after atomic staging")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(prune_module.os, "unlink", interrupt_once)
    first = apply_prune_plan(
        tmp_path,
        plan,
        disposition_manifest=manifest,
        certificate_release_context=_certificate_prune_context(
            request, certificate, manifest, plan
        ),
    )
    assert first["failures"]
    assert not (tmp_path / "raw/a.bin").exists()
    monkeypatch.setattr(prune_module.os, "unlink", real_unlink)
    second = apply_prune_plan(
        tmp_path,
        plan,
        disposition_manifest=manifest,
        certificate_release_context=_certificate_prune_context(
            request, certificate, manifest, plan
        ),
    )
    assert second["recovered_staged"] == ["raw/a.bin"]
    assert second["failures"] == []


def test_14_caller_cannot_supply_release_staging_authority(tmp_path: Path) -> None:
    request = _request([_artifact("raw-a", "raw/a.bin")])
    certificate = _certificate(tmp_path, request)
    manifest = certificate_disposition_manifest_v1(
        request, certificate, evidence_root=tmp_path
    )
    plan = build_prune_plan(tmp_path, set(prune_module.CATEGORIES), manifest)
    with pytest.raises(TypeError, match="staging_root"):
        apply_prune_plan(
            tmp_path,
            plan,
            disposition_manifest=manifest,
            staging_root=tmp_path / "caller-controlled-stage",  # type: ignore[call-arg]
            certificate_release_context=_certificate_prune_context(
                request, certificate, manifest, plan
            ),
        )
    assert (tmp_path / "raw/a.bin").is_file()


def test_15_in_place_mutation_after_certificate_is_refused(tmp_path: Path) -> None:
    request = _request([_artifact("raw-a", "raw/a.bin")])
    certificate = _certificate(tmp_path, request)
    manifest = certificate_disposition_manifest_v1(
        request, certificate, evidence_root=tmp_path
    )
    plan = build_prune_plan(tmp_path, set(prune_module.CATEGORIES), manifest)
    (tmp_path / "raw/a.bin").write_bytes(b"mutated-after-certificate")
    with pytest.raises(RuntimeError, match="differs from manifest"):
        apply_prune_plan(
            tmp_path,
            plan,
            disposition_manifest=manifest,
            certificate_release_context=_certificate_prune_context(
                request, certificate, manifest, plan
            ),
        )
    assert (tmp_path / "raw/a.bin").is_file()


def test_16_symlink_substitution_and_traversal_are_refused(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.bin"
    outside.write_bytes(b"outside")
    link = tmp_path / "raw/link.bin"
    link.parent.mkdir()
    link.symlink_to(outside)
    request = _request([_artifact("link", "raw/link.bin")])
    with pytest.raises(Exception, match="symlink"):
        build_evidence_certificate_v1(request, evidence_root=tmp_path)
    traversal = _request([_artifact("escape", "../outside.bin")])
    with pytest.raises(EvidenceCertificateError, match="safe relative path"):
        build_evidence_certificate_v1(traversal, evidence_root=tmp_path)
    assert outside.read_bytes() == b"outside"


def test_17_ordinary_release_never_invokes_archive(tmp_path: Path) -> None:
    request = _request([_artifact("raw-a", "raw/a.bin")])
    _write_artifacts(tmp_path, request["artifacts"])
    args = _cli_args(tmp_path, request)
    dry = _dry_run(args)
    applied = _apply(args)
    assert dry["archive_invoked"] is False
    assert applied["archive_invoked"] is False


def test_18_mid_programme_gate_close_reclaims_before_programme_complete(
    tmp_path: Path,
) -> None:
    request = _request([_artifact("raw-a", "raw/a.bin")], ordered=["G1", "G2", "G3"])
    _write_artifacts(tmp_path, request["artifacts"])
    args = _cli_args(tmp_path, request)
    _dry_run(args)
    result = _apply(args)
    assert result["programme_complete"] is False
    assert result["released_count"] == 1
    assert not (tmp_path / "raw/a.bin").exists()


def test_19_later_gate_consumes_hold_then_triggers_release(tmp_path: Path) -> None:
    artifact = _artifact(
        "dependency", "raw/dependency.bin", consumers=["G2"], last_use="G2"
    )
    request_g1 = _request([artifact])
    _write_artifacts(tmp_path, request_g1["artifacts"])
    args_g1 = _cli_args(tmp_path, request_g1, "g1")
    dry_g1 = _dry_run(args_g1)
    assert dry_g1["selected_count"] == 0
    assert (tmp_path / "raw/dependency.bin").is_file()
    request_g2 = _request([artifact], gate_id="G2", closed=["G1", "G2"])
    args_g2 = _cli_args(tmp_path, request_g2, "g2")
    _dry_run(args_g2)
    result = _apply(args_g2)
    assert result["released_count"] == 1
    assert not (tmp_path / "raw/dependency.bin").exists()


def test_20_reproduction_recipe_remains_after_raw_deletion(tmp_path: Path) -> None:
    request = _request([_artifact("raw-a", "raw/a.bin")])
    _write_artifacts(tmp_path, request["artifacts"])
    args = _cli_args(tmp_path, request)
    _dry_run(args)
    result = _apply(args)
    final = json.loads(Path(result["certificate_path"]).read_text())
    assert final["reproduction"]["commands"] == request["reproduction"]["commands"]
    assert final["phase"] == "FINAL"


def test_21_exact_qualification_required_retention_is_protected(tmp_path: Path) -> None:
    request = _request(
        [
            _artifact("qualification", "raw/qualification.bin"),
            _artifact("scratch", "raw/scratch.bin"),
        ],
        qualification_status="active",
        qualification_ids=["qualification"],
    )
    certificate = _certificate(tmp_path, request)
    rows = {row["artifact_id"]: row for row in certificate["artifacts"]}
    assert rows["qualification"]["disposition"] == "RETAIN_DURABLE"
    assert rows["scratch"]["release_eligible"] is True


def test_22_dry_run_byte_accounting_is_exact(tmp_path: Path) -> None:
    request = _request(
        [_artifact("raw-a", "raw/a.bin"), _artifact("raw-b", "raw/b.bin")]
    )
    _write_artifacts(tmp_path, request["artifacts"])
    expected = sum(
        (tmp_path / row["path"]).stat().st_size for row in request["artifacts"]
    )
    result = _dry_run(_cli_args(tmp_path, request))
    assert result["selected_bytes"] == expected
    assert all((tmp_path / row["path"]).is_file() for row in request["artifacts"])


def test_23_apply_writes_durable_disposition_receipt(tmp_path: Path) -> None:
    request = _request([_artifact("raw-a", "raw/a.bin")])
    _write_artifacts(tmp_path, request["artifacts"])
    args = _cli_args(tmp_path, request)
    _dry_run(args)
    result = _apply(args)
    receipt = json.loads(Path(result["storage_disposition_receipt_path"]).read_text())
    supplied = receipt.pop("receipt_sha256")
    assert supplied == canonical_sha256(receipt)
    assert receipt["bytes_released"] == result["released_bytes"]


def test_24_legacy_inputs_remain_conservative_and_read_only(tmp_path: Path) -> None:
    (tmp_path / "scratch").mkdir()
    (tmp_path / "scratch/cache.bin").write_bytes(b"legacy-cache")
    (tmp_path / "mystery.bin").write_bytes(b"legacy-unknown")
    before = sorted(
        (str(path.relative_to(tmp_path)), path.read_bytes())
        for path in tmp_path.rglob("*")
        if path.is_file()
    )
    audit = historical_audit(tmp_path, max_objects=20, max_bytes=10_000)
    after = sorted(
        (str(path.relative_to(tmp_path)), path.read_bytes())
        for path in tmp_path.rglob("*")
        if path.is_file()
    )
    assert audit["historical_deletion_authorized"] is False
    assert audit["classes_bytes"]["needs_reduction_certificate"] > 0
    assert audit["classes_bytes"]["unknown_policy"] > 0
    assert before == after


def test_25_unrelated_frozen_execution_is_untouched(tmp_path: Path) -> None:
    evidence = tmp_path / "gate-evidence"
    evidence.mkdir()
    frozen = tmp_path / "frozen-qualification"
    frozen.mkdir()
    frozen_file = frozen / "qualification.bin"
    frozen_file.write_bytes(b"immutable-frozen-evidence")
    request = _request([_artifact("raw-a", "raw/a.bin")])
    _write_artifacts(evidence, request["artifacts"])
    args = _cli_args(evidence, request)
    _dry_run(args)
    _apply(args)
    assert frozen_file.read_bytes() == b"immutable-frozen-evidence"


def test_consumer_added_after_dry_run_invalidates_apply(tmp_path: Path) -> None:
    request = _request([_artifact("raw-a", "raw/a.bin")])
    _write_artifacts(tmp_path, request["artifacts"])
    args = _cli_args(tmp_path, request)
    _dry_run(args)
    mutated = copy.deepcopy(request)
    mutated["artifacts"][0]["consumers"] = ["G2"]
    mutated["artifacts"][0]["last_use"] = "G2"
    args.request.write_text(json.dumps(mutated, indent=2, sort_keys=True) + "\n")
    with pytest.raises(EvidenceCertificateError, match="changed after dry-run"):
        _apply(args)
    assert (tmp_path / "raw/a.bin").is_file()


def test_malformed_dependency_graph_and_missing_future_metadata_fail_closed(
    tmp_path: Path,
) -> None:
    malformed = _request(
        [_artifact("raw-a", "raw/a.bin", consumers=["G2"], last_use="G1")]
    )
    _write_artifacts(tmp_path, malformed["artifacts"])
    with pytest.raises(EvidenceCertificateError, match="final named consumer"):
        build_evidence_certificate_v1(malformed, evidence_root=tmp_path)
    missing = _request([_artifact("raw-a", "raw/a.bin")])
    missing.pop("programme_plan")
    with pytest.raises(EvidenceCertificateError, match="programme_plan"):
        build_evidence_certificate_v1(missing, evidence_root=tmp_path)


def test_duplicate_materialization_group_with_live_copy_holds_all(
    tmp_path: Path,
) -> None:
    request = _request(
        [
            _artifact("copy-a", "raw/a.bin", materialization_group="duplicate-1"),
            _artifact(
                "copy-b",
                "raw/b.bin",
                live="consumer",
                materialization_group="duplicate-1",
            ),
        ]
    )
    certificate = _certificate(tmp_path, request)
    assert {row["disposition"] for row in certificate["artifacts"]} == {
        "HOLD_UNTIL_LAST_USE"
    }


def test_manifest_binds_certificate_and_direct_release_policy(tmp_path: Path) -> None:
    request = _request([_artifact("raw-a", "raw/a.bin")])
    certificate = _certificate(tmp_path, request)
    manifest = certificate_disposition_manifest_v1(
        request, certificate, evidence_root=tmp_path
    )
    validated = validate_artifact_disposition_manifest_v1(manifest)
    assert (
        validated["release_authority"]["certificate_sha256"]
        == certificate["certificate_sha256"]
    )
    assert (
        validated["release_authority"]["ordinary_releasable_archive_policy"]
        == "DIRECT_RELEASE_NO_ARCHIVE"
    )


def test_certificate_manifest_cannot_bypass_close_gate_context(tmp_path: Path) -> None:
    request = _request([_artifact("raw-a", "raw/a.bin")])
    certificate = _certificate(tmp_path, request)
    manifest = certificate_disposition_manifest_v1(
        request, certificate, evidence_root=tmp_path
    )
    plan = build_prune_plan(tmp_path, set(prune_module.CATEGORIES), manifest)
    with pytest.raises(ArtifactLifecycleError, match="close-gate context"):
        apply_prune_plan(
            tmp_path,
            plan,
            disposition_manifest=manifest,
        )
    assert (tmp_path / "raw/a.bin").is_file()


def test_gate_closure_must_be_prefix_and_producer_must_be_bound(tmp_path: Path) -> None:
    non_prefix = _request(
        [
            _artifact(
                "dependency", "raw/dependency.bin", consumers=["G3"], last_use="G3"
            )
        ],
        ordered=["G1", "G2", "G3"],
        closed=["G1", "G3"],
        gate_id="G1",
    )
    _write_artifacts(tmp_path, non_prefix["artifacts"])
    with pytest.raises(EvidenceCertificateError, match="ordered closure prefix"):
        build_evidence_certificate_v1(non_prefix, evidence_root=tmp_path)
    foreign_producer = _request([_artifact("raw-a", "raw/a.bin")])
    foreign_producer["artifacts"][0]["produced_by"] = "G9"
    with pytest.raises(EvidenceCertificateError, match="producer is outside"):
        build_evidence_certificate_v1(foreign_producer, evidence_root=tmp_path)


def test_omitted_consumers_and_unverified_closure_authority_fail_closed(
    tmp_path: Path,
) -> None:
    omitted = _request([_artifact("raw-a", "raw/a.bin")])
    omitted["artifacts"][0].pop("consumers")
    omitted["artifacts"][0].pop("last_use")
    _write_artifacts(tmp_path, omitted["artifacts"])
    with pytest.raises(EvidenceCertificateError, match="omits explicit dependency"):
        build_evidence_certificate_v1(omitted, evidence_root=tmp_path)
    stale_closure = _request([_artifact("raw-a", "raw/a.bin")])
    stale_closure["programme_plan"]["closure_receipt"]["status"] = "OPEN"
    with pytest.raises(EvidenceCertificateError, match="stale or semantically invalid"):
        build_evidence_certificate_v1(stale_closure, evidence_root=tmp_path)


def test_late_member_in_staged_tree_is_preserved_and_blocks_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tree = tmp_path / "raw" / "tree"
    tree.mkdir(parents=True)
    (tree / "sealed.bin").write_bytes(b"sealed")
    request = _request([_artifact("tree", "raw/tree")])
    certificate = build_evidence_certificate_v1(request, evidence_root=tmp_path)
    manifest = certificate_disposition_manifest_v1(
        request, certificate, evidence_root=tmp_path
    )
    plan = build_prune_plan(tmp_path, set(prune_module.CATEGORIES), manifest)
    real_remove = prune_module._remove_sealed_tree_remainder

    def add_late_member(path: Path, **kwargs):
        (path / "late.bin").write_bytes(b"late-writer")
        return real_remove(path, **kwargs)

    monkeypatch.setattr(prune_module, "_remove_sealed_tree_remainder", add_late_member)
    result = apply_prune_plan(
        tmp_path,
        plan,
        disposition_manifest=manifest,
        certificate_release_context=_certificate_prune_context(
            request, certificate, manifest, plan
        ),
    )
    assert result["failures"]
    journal = json.loads(Path(result["release_journal"]).read_text())
    staged_tree = next(Path(journal["staging_root"]).iterdir())
    assert (staged_tree / "sealed.bin").read_bytes() == b"sealed"
    assert (staged_tree / "late.bin").read_bytes() == b"late-writer"


def test_post_delete_receipt_failure_accounts_bytes_and_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request([_artifact("raw-a", "raw/a.bin")])
    _write_artifacts(tmp_path, request["artifacts"])
    expected_bytes = (tmp_path / "raw/a.bin").stat().st_size
    args = _cli_args(tmp_path, request)
    _dry_run(args)
    real_write = reclaim_module.write_storage_disposition_receipt_v1
    failed = False

    def fail_first_receipt(path: Path, value: Mapping[str, Any]):
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("simulated receipt persistence interruption")
        return real_write(path, value)

    monkeypatch.setattr(
        reclaim_module,
        "write_storage_disposition_receipt_v1",
        fail_first_receipt,
    )
    with pytest.raises(OSError, match="receipt persistence"):
        _apply(args)
    assert not (tmp_path / "raw/a.bin").exists()
    monkeypatch.setattr(
        reclaim_module,
        "write_storage_disposition_receipt_v1",
        real_write,
    )
    second = _apply(args)
    assert second["ok"] is True
    assert second["released_bytes"] == expected_bytes


def test_partial_failure_retry_uses_only_journal_authenticated_absence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(
        [_artifact("raw-a", "raw/a.bin"), _artifact("raw-b", "raw/b.bin")]
    )
    _write_artifacts(tmp_path, request["artifacts"])
    _, source_path = _write_source_manifest(tmp_path, request)
    upgraded = upgrade_evidence_certificate_request_source_bindings_v1(
        request,
        evidence_root=tmp_path,
        source_manifest_paths=[source_path],
    )
    monkeypatch.setattr(
        certificate_module,
        "evidence_root_requires_permanent_capsule_v1",
        lambda _root: True,
    )
    monkeypatch.setattr(
        reclaim_module,
        "evidence_root_requires_permanent_capsule_v1",
        lambda _root: True,
    )
    args = _cli_args(tmp_path, upgraded, "partial-journal-retry")
    args.permanent_root = tmp_path.parent / f"{tmp_path.name}-partial-permanent"
    _dry_run(args)
    real_rename = prune_module._rename_noreplace
    calls = 0

    def fail_second_rename(*call_args, **call_kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated second-row staging failure")
        return real_rename(*call_args, **call_kwargs)

    monkeypatch.setattr(prune_module, "_rename_noreplace", fail_second_rename)
    partial = _apply(args)
    assert partial["ok"] is False
    assert partial["mode"] == "partial_failure"
    assert partial["released_count"] == 1
    assert (
        json.loads((args.output_root / "reclamation_intent.json").read_text())["status"]
        == "PARTIAL_FAILURE"
    )
    assert sum((tmp_path / path).exists() for path in ("raw/a.bin", "raw/b.bin")) == 1

    monkeypatch.setattr(prune_module, "_rename_noreplace", real_rename)
    recovered = _apply(args)
    assert recovered["ok"] is True
    assert recovered["released_count"] == 2
    assert not (tmp_path / "raw/a.bin").exists()
    assert not (tmp_path / "raw/b.bin").exists()
    assert recovered["release_journal_status"] == "COMPLETE"
    assert (
        json.loads((args.output_root / "reclamation_intent.json").read_text())["status"]
        == "APPLIED"
    )


def test_stale_debug_review_recovers_authenticated_partial_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, _ = _debug_request(
        tmp_path,
        [
            _artifact("raw-a", "raw/a.bin"),
            _artifact("raw-b", "raw/b.bin"),
        ],
    )
    args = _cli_args(tmp_path, request, "stale-partial-debug")
    args.permanent_root = tmp_path.parent / f"{tmp_path.name}-stale-permanent"
    _dry_run(args)
    real_rename = prune_module._rename_noreplace
    calls = 0

    def fail_second_rename(*call_args, **call_kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated second-row staging failure")
        return real_rename(*call_args, **call_kwargs)

    monkeypatch.setattr(prune_module, "_rename_noreplace", fail_second_rename)
    partial = _apply(args)
    assert partial["mode"] == "partial_failure"
    assert partial["released_count"] == 1

    real_datetime = certificate_module.datetime

    class FutureDatetime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return real_datetime.now(tz) + timedelta(hours=2)

    monkeypatch.setattr(certificate_module, "datetime", FutureDatetime)
    monkeypatch.setattr(prune_module, "_rename_noreplace", real_rename)
    recovered = _apply(args)
    assert recovered["ok"] is True
    assert recovered["released_count"] == 2
    assert recovered["release_journal_status"] == "COMPLETE"
    assert not (tmp_path / "raw/a.bin").exists()
    assert not (tmp_path / "raw/b.bin").exists()


def test_all_staged_debug_release_recovers_despite_new_root_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, _ = _debug_request(
        tmp_path,
        [_artifact("raw-a", "raw/a.bin")],
    )
    args = _cli_args(tmp_path, request, "all-staged-live-retry")
    args.permanent_root = tmp_path.parent / f"{tmp_path.name}-all-staged-permanent"
    _dry_run(args)
    real_unlink = prune_module.os.unlink
    failed = False

    def fail_after_staging(path, *call_args, **call_kwargs):
        nonlocal failed
        if call_kwargs.get("dir_fd") is not None and not failed:
            failed = True
            raise OSError("simulated crash after all rows staged")
        return real_unlink(path, *call_args, **call_kwargs)

    monkeypatch.setattr(prune_module.os, "unlink", fail_after_staging)
    partial = _apply(args)
    assert partial["mode"] == "partial_failure"
    assert failed is True
    assert not (tmp_path / "raw/a.bin").exists()
    journal = json.loads(Path(partial["release_journal"]).read_text())
    assert journal["progress"][0]["state"] == "STAGED"

    monkeypatch.setattr(prune_module.os, "unlink", real_unlink)
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
            str(tmp_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        recovered = _apply(args)
    finally:
        child.terminate()
        child.wait(timeout=5)

    assert recovered["ok"] is True
    assert recovered["released_count"] == 1
    assert recovered["release_journal_status"] == "COMPLETE"
    assert not (tmp_path / "raw/a.bin").exists()


def test_staged_unlinked_debug_source_reappearance_gets_fresh_live_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, _ = _debug_request(
        tmp_path,
        [_artifact("raw-a", "raw/a.bin")],
    )
    args = _cli_args(tmp_path, request, "staged-unlinked-reappearance")
    args.permanent_root = tmp_path.parent / (
        f"{tmp_path.name}-staged-unlinked-permanent"
    )
    source = tmp_path / "raw/a.bin"
    source_bytes = source.read_bytes()
    _dry_run(args)

    real_write_journal = prune_module._write_archive_release_journal
    crashed = False

    def crash_before_released_journal(path, journal):
        nonlocal crashed
        progress = journal.get("progress", [])
        if not crashed and any(
            isinstance(row, Mapping) and row.get("state") == "RELEASED"
            for row in progress
        ):
            crashed = True
            raise OSError("simulated crash after unlink before RELEASED journal")
        return real_write_journal(path, journal)

    monkeypatch.setattr(
        prune_module,
        "_write_archive_release_journal",
        crash_before_released_journal,
    )
    partial = _apply(args)
    assert partial["mode"] == "partial_failure"
    assert crashed is True
    assert not source.exists()
    journal = json.loads(Path(partial["release_journal"]).read_text())
    assert journal["progress"][0]["state"] == "STAGED"
    assert not Path(journal["progress"][0]["staging_path"]).exists()

    monkeypatch.setattr(
        prune_module,
        "_write_archive_release_journal",
        real_write_journal,
    )
    real_open_directory = prune_module._open_directory_beneath
    real_rename = prune_module._rename_noreplace
    child: subprocess.Popen[str] | None = None
    injected = False
    rename_calls = 0

    def open_then_reappear(*call_args, **call_kwargs):
        nonlocal child, injected
        descriptor = real_open_directory(*call_args, **call_kwargs)
        if not injected and call_kwargs.get("create") is True:
            injected = True
            source.write_bytes(source_bytes)
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    (
                        "import sys,time; handle=open(sys.argv[1], 'rb'); "
                        "print('ready', flush=True); time.sleep(30)"
                    ),
                    str(source),
                ],
                stdout=subprocess.PIPE,
                text=True,
            )
            assert child.stdout is not None
            assert child.stdout.readline().strip() == "ready"
        return descriptor

    def count_rename(*call_args, **call_kwargs):
        nonlocal rename_calls
        rename_calls += 1
        return real_rename(*call_args, **call_kwargs)

    monkeypatch.setattr(
        prune_module,
        "_open_directory_beneath",
        open_then_reappear,
    )
    monkeypatch.setattr(prune_module, "_rename_noreplace", count_rename)
    try:
        retry = _apply(args)
    finally:
        if child is not None:
            child.terminate()
            child.wait(timeout=5)

    assert injected is True
    assert retry["mode"] == "partial_failure"
    assert any(
        "process/relay references" in str(failure.get("message"))
        for failure in retry["failures"]
    )
    assert rename_calls == 0
    assert source.is_file()
    retry_journal = json.loads(Path(retry["release_journal"]).read_text())
    assert retry_journal["progress"][0]["state"] == "STAGED"


def test_successful_apply_replays_without_sources_or_saved_result(
    tmp_path: Path,
) -> None:
    request = _request([_artifact("raw-a", "raw/a.bin")])
    _write_artifacts(tmp_path, request["artifacts"])
    args = _cli_args(tmp_path, request)
    _dry_run(args)
    first = _apply(args)
    assert first["ok"] is True
    replay = _apply(args)
    assert replay["idempotent_replay"] is True
    Path(first["intent_path"]).parent.joinpath(
        "gate_G1_reclamation_result.json"
    ).unlink()
    reconstructed = _apply(args)
    assert reconstructed["idempotent_replay"] is True
    assert reconstructed["released_bytes"] == first["released_bytes"]


def test_concurrent_apply_serializes_and_loser_replays_applied_tuple(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request([_artifact("raw-a", "raw/a.bin")])
    _write_artifacts(tmp_path, request["artifacts"])
    args = _cli_args(tmp_path, request, "concurrent-apply")
    _dry_run(args)

    first_inside_transaction = threading.Event()
    release_first = threading.Event()
    second_lock_attempt = threading.Event()
    call_guard = threading.Lock()
    under_lock_calls = 0
    exclusive_lock_attempts = 0
    real_under_lock = reclaim_module._apply_under_lock
    real_flock = reclaim_module.fcntl.flock

    def pause_first_under_lock(*call_args, **call_kwargs):
        nonlocal under_lock_calls
        with call_guard:
            under_lock_calls += 1
            ordinal = under_lock_calls
        if ordinal == 1:
            first_inside_transaction.set()
            assert release_first.wait(timeout=5)
        return real_under_lock(*call_args, **call_kwargs)

    def observe_flock(fd: int, operation: int):
        nonlocal exclusive_lock_attempts
        if operation & reclaim_module.fcntl.LOCK_EX:
            with call_guard:
                exclusive_lock_attempts += 1
                if exclusive_lock_attempts == 2:
                    second_lock_attempt.set()
        return real_flock(fd, operation)

    monkeypatch.setattr(reclaim_module, "_apply_under_lock", pause_first_under_lock)
    monkeypatch.setattr(reclaim_module.fcntl, "flock", observe_flock)
    with ThreadPoolExecutor(max_workers=2) as executor:
        winner = executor.submit(_apply, args)
        assert first_inside_transaction.wait(timeout=5)
        loser = executor.submit(_apply, args)
        assert second_lock_attempt.wait(timeout=5)
        assert loser.done() is False
        release_first.set()
        winner_result = winner.result(timeout=10)
        loser_result = loser.result(timeout=10)

    assert winner_result["ok"] is True
    assert winner_result.get("idempotent_replay") is not True
    assert loser_result["ok"] is True
    assert loser_result["idempotent_replay"] is True
    assert loser_result["released_bytes"] == winner_result["released_bytes"]
    output_root = Path(args.output_root)
    assert (
        json.loads((output_root / "reclamation_intent.json").read_text())["status"]
        == "APPLIED"
    )
    assert (
        json.loads((output_root / "storage_disposition_receipt.json").read_text())[
            "result"
        ]
        == "applied"
    )


def test_debug_forensic_requires_exact_diagnostic_closeout(tmp_path: Path) -> None:
    request, _ = _debug_request(tmp_path, [_artifact("raw-a", "raw/a.bin")])

    missing = copy.deepcopy(request)
    missing.pop("diagnostic_closeout")
    with pytest.raises(EvidenceCertificateError, match="diagnostic_closeout"):
        validate_evidence_certificate_request_v1(missing)

    rejected = copy.deepcopy(request)
    rejected["diagnostic_closeout"]["substantive_review"]["status"] = "rejected"
    with pytest.raises(EvidenceCertificateError, match="review must be accepted"):
        validate_evidence_certificate_request_v1(rejected)

    anomaly = copy.deepcopy(request)
    anomaly["diagnostic_closeout"]["unresolved_anomalies"] = ["open anomaly"]
    with pytest.raises(EvidenceCertificateError, match="unresolved anomalies"):
        validate_evidence_certificate_request_v1(anomaly)

    live = copy.deepcopy(request)
    live["diagnostic_closeout"]["live_state"]["consumer"] = True
    with pytest.raises(EvidenceCertificateError, match="has live state"):
        validate_evidence_certificate_request_v1(live)

    held = copy.deepcopy(request)
    held["diagnostic_closeout"]["active_exact_byte_holds"]["replay"] = ["replay-hold-1"]
    with pytest.raises(EvidenceCertificateError, match="active exact-byte holds"):
        validate_evidence_certificate_request_v1(held)


def test_handcrafted_debug_request_cannot_close_active_source_marker(
    tmp_path: Path,
) -> None:
    request, _ = _debug_request(
        tmp_path,
        [_artifact("raw-a", "raw/a.bin")],
        include_source_live_marker_closeout=False,
    )
    with pytest.raises(EvidenceCertificateError, match="active diagnostic source live"):
        build_evidence_certificate_v1(request, evidence_root=tmp_path)


def test_debug_source_marker_none_still_rechecks_foreign_process(
    tmp_path: Path,
) -> None:
    request, _ = _debug_request(
        tmp_path,
        [_artifact("raw-a", "raw/a.bin")],
        source_live_reference_status="none",
        include_source_live_marker_closeout=False,
    )
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import sys,time; handle=open(sys.argv[1], 'rb'); "
                "print('ready', flush=True); time.sleep(30)"
            ),
            str(tmp_path / "raw/a.bin"),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "ready"
        with pytest.raises(EvidenceCertificateError, match="process/relay references"):
            build_evidence_certificate_v1(request, evidence_root=tmp_path)
    finally:
        child.terminate()
        child.wait(timeout=5)


def test_diagnostic_local_audit_excludes_only_current_process(
    tmp_path: Path,
) -> None:
    request, _ = _debug_request(
        tmp_path,
        [_artifact("raw-a", "raw/a.bin")],
    )
    with (tmp_path / "raw/a.bin").open("rb"):
        audit = diagnostic_local_live_state_audit_v1(tmp_path)
        assert all(row["pid"] != os.getpid() for row in audit["process_references"])
        build_evidence_certificate_v1(request, evidence_root=tmp_path)


def test_diagnostic_destructive_authority_rechecks_foreign_open_fd_and_cmdline(
    tmp_path: Path,
) -> None:
    request, _ = _debug_request(
        tmp_path,
        [_artifact("raw-a", "raw/a.bin")],
    )
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import sys,time; handle=open(sys.argv[1], 'rb'); "
                "print('ready', flush=True); time.sleep(30)"
            ),
            str(tmp_path / "raw/a.bin"),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "ready"
        audit = diagnostic_local_live_state_audit_v1(tmp_path)
        observed = next(
            row for row in audit["process_references"] if row["pid"] == child.pid
        )
        assert {"cmdline", "fd"} <= set(observed["reasons"])
        with pytest.raises(EvidenceCertificateError, match="process/relay references"):
            build_evidence_certificate_v1(request, evidence_root=tmp_path)
    finally:
        child.terminate()
        child.wait(timeout=5)


def test_diagnostic_local_audit_fails_closed_on_process_scan_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, _ = _debug_request(
        tmp_path,
        [_artifact("raw-a", "raw/a.bin")],
    )
    child = subprocess.Popen(["sleep", "30"])
    real_readlink = certificate_module.os.readlink

    def fail_child_cwd(path):
        if str(path) == f"/proc/{child.pid}/cwd":
            raise PermissionError("simulated process inspection denial")
        return real_readlink(path)

    monkeypatch.setattr(certificate_module.os, "readlink", fail_child_cwd)
    try:
        with pytest.raises(EvidenceCertificateError, match="inspection was incomplete"):
            build_evidence_certificate_v1(request, evidence_root=tmp_path)
    finally:
        child.terminate()
        child.wait(timeout=5)


def test_diagnostic_local_audit_rechecks_canonical_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, _ = _debug_request(
        tmp_path,
        [_artifact("raw-a", "raw/a.bin")],
    )
    monkeypatch.setattr(
        certificate_module.RunStorageGuard,
        "run_root_is_reserved",
        lambda _root: True,
    )
    with pytest.raises(EvidenceCertificateError, match="canonical storage reservation"):
        build_evidence_certificate_v1(request, evidence_root=tmp_path)


def test_diagnostic_local_audit_rechecks_recorded_canopy_relay(
    tmp_path: Path,
) -> None:
    request, _ = _debug_request(
        tmp_path,
        [_artifact("raw-a", "raw/a.bin")],
    )
    output_root = tmp_path.parent / f"{tmp_path.name}-relay-output"
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
            "living_lineage_canopy.python.relay_cli watch",
            str(output_root),
        ]
    )
    relay_core = {
        "schema_version": "eiel_canopy_relay_supervision_v1",
        "state": "watching",
        "run_root": str(tmp_path),
        "relays": [
            {
                "cell_id": "cell-a",
                "pid": child.pid,
                "output_root": str(output_root),
            }
        ],
    }
    (tmp_path / "canopy_relay_supervision.json").write_text(
        json.dumps(
            {**relay_core, "receipt_sha256": canonical_sha256(relay_core)},
            sort_keys=True,
        )
        + "\n"
    )
    try:
        with pytest.raises(EvidenceCertificateError, match="process/relay references"):
            build_evidence_certificate_v1(request, evidence_root=tmp_path)
    finally:
        child.terminate()
        child.wait(timeout=5)


@pytest.mark.parametrize(
    ("audit_status", "qualification_status"),
    [("active", "not_required"), ("not_required", "active"), ("unknown", "unknown")],
)
def test_debug_live_marker_never_supersedes_source_exact_byte_holds(
    tmp_path: Path,
    audit_status: str,
    qualification_status: str,
) -> None:
    request, _ = _debug_request(
        tmp_path,
        [_artifact("raw-a", "raw/a.bin")],
        source_audit_hold_status=audit_status,
        source_qualification_hold_status=qualification_status,
    )
    with pytest.raises(EvidenceCertificateError, match="audit/qualification hold"):
        build_evidence_certificate_v1(request, evidence_root=tmp_path)


def test_debug_source_identity_is_per_run_not_programme_assumed(
    tmp_path: Path,
) -> None:
    request, source_manifest = _debug_request(
        tmp_path, [_artifact("raw-a", "raw/a.bin")]
    )
    assert source_manifest["campaign_id"] != request["programme_id"]
    assert source_manifest["plan_sha256"] != request["identity"]["plan_sha256"]
    certificate = build_evidence_certificate_v1(request, evidence_root=tmp_path)
    assert certificate["diagnostic_closeout"]["source_manifest"]["campaign_id"] == (
        "twbb-v3-baseline-r18"
    )

    stale = copy.deepcopy(request)
    stale["diagnostic_closeout"]["source_manifest"]["run_id"] = "foreign-run"
    with pytest.raises(
        EvidenceCertificateError, match="authority binding|identity differs"
    ):
        build_evidence_certificate_v1(stale, evidence_root=tmp_path)


def test_debug_closeout_preserves_holds_r4_r5_rx_and_only_supersedes_exact_rows(
    tmp_path: Path,
) -> None:
    artifacts = [
        _artifact("raw-r1", "raw/r1.bin"),
        _artifact("asset-r4", "raw/r4.bin"),
        _artifact("authority-r5", "raw/r5.bin"),
        _artifact("unknown-rx", "raw/rx.bin"),
    ]
    request, _ = _debug_request(
        tmp_path,
        artifacts,
        source_classes={
            "raw-r1": "R1",
            "asset-r4": "R4",
            "authority-r5": "R5",
            "unknown-rx": "RX",
        },
        closeout_dispositions={
            "asset-r4": "RETAIN",
            "authority-r5": "RETAIN",
            "unknown-rx": "RETAIN",
        },
    )
    certificate = build_evidence_certificate_v1(request, evidence_root=tmp_path)
    rows = {row["artifact_id"]: row for row in certificate["artifacts"]}
    assert rows["raw-r1"]["release_eligible"] is True
    assert rows["raw-r1"]["retention_class"] == "R1"
    assert rows["asset-r4"]["retention_class"] == "R4"
    assert rows["authority-r5"]["retention_class"] == "R5"
    assert rows["unknown-rx"]["retention_class"] == "RX"
    assert all(
        rows[name]["release_eligible"] is False
        for name in ("asset-r4", "authority-r5", "unknown-rx")
    )

    invalid = copy.deepcopy(request)
    invalid["artifacts"][1]["diagnostic_closeout_disposition"] = (
        "SUPERSEDED_BY_CERTIFICATE"
    )
    invalid["diagnostic_closeout"]["superseded_artifact_ids"].append("asset-r4")
    with pytest.raises(EvidenceCertificateError, match="authority binding|source R4"):
        validate_evidence_certificate_request_v1(invalid)


def test_diagnostic_manifest_closes_only_selected_live_marker_and_preserves_holds(
    tmp_path: Path,
) -> None:
    request, source = _debug_request(
        tmp_path,
        [
            _artifact("raw-r1", "raw/r1.bin"),
            _artifact("asset-r4", "raw/r4.bin"),
            _artifact("authority-r5", "raw/r5.bin"),
            _artifact("unknown-rx", "raw/rx.bin"),
        ],
        source_classes={
            "raw-r1": "R1",
            "asset-r4": "R4",
            "authority-r5": "R5",
            "unknown-rx": "RX",
        },
        closeout_dispositions={
            "asset-r4": "RETAIN",
            "authority-r5": "RETAIN",
            "unknown-rx": "RETAIN",
        },
        source_row_statuses={
            "asset-r4": {
                "live_reference_status": "active",
                "audit_hold_status": "active",
                "qualification_hold_status": "not_required",
            },
            "authority-r5": {
                "live_reference_status": "active",
                "audit_hold_status": "not_required",
                "qualification_hold_status": "active",
            },
            "unknown-rx": {
                "live_reference_status": "active",
                "audit_hold_status": "active",
                "qualification_hold_status": "active",
            },
        },
    )
    certificate = build_evidence_certificate_v1(request, evidence_root=tmp_path)
    permanent = publish_permanent_evidence_capsule_v1(
        certificate,
        permanent_root=tmp_path.parent / f"{tmp_path.name}-protected-capsules",
        source_root=tmp_path,
    )
    manifest = certificate_disposition_manifest_v1(
        request,
        certificate,
        evidence_root=tmp_path,
        permanent_publication=permanent,
    )
    source_by_path = {row["path"]: row for row in source["artifacts"]}
    derived_by_path = {row["path"]: row for row in manifest["artifacts"]}
    selected = derived_by_path["raw/r1.bin"]
    assert selected["live_reference_status"] == "none"
    assert (
        selected["audit_hold_status"]
        == source_by_path["raw/r1.bin"]["audit_hold_status"]
    )
    assert (
        selected["qualification_hold_status"]
        == source_by_path["raw/r1.bin"]["qualification_hold_status"]
    )
    for path in ("raw/r4.bin", "raw/r5.bin", "raw/rx.bin"):
        for field in (
            "live_reference_status",
            "audit_hold_status",
            "qualification_hold_status",
        ):
            assert derived_by_path[path][field] == source_by_path[path][field]
    assert manifest["retention"]["audit_hold_status"] == "active"
    assert manifest["retention"]["qualification_hold_status"] == "active"


def test_direct_prune_rejects_erased_protected_diagnostic_source_state(
    tmp_path: Path,
) -> None:
    request, _ = _debug_request(
        tmp_path,
        [
            _artifact("raw-r1", "raw/r1.bin"),
            _artifact("asset-r4", "raw/r4.bin"),
        ],
        source_classes={"raw-r1": "R1", "asset-r4": "R4"},
        closeout_dispositions={"asset-r4": "RETAIN"},
        source_row_statuses={
            "asset-r4": {
                "live_reference_status": "active",
                "audit_hold_status": "active",
                "qualification_hold_status": "not_required",
            }
        },
    )
    certificate = build_evidence_certificate_v1(request, evidence_root=tmp_path)
    permanent = publish_permanent_evidence_capsule_v1(
        certificate,
        permanent_root=tmp_path.parent / f"{tmp_path.name}-prune-state-capsules",
        source_root=tmp_path,
    )
    manifest = certificate_disposition_manifest_v1(
        request,
        certificate,
        evidence_root=tmp_path,
        permanent_publication=permanent,
    )
    tampered = copy.deepcopy(manifest)
    protected = next(
        row for row in tampered["artifacts"] if row["path"] == "raw/r4.bin"
    )
    protected["live_reference_status"] = "none"
    protected["audit_hold_status"] = "not_required"
    tampered["retention"]["audit_hold_status"] = "not_required"
    tampered.pop("manifest_sha256")
    tampered["manifest_sha256"] = canonical_sha256(tampered)
    plan = build_prune_plan(tmp_path, set(prune_module.CATEGORIES), tampered)
    context = _certificate_prune_context(request, certificate, tampered, plan)
    context["permanent_publication"] = permanent

    with pytest.raises(ArtifactLifecycleError, match="manifest row differs"):
        apply_prune_plan(
            tmp_path,
            plan,
            disposition_manifest=tampered,
            certificate_release_context=context,
        )
    assert (tmp_path / "raw/r1.bin").is_file()
    assert (tmp_path / "raw/r4.bin").is_file()


def test_debug_apply_rechecks_process_references_at_destructive_boundary(
    tmp_path: Path,
) -> None:
    request, _ = _debug_request(
        tmp_path,
        [_artifact("raw-a", "raw/a.bin")],
    )
    args = _cli_args(tmp_path, request, "debug-process-recheck")
    args.permanent_root = tmp_path.parent / f"{tmp_path.name}-process-permanent"
    _dry_run(args)
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import sys,time; handle=open(sys.argv[1], 'rb'); "
                "print('ready', flush=True); time.sleep(30)"
            ),
            str(tmp_path / "raw/a.bin"),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "ready"
        with pytest.raises(EvidenceCertificateError, match="process/relay references"):
            _apply(args)
        assert (tmp_path / "raw/a.bin").is_file()
    finally:
        child.terminate()
        child.wait(timeout=5)


def test_debug_prune_rechecks_live_state_after_target_validation_before_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, _ = _debug_request(
        tmp_path,
        [_artifact("raw-a", "raw/a.bin")],
    )
    certificate = build_evidence_certificate_v1(request, evidence_root=tmp_path)
    permanent = publish_permanent_evidence_capsule_v1(
        certificate,
        permanent_root=tmp_path.parent / f"{tmp_path.name}-late-process-capsules",
        source_root=tmp_path,
    )
    manifest = certificate_disposition_manifest_v1(
        request,
        certificate,
        evidence_root=tmp_path,
        permanent_publication=permanent,
    )
    plan = build_prune_plan(tmp_path, set(prune_module.CATEGORIES), manifest)
    context = _certificate_prune_context(request, certificate, manifest, plan)
    context["permanent_publication"] = permanent

    original_validate = prune_module.validate_certificate_source_manifest_bindings_v1
    original_rename = prune_module._rename_noreplace
    child: subprocess.Popen[str] | None = None
    rename_calls = 0

    def validate_then_attach(*call_args, **call_kwargs):
        nonlocal child
        result = original_validate(*call_args, **call_kwargs)
        assert child is None
        child = subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import sys,time; handle=open(sys.argv[1], 'rb'); "
                    "print('ready', flush=True); time.sleep(30)"
                ),
                str(tmp_path / "raw/a.bin"),
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "ready"
        return result

    def count_rename(*call_args, **call_kwargs):
        nonlocal rename_calls
        rename_calls += 1
        return original_rename(*call_args, **call_kwargs)

    monkeypatch.setattr(
        prune_module,
        "validate_certificate_source_manifest_bindings_v1",
        validate_then_attach,
    )
    monkeypatch.setattr(prune_module, "_rename_noreplace", count_rename)
    try:
        result = apply_prune_plan(
            tmp_path,
            plan,
            disposition_manifest=manifest,
            certificate_release_context=context,
        )
        assert result["failures"]
        assert "process/relay references" in result["failures"][0]["message"]
        assert rename_calls == 0
        assert (tmp_path / "raw/a.bin").is_file()
    finally:
        if child is not None:
            child.terminate()
            child.wait(timeout=5)


def test_debug_apply_rechecks_reservation_at_destructive_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, _ = _debug_request(
        tmp_path,
        [_artifact("raw-a", "raw/a.bin")],
    )
    args = _cli_args(tmp_path, request, "debug-reservation-recheck")
    args.permanent_root = tmp_path.parent / f"{tmp_path.name}-reserve-permanent"
    _dry_run(args)
    monkeypatch.setattr(
        certificate_module.RunStorageGuard,
        "run_root_is_reserved",
        lambda _root: True,
    )
    with pytest.raises(ArtifactLifecycleError, match="reservation"):
        _apply(args)
    assert (tmp_path / "raw/a.bin").is_file()


def test_debug_compact_capsule_precedes_release_and_final_follows_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = [
        _artifact("decision-r3", "raw/decision.bin"),
        _artifact("durable-r5", "raw/durable.bin"),
    ]
    request, _ = _debug_request(
        tmp_path,
        artifacts,
        source_classes={"decision-r3": "R3", "durable-r5": "R5"},
        closeout_dispositions={"durable-r5": "RETAIN"},
    )
    args = _cli_args(tmp_path, request, "debug-close")
    args.permanent_root = tmp_path.parent / f"{tmp_path.name}-permanent"
    dry = _dry_run(args)
    pre_receipt_path = Path(dry["permanent_capsule_publication_path"])
    pre_publication = json.loads(pre_receipt_path.read_text())
    assert pre_publication["phase"] == "SEALED_PRE_RELEASE"
    assert pre_publication["total_bytes"] < 32 * 1024
    assert Path(pre_publication["json_path"]).is_file()
    assert Path(pre_publication["markdown_path"]).is_file()
    assert Path(pre_publication["complete_path"]).is_file()
    pre_capsule = validate_permanent_evidence_capsule_v1(
        json.loads(Path(pre_publication["json_path"]).read_text())
    )
    assert "artifacts" not in pre_capsule
    assert pre_capsule["artifact_accounting"]["object_count"] == 2

    manifest = json.loads(Path(dry["manifest_path"]).read_text())
    assert manifest["lifecycle_profile"] == DEBUG_FORENSIC
    assert manifest["release_authority"]["kind"] == (
        "EVIDENCE_CERTIFICATE_DIAGNOSTIC_CLOSEOUT"
    )
    assert manifest["release_authority"]["source_row_count"] == 2
    assert manifest["release_authority"]["accounted_source_row_count"] == 2
    assert manifest["release_authority"]["superseded_row_count"] == 1
    assert manifest["release_authority"]["preserved_row_count"] == 1

    real_apply = reclaim_module.apply_prune_plan
    publication_seen_before_release = False

    def observe_publication(*call_args, **call_kwargs):
        nonlocal publication_seen_before_release
        publication = call_kwargs["certificate_release_context"][
            "permanent_publication"
        ]
        assert Path(publication["json_path"]).is_file()
        assert Path(publication["markdown_path"]).is_file()
        assert Path(publication["complete_path"]).is_file()
        assert (tmp_path / "raw/decision.bin").is_file()
        publication_seen_before_release = True
        return real_apply(*call_args, **call_kwargs)

    monkeypatch.setattr(reclaim_module, "apply_prune_plan", observe_publication)
    applied = _apply(args)
    assert applied["ok"] is True
    assert publication_seen_before_release is True
    assert not (tmp_path / "raw/decision.bin").exists()
    assert (tmp_path / "raw/durable.bin").is_file()
    final_publication = json.loads(
        Path(applied["permanent_final_publication_path"]).read_text()
    )
    assert final_publication["phase"] == "FINAL"
    assert final_publication["total_bytes"] < 32 * 1024
    final_capsule = validate_permanent_evidence_capsule_v1(
        json.loads(Path(final_publication["json_path"]).read_text())
    )
    assert final_capsule["phase"] == "FINAL"
    assert final_capsule["storage_disposition_receipt_sha256"]


def test_pre_release_reserves_near_cap_final_publication_before_release(
    tmp_path: Path,
) -> None:
    request, _ = _debug_request(
        tmp_path,
        [_artifact("raw-a", "raw/a.bin")],
    )
    args = _cli_args(tmp_path, request, "near-cap-family")
    permanent_root, root_limit = _configure_near_cap_permanent_root(
        tmp_path,
        request,
        args,
        label="near-cap-family",
    )
    dry = _dry_run(args)
    pre_publication = json.loads(
        Path(dry["permanent_capsule_publication_path"]).read_text()
    )
    actual_before, reserved_before, consumed_before = capsule_module._root_usage(
        permanent_root
    )
    assert actual_before + reserved_before == root_limit
    assert reserved_before == DEFAULT_MAX_CAPSULE_BYTES
    assert pre_publication["final_capacity_reservation_sha256"] not in consumed_before

    applied = _apply(args)
    assert applied["ok"] is True
    final_publication = json.loads(
        Path(applied["permanent_final_publication_path"]).read_text()
    )
    actual_after, reserved_after, consumed_after = capsule_module._root_usage(
        permanent_root
    )
    assert actual_after <= root_limit
    assert reserved_after == 0
    assert final_publication["final_capacity_reservation_status"] == "CONSUMED"
    assert final_publication["final_capacity_reservation_sha256"] in consumed_after
    assert not (tmp_path / "raw/a.bin").exists()


def test_dry_run_recovers_exact_pre_publication_before_local_certificate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request([_artifact("raw-a", "raw/a.bin")])
    _write_artifacts(tmp_path, request["artifacts"])
    args = _cli_args(tmp_path, request, "crash-pre-certificate")
    permanent_root = tmp_path.parent / f"{tmp_path.name}-crash-pre-permanent"
    args.permanent_root = permanent_root
    real_write = reclaim_module.write_json_atomic
    crashed = False

    def crash_before_local_certificate(path: Path, value: Mapping[str, Any]):
        nonlocal crashed
        if path.name == "evidence_certificate.pre_release.json" and not crashed:
            crashed = True
            raise OSError("simulated crash before local pre-release certificate")
        return real_write(path, value)

    monkeypatch.setattr(
        reclaim_module,
        "write_json_atomic",
        crash_before_local_certificate,
    )
    with pytest.raises(OSError, match="before local pre-release certificate"):
        _dry_run(args)
    assert crashed is True
    publication_path = args.output_root / "permanent_capsule.pre_release.json"
    certificate_path = args.output_root / "evidence_certificate.pre_release.json"
    publication_before = publication_path.read_bytes()
    names_before = sorted(path.name for path in permanent_root.iterdir())
    assert not certificate_path.exists()

    monkeypatch.setattr(reclaim_module, "write_json_atomic", real_write)
    artifact_path = tmp_path / "raw/a.bin"
    artifact_before = artifact_path.read_bytes()
    artifact_path.write_bytes(b"changed-after-PRE-publication")
    with pytest.raises(
        EvidenceCertificateError, match="differs from the rebuilt certificate"
    ):
        _dry_run(args)
    assert sorted(path.name for path in permanent_root.iterdir()) == names_before
    assert not certificate_path.exists()
    artifact_path.write_bytes(artifact_before)

    recovered = _dry_run(args)
    assert recovered["ok"] is True
    assert publication_path.read_bytes() == publication_before
    assert sorted(path.name for path in permanent_root.iterdir()) == names_before
    assert certificate_path.is_file()
    assert (args.output_root / "reclamation_intent.json").is_file()


@pytest.mark.parametrize(
    ("crash_filename", "receipt_writer"),
    [
        ("evidence_certificate.pre_release.json", False),
        ("artifact_disposition_manifest.json", False),
        ("storage_disposition_receipt.dry_run.json", True),
        ("reclamation_intent.json", False),
        ("gate_G1_reclamation_result.json", False),
    ],
)
@pytest.mark.parametrize("permanent_enabled", [False, True])
def test_dry_run_recovers_every_exact_durable_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash_filename: str,
    receipt_writer: bool,
    permanent_enabled: bool,
) -> None:
    request = _request([_artifact("raw-a", "raw/a.bin")])
    _write_artifacts(tmp_path, request["artifacts"])
    args = _cli_args(tmp_path, request, f"prefix-{crash_filename}")
    permanent_root = (
        tmp_path.parent / f"{tmp_path.name}-{crash_filename}-permanent"
        if permanent_enabled
        else None
    )
    args.permanent_root = permanent_root
    real_json_write = reclaim_module.write_json_atomic
    real_receipt_write = reclaim_module.write_storage_disposition_receipt_v1
    crashed = False

    def crash_after_json_write(path: Path, value: Mapping[str, Any]):
        nonlocal crashed
        result = real_json_write(path, value)
        if not receipt_writer and path.name == crash_filename and not crashed:
            crashed = True
            raise OSError("simulated crash after durable prefix write")
        return result

    def crash_after_receipt_write(path: Path, value: Mapping[str, Any]):
        nonlocal crashed
        result = real_receipt_write(path, value)
        if receipt_writer and path.name == crash_filename and not crashed:
            crashed = True
            raise OSError("simulated crash after durable prefix write")
        return result

    monkeypatch.setattr(reclaim_module, "write_json_atomic", crash_after_json_write)
    monkeypatch.setattr(
        reclaim_module,
        "write_storage_disposition_receipt_v1",
        crash_after_receipt_write,
    )
    with pytest.raises(OSError, match="after durable prefix write"):
        _dry_run(args)
    assert crashed is True
    durable_before = {
        path.name: path.read_bytes()
        for path in args.output_root.iterdir()
        if path.is_file() and path.name != ".eiel-evidence-closeout.lock"
    }
    permanent_names_before = (
        sorted(path.name for path in permanent_root.iterdir())
        if permanent_root is not None
        else []
    )

    monkeypatch.setattr(reclaim_module, "write_json_atomic", real_json_write)
    monkeypatch.setattr(
        reclaim_module,
        "write_storage_disposition_receipt_v1",
        real_receipt_write,
    )
    recovered = _dry_run(args)

    assert recovered["ok"] is True
    for name, payload in durable_before.items():
        assert (args.output_root / name).read_bytes() == payload
    if permanent_root is not None:
        assert sorted(path.name for path in permanent_root.iterdir()) == (
            permanent_names_before
        )


def test_dry_run_rejects_nonprefix_and_tampered_recovery_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request([_artifact("raw-a", "raw/a.bin")])
    _write_artifacts(tmp_path, request["artifacts"])
    args = _cli_args(tmp_path, request, "bad-prefix")
    args.permanent_root = tmp_path.parent / f"{tmp_path.name}-bad-prefix-permanent"
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "artifact_disposition_manifest.json").write_text("{}\n")
    with pytest.raises(EvidenceCertificateError, match="exact durable prefix"):
        _dry_run(args)

    (args.output_root / "artifact_disposition_manifest.json").unlink()
    unexpected = args.output_root / "unrecognized-state.json"
    unexpected.write_text("{}\n")
    with pytest.raises(EvidenceCertificateError, match="outside its exact durable"):
        _dry_run(args)
    unexpected.unlink()

    hardlink_args = _cli_args(tmp_path, request, "hard-linked-prefix")
    hardlink_args.output_root.mkdir(parents=True, exist_ok=True)
    external = tmp_path / "hard-linked-certificate.json"
    external.write_text("{}\n")
    os.link(
        external,
        hardlink_args.output_root / "evidence_certificate.pre_release.json",
    )
    with pytest.raises(EvidenceCertificateError, match="single-link regular"):
        _dry_run(hardlink_args)

    real_write = reclaim_module.write_json_atomic

    def crash_after_manifest(path: Path, value: Mapping[str, Any]):
        result = real_write(path, value)
        if path.name == "artifact_disposition_manifest.json":
            raise OSError("simulated crash after manifest")
        return result

    monkeypatch.setattr(reclaim_module, "write_json_atomic", crash_after_manifest)
    with pytest.raises(OSError, match="after manifest"):
        _dry_run(args)
    monkeypatch.setattr(reclaim_module, "write_json_atomic", real_write)
    (args.output_root / "artifact_disposition_manifest.json").write_text("{}\n")
    with pytest.raises((ArtifactLifecycleError, EvidenceCertificateError)):
        _dry_run(args)


def test_permanent_capsule_is_disjoint_bounded_idempotent_and_recoverable(
    tmp_path: Path,
) -> None:
    request = _request([_artifact("raw-a", "raw/a.bin")])
    certificate = _certificate(tmp_path, request)
    with pytest.raises(EvidenceCertificateError, match="must be disjoint"):
        publish_permanent_evidence_capsule_v1(
            certificate,
            permanent_root=tmp_path / "permanent",
            source_root=tmp_path,
        )

    permanent_root = tmp_path.parent / f"{tmp_path.name}-capsules"
    first = publish_permanent_evidence_capsule_v1(
        certificate,
        permanent_root=permanent_root,
        source_root=tmp_path,
    )
    second = publish_permanent_evidence_capsule_v1(
        certificate,
        permanent_root=permanent_root,
        source_root=tmp_path,
    )
    assert first == second
    assert first["total_bytes"] == (
        first["json_bytes"] + first["markdown_bytes"] + first["complete_bytes"]
    )
    assert len(list(permanent_root.iterdir())) == 4
    reservation_path = Path(first["final_capacity_reservation_path"])
    assert reservation_path.is_file()
    assert first["final_capacity_reservation_status"] == "RESERVED"
    assert first["final_capacity_reserved_bytes"] == DEFAULT_MAX_CAPSULE_BYTES
    assert (
        validate_permanent_evidence_publication_v1(
            first, certificate=certificate, source_root=tmp_path
        )
        == first
    )

    Path(first["complete_path"]).unlink()
    recovered = publish_permanent_evidence_capsule_v1(
        certificate,
        permanent_root=permanent_root,
        source_root=tmp_path,
    )
    assert recovered == first
    assert Path(first["complete_path"]).is_file()

    with pytest.raises(EvidenceCertificateError, match="exceeds max_capsule_bytes"):
        publish_permanent_evidence_capsule_v1(
            certificate,
            permanent_root=tmp_path.parent / f"{tmp_path.name}-too-small",
            source_root=tmp_path,
            max_capsule_bytes=first["total_bytes"] - 1,
        )
    with pytest.raises(EvidenceCertificateError, match="byte budget"):
        publish_permanent_evidence_capsule_v1(
            certificate,
            permanent_root=tmp_path.parent / f"{tmp_path.name}-root-too-small",
            source_root=tmp_path,
            max_root_bytes=first["total_bytes"] - 1,
        )
    with pytest.raises(EvidenceCertificateError, match="canonical storage policy"):
        publish_permanent_evidence_capsule_v1(
            certificate,
            permanent_root=tmp_path.parent / f"{tmp_path.name}-oversized-policy",
            source_root=tmp_path,
            max_capsule_bytes=DEFAULT_MAX_CAPSULE_BYTES + 1,
        )


def test_data2_permanent_root_override_is_rejected_before_publication(
    tmp_path: Path,
) -> None:
    request = validate_evidence_certificate_request_v1(
        _request([_artifact("raw-a", "raw/a.bin")])
    )
    args = argparse.Namespace(permanent_root=tmp_path / "bypass")
    with pytest.raises(EvidenceCertificateError, match="cannot override"):
        reclaim_module._permanent_root_for_closeout(
            args,
            request,
            evidence_root=Path("/tmp/emergence-test/runtime/fake-run"),
        )


def test_applied_legacy_certificate_replay_can_publish_final_without_raw(
    tmp_path: Path,
) -> None:
    request = _request([_artifact("raw-a", "raw/a.bin")])
    _write_artifacts(tmp_path, request["artifacts"])
    args = _cli_args(tmp_path, request, "migration-close")
    _dry_run(args)
    first = _apply(args)
    assert first["ok"] is True
    assert not (tmp_path / "raw/a.bin").exists()
    assert first["permanent_final_publication_path"] is None

    args.permanent_root = tmp_path.parent / f"{tmp_path.name}-migration-capsule"
    replay = _apply(args)
    assert replay["idempotent_replay"] is True
    assert replay["permanent_pre_release_publication_path"] is None
    assert Path(replay["permanent_final_publication_path"]).is_file()
    final_publication = json.loads(
        Path(replay["permanent_final_publication_path"]).read_text()
    )
    assert final_publication["phase"] == "FINAL"
    assert Path(final_publication["json_path"]).is_file()


def test_non_debug_request_and_certificate_shape_remains_compatible(
    tmp_path: Path,
) -> None:
    request = _request([_artifact("raw-a", "raw/a.bin")])
    normalized = validate_evidence_certificate_request_v1(request)
    certificate = _certificate(tmp_path, request)
    assert "diagnostic_closeout" not in normalized
    assert "diagnostic_closeout" not in certificate
    assert "source_retention_class" not in certificate["artifacts"][0]
    assert certificate["artifacts"][0]["release_eligible"] is True


@pytest.mark.parametrize("source_class", ["R0", "R1", "R2"])
def test_data2_bound_ordinary_r0_r2_rows_preserve_source_class_and_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_class: str,
) -> None:
    request = _request([_artifact("raw-a", "raw/a.bin")])
    _write_artifacts(tmp_path, request["artifacts"])
    _, source_path = _write_source_manifest(
        tmp_path,
        request,
        retention_classes={"raw-a": source_class},
    )
    upgraded = upgrade_evidence_certificate_request_source_bindings_v1(
        request,
        evidence_root=tmp_path,
        source_manifest_paths=[source_path],
    )
    monkeypatch.setattr(
        certificate_module,
        "evidence_root_requires_permanent_capsule_v1",
        lambda _root: True,
    )
    certificate = build_evidence_certificate_v1(upgraded, evidence_root=tmp_path)
    assert (
        validate_certificate_source_manifest_bindings_v1(
            upgraded,
            certificate,
            evidence_root=tmp_path,
        )["binding_count"]
        == 1
    )
    row = certificate["artifacts"][0]
    assert row["source_retention_class"] == source_class
    assert row["retention_class"] == source_class
    assert row["release_eligible"] is True
    capsule = build_permanent_evidence_capsule_v1(certificate)
    assert source_class in capsule["artifact_accounting"]["by_source_retention_class"]


def test_source_binding_validator_allows_only_exact_authenticated_absence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request([_artifact("raw-a", "raw/a.bin")])
    _write_artifacts(tmp_path, request["artifacts"])
    _, source_path = _write_source_manifest(tmp_path, request)
    upgraded = upgrade_evidence_certificate_request_source_bindings_v1(
        request,
        evidence_root=tmp_path,
        source_manifest_paths=[source_path],
    )
    monkeypatch.setattr(
        certificate_module,
        "evidence_root_requires_permanent_capsule_v1",
        lambda _root: True,
    )
    certificate = build_evidence_certificate_v1(upgraded, evidence_root=tmp_path)
    (tmp_path / "raw/a.bin").unlink()
    with pytest.raises(FileNotFoundError):
        validate_certificate_source_manifest_bindings_v1(
            upgraded,
            certificate,
            evidence_root=tmp_path,
        )
    recovered = validate_certificate_source_manifest_bindings_v1(
        upgraded,
        certificate,
        evidence_root=tmp_path,
        authenticated_absent_paths={"raw/a.bin"},
    )
    assert recovered["binding_count"] == 1
    with pytest.raises(EvidenceCertificateError, match="exceed"):
        validate_certificate_source_manifest_bindings_v1(
            upgraded,
            certificate,
            evidence_root=tmp_path,
            authenticated_absent_paths={"raw/not-selected.bin"},
        )
    _write_artifacts(tmp_path, request["artifacts"])
    with pytest.raises(EvidenceCertificateError, match="currently absent"):
        validate_certificate_source_manifest_bindings_v1(
            upgraded,
            certificate,
            evidence_root=tmp_path,
            authenticated_absent_paths={"raw/a.bin"},
        )


@pytest.mark.parametrize("source_class", ["R3", "R4", "R5", "RX"])
def test_data2_bound_ordinary_durable_or_unknown_rows_cannot_be_reclassified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_class: str,
) -> None:
    request = _request([_artifact("protected", "raw/protected.bin")])
    _write_artifacts(tmp_path, request["artifacts"])
    _, source_path = _write_source_manifest(
        tmp_path,
        request,
        retention_classes={"protected": source_class},
    )
    upgraded = upgrade_evidence_certificate_request_source_bindings_v1(
        request,
        evidence_root=tmp_path,
        source_manifest_paths=[source_path],
    )
    monkeypatch.setattr(
        certificate_module,
        "evidence_root_requires_permanent_capsule_v1",
        lambda _root: True,
    )
    certificate = build_evidence_certificate_v1(upgraded, evidence_root=tmp_path)
    row = certificate["artifacts"][0]
    assert row["source_retention_class"] == source_class
    assert row["release_eligible"] is False
    assert row["retention_class"] == source_class

    malicious = copy.deepcopy(certificate)
    malicious.pop("certificate_sha256")
    malicious_row = malicious["artifacts"][0]
    malicious_row.update(
        {
            "disposition": "RELEASE_NOW",
            "retention_class": "R2",
            "release_eligible": True,
        }
    )
    size = int(malicious_row["bytes"])
    malicious["disposition_summary"].update(
        {
            "certified_releasable_bytes": size,
            "released_bytes": 0,
            "retained_bytes": 0,
        }
    )
    malicious.pop("certificate_id")
    malicious["certificate_id"] = "eiel-cert-" + canonical_sha256(malicious)[:24]
    malicious["certificate_sha256"] = canonical_sha256(malicious)
    with pytest.raises(
        EvidenceCertificateError,
        match="cannot relabel or release a bound",
    ):
        validate_evidence_certificate_v1(malicious)


def test_data2_mixed_source_inventory_releases_only_r0_r2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(
        [
            _artifact("raw-r1", "raw/r1.bin"),
            _artifact("durable-r4", "raw/r4.bin"),
            _artifact("unknown-rx", "raw/rx.bin"),
        ]
    )
    _write_artifacts(tmp_path, request["artifacts"])
    _, source_path = _write_source_manifest(
        tmp_path,
        request,
        retention_classes={
            "raw-r1": "R1",
            "durable-r4": "R4",
            "unknown-rx": "RX",
        },
    )
    upgraded = upgrade_evidence_certificate_request_source_bindings_v1(
        request,
        evidence_root=tmp_path,
        source_manifest_paths=[source_path],
    )
    monkeypatch.setattr(
        certificate_module,
        "evidence_root_requires_permanent_capsule_v1",
        lambda _root: True,
    )
    certificate = build_evidence_certificate_v1(upgraded, evidence_root=tmp_path)
    rows = {row["artifact_id"]: row for row in certificate["artifacts"]}
    assert rows["raw-r1"]["release_eligible"] is True
    assert rows["durable-r4"]["release_eligible"] is False
    assert rows["unknown-rx"]["release_eligible"] is False


def test_data2_direct_prune_rechecks_capsule_and_current_source_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request([_artifact("raw-a", "raw/a.bin")])
    _write_artifacts(tmp_path, request["artifacts"])
    _, source_path = _write_source_manifest(tmp_path, request)
    upgraded = upgrade_evidence_certificate_request_source_bindings_v1(
        request,
        evidence_root=tmp_path,
        source_manifest_paths=[source_path],
    )
    monkeypatch.setattr(
        certificate_module,
        "evidence_root_requires_permanent_capsule_v1",
        lambda _root: True,
    )
    certificate = build_evidence_certificate_v1(upgraded, evidence_root=tmp_path)
    permanent = publish_permanent_evidence_capsule_v1(
        certificate,
        permanent_root=tmp_path.parent / f"{tmp_path.name}-direct-permanent",
        source_root=tmp_path,
    )
    manifest = certificate_disposition_manifest_v1(
        upgraded,
        certificate,
        evidence_root=tmp_path,
        permanent_publication=permanent,
    )
    plan = build_prune_plan(tmp_path, set(prune_module.CATEGORIES), manifest)
    context = _certificate_prune_context(
        upgraded,
        certificate,
        manifest,
        plan,
    )
    with pytest.raises(EvidenceCertificateError, match="publication"):
        apply_prune_plan(
            tmp_path,
            plan,
            disposition_manifest=manifest,
            certificate_release_context=context,
        )
    assert (tmp_path / "raw/a.bin").is_file()

    context["permanent_publication"] = permanent
    original_source = json.loads(source_path.read_text())
    replacement_source = copy.deepcopy(original_source)
    replacement_source.pop("manifest_sha256")
    replacement_source["timestamp_utc"] = "2026-08-27T13:00:00Z"
    replacement_source["manifest_sha256"] = canonical_sha256(replacement_source)
    source_path.write_text(
        json.dumps(replacement_source, indent=2, sort_keys=True) + "\n"
    )
    with pytest.raises(EvidenceCertificateError, match="manifest SHA-256"):
        apply_prune_plan(
            tmp_path,
            plan,
            disposition_manifest=manifest,
            certificate_release_context=context,
        )
    assert (tmp_path / "raw/a.bin").is_file()

    source_path.write_text(json.dumps(original_source, indent=2, sort_keys=True) + "\n")
    (tmp_path / "raw/a.bin").write_bytes(b"changed-after-certificate")
    with pytest.raises(EvidenceCertificateError, match="actual bytes"):
        apply_prune_plan(
            tmp_path,
            plan,
            disposition_manifest=manifest,
            certificate_release_context=context,
        )
    assert (tmp_path / "raw/a.bin").is_file()


def test_direct_prune_holds_bound_source_fence_after_recheck_before_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, _, _, source_root = _combined_bound_fixture(tmp_path)
    monkeypatch.setattr(
        certificate_module,
        "evidence_root_requires_permanent_capsule_v1",
        lambda _root: True,
    )
    certificate = build_evidence_certificate_v1(request, evidence_root=tmp_path)
    permanent = publish_permanent_evidence_capsule_v1(
        certificate,
        permanent_root=tmp_path.parent / f"{tmp_path.name}-fenced-direct",
        source_root=tmp_path,
    )
    manifest = certificate_disposition_manifest_v1(
        request,
        certificate,
        evidence_root=tmp_path,
        permanent_publication=permanent,
    )
    plan = build_prune_plan(tmp_path, set(prune_module.CATEGORIES), manifest)
    context = _certificate_prune_context(request, certificate, manifest, plan)
    context["permanent_publication"] = permanent
    original_rename = prune_module._rename_noreplace
    attempted = threading.Event()
    acquired = threading.Event()
    reseal_thread: threading.Thread | None = None

    def rename_after_reseal_attempt(src, dst, *call_args, **call_kwargs):
        nonlocal reseal_thread
        if reseal_thread is None:

            def attempt_canonical_reseal() -> None:
                attempted.set()
                with reclaim_module.run_artifact_transaction_fence_v1(source_root):
                    acquired.set()

            reseal_thread = threading.Thread(target=attempt_canonical_reseal)
            reseal_thread.start()
            assert attempted.wait(timeout=2)
            assert acquired.wait(timeout=0.1) is False
        return original_rename(src, dst, *call_args, **call_kwargs)

    monkeypatch.setattr(prune_module, "_rename_noreplace", rename_after_reseal_attempt)
    result = apply_prune_plan(
        tmp_path,
        plan,
        disposition_manifest=manifest,
        certificate_release_context=context,
    )
    assert result["failures"] == []
    assert reseal_thread is not None
    reseal_thread.join(timeout=2)
    assert acquired.is_set()


def test_close_gate_holds_bound_source_fence_through_durable_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, _, _, source_root = _combined_bound_fixture(tmp_path)
    monkeypatch.setattr(
        certificate_module,
        "evidence_root_requires_permanent_capsule_v1",
        lambda _root: True,
    )
    monkeypatch.setattr(
        reclaim_module,
        "evidence_root_requires_permanent_capsule_v1",
        lambda _root: True,
    )
    args = _cli_args(tmp_path, request, "source-fenced-close")
    args.permanent_root = tmp_path.parent / f"{tmp_path.name}-fenced-close"
    _dry_run(args)
    original_rename = prune_module._rename_noreplace
    original_write_receipt = reclaim_module.write_storage_disposition_receipt_v1
    attempted = threading.Event()
    acquired = threading.Event()
    receipt_written_while_blocked = False
    reseal_thread: threading.Thread | None = None

    def rename_after_reseal_attempt(src, dst, *call_args, **call_kwargs):
        nonlocal reseal_thread
        if reseal_thread is None:

            def attempt_canonical_reseal() -> None:
                attempted.set()
                with reclaim_module.run_artifact_transaction_fence_v1(source_root):
                    acquired.set()

            reseal_thread = threading.Thread(target=attempt_canonical_reseal)
            reseal_thread.start()
            assert attempted.wait(timeout=2)
            assert acquired.wait(timeout=0.1) is False
        return original_rename(src, dst, *call_args, **call_kwargs)

    def write_receipt_while_fenced(path: Path, receipt: Mapping[str, Any]) -> Path:
        nonlocal receipt_written_while_blocked
        assert attempted.is_set()
        assert acquired.is_set() is False
        receipt_written_while_blocked = True
        return original_write_receipt(path, receipt)

    monkeypatch.setattr(prune_module, "_rename_noreplace", rename_after_reseal_attempt)
    monkeypatch.setattr(
        reclaim_module,
        "write_storage_disposition_receipt_v1",
        write_receipt_while_fenced,
    )
    result = _apply(args)
    assert result["ok"] is True
    assert receipt_written_while_blocked is True
    assert reseal_thread is not None
    reseal_thread.join(timeout=2)
    assert acquired.is_set()


def test_source_binding_upgrade_accepts_only_exact_legacy_partial_and_cli_copy(
    tmp_path: Path,
) -> None:
    request = _request([_artifact("raw-a", "raw/a.bin")])
    _write_artifacts(tmp_path, request["artifacts"])
    _, source_path = _write_source_manifest(tmp_path, request)
    complete = upgrade_evidence_certificate_request_source_bindings_v1(
        request,
        evidence_root=tmp_path,
        source_manifest_paths=[source_path],
    )
    binding = complete["artifacts"][0]["source_manifest_binding"]
    digest_name = "sha256" if "sha256" in binding else "tree_sha256"
    legacy = copy.deepcopy(request)
    legacy["artifacts"][0]["source_manifest_binding"] = {
        name: binding[name]
        for name in ("artifact_id", "bytes", "retention_class", digest_name)
    }
    with pytest.raises(EvidenceCertificateError, match="exact manifest"):
        validate_evidence_certificate_request_v1(legacy)
    upgraded = upgrade_evidence_certificate_request_source_bindings_v1(
        legacy,
        evidence_root=tmp_path,
        source_manifest_paths=[source_path],
    )
    assert upgraded == complete
    assert "manifest_path" not in legacy["artifacts"][0]["source_manifest_binding"]

    request_path = tmp_path / "legacy-request.json"
    request_path.write_text(json.dumps(legacy, indent=2, sort_keys=True) + "\n")
    request_before = request_path.read_bytes()
    output_path = tmp_path / "upgraded-request.json"
    result = reclaim_module._upgrade_request_bindings(
        argparse.Namespace(
            request=request_path,
            evidence_root=tmp_path,
            source_manifest=[source_path],
            diagnostic_upgrade_spec=None,
            out=output_path,
            max_manifests=64,
            max_rows=16_384,
        )
    )
    assert result["apply_performed"] is False
    assert result["source_binding_count"] == 1
    assert request_path.read_bytes() == request_before
    assert (
        validate_evidence_certificate_request_v1(json.loads(output_path.read_text()))
        == complete
    )

    conflict = copy.deepcopy(legacy)
    conflict["artifacts"][0]["source_manifest_binding"][digest_name] = "0" * 64
    with pytest.raises(EvidenceCertificateError, match="differs from the named"):
        upgrade_evidence_certificate_request_source_bindings_v1(
            conflict,
            evidence_root=tmp_path,
            source_manifest_paths=[source_path],
        )


def test_debug_upgrade_spec_requires_exact_one_to_one_source_mapping(
    tmp_path: Path,
) -> None:
    complete, _ = _debug_request(
        tmp_path,
        [_artifact("raw-a", "raw/a.bin"), _artifact("raw-b", "raw/b.bin")],
    )
    old = copy.deepcopy(complete)
    diagnostic_closeout = old.pop("diagnostic_closeout")
    mappings = []
    for row in old["artifacts"]:
        mappings.append(
            {
                "artifact_id": row["artifact_id"],
                "source_artifact_id": row.pop("source_artifact_id"),
                "source_retention_class": row.pop("source_retention_class"),
                "diagnostic_closeout_disposition": row.pop(
                    "diagnostic_closeout_disposition"
                ),
            }
        )
        row.pop("source_manifest_binding")
    spec = {
        "schema_version": DIAGNOSTIC_UPGRADE_SPEC_SCHEMA_VERSION,
        "diagnostic_closeout": diagnostic_closeout,
        "artifact_mappings": mappings,
    }
    source_path = tmp_path / diagnostic_closeout["source_manifest"]["path"]
    upgraded = upgrade_evidence_certificate_request_source_bindings_v1(
        old,
        evidence_root=tmp_path,
        source_manifest_paths=[source_path],
        diagnostic_upgrade_spec=spec,
    )
    assert upgraded == validate_evidence_certificate_request_v1(complete)

    with pytest.raises(EvidenceCertificateError, match="explicit diagnostic"):
        upgrade_evidence_certificate_request_source_bindings_v1(
            old,
            evidence_root=tmp_path,
            source_manifest_paths=[source_path],
        )
    incomplete = copy.deepcopy(spec)
    incomplete["artifact_mappings"].pop()
    with pytest.raises(EvidenceCertificateError, match="exactly cover"):
        upgrade_evidence_certificate_request_source_bindings_v1(
            old,
            evidence_root=tmp_path,
            source_manifest_paths=[source_path],
            diagnostic_upgrade_spec=incomplete,
        )


def test_debug_upgrade_spec_supports_marker_free_source_identity(
    tmp_path: Path,
) -> None:
    complete, _ = _debug_request(
        tmp_path,
        [_artifact("raw-a", "raw/a.bin")],
        source_live_reference_status="none",
        include_source_live_marker_closeout=False,
    )
    old = copy.deepcopy(complete)
    diagnostic_closeout = old.pop("diagnostic_closeout")
    mappings = []
    for row in old["artifacts"]:
        mappings.append(
            {
                "artifact_id": row["artifact_id"],
                "source_artifact_id": row.pop("source_artifact_id"),
                "source_retention_class": row.pop("source_retention_class"),
                "diagnostic_closeout_disposition": row.pop(
                    "diagnostic_closeout_disposition"
                ),
            }
        )
        row.pop("source_manifest_binding")
    spec = {
        "schema_version": DIAGNOSTIC_UPGRADE_SPEC_SCHEMA_VERSION,
        "diagnostic_closeout": diagnostic_closeout,
        "artifact_mappings": mappings,
    }
    source_path = tmp_path / diagnostic_closeout["source_manifest"]["path"]

    upgraded = upgrade_evidence_certificate_request_source_bindings_v1(
        old,
        evidence_root=tmp_path,
        source_manifest_paths=[source_path],
        diagnostic_upgrade_spec=spec,
    )

    assert upgraded == validate_evidence_certificate_request_v1(complete)
    assert "runtime_root" not in upgraded["diagnostic_closeout"]["source_manifest"]


def _applied_receipt_fixture(
    root: Path,
    request: Mapping[str, Any],
    certificate: Mapping[str, Any],
    manifest: Mapping[str, Any],
    pre_certificate_path: Path,
) -> dict[str, Any]:
    plan = build_prune_plan(root, set(prune_module.CATEGORIES), manifest)
    released = int(plan["selected_bytes"])
    return build_storage_disposition_receipt_v1(
        manifest=manifest,
        bytes_before=int(manifest["summary"]["produced_bytes"]),
        bytes_retained=int(manifest["summary"]["produced_bytes"]) - released,
        bytes_released=released,
        bytes_unresolved_rx=int(manifest["summary"]["unresolved_rx_bytes"]),
        removed=plan["selected"],
        authorization_reason="test exact certificate release",
        replacement_references=[
            {
                "certificate_id": certificate["certificate_id"],
                "certificate_sha256": certificate["certificate_sha256"],
            }
        ],
        retention_references=[str(pre_certificate_path)],
        plan_sha256=manifest["plan_sha256"],
        timestamp_utc="2026-08-27T12:00:00Z",
        result="applied",
    )


def test_finalization_requires_exact_receipt_and_actual_completed_release(
    tmp_path: Path,
) -> None:
    request = _request([_artifact("raw-a", "raw/a.bin")])
    _write_artifacts(tmp_path, request["artifacts"])
    args = _cli_args(tmp_path, request, "final-integrity")
    dry = _dry_run(args)
    certificate = json.loads(Path(dry["certificate_path"]).read_text())
    manifest = json.loads(Path(dry["manifest_path"]).read_text())
    false_receipt = _applied_receipt_fixture(
        tmp_path,
        request,
        certificate,
        manifest,
        Path(dry["certificate_path"]),
    )
    with pytest.raises(EvidenceCertificateError, match="still exist"):
        finalize_evidence_certificate_v1(
            certificate,
            storage_receipt=false_receipt,
            manifest=manifest,
        )

    applied = _apply(args)
    receipt = json.loads(Path(applied["storage_disposition_receipt_path"]).read_text())
    final = json.loads(Path(applied["certificate_path"]).read_text())
    assert (
        validate_storage_receipt_certificate_binding_v1(
            receipt,
            manifest=manifest,
            pre_certificate=certificate,
        )["receipt_sha256"]
        == receipt["receipt_sha256"]
    )
    assert validate_evidence_certificate_v1(final) == final
    assert (
        finalize_evidence_certificate_v1(
            certificate,
            storage_receipt=receipt,
            manifest=manifest,
        )
        == final
    )

    for field, value, message in (
        ("manifest_sha256", "0" * 64, "differs from manifest"),
        ("bytes_released", int(receipt["bytes_released"]) + 1, "byte totals"),
    ):
        tampered = copy.deepcopy(receipt)
        tampered[field] = value
        tampered.pop("receipt_sha256")
        tampered["receipt_sha256"] = canonical_sha256(tampered)
        with pytest.raises(EvidenceCertificateError, match=message):
            finalize_evidence_certificate_v1(
                certificate,
                storage_receipt=tampered,
                manifest=manifest,
            )
    extra_schema = copy.deepcopy(receipt)
    extra_schema["action_failures"] = []
    extra_schema.pop("receipt_sha256")
    extra_schema["receipt_sha256"] = canonical_sha256(extra_schema)
    with pytest.raises(EvidenceCertificateError, match="exact schema"):
        finalize_evidence_certificate_v1(
            certificate,
            storage_receipt=extra_schema,
            manifest=manifest,
        )

    bad_final = copy.deepcopy(final)
    bad_final["artifacts"][0]["released"] = False
    bad_final["artifacts"][0]["released_at_utc"] = None
    bad_final["artifacts"][0]["release_receipt_sha256"] = None
    bad_final.pop("certificate_sha256")
    bad_final["certificate_sha256"] = canonical_sha256(bad_final)
    with pytest.raises(EvidenceCertificateError, match="omits an eligible"):
        validate_evidence_certificate_v1(bad_final)


def test_apply_recovers_after_final_publication_before_applied_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request([_artifact("raw-a", "raw/a.bin")])
    _write_artifacts(tmp_path, request["artifacts"])
    _, source_path = _write_source_manifest(tmp_path, request)
    upgraded = upgrade_evidence_certificate_request_source_bindings_v1(
        request,
        evidence_root=tmp_path,
        source_manifest_paths=[source_path],
    )
    monkeypatch.setattr(
        certificate_module,
        "evidence_root_requires_permanent_capsule_v1",
        lambda _root: True,
    )
    monkeypatch.setattr(
        reclaim_module,
        "evidence_root_requires_permanent_capsule_v1",
        lambda _root: True,
    )
    args = _cli_args(tmp_path, upgraded, "crash-final")
    permanent_root, root_limit = _configure_near_cap_permanent_root(
        tmp_path,
        upgraded,
        args,
        label="crash-final",
    )
    _dry_run(args)
    real_write = reclaim_module.write_json_atomic
    crashed = False

    def crash_before_applied_intent(path: Path, value: Mapping[str, Any]):
        nonlocal crashed
        if value.get("status") == "APPLIED" and path.name == "reclamation_intent.json":
            crashed = True
            raise OSError("simulated crash before APPLIED intent")
        return real_write(path, value)

    monkeypatch.setattr(
        reclaim_module,
        "write_json_atomic",
        crash_before_applied_intent,
    )
    with pytest.raises(OSError, match="simulated crash"):
        _apply(args)
    assert crashed is True
    output_root = args.output_root
    final_before = json.loads(
        (output_root / "permanent_capsule.final.json").read_text()
    )
    names_before = sorted(path.name for path in permanent_root.iterdir())
    assert (
        json.loads((output_root / "reclamation_intent.json").read_text())["status"]
        == "APPLYING"
    )

    monkeypatch.setattr(reclaim_module, "write_json_atomic", real_write)
    recovered = _apply(args)
    assert recovered["idempotent_replay"] is True
    final_after = json.loads((output_root / "permanent_capsule.final.json").read_text())
    assert final_after == final_before
    assert sorted(path.name for path in permanent_root.iterdir()) == names_before
    actual_bytes, reserved_bytes, consumed = capsule_module._root_usage(permanent_root)
    assert actual_bytes <= root_limit
    assert reserved_bytes == 0
    assert final_after["final_capacity_reservation_sha256"] in consumed
    assert (
        json.loads((output_root / "reclamation_intent.json").read_text())["status"]
        == "APPLIED"
    )


def test_apply_recovers_global_final_before_local_publication_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request([_artifact("raw-a", "raw/a.bin")])
    _write_artifacts(tmp_path, request["artifacts"])
    _, source_path = _write_source_manifest(tmp_path, request)
    upgraded = upgrade_evidence_certificate_request_source_bindings_v1(
        request,
        evidence_root=tmp_path,
        source_manifest_paths=[source_path],
    )
    monkeypatch.setattr(
        certificate_module,
        "evidence_root_requires_permanent_capsule_v1",
        lambda _root: True,
    )
    monkeypatch.setattr(
        reclaim_module,
        "evidence_root_requires_permanent_capsule_v1",
        lambda _root: True,
    )
    args = _cli_args(tmp_path, upgraded, "crash-local-final-publication")
    permanent_root, root_limit = _configure_near_cap_permanent_root(
        tmp_path,
        upgraded,
        args,
        label="crash-local-final-publication",
    )
    _dry_run(args)
    real_write = reclaim_module.write_json_atomic
    crashed = False

    def crash_before_local_final_publication(path: Path, value: Mapping[str, Any]):
        nonlocal crashed
        if path.name == "permanent_capsule.final.json" and not crashed:
            crashed = True
            raise OSError("simulated crash before local FINAL publication receipt")
        return real_write(path, value)

    monkeypatch.setattr(
        reclaim_module,
        "write_json_atomic",
        crash_before_local_final_publication,
    )
    with pytest.raises(OSError, match="before local FINAL publication receipt"):
        _apply(args)
    assert crashed is True
    local_final = args.output_root / "permanent_capsule.final.json"
    assert not local_final.exists()
    names_before = sorted(path.name for path in permanent_root.iterdir())
    actual_bytes, reserved_bytes, consumed = capsule_module._root_usage(permanent_root)
    pre_publication = json.loads(
        (args.output_root / "permanent_capsule.pre_release.json").read_text()
    )
    assert actual_bytes <= root_limit
    assert reserved_bytes == 0
    assert pre_publication["final_capacity_reservation_sha256"] in consumed
    assert (
        json.loads((args.output_root / "reclamation_intent.json").read_text())["status"]
        == "APPLYING"
    )

    monkeypatch.setattr(reclaim_module, "write_json_atomic", real_write)
    recovered = _apply(args)
    assert recovered["ok"] is True
    assert recovered["idempotent_replay"] is True
    assert local_final.is_file()
    assert sorted(path.name for path in permanent_root.iterdir()) == names_before
    final_publication = json.loads(local_final.read_text())
    assert final_publication["phase"] == "FINAL"
    assert final_publication["final_capacity_reservation_sha256"] in consumed
    assert (
        json.loads((args.output_root / "reclamation_intent.json").read_text())["status"]
        == "APPLIED"
    )
