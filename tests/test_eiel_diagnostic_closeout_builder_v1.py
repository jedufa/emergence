from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pytest

from engine.campaigns.artifact_lifecycle import (
    DEBUG_FORENSIC,
    build_artifact_disposition_manifest_v1,
    build_sealed_artifact_inventory_row_v1,
    lifecycle_profile_contract_v1,
    validate_run_artifact_lifecycle_contract_v1,
)
from engine.campaigns.canonical import canonical_sha256
from engine.campaigns.diagnostic_closeout_builder import (
    DEBUG_CLOSEOUT_REVIEW_INPUT_SCHEMA_VERSION,
    MAX_DEBUG_CLOSEOUT_REPORT_BYTES,
    build_debug_closeout_bundle_v1,
    draft_debug_closeout_review_spec_v1,
    validate_debug_closeout_review_receipt_v1,
    validate_debug_closeout_review_spec_v1,
)
from engine.campaigns.evidence_certificate import (
    EvidenceCertificateError,
    build_evidence_certificate_v1,
    utc_now,
)
from engine.campaigns.permanent_evidence_capsule import (
    build_permanent_evidence_capsule_v1,
    render_permanent_evidence_report_v1,
)
from tools.scripts import eiel_evidence_certificate_reclaim as reclaim


def _debug_contract() -> dict[str, Any]:
    contract = lifecycle_profile_contract_v1(
        DEBUG_FORENSIC,
        total_max_bytes=10_000_000,
        checkpoint_schedule=[1],
    )
    contract["retention"]["unknown_artifact_scope"] = "exact_object"
    contract.pop("contract_sha256", None)
    return validate_run_artifact_lifecycle_contract_v1(contract)


def _fixture(tmp_path: Path, *, live_reference_status: str = "none") -> dict[str, Any]:
    root = tmp_path / "evidence"
    root.mkdir(parents=True)
    git_identity = {
        "repository_root": "/repo/emergence",
        "commit": "1" * 40,
        "working_tree_clean": True,
        "working_tree_status_lines": [],
    }
    payloads = {
        "raw/state.bin": b"raw-state-not-for-report",
        "raw/decision.bin": b"raw-decision-input",
        "reports/execution_request.json": (
            json.dumps({"source_identity": {"git": git_identity}}, sort_keys=True)
            + "\n"
        ).encode(),
        "reports/execution_result.json": (
            json.dumps(
                {
                    "status": "completed_without_supported_result",
                    "source_identity": {"git": git_identity},
                },
                sort_keys=True,
            )
            + "\n"
        ).encode(),
        "assets/model.json": b'{"asset":true}\n',
        "audit/certificate.json": b'{"audit":"compact"}\n',
        "unknown/foreign.bin": b"unknown",
    }
    classes = {
        "raw/state.bin": "R1",
        "raw/decision.bin": "R3",
        "reports/execution_request.json": "R4",
        "reports/execution_result.json": "R4",
        "assets/model.json": "R4",
        "audit/certificate.json": "R5",
        "unknown/foreign.bin": "RX",
    }
    inventory = []
    for relative, payload in payloads.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        inventory.append(
            build_sealed_artifact_inventory_row_v1(
                run_root=root,
                path=path,
                artifact_family="debug-fixture",
                retention_class=classes[relative],
                owner_writer="debug-fixture-writer",
            )
        )
    contract = _debug_contract()
    manifest = build_artifact_disposition_manifest_v1(
        {
            "campaign_id": "debug-fixture-campaign",
            "plan_sha256": "a" * 64,
            "artifact_lifecycle": contract,
        },
        run_root=root,
        terminal_status="completed",
        live_reference_status=live_reference_status,
        audit_hold_status="not_required",
        qualification_hold_status="not_required",
        artifact_inventory=inventory,
        inventory_complete=True,
    )
    manifest_path = root / "control" / "artifact_disposition_manifest.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    contract_path = inputs / "lifecycle_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    review_input = {
        "schema_version": DEBUG_CLOSEOUT_REVIEW_INPUT_SCHEMA_VERSION,
        "programme_id": "debug-fixture-closeout",
        "work_id": "work-debug-fixture-closeout",
        "obligation_id": "obligation-debug-closeout",
        "bounded_claim": (
            "the exact completed diagnostic inventory is accounted for and raw rows "
            "are no longer needed for the bounded diagnostic purpose"
        ),
        "reviewed_at_utc": "2026-08-28T12:00:00Z",
        "reviewer": {
            "identity": "same-identity-reviewer",
            "independence": "REVIEWER_INDEPENDENCE_UNAVAILABLE",
            "status": "accepted_with_limitations",
        },
        "decision": "NO_SUPPORTED_RESULT",
        "conclusion": (
            "execution completed, but the named result artifact supports no scientific "
            "or qualification result; the raw diagnostic is closed"
        ),
        "limitations": [
            "same-identity reduced-assurance review",
            "the run supports no scientific or qualification conclusion",
        ],
        "criteria": [
            {
                "criterion": "supported-result availability",
                "status": "failed",
                "detail": "the manifest-bound execution result records no supported result",
            },
            {
                "criterion": "complete exact inventory accounting",
                "status": "passed",
                "detail": "every producer-declared source row is bound by the request",
            },
        ],
        "measurements": {"supported_result_count": 0},
        "supersession": {
            "retention_classes": ["R1", "R3"],
            "artifact_ids": [],
            "preserve_artifact_ids": [],
            "evidence_class": "RAW_TRANSIENT_EVIDENCE",
            "rationale": "raw diagnostic bytes are superseded by the compact closeout",
        },
        "source_live_marker_audit": {
            "status": "CLOSED_BY_CURRENT_STATE_AUDIT",
            "audited_at_utc": utc_now(),
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
            "audit_method": (
                "exact run-root process, relay, current owner reference, and canonical "
                "reservation registry checks"
            ),
        },
        "reproduction": {
            "prerequisites": ["source commit and the declared execution inputs"],
            "commands": ["rerun the exact manifest-bound diagnostic request"],
            "verification_rules": [
                "compare the new result against the named execution result contract"
            ],
            "expected_results": [
                {"name": "supported_result", "status": "not_established"}
            ],
            "guarantee": "REVALIDATION_ONLY",
        },
        "environment": {"python": "3.14", "route": "fixture"},
        "device": {"kind": "host-fixture"},
        "no_claims": [
            "no scientific claim is granted by this diagnostic closeout",
            "no qualification claim is granted by this diagnostic closeout",
        ],
        "confirmations": {
            "complete_inventory_reviewed": True,
            "no_unresolved_anomalies": True,
            "no_live_state": True,
            "no_active_exact_byte_holds": True,
        },
    }
    review_input_path = inputs / "review_input.json"
    review_input_path.write_text(
        json.dumps(review_input, indent=2, sort_keys=True) + "\n"
    )
    return {
        "root": root,
        "contract": contract,
        "contract_path": contract_path,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "review_input": review_input,
        "review_input_path": review_input_path,
    }


def _draft(fixture: dict[str, Any]) -> dict[str, Any]:
    return draft_debug_closeout_review_spec_v1(
        evidence_root=fixture["root"],
        source_manifest_path=fixture["manifest_path"],
        artifact_lifecycle_contract=fixture["contract"],
        review_input=fixture["review_input"],
        supporting_artifacts=[
            {
                "role": "execution-request",
                "path": "reports/execution_request.json",
            },
            {"role": "execution-result", "path": "reports/execution_result.json"},
        ],
    )


@pytest.mark.parametrize(
    "independence",
    ["REVIEWER_INDEPENDENCE_UNAVAILABLE", "INDEPENDENT_REVIEW"],
)
def test_builder_preserves_durable_rows_and_binds_support(
    tmp_path: Path, independence: str
) -> None:
    fixture = _fixture(tmp_path)
    fixture["review_input"]["reviewer"]["independence"] = independence
    fixture["review_input"]["reviewer"]["identity"] = (
        "independent-reviewer"
        if independence == "INDEPENDENT_REVIEW"
        else "same-identity-reviewer"
    )
    spec = _draft(fixture)
    assert validate_debug_closeout_review_spec_v1(spec) == spec
    assert spec["reviewer"]["independence"] == independence
    support = next(
        row for row in spec["control_receipts"] if row["role"] == "execution-result"
    )
    result_row = next(
        row
        for row in fixture["manifest"]["artifacts"]
        if row["path"] == "reports/execution_result.json"
    )
    assert support["artifact_id"] == result_row["artifact_id"]
    assert support["sha256"] == result_row["sha256"]
    assert spec["source"] == {
        "repository": "/repo/emergence",
        "commit_sha": "1" * 40,
    }

    bundle = build_debug_closeout_bundle_v1(
        evidence_root=fixture["root"],
        source_manifest_path=fixture["manifest_path"],
        review_spec=spec,
    )
    assert (
        validate_debug_closeout_review_receipt_v1(bundle["review_receipt"])
        == (bundle["review_receipt"])
    )
    request = bundle["request"]
    assert len(request["artifacts"]) == len(fixture["manifest"]["artifacts"])
    by_path = {row["path"]: row for row in request["artifacts"]}
    assert by_path["raw/state.bin"]["diagnostic_closeout_disposition"] == (
        "SUPERSEDED_BY_CERTIFICATE"
    )
    assert by_path["raw/decision.bin"]["diagnostic_closeout_disposition"] == (
        "SUPERSEDED_BY_CERTIFICATE"
    )
    for relative in (
        "reports/execution_result.json",
        "reports/execution_request.json",
        "assets/model.json",
        "audit/certificate.json",
        "unknown/foreign.bin",
    ):
        assert by_path[relative]["diagnostic_closeout_disposition"] == "RETAIN"
    assert request["result"]["decision"] == "NO_SUPPORTED_RESULT"
    supporting = request["result"]["digests"]["supporting_artifacts"]
    supporting_result = next(
        row for row in supporting if row["role"] == "execution-result"
    )
    assert supporting_result["path"] == "reports/execution_result.json"
    assert "reports/execution_result.json" in bundle["report_markdown"]
    assert len(bundle["report_markdown"].encode()) < MAX_DEBUG_CLOSEOUT_REPORT_BYTES
    assert "raw-state-not-for-report" not in bundle["report_markdown"]

    certificate = build_evidence_certificate_v1(request, evidence_root=fixture["root"])
    certificate_by_path = {row["path"]: row for row in certificate["artifacts"]}
    assert certificate_by_path["raw/state.bin"]["release_eligible"] is True
    assert certificate_by_path["raw/decision.bin"]["retention_class"] == "R2"
    assert certificate_by_path["assets/model.json"]["retention_class"] == "R4"
    assert certificate_by_path["audit/certificate.json"]["retention_class"] == "R5"
    assert certificate_by_path["unknown/foreign.bin"]["retention_class"] == "RX"
    capsule = build_permanent_evidence_capsule_v1(certificate)
    permanent_report = render_permanent_evidence_report_v1(capsule)
    assert "reports/execution_result.json" in permanent_report
    assert result_row["sha256"] in permanent_report


@pytest.mark.parametrize("protected_class", ["R4", "R5", "RX"])
def test_draft_rejects_explicit_protected_supersession(
    tmp_path: Path, protected_class: str
) -> None:
    fixture = _fixture(tmp_path)
    protected = next(
        row
        for row in fixture["manifest"]["artifacts"]
        if row["retention_class"] == protected_class
        and row["path"]
        not in {"reports/execution_request.json", "reports/execution_result.json"}
    )
    fixture["review_input"]["supersession"]["retention_classes"] = []
    fixture["review_input"]["supersession"]["artifact_ids"] = [protected["artifact_id"]]
    with pytest.raises(EvidenceCertificateError, match="protected R4/R5/RX"):
        _draft(fixture)


def test_draft_rejects_terminal_status_only_and_tampered_support(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    with pytest.raises(EvidenceCertificateError, match="result/control artifact"):
        draft_debug_closeout_review_spec_v1(
            evidence_root=fixture["root"],
            source_manifest_path=fixture["manifest_path"],
            artifact_lifecycle_contract=fixture["contract"],
            review_input=fixture["review_input"],
            supporting_artifacts=[],
        )
    result_path = fixture["root"] / "reports" / "execution_result.json"
    payload = bytearray(result_path.read_bytes())
    payload[-2] = ord(" ")
    result_path.write_bytes(payload)
    with pytest.raises(EvidenceCertificateError, match="SHA-256 mismatch"):
        _draft(fixture)


def test_draft_rejects_result_git_identity_mismatch(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    result_path = fixture["root"] / "reports" / "execution_result.json"
    payload = (
        json.dumps(
            {
                "status": "completed_without_supported_result",
                "source_identity": {
                    "git": {
                        "repository_root": "/repo/emergence",
                        "commit": "2" * 40,
                    }
                },
            },
            sort_keys=True,
        )
        + "\n"
    ).encode()
    result_path.write_bytes(payload)
    manifest = copy.deepcopy(fixture["manifest"])
    for row in manifest["artifacts"]:
        if row["path"] == "reports/execution_result.json":
            row["bytes"] = len(payload)
            row["sha256"] = hashlib.sha256(payload).hexdigest()
    manifest["summary"]["produced_bytes"] = sum(
        int(row["bytes"]) for row in manifest["artifacts"]
    )
    for class_name, summary in manifest["summary"]["by_retention_class"].items():
        class_rows = [
            row for row in manifest["artifacts"] if row["retention_class"] == class_name
        ]
        summary["objects"] = len(class_rows)
        summary["bytes"] = sum(int(row["bytes"]) for row in class_rows)
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    fixture["manifest"] = manifest
    fixture["manifest_path"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    with pytest.raises(EvidenceCertificateError, match="Git identity differs"):
        _draft(fixture)


def test_draft_requires_explicit_closed_state_and_no_claims(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    unclosed = copy.deepcopy(fixture["review_input"])
    unclosed["confirmations"]["no_live_state"] = False
    fixture["review_input"] = unclosed
    with pytest.raises(EvidenceCertificateError, match="every explicit closeout"):
        _draft(fixture)

    fixture = _fixture(tmp_path / "second")
    fixture["review_input"]["no_claims"] = ["no scientific claim"]
    with pytest.raises(EvidenceCertificateError, match="qualification"):
        _draft(fixture)


def test_active_source_marker_requires_exact_selected_set_closeout(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, live_reference_status="active")
    spec = _draft(fixture)
    bundle = build_debug_closeout_bundle_v1(
        evidence_root=fixture["root"],
        source_manifest_path=fixture["manifest_path"],
        review_spec=spec,
    )
    certificate = build_evidence_certificate_v1(
        bundle["request"], evidence_root=fixture["root"]
    )
    assert all(
        row["release_eligible"] is True
        for row in certificate["artifacts"]
        if row["path"].startswith("raw/")
    )
    assert spec["source_live_marker_closeout"]["scope"] == (
        "SOURCE_MANIFEST_LIVE_REFERENCE_MARKER_ONLY"
    )

    omitted = copy.deepcopy(spec)
    omitted.pop("source_live_marker_closeout")
    omitted.pop("spec_sha256")
    omitted["spec_sha256"] = canonical_sha256(omitted)
    with pytest.raises(EvidenceCertificateError, match="fields differ"):
        build_debug_closeout_bundle_v1(
            evidence_root=fixture["root"],
            source_manifest_path=fixture["manifest_path"],
            review_spec=omitted,
        )

    mismatched = copy.deepcopy(spec)
    mismatched["superseded_artifacts"] = mismatched["superseded_artifacts"][:1]
    mismatched.pop("spec_sha256")
    mismatched["spec_sha256"] = canonical_sha256(mismatched)
    with pytest.raises(EvidenceCertificateError, match="authority binding"):
        build_debug_closeout_bundle_v1(
            evidence_root=fixture["root"],
            source_manifest_path=fixture["manifest_path"],
            review_spec=mismatched,
        )


def test_review_spec_cannot_be_transplanted_to_another_source_root(
    tmp_path: Path,
) -> None:
    source_a = _fixture(tmp_path / "run-a", live_reference_status="active")
    source_b = _fixture(tmp_path / "run-b", live_reference_status="active")
    spec = _draft(source_a)

    with pytest.raises(EvidenceCertificateError, match="different source manifest"):
        build_debug_closeout_bundle_v1(
            evidence_root=source_b["root"],
            source_manifest_path=source_b["manifest_path"],
            review_spec=spec,
        )


def test_draft_rejects_stale_source_live_marker_audit(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, live_reference_status="active")
    fixture["review_input"]["source_live_marker_audit"]["audited_at_utc"] = (
        "2000-01-01T00:00:00Z"
    )
    with pytest.raises(EvidenceCertificateError, match="current-state audit is stale"):
        _draft(fixture)


def _run_cli(monkeypatch: pytest.MonkeyPatch, capsys, arguments: list[str]) -> dict:
    monkeypatch.setattr(
        sys, "argv", ["eiel_evidence_certificate_reclaim.py", *arguments]
    )
    assert reclaim.main() == 0
    return json.loads(capsys.readouterr().out)


def test_cli_draft_then_prepare_is_dry_run_first_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "closeout-output"
    draft_args = [
        "draft-debug-review",
        "--evidence-root",
        str(fixture["root"]),
        "--source-manifest",
        str(fixture["manifest_path"]),
        "--lifecycle-contract",
        str(fixture["contract_path"]),
        "--review-input",
        str(fixture["review_input_path"]),
        "--supporting-artifact",
        "execution-request=reports/execution_request.json",
        "--supporting-artifact",
        "execution-result=reports/execution_result.json",
        "--output-root",
        str(output),
    ]
    dry = _run_cli(monkeypatch, capsys, draft_args)
    assert dry["mode"] == "dry_run_review_draft"
    assert not output.exists()
    applied = _run_cli(
        monkeypatch,
        capsys,
        [
            *draft_args,
            "--apply",
            "--expected-plan-sha256",
            dry["plan"]["plan_sha256"],
        ],
    )
    assert applied["mode"] == "review_spec_materialized"
    spec_path = output / "diagnostic_closeout_review_spec.json"
    assert spec_path.is_file()
    replay = _run_cli(
        monkeypatch,
        capsys,
        [
            *draft_args,
            "--apply",
            "--expected-plan-sha256",
            dry["plan"]["plan_sha256"],
        ],
    )
    assert replay == applied

    prepare_args = [
        "prepare-debug-closeout",
        "--evidence-root",
        str(fixture["root"]),
        "--source-manifest",
        str(fixture["manifest_path"]),
        "--review-spec",
        str(spec_path),
        "--output-root",
        str(output),
    ]
    prepare_dry = _run_cli(monkeypatch, capsys, prepare_args)
    assert prepare_dry["mode"] == "dry_run_review_preparation"
    request_path = output / "diagnostic_closeout_request.json"
    assert not request_path.exists()
    prepared = _run_cli(
        monkeypatch,
        capsys,
        [
            *prepare_args,
            "--apply",
            "--expected-plan-sha256",
            prepare_dry["plan"]["plan_sha256"],
        ],
    )
    assert prepared["mode"] == "review_bundle_materialized"
    assert request_path.is_file()
    assert (output / "diagnostic_closeout_report.md").is_file()
    assert (fixture["root"] / "raw" / "state.bin").is_file()
    prepared_replay = _run_cli(
        monkeypatch,
        capsys,
        [
            *prepare_args,
            "--apply",
            "--expected-plan-sha256",
            prepare_dry["plan"]["plan_sha256"],
        ],
    )
    assert prepared_replay == prepared
    assert prepared["raw_release_performed"] is False
    assert (
        canonical_sha256(json.loads(request_path.read_text()))
        == prepared["request_sha256"]
    )
