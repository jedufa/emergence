from __future__ import annotations

import json
import shutil
from pathlib import Path

from tests.eiel_actor_v11_helpers import REPO_ROOT, write_preflight_receipt
from tools.actors import authority


def _receipt(path: Path, **kwargs: object) -> tuple[Path, dict[str, object]]:
    receipt_path = write_preflight_receipt(path, **kwargs)
    return receipt_path, json.loads(receipt_path.read_text(encoding="utf-8"))


def _bound_context(tmp_path: Path, *, fail_closed: bool = False, acknowledge: bool = False) -> dict[str, object]:
    receipt_path, receipt = _receipt(
        tmp_path / "preflight.json",
        fail_closed=fail_closed,
        unknown_gaps=["multiple authority domains remain"] if fail_closed else [],
    )
    retained = authority.retain_preflight_receipt(receipt, state_root=tmp_path / "state")
    return authority.build_authority_context(
        authorization_basis="operator_instruction",
        repo_root=REPO_ROOT,
        preflight_receipt=receipt,
        preflight_receipt_path=retained,
        operator_instruction_ref="explicit operator authority for bounded actor verification",
        preflight_fail_closed_acknowledged=acknowledge,
    )


def test_authority_context_binds_and_reverifies_preflight_receipt(tmp_path: Path) -> None:
    context = _bound_context(tmp_path)
    result = authority.validate_authority_context(
        context,
        mission_type="CONFIRMATION",
        resource_demand={"device": "cpu"},
        repo_root=REPO_ROOT,
    )
    assert result["valid"] is True
    binding = result["preflight"]
    assert binding["receipt_sha256"]
    assert Path(binding["receipt_path"]).is_file()
    assert binding["catalog_build_id"].startswith("eiel-build-wp4-")
    assert len(binding["maintained_source_digest"]) == 64

    # A fresh process/path verification must fail closed after tampering.
    retained = Path(binding["receipt_path"])
    payload = json.loads(retained.read_text(encoding="utf-8"))
    payload["topic"] = "tampered topic"
    retained.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    tampered = authority.validate_authority_context(
        context,
        mission_type="CONFIRMATION",
        resource_demand={"device": "cpu"},
        repo_root=REPO_ROOT,
    )
    assert tampered["valid"] is False
    assert any("hash changed" in problem for problem in tampered["errors"])


def test_fail_closed_preflight_requires_higher_authority_acknowledgement(tmp_path: Path) -> None:
    blocked_context = _bound_context(tmp_path / "blocked", fail_closed=True, acknowledge=False)
    blocked = authority.validate_authority_context(
        blocked_context,
        mission_type="CONFIRMATION",
        resource_demand={"device": "cpu"},
        repo_root=REPO_ROOT,
    )
    assert blocked["valid"] is False
    assert any("fail-closed" in problem for problem in blocked["errors"])

    acknowledged_context = _bound_context(tmp_path / "ack", fail_closed=True, acknowledge=True)
    acknowledged = authority.validate_authority_context(
        acknowledged_context,
        mission_type="CONFIRMATION",
        resource_demand={"device": "cpu"},
        repo_root=REPO_ROOT,
    )
    assert acknowledged["valid"] is True
    assert acknowledged["warnings"]
    assert acknowledged["preflight"]["unknown_authority_gaps"] == [
        "multiple authority domains remain"
    ]


def test_work_package_and_active_goal_bindings_detect_drift(tmp_path: Path) -> None:
    work_package = tmp_path / "approved-work-package.md"
    active_goal = tmp_path / "ACTIVE_GOAL.md"
    work_package.write_text("# Approved work package\n", encoding="utf-8")
    active_goal.write_text("# Active goal\n\nWork package: approved-work-package.md\n", encoding="utf-8")
    receipt_path, receipt = _receipt(tmp_path / "preflight.json")
    retained = authority.retain_preflight_receipt(receipt, state_root=tmp_path / "state")
    context = authority.build_authority_context(
        authorization_basis="active_goal",
        repo_root=REPO_ROOT,
        preflight_receipt=receipt,
        preflight_receipt_path=retained,
        work_package_path=work_package,
        active_goal_path=active_goal,
    )
    valid = authority.validate_authority_context(
        context,
        mission_type="IMPLEMENTATION_VALIDATION",
        resource_demand={"device": "cpu"},
        repo_root=REPO_ROOT,
    )
    assert valid["valid"] is True

    work_package.write_text("# Mutated work package\n", encoding="utf-8")
    drifted = authority.validate_authority_context(
        context,
        mission_type="IMPLEMENTATION_VALIDATION",
        resource_demand={"device": "cpu"},
        repo_root=REPO_ROOT,
    )
    assert drifted["valid"] is False
    assert any("approved work package hash changed" in problem for problem in drifted["errors"])


def test_catalog_identity_drift_requires_new_preflight(tmp_path: Path) -> None:
    source = tmp_path / "catalog"
    shutil.copytree(REPO_ROOT / "tools/evidence/librarian/catalog_source/00-catalog", source)
    receipt = authority.run_librarian_preflight(
        repo_root=REPO_ROOT,
        source_root=source,
        topic="qualification authority register",
        collection="all",
        authority_domain="execution_route",
    )
    retained = authority.retain_preflight_receipt(receipt, state_root=tmp_path / "state")
    context = authority.build_authority_context(
        authorization_basis="operator_instruction",
        repo_root=REPO_ROOT,
        preflight_receipt=receipt,
        preflight_receipt_path=retained,
        operator_instruction_ref="test catalog drift",
    )
    before = authority.validate_authority_context(
        context,
        mission_type="TECHNICAL_QUALIFICATION",
        resource_demand={"device": "cpu"},
        repo_root=REPO_ROOT,
        source_root=source,
    )
    assert before["valid"] is True

    target = next(path for path in (source / "dossiers").rglob("*.json") if path.is_file())
    target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    after = authority.validate_authority_context(
        context,
        mission_type="TECHNICAL_QUALIFICATION",
        resource_demand={"device": "cpu"},
        repo_root=REPO_ROOT,
        source_root=source,
    )
    assert after["valid"] is False
    assert any("maintained source changed" in problem for problem in after["errors"])


def test_mechanical_exemption_is_narrow_and_non_scientific(tmp_path: Path) -> None:
    context = authority.build_authority_context(
        authorization_basis="mechanical_preflight_exemption",
        repo_root=REPO_ROOT,
        operator_instruction_ref="operator requested a bounded mechanical smoke",
        preflight_exemption_reason="single CPU CLI help and schema smoke only",
    )
    allowed = authority.validate_authority_context(
        context,
        mission_type="SMOKE",
        resource_demand={"device": "cpu"},
        repo_root=REPO_ROOT,
    )
    assert allowed["valid"] is True

    scientific = authority.validate_authority_context(
        context,
        mission_type="SCIENTIFIC_PROBE",
        resource_demand={"device": "cpu"},
        repo_root=REPO_ROOT,
    )
    gpu = authority.validate_authority_context(
        context,
        mission_type="SMOKE",
        resource_demand={"device": "cuda"},
        repo_root=REPO_ROOT,
    )
    assert scientific["valid"] is False
    assert gpu["valid"] is False


def test_evidence_preflight_markdown_contains_mandatory_header(tmp_path: Path) -> None:
    context = _bound_context(tmp_path)
    text = authority.evidence_preflight_markdown(context["preflight"])
    assert text.startswith("## Evidence Preflight\n")
    for label in (
        "Authority consulted:",
        "Evidence IDs:",
        "Document IDs:",
        "Claims consulted:",
        "Limitations reviewed:",
        "Unknown authority:",
        "Catalog build ID:",
        "Maintained source digest:",
    ):
        assert f"- {label}" in text
