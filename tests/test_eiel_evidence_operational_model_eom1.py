from __future__ import annotations

import json
import shutil
from pathlib import Path

from tools.evidence.librarian import wp4
from tools.evidence.librarian.cli import main


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "tools/evidence/librarian/catalog_source/00-catalog"


def _copy_source(tmp_path: Path) -> Path:
    target = tmp_path / "catalog_source"
    shutil.copytree(SOURCE_ROOT, target)
    return target


def test_eom1_preflight_receipt_contains_required_report_fields(tmp_path: Path) -> None:
    src = _copy_source(tmp_path)
    catalog = wp4.build_catalog(src, REPO_ROOT)

    receipt = wp4.evidence_preflight(catalog, "campaign_decision")

    assert receipt["schema_version"] == "eiel_evidence_preflight_receipt_v1"
    assert receipt["authority_consulted"]
    assert receipt["catalog_build_id"] == catalog["build_id"]
    assert receipt["source_digest"] == catalog["maintained_source_digest"]
    assert "evidence_ids" in receipt
    assert "claims_consulted" in receipt
    assert "limitations_reviewed" in receipt
    assert "unresolved_authority" in receipt
    assert "applicable_dossiers" in receipt


def test_eom1_preflight_fails_closed_for_unknown_topic(tmp_path: Path) -> None:
    src = _copy_source(tmp_path)
    catalog = wp4.build_catalog(src, REPO_ROOT)

    receipt = wp4.evidence_preflight(catalog, "definitely-not-an-eiel-topic")

    assert receipt["fail_closed"] is True
    assert receipt["fallback_repository_search_allowed"] is True
    assert receipt["unresolved_authority"] is True


def test_eom1_preflight_fails_closed_when_evidence_exists_without_authority(tmp_path: Path) -> None:
    src = _copy_source(tmp_path)
    catalog = wp4.build_catalog(src, REPO_ROOT)

    receipt = wp4.evidence_preflight(catalog, "memory")

    assert receipt["artifact_ids"] or receipt["limitations_reviewed"] or receipt["applicable_dossiers"]
    assert receipt["authority_consulted"] == []
    assert receipt["unresolved_authority"] is True
    assert receipt["fail_closed"] is True
    assert receipt["fallback_repository_search_allowed"] is True


def test_eom1_preflight_cli_writes_receipt(tmp_path: Path) -> None:
    src = _copy_source(tmp_path)
    receipt = tmp_path / "preflight.json"

    exit_code = main(
        [
            "preflight",
            "--repo-root",
            str(REPO_ROOT),
            "--source-root",
            str(src),
            "--topic",
            "campaign_decision",
            "--receipt",
            str(receipt),
            "--json",
        ]
    )

    payload = json.loads(receipt.read_text())
    assert exit_code == 0
    assert payload["authority_consulted"]
    assert payload["catalog_build_id"]


def test_eom1_registration_helper_adds_metadata_object_and_regenerates_aggregates(tmp_path: Path) -> None:
    src = _copy_source(tmp_path)
    template = json.loads(sorted((src / "issues").glob("*.json"))[0].read_text())
    template["issue_id"] = "eiel-issue-eom1-test-registration"
    template["category"] = "eom1_test_registration"
    template["paths"] = ["docs/400_implementation-engineering/410_implementation-reports/000_programme-cross-cutting/EIEL-DOC-000328--eiel-evidence-operational-model-eom1-closeout-report-2026-08-02.md"]

    receipt = wp4.register_accession_object(src, "issues", template)
    aggregate = json.loads((src / "accessions/issues.json").read_text())

    assert receipt["object_id"] == "eiel-issue-eom1-test-registration"
    assert receipt["created"] is True
    assert receipt["referential_integrity"]["ok"] is True
    assert any(row["issue_id"] == "eiel-issue-eom1-test-registration" for row in aggregate)


def test_eom1_registration_helper_is_fail_closed_before_writing_invalid_object(tmp_path: Path) -> None:
    src = _copy_source(tmp_path)
    bad = json.loads(sorted((src / "records").glob("*.json"))[0].read_text())
    bad["evidence_id"] = "eiel-rec-eom1-invalid-registration"
    bad["claim_ids"] = ["missing-claim-for-eom1-test"]
    target = src / "records" / "eiel-rec-eom1-invalid-registration.json"

    try:
        wp4.register_accession_object(src, "records", bad)
    except Exception as exc:
        assert "referential integrity" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("invalid registration did not fail closed")

    assert not target.exists()


def test_eom1_view_root_compatibility_is_preserved(tmp_path: Path) -> None:
    src = _copy_source(tmp_path)
    views = tmp_path / "views"
    wp4.rebuild_views(REPO_ROOT, src, views)

    assert main(
        [
            "verify-catalog",
            "--repo-root",
            str(REPO_ROOT),
            "--source-root",
            str(src),
            "--view-root",
            str(views),
            "--json",
        ]
    ) == 0


def test_eom1_menu_contract_includes_preflight_and_validation(tmp_path: Path) -> None:
    output = tmp_path / "menu.json"
    wp4.write_menu_contract(output)
    contract = json.loads(output.read_text())
    actions = {row["action_id"]: row for row in contract["actions"]}

    assert "evidence_preflight" in actions
    assert "validate_evidence" in actions
    assert actions["evidence_preflight"]["duplicates_authority_logic"] is False


def test_eom1_agents_declares_mandatory_preflight_receipt() -> None:
    text = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")

    assert "Knowledge Librarian is the mandatory front door" in text
    assert "Searching the repository for authority before an" in text
    assert "## Evidence Preflight" in text
    assert "Catalog build ID" in text
