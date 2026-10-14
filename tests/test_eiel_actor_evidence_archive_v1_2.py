from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from tests.eiel_actor_v11_helpers import REPO_ROOT, run_eiel
from tools.actors.evidence_archive import inspect_published_evidence_archive


REAL_ARCHIVE = (
    REPO_ROOT
    / "evidences/400_worlds-ecology/"
    "EIEL-EV-001463--19-predrisk-fixed1-validation.evidence.zip"
)


def test_real_published_archive_identity_sidecar_and_integrity() -> None:
    if not REAL_ARCHIVE.is_file():
        pytest.skip("published evidence archive is not included in this core transfer")
    result = inspect_published_evidence_archive(REAL_ARCHIVE)
    assert result["status"] == "PUBLISHED_EVIDENCE_ARCHIVE_VERIFIED"
    assert result["evidence_id"] == "EIEL-EV-001463"
    assert result["sidecar"]["evidence_id"] == "EIEL-EV-001464"
    assert result["sidecar"]["matches"] is True
    assert result["zip_safety"]["safety_state"] == "safe"
    assert result["assessment_eligibility"] == "not_raw_campaign_input"
    assert result["authority_effect"] == "none_from_integrity_verification"
    assert result["scientific_claim_effect"] == "none"


def test_evidence_manager_inspects_published_archive_without_reassessment(tmp_path: Path) -> None:
    if not REAL_ARCHIVE.is_file():
        pytest.skip("published evidence archive is not included in this core transfer")
    state = tmp_path / "state"
    first = run_eiel(
        ["evidenceManager", "inspect", str(REAL_ARCHIVE)],
        state_root=state,
    )
    second = run_eiel(
        ["evidenceManager", "inspect", str(REAL_ARCHIVE)],
        state_root=state,
    )
    assert first == second
    row = first["archive_inspection"]
    assert first["status"] == "PUBLISHED_EVIDENCE_ARCHIVE_VERIFIED"
    assert row["historical_archive_only"] is True
    assert row["automatic_extraction"] is False
    assert row["procedural_decision"] is None
    assert row["assessment_eligibility"] == "not_raw_campaign_input"
    assert first["next_actions"][0]["command"].startswith(
        "python -m tools.evidence.librarian preflight"
    )
    assert not list(state.rglob("metrics.csv"))
    assert not list(state.rglob("campaign_evidence_manifest.json"))


def test_unregistered_safe_zip_has_no_authority_and_no_assessment_eligibility(tmp_path: Path) -> None:
    archive = tmp_path / "unregistered.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("campaign_evidence_manifest.json", "{}")
        handle.writestr("metrics.csv", "generation,value\n0,1\n")
    result = run_eiel(
        ["evidenceManager", "inspect", str(archive)],
        state_root=tmp_path / "state",
    )
    row = result["archive_inspection"]
    assert result["status"] == "UNREGISTERED_ARCHIVE_INSPECTED"
    assert row["registered_filename"] is False
    assert row["authority_effect"] == "none"
    assert row["assessment_eligibility"] == "not_raw_campaign_input"
    assert row["automatic_extraction"] is False


def test_unsafe_zip_path_traversal_fails_closed(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape.txt", "no")
    result = run_eiel(
        ["evidenceManager", "inspect", str(archive)],
        state_root=tmp_path / "state",
        expected=1,
    )
    assert result["ok"] is False
    assert result["status"] == "INVALID_ARCHIVE"
    assert result["archive_inspection"]["automatic_extraction"] is False
    assert result["problems"]
