from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.eiel_actor_v11_helpers import REPO_ROOT
from tools.actors.evidence_archive import inspect_published_evidence_archive


EVIDENCE_ROOT = REPO_ROOT / "evidences"
ISSUE_ROOT = REPO_ROOT / "tools/evidence/librarian/catalog_source/00-catalog/issues"
PUBLICATION_MAP = REPO_ROOT / "evidences/990_legacy-redirects/archive-publication-v1-map.json"


def _accepted_unsafe_archives() -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for path in sorted(ISSUE_ROOT.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("category") != "unsafe_package_integrity_restricted":
            continue
        if payload.get("disposition") != "accepted_exception":
            continue
        evidence = payload.get("evidence") if isinstance(payload.get("evidence"), dict) else {}
        package = str(evidence.get("package_path") or "")
        if package:
            rows[package] = payload
    return rows



def _published_archives() -> dict[str, dict[str, object]]:
    payload = json.loads(PUBLICATION_MAP.read_text(encoding="utf-8"))
    rows: dict[str, dict[str, object]] = {}
    for row in payload.get("files") or []:
        if not isinstance(row, dict):
            continue
        path = str(row.get("new_path") or "")
        if not path.endswith(".evidence.zip"):
            continue
        assert path not in rows, f"duplicate Archive 1.0 published path: {path}"
        rows[path] = row
    return rows

def test_complete_published_evidence_archive_corpus_is_integrity_classified() -> None:
    """Every shipped evidence ZIP is either verified or a catalogued exception.

    This is intentionally a corpus-level regression gate.  It never extracts a
    ZIP and never changes historical evidence.  New unsafe archives therefore
    fail loudly instead of becoming another bit of inherited folklore.
    """

    archives = sorted(EVIDENCE_ROOT.rglob("*.evidence.zip"))
    if not archives:
        pytest.skip("published evidence corpus is not included in this core transfer")
    accepted_unsafe = _accepted_unsafe_archives()
    published = _published_archives()
    seen_ids: set[str] = set()
    invalid_paths: set[str] = set()

    for archive in archives:
        result = inspect_published_evidence_archive(archive)
        evidence_id = str(result.get("evidence_id") or "")
        assert evidence_id and evidence_id not in seen_ids, archive
        seen_ids.add(evidence_id)
        assert result["registered_filename"] is True
        assert result["assessment_eligibility"] == "not_raw_campaign_input"
        assert result["authority_effect"] == "none_from_integrity_verification"
        assert result["scientific_claim_effect"] == "none"
        sidecar = result["sidecar"]
        if sidecar["expected_sha256"] is not None:
            assert sidecar["matches"] is True, archive

        relative = archive.relative_to(REPO_ROOT).as_posix()
        publication = published.get(relative)
        assert publication is not None, f"evidence ZIP lacks an Archive 1.0 publication record: {relative}"
        assert publication["permanent_file_id"] == evidence_id
        assert publication["sha256"] == result["archive_sha256"]
        assert publication["verified_destination_sha256"] == result["archive_sha256"]
        assert publication["tracked"] is True
        if result["status"] == "PUBLISHED_EVIDENCE_ARCHIVE_VERIFIED":
            continue
        assert result["status"] == "INVALID_PUBLISHED_EVIDENCE_ARCHIVE", result
        invalid_paths.add(relative)
        issue = accepted_unsafe.get(relative)
        assert issue is not None, f"unsafe archive has no accepted Librarian exception: {relative}"
        evidence = issue["evidence"]
        assert evidence["package_sha256"] == result["archive_sha256"]
        assert evidence["direct_extraction_performed"] is False
        assert evidence["historical_package_modified"] is False

    assert set(published) == {path.relative_to(REPO_ROOT).as_posix() for path in archives}, (
        "Archive 1.0 publication records and shipped evidence ZIPs must remain one-to-one"
    )
    assert invalid_paths == set(accepted_unsafe), (
        "catalogued unsafe evidence and observed unsafe evidence must remain one-to-one"
    )
