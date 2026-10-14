from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

from tools.evidence.librarian.wp2r import (
    EXPECTED_V1R_TREE_DIGEST,
    SAFE_COUNTERPART,
    UNSAFE_MEMBER,
    UNSAFE_ZIP_REL,
    _classify_pair,
    _normalized_semantic,
    build_accessions,
    build_authority_map,
    build_review_queue,
    build_v1r_provenance_receipt,
    build_wp2r_catalog,
    reconcile_divergent_pairs,
    requirement_matrix,
    unsafe_zip_record,
    write_accession_object_shards,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = Path(
    "/tmp/emergence-test/runtime/evidence_raw/eiel_evidence_librarian_v1r/"
    "20260802T000000Z_baseline_audit_final"
)
REPEAT_AUDIT_ROOT = Path(
    "/tmp/emergence-test/runtime/evidence_raw/eiel_evidence_librarian_v1r/"
    "20260802T000000Z_baseline_audit_final_repeat"
)
SUPPLEMENT_ZIP = Path(
    "/tmp/emergence-test/runtime/evidence_raw/eiel_evidence_librarian_v1r/"
    "EIEL_Evidence_Librarian_V1R_Audit_Supplement_2026_08_02_final.zip"
)


def _init_repo(root: Path) -> None:
    os.system(f"git -C {root} init -q")
    os.system(f"git -C {root} config user.email test@example.invalid")
    os.system(f"git -C {root} config user.name Test")


def _commit_all(root: Path) -> None:
    os.system(f"git -C {root} add .")
    os.system(f"git -C {root} commit -q -m baseline")


def _fixture_audit(tmp_path: Path) -> Path:
    audit = tmp_path / "audit"
    generated = audit / "generated"
    generated.mkdir(parents=True)
    rows = []
    for index in range(3):
        rows.append(
            {
                "basename": f"Report_{index}.md",
                "docs_paths": [f"docs/Report_{index}.md"],
                "docs_sha256": ["d" * 64],
                "evidence_paths": [f"Report_{index}.md"],
                "evidence_sha256": ["e" * 64],
                "has_exact_content_match": False,
            }
        )
    (audit / "docs_evidences_same_basename_report.json").write_text(
        json.dumps({"rows": rows, "divergent_same_basename_count": 3}),
        encoding="utf-8",
    )
    (audit / "issue_ledger.json").write_text("[]\n", encoding="utf-8")
    (audit / "authority_drift_curation_queue.json").write_text(
        json.dumps({"rows": []}), encoding="utf-8"
    )
    (audit / "file_inventory.jsonl").write_text("", encoding="utf-8")
    (audit / "baseline_manifest.json").write_text(
        json.dumps({"tree_digest_sha256": "0" * 64}), encoding="utf-8"
    )
    (generated / "EIEL_Evidence_Catalog.json").write_text(
        json.dumps({"artifact_inventory": []}), encoding="utf-8"
    )
    (generated / "EIEL_Evidence_Catalog_Build.json").write_text(
        json.dumps({"build_status": "success"}), encoding="utf-8"
    )
    return audit


def test_wp2r_classification_keeps_raw_and_normalized_diff_separate() -> None:
    left = "# A\n\nDecision: keep\n"
    right = "# A\nDecision: keep\n"
    facts = {"decision": "Decision: keep", "line_count": 3, "source_commit_mentions": []}

    assert left.encode() != right.encode()
    assert _normalized_semantic(left) == _normalized_semantic(right)
    assert _classify_pair(facts, facts, {"raw_byte_equal": False, "normalized_equal": True}) == (
        "same_record_non_decision_text_drift"
    )


def test_wp2r_reconciles_complete_fixture_pairs(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "evidences").mkdir()
    for index in range(3):
        (repo / "docs" / f"Report_{index}.md").write_text(
            f"# Report {index}\n\nDecision: docs\n", encoding="utf-8"
        )
        (repo / "evidences" / f"Report_{index}.md").write_text(
            f"# Report {index}\n\nDecision: evidence\n", encoding="utf-8"
        )
    _init_repo(repo)
    _commit_all(repo)

    rows = reconcile_divergent_pairs(repo, _fixture_audit(tmp_path))

    assert len(rows) == 3
    assert {row["classification"] for row in rows} == {
        "same_record_decision_or_limitation_drift"
    }
    assert all(row["raw_byte_comparison"]["equal"] is False for row in rows)


def test_wp2r_real_audit_has_one_to_one_coverage_of_51_pairs() -> None:
    rows = reconcile_divergent_pairs(REPO_ROOT, AUDIT_ROOT)

    assert len(rows) == 51
    assert len({row["reconciliation_id"] for row in rows}) == 51
    assert {row["classification"] for row in rows}


def test_wp2r_authority_map_requires_explicit_sources() -> None:
    rows = build_authority_map(REPO_ROOT)

    assert {row["confidence"] for row in rows} <= {"explicit", "unknown"}
    assert all(row["authority_source"] or row["confidence"] == "unknown" for row in rows)
    assert len({row["domain"] for row in rows}) == len(rows)


def test_wp2r_unsafe_zip_is_streamed_and_not_extracted(monkeypatch) -> None:
    def forbidden_extract(*args, **kwargs):  # pragma: no cover - failure path
        raise AssertionError("direct extraction is forbidden")

    monkeypatch.setattr(zipfile.ZipFile, "extract", forbidden_extract)
    monkeypatch.setattr(zipfile.ZipFile, "extractall", forbidden_extract)

    result = unsafe_zip_record(REPO_ROOT)

    assert result["package_path"] == "evidences/" + UNSAFE_ZIP_REL
    assert result["known_unsafe_member"] == UNSAFE_MEMBER
    assert result["safe_counterpart_path"] == SAFE_COUNTERPART
    assert result["read_only_streamed_member_equals_safe_counterpart"] is True
    assert result["direct_extraction_performed"] is False


def test_wp2r_accession_integrity_and_no_dangling_ids() -> None:
    reconciliations = reconcile_divergent_pairs(REPO_ROOT, AUDIT_ROOT)
    authorities = build_authority_map(REPO_ROOT)
    unsafe = unsafe_zip_record(REPO_ROOT)
    accessions = build_accessions(
        repo_root=REPO_ROOT,
        reconciliations=reconciliations,
        authority_map=authorities,
        unsafe_record=unsafe,
    )
    artifact_ids = {row["artifact_id"] for row in accessions["artifacts"]}
    record_ids = {row["evidence_id"] for row in accessions["records"]}

    assert len(reconciliations) == 51
    assert 0 < len(accessions["records"]) <= len(reconciliations) * 2
    assert all(set(row["artifact_ids"]) <= artifact_ids for row in accessions["records"])
    assert all(set(row["supporting_record_ids"]) <= record_ids for row in accessions["claims"])
    assert all(not row["supersedes_record_ids"] for row in accessions["records"])
    assert all(not row["superseded_by_record_ids"] for row in accessions["records"])


def test_wp2r_writes_object_level_accession_shards(tmp_path: Path) -> None:
    reconciliations = reconcile_divergent_pairs(REPO_ROOT, AUDIT_ROOT)
    accessions = build_accessions(
        repo_root=REPO_ROOT,
        reconciliations=reconciliations,
        authority_map=build_authority_map(REPO_ROOT),
        unsafe_record=unsafe_zip_record(REPO_ROOT),
    )

    manifest = write_accession_object_shards(tmp_path, accessions)

    assert manifest["shard_count"] == sum(len(rows) for rows in accessions.values())
    assert manifest["collection_counts"]["records"] == len(accessions["records"])
    first = manifest["shards"][0]
    assert (tmp_path / first["path"]).is_file()


def test_wp2r_catalog_is_deterministic_and_separates_views() -> None:
    reconciliations = reconcile_divergent_pairs(REPO_ROOT, AUDIT_ROOT)
    authorities = build_authority_map(REPO_ROOT)
    unsafe = unsafe_zip_record(REPO_ROOT)
    accessions = build_accessions(
        repo_root=REPO_ROOT,
        reconciliations=reconciliations,
        authority_map=authorities,
        unsafe_record=unsafe,
    )
    review = build_review_queue(reconciliations, authorities)
    baseline = json.loads((AUDIT_ROOT / "generated/EIEL_Evidence_Catalog.json").read_text())
    catalog_a = build_wp2r_catalog(
        baseline_catalog=baseline,
        provenance={"relationship": "sibling_report_only_amend"},
        input_manifest={"input_set_digest": "a" * 64, "source_repository_commit": "b" * 40},
        reconciliations=reconciliations,
        accessions=accessions,
        authority_map=authorities,
        review_queue=review,
        unsafe_record=unsafe,
    )
    catalog_b = build_wp2r_catalog(
        baseline_catalog=baseline,
        provenance={"relationship": "sibling_report_only_amend"},
        input_manifest={"input_set_digest": "a" * 64, "source_repository_commit": "b" * 40},
        reconciliations=reconciliations,
        accessions=accessions,
        authority_map=authorities,
        review_queue=review,
        unsafe_record=unsafe,
    )

    assert catalog_a == catalog_b
    assert "physical_files" in catalog_a
    assert "atomic_evidence_records" in catalog_a
    assert "authority_assertions" in catalog_a
    assert catalog_a["authority_policy"]["inference_allowed"] is False


def test_wp2r_stage0r_receipt_and_tree_digest() -> None:
    receipt = build_v1r_provenance_receipt(
        repo_root=REPO_ROOT,
        audit_root=AUDIT_ROOT,
        repeat_audit_root=REPEAT_AUDIT_ROOT,
        supplement_zip=SUPPLEMENT_ZIP,
    )

    assert receipt["conclusion"] == "passed"
    assert receipt["relationship"] == "sibling_report_only_amend"
    assert receipt["baseline"]["audit_tree_digest"] == EXPECTED_V1R_TREE_DIGEST
    assert receipt["baseline"]["divergent_same_basename_count"] == 51


def test_wp2r_requirement_matrix_has_21_covered_items() -> None:
    matrix = requirement_matrix()

    assert len(matrix) == 21
    assert {row["coverage_state"] for row in matrix} == {"covered"}
