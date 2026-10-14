from __future__ import annotations

import zipfile
from pathlib import Path

from tools.evidence.librarian import wp3
from tools.evidence.librarian.core import deterministic_zip


REPO_ROOT = Path(__file__).resolve().parents[1]
WP2R_ROOT = Path(
    "/tmp/emergence-test/runtime/evidence_raw/eiel_evidence_librarian_wp2r/"
    "20260802T000000Z_wp2r_curated_final"
)
WP2R_PACKAGE = Path(
    "/tmp/emergence-test/runtime/evidence_raw/eiel_evidence_librarian_wp2r/"
    "EIEL_Evidence_Librarian_WP2R_Evidence_Package_2026_08_02.zip"
)


def test_wp3_compare_packages_records_reported_hash_discrepancy() -> None:
    receipt = wp3.compare_packages(
        WP2R_PACKAGE,
        [],
        reported_hashes=[wp3.REPORTED_WP2R_PACKAGE_HASH, wp3.VERIFIED_WP2R_PACKAGE_HASH],
    )

    assert receipt["decision"] == "proceed"
    assert receipt["canonical_sidecar_ok"] is True
    assert receipt["canonical_member_manifest_verification"]["ok"] is True
    assert receipt["reported_hash_resolution"]["classification"] in {
        "matches_report",
        "report_text_hash_stale_or_pre_refresh",
    }


def test_wp3_compare_packages_detects_member_content_divergence(tmp_path: Path) -> None:
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()
    (root_a / "x.txt").write_text("a", encoding="utf-8")
    (root_b / "x.txt").write_text("b", encoding="utf-8")
    zip_a = tmp_path / "a.zip"
    zip_b = tmp_path / "b.zip"
    deterministic_zip(zip_a, [root_a / "x.txt"], root=root_a)
    deterministic_zip(zip_b, [root_b / "x.txt"], root=root_b)

    receipt = wp3.compare_packages(zip_a, [zip_b])

    assert receipt["available_copy_comparisons"][0]["relationship"] == "content_divergent"
    assert receipt["decision"] == "stop_content_divergent"


def _wp2_review_inputs() -> tuple[list[dict], dict[str, dict]]:
    catalog = wp3._load(WP2R_ROOT / "generated/EIEL_Evidence_Catalog_WP2R.json")
    queue = wp3._load(WP2R_ROOT / "authority_review_queue.json")
    return queue, wp3._artifact_hash_lookup(catalog)


def test_wp3_adjudicates_24_queue_items_to_zero_remaining() -> None:
    queue, lookup = _wp2_review_inputs()
    ledger, remaining = wp3.adjudicate_review_queue(queue, artifact_lookup=lookup)

    assert len(queue) == 24
    assert len(ledger) == 24
    assert remaining == []
    assert {row["current_authority_inferred"] for row in ledger} == {False}


def test_wp3_policy_rows_scope_campaign_and_scientific_claims() -> None:
    catalog = wp3._load(WP2R_ROOT / "generated/EIEL_Evidence_Catalog_WP2R.json")
    rows = wp3.authority_policy_rows(
        catalog["authority_map"],
        observed_at_repo_commit="f" * 40,
    )
    by_domain = {row["domain"]: row for row in rows}

    assert by_domain["campaign_decision"]["asserted_status"] == "exact_evidence_record_and_decision_rule_scoped"
    assert by_domain["scientific_claim"]["asserted_status"] == "exact_claim_assertion_and_supporting_record_scoped"
    assert by_domain["campaign_decision"]["observed_at_repo_commit"] == "f" * 40
    assert by_domain["campaign_decision"]["authority_granting_commit"] is None


def test_wp3_promotes_accessions_and_builds_deterministic_views(tmp_path: Path) -> None:
    package = wp3.compare_packages(WP2R_PACKAGE, [])
    catalog = wp3._load(WP2R_ROOT / "generated/EIEL_Evidence_Catalog_WP2R.json")
    queue, lookup = _wp2_review_inputs()
    authority = wp3.authority_policy_rows(catalog["authority_map"], observed_at_repo_commit="f" * 40)
    ledger, remaining = wp3.adjudicate_review_queue(queue, artifact_lookup=lookup)
    source_root = tmp_path / "source"

    manifest = wp3.promote_accessions(
        repo_root=REPO_ROOT,
        wp2_root=WP2R_ROOT,
        source_root=source_root,
        package_receipt=package,
        authority_policies=authority,
        disposition_ledger=ledger,
        remaining_queue=remaining,
        observed_at_repo_commit="f" * 40,
    )
    first = wp3.build_views(
        source_manifest=manifest,
        source_root=source_root,
        output_root=tmp_path / "out1",
        package_receipt=package,
        disposition_ledger=ledger,
        remaining_queue=remaining,
    )
    second = wp3.build_views(
        source_manifest=manifest,
        source_root=source_root,
        output_root=tmp_path / "out2",
        package_receipt=package,
        disposition_ledger=ledger,
        remaining_queue=remaining,
    )

    assert (source_root / "SOURCE_MANIFEST.json").is_file()
    assert (source_root / "builds/eiel_catalog_build_wp3.json").is_file()
    assert first["output_hashes"] == second["output_hashes"]


def test_wp3_unsafe_zip_fixture_prevents_direct_extraction(monkeypatch) -> None:
    def forbidden_extract(*args, **kwargs):  # pragma: no cover
        raise AssertionError("direct extraction forbidden")

    monkeypatch.setattr(zipfile.ZipFile, "extract", forbidden_extract)
    monkeypatch.setattr(zipfile.ZipFile, "extractall", forbidden_extract)

    receipt = wp3.compare_packages(WP2R_PACKAGE, [])

    assert receipt["canonical_zip_safety"]["safety_state"] == "safe"


def test_wp3_requirement_matrix_has_18_items() -> None:
    matrix = wp3.requirement_matrix()

    assert len(matrix) == 18
    assert {row["coverage_state"] for row in matrix} == {"covered"}

