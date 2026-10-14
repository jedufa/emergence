from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from tools.evidence.librarian import wp4
from tools.evidence.librarian.cli import main


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "tools/evidence/librarian/catalog_source/00-catalog"


def _copy_source(tmp_path: Path) -> Path:
    target = tmp_path / "catalog_source"
    shutil.copytree(SOURCE_ROOT, target)
    return target


def test_wp4_splits_aggregate_ledgers_into_semantically_equivalent_object_shards(tmp_path: Path) -> None:
    src = _copy_source(tmp_path)

    receipt = wp4.shard_source_from_aggregates(src)

    expected_document_relation_count = len(json.loads((src / "accessions/document_relations.json").read_text()))
    assert receipt["semantic_equivalence"]["semantically_equal"] is True
    assert receipt["historical_payloads_touched"] is False
    assert receipt["shard_counts"] == {
        "artifacts": 103,
        "authorities": 11,
        "claims": 38,
        "document_relations": expected_document_relation_count,
        "documents": len(json.loads((src / "accessions/documents.json").read_text())),
        "issues": 479,
        "raw_assets": 203,
        "records": 39,
    }
    assert expected_document_relation_count == len(list((src / "document_relations").glob("*.json")))
    assert len(list((src / "records").glob("*.json"))) == 39
    assert (src / "accessions/records.json").is_file()


def test_wp4_maintained_source_digest_is_stable_after_split(tmp_path: Path) -> None:
    src = _copy_source(tmp_path)
    wp4.shard_source_from_aggregates(src)
    wp4.ensure_dossier_definition_files(src)

    first = wp4.maintained_source_digest(src)
    second = wp4.maintained_source_digest(src)

    assert first == second
    assert len(first) == 64


def test_wp4_referential_integrity_passes_on_curated_source(tmp_path: Path) -> None:
    src = _copy_source(tmp_path)
    wp4.shard_source_from_aggregates(src)
    catalog = wp4.build_catalog(src, REPO_ROOT)

    report = wp4.referential_integrity(catalog["accessions"])

    assert report["ok"] is True
    assert report["failure_count"] == 0


def test_wp4_builds_exactly_eight_synthesis_dossiers(tmp_path: Path) -> None:
    src = _copy_source(tmp_path)
    wp4.shard_source_from_aggregates(src)
    catalog = wp4.build_catalog(src, REPO_ROOT)

    manifests = wp4.build_dossier_manifests(catalog)

    assert [row["title"] for row in manifests] == [row["title"] for row in wp4.DOSSIER_DEFINITIONS]
    assert len(manifests) == 8
    assert all(row["covered_record_ids"] or row["covered_claim_ids"] or row["covered_issue_ids"] for row in manifests)


def test_wp4_query_and_show_resolve_exact_object_ids(tmp_path: Path) -> None:
    src = _copy_source(tmp_path)
    wp4.shard_source_from_aggregates(src)
    catalog = wp4.build_catalog(src, REPO_ROOT)
    first_record = json.loads(sorted((src / "records").glob("*.json"))[0].read_text())

    result = wp4.show_object(catalog, first_record["evidence_id"])

    assert result["object_type"] == "records"
    assert result["object"]["evidence_id"] == first_record["evidence_id"]


def test_wp4_authority_query_is_domain_scoped_and_fail_closed_for_missing_domain(tmp_path: Path) -> None:
    src = _copy_source(tmp_path)
    wp4.shard_source_from_aggregates(src)
    catalog = wp4.build_catalog(src, REPO_ROOT)

    result = wp4.authority_query(catalog, "campaign_decision")
    assert result["domain"] == "campaign_decision"
    assert "fail_closed" in result["policy"]

    try:
        wp4.authority_query(catalog, "not-a-domain")
    except wp4.CliExit as exc:
        assert exc.exit_code == wp4.EXIT_NO_MATCH
    else:  # pragma: no cover
        raise AssertionError("missing authority domain did not fail closed")


def test_wp4_generated_catalog_stale_detection(tmp_path: Path) -> None:
    src = _copy_source(tmp_path)
    wp4.shard_source_from_aggregates(src)
    out = tmp_path / "views"
    wp4.rebuild_views(REPO_ROOT, src, out)

    ok = wp4.verify_catalog(src, out / wp4.KNOWLEDGE_CATALOG_JSON, REPO_ROOT)
    assert ok["ok"] is True
    assert ok["stale"] is False

    record_path = sorted((src / "records").glob("*.json"))[0]
    record = json.loads(record_path.read_text())
    record["tags"] = sorted(set(record.get("tags", [])) | {"stale-test-marker"})
    record_path.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")

    stale = wp4.verify_catalog(src, out / wp4.KNOWLEDGE_CATALOG_JSON, REPO_ROOT)
    assert stale["ok"] is False
    assert stale["stale"] is True


def test_wp4_decision_ready_preserves_original_preflight_receipt(tmp_path: Path) -> None:
    src = _copy_source(tmp_path)
    wp4.shard_source_from_aggregates(src)
    catalog = wp4.build_catalog(src, REPO_ROOT)
    topic = "CPU campaign worker core utilization and parallelism"
    receipt = wp4.evidence_preflight(catalog, topic)
    before = json.dumps(receipt, sort_keys=True)

    summary = wp4.decision_ready_preflight(catalog, topic)

    assert json.dumps(receipt, sort_keys=True) == before
    assert summary["preflight_receipt"] == receipt
    assert summary["preflight_receipt_sha256"] == wp4.hashlib.sha256(
        wp4.canonical_json_bytes(receipt)
    ).hexdigest()


def test_wp4_decision_ready_ambiguous_preflight_remains_fail_closed(tmp_path: Path) -> None:
    src = _copy_source(tmp_path)
    wp4.shard_source_from_aggregates(src)
    catalog = wp4.build_catalog(src, REPO_ROOT)

    summary = wp4.decision_ready_preflight(
        catalog, "CPU campaign worker core utilization and parallelism"
    )

    assert summary["preflight_receipt"]["fail_closed"] is True
    assert summary["applicability"]["state"] == "not_established"
    assert summary["sufficiency"]["state"] == "not_established"
    assert {row["domain"] for row in summary["authority_families"]} == {
        "campaign_decision",
        "scientific_claim",
    }
    assert "Do not implement" in summary["next_prohibited_action"]


def test_wp4_decision_ready_no_authority_reports_bounded_fallback(tmp_path: Path) -> None:
    src = _copy_source(tmp_path)
    wp4.shard_source_from_aggregates(src)
    catalog = wp4.build_catalog(src, REPO_ROOT)

    summary = wp4.decision_ready_preflight(catalog, "zqxjv librarium frobnicate")

    assert summary["preflight_receipt"]["authority_resolution_state"] == "none"
    assert summary["applicability_states"] == ["established", "not_established", "excluded"]
    assert summary["sufficiency_states"] == ["sufficient", "insufficient", "not_established"]
    assert summary["applicability"]["state"] == "not_established"
    assert summary["sufficiency"]["state"] == "insufficient"
    assert "bounded repository search" in summary["fallback_repository_scope"]
    assert "repository prose cannot become authority" in summary["fallback_repository_scope"]


def test_wp4_decision_ready_summary_is_deterministic(tmp_path: Path) -> None:
    src = _copy_source(tmp_path)
    wp4.shard_source_from_aggregates(src)
    catalog = wp4.build_catalog(src, REPO_ROOT)
    topic = "Knowledge Librarian decision-ready preflight UX mode"

    first = wp4.format_decision_ready_summary(
        wp4.decision_ready_preflight(catalog, topic)
    )
    second = wp4.format_decision_ready_summary(
        wp4.decision_ready_preflight(catalog, topic)
    )

    assert first == second
    assert "Applicability: established" in first
    assert "Sufficiency: sufficient" in first


def test_wp4_search_adds_buckets_without_changing_legacy_fields(tmp_path: Path) -> None:
    src = _copy_source(tmp_path)
    wp4.shard_source_from_aggregates(src)
    catalog = wp4.build_catalog(src, REPO_ROOT)

    result = wp4.search_catalog(catalog, "route", limit=5)

    assert result["schema_version"] == "eiel_knowledge_search_result_v1"
    assert result["search_ux_schema_version"] == wp4.SEARCH_UX_SCHEMA_VERSION
    assert len(result["matches"]) <= 5
    assert result["match_count"] >= len(result["matches"])
    assert "exact_matches" in result
    assert "nearby_matches" in result
    assert "rejected_matches" in result
    assert result["returned_rejected_count"] <= 5
    assert result["rejected_candidate_count"] >= result["returned_rejected_count"]


def test_wp4_catalog_html_overview_uses_browser_recent_files_panel(tmp_path: Path) -> None:
    src = _copy_source(tmp_path)
    wp4.shard_source_from_aggregates(src)
    catalog = wp4.build_catalog(src, REPO_ROOT)
    output = tmp_path / "catalog.html"

    wp4.write_catalog_html(output, catalog, wp4.build_dossier_manifests(catalog))

    text = output.read_text(encoding="utf-8")
    assert "<h2>Last opened in this catalog</h2>" in text
    assert '<div id="recent-files" class="box"></div>' in text
    assert "eielKnowledgeCatalog.recentFiles.v1" in text
    assert "<h2>Collections</h2>" not in text
    assert "Decision-Ready Guidance" in text
    assert "python -m tools.evidence.librarian decision" in text
    assert "established, not_established, or excluded" in text
    assert "Topic:" not in text
    assert "<details><summary>Synthesis dossiers</summary>" in text


def test_wp4_decision_cli_and_preflight_decision_ready_reuse_same_summary(
    tmp_path: Path, capsys
) -> None:
    src = _copy_source(tmp_path)
    wp4.shard_source_from_aggregates(src)
    topic = "Knowledge Librarian decision-ready preflight UX mode"

    assert main(["decision", "--repo-root", str(REPO_ROOT), "--source-root", str(src), topic]) == 0
    decision_output = capsys.readouterr().out
    assert main(
        [
            "preflight",
            "--repo-root",
            str(REPO_ROOT),
            "--source-root",
            str(src),
            "--topic",
            topic,
            "--decision-ready",
        ]
    ) == 0
    preflight_output = capsys.readouterr().out

    assert decision_output.strip() in preflight_output
    assert "Detailed preflight audit:" in preflight_output


def test_wp4_decision_cli_json_returns_decision_projection(tmp_path: Path, capsys) -> None:
    src = _copy_source(tmp_path)
    wp4.shard_source_from_aggregates(src)

    assert main(
        [
            "decision",
            "--repo-root",
            str(REPO_ROOT),
            "--source-root",
            str(src),
            "--json",
            "Knowledge Librarian decision-ready preflight UX mode",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["schema_version"] == wp4.DECISION_READY_SCHEMA_VERSION
    assert payload["applicability"]["state"] == "established"
    assert payload["sufficiency"]["state"] == "sufficient"


def test_wp4_menu_contract_maps_to_existing_cli_commands(tmp_path: Path) -> None:
    output = tmp_path / "EIEL_Menu_Integration_Contract_WP4.json"
    wp4.write_menu_contract(output)
    contract = json.loads(output.read_text())
    commands = {row["command"] for row in wp4.cli_command_inventory()["commands"]}

    for action in contract["actions"]:
        argv = action["cli_argv_template"]
        assert argv[3] in commands
        assert action["duplicates_authority_logic"] is False


def test_wp4_cli_exit_codes_are_stable(tmp_path: Path, capsys) -> None:
    src = _copy_source(tmp_path)
    wp4.shard_source_from_aggregates(src)

    assert main(["search", "--repo-root", str(REPO_ROOT), "--source-root", str(src), "definitely-no-such-wp4-object"]) == wp4.EXIT_NO_MATCH
    assert main(["authority", "--repo-root", str(REPO_ROOT), "--source-root", str(src), "definitely-no-such-domain"]) == wp4.EXIT_NO_MATCH
    captured = capsys.readouterr()
    assert "no authority assertion" in captured.err


def test_wp4_no_historical_evidence_tree_changes_from_tests() -> None:
    status = subprocess.run(
        ["git", "status", "--short", "--", "evidences"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()

    assert status == ""
