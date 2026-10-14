from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from tools.evidence.librarian import wp4
from tools.evidence.librarian.documents import (
    DOCUMENT_SCHEMA_VERSION,
    RUNTIME_LIFECYCLES,
    publication_plan,
    validate_document_objects,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "tools/evidence/librarian/catalog_source/00-catalog"
INITIAL_PUBLICATION_DOCUMENT_COUNT = 2368
POST_PUBLICATION_FIRST_SEQUENCE = INITIAL_PUBLICATION_DOCUMENT_COUNT + 1


def _current_document_count() -> int:
    return len(json.loads((SOURCE_ROOT / "accessions/documents.json").read_text()))


CURRENT_DOCUMENT_COUNT = _current_document_count()


@pytest.fixture(scope="module")
def catalog() -> dict[str, object]:
    return wp4.build_catalog(SOURCE_ROOT, REPO_ROOT)


def _documents(catalog: dict[str, object]) -> list[dict[str, object]]:
    return catalog["accessions"]["documents"]  # type: ignore[index,return-value]


def _relations(catalog: dict[str, object]) -> list[dict[str, object]]:
    return catalog["accessions"]["document_relations"]  # type: ignore[index,return-value]


def _issues(catalog: dict[str, object]) -> list[dict[str, object]]:
    return catalog["accessions"]["issues"]  # type: ignore[index,return-value]


def test_documents_are_maintained_exactly_once_with_stable_frozen_ids(catalog: dict[str, object]) -> None:
    documents = _documents(catalog)
    ids = [str(row["document_id"]) for row in documents]
    paths = [str(row["path"]) for row in documents]

    assert len(documents) == CURRENT_DOCUMENT_COUNT
    assert len(ids) == len(set(ids)) == CURRENT_DOCUMENT_COUNT
    assert len(paths) == len(set(paths)) == CURRENT_DOCUMENT_COUNT
    assert ids == [
        f"EIEL-DOC-{index:06d}" for index in range(1, CURRENT_DOCUMENT_COUNT + 1)
    ]
    assert all(re.fullmatch(r"EIEL-DOC-\d{6}", document_id) for document_id in ids)

    for row in documents:
        identity = row["identity"]
        sequence = int(str(row["document_id"]).rsplit("-", 1)[1])
        assert row["schema_version"] == DOCUMENT_SCHEMA_VERSION
        assert row["collection"] == "documents"
        assert identity["frozen"] is True
        assert identity["path_is_identity"] is False
        assert identity["reuse_policy"] == "never_reuse"
        assert identity["path_at_freeze"] != row["path"]
        assert len(identity["path_history"]) == 1
        path_history = identity["path_history"][0]
        expected_publication_id = (
            "EIEL-KNOWLEDGE-ARCHIVE-PUBLICATION-1.0"
            if sequence <= INITIAL_PUBLICATION_DOCUMENT_COUNT
            else str(identity["path_history"][0]["publication_id"])
        )
        assert path_history["publication_id"] == expected_publication_id
        assert path_history["from_path"] == identity["path_at_freeze"]
        assert path_history["to_path"] == row["path"]
        assert path_history["sha256"] == identity["sha256_at_freeze"]
        assert path_history["size_bytes"] == identity["size_bytes_at_freeze"]
        assert path_history["content_unchanged"] is True
        assert path_history["authority_effect"] == "none"
        content_revisions = identity.get("content_revisions") or []
        if content_revisions:
            assert content_revisions[0]["from_sha256"] == identity["sha256_at_freeze"]
            assert content_revisions[0]["from_size_bytes"] == identity["size_bytes_at_freeze"]
            assert content_revisions[-1]["to_sha256"] == row["sha256"]
            assert content_revisions[-1]["to_size_bytes"] == row["size_bytes"]
            assert all(revision["authority_effect"] == "none" for revision in content_revisions)
            assert all(revision["identity_effect"] == "none" for revision in content_revisions)
        else:
            assert identity["sha256_at_freeze"] == row["sha256"]
            assert identity["size_bytes_at_freeze"] == row["size_bytes"]
        if sequence <= 2366:
            assert identity["freeze_commit"] == wp4.DOCS_AUDIT_EXPECTED["observed_commit"]
            assert identity["freeze_basis"] == "accepted_read_only_docs_audit_v1"
        elif sequence <= INITIAL_PUBLICATION_DOCUMENT_COUNT:
            assert identity["freeze_commit"] == "1a8414967f5835cee875a15d6eccd80574df5cbd"
            assert identity["freeze_basis"] == "explicit_post_audit_registration_before_archive_publication_v1"
        else:
            assert identity["freeze_basis"] == ["explicit_post_publication_ingress"]
            assert re.fullmatch(r"[0-9a-f]{40}", identity["freeze_commit"])
        assert Path(str(row["proposed_publication"]["proposed_destination"])).name.startswith(
            f"{row['document_id']}--"
        )
        assert row["proposed_publication"]["authority_effect"] == "none"
        assert row["proposed_publication"]["physical_move_authorized"] is True
        assert row["proposed_publication"]["publication_state"] == "published"
        assert row["publication"]["publication_id"] == expected_publication_id
        assert row["publication"]["path_before_publication"] == identity["path_at_freeze"]
        assert row["publication"]["current_path"] == row["path"]
        assert row["publication"]["content_unchanged"] is True
        assert row["publication"]["authority_effect"] == "none"


def test_document_identity_freeze_and_publication_plan_are_path_independent_metadata(
    catalog: dict[str, object],
) -> None:
    documents = _documents(catalog)
    freeze = json.loads((SOURCE_ROOT / "manifests/DOCUMENT_IDENTITY_FREEZE.json").read_text())
    plan = json.loads((SOURCE_ROOT / "manifests/DOCUMENT_PUBLICATION_PLAN_CANDIDATE.json").read_text())
    computed_plan = publication_plan(documents)

    assert freeze["schema_version"] == "eiel_document_identity_freeze_manifest_v1"
    assert freeze["record_count"] == CURRENT_DOCUMENT_COUNT
    assert freeze["next_document_sequence"] == CURRENT_DOCUMENT_COUNT + 1
    assert freeze["path_is_identity"] is False
    assert freeze["id_reuse_policy"] == "never_reuse"
    assert plan["document_count"] == INITIAL_PUBLICATION_DOCUMENT_COUNT
    assert computed_plan["document_count"] == CURRENT_DOCUMENT_COUNT
    assert plan["all_ids_prefixed_in_filenames"] is True
    assert computed_plan["all_ids_prefixed_in_filenames"] is True
    assert plan["physical_move_authorized"] is False
    assert computed_plan["physical_move_authorized"] is False


def test_document_and_relation_schema_validation_passes(catalog: dict[str, object]) -> None:
    objects = catalog["accessions"]  # type: ignore[index]

    document_validation = validate_document_objects(objects, source_root=SOURCE_ROOT)
    integrity = wp4.referential_integrity(objects)

    assert document_validation["ok"] is True
    assert document_validation["failure_count"] == 0
    assert document_validation["document_count"] == CURRENT_DOCUMENT_COUNT
    assert document_validation["document_relation_count"] == len(_relations(catalog))
    assert integrity["ok"] is True
    assert integrity["failure_count"] == 0


def test_duplicate_and_same_basename_relations_are_review_only(catalog: dict[str, object]) -> None:
    relations = _relations(catalog)
    duplicate_relations = [row for row in relations if row["relation_type"] == "duplicate_content"]
    basename_relations = [row for row in relations if row["relation_type"] == "same_basename_unproven"]
    path_changed_relations = [row for row in relations if row["relation_type"] == "path_changed"]
    content_revision_relations = [row for row in relations if row["relation_type"] == "content_revision"]

    assert len(duplicate_relations) == 114
    assert len(basename_relations) == 426
    assert len(path_changed_relations) == CURRENT_DOCUMENT_COUNT
    assert len(content_revision_relations) == sum(
        len(row["identity"].get("content_revisions") or [])
        for row in _documents(catalog)
    )
    assert all(
        row["authority_effect"] == "none"
        for row in duplicate_relations + basename_relations + path_changed_relations + content_revision_relations
    )
    assert all(row["basis"] == ["identical_sha256"] for row in duplicate_relations)
    assert all(row["basis"] == ["basename_only_unproven"] for row in basename_relations)
    assert all(row["basis"] == ["operator_governance_decision", "path_history"] for row in path_changed_relations)
    assert all(row["basis"] == ["operator_governance_decision", "manual_review"] for row in content_revision_relations)
    assert all(row["review_state"] == "candidate" for row in duplicate_relations)
    assert all(row["review_state"] == "needs_review" for row in basename_relations)
    assert all(row["review_state"] == "accepted" for row in path_changed_relations)
    assert all(row["review_state"] == "accepted" for row in content_revision_relations)
    assert all("#eiel-docrev-" in row["target"]["target_id_or_path"] for row in content_revision_relations)


def test_authority_review_candidates_do_not_create_document_authority_assertions(
    catalog: dict[str, object],
) -> None:
    documents = _documents(catalog)
    issues = _issues(catalog)
    authority_review_documents = [
        row for row in documents if "authority_review_required" in (row.get("review_flags") or [])
    ]
    authority_review_issues = [
        row for row in issues if row.get("owner_collection") == "documents" and row.get("issue_kind") == "authority_drift_candidate"
    ]

    assert catalog["object_counts"]["authorities"] == 11  # type: ignore[index]
    assert len(authority_review_documents) == 210
    assert len(authority_review_issues) == 210
    assert all(row["currentness_status"] == "unreviewed" for row in authority_review_documents)
    assert {row["review_state"] for row in authority_review_documents} == {
        "needs_authority_review",
        "physical_cleanup_candidate",
    }
    assert sum(row["review_state"] == "needs_authority_review" for row in authority_review_documents) == 207
    assert sum(row["review_state"] == "physical_cleanup_candidate" for row in authority_review_documents) == 3
    assert all("asserted_status" not in row for row in authority_review_documents)
    assert all(row["authority_effect"] == "none" for row in authority_review_issues)


def test_runtime_candidates_and_machine_only_companions_are_published_as_runtime_relocations(
    catalog: dict[str, object],
) -> None:
    runtime = [
        row for row in _documents(catalog) if row["proposed_publication"]["destination_class"] == "runtime_relocation"
    ]
    audit_runtime = [row for row in runtime if row["refined_lifecycle_candidate"] in RUNTIME_LIFECYCLES]
    package_companions = [row for row in runtime if row not in audit_runtime]

    assert len(runtime) == 518
    assert len(audit_runtime) == 447
    assert len(package_companions) == 71
    assert {row["retention_class"] for row in audit_runtime} == {"cleanup_candidate"}
    assert {row["review_state"] for row in audit_runtime} == {"physical_cleanup_candidate"}
    assert all("machine_only_same_stem_package" in row["proposed_publication"]["basis"] for row in package_companions)
    assert all(str(row["path"]).startswith("/tmp/emergence-test/storage/") for row in runtime)
    assert all(Path(str(row["path"])).is_file() for row in runtime)
    assert all(row["git_state"] == "runtime_relocated" for row in runtime)
    assert all(row["publication"]["action"] == "runtime_relocate" for row in runtime)
    assert all(str(row["proposed_publication"]["proposed_destination"]).startswith("/tmp/emergence-test/storage/") for row in runtime)
    assert all(row["proposed_publication"]["physical_move_authorized"] is True for row in runtime)


def test_same_stem_document_packages_share_destination_class_and_parent(catalog: dict[str, object]) -> None:
    reading_copy_pairs = {
        (str(row["source_document_id"]), str(row["target"]["target_id_or_path"]))
        for row in _relations(catalog)
        if row["relation_type"] == "reading_copy_of"
    }
    packages: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in _documents(catalog):
        path = Path(str(row["identity"]["path_at_freeze"]))
        packages.setdefault((path.parent.as_posix().casefold(), path.stem.casefold()), []).append(row)

    paired = [
        members
        for members in packages.values()
        if len({Path(str(row["identity"]["path_at_freeze"])).suffix for row in members}) > 1
    ]
    assert paired
    for members in paired:
        proposals = [row["proposed_publication"] for row in members]
        destination_classes = {
            str(proposal["destination_class"]) for proposal in proposals
        }
        destination_parents = {
            Path(str(proposal["proposed_destination"])).parent
            for proposal in proposals
        }
        if len(destination_classes) > 1 or len(destination_parents) > 1:
            member_ids = {str(row["document_id"]) for row in members}
            assert any(
                source_id in member_ids and target_id in member_ids
                for source_id, target_id in reading_copy_pairs
            )
            continue
        assert len(destination_classes) == 1
        assert len(destination_parents) == 1


def test_structured_specs_are_not_blanket_runtime_cleanup_candidates(catalog: dict[str, object]) -> None:
    structured_specs = [
        row
        for row in _documents(catalog)
        if row["refined_lifecycle_candidate"] == "structured_spec_or_contract_support"
    ]

    assert len(structured_specs) == 49
    assert {row["retention_class"] for row in structured_specs} == {"machine_readable_support"}
    assert all(row["review_state"] != "physical_cleanup_candidate" for row in structured_specs)
    assert all(row["proposed_publication"]["destination_class"] != "runtime_relocation" for row in structured_specs)


def test_existing_catalog_links_are_direct_navigation_not_identity_merge(catalog: dict[str, object]) -> None:
    linked = [
        row
        for row in _documents(catalog)
        if any(
            row.get(field)
            for field in (
                "linked_evidence_ids",
                "linked_claim_ids",
                "linked_authority_assertion_ids",
                "linked_raw_asset_ids",
                "linked_artifact_ids",
                "linked_issue_ids",
            )
        )
    ]
    linked_to_evidence = [row for row in _relations(catalog) if row["relation_type"] == "linked_to_evidence_record"]

    assert len(linked) == 57
    assert len(linked_to_evidence) == 34
    assert all(row["authority_effect"] == "none" for row in linked_to_evidence)
    assert all(row["target"]["target_type"] == "evidence_record" for row in linked_to_evidence)


def test_unified_issue_ledger_has_owner_collection_for_evidence_and_documents(
    catalog: dict[str, object],
) -> None:
    issues = _issues(catalog)
    owner_counts = {}
    for row in issues:
        owner = row.get("owner_collection")
        owner_counts[owner] = owner_counts.get(owner, 0) + 1

    assert owner_counts == {"documents": 456, "evidence": 23}
    assert all(row.get("owner_collection") in {"documents", "evidence", "shared"} for row in issues)
    assert all(row["document_ids"] for row in issues if row.get("owner_collection") == "documents")


def test_collection_filters_apply_to_search_show_and_preflight(catalog: dict[str, object]) -> None:
    first_document_id = _documents(catalog)[0]["document_id"]

    document_search = wp4.search_catalog(catalog, "production runtime", collection="documents", limit=50)
    evidence_search = wp4.search_catalog(catalog, "production runtime", collection="evidence", limit=50)
    document_show = wp4.show_object(catalog, str(first_document_id), collection="documents")
    document_preflight = wp4.evidence_preflight(catalog, "production runtime", collection="documents", limit=10)
    evidence_preflight = wp4.evidence_preflight(catalog, "production runtime", collection="evidence", limit=10)

    assert document_search["match_count"] > 0
    assert all(row["object_class"] in wp4.DOCUMENT_OBJECT_CLASSES for row in document_search["matches"])
    assert all(row["object_class"] in wp4.EVIDENCE_OBJECT_CLASSES for row in evidence_search["matches"])
    assert document_show["object_type"] == "documents"
    assert document_show["collection"] == "documents"
    with pytest.raises(wp4.CliExit):
        wp4.show_object(catalog, str(first_document_id), collection="evidence")
    assert document_preflight["collection"] == "documents"
    assert document_preflight["document_ids"]
    assert document_preflight["evidence_ids"] == []
    assert document_preflight["artifact_ids"] == []
    assert document_preflight["raw_asset_ids"] == []
    assert evidence_preflight["collection"] == "evidence"
    assert evidence_preflight["document_ids"] == []
    assert evidence_preflight["document_relation_ids"] == []


def test_generated_knowledge_views_are_deterministic_from_temp_catalog_copy(tmp_path: Path) -> None:
    source_copy = tmp_path / "catalog_source"
    shutil.copytree(SOURCE_ROOT, source_copy)
    first = wp4.rebuild_views(REPO_ROOT, source_copy, tmp_path / "views_a")
    second = wp4.rebuild_views(REPO_ROOT, source_copy, tmp_path / "views_b")
    comparison = wp4.compare_output_trees(tmp_path / "views_a", tmp_path / "views_b")

    assert first["referential_integrity"]["ok"] is True
    assert second["referential_integrity"]["ok"] is True
    assert first["output_hashes"][wp4.KNOWLEDGE_CATALOG_JSON] == second["output_hashes"][wp4.KNOWLEDGE_CATALOG_JSON]
    assert first["output_hashes"][wp4.KNOWLEDGE_CATALOG_HTML] == second["output_hashes"][wp4.KNOWLEDGE_CATALOG_HTML]
    assert first["output_hashes"][wp4.KNOWLEDGE_SEARCH_INDEX_JSON] == second["output_hashes"][wp4.KNOWLEDGE_SEARCH_INDEX_JSON]
    assert comparison["byte_identical"] is True
    assert comparison["only_first"] == []
    assert comparison["only_second"] == []
    assert comparison["changed"] == []


def test_metadata_queries_do_not_mutate_docs_or_evidences() -> None:
    before = {}
    for tree in ("docs", "evidences"):
        before[tree] = subprocess.run(
            ["git", "status", "--short", "--", tree],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout

    catalog = wp4.build_catalog(SOURCE_ROOT, REPO_ROOT)
    wp4.search_catalog(catalog, "production runtime", collection="all", limit=5)
    wp4.evidence_preflight(catalog, "production runtime", collection="all", limit=5)

    for tree in ("docs", "evidences"):
        after = subprocess.run(
            ["git", "status", "--short", "--", tree],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout
        assert after == before[tree]
