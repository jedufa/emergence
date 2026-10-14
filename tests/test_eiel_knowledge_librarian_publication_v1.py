from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tools.evidence.librarian.core import LibrarianError, file_sha256
from tools.evidence.librarian.publication import (
    ALLOWED_DOC_SHELVES,
    CONTENT_REVISION_AUTHORIZATION,
    DOCUMENT_REFERENCE_REWRITE_AUTHORIZATION,
    REFERENCE_REWRITE_AUTHORIZATION,
    finalize_catalog_publication,
    prepare_publication_candidate,
    record_document_content_revisions,
    register_and_publish_repo_document,
    resolve_document_locator,
    rewrite_published_document_references,
    rewrite_repository_path_references,
    validate_publication_plan,
    verify_published_document_archive,
    verify_publication_candidate,
)
from tools.evidence.librarian.cli import main


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_plan(path: Path, plan: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _plan_entry(
    *,
    repo_root: Path,
    source_path: str,
    document_id: str,
    action: str,
    proposed_destination: str,
) -> dict[str, Any]:
    source = repo_root / source_path
    return {
        "action": action,
        "authority_effect": "none",
        "document_id": document_id,
        "physical_move_authorized": False,
        "proposed_destination": proposed_destination,
        "source_path": source_path,
        "source_sha256": file_sha256(source),
        "source_size_bytes": source.stat().st_size,
    }


def _synthetic_plan(*, repo_root: Path, runtime_root: Path) -> dict[str, Any]:
    _write(repo_root / "docs/old/current-report.md", "# Current report\n\nBounded conclusion.\n")
    _write(repo_root / "docs/40_eiel_design/run/main_results.csv", "seed,value\n1,3\n")

    entries = [
        _plan_entry(
            repo_root=repo_root,
            source_path="docs/old/current-report.md",
            document_id="EIEL-DOC-000001",
            action="repo_move",
            proposed_destination=(
                "docs/100_governance-authority/100_governance/"
                "EIEL-DOC-000001--current-report.md"
            ),
        ),
        _plan_entry(
            repo_root=repo_root,
            source_path="docs/40_eiel_design/run/main_results.csv",
            document_id="EIEL-DOC-000002",
            action="runtime_relocate",
            proposed_destination=(
                runtime_root
                / "eiel_knowledge_librarian_docs_v1/relocated_runtime/"
                "EIEL-DOC-000002--main-results.csv"
            ).as_posix(),
        ),
    ]
    return {
        "all_ids_prefixed_in_filenames": True,
        "collision_count": 0,
        "collisions": [],
        "document_count": len(entries),
        "entries": entries,
        "physical_move_authorized": False,
        "publication_namespace": "synthetic-publication-test",
        "repo_move_count": 1,
        "runtime_relocation_count": 1,
        "schema_version": "eiel_document_publication_plan_v1",
    }


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _validate(plan_path: Path, *, repo_root: Path, runtime_root: Path) -> dict[str, Any]:
    return validate_publication_plan(
        repo_root,
        plan_path,
        runtime_root=runtime_root,
        require_clean_docs=False,
        require_complete_docs_tree=False,
    )


def _init_git_repo(repo_root: Path) -> str:
    subprocess.run(["git", "init"], cwd=repo_root, check=True, stdout=subprocess.PIPE)
    subprocess.run(
        ["git", "config", "user.email", "tests@example.invalid"],
        cwd=repo_root,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Publication Tests"],
        cwd=repo_root,
        check=True,
    )
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(
        ["git", "commit", "-m", "synthetic publication fixture"],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
    )
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()


def _document_record(entry: dict[str, Any], *, freeze_commit: str) -> dict[str, Any]:
    return {
        "schema_version": "eiel_document_record_v1",
        "collection": "documents",
        "document_id": entry["document_id"],
        "path": entry["source_path"],
        "sha256": entry["source_sha256"],
        "size_bytes": entry["source_size_bytes"],
        "document_class": "campaign_report"
        if entry["action"] == "repo_move"
        else "raw_or_machine_output_candidate",
        "currentness_status": "unreviewed",
        "canonical_reading_location": entry["source_path"] if entry["action"] == "repo_move" else None,
        "supersedes_document_ids": [],
        "superseded_by_document_ids": [],
        "linked_evidence_ids": [],
        "linked_claim_ids": [],
        "linked_authority_assertion_ids": [],
        "linked_dossier_ids": [],
        "linked_raw_asset_ids": [],
        "linked_artifact_ids": [],
        "linked_issue_ids": [],
        "identity": {
            "frozen": True,
            "path_is_identity": False,
            "reuse_policy": "never_reuse",
            "path_at_freeze": entry["source_path"],
            "sha256_at_freeze": entry["source_sha256"],
            "size_bytes_at_freeze": entry["source_size_bytes"],
            "freeze_commit": freeze_commit,
            "freeze_source_digest": "synthetic-freeze",
            "freeze_basis": ["synthetic_publication_test"],
        },
        "proposed_publication": {
            "proposal_version": "eiel_docs_publication_namespace_v1",
            "destination_class": "runtime_relocation"
            if entry["action"] == "runtime_relocate"
            else "repo_document",
            "proposed_destination": entry["proposed_destination"],
            "basis": ["synthetic_publication_test", "document_id_filename_prefix"],
            "authority_effect": "none",
            "physical_move_authorized": False,
        },
        "source_state": {"state": "identity_freeze", "observed_at_commit": freeze_commit},
        "git_state": "tracked",
    }


def _write_catalog_source(source_root: Path, plan: dict[str, Any], *, freeze_commit: str) -> None:
    for entry in plan["entries"]:
        record = _document_record(entry, freeze_commit=freeze_commit)
        _write_plan(source_root / "documents" / f"{entry['document_id']}.json", record)
    _write_plan(
        source_root / "issues/eiel-issue-docs-v1-runtime.json",
        {
            "schema_version": "eiel_issue_v1",
            "issue_id": "eiel-issue-docs-v1-runtime",
            "owner_collection": "documents",
            "issue_kind": "probable_runtime_output_in_docs",
            "review_state": "open",
            "remediation_state": "open",
            "document_ids": ["EIEL-DOC-000002"],
            "document_paths": ["docs/40_eiel_design/run/main_results.csv"],
            "paths": ["docs/40_eiel_design/run/main_results.csv"],
        },
    )
    _write_plan(
        source_root / "policies/document_publication_namespace_v1.json",
        {
            "schema_version": "eiel_document_publication_namespace_policy_v1",
            "status": "candidate",
            "proposed_destination_rule": {"physical_move_authorized": False},
            "forbidden": ["move_files_in_docs_v1"],
        },
    )
    (source_root / "document_relations").mkdir(parents=True, exist_ok=True)
    (source_root / "manifests").mkdir(parents=True, exist_ok=True)


def _post_publication_document_record(
    *,
    repo_root: Path,
    document_id: str,
    path: str,
    freeze_commit: str,
) -> dict[str, Any]:
    source = repo_root / path
    digest = file_sha256(source)
    size_bytes = source.stat().st_size
    return {
        "schema_version": "eiel_document_record_v1",
        "collection": "documents",
        "document_id": document_id,
        "title": f"Synthetic {document_id}",
        "path": path,
        "canonical_reading_location": path,
        "sha256": digest,
        "size_bytes": size_bytes,
        "git_state": "tracked",
        "document_class": "decision_report",
        "document_class_candidates": ["decision_report"],
        "refined_lifecycle_candidate": "synthetic_post_publication_fixture",
        "currentness_status": "unreviewed",
        "currentness_candidate": "synthetic_review_required",
        "authored_or_generated": "candidate",
        "authorship_candidate": "authored_candidate",
        "retention_class": "candidate",
        "retention_candidate": "repo_publication",
        "review_state": "candidate",
        "review_flags": ["synthetic_post_publication_fixture"],
        "supersedes_document_ids": [],
        "superseded_by_document_ids": [],
        "linked_evidence_ids": [],
        "linked_claim_ids": [],
        "linked_authority_assertion_ids": [],
        "linked_dossier_ids": [],
        "linked_raw_asset_ids": [],
        "linked_artifact_ids": [],
        "linked_issue_ids": [],
        "source_state": {
            "state": "archive_publication_v1",
            "observed_at_commit": freeze_commit,
            "source_digest": digest,
        },
        "identity": {
            "frozen": True,
            "path_at_freeze": "docs/40_eiel_design/original.md",
            "sha256_at_freeze": digest,
            "size_bytes_at_freeze": size_bytes,
            "freeze_commit": freeze_commit,
            "freeze_source_digest": digest,
            "freeze_basis": ["synthetic_archive_publication_fixture"],
            "reuse_policy": "never_reuse",
            "path_is_identity": False,
            "path_history": [
                {
                    "publication_id": "synthetic-archive-publication-v1",
                    "from_path": "docs/40_eiel_design/original.md",
                    "to_path": path,
                    "sha256": digest,
                    "size_bytes": size_bytes,
                    "content_unchanged": True,
                    "authority_effect": "none",
                }
            ],
            "content_revisions": [],
        },
        "proposed_publication": {
            "proposal_version": "eiel_docs_publication_namespace_v1",
            "destination_class": "repo_document",
            "proposed_destination": path,
            "basis": ["synthetic_archive_publication_fixture"],
            "confidence": "candidate",
            "authority_effect": "none",
            "physical_move_authorized": True,
            "publication_state": "published",
        },
        "publication": {
            "publication_id": "synthetic-archive-publication-v1",
            "action": "repo_move",
            "path_before_publication": "docs/40_eiel_design/original.md",
            "current_path": path,
            "content_unchanged": True,
            "authority_effect": "none",
            "prepublication_head": freeze_commit,
        },
        "notes": "Synthetic published document fixture.",
    }


def _write_post_publication_source(
    *,
    repo_root: Path,
    source_root: Path,
    head: str,
    existing_path: str,
) -> None:
    for directory in (
        "artifacts",
        "records",
        "claims",
        "authorities",
        "raw_assets",
        "issues",
        "documents",
        "document_relations",
        "accessions",
        "manifests",
        "policies",
        "schemas",
    ):
        (source_root / directory).mkdir(parents=True, exist_ok=True)
    _write_plan(
        source_root / "schemas/document_record_v1.schema.json",
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )
    _write_plan(
        source_root / "schemas/document_relation_v1.schema.json",
        {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"},
    )
    _write_plan(
        source_root / "documents/EIEL-DOC-000001.json",
        _post_publication_document_record(
            repo_root=repo_root,
            document_id="EIEL-DOC-000001",
            path=existing_path,
            freeze_commit=head,
        ),
    )
    _write_plan(
        source_root / "document_relations/eiel-docrel-existing-path.json",
        {
            "schema_version": "eiel_document_relation_v1",
            "collection": "document_relations",
            "relation_id": "eiel-docrel-existing-path",
            "relation_type": "path_changed",
            "source_document_id": "EIEL-DOC-000001",
            "target": {"target_type": "path", "target_id_or_path": existing_path},
            "basis": ["synthetic_existing_relation"],
            "authority_effect": "none",
            "review_state": "accepted",
            "notes": "Existing path relation that must be preserved.",
        },
    )
    _write_plan(
        source_root / "document_relations/eiel-docrel-existing-revision.json",
        {
            "schema_version": "eiel_document_relation_v1",
            "collection": "document_relations",
            "relation_id": "eiel-docrel-existing-revision",
            "relation_type": "content_revision",
            "source_document_id": "EIEL-DOC-000001",
            "target": {"target_type": "path", "target_id_or_path": existing_path},
            "basis": ["synthetic_existing_relation"],
            "authority_effect": "none",
            "review_state": "accepted",
            "notes": "Existing content revision relation that must be preserved.",
        },
    )
    _write_plan(
        source_root / "manifests/DOCUMENT_LEGACY_PATH_REDIRECTS.json",
        {
            "schema_version": "eiel_document_legacy_path_redirects_v1",
            "publication_id": "synthetic-archive-publication-v1",
            "path_is_identity": False,
            "authority_effect": "none",
            "entry_count": 1,
            "entries": [
                {
                    "document_id": "EIEL-DOC-000001",
                    "old_path": "docs/40_eiel_design/original.md",
                    "current_path": existing_path,
                    "action": "repo_move",
                    "sha256": file_sha256(repo_root / existing_path),
                    "size_bytes": (repo_root / existing_path).stat().st_size,
                }
            ],
        },
    )
    _write_plan(
        source_root / "policies/document_identity_policy_v1.json",
        {
            "schema_version": "eiel_document_identity_policy_v1",
            "id_reuse_policy": "never_reuse",
            "path_is_identity": False,
            "allocation": {"next_sequence_after_current_freeze": 2},
            "current_freeze_range": {
                "count": 1,
                "first_id": "EIEL-DOC-000001",
                "last_id": "EIEL-DOC-000001",
                "status": "frozen_for_archive_publication_v1",
            },
        },
    )
    _write_plan(
        source_root / "policies/document_publication_namespace_v1.json",
        {
            "schema_version": "eiel_document_publication_namespace_policy_v1",
            "status": "archive_publication_v1_published",
            "proposed_destination_rule": {"physical_move_authorized": True},
            "forbidden": [],
        },
    )
    _write_plan(
        source_root / "manifests/DOCUMENT_ARCHIVE_PUBLICATION_V1.json",
        {
            "schema_version": "eiel_document_archive_publication_manifest_v1",
            "publication_id": "synthetic-archive-publication-v1",
            "document_count": 1,
            "repo_document_count": 1,
            "runtime_relocation_count": 0,
            "authority_effect": "none",
            "publication_state": "published_pending_commit",
            "redirect_manifest": "manifests/DOCUMENT_LEGACY_PATH_REDIRECTS.json",
        },
    )


def _publish_synthetic_destinations(repo_root: Path, runtime_root: Path, plan: dict[str, Any]) -> None:
    for entry in plan["entries"]:
        source = repo_root / entry["source_path"]
        destination = repo_root / entry["proposed_destination"] if entry["action"] == "repo_move" else Path(entry["proposed_destination"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())


def test_publication_plan_validates_small_synthetic_repo_and_runtime_plan(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    runtime_root = tmp_path / "runtime-publication"
    plan = _synthetic_plan(repo_root=repo_root, runtime_root=runtime_root)
    plan_path = _write_plan(tmp_path / "plan.json", plan)

    result = _validate(plan_path, repo_root=repo_root, runtime_root=runtime_root)

    assert result["ok"] is True
    assert result["document_count"] == 2
    assert result["repo_move_count"] == 1
    assert result["runtime_relocation_count"] == 1
    assert result["physical_documents_modified"] is False


def test_publication_plan_requires_uid_prefixed_destinations(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    runtime_root = tmp_path / "runtime-publication"
    plan = _synthetic_plan(repo_root=repo_root, runtime_root=runtime_root)
    plan["entries"][0]["proposed_destination"] = "docs/100_governance-authority/current-report.md"
    plan_path = _write_plan(tmp_path / "plan.json", plan)

    with pytest.raises(LibrarianError, match="UID|document_id|prefix"):
        _validate(plan_path, repo_root=repo_root, runtime_root=runtime_root)


def test_publication_plan_rejects_path_escape_and_unknown_destination_roots(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    runtime_root = tmp_path / "runtime-publication"
    plan = _synthetic_plan(repo_root=repo_root, runtime_root=runtime_root)

    escaped_source = json.loads(json.dumps(plan))
    escaped_source["entries"][0]["source_path"] = "../outside.md"
    escaped_source_path = _write_plan(tmp_path / "escaped-source-plan.json", escaped_source)
    with pytest.raises(LibrarianError, match="escape|outside|source"):
        _validate(escaped_source_path, repo_root=repo_root, runtime_root=runtime_root)

    escaped_destination = json.loads(json.dumps(plan))
    escaped_destination["entries"][1]["proposed_destination"] = (
        tmp_path / "other-runtime/EIEL-DOC-000002--main-results.csv"
    ).as_posix()
    escaped_destination_path = _write_plan(
        tmp_path / "escaped-destination-plan.json",
        escaped_destination,
    )
    with pytest.raises(LibrarianError, match="escape|outside|destination|runtime"):
        _validate(escaped_destination_path, repo_root=repo_root, runtime_root=runtime_root)


def test_publication_plan_rejects_destination_collisions(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    runtime_root = tmp_path / "runtime-publication"
    plan = _synthetic_plan(repo_root=repo_root, runtime_root=runtime_root)
    plan["entries"][1]["action"] = "repo_move"
    plan["entries"][1]["proposed_destination"] = plan["entries"][0]["proposed_destination"]
    plan_path = _write_plan(tmp_path / "plan.json", plan)

    with pytest.raises(LibrarianError, match="collision|duplicate"):
        _validate(plan_path, repo_root=repo_root, runtime_root=runtime_root)


def test_publication_plan_rejects_source_hash_or_size_drift(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    runtime_root = tmp_path / "runtime-publication"
    plan = _synthetic_plan(repo_root=repo_root, runtime_root=runtime_root)
    plan_path = _write_plan(tmp_path / "plan.json", plan)
    _write(repo_root / plan["entries"][0]["source_path"], "# Current report\n\nChanged after freeze.\n")

    with pytest.raises(LibrarianError, match="hash|size|drift"):
        _validate(plan_path, repo_root=repo_root, runtime_root=runtime_root)


def test_prepare_publication_candidate_copies_deterministically_and_stages_runtime(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    runtime_root = tmp_path / "runtime-publication"
    plan = _synthetic_plan(repo_root=repo_root, runtime_root=runtime_root)
    plan_path = _write_plan(tmp_path / "plan.json", plan)
    candidate_a = tmp_path / "candidate-a"
    candidate_b = tmp_path / "candidate-b"
    runtime_stage_a = tmp_path / "runtime-stage-a"
    runtime_stage_b = tmp_path / "runtime-stage-b"

    receipt_a = prepare_publication_candidate(
        repo_root,
        plan_path,
        candidate_a,
        runtime_stage_a,
        runtime_root=runtime_root,
        require_clean_docs=False,
        require_complete_docs_tree=False,
    )
    receipt_b = prepare_publication_candidate(
        repo_root,
        plan_path,
        candidate_b,
        runtime_stage_b,
        runtime_root=runtime_root,
        require_clean_docs=False,
        require_complete_docs_tree=False,
    )

    repo_candidate = (
        candidate_a
        / "docs/100_governance-authority/100_governance/"
        "EIEL-DOC-000001--current-report.md"
    )
    runtime_candidate = (
        runtime_stage_a
        / "eiel_knowledge_librarian_docs_v1/relocated_runtime/"
        "EIEL-DOC-000002--main-results.csv"
    )
    assert receipt_a["ok"] is True
    assert receipt_b["ok"] is True
    assert receipt_a["candidate_verification"]["repo_file_count"] == 1
    assert receipt_a["candidate_verification"]["runtime_file_count"] == 1
    assert receipt_b["candidate_verification"]["repo_file_count"] == 1
    assert receipt_b["candidate_verification"]["runtime_file_count"] == 1
    assert repo_candidate.read_text(encoding="utf-8") == "# Current report\n\nBounded conclusion.\n"
    assert runtime_candidate.read_text(encoding="utf-8") == "seed,value\n1,3\n"
    assert (repo_root / "docs/old/current-report.md").is_file()
    assert (repo_root / "docs/40_eiel_design/run/main_results.csv").is_file()
    assert not (runtime_root / "eiel_knowledge_librarian_docs_v1").exists()
    assert _tree_digest(candidate_a) == _tree_digest(candidate_b)
    assert _tree_digest(runtime_stage_a) == _tree_digest(runtime_stage_b)


def test_verify_publication_candidate_accepts_candidate_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    runtime_root = tmp_path / "runtime-publication"
    plan = _synthetic_plan(repo_root=repo_root, runtime_root=runtime_root)
    plan_path = _write_plan(tmp_path / "plan.json", plan)
    candidate_root = tmp_path / "candidate"
    runtime_stage_root = tmp_path / "runtime-stage"
    prepare_publication_candidate(
        repo_root,
        plan_path,
        candidate_root,
        runtime_stage_root,
        runtime_root=runtime_root,
        require_clean_docs=False,
        require_complete_docs_tree=False,
    )

    verification = verify_publication_candidate(
        repo_root,
        plan_path,
        candidate_root,
        runtime_stage_root,
        runtime_root=runtime_root,
    )
    assert verification["ok"] is True
    assert verification["repo_file_count"] == 1
    assert verification["runtime_file_count"] == 1

    tampered = (
        candidate_root
        / "docs/100_governance-authority/100_governance/"
        "EIEL-DOC-000001--current-report.md"
    )
    _write(tampered, "# Current report\n\nTampered candidate.\n")

    with pytest.raises(LibrarianError, match="candidate|hash|mismatch"):
        verify_publication_candidate(
            repo_root,
            plan_path,
            candidate_root,
            runtime_stage_root,
            runtime_root=runtime_root,
        )


def test_finalize_catalog_publication_records_published_paths_redirects_and_resolves_locators(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    runtime_root = tmp_path / "runtime-publication"
    source_root = tmp_path / "catalog-source"
    plan = _synthetic_plan(repo_root=repo_root, runtime_root=runtime_root)
    plan_path = _write_plan(tmp_path / "plan.json", plan)
    _publish_synthetic_destinations(repo_root, runtime_root, plan)
    prepublication_head = _init_git_repo(repo_root)
    _write_catalog_source(source_root, plan, freeze_commit=prepublication_head)

    receipt = finalize_catalog_publication(
        repo_root,
        source_root,
        plan_path,
        publication_id="synthetic-publication-v1",
        prepublication_head=prepublication_head,
        runtime_root=runtime_root,
    )

    assert receipt["semantic_equivalence"] is True
    assert receipt["repo_document_count"] == 1
    assert receipt["runtime_relocation_count"] == 1
    assert receipt["redirect_entry_count"] == 2
    assert receipt["physical_content_modified"] is False

    repo_record = json.loads((source_root / "documents/EIEL-DOC-000001.json").read_text())
    runtime_record = json.loads((source_root / "documents/EIEL-DOC-000002.json").read_text())
    assert repo_record["path"] == plan["entries"][0]["proposed_destination"]
    assert repo_record["canonical_reading_location"] == plan["entries"][0]["proposed_destination"]
    assert runtime_record["path"] == plan["entries"][1]["proposed_destination"]
    assert runtime_record["git_state"] == "runtime_relocated"
    assert repo_record["publication"]["authority_effect"] == "none"
    assert repo_record["identity"]["path_history"] == [
        {
            "publication_id": "synthetic-publication-v1",
            "from_path": plan["entries"][0]["source_path"],
            "to_path": plan["entries"][0]["proposed_destination"],
            "sha256": plan["entries"][0]["source_sha256"],
            "size_bytes": plan["entries"][0]["source_size_bytes"],
            "content_unchanged": True,
            "authority_effect": "none",
        }
    ]

    redirects = json.loads((source_root / "manifests/DOCUMENT_LEGACY_PATH_REDIRECTS.json").read_text())
    assert redirects["entry_count"] == 2
    assert {
        (row["document_id"], row["old_path"], row["current_path"])
        for row in redirects["entries"]
    } == {
        (
            "EIEL-DOC-000001",
            plan["entries"][0]["source_path"],
            plan["entries"][0]["proposed_destination"],
        ),
        (
            "EIEL-DOC-000002",
            plan["entries"][1]["source_path"],
            plan["entries"][1]["proposed_destination"],
        ),
    }

    issue = json.loads((source_root / "issues/eiel-issue-docs-v1-runtime.json").read_text())
    assert issue["review_state"] == "resolved"
    assert issue["document_paths_at_detection"] == ["docs/40_eiel_design/run/main_results.csv"]
    assert issue["document_paths"] == [plan["entries"][1]["proposed_destination"]]

    by_id = resolve_document_locator(source_root, "EIEL-DOC-000001")
    by_old_path = resolve_document_locator(source_root, plan["entries"][0]["source_path"])
    by_current_path = resolve_document_locator(source_root, plan["entries"][0]["proposed_destination"])
    assert by_id["current_path"] == plan["entries"][0]["proposed_destination"]
    for resolution in (by_old_path, by_current_path):
        assert resolution["document_id"] == by_id["document_id"]
        assert resolution["current_path"] == by_id["current_path"]
        assert resolution["path_at_freeze"] == by_id["path_at_freeze"]
        assert resolution["sha256"] == by_id["sha256"]
        assert resolution["authority_effect"] == "none"
    with pytest.raises(LibrarianError, match="not found"):
        resolve_document_locator(source_root, "docs/missing.md")


def test_rewrite_repository_path_references_is_dry_run_then_authorized_active_file_rewrite(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    runtime_root = tmp_path / "runtime-publication"
    source_root = tmp_path / "catalog-source"
    plan = _synthetic_plan(repo_root=repo_root, runtime_root=runtime_root)
    plan_path = _write_plan(tmp_path / "plan.json", plan)
    _publish_synthetic_destinations(repo_root, runtime_root, plan)
    _write(repo_root / "tools/active_consumer.py", 'REPORT = "docs/old/current-report.md"\n')
    _write(repo_root / "docs/historical_note.md", "docs/old/current-report.md remains historical here\n")
    prepublication_head = _init_git_repo(repo_root)
    _write_catalog_source(source_root, plan, freeze_commit=prepublication_head)
    finalize_catalog_publication(
        repo_root,
        source_root,
        plan_path,
        publication_id="synthetic-publication-v1",
        prepublication_head=prepublication_head,
        runtime_root=runtime_root,
    )
    redirect_manifest = source_root / "manifests/DOCUMENT_LEGACY_PATH_REDIRECTS.json"

    dry_run = rewrite_repository_path_references(repo_root, redirect_manifest)

    assert dry_run["mode"] == "dry_run"
    assert dry_run["changed_file_count"] == 1
    assert dry_run["replacement_count"] == 1
    assert dry_run["changed_files"][0]["path"] == "tools/active_consumer.py"
    assert (repo_root / "tools/active_consumer.py").read_text(encoding="utf-8") == (
        'REPORT = "docs/old/current-report.md"\n'
    )

    with pytest.raises(LibrarianError, match="authorization"):
        rewrite_repository_path_references(
            repo_root,
            redirect_manifest,
            authorization="wrong-token",
        )

    executed = rewrite_repository_path_references(
        repo_root,
        redirect_manifest,
        authorization=REFERENCE_REWRITE_AUTHORIZATION,
    )

    assert executed["mode"] == "execute"
    assert executed["changed_file_count"] == 1
    assert executed["replacement_count"] == 1
    assert (repo_root / "tools/active_consumer.py").read_text(encoding="utf-8") == (
        f'REPORT = "{plan["entries"][0]["proposed_destination"]}"\n'
    )
    assert (repo_root / "docs/historical_note.md").read_text(encoding="utf-8") == (
        "docs/old/current-report.md remains historical here\n"
    )


def test_selected_living_document_rewrite_and_content_revision_preserve_stable_id(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    runtime_root = tmp_path / "runtime-publication"
    source_root = tmp_path / "catalog-source"
    plan = _synthetic_plan(repo_root=repo_root, runtime_root=runtime_root)
    plan_path = _write_plan(tmp_path / "plan.json", plan)
    _publish_synthetic_destinations(repo_root, runtime_root, plan)
    prepublication_head = _init_git_repo(repo_root)
    _write_catalog_source(source_root, plan, freeze_commit=prepublication_head)
    finalize_catalog_publication(
        repo_root,
        source_root,
        plan_path,
        publication_id="synthetic-publication-v1",
        prepublication_head=prepublication_head,
        runtime_root=runtime_root,
    )
    current_path = repo_root / plan["entries"][0]["proposed_destination"]
    _write(
        current_path,
        "# Current report\n\nSee docs/40_eiel_design/run/main_results.csv.\n",
    )
    redirect_manifest = source_root / "manifests/DOCUMENT_LEGACY_PATH_REDIRECTS.json"

    dry_run = rewrite_published_document_references(
        repo_root,
        source_root,
        redirect_manifest,
        ["EIEL-DOC-000001"],
    )
    assert dry_run["mode"] == "dry_run"
    assert dry_run["replacement_count"] == 1
    assert "docs/40_eiel_design/run/main_results.csv" in current_path.read_text()

    with pytest.raises(LibrarianError, match="authorization"):
        rewrite_published_document_references(
            repo_root,
            source_root,
            redirect_manifest,
            ["EIEL-DOC-000001"],
            authorization="wrong-token",
        )

    executed = rewrite_published_document_references(
        repo_root,
        source_root,
        redirect_manifest,
        ["EIEL-DOC-000001"],
        authorization=DOCUMENT_REFERENCE_REWRITE_AUTHORIZATION,
    )
    assert executed["replacement_count"] == 1
    assert plan["entries"][1]["proposed_destination"] in current_path.read_text()

    receipt = record_document_content_revisions(
        repo_root,
        source_root,
        ["EIEL-DOC-000001"],
        reason="Refresh living document locators after synthetic publication.",
        authorization=CONTENT_REVISION_AUTHORIZATION,
    )
    assert receipt["revised_document_count"] == 1
    record = json.loads((source_root / "documents/EIEL-DOC-000001.json").read_text())
    assert record["document_id"] == "EIEL-DOC-000001"
    assert record["sha256"] == file_sha256(current_path)
    assert record["identity"]["sha256_at_freeze"] == plan["entries"][0]["source_sha256"]
    assert len(record["identity"]["content_revisions"]) == 1
    assert record["identity"]["content_revisions"][0]["authority_effect"] == "none"
    revision_relations = [
        json.loads(path.read_text())
        for path in (source_root / "document_relations").glob("*.json")
        if json.loads(path.read_text()).get("relation_type") == "content_revision"
    ]
    assert len(revision_relations) == 1


def test_complete_published_archive_verification_checks_membership_hashes_and_shelves(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    runtime_root = tmp_path / "runtime-publication"
    source_root = tmp_path / "catalog-source"
    plan = _synthetic_plan(repo_root=repo_root, runtime_root=runtime_root)
    plan_path = _write_plan(tmp_path / "plan.json", plan)
    _publish_synthetic_destinations(repo_root, runtime_root, plan)
    prepublication_head = _init_git_repo(repo_root)
    _write_catalog_source(source_root, plan, freeze_commit=prepublication_head)
    finalize_catalog_publication(
        repo_root,
        source_root,
        plan_path,
        publication_id="synthetic-publication-v1",
        prepublication_head=prepublication_head,
        runtime_root=runtime_root,
    )
    (repo_root / "docs/old/current-report.md").unlink()
    (repo_root / "docs/old").rmdir()
    (repo_root / "docs/40_eiel_design/run/main_results.csv").unlink()
    (repo_root / "docs/40_eiel_design/run").rmdir()
    (repo_root / "docs/40_eiel_design").rmdir()
    for shelf in ALLOWED_DOC_SHELVES:
        (repo_root / "docs" / shelf).mkdir(parents=True, exist_ok=True)

    receipt = verify_published_document_archive(
        repo_root,
        source_root,
        runtime_root=runtime_root,
    )
    assert receipt["ok"] is True
    assert receipt["document_count"] == 2
    assert receipt["repo_document_count"] == 1
    assert receipt["runtime_document_count"] == 1
    assert receipt["redirect_count"] == 2

    current_path = repo_root / plan["entries"][0]["proposed_destination"]
    _write(current_path, "tampered\n")
    with pytest.raises(LibrarianError, match="SHA-256|size|verification"):
        verify_published_document_archive(
            repo_root,
            source_root,
            runtime_root=runtime_root,
        )


def test_register_and_publish_document_preserves_existing_relations_and_redirects(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    source_root = tmp_path / "catalog-source"
    existing_path = (
        "docs/500_campaigns-decisions-closeouts/520_campaign-reports/"
        "EIEL-DOC-000001--old.md"
    )
    incoming_path = "docs/900_lifecycle/940_incoming/new-closeout.md"
    _write(repo_root / existing_path, "# Old\n\nPublished.\n")
    _write(repo_root / incoming_path, "# New\n\nReady for publication.\n")
    head = _init_git_repo(repo_root)
    _write_post_publication_source(
        repo_root=repo_root,
        source_root=source_root,
        head=head,
        existing_path=existing_path,
    )
    existing_path_relation = json.loads(
        (source_root / "document_relations/eiel-docrel-existing-path.json").read_text()
    )
    existing_revision_relation = json.loads(
        (source_root / "document_relations/eiel-docrel-existing-revision.json").read_text()
    )
    existing_redirects = json.loads(
        (source_root / "manifests/DOCUMENT_LEGACY_PATH_REDIRECTS.json").read_text()
    )

    receipt = register_and_publish_repo_document(
        repo_root=repo_root,
        source_root=source_root,
        path=incoming_path,
        title="New closeout",
        document_class="decision_report",
        destination_shelf="500_campaigns-decisions-closeouts/540_handoffs",
        observed_commit=head,
        linked_dossier_ids=[],
        publication_id="synthetic-post-publication-ingress-v1",
    )

    destination = (
        "docs/500_campaigns-decisions-closeouts/540_handoffs/"
        "EIEL-DOC-000002--new-closeout.md"
    )
    assert receipt["document_id"] == "EIEL-DOC-000002"
    assert receipt["current_path"] == destination
    assert receipt["physical_document_moved"] is True
    assert not (repo_root / incoming_path).exists()
    assert (repo_root / destination).read_text(encoding="utf-8") == (
        "# New\n\nReady for publication.\n"
    )

    record = json.loads((source_root / "documents/EIEL-DOC-000002.json").read_text())
    assert record["path"] == destination
    assert record["identity"]["path_at_freeze"] == incoming_path
    assert record["identity"]["path_history"][0]["from_path"] == incoming_path
    assert record["identity"]["path_history"][0]["to_path"] == destination
    assert record["publication"]["authority_effect"] == "none"

    assert json.loads(
        (source_root / "document_relations/eiel-docrel-existing-path.json").read_text()
    ) == existing_path_relation
    assert json.loads(
        (source_root / "document_relations/eiel-docrel-existing-revision.json").read_text()
    ) == existing_revision_relation
    all_relations = [
        json.loads(path.read_text())
        for path in sorted((source_root / "document_relations").glob("*.json"))
    ]
    assert len(all_relations) == 3
    assert {
        row["relation_type"] for row in all_relations if row["source_document_id"] == "EIEL-DOC-000002"
    } == {"path_changed"}

    redirects = json.loads((source_root / "manifests/DOCUMENT_LEGACY_PATH_REDIRECTS.json").read_text())
    assert redirects["entries"][0] == existing_redirects["entries"][0]
    assert redirects["entry_count"] == 2
    assert {
        (row["document_id"], row["old_path"], row["current_path"])
        for row in redirects["entries"]
    } == {
        ("EIEL-DOC-000001", "docs/40_eiel_design/original.md", existing_path),
        ("EIEL-DOC-000002", incoming_path, destination),
    }


def test_register_and_publish_document_handles_newline_in_tracked_path(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    source_root = tmp_path / "catalog-source"
    existing_path = (
        "docs/500_campaigns-decisions-closeouts/520_campaign-reports/"
        "EIEL-DOC-000001--old.md"
    )
    incoming_path = "docs/900_lifecycle/940_incoming/research-reference\n.md"
    _write(repo_root / existing_path, "# Old\n\nPublished.\n")
    _write(repo_root / incoming_path, "# Research reference\n")
    head = _init_git_repo(repo_root)
    _write_post_publication_source(
        repo_root=repo_root,
        source_root=source_root,
        head=head,
        existing_path=existing_path,
    )

    receipt = register_and_publish_repo_document(
        repo_root=repo_root,
        source_root=source_root,
        path=incoming_path,
        title="Research reference",
        document_class="research_reference",
        destination_shelf="600_references-publications/610_books-papers-external",
        observed_commit=head,
    )

    destination = (
        "docs/600_references-publications/610_books-papers-external/"
        "EIEL-DOC-000002--research-reference.md"
    )
    assert receipt["current_path"] == destination
    assert not (repo_root / incoming_path).exists()
    assert (repo_root / destination).read_text(encoding="utf-8") == (
        "# Research reference\n"
    )


def test_register_and_publish_document_cli_generates_aggregates_and_reading_copy_relation(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    source_root = tmp_path / "catalog-source"
    existing_path = (
        "docs/500_campaigns-decisions-closeouts/520_campaign-reports/"
        "EIEL-DOC-000001--old.md"
    )
    incoming_path = "docs/900_lifecycle/940_incoming/new-reading-copy.html"
    _write(repo_root / existing_path, "# Old\n\nPublished.\n")
    _write(repo_root / incoming_path, "<!doctype html><title>New reading copy</title>\n")
    head = _init_git_repo(repo_root)
    _write_post_publication_source(
        repo_root=repo_root,
        source_root=source_root,
        head=head,
        existing_path=existing_path,
    )

    exit_code = main(
        [
            "register-and-publish-document",
            "--repo-root",
            str(repo_root),
            "--source-root",
            str(source_root),
            "--path",
            incoming_path,
            "--title",
            "New reading copy",
            "--document-class",
            "generated_support",
            "--destination-shelf",
            "700_generated-reading-support/710_html-reading-copies",
            "--observed-commit",
            head,
            "--dossier",
            "dossier-01-programme-evidence-map-methods-claim-boundaries",
            "--reading-copy-of",
            "EIEL-DOC-000001",
            "--json",
        ]
    )

    assert exit_code == 0
    assert (source_root / "accessions/documents.json").is_file()
    assert (source_root / "accessions/document_relations.json").is_file()
    assert (source_root / "SOURCE_MANIFEST.json").is_file()
    destination = (
        "docs/700_generated-reading-support/710_html-reading-copies/"
        "EIEL-DOC-000002--new-reading-copy.html"
    )
    assert (repo_root / destination).is_file()

    documents = json.loads((source_root / "accessions/documents.json").read_text())
    new_record = [row for row in documents if row["document_id"] == "EIEL-DOC-000002"][0]
    assert new_record["path"] == destination
    assert new_record["linked_dossier_ids"] == [
        "dossier-01-programme-evidence-map-methods-claim-boundaries"
    ]
    relations = json.loads((source_root / "accessions/document_relations.json").read_text())
    reading_relations = [
        row
        for row in relations
        if row["relation_type"] == "reading_copy_of"
        and row["source_document_id"] == "EIEL-DOC-000002"
    ]
    assert len(reading_relations) == 1
    assert reading_relations[0]["target"] == {
        "target_type": "document",
        "target_id_or_path": "EIEL-DOC-000001",
    }
    manifest = json.loads((source_root / "SOURCE_MANIFEST.json").read_text())
    assert manifest["object_counts"]["documents"] == 2
    assert manifest["object_counts"]["document_relations"] == 4


def test_register_and_publish_document_fails_closed_on_guard_violations(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    source_root = tmp_path / "catalog-source"
    existing_path = (
        "docs/500_campaigns-decisions-closeouts/520_campaign-reports/"
        "EIEL-DOC-000001--old.md"
    )
    incoming_path = "docs/900_lifecycle/940_incoming/new-closeout.md"
    _write(repo_root / existing_path, "# Old\n\nPublished.\n")
    _write(repo_root / incoming_path, "# New\n\nReady for publication.\n")
    head = _init_git_repo(repo_root)
    _write_post_publication_source(
        repo_root=repo_root,
        source_root=source_root,
        head=head,
        existing_path=existing_path,
    )

    with pytest.raises(LibrarianError, match="HEAD mismatch"):
        register_and_publish_repo_document(
            repo_root=repo_root,
            source_root=source_root,
            path=incoming_path,
            title="New closeout",
            document_class="decision_report",
            destination_shelf="500_campaigns-decisions-closeouts/540_handoffs",
            observed_commit="0" * 40,
        )
    assert (repo_root / incoming_path).is_file()

    with pytest.raises(LibrarianError, match="numbered|publication namespace"):
        register_and_publish_repo_document(
            repo_root=repo_root,
            source_root=source_root,
            path=incoming_path,
            title="New closeout",
            document_class="decision_report",
            destination_shelf="campaigns-decisions-closeouts/540_handoffs",
            observed_commit=head,
        )
    assert (repo_root / incoming_path).is_file()

    duplicate_destination = (
        repo_root
        / "docs/500_campaigns-decisions-closeouts/540_handoffs/"
        "EIEL-DOC-000002--new-closeout.md"
    )
    _write(duplicate_destination, "collision\n")
    with pytest.raises(LibrarianError, match="destination already exists"):
        register_and_publish_repo_document(
            repo_root=repo_root,
            source_root=source_root,
            path=incoming_path,
            title="New closeout",
            document_class="decision_report",
            destination_shelf="500_campaigns-decisions-closeouts/540_handoffs",
            observed_commit=head,
        )
    duplicate_destination.unlink()
    assert (repo_root / incoming_path).is_file()

    untracked_path = "docs/900_lifecycle/940_incoming/untracked.md"
    _write(repo_root / untracked_path, "untracked\n")
    with pytest.raises(LibrarianError, match="not tracked"):
        register_and_publish_repo_document(
            repo_root=repo_root,
            source_root=source_root,
            path=untracked_path,
            title="Untracked",
            document_class="handoff",
            destination_shelf="500_campaigns-decisions-closeouts/540_handoffs",
            observed_commit=head,
        )
    assert (repo_root / untracked_path).is_file()

    _write_plan(
        source_root / "schemas/document_record_v1.schema.json",
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {
                "document_class": {"enum": ["decision_report", "generated_support"]}
            },
        },
    )
    with pytest.raises(LibrarianError, match="controlled vocabulary"):
        register_and_publish_repo_document(
            repo_root=repo_root,
            source_root=source_root,
            path=incoming_path,
            title="New closeout",
            document_class="generated_reading_copy",
            destination_shelf="500_campaigns-decisions-closeouts/540_handoffs",
            observed_commit=head,
        )
    assert (repo_root / incoming_path).is_file()


def test_register_and_publish_document_adopts_unique_preassigned_id(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    source_root = tmp_path / "catalog-source"
    existing_path = (
        "docs/500_campaigns-decisions-closeouts/520_campaign-reports/"
        "EIEL-DOC-000001--old.md"
    )
    incoming_path = (
        "docs/900_lifecycle/940_incoming/"
        "EIEL-DOC-000003--preassigned-closeout.md"
    )
    _write(repo_root / existing_path, "# Old\n\nPublished.\n")
    _write(repo_root / incoming_path, "# Preassigned\n\nReady.\n")
    head = _init_git_repo(repo_root)
    _write_post_publication_source(
        repo_root=repo_root,
        source_root=source_root,
        head=head,
        existing_path=existing_path,
    )

    receipt = register_and_publish_repo_document(
        repo_root=repo_root,
        source_root=source_root,
        path=incoming_path,
        title="Preassigned closeout",
        document_class="decision_report",
        destination_shelf="500_campaigns-decisions-closeouts/540_handoffs",
        observed_commit=head,
    )

    destination = (
        "docs/500_campaigns-decisions-closeouts/540_handoffs/"
        "EIEL-DOC-000003--preassigned-closeout.md"
    )
    assert receipt["document_id"] == "EIEL-DOC-000003"
    assert receipt["current_path"] == destination
    assert (repo_root / destination).read_text(encoding="utf-8") == "# Preassigned\n\nReady.\n"
    policy = json.loads(
        (source_root / "policies/document_identity_policy_v1.json").read_text()
    )
    assert policy["allocation"]["next_sequence_after_current_freeze"] == 4


def test_register_and_publish_document_rejects_ambiguous_preassigned_id(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    source_root = tmp_path / "catalog-source"
    existing_path = (
        "docs/500_campaigns-decisions-closeouts/520_campaign-reports/"
        "EIEL-DOC-000001--old.md"
    )
    incoming_path = (
        "docs/900_lifecycle/940_incoming/"
        "EIEL-DOC-000003--preassigned-closeout.md"
    )
    conflicting_path = (
        "docs/900_lifecycle/940_incoming/"
        "EIEL-DOC-000003--conflicting-reservation.md"
    )
    _write(repo_root / existing_path, "# Old\n")
    _write(repo_root / incoming_path, "# Preassigned\n")
    _write(repo_root / conflicting_path, "# Conflict\n")
    head = _init_git_repo(repo_root)
    _write_post_publication_source(
        repo_root=repo_root,
        source_root=source_root,
        head=head,
        existing_path=existing_path,
    )

    with pytest.raises(LibrarianError, match="not uniquely reserved"):
        register_and_publish_repo_document(
            repo_root=repo_root,
            source_root=source_root,
            path=incoming_path,
            title="Preassigned closeout",
            document_class="decision_report",
            destination_shelf="500_campaigns-decisions-closeouts/540_handoffs",
            observed_commit=head,
        )
    assert (repo_root / incoming_path).is_file()
