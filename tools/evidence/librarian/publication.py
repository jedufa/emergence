"""Fail-closed physical publication for the EIEL documentation collection.

The publication workflow is deliberately split into validation, candidate
preparation, candidate verification, and an explicitly authorized swap.  No
source file is changed while a candidate is prepared.  Runtime payloads are
copied and verified before the original documentation tree is replaced.
"""

from __future__ import annotations

from engine.runtime_roots import public_runtime_path

import hashlib
import json
import os
import re
import shutil
import subprocess
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .core import LibrarianError, canonical_json_bytes, file_sha256, write_json


PUBLICATION_SCHEMA_VERSION = "eiel_document_archive_publication_v1"
PUBLICATION_AUTHORIZATION = "EIEL_KNOWLEDGE_ARCHIVE_PUBLICATION_1_0"
REFERENCE_REWRITE_AUTHORIZATION = "EIEL_KNOWLEDGE_PATH_REFERENCE_REWRITE_1_0"
DOCUMENT_REFERENCE_REWRITE_AUTHORIZATION = (
    "EIEL_KNOWLEDGE_DOCUMENT_REFERENCE_REWRITE_1_0"
)
CONTENT_REVISION_AUTHORIZATION = "EIEL_KNOWLEDGE_DOCUMENT_CONTENT_REVISION_1_0"
DOCUMENT_ID_PATTERN = re.compile(r"^EIEL-DOC-[0-9]{6}$")
DEFAULT_RUNTIME_ROOT = Path(
    public_runtime_path('evidence_raw/eiel_knowledge_librarian_docs_v1/relocated_runtime')
)
ALLOWED_DOC_SHELVES = {
    "000_start-here",
    "100_governance-authority",
    "200_strategy-science-programme",
    "300_architecture-specifications",
    "400_implementation-engineering",
    "500_campaigns-decisions-closeouts",
    "600_references-publications",
    "700_generated-reading-support",
    "800_machine-readable-support",
    "900_lifecycle",
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _git(repo_root: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _relative_docs_path(value: str, *, label: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or path.parts[0] != "docs" or ".." in path.parts:
        raise LibrarianError(f"{label} must be a safe repo-relative docs path: {value}")
    return path


def _resolved_under(base: Path, path: Path, *, label: str) -> Path:
    base_resolved = base.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(base_resolved)
    except ValueError as exc:
        raise LibrarianError(f"{label} escapes {base_resolved}: {path}") from exc
    return resolved


def _runtime_relative(value: str, runtime_root: Path) -> PurePosixPath:
    destination = Path(value)
    if not destination.is_absolute() or ".." in destination.parts:
        raise LibrarianError(f"runtime destination must be absolute and normalized: {value}")
    root = runtime_root.resolve()
    try:
        relative = destination.resolve().relative_to(root)
    except ValueError as exc:
        raise LibrarianError(f"runtime destination escapes {root}: {value}") from exc
    if not relative.parts:
        raise LibrarianError(f"runtime destination may not equal runtime root: {value}")
    return PurePosixPath(relative.as_posix())


def _entry_paths(
    repo_root: Path,
    entry: Mapping[str, Any],
    runtime_root: Path,
) -> tuple[Path, PurePosixPath]:
    document_id = str(entry.get("document_id") or "")
    if not DOCUMENT_ID_PATTERN.fullmatch(document_id):
        raise LibrarianError(f"invalid document ID in publication plan: {document_id}")
    source_rel = _relative_docs_path(str(entry.get("source_path") or ""), label="source_path")
    source = _resolved_under(repo_root, repo_root / Path(source_rel.as_posix()), label="source_path")
    destination = str(entry.get("proposed_destination") or "")
    action = str(entry.get("action") or "")
    if action == "repo_move":
        destination_rel = _relative_docs_path(destination, label="repo destination")
        if len(destination_rel.parts) < 3 or destination_rel.parts[1] not in ALLOWED_DOC_SHELVES:
            raise LibrarianError(f"repo destination is outside the published shelf namespace: {destination}")
    elif action == "runtime_relocate":
        destination_rel = _runtime_relative(destination, runtime_root)
    else:
        raise LibrarianError(f"unsupported publication action for {document_id}: {action}")
    if not destination_rel.name.startswith(f"{document_id}--"):
        raise LibrarianError(f"destination filename lacks permanent UID prefix for {document_id}: {destination}")
    return source, destination_rel


def _candidate_digest(entries: list[Mapping[str, Any]]) -> str:
    rows = [
        {
            "document_id": row["document_id"],
            "source_sha256": row["source_sha256"],
            "action": row["action"],
            "proposed_destination": row["proposed_destination"],
        }
        for row in sorted(entries, key=lambda item: str(item["document_id"]))
    ]
    return hashlib.sha256(canonical_json_bytes(rows)).hexdigest()


def validate_publication_plan(
    repo_root: Path,
    plan_path: Path,
    *,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    expected_head: str | None = None,
    expected_document_count: int | None = None,
    require_clean_docs: bool = True,
    require_complete_docs_tree: bool = True,
) -> dict[str, Any]:
    """Validate every source and destination without modifying the filesystem."""

    repo_root = repo_root.resolve()
    runtime_root = runtime_root.resolve()
    plan = _load_json(plan_path)
    entries = list(plan.get("entries") or [])
    failures: list[str] = []
    if plan.get("schema_version") != "eiel_document_publication_plan_v1":
        failures.append("unexpected publication plan schema")
    if int(plan.get("document_count") or -1) != len(entries):
        failures.append("publication plan document_count mismatch")
    if expected_document_count is not None and len(entries) != expected_document_count:
        failures.append(
            f"expected {expected_document_count} document entries, observed {len(entries)}"
        )
    if int(plan.get("collision_count") or 0) != 0 or plan.get("collisions"):
        failures.append("publication plan contains destination collisions")
    if plan.get("all_ids_prefixed_in_filenames") is not True:
        failures.append("publication plan did not certify UID-prefixed filenames")
    if require_clean_docs:
        status = _git(repo_root, ["status", "--porcelain=v1", "--", "docs"])
        if status.returncode != 0:
            failures.append(status.stderr.strip() or "unable to inspect docs Git state")
        elif status.stdout:
            failures.append("docs tree is not clean at publication validation")
    if expected_head is not None:
        head = _git(repo_root, ["rev-parse", "HEAD"])
        if head.returncode != 0 or head.stdout.strip() != expected_head:
            failures.append(
                f"repository HEAD mismatch: expected {expected_head}, observed {head.stdout.strip() or 'unknown'}"
            )

    document_ids: set[str] = set()
    sources: set[str] = set()
    destinations: set[str] = set()
    repo_count = 0
    runtime_count = 0
    for entry in entries:
        document_id = str(entry.get("document_id") or "")
        source_rel = str(entry.get("source_path") or "")
        destination = str(entry.get("proposed_destination") or "")
        if document_id in document_ids:
            failures.append(f"duplicate document ID: {document_id}")
        if source_rel in sources:
            failures.append(f"duplicate source path: {source_rel}")
        if destination in destinations:
            failures.append(f"duplicate destination path: {destination}")
        document_ids.add(document_id)
        sources.add(source_rel)
        destinations.add(destination)
        if entry.get("action") == "repo_move":
            repo_count += 1
        elif entry.get("action") == "runtime_relocate":
            runtime_count += 1
        try:
            source, _destination_rel = _entry_paths(repo_root, entry, runtime_root)
        except LibrarianError as exc:
            failures.append(str(exc))
            continue
        if not source.is_file() or source.is_symlink():
            failures.append(f"source is missing, non-regular, or a symlink: {source_rel}")
            continue
        actual_size = source.stat().st_size
        expected_size_value = entry.get("source_size_bytes")
        expected_size = int(expected_size_value) if expected_size_value is not None else -1
        if actual_size != expected_size:
            failures.append(
                f"source size drift for {document_id}: expected {expected_size}, observed {actual_size}"
            )
            continue
        actual_hash = file_sha256(source)
        expected_hash = str(entry.get("source_sha256") or "")
        if actual_hash != expected_hash:
            failures.append(
                f"source SHA-256 drift for {document_id}: expected {expected_hash}, observed {actual_hash}"
            )
        if entry.get("authority_effect") != "none":
            failures.append(f"publication entry changes authority: {document_id}")

    expected_repo_count = plan.get("repo_move_count")
    expected_runtime_count = plan.get("runtime_relocation_count")
    if repo_count != (int(expected_repo_count) if expected_repo_count is not None else -1):
        failures.append("repo_move_count mismatch")
    if runtime_count != (int(expected_runtime_count) if expected_runtime_count is not None else -1):
        failures.append("runtime_relocation_count mismatch")
    if require_complete_docs_tree:
        tracked = _git(repo_root, ["ls-files", "-z", "--", "docs"])
        if tracked.returncode != 0:
            failures.append(tracked.stderr.strip() or "unable to enumerate tracked docs")
        else:
            tracked_paths = {row for row in tracked.stdout.split("\0") if row}
            if tracked_paths != sources:
                missing = sorted(tracked_paths - sources)[:10]
                extra = sorted(sources - tracked_paths)[:10]
                failures.append(
                    f"plan/source tree mismatch: unplanned_tracked={missing}, nontracked_sources={extra}"
                )
    result = {
        "schema_version": "eiel_document_publication_plan_validation_v1",
        "ok": not failures,
        "failure_count": len(failures),
        "failures": failures[:200],
        "document_count": len(entries),
        "repo_move_count": repo_count,
        "runtime_relocation_count": runtime_count,
        "candidate_digest": _candidate_digest(entries),
        "runtime_root": str(runtime_root),
        "physical_documents_modified": False,
    }
    if failures:
        raise LibrarianError("publication plan validation failed: " + "; ".join(failures[:5]))
    return result


def prepare_publication_candidate(
    repo_root: Path,
    plan_path: Path,
    candidate_root: Path,
    runtime_stage_root: Path,
    *,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    expected_head: str | None = None,
    expected_document_count: int | None = None,
    require_clean_docs: bool = True,
    require_complete_docs_tree: bool = True,
) -> dict[str, Any]:
    """Copy a complete publication candidate without changing source files."""

    validation = validate_publication_plan(
        repo_root,
        plan_path,
        runtime_root=runtime_root,
        expected_head=expected_head,
        expected_document_count=expected_document_count,
        require_clean_docs=require_clean_docs,
        require_complete_docs_tree=require_complete_docs_tree,
    )
    candidate_root = candidate_root.resolve()
    runtime_stage_root = runtime_stage_root.resolve()
    for path, label in ((candidate_root, "candidate root"), (runtime_stage_root, "runtime stage root")):
        if path.exists():
            raise LibrarianError(f"{label} already exists: {path}")
    candidate_docs = candidate_root / "docs"
    candidate_docs.mkdir(parents=True)
    runtime_stage_root.mkdir(parents=True)
    plan = _load_json(plan_path)
    copied_bytes = 0
    for entry in plan["entries"]:
        source, destination_rel = _entry_paths(repo_root.resolve(), entry, runtime_root.resolve())
        if entry["action"] == "repo_move":
            destination = candidate_root / Path(str(entry["proposed_destination"]))
            _resolved_under(candidate_docs, destination, label="candidate repo destination")
        else:
            destination = runtime_stage_root / Path(destination_rel.as_posix())
            _resolved_under(runtime_stage_root, destination, label="candidate runtime destination")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise LibrarianError(f"candidate destination already exists: {destination}")
        shutil.copy2(source, destination)
        copied_bytes += destination.stat().st_size
        if destination.stat().st_size != int(entry["source_size_bytes"]):
            raise LibrarianError(f"candidate size verification failed: {entry['document_id']}")
        if file_sha256(destination) != entry["source_sha256"]:
            raise LibrarianError(f"candidate SHA-256 verification failed: {entry['document_id']}")
    verified = verify_publication_candidate(
        repo_root,
        plan_path,
        candidate_root,
        runtime_stage_root,
        runtime_root=runtime_root,
    )
    return {
        "schema_version": "eiel_document_publication_candidate_receipt_v1",
        "ok": True,
        "candidate_root": str(candidate_root),
        "runtime_stage_root": str(runtime_stage_root),
        "copied_bytes": copied_bytes,
        "plan_validation": validation,
        "candidate_verification": verified,
        "physical_documents_modified": False,
    }


def verify_publication_candidate(
    repo_root: Path,
    plan_path: Path,
    candidate_root: Path,
    runtime_stage_root: Path,
    *,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    """Verify exact candidate membership, size, and content identity."""

    repo_root = repo_root.resolve()
    candidate_root = candidate_root.resolve()
    runtime_stage_root = runtime_stage_root.resolve()
    runtime_root = runtime_root.resolve()
    plan = _load_json(plan_path)
    expected_repo: set[str] = set()
    expected_runtime: set[str] = set()
    failures: list[str] = []
    verified_bytes = 0
    for entry in plan.get("entries") or []:
        _source, destination_rel = _entry_paths(repo_root, entry, runtime_root)
        if entry["action"] == "repo_move":
            destination = candidate_root / Path(str(entry["proposed_destination"]))
            expected_repo.add(destination.relative_to(candidate_root / "docs").as_posix())
        else:
            destination = runtime_stage_root / Path(destination_rel.as_posix())
            expected_runtime.add(destination.relative_to(runtime_stage_root).as_posix())
        if not destination.is_file() or destination.is_symlink():
            failures.append(f"candidate file missing or invalid: {destination}")
            continue
        verified_bytes += destination.stat().st_size
        if destination.stat().st_size != int(entry["source_size_bytes"]):
            failures.append(f"candidate size mismatch: {entry['document_id']}")
            continue
        if file_sha256(destination) != entry["source_sha256"]:
            failures.append(f"candidate SHA-256 mismatch: {entry['document_id']}")
    candidate_docs = candidate_root / "docs"
    actual_repo = {
        path.relative_to(candidate_docs).as_posix()
        for path in candidate_docs.rglob("*")
        if path.is_file()
    } if candidate_docs.is_dir() else set()
    actual_runtime = {
        path.relative_to(runtime_stage_root).as_posix()
        for path in runtime_stage_root.rglob("*")
        if path.is_file()
    } if runtime_stage_root.is_dir() else set()
    if actual_repo != expected_repo:
        failures.append(
            f"candidate repo membership mismatch: missing={sorted(expected_repo - actual_repo)[:10]}, "
            f"extra={sorted(actual_repo - expected_repo)[:10]}"
        )
    if actual_runtime != expected_runtime:
        failures.append(
            f"candidate runtime membership mismatch: missing={sorted(expected_runtime - actual_runtime)[:10]}, "
            f"extra={sorted(actual_runtime - expected_runtime)[:10]}"
        )
    result = {
        "schema_version": "eiel_document_publication_candidate_verification_v1",
        "ok": not failures,
        "failure_count": len(failures),
        "failures": failures[:200],
        "candidate_digest": _candidate_digest(list(plan.get("entries") or [])),
        "repo_file_count": len(actual_repo),
        "runtime_file_count": len(actual_runtime),
        "verified_bytes": verified_bytes,
        "physical_documents_modified": False,
    }
    if failures:
        raise LibrarianError("publication candidate verification failed: " + "; ".join(failures[:5]))
    return result


def _verify_backup(backup_zip: Path, expected_member_count: int) -> dict[str, Any]:
    if not backup_zip.is_file():
        raise LibrarianError(f"publication backup is missing: {backup_zip}")
    with zipfile.ZipFile(backup_zip) as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise LibrarianError(f"publication backup contains a corrupt member: {corrupt}")
        member_count = sum(not info.is_dir() for info in archive.infolist())
    if member_count != expected_member_count:
        raise LibrarianError(
            f"publication backup member count mismatch: expected {expected_member_count}, observed {member_count}"
        )
    return {
        "path": str(backup_zip.resolve()),
        "sha256": file_sha256(backup_zip),
        "member_count": member_count,
        "zip_integrity": "verified",
    }


def execute_publication(
    repo_root: Path,
    plan_path: Path,
    candidate_root: Path,
    runtime_stage_root: Path,
    old_docs_root: Path,
    backup_zip: Path,
    *,
    authorization: str,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    expected_head: str,
    expected_document_count: int,
) -> dict[str, Any]:
    """Publish a verified candidate using guarded directory swaps.

    The pre-publication docs directory is retained at ``old_docs_root`` until a
    later closeout step verifies the full repository and explicitly removes it.
    """

    if authorization != PUBLICATION_AUTHORIZATION:
        raise LibrarianError("physical publication authorization token is missing or incorrect")
    repo_root = repo_root.resolve()
    candidate_root = candidate_root.resolve()
    runtime_stage_root = runtime_stage_root.resolve()
    runtime_root = runtime_root.resolve()
    old_docs_root = old_docs_root.resolve()
    plan = _load_json(plan_path)
    validation = validate_publication_plan(
        repo_root,
        plan_path,
        runtime_root=runtime_root,
        expected_head=expected_head,
        expected_document_count=expected_document_count,
        require_clean_docs=True,
        require_complete_docs_tree=True,
    )
    candidate = verify_publication_candidate(
        repo_root,
        plan_path,
        candidate_root,
        runtime_stage_root,
        runtime_root=runtime_root,
    )
    backup = _verify_backup(backup_zip.resolve(), expected_document_count)
    published_docs = repo_root / "docs"
    candidate_docs = candidate_root / "docs"
    if old_docs_root.exists():
        raise LibrarianError(f"pre-publication docs retention path already exists: {old_docs_root}")
    if runtime_root.exists():
        raise LibrarianError(f"runtime publication root already exists: {runtime_root}")
    if published_docs.stat().st_dev != candidate_docs.stat().st_dev:
        raise LibrarianError("docs candidate is not on the same filesystem as the published docs tree")
    runtime_root.parent.mkdir(parents=True, exist_ok=True)
    if runtime_stage_root.stat().st_dev != runtime_root.parent.stat().st_dev:
        raise LibrarianError("runtime staging root is not on the same filesystem as its publication root")

    os.replace(runtime_stage_root, runtime_root)
    os.replace(published_docs, old_docs_root)
    try:
        os.replace(candidate_docs, published_docs)
    except BaseException:
        os.replace(old_docs_root, published_docs)
        raise

    repo_members = [path for path in published_docs.rglob("*") if path.is_file()]
    runtime_members = [path for path in runtime_root.rglob("*") if path.is_file()]
    if len(repo_members) != int(plan["repo_move_count"]):
        raise LibrarianError("published docs membership count differs from the approved plan")
    if len(runtime_members) != int(plan["runtime_relocation_count"]):
        raise LibrarianError("published runtime membership count differs from the approved plan")
    for entry in plan["entries"]:
        if entry["action"] == "repo_move":
            destination = repo_root / str(entry["proposed_destination"])
        else:
            destination = Path(str(entry["proposed_destination"]))
        if not destination.is_file() or destination.stat().st_size != int(entry["source_size_bytes"]):
            raise LibrarianError(f"published destination missing or wrong size: {entry['document_id']}")
        if file_sha256(destination) != entry["source_sha256"]:
            raise LibrarianError(f"published destination hash mismatch: {entry['document_id']}")
    return {
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "ok": True,
        "authorization": authorization,
        "expected_head": expected_head,
        "candidate_digest": validation["candidate_digest"],
        "repo_document_count": int(plan["repo_move_count"]),
        "runtime_relocation_count": int(plan["runtime_relocation_count"]),
        "published_docs_root": str(published_docs),
        "published_runtime_root": str(runtime_root),
        "retained_prepublication_docs_root": str(old_docs_root),
        "backup": backup,
        "candidate_verification": candidate,
        "authority_effect": "none",
        "source_files_deleted": False,
        "publication_state": "candidate_published_pending_repository_closeout",
    }


def _document_semantic_projection(record: Mapping[str, Any]) -> dict[str, Any]:
    """Project fields that a path-only publication is forbidden to change."""

    return {
        "document_id": record["document_id"],
        "sha256": record["sha256"],
        "size_bytes": record["size_bytes"],
        "document_class": record["document_class"],
        "currentness_status": record["currentness_status"],
        "supersedes_document_ids": record.get("supersedes_document_ids", []),
        "superseded_by_document_ids": record.get("superseded_by_document_ids", []),
        "linked_evidence_ids": record.get("linked_evidence_ids", []),
        "linked_claim_ids": record.get("linked_claim_ids", []),
        "linked_authority_assertion_ids": record.get("linked_authority_assertion_ids", []),
        "linked_dossier_ids": record.get("linked_dossier_ids", []),
        "linked_raw_asset_ids": record.get("linked_raw_asset_ids", []),
        "linked_artifact_ids": record.get("linked_artifact_ids", []),
        "linked_issue_ids": record.get("linked_issue_ids", []),
    }


def _semantic_digest(records: list[Mapping[str, Any]]) -> str:
    projection = [
        _document_semantic_projection(record)
        for record in sorted(records, key=lambda row: str(row["document_id"]))
    ]
    return hashlib.sha256(canonical_json_bytes(projection)).hexdigest()


def finalize_catalog_publication(
    repo_root: Path,
    source_root: Path,
    plan_path: Path,
    *,
    publication_id: str,
    prepublication_head: str,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    """Record current paths and path history after the physical swap."""

    from .documents import _relation, identity_freeze_manifest

    repo_root = repo_root.resolve()
    source_root = source_root.resolve()
    runtime_root = runtime_root.resolve()
    current_head = _git(repo_root, ["rev-parse", "HEAD"])
    if current_head.returncode != 0 or current_head.stdout.strip() != prepublication_head:
        raise LibrarianError(
            "catalog finalization HEAD mismatch: "
            f"expected {prepublication_head}, observed {current_head.stdout.strip() or 'unknown'}"
        )
    plan = _load_json(plan_path)
    entries = list(plan.get("entries") or [])
    entry_by_id = {str(row["document_id"]): row for row in entries}
    records = [
        _load_json(path)
        for path in sorted((source_root / "documents").glob("*.json"))
    ]
    if len(entry_by_id) != len(entries) or set(entry_by_id) != {
        str(row["document_id"]) for row in records
    }:
        raise LibrarianError("publication plan and maintained document IDs differ")
    semantic_digest_before = _semantic_digest(records)
    old_to_new: dict[str, str] = {}
    updated: list[dict[str, Any]] = []
    repo_count = 0
    runtime_count = 0
    for record in records:
        document_id = str(record["document_id"])
        entry = entry_by_id[document_id]
        old_path = str(entry["source_path"])
        if record["path"] != old_path:
            raise LibrarianError(
                f"maintained path differs from publication source for {document_id}: "
                f"{record['path']} != {old_path}"
            )
        new_path = str(entry["proposed_destination"])
        destination = repo_root / new_path if entry["action"] == "repo_move" else Path(new_path)
        if not destination.is_file() or destination.is_symlink():
            raise LibrarianError(f"published destination is missing or invalid: {document_id}")
        if destination.stat().st_size != int(entry["source_size_bytes"]):
            raise LibrarianError(f"published destination size differs: {document_id}")
        if file_sha256(destination) != entry["source_sha256"]:
            raise LibrarianError(f"published destination SHA-256 differs: {document_id}")
        old_to_new[old_path] = new_path
        row = dict(record)
        identity = dict(row["identity"])
        path_history = list(identity.get("path_history") or [])
        path_history.append(
            {
                "publication_id": publication_id,
                "from_path": old_path,
                "to_path": new_path,
                "sha256": row["sha256"],
                "size_bytes": row["size_bytes"],
                "content_unchanged": True,
                "authority_effect": "none",
            }
        )
        identity["path_history"] = path_history
        row["identity"] = identity
        if row.get("canonical_reading_location") is not None:
            row["canonical_reading_location"] = new_path
        row["path"] = new_path
        row["git_state"] = "tracked" if entry["action"] == "repo_move" else "runtime_relocated"
        proposal = dict(row["proposed_publication"])
        proposal.update(
            {
                "physical_move_authorized": True,
                "publication_state": "published",
                "publication_id": publication_id,
            }
        )
        row["proposed_publication"] = proposal
        row["publication"] = {
            "publication_id": publication_id,
            "action": entry["action"],
            "path_before_publication": old_path,
            "current_path": new_path,
            "content_unchanged": True,
            "authority_effect": "none",
            "prepublication_head": prepublication_head,
        }
        source_state = dict(row["source_state"])
        source_state.update(
            {
                "state": "archive_publication_v1",
                "observed_at_commit": prepublication_head,
                "source_digest": plan.get("candidate_digest") or _candidate_digest(entries),
                "publication_id": publication_id,
            }
        )
        row["source_state"] = source_state
        updated.append(row)
        if entry["action"] == "repo_move":
            repo_count += 1
        else:
            runtime_count += 1
    semantic_digest_after = _semantic_digest(updated)
    if semantic_digest_after != semantic_digest_before:
        raise LibrarianError("path-only publication changed document semantic projection")

    relation_paths = sorted((source_root / "document_relations").glob("*.json"))
    relations = {_load_json(path)["relation_id"]: _load_json(path) for path in relation_paths}
    baseline_relation_ids = sorted(relations)
    for record in updated:
        relation = _relation(
            "path_changed",
            str(record["document_id"]),
            target_type="path",
            target_id_or_path=str(record["path"]),
            basis=["operator_governance_decision", "path_history"],
            review_state="accepted",
            notes="Archive Publication 1.0 path transition; content and authority semantics are unchanged.",
        )
        relations[str(relation["relation_id"])] = relation

    document_issues: list[dict[str, Any]] = []
    for issue_path in sorted((source_root / "issues").glob("*.json")):
        issue = _load_json(issue_path)
        if issue.get("owner_collection") != "documents":
            continue
        row = dict(issue)
        prior_paths = list(row.get("document_paths") or row.get("paths") or [])
        row["document_paths_at_detection"] = prior_paths
        row["document_paths"] = [old_to_new.get(str(path), str(path)) for path in prior_paths]
        row["paths"] = [old_to_new.get(str(path), str(path)) for path in row.get("paths", prior_paths)]
        row["last_observed"] = publication_id
        if row.get("issue_kind") == "probable_runtime_output_in_docs":
            row["review_state"] = "resolved"
            row["remediation_state"] = "resolved"
            row["disposition"] = "resolved_by_archive_publication_v1"
        document_issues.append(row)

    expected_document_names = {f"{row['document_id']}.json" for row in updated}
    for path in (source_root / "documents").glob("*.json"):
        if path.name not in expected_document_names:
            raise LibrarianError(f"unexpected maintained document shard during finalization: {path}")
    for row in updated:
        write_json(source_root / "documents" / f"{row['document_id']}.json", row, pretty=False)
    expected_relation_names = {f"{row['relation_id']}.json" for row in relations.values()}
    for path in (source_root / "document_relations").glob("*.json"):
        if path.name not in expected_relation_names:
            path.unlink()
    for row in relations.values():
        write_json(source_root / "document_relations" / f"{row['relation_id']}.json", row, pretty=False)
    expected_issue_names = {f"{row['issue_id']}.json" for row in document_issues}
    for path in (source_root / "issues").glob("eiel-issue-docs-v1-*.json"):
        if path.name not in expected_issue_names:
            path.unlink()
    for row in document_issues:
        write_json(source_root / "issues" / f"{row['issue_id']}.json", row, pretty=False)

    freeze = identity_freeze_manifest(updated)
    write_json(source_root / "manifests/DOCUMENT_IDENTITY_FREEZE.json", freeze, pretty=False)
    redirects = {
        "schema_version": "eiel_document_legacy_path_redirects_v1",
        "publication_id": publication_id,
        "path_is_identity": False,
        "authority_effect": "none",
        "entry_count": len(entries),
        "entries": [
            {
                "document_id": entry["document_id"],
                "old_path": entry["source_path"],
                "current_path": entry["proposed_destination"],
                "action": entry["action"],
                "sha256": entry["source_sha256"],
                "size_bytes": entry["source_size_bytes"],
            }
            for entry in sorted(entries, key=lambda row: str(row["document_id"]))
        ],
    }
    final_manifest = {
        "schema_version": "eiel_document_archive_publication_manifest_v1",
        "publication_id": publication_id,
        "prepublication_head": prepublication_head,
        "document_count": len(updated),
        "repo_document_count": repo_count,
        "runtime_relocation_count": runtime_count,
        "baseline_relation_count": len(baseline_relation_ids),
        "path_changed_relation_count": len(updated),
        "semantic_projection_sha256_before": semantic_digest_before,
        "semantic_projection_sha256_after": semantic_digest_after,
        "semantic_equivalence": semantic_digest_before == semantic_digest_after,
        "candidate_digest": _candidate_digest(entries),
        "all_current_filenames_uid_prefixed": all(
            Path(str(row["path"])).name.startswith(f"{row['document_id']}--")
            for row in updated
        ),
        "authority_effect": "none",
        "publication_state": "published_pending_commit",
        "redirect_manifest": "manifests/DOCUMENT_LEGACY_PATH_REDIRECTS.json",
    }
    write_json(source_root / "manifests/DOCUMENT_LEGACY_PATH_REDIRECTS.json", redirects, pretty=False)
    write_json(source_root / "manifests/DOCUMENT_ARCHIVE_PUBLICATION_V1.json", final_manifest, pretty=False)
    policy_path = source_root / "policies/document_publication_namespace_v1.json"
    policy = _load_json(policy_path)
    policy["status"] = "archive_publication_v1_published"
    policy["published_publication_id"] = publication_id
    policy["proposed_destination_rule"]["physical_move_authorized"] = True
    policy["forbidden"] = [
        value for value in policy.get("forbidden", []) if value != "move_files_in_docs_v1"
    ]
    policy["future_ingress_rule"] = "new documents enter through incoming and the Knowledge Librarian publication workflow"
    write_json(policy_path, policy, pretty=True)
    return {
        **final_manifest,
        "redirect_entry_count": len(redirects["entries"]),
        "document_issue_count": len(document_issues),
        "document_relation_count": len(relations),
        "physical_content_modified": False,
    }


def _load_document_records(source_root: Path) -> list[dict[str, Any]]:
    return [
        _load_json(path)
        for path in sorted((source_root / "documents").glob("*.json"))
    ]


def _next_document_id(
    records: list[Mapping[str, Any]],
    *,
    reserved_document_ids: set[str] | None = None,
) -> tuple[str, int]:
    """Allocate the next stable document ID without reusing reserved IDs.

    The maintained catalog is authoritative, but post-publication ingress can
    encounter already-tracked incoming documents that carry manually reserved
    EIEL-DOC prefixes before they are catalogued.  Treat those IDs as reserved
    too so automatic accession never collides with a visible repo document.
    """

    seen = {str(row["document_id"]) for row in records}
    if reserved_document_ids:
        seen.update(str(value) for value in reserved_document_ids if value)
    if not seen:
        return "EIEL-DOC-000001", 1
    next_sequence = max(int(value.rsplit("-", 1)[1]) for value in seen if DOCUMENT_ID_PATTERN.fullmatch(value)) + 1
    while True:
        candidate = f"EIEL-DOC-{next_sequence:06d}"
        if candidate not in seen:
            return candidate, next_sequence
        next_sequence += 1


def _reserved_document_ids_from_docs_tree(repo_root: Path) -> set[str]:
    """Return EIEL-DOC prefixes already visible in the repo docs tree."""

    docs_root = (repo_root / "docs").resolve()
    if not docs_root.is_dir():
        return set()
    reserved: set[str] = set()
    prefix_re = re.compile(r"^(EIEL-DOC-[0-9]{6})--")
    for path in docs_root.rglob("EIEL-DOC-[0-9][0-9][0-9][0-9][0-9][0-9]--*"):
        if path.is_symlink():
            continue
        match = prefix_re.match(path.name)
        if match:
            reserved.add(match.group(1))
    return reserved


def _reserved_document_ids_from_incoming(repo_root: Path) -> set[str]:
    """Return preassigned IDs from the controlled incoming shelf only.

    Unregistered UID-prefixed files elsewhere in the published namespace are
    not reservations; they are collisions and must be caught by the normal
    destination guards.
    """

    incoming = (repo_root / "docs/900_lifecycle/940_incoming").resolve()
    if not incoming.is_dir():
        return set()
    reserved: set[str] = set()
    prefix_re = re.compile(r"^(EIEL-DOC-[0-9]{6})--")
    for path in incoming.glob("EIEL-DOC-[0-9][0-9][0-9][0-9][0-9][0-9]--*"):
        if path.is_symlink() or not path.is_file():
            continue
        match = prefix_re.match(path.name)
        if match:
            reserved.add(match.group(1))
    return reserved


def _preassigned_document_id_from_source(repo_root: Path, source_abs: Path) -> str | None:
    """Return a uniquely reserved EIEL-DOC ID carried by the source filename.

    Human and operator workflows sometimes reserve a durable ID before the
    catalog record is written.  Adoption is allowed only when exactly one
    non-symlink document in the repository docs tree carries that prefix and it
    is the source being registered.  This prevents a convenient prefix from
    becoming an authority-smuggling or collision mechanism.
    """

    match = re.match(r"^(EIEL-DOC-[0-9]{6})--", source_abs.name)
    if not match:
        return None
    document_id = match.group(1)
    docs_root = (repo_root / "docs").resolve()
    matches = [
        path.resolve()
        for path in docs_root.rglob(f"{document_id}--*")
        if path.is_file() and not path.is_symlink()
    ]
    unique = sorted(set(matches))
    if unique != [source_abs.resolve()]:
        rendered = ", ".join(str(path) for path in unique) or "none"
        raise LibrarianError(
            f"preassigned document ID is not uniquely reserved by the ingress source: "
            f"{document_id}; matches={rendered}"
        )
    return document_id


def _validate_destination_shelf(destination_shelf: str) -> PurePosixPath:
    shelf = PurePosixPath(destination_shelf)
    if shelf.is_absolute() or not shelf.parts or ".." in shelf.parts:
        raise LibrarianError(f"invalid destination shelf: {destination_shelf}")
    if shelf.parts[0] not in ALLOWED_DOC_SHELVES:
        raise LibrarianError(
            f"destination shelf is outside the publication namespace: {destination_shelf}"
        )
    if not re.fullmatch(r"[0-9]{3}_.+", shelf.parts[0]):
        raise LibrarianError(
            f"destination shelf must begin with a numbered top-level shelf: {destination_shelf}"
        )
    return shelf


def _registered_paths(records: list[Mapping[str, Any]]) -> set[str]:
    paths: set[str] = set()
    for row in records:
        for value in (
            row.get("path"),
            row.get("canonical_reading_location"),
            row.get("identity", {}).get("path_at_freeze"),
        ):
            if value:
                paths.add(str(value))
        for event in row.get("identity", {}).get("path_history", []) or []:
            for field in ("from_path", "to_path"):
                value = event.get(field)
                if value:
                    paths.add(str(value))
    return paths


def _load_redirect_manifest(source_root: Path) -> dict[str, Any]:
    path = source_root / "manifests/DOCUMENT_LEGACY_PATH_REDIRECTS.json"
    if path.is_file():
        manifest = _load_json(path)
        manifest["entries"] = list(manifest.get("entries") or [])
        return manifest
    return {
        "schema_version": "eiel_document_legacy_path_redirects_v1",
        "publication_id": None,
        "path_is_identity": False,
        "authority_effect": "none",
        "entry_count": 0,
        "entries": [],
    }


def _write_identity_policy_after_ingress(
    source_root: Path,
    records: list[Mapping[str, Any]],
    *,
    next_sequence: int,
) -> None:
    policy_path = source_root / "policies/document_identity_policy_v1.json"
    if not policy_path.is_file():
        return
    policy = _load_json(policy_path)
    sorted_records = sorted(records, key=lambda row: str(row["document_id"]))
    policy["current_freeze_range"] = {
        "count": len(sorted_records),
        "first_id": sorted_records[0]["document_id"] if sorted_records else None,
        "last_id": sorted_records[-1]["document_id"] if sorted_records else None,
        "status": "post_publication_ingress_active",
    }
    allocation = dict(policy.get("allocation") or {})
    allocation["next_sequence_after_current_freeze"] = next_sequence + 1
    policy["allocation"] = allocation
    policy["post_publication_ingress_rule"] = (
        "new documents are assigned the next stable EIEL-DOC ID, physically "
        "published with a UID-prefixed filename, and recorded without rebuilding "
        "existing path_changed or content_revision relations"
    )
    policy["path_is_identity"] = False
    policy["id_reuse_policy"] = "never_reuse"
    write_json(policy_path, policy, pretty=True)


def _write_publication_policy_after_ingress(source_root: Path) -> None:
    policy_path = source_root / "policies/document_publication_namespace_v1.json"
    if not policy_path.is_file():
        return
    policy = _load_json(policy_path)
    policy["status"] = "archive_publication_v1_published_post_publication_ingress_active"
    policy["future_ingress_rule"] = (
        "new documents enter through incoming and the Knowledge Librarian "
        "post-publication document ingress workflow"
    )
    proposed_rule = dict(policy.get("proposed_destination_rule") or {})
    proposed_rule["physical_move_authorized"] = True
    policy["proposed_destination_rule"] = proposed_rule
    policy["path_is_identity"] = False
    write_json(policy_path, policy, pretty=True)


def _write_archive_manifest_after_ingress(
    source_root: Path,
    records: list[Mapping[str, Any]],
    *,
    document_id: str,
    publication_id: str,
) -> None:
    path = source_root / "manifests/DOCUMENT_ARCHIVE_PUBLICATION_V1.json"
    manifest = _load_json(path) if path.is_file() else {
        "schema_version": "eiel_document_archive_publication_manifest_v1",
        "publication_state": "published_pending_commit",
        "authority_effect": "none",
    }
    manifest["document_count"] = len(records)
    manifest["repo_document_count"] = sum(
        str(row.get("path") or "").startswith("docs/") for row in records
    )
    manifest["path_changed_relation_count"] = sum(
        _load_json(path).get("relation_type") == "path_changed"
        for path in (source_root / "document_relations").glob("*.json")
    )
    manifest["post_publication_ingress_count"] = (
        int(manifest.get("post_publication_ingress_count") or 0) + 1
    )
    manifest["latest_ingress_document_id"] = document_id
    manifest["latest_ingress_publication_id"] = publication_id
    manifest["authority_effect"] = "none"
    manifest["publication_state"] = "published_pending_commit"
    manifest["redirect_manifest"] = "manifests/DOCUMENT_LEGACY_PATH_REDIRECTS.json"
    write_json(path, manifest, pretty=False)


def register_and_publish_repo_document(
    *,
    repo_root: Path,
    source_root: Path,
    path: str,
    title: str,
    document_class: str,
    destination_shelf: str,
    observed_commit: str,
    linked_dossier_ids: list[str] | tuple[str, ...] = (),
    reading_copy_of: str | None = None,
    publication_id: str = "post-publication-document-ingress-v1",
) -> dict[str, Any]:
    """Accession and physically publish one already tracked repo document.

    This is intentionally separate from the pre-publication registration helper:
    it appends one document record and one path relation, preserving all
    existing relations and redirects exactly instead of rebuilding them.
    """

    from .documents import _relation, identity_freeze_manifest, uid_filename

    repo_root = repo_root.resolve()
    source_root = source_root.resolve()
    source_rel = _relative_docs_path(path, label="document path")
    source_abs = _resolved_under(
        repo_root / "docs",
        repo_root / Path(source_rel.as_posix()),
        label="document path",
    )
    if not source_abs.is_file() or source_abs.is_symlink():
        raise LibrarianError(f"post-publication ingress source is missing or invalid: {path}")

    head = _git(repo_root, ["rev-parse", "HEAD"])
    if head.returncode != 0 or head.stdout.strip() != observed_commit:
        raise LibrarianError(
            "post-publication ingress HEAD mismatch: "
            f"expected {observed_commit}, observed {head.stdout.strip() or 'unknown'}"
        )
    tracked = _git(
        repo_root,
        ["ls-files", "-z", "--error-unmatch", "--", source_rel.as_posix()],
    )
    if tracked.returncode != 0 or tracked.stdout.removesuffix("\0") != source_rel.as_posix():
        raise LibrarianError(
            f"post-publication ingress source is not tracked exactly: {source_rel.as_posix()}"
        )

    shelf = _validate_destination_shelf(destination_shelf)
    records = _load_document_records(source_root)
    document_schema_path = source_root / "schemas/document_record_v1.schema.json"
    if document_schema_path.is_file():
        document_schema = _load_json(document_schema_path)
        allowed_document_classes = set(
            document_schema.get("properties", {})
            .get("document_class", {})
            .get("enum", [])
        )
        if allowed_document_classes and document_class not in allowed_document_classes:
            raise LibrarianError(
                "document class is outside the controlled vocabulary: "
                f"{document_class}"
            )
    existing_document_ids = {str(row["document_id"]) for row in records}
    reserved_document_ids = _reserved_document_ids_from_incoming(repo_root)
    preassigned_document_id = _preassigned_document_id_from_source(repo_root, source_abs)
    if preassigned_document_id is not None:
        if preassigned_document_id in existing_document_ids:
            raise LibrarianError(
                f"preassigned document ID is already registered: {preassigned_document_id}"
            )
        document_id = preassigned_document_id
        known_ids = existing_document_ids | reserved_document_ids
        next_sequence = max(
            int(value.rsplit("-", 1)[1])
            for value in known_ids
            if DOCUMENT_ID_PATTERN.fullmatch(value)
        )
    else:
        document_id, next_sequence = _next_document_id(
            records, reserved_document_ids=reserved_document_ids
        )
    if not DOCUMENT_ID_PATTERN.fullmatch(document_id):
        raise LibrarianError(f"allocated invalid document ID: {document_id}")
    if reading_copy_of is not None:
        if not DOCUMENT_ID_PATTERN.fullmatch(reading_copy_of):
            raise LibrarianError(f"invalid reading-copy target document ID: {reading_copy_of}")
        if reading_copy_of not in existing_document_ids:
            raise LibrarianError(f"reading-copy target document is missing: {reading_copy_of}")
        if reading_copy_of == document_id:
            raise LibrarianError("document may not be a reading copy of itself")

    existing_paths = _registered_paths(records)
    if source_rel.as_posix() in existing_paths:
        raise LibrarianError(f"document path is already registered: {source_rel.as_posix()}")

    digest = file_sha256(source_abs)
    size_bytes = source_abs.stat().st_size
    destination_filename = (
        source_rel.name
        if preassigned_document_id is not None
        else uid_filename(document_id, source_rel.as_posix())
    )
    destination_rel = PurePosixPath("docs") / shelf / destination_filename
    destination = destination_rel.as_posix()
    destination_abs = _resolved_under(
        repo_root / "docs",
        repo_root / Path(destination),
        label="document destination",
    )
    if destination in existing_paths:
        raise LibrarianError(f"document destination is already registered: {destination}")
    if destination_abs.exists():
        raise LibrarianError(f"document destination already exists: {destination}")

    redirects = _load_redirect_manifest(source_root)
    redirect_entries = list(redirects.get("entries") or [])
    redirect_paths = {
        str(row.get(field) or "")
        for row in redirect_entries
        for field in ("old_path", "current_path")
        if row.get(field)
    }
    if source_rel.as_posix() in redirect_paths or destination in redirect_paths:
        raise LibrarianError("document source or destination already appears in legacy redirects")

    path_relation = _relation(
        "path_changed",
        document_id,
        target_type="path",
        target_id_or_path=destination,
        basis=["operator_governance_decision", "path_history"],
        review_state="accepted",
        notes="Post-publication document ingress path transition; content and authority semantics are unchanged.",
    )
    relations = [path_relation]
    if reading_copy_of is not None:
        relations.append(
            _relation(
                "reading_copy_of",
                document_id,
                target_type="document",
                target_id_or_path=reading_copy_of,
                basis=["operator_governance_decision", "explicit_document_declaration"],
                review_state="accepted",
                notes="Operator-supplied reading-copy relation; it does not change authority or currentness.",
            )
        )
    relation_paths = [
        source_root / "document_relations" / f"{relation['relation_id']}.json"
        for relation in relations
    ]
    for relation, relation_path in zip(relations, relation_paths, strict=True):
        if relation_path.exists():
            raise LibrarianError(f"document relation already exists: {relation['relation_id']}")

    dossiers = sorted(set(str(value) for value in linked_dossier_ids))
    path_event = {
        "publication_id": publication_id,
        "from_path": source_rel.as_posix(),
        "to_path": destination,
        "sha256": digest,
        "size_bytes": size_bytes,
        "content_unchanged": True,
        "authority_effect": "none",
    }
    record = {
        "schema_version": "eiel_document_record_v1",
        "collection": "documents",
        "document_id": document_id,
        "title": title,
        "path": destination,
        "canonical_reading_location": destination,
        "sha256": digest,
        "size_bytes": size_bytes,
        "git_state": "tracked",
        "document_class": document_class,
        "document_class_candidates": [document_class],
        "refined_lifecycle_candidate": "post_publication_explicit_ingress",
        "currentness_status": "unreviewed",
        "currentness_candidate": "explicit_ingress_review_required",
        "authored_or_generated": "candidate",
        "authorship_candidate": "authored_candidate",
        "retention_class": "candidate",
        "retention_candidate": "repo_publication",
        "review_state": "candidate",
        "review_flags": ["post_publication_explicit_ingress"],
        "supersedes_document_ids": [],
        "superseded_by_document_ids": [],
        "linked_evidence_ids": [],
        "linked_claim_ids": [],
        "linked_authority_assertion_ids": [],
        "linked_dossier_ids": dossiers,
        "linked_raw_asset_ids": [],
        "linked_artifact_ids": [],
        "linked_issue_ids": [],
        "source_state": {
            "state": "post_publication_explicit_ingress",
            "observed_at_commit": observed_commit,
            "source_digest": digest,
            "publication_id": publication_id,
        },
        "identity": {
            "frozen": True,
            "path_at_freeze": source_rel.as_posix(),
            "sha256_at_freeze": digest,
            "size_bytes_at_freeze": size_bytes,
            "freeze_commit": observed_commit,
            "freeze_source_digest": digest,
            "freeze_basis": ["explicit_post_publication_ingress"],
            "reuse_policy": "never_reuse",
            "path_is_identity": False,
            "path_history": [path_event],
            "content_revisions": [],
        },
        "proposed_publication": {
            "proposal_version": "eiel_docs_publication_namespace_v1",
            "destination_class": "repo_document",
            "proposed_destination": destination,
            "basis": ["explicit_post_publication_ingress", "document_id_filename_prefix"],
            "confidence": "candidate",
            "authority_effect": "none",
            "physical_move_authorized": True,
            "publication_state": "published",
            "publication_id": publication_id,
            "requires_operator_review": False,
        },
        "publication": {
            "publication_id": publication_id,
            "action": "repo_move",
            "path_before_publication": source_rel.as_posix(),
            "current_path": destination,
            "content_unchanged": True,
            "authority_effect": "none",
            "prepublication_head": observed_commit,
        },
        "notes": (
            "Explicitly accessioned after Archive Publication 1.0 and physically "
            "published through the Knowledge Librarian; no authority or currentness is inferred."
        ),
    }

    updated_records = sorted([*records, record], key=lambda row: str(row["document_id"]))
    destination_abs.parent.mkdir(parents=True, exist_ok=True)
    moved = False
    try:
        os.replace(source_abs, destination_abs)
        moved = True
        if file_sha256(destination_abs) != digest or destination_abs.stat().st_size != size_bytes:
            raise LibrarianError("published document failed post-move content verification")
        write_json(source_root / "documents" / f"{document_id}.json", record, pretty=False)
        for relation, relation_path in zip(relations, relation_paths, strict=True):
            write_json(relation_path, relation, pretty=False)
        redirect_entries.append(
            {
                "document_id": document_id,
                "old_path": source_rel.as_posix(),
                "current_path": destination,
                "action": "repo_move",
                "sha256": digest,
                "size_bytes": size_bytes,
            }
        )
        redirects.update(
            {
                "latest_publication_id": publication_id,
                "path_is_identity": False,
                "authority_effect": "none",
                "entry_count": len(redirect_entries),
                "entries": sorted(
                    redirect_entries,
                    key=lambda row: (str(row["document_id"]), str(row["old_path"])),
                ),
            }
        )
        write_json(
            source_root / "manifests/DOCUMENT_LEGACY_PATH_REDIRECTS.json",
            redirects,
            pretty=False,
        )
        write_json(
            source_root / "manifests/DOCUMENT_IDENTITY_FREEZE.json",
            identity_freeze_manifest(updated_records),
            pretty=False,
        )
        _write_identity_policy_after_ingress(
            source_root,
            updated_records,
            next_sequence=next_sequence,
        )
        _write_publication_policy_after_ingress(source_root)
        _write_archive_manifest_after_ingress(
            source_root,
            updated_records,
            document_id=document_id,
            publication_id=publication_id,
        )
    except Exception:
        if moved and destination_abs.exists() and not source_abs.exists():
            source_abs.parent.mkdir(parents=True, exist_ok=True)
            os.replace(destination_abs, source_abs)
        raise

    return {
        "schema_version": "eiel_post_publication_document_ingress_receipt_v1",
        "document_id": document_id,
        "path_before_publication": source_rel.as_posix(),
        "current_path": destination,
        "sha256": digest,
        "size_bytes": size_bytes,
        "publication_id": publication_id,
        "document_count": len(updated_records),
        "redirect_entry_count_before": len(redirect_entries) - 1,
        "redirect_entry_count_after": len(redirect_entries),
        "existing_document_relations_preserved": True,
        "path_changed_relation_id": path_relation["relation_id"],
        "reading_copy_of": reading_copy_of,
        "appended_relation_ids": [relation["relation_id"] for relation in relations],
        "authority_effect": "none",
        "physical_document_moved": True,
    }


def resolve_document_locator(source_root: Path, identifier_or_path: str) -> dict[str, Any]:
    """Resolve a stable document ID, legacy path, or current path."""

    matches: list[dict[str, Any]] = []
    for shard in sorted((source_root.resolve() / "documents").glob("*.json")):
        row = _load_json(shard)
        legacy_paths = [str(row.get("identity", {}).get("path_at_freeze") or "")]
        legacy_paths.extend(
            str(item.get("from_path") or "")
            for item in row.get("identity", {}).get("path_history", [])
        )
        if identifier_or_path in {
            str(row["document_id"]),
            str(row["path"]),
            *legacy_paths,
        }:
            matches.append(row)
    if not matches:
        raise LibrarianError(f"document locator not found: {identifier_or_path}")
    if len(matches) > 1:
        raise LibrarianError(f"document locator is ambiguous: {identifier_or_path}")
    row = matches[0]
    return {
        "schema_version": "eiel_document_locator_resolution_v1",
        "query": identifier_or_path,
        "document_id": row["document_id"],
        "current_path": row["path"],
        "path_at_freeze": row["identity"]["path_at_freeze"],
        "sha256": row["sha256"],
        "size_bytes": row["size_bytes"],
        "publication": row.get("publication"),
        "authority_effect": "none",
    }


def rewrite_repository_path_references(
    repo_root: Path,
    redirect_manifest_path: Path,
    *,
    authorization: str | None = None,
) -> dict[str, Any]:
    """Plan or apply exact legacy-path replacements in active tracked text files.

    Published documents, historical evidence, and maintained librarian metadata
    are excluded.  Their legacy locators remain available through the redirect
    manifest and ``resolve-document``.
    """

    repo_root = repo_root.resolve()
    redirect_manifest = _load_json(redirect_manifest_path)
    redirects = {
        str(row["old_path"]): str(row["current_path"])
        for row in redirect_manifest.get("entries") or []
    }
    if len(redirects) != int(redirect_manifest.get("entry_count") or -1):
        raise LibrarianError("redirect manifest count or path uniqueness is invalid")
    tracked = _git(repo_root, ["ls-files", "-z"])
    if tracked.returncode != 0:
        raise LibrarianError(tracked.stderr.strip() or "unable to enumerate tracked files")
    excluded_prefixes = (
        "docs/",
        "evidences/",
        "tools/evidence/librarian/catalog_source/",
    )
    changed: list[dict[str, Any]] = []
    skipped_binary = 0
    skipped_large = 0
    apply_changes = authorization is not None
    if apply_changes and authorization != REFERENCE_REWRITE_AUTHORIZATION:
        raise LibrarianError("path-reference rewrite authorization token is incorrect")
    for rel_path in (row for row in tracked.stdout.split("\0") if row):
        if rel_path.startswith(excluded_prefixes):
            continue
        path = repo_root / rel_path
        if not path.is_file() or path.is_symlink():
            continue
        if path.stat().st_size > 10_000_000:
            skipped_large += 1
            continue
        raw = path.read_bytes()
        if b"\0" in raw:
            skipped_binary += 1
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            skipped_binary += 1
            continue
        replacements = [
            (old_path, new_path, text.count(old_path))
            for old_path, new_path in redirects.items()
            if old_path in text
        ]
        if not replacements:
            continue
        revised = text
        for old_path, new_path, _count in sorted(
            replacements,
            key=lambda item: (-len(item[0]), item[0]),
        ):
            revised = revised.replace(old_path, new_path)
        if revised == text:
            continue
        if apply_changes:
            path.write_text(revised, encoding="utf-8")
        changed.append(
            {
                "path": rel_path,
                "replacement_count": sum(row[2] for row in replacements),
                "legacy_paths": sorted(row[0] for row in replacements),
                "sha256_before": hashlib.sha256(raw).hexdigest(),
                "sha256_after": hashlib.sha256(revised.encode("utf-8")).hexdigest(),
            }
        )
    return {
        "schema_version": "eiel_document_repository_path_reference_rewrite_v1",
        "mode": "execute" if apply_changes else "dry_run",
        "authorization": authorization,
        "redirect_manifest": str(redirect_manifest_path),
        "redirect_count": len(redirects),
        "changed_file_count": len(changed),
        "replacement_count": sum(row["replacement_count"] for row in changed),
        "changed_files": changed,
        "skipped_binary_file_count": skipped_binary,
        "skipped_large_file_count": skipped_large,
        "excluded_prefixes": list(excluded_prefixes),
        "authority_effect": "none",
        "historical_document_content_modified": False,
    }


def rewrite_published_document_references(
    repo_root: Path,
    source_root: Path,
    redirect_manifest_path: Path,
    document_ids: list[str],
    *,
    authorization: str | None = None,
) -> dict[str, Any]:
    """Rewrite legacy locators in explicitly selected living documents only.

    The broad repository rewrite intentionally excludes ``docs/`` so archived
    reports keep their original historical wording.  This narrower operation
    exists for living registers and governance documents whose references must
    follow Archive Publication 1.0.  Selection is by stable document ID, never
    by path inference.
    """

    repo_root = repo_root.resolve()
    source_root = source_root.resolve()
    redirect_manifest = _load_json(redirect_manifest_path)
    redirects = {
        str(row["old_path"]): str(row["current_path"])
        for row in redirect_manifest.get("entries") or []
    }
    if len(redirects) != int(redirect_manifest.get("entry_count") or -1):
        raise LibrarianError("redirect manifest count or path uniqueness is invalid")
    selected_ids = sorted(set(document_ids))
    if not selected_ids:
        raise LibrarianError("at least one --document-id is required")
    apply_changes = authorization is not None
    if apply_changes and authorization != DOCUMENT_REFERENCE_REWRITE_AUTHORIZATION:
        raise LibrarianError("document-reference rewrite authorization token is incorrect")

    changed: list[dict[str, Any]] = []
    unchanged: list[str] = []
    for document_id in selected_ids:
        if not DOCUMENT_ID_PATTERN.fullmatch(document_id):
            raise LibrarianError(f"invalid selected document ID: {document_id}")
        shard = source_root / "documents" / f"{document_id}.json"
        if not shard.is_file():
            raise LibrarianError(f"selected document record is missing: {document_id}")
        record = _load_json(shard)
        rel_path = _relative_docs_path(str(record.get("path") or ""), label="document path")
        path = _resolved_under(repo_root / "docs", repo_root / Path(rel_path.as_posix()), label="document path")
        if not path.is_file() or path.is_symlink():
            raise LibrarianError(f"selected published document is missing or invalid: {document_id}")
        raw = path.read_bytes()
        if b"\0" in raw:
            raise LibrarianError(f"selected published document is binary: {document_id}")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise LibrarianError(f"selected published document is not UTF-8: {document_id}") from exc
        replacements = [
            (old_path, new_path, text.count(old_path))
            for old_path, new_path in redirects.items()
            if old_path in text
        ]
        if not replacements:
            unchanged.append(document_id)
            continue
        revised = text
        for old_path, new_path, _count in sorted(
            replacements,
            key=lambda item: (-len(item[0]), item[0]),
        ):
            revised = revised.replace(old_path, new_path)
        if apply_changes:
            path.write_text(revised, encoding="utf-8")
        changed.append(
            {
                "document_id": document_id,
                "path": rel_path.as_posix(),
                "replacement_count": sum(row[2] for row in replacements),
                "legacy_paths": sorted(row[0] for row in replacements),
                "sha256_before": hashlib.sha256(raw).hexdigest(),
                "sha256_after": hashlib.sha256(revised.encode("utf-8")).hexdigest(),
            }
        )
    return {
        "schema_version": "eiel_document_selected_reference_rewrite_v1",
        "mode": "execute" if apply_changes else "dry_run",
        "authorization": authorization,
        "redirect_manifest": str(redirect_manifest_path),
        "redirect_count": len(redirects),
        "selected_document_ids": selected_ids,
        "changed_document_count": len(changed),
        "unchanged_document_ids": unchanged,
        "replacement_count": sum(row["replacement_count"] for row in changed),
        "changed_documents": changed,
        "authority_effect": "none",
        "selection_basis": "explicit_stable_document_ids",
    }


def record_document_content_revisions(
    repo_root: Path,
    source_root: Path,
    document_ids: list[str],
    *,
    reason: str,
    authorization: str,
) -> dict[str, Any]:
    """Reconcile deliberate edits to living documents with their stable IDs."""

    from .documents import _relation, identity_freeze_manifest

    if authorization != CONTENT_REVISION_AUTHORIZATION:
        raise LibrarianError("document content-revision authorization token is incorrect")
    if not reason.strip():
        raise LibrarianError("content revision reason may not be empty")
    repo_root = repo_root.resolve()
    source_root = source_root.resolve()
    selected_ids = sorted(set(document_ids))
    if not selected_ids:
        raise LibrarianError("at least one --document-id is required")
    head = _git(repo_root, ["rev-parse", "HEAD"])
    if head.returncode != 0:
        raise LibrarianError(head.stderr.strip() or "unable to resolve repository HEAD")
    observed_at_commit = head.stdout.strip()
    relation_paths = sorted((source_root / "document_relations").glob("*.json"))
    relations = {_load_json(path)["relation_id"]: _load_json(path) for path in relation_paths}
    revised_records: list[dict[str, Any]] = []
    unchanged_ids: list[str] = []
    revisions: list[dict[str, Any]] = []

    for document_id in selected_ids:
        if not DOCUMENT_ID_PATTERN.fullmatch(document_id):
            raise LibrarianError(f"invalid selected document ID: {document_id}")
        shard = source_root / "documents" / f"{document_id}.json"
        if not shard.is_file():
            raise LibrarianError(f"selected document record is missing: {document_id}")
        record = _load_json(shard)
        rel_path = _relative_docs_path(str(record.get("path") or ""), label="document path")
        path = _resolved_under(repo_root / "docs", repo_root / Path(rel_path.as_posix()), label="document path")
        if not path.is_file() or path.is_symlink():
            raise LibrarianError(f"selected published document is missing or invalid: {document_id}")
        current_sha256 = file_sha256(path)
        current_size = path.stat().st_size
        previous_sha256 = str(record["sha256"])
        previous_size = int(record["size_bytes"])
        if current_sha256 == previous_sha256 and current_size == previous_size:
            unchanged_ids.append(document_id)
            continue
        revision_identity = {
            "document_id": document_id,
            "path": rel_path.as_posix(),
            "from_sha256": previous_sha256,
            "to_sha256": current_sha256,
            "from_size_bytes": previous_size,
            "to_size_bytes": current_size,
            "reason": reason.strip(),
            "recorded_at_commit": observed_at_commit,
        }
        revision_id = "eiel-docrev-" + hashlib.sha256(
            canonical_json_bytes(revision_identity)
        ).hexdigest()[:20]
        revision = {
            "revision_id": revision_id,
            **revision_identity,
            "authority_effect": "none",
            "identity_effect": "none",
        }
        row = dict(record)
        identity = dict(row["identity"])
        content_revisions = list(identity.get("content_revisions") or [])
        content_revisions.append(revision)
        identity["content_revisions"] = content_revisions
        row["identity"] = identity
        row["sha256"] = current_sha256
        row["size_bytes"] = current_size
        source_state = dict(row["source_state"])
        source_state["state"] = "archive_publication_v1_postpublication_maintenance"
        source_state["latest_content_revision_id"] = revision_id
        row["source_state"] = source_state
        publication = dict(row.get("publication") or {})
        publication["post_publication_content_revision_count"] = len(content_revisions)
        publication["latest_content_revision_id"] = revision_id
        row["publication"] = publication
        relation = _relation(
            "content_revision",
            document_id,
            target_type="path",
            target_id_or_path=f"{rel_path.as_posix()}#{revision_id}",
            basis=["operator_governance_decision", "manual_review"],
            review_state="accepted",
            notes=(
                "Controlled post-publication maintenance revision; stable document "
                "identity and authority semantics are unchanged."
            ),
        )
        relations[str(relation["relation_id"])] = relation
        write_json(shard, row, pretty=False)
        revised_records.append(row)
        revisions.append(revision)

    all_records = [
        _load_json(path)
        for path in sorted((source_root / "documents").glob("*.json"))
    ]
    write_json(
        source_root / "manifests" / "DOCUMENT_IDENTITY_FREEZE.json",
        identity_freeze_manifest(all_records),
        pretty=False,
    )
    for row in relations.values():
        write_json(
            source_root / "document_relations" / f"{row['relation_id']}.json",
            row,
            pretty=False,
        )
    publication_manifest_path = source_root / "manifests" / "DOCUMENT_ARCHIVE_PUBLICATION_V1.json"
    if publication_manifest_path.is_file():
        publication_manifest = _load_json(publication_manifest_path)
        publication_manifest["post_publication_content_revision_count"] = sum(
            len(row.get("identity", {}).get("content_revisions") or [])
            for row in all_records
        )
        publication_manifest["content_revision_authority_effect"] = "none"
        write_json(publication_manifest_path, publication_manifest, pretty=False)
    return {
        "schema_version": "eiel_document_content_revision_receipt_v1",
        "authorization": authorization,
        "reason": reason.strip(),
        "recorded_at_commit": observed_at_commit,
        "selected_document_ids": selected_ids,
        "revised_document_count": len(revised_records),
        "unchanged_document_ids": unchanged_ids,
        "revisions": revisions,
        "authority_effect": "none",
        "identity_effect": "none",
    }


def verify_published_document_archive(
    repo_root: Path,
    source_root: Path,
    *,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
) -> dict[str, Any]:
    """Verify the complete current archive against maintained document records."""

    repo_root = repo_root.resolve()
    source_root = source_root.resolve()
    runtime_root = runtime_root.resolve()
    records = [
        _load_json(path)
        for path in sorted((source_root / "documents").glob("*.json"))
    ]
    failures: list[str] = []
    ids: set[str] = set()
    current_paths: set[str] = set()
    expected_repo_paths: set[str] = set()
    expected_runtime_paths: set[str] = set()
    verified_bytes = 0
    revised_document_count = 0

    for record in records:
        document_id = str(record.get("document_id") or "")
        current_path = str(record.get("path") or "")
        if document_id in ids:
            failures.append(f"duplicate document ID: {document_id}")
        if current_path in current_paths:
            failures.append(f"duplicate current document path: {current_path}")
        ids.add(document_id)
        current_paths.add(current_path)
        if not DOCUMENT_ID_PATTERN.fullmatch(document_id):
            failures.append(f"invalid document ID: {document_id}")
            continue
        if current_path.startswith("docs/"):
            try:
                rel = _relative_docs_path(current_path, label="current document path")
                destination = _resolved_under(
                    repo_root / "docs",
                    repo_root / Path(rel.as_posix()),
                    label="current document path",
                )
            except LibrarianError as exc:
                failures.append(str(exc))
                continue
            if len(rel.parts) < 3 or rel.parts[1] not in ALLOWED_DOC_SHELVES:
                failures.append(f"published document is outside numbered shelves: {document_id}")
            expected_repo_paths.add(current_path)
        else:
            try:
                _runtime_relative(current_path, runtime_root)
            except LibrarianError as exc:
                failures.append(str(exc))
                continue
            destination = Path(current_path)
            expected_runtime_paths.add(current_path)
        if Path(current_path).name.startswith(f"{document_id}--") is False:
            failures.append(f"current filename lacks stable UID prefix: {document_id}")
        if not destination.is_file() or destination.is_symlink():
            failures.append(f"current document is missing or invalid: {document_id}")
            continue
        actual_size = destination.stat().st_size
        actual_hash = file_sha256(destination)
        verified_bytes += actual_size
        recorded_size = record.get("size_bytes")
        if actual_size != (int(recorded_size) if recorded_size is not None else -1):
            failures.append(f"current document size differs from record: {document_id}")
        if actual_hash != str(record.get("sha256") or ""):
            failures.append(f"current document SHA-256 differs from record: {document_id}")
        revisions = list(record.get("identity", {}).get("content_revisions") or [])
        if revisions:
            revised_document_count += 1
            latest = revisions[-1]
            if latest.get("to_sha256") != record.get("sha256"):
                failures.append(f"latest content revision hash is stale: {document_id}")
            latest_size = latest.get("to_size_bytes")
            if (int(latest_size) if latest_size is not None else -1) != (
                int(recorded_size) if recorded_size is not None else -1
            ):
                failures.append(f"latest content revision size is stale: {document_id}")

    actual_repo_paths = {
        path.relative_to(repo_root).as_posix()
        for path in (repo_root / "docs").rglob("*")
        if path.is_file() and path != repo_root / "docs/000_knowledge-catalog.html"
    }
    actual_runtime_paths = {
        str(path.resolve())
        for path in runtime_root.rglob("*")
        if path.is_file()
    } if runtime_root.is_dir() else set()
    if actual_repo_paths != expected_repo_paths:
        failures.append(
            "repo document membership differs: "
            f"missing={sorted(expected_repo_paths - actual_repo_paths)[:10]}, "
            f"extra={sorted(actual_repo_paths - expected_repo_paths)[:10]}"
        )
    if actual_runtime_paths != expected_runtime_paths:
        failures.append(
            "runtime document membership differs: "
            f"missing={sorted(expected_runtime_paths - actual_runtime_paths)[:10]}, "
            f"extra={sorted(actual_runtime_paths - expected_runtime_paths)[:10]}"
        )

    redirect_path = source_root / "manifests" / "DOCUMENT_LEGACY_PATH_REDIRECTS.json"
    if not redirect_path.is_file():
        failures.append("legacy path redirect manifest is missing")
        redirects: list[dict[str, Any]] = []
    else:
        redirect_manifest = _load_json(redirect_path)
        redirects = list(redirect_manifest.get("entries") or [])
        if len(redirects) != len(records):
            failures.append("legacy path redirect count differs from document count")
        redirects_by_id = {str(row.get("document_id")): row for row in redirects}
        if len(redirects_by_id) != len(redirects):
            failures.append("legacy path redirect IDs are not unique")
        for record in records:
            document_id = str(record["document_id"])
            redirect = redirects_by_id.get(document_id)
            if redirect is None:
                failures.append(f"legacy path redirect is missing: {document_id}")
                continue
            if redirect.get("old_path") != record.get("identity", {}).get("path_at_freeze"):
                failures.append(f"legacy path redirect freeze path differs: {document_id}")
            if redirect.get("current_path") != record.get("path"):
                failures.append(f"legacy path redirect current path differs: {document_id}")

    top_level_shelves = sorted(
        path.name for path in (repo_root / "docs").iterdir() if path.is_dir()
    )
    if set(top_level_shelves) != ALLOWED_DOC_SHELVES:
        failures.append(
            "top-level shelf set differs: "
            f"expected={sorted(ALLOWED_DOC_SHELVES)}, observed={top_level_shelves}"
        )
    site_path = repo_root / "docs/000_knowledge-catalog.html"
    result = {
        "schema_version": "eiel_published_document_archive_verification_v1",
        "ok": not failures,
        "failure_count": len(failures),
        "failures": failures[:200],
        "document_count": len(records),
        "repo_document_count": len(expected_repo_paths),
        "runtime_document_count": len(expected_runtime_paths),
        "redirect_count": len(redirects),
        "content_revised_document_count": revised_document_count,
        "verified_bytes": verified_bytes,
        "top_level_shelves": top_level_shelves,
        "all_managed_filenames_uid_prefixed": not any(
            "lacks stable UID prefix" in failure for failure in failures
        ),
        "knowledge_site": {
            "path": "docs/000_knowledge-catalog.html",
            "present": site_path.is_file(),
            "sha256": file_sha256(site_path) if site_path.is_file() else None,
            "managed_document": False,
        },
        "runtime_root": str(runtime_root),
        "authority_effect": "none",
    }
    if failures:
        raise LibrarianError(
            "published document archive verification failed: "
            + "; ".join(failures[:5])
        )
    return result
