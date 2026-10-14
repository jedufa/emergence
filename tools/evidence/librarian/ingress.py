"""Human-facing external document ingress for the EIEL Knowledge Librarian."""

from __future__ import annotations

from engine.runtime_roots import public_runtime_path

import argparse
import json
import os
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from . import wp4
from .core import LibrarianError, file_sha256, write_json
from .documents import _relation, identity_freeze_manifest, uid_filename
from .publication import (
    DOCUMENT_ID_PATTERN,
    _load_document_records,
    _load_json,
    _load_redirect_manifest,
    _next_document_id,
    _registered_paths,
    _validate_destination_shelf,
    _write_archive_manifest_after_ingress,
    _write_identity_policy_after_ingress,
    _write_publication_policy_after_ingress,
)


DEFAULT_HUMAN_INGRESS_RUNTIME_ROOT = Path(
    public_runtime_path('evidence_raw/eiel_knowledge_librarian_human_ingress_v1')
)


@dataclass(frozen=True)
class IngressRequest:
    source: Path
    title: str
    document_class: str
    destination_shelf: str
    linked_dossier_ids: tuple[str, ...] = ()
    reading_copy_of: str | None = None
    publication_id: str = "human-ingress-v1"
    source_label: str | None = None


def _git(repo_root: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _repo_head(repo_root: Path) -> str:
    head = _git(repo_root, ["rev-parse", "HEAD"])
    if head.returncode != 0:
        raise LibrarianError(head.stderr.strip() or "unable to read repository HEAD")
    return head.stdout.strip()


def _require_branch(repo_root: Path) -> str:
    branch = _git(repo_root, ["symbolic-ref", "--quiet", "--short", "HEAD"])
    if branch.returncode != 0 or not branch.stdout.strip():
        raise LibrarianError("ingest commit/stage requires a non-detached Git branch")
    return branch.stdout.strip()


def _require_clean(repo_root: Path) -> None:
    status = _git(repo_root, ["status", "--porcelain=v1"])
    if status.returncode != 0:
        raise LibrarianError(status.stderr.strip() or "unable to inspect Git status")
    if status.stdout.strip():
        raise LibrarianError(
            "repository has pre-existing dirty changes; refusing to mix them with "
            "human document ingress"
        )


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _validate_external_source(repo_root: Path, source_root: Path, source: Path) -> Path:
    source_abs = source.expanduser().resolve()
    if not source_abs.is_file() or source_abs.is_symlink():
        raise LibrarianError(f"external source is missing, non-regular, or a symlink: {source}")
    managed_roots = [
        repo_root.resolve(),
        source_root.resolve(),
    ]
    if any(_inside(source_abs, root) for root in managed_roots):
        raise LibrarianError(
            "external ingress source must be outside managed repository and "
            f"catalog roots: {source_abs}"
        )
    return source_abs


def _allowed_document_classes(source_root: Path) -> set[str]:
    schema_path = source_root / "schemas/document_record_v1.schema.json"
    if not schema_path.is_file():
        return set()
    schema = _load_json(schema_path)
    return set(
        schema.get("properties", {})
        .get("document_class", {})
        .get("enum", [])
    )


def _duplicate_matches(records: list[Mapping[str, Any]], digest: str) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for row in records:
        identity = row.get("identity") or {}
        if digest in {str(row.get("sha256") or ""), str(identity.get("sha256_at_freeze") or "")}:
            matches.append(
                {
                    "document_id": str(row.get("document_id") or ""),
                    "path": str(row.get("path") or ""),
                    "title": str(row.get("title") or ""),
                    "sha256": str(row.get("sha256") or ""),
                }
            )
    return matches


def preview_external_document(
    *,
    repo_root: Path,
    source_root: Path,
    request: IngressRequest,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    source_root = source_root.resolve()
    source_abs = _validate_external_source(repo_root, source_root, request.source)
    shelf = _validate_destination_shelf(request.destination_shelf)
    records = _load_document_records(source_root)
    document_id, next_sequence = _next_document_id(records)
    if not DOCUMENT_ID_PATTERN.fullmatch(document_id):
        raise LibrarianError(f"allocated invalid document ID: {document_id}")

    allowed_classes = _allowed_document_classes(source_root)
    blockers: list[str] = []
    warnings: list[str] = []
    if allowed_classes and request.document_class not in allowed_classes:
        blockers.append(f"document class is outside the controlled vocabulary: {request.document_class}")

    existing_ids = {str(row.get("document_id") or "") for row in records}
    if request.reading_copy_of:
        if request.reading_copy_of not in existing_ids:
            blockers.append(f"reading-copy target document is missing: {request.reading_copy_of}")

    digest = file_sha256(source_abs)
    size_bytes = source_abs.stat().st_size
    exact_duplicates = _duplicate_matches(records, digest)
    if exact_duplicates:
        blockers.append("exact duplicate content already exists; no new EIEL-DOC ID may be consumed by default")

    source_label = request.source_label or source_abs.as_posix()
    destination = (Path("docs") / Path(shelf.as_posix()) / uid_filename(document_id, source_abs.name)).as_posix()
    existing_paths = _registered_paths(records)
    if destination in existing_paths:
        blockers.append(f"proposed destination is already registered: {destination}")
    if (repo_root / destination).exists():
        blockers.append(f"proposed destination already exists: {destination}")
    same_titles = [
        {"document_id": str(row.get("document_id")), "path": str(row.get("path"))}
        for row in records
        if str(row.get("title") or "").casefold() == request.title.casefold()
    ]
    if same_titles:
        warnings.append("same title already exists; this is not authority and requires human review if intentional")

    return {
        "schema_version": "eiel_human_document_ingress_preview_v1",
        "ok": not blockers,
        "mode": "dry-run",
        "document_id": document_id,
        "next_sequence_if_committed": next_sequence,
        "title": request.title,
        "document_class": request.document_class,
        "destination_shelf": shelf.as_posix(),
        "proposed_destination": destination,
        "source_path": source_abs.as_posix(),
        "source_label": source_label,
        "source_sha256": digest,
        "source_size_bytes": size_bytes,
        "linked_dossier_ids": sorted(set(request.linked_dossier_ids)),
        "publication_id": request.publication_id,
        "currentness_status": "unreviewed",
        "authority_effect": "none",
        "scientific_claim_effect": "none",
        "exact_duplicates": exact_duplicates,
        "same_title_matches": same_titles,
        "warnings": warnings,
        "blockers": blockers,
        "next_action": "confirm publish" if not blockers else "resolve blockers before publishing",
        "durable_writes": False,
    }


class _Rollback:
    def __init__(self, paths: list[Path]) -> None:
        self._snapshots: dict[Path, bytes | None] = {}
        for path in paths:
            resolved = path.resolve()
            self._snapshots[resolved] = resolved.read_bytes() if resolved.exists() else None

    def restore(self) -> None:
        for path, content in self._snapshots.items():
            if content is None:
                if path.exists():
                    path.unlink()
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)


def _write_knowledge_catalog_site(repo_root: Path, source_root: Path) -> dict[str, Any]:
    catalog = wp4.build_catalog(source_root, repo_root)
    integrity = wp4.referential_integrity(catalog["accessions"])
    document_validation = wp4.validate_document_objects(catalog["accessions"], source_root=source_root)
    if not integrity["ok"] or not document_validation["ok"]:
        raise LibrarianError("human document ingress failed catalog validation")
    dossier_manifests = wp4.build_dossier_manifests(catalog)
    output = repo_root / "docs/000_knowledge-catalog.html"
    output_sha256 = wp4.write_catalog_html(output, catalog, dossier_manifests)
    return {
        "output": output.as_posix(),
        "output_sha256": output_sha256,
        "build_id": catalog["build_id"],
        "maintained_source_digest": catalog["maintained_source_digest"],
        "document_count": catalog["object_counts"]["documents"],
        "referential_integrity": integrity,
        "document_validation": document_validation,
    }


def _stage_bytes(source_abs: Path, runtime_root: Path, transaction_id: str) -> Path:
    target = runtime_root / "staging" / transaction_id / source_abs.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_abs, target)
    if file_sha256(target) != file_sha256(source_abs):
        raise LibrarianError("private ingress staging copy failed SHA-256 verification")
    return target


def _trash_source(source_abs: Path) -> dict[str, Any]:
    gio = shutil.which("gio")
    trash_put = shutil.which("trash-put")
    if gio:
        cmd = [gio, "trash", source_abs.as_posix()]
    elif trash_put:
        cmd = [trash_put, source_abs.as_posix()]
    else:
        return {"requested": True, "trashed": False, "error": "no safe trash command found"}
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return {
        "requested": True,
        "trashed": result.returncode == 0,
        "error": None if result.returncode == 0 else (result.stderr.strip() or result.stdout.strip()),
    }


def execute_external_document_ingress(
    *,
    repo_root: Path,
    source_root: Path,
    request: IngressRequest,
    mode: str,
    runtime_root: Path = DEFAULT_HUMAN_INGRESS_RUNTIME_ROOT,
    trash_source: bool = False,
) -> dict[str, Any]:
    if mode not in {"dry-run", "stage", "commit"}:
        raise LibrarianError(f"unsupported ingress mode: {mode}")
    preview = preview_external_document(repo_root=repo_root, source_root=source_root, request=request)
    if mode == "dry-run":
        return preview
    if not preview["ok"]:
        raise LibrarianError("human document ingress preview has blockers: " + "; ".join(preview["blockers"]))

    repo_root = repo_root.resolve()
    source_root = source_root.resolve()
    runtime_root = runtime_root.resolve()
    _require_branch(repo_root)
    _require_clean(repo_root)
    observed_commit = _repo_head(repo_root)
    transaction_id = f"{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:12]}"
    source_abs = Path(str(preview["source_path"]))
    staged_source = _stage_bytes(source_abs, runtime_root, transaction_id)
    if file_sha256(staged_source) != str(preview["source_sha256"]):
        raise LibrarianError("external source changed during private staging")

    records = _load_document_records(source_root)
    document_id = str(preview["document_id"])
    _, next_sequence = _next_document_id(records)
    destination = str(preview["proposed_destination"])
    destination_abs = repo_root / destination
    digest = str(preview["source_sha256"])
    size_bytes = int(preview["source_size_bytes"])
    source_label = str(preview["source_label"])
    dossiers = sorted(set(str(value) for value in request.linked_dossier_ids))

    path_relation = _relation(
        "path_changed",
        document_id,
        target_type="path",
        target_id_or_path=destination,
        basis=["operator_governance_decision", "path_history"],
        review_state="accepted",
        notes="Human external document ingress path transition; content and authority semantics are unchanged.",
    )
    relations = [path_relation]
    if request.reading_copy_of is not None:
        relations.append(
            _relation(
                "reading_copy_of",
                document_id,
                target_type="document",
                target_id_or_path=request.reading_copy_of,
                basis=["operator_governance_decision", "explicit_document_declaration"],
                review_state="accepted",
                notes="Operator-supplied reading-copy relation; it does not change authority or currentness.",
            )
        )
    relation_paths = [
        source_root / "document_relations" / f"{relation['relation_id']}.json"
        for relation in relations
    ]
    for relation_path in relation_paths:
        if relation_path.exists():
            raise LibrarianError(f"document relation already exists: {relation_path.stem}")

    path_event = {
        "publication_id": request.publication_id,
        "from_path": source_label,
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
        "title": request.title,
        "path": destination,
        "canonical_reading_location": destination,
        "sha256": digest,
        "size_bytes": size_bytes,
        "git_state": "tracked",
        "document_class": request.document_class,
        "document_class_candidates": [request.document_class],
        "refined_lifecycle_candidate": "human_external_ingress",
        "currentness_status": "unreviewed",
        "currentness_candidate": "explicit_ingress_review_required",
        "authored_or_generated": "candidate",
        "authorship_candidate": "authored_candidate",
        "retention_class": "candidate",
        "retention_candidate": "repo_publication",
        "review_state": "candidate",
        "review_flags": ["human_external_ingress", "post_publication_explicit_ingress"],
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
            "state": "human_external_ingress",
            "observed_at_commit": observed_commit,
            "source_digest": digest,
            "publication_id": request.publication_id,
            "original_source_path": source_label,
            "private_staging_path": staged_source.as_posix(),
        },
        "identity": {
            "frozen": True,
            "path_at_freeze": source_label,
            "sha256_at_freeze": digest,
            "size_bytes_at_freeze": size_bytes,
            "freeze_commit": observed_commit,
            "freeze_source_digest": digest,
            "freeze_basis": ["human_external_ingress", "explicit_post_publication_ingress"],
            "reuse_policy": "never_reuse",
            "path_is_identity": False,
            "path_history": [path_event],
            "content_revisions": [],
        },
        "proposed_publication": {
            "proposal_version": "eiel_docs_publication_namespace_v1",
            "destination_class": "repo_document",
            "proposed_destination": destination,
            "basis": ["human_external_ingress", "document_id_filename_prefix"],
            "confidence": "candidate",
            "authority_effect": "none",
            "physical_move_authorized": True,
            "publication_state": "published",
            "publication_id": request.publication_id,
            "requires_operator_review": False,
        },
        "publication": {
            "publication_id": request.publication_id,
            "action": "repo_copy",
            "path_before_publication": source_label,
            "current_path": destination,
            "content_unchanged": True,
            "authority_effect": "none",
            "prepublication_head": observed_commit,
        },
        "notes": (
            "Externally supplied by human ingress after Archive Publication 1.0; "
            "no authority, currentness, supersession, or scientific claim is inferred."
        ),
    }
    updated_records = sorted([*records, record], key=lambda row: str(row["document_id"]))

    paths_to_snapshot = [
        destination_abs,
        source_root / "documents" / f"{document_id}.json",
        *relation_paths,
        source_root / "manifests/DOCUMENT_LEGACY_PATH_REDIRECTS.json",
        source_root / "manifests/DOCUMENT_IDENTITY_FREEZE.json",
        source_root / "manifests/DOCUMENT_ARCHIVE_PUBLICATION_V1.json",
        source_root / "policies/document_identity_policy_v1.json",
        source_root / "policies/document_publication_namespace_v1.json",
        *[
            source_root / "accessions" / f"{collection}.json"
            for collection in wp4.OBJECT_CLASSES
        ],
        source_root / "manifests/GENERATED_AGGREGATES.json",
        source_root / "SOURCE_MANIFEST.json",
        repo_root / "docs/000_knowledge-catalog.html",
    ]
    rollback = _Rollback(paths_to_snapshot)
    git_paths: list[str] = []
    try:
        destination_abs.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(staged_source, destination_abs)
        if file_sha256(destination_abs) != digest or destination_abs.stat().st_size != size_bytes:
            raise LibrarianError("published document failed post-copy content verification")
        write_json(source_root / "documents" / f"{document_id}.json", record, pretty=False)
        for relation, relation_path in zip(relations, relation_paths, strict=True):
            write_json(relation_path, relation, pretty=False)

        redirects = _load_redirect_manifest(source_root)
        redirect_entries = list(redirects.get("entries") or [])
        redirect_entries.append(
            {
                "document_id": document_id,
                "old_path": source_label,
                "current_path": destination,
                "action": "repo_copy",
                "sha256": digest,
                "size_bytes": size_bytes,
            }
        )
        redirects.update(
            {
                "latest_publication_id": request.publication_id,
                "path_is_identity": False,
                "authority_effect": "none",
                "entry_count": len(redirect_entries),
                "entries": sorted(
                    redirect_entries,
                    key=lambda row: (str(row["document_id"]), str(row["old_path"])),
                ),
            }
        )
        write_json(source_root / "manifests/DOCUMENT_LEGACY_PATH_REDIRECTS.json", redirects, pretty=False)
        write_json(source_root / "manifests/DOCUMENT_IDENTITY_FREEZE.json", identity_freeze_manifest(updated_records), pretty=False)
        _write_identity_policy_after_ingress(source_root, updated_records, next_sequence=next_sequence)
        _write_publication_policy_after_ingress(source_root)
        _write_archive_manifest_after_ingress(
            source_root,
            updated_records,
            document_id=document_id,
            publication_id=request.publication_id,
        )
        objects = wp4.load_sharded_accessions(source_root)
        generated_aggregates = wp4.generate_aggregate_ledgers(source_root, objects)
        source_manifest = wp4.write_source_manifest(source_root, source_layout="per_object_shards")
        site = _write_knowledge_catalog_site(repo_root, source_root)

        git_paths = [
            destination,
            str((source_root / "documents" / f"{document_id}.json").relative_to(repo_root)),
            *[str(path.relative_to(repo_root)) for path in relation_paths],
            str((source_root / "manifests/DOCUMENT_LEGACY_PATH_REDIRECTS.json").relative_to(repo_root)),
            str((source_root / "manifests/DOCUMENT_IDENTITY_FREEZE.json").relative_to(repo_root)),
            str((source_root / "manifests/DOCUMENT_ARCHIVE_PUBLICATION_V1.json").relative_to(repo_root)),
            str((source_root / "policies/document_identity_policy_v1.json").relative_to(repo_root)),
            str((source_root / "policies/document_publication_namespace_v1.json").relative_to(repo_root)),
            *[
                str((source_root / "accessions" / f"{collection}.json").relative_to(repo_root))
                for collection in wp4.OBJECT_CLASSES
            ],
            str((source_root / "manifests/GENERATED_AGGREGATES.json").relative_to(repo_root)),
            str((source_root / "SOURCE_MANIFEST.json").relative_to(repo_root)),
            "docs/000_knowledge-catalog.html",
        ]
        git_add = _git(repo_root, ["add", "--", *git_paths])
        if git_add.returncode != 0:
            raise LibrarianError(git_add.stderr.strip() or "failed to stage human ingress files")
        commit_hash: str | None = None
        if mode == "commit":
            commit = _git(repo_root, ["commit", "-m", f"Ingest {document_id} {request.title}"])
            if commit.returncode != 0:
                raise LibrarianError(commit.stderr.strip() or "failed to commit human ingress")
            commit_hash = _repo_head(repo_root)
        trash_result = _trash_source(source_abs) if trash_source else {"requested": False, "trashed": False, "error": None}
    except Exception:
        if git_paths:
            _git(repo_root, ["reset", "-q", "HEAD", "--", *git_paths])
        rollback.restore()
        raise

    return {
        "schema_version": "eiel_human_document_ingress_receipt_v1",
        "mode": mode,
        "document_id": document_id,
        "title": request.title,
        "path_before_publication": source_label,
        "current_path": destination,
        "sha256": digest,
        "size_bytes": size_bytes,
        "publication_id": request.publication_id,
        "document_count": len(updated_records),
        "path_changed_relation_id": path_relation["relation_id"],
        "appended_relation_ids": [relation["relation_id"] for relation in relations],
        "generated_aggregates": generated_aggregates,
        "source_manifest": source_manifest,
        "knowledge_catalog_site": site,
        "authority_effect": "none",
        "scientific_claim_effect": "none",
        "physical_document_copied": True,
        "source_file_retained": not bool(trash_result["trashed"]),
        "trash_source": trash_result,
        "git": {
            "mode": mode,
            "branch": _require_branch(repo_root),
            "pre_ingress_head": observed_commit,
            "commit": commit_hash,
            "staged_paths": git_paths,
        },
        "transaction": {
            "id": transaction_id,
            "rollback_model": "file snapshots restored and created files removed if publication fails before successful commit",
            "private_staging_path": staged_source.as_posix(),
        },
    }


def _request_from_args(args: argparse.Namespace) -> IngressRequest:
    publication_id = args.publication_id or f"human-ingress-v1-{time.strftime('%Y-%m-%d')}"
    return IngressRequest(
        source=args.source,
        title=args.title,
        document_class=args.document_class,
        destination_shelf=args.shelf,
        linked_dossier_ids=tuple(args.dossier or ()),
        reading_copy_of=args.reading_copy_of,
        publication_id=publication_id,
    )


def ingest_document_from_args(args: argparse.Namespace) -> dict[str, Any]:
    modes = [args.dry_run, args.stage, args.commit]
    if sum(bool(value) for value in modes) != 1:
        raise LibrarianError("choose exactly one of --dry-run, --stage, or --commit")
    mode = "dry-run" if args.dry_run else "stage" if args.stage else "commit"
    return execute_external_document_ingress(
        repo_root=args.repo_root,
        source_root=args.source_root,
        request=_request_from_args(args),
        mode=mode,
        runtime_root=args.runtime_root,
        trash_source=args.trash_source,
    )


def download_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eiel-ingest-download")
    parser.add_argument("source", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-root", type=Path, default=Path("tools/evidence/librarian/catalog_source/00-catalog"))
    parser.add_argument("--title", required=True)
    parser.add_argument("--research-reference", action="store_true")
    parser.add_argument("--document-class", default=None)
    parser.add_argument("--shelf", default=None)
    parser.add_argument("--dossier", action="append", default=[])
    parser.add_argument("--publication-id")
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_HUMAN_INGRESS_RUNTIME_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stage", action="store_true")
    parser.add_argument("--commit", action="store_true")
    parser.add_argument("--trash-source", action="store_true")
    args = parser.parse_args(argv)
    if args.research_reference:
        args.document_class = args.document_class or "research_reference"
        args.shelf = args.shelf or "600_references-publications/600_research-references"
    if not args.document_class or not args.shelf:
        parser.error("--document-class and --shelf are required unless --research-reference is used")
    args.source_root = args.source_root if args.source_root.is_absolute() else args.repo_root / args.source_root
    try:
        result = ingest_document_from_args(args)
    except LibrarianError as exc:
        print(f"error: {exc}", file=os.sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
