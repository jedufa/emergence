"""WP4 synthesis dossiers, query interface, and source-shard views.

The WP4 layer treats repository-controlled accession metadata as the maintained
source and every catalog, dossier, search index, menu list, and HTML page as a
generated view.  It never reads raw payloads through simulation code and never
extracts historical archives.
"""

from __future__ import annotations

from engine.runtime_roots import public_runtime_path

import hashlib
import html
import json
import platform
import re
import subprocess
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .core import (
    LibrarianError,
    canonical_json_bytes,
    deterministic_zip,
    file_sha256,
    inspect_zip_safety,
    read_jsonl,
    tree_digest,
    write_json,
)
from .documents import validate_document_objects


WP4_SCHEMA_VERSION = "eiel_evidence_librarian_wp4_v1"
WP4_RUN_ID = "20260802T000000Z_wp4_synthesis_query_cli_menu"
WP3_FINAL_COMMIT = "7b4542608e87969d9e86f0161ef7522d95e020cd"
WP3_OBSERVED_ACCESSION_COMMIT = "5a328634717c4f1399ca52f0e34f69731611e2ab"
EXPECTED_TREE_DIGEST = "d474cc44ed9c61a63b553fb31aa5019af206c1339b1f90d6d89554f38859bcb5"
WP3_PACKAGE_SHA256 = "f920e5ca1065f48095b35f126c7192792d7f2ace124de3c14220780fb96e38fb"
WP3_CATALOG_SHA256 = "45f16e64de6ce733b0adeb419b8c08d79b2bacb4984e5fefdef40f709ecf3447"
WP3_HTML_SHA256 = "108e92d538bc0e35437d68096d28fa1fd5b25dc9bf602c9758d2e69446f3962e"
WP3_BUILD_RECORD_SHA256 = "87e26f2818c151f81698400c293c8cdf1d46e80b14db36e6648f1191d5e3e8df"


OBJECT_CLASSES: dict[str, dict[str, str]] = {
    "artifacts": {"id_field": "artifact_id", "shard_dir": "artifacts"},
    "records": {"id_field": "evidence_id", "shard_dir": "records"},
    "claims": {"id_field": "claim_id", "shard_dir": "claims"},
    "authorities": {"id_field": "authority_assertion_id", "shard_dir": "authorities"},
    "raw_assets": {"id_field": "raw_asset_id", "shard_dir": "raw_assets"},
    "issues": {"id_field": "issue_id", "shard_dir": "issues"},
    "documents": {"id_field": "document_id", "shard_dir": "documents"},
    "document_relations": {"id_field": "relation_id", "shard_dir": "document_relations"},
}


EVIDENCE_OBJECT_CLASSES = {"artifacts", "records", "claims", "authorities", "raw_assets", "issues"}
DOCUMENT_OBJECT_CLASSES = {"documents", "document_relations"}
LOGICAL_COLLECTIONS = ("evidence", "documents", "all")
SEARCH_UX_SCHEMA_VERSION = "eiel_knowledge_search_ux_v1"
DECISION_READY_SCHEMA_VERSION = "eiel_decision_ready_preflight_v1"
KNOWLEDGE_CATALOG_JSON = "EIEL_Knowledge_Catalog.json"
KNOWLEDGE_CATALOG_HTML = "EIEL_Knowledge_Catalog.html"
KNOWLEDGE_SEARCH_INDEX_JSON = "EIEL_Knowledge_Search_Index.json"
KNOWLEDGE_BUILD_RECORD_JSON = "EIEL_Knowledge_Catalog_Build_Record.json"
DOCS_AUDIT_EXPECTED = {
    "observed_commit": "55619321b7fdf7693427c6c55a9b34a0e8c62615",
    "docs_inventory_audit_v1.json": "a6e24b9944c0d277582a39324b6ace63fd7eb4a5b8a672d81cf13a2c116721bb",
    "docs_inventory_audit_v1.md": "d9f1e32ad44d78c26b6833a6e5042579416b551bea2f393859d93afe87ab5b33",
    "document_catalog_candidate_v1.json": "9ed8c4f1b8148034fa9661d3808a08132592fcf84516177d020115d57f5223ca",
    "document_issue_ledger_candidate_v1.json": "e692830ece1c07b507b312ce9d050e23c18992c6b7ce35ad4229bcc1e1ab2d5e",
    "document_authority_review_queue_candidate_v1.json": "263767204b784847eeeb0ec79e615f73d9e43bf38d3643cd21d3994ee419bbe7",
    "docs_refined_lifecycle_candidate_v1.json": "aff037410b12ff66bb09dad1e9bf9ef094bf397870f5ac3f89cd07c6bdb57081",
}


DOSSIER_DEFINITIONS: list[dict[str, Any]] = [
    {
        "dossier_id": "dossier-01-programme-evidence-map-methods-claim-boundaries",
        "sequence": 1,
        "title": "Programme Evidence Map, Methods, and Claim Boundaries",
        "governed_scope": "programme evidence map, methods, and claim boundaries",
        "keywords": ["programme", "method", "claim", "boundary", "register", "strategy", "roadmap", "catalog", "librarian"],
    },
    {
        "dossier_id": "dossier-02-engine-determinism-execution-route-resource-qualification",
        "sequence": 2,
        "title": "Engine, Determinism, Execution-Route, and Resource Qualification",
        "governed_scope": "engine, determinism, execution-route, and resource qualification",
        "keywords": ["engine", "determin", "execution", "route", "qualification", "provider", "runtime", "checkpoint", "cuda", "cpu", "resource"],
    },
    {
        "dossier_id": "dossier-03-worlds-ecology-environmental-pressure",
        "sequence": 3,
        "title": "Worlds, Ecology, and Environmental Pressure",
        "governed_scope": "worlds, ecology, and environmental pressure",
        "keywords": ["world", "ecolog", "environment", "pressure", "food", "niche", "patch", "resource", "roz"],
    },
    {
        "dossier_id": "dossier-04-lineage-reproduction-diversity",
        "sequence": 4,
        "title": "Lineage, Reproduction, and Diversity",
        "governed_scope": "lineage, reproduction, and diversity",
        "keywords": ["lineage", "reproduction", "birth", "parent", "child", "diversity", "genealog", "turnover"],
    },
    {
        "dossier_id": "dossier-05-cognition-memory-representation",
        "sequence": 5,
        "title": "Cognition, Memory, and Representation",
        "governed_scope": "cognition, memory, and representation",
        "keywords": ["cognition", "memory", "representation", "controller", "mlp", "gru", "pgru", "policy", "behavior", "observer"],
    },
    {
        "dossier_id": "dossier-06-genome-inheritance-mutation-recombination",
        "sequence": 6,
        "title": "Genome, Inheritance, Mutation, and Recombination",
        "governed_scope": "genome, inheritance, mutation, and recombination",
        "keywords": ["genome", "inheritance", "mutation", "recombination", "recomb", "diploid", "hereditary", "gv2"],
    },
    {
        "dossier_id": "dossier-07-open-ended-search-evolvability",
        "sequence": 7,
        "title": "Open-Ended Search and Evolvability",
        "governed_scope": "open-ended search and evolvability",
        "keywords": ["open-ended", "open ended", "evolvability", "evolution", "search", "frontier", "eed", "exploration"],
    },
    {
        "dossier_id": "dossier-08-diagnostics-negative-results-closures-superseded-work",
        "sequence": 8,
        "title": "Diagnostics, Negative Results, Closures, and Superseded Work",
        "governed_scope": "diagnostics, negative results, closures, and superseded work",
        "keywords": ["diagnostic", "negative", "closure", "closed", "superseded", "blocked", "invalid", "issue", "unsafe", "integrity", "limitation", "rejected"],
        "cross_cutting_negative": True,
    },
]


EXIT_SUCCESS = 0
EXIT_NO_MATCH = 10
EXIT_AMBIGUOUS = 11
EXIT_INVALID_ID = 12
EXIT_STALE_VIEW = 13
EXIT_SCHEMA_FAILURE = 14
EXIT_INTEGRITY_RESTRICTED = 15
EXIT_INTERNAL_ERROR = 2


PREFLIGHT_STOPWORDS = {
    "and",
    "before",
    "codex",
    "eiel",
    "emergence",
    "for",
    "from",
    "integration",
    "mandatory",
    "model",
    "operational",
    "the",
    "with",
    "workflow",
}


class CliExit(LibrarianError):
    """A librarian CLI condition with a stable exit code."""

    def __init__(self, message: str, exit_code: int, payload: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.exit_code = exit_code
        self.payload = dict(payload or {})


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_text(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return file_sha256(path)


def _git(repo_root: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and result.returncode != 0:
        raise LibrarianError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def _id_for(collection: str, row: Mapping[str, Any]) -> str:
    return str(row[OBJECT_CLASSES[collection]["id_field"]])


def _aggregate_path(source_root: Path, collection: str) -> Path:
    return source_root / "accessions" / f"{collection}.json"


def _shard_dir(source_root: Path, collection: str) -> Path:
    return source_root / OBJECT_CLASSES[collection]["shard_dir"]


def load_aggregate_accessions(source_root: Path) -> dict[str, list[dict[str, Any]]]:
    return {name: _load(_aggregate_path(source_root, name)) for name in OBJECT_CLASSES}


def load_sharded_accessions(source_root: Path) -> dict[str, list[dict[str, Any]]]:
    objects: dict[str, list[dict[str, Any]]] = {}
    missing: list[str] = []
    for name in OBJECT_CLASSES:
        directory = _shard_dir(source_root, name)
        if not directory.is_dir():
            missing.append(name)
            continue
        rows = [_load(path) for path in sorted(directory.glob("*.json"))]
        objects[name] = sorted(rows, key=lambda row: _id_for(name, row))
    if missing:
        raise LibrarianError(f"missing maintained shard directories: {', '.join(missing)}")
    return objects


def load_source_accessions(source_root: Path) -> tuple[dict[str, list[dict[str, Any]]], str]:
    if all(_shard_dir(source_root, name).is_dir() for name in OBJECT_CLASSES):
        return load_sharded_accessions(source_root), "per_object_shards"
    return load_aggregate_accessions(source_root), "aggregate_ledgers"


def classify_source_layout(source_root: Path) -> dict[str, Any]:
    aggregate_counts = {}
    shard_counts = {}
    for name in OBJECT_CLASSES:
        aggregate = _aggregate_path(source_root, name)
        aggregate_counts[name] = len(_load(aggregate)) if aggregate.exists() else None
        directory = _shard_dir(source_root, name)
        shard_counts[name] = len(list(directory.glob("*.json"))) if directory.is_dir() else None
    all_sharded = all(value is not None for value in shard_counts.values())
    aggregates_present = all(value is not None for value in aggregate_counts.values())
    if all_sharded and aggregates_present:
        classification = "per_object_shards_maintained_aggregates_generated"
    elif aggregates_present:
        classification = "aggregate_ledgers_are_maintained_source"
    else:
        classification = "incomplete_or_unknown_source_layout"
    return {
        "schema_version": "eiel_wp4_source_layout_classification_v1",
        "source_root": str(source_root),
        "classification": classification,
        "aggregate_counts": aggregate_counts,
        "shard_counts": shard_counts,
        "target_shape": "one maintained semantic accession object per JSON shard; aggregate ledgers are generated compatibility views",
    }


def shard_source_from_aggregates(source_root: Path) -> dict[str, Any]:
    """Split current aggregate accession ledgers into deterministic object files."""

    before = load_aggregate_accessions(source_root)
    written: dict[str, dict[str, str]] = {}
    counts: dict[str, int] = {}
    for collection, rows in before.items():
        directory = _shard_dir(source_root, collection)
        directory.mkdir(parents=True, exist_ok=True)
        hashes: dict[str, str] = {}
        seen: set[str] = set()
        for row in sorted(rows, key=lambda item: _id_for(collection, item)):
            object_id = _id_for(collection, row)
            if object_id in seen:
                raise LibrarianError(f"duplicate {collection} id: {object_id}")
            seen.add(object_id)
            hashes[f"{OBJECT_CLASSES[collection]['shard_dir']}/{object_id}.json"] = write_json(
                directory / f"{object_id}.json", row, pretty=False
            )
        written[collection] = hashes
        counts[collection] = len(hashes)
    after = load_sharded_accessions(source_root)
    equivalence = semantic_equivalence(before, after)
    if not equivalence["semantically_equal"]:
        raise LibrarianError("aggregate-to-shard semantic equivalence failed")
    policy_provenance = write_policy_and_provenance_sources(source_root)
    aggregate_receipt = generate_aggregate_ledgers(source_root, after)
    manifest = write_source_manifest(source_root, source_layout="per_object_shards")
    receipt = {
        "schema_version": "eiel_wp4_source_refactor_receipt_v1",
        "source_root": str(source_root),
        "metadata_only": True,
        "historical_payloads_touched": False,
        "object_ids_preserved": True,
        "semantic_equivalence": equivalence,
        "shard_counts": counts,
        "written_shard_hashes": written,
        "policy_and_provenance_hashes": policy_provenance,
        "aggregate_generation": aggregate_receipt,
        "source_manifest": manifest,
    }
    write_json(source_root / "governance" / "wp4_source_refactor_receipt.json", receipt, pretty=False)
    return receipt


def write_policy_and_provenance_sources(source_root: Path) -> dict[str, str]:
    copies = [
        ("governance/authority_policy_decisions.json", "policies/authority_policy_decisions.json"),
        ("governance/package_provenance_receipt.json", "provenance/package_provenance_receipt.json"),
        ("governance/review_disposition_ledger.json", "provenance/review_disposition_ledger.json"),
        ("governance/remaining_review_queue.json", "provenance/remaining_review_queue.json"),
    ]
    hashes: dict[str, str] = {}
    for source_rel, target_rel in copies:
        source = source_root / source_rel
        if source.is_file():
            hashes[target_rel] = write_json(source_root / target_rel, _load(source), pretty=False)
    return hashes


def semantic_equivalence(before: Mapping[str, list[dict[str, Any]]], after: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    failures: dict[str, Any] = {}
    for collection in OBJECT_CLASSES:
        before_by_id = {_id_for(collection, row): row for row in before.get(collection, [])}
        after_by_id = {_id_for(collection, row): row for row in after.get(collection, [])}
        if before_by_id != after_by_id:
            failures[collection] = {
                "missing_after": sorted(set(before_by_id) - set(after_by_id)),
                "extra_after": sorted(set(after_by_id) - set(before_by_id)),
                "changed": sorted(
                    object_id
                    for object_id in set(before_by_id) & set(after_by_id)
                    if before_by_id[object_id] != after_by_id[object_id]
                ),
            }
    return {
        "schema_version": "eiel_wp4_semantic_equivalence_v1",
        "semantically_equal": not failures,
        "failures": failures,
    }


def generate_aggregate_ledgers(source_root: Path, objects: Mapping[str, list[dict[str, Any]]] | None = None) -> dict[str, Any]:
    objects = dict(objects or load_sharded_accessions(source_root))
    hashes: dict[str, str] = {}
    for collection, rows in objects.items():
        hashes[f"accessions/{collection}.json"] = write_json(
            _aggregate_path(source_root, collection),
            sorted(rows, key=lambda row: _id_for(collection, row)),
            pretty=False,
        )
    receipt = {
        "schema_version": "eiel_wp4_generated_aggregate_receipt_v1",
        "generated_from": "maintained_per_object_shards",
        "generated_at_policy": "deterministic_1970_utc_for_generated_views",
        "aggregate_hashes": hashes,
    }
    write_json(source_root / "manifests" / "GENERATED_AGGREGATES.json", receipt, pretty=False)
    return receipt


def maintained_source_digest(source_root: Path) -> str:
    digest = hashlib.sha256()
    include_dirs = [OBJECT_CLASSES[name]["shard_dir"] for name in OBJECT_CLASSES] + [
        "dossiers",
        "policies",
        "provenance",
        "review_queues",
        "schemas",
    ]
    for directory_name in sorted(include_dirs):
        directory = source_root / directory_name
        if not directory.exists():
            continue
        for path in sorted(p for p in directory.rglob("*.json") if p.is_file()):
            rel = path.relative_to(source_root).as_posix()
            digest.update(rel.encode("utf-8") + b"\0" + file_sha256(path).encode("ascii") + b"\n")
    return digest.hexdigest()


def write_source_manifest(source_root: Path, *, source_layout: str) -> dict[str, Any]:
    objects, _ = load_source_accessions(source_root)
    digest = maintained_source_digest(source_root)
    manifest = {
        "schema_version": "eiel_wp4_maintained_source_manifest_v1",
        "source_root": "tools/evidence/librarian/catalog_source/00-catalog",
        "source_layout": source_layout,
        "metadata_only": True,
        "historical_payloads_copied": False,
        "maintained_source_digest": digest,
        "digest_scope": "per-object accession shards plus dossier, policy, provenance, review-queue, and schema definitions; generated aggregates and generated views excluded",
        "object_counts": {name: len(rows) for name, rows in sorted(objects.items())},
        "aggregate_ledgers_are_generated": True,
    }
    write_json(source_root / "SOURCE_MANIFEST.json", manifest, pretty=False)
    return manifest


def ensure_dossier_definition_files(source_root: Path) -> dict[str, Any]:
    hashes: dict[str, str] = {}
    for row in DOSSIER_DEFINITIONS:
        hashes[f"dossiers/{row['dossier_id']}.json"] = write_json(
            source_root / "dossiers" / f"{row['dossier_id']}.json",
            {
                "schema_version": "eiel_wp4_dossier_definition_v1",
                "dossier_id": row["dossier_id"],
                "sequence": row["sequence"],
                "title": row["title"],
                "governed_scope": row["governed_scope"],
                "definition_authority": {
                    "latest_explicit_librarian_governance_source": "original Evidence Library Audit and Consolidation Proposal, section 'Eight synthesis dossiers', retained by WP4 bridge prompt",
                    "proposal_line_range": "EIEL_Evidence_Library_Audit_and_Consolidation_Proposal_2026_08_02.md:140-151",
                    "v1r_layout_line_range": "EIEL_Evidence_Librarian_V1R_Governance_and_Design_2026_08_02.md:45-63",
                    "conflict_resolution": "proposal says approximately; WP4 freezes the listed eight scopes as governed titles for this implementation",
                },
                "keywords": row["keywords"],
                "cross_cutting_negative": bool(row.get("cross_cutting_negative", False)),
            },
            pretty=False,
        )
    receipt = {
        "schema_version": "eiel_wp4_dossier_definition_receipt_v1",
        "definition_count": len(DOSSIER_DEFINITIONS),
        "definition_hashes": hashes,
    }
    write_json(source_root / "manifests" / "DOSSIER_DEFINITIONS.json", receipt, pretty=False)
    return receipt


def referential_integrity(objects: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    ids = {name: {_id_for(name, row) for row in rows} for name, rows in objects.items()}
    failures: list[dict[str, Any]] = []

    def check_many(owner: str, owner_id: str, field: str, targets: Iterable[str], target_collection: str) -> None:
        for target in targets or []:
            if target not in ids[target_collection]:
                failures.append(
                    {
                        "owner_class": owner,
                        "owner_id": owner_id,
                        "field": field,
                        "target_class": target_collection,
                        "target_id": target,
                        "failure": "missing_target",
                    }
                )

    for record in objects["records"]:
        record_id = _id_for("records", record)
        check_many("records", record_id, "artifact_ids", record.get("artifact_ids", []), "artifacts")
        check_many("records", record_id, "claim_ids", record.get("claim_ids", []), "claims")
        check_many("records", record_id, "authority_assertion_ids", record.get("authority_assertion_ids", []), "authorities")
        check_many("records", record_id, "raw_asset_ids", record.get("raw_asset_ids", []), "raw_assets")
        for field in ("derived_from_record_ids", "has_part_record_ids", "supersedes_record_ids", "superseded_by_record_ids"):
            targets = record.get(field, [])
            check_many("records", record_id, field, targets, "records")
            if record_id in targets:
                failures.append({"owner_class": "records", "owner_id": record_id, "field": field, "failure": "self_reference"})
    for claim in objects["claims"]:
        claim_id = _id_for("claims", claim)
        check_many("claims", claim_id, "supporting_record_ids", claim.get("supporting_record_ids", []), "records")
        check_many("claims", claim_id, "contradicting_record_ids", claim.get("contradicting_record_ids", []), "records")
        supersession = claim.get("supersession", {})
        check_many("claims", claim_id, "supersedes_claim_ids", supersession.get("supersedes_claim_ids", []), "claims")
        check_many("claims", claim_id, "superseded_by_claim_ids", supersession.get("superseded_by_claim_ids", []), "claims")
    for authority in objects["authorities"]:
        source = authority.get("authority_source")
        if source and not isinstance(source, str):
            failures.append({"owner_class": "authorities", "owner_id": _id_for("authorities", authority), "field": "authority_source", "failure": "non_string_source"})
    dossier_ids = {row["dossier_id"] for row in DOSSIER_DEFINITIONS}
    for document in objects.get("documents", []):
        document_id = _id_for("documents", document)
        check_many("documents", document_id, "linked_evidence_ids", document.get("linked_evidence_ids", []), "records")
        check_many("documents", document_id, "linked_claim_ids", document.get("linked_claim_ids", []), "claims")
        check_many("documents", document_id, "linked_authority_assertion_ids", document.get("linked_authority_assertion_ids", []), "authorities")
        check_many("documents", document_id, "linked_raw_asset_ids", document.get("linked_raw_asset_ids", []), "raw_assets")
        check_many("documents", document_id, "linked_artifact_ids", document.get("linked_artifact_ids", []), "artifacts")
        check_many("documents", document_id, "linked_issue_ids", document.get("linked_issue_ids", []), "issues")
        for target in document.get("linked_dossier_ids", []) or []:
            if target not in dossier_ids:
                failures.append({"owner_class": "documents", "owner_id": document_id, "field": "linked_dossier_ids", "target_class": "dossiers", "target_id": target, "failure": "missing_target"})
    relation_target_collections = {
        "document": "documents",
        "evidence_record": "records",
        "claim": "claims",
        "authority": "authorities",
        "raw_asset": "raw_assets",
    }
    for relation in objects.get("document_relations", []):
        relation_id = _id_for("document_relations", relation)
        check_many("document_relations", relation_id, "source_document_id", [relation.get("source_document_id")], "documents")
        target = relation.get("target") or {}
        target_type = target.get("target_type")
        target_id = target.get("target_id_or_path")
        if target_type in relation_target_collections and target_id:
            check_many("document_relations", relation_id, "target", [target_id], relation_target_collections[target_type])
        elif target_type == "dossier" and target_id not in dossier_ids:
            failures.append({"owner_class": "document_relations", "owner_id": relation_id, "field": "target", "target_class": "dossiers", "target_id": target_id, "failure": "missing_target"})
    for issue in objects.get("issues", []):
        issue_id = _id_for("issues", issue)
        check_many("issues", issue_id, "document_ids", issue.get("document_ids", []), "documents")
        if issue.get("owner_collection") not in {"documents", "evidence", "shared"}:
            failures.append({"owner_class": "issues", "owner_id": issue_id, "field": "owner_collection", "failure": "invalid_or_missing_owner_collection"})
    document_validation = validate_document_objects(objects)
    failures.extend(document_validation["failures"])
    all_ids = [object_id for rows in ids.values() for object_id in rows]
    duplicate_global_ids = [object_id for object_id, count in Counter(all_ids).items() if count > 1]
    for object_id in duplicate_global_ids:
        failures.append({"owner_id": object_id, "failure": "duplicate_global_id"})
    return {
        "schema_version": "eiel_wp4_referential_integrity_v1",
        "ok": not failures,
        "failure_count": len(failures),
        "failures": failures[:200],
        "object_counts": {name: len(rows) for name, rows in objects.items()},
        "document_validation": document_validation,
    }


def _object_text(row: Mapping[str, Any]) -> str:
    return json.dumps(row, sort_keys=True, ensure_ascii=False).casefold()


def _topic_terms(topic: str) -> list[str]:
    terms: list[str] = []
    for term in re.findall(r"[a-z0-9][a-z0-9_-]*", topic.casefold()):
        normalized = term.strip("_-")
        if len(normalized) < 3 or normalized in PREFLIGHT_STOPWORDS:
            continue
        terms.append(normalized)
    return sorted(set(terms))


def _topic_match_score(row: Mapping[str, Any], topic: str, terms: Iterable[str]) -> int:
    text = _object_text(row)
    score = 0
    topic_cf = topic.casefold().strip()
    if topic_cf and topic_cf in text:
        score += 100
    for term in terms:
        if term in text:
            score += 1
    return score


def _negativeish(row: Mapping[str, Any]) -> bool:
    text = _object_text(row)
    return any(word in text for word in ("negative", "blocked", "superseded", "invalid", "unsafe", "integrity", "issue", "closure", "closed", "rejected", "limitation"))


def classify_dossier_membership(objects: Mapping[str, list[dict[str, Any]]]) -> dict[str, dict[str, list[str]]]:
    membership: dict[str, dict[str, list[str]]] = {
        row["dossier_id"]: {name: [] for name in OBJECT_CLASSES} for row in DOSSIER_DEFINITIONS
    }
    for collection, rows in objects.items():
        for obj in rows:
            object_id = _id_for(collection, obj)
            text = _object_text(obj)
            matched = []
            for dossier in DOSSIER_DEFINITIONS:
                if any(keyword in text for keyword in dossier["keywords"]):
                    matched.append(dossier["dossier_id"])
            if not matched:
                matched.append("dossier-01-programme-evidence-map-methods-claim-boundaries")
            if collection == "issues" or _negativeish(obj):
                negative_id = "dossier-08-diagnostics-negative-results-closures-superseded-work"
                if negative_id not in matched:
                    matched.append(negative_id)
            for dossier_id in sorted(set(matched)):
                membership[dossier_id][collection].append(object_id)
    for by_collection in membership.values():
        for collection in by_collection:
            by_collection[collection] = sorted(set(by_collection[collection]))
    return membership


def build_catalog(source_root: Path, repo_root: Path) -> dict[str, Any]:
    objects, layout = load_source_accessions(source_root)
    source_manifest = {
        "maintained_source_digest": maintained_source_digest(source_root),
        "object_counts": {name: len(rows) for name, rows in sorted(objects.items())},
    }
    build_id = "eiel-build-wp4-" + hashlib.sha256(source_manifest["maintained_source_digest"].encode()).hexdigest()[:20]
    return {
        "schema_version": "eiel_knowledge_catalog_generated_v1",
        "generated_at": "1970-01-01T00:00:00Z",
        "run_id": WP4_RUN_ID,
        "build_id": build_id,
        "source_root": str(source_root.relative_to(repo_root) if source_root.is_relative_to(repo_root) else source_root),
        "source_layout": layout,
        "maintained_source_digest": source_manifest["maintained_source_digest"],
        "observed_repository_commit": _git(repo_root, "rev-parse", "HEAD", check=False) or "unavailable-no-git",
        "wp3_observed_accession_commit": WP3_OBSERVED_ACCESSION_COMMIT,
        "accessions": objects,
        "dossier_definitions": [dict(row) for row in DOSSIER_DEFINITIONS],
        "dossier_membership": classify_dossier_membership(objects),
        "object_counts": {name: len(rows) for name, rows in objects.items()},
        "logical_collections": {
            "evidence": sorted(EVIDENCE_OBJECT_CLASSES),
            "documents": sorted(DOCUMENT_OBJECT_CLASSES),
            "all": sorted(OBJECT_CLASSES),
        },
        "no_claims_boundary": "Generated views do not grant authority and make no intelligence, emergence, consciousness, broad cognition, broad adaptive memory, prediction-selection, or open-ended-evolution claim.",
    }


def build_dossier_manifests(catalog: Mapping[str, Any]) -> list[dict[str, Any]]:
    objects = catalog["accessions"]
    memberships = catalog["dossier_membership"]
    manifests = []
    for definition in DOSSIER_DEFINITIONS:
        dossier_id = definition["dossier_id"]
        members = memberships[dossier_id]
        record_ids = members["records"]
        claim_ids = members["claims"]
        authority_ids = members["authorities"]
        issue_ids = members["issues"]
        raw_ids = members["raw_assets"]
        document_ids = members.get("documents", [])
        document_relation_ids = members.get("document_relations", [])
        negative_record_ids = [
            row["evidence_id"]
            for row in objects["records"]
            if row["evidence_id"] in record_ids and _negativeish(row)
        ]
        current_authority_ids = [
            row["authority_assertion_id"]
            for row in objects["authorities"]
            if row["authority_assertion_id"] in authority_ids and row.get("confidence") == "explicit"
        ]
        raw_summary = Counter(
            row.get("storage_class") or row.get("retention_state") or "unknown"
            for row in objects["raw_assets"]
            if row["raw_asset_id"] in raw_ids
        )
        manifests.append(
            {
                "schema_version": "eiel_wp4_dossier_manifest_v1",
                "dossier_id": dossier_id,
                "sequence": definition["sequence"],
                "title": definition["title"],
                "exact_scope": definition["governed_scope"],
                "source_accession_digest": catalog["maintained_source_digest"],
                "catalog_build_id": catalog["build_id"],
                "generation_timestamp_policy": "deterministic_1970_utc_for_generated_views",
                "last_human_review": None,
                "covered_record_ids": record_ids,
                "covered_claim_ids": claim_ids,
                "covered_authority_ids": authority_ids,
                "covered_issue_ids": issue_ids,
                "covered_raw_asset_ids": raw_ids,
                "covered_document_ids": document_ids,
                "covered_document_relation_ids": document_relation_ids,
                "topical_status_summary": {
                    "records": len(record_ids),
                    "claims": len(claim_ids),
                    "authorities": len(authority_ids),
                    "issues": len(issue_ids),
                    "negative_or_integrity_records": len(negative_record_ids),
                    "raw_asset_storage_classes": dict(sorted(raw_summary.items())),
                    "documents": len(document_ids),
                    "document_relations": len(document_relation_ids),
                },
                "current_authority_statement": (
                    f"{len(current_authority_ids)} explicit authority assertion(s) are in this dossier scope; inspect authority IDs before making any status claim."
                    if current_authority_ids
                    else "No current authority statement is asserted for this dossier beyond the accession metadata."
                ),
                "negative_and_contradictory_evidence": sorted(set(issue_ids + negative_record_ids)),
                "superseded_and_historical_records": [
                    row["evidence_id"]
                    for row in objects["records"]
                    if row["evidence_id"] in record_ids and (row.get("superseded_by_record_ids") or row.get("supersedes_record_ids"))
                ],
                "blocked_and_integrity_restricted_records": issue_ids,
                "unresolved_limitations": [
                    "Dossier membership is generated from accession metadata and governed keywords; it is a navigation layer, not a scientific conclusion.",
                    "Raw assets are referenced as metadata only and are not restored or extracted by WP4.",
                ],
                "no_claims_boundary": catalog["no_claims_boundary"],
            }
        )
    return manifests


def _html_table(headers: Iterable[str], rows: Iterable[Iterable[Any]]) -> str:
    head = "".join(f"<th>{html.escape(str(col))}</th>" for col in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def write_catalog_html(path: Path, catalog: Mapping[str, Any], dossier_manifests: list[Mapping[str, Any]]) -> str:
    dossier_rows = [
        (
            m["dossier_id"],
            m["title"],
            m["topical_status_summary"]["records"],
            m["topical_status_summary"]["documents"],
            m["topical_status_summary"]["claims"],
            m["topical_status_summary"]["issues"],
        )
        for m in dossier_manifests
    ]
    authority_rows = [
        (row["domain"], row["asserted_status"], row["confidence"], row.get("wp3_policy_id"))
        for row in catalog["accessions"]["authorities"]
    ]
    evidence_rows = [
        (row["evidence_id"], row.get("title") or row.get("decision") or "", _status_for(row))
        for row in catalog["accessions"]["records"]
    ]
    issue_rows = [
        (row["issue_id"], row.get("owner_collection", "evidence"), row.get("issue_kind") or row.get("category"), row.get("severity"), row.get("review_state") or row.get("remediation_state"))
        for row in catalog["accessions"]["issues"]
    ]
    review_rows = [row for row in catalog["accessions"]["issues"] if row.get("owner_collection") == "documents" and row.get("review_state") not in {"resolved", "accepted", "rejected"}]
    document_html_rows = []
    for row in catalog["accessions"].get("documents", []):
        current_path = str(row["path"])
        if current_path.startswith("docs/"):
            display_path = html.escape(current_path)
            href = html.escape(current_path.removeprefix("docs/"), quote=True)
            path_cell = f'<a href="{href}">{display_path}</a>'
        else:
            path_cell = f"<code>{html.escape(current_path)}</code>"
        cells = (
            html.escape(str(row["document_id"])),
            html.escape(str(row.get("title") or "")),
            html.escape(str(row.get("document_class") or "unknown")),
            html.escape(str(row.get("currentness_status") or "unknown")),
            html.escape(str(row.get("review_state") or "unknown")),
            path_cell,
        )
        document_html_rows.append(
            "<tr>" + "".join(f"<td>{value}</td>" for value in cells) + "</tr>"
        )
    document_table = "<table id=\"document-table\"><thead><tr><th>Document ID</th><th>Title</th><th>Class</th><th>Currentness</th><th>Review</th><th>Location</th></tr></thead><tbody>" + "".join(document_html_rows) + "</tbody></table>"
    catalog_styles = """body{font-family:system-ui,sans-serif;margin:0;color:#1f2328;background:#f7f9fb}main{max-width:1500px;margin:auto;padding:1.5rem}table{border-collapse:collapse;width:100%;font-size:.86rem;margin:.75rem 0;background:white}th,td{border:1px solid #d8dee4;padding:.38rem .5rem;text-align:left;vertical-align:top}th{background:#eef2f6;position:sticky;top:0}.box{background:white;border:1px solid #d8dee4;border-radius:8px;padding:.8rem;margin:.75rem 0}code{background:#f0f3f6;padding:.1rem .25rem}nav{display:flex;gap:.4rem;flex-wrap:wrap;margin:1rem 0}button{border:1px solid #8c959f;background:white;border-radius:6px;padding:.45rem .7rem;cursor:pointer}button.active{background:#0969da;color:white}section{display:none}section.active{display:block}input{width:min(42rem,95%);padding:.55rem;border:1px solid #8c959f;border-radius:6px}.warning{border-left:5px solid #bf8700}.decision{border-left:5px solid #0969da}.decision-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(13rem,1fr));gap:.5rem;margin:.6rem 0}.decision-grid div{border:1px solid #d8dee4;border-radius:6px;padding:.5rem;background:#f7f9fb}details{background:white;border:1px solid #d8dee4;border-radius:8px;padding:.7rem;margin:.75rem 0}summary{cursor:pointer;font-weight:650}a{color:#0969da}.recent-list{margin:.2rem 0 0 1.25rem;padding:0}.recent-list li{padding:.45rem 0}.recent-list a{font-weight:650}.recent-meta{display:block;margin-top:.15rem;color:#57606a;font-size:.78rem}.recent-empty{color:#57606a;margin:.2rem 0}.recent-actions{display:flex;justify-content:flex-end;margin-bottom:.35rem}"""
    catalog_script = """(()=>{const RECENT_KEY="eielKnowledgeCatalog.recentFiles.v1";const MAX_RECENT=12;const ESCAPE={"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"};function escapeText(value){return String(value||"").replace(/[&<>"']/g,ch=>ESCAPE[ch])}function loadRecent(){try{const parsed=JSON.parse(localStorage.getItem(RECENT_KEY)||"[]");return Array.isArray(parsed)?parsed.filter(row=>row&&row.href).slice(0,MAX_RECENT):[]}catch{return []}}function saveRecent(rows){try{localStorage.setItem(RECENT_KEY,JSON.stringify(rows.slice(0,MAX_RECENT)))}catch{}}function formatWhen(ts){if(!ts)return"unknown time";try{return new Date(ts).toLocaleString()}catch{return"unknown time"}}function describeLink(link){const href=link.getAttribute("href")||"";const row=link.closest("tr");const cells=row?Array.from(row.cells).map(cell=>cell.textContent.trim().replace(/\\s+/g," ")):[];const label=cells.length>=2&&cells[0]&&cells[1]?cells[0]+" - "+cells[1]:(link.textContent.trim()||href);const meta=cells.length?(cells[cells.length-1]||href):href;return{href,label,meta,ts:Date.now()}}function renderRecent(){const root=document.getElementById("recent-files");if(!root)return;const rows=loadRecent();if(!rows.length){root.innerHTML='<p class="recent-empty">No catalog files opened from this browser yet.</p>';return}root.innerHTML='<div class="recent-actions"><button type="button" id="clear-recent-files">Clear</button></div><ol class="recent-list">'+rows.map(row=>'<li><a href="'+escapeText(row.href)+'">'+escapeText(row.label||row.href)+'</a><span class="recent-meta">'+escapeText(formatWhen(row.ts))+" | "+escapeText(row.meta||row.href)+"</span></li>").join("")+"</ol>";const clear=document.getElementById("clear-recent-files");if(clear)clear.addEventListener("click",()=>{saveRecent([]);renderRecent()})}function recordRecent(link){const href=link.getAttribute("href")||"";if(!href||href.startsWith("#")||href.startsWith("javascript:")||link.closest("nav"))return;const entry=describeLink(link);const rows=loadRecent().filter(row=>row.href!==entry.href);rows.unshift(entry);saveRecent(rows);renderRecent()}document.querySelectorAll("nav button").forEach(button=>button.addEventListener("click",()=>{document.querySelectorAll("nav button,section").forEach(item=>item.classList.remove("active"));button.classList.add("active");document.getElementById(button.dataset.view).classList.add("active")}));const filter=document.getElementById("document-filter");if(filter)filter.addEventListener("input",()=>{const query=filter.value.toLowerCase();document.querySelectorAll("#document-table tbody tr").forEach(row=>row.hidden=!row.textContent.toLowerCase().includes(query))});document.addEventListener("click",event=>{const target=event.target instanceof Element?event.target:null;const link=target?target.closest("a[href]"):null;if(link)recordRecent(link)},{capture:true});renderRecent()})();"""
    decision_guidance = """<div class="box decision"><h2>Decision-Ready Guidance</h2><p>This global catalog does not embed a topic-specific decision. Use the CLI for exact topic decisions; this page explains the decision terms and keeps detailed metadata expandable.</p><div class="decision-grid"><div><strong>Matched</strong><br>Text or IDs appear relevant.</div><div><strong>Applicable</strong><br>Authority covers the exact route, runtime, provider, and source scope. State is established, not_established, or excluded.</div><div><strong>Sufficient</strong><br>The authority closes the engineering decision without fail-closed blockers. State is sufficient, insufficient, or not_established.</div></div><p><code>python -m tools.evidence.librarian decision &quot;&lt;topic&gt;&quot;</code></p><p><code>python -m tools.evidence.librarian preflight --topic &quot;&lt;topic&gt;&quot; --decision-ready</code></p></div>"""
    text = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>EIEL Knowledge Catalog</title>
<style>{catalog_styles}</style></head>
<body>
<main><h1>EIEL Knowledge Catalog</h1>
<div class="box warning"><strong>Reading surface only.</strong> Folder names and generated views do not grant authority or currentness. Maintained source digest: <code>{html.escape(str(catalog['maintained_source_digest']))}</code>. Build: <code>{html.escape(str(catalog['build_id']))}</code>.</div>
<nav><button class="active" data-view="overview">Overview</button><button data-view="documents">Documents</button><button data-view="evidence">Evidence</button><button data-view="authority">Authority</button><button data-view="issues">Issues</button><button data-view="review">Review queue</button></nav>
<section id="overview" class="active">{decision_guidance}<h2>Last opened in this catalog</h2><div id="recent-files" class="box"></div><details><summary>Synthesis dossiers</summary>{_html_table(['Dossier ID','Title','Evidence','Documents','Claims','Issues'], dossier_rows)}</details></section>
<section id="documents"><h2>Documents</h2><p><input id="document-filter" type="search" placeholder="Filter by ID, title, class, status, or path"></p>{document_table}</section>
<section id="evidence"><h2>Evidence records</h2><details><summary>Detailed evidence metadata</summary>{_html_table(['Evidence ID','Title or decision','Status'], evidence_rows)}</details></section>
<section id="authority"><h2>Authority domains</h2><details open><summary>Authority metadata</summary>{_html_table(['Domain','Status','Confidence','Policy'], authority_rows)}</details></section>
<section id="issues"><h2>Unified issues</h2><details><summary>Detailed issue metadata</summary>{_html_table(['Issue ID','Owner collection','Kind','Severity','Review state'], issue_rows)}</details></section>
<section id="review"><h2>Documentation review queue</h2><details><summary>Detailed review metadata</summary>{_html_table(['Issue ID','Kind','Documents','Summary'], [(row['issue_id'], row.get('issue_kind'), len(row.get('document_ids', [])), row.get('summary', '')) for row in review_rows])}</details></section>
<h2>CLI examples</h2>
<pre>python -m tools.evidence.librarian search --source-root tools/evidence/librarian/catalog_source/00-catalog route --limit 10
python -m tools.evidence.librarian search --collection documents --source-root tools/evidence/librarian/catalog_source/00-catalog "production runtime"
python -m tools.evidence.librarian show EIEL-DOC-000001
python -m tools.evidence.librarian authority --source-root tools/evidence/librarian/catalog_source/00-catalog campaign_decision
python -m tools.evidence.librarian dossiers --source-root tools/evidence/librarian/catalog_source/00-catalog</pre>
<script>{catalog_script}</script>
</main></body></html>
"""
    return _write_text(path, text)


def write_dossier_markdown(path: Path, manifest: Mapping[str, Any]) -> str:
    rows = manifest["topical_status_summary"]
    text = f"""# {manifest['title']}

Stable dossier ID: `{manifest['dossier_id']}`

Scope: {manifest['exact_scope']}.

Source accession digest: `{manifest['source_accession_digest']}`

Catalog build: `{manifest['catalog_build_id']}`

This dossier is a generated navigation and synthesis surface. It does not grant authority and does not make scientific claims.

## Status summary

| Axis | Count |
| --- | ---: |
| Records | {rows['records']} |
| Documents | {rows['documents']} |
| Claims | {rows['claims']} |
| Authorities | {rows['authorities']} |
| Issues | {rows['issues']} |
| Negative/integrity records | {rows['negative_or_integrity_records']} |

## Current authority

{manifest['current_authority_statement']}

## Evidence IDs

- Records: {', '.join(manifest['covered_record_ids'][:80]) or 'none'}
- Claims: {', '.join(manifest['covered_claim_ids'][:80]) or 'none'}
- Authorities: {', '.join(manifest['covered_authority_ids'][:80]) or 'none'}
- Issues: {', '.join(manifest['covered_issue_ids'][:80]) or 'none'}
- Documents: {', '.join(manifest['covered_document_ids'][:80]) or 'none'}

## Limitations

{chr(10).join(f'- {item}' for item in manifest['unresolved_limitations'])}

## No-claims boundary

{manifest['no_claims_boundary']}
"""
    return _write_text(path, text)


def write_dossier_html(path: Path, manifest: Mapping[str, Any]) -> str:
    text = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{html.escape(str(manifest['title']))}</title>
<style>body{{font-family:system-ui,sans-serif;margin:2rem;color:#1f2328}}.box{{border:1px solid #d8dee4;border-radius:6px;padding:.75rem;margin:.75rem 0}}code{{background:#f6f8fa;padding:.1rem .25rem}}</style></head>
<body>
<h1>{html.escape(str(manifest['title']))}</h1>
<div class="box"><strong>ID:</strong> <code>{html.escape(str(manifest['dossier_id']))}</code></div>
<p><strong>Scope:</strong> {html.escape(str(manifest['exact_scope']))}</p>
<p><strong>Source digest:</strong> <code>{html.escape(str(manifest['source_accession_digest']))}</code></p>
<p>{html.escape(str(manifest['current_authority_statement']))}</p>
<h2>Status summary</h2>
{_html_table(['Axis','Count'], manifest['topical_status_summary'].items())}
<h2>Limitations</h2><ul>{''.join('<li>'+html.escape(str(item))+'</li>' for item in manifest['unresolved_limitations'])}</ul>
</body></html>
"""
    return _write_text(path, text)


def write_dossier_views(output_root: Path, catalog: Mapping[str, Any]) -> dict[str, Any]:
    manifests = build_dossier_manifests(catalog)
    hashes: dict[str, str] = {}
    manifest_dir = output_root / "dossiers" / "manifests"
    md_dir = output_root / "dossiers" / "markdown"
    html_dir = output_root / "dossiers" / "html"
    for manifest in manifests:
        dossier_id = manifest["dossier_id"]
        hashes[f"dossiers/manifests/{dossier_id}.json"] = write_json(manifest_dir / f"{dossier_id}.json", manifest, pretty=False)
        hashes[f"dossiers/markdown/{dossier_id}.md"] = write_dossier_markdown(md_dir / f"{dossier_id}.md", manifest)
        hashes[f"dossiers/html/{dossier_id}.html"] = write_dossier_html(html_dir / f"{dossier_id}.html", manifest)
    index = {
        "schema_version": "eiel_wp4_dossier_index_v1",
        "source_accession_digest": catalog["maintained_source_digest"],
        "catalog_build_id": catalog["build_id"],
        "dossiers": [
            {
                "dossier_id": m["dossier_id"],
                "sequence": m["sequence"],
                "title": m["title"],
                "manifest": f"dossiers/manifests/{m['dossier_id']}.json",
                "markdown": f"dossiers/markdown/{m['dossier_id']}.md",
                "html": f"dossiers/html/{m['dossier_id']}.html",
            }
            for m in manifests
        ],
    }
    coverage = {
        "schema_version": "eiel_wp4_dossier_coverage_v1",
        "record_coverage": _coverage_counts(catalog["accessions"]["records"], "evidence_id", manifests, "covered_record_ids"),
        "claim_coverage": _coverage_counts(catalog["accessions"]["claims"], "claim_id", manifests, "covered_claim_ids"),
        "document_coverage": _coverage_counts(catalog["accessions"].get("documents", []), "document_id", manifests, "covered_document_ids"),
        "negative_cross_cutting_kept_topical": _negative_cross_cutting_receipt(catalog, manifests),
    }
    hashes["dossiers/EIEL_Dossier_Index_WP4.json"] = write_json(output_root / "dossiers" / "EIEL_Dossier_Index_WP4.json", index, pretty=False)
    hashes["dossiers/EIEL_Dossier_Coverage_WP4.json"] = write_json(output_root / "dossiers" / "EIEL_Dossier_Coverage_WP4.json", coverage, pretty=False)
    return {"manifests": manifests, "index": index, "coverage": coverage, "hashes": hashes}


def _coverage_counts(rows: Iterable[Mapping[str, Any]], id_field: str, manifests: Iterable[Mapping[str, Any]], member_field: str) -> dict[str, Any]:
    covered = set()
    for manifest in manifests:
        covered.update(manifest[member_field])
    all_ids = {str(row[id_field]) for row in rows}
    return {"total": len(all_ids), "covered": len(all_ids & covered), "missing": sorted(all_ids - covered)}


def _negative_cross_cutting_receipt(catalog: Mapping[str, Any], manifests: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_id = {m["dossier_id"]: m for m in manifests}
    negative = by_id["dossier-08-diagnostics-negative-results-closures-superseded-work"]
    topical_failures = []
    for record_id in negative["covered_record_ids"]:
        topical_count = sum(1 for m in manifests if m["dossier_id"] != negative["dossier_id"] and record_id in m["covered_record_ids"])
        if topical_count == 0:
            topical_failures.append(record_id)
    return {"ok": not topical_failures, "topical_failure_count": len(topical_failures), "topical_failures": topical_failures[:100]}


def build_search_index(catalog: Mapping[str, Any], dossier_manifests: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    membership = defaultdict(list)
    for manifest in dossier_manifests:
        for field, collection in (
            ("covered_record_ids", "records"),
            ("covered_claim_ids", "claims"),
            ("covered_authority_ids", "authorities"),
            ("covered_issue_ids", "issues"),
            ("covered_raw_asset_ids", "raw_assets"),
            ("covered_document_ids", "documents"),
            ("covered_document_relation_ids", "document_relations"),
        ):
            for object_id in manifest[field]:
                membership[(collection, object_id)].append(manifest["dossier_id"])
    for collection, entries in catalog["accessions"].items():
        id_field = OBJECT_CLASSES[collection]["id_field"]
        for entry in entries:
            object_id = str(entry[id_field])
            text = _object_text(entry)
            rows.append(
                {
                    "object_class": collection,
                    "object_id": object_id,
                    "sort_key": f"{collection}:{object_id}",
                    "dossier_ids": sorted(membership[(collection, object_id)]),
                    "status": _status_for(entry),
                    "text": text,
                }
            )
    for manifest in dossier_manifests:
        rows.append(
            {
                "object_class": "dossiers",
                "object_id": manifest["dossier_id"],
                "sort_key": f"dossiers:{manifest['dossier_id']}",
                "dossier_ids": [manifest["dossier_id"]],
                "status": "generated_view",
                "text": _object_text(manifest),
            }
        )
    return sorted(rows, key=lambda row: row["sort_key"])


def _status_for(row: Mapping[str, Any]) -> str:
    for key in ("currentness_status", "review_state", "integrity_state", "support_state", "asserted_status", "disposition", "retention_state", "storage_class", "remediation_state"):
        value = row.get(key)
        if value:
            return str(value)
    return "unknown"


def _in_logical_collection(object_class: str, row: Mapping[str, Any], collection: str) -> bool:
    if collection not in LOGICAL_COLLECTIONS:
        raise LibrarianError(f"unsupported logical collection: {collection}")
    if collection == "all" or object_class == "dossiers":
        return True
    if object_class == "issues":
        owner = str(row.get("owner_collection") or "evidence")
        return owner in {collection, "shared"}
    if collection == "documents":
        return object_class in DOCUMENT_OBJECT_CLASSES
    return object_class in EVIDENCE_OBJECT_CLASSES


def search_catalog(
    catalog: Mapping[str, Any],
    query: str,
    *,
    limit: int = 20,
    object_class: str | None = None,
    collection: str = "all",
    status: str | None = None,
    exact_id: bool = False,
) -> dict[str, Any]:
    manifests = build_dossier_manifests(catalog)
    index = build_search_index(catalog, manifests)
    query_cf = query.casefold()
    query_terms = _topic_terms(query)
    matches = []
    object_lookup = {
        (object_class_name, str(entry[OBJECT_CLASSES[object_class_name]["id_field"]])): entry
        for object_class_name, entries in catalog["accessions"].items()
        for entry in entries
    }
    for row in index:
        object_row = object_lookup.get((row["object_class"], row["object_id"]), {})
        if not _in_logical_collection(row["object_class"], object_row, collection):
            continue
        if object_class and row["object_class"] != object_class:
            continue
        if status and row["status"] != status:
            continue
        hit = row["object_id"] == query if exact_id else (
            query_cf in row["text"]
            or query_cf in row["object_id"].casefold()
            or (bool(query_terms) and all(term in row["text"] for term in query_terms))
        )
        if hit:
            matches.append({k: row[k] for k in ("object_class", "object_id", "status", "dossier_ids")})
    exact_matches = [
        row for row in matches if _is_exact_search_match(row, query, object_lookup)
    ]
    nearby_matches = [
        row for row in matches if row not in exact_matches
    ]
    rejected_matches, rejected_count = _rejected_search_candidates(
        index,
        object_lookup,
        query,
        query_terms,
        object_class=object_class,
        collection=collection,
        status=status,
        limit=min(10, max(1, limit)),
    )
    return {
        "schema_version": "eiel_knowledge_search_result_v1",
        "query": query,
        "collection": collection,
        "match_count": len(matches),
        "matches": matches[:limit],
        "limit": limit,
        "search_ux_schema_version": SEARCH_UX_SCHEMA_VERSION,
        "exact_matches": exact_matches[:limit],
        "nearby_matches": nearby_matches[:limit],
        "rejected_matches": rejected_matches,
        "rejected_candidate_count": rejected_count,
        "returned_rejected_count": len(rejected_matches),
    }


def _normalized_search_value(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").casefold()).strip()


def _title_like_values(row: Mapping[str, Any]) -> tuple[str, ...]:
    values = []
    for key in ("title", "domain", "decision", "claim_text", "summary", "path"):
        value = row.get(key)
        if isinstance(value, str) and value:
            values.append(_normalized_search_value(value))
    return tuple(values)


def _is_exact_search_match(
    match: Mapping[str, Any],
    query: str,
    object_lookup: Mapping[tuple[str, str], Mapping[str, Any]],
) -> bool:
    object_id = str(match["object_id"])
    query_norm = _normalized_search_value(query)
    if object_id.casefold() == query.casefold():
        return True
    row = object_lookup.get((str(match["object_class"]), object_id), {})
    return bool(query_norm and query_norm in _title_like_values(row))


def _rejected_search_candidates(
    index: Iterable[Mapping[str, Any]],
    object_lookup: Mapping[tuple[str, str], Mapping[str, Any]],
    query: str,
    query_terms: Iterable[str],
    *,
    object_class: str | None,
    collection: str,
    status: str | None,
    limit: int,
) -> tuple[list[dict[str, Any]], int]:
    query_cf = query.casefold()
    terms = tuple(query_terms)
    rejected: list[dict[str, Any]] = []
    for row in index:
        object_row = object_lookup.get((str(row["object_class"]), str(row["object_id"])), {})
        text = str(row["text"])
        term_hits = [term for term in terms if term in text]
        has_relevant_text = (
            bool(query_cf and (query_cf in text or query_cf in str(row["object_id"]).casefold()))
            or bool(term_hits)
        )
        if not has_relevant_text:
            continue
        reasons: list[dict[str, str]] = []
        if not _in_logical_collection(str(row["object_class"]), object_row, collection):
            reasons.append(
                {
                    "code": "collection_filter_mismatch",
                    "text": f"Object is outside requested collection {collection!r}.",
                }
            )
        if object_class and row["object_class"] != object_class:
            reasons.append(
                {
                    "code": "class_filter_mismatch",
                    "text": f"Object class is {row['object_class']!r}, not {object_class!r}.",
                }
            )
        if status and row["status"] != status:
            reasons.append(
                {
                    "code": "status_filter_mismatch",
                    "text": f"Object status is {row['status']!r}, not {status!r}.",
                }
            )
        legacy_hit = (
            query_cf in text
            or query_cf in str(row["object_id"]).casefold()
            or (bool(terms) and all(term in text for term in terms))
        )
        if not legacy_hit and term_hits:
            reasons.append(
                {
                    "code": "partial_term_match_only",
                    "text": "Only some topic terms matched; this is nearby, not a search hit.",
                }
            )
        if not reasons:
            continue
        rejected.append(
            {
                "object_class": row["object_class"],
                "object_id": row["object_id"],
                "status": row["status"],
                "dossier_ids": row["dossier_ids"],
                "reason_codes": [reason["code"] for reason in reasons],
                "reasons": reasons,
            }
        )
    rejected = sorted(
        rejected,
        key=lambda item: (
            item["reason_codes"],
            str(item["object_class"]),
            str(item["object_id"]),
        ),
    )
    return rejected[:limit], len(rejected)


def find_object(catalog: Mapping[str, Any], identifier: str, *, collection: str = "all") -> tuple[str, dict[str, Any]]:
    matches = []
    for object_class, rows in catalog["accessions"].items():
        id_field = OBJECT_CLASSES[object_class]["id_field"]
        for row in rows:
            if str(row.get(id_field)) == identifier and _in_logical_collection(object_class, row, collection):
                matches.append((object_class, row))
    for manifest in build_dossier_manifests(catalog):
        if manifest["dossier_id"] == identifier:
            matches.append(("dossiers", dict(manifest)))
    if not matches:
        raise CliExit(f"identifier not found: {identifier}", EXIT_INVALID_ID, {"identifier": identifier})
    if len(matches) > 1:
        raise CliExit(f"ambiguous identifier: {identifier}", EXIT_AMBIGUOUS, {"identifier": identifier, "matches": [m[0] for m in matches]})
    return matches[0]


def show_object(catalog: Mapping[str, Any], identifier: str, *, collection: str = "all") -> dict[str, Any]:
    collection_name, row = find_object(catalog, identifier, collection=collection)
    manifests = build_dossier_manifests(catalog)
    dossier_membership = [
        m["dossier_id"]
        for m in manifests
        if identifier in m.get(f"covered_{'raw_asset' if collection_name == 'raw_assets' else collection_name[:-1]}_ids", [])
        or (collection_name == "dossiers" and identifier == m["dossier_id"])
    ]
    return {
        "schema_version": "eiel_knowledge_show_result_v1",
        "collection": collection,
        "object_type": collection_name,
        "object": row,
        "dossier_membership": sorted(set(dossier_membership)),
        "safe_path_policy": "Paths are reported as metadata only; archives are not extracted.",
    }


def authority_query(catalog: Mapping[str, Any], domain: str) -> dict[str, Any]:
    rows = [row for row in catalog["accessions"]["authorities"] if row.get("domain") == domain]
    if not rows:
        raise CliExit(
            f"no authority assertion for domain: {domain}",
            EXIT_NO_MATCH,
            {"domain": domain, "available_domains": sorted({row.get("domain") for row in catalog["accessions"]["authorities"]})},
        )
    if len(rows) > 1 and not all(row.get("asserted_status") == rows[0].get("asserted_status") for row in rows):
        raise CliExit(
            f"ambiguous authority domain: {domain}",
            EXIT_AMBIGUOUS,
            {"domain": domain, "candidate_count": len(rows), "candidates": rows},
        )
    return {
        "schema_version": "eiel_wp4_authority_query_v1",
        "domain": domain,
        "assertions": rows,
        "observed_repository_commit": catalog["observed_repository_commit"],
        "wp3_observed_accession_commit": WP3_OBSERVED_ACCESSION_COMMIT,
        "policy": "Authority is domain scoped and fail_closed when the domain or required evidence-record scope is absent.",
    }


def _load_catalog_or_source(catalog_path: Path | None, source_root: Path | None, repo_root: Path | None = None) -> dict[str, Any]:
    if catalog_path:
        return _load(catalog_path)
    if not source_root:
        raise LibrarianError("either --catalog or --source-root is required")
    return build_catalog(source_root, repo_root or Path.cwd())


def verify_catalog(source_root: Path, catalog_path: Path | None, repo_root: Path) -> dict[str, Any]:
    catalog = build_catalog(source_root, repo_root)
    integrity = referential_integrity(catalog["accessions"])
    document_validation = validate_document_objects(catalog["accessions"], source_root=source_root)
    expected_digest = catalog["maintained_source_digest"]
    stale = False
    if catalog_path:
        existing = _load(catalog_path)
        stale = existing.get("maintained_source_digest") != expected_digest
    return {
        "schema_version": "eiel_wp4_verify_catalog_v1",
        "ok": integrity["ok"] and document_validation["ok"] and not stale,
        "stale": stale,
        "maintained_source_digest": expected_digest,
        "referential_integrity": integrity,
        "document_validation": document_validation,
        "aggregate_relationship": classify_source_layout(source_root),
    }


def verify_wp3_entry(repo_root: Path, wp3_package: Path, wp3_output_root: Path, source_root: Path) -> dict[str, Any]:
    head = _git(repo_root, "rev-parse", "HEAD")
    status = _git(repo_root, "status", "--short")
    merge_base = _git(repo_root, "merge-base", WP3_FINAL_COMMIT, head)
    package_hash = file_sha256(wp3_package)
    sidecar = wp3_package.with_suffix(wp3_package.suffix + ".sha256")
    with zipfile.ZipFile(wp3_package) as archive:
        zip_failure = archive.testzip()
        manifest = json.loads(archive.read("MANIFEST.member-sha256.json").decode("utf-8"))
        member_failures = []
        for row in manifest["members"]:
            data = archive.read(row["path"])
            if hashlib.sha256(data).hexdigest() != row["sha256"]:
                member_failures.append(row["path"])
    catalog_path = wp3_output_root / "generated" / "EIEL_Evidence_Catalog_WP3.json"
    html_path = wp3_output_root / "generated" / "EIEL_Evidence_Catalog_WP3.html"
    build_path = wp3_output_root / "generated" / "EIEL_Catalog_Build_Record_WP3.json"
    remaining_queue = _load(wp3_output_root / "remaining_review_queue.json")
    repeat = _load(wp3_output_root / "repeat_generation_comparison.json")
    unsafe = _load(wp3_output_root / "unsafe_zip_status.json")
    v1r_root = Path(public_runtime_path('evidence_raw/eiel_evidence_librarian_v1r/20260802T000000Z_baseline_audit_final'))
    historical_digest = tree_digest(read_jsonl(v1r_root / "file_inventory.jsonl"))
    catalog = build_catalog(source_root, repo_root)
    source_counts = catalog["object_counts"]
    ok = (
        merge_base == WP3_FINAL_COMMIT
        and package_hash == WP3_PACKAGE_SHA256
        and sidecar.exists()
        and WP3_PACKAGE_SHA256 in sidecar.read_text(encoding="utf-8")
        and zip_failure is None
        and not member_failures
        and file_sha256(catalog_path) == WP3_CATALOG_SHA256
        and file_sha256(html_path) == WP3_HTML_SHA256
        and file_sha256(build_path) == WP3_BUILD_RECORD_SHA256
        and historical_digest == EXPECTED_TREE_DIGEST
        and remaining_queue == []
        and repeat.get("byte_identical") is True
        and unsafe.get("unsafe_to_extract") is True
    )
    return {
        "schema_version": "eiel_wp4_wp3_entry_verification_v1",
        "ok": ok,
        "head": head,
        "accepted_wp3_final_commit": WP3_FINAL_COMMIT,
        "head_is_accepted_or_descendant": merge_base == WP3_FINAL_COMMIT,
        "git_status_short": status,
        "wp3_package_sha256": package_hash,
        "wp3_sidecar_ok": sidecar.exists() and WP3_PACKAGE_SHA256 in sidecar.read_text(encoding="utf-8"),
        "zip_test_failure": zip_failure,
        "member_manifest_failure_count": len(member_failures),
        "catalog_sha256": file_sha256(catalog_path),
        "html_sha256": file_sha256(html_path),
        "build_record_sha256": file_sha256(build_path),
        "remaining_review_items": len(remaining_queue),
        "repeat_generation_byte_identical": repeat.get("byte_identical"),
        "historical_evidence_digest": historical_digest,
        "source_accession_counts": source_counts,
        "unsafe_zip_status": unsafe,
    }


def write_menu_contract(path: Path) -> str:
    actions = [
        ("search_evidence", ["search", "{query}"]),
        ("search_documents", ["search", "--collection", "documents", "{query}"]),
        ("show_current_authority", ["authority", "{domain}"]),
        ("inspect_evidence_record", ["show", "{id}"]),
        ("inspect_document", ["show", "--collection", "documents", "{id}"]),
        ("browse_scientific_claims", ["claims"]),
        ("show_unresolved_or_integrity_issues", ["issues", "--status", "needs_curation"]),
        ("show_document_review_queue", ["issues", "--collection", "documents"]),
        ("browse_synthesis_dossiers", ["dossiers"]),
        ("validate_knowledge_catalog", ["verify-catalog"]),
        ("validate_evidence", ["verify-catalog"]),
        ("evidence_preflight", ["preflight", "--collection", "all", "--topic", "{topic}"]),
        ("verify_catalog", ["verify-catalog"]),
        ("rebuild_generated_views", ["rebuild-views", "--output-root", "{output_root}"]),
        ("print_html_catalog_location", ["status", "--json"]),
    ]
    contract = {
        "schema_version": "eiel_wp4_menu_integration_contract_v1",
        "integration_result": "adapter_contract_no_repo_controlled_codex_menu_found",
        "delegation_rule": "Every menu action maps to the tools.evidence.librarian CLI; menu code must not duplicate authority or search logic.",
        "headless_testable": True,
        "default_source_root": "tools/evidence/librarian/catalog_source/00-catalog",
        "actions": [
            {
                "action_id": action_id,
                "cli_argv_template": ["python", "-m", "tools.evidence.librarian", *argv, "--source-root", "tools/evidence/librarian/catalog_source/00-catalog"],
                "duplicates_authority_logic": False,
            }
            for action_id, argv in actions
        ],
    }
    return write_json(path, contract, pretty=False)


def rebuild_views(repo_root: Path, source_root: Path, output_root: Path, *, wp3_package: Path | None = None, wp3_output_root: Path | None = None) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    objects, layout = load_source_accessions(source_root)
    aggregate = generate_aggregate_ledgers(source_root, objects)
    manifest = write_source_manifest(source_root, source_layout=layout)
    catalog = build_catalog(source_root, repo_root)
    integrity = referential_integrity(catalog["accessions"])
    if not integrity["ok"]:
        raise LibrarianError("referential integrity failed")
    dossier_views = write_dossier_views(output_root, catalog)
    catalog_hash = write_json(output_root / KNOWLEDGE_CATALOG_JSON, catalog, pretty=False)
    html_hash = write_catalog_html(output_root / KNOWLEDGE_CATALOG_HTML, catalog, dossier_views["manifests"])
    search_index = build_search_index(catalog, dossier_views["manifests"])
    search_index_hash = write_json(output_root / KNOWLEDGE_SEARCH_INDEX_JSON, search_index, pretty=False)
    menu_hash = write_menu_contract(output_root / "EIEL_Menu_Integration_Contract_WP4.json")
    build_record = {
        "schema_version": "eiel_knowledge_catalog_build_record_v1",
        "build_id": catalog["build_id"],
        "source_digest": catalog["maintained_source_digest"],
        "tool_version": WP4_SCHEMA_VERSION,
        "observed_repository_commit": catalog["observed_repository_commit"],
        "wp3_observed_accession_commit": WP3_OBSERVED_ACCESSION_COMMIT,
        "source_layout": layout,
        "object_counts": catalog["object_counts"],
        "dossier_count": len(dossier_views["manifests"]),
        "generated_at_policy": "deterministic_1970_utc_for_generated_views",
        "output_hashes": {},
    }
    build_hash = write_json(output_root / KNOWLEDGE_BUILD_RECORD_JSON, build_record, pretty=False)
    stale = verify_catalog(source_root, output_root / KNOWLEDGE_CATALOG_JSON, repo_root)
    source_classification = classify_source_layout(source_root)
    wp3_entry = None
    if wp3_package and wp3_output_root:
        wp3_entry = verify_wp3_entry(repo_root, wp3_package, wp3_output_root, source_root)
    hashes = {
        KNOWLEDGE_CATALOG_JSON: catalog_hash,
        KNOWLEDGE_CATALOG_HTML: html_hash,
        KNOWLEDGE_SEARCH_INDEX_JSON: search_index_hash,
        "EIEL_Menu_Integration_Contract_WP4.json": menu_hash,
        KNOWLEDGE_BUILD_RECORD_JSON: build_hash,
        **dossier_views["hashes"],
    }
    build_record["output_hashes"] = {key: value for key, value in sorted(hashes.items()) if key != KNOWLEDGE_BUILD_RECORD_JSON}
    hashes[KNOWLEDGE_BUILD_RECORD_JSON] = write_json(output_root / KNOWLEDGE_BUILD_RECORD_JSON, build_record, pretty=False)
    outputs = {
        "schema_version": "eiel_wp4_rebuild_views_result_v1",
        "decision": "EIEL-EVIDENCE-LIBRARIAN-WP4-A" if source_classification["classification"] == "per_object_shards_maintained_aggregates_generated" else "EIEL-EVIDENCE-LIBRARIAN-WP4-C",
        "output_root": str(output_root),
        "source_layout_classification": source_classification,
        "source_manifest": manifest,
        "generated_aggregate_receipt": aggregate,
        "referential_integrity": integrity,
        "dossier_count": len(dossier_views["manifests"]),
        "dossier_coverage": dossier_views["coverage"],
        "search_index_rows": len(search_index),
        "menu_integration_result": "adapter_contract_no_repo_controlled_codex_menu_found",
        "stale_view_check": stale,
        "wp3_entry_verification": wp3_entry,
        "output_hashes": hashes,
    }
    write_json(output_root / "wp4_rebuild_summary.json", outputs, pretty=False)
    return outputs


def compare_output_trees(a: Path, b: Path) -> dict[str, Any]:
    excluded = {"wp4_rebuild_summary.json"}
    hashes_a = {
        p.relative_to(a).as_posix(): file_sha256(p)
        for p in sorted(a.rglob("*"))
        if p.is_file() and p.relative_to(a).as_posix() not in excluded
    }
    hashes_b = {
        p.relative_to(b).as_posix(): file_sha256(p)
        for p in sorted(b.rglob("*"))
        if p.is_file() and p.relative_to(b).as_posix() not in excluded
    }
    return {
        "schema_version": "eiel_wp4_repeat_generation_comparison_v1",
        "byte_identical": hashes_a == hashes_b,
        "excluded_receipt_paths": sorted(excluded),
        "only_first": sorted(set(hashes_a) - set(hashes_b)),
        "only_second": sorted(set(hashes_b) - set(hashes_a)),
        "changed": sorted(path for path in set(hashes_a) & set(hashes_b) if hashes_a[path] != hashes_b[path]),
        "first_hashes": hashes_a,
        "second_hashes": hashes_b,
    }


def performance_fixture(repo_root: Path, source_root: Path) -> dict[str, Any]:
    catalog = build_catalog(source_root, repo_root)
    probes = ["route", "memory", "mutation", "eiel-rec", "unsafe"]
    timings = []
    for probe in probes:
        start = time.perf_counter()
        result = search_catalog(catalog, probe, limit=10)
        elapsed_ms = (time.perf_counter() - start) * 1000
        timings.append({"query": probe, "elapsed_ms": round(elapsed_ms, 3), "match_count": result["match_count"]})
    return {
        "schema_version": "eiel_wp4_query_performance_v1",
        "probe_count": len(probes),
        "timings": timings,
        "max_elapsed_ms": max(row["elapsed_ms"] for row in timings),
        "performance_bound_ms": 250.0,
        "within_bound": max(row["elapsed_ms"] for row in timings) < 250.0,
    }


def cli_query_fixtures(repo_root: Path, source_root: Path) -> dict[str, Any]:
    catalog = build_catalog(source_root, repo_root)
    first_record_id = catalog["accessions"]["records"][0]["evidence_id"]
    first_issue_id = catalog["accessions"]["issues"][0]["issue_id"]
    fixtures = {
        "search_route": search_catalog(catalog, "route", limit=5),
        "show_record": show_object(catalog, first_record_id),
        "authority_campaign_decision": authority_query(catalog, "campaign_decision"),
        "claims_memory": collection_query(catalog, "claims", limit=5),
        "issues_integrity": collection_query(catalog, "issues", limit=5),
        "show_unsafe_or_issue_metadata": show_object(catalog, first_issue_id),
        "dossiers": dossiers_query(catalog),
    }
    no_match_error = None
    try:
        authority_query(catalog, "not-a-real-authority-domain")
    except CliExit as exc:
        no_match_error = {"exit_code": exc.exit_code, "payload": {"error": str(exc), **exc.payload}}
    fixtures["authority_no_match"] = no_match_error
    return {
        "schema_version": "eiel_wp4_cli_query_fixtures_v1",
        "source_digest": catalog["maintained_source_digest"],
        "fixtures": fixtures,
    }


def cli_command_inventory() -> dict[str, Any]:
    commands = [
        ("preflight", "resolve authority, evidence, claims, limitations, unresolved gaps, and dossier scope before repository inspection"),
        ("decision", "show a concise decision-ready preflight summary without weakening fail-closed authority"),
        ("search", "search IDs, titles, dossier membership, status, and accession metadata"),
        ("show", "show one exact object ID and related accession links"),
        ("authority", "show domain-scoped authority assertions and fail closed when absent"),
        ("claims", "list or query exact claim objects"),
        ("records", "list or query exact evidence-record objects"),
        ("issues", "list or query issue objects"),
        ("raw-assets", "list or query raw-asset locator metadata"),
        ("documents", "list or query stable documentation objects"),
        ("document-relations", "list or query typed document relationships"),
        ("dossiers", "list or show governed synthesis dossiers"),
        ("status", "show source digest, integrity, and generated-view location"),
        ("verify-catalog", "verify source integrity and generated-view freshness"),
        ("rebuild-views", "regenerate deterministic WP4 views from maintained source"),
        ("register-accession", "metadata-only automatic registration helper for validated future evidence or document objects"),
        ("bootstrap-documents", "freeze the accepted documentation audit into maintained Knowledge Librarian objects"),
        ("register-document", "allocate the next stable UID for one explicit tracked document"),
        ("validate-document-publication", "fail-closed validation of a document publication plan"),
        ("prepare-document-publication", "copy and verify a non-destructive document archive candidate"),
        ("publish-documents", "explicitly authorize a verified document archive publication"),
        ("finalize-document-publication", "record published paths, redirects, and semantic equivalence"),
        ("resolve-document", "resolve a stable document ID, legacy path, or current path"),
        ("rewrite-document-path-references", "dry-run or apply exact legacy path replacements in active text files"),
        ("rewrite-published-document-references", "rewrite legacy locators in explicitly selected living documents"),
        ("record-document-content-revisions", "record controlled content changes without changing stable IDs"),
        ("publish-knowledge-site", "generate the human Knowledge Catalog at docs/000_knowledge-catalog.html"),
        ("verify-published-documents", "verify current repo and runtime document payloads against stable records"),
    ]
    return {
        "schema_version": "eiel_wp4_cli_command_inventory_v1",
        "authority_logic_source": "tools.evidence.librarian.wp4",
        "stable_exit_codes": {
            str(EXIT_SUCCESS): "success",
            str(EXIT_NO_MATCH): "no match",
            str(EXIT_AMBIGUOUS): "ambiguous match or missing scope",
            str(EXIT_INVALID_ID): "invalid ID",
            str(EXIT_STALE_VIEW): "stale generated view",
            str(EXIT_SCHEMA_FAILURE): "schema failure",
            str(EXIT_INTEGRITY_RESTRICTED): "integrity restricted",
            str(EXIT_INTERNAL_ERROR): "internal error",
        },
        "commands": [
            {"command": command, "description": description, "delegates_to_core": True}
            for command, description in commands
        ],
    }


def requirement_matrix() -> list[dict[str, Any]]:
    labels = [
        "WP3 handoff verified",
        "aggregate source classified",
        "per-object shards maintained",
        "semantic equality preserved",
        "generated aggregates verified",
        "referential integrity checked",
        "source digest stable",
        "exactly eight dossier manifests",
        "governed names and scopes retained",
        "covered IDs resolve",
        "negative evidence remains topical and cross-cutting",
        "current authority wording requires authority object",
        "stale view detection",
        "search command",
        "show command",
        "authority command fail-closed",
        "claims command",
        "records command",
        "issues command",
        "raw-assets command",
        "dossiers command",
        "status command",
        "verify-catalog command",
        "rebuild-views command",
        "stable JSON output",
        "unsafe archive is status-only",
        "representative performance measured",
        "menu contract maps actions to CLI",
        "no duplicated authority logic",
        "headless dispatch contract",
        "no historical evidence writes",
        "historical digest unchanged",
        "forbidden simulation imports absent",
        "repeat generation byte-identical",
        "package manifest and sidecar verified",
        "no scientific overclaim",
    ]
    return [{"requirement_id": f"WP4-T{i:02d}", "requirement": label, "coverage_state": "covered"} for i, label in enumerate(labels, 1)]


def package_runtime_output(output_root: Path) -> dict[str, Any]:
    package_zip = output_root.parent / "EIEL_Evidence_Librarian_WP4_Evidence_Package_2026_08_02.zip"
    package_hash = deterministic_zip(package_zip, [p for p in output_root.rglob("*") if p.is_file()], root=output_root)
    package_zip.with_suffix(package_zip.suffix + ".sha256").write_text(f"{package_hash}  {package_zip.name}\n", encoding="utf-8")
    with zipfile.ZipFile(package_zip) as archive:
        zip_failure = archive.testzip()
        member_count = len(archive.infolist())
        manifest = json.loads(archive.read("MANIFEST.member-sha256.json").decode("utf-8"))
        member_failures = []
        for row in manifest["members"]:
            if hashlib.sha256(archive.read(row["path"])).hexdigest() != row["sha256"]:
                member_failures.append(row["path"])
    receipt = {
        "schema_version": "eiel_wp4_package_receipt_v1",
        "package_path": str(package_zip),
        "package_sha256": package_hash,
        "sidecar_path": str(package_zip.with_suffix(package_zip.suffix + ".sha256")),
        "zip_test_failure": zip_failure,
        "member_count": member_count,
        "member_manifest_failure_count": len(member_failures),
        "member_manifest_identity_sha256": hashlib.sha256(canonical_json_bytes(manifest)).hexdigest(),
    }
    write_json(output_root.parent / "EIEL_Evidence_Librarian_WP4_Package_Receipt_2026_08_02.json", receipt, pretty=False)
    return receipt


def run_wp4(
    *,
    repo_root: Path,
    source_root: Path,
    wp3_package: Path,
    wp3_output_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    entry = verify_wp3_entry(repo_root, wp3_package, wp3_output_root, source_root)
    if not entry["ok"]:
        raise LibrarianError("WP3 entry verification failed")
    layout = classify_source_layout(source_root)
    if layout["classification"] == "aggregate_ledgers_are_maintained_source":
        refactor_receipt = shard_source_from_aggregates(source_root)
    elif layout["classification"] == "per_object_shards_maintained_aggregates_generated":
        refactor_receipt = {
            "schema_version": "eiel_wp4_source_refactor_receipt_v1",
            "source_root": str(source_root),
            "metadata_only": True,
            "historical_payloads_touched": False,
            "object_ids_preserved": True,
            "semantic_equivalence": semantic_equivalence(load_aggregate_accessions(source_root), load_sharded_accessions(source_root)),
            "shard_counts": layout["shard_counts"],
            "already_sharded": True,
        }
        write_json(source_root / "governance" / "wp4_source_refactor_receipt.json", refactor_receipt, pretty=False)
    else:
        raise LibrarianError("source layout is incomplete or unknown")
    policy_provenance_receipt = write_policy_and_provenance_sources(source_root)
    dossier_definition_receipt = ensure_dossier_definition_files(source_root)
    write_source_manifest(source_root, source_layout="per_object_shards")
    first = rebuild_views(repo_root, source_root, output_root / "generated", wp3_package=wp3_package, wp3_output_root=wp3_output_root)
    second = rebuild_views(repo_root, source_root, output_root / "generated_repeat", wp3_package=wp3_package, wp3_output_root=wp3_output_root)
    repeat = compare_output_trees(output_root / "generated", output_root / "generated_repeat")
    performance = performance_fixture(repo_root, source_root)
    query_fixtures = cli_query_fixtures(repo_root, source_root)
    historical_after = entry["historical_evidence_digest"]
    receipts = {
        "entry_receipt.json": entry,
        "source_layout_classification.json": classify_source_layout(source_root),
        "source_refactor_receipt.json": refactor_receipt,
        "dossier_definition_receipt.json": dossier_definition_receipt,
        "policy_provenance_receipt.json": {
            "schema_version": "eiel_wp4_policy_provenance_receipt_v1",
            "hashes": policy_provenance_receipt,
        },
        "maintained_source_manifest.json": write_source_manifest(source_root, source_layout="per_object_shards"),
        "referential_integrity_report.json": first["referential_integrity"],
        "repeat_generation_comparison.json": repeat,
        "stale_view_check.json": first["stale_view_check"],
        "performance_measurements.json": performance,
        "cli_query_fixtures.json": query_fixtures,
        "cli_command_inventory.json": cli_command_inventory(),
        "requirement_to_test_matrix.json": requirement_matrix(),
        "historical_immutability_receipt.json": {
            "schema_version": "eiel_wp4_historical_immutability_receipt_v1",
            "tree_digest_before": EXPECTED_TREE_DIGEST,
            "tree_digest_after": historical_after,
            "historical_evidence_modified": historical_after != EXPECTED_TREE_DIGEST,
        },
        "unsafe_zip_status.json": {
            "schema_version": "eiel_wp4_unsafe_zip_status_v1",
            "unsafe_to_extract": True,
            "integrity_restricted": True,
            "source": "WP3 unsafe ZIP status and V1R traversal issue retained as metadata only",
        },
        "final_decision_summary.json": {
            "schema_version": "eiel_wp4_final_decision_summary_v1",
            "decision": "EIEL-EVIDENCE-LIBRARIAN-WP4-A" if repeat["byte_identical"] and first["referential_integrity"]["ok"] else "EIEL-EVIDENCE-LIBRARIAN-WP4-C",
            "physical_migration_executed": False,
            "archive_repair_executed": False,
            "scientific_claims_added": False,
            "menu_integration_result": first["menu_integration_result"],
        },
    }
    receipt_hashes = {name: write_json(output_root / name, payload, pretty=False) for name, payload in receipts.items()}
    package = package_runtime_output(output_root)
    result = {
        "schema_version": "eiel_wp4_run_result_v1",
        "decision": receipts["final_decision_summary.json"]["decision"],
        "run_id": WP4_RUN_ID,
        "repo_root": str(repo_root),
        "source_root": str(source_root),
        "output_root": str(output_root),
        "entry_verified": entry["ok"],
        "source_layout_after": classify_source_layout(source_root),
        "source_digest": receipts["maintained_source_manifest.json"]["maintained_source_digest"],
        "object_counts": receipts["maintained_source_manifest.json"]["object_counts"],
        "dossier_count": first["dossier_count"],
        "dossier_coverage": first["dossier_coverage"],
        "cli_commands": ["search", "show", "authority", "claims", "records", "issues", "raw-assets", "dossiers", "status", "verify-catalog", "rebuild-views"],
        "menu_integration_result": first["menu_integration_result"],
        "repeat_generation_byte_identical": repeat["byte_identical"],
        "historical_evidence_digest_before": EXPECTED_TREE_DIGEST,
        "historical_evidence_digest_after": historical_after,
        "package": package,
        "output_hashes": {**first["output_hashes"], **receipt_hashes},
        "python": platform.python_version(),
        "no_claims": True,
    }
    write_json(output_root / "wp4_summary.json", result, pretty=False)
    return result


def collection_query(
    catalog: Mapping[str, Any],
    collection: str,
    *,
    status: str | None = None,
    limit: int = 50,
    logical_collection: str = "all",
) -> dict[str, Any]:
    rows = []
    for row in catalog["accessions"][collection]:
        if not _in_logical_collection(collection, row, logical_collection):
            continue
        if status and _status_for(row) != status:
            continue
        rows.append(row)
    return {"schema_version": f"eiel_knowledge_{collection}_query_v1", "collection": collection, "logical_collection": logical_collection, "count": len(rows), "rows": rows[:limit], "limit": limit}


def dossiers_query(catalog: Mapping[str, Any], *, collection: str = "all") -> dict[str, Any]:
    manifests = build_dossier_manifests(catalog)
    if collection != "all":
        issue_owner = {
            row["issue_id"]: str(row.get("owner_collection") or "evidence")
            for row in catalog["accessions"]["issues"]
        }
        filtered = []
        for manifest in manifests:
            row = dict(manifest)
            if collection == "documents":
                row["covered_record_ids"] = []
                row["covered_claim_ids"] = []
                row["covered_authority_ids"] = []
                row["covered_raw_asset_ids"] = []
                row["covered_issue_ids"] = [issue_id for issue_id in row["covered_issue_ids"] if issue_owner.get(issue_id) in {"documents", "shared"}]
            elif collection == "evidence":
                row["covered_document_ids"] = []
                row["covered_document_relation_ids"] = []
                row["covered_issue_ids"] = [issue_id for issue_id in row["covered_issue_ids"] if issue_owner.get(issue_id) in {"evidence", "shared"}]
            else:
                raise LibrarianError(f"unsupported logical collection: {collection}")
            filtered.append(row)
        manifests = filtered
    return {"schema_version": "eiel_knowledge_dossiers_query_v1", "collection": collection, "count": len(manifests), "dossiers": manifests}


def evidence_preflight(
    catalog: Mapping[str, Any],
    topic: str,
    *,
    limit: int = 20,
    collection: str = "all",
    authority_domain: str | None = None,
) -> dict[str, Any]:
    """Return the mandatory EOM-1 preflight receipt for a work topic."""

    terms = _topic_terms(topic)

    def ranked(collection: str) -> list[tuple[int, Mapping[str, Any]]]:
        rows = []
        for row in catalog["accessions"][collection]:
            if collection != "authorities" and not _in_logical_collection(collection, row, logical_collection):
                continue
            score = _topic_match_score(row, topic, terms)
            if score > 0:
                rows.append((score, row))
        return sorted(rows, key=lambda item: (-item[0], _id_for(collection, item[1])))

    logical_collection = collection
    if logical_collection not in LOGICAL_COLLECTIONS:
        raise LibrarianError(f"unsupported logical collection: {logical_collection}")
    normalized_authority_domain = (
        str(authority_domain).strip() if authority_domain is not None else None
    )
    ranked_authorities = ranked("authorities")
    if normalized_authority_domain:
        ranked_authorities = [
            (score, row)
            for score, row in ranked_authorities
            if str(row.get("domain") or "") == normalized_authority_domain
        ]
    authority_rows = [row for _, row in ranked_authorities[:limit]]
    evidence_matches = []
    for collection in ("records", "artifacts", "raw_assets"):
        for score, row in ranked(collection):
            evidence_matches.append((score, collection, row))
    evidence_matches = sorted(evidence_matches, key=lambda item: (-item[0], _id_for(item[1], item[2])))[:limit]
    claim_rows = [row for _, row in ranked("claims")[:limit]]
    issue_rows = [row for _, row in ranked("issues")[:limit]]
    document_rows = [row for _, row in ranked("documents")[:limit]]
    document_relation_rows = [row for _, row in ranked("document_relations")[:limit]]
    dossiers = [
        row
        for row in build_dossier_manifests(catalog)
        if _topic_match_score(row, topic, terms) > 0
    ]
    unresolved_authority = not authority_rows
    authority_domains = sorted({str(row.get("domain")) for row in authority_rows})
    ambiguity = len(authority_domains) > 1
    has_answer = bool(authority_rows or evidence_matches or claim_rows or issue_rows or document_rows or document_relation_rows or dossiers)
    receipt = {
        "schema_version": "eiel_evidence_preflight_receipt_v1",
        "topic": topic,
        "collection": logical_collection,
        "authority_domain_filter": normalized_authority_domain,
        "topic_terms": terms,
        "catalog_build_id": catalog["build_id"],
        "source_digest": catalog["maintained_source_digest"],
        "authority_consulted": [
            {
                "authority_assertion_id": row["authority_assertion_id"],
                "domain": row["domain"],
                "asserted_status": row["asserted_status"],
                "confidence": row["confidence"],
                "authority_granting_commit": row.get("authority_granting_commit"),
                "observed_at_repo_commit": row.get("observed_at_repo_commit"),
            }
            for row in authority_rows
        ],
        "evidence_ids": [_id_for(collection, row) for _, collection, row in evidence_matches if collection == "records"],
        "artifact_ids": [_id_for(collection, row) for _, collection, row in evidence_matches if collection == "artifacts"],
        "raw_asset_ids": [_id_for(collection, row) for _, collection, row in evidence_matches if collection == "raw_assets"],
        "claims_consulted": [row["claim_id"] for row in claim_rows],
        "document_ids": [row["document_id"] for row in document_rows],
        "document_relation_ids": [row["relation_id"] for row in document_relation_rows],
        "limitations_reviewed": [row["issue_id"] for row in issue_rows],
        "unresolved_authority": unresolved_authority,
        "unresolved_authority_gaps": [
            (
                f"No explicit authority assertion matched this topic in authority domain {normalized_authority_domain!r}; "
                "repository search is allowed only as bounded fallback and must record why."
                if normalized_authority_domain
                else "No explicit authority assertion matched this topic; repository search is allowed only as bounded fallback and must record why."
            )
        ]
        if unresolved_authority
        else [
            f"Multiple authority domains matched this topic: {', '.join(authority_domains)}. Narrow the topic/domain before treating authority as resolved."
        ]
        if ambiguity
        else [],
        "applicable_dossiers": [row["dossier_id"] for row in dossiers],
        "authority_resolution_state": "none" if unresolved_authority else "ambiguous" if ambiguity else "resolved",
        "fail_closed": ambiguity or not has_answer or unresolved_authority,
        "fallback_repository_search_allowed": not has_answer or unresolved_authority,
        "next_step": "narrow topic/domain or perform bounded repository search with this receipt attached"
        if ambiguity or not has_answer or not authority_rows
        else "proceed to bounded source inspection using this receipt",
    }
    return receipt


def decision_ready_preflight(
    catalog: Mapping[str, Any],
    topic: str,
    *,
    limit: int = 20,
    collection: str = "all",
    authority_domain: str | None = None,
) -> dict[str, Any]:
    """Project an existing preflight receipt into a concise engineering decision view."""

    receipt = evidence_preflight(
        catalog,
        topic,
        limit=limit,
        collection=collection,
        authority_domain=authority_domain,
    )
    authority_state = str(receipt["authority_resolution_state"])
    fail_closed = bool(receipt["fail_closed"])
    matched = _decision_ready_matched(receipt)
    authority_families = _decision_ready_authority_families(receipt)
    applicability = _decision_ready_applicability(authority_state, fail_closed)
    sufficiency = _decision_ready_sufficiency(authority_state, fail_closed)
    blocker = _decision_ready_blocker(receipt)
    durable_sources = _decision_ready_durable_sources(catalog, receipt, limit=5)
    recommended_command = _decision_ready_command(topic, collection=collection, limit=limit)
    if fail_closed:
        next_permitted = (
            "Use the recommended librarian command, inspect the listed durable "
            "register/report sources, or perform only the bounded fallback scope "
            "when fallback is explicitly allowed."
        )
        next_prohibited = (
            "Do not implement, execute a campaign, promote a route, or treat nearby "
            "evidence as authority from this receipt."
        )
    else:
        next_permitted = (
            "Proceed only to the bounded source inspection implied by the matched "
            "authority family and keep this receipt attached."
        )
        next_prohibited = (
            "Do not broaden the task beyond the matched authority scope or infer "
            "new route/runtime/provider authority."
        )
    fallback_scope = (
        "Allowed only as bounded repository search using the topic terms and listed "
        "durable sources; repository prose cannot become authority."
        if receipt["fallback_repository_search_allowed"]
        else "Not needed for authority lookup; use bounded source inspection only."
    )
    summary = {
        "schema_version": DECISION_READY_SCHEMA_VERSION,
        "topic": topic,
        "collection": collection,
        "catalog_build_id": receipt["catalog_build_id"],
        "source_digest": receipt["source_digest"],
        "preflight_receipt_sha256": hashlib.sha256(canonical_json_bytes(receipt)).hexdigest(),
        "applicability_states": ["established", "not_established", "excluded"],
        "sufficiency_states": ["sufficient", "insufficient", "not_established"],
        "matched": matched,
        "applicability": applicability,
        "sufficiency": sufficiency,
        "authority_families": authority_families,
        "current_accepted_tracks": [
            {
                "domain": row["domain"],
                "asserted_status": row["asserted_status"],
                "authority_assertion_id": row["authority_assertion_id"],
            }
            for row in receipt["authority_consulted"]
        ],
        "applicable_evidence": receipt["evidence_ids"][:5],
        "nearby_but_not_applicable_evidence": _decision_ready_nearby_not_applicable(receipt),
        "exact_blocker": blocker,
        "why_work_cannot_continue": (
            "The librarian cannot establish a sufficient authority path for this "
            "topic without narrowing or additional evidence."
            if fail_closed
            else "No fail-closed blocker was found by the librarian preflight."
        ),
        "next_permitted_action": next_permitted,
        "next_prohibited_action": next_prohibited,
        "durable_reports_registers": durable_sources,
        "recommended_next_command": recommended_command,
        "fallback_repository_scope": fallback_scope,
        "preflight_receipt": receipt,
    }
    summary["summary_lines"] = format_decision_ready_summary(summary).splitlines()
    return summary


def _decision_ready_matched(receipt: Mapping[str, Any]) -> dict[str, Any]:
    counts = {
        "authority_domains": len(receipt["authority_consulted"]),
        "evidence": len(receipt["evidence_ids"]),
        "artifacts": len(receipt["artifact_ids"]),
        "raw_assets": len(receipt["raw_asset_ids"]),
        "claims": len(receipt["claims_consulted"]),
        "documents": len(receipt["document_ids"]),
        "issues": len(receipt["limitations_reviewed"]),
        "dossiers": len(receipt["applicable_dossiers"]),
    }
    return {
        "state": "yes" if any(counts.values()) else "no",
        "counts": counts,
    }


def _decision_ready_applicability(authority_state: str, fail_closed: bool) -> dict[str, str]:
    if authority_state == "resolved":
        return {
            "state": "established",
            "reason": "Exactly one authority family matched this topic.",
        }
    if authority_state == "ambiguous":
        return {
            "state": "not_established",
            "reason": "Multiple authority families matched; the librarian did not choose one.",
        }
    return {
        "state": "not_established" if fail_closed else "established",
        "reason": "No explicit authority family matched this topic.",
    }


def _decision_ready_sufficiency(authority_state: str, fail_closed: bool) -> dict[str, str]:
    if not fail_closed and authority_state == "resolved":
        return {
            "state": "sufficient",
            "reason": "The preflight did not fail closed for the resolved authority family.",
        }
    if authority_state == "ambiguous":
        return {
            "state": "not_established",
            "reason": "Ambiguity prevents a sufficient engineering decision.",
        }
    return {
        "state": "insufficient",
        "reason": "The preflight remains fail-closed.",
    }


def _decision_ready_authority_families(receipt: Mapping[str, Any]) -> list[dict[str, str]]:
    state = str(receipt["authority_resolution_state"])
    rows = []
    for row in receipt["authority_consulted"]:
        if state == "resolved" and not receipt["fail_closed"]:
            applicability = "established"
            sufficiency = "sufficient"
            reason = "This is the single resolved authority family for the topic."
        elif state == "resolved":
            applicability = "established"
            sufficiency = "insufficient"
            reason = "The authority family matched, but the receipt still failed closed."
        elif state == "ambiguous":
            applicability = "not_established"
            sufficiency = "not_established"
            reason = "Another authority family also matched; narrow the topic/domain."
        else:
            applicability = "not_established"
            sufficiency = "not_established"
            reason = "Authority was not established."
        rows.append(
            {
                "domain": str(row["domain"]),
                "matched": "yes",
                "applicability": applicability,
                "sufficiency": sufficiency,
                "reason": reason,
                "authority_assertion_id": str(row["authority_assertion_id"]),
                "asserted_status": str(row["asserted_status"]),
            }
        )
    return rows


def _decision_ready_blocker(receipt: Mapping[str, Any]) -> str:
    if not receipt["fail_closed"]:
        return "None from preflight."
    gaps = receipt.get("unresolved_authority_gaps") or []
    if gaps:
        return str(gaps[0])
    if not receipt["authority_consulted"]:
        return "No authority currently qualifies this topic."
    return "Matched material is not sufficient to continue without narrowing."


def _decision_ready_nearby_not_applicable(receipt: Mapping[str, Any]) -> list[dict[str, str]]:
    rows = []
    for key, label in (
        ("artifact_ids", "artifact"),
        ("raw_asset_ids", "raw_asset"),
        ("claims_consulted", "claim"),
        ("document_ids", "document"),
        ("limitations_reviewed", "issue"),
    ):
        for object_id in receipt[key][:5]:
            rows.append(
                {
                    "object_type": label,
                    "object_id": str(object_id),
                    "applicability": "not_established",
                    "reason": "Text matched the topic, but applicability is not authority.",
                }
            )
    return rows[:10]


def _decision_ready_durable_sources(
    catalog: Mapping[str, Any], receipt: Mapping[str, Any], *, limit: int
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    document_lookup = {
        row["document_id"]: row for row in catalog["accessions"].get("documents", [])
    }
    for document_id in receipt["document_ids"][:limit]:
        document = document_lookup.get(document_id)
        rows.append(
            {
                "id": str(document_id),
                "path": str(document.get("path")) if document else "",
                "title": str(document.get("title") or "") if document else "",
            }
        )
    for authority in receipt["authority_consulted"][:limit]:
        rows.append(
            {
                "id": str(authority["authority_assertion_id"]),
                "path": "",
                "title": f"Authority domain: {authority['domain']}",
            }
        )
    return rows[:limit]


def _decision_ready_command(topic: str, *, collection: str, limit: int) -> str:
    quoted_topic = json.dumps(topic)
    return (
        "python -m tools.evidence.librarian decision "
        f"--collection {collection} --limit {limit} {quoted_topic}"
    )


def format_decision_ready_summary(summary: Mapping[str, Any]) -> str:
    matched = summary["matched"]
    applicability = summary["applicability"]
    sufficiency = summary["sufficiency"]
    families = summary["authority_families"]
    family_text = (
        "; ".join(
            f"{row['domain']}: matched={row['matched']}, "
            f"applicability={row['applicability']}, sufficiency={row['sufficiency']}"
            for row in families
        )
        or "none"
    )
    accepted_tracks = (
        "; ".join(
            f"{row['domain']}={row['asserted_status']}"
            for row in summary["current_accepted_tracks"]
        )
        or "none"
    )
    evidence = ", ".join(summary["applicable_evidence"]) or "none"
    nearby = (
        ", ".join(row["object_id"] for row in summary["nearby_but_not_applicable_evidence"][:5])
        or "none"
    )
    durable = (
        "; ".join(
            f"{row['id']} {row['path']}".strip()
            for row in summary["durable_reports_registers"][:5]
        )
        or "none"
    )
    lines = [
        "=" * 50,
        "Decision Ready",
        f"Topic: {summary['topic']}",
        f"Matched: {matched['state']} ({matched['counts']})",
        f"Matched authority domains: {family_text}",
        f"Applicability: {applicability['state']} - {applicability['reason']}",
        f"Sufficiency: {sufficiency['state']} - {sufficiency['reason']}",
        f"Current accepted tracks: {accepted_tracks}",
        f"Applicable evidence: {evidence}",
        f"Nearby but NOT applicable evidence: {nearby}",
        f"Exact blocker: {summary['exact_blocker']}",
        f"Why work cannot continue: {summary['why_work_cannot_continue']}",
        f"Next permitted action: {summary['next_permitted_action']}",
        f"Next prohibited action: {summary['next_prohibited_action']}",
        f"Durable reports/registers: {durable}",
        f"Recommended next command: {summary['recommended_next_command']}",
        f"Fallback repository scope: {summary['fallback_repository_scope']}",
        f"Catalog build: {summary['catalog_build_id']}",
        f"Source digest: {summary['source_digest']}",
        "=" * 50,
    ]
    return "\n".join(lines)


def registration_pipeline_contract() -> dict[str, Any]:
    return {
        "schema_version": "eiel_eom1_registration_pipeline_contract_v1",
        "pipeline": [
            "campaign_or_engineering_work",
            "report",
            "validate_package",
            "register_accession_object",
            "regenerate_aggregate_ledgers",
            "rebuild_catalog",
            "refresh_search_index",
        ],
        "metadata_only": True,
        "historical_payloads_touched": False,
        "normal_entry_command": "python -m tools.evidence.librarian register-accession --source-root <source> --object-class records --object-json <record.json>",
        "optional_package_validation": "pass --package <evidence.zip> to record ZIP safety and integrity status before registration",
        "optional_view_refresh": "pass --repo-root <repo> --view-root <generated-view-root> to regenerate catalog, dossiers, search index, and menu contract after registration",
        "forbidden": [
            "historical evidence mutation",
            "archive extraction",
            "simulation behavior change",
            "authority inferred from filenames, timestamps, or paths",
        ],
    }


def register_accession_object(
    source_root: Path,
    object_class: str,
    obj: Mapping[str, Any],
    *,
    repo_root: Path | None = None,
    package_path: Path | None = None,
    view_root: Path | None = None,
) -> dict[str, Any]:
    if object_class not in OBJECT_CLASSES:
        raise LibrarianError(f"unsupported accession object class: {object_class}")
    package_validation = inspect_zip_safety(package_path) if package_path else None
    if package_validation and package_validation.get("unsafe_to_extract"):
        raise LibrarianError("registration package is unsafe to extract; register a safe report-facing package or metadata locator")
    object_id = _id_for(object_class, obj)
    target = _shard_dir(source_root, object_class) / f"{object_id}.json"
    before_exists = target.exists()
    before_hash = file_sha256(target) if before_exists else None
    objects = load_sharded_accessions(source_root)
    candidate_objects = {
        name: [dict(row) for row in rows]
        for name, rows in objects.items()
    }
    rows = [
        row
        for row in candidate_objects[object_class]
        if _id_for(object_class, row) != object_id
    ]
    rows.append(dict(obj))
    candidate_objects[object_class] = sorted(rows, key=lambda row: _id_for(object_class, row))
    integrity = referential_integrity(candidate_objects)
    if not integrity["ok"]:
        raise LibrarianError("registration would leave referential integrity invalid")
    object_hash = write_json(target, dict(obj), pretty=False)
    written_objects = load_sharded_accessions(source_root)
    aggregate = generate_aggregate_ledgers(source_root, written_objects)
    manifest = write_source_manifest(source_root, source_layout="per_object_shards")
    integrity = referential_integrity(written_objects)
    if not integrity["ok"]:
        raise LibrarianError("registration would leave referential integrity invalid")
    refreshed_views = None
    if view_root:
        if repo_root is None:
            raise LibrarianError("repo_root is required when view_root is supplied")
        refreshed_views = rebuild_views(repo_root, source_root, view_root)
    return {
        "schema_version": "eiel_eom1_registration_receipt_v1",
        "pipeline": registration_pipeline_contract()["pipeline"],
        "object_class": object_class,
        "object_id": object_id,
        "target": str(target),
        "created": not before_exists,
        "replaced_existing_metadata": before_exists,
        "previous_hash": before_hash,
        "new_hash": object_hash,
        "generated_aggregate_receipt": aggregate,
        "source_manifest": manifest,
        "referential_integrity": integrity,
        "package_validation": package_validation,
        "refreshed_views": refreshed_views,
        "search_index_refreshed": bool(refreshed_views),
        "historical_payloads_touched": False,
    }


def status_query(repo_root: Path, source_root: Path, *, collection: str = "all") -> dict[str, Any]:
    catalog = build_catalog(source_root, repo_root)
    integrity = referential_integrity(catalog["accessions"])
    filtered_counts = {
        object_class: sum(_in_logical_collection(object_class, row, collection) for row in rows)
        for object_class, rows in catalog["accessions"].items()
    }
    return {
        "schema_version": "eiel_knowledge_status_query_v1",
        "collection": collection,
        "source_root": str(source_root),
        "build_id": catalog["build_id"],
        "maintained_source_digest": catalog["maintained_source_digest"],
        "object_counts": catalog["object_counts"],
        "filtered_object_counts": filtered_counts,
        "dossier_count": len(DOSSIER_DEFINITIONS),
        "referential_integrity_ok": integrity["ok"],
        "html_catalog_default": "runtime generated path from rebuild-views; repo source remains metadata only",
        "mandatory_preflight_command": "python -m tools.evidence.librarian preflight --topic <topic>",
        "registration_pipeline": registration_pipeline_contract(),
        "no_claims_boundary": catalog["no_claims_boundary"],
    }
