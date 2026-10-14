"""Documentation collection support for the shared EIEL Knowledge Librarian.

This module handles metadata only.  It does not move, rename, delete, extract,
or rewrite files.  Physical publication is implemented by the guarded
publication layer after an identity freeze and an approved dry-run plan.
"""

from __future__ import annotations

from engine.runtime_roots import public_runtime_path

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from .core import LibrarianError, canonical_json_bytes, file_sha256, git_output, write_json


DOCUMENT_SCHEMA_VERSION = "eiel_document_record_v1"
DOCUMENT_RELATION_SCHEMA_VERSION = "eiel_document_relation_v1"
DOCUMENT_BOOTSTRAP_SCHEMA_VERSION = "eiel_knowledge_librarian_documents_bootstrap_v1"
DOCUMENT_ID_PATTERN = re.compile(r"^EIEL-DOC-(\d{6})$")
RUNTIME_LIFECYCLES = {"probable_runtime_result_json", "probable_runtime_result_table"}
HUMAN_READING_EXTENSIONS = {".md", ".html", ".pdf", ".epub", ".docx", ".txt"}
MACHINE_COMPANION_EXTENSIONS = {".csv", ".json", ".jsonl", ".sha256", ".xlsx", ".yaml", ".yml"}


THEME_FOLDERS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("200_genome-inheritance", ("genome", "inherit", "mutation", "recomb", "diploid", "gv2"), "dossier-06-genome-inheritance-mutation-recombination"),
    ("300_lineage-reproduction", ("lineage", "reproduction", "birth", "parent", "newborn", "turnover", "genealog"), "dossier-04-lineage-reproduction-diversity"),
    ("400_worlds-ecology", ("world", "ecolog", "environment", "pressure", "food", "niche", "resource", "roz"), "dossier-03-worlds-ecology-environmental-pressure"),
    ("500_cognition-memory", ("cognition", "memory", "controller", "gru", "mlp", "observer"), "dossier-05-cognition-memory-representation"),
    ("600_representation-policy", ("representation", "policy", "action", "sensory", "channel", "routing"), "dossier-05-cognition-memory-representation"),
    ("700_open-ended-search", ("open-ended", "open_ended", "evolvability", "evolution", "frontier", "eed", "exploration"), "dossier-07-open-ended-search-evolvability"),
    ("800_diagnostics-closures", ("diagnostic", "negative", "closure", "superseded", "blocked", "failure", "invalid", "audit"), "dossier-08-diagnostics-negative-results-closures-superseded-work"),
    ("100_engine-route-qualification", ("engine", "cuda", "route", "qualification", "runtime", "checkpoint", "determin", "provider", "cpu", "gpu"), "dossier-02-engine-determinism-execution-route-resource-qualification"),
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _stable_id(prefix: str, payload: Mapping[str, Any] | Iterable[Any], *, length: int = 20) -> str:
    digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()[:length]
    return f"{prefix}{digest}"


def _slug(value: str, *, limit: int = 120) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", normalized).strip("-").lower()
    return (normalized or "document")[:limit].rstrip("-")


def uid_filename(document_id: str, path: str) -> str:
    source = Path(path)
    suffix = "".join(source.suffixes).lower()
    stem = source.name[: -len(suffix)] if suffix else source.name
    return f"{document_id}--{_slug(stem)}{suffix}"


def _primary_document_class(candidate: Mapping[str, Any], refined: Mapping[str, Any]) -> str:
    classes = set(candidate.get("document_classes") or [])
    lifecycle = str(refined.get("refined_lifecycle_candidate") or "")
    path_cf = str(candidate["path"]).casefold()
    name_cf = Path(str(candidate["path"])).name.casefold()
    if "authority_register" in classes:
        return "authority_register"
    if "strategy" in classes:
        return "strategy"
    if "governance" in classes:
        return "governance"
    if "prompt" in classes:
        return "prompt"
    if "research_reference" in classes or lifecycle in {"research_reference", "reference_or_publication_binary"}:
        return "research_reference"
    if "historical" in classes or lifecycle == "historical_archive_doc" or "/obsolete/" in path_cf or "/90_archive/" in path_cf:
        return "historical"
    if lifecycle in RUNTIME_LIFECYCLES:
        return "raw_or_machine_output_candidate"
    if lifecycle == "structured_spec_or_contract_support" or any(word in name_cf for word in ("schema", "specification", "contract")):
        return "specification"
    if "implementation_report" in classes:
        if "plan" in name_cf and "report" not in name_cf and "closeout" not in name_cf:
            return "implementation_plan"
        return "implementation_report"
    if any(word in name_cf for word in ("operating_model", "operating-model", "authority_register", "work_register", "policy")):
        return "governance"
    if any(word in name_cf for word in ("strategy", "roadmap", "scientific_programme")):
        return "strategy"
    if any(word in name_cf for word in ("implementation_plan", "migration_plan", "campaign_plan")) or ("plan" in name_cf and "explain" not in name_cf):
        return "implementation_plan"
    if any(word in name_cf for word in ("implementation", "migration", "integration_record")):
        return "implementation_report"
    if any(word in name_cf for word in ("decision", "closeout", "closure")):
        return "decision_report"
    if "handoff" in name_cf:
        return "handoff"
    if "campaign" in name_cf or "report" in name_cf:
        return "campaign_report"
    if lifecycle == "generated_reading_copy" or "generated_support" in classes:
        return "generated_support"
    if lifecycle.startswith("structured_") or "machine_output" in classes:
        return "machine_readable_support"
    if lifecycle in {"durable_report_or_design_doc", "markdown_document_review"}:
        if any(word in name_cf for word in ("architecture", "design", "map")):
            return "architecture"
        if any(word in name_cf for word in ("audit", "validation", "verification", "gate", "probe", "receipt", "status", "boundary", "readiness")):
            return "campaign_report"
        return "durable_document_candidate"
    return "unclassified"


def _authorship(candidate: Mapping[str, Any]) -> str:
    value = str(candidate.get("authorship_candidate") or "unknown")
    if value == "authored_candidate":
        return "candidate"
    if value in {"generated_candidate", "generated_or_machine_candidate"}:
        return "candidate"
    return "unknown"


def _retention_class(primary_class: str, lifecycle: str) -> str:
    if lifecycle in RUNTIME_LIFECYCLES:
        return "cleanup_candidate"
    if primary_class == "research_reference":
        return "external_reference"
    if primary_class == "historical":
        return "historical"
    if lifecycle == "generated_reading_copy":
        return "generated_reading_copy"
    if primary_class in {"machine_readable_support", "specification"} and lifecycle.startswith("structured_"):
        return "machine_readable_support"
    return "candidate"


def _review_state(candidate: Mapping[str, Any], refined: Mapping[str, Any], authority_ids: set[str]) -> tuple[str, list[str]]:
    document_id = str(candidate["document_id"])
    lifecycle = str(refined.get("refined_lifecycle_candidate") or "other_review")
    flags: set[str] = set(candidate.get("issue_candidates") or [])
    if lifecycle in RUNTIME_LIFECYCLES:
        flags.add("physical_cleanup_candidate")
    if document_id in authority_ids:
        flags.add("authority_review_required")
    if lifecycle in {"other_review", "markdown_document_review", "structured_support_review", "root_structured_support_review"}:
        flags.add("lifecycle_review_required")
    if "physical_cleanup_candidate" in flags:
        state = "physical_cleanup_candidate"
    elif "authority_review_required" in flags:
        state = "needs_authority_review"
    elif "lifecycle_review_required" in flags:
        state = "needs_lifecycle_review"
    else:
        state = "candidate"
    return state, sorted(flags)


def infer_theme(path: str, title: str | None = None) -> tuple[str, list[str]]:
    text = f"{path} {title or ''}".casefold()
    matches: list[tuple[str, str]] = []
    for folder, keywords, dossier_id in THEME_FOLDERS:
        if any(keyword in text for keyword in keywords):
            matches.append((folder, dossier_id))
    if not matches:
        return "000_programme-cross-cutting", ["dossier-01-programme-evidence-map-methods-claim-boundaries"]
    folder, dossier = matches[0]
    dossiers = sorted({row[1] for row in matches})
    return folder, dossiers


def _paired_markdown_path(path: str, all_paths: set[str]) -> str | None:
    source = Path(path)
    if source.suffix.casefold() != ".html":
        return None
    markdown = source.with_suffix(".md").as_posix()
    return markdown if markdown in all_paths else None


def _same_stem_key(path: str) -> tuple[str, str]:
    source = Path(path)
    return source.parent.as_posix().casefold(), source.stem.casefold()


def _package_anchor(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose a navigation anchor without inferring authority or currentness."""

    priority = {
        "authority_register": 0,
        "decision_report": 1,
        "governance": 2,
        "strategy": 3,
        "architecture": 4,
        "specification": 5,
        "implementation_plan": 6,
        "implementation_report": 7,
        "campaign_report": 8,
        "handoff": 9,
        "prompt": 10,
        "research_reference": 11,
    }

    def sort_key(record: Mapping[str, Any]) -> tuple[int, int, str]:
        extension = Path(str(record["path"])).suffix.casefold()
        primary = str(record.get("document_class") or "unknown")
        if primary == "research_reference":
            extension_rank = {".pdf": 0, ".epub": 1, ".docx": 2}.get(extension, 3)
        else:
            extension_rank = {".md": 0, ".json": 1, ".html": 2}.get(extension, 3)
        return priority.get(primary, 50), extension_rank, str(record["document_id"])

    return min(records, key=sort_key)


def _co_locate_document_packages(records: list[dict[str, Any]]) -> None:
    """Keep same-stem companions together while preserving individual UIDs.

    A package rule affects navigation only.  It never merges files, establishes
    authority, or adjudicates whether a companion was generated from another.
    Machine-only result packages follow a runtime-classified member out of Git;
    packages containing a human reading document remain in the repository.
    """

    packages: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        packages[_same_stem_key(str(record["path"]))].append(record)
    runtime_root = Path(public_runtime_path('evidence_raw/eiel_knowledge_librarian_docs_v1/relocated_runtime'))
    for members in packages.values():
        extensions = {Path(str(record["path"])).suffix.casefold() for record in members}
        if len(members) < 2 or len(extensions) < 2:
            continue
        has_human_reading = bool(extensions & HUMAN_READING_EXTENSIONS)
        has_runtime_member = any(
            record["proposed_publication"]["destination_class"] == "runtime_relocation"
            for record in members
        )
        machine_only = bool(extensions) and extensions <= MACHINE_COMPANION_EXTENSIONS
        if has_runtime_member and machine_only and not has_human_reading:
            anchor = min(members, key=lambda record: str(record["document_id"]))
            for record in members:
                proposal = record["proposed_publication"]
                proposal["destination_class"] = "runtime_relocation"
                proposal["proposed_destination"] = str(
                    runtime_root / uid_filename(str(record["document_id"]), str(record["path"]))
                )
                proposal["basis"] = sorted(
                    set(proposal["basis"])
                    | {
                        "machine_only_same_stem_package",
                        f"package_anchor:{anchor['document_id']}",
                    }
                )
            continue
        if any(record["proposed_publication"]["destination_class"] != "repo_document" for record in members):
            continue
        anchor = _package_anchor(members)
        anchor_parent = Path(str(anchor["proposed_publication"]["proposed_destination"])).parent
        for record in members:
            proposal = record["proposed_publication"]
            proposal["proposed_destination"] = str(
                anchor_parent / uid_filename(str(record["document_id"]), str(record["path"]))
            )
            proposal["basis"] = sorted(
                set(proposal["basis"])
                | {
                    "same_stem_document_package",
                    f"package_anchor:{anchor['document_id']}",
                }
            )


def proposed_publication(
    record: Mapping[str, Any],
    *,
    refined_lifecycle: str,
    all_paths: set[str],
    records_by_path: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Return a navigation-only future destination proposal.

    A proposed destination is deliberately excluded from authority resolution.
    Runtime candidates receive an external destination class and no repo move.
    """

    path = str(record["path"])
    primary = str(record["document_class"])
    name_cf = Path(path).name.casefold()
    extension = Path(path).suffix.casefold()
    theme, _ = infer_theme(path, str(record.get("title") or ""))
    paired_md = _paired_markdown_path(path, all_paths)
    if paired_md:
        paired_record = records_by_path.get(paired_md)
        if paired_record:
            paired = proposed_publication(
                paired_record,
                refined_lifecycle=str(paired_record.get("refined_lifecycle_candidate") or ""),
                all_paths=all_paths,
                records_by_path=records_by_path,
            )
            destination = str(Path(str(paired["proposed_destination"])).parent / uid_filename(str(record["document_id"]), path))
            return {
                "proposal_version": "eiel_docs_publication_namespace_v1",
                "destination_class": "repo_document",
                "proposed_destination": destination,
                "basis": ["paired_markdown_destination", "document_id_filename_prefix"],
                "confidence": "candidate",
                "authority_effect": "none",
                "physical_move_authorized": False,
                "requires_operator_review": True,
            }
    if refined_lifecycle in RUNTIME_LIFECYCLES:
        destination = str(Path(public_runtime_path("evidence_raw/eiel_knowledge_librarian_docs_v1/relocated_runtime")) / uid_filename(str(record["document_id"]), path))
        return {
            "proposal_version": "eiel_docs_publication_namespace_v1",
            "destination_class": "runtime_relocation",
            "proposed_destination": destination,
            "basis": [f"refined_lifecycle:{refined_lifecycle}", "document_id_filename_prefix"],
            "confidence": "candidate",
            "authority_effect": "none",
            "physical_move_authorized": False,
            "requires_operator_review": True,
        }
    if primary == "authority_register":
        shelf = "100_governance-authority/110_authority-registers"
    elif primary == "governance":
        shelf = "100_governance-authority/100_governance"
    elif primary == "strategy":
        shelf = "200_strategy-science-programme/200_current-strategy"
    elif primary == "architecture":
        shelf = f"300_architecture-specifications/300_system-architecture/{theme}"
    elif primary == "specification":
        shelf = "300_architecture-specifications/370_specifications"
    elif primary == "implementation_plan":
        shelf = f"400_implementation-engineering/400_implementation-plans/{theme}"
    elif primary == "implementation_report":
        shelf = f"400_implementation-engineering/410_implementation-reports/{theme}"
    elif primary == "decision_report":
        if "qualification" in name_cf or "closeout" in name_cf:
            shelf = f"500_campaigns-decisions-closeouts/510_qualification-closeouts/{theme}"
        else:
            shelf = f"500_campaigns-decisions-closeouts/500_decision-reports/{theme}"
    elif primary == "campaign_report":
        shelf = f"500_campaigns-decisions-closeouts/520_campaign-reports/{theme}"
    elif primary == "handoff":
        shelf = "500_campaigns-decisions-closeouts/540_handoffs"
    elif primary == "prompt":
        shelf = "500_campaigns-decisions-closeouts/550_prompts"
    elif primary == "research_reference":
        shelf = "600_references-publications/610_books-papers-external" if extension in {".pdf", ".epub"} else "600_references-publications/600_research-references"
    elif primary == "historical":
        shelf = "900_lifecycle/900_historical"
    elif refined_lifecycle == "generated_reading_copy" or primary == "generated_support":
        shelf = "700_generated-reading-support/700_html-reading-copies"
    elif primary == "machine_readable_support":
        parent = Path(path).parent.name
        campaign = _slug(parent, limit=80) if parent and parent not in {"docs", "40_eiel_design"} else "root-support"
        shelf = f"800_machine-readable-support/860_small-derived-tables/{theme}/{campaign}"
    elif path in {"docs/700_generated-reading-support/700_html-reading-copies/EIEL-DOC-002349--index.html"} or Path(path).name.casefold() in {"readme.md", "index.md"}:
        shelf = "000_start-here/030_indexes-maps"
    elif "white_paper" in path.casefold() or "white-paper" in path.casefold():
        shelf = "600_references-publications/620_white-papers"
    elif primary == "durable_document_candidate":
        path_cf = path.casefold()
        if any(word in path_cf for word in ("persistent_context", "working_style", "notebooklm_guide", "user_history")):
            shelf = "000_start-here/010_operator-guides"
        elif "/campaign_plans/" in path_cf or "plan" in name_cf:
            shelf = f"400_implementation-engineering/400_implementation-plans/{theme}"
        elif any(word in name_cf for word in ("framework", "architecture", "design")):
            shelf = f"300_architecture-specifications/300_system-architecture/{theme}"
        else:
            shelf = f"500_campaigns-decisions-closeouts/520_campaign-reports/{theme}"
    elif "/ddjf/" in path.casefold() or "ddjf" in name_cf:
        shelf = "600_references-publications/630_public-narratives"
    elif "/generated/" in path.casefold():
        shelf = "700_generated-reading-support/720_generated-indexes"
    elif "pressure_knob" in path.casefold() or "/specs/" in path.casefold():
        shelf = "800_machine-readable-support/810_contract-json"
    elif "manifest" in name_cf:
        shelf = "800_machine-readable-support/840_manifests"
    elif "history" in name_cf:
        shelf = "900_lifecycle/900_historical"
    elif any(word in name_cf for word in ("validation", "preflight", "no_claims")):
        shelf = f"500_campaigns-decisions-closeouts/520_campaign-reports/{theme}"
    elif "guide" in name_cf:
        shelf = "000_start-here/010_operator-guides"
    elif "lessons" in name_cf or "ledger" in name_cf:
        shelf = "200_strategy-science-programme/210_scientific-ground-rules"
    elif extension == ".zip" or "thoughts" in name_cf:
        shelf = "900_lifecycle/930_quarantine-review"
    else:
        shelf = "900_lifecycle/950_unclassified"
    return {
        "proposal_version": "eiel_docs_publication_namespace_v1",
        "destination_class": "repo_document",
        "proposed_destination": f"docs/{shelf}/{uid_filename(str(record['document_id']), path)}",
        "basis": [f"document_class:{primary}", f"refined_lifecycle:{refined_lifecycle}", "document_id_filename_prefix"],
        "confidence": "candidate",
        "authority_effect": "none",
        "physical_move_authorized": False,
        "requires_operator_review": True,
    }


def build_document_records(
    *,
    repo_root: Path,
    catalog_candidate_path: Path,
    refined_lifecycle_path: Path,
    authority_queue_path: Path,
    observed_commit: str,
) -> list[dict[str, Any]]:
    candidate_payload = _load(catalog_candidate_path)
    refined_payload = _load(refined_lifecycle_path)
    queue_payload = _load(authority_queue_path)
    candidates = list(candidate_payload["document_records"])
    refined_by_path = {str(row["path"]): row for row in refined_payload["records"]}
    authority_ids = {str(row["document_id"]) for row in queue_payload["queue"]}
    ids = [str(row["document_id"]) for row in candidates]
    paths = [str(row["path"]) for row in candidates]
    if len(ids) != len(set(ids)) or len(paths) != len(set(paths)):
        raise LibrarianError("candidate document IDs or paths are not unique")
    all_paths = set(paths)
    source_digest = file_sha256(catalog_candidate_path)
    preliminary: list[dict[str, Any]] = []
    for candidate in candidates:
        document_id = str(candidate["document_id"])
        if not DOCUMENT_ID_PATTERN.fullmatch(document_id):
            raise LibrarianError(f"invalid candidate document ID: {document_id}")
        path = str(candidate["path"])
        absolute = repo_root / path
        if not absolute.is_file():
            raise LibrarianError(f"candidate document path is missing: {path}")
        actual_hash = file_sha256(absolute)
        if actual_hash != candidate["sha256"] or absolute.stat().st_size != candidate["size_bytes"]:
            raise LibrarianError(f"candidate document changed after audit: {path}")
        refined = refined_by_path.get(path)
        if not refined:
            raise LibrarianError(f"missing refined lifecycle record: {path}")
        primary_class = _primary_document_class(candidate, refined)
        review_state, review_flags = _review_state(candidate, refined, authority_ids)
        linked = candidate.get("linked_evidence_catalog_objects") or {}
        title = candidate.get("title_candidate") or Path(path).stem.replace("_", " ")
        _, dossier_ids = infer_theme(path, str(title))
        record = {
            "schema_version": DOCUMENT_SCHEMA_VERSION,
            "collection": "documents",
            "document_id": document_id,
            "title": title,
            "path": path,
            "canonical_reading_location": path if candidate.get("canonical_reading_location_candidate") == "yes_candidate" else None,
            "sha256": actual_hash,
            "size_bytes": int(candidate["size_bytes"]),
            "git_state": str(candidate.get("git_state") or "unknown"),
            "document_class": primary_class,
            "document_class_candidates": sorted(set(candidate.get("document_classes") or [])),
            "refined_lifecycle_candidate": str(refined.get("refined_lifecycle_candidate") or "other_review"),
            "currentness_status": "unreviewed",
            "currentness_candidate": str(candidate.get("currentness_status_candidate") or "unknown"),
            "authored_or_generated": _authorship(candidate),
            "authorship_candidate": str(candidate.get("authorship_candidate") or "unknown"),
            "retention_class": _retention_class(primary_class, str(refined.get("refined_lifecycle_candidate") or "")),
            "retention_candidate": str(candidate.get("retention_class_candidate") or "unknown"),
            "review_state": review_state,
            "review_flags": review_flags,
            "supersedes_document_ids": [],
            "superseded_by_document_ids": [],
            "linked_evidence_ids": sorted(set(linked.get("records") or [])),
            "linked_claim_ids": sorted(set(linked.get("claims") or [])),
            "linked_authority_assertion_ids": sorted(set(linked.get("authorities") or [])),
            "linked_dossier_ids": dossier_ids,
            "linked_raw_asset_ids": sorted(set(linked.get("raw_assets") or [])),
            "linked_artifact_ids": sorted(set(linked.get("artifacts") or [])),
            "linked_issue_ids": sorted(set(linked.get("issues") or [])),
            "source_state": {
                "state": "frozen_audit_import",
                "observed_at_commit": observed_commit,
                "source_digest": source_digest,
                "audit_artifact": str(catalog_candidate_path),
            },
            "identity": {
                "frozen": True,
                "path_at_freeze": path,
                "sha256_at_freeze": actual_hash,
                "size_bytes_at_freeze": int(candidate["size_bytes"]),
                "freeze_commit": observed_commit,
                "freeze_source_digest": source_digest,
                "freeze_basis": "accepted_read_only_docs_audit_v1",
                "reuse_policy": "never_reuse",
                "path_is_identity": False,
                "path_history": [],
                "content_revisions": [],
            },
            "notes": "Imported from the accepted read-only audit as conservative metadata; no authority or currentness is inferred.",
        }
        preliminary.append(record)
    records_by_path = {str(row["path"]): row for row in preliminary}
    for record in preliminary:
        lifecycle = str(record["refined_lifecycle_candidate"])
        record["proposed_publication"] = proposed_publication(
            record,
            refined_lifecycle=lifecycle,
            all_paths=all_paths,
            records_by_path=records_by_path,
        )
    _co_locate_document_packages(preliminary)
    return sorted(preliminary, key=lambda row: str(row["document_id"]))


def _relation(
    relation_type: str,
    source_document_id: str,
    *,
    target_type: str,
    target_id_or_path: str,
    basis: list[str],
    review_state: str,
    notes: str,
) -> dict[str, Any]:
    identity = {
        "relation_type": relation_type,
        "source_document_id": source_document_id,
        "target_type": target_type,
        "target_id_or_path": target_id_or_path,
        "basis": basis,
    }
    return {
        "schema_version": DOCUMENT_RELATION_SCHEMA_VERSION,
        "collection": "document_relations",
        "relation_id": _stable_id("eiel-docrel-", identity),
        "relation_type": relation_type,
        "source_document_id": source_document_id,
        "target": {"target_type": target_type, "target_id_or_path": target_id_or_path},
        "basis": basis,
        "authority_effect": "none",
        "review_state": review_state,
        "notes": notes,
    }


def build_document_relations(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [dict(row) for row in records]
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_basename: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_path = {str(row["path"]): row for row in rows}
    relations: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_hash[str(row["sha256"])].append(row)
        by_basename[Path(str(row["path"])).name.casefold()].append(row)
    for members in by_hash.values():
        members = sorted(members, key=lambda row: str(row["document_id"]))
        if len(members) < 2:
            continue
        source = members[0]
        for target in members[1:]:
            relation = _relation(
                "duplicate_content",
                str(source["document_id"]),
                target_type="document",
                target_id_or_path=str(target["document_id"]),
                basis=["identical_sha256"],
                review_state="candidate",
                notes="Exact content identity is recorded without merging the physical documents.",
            )
            relations[str(relation["relation_id"])] = relation
    for members in by_basename.values():
        members = sorted(members, key=lambda row: str(row["document_id"]))
        if len(members) < 2 or len({str(row["sha256"]) for row in members}) < 2:
            continue
        source = members[0]
        for target in members[1:]:
            relation = _relation(
                "same_basename_unproven",
                str(source["document_id"]),
                target_type="document",
                target_id_or_path=str(target["document_id"]),
                basis=["basename_only_unproven"],
                review_state="needs_review",
                notes="A basename match alone does not establish supersession, copy, or authority.",
            )
            relations[str(relation["relation_id"])] = relation
    target_fields = (
        ("linked_evidence_ids", "evidence_record", "linked_to_evidence_record"),
        ("linked_claim_ids", "claim", "references"),
        ("linked_authority_assertion_ids", "authority", "references"),
        ("linked_raw_asset_ids", "raw_asset", "references"),
        ("linked_artifact_ids", "external_reference", "references"),
        ("linked_issue_ids", "external_reference", "references"),
    )
    for row in rows:
        for field, target_type, relation_type in target_fields:
            for target_id in row.get(field, []) or []:
                relation = _relation(
                    relation_type,
                    str(row["document_id"]),
                    target_type=target_type,
                    target_id_or_path=str(target_id),
                    basis=["existing_librarian_relation"],
                    review_state="accepted",
                    notes="Cross-collection navigation link; it does not grant or broaden authority.",
                )
                relations[str(relation["relation_id"])] = relation
        paired = _paired_markdown_path(str(row["path"]), set(by_path))
        if paired:
            target = by_path[paired]
            relation = _relation(
                "same_basename_unproven",
                str(row["document_id"]),
                target_type="document",
                target_id_or_path=str(target["document_id"]),
                basis=["basename_only_unproven"],
                review_state="needs_review",
                notes="HTML and Markdown share a stem; generated-copy status remains unadjudicated.",
            )
            relations[str(relation["relation_id"])] = relation
    return sorted(relations.values(), key=lambda row: str(row["relation_id"]))


def _document_issue(
    kind: str,
    document_rows: Iterable[Mapping[str, Any]],
    *,
    severity: str,
    summary: str,
    trigger: str,
) -> dict[str, Any]:
    rows = sorted((dict(row) for row in document_rows), key=lambda row: str(row["document_id"]))
    document_ids = [str(row["document_id"]) for row in rows]
    paths = [str(row["path"]) for row in rows]
    identity = {"kind": kind, "document_ids": document_ids, "trigger": trigger}
    return {
        "schema_version": "eiel_issue_v1",
        "issue_id": _stable_id("eiel-issue-docs-v1-", identity),
        "category": kind,
        "issue_kind": kind,
        "owner_collection": "documents",
        "document_ids": document_ids,
        "document_paths": paths,
        "paths": paths,
        "severity": severity,
        "summary": summary,
        "detection_rule": trigger,
        "disposition": "review_required_non_destructive",
        "review_state": "candidate",
        "owner": "knowledge_librarian_review_queue",
        "first_observed": "docs_audit_v1",
        "last_observed": "docs_collection_v1",
        "remediation_state": "planned",
        "exception_expiry": None,
        "authority_effect": "none",
        "evidence": {"document_count": len(document_ids), "no_authority_inference": True},
    }


def build_document_issues(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [dict(row) for row in records]
    by_id = {str(row["document_id"]): row for row in rows}
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_basename: dict[str, list[dict[str, Any]]] = defaultdict(list)
    runtime_by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    historical_by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    issues: list[dict[str, Any]] = []
    for row in rows:
        by_hash[str(row["sha256"])].append(row)
        by_basename[Path(str(row["path"])).name.casefold()].append(row)
        if row.get("review_state") == "physical_cleanup_candidate":
            runtime_by_parent[str(Path(str(row["path"])).parent)].append(row)
        if row.get("document_class") == "historical":
            parts = Path(str(row["path"])).parts
            historical_by_root[parts[1] if len(parts) > 1 else "docs"].append(row)
        if "authority_review_required" in (row.get("review_flags") or []):
            issues.append(
                _document_issue(
                    "authority_drift_candidate",
                    [row],
                    severity="high",
                    summary="Authority or currentness language requires explicit adjudication.",
                    trigger="accepted_docs_audit_authority_review_queue",
                )
            )
    for members in by_hash.values():
        if len(members) > 1:
            issues.append(
                _document_issue(
                    "exact_duplicate_content",
                    members,
                    severity="medium",
                    summary="Multiple physical documents have identical SHA-256 content; no merge is inferred.",
                    trigger="identical_sha256",
                )
            )
    for basename, members in by_basename.items():
        if len(members) > 1 and len({str(row["sha256"]) for row in members}) > 1:
            issues.append(
                _document_issue(
                    "same_basename_divergence",
                    members,
                    severity="high",
                    summary=f"Divergent files share basename {basename}; relationship and currentness are unproven.",
                    trigger="casefolded_basename_with_divergent_sha256",
                )
            )
    for parent, members in sorted(runtime_by_parent.items()):
        issues.append(
            _document_issue(
                "probable_runtime_output_in_docs",
                members,
                severity="high",
                summary=f"Probable runtime results under {parent} require verified relocation and a pointer manifest.",
                trigger="refined_lifecycle_probable_runtime_result",
            )
        )
    for root, members in sorted(historical_by_root.items()):
        issues.append(
            _document_issue(
                "obsolete_or_historical_candidate",
                members,
                severity="low",
                summary=f"Historical candidates under {root} require lifecycle review; no authority change is inferred.",
                trigger="accepted_audit_historical_candidate",
            )
        )
    for row in rows:
        if row.get("document_class") == "research_reference" and int(row.get("size_bytes") or 0) >= 10_000_000:
            issues.append(
                _document_issue(
                    "large_reference_in_repo",
                    [row],
                    severity="medium",
                    summary="Large external reference should remain reference-scoped and may later be cold-storage reviewed.",
                    trigger="research_reference_size_at_least_10MB",
                )
            )
    unique = {str(row["issue_id"]): row for row in issues}
    missing = [document_id for row in unique.values() for document_id in row["document_ids"] if document_id not in by_id]
    if missing:
        raise LibrarianError(f"document issues reference missing IDs: {missing[:5]}")
    return sorted(unique.values(), key=lambda row: str(row["issue_id"]))


def identity_freeze_manifest(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [
        {
            "document_id": row["document_id"],
            "path_at_freeze": row["identity"]["path_at_freeze"],
            "sha256_at_freeze": row["identity"]["sha256_at_freeze"],
            "size_bytes_at_freeze": row["identity"]["size_bytes_at_freeze"],
            "freeze_commit": row["identity"]["freeze_commit"],
            "freeze_source_digest": row["identity"]["freeze_source_digest"],
            "freeze_basis": row["identity"]["freeze_basis"],
        }
        for row in sorted(records, key=lambda item: str(item["document_id"]))
    ]
    return {
        "schema_version": "eiel_document_identity_freeze_manifest_v1",
        "identity_policy": "eiel-doc-identity-policy-v1",
        "id_reuse_policy": "never_reuse",
        "path_is_identity": False,
        "record_count": len(rows),
        "next_document_sequence": max(int(str(row["document_id"]).rsplit("-", 1)[1]) for row in rows) + 1,
        "records": rows,
    }


def publication_plan(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = []
    destinations: dict[str, list[str]] = defaultdict(list)
    for record in sorted(records, key=lambda row: str(row["document_id"])):
        proposal = record["proposed_publication"]
        destination = str(proposal["proposed_destination"])
        action = "runtime_relocate" if proposal["destination_class"] == "runtime_relocation" else "repo_move"
        row = {
            "document_id": record["document_id"],
            "source_path": record["path"],
            "source_sha256": record["sha256"],
            "source_size_bytes": record["size_bytes"],
            "action": action,
            "proposed_destination": destination,
            "authority_effect": "none",
            "physical_move_authorized": False,
        }
        rows.append(row)
        destinations[destination].append(str(record["document_id"]))
    collisions = [
        {"proposed_destination": path, "document_ids": ids}
        for path, ids in sorted(destinations.items())
        if len(ids) > 1
    ]
    return {
        "schema_version": "eiel_document_publication_plan_v1",
        "publication_namespace": "eiel-doc-publication-namespace-v1",
        "document_count": len(rows),
        "repo_move_count": sum(row["action"] == "repo_move" for row in rows),
        "runtime_relocation_count": sum(row["action"] == "runtime_relocate" for row in rows),
        "collision_count": len(collisions),
        "collisions": collisions,
        "all_ids_prefixed_in_filenames": all(Path(row["proposed_destination"]).name.startswith(str(row["document_id"])) for row in rows),
        "physical_move_authorized": False,
        "entries": rows,
    }


def validate_document_objects(
    objects: Mapping[str, list[dict[str, Any]]],
    *,
    source_root: Path | None = None,
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    document_ids = {str(row.get("document_id")) for row in objects.get("documents", [])}
    relation_ids: set[str] = set()
    required = {
        "document_id", "collection", "schema_version", "path", "canonical_reading_location", "sha256", "size_bytes",
        "git_state", "document_class", "currentness_status", "authored_or_generated", "retention_class", "review_state",
        "supersedes_document_ids", "superseded_by_document_ids", "linked_evidence_ids", "linked_claim_ids",
        "linked_authority_assertion_ids", "linked_dossier_ids", "linked_raw_asset_ids", "source_state", "notes",
    }
    for row in objects.get("documents", []):
        document_id = str(row.get("document_id"))
        missing = sorted(required - set(row))
        if missing:
            failures.append({"object_class": "documents", "object_id": document_id, "failure": "missing_fields", "fields": missing})
        if not DOCUMENT_ID_PATTERN.fullmatch(document_id):
            failures.append({"object_class": "documents", "object_id": document_id, "failure": "invalid_id"})
        if row.get("collection") != "documents" or row.get("schema_version") != DOCUMENT_SCHEMA_VERSION:
            failures.append({"object_class": "documents", "object_id": document_id, "failure": "schema_or_collection_mismatch"})
        for field in ("supersedes_document_ids", "superseded_by_document_ids"):
            for target in row.get(field, []) or []:
                if target not in document_ids:
                    failures.append({"object_class": "documents", "object_id": document_id, "field": field, "target_id": target, "failure": "missing_target"})
    for row in objects.get("document_relations", []):
        relation_id = str(row.get("relation_id"))
        if relation_id in relation_ids:
            failures.append({"object_class": "document_relations", "object_id": relation_id, "failure": "duplicate_id"})
        relation_ids.add(relation_id)
        source_id = str(row.get("source_document_id"))
        if source_id not in document_ids:
            failures.append({"object_class": "document_relations", "object_id": relation_id, "source_document_id": source_id, "failure": "missing_source_document"})
        target = row.get("target") or {}
        if target.get("target_type") == "document" and target.get("target_id_or_path") not in document_ids:
            failures.append({"object_class": "document_relations", "object_id": relation_id, "target_id": target.get("target_id_or_path"), "failure": "missing_target_document"})
    schema_validation = {"available": False, "failure_count": 0, "failures": []}
    if source_root:
        try:
            from jsonschema import Draft202012Validator
        except ImportError:
            pass
        else:
            schema_validation["available"] = True
            schemas = {
                "documents": _load(source_root / "schemas/document_record_v1.schema.json"),
                "document_relations": _load(source_root / "schemas/document_relation_v1.schema.json"),
            }
            for collection, schema in schemas.items():
                validator = Draft202012Validator(schema)
                for row in objects.get(collection, []):
                    object_id = str(row.get("document_id") or row.get("relation_id"))
                    for error in validator.iter_errors(row):
                        schema_validation["failures"].append({"object_class": collection, "object_id": object_id, "path": list(error.path), "message": error.message})
            schema_validation["failure_count"] = len(schema_validation["failures"])
            failures.extend({"failure": "json_schema", **row} for row in schema_validation["failures"])
    return {
        "schema_version": "eiel_document_collection_validation_v1",
        "ok": not failures,
        "failure_count": len(failures),
        "failures": failures[:500],
        "document_count": len(objects.get("documents", [])),
        "document_relation_count": len(objects.get("document_relations", [])),
        "schema_validation": schema_validation,
    }


def write_bootstrap_source(
    *,
    source_root: Path,
    records: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    document_issues: list[dict[str, Any]],
) -> dict[str, Any]:
    for directory in (source_root / "documents", source_root / "document_relations"):
        directory.mkdir(parents=True, exist_ok=True)
    expected_documents = {f"{row['document_id']}.json" for row in records}
    expected_relations = {f"{row['relation_id']}.json" for row in relations}
    for path in (source_root / "documents").glob("*.json"):
        if path.name not in expected_documents:
            path.unlink()
    for path in (source_root / "document_relations").glob("*.json"):
        if path.name not in expected_relations:
            path.unlink()
    for row in records:
        write_json(source_root / "documents" / f"{row['document_id']}.json", row, pretty=False)
    for row in relations:
        write_json(source_root / "document_relations" / f"{row['relation_id']}.json", row, pretty=False)
    existing_issue_paths = sorted((source_root / "issues").glob("*.json"))
    for path in existing_issue_paths:
        row = _load(path)
        if "owner_collection" not in row:
            row["owner_collection"] = "evidence"
            write_json(path, row, pretty=False)
    expected_document_issues = {f"{row['issue_id']}.json" for row in document_issues}
    for path in (source_root / "issues").glob("eiel-issue-docs-v1-*.json"):
        if path.name not in expected_document_issues:
            path.unlink()
    for row in document_issues:
        write_json(source_root / "issues" / f"{row['issue_id']}.json", row, pretty=False)
    freeze = identity_freeze_manifest(records)
    plan = publication_plan(records)
    write_json(source_root / "manifests" / "DOCUMENT_IDENTITY_FREEZE.json", freeze, pretty=False)
    write_json(source_root / "manifests" / "DOCUMENT_PUBLICATION_PLAN_CANDIDATE.json", plan, pretty=False)
    return {
        "schema_version": DOCUMENT_BOOTSTRAP_SCHEMA_VERSION,
        "document_count": len(records),
        "document_relation_count": len(relations),
        "document_issue_count": len(document_issues),
        "identity_freeze_record_count": freeze["record_count"],
        "next_document_sequence": freeze["next_document_sequence"],
        "publication_plan_collision_count": plan["collision_count"],
        "runtime_relocation_candidate_count": plan["runtime_relocation_count"],
        "physical_documents_modified": False,
    }


def register_repo_document(
    *,
    repo_root: Path,
    source_root: Path,
    path: str,
    title: str,
    document_class: str,
    destination_shelf: str,
    observed_commit: str,
    linked_dossier_ids: Iterable[str] = (),
) -> dict[str, Any]:
    """Append one explicitly supplied post-audit document with the next UID."""

    repo_root = repo_root.resolve()
    source_root = source_root.resolve()
    source_rel = PurePosixPath(path)
    if source_rel.is_absolute() or not source_rel.parts or source_rel.parts[0] != "docs" or ".." in source_rel.parts:
        raise LibrarianError(f"document registration path must be safely under docs/: {path}")
    absolute = (repo_root / Path(source_rel.as_posix())).resolve()
    if not absolute.is_relative_to((repo_root / "docs").resolve()) or not absolute.is_file() or absolute.is_symlink():
        raise LibrarianError(f"document registration source is missing or invalid: {path}")
    tracked = git_output(repo_root, ["ls-files", "--error-unmatch", path])
    if tracked.strip() != path:
        raise LibrarianError(f"document registration source is not tracked exactly: {path}")
    current_head = git_output(repo_root, ["rev-parse", "HEAD"], check=True).strip()
    if current_head != observed_commit:
        raise LibrarianError(
            f"document registration commit mismatch: expected {observed_commit}, observed {current_head}"
        )
    shelf = PurePosixPath(destination_shelf)
    if shelf.is_absolute() or not shelf.parts or ".." in shelf.parts:
        raise LibrarianError(f"invalid destination shelf: {destination_shelf}")
    top_level = shelf.parts[0]
    if top_level not in {
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
    }:
        raise LibrarianError(f"destination shelf is outside the publication namespace: {destination_shelf}")
    records = [_load(item) for item in sorted((source_root / "documents").glob("*.json"))]
    if any(str(row["path"]) == path for row in records):
        raise LibrarianError(f"document path is already registered: {path}")
    next_sequence = max(int(str(row["document_id"]).rsplit("-", 1)[1]) for row in records) + 1
    document_id = f"EIEL-DOC-{next_sequence:06d}"
    digest = file_sha256(absolute)
    destination = f"docs/{shelf.as_posix()}/{uid_filename(document_id, path)}"
    dossiers = sorted(set(str(value) for value in linked_dossier_ids))
    record = {
        "schema_version": DOCUMENT_SCHEMA_VERSION,
        "collection": "documents",
        "document_id": document_id,
        "title": title,
        "path": path,
        "canonical_reading_location": path,
        "sha256": digest,
        "size_bytes": absolute.stat().st_size,
        "git_state": "tracked",
        "document_class": document_class,
        "document_class_candidates": [document_class],
        "refined_lifecycle_candidate": "post_audit_explicit_registration",
        "currentness_status": "unreviewed",
        "currentness_candidate": "explicit_registration_review_required",
        "authored_or_generated": "candidate",
        "authorship_candidate": "authored_candidate",
        "retention_class": "candidate",
        "retention_candidate": "repo_publication",
        "review_state": "candidate",
        "review_flags": ["post_audit_explicit_registration"],
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
            "state": "post_audit_explicit_registration",
            "observed_at_commit": observed_commit,
            "source_digest": digest,
            "audit_artifact": None,
        },
        "identity": {
            "frozen": True,
            "path_at_freeze": path,
            "sha256_at_freeze": digest,
            "size_bytes_at_freeze": absolute.stat().st_size,
            "freeze_commit": observed_commit,
            "freeze_source_digest": digest,
            "freeze_basis": "explicit_post_audit_registration_before_archive_publication_v1",
            "reuse_policy": "never_reuse",
            "path_is_identity": False,
            "path_history": [],
            "content_revisions": [],
        },
        "proposed_publication": {
            "proposal_version": "eiel_docs_publication_namespace_v1",
            "destination_class": "repo_document",
            "proposed_destination": destination,
            "basis": ["explicit_post_audit_registration", "document_id_filename_prefix"],
            "confidence": "candidate",
            "authority_effect": "none",
            "physical_move_authorized": False,
            "requires_operator_review": True,
        },
        "notes": "Explicitly registered after the accepted audit and before Archive Publication 1.0; no authority or currentness is inferred.",
    }
    records.append(record)
    records = sorted(records, key=lambda row: str(row["document_id"]))
    relations = build_document_relations(records)
    issues = build_document_issues(records)
    bootstrap = write_bootstrap_source(
        source_root=source_root,
        records=records,
        relations=relations,
        document_issues=issues,
    )
    identity_policy_path = source_root / "policies/document_identity_policy_v1.json"
    identity_policy = _load(identity_policy_path)
    identity_policy["current_freeze_range"] = {
        "count": len(records),
        "first_id": records[0]["document_id"],
        "last_id": records[-1]["document_id"],
        "status": "frozen_for_archive_publication_v1",
    }
    identity_policy["allocation"]["next_sequence_after_current_freeze"] = next_sequence + 1
    write_json(identity_policy_path, identity_policy, pretty=True)
    return {
        "schema_version": "eiel_document_explicit_registration_v1",
        "document_id": document_id,
        "path": path,
        "sha256": digest,
        "size_bytes": absolute.stat().st_size,
        "proposed_destination": destination,
        "bootstrap": bootstrap,
        "authority_effect": "none",
        "physical_documents_modified": False,
    }
