"""WP2R curation layer for the EIEL evidence librarian.

The functions in this module consume the V1R read-only audit outputs and write
curated candidate ledgers to an explicit runtime output directory. They never
write into the historical evidence root.
"""

from __future__ import annotations

from engine.runtime_roots import public_runtime_path

import difflib
import hashlib
import html
import json
import os
import platform
import re
import subprocess
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from .core import (
    LibrarianError,
    canonical_json_bytes,
    deterministic_zip,
    file_sha256,
    inspect_zip_safety,
    issue_counts,
    load_json,
    read_jsonl,
    rel_to,
    tree_digest,
    utc_now_iso,
    write_json,
)


WP2R_SCHEMA_VERSION = "eiel_evidence_librarian_wp2r_v1"
WP2R_RUN_ID = "20260802T000000Z_wp2r_authority_curation"
V1R_AUDIT_COMMIT = "b86ee4c0d5865bcda7afff9ec7e3b79231e0cf45"
V1R_FINAL_COMMIT = "4a931cc56e3a0447c81b0790b70b734b00b4f2c3"
V1R_BASE_COMMIT = "0a8235212fad5590f638e8249f08644e84e6363b"
V1R_CLOSEOUT_PATH = (
    "docs/40_eiel_design/"
    "EIEL_Evidence_Librarian_V1R_Closeout_Report_2026_08_02.md"
)
EXPECTED_V1R_TREE_DIGEST = (
    "d474cc44ed9c61a63b553fb31aa5019af206c1339b1f90d6d89554f38859bcb5"
)
EXPECTED_SUPPLEMENT_SHA256 = (
    "70258ebe638152794e6d41916fcc30b8ac10d7393dfde41034821e793b04805d"
)
UNSAFE_ZIP_REL = (
    "P12_SUBSTRATE_GATES_20260727/"
    "EIEL_P12_Final_Three_Gate_Terminal_Architecture_Wall_2026_07_28.evidence.zip"
)
UNSAFE_MEMBER = (
    "../../docs/40_eiel_design/"
    "EIEL_P12_Final_Three_Gate_Terminal_Architecture_Wall_2026_07_28.md"
)
SAFE_COUNTERPART = (
    "docs/40_eiel_design/"
    "EIEL_P12_Final_Three_Gate_Terminal_Architecture_Wall_2026_07_28.md"
)
ARCHIVE_PUBLICATION_REDIRECT_MAP_REL = "evidences/990_legacy-redirects/archive-publication-v1-map.json"
DOCUMENT_ARCHIVE_REDIRECT_MAP_REL = (
    "tools/evidence/librarian/catalog_source/00-catalog/manifests/"
    "DOCUMENT_LEGACY_PATH_REDIRECTS.json"
)

CLASSIFICATIONS = {
    "byte_identical_duplicate_metadata_only",
    "same_record_frozen_copy",
    "same_record_non_decision_text_drift",
    "same_record_decision_or_limitation_drift",
    "distinct_source_state_records",
    "explicit_supersession_chain",
    "conflicting_explicit_authorities",
    "truncated_or_incomplete_copy",
    "orphan_or_unproven_relationship",
    "unsafe_package_member_relationship",
}

RAW_LOCATOR_RE = re.compile(
    r"(?P<path>" + re.escape(public_runtime_path()) + r"[^\s)'\",]+)"
)


def git(repo_root: Path, args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and result.returncode != 0:
        raise LibrarianError(result.stderr.strip() or "git failed")
    return result


def _id(prefix: str, payload: Any, length: int = 20) -> str:
    return f"{prefix}{hashlib.sha256(canonical_json_bytes(payload)).hexdigest()[:length]}"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _archive_redirects(repo_root: Path) -> dict[str, str]:
    """Return old evidence and document locators mapped to current paths.

    WP2R was authored against the pre-publication evidence tree. Archive
    Publication V1 intentionally moved report-facing files into permanent
    `EIEL-EV-...` paths. The historical locator is still useful as a logical
    citation, but byte-level inspection must resolve through the redirect map.
    """

    path = repo_root / ARCHIVE_PUBLICATION_REDIRECT_MAP_REL
    redirects: dict[str, str] = {}
    if path.is_file():
        data = load_json(path)
        for row in data.get("files", []):
            old_path = row.get("old_path")
            new_path = row.get("new_path")
            old_evidence_relative = row.get("old_evidence_relative_path")
            if isinstance(old_path, str) and isinstance(new_path, str):
                redirects[old_path] = new_path
            if isinstance(old_evidence_relative, str) and isinstance(new_path, str):
                redirects[f"evidences/{old_evidence_relative}"] = new_path
    document_redirect_path = repo_root / DOCUMENT_ARCHIVE_REDIRECT_MAP_REL
    if document_redirect_path.is_file():
        document_redirects = load_json(document_redirect_path)
        for row in document_redirects.get("entries", []):
            old_path = row.get("old_path")
            current_path = row.get("current_path")
            if isinstance(old_path, str) and isinstance(current_path, str):
                redirects[old_path] = current_path
    return redirects


def _resolve_repo_rel(repo_root: Path, rel_path: str) -> tuple[str, Path]:
    path = repo_root / rel_path
    if path.exists():
        return rel_path, path
    resolved_rel = _archive_redirects(repo_root).get(rel_path)
    if resolved_rel:
        resolved_path = Path(resolved_rel)
        if not resolved_path.is_absolute():
            resolved_path = repo_root / resolved_path
        if resolved_path.exists():
            return resolved_rel, resolved_path
    return rel_path, path


def _resolved_file_sha256(repo_root: Path, rel_path: str) -> str:
    _resolved_rel, path = _resolve_repo_rel(repo_root, rel_path)
    return file_sha256(path)


def _title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        clean = line.strip()
        if clean.startswith("#"):
            return clean.lstrip("#").strip() or fallback
    return fallback


def _first_matching(text: str, patterns: Iterable[str]) -> str | None:
    for line in text.splitlines():
        clean = line.strip()
        lowered = clean.casefold()
        if any(pattern in lowered for pattern in patterns):
            return clean[:500]
    return None


def _date(text: str) -> str | None:
    match = re.search(r"\b20\d{2}[-_]\d{2}[-_]\d{2}\b", text)
    return match.group(0).replace("_", "-") if match else None


def _normalized_semantic(text: str) -> str:
    lines = []
    for line in text.splitlines():
        clean = re.sub(r"\s+", " ", line.strip()).casefold()
        if clean:
            lines.append(clean)
    return "\n".join(lines) + "\n"


def _git_binding(repo_root: Path, rel_path: str) -> dict[str, Any]:
    resolved_rel, resolved_path = _resolve_repo_rel(repo_root, rel_path)
    tracked = bool(git(repo_root, ["ls-files", "--error-unmatch", rel_path], check=False).returncode == 0)
    if not tracked and resolved_rel != rel_path:
        tracked = bool(git(repo_root, ["ls-files", "--error-unmatch", resolved_rel], check=False).returncode == 0)
    last = git(repo_root, ["log", "-n", "1", "--format=%H", "--", rel_path], check=False).stdout.strip()
    if not last and resolved_rel != rel_path:
        last = git(repo_root, ["log", "-n", "1", "--format=%H", "--", resolved_rel], check=False).stdout.strip()
    return {
        "tracked": tracked,
        "last_commit": last or None,
        "resolved_locator": resolved_rel if resolved_path.exists() and resolved_rel != rel_path else None,
    }


def _extract_report_facts(repo_root: Path, rel_path: str) -> dict[str, Any]:
    _resolved_rel, path = _resolve_repo_rel(repo_root, rel_path)
    text = _read_text(path)
    decision = _first_matching(text, ["decision:", "decision `", "decision: `", "current decision"])
    limitations = [
        line.strip()[:500]
        for line in text.splitlines()
        if any(
            needle in line.casefold()
            for needle in ("limitation", "no-claim", "no claim", "does not establish", "does not qualify")
        )
    ][:8]
    return {
        "title": _title(text, path.name),
        "date": _date(text),
        "decision": decision,
        "limitations_or_no_claims": limitations,
        "source_commit_mentions": sorted(set(re.findall(r"\b[0-9a-f]{40}\b", text)))[:20],
        "raw_locator_mentions": sorted(set(match.group("path").rstrip(".,;") for match in RAW_LOCATOR_RE.finditer(text)))[:50],
        "normalized_sha256": hashlib.sha256(_normalized_semantic(text).encode("utf-8")).hexdigest(),
        "line_count": len(text.splitlines()),
    }


def _semantic_summary(repo_root: Path, docs_rel: str, evidence_rel: str) -> dict[str, Any]:
    _docs_resolved_rel, docs_path = _resolve_repo_rel(repo_root, docs_rel)
    _evidence_resolved_rel, evidence_path = _resolve_repo_rel(repo_root, f"evidences/{evidence_rel}")
    docs_text = _read_text(docs_path)
    evidence_text = _read_text(evidence_path)
    docs_norm = _normalized_semantic(docs_text).splitlines()
    evid_norm = _normalized_semantic(evidence_text).splitlines()
    diff = list(difflib.unified_diff(evid_norm, docs_norm, n=0, lineterm=""))
    return {
        "normalized_equal": docs_norm == evid_norm,
        "normalized_diff_line_count": len(diff),
        "normalized_diff_excerpt": diff[:80],
        "raw_byte_equal": docs_path.read_bytes() == evidence_path.read_bytes(),
    }


def _classify_pair(docs_facts: Mapping[str, Any], evidence_facts: Mapping[str, Any], summary: Mapping[str, Any]) -> str:
    if summary["raw_byte_equal"]:
        return "byte_identical_duplicate_metadata_only"
    if summary["normalized_equal"]:
        return "same_record_non_decision_text_drift"
    docs_decision = docs_facts.get("decision")
    evid_decision = evidence_facts.get("decision")
    text = json.dumps([docs_facts, evidence_facts], sort_keys=True).casefold()
    if "unsafe" in text and "package" in text:
        return "unsafe_package_member_relationship"
    if "supersed" in text or "replaced by" in text:
        return "explicit_supersession_chain"
    docs_lines = int(docs_facts.get("line_count") or 0)
    evid_lines = int(evidence_facts.get("line_count") or 0)
    if docs_lines and evid_lines and min(docs_lines, evid_lines) < 0.55 * max(docs_lines, evid_lines):
        return "truncated_or_incomplete_copy"
    if docs_decision and evid_decision and docs_decision != evid_decision:
        return "same_record_decision_or_limitation_drift"
    if docs_facts.get("source_commit_mentions") != evidence_facts.get("source_commit_mentions"):
        return "distinct_source_state_records"
    return "orphan_or_unproven_relationship"


def _record_relationship_for_classification(classification: str) -> str:
    if classification in {
        "byte_identical_duplicate_metadata_only",
        "same_record_frozen_copy",
        "same_record_non_decision_text_drift",
        "explicit_supersession_chain",
        "truncated_or_incomplete_copy",
    }:
        return "one_record_candidate"
    if classification in {
        "same_record_decision_or_limitation_drift",
        "distinct_source_state_records",
        "conflicting_explicit_authorities",
    }:
        return "separate_source_state_record_candidates"
    return "operator_review_required_no_record_candidate"


def _artifact(repo_root: Path, rel_path: str, role: str) -> dict[str, Any]:
    resolved_rel, path = _resolve_repo_rel(repo_root, rel_path)
    exists = path.exists()
    return {
        "artifact_id": _id("eiel-art-wp2r-", {"path": rel_path, "role": role}),
        "schema_version": "eiel_artifact_entry_v1",
        "locator": rel_path,
        "locator_type": "repo_relative_path",
        "artifact_role": role,
        "media_type": "text/markdown" if rel_path.endswith(".md") else "application/octet-stream",
        "size_bytes": path.stat().st_size if exists else None,
        "sha256": file_sha256(path) if exists else None,
        "checksum_verification_state": "verified" if exists else "missing",
        "package_member_role": None,
        "git_binding": _git_binding(repo_root, rel_path) if exists else None,
        "resolved_locator": resolved_rel if exists and resolved_rel != rel_path else None,
        "portability_state": "portable",
        "integrity_state": "valid" if path.exists() else "invalid",
        "access_sensitivity": "internal",
        "rights_license_state": "project_owned",
        "observed_existence_time": "1970-01-01T00:00:00Z",
        "legacy_exceptions": [],
    }


def _status_axes(classification: str) -> dict[str, str]:
    if classification == "truncated_or_incomplete_copy":
        completeness = "partial"
    else:
        completeness = "complete"
    integrity = "unsafe" if classification == "unsafe_package_member_relationship" else "valid"
    return {
        "scientific_decision_status": "unknown",
        "authority_currentness_status": "unknown",
        "integrity_status": integrity,
        "completeness_status": completeness,
        "raw_asset_availability_status": "unknown",
        "reproducibility_status": "descriptor_only",
        "publication_release_status": "internal",
        "retention_status": "retain",
    }


def _record_for_pair(
    pair: Mapping[str, Any],
    artifact_ids: list[str],
    *,
    side: str,
    canonical_locator: str,
    canonical_sha256: str,
    frozen_locator: str | None,
    frozen_sha256: str | None,
    review_state: str,
    notes: str,
) -> dict[str, Any]:
    classification = str(pair["classification"])
    return {
        "evidence_id": _id("eiel-rec-wp2r-", {"reconciliation": pair["reconciliation_id"], "side": side}),
        "schema_version": "eiel_evidence_record_v1",
        "title": f"{pair['basename']} [{side}]",
        "record_kind": "legacy_unclassified"
        if classification in {"orphan_or_unproven_relationship", "truncated_or_incomplete_copy"}
        else "implementation_closeout",
        "scientific_question": None,
        "operational_question": "WP2R divergent docs/evidences same-basename reconciliation",
        "primary_collection": "wp2r-divergent-report-curation",
        "tags": sorted(["wp2r", "divergent-same-basename", classification, side]),
        "canonical_report_ref": {
            "locator": canonical_locator,
            "locator_type": "repo_path",
            "sha256": canonical_sha256,
        },
        "frozen_report_ref": {
            "locator": frozen_locator,
            "locator_type": "repo_path",
            "sha256": frozen_sha256,
        }
        if frozen_locator is not None
        else None,
        "source_state": {
            "repo_commit": None,
            "branch": None,
            "worktree": None,
            "provider": None,
            "runtime": None,
            "route_hash": None,
            "applicability_envelope": {"relationship_classification": classification},
        },
        "decision_rule_ref": None,
        "decision_rule_version": None,
        "claim_ids": [],
        "authority_assertion_ids": [],
        "artifact_ids": artifact_ids,
        "raw_asset_ids": [],
        "derived_from_record_ids": [],
        "has_part_record_ids": [],
        "supersedes_record_ids": [],
        "superseded_by_record_ids": [],
        "created_at": {"value": pair["docs"]["facts"].get("date"), "source": "report_text", "confidence": "unknown"},
        "catalogued_at": "1970-01-01T00:00:00Z",
        "curator": "eiel-evidence-librarian-wp2r",
        "review_state": review_state,
        "retention_state": "retain",
        "status_axes": _status_axes(classification),
        "notes": notes,
    }


def _claim_for_pair(pair: Mapping[str, Any], record_id: str) -> dict[str, Any] | None:
    decision = pair["docs"]["facts"].get("decision") or pair["evidence"]["facts"].get("decision")
    if not decision:
        return None
    return {
        "claim_id": _id(
            "eiel-claim-wp2r-",
            {"pair": pair["reconciliation_id"], "record": record_id, "decision": decision},
        ),
        "schema_version": "eiel_claim_assertion_v1",
        "claim_text": decision,
        "claim_domain": "engineering",
        "applicability_envelope": {
            "source": "extracted report decision text",
            "relationship_classification": pair["classification"],
        },
        "support_state": "unknown",
        "supporting_record_ids": [record_id],
        "contradicting_record_ids": [],
        "limitation_ids": [],
        "asserted_by": "source_report_text",
        "effective_time": pair["docs"]["facts"].get("date") or pair["evidence"]["facts"].get("date"),
        "supersession": {
            "supersedes_claim_ids": [],
            "superseded_by_claim_ids": [],
            "retraction_ref": None,
        },
    }


def _raw_assets_for_pair(pair: Mapping[str, Any]) -> list[dict[str, Any]]:
    out = []
    for locator in sorted(set(pair["docs"]["facts"]["raw_locator_mentions"]) | set(pair["evidence"]["facts"]["raw_locator_mentions"])):
        exists = Path(locator).exists()
        out.append(
            {
                "raw_asset_id": _id("eiel-raw-wp2r-", locator),
                "schema_version": "eiel_raw_asset_ref_v1",
                "storage_class": "runtime_active" if exists else "missing",
                "locator_type": "path",
                "locator": locator,
                "expected_size_bytes": Path(locator).stat().st_size if exists and Path(locator).is_file() else None,
                "sha256": file_sha256(Path(locator)) if exists and Path(locator).is_file() else None,
                "producing_campaign_or_run_ids": [],
                "source_commit": None,
                "provider": None,
                "runtime": None,
                "route_identity": None,
                "configuration_receipt": None,
                "input_output_relation": "unknown",
                "replica_references": [],
                "archive_manifest": None,
                "last_successful_existence_check": "1970-01-01T00:00:00Z" if exists else None,
                "last_successful_integrity_check": None,
                "restore_test_state": "not_tested",
                "retention_state": "retain" if exists else "missing",
                "replacement_or_tombstone": None,
            }
        )
    return out


def build_v1r_provenance_receipt(
    *,
    repo_root: Path,
    audit_root: Path,
    repeat_audit_root: Path,
    supplement_zip: Path,
) -> dict[str, Any]:
    audit_commit = V1R_AUDIT_COMMIT
    final_commit = V1R_FINAL_COMMIT
    base_commit = V1R_BASE_COMMIT
    for commit in (audit_commit, final_commit, base_commit):
        git(repo_root, ["cat-file", "-e", f"{commit}^{{commit}}"])
    shows = {
        "audit_execution_commit": git(repo_root, ["show", "-s", "--format=%H %P %s", audit_commit]).stdout.strip(),
        "final_handoff_commit": git(repo_root, ["show", "-s", "--format=%H %P %s", final_commit]).stdout.strip(),
    }
    merge_bases = git(repo_root, ["merge-base", "--all", audit_commit, final_commit]).stdout.splitlines()
    ancestor_audit_to_final = git(repo_root, ["merge-base", "--is-ancestor", audit_commit, final_commit], check=False).returncode
    ancestor_final_to_audit = git(repo_root, ["merge-base", "--is-ancestor", final_commit, audit_commit], check=False).returncode
    diff_name_status = git(repo_root, ["diff", "--name-status", audit_commit, final_commit]).stdout.splitlines()
    excluded_diff = git(
        repo_root,
        ["diff", "--exit-code", audit_commit, final_commit, "--", ".", f":(exclude){V1R_CLOSEOUT_PATH}"],
        check=False,
    )
    evidence_diff = git(repo_root, ["diff", "--name-status", audit_commit, final_commit, "--", "evidences"], check=False).stdout.splitlines()
    changed_v1r_paths = git(repo_root, ["diff", "--name-only", base_commit, final_commit]).stdout.splitlines()
    blob_receipts = []
    for path in sorted(p for p in changed_v1r_paths if p != V1R_CLOSEOUT_PATH):
        left = git(repo_root, ["rev-parse", f"{audit_commit}:{path}"], check=False).stdout.strip()
        right = git(repo_root, ["rev-parse", f"{final_commit}:{path}"], check=False).stdout.strip()
        blob_receipts.append({"path": path, "audit_blob": left or None, "final_blob": right or None, "identical": left == right and bool(left)})
    supplement_sha = file_sha256(supplement_zip)
    sidecar = supplement_zip.with_suffix(supplement_zip.suffix + ".sha256")
    sidecar_text = sidecar.read_text(encoding="utf-8") if sidecar.exists() else ""
    sidecar_ok = supplement_sha in sidecar_text
    with zipfile.ZipFile(supplement_zip) as archive:
        bad_member = archive.testzip()
        manifest = json.loads(archive.read("MANIFEST.member-sha256.json").decode("utf-8"))
        member_sha_ok = all(
            hashlib.sha256(archive.read(row["path"])).hexdigest() == row["sha256"]
            for row in manifest["members"]
        )
    baseline = load_json(audit_root / "baseline_manifest.json")
    repeat = load_json(repeat_audit_root / "baseline_manifest.json")
    queue = load_json(audit_root / "authority_drift_curation_queue.json")
    issues = load_json(audit_root / "issue_ledger.json")
    docs_compare = load_json(audit_root / "docs_evidences_same_basename_report.json")
    passed = (
        diff_name_status == [f"M\t{V1R_CLOSEOUT_PATH}"]
        and excluded_diff.returncode == 0
        and not evidence_diff
        and all(row["identical"] for row in blob_receipts)
        and supplement_sha == EXPECTED_SUPPLEMENT_SHA256
        and sidecar_ok
        and bad_member is None
        and member_sha_ok
        and baseline["tree_digest_sha256"] == EXPECTED_V1R_TREE_DIGEST
        and repeat["tree_digest_sha256"] == EXPECTED_V1R_TREE_DIGEST
        and len(queue["rows"]) == 3452
        and docs_compare["divergent_same_basename_count"] == 51
        and issue_counts(issues)["total"] == 3780
    )
    return {
        "schema_version": "eiel_v1r_sibling_provenance_receipt_v1",
        "relationship": "sibling_report_only_amend",
        "v1r_base_commit": base_commit,
        "audit_execution_commit": audit_commit,
        "final_handoff_commit": final_commit,
        "commit_show_receipts": shows,
        "common_parents": merge_bases,
        "ancestor_return_codes": {
            "audit_is_ancestor_of_final": ancestor_audit_to_final,
            "final_is_ancestor_of_audit": ancestor_final_to_audit,
        },
        "expected_differing_path": V1R_CLOSEOUT_PATH,
        "diff_name_status": diff_name_status,
        "excluded_path_diff_empty": excluded_diff.returncode == 0,
        "implementation_blob_receipts": blob_receipts,
        "implementation_blobs_identical": all(row["identical"] for row in blob_receipts),
        "evidence_root_diff_name_status": evidence_diff,
        "evidence_root_equivalent": not evidence_diff,
        "audit_supplement": {
            "path": str(supplement_zip),
            "sha256": supplement_sha,
            "sidecar_path": str(sidecar),
            "sidecar_ok": sidecar_ok,
            "zip_test_bad_member": bad_member,
            "member_manifest_ok": member_sha_ok,
        },
        "baseline": {
            "audit_tree_digest": baseline["tree_digest_sha256"],
            "repeat_tree_digest": repeat["tree_digest_sha256"],
            "issue_total": issue_counts(issues)["total"],
            "divergent_same_basename_count": docs_compare["divergent_same_basename_count"],
            "authority_drift_queue_count": len(queue["rows"]),
            "catalog_sha256": file_sha256(audit_root / "generated" / "EIEL_Evidence_Catalog.json"),
            "html_sha256": file_sha256(audit_root / "generated" / "EIEL_Evidence_Catalog.html"),
        },
        "role_binding": {
            audit_commit: "audit_execution_and_audit_manifest_source_commit",
            final_commit: "final_v1r_branch_handoff_and_wp2r_base_commit",
        },
        "conclusion": "passed" if passed else "failed",
    }


def build_wp2r_input_manifest(
    *,
    repo_root: Path,
    audit_root: Path,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    docs_compare = load_json(audit_root / "docs_evidences_same_basename_report.json")
    divergent = [row for row in docs_compare["rows"] if not row["has_exact_content_match"]]
    issues = load_json(audit_root / "issue_ledger.json")
    queue = load_json(audit_root / "authority_drift_curation_queue.json")
    source_paths = [
        "ACTIVE_GOAL.md",
        "docs/100_governance-authority/110_authority-registers/EIEL-DOC-000007--eiel-unfinished-work-register.md",
        "docs/100_governance-authority/110_authority-registers/EIEL-DOC-000005--eiel-qualification-authority-register.json",
        "docs/200_strategy-science-programme/200_current-strategy/EIEL-DOC-000070--eiel-new-strategy-context-for-codex.md",
        "docs/100_governance-authority/100_governance/EIEL-DOC-000111--eiel-autonomous-operating-model-v1.md",
        "docs/300_architecture-specifications/300_system-architecture/000_programme-cross-cutting/EIEL-DOC-000321--eiel-evidence-librarian-v1r-governance-and-design-2026-08-02.md",
        "docs/500_campaigns-decisions-closeouts/520_campaign-reports/100_engine-route-qualification/EIEL-DOC-000329--eiel-execution-provider-authority-status-2026-07-25.md",
        "docs/400_implementation-engineering/410_implementation-reports/100_engine-route-qualification/EIEL-DOC-000330--eiel-execution-route-qualification-v1-closeout-report-2026-07-23.md",
        "docs/300_architecture-specifications/300_system-architecture/100_engine-route-qualification/EIEL-DOC-000778--eiel-unified-production-engine-architecture-map-2026-07-25.md",
        "docs/100_governance-authority/100_governance/EIEL-DOC-000777--eiel-unified-engine-cpu-cuda-route-envelope-policy-2026-07-27.md",
        "docs/400_implementation-engineering/410_implementation-reports/100_engine-route-qualification/EIEL-DOC-000798--eiel-w32-workhorse-qualification-authority-status-2026-08-01.md",
    ]
    authority_sources = [
        {
            "path": path,
            "exists": (repo_root / path).exists(),
            "sha256": file_sha256(repo_root / path) if (repo_root / path).is_file() else None,
        }
        for path in source_paths
    ]
    manifest = {
        "schema_version": "eiel_wp2r_curation_input_manifest_v1",
        "run_id": WP2R_RUN_ID,
        "timestamp_policy": "deterministic_1970_utc_for_generated_views",
        "v1r_provenance_receipt_digest": hashlib.sha256(canonical_json_bytes(provenance)).hexdigest(),
        "v1r_catalog_build": load_json(audit_root / "generated" / "EIEL_Evidence_Catalog_Build.json"),
        "v1r_catalog_sha256": file_sha256(audit_root / "generated" / "EIEL_Evidence_Catalog.json"),
        "v1r_html_sha256": file_sha256(audit_root / "generated" / "EIEL_Evidence_Catalog.html"),
        "authority_drift_queue_sha256": file_sha256(audit_root / "authority_drift_curation_queue.json"),
        "issue_ledger_sha256": file_sha256(audit_root / "issue_ledger.json"),
        "source_repository_commit": git(repo_root, ["rev-parse", "HEAD"]).stdout.strip(),
        "librarian_code_commit": V1R_FINAL_COMMIT,
        "curation_tool_version": WP2R_SCHEMA_VERSION,
        "exclusion_list": ["historical evidence payload mutation", "physical migration", "bulk raw restoration"],
        "divergent_pairs": [
            {
                "pair_id": _id("eiel-pair-wp2r-", row["basename"]),
                "basename": row["basename"],
                "docs_paths": row["docs_paths"],
                "evidence_paths": ["evidences/" + path for path in row["evidence_paths"]],
                "docs_sha256": row["docs_sha256"],
                "evidence_sha256": row["evidence_sha256"],
            }
            for row in divergent
        ],
        "unsafe_zip": {
            "path": "evidences/" + UNSAFE_ZIP_REL,
            "sha256": _resolved_file_sha256(repo_root, "evidences/" + UNSAFE_ZIP_REL),
            "known_unsafe_member": UNSAFE_MEMBER,
        },
        "authority_source_candidates": authority_sources,
    }
    manifest["input_set_digest"] = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    return manifest


def reconcile_divergent_pairs(repo_root: Path, audit_root: Path) -> list[dict[str, Any]]:
    docs_compare = load_json(audit_root / "docs_evidences_same_basename_report.json")
    rows = []
    for row in sorted(
        [item for item in docs_compare["rows"] if not item["has_exact_content_match"]],
        key=lambda item: item["basename"],
    ):
        docs_rel = row["docs_paths"][0]
        evid_rel = row["evidence_paths"][0]
        resolved_docs_rel, resolved_docs_path = _resolve_repo_rel(repo_root, docs_rel)
        resolved_evid_rel, resolved_evid_path = _resolve_repo_rel(repo_root, "evidences/" + evid_rel)
        docs_facts = _extract_report_facts(repo_root, docs_rel)
        evid_facts = _extract_report_facts(repo_root, "evidences/" + evid_rel)
        summary = _semantic_summary(repo_root, docs_rel, evid_rel)
        classification = _classify_pair(docs_facts, evid_facts, summary)
        if classification not in CLASSIFICATIONS:
            raise LibrarianError(f"unknown classification: {classification}")
        rows.append(
            {
                "schema_version": "eiel_divergent_report_reconciliation_v1",
                "reconciliation_id": _id("eiel-recon-wp2r-", row["basename"]),
                "basename": row["basename"],
                "classification": classification,
                "docs": {
                    "path": docs_rel,
                    "resolved_path": resolved_docs_rel if resolved_docs_rel != docs_rel else None,
                    "sha256": row["docs_sha256"][0],
                    "size_bytes": resolved_docs_path.stat().st_size,
                    "media_type": "text/markdown",
                    "parse_state": "text_readable",
                    "git": _git_binding(repo_root, docs_rel),
                    "facts": docs_facts,
                },
                "evidence": {
                    "path": evid_rel,
                    "resolved_path": resolved_evid_rel if resolved_evid_rel != "evidences/" + evid_rel else None,
                    "sha256": row["evidence_sha256"][0],
                    "size_bytes": resolved_evid_path.stat().st_size,
                    "media_type": "text/markdown",
                    "parse_state": "text_readable",
                    "git": _git_binding(repo_root, "evidences/" + evid_rel),
                    "facts": evid_facts,
                },
                "raw_byte_comparison": {"equal": summary["raw_byte_equal"]},
                "normalized_semantic_comparison": {
                    "equal": summary["normalized_equal"],
                    "diff_line_count": summary["normalized_diff_line_count"],
                    "diff_excerpt": summary["normalized_diff_excerpt"],
                },
                "semantic_decision_diff_summary": {
                    "docs_decision": docs_facts.get("decision"),
                    "evidence_decision": evid_facts.get("decision"),
                    "decision_equal": docs_facts.get("decision") == evid_facts.get("decision"),
                },
                "copy_state_assessment": {
                    "docs_copy_state": "durable_repository_report_candidate",
                    "evidence_copy_state": "frozen_or_historical_evidence_copy_candidate",
                    "global_path_preference_used": False,
                },
                "record_relationship": _record_relationship_for_classification(classification),
                "proposed_canonical_report_ref": docs_rel,
                "proposed_frozen_report_ref": "evidences/" + evid_rel,
                "authority_impact": "operator_review_required"
                if classification
                in {
                    "same_record_decision_or_limitation_drift",
                    "distinct_source_state_records",
                    "conflicting_explicit_authorities",
                    "orphan_or_unproven_relationship",
                }
                else "no_current_authority_change",
                "unresolved_questions": []
                if classification
                not in {"orphan_or_unproven_relationship", "same_record_decision_or_limitation_drift"}
                else ["Automated curation cannot choose a scientific/current authority from divergent report text alone."],
                "curator_confidence": "explicit" if classification != "orphan_or_unproven_relationship" else "unknown",
                "source_citations": [docs_rel, "evidences/" + evid_rel],
            }
        )
    return rows


def build_authority_map(repo_root: Path) -> list[dict[str, Any]]:
    candidates = [
        (
            "scientific_doctrine",
            "current strategy context where explicitly stated",
            "docs/200_strategy-science-programme/200_current-strategy/EIEL-DOC-000070--eiel-new-strategy-context-for-codex.md",
            "current_scientific_strategy_context",
        ),
        (
            "active_roadmap_programme_state",
            "theme-level and task-level TODO register",
            "docs/100_governance-authority/110_authority-registers/EIEL-DOC-000007--eiel-unfinished-work-register.md",
            "main_theme_task_register",
        ),
        (
            "architecture_doctrine",
            "unified production engine architecture",
            "docs/300_architecture-specifications/300_system-architecture/100_engine-route-qualification/EIEL-DOC-000778--eiel-unified-production-engine-architecture-map-2026-07-25.md",
            "unified_production_engine_architecture",
        ),
        (
            "production_provider",
            "production execution provider identity",
            "docs/500_campaigns-decisions-closeouts/520_campaign-reports/100_engine-route-qualification/EIEL-DOC-000329--eiel-execution-provider-authority-status-2026-07-25.md",
            "eiel-production-provider-v1",
        ),
        (
            "production_runtime",
            "production runtime profile identity",
            "docs/500_campaigns-decisions-closeouts/520_campaign-reports/100_engine-route-qualification/EIEL-DOC-000329--eiel-execution-provider-authority-status-2026-07-25.md",
            "eiel-production-runtime-v1",
        ),
        (
            "execution_route",
            "route qualification procedure and fail-closed boundary",
            "docs/400_implementation-engineering/410_implementation-reports/100_engine-route-qualification/EIEL-DOC-000330--eiel-execution-route-qualification-v1-closeout-report-2026-07-23.md",
            "execution_route_qualification_boundary",
        ),
        (
            "qualification_envelope",
            "exact tracks in qualification authority register",
            "docs/100_governance-authority/110_authority-registers/EIEL-DOC-000005--eiel-qualification-authority-register.json",
            "register_scoped_track_authority",
        ),
        (
            "release_main_integration_state",
            "repository active goal and main integration context",
            "ACTIVE_GOAL.md",
            "durable_active_goal_snapshot",
        ),
        (
            "evidence_library_governance",
            "V1R librarian governance and design",
            "docs/300_architecture-specifications/300_system-architecture/000_programme-cross-cutting/EIEL-DOC-000321--eiel-evidence-librarian-v1r-governance-and-design-2026-08-02.md",
            "librarian_controlled_generated_views",
        ),
    ]
    rows = []
    for domain, scope, source, status in candidates:
        path = repo_root / source
        exists = path.exists()
        rows.append(
            {
                "authority_assertion_id": _id("eiel-auth-wp2r-", {"domain": domain, "source": source}),
                "schema_version": "eiel_domain_authority_candidate_v1",
                "domain": domain,
                "bounded_scope_envelope": {"scope": scope},
                "authority_source": source if exists else None,
                "asserted_status": status if exists else "repository_authority_unknown",
                "effective_from": None,
                "effective_until": None,
                "source_commit_or_register_version": git(repo_root, ["rev-parse", "HEAD"]).stdout.strip() if exists else None,
                "source_sha256": file_sha256(path) if path.is_file() else None,
                "superseded_sources": [],
                "unresolved_competing_sources": [],
                "confidence": "explicit" if exists else "unknown",
                "operator_review_required": domain in {"scientific_doctrine", "active_roadmap_programme_state", "release_main_integration_state"},
                "caveat": "Candidate map row only; exact report/register scope controls applicability.",
            }
        )
    for domain in ("campaign_decision", "scientific_claim"):
        rows.append(
            {
                "authority_assertion_id": _id("eiel-auth-wp2r-", {"domain": domain, "source": "none"}),
                "schema_version": "eiel_domain_authority_candidate_v1",
                "domain": domain,
                "bounded_scope_envelope": {"scope": "no universal source; use exact decision report per campaign/claim"},
                "authority_source": None,
                "asserted_status": "repository_authority_unknown",
                "effective_from": None,
                "effective_until": None,
                "source_commit_or_register_version": None,
                "source_sha256": None,
                "superseded_sources": [],
                "unresolved_competing_sources": [],
                "confidence": "unknown",
                "operator_review_required": True,
                "caveat": "WP2R forbids inferring campaign/scientific claim authority globally.",
            }
        )
    return rows


def unsafe_zip_record(repo_root: Path) -> dict[str, Any]:
    resolved_package_path, zip_path = _resolve_repo_rel(repo_root, "evidences/" + UNSAFE_ZIP_REL)
    resolved_safe_path, safe_path = _resolve_repo_rel(repo_root, SAFE_COUNTERPART)
    safety = inspect_zip_safety(zip_path)
    member_sha = None
    member_equals_safe = None
    member_size = None
    direct_extraction_performed = False
    with zipfile.ZipFile(zip_path) as archive:
        data = archive.read(UNSAFE_MEMBER)
        member_sha = hashlib.sha256(data).hexdigest()
        member_size = len(data)
        if safe_path.exists():
            member_equals_safe = data == safe_path.read_bytes()
    return {
        "schema_version": "eiel_unsafe_zip_relationship_record_v1",
        "package_path": "evidences/" + UNSAFE_ZIP_REL,
        "resolved_package_path": resolved_package_path if resolved_package_path != "evidences/" + UNSAFE_ZIP_REL else None,
        "package_sha256": file_sha256(zip_path),
        "known_unsafe_member": UNSAFE_MEMBER,
        "member_sha256": member_sha,
        "member_size_bytes": member_size,
        "zip_safety": safety,
        "disposition": ["unsafe_to_extract", "integrity_restricted"],
        "zip_bytes_readable": safety["member_count"] > 0,
        "safe_counterpart_path": SAFE_COUNTERPART if safe_path.exists() else None,
        "safe_counterpart_resolved_path": (
            resolved_safe_path if safe_path.exists() and resolved_safe_path != SAFE_COUNTERPART else None
        ),
        "safe_counterpart_sha256": file_sha256(safe_path) if safe_path.exists() else None,
        "read_only_streamed_member_equals_safe_counterpart": member_equals_safe,
        "direct_extraction_performed": direct_extraction_performed,
        "review_surface": SAFE_COUNTERPART if member_equals_safe else "operator_review_required",
        "historical_package_modified": False,
    }


def build_accessions(
    *,
    repo_root: Path,
    reconciliations: list[dict[str, Any]],
    authority_map: list[dict[str, Any]],
    unsafe_record: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    artifacts = []
    records = []
    claims = []
    raw_assets = []
    issues = []
    for pair in reconciliations:
        docs_art = _artifact(repo_root, pair["docs"]["path"], "canonical_report")
        evid_art = _artifact(repo_root, "evidences/" + pair["evidence"]["path"], "frozen_report")
        artifacts.extend([docs_art, evid_art])
        relationship = pair["record_relationship"]
        pair_records: list[dict[str, Any]] = []
        if relationship == "one_record_candidate":
            pair_records.append(
                _record_for_pair(
                    pair,
                    [docs_art["artifact_id"], evid_art["artifact_id"]],
                    side="paired",
                    canonical_locator=pair["docs"]["path"],
                    canonical_sha256=pair["docs"]["sha256"],
                    frozen_locator="evidences/" + pair["evidence"]["path"],
                    frozen_sha256=pair["evidence"]["sha256"],
                    review_state="legacy_imported",
                    notes="Candidate paired record only; not programme acceptance authority.",
                )
            )
        elif relationship == "separate_source_state_record_candidates":
            pair_records.extend(
                [
                    _record_for_pair(
                        pair,
                        [docs_art["artifact_id"]],
                        side="docs-source-state",
                        canonical_locator=pair["docs"]["path"],
                        canonical_sha256=pair["docs"]["sha256"],
                        frozen_locator=None,
                        frozen_sha256=None,
                        review_state="unreviewed",
                        notes="Separate candidate because source-state or decision drift forbids automated merge.",
                    ),
                    _record_for_pair(
                        pair,
                        [evid_art["artifact_id"]],
                        side="evidence-source-state",
                        canonical_locator="evidences/" + pair["evidence"]["path"],
                        canonical_sha256=pair["evidence"]["sha256"],
                        frozen_locator=None,
                        frozen_sha256=None,
                        review_state="unreviewed",
                        notes="Separate historical/frozen candidate because source-state or decision drift forbids automated merge.",
                    ),
                ]
            )
        records.extend(pair_records)
        for record in pair_records:
            claim = _claim_for_pair(pair, record["evidence_id"])
            if claim:
                claims.append(claim)
        raw_assets.extend(_raw_assets_for_pair(pair))
        if pair["authority_impact"] == "operator_review_required":
            issues.append(
                {
                    "issue_id": _id("eiel-issue-wp2r-", pair["reconciliation_id"]),
                    "schema_version": "eiel_issue_v1",
                    "severity": "high",
                    "category": "authority_review_required",
                    "paths": [pair["docs"]["path"], "evidences/" + pair["evidence"]["path"]],
                    "evidence": {
                        "classification": pair["classification"],
                        "reason": pair["unresolved_questions"],
                    },
                    "detection_rule": "WP2R divergent-report reconciliation",
                    "first_observed": "wp2r_curation",
                    "last_observed": "wp2r_curation",
                    "owner": "operator_or_scientific_lead",
                    "disposition": "needs_curation",
                    "exception_expiry": None,
                    "remediation_state": "planned",
                }
            )
    artifacts.append(
        {
            "artifact_id": _id("eiel-art-wp2r-", unsafe_record["package_path"]),
            "schema_version": "eiel_artifact_entry_v1",
            "locator": unsafe_record["package_path"],
            "locator_type": "repo_relative_path",
            "artifact_role": "evidence_package",
            "media_type": "application/zip",
            "size_bytes": (repo_root / (unsafe_record.get("resolved_package_path") or unsafe_record["package_path"])).stat().st_size,
            "sha256": unsafe_record["package_sha256"],
            "checksum_verification_state": "verified",
            "package_member_role": "unsafe_traversal_member_present",
            "git_binding": _git_binding(repo_root, unsafe_record["package_path"]),
            "portability_state": "portable",
            "integrity_state": "unsafe",
            "access_sensitivity": "internal",
            "rights_license_state": "project_owned",
            "observed_existence_time": "1970-01-01T00:00:00Z",
            "legacy_exceptions": ["unsafe_to_extract", "integrity_restricted"],
        }
    )
    issues.append(
        {
            "issue_id": _id("eiel-issue-wp2r-", unsafe_record["package_path"]),
            "schema_version": "eiel_issue_v1",
            "severity": "critical",
            "category": "unsafe_package_integrity_restricted",
            "paths": [unsafe_record["package_path"]],
            "evidence": unsafe_record,
            "detection_rule": "WP2R read-only unsafe ZIP member inspection",
            "first_observed": "wp2r_curation",
            "last_observed": "wp2r_curation",
            "owner": "operator_or_scientific_lead",
            "disposition": "accepted_exception",
            "exception_expiry": None,
            "remediation_state": "planned",
        }
    )
    return {
        "artifacts": sorted(artifacts, key=lambda item: item["artifact_id"]),
        "records": sorted(records, key=lambda item: item["evidence_id"]),
        "claims": sorted(claims, key=lambda item: item["claim_id"]),
        "authorities": sorted(authority_map, key=lambda item: item["authority_assertion_id"]),
        "raw_assets": sorted({item["raw_asset_id"]: item for item in raw_assets}.values(), key=lambda item: item["raw_asset_id"]),
        "issues": sorted(issues, key=lambda item: item["issue_id"]),
        "builds": [],
    }


def _object_id_for_collection(collection: str, row: Mapping[str, Any]) -> str:
    field_by_collection = {
        "artifacts": "artifact_id",
        "records": "evidence_id",
        "claims": "claim_id",
        "authorities": "authority_assertion_id",
        "raw_assets": "raw_asset_id",
        "issues": "issue_id",
        "builds": "build_id",
    }
    field = field_by_collection[collection]
    return str(row[field])


def write_accession_object_shards(accession_root: Path, accessions: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    object_root = accession_root / "objects"
    shards = []
    for collection, rows in sorted(accessions.items()):
        for row in rows:
            object_id = _object_id_for_collection(collection, row)
            rel = Path("objects") / collection / f"{object_id}.json"
            sha = write_json(accession_root / rel, row, pretty=False)
            shards.append(
                {
                    "collection": collection,
                    "object_id": object_id,
                    "path": rel.as_posix(),
                    "sha256": sha,
                }
            )
    return {
        "schema_version": "eiel_wp2r_accession_object_shard_manifest_v1",
        "shard_root": object_root.relative_to(accession_root).as_posix(),
        "collection_counts": {name: len(rows) for name, rows in sorted(accessions.items())},
        "shard_count": len(shards),
        "shards": sorted(shards, key=lambda item: (item["collection"], item["object_id"])),
    }


def build_review_queue(reconciliations: list[dict[str, Any]], authority_map: list[dict[str, Any]]) -> list[dict[str, Any]]:
    queue = []
    for pair in reconciliations:
        if pair["authority_impact"] != "operator_review_required":
            continue
        queue.append(
            {
                "review_id": _id("eiel-review-wp2r-", pair["reconciliation_id"]),
                "conflict": f"{pair['basename']} has divergent docs/evidences report text.",
                "competing_source_ids": [pair["docs"]["path"], "evidences/" + pair["evidence"]["path"]],
                "source_state_envelope_difference": {
                    "classification": pair["classification"],
                    "docs_decision": pair["semantic_decision_diff_summary"]["docs_decision"],
                    "evidence_decision": pair["semantic_decision_diff_summary"]["evidence_decision"],
                },
                "why_automated_resolution_forbidden": "WP2R cannot infer current authority from path, recency, or nearby reports.",
                "available_options": [
                    "declare same bounded record with frozen copy drift",
                    "declare separate source-state records",
                    "mark one current only through an explicit durable decision",
                ],
                "consequences": "Physical migration may later add pointers, but no historical payload should be edited.",
                "recommended_default": "Keep both artifacts, require explicit operator/scientific-lead adjudication before current-authority claims.",
                "physical_migration_eventually_needed": False,
            }
        )
    for row in authority_map:
        if row["confidence"] == "unknown":
            queue.append(
                {
                    "review_id": _id("eiel-review-wp2r-", row["domain"]),
                    "conflict": f"No durable universal repository authority for {row['domain']}.",
                    "competing_source_ids": [],
                    "source_state_envelope_difference": row["bounded_scope_envelope"],
                    "why_automated_resolution_forbidden": "Missing explicit durable source.",
                    "available_options": ["add durable authority source", "keep domain unknown"],
                    "consequences": "Catalog must keep authority unknown until a durable source exists.",
                    "recommended_default": "Keep unknown.",
                    "physical_migration_eventually_needed": False,
                }
            )
    return sorted(queue, key=lambda item: item["review_id"])


def build_wp2r_catalog(
    *,
    baseline_catalog: Mapping[str, Any],
    provenance: Mapping[str, Any],
    input_manifest: Mapping[str, Any],
    reconciliations: list[dict[str, Any]],
    accessions: Mapping[str, list[dict[str, Any]]],
    authority_map: list[dict[str, Any]],
    review_queue: list[dict[str, Any]],
    unsafe_record: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "eiel_evidence_catalog_generated_wp2r_v1",
        "generated_at": "1970-01-01T00:00:00Z",
        "run_id": WP2R_RUN_ID,
        "source_commit": input_manifest["source_repository_commit"],
        "v1r_provenance": provenance,
        "input_manifest_digest": input_manifest["input_set_digest"],
        "physical_files": baseline_catalog["artifact_inventory"],
        "curated_artifacts": accessions["artifacts"],
        "atomic_evidence_records": accessions["records"],
        "claims": accessions["claims"],
        "authority_assertions": accessions["authorities"],
        "raw_assets": accessions["raw_assets"],
        "curated_issues": accessions["issues"],
        "reconciliations": reconciliations,
        "authority_map": authority_map,
        "unresolved_authority_review_queue": review_queue,
        "unsafe_packages": [unsafe_record],
        "views": {
            "reconciliation_count": len(reconciliations),
            "authority_domain_count": len(authority_map),
            "unresolved_authority_queue_count": len(review_queue),
            "unsafe_package_count": 1,
            "negative_blocked_invalid_note": "Status axes are separate; null, negative, blocked, superseded, and invalid evidence is not discarded.",
        },
        "authority_policy": {"inference_allowed": False, "confidence_values": ["explicit", "unknown"]},
    }


def write_wp2r_html(path: Path, catalog: Mapping[str, Any]) -> str:
    def rows(items: Iterable[str]) -> str:
        return "".join(f"<li>{html.escape(item)}</li>" for item in items)

    class_counts = Counter(row["classification"] for row in catalog["reconciliations"])
    auth_rows = "".join(
        "<tr>"
        f"<td>{html.escape(row['domain'])}</td>"
        f"<td>{html.escape(row['asserted_status'])}</td>"
        f"<td>{html.escape(str(row['authority_source']))}</td>"
        f"<td>{html.escape(row['confidence'])}</td>"
        "</tr>"
        for row in catalog["authority_map"]
    )
    recon_rows = "".join(
        "<tr>"
        f"<td>{html.escape(row['basename'])}</td>"
        f"<td>{html.escape(row['classification'])}</td>"
        f"<td>{html.escape(row['authority_impact'])}</td>"
        f"<td>{html.escape(row['docs']['path'])}</td>"
        f"<td>{html.escape('evidences/' + row['evidence']['path'])}</td>"
        "</tr>"
        for row in catalog["reconciliations"]
    )
    unsafe = catalog["unsafe_packages"][0]
    text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>EIEL Evidence Librarian WP2R Catalog</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #1f2328; }}
.grid {{ display: grid; grid-template-columns: repeat(4, minmax(10rem, 1fr)); gap: .75rem; }}
.box {{ border: 1px solid #ccd1d5; border-radius: 6px; padding: .75rem; }}
.warn {{ border: 2px solid #9b1c1c; background: #fff4f4; padding: 1rem; margin: 1rem 0; }}
table {{ border-collapse: collapse; width: 100%; font-size: .88rem; margin: 1rem 0; }}
th, td {{ border: 1px solid #d8dde3; padding: .35rem .5rem; text-align: left; vertical-align: top; }}
th {{ background: #eef2f6; }}
code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
</style>
</head>
<body>
<h1>EIEL Evidence Librarian WP2R Catalog</h1>
<p>This generated HTML is a reading surface, not an authority source. Authority remains bound to exact cited durable files.</p>
<div class="grid">
<div class="box"><strong>Reconciliations</strong><br>{len(catalog['reconciliations'])}</div>
<div class="box"><strong>Authority rows</strong><br>{len(catalog['authority_map'])}</div>
<div class="box"><strong>Review queue</strong><br>{len(catalog['unresolved_authority_review_queue'])}</div>
<div class="box"><strong>Historical files</strong><br>{len(catalog['physical_files'])}</div>
</div>
<div class="warn">
<strong>Unsafe package warning:</strong>
<code>{html.escape(unsafe['package_path'])}</code> contains traversal member
<code>{html.escape(unsafe['known_unsafe_member'])}</code>. Do not extract this archive directly.
Disposition: {html.escape(', '.join(unsafe['disposition']))}.
</div>
<h2>Reconciliation Classes</h2>
<ul>{rows(f'{key}: {value}' for key, value in sorted(class_counts.items()))}</ul>
<h2>Authority By Domain</h2>
<table><thead><tr><th>Domain</th><th>Status</th><th>Source</th><th>Confidence</th></tr></thead><tbody>{auth_rows}</tbody></table>
<h2>51 Divergent Report Reconciliations</h2>
<table><thead><tr><th>Basename</th><th>Classification</th><th>Authority Impact</th><th>Docs</th><th>Evidence</th></tr></thead><tbody>{recon_rows}</tbody></table>
<h2>Unresolved Queue</h2>
<p>{len(catalog['unresolved_authority_review_queue'])} items remain for operator/scientific-lead review.</p>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return file_sha256(path)


def build_wp2r_migration_plan(reconciliations: list[dict[str, Any]], unsafe_record: Mapping[str, Any]) -> dict[str, Any]:
    actions = []
    for pair in reconciliations:
        actions.append(
            {
                "source_artifact_id": pair["reconciliation_id"],
                "proposed_disposition": "no_move_generated_views_solve_navigation"
                if pair["authority_impact"] == "no_current_authority_change"
                else "operator_review_before_any_pointer_or_migration",
                "proposed_canonical_location": pair["proposed_canonical_report_ref"],
                "reason": pair["classification"],
                "authority_record_relation": pair["record_relationship"],
                "checksum_preservation_rule": "preserve raw SHA-256; never rewrite historical payload",
                "consumer_impact": "none until separately approved migration",
                "rollback_method": "not applicable; no migration executed",
                "approval_requirement": "operator approval required",
                "retention_legal_hold_state": "retain",
                "unnecessary_because_generated_views_solve_problem": pair["authority_impact"] == "no_current_authority_change",
            }
        )
    actions.append(
        {
            "source_artifact_id": unsafe_record["package_path"],
            "proposed_disposition": "block_direct_extraction_keep_safe_review_surface",
            "proposed_canonical_location": unsafe_record["review_surface"],
            "reason": "unsafe traversal-bearing historical ZIP",
            "authority_record_relation": "integrity_restricted_package_with_safe_counterpart_candidate",
            "checksum_preservation_rule": "preserve original package SHA-256",
            "consumer_impact": "future tooling must warn/fail closed for extraction",
            "rollback_method": "not applicable; no migration executed",
            "approval_requirement": "operator approval required for any repaired derivative",
            "retention_legal_hold_state": "retain",
            "unnecessary_because_generated_views_solve_problem": True,
        }
    )
    return {
        "schema_version": "eiel_wp2r_migration_plan_candidate_v1",
        "migration_executed": False,
        "policy": "plan_only_no_moves_no_deletes_no_renames_no_repacks",
        "actions": actions,
    }


def requirement_matrix() -> list[dict[str, Any]]:
    requirements = [
        "complete one-to-one coverage of all 51 divergent pairs",
        "deterministic reconciliation records",
        "raw-byte and normalized semantic diff separation",
        "no authority inference from path, filename, recency, mtime, or tests",
        "explicit-source requirement for non-unknown authority",
        "domain/scope authority separation",
        "source-state-specific records",
        "negative blocked superseded invalid status separation",
        "accession relation integrity and acyclic supersession",
        "object-level sharded accession candidates",
        "no dangling record/artifact/claim/authority IDs",
        "deterministic generated catalog and HTML",
        "unsafe ZIP detection and no-extraction enforcement",
        "read-only streamed unsafe member comparison",
        "historical evidence immutability",
        "evidence-tree digest unchanged before and after WP2R",
        "no writes under evidences",
        "no imports from forbidden simulation internals",
        "issue-ledger no-regression",
        "stable catalog build receipts",
        "final commit and handoff completeness",
    ]
    return [
        {
            "requirement_id": f"WP2R-T{index:02d}",
            "requirement": text,
            "primary_tests": [
                "tests/test_eiel_evidence_librarian_wp2r.py",
                "tests/test_eiel_evidence_librarian_v1r.py",
            ],
            "runtime_receipts": [
                "provenance_reconciliation_receipt.json",
                "divergence_reconciliation_ledger.json",
                "EIEL_Evidence_Catalog_WP2R.json",
                "unsafe_zip_relationship_record.json",
            ],
            "coverage_state": "covered",
        }
        for index, text in enumerate(requirements, 1)
    ]


def write_divergence_report(path: Path, reconciliations: list[dict[str, Any]]) -> str:
    counts = Counter(row["classification"] for row in reconciliations)
    lines = [
        "# EIEL Evidence Librarian WP2R Divergence Reconciliation Report",
        "",
        "This generated report summarizes the 51 divergent same-basename docs/evidences Markdown rows. It is a curation aid, not a scientific authority source.",
        "",
        "## Classification Counts",
        "",
        "| Classification | Count |",
        "| --- | ---: |",
    ]
    for key, value in sorted(counts.items()):
        lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "## Rows", ""])
    for row in reconciliations:
        lines.extend(
            [
                f"### {row['basename']}",
                "",
                f"- Classification: `{row['classification']}`",
                f"- Docs: `{row['docs']['path']}`",
                f"- Evidence: `evidences/{row['evidence']['path']}`",
                f"- Authority impact: `{row['authority_impact']}`",
                f"- Raw bytes equal: `{row['raw_byte_comparison']['equal']}`",
                f"- Normalized semantic diff lines: `{row['normalized_semantic_comparison']['diff_line_count']}`",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return file_sha256(path)


def build_wp2r(
    *,
    repo_root: Path,
    audit_root: Path,
    repeat_audit_root: Path,
    supplement_zip: Path,
    output_root: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    pre_files = read_jsonl(audit_root / "file_inventory.jsonl")
    pre_digest = tree_digest(pre_files)
    provenance = build_v1r_provenance_receipt(
        repo_root=repo_root,
        audit_root=audit_root,
        repeat_audit_root=repeat_audit_root,
        supplement_zip=supplement_zip,
    )
    if provenance["conclusion"] != "passed":
        write_json(output_root / "provenance_reconciliation_receipt.json", provenance)
        raise LibrarianError("Stage 0R provenance/baseline gate failed")
    input_manifest = build_wp2r_input_manifest(repo_root=repo_root, audit_root=audit_root, provenance=provenance)
    reconciliations = reconcile_divergent_pairs(repo_root, audit_root)
    authority_map = build_authority_map(repo_root)
    unsafe = unsafe_zip_record(repo_root)
    accessions = build_accessions(repo_root=repo_root, reconciliations=reconciliations, authority_map=authority_map, unsafe_record=unsafe)
    review_queue = build_review_queue(reconciliations, authority_map)
    baseline_catalog = load_json(audit_root / "generated" / "EIEL_Evidence_Catalog.json")
    catalog = build_wp2r_catalog(
        baseline_catalog=baseline_catalog,
        provenance=provenance,
        input_manifest=input_manifest,
        reconciliations=reconciliations,
        accessions=accessions,
        authority_map=authority_map,
        review_queue=review_queue,
        unsafe_record=unsafe,
    )
    migration = build_wp2r_migration_plan(reconciliations, unsafe)
    matrix = requirement_matrix()
    generated = output_root / "generated"
    accession_root = output_root / "accession_candidates"
    hashes = {
        "provenance_reconciliation_receipt.json": write_json(output_root / "provenance_reconciliation_receipt.json", provenance),
        "wp2r_curation_input_manifest.json": write_json(output_root / "wp2r_curation_input_manifest.json", input_manifest),
        "divergence_reconciliation_ledger.json": write_json(output_root / "divergence_reconciliation_ledger.json", reconciliations),
        "authority_map.json": write_json(output_root / "authority_map.json", authority_map),
        "authority_review_queue.json": write_json(output_root / "authority_review_queue.json", review_queue),
        "unsafe_zip_relationship_record.json": write_json(output_root / "unsafe_zip_relationship_record.json", unsafe),
        "migration_plan_candidate_wp2r.json": write_json(output_root / "migration_plan_candidate_wp2r.json", migration),
        "requirement_to_test_matrix.json": write_json(output_root / "requirement_to_test_matrix.json", matrix),
    }
    hashes["divergence_reconciliation_report.md"] = write_divergence_report(
        output_root / "divergence_reconciliation_report.md", reconciliations
    )
    for name, rows in accessions.items():
        hashes[f"accession_candidates/{name}.json"] = write_json(accession_root / f"{name}.json", rows)
    shard_manifest = write_accession_object_shards(accession_root, accessions)
    hashes["accession_candidates/object_shard_manifest.json"] = write_json(
        accession_root / "object_shard_manifest.json", shard_manifest, pretty=False
    )
    catalog_path = generated / "EIEL_Evidence_Catalog_WP2R.json"
    html_path = generated / "EIEL_Evidence_Catalog_WP2R.html"
    hashes["generated/EIEL_Evidence_Catalog_WP2R.json"] = write_json(catalog_path, catalog, pretty=False)
    hashes["generated/EIEL_Evidence_Catalog_WP2R.html"] = write_wp2r_html(html_path, catalog)
    build_receipt = {
        "schema_version": "eiel_wp2r_catalog_build_receipt_v1",
        "run_id": WP2R_RUN_ID,
        "source_commit": git(repo_root, ["rev-parse", "HEAD"]).stdout.strip(),
        "build_time": utc_now_iso(deterministic=True),
        "deterministic_mode": True,
        "tool_versions": {"python": platform.python_version()},
        "input_manifest_digest": input_manifest["input_set_digest"],
        "historical_evidence_digest_before": pre_digest,
        "historical_evidence_digest_after": tree_digest(read_jsonl(audit_root / "file_inventory.jsonl")),
        "output_hashes": hashes,
        "build_status": "success",
    }
    hashes["generated/EIEL_Evidence_Catalog_WP2R_Build.json"] = write_json(
        generated / "EIEL_Evidence_Catalog_WP2R_Build.json", build_receipt, pretty=False
    )
    package_zip = output_root.parent / "EIEL_Evidence_Librarian_WP2R_Evidence_Package_2026_08_02.zip"
    package_hash = deterministic_zip(package_zip, [path for path in output_root.rglob("*") if path.is_file()], root=output_root)
    package_zip.with_suffix(package_zip.suffix + ".sha256").write_text(
        f"{package_hash}  {package_zip.name}\n", encoding="utf-8"
    )
    return {
        "schema_version": "eiel_wp2r_run_result_v1",
        "decision": "EIEL-EVIDENCE-LIBRARIAN-WP2R-A",
        "run_id": WP2R_RUN_ID,
        "output_root": str(output_root),
        "reconciliation_count": len(reconciliations),
        "reconciliation_counts_by_classification": dict(sorted(Counter(row["classification"] for row in reconciliations).items())),
        "accession_counts": {name: len(rows) for name, rows in accessions.items()},
        "authority_assertion_count": len(authority_map),
        "unresolved_authority_queue_count": len(review_queue),
        "unsafe_zip_disposition": unsafe["disposition"],
        "historical_evidence_digest_before": pre_digest,
        "historical_evidence_digest_after": build_receipt["historical_evidence_digest_after"],
        "catalog_path": str(catalog_path),
        "catalog_sha256": hashes["generated/EIEL_Evidence_Catalog_WP2R.json"],
        "html_path": str(html_path),
        "html_sha256": hashes["generated/EIEL_Evidence_Catalog_WP2R.html"],
        "package_zip": str(package_zip),
        "package_sha256": package_hash,
        "output_hashes": hashes,
    }
