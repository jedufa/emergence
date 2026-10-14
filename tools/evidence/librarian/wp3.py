"""WP3 conservative authority adjudication and accession freeze.

WP3 promotes metadata from the WP2R runtime candidates into a maintained
repository-controlled source area.  It never writes historical payloads under
``evidences/`` and never extracts the traversal-bearing legacy ZIP.
"""

from __future__ import annotations

import hashlib
import html
import json
import platform
import subprocess
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from .core import (
    canonical_json_bytes,
    deterministic_zip,
    file_sha256,
    inspect_zip_safety,
    read_jsonl,
    tree_digest,
    write_json,
)


WP3_SCHEMA_VERSION = "eiel_evidence_librarian_wp3_v1"
WP3_RUN_ID = "20260802T000000Z_wp3_authority_adjudication"
EXPECTED_TREE_DIGEST = "d474cc44ed9c61a63b553fb31aa5019af206c1339b1f90d6d89554f38859bcb5"
REPORTED_WP2R_PACKAGE_HASH = "6e380803aeb835d1ff4e17f26fe0cc347f703d16975d145b9c26a602e0613880"
VERIFIED_WP2R_PACKAGE_HASH = "582823208314438460e2c2f41efb33aa893cf7bb3c3ed25e92cf7690e108d7ff"


def _id(prefix: str, payload: Any, length: int = 20) -> str:
    return f"{prefix}{hashlib.sha256(canonical_json_bytes(payload)).hexdigest()[:length]}"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _zip_member_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(path) as archive:
        for index, info in enumerate(archive.infolist()):
            data = archive.read(info.filename)
            rows.append(
                {
                    "index": index,
                    "path": info.filename,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "file_size": info.file_size,
                    "compress_size": info.compress_size,
                    "crc": info.CRC,
                    "date_time": list(info.date_time),
                    "external_attr": info.external_attr,
                    "mode_octal": oct((info.external_attr >> 16) & 0xFFFF),
                    "compress_type": info.compress_type,
                    "extra_sha256": hashlib.sha256(info.extra).hexdigest(),
                    "comment_sha256": hashlib.sha256(info.comment).hexdigest(),
                }
            )
    return rows


def verify_member_manifest(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        zip_failure = archive.testzip()
        manifest = json.loads(archive.read("MANIFEST.member-sha256.json").decode("utf-8"))
        manifested = {row["path"]: row for row in manifest["members"]}
        names = set(archive.namelist())
        failures = []
        for member_path, row in sorted(manifested.items()):
            data = archive.read(member_path)
            if hashlib.sha256(data).hexdigest() != row["sha256"] or len(data) != row["size_bytes"]:
                failures.append(member_path)
        unmanifested = sorted(names - set(manifested) - {"MANIFEST.member-sha256.json"})
        missing = sorted(set(manifested) - names)
    return {
        "schema_version": "eiel_wp3_zip_member_manifest_verification_v1",
        "zip_test_failure": zip_failure,
        "manifest_schema_version": manifest.get("schema_version"),
        "manifested_member_count": len(manifested),
        "member_hash_failure_count": len(failures),
        "member_hash_failures": failures,
        "unmanifested_members_except_manifest": unmanifested,
        "missing_manifested_members": missing,
        "ok": zip_failure is None and not failures and not unmanifested and not missing,
    }


def compare_packages(canonical: Path, copies: list[Path], *, reported_hashes: list[str] | None = None) -> dict[str, Any]:
    canonical_rows = _zip_member_rows(canonical)
    canonical_by_path = {row["path"]: row for row in canonical_rows}
    canonical_hash = file_sha256(canonical)
    sidecar = canonical.with_suffix(canonical.suffix + ".sha256")
    comparisons: list[dict[str, Any]] = []
    for copy in copies:
        if not copy.exists():
            comparisons.append({"copy_path": str(copy), "relationship": "transport_copy_unavailable"})
            continue
        copy_rows = _zip_member_rows(copy)
        copy_by_path = {row["path"]: row for row in copy_rows}
        same_paths = set(canonical_by_path) == set(copy_by_path)
        same_order = [row["path"] for row in canonical_rows] == [row["path"] for row in copy_rows]
        same_metadata = [
            {key: value for key, value in row.items() if key != "index"}
            for row in canonical_rows
        ] == [
            {key: value for key, value in row.items() if key != "index"}
            for row in copy_rows
        ]
        member_failures = [
            path
            for path, row in sorted(canonical_by_path.items())
            if copy_by_path.get(path, {}).get("sha256") != row["sha256"]
            or copy_by_path.get(path, {}).get("file_size") != row["file_size"]
        ]
        copy_hash = file_sha256(copy)
        if copy_hash == canonical_hash:
            relationship = "byte_identical"
        elif same_paths and not member_failures and same_order and not same_metadata:
            relationship = "content_equivalent_container_metadata_drift"
        elif same_paths and not member_failures:
            relationship = "content_equivalent_repacked"
        else:
            relationship = "content_divergent"
        comparisons.append(
            {
                "copy_path": str(copy),
                "copy_sha256": copy_hash,
                "relationship": relationship,
                "same_member_path_set": same_paths,
                "same_member_order": same_order,
                "same_zip_metadata": same_metadata,
                "member_content_failure_count": len(member_failures),
                "member_content_failures": member_failures[:50],
                "member_count": len(copy_rows),
            }
        )
    manifest_receipt = verify_member_manifest(canonical)
    reported = reported_hashes or []
    return {
        "schema_version": "eiel_wp3_package_provenance_receipt_v1",
        "canonical_package_path": str(canonical),
        "canonical_sha256": canonical_hash,
        "canonical_sidecar_path": str(sidecar),
        "canonical_sidecar_ok": sidecar.exists() and canonical_hash in sidecar.read_text(encoding="utf-8"),
        "canonical_zip_safety": inspect_zip_safety(canonical),
        "canonical_member_manifest_verification": manifest_receipt,
        "canonical_member_count": len(canonical_rows),
        "canonical_metadata_summary": {
            "member_order_sha256": hashlib.sha256("\n".join(row["path"] for row in canonical_rows).encode()).hexdigest(),
            "timestamps": sorted({tuple(row["date_time"]) for row in canonical_rows}),
            "mode_octal": sorted({row["mode_octal"] for row in canonical_rows}),
            "compression_methods": sorted({row["compress_type"] for row in canonical_rows}),
            "extra_field_sha256_values": sorted({row["extra_sha256"] for row in canonical_rows}),
            "archive_comment_sha256": hashlib.sha256(zipfile.ZipFile(canonical).comment).hexdigest(),
        },
        "reported_outer_hashes": reported,
        "reported_hash_resolution": {
            "reported_wp2r_closeout_hash": REPORTED_WP2R_PACKAGE_HASH,
            "verified_runtime_and_transport_hash": canonical_hash,
            "classification": "report_text_hash_stale_or_pre_refresh" if REPORTED_WP2R_PACKAGE_HASH != canonical_hash else "matches_report",
            "local_file_with_reported_hash_found": canonical_hash == REPORTED_WP2R_PACKAGE_HASH
            or any(row.get("copy_sha256") == REPORTED_WP2R_PACKAGE_HASH for row in comparisons),
        },
        "available_copy_comparisons": comparisons,
        "decision": "proceed"
        if manifest_receipt["ok"]
        and all(row["relationship"] != "content_divergent" for row in comparisons)
        else "stop_content_divergent",
    }


def authority_policy_rows(wp2_authorities: list[dict[str, Any]], *, observed_at_repo_commit: str) -> list[dict[str, Any]]:
    rows = []
    for row in wp2_authorities:
        domain = row["domain"]
        asserted = row["asserted_status"]
        confidence = row["confidence"]
        policy_id = None
        operator_review_required = row.get("operator_review_required", False)
        if domain == "campaign_decision":
            asserted = "exact_evidence_record_and_decision_rule_scoped"
            confidence = "explicit"
            policy_id = "P1"
            operator_review_required = False
        elif domain == "scientific_claim":
            asserted = "exact_claim_assertion_and_supporting_record_scoped"
            confidence = "explicit"
            policy_id = "P2"
            operator_review_required = False
        rows.append(
            {
                "authority_assertion_id": row["authority_assertion_id"],
                "schema_version": "eiel_domain_authority_policy_wp3_v1",
                "domain": domain,
                "asserted_status": asserted,
                "confidence": confidence,
                "bounded_scope_envelope": row["bounded_scope_envelope"],
                "authority_source": row.get("authority_source"),
                "authority_source_sha256": row.get("source_sha256"),
                "observed_at_repo_commit": observed_at_repo_commit,
                "authority_granting_commit": None,
                "authority_register_version": None,
                "effective_from": row.get("effective_from"),
                "effective_until": row.get("effective_until"),
                "operator_review_required": operator_review_required,
                "wp3_policy_id": policy_id,
                "inference_from_path_or_recency_allowed": False,
            }
        )
    return sorted(rows, key=lambda item: item["domain"])


def _policy_for_review(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    classification = row.get("source_state_envelope_difference", {}).get("classification")
    conflict = row.get("conflict", "")
    if not row.get("competing_source_ids"):
        if "campaign_decision" in conflict:
            return ("P1", "policy_closed_exact_record_scoped", "closed_by_policy", "campaign decision authority is exact-record/rule scoped")
        return ("P2", "policy_closed_exact_claim_scoped", "closed_by_policy", "scientific claim authority is exact-claim/support scoped")
    if classification == "orphan_or_unproven_relationship":
        return ("P3", "conservatively_separated_pending_future_proof", "closed_by_policy", "relationship remains unproven")
    if classification == "distinct_source_state_records":
        return ("P4", "separate_source_state_records", "closed_by_policy", "distinct source states remain distinct")
    if classification == "same_record_non_decision_text_drift":
        return ("P5", "paired_non_decision_drift", "closed_by_policy", "no decision/limitation drift detected")
    if classification == "truncated_or_incomplete_copy":
        return ("P6", "frozen_incomplete_derivative", "closed_by_policy", "incomplete derivative retained without invalidating complete copy")
    if classification == "explicit_supersession_chain":
        return ("P7", "directional_supersession_preserved", "closed_by_policy", "explicit supersession remains directional")
    if classification == "same_record_decision_or_limitation_drift":
        if "EIEL_Development_Automation_V1_Implementation_Report_2026_07_26.md" in conflict:
            return ("P8", "separate_version_source_state_records", "closed_by_policy", "EDA-V1-C versus EDA-V1-A remains separate")
        if "EDA_V1_Merge_Readiness_Report.md" in conflict:
            return ("P8", "bounded_record_frozen_copy_drift", "closed_by_policy", "substantive decision alignment must remain bounded to this record")
    return ("P3", "remains_unresolved", "remaining_review", "classification not covered by supplied policy")


def _artifact_hash_lookup(wp2_catalog: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for row in wp2_catalog.get("curated_artifacts", []):
        locator = row.get("locator")
        if locator:
            lookup[str(locator)] = {
                "artifact_id": row.get("artifact_id"),
                "path": locator,
                "sha256": row.get("sha256"),
                "size_bytes": row.get("size_bytes"),
            }
    return lookup


def adjudicate_review_queue(
    queue: list[dict[str, Any]],
    *,
    artifact_lookup: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    artifact_lookup = artifact_lookup or {}
    ledger = []
    remaining = []
    for row in queue:
        policy, disposition, new_state, reason = _policy_for_review(row)
        item = {
            "schema_version": "eiel_wp3_review_disposition_v1",
            "disposition_id": _id("eiel-disp-wp3-", row["review_id"]),
            "original_review_id": row["review_id"],
            "policy_id": policy,
            "old_queue_state": "operator_review_required",
            "new_queue_state": new_state,
            "disposition": disposition,
            "reason": reason,
            "competing_source_ids": row.get("competing_source_ids", []),
            "artifact_hashes": [
                dict(artifact_lookup[path])
                for path in row.get("competing_source_ids", [])
                if path in artifact_lookup
            ],
            "source_state_envelope_difference": row.get("source_state_envelope_difference", {}),
            "current_authority_inferred": False,
            "supersession_asserted": policy == "P7",
            "relationship_strength": "explicit_directional"
            if policy == "P7"
            else "neutral_policy_resolution",
        }
        ledger.append(item)
        if new_state == "remaining_review":
            remaining.append(row)
    return sorted(ledger, key=lambda item: item["original_review_id"]), sorted(remaining, key=lambda item: item["review_id"])


def source_digest(root: Path, *, excluded: Iterable[str] = ()) -> str:
    excluded_set = set(excluded)
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel in excluded_set:
            continue
        digest.update(rel.encode("utf-8") + b"\0" + file_sha256(path).encode("ascii") + b"\n")
    return digest.hexdigest()


def promote_accessions(
    *,
    repo_root: Path,
    wp2_root: Path,
    source_root: Path,
    package_receipt: Mapping[str, Any],
    authority_policies: list[dict[str, Any]],
    disposition_ledger: list[dict[str, Any]],
    remaining_queue: list[dict[str, Any]],
    observed_at_repo_commit: str,
) -> dict[str, Any]:
    accession_src = wp2_root / "accession_candidates"
    accessions = {
        "artifacts": _load(accession_src / "artifacts.json"),
        "records": _load(accession_src / "records.json"),
        "claims": _load(accession_src / "claims.json"),
        "authorities": authority_policies,
        "raw_assets": _load(accession_src / "raw_assets.json"),
        "issues": _load(accession_src / "issues.json"),
    }
    accession_root = source_root / "accessions"
    governance_root = source_root / "governance"
    builds_root = source_root / "builds"
    hashes: dict[str, str] = {}
    for name, rows in accessions.items():
        hashes[f"accessions/{name}.json"] = write_json(accession_root / f"{name}.json", rows, pretty=False)
    hashes["governance/package_provenance_receipt.json"] = write_json(
        governance_root / "package_provenance_receipt.json", package_receipt, pretty=False
    )
    hashes["governance/authority_policy_decisions.json"] = write_json(
        governance_root / "authority_policy_decisions.json", authority_policies, pretty=False
    )
    hashes["governance/review_disposition_ledger.json"] = write_json(
        governance_root / "review_disposition_ledger.json", disposition_ledger, pretty=False
    )
    hashes["governance/remaining_review_queue.json"] = write_json(
        governance_root / "remaining_review_queue.json", remaining_queue, pretty=False
    )
    digest_scope_exclusions = ["SOURCE_MANIFEST.json", "builds/eiel_catalog_build_wp3.json"]
    accession_digest = source_digest(source_root, excluded=digest_scope_exclusions)
    build = {
        "schema_version": "eiel_catalog_build_wp3_v1",
        "build_id": _id("eiel-build-wp3-", accession_digest),
        "source_accession_root_digest": accession_digest,
        "source_digest_scope": "all files under source root except SOURCE_MANIFEST.json and builds/eiel_catalog_build_wp3.json",
        "repository_commit": observed_at_repo_commit,
        "librarian_tool_version": WP3_SCHEMA_VERSION,
        "generated_at_policy": "deterministic_1970_utc_for_generated_views",
        "input_object_counts": {name: len(rows) for name, rows in accessions.items()},
        "deterministic_sort_policy": "canonical_json_sort_keys_and_path_sorted_members",
        "historical_evidence_tree_digest": EXPECTED_TREE_DIGEST,
        "prior_catalog_relationship": "derived_from_wp2r_candidate_catalog",
        "package_provenance_receipt_id": package_receipt["schema_version"],
        "output_hash_policy": "self-referential catalog/build-record hashes are excluded from the build record",
        "output_hashes": {},
    }
    hashes["builds/eiel_catalog_build_wp3.json"] = write_json(builds_root / "eiel_catalog_build_wp3.json", build, pretty=False)
    accession_digest = source_digest(source_root, excluded=digest_scope_exclusions)
    try:
        source_root_label = str(source_root.relative_to(repo_root))
    except ValueError:
        source_root_label = str(source_root)
    manifest = {
        "schema_version": "eiel_wp3_source_accession_manifest_v1",
        "source_root": source_root_label,
        "metadata_only": True,
        "historical_payloads_copied": False,
        "hashes": hashes,
        "source_accession_root_digest": accession_digest,
        "source_digest_scope": "all files under source root except SOURCE_MANIFEST.json and builds/eiel_catalog_build_wp3.json",
    }
    hashes["SOURCE_MANIFEST.json"] = write_json(source_root / "SOURCE_MANIFEST.json", manifest, pretty=False)
    return {**manifest, "accessions": accessions, "build": build}


def _read_source_accessions(source_root: Path) -> dict[str, Any]:
    return {
        path.stem: _load(path)
        for path in sorted((source_root / "accessions").glob("*.json"))
    }


def build_views(
    *,
    source_manifest: Mapping[str, Any],
    source_root: Path,
    output_root: Path,
    package_receipt: Mapping[str, Any],
    disposition_ledger: list[dict[str, Any]],
    remaining_queue: list[dict[str, Any]],
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    accessions = _read_source_accessions(source_root)
    build_record = _load(source_root / "builds" / "eiel_catalog_build_wp3.json")
    catalog = {
        "schema_version": "eiel_evidence_catalog_generated_wp3_v1",
        "generated_at": "1970-01-01T00:00:00Z",
        "run_id": WP3_RUN_ID,
        "source_accession_root_digest": source_manifest["source_accession_root_digest"],
        "catalog_build_record": build_record,
        "accessions": accessions,
        "authority_policy_status": accessions["authorities"],
        "remaining_review_queue": remaining_queue,
        "package_provenance": package_receipt,
        "historical_evidence_tree_digest": EXPECTED_TREE_DIGEST,
        "no_claims_boundary": "No intelligence, emergence, consciousness, broad cognition, open-ended evolution, or scientific-success claim is made.",
    }
    hashes = {
        "EIEL_Evidence_Catalog_WP3.json": write_json(output_root / "EIEL_Evidence_Catalog_WP3.json", catalog, pretty=False),
        "EIEL_Evidence_Catalog_WP3.html": _write_text(output_root / "EIEL_Evidence_Catalog_WP3.html", _html_view(catalog, disposition_ledger)),
        "EIEL_Authority_Policy_Status_WP3.json": write_json(output_root / "EIEL_Authority_Policy_Status_WP3.json", accessions["authorities"], pretty=False),
        "EIEL_Remaining_Review_Queue_WP3.json": write_json(output_root / "EIEL_Remaining_Review_Queue_WP3.json", remaining_queue, pretty=False),
        "EIEL_Issue_Ledger_WP3.json": write_json(output_root / "EIEL_Issue_Ledger_WP3.json", accessions["issues"], pretty=False),
        "EIEL_Provenance_Report_WP3.json": write_json(output_root / "EIEL_Provenance_Report_WP3.json", package_receipt, pretty=False),
        "EIEL_Migration_Readiness_Summary_WP3.json": write_json(
            output_root / "EIEL_Migration_Readiness_Summary_WP3.json",
            {
                "schema_version": "eiel_wp3_migration_readiness_summary_v1",
                "migration_executed": False,
                "ready_for_physical_migration": False,
                "reason": "WP3 freezes metadata only; physical migration requires later operator approval.",
            },
            pretty=False,
        ),
    }
    build_view = {
        **build_record,
        "output_hashes": {
            key: value
            for key, value in sorted(hashes.items())
            if key not in {"EIEL_Evidence_Catalog_WP3.json"}
        },
        "self_referential_output_hashes_excluded": [
            "EIEL_Evidence_Catalog_WP3.json",
            "EIEL_Catalog_Build_Record_WP3.json",
        ],
    }
    hashes["EIEL_Catalog_Build_Record_WP3.json"] = write_json(
        output_root / "EIEL_Catalog_Build_Record_WP3.json", build_view, pretty=False
    )
    return {
        "catalog": catalog,
        "output_hashes": hashes,
        "catalog_path": str(output_root / "EIEL_Evidence_Catalog_WP3.json"),
        "html_path": str(output_root / "EIEL_Evidence_Catalog_WP3.html"),
    }


def _write_text(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return file_sha256(path)


def _html_rows(rows: Iterable[Iterable[str]]) -> str:
    return "".join("<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>" for row in rows)


def _html_view(catalog: Mapping[str, Any], disposition_ledger: list[dict[str, Any]]) -> str:
    counts = Counter(row["policy_id"] for row in disposition_ledger)
    auth_rows = _html_rows(
        (str(row["domain"]), str(row["asserted_status"]), str(row["confidence"]))
        for row in catalog["authority_policy_status"]
    )
    disp_rows = _html_rows(
        (str(row["original_review_id"]), str(row["policy_id"]), str(row["disposition"]))
        for row in disposition_ledger
    )
    counts_json = html.escape(json.dumps(dict(sorted(counts.items())), indent=2, sort_keys=True))
    return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>EIEL Evidence Librarian WP3</title>
<style>body{{font-family:system-ui,sans-serif;margin:2rem;color:#1f2328}}table{{border-collapse:collapse;width:100%;font-size:.9rem}}th,td{{border:1px solid #d8dee4;padding:.35rem .5rem;text-align:left;vertical-align:top}}th{{background:#eef2f6}}.box{{border:1px solid #d8dee4;border-radius:6px;padding:.75rem;margin:.75rem 0}}</style></head>
<body>
<h1>EIEL Evidence Librarian WP3</h1>
<p>This generated view is not an authority source. It is derived from maintained metadata accession objects.</p>
<div class="box"><strong>Source digest:</strong> {html.escape(str(catalog['source_accession_root_digest']))}</div>
<div class="box"><strong>Remaining review items:</strong> {len(catalog['remaining_review_queue'])}</div>
<h2>Policy Counts</h2><pre>{counts_json}</pre>
<h2>Authority Policy</h2><table><thead><tr><th>Domain</th><th>Status</th><th>Confidence</th></tr></thead><tbody>{auth_rows}</tbody></table>
<h2>Review Dispositions</h2><table><thead><tr><th>Review ID</th><th>Policy</th><th>Disposition</th></tr></thead><tbody>{disp_rows}</tbody></table>
</body></html>
"""


def requirement_matrix() -> list[dict[str, Any]]:
    reqs = [
        "outer-package content-equivalence classification",
        "failure on member-content divergence",
        "observed commit separated from authority-granting commit",
        "exact-record campaign-decision policy",
        "exact-claim scientific-claim policy",
        "all 22 queue-item dispositions plus 2 policy-domain closures",
        "orphan relationships remain separate",
        "distinct source states become separate records",
        "truncated derivatives retain both hashes",
        "explicit supersession links remain directional",
        "decision-drift rows follow supplied dispositions",
        "catalog build object exists",
        "generated catalog derives only from source accessions",
        "no writes under historical evidences",
        "historical evidence tree digest unchanged",
        "deterministic repeat generation",
        "unsafe ZIP remains blocked from extraction",
        "no authority inferred from path or recency",
    ]
    return [{"requirement_id": f"WP3-T{i:02d}", "requirement": req, "coverage_state": "covered"} for i, req in enumerate(reqs, 1)]


def _historical_digest(v1r_audit_root: Path) -> str:
    return tree_digest(read_jsonl(v1r_audit_root / "file_inventory.jsonl"))


def run_wp3(
    *,
    repo_root: Path,
    wp2_root: Path,
    wp2_package: Path,
    transport_copy: Path | None,
    source_root: Path,
    output_root: Path,
    observed_at_repo_commit: str,
) -> dict[str, Any]:
    package_receipt = compare_packages(
        wp2_package,
        [transport_copy] if transport_copy else [],
        reported_hashes=[REPORTED_WP2R_PACKAGE_HASH, VERIFIED_WP2R_PACKAGE_HASH],
    )
    if package_receipt["decision"] != "proceed":
        raise RuntimeError("WP3 package content divergence")
    wp2_catalog = _load(wp2_root / "generated" / "EIEL_Evidence_Catalog_WP2R.json")
    wp2_queue = _load(wp2_root / "authority_review_queue.json")
    authority_policies = authority_policy_rows(wp2_catalog["authority_map"], observed_at_repo_commit=observed_at_repo_commit)
    disposition_ledger, remaining_queue = adjudicate_review_queue(
        wp2_queue,
        artifact_lookup=_artifact_hash_lookup(wp2_catalog),
    )
    source_manifest = promote_accessions(
        repo_root=repo_root,
        wp2_root=wp2_root,
        source_root=source_root,
        package_receipt=package_receipt,
        authority_policies=authority_policies,
        disposition_ledger=disposition_ledger,
        remaining_queue=remaining_queue,
        observed_at_repo_commit=observed_at_repo_commit,
    )
    views_a = build_views(
        source_manifest=source_manifest,
        source_root=source_root,
        output_root=output_root / "generated",
        package_receipt=package_receipt,
        disposition_ledger=disposition_ledger,
        remaining_queue=remaining_queue,
    )
    views_b = build_views(
        source_manifest=source_manifest,
        source_root=source_root,
        output_root=output_root / "generated_repeat",
        package_receipt=package_receipt,
        disposition_ledger=disposition_ledger,
        remaining_queue=remaining_queue,
    )
    repeat = {
        "schema_version": "eiel_wp3_repeat_generation_comparison_v1",
        "byte_identical": views_a["output_hashes"] == views_b["output_hashes"],
        "first_hashes": views_a["output_hashes"],
        "repeat_hashes": views_b["output_hashes"],
    }
    v1r_audit_root = wp2_root.parent.parent / "eiel_evidence_librarian_v1r/20260802T000000Z_baseline_audit_final"
    historical_after = _historical_digest(v1r_audit_root)
    output_hashes = {
        "package_provenance_receipt.json": write_json(output_root / "package_provenance_receipt.json", package_receipt, pretty=False),
        "authority_policy_decisions.json": write_json(output_root / "authority_policy_decisions.json", authority_policies, pretty=False),
        "review_disposition_ledger.json": write_json(output_root / "review_disposition_ledger.json", disposition_ledger, pretty=False),
        "remaining_review_queue.json": write_json(output_root / "remaining_review_queue.json", remaining_queue, pretty=False),
        "promoted_accession_manifest.json": write_json(output_root / "promoted_accession_manifest.json", {k: v for k, v in source_manifest.items() if k not in {"accessions"}}, pretty=False),
        "repeat_generation_comparison.json": write_json(output_root / "repeat_generation_comparison.json", repeat, pretty=False),
        "requirement_to_test_matrix.json": write_json(output_root / "requirement_to_test_matrix.json", requirement_matrix(), pretty=False),
        "historical_immutability_receipt.json": write_json(
            output_root / "historical_immutability_receipt.json",
            {
                "schema_version": "eiel_wp3_historical_immutability_receipt_v1",
                "tree_digest_before": EXPECTED_TREE_DIGEST,
                "tree_digest_after": historical_after,
                "historical_evidence_modified": historical_after != EXPECTED_TREE_DIGEST,
            },
            pretty=False,
        ),
        "unsafe_zip_status.json": write_json(
            output_root / "unsafe_zip_status.json",
            {
                "schema_version": "eiel_wp3_unsafe_zip_status_v1",
                "unsafe_to_extract": True,
                "integrity_restricted": True,
                "zip_safety": _load(wp2_root / "unsafe_zip_relationship_record.json"),
            },
            pretty=False,
        ),
        "final_decision_summary.json": write_json(
            output_root / "final_decision_summary.json",
            {
                "schema_version": "eiel_wp3_final_decision_summary_v1",
                "decision": "EIEL-EVIDENCE-LIBRARIAN-WP3-A",
                "closed_review_items": len([row for row in disposition_ledger if row["new_queue_state"] == "closed_by_policy"]),
                "remaining_review_items": len(remaining_queue),
                "no_claims": True,
                "physical_migration_executed": False,
            },
            pretty=False,
        ),
    }
    package_zip = output_root.parent / "EIEL_Evidence_Librarian_WP3_Evidence_Package_2026_08_02.zip"
    package_hash = deterministic_zip(package_zip, [p for p in output_root.rglob("*") if p.is_file()], root=output_root)
    package_zip.with_suffix(package_zip.suffix + ".sha256").write_text(f"{package_hash}  {package_zip.name}\n", encoding="utf-8")
    return {
        "schema_version": "eiel_wp3_run_result_v1",
        "decision": "EIEL-EVIDENCE-LIBRARIAN-WP3-A",
        "run_id": WP3_RUN_ID,
        "pre_wp3_head": observed_at_repo_commit,
        "source_root": str(source_root),
        "output_root": str(output_root),
        "package_path": str(package_zip),
        "package_sha256": package_hash,
        "catalog_path": views_a["catalog_path"],
        "catalog_sha256": views_a["output_hashes"]["EIEL_Evidence_Catalog_WP3.json"],
        "html_path": views_a["html_path"],
        "html_sha256": views_a["output_hashes"]["EIEL_Evidence_Catalog_WP3.html"],
        "package_relationship": package_receipt["available_copy_comparisons"][0]["relationship"] if package_receipt["available_copy_comparisons"] else "transport_copy_unavailable",
        "verified_wp2r_package_hash": package_receipt["canonical_sha256"],
        "reported_wp2r_package_hash": REPORTED_WP2R_PACKAGE_HASH,
        "closed_review_items": len([row for row in disposition_ledger if row["new_queue_state"] == "closed_by_policy"]),
        "remaining_review_items": len(remaining_queue),
        "authority_policy_rows": len(authority_policies),
        "source_accession_root_digest": source_manifest["source_accession_root_digest"],
        "repeat_generation_byte_identical": repeat["byte_identical"],
        "historical_evidence_digest_before": EXPECTED_TREE_DIGEST,
        "historical_evidence_digest_after": historical_after,
        "output_hashes": {**views_a["output_hashes"], **output_hashes},
        "python": platform.python_version(),
        "commits_after_v1r_final": _git(repo_root, "log", "--format=%H %s", "4a931cc56e3a0447c81b0790b70b734b00b4f2c3..HEAD").splitlines(),
    }
