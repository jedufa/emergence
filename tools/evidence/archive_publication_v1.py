"""EIEL Evidence Archive Publication 1.0 planner, verifier, and publisher.

This module treats archive cleanup as publication of a new canonical evidence
layout, not as ad-hoc folder tidying.  It never merges evidence records and it
does not interpret scientific content.  Every input file receives a permanent
file ID, and every move is represented in a manifest that can be verified by
path, size, SHA-256, destination, and semantic catalog identity.
"""

from __future__ import annotations

from engine.runtime_roots import public_runtime_path

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from tools.evidence.librarian import wp4


SCHEMA_VERSION = "eiel_evidence_archive_publication_v1"
DEFAULT_SOURCE_ROOT = Path("tools/evidence/librarian/catalog_source/00-catalog")
DEFAULT_RUNTIME_ROOT = Path(
    public_runtime_path('evidence_raw/eiel_evidence_archive_publication_v1')
)

KEEP_ARCHIVE_FOLDERS = {
    "000_programme-methods",
    "010_archive-indexes",
    "100_engine-route-qualification",
    "200_genome-inheritance",
    "300_lineage-reproduction",
    "400_worlds-ecology",
    "500_cognition-memory",
    "600_representation-policy",
    "700_open-ended-search",
    "800_diagnostics-closures",
    "850_handoffs",
    "900_incoming",
    "990_legacy-redirects",
}
POST_PUBLICATION_ACCESSION_SHELVES = KEEP_ARCHIVE_FOLDERS - {
    "900_incoming",
    "990_legacy-redirects",
}
LEGACY_REDIRECT_MAP_REL = Path("evidences/990_legacy-redirects/archive-publication-v1-map.json")
ARCHIVE_SUMMARY_REL = Path("evidences/990_legacy-redirects/archive-publication-v1-summary.json")

RAW_EXTENSIONS = {".csv", ".tsv", ".json", ".yaml", ".yml", ".html", ".txt", ""}
ALWAYS_KEEP_FILENAMES = {"README.md", "INDEX.md"}


class PublicationError(RuntimeError):
    """Fail-closed publication error."""


@dataclass(frozen=True)
class PublicationPaths:
    repo_root: Path
    evidence_root: Path
    source_root: Path
    runtime_root: Path
    freeze_root: Path
    candidate_root: Path
    raw_root: Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return file_sha256(path)


def write_text(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return file_sha256(path)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PublicationError(f"missing required JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def git(repo_root: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and result.returncode != 0:
        raise PublicationError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def git_tracked(repo_root: Path, rel_path: str) -> bool:
    try:
        git(repo_root, "ls-files", "--error-unmatch", rel_path)
    except PublicationError:
        return False
    return True


def rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def slugify(value: str, *, limit: int = 72) -> str:
    text = value.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    text = re.sub(r"-+", "-", text)
    return (text or "evidence")[:limit].strip("-") or "evidence"


def publication_suffix(path: Path, *, raw: bool) -> str:
    name = path.name
    if raw:
        if name.endswith(".evidence.zip.sha256"):
            return ".evidence.zip.sha256"
        if name.endswith(".zip.sha256"):
            return ".zip.sha256"
        if name.endswith(".evidence.zip"):
            return ".evidence.zip"
        return "".join(path.suffixes) or ".bin"
    if name.endswith(".evidence.zip.sha256"):
        return ".evidence.zip.sha256"
    if name.endswith(".zip.sha256"):
        return ".evidence.zip.sha256"
    if name.endswith(".evidence.zip") or name.endswith(".zip"):
        return ".evidence.zip"
    if path.suffix.lower() == ".md":
        return ".md"
    return path.suffix.lower() or ".bin"


def stem_without_suffix(path: Path) -> str:
    name = path.name
    for suffix in (".evidence.zip.sha256", ".zip.sha256", ".evidence.zip"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    if path.suffix:
        return name[: -len(path.suffix)]
    return name


def filesystem_evidence_id_paths(evidence_root: Path) -> dict[str, str]:
    filesystem_ids: dict[str, Path] = {}
    for path in evidence_root.rglob("*"):
        if not path.is_file():
            continue
        match = re.match(r"^(EIEL-EV-(\d{6}))--", path.name)
        if not match:
            continue
        permanent_id = match.group(1)
        if permanent_id in filesystem_ids:
            raise PublicationError(
                "duplicate permanent evidence ID already exists in the archive: "
                f"{permanent_id}"
            )
        filesystem_ids[permanent_id] = path
    return {
        permanent_id: path.relative_to(evidence_root.parent).as_posix()
        for permanent_id, path in filesystem_ids.items()
    }


def next_permanent_evidence_id(
    redirect: Mapping[str, Any],
    evidence_root: Path | None = None,
) -> str:
    sequences = [
        int(match.group(1))
        for row in redirect.get("files", [])
        if (match := re.fullmatch(r"EIEL-EV-(\d{6})", str(row.get("permanent_file_id", ""))))
    ]
    if evidence_root is not None:
        sequences.extend(
            int(permanent_id.rsplit("-", 1)[1])
            for permanent_id in filesystem_evidence_id_paths(evidence_root)
        )
    max_seen = max(sequences, default=0)
    return f"EIEL-EV-{max_seen + 1:06d}"


def archive_1_0_folder_name(shelf: str) -> str:
    return shelf.split("_", 1)[1] if "_" in shelf else shelf


def validate_post_publication_shelf(shelf: str) -> None:
    if shelf not in POST_PUBLICATION_ACCESSION_SHELVES:
        raise PublicationError(f"bad or unapproved evidence destination shelf: {shelf}")
    if not re.fullmatch(r"\d{3}_[a-z0-9-]+", shelf):
        raise PublicationError(f"destination shelf is not numbered: {shelf}")


def post_publication_filename(permanent_id: str, source: Path) -> str:
    return f"{permanent_id}--{slugify(stem_without_suffix(source))}{publication_suffix(source, raw=False)}"


def accession_post_publication_file(
    *,
    repo_root: Path,
    source: Path,
    shelf: str,
    expected_head: str,
    redirect_map: Path | None = None,
    summary_path: Path | None = None,
) -> dict[str, Any]:
    """Accession one already tracked report-facing evidence file after publication.

    This is intentionally narrower than Archive Publication 1.0. It does not
    rebuild the archive, alter authority/claims, relocate raw payloads, or touch
    more than the source file, redirect map, and summary counters.
    """

    repo_root = repo_root.resolve()
    evidence_root = repo_root / "evidences"
    source_path = (repo_root / source) if not source.is_absolute() else source
    source_path = source_path.absolute()
    redirect_path = (repo_root / (redirect_map or LEGACY_REDIRECT_MAP_REL)).resolve()
    summary_file = (repo_root / (summary_path or ARCHIVE_SUMMARY_REL)).resolve()
    validate_post_publication_shelf(shelf)
    if not expected_head or git(repo_root, "rev-parse", "HEAD") != expected_head:
        raise PublicationError("observed HEAD does not match required expected head")
    if git(repo_root, "status", "--short"):
        raise PublicationError("worktree must be clean before post-publication accession")
    try:
        source_rel = source_path.relative_to(repo_root).as_posix()
        source_evidence_rel = source_path.relative_to(evidence_root).as_posix()
    except ValueError as exc:
        raise PublicationError("source must be under evidences") from exc
    if not source_rel.startswith("evidences/"):
        raise PublicationError("source must be under evidences")
    if source_path.is_symlink():
        raise PublicationError(f"source symlinks are forbidden: {source_rel}")
    if not source_path.exists():
        raise PublicationError(f"missing source evidence file: {source_rel}")
    if not source_path.is_file():
        raise PublicationError(f"source is not a regular file: {source_rel}")
    if re.search(r"EIEL-EV-\d{6}--", source_path.name):
        raise PublicationError(f"source already appears to be a current EIEL-EV path: {source_rel}")
    if not git_tracked(repo_root, source_rel):
        raise PublicationError(f"source evidence file is not tracked: {source_rel}")
    size = source_path.stat().st_size
    keep, reason = should_keep_report_facing(source_path, size)
    if not keep:
        raise PublicationError(f"source is not report-facing evidence: {source_rel}")
    redirect = load_json(redirect_path)
    summary = load_json(summary_file)
    files = list(redirect.get("files", []))
    filesystem_ids = filesystem_evidence_id_paths(evidence_root)
    mapped_filesystem_pairs = {
        (str(row.get("permanent_file_id")), str(row.get("new_path")))
        for row in files
        if row.get("new_path")
    }
    unmapped_filesystem_ids = sorted(
        (permanent_id, path)
        for permanent_id, path in filesystem_ids.items()
        if (permanent_id, path) not in mapped_filesystem_pairs
    )
    if unmapped_filesystem_ids:
        raise PublicationError(
            "UID-prefixed evidence files are missing from the redirect map: "
            + ", ".join(
                f"{permanent_id}={path}"
                for permanent_id, path in unmapped_filesystem_ids[:5]
            )
        )
    known_paths = {
        str(row.get(key))
        for row in files
        for key in ("old_path", "new_path", "runtime_path")
        if row.get(key)
    }
    if source_rel in known_paths:
        raise PublicationError(f"source path already exists in archive redirect map: {source_rel}")
    permanent_id = next_permanent_evidence_id(redirect, evidence_root)
    if permanent_id in {str(row.get("permanent_file_id")) for row in files}:
        raise PublicationError(f"permanent evidence ID collision: {permanent_id}")
    destination_rel = f"evidences/{shelf}/{post_publication_filename(permanent_id, source_path)}"
    if destination_rel in known_paths:
        raise PublicationError(f"destination already exists in archive redirect map: {destination_rel}")
    destination_path = repo_root / destination_rel
    if destination_path.exists():
        raise PublicationError(f"destination already exists: {destination_rel}")
    source_sha = file_sha256(source_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.rename(destination_path)
    destination_sha = file_sha256(destination_path)
    if destination_sha != source_sha or destination_path.stat().st_size != size:
        raise PublicationError(f"post-publication move identity mismatch for {source_rel}")
    entry = {
        "archive_1_0_publication_folder": archive_1_0_folder_name(shelf),
        "archive_disposition": "report_facing",
        "authority_effect": "none",
        "claims_altered": False,
        "new_path": destination_rel,
        "old_evidence_relative_path": source_evidence_rel,
        "old_path": source_rel,
        "permanent_file_id": permanent_id,
        "post_publication_accession": True,
        "publication_folder": shelf,
        "publication_reason": f"post-publication accession: {reason}",
        "runtime_path": None,
        "sha256": source_sha,
        "size_bytes": size,
        "tracked": True,
        "verified_destination_sha256": destination_sha,
    }
    redirect["files"] = [*files, entry]
    if "file_count" in summary:
        summary["file_count"] = int(summary.get("file_count") or 0) + 1
    summary.setdefault("disposition_counts", {})
    summary["disposition_counts"]["report_facing"] = int(summary["disposition_counts"].get("report_facing") or 0) + 1
    summary.setdefault("folder_counts", {})
    summary["folder_counts"][shelf] = int(summary["folder_counts"].get(shelf) or 0) + 1
    if "archive_1_0_folder_counts" in summary:
        folder_name = archive_1_0_folder_name(shelf)
        summary["archive_1_0_folder_counts"][folder_name] = int(summary["archive_1_0_folder_counts"].get(folder_name) or 0) + 1
    write_json(redirect_path, redirect)
    write_json(summary_file, summary)
    return {
        "schema_version": f"{SCHEMA_VERSION}_post_publication_accession_receipt_v1",
        "ok": True,
        "expected_head": expected_head,
        "source_path": source_rel,
        "destination_path": destination_rel,
        "permanent_file_id": permanent_id,
        "sha256": source_sha,
        "size_bytes": size,
        "redirect_map": rel(redirect_path, repo_root),
        "summary_path": rel(summary_file, repo_root),
        "authority_effect": "none",
        "claims_altered": False,
    }


def top_component(old_rel: str) -> str:
    return old_rel.split("/", 1)[0]


def classify_folder(old_rel: str) -> str:
    top = top_component(old_rel)
    text = old_rel.lower()
    if top == "00-handoffs" or "handoff" in text:
        return "850_handoffs"
    if top in {"00-campaign-index.md", "00-runtime-relocation-index.md", "README.md"}:
        return "000_programme-methods"
    if top.startswith("90-") or "diagnostic" in text or "closure" in text:
        return "800_diagnostics-closures"
    if any(term in text for term in ("route", "qualification", "provider", "runtime", "automation", "eda_v1", "authoritative-state", "unified-production-engine")):
        return "100_engine-route-qualification"
    if any(term in text for term in ("genome", "gv2", "p12", "recomb", "diploid", "inheritance")):
        return "200_genome-inheritance"
    if any(term in text for term in ("lineage", "reproduction", "birth", "parentage")):
        return "300_lineage-reproduction"
    if any(term in text for term in ("cognition", "memory", "prediction", "controller", "perception", "crb")):
        return "500_cognition-memory"
    if any(term in text for term in ("open-ended", "evolvability", "lro", "branch", "eed", "search")):
        return "700_open-ended-search"
    if any(term in text for term in ("representation", "polarch", "policy", "mutrad", "structural", "rep0", "rep1", "rep2")):
        return "600_representation-policy"
    if any(term in text for term in ("world", "ecology", "jungle", "resource", "pressure", "path1", "path2", "respec", "mosaic", "predrisk", "habacclim", "corridor")):
        return "400_worlds-ecology"
    return "000_programme-methods"


def should_keep_report_facing(path: Path, size: int) -> tuple[bool, str]:
    name = path.name
    suffix = path.suffix.lower()
    if name in ALWAYS_KEEP_FILENAMES:
        return True, "archive index or README"
    if suffix == ".md":
        return True, "Markdown report-facing evidence"
    if name.endswith(".evidence.zip") or suffix == ".zip":
        if size <= 1_000_000:
            return True, "small complementary evidence ZIP"
        return False, "bulky evidence ZIP relocated to runtime"
    if name.endswith(".zip.sha256") or name.endswith(".evidence.zip.sha256"):
        return True, "ZIP checksum sidecar"
    if suffix in RAW_EXTENSIONS or suffix == "":
        return False, "raw or machine-oriented payload relocated to runtime"
    return False, "non-report-facing payload relocated to runtime"


def iter_evidence_files(evidence_root: Path) -> list[Path]:
    return sorted(path for path in evidence_root.rglob("*") if path.is_file())


def semantic_snapshot(repo_root: Path, source_root: Path) -> dict[str, Any]:
    catalog = wp4.build_catalog(source_root, repo_root)
    dossiers = wp4.build_dossier_manifests(catalog)
    accessions = catalog["accessions"]
    def stable_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            (dict(row) for row in rows),
            key=lambda row: json.dumps(row, sort_keys=True, separators=(",", ":")),
        )

    return {
        "schema_version": "eiel_archive_publication_semantic_snapshot_v1",
        "source_digest": catalog["maintained_source_digest"],
        "build_id": catalog["build_id"],
        "object_counts": catalog["object_counts"],
        "authorities": stable_rows(
            {
                "id": row.get("authority_assertion_id"),
                "domain": row.get("domain"),
                "asserted_status": row.get("asserted_status"),
                "confidence": row.get("confidence"),
                "scope": row.get("bounded_scope_envelope"),
            }
            for row in accessions["authorities"]
        ),
        "claims": stable_rows(
            {
                "id": row.get("claim_id"),
                "status": row.get("support_state"),
                "record_ids": row.get("supporting_record_ids") or row.get("evidence_record_ids") or [],
                "authority_ids": row.get("authority_assertion_ids", []),
            }
            for row in accessions["claims"]
        ),
        "records": stable_rows(
            {
                "id": row.get("evidence_id"),
                "claim_ids": row.get("claim_ids", []),
                "authority_ids": row.get("authority_assertion_ids", []),
                "artifact_ids": row.get("artifact_ids", []),
                "raw_asset_ids": row.get("raw_asset_ids", []),
                "supersedes": row.get("supersedes_record_ids", []),
                "superseded_by": row.get("superseded_by_record_ids", []),
            }
            for row in accessions["records"]
        ),
        "raw_assets": stable_rows(
            {
                "id": row.get("raw_asset_id"),
                "locator": row.get("locator"),
                "sha256": row.get("sha256"),
                "storage_class": row.get("storage_class"),
                "retention_state": row.get("retention_state"),
            }
            for row in accessions["raw_assets"]
        ),
        "dossier_membership": stable_rows(
            {
                "dossier_id": row["dossier_id"],
                "records": row["covered_record_ids"],
                "claims": row["covered_claim_ids"],
                "issues": row["covered_issue_ids"],
            }
            for row in dossiers
        ),
    }


def freeze_archive(paths: PublicationPaths) -> dict[str, Any]:
    files = iter_evidence_files(paths.evidence_root)
    tracked = set(git(paths.repo_root, "ls-files", "evidences").splitlines())
    rows = []
    for index, path in enumerate(files, start=1):
        old_rel = rel(path, paths.repo_root)
        rows.append(
            {
                "permanent_file_id": f"EIEL-EV-{index:06d}",
                "old_path": old_rel,
                "old_evidence_relative_path": rel(path, paths.evidence_root),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
                "tracked": old_rel in tracked,
            }
        )
    payload = {
        "schema_version": f"{SCHEMA_VERSION}_freeze_manifest",
        "repo_head": git(paths.repo_root, "rev-parse", "HEAD"),
        "repo_branch": git(paths.repo_root, "branch", "--show-current"),
        "evidence_root": str(paths.evidence_root),
        "file_count": len(rows),
        "tracked_file_count": sum(1 for row in rows if row["tracked"]),
        "directory_count": sum(1 for path in paths.evidence_root.rglob("*") if path.is_dir()),
        "total_bytes": sum(row["size_bytes"] for row in rows),
        "semantic_snapshot": semantic_snapshot(paths.repo_root, paths.source_root),
        "files": rows,
    }
    write_json(paths.freeze_root / "freeze_manifest.json", payload)
    return payload


def destination_for(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    old_path = Path(str(row["old_evidence_relative_path"]))
    size = int(row["size_bytes"])
    keep, reason = should_keep_report_facing(old_path, size)
    folder = classify_folder(str(row["old_evidence_relative_path"]))
    slug = slugify(stem_without_suffix(old_path))
    permanent_id = str(row["permanent_file_id"])
    suffix = publication_suffix(old_path, raw=not keep)
    filename = f"{permanent_id}--{slug}{suffix}"
    if keep:
        return "report_facing", reason, folder, f"{folder}/{filename}"
    return "runtime_relocated", reason, "legacy-redirects", f"relocated_raw/{filename}"


def build_plan(paths: PublicationPaths, freeze: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    destinations: set[str] = set()
    for row in freeze["files"]:
        disposition, reason, folder, destination = destination_for(row)
        if disposition == "report_facing":
            candidate_path = f"evidences/{destination}"
            runtime_path = None
        else:
            candidate_path = None
            runtime_path = str(paths.raw_root / destination.removeprefix("relocated_raw/"))
        if candidate_path in destinations:
            raise PublicationError(f"duplicate candidate destination: {candidate_path}")
        if candidate_path:
            destinations.add(candidate_path)
        rows.append(
            {
                **row,
                "archive_disposition": disposition,
                "publication_reason": reason,
                "publication_folder": folder,
                "new_path": candidate_path,
                "runtime_path": runtime_path,
            }
        )
    counts = Counter(row["archive_disposition"] for row in rows)
    folder_counts = Counter(row["publication_folder"] for row in rows if row["archive_disposition"] == "report_facing")
    payload = {
        "schema_version": f"{SCHEMA_VERSION}_publication_plan",
        "repo_head": freeze["repo_head"],
        "file_count": len(rows),
        "disposition_counts": dict(sorted(counts.items())),
        "folder_counts": dict(sorted(folder_counts.items())),
        "maintenance_rule": "Every move must reduce maintenance cost by standardizing folder, filename, report/raw boundary, or permanent ID traceability.",
        "no_merge_policy": True,
        "files": rows,
    }
    write_json(paths.freeze_root / "publication_plan.json", payload)
    return payload


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def write_candidate_indexes(paths: PublicationPaths, plan: Mapping[str, Any]) -> None:
    lines = [
        "# EIEL Evidence Archive Publication 1.0",
        "",
        "This is the canonical report-facing evidence archive after physical",
        "publication.  Files are addressed by permanent `EIEL-EV-......` file IDs.",
        "",
        "Raw, bulky, or machine-oriented payloads are not stored here; see",
        "`990_legacy-redirects/archive-publication-v1-map.json` for their runtime",
        "locations and hashes.",
        "",
        "New evidence enters through `900_incoming/` and is promoted only after",
        "Evidence Librarian validation.",
        "",
    ]
    write_text(paths.candidate_root / "README.md", "\n".join(lines))
    index_lines = [
        "# EIEL Evidence Archive Publication 1.0 Index",
        "",
        f"- Total source files: {plan['file_count']}",
        f"- Report-facing files retained in archive: {plan['disposition_counts'].get('report_facing', 0)}",
        f"- Runtime-relocated files: {plan['disposition_counts'].get('runtime_relocated', 0)}",
        "",
        "## Folders",
        "",
    ]
    for folder in sorted(KEEP_ARCHIVE_FOLDERS):
        index_lines.append(f"- `{folder}/`")
    write_text(paths.candidate_root / "INDEX.md", "\n".join(index_lines) + "\n")
    for folder in KEEP_ARCHIVE_FOLDERS:
        (paths.candidate_root / folder).mkdir(parents=True, exist_ok=True)
    write_text(paths.candidate_root / "900_incoming" / "README.md", "# Incoming Evidence\n\nNew evidence lands here before librarian validation and promotion.\n")


def build_candidate(paths: PublicationPaths, plan: Mapping[str, Any]) -> dict[str, Any]:
    if paths.candidate_root.exists():
        raise PublicationError(f"candidate root already exists: {paths.candidate_root}")
    paths.candidate_root.mkdir(parents=True)
    paths.raw_root.mkdir(parents=True, exist_ok=True)
    write_candidate_indexes(paths, plan)
    manifest_rows = []
    for row in plan["files"]:
        source = paths.repo_root / str(row["old_path"])
        if row["archive_disposition"] == "report_facing":
            target = paths.candidate_root.parent / str(row["new_path"])
        else:
            target = Path(str(row["runtime_path"]))
        copy_file(source, target)
        copied_hash = file_sha256(target)
        if copied_hash != row["sha256"]:
            raise PublicationError(f"copy hash mismatch for {row['old_path']}")
        manifest_rows.append({**row, "verified_destination_sha256": copied_hash})
    redirect = {
        "schema_version": f"{SCHEMA_VERSION}_legacy_redirect_map",
        "policy": "Old evidence paths are preserved as manifest redirects. Files were not merged.",
        "files": manifest_rows,
    }
    write_json(paths.candidate_root / "990_legacy-redirects" / "archive-publication-v1-map.json", redirect)
    summary = {
        "schema_version": f"{SCHEMA_VERSION}_candidate_summary",
        "candidate_root": str(paths.candidate_root),
        "runtime_raw_root": str(paths.raw_root),
        "disposition_counts": plan["disposition_counts"],
        "folder_counts": plan["folder_counts"],
        "file_count": plan["file_count"],
    }
    write_json(paths.candidate_root / "990_legacy-redirects" / "archive-publication-v1-summary.json", summary)
    return summary


def verify_candidate(paths: PublicationPaths, freeze: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    redirect_path = paths.candidate_root / "990_legacy-redirects" / "archive-publication-v1-map.json"
    if not redirect_path.exists():
        failures.append({"kind": "missing_redirect_map", "path": str(redirect_path)})
        redirect_rows = []
    else:
        redirect_rows = json.loads(redirect_path.read_text(encoding="utf-8"))["files"]
    by_old = {row["old_path"]: row for row in redirect_rows}
    for row in freeze["files"]:
        old_path = str(row["old_path"])
        mapped = by_old.get(old_path)
        if mapped is None:
            failures.append({"kind": "missing_mapping", "old_path": old_path})
            continue
        if mapped["sha256"] != row["sha256"] or mapped["size_bytes"] != row["size_bytes"]:
            failures.append({"kind": "mapping_identity_mismatch", "old_path": old_path})
        if mapped["archive_disposition"] == "report_facing":
            target = paths.candidate_root.parent / mapped["new_path"]
        else:
            target = Path(mapped["runtime_path"])
        if not target.exists():
            failures.append({"kind": "missing_destination", "old_path": old_path, "target": str(target)})
            continue
        if file_sha256(target) != row["sha256"]:
            failures.append({"kind": "destination_hash_mismatch", "old_path": old_path, "target": str(target)})
    after_semantic = semantic_snapshot(paths.repo_root, paths.source_root)
    semantic_equal = freeze["semantic_snapshot"] == after_semantic
    if not semantic_equal:
        failures.append({"kind": "semantic_snapshot_changed"})
    candidate_files = [path for path in paths.candidate_root.rglob("*") if path.is_file()]
    raw_like_in_candidate = [
        rel(path, paths.candidate_root)
        for path in candidate_files
        if path.suffix.lower() in {".csv", ".tsv", ".html", ".txt", ".yaml", ".yml"}
    ]
    if raw_like_in_candidate:
        failures.append({"kind": "raw_like_files_remain_in_candidate", "paths": raw_like_in_candidate[:50]})
    result = {
        "schema_version": f"{SCHEMA_VERSION}_verification_receipt",
        "ok": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "source_file_count": freeze["file_count"],
        "mapped_file_count": len(by_old),
        "candidate_file_count": len(candidate_files),
        "semantic_equivalence": {
            "ok": semantic_equal,
            "authority_unchanged": semantic_equal,
            "claim_unchanged": semantic_equal,
            "dossier_membership_unchanged": semantic_equal,
            "accession_ids_unchanged": semantic_equal,
            "supersession_unchanged": semantic_equal,
            "raw_reference_unchanged": semantic_equal,
        },
        "candidate_root": str(paths.candidate_root),
        "runtime_raw_root": str(paths.raw_root),
        "publication_plan_disposition_counts": plan["disposition_counts"],
    }
    write_json(paths.freeze_root / "verification_receipt.json", result)
    return result


def publish_candidate(paths: PublicationPaths, verification: Mapping[str, Any]) -> dict[str, Any]:
    if not verification.get("ok"):
        raise PublicationError("candidate verification is not ok; refusing to publish")
    if git(paths.repo_root, "status", "--short"):
        raise PublicationError("worktree must be clean before publishing archive")
    backup_root = paths.runtime_root / "pre_publication_evidences_git_backup"
    if backup_root.exists():
        raise PublicationError(f"backup root already exists: {backup_root}")
    shutil.copytree(paths.evidence_root, backup_root)
    git(paths.repo_root, "rm", "-r", "--quiet", "evidences")
    shutil.copytree(paths.candidate_root, paths.evidence_root)
    git(paths.repo_root, "add", "evidences")
    result = {
        "schema_version": f"{SCHEMA_VERSION}_publish_receipt",
        "published": True,
        "published_evidence_root": str(paths.evidence_root),
        "pre_publication_backup_root": str(backup_root),
        "candidate_root": str(paths.candidate_root),
        "runtime_raw_root": str(paths.raw_root),
        "git_status_after_add": git(paths.repo_root, "status", "--short"),
        "historical_files_merged": False,
        "historical_payloads_rewritten": False,
    }
    write_json(paths.freeze_root / "publish_receipt.json", result)
    return result


def build_paths(args: argparse.Namespace) -> PublicationPaths:
    repo_root = args.repo_root.resolve()
    evidence_root = (repo_root / "evidences").resolve()
    runtime_root = args.runtime_root.resolve()
    return PublicationPaths(
        repo_root=repo_root,
        evidence_root=evidence_root,
        source_root=(repo_root / args.source_root).resolve()
        if not args.source_root.is_absolute()
        else args.source_root.resolve(),
        runtime_root=runtime_root,
        freeze_root=runtime_root / "freeze",
        candidate_root=runtime_root / "published_archive_candidate" / "evidences",
        raw_root=runtime_root / "relocated_raw",
    )


def load_required(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise PublicationError(f"missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    paths = build_paths(args)
    freeze = freeze_archive(paths)
    plan = build_plan(paths, freeze)
    candidate = build_candidate(paths, plan)
    verification = verify_candidate(paths, freeze, plan)
    result: dict[str, Any] = {
        "schema_version": f"{SCHEMA_VERSION}_run_all_result",
        "freeze_manifest": str(paths.freeze_root / "freeze_manifest.json"),
        "publication_plan": str(paths.freeze_root / "publication_plan.json"),
        "candidate_summary": candidate,
        "verification": verification,
    }
    if args.publish:
        result["publish"] = publish_candidate(paths, verification)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eiel-archive-publication-v1")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze")
    sub.add_parser("plan")
    sub.add_parser("build-candidate")
    sub.add_parser("verify-candidate")
    run = sub.add_parser("run-all")
    run.add_argument("--publish", action="store_true")
    pub = sub.add_parser("publish")
    pub.add_argument("--verification", type=Path)
    accession = sub.add_parser("accession-file")
    accession.add_argument("--source", type=Path, required=True)
    accession.add_argument("--shelf", required=True)
    accession.add_argument("--expected-head", required=True)
    accession.add_argument("--redirect-map", type=Path)
    accession.add_argument("--summary", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        paths = build_paths(args)
        if args.command == "freeze":
            result = freeze_archive(paths)
        elif args.command == "plan":
            result = build_plan(paths, load_required(paths.freeze_root / "freeze_manifest.json"))
        elif args.command == "build-candidate":
            result = build_candidate(paths, load_required(paths.freeze_root / "publication_plan.json"))
        elif args.command == "verify-candidate":
            result = verify_candidate(
                paths,
                load_required(paths.freeze_root / "freeze_manifest.json"),
                load_required(paths.freeze_root / "publication_plan.json"),
            )
        elif args.command == "publish":
            verification = load_required(args.verification) if args.verification else load_required(paths.freeze_root / "verification_receipt.json")
            result = publish_candidate(paths, verification)
        elif args.command == "accession-file":
            result = accession_post_publication_file(
                repo_root=paths.repo_root,
                source=args.source,
                shelf=args.shelf,
                expected_head=args.expected_head,
                redirect_map=args.redirect_map,
                summary_path=args.summary,
            )
        elif args.command == "run-all":
            result = run_all(args)
        else:  # pragma: no cover
            parser.error(f"unsupported command {args.command}")
            return 2
    except PublicationError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ok", True) is not False else 1


if __name__ == "__main__":
    raise SystemExit(main())
