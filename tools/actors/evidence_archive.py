"""Integrity-only inspection for published EIEL evidence archives.

Published ``EIEL-EV-xxxxxx--*.evidence.zip`` packages are Librarian-managed
archive objects.  They are not raw campaign trees and must never be silently
fed back into the Evidence Decision Layer as though they were fresh outputs.
This module verifies packaging identity and ZIP integrity only.  Scientific and
authority interpretation remains with the Knowledge Librarian and the relevant
decision/qualification actors.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from tools.actors.common import ActorError, file_sha256
from tools.evidence.librarian.core import inspect_zip_safety

ARCHIVE_SCHEMA_VERSION = "eiel_published_evidence_archive_inspection_v1"
EVIDENCE_ARCHIVE_RE = re.compile(
    r"^(?P<evidence_id>EIEL-EV-\d{6})--(?P<slug>.+)\.evidence\.zip$",
    re.IGNORECASE,
)
EVIDENCE_SIDECAR_RE = re.compile(
    r"^(?P<evidence_id>EIEL-EV-\d{6})--(?P<slug>.+)\.evidence\.zip\.sha256$",
    re.IGNORECASE,
)
SHA256_RE = re.compile(r"\b([0-9a-fA-F]{64})\b")


def is_evidence_archive(path: str | Path) -> bool:
    source = Path(path)
    return source.is_file() and source.name.lower().endswith(".evidence.zip")


def _sidecar_candidates(path: Path, *, slug: str | None) -> list[Path]:
    """Return sidecar candidates under both legacy and Archive 1.0 naming.

    Archive Publication 1.0 assigns a distinct permanent ``EIEL-EV`` ID to
    every file, including a checksum sidecar.  The sidecar therefore normally
    shares the semantic slug, not the ZIP's permanent ID.
    """

    candidates = [Path(str(path) + ".sha256"), path.with_suffix(".sha256")]
    if slug:
        candidates.extend(sorted(path.parent.glob(f"EIEL-EV-*--{slug}.evidence.zip.sha256")))
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def _read_sidecar(path: Path) -> tuple[str | None, list[str]]:
    warnings: list[str] = []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return None, [f"cannot read SHA-256 sidecar {path}: {exc}"]
    match = SHA256_RE.search(text)
    if not match:
        return None, [f"SHA-256 sidecar contains no 64-hex digest: {path}"]
    return match.group(1).lower(), warnings


def inspect_published_evidence_archive(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ActorError(
            f"published evidence archive does not exist: {source}",
            code="published_evidence_archive_missing",
        )
    if not source.name.lower().endswith(".evidence.zip"):
        raise ActorError(
            f"not an EIEL evidence archive: {source}",
            code="not_published_evidence_archive",
        )

    match = EVIDENCE_ARCHIVE_RE.match(source.name)
    evidence_id = match.group("evidence_id").upper() if match else None
    slug = match.group("slug") if match else None
    archive_sha = file_sha256(source)
    safety = inspect_zip_safety(source)

    candidate_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for candidate in _sidecar_candidates(source, slug=slug):
        if not candidate.is_file():
            continue
        expected, sidecar_warnings = _read_sidecar(candidate)
        warnings.extend(sidecar_warnings)
        sidecar_match = expected == archive_sha if expected else None
        sidecar_name_match = EVIDENCE_SIDECAR_RE.match(candidate.name)
        candidate_rows.append({
            "path": str(candidate),
            "evidence_id": (
                sidecar_name_match.group("evidence_id").upper()
                if sidecar_name_match else None
            ),
            "expected_sha256": expected,
            "matches": sidecar_match,
        })

    matching_rows = [row for row in candidate_rows if row.get("matches") is True]
    sidecar_row = matching_rows[0] if matching_rows else (candidate_rows[0] if candidate_rows else None)
    if not candidate_rows:
        warnings.append("no published SHA-256 sidecar is present; archive file hash was still computed")
    elif len(matching_rows) > 1:
        warnings.append("multiple checksum sidecars match the archive; the first stable path is reported")
    elif not matching_rows:
        warnings.append("published SHA-256 sidecar does not match the archive")

    sidecar_path = Path(str(sidecar_row["path"])) if sidecar_row else None
    sidecar_expected = str(sidecar_row.get("expected_sha256")) if sidecar_row and sidecar_row.get("expected_sha256") else None
    sidecar_match = bool(sidecar_row.get("matches")) if sidecar_row and sidecar_row.get("matches") is not None else None

    members = [str(row.get("path")) for row in safety.get("members") or []]
    lower_members = [name.lower() for name in members]
    decision_members = sorted(
        name for name in members
        if "decision" in name.lower() and name.lower().endswith((".json", ".md", ".csv"))
    )
    report_members = sorted(name for name in members if name.lower().endswith(".md"))
    raw_manifest_members = sorted(
        name for name in members
        if Path(name).name.lower() == "campaign_evidence_manifest.json"
    )
    metrics_members = sorted(name for name in members if Path(name).name.lower() == "metrics.csv")

    problems: list[str] = []
    if evidence_id is None:
        problems.append(
            "archive filename lacks permanent EIEL-EV-xxxxxx identity; it is not a registered Archive 1.0 object"
        )
    if safety.get("safety_state") != "safe":
        problems.append("ZIP safety/integrity inspection reported one or more issues")
    if sidecar_expected and not sidecar_match:
        problems.append("published SHA-256 sidecar mismatch")

    if problems:
        status = "INVALID_PUBLISHED_EVIDENCE_ARCHIVE"
    elif evidence_id is None:
        status = "UNREGISTERED_EVIDENCE_ARCHIVE"
    else:
        status = "PUBLISHED_EVIDENCE_ARCHIVE_VERIFIED"

    return {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "archive_path": str(source),
        "archive_name": source.name,
        "archive_sha256": archive_sha,
        "evidence_id": evidence_id,
        "slug": slug,
        "status": status,
        "registered_filename": evidence_id is not None,
        "sidecar": {
            "path": str(sidecar_path) if sidecar_path else None,
            "evidence_id": sidecar_row.get("evidence_id") if sidecar_row else None,
            "expected_sha256": sidecar_expected,
            "matches": sidecar_match,
            "candidate_count": len(candidate_rows),
            "candidates": candidate_rows,
        },
        "zip_safety": safety,
        "member_count": int(safety.get("member_count") or 0),
        "inventory": {
            "decision_members": decision_members,
            "report_members": report_members,
            "campaign_evidence_manifest_members": raw_manifest_members,
            "metrics_members": metrics_members,
            "has_decision_material": bool(decision_members),
            "has_report_material": bool(report_members),
            "contains_raw_campaign_manifest": bool(raw_manifest_members),
            "contains_metrics": bool(metrics_members),
        },
        "problems": problems,
        "warnings": warnings,
        "authority_effect": "none_from_integrity_verification",
        "scientific_claim_effect": "none",
        "assessment_eligibility": "not_raw_campaign_input",
        "interpretation": (
            "Packaging integrity and permanent filename identity were inspected. "
            "This does not re-assess campaign outputs, qualify a route, or establish a scientific claim."
        ),
    }
