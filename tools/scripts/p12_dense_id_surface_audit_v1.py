#!/usr/bin/env python3
"""Deterministic dense historical-ID surface audit for P1.2."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
for import_root in (ROOT, ROOT / "engine"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from engine.authoritative_state.schema import content_sha256  # noqa: E402


SCHEMA_VERSION = "p12_dense_id_surface_audit_v2"
DECISION = "P12-DENSE-ID-SURFACES-OPEN"
AUTHORITY_DECISION = "P12-AUTHORITY-TRIAGE-OPEN"
DEFAULT_SCAN_ROOTS = ("engine", "tools", "tests")
STAGE1_RAW_SEARCH_PATHS = (
    "engine/perf",
    "engine/native_runtime",
    "engine/env",
    "engine/evolution/population_pkg",
    "tests/perf",
    "tests/test_gridworld_stepping_sparse_identity.py",
    "tests/test_eiel_wdl_checkpoint_descriptor_plan.py",
    "tests/test_eiel_eed0_resident_checkpoint_gate_v1.py",
)
TEXT_SUFFIXES = {".py", ".cpp", ".cu", ".hpp", ".h", ".cuh", ".json"}
EXCLUDED_PATH_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
}
EXCLUDED_RELATIVE_PATHS = {
    "tools/scripts/p12_dense_id_surface_audit_v1.py",
    "tests/test_p12_dense_id_surface_audit_v1.py",
}


@dataclass(frozen=True)
class SurfacePattern:
    surface_id: str
    category: str
    status: str
    rationale: str
    regex: re.Pattern[str]


@dataclass(frozen=True)
class SurfaceMatch:
    surface_id: str
    category: str
    status: str
    domain: str
    path: str
    line: int
    text: str
    rationale: str
    authority_category: str
    authority_action: str
    is_authoritative_defect: bool
    is_compatibility_debt: bool


PATTERNS: tuple[SurfacePattern, ...] = (
    SurfacePattern(
        surface_id="dense_agent_id_to_slot_index",
        category="array index",
        status="unsafe_or_unqualified_dense_index",
        rationale="stable ID is used as a direct index into a dense ID-to-slot map",
        regex=re.compile(
            r"agent_id_to_slot\s*\[[^\]\n]*(agent_id|child_id|parent_id|"
            r"selected_agent_id|prepared_child_id|mismatch_agent_id)"
        ),
    ),
    SurfacePattern(
        surface_id="dense_physical_state_index_by_agent_id",
        category="array index",
        status="unsafe_or_unqualified_dense_index",
        rationale="stable ID indexes physical resident state arrays",
        regex=re.compile(
            r"(agent_alive|agent_pos_y|agent_pos_x|pos_y_by_id|pos_x_by_id|"
            r"energy_by_id|health_by_id|injury_by_id|scar_by_id)\s*"
            r"\[[^\]\n]*(agent_id|agent_ids|aid|ids)\]"
        ),
    ),
    SurfacePattern(
        surface_id="int32_stable_id_transport",
        category="public identity boundary",
        status="needs_public_identity_decision",
        rationale="stable or public identity appears to be narrowed to int32",
        regex=re.compile(
            r"(agent_ids?|parent_id|child_id|selected_agent_id|event_tape_.*id|"
            r"typed_lifecycle_event_.*agent_id|agent_id_to_slot).*"
            r"(np\.int32|torch\.int32|torch::kInt32|kInt32|int32_t)"
        ),
    ),
    SurfacePattern(
        surface_id="agent_grid_public_occupant_value",
        category="grid occupant value",
        status="needs_public_identity_decision",
        rationale="agent_grid is an int32 public occupant surface whose slot-vs-stable-ID meaning must be frozen",
        regex=re.compile(
            r"(agent_grid|grid_a).*(np\.int32|torch\.int32|torch::kInt32|"
            r"kInt32|int32_t|accessor<int32_t)"
        ),
    ),
    SurfacePattern(
        surface_id="accepted_sparse_identity_contract",
        category="cache key",
        status="accepted_sparse_identity_evidence",
        rationale="sparse or slot-owned identity surface uses int64 IDs with bounded slot rows",
        regex=re.compile(
            r"(agent_positions_by_id|agent_id_to_slot_agent_ids|"
            r"agent_id_to_slot_slots|slot_to_agent_id.*(int64|kInt64|torch::kInt64)|"
            r"active_agent_ids.*(int64|kInt64|torch::kInt64)|"
            r"compact identity v2 rejects dense)"
        ),
    ),
    SurfacePattern(
        surface_id="identity_cache_or_lookup_by_stable_id",
        category="cache key",
        status="needs_slot_token_cursor_review",
        rationale="dictionary/cache lookup by stable ID needs explicit slot-token ownership review",
        regex=re.compile(
            r"(by_id|by_agent|source_by_id|family_by_agent|origin_by_id|"
            r"current_slots|slot_by_id|positions|ancestry)\s*"
            r"\[[^\]\n]*(agent_id|child_id|parent_id|aid|stable_id)\]"
        ),
    ),
    SurfacePattern(
        surface_id="checkpoint_dense_id_reconstruction",
        category="checkpoint shape",
        status="needs_checkpoint_migration_or_rejection",
        rationale="checkpoint import/export touches stable-ID keyed dense or reconstructed state",
        regex=re.compile(
            r"(dense_[xy]|source\[agent_id\]|positions\[agent_id\]|"
            r"grid_positions\[agent_id\]|cache\[agent_id\]|by_id\[agent_id\])"
        ),
    ),
    SurfacePattern(
        surface_id="wdl_eed_respec_agent_keyed_reporting",
        category="report-buffer index",
        status="needs_compact_reporting_capacity_contract",
        rationale="WDL/EED/RESPEC reporting or cursor surface is keyed by agent or lineage identity",
        regex=re.compile(
            r"(wdl|eed|respec|exposure|settlement|attribution).*"
            r"(agent_id|agent_ids|lineage_id|slot_token|cursor|snapshot)"
        ),
    ),
    SurfacePattern(
        surface_id="event_agent_id_surface",
        category="event key",
        status="needs_high_id_event_parity",
        rationale="event tape carries stable IDs and needs high-ID/ragged parity evidence",
        regex=re.compile(
            r"(event_tape_(actor|target)_id|typed_lifecycle_event_"
            r"(actor|target|parent|child)_agent_id)"
        ),
    ),
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )


def _is_scannable(path: Path, root: Path) -> bool:
    rel = path.relative_to(root).as_posix()
    if rel in EXCLUDED_RELATIVE_PATHS:
        return False
    if any(part in EXCLUDED_PATH_PARTS for part in path.parts):
        return False
    return path.is_file() and path.suffix in TEXT_SUFFIXES


def _iter_files(root: Path, scan_roots: Sequence[str]) -> Iterable[Path]:
    for scan_root in scan_roots:
        base = root / scan_root
        if not base.exists():
            continue
        if _is_scannable(base, root):
            yield base
            continue
        for path in sorted(base.rglob("*")):
            if _is_scannable(path, root):
                yield path


def _domain(path: str) -> str:
    lowered = path.lower()
    if "checkpoint" in lowered:
        return "checkpoint"
    if "wdl" in lowered or "delayed" in lowered:
        return "wdl"
    if "eed" in lowered:
        return "eed"
    if "respec" in lowered:
        return "respec"
    if "cuda" in lowered or lowered.endswith(".cu") or "_kernel" in lowered:
        return "cuda"
    if "native" in lowered or "cxx" in lowered:
        return "native_cpu"
    if "patch" in lowered:
        return "patch"
    if "chronicle" in lowered:
        return "chronicle"
    if "campaign" in lowered or "route" in lowered or "qualification" in lowered:
        return "route_campaign"
    if lowered.startswith("tests/"):
        return "test_fixture"
    return "python_runtime"


def _status_for_path(status: str, path: str) -> str:
    if path.startswith("tests/") and status.startswith("unsafe"):
        return "test_fixture_or_legacy_dense_index"
    if "/historical/" in path or path.startswith("tools/historical/"):
        return "historical_or_analysis_only"
    return status


def _authority_category(path: str, status: str, domain: str, text: str) -> str:
    lowered = path.lower()
    lowered_text = text.lower()
    if lowered.startswith("tests/"):
        return "test-only"
    if "generated" in lowered or "/_generated" in lowered:
        return "generated source"
    if "compat" in lowered_text or "legacy_dense" in lowered_text:
        return "compatibility-only"
    if domain == "checkpoint":
        return "checkpoint authority"
    if domain in {"wdl", "eed", "respec", "chronicle"}:
        return "reporting/Chronicle authority"
    if domain == "route_campaign":
        return "route/package ABI"
    if status in {
        "accepted_sparse_identity_evidence",
        "needs_public_identity_decision",
    }:
        return "production identity authority"
    if status in {
        "unsafe_or_unqualified_dense_index",
        "needs_slot_token_cursor_review",
    }:
        return "production mutable-state authority"
    return "false positive"


def _is_authoritative_defect(status: str, authority_category: str) -> bool:
    if authority_category in {
        "test-only",
        "generated source",
        "comments/docs",
        "false positive",
        "compatibility-only",
    }:
        return False
    return status in {
        "unsafe_or_unqualified_dense_index",
        "needs_public_identity_decision",
        "needs_checkpoint_migration_or_rejection",
        "needs_compact_reporting_capacity_contract",
        "needs_high_id_event_parity",
        "needs_slot_token_cursor_review",
    }


def _is_compatibility_debt(status: str, authority_category: str, text: str) -> bool:
    lowered = text.lower()
    return (
        authority_category == "compatibility-only"
        or "compat" in lowered
        or status == "test_fixture_or_legacy_dense_index"
    )


def _authority_action(
    *,
    status: str,
    authority_category: str,
    is_authoritative_defect: bool,
    is_compatibility_debt: bool,
) -> str:
    if authority_category == "test-only":
        return "update or retain only after production ABI decision"
    if authority_category == "generated source":
        return "regenerate from authoritative schema if source changes"
    if is_compatibility_debt:
        return "retire or quarantine as non-authoritative compatibility"
    if authority_category == "false positive":
        return "no P1.2 action"
    if status == "accepted_sparse_identity_evidence":
        return "preserve as accepted sparse identity evidence"
    if is_authoritative_defect:
        return "repair or qualify before P12-A"
    return "review before claim"


def scan_dense_id_surfaces_v1(
    *,
    source_root: Path | str = ROOT,
    scan_roots: Sequence[str] = DEFAULT_SCAN_ROOTS,
) -> list[SurfaceMatch]:
    root = Path(source_root).resolve()
    matches: list[SurfaceMatch] = []
    for path in _iter_files(root, scan_roots):
        rel = path.relative_to(root).as_posix()
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            lines = path.read_text(encoding="latin-1").splitlines()
        for line_no, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            for pattern in PATTERNS:
                if not pattern.regex.search(stripped):
                    continue
                status = _status_for_path(pattern.status, rel)
                domain = _domain(rel)
                authority_category = _authority_category(
                    rel,
                    status,
                    domain,
                    stripped,
                )
                is_authoritative_defect = _is_authoritative_defect(
                    status,
                    authority_category,
                )
                is_compatibility_debt = _is_compatibility_debt(
                    status,
                    authority_category,
                    stripped,
                )
                matches.append(
                    SurfaceMatch(
                        surface_id=pattern.surface_id,
                        category=pattern.category,
                        status=status,
                        domain=domain,
                        path=rel,
                        line=line_no,
                        text=stripped[:240],
                        rationale=pattern.rationale,
                        authority_category=authority_category,
                        authority_action=_authority_action(
                            status=status,
                            authority_category=authority_category,
                            is_authoritative_defect=is_authoritative_defect,
                            is_compatibility_debt=is_compatibility_debt,
                        ),
                        is_authoritative_defect=is_authoritative_defect,
                        is_compatibility_debt=is_compatibility_debt,
                    )
                )
    return matches


def _aggregate(matches: Sequence[SurfaceMatch]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    for match in matches:
        key = (
            match.surface_id,
            match.category,
            match.status,
            match.domain,
            match.path,
            match.authority_category,
        )
        row = grouped.get(key)
        if row is None:
            row = {
                "surface_id": match.surface_id,
                "category": match.category,
                "status": match.status,
                "domain": match.domain,
                "path": match.path,
                "first_line": match.line,
                "match_count": 0,
                "sample": match.text,
                "rationale": match.rationale,
                "authority_category": match.authority_category,
                "authority_action": match.authority_action,
                "is_authoritative_defect": match.is_authoritative_defect,
                "is_compatibility_debt": match.is_compatibility_debt,
            }
            grouped[key] = row
        row["match_count"] += 1
    return sorted(
        grouped.values(),
        key=lambda row: (
            str(row["status"]),
            str(row["domain"]),
            str(row["path"]),
            str(row["authority_category"]),
            int(row["first_line"]),
            str(row["surface_id"]),
        ),
    )


def _count_raw_identity_files(
    *,
    source_root: Path,
    scan_roots: Sequence[str],
) -> int:
    raw_terms = re.compile(
        r"agent_id_to_slot|slot_to_agent_id|population_next_id|next_id|"
        r"agent_grid|agent_pos_[xy]|slot_token|pending_free_agent_ids"
    )
    count = 0
    for path in _iter_files(source_root, scan_roots):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="latin-1")
        if raw_terms.search(text):
            count += 1
    return count


def _authority_summary(surfaces: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    non_authoritative = {
        "test-only",
        "generated source",
        "comments/docs",
        "false positive",
        "compatibility-only",
    }
    for row in surfaces:
        category = str(row["authority_category"])
        summary = grouped.setdefault(
            category,
            {
                "authority_category": category,
                "files": set(),
                "authoritative_files": set(),
                "defect_files": set(),
                "compatibility_debt_files": set(),
                "action": str(row["authority_action"]),
            },
        )
        path = str(row["path"])
        summary["files"].add(path)
        if category not in non_authoritative:
            summary["authoritative_files"].add(path)
        if bool(row["is_authoritative_defect"]):
            summary["defect_files"].add(path)
        if bool(row["is_compatibility_debt"]):
            summary["compatibility_debt_files"].add(path)
    result: list[dict[str, Any]] = []
    for summary in grouped.values():
        if summary["authority_category"] == "test-only":
            action = "update or retain only after production ABI decision"
        elif summary["compatibility_debt_files"]:
            action = "retire or quarantine as non-authoritative compatibility"
        elif summary["defect_files"]:
            action = "repair or qualify before P12-A"
        else:
            action = str(summary["action"])
        result.append(
            {
                "authority_category": summary["authority_category"],
                "files": len(summary["files"]),
                "authoritative_files": len(summary["authoritative_files"]),
                "defects": len(summary["defect_files"]),
                "compatibility_debt": len(summary["compatibility_debt_files"]),
                "action": action,
            }
        )
    return sorted(result, key=lambda row: str(row["authority_category"]))


def build_dense_id_surface_audit_v1(
    *,
    source_root: Path | str = ROOT,
    source_commit: str | None = None,
) -> dict[str, Any]:
    root = Path(source_root).resolve()
    matches = scan_dense_id_surfaces_v1(source_root=root)
    surfaces = _aggregate(matches)
    counts_by_status: dict[str, int] = {}
    counts_by_category: dict[str, int] = {}
    counts_by_domain: dict[str, int] = {}
    for row in surfaces:
        counts_by_status[str(row["status"])] = counts_by_status.get(str(row["status"]), 0) + 1
        counts_by_category[str(row["category"])] = counts_by_category.get(str(row["category"]), 0) + 1
        counts_by_domain[str(row["domain"])] = counts_by_domain.get(str(row["domain"]), 0) + 1
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "decision": DECISION,
        "authority_decision": AUTHORITY_DECISION,
        "source_commit": source_commit,
        "scan_roots": list(DEFAULT_SCAN_ROOTS),
        "pattern_count": len(PATTERNS),
        "raw_identity_file_count": _count_raw_identity_files(
            source_root=root,
            scan_roots=STAGE1_RAW_SEARCH_PATHS,
        ),
        "expanded_triage_identity_file_count": _count_raw_identity_files(
            source_root=root,
            scan_roots=DEFAULT_SCAN_ROOTS,
        ),
        "raw_match_count": len(matches),
        "surface_count": len(surfaces),
        "authoritative_defect_file_count": len(
            {
                str(row["path"])
                for row in surfaces
                if bool(row["is_authoritative_defect"])
            }
        ),
        "compatibility_debt_file_count": len(
            {
                str(row["path"])
                for row in surfaces
                if bool(row["is_compatibility_debt"])
            }
        ),
        "counts_by_status": dict(sorted(counts_by_status.items())),
        "counts_by_category": dict(sorted(counts_by_category.items())),
        "counts_by_domain": dict(sorted(counts_by_domain.items())),
        "authority_summary": _authority_summary(surfaces),
        "surfaces": surfaces,
        "stage1_boundary": (
            "This is a static surface and authority triage. It does not close "
            "dense-ID scaling without follow-up high-ID/ragged/slot-reuse route "
            "evidence and public identity ABI decisions."
        ),
    }
    payload["audit_sha256"] = content_sha256(payload)
    return payload


def render_markdown_v1(payload: Mapping[str, Any], json_path: Path) -> str:
    counts = payload["counts_by_status"]
    lines = [
        "# EIEL P1.2 Stage 2 Dense-ID Authority Triage",
        "",
        "Date: 2026-07-27",
        f"Decision: `{payload['decision']}`",
        f"Source commit: `{payload.get('source_commit')}`",
        f"Audit SHA-256: `{payload['audit_sha256']}`",
        "",
        "Win/Loss: Win for converting the accepted Stage 1 search space into "
        "authority categories; loss for final dense-ID closure. The audit table "
        "identifies closure-critical authoritative surfaces, so dense historical-ID "
        "scaling remains open until those surfaces are repaired or qualified with "
        "route-bound evidence.",
        "",
        "## Counts",
        "",
        f"- Raw Stage 1 identity file count: `{payload['raw_identity_file_count']}`",
        f"- Expanded triage identity file count: `{payload['expanded_triage_identity_file_count']}`",
        f"- Raw pattern matches: `{payload['raw_match_count']}`",
        f"- Aggregated surfaces: `{payload['surface_count']}`",
        f"- Authoritative defect files: `{payload['authoritative_defect_file_count']}`",
        f"- Compatibility-debt files: `{payload['compatibility_debt_file_count']}`",
    ]
    for status, count in sorted(counts.items()):
        lines.append(f"- `{status}`: `{count}` surfaces")
    lines.extend(
        [
            "",
            "## Authority Triage",
            "",
            "| Category | Files | Authoritative files | Defects | Compatibility debt | Action |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in payload["authority_summary"]:
        lines.append(
            "| {category} | {files} | {authoritative} | {defects} | {compat} | {action} |".format(
                category=row["authority_category"],
                files=row["files"],
                authoritative=row["authoritative_files"],
                defects=row["defects"],
                compat=row["compatibility_debt"],
                action=row["action"],
            )
        )
    lines.extend(
        [
            "",
            "## Closure-Critical Surface Set",
            "",
            "- Public identity ABI: production identity authority surfaces with "
            "`needs_public_identity_decision`.",
            "- CUDA compatibility shadow: compatibility-debt and legacy dense-index "
            "test surfaces that must not become production authority.",
            "- Checkpoint identity authority: checkpoint surfaces carrying dense, "
            "public, or event identity fields.",
            "- WDL compact reporting: WDL/EED/RESPEC/Chronicle report-buffer and "
            "cursor surfaces requiring capacity and ownership contracts.",
            "- Route/package ABI: campaign/provider/materializer surfaces that must "
            "bind the chosen identity and reporting contracts.",
            "- Mutable-state lookup: production mutable-state authority surfaces that "
            "must use slots plus slot tokens rather than stable-ID indexing.",
            "",
            "## High-Risk Surface Table",
            "",
            "| Status | Authority | Domain | Category | Surface | File:line | Count |",
            "| --- | --- | --- | --- | --- | --- | ---: |",
        ]
    )
    high_risk_statuses = {
        "unsafe_or_unqualified_dense_index",
        "needs_public_identity_decision",
        "needs_checkpoint_migration_or_rejection",
        "needs_compact_reporting_capacity_contract",
        "needs_high_id_event_parity",
        "needs_slot_token_cursor_review",
    }
    shown = 0
    for row in payload["surfaces"]:
        if row["status"] not in high_risk_statuses:
            continue
        lines.append(
            "| {status} | {authority} | {domain} | {category} | {surface} | `{path}:{line}` | {count} |".format(
                status=row["status"],
                authority=row["authority_category"],
                domain=row["domain"],
                category=row["category"],
                surface=row["surface_id"],
                path=row["path"],
                line=row["first_line"],
                count=row["match_count"],
            )
        )
        shown += 1
        if shown >= 40:
            break
    lines.extend(
        [
            "",
            "Full table:",
            f"`{json_path.name}`",
            "",
            "## Boundary",
            "",
            "This Stage 2 triage is static evidence only. It does not prove checkpoint "
            "migration, WDL compact-reporting capacity, EED/RESPEC cursor ownership, "
            "CUDA route qualification, or `INT64_MAX` exhaustion safety. Those remain "
            "P1.2 work.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_outputs(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.source_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = build_dense_id_surface_audit_v1(
        source_root=root,
        source_commit=args.source_commit,
    )
    json_path = output_dir / args.json_name
    report_path = output_dir / args.report_name
    _write_json(json_path, payload)
    report_path.write_text(render_markdown_v1(payload, json_path), encoding="ascii")
    return {
        "ok": True,
        "decision": payload["decision"],
        "surface_count": payload["surface_count"],
        "raw_match_count": payload["raw_match_count"],
        "audit_sha256": payload["audit_sha256"],
        "json_path": str(json_path),
        "json_sha256": _file_sha256(json_path),
        "report_path": str(report_path),
        "report_sha256": _file_sha256(report_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(ROOT))
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--json-name",
        default="p12_dense_id_surface_audit_v1.json",
    )
    parser.add_argument(
        "--report-name",
        default="EIEL_P12_Stage1_Dense_ID_Surface_Audit_2026_07_27.md",
    )
    args = parser.parse_args(argv)
    print(json.dumps(build_outputs(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
