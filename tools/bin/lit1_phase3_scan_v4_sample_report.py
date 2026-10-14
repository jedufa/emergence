#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = 1
RUN_ID_RE = re.compile(
    r"__context_(?P<context>.+?)__init_(?P<init_arm>.+?)__seed(?P<seed>\d+)$"
)

SCAN_FIELDS = [
    "run_id",
    "context",
    "init_arm",
    "seed",
    "status",
    "event_file_present",
    "event_rows_read",
    "event_schema_versions_seen",
    "reproduction_scan_exposure_rows",
    "reproduction_scan_eligible_rows",
    "reproduction_scan_ineligible_rows",
    "reproduction_scan_no_attempt_rows",
    "reproduction_scan_attempt_recorded_rows",
    "attempt_sources",
    "scan_exposure_metric_status",
    "missing_required_metrics",
    "top1_dominance_generation",
    "effective_lineage_collapse_generation",
    "run_dir",
    "phase1_output_dir",
]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _parse_run_id(run_id: str) -> dict[str, Any]:
    match = RUN_ID_RE.search(run_id)
    if not match:
        raise ValueError(f"cannot parse LIT-1 scan-v4 run_id: {run_id}")
    return {
        "context": match.group("context"),
        "init_arm": match.group("init_arm"),
        "seed": int(match.group("seed")),
    }


def _phase1_dir_for_run(phase1_root: Path, run_dir: Path) -> Path:
    matches = sorted(phase1_root.glob(f"*_{run_dir.name}"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one Phase 1 output directory for run {run_dir.name}, found {len(matches)}"
        )
    return matches[0]


def _cell_from_result(result: Mapping[str, Any], phase1_root: Path) -> dict[str, Any]:
    run_id = str(result["run_id"])
    run_dir = Path(str(result["run_dir"]))
    parsed = _parse_run_id(run_id)
    phase1_dir = _phase1_dir_for_run(phase1_root, run_dir)
    summary = _read_json(phase1_dir / "lit1_lineage_funnel_phase1_summary.json")
    coverage = _read_json(phase1_dir / "lit1_lineage_funnel_phase1_metric_coverage.json")
    collapse = _read_json(phase1_dir / "lit1_lineage_funnel_phase1_collapse_stage_summary.json")
    event_summary = summary.get("event_summary", {}) or {}
    counts = event_summary.get("event_type_counts", {}) or {}
    metric_status = coverage.get("required_metric_status", {}) or {}
    return {
        "run_id": run_id,
        "context": parsed["context"],
        "init_arm": parsed["init_arm"],
        "seed": parsed["seed"],
        "status": str(result.get("status", "")),
        "event_file_present": bool(event_summary.get("event_file_present", False)),
        "event_rows_read": int(event_summary.get("event_rows_read", 0)),
        "event_schema_versions_seen": ",".join(str(item) for item in event_summary.get("event_schema_versions_seen", [])),
        "reproduction_scan_exposure_rows": int(counts.get("reproduction_scan_exposure", 0)),
        "reproduction_scan_eligible_rows": int(counts.get("reproduction_scan_eligible", 0)),
        "reproduction_scan_ineligible_rows": int(counts.get("reproduction_scan_ineligible", 0)),
        "reproduction_scan_no_attempt_rows": int(counts.get("reproduction_scan_no_attempt", 0)),
        "reproduction_scan_attempt_recorded_rows": int(counts.get("reproduction_scan_attempt_recorded", 0)),
        "attempt_sources": ",".join(str(item) for item in summary.get("attempt_sources", [])),
        "scan_exposure_metric_status": str(metric_status.get("scan_exposure_count", "")),
        "missing_required_metrics": ",".join(str(item) for item in coverage.get("missing_required_metrics", [])),
        "top1_dominance_generation": int(collapse.get("top1_dominance_generation", -1)),
        "effective_lineage_collapse_generation": int(collapse.get("effective_lineage_collapse_generation", -1)),
        "run_dir": str(run_dir),
        "phase1_output_dir": str(phase1_dir),
    }


def _totals(cells: list[Mapping[str, Any]]) -> dict[str, Any]:
    scan_fields = [
        "event_rows_read",
        "reproduction_scan_exposure_rows",
        "reproduction_scan_eligible_rows",
        "reproduction_scan_ineligible_rows",
        "reproduction_scan_no_attempt_rows",
        "reproduction_scan_attempt_recorded_rows",
    ]
    out = {field: sum(int(row.get(field, 0)) for row in cells) for field in scan_fields}
    out["event_files_present"] = sum(1 for row in cells if bool(row.get("event_file_present", False)))
    out["event_schema_versions_seen"] = sorted(
        {version for row in cells for version in str(row.get("event_schema_versions_seen", "")).split(",") if version}
    )
    out["attempt_sources"] = sorted(
        {source for row in cells for source in str(row.get("attempt_sources", "")).split(",") if source}
    )
    out["scan_exposure_metric_statuses"] = sorted(
        {str(row.get("scan_exposure_metric_status", "")) for row in cells if row.get("scan_exposure_metric_status", "")}
    )
    return out


def _decision(campaign: Mapping[str, Any], cells: list[Mapping[str, Any]], expected_runs: int) -> dict[str, Any]:
    failures = list(campaign.get("failures", []) or [])
    totals = _totals(cells)
    failed_criteria: list[str] = []
    if int(campaign.get("planned_runs", 0)) != expected_runs:
        failed_criteria.append("planned_run_count")
    if sum(1 for row in cells if str(row.get("status", "")) == "ok") != expected_runs or failures:
        failed_criteria.append("campaign_completion")
    if len(cells) != expected_runs:
        failed_criteria.append("phase1_output_count")
    if int(totals["event_files_present"]) != expected_runs:
        failed_criteria.append("event_file_presence")
    if totals["event_schema_versions_seen"] != ["4"]:
        failed_criteria.append("schema_v4_only")
    if int(totals["reproduction_scan_exposure_rows"]) <= 0:
        failed_criteria.append("scan_exposure_rows")
    if int(totals["reproduction_scan_attempt_recorded_rows"]) <= 0:
        failed_criteria.append("scan_attempt_rows")
    if totals["scan_exposure_metric_statuses"] != ["available_from_schema_v4_events"]:
        failed_criteria.append("scan_metric_coverage")
    if any(str(row.get("missing_required_metrics", "")) for row in cells):
        failed_criteria.append("missing_required_metrics")

    if failed_criteria:
        decision = "LIT-1-PHASE3-SCANV4-C"
        interpretation = "Scan-v4 sampled event telemetry is incomplete or not schema-v4 scan-exposure capable."
    else:
        decision = "LIT-1-PHASE3-SCANV4-A"
        interpretation = (
            "Scan-v4 sampled event telemetry is valid for measuring live reproduction scan exposure, "
            "eligibility, and scan-attempt records on the selected LIT-1 corners."
        )
    return {
        "decision": decision,
        "interpretation": interpretation,
        "failed_criteria": failed_criteria,
        "expected_runs": int(expected_runs),
        "completed_runs": sum(1 for row in cells if str(row.get("status", "")) == "ok"),
        "totals": totals,
        "recommended_next_step": (
            "Integrate Phase 2 CSV localization and Phase 3 scan-v4 sampled telemetry into the final "
            "LIT-1 A/B/C decision package without changing lineage mechanics."
        ),
    }


def build_report(
    *,
    campaign_summary_path: Path,
    phase1_output_root: Path,
    expected_runs: int = 6,
    uncorrected_phase1_output_root: Path | None = None,
) -> dict[str, Any]:
    campaign = _read_json(campaign_summary_path)
    cells = [
        _cell_from_result(result, phase1_output_root)
        for result in sorted(campaign.get("results", []), key=lambda row: int(row.get("index", 0)))
    ]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "campaign_summary_path": str(campaign_summary_path),
        "phase1_output_root": str(phase1_output_root),
        "registry": str(campaign.get("registry", "")),
        "batch_tag": str(campaign.get("batch_tag", "")),
        "campaign_output_profile": str(campaign.get("campaign_output_profile", "")),
        "planned_runs": int(campaign.get("planned_runs", 0)),
        "result_rows": len(cells),
        "workers": int(campaign.get("workers", 0)),
        "total_wall_s": float(campaign.get("total_wall_s", 0.0) or 0.0),
        "mean_child_wall_s": float(campaign.get("mean_child_wall_s", 0.0) or 0.0),
        "progress_ui_attach_command": str((campaign.get("progress_ui", {}) or {}).get("attach_command", "")),
        "decision": _decision(campaign, cells, expected_runs),
        "no_claims": (
            "This LIT-1 Phase 3 scan-v4 sample is telemetry validation only. It does not show intelligence, "
            "emergence, consciousness, broad cognition, memory selection, prediction selection, solved "
            "lineage diversity, or a lineage repair."
        ),
    }
    if uncorrected_phase1_output_root:
        uncorrected_cells = []
        for summary_path in sorted(uncorrected_phase1_output_root.glob("*/lit1_lineage_funnel_phase1_summary.json")):
            event_summary = _read_json(summary_path).get("event_summary", {}) or {}
            counts = event_summary.get("event_type_counts", {}) or {}
            uncorrected_cells.append(
                {
                    "event_rows_read": int(event_summary.get("event_rows_read", 0)),
                    "event_schema_versions_seen": ",".join(
                        str(item) for item in event_summary.get("event_schema_versions_seen", [])
                    ),
                    "reproduction_scan_exposure_rows": int(counts.get("reproduction_scan_exposure", 0)),
                    "reproduction_scan_attempt_recorded_rows": int(
                        counts.get("reproduction_scan_attempt_recorded", 0)
                    ),
                }
            )
        counter = Counter()
        versions = set()
        for row in uncorrected_cells:
            counter["event_rows_read"] += int(row["event_rows_read"])
            counter["reproduction_scan_exposure_rows"] += int(row["reproduction_scan_exposure_rows"])
            counter["reproduction_scan_attempt_recorded_rows"] += int(row["reproduction_scan_attempt_recorded_rows"])
            versions.update(str(row["event_schema_versions_seen"]).split(","))
        summary["uncorrected_attempt"] = {
            "phase1_output_root": str(uncorrected_phase1_output_root),
            "cells": len(uncorrected_cells),
            "event_schema_versions_seen": sorted(version for version in versions if version),
            "event_rows_read": int(counter["event_rows_read"]),
            "reproduction_scan_exposure_rows": int(counter["reproduction_scan_exposure_rows"]),
            "reproduction_scan_attempt_recorded_rows": int(counter["reproduction_scan_attempt_recorded_rows"]),
            "interpretation": (
                "Initial event run enabled event JSONL but not schema-v4 scan-action telemetry; it is "
                "recorded as a failed instrumentation attempt, not as scan evidence."
            ),
        }
    return {"summary": summary, "cells": cells}


def write_report_outputs(report: Mapping[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "scan_v4_sample_summary.json",
        "cell_summary": output_dir / "scan_v4_sample_cell_summary.csv",
    }
    _write_json(paths["summary"], report["summary"])
    _write_csv(paths["cell_summary"], report["cells"], SCAN_FIELDS)
    return {key: str(value) for key, value in paths.items()}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LIT-1 Phase 3 scan-v4 sample report tables.")
    parser.add_argument("--campaign-summary", type=Path, required=True)
    parser.add_argument("--phase1-output-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-runs", type=int, default=6)
    parser.add_argument("--uncorrected-phase1-output-root", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        campaign_summary_path=args.campaign_summary,
        phase1_output_root=args.phase1_output_root,
        expected_runs=int(args.expected_runs),
        uncorrected_phase1_output_root=args.uncorrected_phase1_output_root,
    )
    outputs = write_report_outputs(report, args.output_dir)
    print(json.dumps({"ok": True, "outputs": outputs}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
