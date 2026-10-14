#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = 1
RUN_ID_RE = re.compile(
    r"__context_(?P<context>.+?)__init_(?P<init_arm>.+?)__seed(?P<seed>\d+)$"
)

COMPACT_FIELDS = [
    "run_id",
    "context",
    "init_arm",
    "seed",
    "status",
    "generation_count",
    "lineage_generation_row_count",
    "top1_dominance_generation",
    "effective_lineage_collapse_generation",
    "final_top1_post_refill_alive_share",
    "final_effective_post_refill_alive_lineages",
    "min_effective_post_refill_alive_lineages",
    "max_top1_post_refill_alive_share",
    "event_file_present",
    "attempt_sources",
    "missing_required_metrics",
    "run_dir",
    "phase1_output_dir",
]

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


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _parse_run_id(run_id: str) -> dict[str, Any]:
    match = RUN_ID_RE.search(run_id)
    if not match:
        raise ValueError(f"cannot parse OES-LITGATE-1 run_id: {run_id}")
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


def _compact_cell_from_result(result: Mapping[str, Any], phase1_root: Path) -> dict[str, Any]:
    run_id = str(result["run_id"])
    run_dir = Path(str(result["run_dir"]))
    parsed = _parse_run_id(run_id)
    phase1_dir = _phase1_dir_for_run(phase1_root, run_dir)
    summary = _read_json(phase1_dir / "lit1_lineage_funnel_phase1_summary.json")
    coverage = _read_json(phase1_dir / "lit1_lineage_funnel_phase1_metric_coverage.json")
    collapse = _read_json(phase1_dir / "lit1_lineage_funnel_phase1_collapse_stage_summary.json")
    generations = list(collapse.get("by_generation", []))
    final = generations[-1] if generations else {}
    effective_values = [
        _safe_float(row.get("effective_post_refill_alive_lineages", 0.0)) for row in generations
    ]
    top1_values = [_safe_float(row.get("top1_post_refill_alive_share", 0.0)) for row in generations]
    return {
        "run_id": run_id,
        "context": parsed["context"],
        "init_arm": parsed["init_arm"],
        "seed": parsed["seed"],
        "status": str(result.get("status", "")),
        "generation_count": int(summary.get("generation_count", 0)),
        "lineage_generation_row_count": int(summary.get("lineage_generation_row_count", 0)),
        "top1_dominance_generation": int(collapse.get("top1_dominance_generation", -1)),
        "effective_lineage_collapse_generation": int(collapse.get("effective_lineage_collapse_generation", -1)),
        "final_top1_post_refill_alive_share": _safe_float(final.get("top1_post_refill_alive_share", 0.0)),
        "final_effective_post_refill_alive_lineages": _safe_float(
            final.get("effective_post_refill_alive_lineages", 0.0)
        ),
        "min_effective_post_refill_alive_lineages": min(effective_values) if effective_values else 0.0,
        "max_top1_post_refill_alive_share": max(top1_values) if top1_values else 0.0,
        "event_file_present": bool((summary.get("event_summary", {}) or {}).get("event_file_present", False)),
        "attempt_sources": ",".join(str(item) for item in summary.get("attempt_sources", [])),
        "missing_required_metrics": ",".join(
            str(item) for item in coverage.get("missing_required_metrics", [])
        ),
        "run_dir": str(run_dir),
        "phase1_output_dir": str(phase1_dir),
    }


def _scan_cell_from_result(result: Mapping[str, Any], phase1_root: Path) -> dict[str, Any]:
    run_id = str(result["run_id"])
    run_dir = Path(str(result["run_dir"]))
    parsed = _parse_run_id(run_id)
    phase1_dir = _phase1_dir_for_run(phase1_root, run_dir)
    summary = _read_json(phase1_dir / "lit1_lineage_funnel_phase1_summary.json")
    coverage = _read_json(phase1_dir / "lit1_lineage_funnel_phase1_metric_coverage.json")
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
        "event_schema_versions_seen": ",".join(
            str(item) for item in event_summary.get("event_schema_versions_seen", [])
        ),
        "reproduction_scan_exposure_rows": int(counts.get("reproduction_scan_exposure", 0)),
        "reproduction_scan_eligible_rows": int(counts.get("reproduction_scan_eligible", 0)),
        "reproduction_scan_ineligible_rows": int(counts.get("reproduction_scan_ineligible", 0)),
        "reproduction_scan_no_attempt_rows": int(counts.get("reproduction_scan_no_attempt", 0)),
        "reproduction_scan_attempt_recorded_rows": int(counts.get("reproduction_scan_attempt_recorded", 0)),
        "attempt_sources": ",".join(str(item) for item in summary.get("attempt_sources", [])),
        "scan_exposure_metric_status": str(metric_status.get("scan_exposure_count", "")),
        "missing_required_metrics": ",".join(
            str(item) for item in coverage.get("missing_required_metrics", [])
        ),
        "run_dir": str(run_dir),
        "phase1_output_dir": str(phase1_dir),
    }


def _campaign_cells(
    campaign_summary: Mapping[str, Any],
    phase1_root: Path,
    *,
    scan: bool,
) -> list[dict[str, Any]]:
    builder = _scan_cell_from_result if scan else _compact_cell_from_result
    return [
        builder(result, phase1_root)
        for result in sorted(campaign_summary.get("results", []), key=lambda row: int(row.get("index", 0)))
    ]


def _compact_context_summary(cells: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for cell in cells:
        groups[(str(cell["context"]), str(cell["init_arm"]))].append(cell)
    rows: list[dict[str, Any]] = []
    for (context, init_arm), group in sorted(groups.items()):
        final_top1 = [_safe_float(row["final_top1_post_refill_alive_share"]) for row in group]
        final_effective = [_safe_float(row["final_effective_post_refill_alive_lineages"]) for row in group]
        rows.append(
            {
                "context": context,
                "init_arm": init_arm,
                "runs": len(group),
                "seeds": ",".join(str(row["seed"]) for row in sorted(group, key=lambda item: int(item["seed"]))),
                "top1_collapse_cells": sum(
                    1 for row in group if int(row["top1_dominance_generation"]) >= 0
                ),
                "effective_collapse_cells": sum(
                    1 for row in group if int(row["effective_lineage_collapse_generation"]) >= 0
                ),
                "final_top1_mean": statistics.fmean(final_top1) if final_top1 else 0.0,
                "final_top1_max": max(final_top1) if final_top1 else 0.0,
                "final_effective_mean": statistics.fmean(final_effective) if final_effective else 0.0,
                "final_effective_min": min(final_effective) if final_effective else 0.0,
            }
        )
    return rows


def _scan_totals(cells: list[Mapping[str, Any]]) -> dict[str, Any]:
    fields = [
        "event_rows_read",
        "reproduction_scan_exposure_rows",
        "reproduction_scan_eligible_rows",
        "reproduction_scan_ineligible_rows",
        "reproduction_scan_no_attempt_rows",
        "reproduction_scan_attempt_recorded_rows",
    ]
    out = {field: sum(int(row.get(field, 0)) for row in cells) for field in fields}
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


def _decision(
    *,
    compact_campaign: Mapping[str, Any],
    compact_cells: list[Mapping[str, Any]],
    scan_campaign: Mapping[str, Any],
    scan_cells: list[Mapping[str, Any]],
    expected_compact_runs: int,
    expected_scan_runs: int,
    repeat_parity: Mapping[str, Any] | None,
) -> dict[str, Any]:
    failed_criteria: list[str] = []
    warning_criteria: list[str] = []
    compact_completed = sum(1 for row in compact_cells if str(row.get("status", "")) == "ok")
    compact_failures = list(compact_campaign.get("failures", []) or [])
    scan_completed = sum(1 for row in scan_cells if str(row.get("status", "")) == "ok")
    scan_failures = list(scan_campaign.get("failures", []) or [])
    compact_missing = [str(row["run_id"]) for row in compact_cells if str(row.get("missing_required_metrics", ""))]
    scan_missing = [str(row["run_id"]) for row in scan_cells if str(row.get("missing_required_metrics", ""))]
    top1_collapse_cells = [str(row["run_id"]) for row in compact_cells if int(row["top1_dominance_generation"]) >= 0]
    effective_collapse_cells = [
        str(row["run_id"]) for row in compact_cells if int(row["effective_lineage_collapse_generation"]) >= 0
    ]
    scan_totals = _scan_totals(scan_cells)

    if int(compact_campaign.get("planned_runs", 0)) != expected_compact_runs:
        failed_criteria.append("compact_planned_run_count")
    if compact_completed != expected_compact_runs or compact_failures:
        failed_criteria.append("compact_campaign_completion")
    if len(compact_cells) != expected_compact_runs:
        failed_criteria.append("compact_helper_output_count")
    if compact_missing:
        failed_criteria.append("compact_required_metric_coverage")
    if int(scan_campaign.get("planned_runs", 0)) != expected_scan_runs:
        failed_criteria.append("scan_planned_run_count")
    if scan_completed != expected_scan_runs or scan_failures:
        failed_criteria.append("scan_campaign_completion")
    if len(scan_cells) != expected_scan_runs:
        failed_criteria.append("scan_helper_output_count")
    if int(scan_totals["event_files_present"]) != expected_scan_runs:
        failed_criteria.append("scan_event_file_presence")
    if scan_totals["event_schema_versions_seen"] != ["4"]:
        failed_criteria.append("scan_schema_v4_only")
    if int(scan_totals["reproduction_scan_exposure_rows"]) <= 0:
        failed_criteria.append("scan_exposure_rows")
    if int(scan_totals["reproduction_scan_attempt_recorded_rows"]) <= 0:
        failed_criteria.append("scan_attempt_recorded_rows")
    if scan_totals["attempt_sources"] != ["schema_v4_scan_event"]:
        failed_criteria.append("scan_attempt_source")
    if scan_totals["scan_exposure_metric_statuses"] != ["available_from_schema_v4_events"]:
        failed_criteria.append("scan_metric_coverage")
    if scan_missing:
        failed_criteria.append("scan_required_metric_coverage")
    if top1_collapse_cells:
        warning_criteria.append("top1_collapse")
    if effective_collapse_cells:
        warning_criteria.append("effective_lineage_collapse")
    if repeat_parity and repeat_parity.get("executed") is True and repeat_parity.get("passed") is not True:
        failed_criteria.append("repeat_parity")

    if failed_criteria:
        decision = "OES-LITGATE1-C"
        gate_status = "fail-closed"
        interpretable = False
    elif warning_criteria:
        decision = "OES-LITGATE1-B"
        gate_status = "warn"
        interpretable = False
    else:
        decision = "OES-LITGATE1-A"
        gate_status = "pass"
        interpretable = True

    return {
        "decision": decision,
        "lit1_gate_status": gate_status,
        "failed_criteria": failed_criteria,
        "warning_criteria": warning_criteria,
        "expected_compact_runs": int(expected_compact_runs),
        "completed_compact_runs": compact_completed,
        "expected_scan_runs": int(expected_scan_runs),
        "completed_scan_runs": scan_completed,
        "compact_missing_required_metric_cells": compact_missing,
        "scan_missing_required_metric_cells": scan_missing,
        "top1_collapse_cells": top1_collapse_cells,
        "effective_lineage_collapse_cells": effective_collapse_cells,
        "scan_totals": scan_totals,
        "repeat_parity": dict(repeat_parity or {}),
        "open_ended_outcomes_interpretable": interpretable,
        "interpretation": (
            "Bounded open-ended-search outputs may be interpreted as telemetry for the tested substrate only."
            if interpretable
            else "Do not interpret adaptive, cognition, or open-ended outcome signals under this gate."
        ),
    }


def build_report(
    *,
    compact_campaign_summary_path: Path,
    compact_phase1_output_root: Path,
    scan_campaign_summary_path: Path,
    scan_phase1_output_root: Path,
    expected_compact_runs: int,
    expected_scan_runs: int,
    repeat_parity_path: Path | None = None,
) -> dict[str, Any]:
    compact_campaign = _read_json(compact_campaign_summary_path)
    scan_campaign = _read_json(scan_campaign_summary_path)
    repeat_parity = _read_json(repeat_parity_path) if repeat_parity_path else {}
    compact_cells = _campaign_cells(compact_campaign, compact_phase1_output_root, scan=False)
    scan_cells = _campaign_cells(scan_campaign, scan_phase1_output_root, scan=True)
    decision = _decision(
        compact_campaign=compact_campaign,
        compact_cells=compact_cells,
        scan_campaign=scan_campaign,
        scan_cells=scan_cells,
        expected_compact_runs=expected_compact_runs,
        expected_scan_runs=expected_scan_runs,
        repeat_parity=repeat_parity,
    )
    run_summary = {
        "schema_version": SCHEMA_VERSION,
        "compact_campaign_summary_path": str(compact_campaign_summary_path),
        "compact_phase1_output_root": str(compact_phase1_output_root),
        "scan_campaign_summary_path": str(scan_campaign_summary_path),
        "scan_phase1_output_root": str(scan_phase1_output_root),
        "compact_registry": str(compact_campaign.get("registry", "")),
        "scan_registry": str(scan_campaign.get("registry", "")),
        "compact_planned_runs": int(compact_campaign.get("planned_runs", 0)),
        "scan_planned_runs": int(scan_campaign.get("planned_runs", 0)),
        "compact_total_wall_s": _safe_float(compact_campaign.get("total_wall_s", 0.0)),
        "scan_total_wall_s": _safe_float(scan_campaign.get("total_wall_s", 0.0)),
        "compact_workers": int(compact_campaign.get("workers", 0)),
        "scan_workers": int(scan_campaign.get("workers", 0)),
        "decision": decision["decision"],
        "lit1_gate_status": decision["lit1_gate_status"],
        "no_claims": (
            "OES-LITGATE-1 is a bounded lineage-gated open-ended-search telemetry run. It does not show "
            "intelligence, emergence, consciousness, broad cognition, memory selection, prediction selection, "
            "culture, superintelligence, solved lineage diversity, lineage repair, or a beneficial adaptive "
            "mechanism. It does not add direct cognition reward, lineage rescue, diversity reward, protected "
            "reproduction, archive reseeding, or survival subsidy."
        ),
    }
    lit_gate_summary = {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "compact_context_summary": _compact_context_summary(compact_cells),
    }
    scan_summary = {
        "schema_version": SCHEMA_VERSION,
        "scan_totals": decision["scan_totals"],
        "sampled_cells": scan_cells,
    }
    decision_payload = {
        "schema_version": SCHEMA_VERSION,
        **decision,
        "main_supported_finding": (
            "The LIT-1 gate is observable on this bounded campaign: compact funnel telemetry covered all cells, "
            "and sampled schema-v4 scan telemetry covered selected corners."
            if decision["decision"] == "OES-LITGATE1-A"
            else "The run produced bounded gate evidence, but the LIT-1 gate did not fully pass."
        ),
        "main_unsupported_finding": (
            "No intelligence, emergence, broad cognition, memory selection, prediction selection, solved lineage "
            "diversity, lineage repair, or beneficial adaptive mechanism claim is supported."
        ),
        "recommended_next_step": (
            "If the decision is A, inspect open-ended telemetry under this gate and plan the next finite "
            "measurement slice. If B/C/D, repair only the reported gate failure before further science."
        ),
    }
    return {
        "run_summary": run_summary,
        "lit_gate_summary": lit_gate_summary,
        "compact_funnel_matrix": compact_cells,
        "scan_v4_sample_summary": scan_summary,
        "decision": decision_payload,
    }


def write_outputs(report: Mapping[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "run_summary": output_dir / "oes_litgate1_run_summary.json",
        "lit_gate_summary": output_dir / "oes_litgate1_lit_gate_summary.json",
        "compact_funnel_matrix": output_dir / "oes_litgate1_compact_funnel_matrix.csv",
        "scan_v4_sample_summary": output_dir / "oes_litgate1_scan_v4_sample_summary.json",
        "decision": output_dir / "oes_litgate1_decision.json",
    }
    _write_json(paths["run_summary"], report["run_summary"])
    _write_json(paths["lit_gate_summary"], report["lit_gate_summary"])
    _write_csv(paths["compact_funnel_matrix"], report["compact_funnel_matrix"], COMPACT_FIELDS)
    _write_json(paths["scan_v4_sample_summary"], report["scan_v4_sample_summary"])
    _write_json(paths["decision"], report["decision"])
    return {key: str(value) for key, value in paths.items()}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build OES-LITGATE-1 report artifacts.")
    parser.add_argument("--compact-campaign-summary", type=Path, required=True)
    parser.add_argument("--compact-phase1-output-root", type=Path, required=True)
    parser.add_argument("--scan-campaign-summary", type=Path, required=True)
    parser.add_argument("--scan-phase1-output-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-compact-runs", type=int, default=40)
    parser.add_argument("--expected-scan-runs", type=int, default=6)
    parser.add_argument("--repeat-parity", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        compact_campaign_summary_path=args.compact_campaign_summary,
        compact_phase1_output_root=args.compact_phase1_output_root,
        scan_campaign_summary_path=args.scan_campaign_summary,
        scan_phase1_output_root=args.scan_phase1_output_root,
        expected_compact_runs=int(args.expected_compact_runs),
        expected_scan_runs=int(args.expected_scan_runs),
        repeat_parity_path=args.repeat_parity,
    )
    outputs = write_outputs(report, args.output_dir)
    print(json.dumps({"ok": True, "outputs": outputs}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
