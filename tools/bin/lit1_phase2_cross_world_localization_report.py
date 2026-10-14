#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = 1
RUN_ID_RE = re.compile(
    r"__context_(?P<context>.+?)__init_(?P<init_arm>.+?)__seed(?P<seed>\d+)$"
)

CELL_FIELDS = [
    "run_id",
    "context",
    "init_arm",
    "seed",
    "status",
    "wall_s",
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

GROUP_FIELDS = [
    "context",
    "init_arm",
    "runs",
    "seeds",
    "top1_collapse_cells",
    "effective_collapse_cells",
    "final_top1_mean",
    "final_top1_min",
    "final_top1_max",
    "final_effective_mean",
    "final_effective_min",
    "final_effective_max",
    "min_effective_mean",
    "min_effective_min",
    "max_top1_mean",
    "max_top1_max",
    "event_files_present",
    "attempt_sources",
    "missing_required_metrics",
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
        raise ValueError(f"cannot parse LIT-1 Phase 2 run_id: {run_id}")
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


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _cell_from_result(result: Mapping[str, Any], phase1_root: Path) -> dict[str, Any]:
    run_id = str(result["run_id"])
    run_dir = Path(str(result["run_dir"]))
    parsed = _parse_run_id(run_id)
    phase1_dir = _phase1_dir_for_run(phase1_root, run_dir)
    summary = _read_json(phase1_dir / "lit1_lineage_funnel_phase1_summary.json")
    coverage = _read_json(phase1_dir / "lit1_lineage_funnel_phase1_metric_coverage.json")
    collapse = _read_json(phase1_dir / "lit1_lineage_funnel_phase1_collapse_stage_summary.json")
    generations = list(collapse.get("by_generation", []))
    if not generations:
        raise ValueError(f"empty collapse-stage generation summary for {run_id}")
    final = generations[-1]
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
        "wall_s": _safe_float(result.get("wall_s", 0.0)),
        "generation_count": int(summary.get("generation_count", 0)),
        "lineage_generation_row_count": int(summary.get("lineage_generation_row_count", 0)),
        "top1_dominance_generation": int(collapse.get("top1_dominance_generation", -1)),
        "effective_lineage_collapse_generation": int(
            collapse.get("effective_lineage_collapse_generation", -1)
        ),
        "final_top1_post_refill_alive_share": _safe_float(
            final.get("top1_post_refill_alive_share", 0.0)
        ),
        "final_effective_post_refill_alive_lineages": _safe_float(
            final.get("effective_post_refill_alive_lineages", 0.0)
        ),
        "min_effective_post_refill_alive_lineages": min(effective_values),
        "max_top1_post_refill_alive_share": max(top1_values),
        "event_file_present": bool(coverage.get("event_file_present", False)),
        "attempt_sources": ",".join(str(item) for item in summary.get("attempt_sources", [])),
        "missing_required_metrics": ",".join(str(item) for item in coverage.get("missing_required_metrics", [])),
        "run_dir": str(run_dir),
        "phase1_output_dir": str(phase1_dir),
    }


def _group_rows(cells: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for cell in cells:
        groups[(str(cell["context"]), str(cell["init_arm"]))].append(cell)

    rows: list[dict[str, Any]] = []
    for (context, init_arm), group in sorted(groups.items()):
        final_top1 = [_safe_float(row["final_top1_post_refill_alive_share"]) for row in group]
        final_effective = [
            _safe_float(row["final_effective_post_refill_alive_lineages"]) for row in group
        ]
        min_effective = [_safe_float(row["min_effective_post_refill_alive_lineages"]) for row in group]
        max_top1 = [_safe_float(row["max_top1_post_refill_alive_share"]) for row in group]
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
                "final_top1_mean": _mean(final_top1),
                "final_top1_min": min(final_top1),
                "final_top1_max": max(final_top1),
                "final_effective_mean": _mean(final_effective),
                "final_effective_min": min(final_effective),
                "final_effective_max": max(final_effective),
                "min_effective_mean": _mean(min_effective),
                "min_effective_min": min(min_effective),
                "max_top1_mean": _mean(max_top1),
                "max_top1_max": max(max_top1),
                "event_files_present": sum(1 for row in group if bool(row["event_file_present"])),
                "attempt_sources": ",".join(sorted({str(row["attempt_sources"]) for row in group})),
                "missing_required_metrics": ",".join(
                    sorted({str(row["missing_required_metrics"]) for row in group if row["missing_required_metrics"]})
                ),
            }
        )
    return rows


def _decision(campaign: Mapping[str, Any], cells: list[Mapping[str, Any]], expected_runs: int) -> dict[str, Any]:
    failures = list(campaign.get("failures", []) or [])
    completed = sum(1 for row in cells if str(row.get("status", "")) == "ok")
    event_files_present = sum(1 for row in cells if bool(row.get("event_file_present", False)))
    top1_collapse_cells = sum(1 for row in cells if int(row["top1_dominance_generation"]) >= 0)
    effective_collapse_cells = sum(
        1 for row in cells if int(row["effective_lineage_collapse_generation"]) >= 0
    )
    missing_required_cells = [str(row["run_id"]) for row in cells if str(row["missing_required_metrics"])]

    failed_criteria: list[str] = []
    if int(campaign.get("planned_runs", 0)) != expected_runs:
        failed_criteria.append("planned_run_count")
    if completed != expected_runs or failures:
        failed_criteria.append("campaign_completion")
    if len(cells) != expected_runs:
        failed_criteria.append("phase1_output_count")
    if missing_required_cells:
        failed_criteria.append("required_metric_coverage")

    if failed_criteria:
        decision = "LIT-1-PHASE2-CSV-C"
        interpretation = "CSV-first localization package is incomplete or missing required compact metrics."
    else:
        decision = "LIT-1-PHASE2-CSV-A"
        interpretation = (
            "CSV-first localization package is valid for bounded cross-world collapse-stage screening; "
            "scan exposure remains unresolved because schema-v4 event logging was intentionally disabled."
        )

    return {
        "decision": decision,
        "interpretation": interpretation,
        "failed_criteria": failed_criteria,
        "expected_runs": int(expected_runs),
        "completed_runs": completed,
        "top1_collapse_cells": top1_collapse_cells,
        "effective_collapse_cells": effective_collapse_cells,
        "event_files_present": event_files_present,
        "missing_required_metric_cells": missing_required_cells,
        "recommended_next_step": (
            "Run sampled schema-v4 scan-event follow-up cells on the same matrix corners before any LIT-1 "
            "lineage-integrity A/B/C decision."
        ),
    }


def build_report(
    *,
    campaign_summary_path: Path,
    phase1_output_root: Path,
    expected_runs: int = 24,
) -> dict[str, Any]:
    campaign = _read_json(campaign_summary_path)
    cells = [
        _cell_from_result(result, phase1_output_root)
        for result in sorted(campaign.get("results", []), key=lambda row: int(row.get("index", 0)))
    ]
    groups = _group_rows(cells)
    return {
        "summary": {
            "schema_version": SCHEMA_VERSION,
            "campaign_summary_path": str(campaign_summary_path),
            "phase1_output_root": str(phase1_output_root),
            "registry": str(campaign.get("registry", "")),
            "batch_tag": str(campaign.get("batch_tag", "")),
            "campaign_output_profile": str(campaign.get("campaign_output_profile", "")),
            "planned_runs": int(campaign.get("planned_runs", 0)),
            "result_rows": len(cells),
            "workers": int(campaign.get("workers", 0)),
            "total_wall_s": _safe_float(campaign.get("total_wall_s", 0.0)),
            "mean_child_wall_s": _safe_float(campaign.get("mean_child_wall_s", 0.0)),
            "progress_ui_attach_command": str(
                (campaign.get("progress_ui", {}) or {}).get("attach_command", "")
            ),
            "decision": _decision(campaign, cells, expected_runs),
            "no_claims": (
                "This LIT-1 Phase 2 CSV-first package is telemetry localization only. It does not show "
                "intelligence, emergence, consciousness, broad cognition, memory selection, prediction "
                "selection, solved lineage diversity, or a lineage repair."
            ),
        },
        "cells": cells,
        "groups": groups,
    }


def write_report_outputs(report: Mapping[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "cross_world_localization_run_summary.json",
        "cell_summary": output_dir / "cross_world_localization_cell_summary.csv",
        "context_summary": output_dir / "cross_world_localization_context_summary.csv",
    }
    _write_json(paths["summary"], report["summary"])
    _write_csv(paths["cell_summary"], report["cells"], CELL_FIELDS)
    _write_csv(paths["context_summary"], report["groups"], GROUP_FIELDS)
    return {key: str(value) for key, value in paths.items()}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LIT-1 Phase 2 cross-world localization report tables.")
    parser.add_argument("--campaign-summary", type=Path, required=True)
    parser.add_argument("--phase1-output-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-runs", type=int, default=24)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        campaign_summary_path=args.campaign_summary,
        phase1_output_root=args.phase1_output_root,
        expected_runs=int(args.expected_runs),
    )
    outputs = write_report_outputs(report, args.output_dir)
    print(json.dumps({"ok": True, "outputs": outputs}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
