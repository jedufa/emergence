#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)


SCHEMA_VERSION = 1
KNOWN_EVENT_SCHEMA_VERSIONS = {3, 4}
SCAN_EVENT_TYPES = {
    "reproduction_scan_exposure",
    "reproduction_scan_eligible",
    "reproduction_scan_ineligible",
    "reproduction_scan_no_attempt",
    "reproduction_scan_attempt_recorded",
}

BY_GENERATION_FIELDS = [
    "schema_version",
    "generation",
    "lineage_id",
    "alive_count",
    "alive_share",
    "eligible_parent_count",
    "eligible_parent_share",
    "selected_parent_count",
    "selected_parent_share",
    "scan_exposure_count",
    "scan_exposure_share",
    "scan_eligible_count",
    "scan_ineligible_count",
    "scan_no_attempt_count",
    "attempt_count",
    "attempt_share",
    "attempt_source",
    "successful_birth_count",
    "successful_birth_share",
    "failed_birth_count",
    "failed_birth_reason_counts",
    "offspring_count",
    "offspring_share",
    "post_refill_alive_count",
    "post_refill_alive_share",
    "child_placement_distance_mean",
    "child_patch_context_summary",
    "first_zero_selected_parent_generation",
    "first_zero_successful_birth_generation",
    "top1_dominance_generation",
    "effective_lineage_collapse_generation",
    "eligible_parent_share_over_alive_share",
    "selected_parent_share_over_eligible_parent_share",
    "scan_share_over_selected_parent_share",
    "attempt_share_over_scan_share",
    "success_share_over_attempt_share",
    "offspring_share_over_selected_parent_share",
    "post_refill_share_over_pre_refill_alive_share",
    "event_schema_versions_seen",
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if float(denominator) > 0.0 else 0.0


def _share(count: int, total: int) -> float:
    return _ratio(float(count), float(total))


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"missing generation CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BY_GENERATION_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in BY_GENERATION_FIELDS})


def _lineage_from_event(row: Mapping[str, Any]) -> str:
    for field in (
        "parent_biological_lineage_id",
        "parent_lineage_id",
        "biological_lineage_id",
        "lineage_id",
        "child_biological_lineage_id",
        "child_lineage_id",
    ):
        value = str(row.get(field, "") or "")
        if value:
            return value
    return ""


def _read_events(path: Path | None, *, ignore_unsupported_event_schema: bool = False) -> tuple[dict[tuple[int, str], Counter[str]], dict[str, Any]]:
    counters: dict[tuple[int, str], Counter[str]] = defaultdict(Counter)
    summary: dict[str, Any] = {
        "event_file_present": bool(path and path.exists()),
        "event_rows_read": 0,
        "event_rows_ignored_missing_lineage": 0,
        "event_type_counts": {},
        "event_schema_versions_seen": [],
        "unsupported_event_schema_versions": [],
        "path2_roz_event_rows": 0,
        "path2_roz_scan_exposure_rows": 0,
        "path2_roz_scan_attempt_recorded_rows": 0,
    }
    if path is None or not path.exists():
        return counters, summary

    event_types: Counter[str] = Counter()
    versions: Counter[int] = Counter()
    unsupported: Counter[int] = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON event row at {path}:{line_no}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"event row must be an object at {path}:{line_no}")
            version = _safe_int(row.get("event_schema_version", 3), 3)
            versions[version] += 1
            if version not in KNOWN_EVENT_SCHEMA_VERSIONS:
                unsupported[version] += 1
                if not ignore_unsupported_event_schema:
                    raise ValueError(f"unsupported event_schema_version {version} at {path}:{line_no}")
                continue
            event_type = str(row.get("event_type", "") or row.get("event", "") or "")
            if not event_type:
                continue
            generation = _safe_int(row.get("generation", 0), 0)
            lineage_id = _lineage_from_event(row)
            event_types[event_type] += 1
            summary["event_rows_read"] += 1
            if _safe_int(row.get("path2_roz_reporting_enable", 0), 0) > 0:
                summary["path2_roz_event_rows"] += 1
                if event_type == "reproduction_scan_exposure":
                    summary["path2_roz_scan_exposure_rows"] += 1
                if event_type == "reproduction_scan_attempt_recorded":
                    summary["path2_roz_scan_attempt_recorded_rows"] += 1
            if not lineage_id:
                summary["event_rows_ignored_missing_lineage"] += 1
                continue
            key = (generation, lineage_id)
            if event_type in SCAN_EVENT_TYPES:
                counters[key][event_type] += 1
            if event_type == "birth_success":
                counters[key]["birth_success_event"] += 1
            if event_type.startswith("birth_fail"):
                counters[key]["birth_fail_event"] += 1

    summary["event_type_counts"] = dict(sorted(event_types.items()))
    summary["event_schema_versions_seen"] = sorted(str(key) for key in versions)
    summary["unsupported_event_schema_versions"] = sorted(str(key) for key in unsupported)
    return counters, summary


def _effective_lineages(rows: Iterable[Mapping[str, Any]], share_field: str) -> float:
    shares = [_safe_float(row.get(share_field, 0.0), 0.0) for row in rows]
    entropy = -sum(share * math.log(share) for share in shares if share > 0.0)
    return math.exp(entropy) if entropy > 0.0 else 0.0


def _top1_share(rows: Iterable[Mapping[str, Any]], share_field: str) -> float:
    return max((_safe_float(row.get(share_field, 0.0), 0.0) for row in rows), default=0.0)


def _derive_collapse_generations(
    csv_rows_by_generation: Mapping[int, list[Mapping[str, Any]]],
    *,
    top1_threshold: float,
    effective_collapse_threshold: float,
) -> tuple[int, int, list[dict[str, Any]]]:
    top1_generation = -1
    effective_generation = -1
    generation_rows: list[dict[str, Any]] = []
    for generation, rows in sorted(csv_rows_by_generation.items()):
        top1 = _top1_share(rows, "post_refill_alive_share")
        effective = _effective_lineages(rows, "post_refill_alive_share")
        alive_total = sum(_safe_int(row.get("post_refill_alive_count", 0), 0) for row in rows)
        if alive_total > 0 and top1_generation < 0 and top1 >= top1_threshold:
            top1_generation = int(generation)
        if alive_total > 0 and effective_generation < 0 and effective <= effective_collapse_threshold:
            effective_generation = int(generation)
        generation_rows.append(
            {
                "generation": int(generation),
                "post_refill_alive_total": int(alive_total),
                "top1_post_refill_alive_share": top1,
                "effective_post_refill_alive_lineages": effective,
            }
        )
    return top1_generation, effective_generation, generation_rows


def _child_context(row: Mapping[str, Any]) -> str:
    payload = {
        "child_patch_id_or_coordinates": str(row.get("child_patch_id_or_coordinates", "") or ""),
        "child_basin_id": str(row.get("child_basin_id", "") or ""),
        "food_child_site_mean": _safe_float(row.get("food_child_site_mean", 0.0), 0.0),
        "waste_child_site_mean": _safe_float(row.get("waste_child_site_mean", 0.0), 0.0),
        "scent_child_site_mean": _safe_float(row.get("scent_child_site_mean", 0.0), 0.0),
        "shelter_cover_child_site_mean": _safe_float(row.get("shelter_cover_child_site_mean", 0.0), 0.0),
        "predator_hazard_exposure_child_site_mean": _safe_float(
            row.get("predator_hazard_exposure_child_site_mean", 0.0),
            0.0,
        ),
    }
    return _stable_json(payload)


def build_phase1_outputs(
    *,
    generation_csv: Path,
    events_jsonl: Path | None = None,
    top1_threshold: float = 0.8,
    effective_collapse_threshold: float = 1.5,
    ignore_unsupported_event_schema: bool = False,
) -> dict[str, Any]:
    csv_rows = _read_csv(generation_csv)
    csv_rows_by_generation: dict[int, list[dict[str, str]]] = defaultdict(list)
    lineages: set[tuple[int, str]] = set()
    for row in csv_rows:
        generation = _safe_int(row.get("generation", 0), 0)
        lineage_id = str(row.get("lineage_id", "") or "")
        csv_rows_by_generation[generation].append(row)
        lineages.add((generation, lineage_id))

    event_counters, event_summary = _read_events(
        events_jsonl,
        ignore_unsupported_event_schema=ignore_unsupported_event_schema,
    )
    lineages.update(event_counters)
    top1_gen, effective_gen, collapse_by_generation = _derive_collapse_generations(
        csv_rows_by_generation,
        top1_threshold=top1_threshold,
        effective_collapse_threshold=effective_collapse_threshold,
    )

    csv_row_by_key = {
        (_safe_int(row.get("generation", 0), 0), str(row.get("lineage_id", "") or "")): row for row in csv_rows
    }
    scan_totals: Counter[int] = Counter()
    attempt_totals: Counter[int] = Counter()
    csv_attempt_totals: Counter[int] = Counter()
    for (generation, _lineage_id), counts in event_counters.items():
        scan_totals[generation] += counts["reproduction_scan_exposure"]
        attempt_totals[generation] += counts["reproduction_scan_attempt_recorded"]
    for row in csv_rows:
        csv_attempt_totals[_safe_int(row.get("generation", 0), 0)] += _safe_int(row.get("birth_attempts", 0), 0)

    by_generation: list[dict[str, Any]] = []
    for generation, lineage_id in sorted(lineages):
        source = csv_row_by_key.get((generation, lineage_id), {})
        counts = event_counters.get((generation, lineage_id), Counter())
        scan_exposure_count = int(counts["reproduction_scan_exposure"])
        scan_exposure_share = _share(scan_exposure_count, int(scan_totals[generation]))
        event_attempt_count = int(counts["reproduction_scan_attempt_recorded"])
        csv_attempt_count = _safe_int(source.get("birth_attempts", 0), 0)
        if event_summary["event_file_present"] and int(attempt_totals[generation]) > 0:
            attempt_count = event_attempt_count
            attempt_share = _share(event_attempt_count, int(attempt_totals[generation]))
            attempt_source = "schema_v4_scan_event"
        else:
            attempt_count = csv_attempt_count
            attempt_share = _share(csv_attempt_count, int(csv_attempt_totals[generation]))
            attempt_source = "generation_csv_birth_attempts"
        selected_share = _safe_float(source.get("selected_parent_share", 0.0), 0.0)
        offspring_share = _safe_float(source.get("offspring_share", 0.0), 0.0)
        row = {
            "schema_version": SCHEMA_VERSION,
            "generation": int(generation),
            "lineage_id": lineage_id,
            "alive_count": _safe_int(source.get("alive_count_before_selection", 0), 0),
            "alive_share": _safe_float(source.get("alive_share_before_selection", 0.0), 0.0),
            "eligible_parent_count": _safe_int(source.get("eligible_parent_count", 0), 0),
            "eligible_parent_share": _safe_float(source.get("eligible_parent_share", 0.0), 0.0),
            "selected_parent_count": _safe_int(source.get("selected_parent_count", 0), 0),
            "selected_parent_share": selected_share,
            "scan_exposure_count": scan_exposure_count,
            "scan_exposure_share": scan_exposure_share,
            "scan_eligible_count": int(counts["reproduction_scan_eligible"]),
            "scan_ineligible_count": int(counts["reproduction_scan_ineligible"]),
            "scan_no_attempt_count": int(counts["reproduction_scan_no_attempt"]),
            "attempt_count": int(attempt_count),
            "attempt_share": attempt_share,
            "attempt_source": attempt_source,
            "successful_birth_count": _safe_int(source.get("successful_births", source.get("offspring_produced_count", 0)), 0),
            "successful_birth_share": offspring_share,
            "failed_birth_count": _safe_int(source.get("failed_births", 0), 0),
            "failed_birth_reason_counts": str(source.get("failed_births_by_reason_json", "{}") or "{}"),
            "offspring_count": _safe_int(source.get("offspring_produced_count", 0), 0),
            "offspring_share": offspring_share,
            "post_refill_alive_count": _safe_int(source.get("post_refill_alive_count", 0), 0),
            "post_refill_alive_share": _safe_float(source.get("post_refill_alive_share", 0.0), 0.0),
            "child_placement_distance_mean": _safe_float(source.get("parent_child_distance_mean", 0.0), 0.0),
            "child_patch_context_summary": _child_context(source),
            "first_zero_selected_parent_generation": _safe_int(
                source.get("first_generation_zero_selected_parents", -1),
                -1,
            ),
            "first_zero_successful_birth_generation": _safe_int(
                source.get("first_generation_zero_successful_births", -1),
                -1,
            ),
            "top1_dominance_generation": top1_gen,
            "effective_lineage_collapse_generation": effective_gen,
            "eligible_parent_share_over_alive_share": _safe_float(
                source.get("eligible_parent_share_over_alive_share", 0.0),
                0.0,
            ),
            "selected_parent_share_over_eligible_parent_share": _safe_float(
                source.get("selected_parent_share_over_eligible_parent_share", 0.0),
                0.0,
            ),
            "scan_share_over_selected_parent_share": _ratio(scan_exposure_share, selected_share),
            "attempt_share_over_scan_share": _ratio(attempt_share, scan_exposure_share),
            "success_share_over_attempt_share": _ratio(offspring_share, attempt_share),
            "offspring_share_over_selected_parent_share": _safe_float(
                source.get("offspring_share_over_selected_parent_share", _ratio(offspring_share, selected_share)),
                0.0,
            ),
            "post_refill_share_over_pre_refill_alive_share": _safe_float(
                source.get("post_refill_share_over_pre_refill_alive_share", 0.0),
                0.0,
            ),
            "event_schema_versions_seen": ",".join(event_summary["event_schema_versions_seen"]),
        }
        by_generation.append(row)

    coverage = _metric_coverage(
        by_generation,
        csv_headers=list(csv_rows[0].keys()) if csv_rows else [],
        event_summary=event_summary,
    )
    collapse_summary = {
        "schema_version": SCHEMA_VERSION,
        "top1_threshold": float(top1_threshold),
        "effective_collapse_threshold": float(effective_collapse_threshold),
        "top1_dominance_generation": top1_gen,
        "effective_lineage_collapse_generation": effective_gen,
        "by_generation": collapse_by_generation,
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generation_csv": str(generation_csv),
        "events_jsonl": str(events_jsonl) if events_jsonl else "",
        "generation_count": len(csv_rows_by_generation),
        "lineage_generation_row_count": len(by_generation),
        "event_summary": event_summary,
        "top1_dominance_generation": top1_gen,
        "effective_lineage_collapse_generation": effective_gen,
        "attempt_sources": sorted({str(row["attempt_source"]) for row in by_generation}),
        "mechanics_changes": "none_offline_reporting_helper_only",
        "no_claims": (
            "LIT-1 Phase 1 audit output is offline reporting telemetry only. It does not show intelligence, "
            "emergence, consciousness, broad cognition, memory selection, prediction selection, or solved lineage diversity."
        ),
    }
    return {
        "by_generation": by_generation,
        "summary": summary,
        "metric_coverage": coverage,
        "collapse_stage_summary": collapse_summary,
    }


def _metric_coverage(
    rows: list[Mapping[str, Any]],
    *,
    csv_headers: list[str],
    event_summary: Mapping[str, Any],
) -> dict[str, Any]:
    row_fields = set(rows[0]) if rows else set()
    event_types = set((event_summary.get("event_type_counts") or {}).keys())
    required = {
        "alive_count": "available" if "alive_count" in row_fields else "missing",
        "alive_share": "available" if "alive_share" in row_fields else "missing",
        "eligible_parent_count": "available" if "eligible_parent_count" in row_fields else "missing",
        "eligible_parent_share": "available" if "eligible_parent_share" in row_fields else "missing",
        "selected_parent_count": "available" if "selected_parent_count" in row_fields else "missing",
        "selected_parent_share": "available" if "selected_parent_share" in row_fields else "missing",
        "scan_exposure_count": "available_from_schema_v4_events"
        if "reproduction_scan_exposure" in event_types
        else "present_zero_or_events_missing",
        "scan_exposure_share": "available_from_schema_v4_events"
        if "reproduction_scan_exposure" in event_types
        else "present_zero_or_events_missing",
        "attempt_count": "available",
        "attempt_share": "available",
        "successful_birth_count": "available",
        "successful_birth_share": "available",
        "failed_birth_count": "available",
        "failed_birth_reason_counts": "available",
        "offspring_count": "available",
        "offspring_share": "available",
        "post_refill_alive_count": "available",
        "post_refill_alive_share": "available",
        "child_placement_distance_mean": "available_alias_parent_child_distance_mean"
        if "parent_child_distance_mean" in csv_headers
        else "present_zero_or_csv_field_missing",
        "child_patch_context_summary": "available_compact_summary",
        "first_zero_selected_parent_generation": "available",
        "first_zero_successful_birth_generation": "available",
        "top1_dominance_generation": "derived",
        "effective_lineage_collapse_generation": "derived",
    }
    missing = [key for key, status in required.items() if status == "missing"]
    return {
        "schema_version": SCHEMA_VERSION,
        "required_metric_status": required,
        "missing_required_metrics": missing,
        "event_file_present": bool(event_summary.get("event_file_present", False)),
        "event_schema_versions_seen": list(event_summary.get("event_schema_versions_seen", [])),
        "event_type_counts": dict(event_summary.get("event_type_counts", {}) or {}),
        "notes": [
            "Schema-v3 selected-parent exposure is accounting exposure, not live scan exposure.",
            "Schema-v4 reproduction_scan_* rows are live scan exposure telemetry, not action-offered/action-taken claims.",
            "Path 2 ROZ counters are opportunity context only and are not lineage repair evidence by themselves.",
        ],
    }


def write_outputs(outputs: Mapping[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "lit1_lineage_funnel_phase1_summary.json",
        "by_generation": output_dir / "lit1_lineage_funnel_phase1_by_generation.csv",
        "metric_coverage": output_dir / "lit1_lineage_funnel_phase1_metric_coverage.json",
        "collapse_stage_summary": output_dir / "lit1_lineage_funnel_phase1_collapse_stage_summary.json",
    }
    _write_json(paths["summary"], outputs["summary"])
    _write_csv(paths["by_generation"], list(outputs["by_generation"]))
    _write_json(paths["metric_coverage"], outputs["metric_coverage"])
    _write_json(paths["collapse_stage_summary"], outputs["collapse_stage_summary"])
    return {key: str(value) for key, value in paths.items()}


def _default_generation_csv(run_dir: Path | None, generation_csv: Path | None) -> Path:
    if generation_csv is not None:
        return generation_csv
    if run_dir is None:
        raise ValueError("either --run-dir or --generation-csv is required")
    return run_dir / "metrics" / "lineage_reproduction_funnel_generation_summary.csv"


def _default_events_jsonl(run_dir: Path | None, events_jsonl: Path | None) -> Path | None:
    if events_jsonl is not None:
        return events_jsonl
    if run_dir is None:
        return None
    path = run_dir / "metrics" / "lineage_reproduction_funnel_events.jsonl"
    return path if path.exists() else None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LIT-1 Phase 1 lineage funnel audit outputs.")
    parser.add_argument("--run-dir", type=Path, help="Run directory containing metrics/ lineage funnel files.")
    parser.add_argument("--generation-csv", type=Path, help="Path to lineage_reproduction_funnel_generation_summary.csv.")
    parser.add_argument("--events-jsonl", type=Path, help="Optional path to lineage_reproduction_funnel_events.jsonl.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for Phase 1 audit outputs.")
    parser.add_argument("--top1-threshold", type=float, default=0.8)
    parser.add_argument("--effective-collapse-threshold", type=float, default=1.5)
    parser.add_argument("--ignore-unsupported-event-schema", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    generation_csv = _default_generation_csv(args.run_dir, args.generation_csv)
    events_jsonl = _default_events_jsonl(args.run_dir, args.events_jsonl)
    outputs = build_phase1_outputs(
        generation_csv=generation_csv,
        events_jsonl=events_jsonl,
        top1_threshold=float(args.top1_threshold),
        effective_collapse_threshold=float(args.effective_collapse_threshold),
        ignore_unsupported_event_schema=bool(args.ignore_unsupported_event_schema),
    )
    written = write_outputs(outputs, args.output_dir)
    print(json.dumps({"ok": True, "outputs": written}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
