#!/usr/bin/env python3
"""Summarize Patch/Jungle V0 Campaign C pressure-tuning batches.

This is reporting-only infrastructure. It reads completed registry batch
directories and writes deterministic JSON, CSV, and Markdown summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


KEY_METRICS: tuple[str, ...] = (
    "alive",
    "deaths",
    "cohort_survival_rate",
    "successful_births",
    "failed_births",
    "patch_food_amount_mean",
    "patch_food_depleted_frac",
    "patch_regrowth_effective_total",
    "patch_regrowth_suppressed_total",
    "patch_regrowth_multiplier_mean",
    "patch_waste_mean",
    "patch_waste_p90",
    "patch_density_mean",
    "patch_density_p90",
    "birth_patch_waste_mean",
    "jungle_predator_occupied_risk_mean",
    "jungle_predator_events",
    "jungle_predator_damage_total",
    "jungle_predator_damage_raw_total",
    "jungle_predator_damage_effective_total",
    "jungle_predator_damage_prevented_total",
    "jungle_predator_event_damage_multiplier_mean",
    "predator_event_patch_scent_mean",
    "predator_event_patch_waste_mean",
    "predator_event_patch_density_mean",
    "predator_event_patch_cover_mean",
    "predator_event_patch_shelter_mean",
    "occupied_patch_cover_mean",
    "occupied_patch_shelter_mean",
    "patch_recovery_activity_mean",
    "jungle_recovery_regrowth_boost_total",
    "patch_recovery_added_regrowth_total",
    "patch_regrowth_after_waste_total",
    "patch_regrowth_after_recovery_total",
    "lineage_active_count",
    "lineage_alive_share_entropy",
    "lineage_effective_alive_lineages",
    "lineage_top1_birth_share",
    "lineage_branch_candidate_count",
    "lineage_patch_waste_mean_std",
    "lineage_patch_shelter_mean_std",
    "lineage_predator_damage_mean_std",
)

CRITICAL_METRICS: tuple[str, ...] = (
    "alive",
    "cohort_survival_rate",
    "successful_births",
    "patch_food_amount_mean",
    "patch_regrowth_suppressed_total",
    "jungle_predator_events",
    "jungle_predator_damage_total",
    "jungle_predator_damage_prevented_total",
    "patch_recovery_activity_mean",
    "patch_recovery_added_regrowth_total",
    "lineage_active_count",
    "lineage_alive_share_entropy",
)

SWEEP_LABELS: dict[str, str] = {
    "patch_jungle_v0_campaign_c1_predator": "C1 predator",
    "patch_jungle_v0_campaign_c2_waste": "C2 waste",
    "patch_jungle_v0_campaign_c3_terrain": "C3 terrain",
    "patch_jungle_v0_campaign_c4_recovery": "C4 recovery",
}

DECISION_METRICS: dict[str, tuple[str, ...]] = {
    "patch_jungle_v0_campaign_c1_predator": (
        "alive",
        "successful_births",
        "jungle_predator_events",
        "jungle_predator_damage_total",
        "lineage_alive_share_entropy",
    ),
    "patch_jungle_v0_campaign_c2_waste": (
        "alive",
        "successful_births",
        "patch_food_amount_mean",
        "patch_regrowth_suppressed_total",
        "patch_waste_mean",
    ),
    "patch_jungle_v0_campaign_c3_terrain": (
        "alive",
        "successful_births",
        "jungle_predator_damage_effective_total",
        "jungle_predator_damage_prevented_total",
        "lineage_alive_share_entropy",
    ),
    "patch_jungle_v0_campaign_c4_recovery": (
        "alive",
        "successful_births",
        "patch_recovery_activity_mean",
        "patch_recovery_added_regrowth_total",
        "patch_regrowth_suppressed_total",
        "jungle_predator_damage_total",
    ),
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_import_path() -> None:
    root = str(_project_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Patch/Jungle V0 Campaign C batches")
    parser.add_argument("--batch-dir", action="append", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--deltas-csv", required=True)
    parser.add_argument("--report-md", default="")
    parser.add_argument("--tail-n", type=int, default=1)
    parser.add_argument("--campaign-name", default="patch_jungle_v0_campaign_c_pressure_tuning")
    parser.add_argument("--output-root", default="/tmp/emergence_patch_jungle_campaign_c")
    parser.add_argument("--preflight-json", default="")
    parser.add_argument("--repeat-parity-json", default="")
    return parser.parse_args(argv)


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _parse_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _mean(values: Iterable[float]) -> float | None:
    items = list(values)
    if not items:
        return None
    return float(sum(items) / len(items))


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return ""
    parsed = _parse_float(value)
    if parsed is None:
        return str(value)
    return f"{parsed:.{digits}f}"


def _last_metrics_row(path: Path) -> dict[str, str] | None:
    rows = _load_csv(path)
    return rows[-1] if rows else None


def _json_list_non_empty(value: str | None) -> bool:
    if value is None or not str(value).strip():
        return False
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return False
    return bool(parsed)


def _scenario_flag_mismatches(scenario_id: str, row: Mapping[str, str]) -> list[dict[str, Any]]:
    from experiments.patch_jungle_eval import PATCH_JUNGLE_FLAGS, scenario_overrides_dict

    expected = scenario_overrides_dict(scenario_id)
    mismatches: list[dict[str, Any]] = []
    for flag in PATCH_JUNGLE_FLAGS:
        if flag not in row:
            continue
        actual = int(float(row[flag]))
        wanted = 1 if bool(expected[flag]) else 0
        if actual != wanted:
            mismatches.append({"scenario": scenario_id, "flag": flag, "expected": wanted, "actual": actual})
    return mismatches


def _summarize_run(
    *,
    batch_dir: Path,
    campaign: str,
    row: Mapping[str, str],
    metric_names: Sequence[str],
    tail_n: int,
) -> dict[str, Any]:
    from experiments.patch_jungle_eval import summarize_metrics_csv

    run_dir = Path(str(row.get("run_dir") or ""))
    metrics_path = run_dir / "metrics.csv"
    scenario = str(row.get("scenario") or "")
    variant = str(row.get("variant") or "default")
    seed = int(row.get("seed") or 0)
    run: dict[str, Any] = {
        "batch_dir": str(batch_dir),
        "campaign": campaign,
        "variant": variant,
        "scenario": scenario,
        "seed": seed,
        "run_id": str(row.get("run_id") or ""),
        "runtime_run_id": str(row.get("runtime_run_id") or ""),
        "run_dir": str(run_dir),
        "metrics_exists": metrics_path.exists(),
        "metrics_rows": 0,
        "metrics": {},
        "missing_metrics": list(metric_names),
        "critical_nonfinite": [],
        "scenario_flag_mismatches": [],
        "lineage_strategy_json_present": False,
        "lineage_branch_candidates_json_present": False,
        "lineage_strategy_json_nonempty": False,
        "lineage_branch_candidates_json_nonempty": False,
    }
    if not metrics_path.exists():
        return run

    summary = summarize_metrics_csv(metrics_path, metric_names, tail_n=tail_n)
    run["metrics_rows"] = int(summary.get("rows") or 0)
    run["metrics"] = dict(summary.get("metrics", {}) or {})
    run["missing_metrics"] = list(summary.get("missing_metrics", []) or [])
    final_row = _last_metrics_row(metrics_path)
    if final_row is None:
        return run

    run["critical_nonfinite"] = [
        {"metric": metric, "value": final_row.get(metric)}
        for metric in CRITICAL_METRICS
        if metric in final_row and _parse_float(final_row.get(metric)) is None
    ]
    run["scenario_flag_mismatches"] = _scenario_flag_mismatches(scenario, final_row) if scenario else []
    strategy_json = final_row.get("lineage_strategy_descriptors_json")
    branches_json = final_row.get("lineage_branch_candidates_json")
    run["lineage_strategy_json_present"] = strategy_json is not None and strategy_json != ""
    run["lineage_branch_candidates_json_present"] = branches_json is not None and branches_json != ""
    run["lineage_strategy_json_nonempty"] = _json_list_non_empty(strategy_json)
    run["lineage_branch_candidates_json_nonempty"] = _json_list_non_empty(branches_json)
    return run


def _variant_order(runs: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], list[str]]:
    order: dict[tuple[str, str], list[str]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()
    for run in runs:
        key = (str(run["campaign"]), str(run["scenario"]))
        variant = str(run["variant"])
        seen_key = (*key, variant)
        if seen_key not in seen:
            seen.add(seen_key)
            order[key].append(variant)
    return dict(order)


def _build_variant_summaries(runs: Sequence[Mapping[str, Any]], metric_names: Sequence[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[(str(run["campaign"]), str(run["variant"]), str(run["scenario"]))].append(run)

    summaries: list[dict[str, Any]] = []
    for (campaign, variant, scenario), entries in sorted(grouped.items()):
        metrics: dict[str, float] = {}
        for metric in metric_names:
            values = [
                parsed
                for entry in entries
                for parsed in [_parse_float((entry.get("metrics") or {}).get(metric))]
                if parsed is not None
            ]
            mean = _mean(values)
            if mean is not None:
                metrics[metric] = mean
        summaries.append(
            {
                "campaign": campaign,
                "variant": variant,
                "scenario": scenario,
                "run_count": len(entries),
                "seeds": sorted(int(entry["seed"]) for entry in entries),
                "metrics": metrics,
                "missing_metrics": sorted(
                    {metric for entry in entries for metric in list(entry.get("missing_metrics", []) or [])}
                ),
            }
        )
    return summaries


def _build_paired_deltas(runs: Sequence[Mapping[str, Any]], metric_names: Sequence[str]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str, int], Mapping[str, Any]] = {}
    for run in runs:
        by_key[(str(run["campaign"]), str(run["scenario"]), str(run["variant"]), int(run["seed"]))] = run

    deltas: list[dict[str, Any]] = []
    for (campaign, scenario), variants in sorted(_variant_order(runs).items()):
        if not variants:
            continue
        baseline = variants[0]
        baseline_seeds = {seed for c, s, v, seed in by_key if c == campaign and s == scenario and v == baseline}
        for variant in variants[1:]:
            variant_seeds = {seed for c, s, v, seed in by_key if c == campaign and s == scenario and v == variant}
            paired_seeds = sorted(baseline_seeds & variant_seeds)
            metric_deltas: dict[str, float] = {}
            for metric in metric_names:
                values: list[float] = []
                for seed in paired_seeds:
                    base = _parse_float((by_key[(campaign, scenario, baseline, seed)].get("metrics") or {}).get(metric))
                    other = _parse_float((by_key[(campaign, scenario, variant, seed)].get("metrics") or {}).get(metric))
                    if base is not None and other is not None:
                        values.append(float(other - base))
                mean = _mean(values)
                if mean is not None:
                    metric_deltas[metric] = mean
            deltas.append(
                {
                    "campaign": campaign,
                    "scenario": scenario,
                    "baseline_variant": baseline,
                    "variant": variant,
                    "paired_seed_count": len(paired_seeds),
                    "paired_seeds": paired_seeds,
                    "deltas": metric_deltas,
                }
            )
    return deltas


def _recommend_band(campaign: str, summaries: Sequence[Mapping[str, Any]]) -> str:
    by_variant: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in summaries:
        if item["campaign"] == campaign:
            by_variant[str(item["variant"])].append(item)
    if not by_variant:
        return "no acceptable band"

    scored: list[tuple[float, str]] = []
    for variant, entries in by_variant.items():
        alive = _mean(_parse_float((e.get("metrics") or {}).get("alive")) or 0.0 for e in entries) or 0.0
        births = _mean(_parse_float((e.get("metrics") or {}).get("successful_births")) or 0.0 for e in entries) or 0.0
        pred_events = _mean(_parse_float((e.get("metrics") or {}).get("jungle_predator_events")) or 0.0 for e in entries) or 0.0
        pred_damage = _mean(_parse_float((e.get("metrics") or {}).get("jungle_predator_damage_total")) or 0.0 for e in entries) or 0.0
        suppressed = _mean(_parse_float((e.get("metrics") or {}).get("patch_regrowth_suppressed_total")) or 0.0 for e in entries) or 0.0
        prevented = _mean(_parse_float((e.get("metrics") or {}).get("jungle_predator_damage_prevented_total")) or 0.0 for e in entries) or 0.0
        recovery = _mean(_parse_float((e.get("metrics") or {}).get("patch_recovery_added_regrowth_total")) or 0.0 for e in entries) or 0.0
        if alive <= 0.0 or births <= 0.0:
            continue
        score = alive + births
        if "predator" in campaign:
            score += min(pred_events, 80.0) * 0.2 + min(pred_damage, 25.0) * 0.5
            if pred_damage <= 0.0:
                score -= 100.0
        elif "waste" in campaign:
            score += min(suppressed, 10.0) * 2.0
            if suppressed <= 0.0:
                score -= 100.0
        elif "terrain" in campaign:
            score += min(prevented, 6.0) * 4.0
            if prevented <= 0.0 or pred_damage <= 0.0:
                score -= 100.0
        elif "recovery" in campaign:
            score += min(recovery, 3.0) * 8.0
            if recovery <= 0.0 and not variant.endswith("_off"):
                score -= 100.0
        scored.append((score, variant))
    if not scored:
        return "no acceptable band"
    return sorted(scored, reverse=True)[0][1]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str], metric_key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for item in rows:
            row = {field: item.get(field, "") for field in fields}
            metric_values = item.get(metric_key) or {}
            for metric in KEY_METRICS:
                if metric in fields:
                    row[metric] = metric_values.get(metric, "")
            if "seeds" in row and isinstance(row["seeds"], list):
                row["seeds"] = " ".join(str(seed) for seed in row["seeds"])
            if "paired_seeds" in row and isinstance(row["paired_seeds"], list):
                row["paired_seeds"] = " ".join(str(seed) for seed in row["paired_seeds"])
            writer.writerow(row)


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(value) for value in row) + " |")
    return out


def _write_markdown_report(
    path: Path,
    *,
    campaign_name: str,
    output_root: str,
    batches: Sequence[Mapping[str, Any]],
    integrity: Mapping[str, Any],
    summaries: Sequence[Mapping[str, Any]],
    deltas: Sequence[Mapping[str, Any]],
    preflight: Mapping[str, Any],
    repeat_parity: Mapping[str, Any],
    recommendations: Mapping[str, str],
) -> None:
    lines: list[str] = [
        "# Patch/Jungle V0 Campaign C Pressure Tuning Report",
        "",
        f"Campaign: `{campaign_name}`",
        "",
        f"Output root: `{output_root}`",
        "",
        "This report covers bounded pressure tuning with existing Patch/Jungle V0 mechanics. It does not claim intelligence, emergence, cognition selection, memory selection, fear selection, prediction selection, or durable lineage-diversity improvement.",
        "",
        "## Settings",
        "",
        "| Setting | Value |",
        "| --- | --- |",
        "| Seeds | `101, 202, 303, 404, 505, 606, 707, 808, 909, 1001` |",
        "| Generations | 10 |",
        "| Steps per generation | 80 |",
        "| Population | 32 |",
        "| Grid size | 12 |",
        "| Device | CPU |",
        "",
        "## Preflight",
        "",
    ]
    preflight_rows = [
        (item.get("command", ""), item.get("returncode", ""), item.get("status", ""))
        for item in preflight.get("commands", [])
    ]
    lines.extend(_markdown_table(("Command", "Return code", "Status"), preflight_rows or (("not recorded", "", ""),)))
    lines.extend(["", "## Repeat Parity", ""])
    parity_rows = [
        (
            item.get("scenario", ""),
            item.get("seed", ""),
            item.get("metrics_hash_match", ""),
            item.get("config_hash_match", ""),
            item.get("run_config_hash_match", ""),
            item.get("final_row_match", ""),
        )
        for item in repeat_parity.get("checks", [])
    ]
    lines.extend(
        _markdown_table(
            ("Scenario", "Seed", "Metrics hash", "Config hash", "Run config hash", "Final row"),
            parity_rows or (("not recorded", "", "", "", "", ""),),
        )
    )
    lines.extend(["", "## Completion", ""])
    completion_rows = [
        (
            SWEEP_LABELS.get(str(batch.get("campaign")), str(batch.get("campaign"))),
            batch.get("planned_runs", ""),
            batch.get("completed_runs", ""),
            batch.get("batch_dir", ""),
        )
        for batch in batches
    ]
    lines.extend(_markdown_table(("Sweep", "Planned", "Completed", "Batch directory"), completion_rows))
    lines.extend(["", "## Integrity", ""])
    lines.extend(
        _markdown_table(
            ("Check", "Value"),
            [
                ("planned_runs", integrity.get("planned_runs", "")),
                ("completed_runs", integrity.get("completed_runs", "")),
                ("missing_metrics_csv", integrity.get("missing_metrics_csv", "")),
                ("empty_metrics_csv", integrity.get("empty_metrics_csv", "")),
                ("critical_nonfinite_issues", integrity.get("critical_nonfinite_issues", "")),
                ("scenario_flag_mismatches", integrity.get("scenario_flag_mismatches", "")),
                ("lineage_branch_candidates_json_nonempty", integrity.get("lineage_branch_candidates_json_nonempty", "")),
            ],
        )
    )
    for campaign in sorted({str(item["campaign"]) for item in summaries}):
        label = SWEEP_LABELS.get(campaign, campaign)
        lines.extend(["", f"## {label}", ""])
        metrics = DECISION_METRICS.get(campaign, ("alive", "successful_births"))
        rows = []
        for item in summaries:
            if item["campaign"] != campaign:
                continue
            values = [item["variant"], item["scenario"], item["run_count"]]
            values.extend(_fmt((item.get("metrics") or {}).get(metric)) for metric in metrics)
            rows.append(values)
        lines.extend(_markdown_table(("Variant", "Scenario", "Runs", *metrics), rows))
        lines.extend(["", f"Recommended band: `{recommendations.get(campaign, 'no acceptable band')}`", ""])
    lines.extend(["", "## Variant Deltas", ""])
    delta_rows = []
    for item in deltas:
        metrics = DECISION_METRICS.get(str(item["campaign"]), ("alive", "successful_births"))
        delta_rows.append(
            [
                SWEEP_LABELS.get(str(item["campaign"]), str(item["campaign"])),
                item["scenario"],
                f"{item['variant']} - {item['baseline_variant']}",
                item["paired_seed_count"],
                ", ".join(f"{metric}={_fmt((item.get('deltas') or {}).get(metric))}" for metric in metrics),
            ]
        )
    lines.extend(_markdown_table(("Sweep", "Scenario", "Comparison", "Paired seeds", "Mean deltas"), delta_rows))
    combined = {
        "predator": recommendations.get("patch_jungle_v0_campaign_c1_predator", "no acceptable band"),
        "waste": recommendations.get("patch_jungle_v0_campaign_c2_waste", "no acceptable band"),
        "terrain protection": recommendations.get("patch_jungle_v0_campaign_c3_terrain", "no acceptable band"),
        "recovery": recommendations.get("patch_jungle_v0_campaign_c4_recovery", "no acceptable band"),
    }
    lines.extend(
        [
            "",
            "## Combined Recommendation",
            "",
            "Recommended Campaign D candidate:",
            "",
            *[f"- {key}: `{value}`" for key, value in combined.items()],
            "",
            "Do not edit defaults from this report alone. Treat this as a candidate for longer validation.",
            "",
            "## What Is Not Concluded",
            "",
            "- No claim is made that intelligence emerged.",
            "- No claim is made that memory, fear, prediction, or GRU use was selected.",
            "- No claim is made that durable lineage diversity improved.",
            "- No claim is made that the overall thesis is proven.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    _ensure_import_path()
    from experiments.patch_jungle_eval import all_metric_names

    args = _parse_args(argv)
    metric_names = tuple(dict.fromkeys((*KEY_METRICS, *all_metric_names())))
    runs: list[dict[str, Any]] = []
    batches: list[dict[str, Any]] = []
    for raw_batch_dir in args.batch_dir:
        batch_dir = Path(raw_batch_dir)
        manifest = _read_json(batch_dir / "manifest.json")
        campaign = str((manifest.get("registry") or {}).get("name") or batch_dir.name)
        matrix_rows = _load_csv(batch_dir / "matrix.csv")
        batch_runs = [
            _summarize_run(
                batch_dir=batch_dir,
                campaign=campaign,
                row=row,
                metric_names=metric_names,
                tail_n=max(1, int(args.tail_n)),
            )
            for row in matrix_rows
        ]
        runs.extend(batch_runs)
        batches.append(
            {
                "batch_dir": str(batch_dir),
                "campaign": campaign,
                "registry": manifest.get("registry") or {},
                "planned_runs": len(matrix_rows),
                "completed_runs": sum(1 for run in batch_runs if run["metrics_exists"] and int(run["metrics_rows"]) > 0),
            }
        )

    summaries = _build_variant_summaries(runs, metric_names)
    deltas = _build_paired_deltas(runs, metric_names)
    integrity = {
        "planned_runs": len(runs),
        "completed_runs": sum(1 for run in runs if run["metrics_exists"] and int(run["metrics_rows"]) > 0),
        "missing_metrics_csv": sum(1 for run in runs if not run["metrics_exists"]),
        "empty_metrics_csv": sum(1 for run in runs if run["metrics_exists"] and int(run["metrics_rows"]) == 0),
        "critical_nonfinite_issues": sum(len(run["critical_nonfinite"]) for run in runs),
        "scenario_flag_mismatches": sum(len(run["scenario_flag_mismatches"]) for run in runs),
        "lineage_strategy_json_present": sum(1 for run in runs if run["lineage_strategy_json_present"]),
        "lineage_branch_candidates_json_present": sum(1 for run in runs if run["lineage_branch_candidates_json_present"]),
        "lineage_strategy_json_nonempty": sum(1 for run in runs if run["lineage_strategy_json_nonempty"]),
        "lineage_branch_candidates_json_nonempty": sum(1 for run in runs if run["lineage_branch_candidates_json_nonempty"]),
    }
    recommendations = {
        campaign: _recommend_band(campaign, summaries)
        for campaign in sorted({str(item["campaign"]) for item in summaries})
    }
    report = {
        "schema_version": "patch_jungle_campaign_c_report_v1",
        "campaign_name": args.campaign_name,
        "output_root": args.output_root,
        "batches": batches,
        "integrity": integrity,
        "variant_summaries": summaries,
        "paired_deltas": deltas,
        "recommendations": recommendations,
        "preflight": _read_json(args.preflight_json) if args.preflight_json else {},
        "repeat_parity": _read_json(args.repeat_parity_json) if args.repeat_parity_json else {},
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(
        Path(args.summary_csv),
        summaries,
        ("campaign", "variant", "scenario", "run_count", "seeds", *KEY_METRICS),
        "metrics",
    )
    _write_csv(
        Path(args.deltas_csv),
        deltas,
        ("campaign", "scenario", "baseline_variant", "variant", "paired_seed_count", "paired_seeds", *KEY_METRICS),
        "deltas",
    )
    if args.report_md:
        _write_markdown_report(
            Path(args.report_md),
            campaign_name=args.campaign_name,
            output_root=args.output_root,
            batches=batches,
            integrity=integrity,
            summaries=summaries,
            deltas=deltas,
            preflight=report["preflight"],
            repeat_parity=report["repeat_parity"],
            recommendations=recommendations,
        )
    print(f"[patch-jungle-campaign-c] wrote {out_json}")
    print(f"[patch-jungle-campaign-c] wrote {args.summary_csv}")
    print(f"[patch-jungle-campaign-c] wrote {args.deltas_csv}")
    if args.report_md:
        print(f"[patch-jungle-campaign-c] wrote {args.report_md}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
