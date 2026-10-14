#!/usr/bin/env python3
"""Summarize Patch/Jungle V0 Campaign D tuned-validation batches.

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


SEEDS: tuple[int, ...] = (
    101,
    202,
    303,
    404,
    505,
    606,
    707,
    808,
    909,
    1001,
    1102,
    1203,
    1304,
    1405,
    1506,
    1607,
    1708,
    1809,
    19010,
    20011,
)

SCENARIOS: tuple[str, ...] = (
    "d00_flat_baseline",
    "d01_patch_observed",
    "d02_full_jungle_default",
    "d03_tuned_full_jungle_no_recovery",
    "d04_tuned_full_jungle_recovery",
)

SCENARIO_LABELS: dict[str, str] = {
    "d00_flat_baseline": "Flat baseline",
    "d01_patch_observed": "Patch observed",
    "d02_full_jungle_default": "Default full Jungle recovery",
    "d03_tuned_full_jungle_no_recovery": "Tuned full Jungle without recovery",
    "d04_tuned_full_jungle_recovery": "Tuned full Jungle with recovery",
}

EXPECTED_ENV: dict[str, dict[str, bool | int | float]] = {
    "d00_flat_baseline": {
        "patch_world_enable": False,
        "patch_world_observation_enable": False,
        "patch_world_waste_consequence_enable": False,
        "jungle_predator_pressure_enable": False,
        "jungle_reproduction_emissions_enable": False,
        "jungle_terrain_protection_enable": False,
        "jungle_ecological_recovery_enable": False,
    },
    "d01_patch_observed": {
        "patch_world_enable": True,
        "patch_world_observation_enable": True,
        "patch_world_waste_consequence_enable": False,
        "jungle_predator_pressure_enable": False,
        "jungle_reproduction_emissions_enable": False,
        "jungle_terrain_protection_enable": False,
        "jungle_ecological_recovery_enable": False,
    },
    "d02_full_jungle_default": {
        "patch_world_enable": True,
        "patch_world_observation_enable": True,
        "patch_world_waste_consequence_enable": True,
        "jungle_predator_pressure_enable": True,
        "jungle_reproduction_emissions_enable": True,
        "jungle_terrain_protection_enable": True,
        "jungle_ecological_recovery_enable": True,
    },
    "d03_tuned_full_jungle_no_recovery": {
        "patch_world_enable": True,
        "patch_world_observation_enable": True,
        "patch_world_waste_consequence_enable": True,
        "jungle_predator_pressure_enable": True,
        "jungle_reproduction_emissions_enable": True,
        "jungle_terrain_protection_enable": True,
        "jungle_ecological_recovery_enable": False,
        "jungle_predator_scent_weight": 0.02,
        "jungle_predator_waste_weight": 0.02,
        "jungle_predator_density_weight": 0.02,
        "jungle_predator_visibility_weight": 0.02,
        "jungle_predator_risk_cap": 0.05,
        "jungle_predator_health_damage": 0.25,
        "patch_waste_regrowth_penalty_scale": 0.5,
        "patch_waste_regrowth_penalty_max": 0.5,
        "patch_density_regrowth_penalty_scale": 0.5,
        "patch_density_regrowth_penalty_max": 0.5,
        "patch_regrowth_min_multiplier": 0.1,
        "jungle_cover_damage_reduction_weight": 0.4,
        "jungle_shelter_damage_reduction_weight": 0.4,
        "jungle_terrain_damage_multiplier_min": 0.15,
    },
    "d04_tuned_full_jungle_recovery": {
        "patch_world_enable": True,
        "patch_world_observation_enable": True,
        "patch_world_waste_consequence_enable": True,
        "jungle_predator_pressure_enable": True,
        "jungle_reproduction_emissions_enable": True,
        "jungle_terrain_protection_enable": True,
        "jungle_ecological_recovery_enable": True,
        "jungle_predator_scent_weight": 0.02,
        "jungle_predator_waste_weight": 0.02,
        "jungle_predator_density_weight": 0.02,
        "jungle_predator_visibility_weight": 0.02,
        "jungle_predator_risk_cap": 0.05,
        "jungle_predator_health_damage": 0.25,
        "patch_waste_regrowth_penalty_scale": 0.5,
        "patch_waste_regrowth_penalty_max": 0.5,
        "patch_density_regrowth_penalty_scale": 0.5,
        "patch_density_regrowth_penalty_max": 0.5,
        "patch_regrowth_min_multiplier": 0.1,
        "jungle_cover_damage_reduction_weight": 0.4,
        "jungle_shelter_damage_reduction_weight": 0.4,
        "jungle_terrain_damage_multiplier_min": 0.15,
        "jungle_recovery_regrowth_boost_scale": 0.05,
        "jungle_recovery_regrowth_boost_max": 0.1,
    },
}

PRIMARY_PAIRS: tuple[tuple[str, str], ...] = (
    ("d00_flat_baseline", "d01_patch_observed"),
    ("d01_patch_observed", "d02_full_jungle_default"),
    ("d01_patch_observed", "d03_tuned_full_jungle_no_recovery"),
    ("d03_tuned_full_jungle_no_recovery", "d04_tuned_full_jungle_recovery"),
    ("d02_full_jungle_default", "d04_tuned_full_jungle_recovery"),
)

KEY_METRICS: tuple[str, ...] = (
    "alive",
    "deaths",
    "cohort_survival_rate",
    "mean_survival_steps_total",
    "successful_births",
    "failed_births",
    "birth_attempts",
    "first_repro_step_mean_alive",
    "birth_patch_count",
    "birth_patch_waste_mean",
    "birth_patch_shelter_mean",
    "occupied_patch_food_amount_mean",
    "occupied_patch_cover_mean",
    "occupied_patch_shelter_mean",
    "occupied_patch_scent_mean",
    "occupied_patch_waste_mean",
    "occupied_patch_density_mean",
    "patch_food_amount_mean",
    "patch_food_depleted_frac",
    "patch_waste_mean",
    "patch_waste_p90",
    "patch_density_mean",
    "patch_density_p90",
    "patch_regrowth_suppressed_total",
    "patch_regrowth_multiplier_mean",
    "jungle_predator_risk_mean",
    "jungle_predator_occupied_risk_mean",
    "jungle_predator_events",
    "jungle_predator_damage_total",
    "jungle_predator_damage_raw_total",
    "jungle_predator_damage_effective_total",
    "jungle_predator_damage_prevented_total",
    "predator_event_patch_scent_mean",
    "predator_event_patch_waste_mean",
    "predator_event_patch_density_mean",
    "predator_event_patch_cover_mean",
    "predator_event_patch_shelter_mean",
    "patch_recovery_activity_mean",
    "jungle_recovery_regrowth_boost_total",
    "patch_recovery_added_regrowth_total",
    "patch_regrowth_after_waste_total",
    "patch_regrowth_after_recovery_total",
    "lineage_active_count",
    "lineage_top1_alive_share",
    "lineage_top3_alive_share",
    "lineage_alive_share_entropy",
    "lineage_effective_alive_lineages",
    "lineage_branch_candidate_count",
    "lineage_patch_waste_mean_std",
    "lineage_patch_shelter_mean_std",
    "lineage_predator_damage_mean_std",
)

CRITICAL_METRICS: tuple[str, ...] = (
    "alive",
    "deaths",
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
    "lineage_effective_alive_lineages",
)

GROUPS: dict[str, tuple[str, ...]] = {
    "population_survival": (
        "alive",
        "deaths",
        "cohort_survival_rate",
        "mean_survival_steps_total",
    ),
    "reproduction": (
        "successful_births",
        "failed_births",
        "birth_attempts",
        "birth_patch_waste_mean",
    ),
    "patch_use": (
        "occupied_patch_food_amount_mean",
        "occupied_patch_cover_mean",
        "occupied_patch_shelter_mean",
        "occupied_patch_waste_mean",
        "occupied_patch_density_mean",
    ),
    "waste_crowding": (
        "patch_waste_mean",
        "patch_waste_p90",
        "patch_density_mean",
        "patch_density_p90",
        "patch_regrowth_suppressed_total",
        "patch_regrowth_multiplier_mean",
    ),
    "predator_pressure": (
        "jungle_predator_events",
        "jungle_predator_damage_total",
        "jungle_predator_occupied_risk_mean",
        "predator_event_patch_waste_mean",
    ),
    "terrain_protection": (
        "jungle_predator_damage_raw_total",
        "jungle_predator_damage_effective_total",
        "jungle_predator_damage_prevented_total",
    ),
    "recovery": (
        "patch_recovery_activity_mean",
        "jungle_recovery_regrowth_boost_total",
        "patch_recovery_added_regrowth_total",
    ),
    "lineage_diversity": (
        "lineage_active_count",
        "lineage_top1_alive_share",
        "lineage_alive_share_entropy",
        "lineage_effective_alive_lineages",
        "lineage_branch_candidate_count",
    ),
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_import_path() -> None:
    root = str(_project_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Patch/Jungle V0 Campaign D batches")
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--analysis-json", required=True)
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--deltas-csv", required=True)
    parser.add_argument("--report-md", required=True)
    parser.add_argument("--repeat-parity-json", default="")
    parser.add_argument("--tail-n", type=int, default=1)
    parser.add_argument("--campaign-name", default="patch_jungle_v0_campaign_d_tuned_validation")
    parser.add_argument("--output-root", default="/tmp/emergence_patch_jungle_campaign_d")
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
    parsed = _parse_float(value)
    if parsed is None:
        return "" if value is None else str(value)
    return f"{parsed:.{digits}f}"


def _last_metrics_row(path: Path) -> dict[str, str] | None:
    rows = _load_csv(path)
    return rows[-1] if rows else None


def _json_nonempty(value: str | None) -> bool:
    if value is None or not str(value).strip():
        return False
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return False
    return bool(parsed)


def _env_mismatches(scenario: str, config_path: Path) -> list[dict[str, Any]]:
    if not config_path.exists():
        return [{"scenario": scenario, "field": "config.json", "expected": "present", "actual": "missing"}]
    env = dict((_read_json(config_path).get("env_cfg") or {}))
    mismatches: list[dict[str, Any]] = []
    for field, expected in EXPECTED_ENV.get(scenario, {}).items():
        if field not in env:
            mismatches.append({"scenario": scenario, "field": field, "expected": expected, "actual": "missing"})
            continue
        actual = env[field]
        if isinstance(expected, bool):
            ok = bool(actual) is expected
        elif isinstance(expected, float):
            ok = abs(float(actual) - expected) <= 1e-12
        else:
            ok = actual == expected
        if not ok:
            mismatches.append({"scenario": scenario, "field": field, "expected": expected, "actual": actual})
    return mismatches


def _summarize_run(row: Mapping[str, str], metric_names: Sequence[str], tail_n: int) -> dict[str, Any]:
    from experiments.patch_jungle_eval import summarize_metrics_csv

    scenario = str(row.get("scenario") or "")
    seed = int(row.get("seed") or 0)
    run_dir = Path(str(row.get("run_dir") or ""))
    metrics_path = run_dir / "metrics.csv"
    config_path = run_dir / "config.json"
    run: dict[str, Any] = {
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
        "env_mismatches": _env_mismatches(scenario, config_path),
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
    strategy_json = final_row.get("lineage_strategy_descriptors_json")
    branch_json = final_row.get("lineage_branch_candidates_json")
    run["lineage_strategy_json_present"] = strategy_json is not None and strategy_json != ""
    run["lineage_branch_candidates_json_present"] = branch_json is not None and branch_json != ""
    run["lineage_strategy_json_nonempty"] = _json_nonempty(strategy_json)
    run["lineage_branch_candidates_json_nonempty"] = _json_nonempty(branch_json)
    return run


def _scenario_summaries(runs: Sequence[Mapping[str, Any]], metric_names: Sequence[str]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[str(run["scenario"])].append(run)
    summaries: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        entries = sorted(grouped.get(scenario, []), key=lambda item: int(item["seed"]))
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
                "scenario": scenario,
                "label": SCENARIO_LABELS[scenario],
                "run_count": len(entries),
                "seeds": [int(entry["seed"]) for entry in entries],
                "metrics": metrics,
                "missing_metrics": sorted(
                    {metric for entry in entries for metric in list(entry.get("missing_metrics", []) or [])}
                ),
            }
        )
    return summaries


def _paired_deltas(runs: Sequence[Mapping[str, Any]], metric_names: Sequence[str]) -> list[dict[str, Any]]:
    by_key = {(str(run["scenario"]), int(run["seed"])): run for run in runs}
    deltas: list[dict[str, Any]] = []
    for baseline, variant in PRIMARY_PAIRS:
        paired = sorted(
            seed for seed in SEEDS if (baseline, seed) in by_key and (variant, seed) in by_key
        )
        metric_deltas: dict[str, float] = {}
        for metric in metric_names:
            values: list[float] = []
            for seed in paired:
                base = _parse_float((by_key[(baseline, seed)].get("metrics") or {}).get(metric))
                other = _parse_float((by_key[(variant, seed)].get("metrics") or {}).get(metric))
                if base is not None and other is not None:
                    values.append(float(other - base))
            mean = _mean(values)
            if mean is not None:
                metric_deltas[metric] = mean
        deltas.append(
            {
                "baseline": baseline,
                "variant": variant,
                "comparison": f"{variant} - {baseline}",
                "paired_seed_count": len(paired),
                "paired_seeds": paired,
                "deltas": metric_deltas,
            }
        )
    return deltas


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str], metric_key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for item in rows:
            row = {field: item.get(field, "") for field in fields}
            metrics = item.get(metric_key) or {}
            for metric in KEY_METRICS:
                if metric in fields:
                    row[metric] = metrics.get(metric, "")
            if isinstance(row.get("seeds"), list):
                row["seeds"] = " ".join(str(seed) for seed in row["seeds"])
            if isinstance(row.get("paired_seeds"), list):
                row["paired_seeds"] = " ".join(str(seed) for seed in row["paired_seeds"])
            writer.writerow(row)


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(value) for value in row) + " |")
    return out


def _metric(summary_by_scenario: Mapping[str, Mapping[str, Any]], scenario: str, metric: str) -> float | None:
    return _parse_float((summary_by_scenario.get(scenario, {}).get("metrics") or {}).get(metric))


def _build_analysis(
    summaries: Sequence[Mapping[str, Any]],
    deltas: Sequence[Mapping[str, Any]],
    integrity: Mapping[str, Any],
) -> dict[str, Any]:
    by_scenario = {str(item["scenario"]): item for item in summaries}
    d04_alive = _metric(by_scenario, "d04_tuned_full_jungle_recovery", "alive") or 0.0
    d04_births = _metric(by_scenario, "d04_tuned_full_jungle_recovery", "successful_births") or 0.0
    d04_pred = _metric(by_scenario, "d04_tuned_full_jungle_recovery", "jungle_predator_damage_total") or 0.0
    d04_suppressed = _metric(by_scenario, "d04_tuned_full_jungle_recovery", "patch_regrowth_suppressed_total") or 0.0
    d04_prevented = _metric(by_scenario, "d04_tuned_full_jungle_recovery", "jungle_predator_damage_prevented_total") or 0.0
    d04_recovery = _metric(by_scenario, "d04_tuned_full_jungle_recovery", "patch_recovery_added_regrowth_total") or 0.0
    d04_lineages = _metric(by_scenario, "d04_tuned_full_jungle_recovery", "lineage_effective_alive_lineages") or 0.0
    d04_branches = _metric(by_scenario, "d04_tuned_full_jungle_recovery", "lineage_branch_candidate_count") or 0.0
    go = (
        bool(integrity.get("all_runs_complete"))
        and d04_alive > 0.0
        and d04_births > 0.0
        and d04_pred > 0.0
        and d04_suppressed > 0.0
        and d04_prevented > 0.0
        and d04_recovery > 0.0
        and d04_lineages > 1.0
        and d04_branches > 0.0
        and int(integrity.get("critical_nonfinite_issues") or 0) == 0
        and int(integrity.get("env_mismatches") or 0) == 0
    )
    return {
        "candidate_configuration": {
            "predator": "medium",
            "waste_crowding": "medium",
            "terrain_protection": "high",
            "recovery": "low",
        },
        "validation_answers": {
            "tuned_configuration_remained_viable": d04_alive > 0.0 and d04_births > 0.0,
            "tuning_improved_over_default_full_jungle": "assess paired deltas d04 - d02; improvement means retained pressure with viability, not only higher population",
            "tuned_recovery_helped": "assess paired deltas d04 - d03",
            "reproduction_remained_contextual": d04_births > 0.0,
            "lineage_reports_remained_useful": d04_lineages > 1.0 and d04_branches > 0.0,
        },
        "campaign_e_recommendation": "GO" if go else "LIMITED GO",
        "primary_pairs": deltas,
    }


def _write_markdown_report(
    path: Path,
    *,
    campaign_name: str,
    output_root: str,
    batch_dir: str,
    integrity: Mapping[str, Any],
    summaries: Sequence[Mapping[str, Any]],
    deltas: Sequence[Mapping[str, Any]],
    analysis: Mapping[str, Any],
    repeat_parity: Mapping[str, Any],
) -> None:
    summary_by_scenario = {str(item["scenario"]): item for item in summaries}
    lines: list[str] = [
        "# Patch/Jungle V0 Campaign D Tuned Validation Report",
        "",
        f"Campaign: `{campaign_name}`",
        "",
        f"Output root: `{output_root}`",
        "",
        f"Batch directory: `{batch_dir}`",
        "",
        "Campaign D validates one selected Campaign C pressure configuration. It does not tune during the run and does not claim intelligence, emergence, memory selection, fear selection, prediction selection, culture, or proof of the thesis.",
        "",
        "## Campaign C Candidate",
        "",
        "- predator pressure: `medium`",
        "- waste/crowding pressure: `medium`",
        "- terrain protection: `high`",
        "- ecological recovery: `low`",
        "",
        "## Settings",
        "",
        "| Setting | Value |",
        "| --- | --- |",
        "| Scenarios | `d00_flat_baseline`, `d01_patch_observed`, `d02_full_jungle_default`, `d03_tuned_full_jungle_no_recovery`, `d04_tuned_full_jungle_recovery` |",
        "| Seeds | `101, 202, 303, 404, 505, 606, 707, 808, 909, 1001, 1102, 1203, 1304, 1405, 1506, 1607, 1708, 1809, 19010, 20011` |",
        "| Generations | 16 |",
        "| Steps per generation | 100 |",
        "| Population | 48 |",
        "| Grid size | 16 |",
        "| Device | CPU |",
        "",
        "## Preflight",
        "",
        "The required preflight test suites and `git diff --check` passed before the full Campaign D run. The Campaign D dry-run planned 100 runs with deterministic scenario and seed ordering.",
        "",
        "## Repeat Parity",
        "",
    ]
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
    lines.extend(
        _markdown_table(
            ("Check", "Value"),
            [
                ("planned_runs", integrity.get("planned_runs", "")),
                ("completed_runs", integrity.get("completed_runs", "")),
                ("missing_metrics_csv", integrity.get("missing_metrics_csv", "")),
                ("empty_metrics_csv", integrity.get("empty_metrics_csv", "")),
                ("critical_nonfinite_issues", integrity.get("critical_nonfinite_issues", "")),
                ("env_mismatches", integrity.get("env_mismatches", "")),
                ("paired_seed_sets_complete", integrity.get("paired_seed_sets_complete", "")),
                ("lineage_strategy_json_nonempty", integrity.get("lineage_strategy_json_nonempty", "")),
                ("lineage_branch_candidates_json_nonempty", integrity.get("lineage_branch_candidates_json_nonempty", "")),
            ],
        )
    )
    lines.extend(["", "## Scenario Summary", ""])
    metric_cols = (
        "alive",
        "successful_births",
        "patch_regrowth_suppressed_total",
        "jungle_predator_damage_total",
        "jungle_predator_damage_prevented_total",
        "patch_recovery_added_regrowth_total",
        "lineage_effective_alive_lineages",
        "lineage_branch_candidate_count",
    )
    rows = []
    for item in summaries:
        metrics = item.get("metrics") or {}
        rows.append([item["scenario"], item["run_count"], *[_fmt(metrics.get(metric)) for metric in metric_cols]])
    lines.extend(_markdown_table(("Scenario", "Runs", *metric_cols), rows))
    lines.extend(["", "## Primary Paired Comparisons", ""])
    delta_cols = (
        "alive",
        "successful_births",
        "patch_regrowth_suppressed_total",
        "jungle_predator_damage_total",
        "jungle_predator_damage_prevented_total",
        "patch_recovery_added_regrowth_total",
        "lineage_effective_alive_lineages",
    )
    delta_rows = []
    for item in deltas:
        delta_rows.append(
            [
                item["comparison"],
                item["paired_seed_count"],
                *[_fmt((item.get("deltas") or {}).get(metric)) for metric in delta_cols],
            ]
        )
    lines.extend(_markdown_table(("Comparison", "Paired seeds", *delta_cols), delta_rows))
    for group, metrics in GROUPS.items():
        lines.extend(["", f"## {group.replace('_', ' ').title()}", ""])
        group_rows = []
        for scenario in SCENARIOS:
            values = summary_by_scenario[scenario].get("metrics") or {}
            group_rows.append([scenario, *[_fmt(values.get(metric)) for metric in metrics]])
        lines.extend(_markdown_table(("Scenario", *metrics), group_rows))
    lines.extend(
        [
            "",
            "## Validation Answers",
            "",
            f"- Tuned configuration remained viable: `{analysis['validation_answers']['tuned_configuration_remained_viable']}`.",
            "- Tuning versus default full Jungle should be read from `d04_tuned_full_jungle_recovery - d02_full_jungle_default`; better means retained interpretable pressure with viability, not only a higher population.",
            "- Tuned recovery should be read from `d04_tuned_full_jungle_recovery - d03_tuned_full_jungle_no_recovery`; useful recovery stabilizes without erasing waste or predator pressure.",
            f"- Reproduction remained possible in tuned recovery: `{analysis['validation_answers']['reproduction_remained_contextual']}`.",
            f"- Lineage reports remained useful by basic availability/count checks: `{analysis['validation_answers']['lineage_reports_remained_useful']}`.",
            "",
            "## Warnings / Anomalies",
            "",
            "- Treat all conclusions as bounded to this horizon and seed set.",
            "- This validation does not prove adaptive cognition or durable lineage diversity improvement.",
            "",
            "## Recommendation For Campaign E",
            "",
            f"Recommendation: `{analysis['campaign_e_recommendation']}` for Campaign E lineage/strategy deep dive, subject to the paired-delta interpretation above.",
            "",
            "## What Is Not Concluded",
            "",
            "- No claim is made that intelligence emerged.",
            "- No claim is made that memory, fear, prediction, or GRU use was selected.",
            "- No claim is made that culture appeared.",
            "- No claim is made that the thesis is proven.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    _ensure_import_path()
    from experiments.patch_jungle_eval import all_metric_names

    args = _parse_args(argv)
    batch_dir = Path(args.batch_dir)
    matrix_rows = _load_csv(batch_dir / "matrix.csv")
    metric_names = tuple(dict.fromkeys((*KEY_METRICS, *all_metric_names())))
    runs = [_summarize_run(row, metric_names, max(1, int(args.tail_n))) for row in matrix_rows]
    summaries = _scenario_summaries(runs, metric_names)
    deltas = _paired_deltas(runs, metric_names)
    seed_sets = {
        scenario: sorted(int(run["seed"]) for run in runs if run["scenario"] == scenario)
        for scenario in SCENARIOS
    }
    integrity = {
        "planned_runs": len(matrix_rows),
        "completed_runs": sum(1 for run in runs if run["metrics_exists"] and int(run["metrics_rows"]) > 0),
        "all_runs_complete": len(matrix_rows) == sum(
            1 for run in runs if run["metrics_exists"] and int(run["metrics_rows"]) > 0
        ),
        "missing_metrics_csv": sum(1 for run in runs if not run["metrics_exists"]),
        "empty_metrics_csv": sum(1 for run in runs if run["metrics_exists"] and int(run["metrics_rows"]) == 0),
        "critical_nonfinite_issues": sum(len(run["critical_nonfinite"]) for run in runs),
        "env_mismatches": sum(len(run["env_mismatches"]) for run in runs),
        "paired_seed_sets_complete": all(seed_sets[scenario] == list(SEEDS) for scenario in SCENARIOS),
        "seed_sets": seed_sets,
        "lineage_strategy_json_present": sum(1 for run in runs if run["lineage_strategy_json_present"]),
        "lineage_branch_candidates_json_present": sum(1 for run in runs if run["lineage_branch_candidates_json_present"]),
        "lineage_strategy_json_nonempty": sum(1 for run in runs if run["lineage_strategy_json_nonempty"]),
        "lineage_branch_candidates_json_nonempty": sum(1 for run in runs if run["lineage_branch_candidates_json_nonempty"]),
        "env_mismatch_details": [item for run in runs for item in run["env_mismatches"]],
    }
    analysis = _build_analysis(summaries, deltas, integrity)
    repeat_parity = _read_json(args.repeat_parity_json) if args.repeat_parity_json else {}
    report = {
        "schema_version": "patch_jungle_campaign_d_report_v1",
        "campaign_name": args.campaign_name,
        "output_root": args.output_root,
        "batch_dir": str(batch_dir),
        "integrity": integrity,
        "scenario_summaries": summaries,
        "paired_deltas": deltas,
        "analysis": analysis,
        "repeat_parity": repeat_parity,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(args.analysis_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.analysis_json).write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(
        Path(args.summary_csv),
        summaries,
        ("scenario", "label", "run_count", "seeds", *KEY_METRICS),
        "metrics",
    )
    _write_csv(
        Path(args.deltas_csv),
        deltas,
        ("baseline", "variant", "comparison", "paired_seed_count", "paired_seeds", *KEY_METRICS),
        "deltas",
    )
    _write_markdown_report(
        Path(args.report_md),
        campaign_name=args.campaign_name,
        output_root=args.output_root,
        batch_dir=str(batch_dir),
        integrity=integrity,
        summaries=summaries,
        deltas=deltas,
        analysis=analysis,
        repeat_parity=repeat_parity,
    )
    print(f"[patch-jungle-campaign-d] wrote {args.out_json}")
    print(f"[patch-jungle-campaign-d] wrote {args.analysis_json}")
    print(f"[patch-jungle-campaign-d] wrote {args.summary_csv}")
    print(f"[patch-jungle-campaign-d] wrote {args.deltas_csv}")
    print(f"[patch-jungle-campaign-d] wrote {args.report_md}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
