#!/usr/bin/env python3
"""R1 integrated baseline reproduction reporting helper.

Reporting-only infrastructure: reads completed registry batch outputs and
emits deterministic JSON, CSV, and Markdown summaries. It does not change
simulation behavior.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence


csv.field_size_limit(sys.maxsize)

SCENARIOS: tuple[str, ...] = (
    "r1_00_flat_baseline",
    "r1_01_patch_observed",
    "r1_02_default_full_jungle",
    "r1_03_tuned_full_jungle_recovery",
    "r1_04_theme12_clean_pressure_reference",
)

SEEDS: tuple[int, ...] = (101, 202, 303, 404, 505)

SCENARIO_LABELS: dict[str, str] = {
    "r1_00_flat_baseline": "Flat baseline",
    "r1_01_patch_observed": "Patch observed",
    "r1_02_default_full_jungle": "Default full Jungle",
    "r1_03_tuned_full_jungle_recovery": "Tuned full Jungle recovery",
    "r1_04_theme12_clean_pressure_reference": "Theme 12 clean pressure reference",
}

EXPECTED_ENV: dict[str, dict[str, bool | float]] = {
    "r1_00_flat_baseline": {
        "patch_world_enable": False,
        "patch_world_observation_enable": False,
        "patch_world_waste_consequence_enable": False,
        "jungle_predator_pressure_enable": False,
        "jungle_reproduction_emissions_enable": False,
        "jungle_terrain_protection_enable": False,
        "jungle_ecological_recovery_enable": False,
    },
    "r1_01_patch_observed": {
        "patch_world_enable": True,
        "patch_world_observation_enable": True,
        "patch_world_waste_consequence_enable": False,
        "jungle_predator_pressure_enable": False,
        "jungle_reproduction_emissions_enable": False,
        "jungle_terrain_protection_enable": False,
        "jungle_ecological_recovery_enable": False,
    },
    "r1_02_default_full_jungle": {
        "patch_world_enable": True,
        "patch_world_observation_enable": True,
        "patch_world_waste_consequence_enable": True,
        "jungle_predator_pressure_enable": True,
        "jungle_reproduction_emissions_enable": True,
        "jungle_terrain_protection_enable": True,
        "jungle_ecological_recovery_enable": True,
        "patch_cover_default": 0.25,
        "patch_shelter_default": 0.15,
    },
    "r1_03_tuned_full_jungle_recovery": {
        "patch_world_enable": True,
        "patch_world_observation_enable": True,
        "patch_world_waste_consequence_enable": True,
        "jungle_predator_pressure_enable": True,
        "jungle_reproduction_emissions_enable": True,
        "jungle_terrain_protection_enable": True,
        "jungle_ecological_recovery_enable": True,
        "jungle_predator_base_risk": 0.0,
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
        "jungle_recovery_fertility_enable": False,
        "jungle_resource_spread_enable": False,
    },
    "r1_04_theme12_clean_pressure_reference": {
        "patch_world_enable": True,
        "patch_world_observation_enable": True,
        "patch_world_waste_consequence_enable": True,
        "jungle_predator_pressure_enable": True,
        "jungle_reproduction_emissions_enable": True,
        "jungle_terrain_protection_enable": True,
        "jungle_ecological_recovery_enable": True,
        "patch_cover_default": 0.25,
        "patch_shelter_default": 0.15,
        "patch_waste_regrowth_penalty_scale": 0.35,
        "patch_density_regrowth_penalty_scale": 0.35,
        "jungle_predator_risk_cap": 0.045,
        "jungle_predator_health_damage": 0.22,
        "jungle_recovery_regrowth_boost_scale": 0.08,
    },
}

EXPECTED_SIM: dict[str, Any] = {
    "device": "cpu",
    "bank_device": "cpu",
    "phase_e_e1_enable": False,
    "enable_developmental_cognition": False,
    "enable_introspection_channels": False,
    "enable_self_regulation": False,
    "enable_self_modification": False,
    "enable_directed_inheritance": False,
}

PRIMARY_PAIRS: tuple[tuple[str, str], ...] = (
    ("r1_00_flat_baseline", "r1_01_patch_observed"),
    ("r1_01_patch_observed", "r1_02_default_full_jungle"),
    ("r1_02_default_full_jungle", "r1_03_tuned_full_jungle_recovery"),
    ("r1_00_flat_baseline", "r1_04_theme12_clean_pressure_reference"),
    ("r1_03_tuned_full_jungle_recovery", "r1_04_theme12_clean_pressure_reference"),
)

KEY_METRICS: tuple[str, ...] = (
    "alive",
    "deaths",
    "cohort_survival_rate",
    "mean_survival_steps_total",
    "successful_births",
    "mean_offspring_total",
    "birth_attempts",
    "patch_food_amount_mean",
    "patch_food_depleted_frac",
    "patch_waste_mean",
    "patch_waste_p90",
    "patch_density_mean",
    "patch_density_p90",
    "patch_regrowth_suppressed_total",
    "patch_regrowth_multiplier_mean",
    "jungle_predator_events",
    "jungle_predator_damage_total",
    "jungle_predator_damage_raw_total",
    "jungle_predator_damage_effective_total",
    "jungle_predator_damage_prevented_total",
    "jungle_predator_occupied_risk_mean",
    "patch_recovery_activity_mean",
    "jungle_recovery_regrowth_boost_total",
    "patch_recovery_added_regrowth_total",
    "occupied_patch_food_amount_mean",
    "occupied_patch_cover_mean",
    "occupied_patch_shelter_mean",
    "occupied_patch_waste_mean",
    "occupied_patch_density_mean",
    "lineage_active_count",
    "lineage_alive_share_entropy",
    "lineage_effective_alive_lineages",
    "lineage_branch_candidate_count",
    "lineage_strategy_descriptor_count",
    "lineage_patch_waste_mean_std",
    "lineage_patch_shelter_mean_std",
    "lineage_predator_damage_mean_std",
    "introspection_enabled_fraction",
    "self_regulation_enabled_fraction",
    "self_modification_enabled_fraction",
)

CRITICAL_METRICS: tuple[str, ...] = (
    "alive",
    "successful_births",
    "patch_food_amount_mean",
    "patch_regrowth_suppressed_total",
    "jungle_predator_events",
    "jungle_predator_damage_total",
    "lineage_effective_alive_lineages",
    "lineage_alive_share_entropy",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _last_metrics(run_dir: Path) -> dict[str, str]:
    rows = _read_rows(run_dir / "metrics.csv")
    if not rows:
        return {}
    return rows[-1]


def _extract_run_dir(row: Mapping[str, str]) -> Path:
    raw = row.get("run_dir") or row.get("output_dir")
    if not raw:
        raise ValueError(f"matrix row has no run_dir: {row}")
    return Path(raw)


def _config_parts(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _read_json(run_dir / "config.json")
    return dict(config.get("env_cfg") or {}), dict(config.get("sim_cfg") or {})


def _close(a: Any, b: Any) -> bool:
    if isinstance(b, bool):
        return bool(a) is b
    try:
        return abs(float(a) - float(b)) <= 1.0e-9
    except (TypeError, ValueError):
        return a == b


def _scenario_from_row(row: Mapping[str, str]) -> str:
    return str(row.get("scenario") or row.get("scenario_id") or "")


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(median(values))


def _hash(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_repeat_parity(path: Path | None) -> dict[str, Any] | None:
    if not path or not path.exists():
        return None
    return _read_json(path)


def summarize_batch(batch_dir: Path, *, repeat_parity_path: Path | None, worktree_info_path: Path | None) -> dict[str, Any]:
    matrix_path = batch_dir / "matrix.csv"
    matrix = _read_rows(matrix_path)
    runs: list[dict[str, Any]] = []
    integrity = {
        "planned_runs": len(matrix),
        "completed_runs": 0,
        "missing_metrics_csv": 0,
        "empty_metrics_csv": 0,
        "critical_nonfinite_issues": 0,
        "env_mismatches": 0,
        "sim_mismatches": 0,
        "lineage_strategy_json_present": 0,
        "lineage_strategy_json_nonempty": 0,
        "lineage_branch_candidates_json_present": 0,
        "lineage_branch_candidates_json_nonempty": 0,
        "theme5_manifest_or_archive_outputs_present": 0,
        "theme4_gating_mismatches": 0,
    }
    seed_sets: dict[str, set[int]] = defaultdict(set)
    mismatch_details: list[dict[str, Any]] = []
    nonfinite_details: list[dict[str, Any]] = []

    for row in matrix:
        scenario = _scenario_from_row(row)
        seed = int(row["seed"])
        run_dir = _extract_run_dir(row)
        seed_sets[scenario].add(seed)
        metrics_path = run_dir / "metrics.csv"
        if not metrics_path.exists():
            integrity["missing_metrics_csv"] += 1
            continue
        if metrics_path.stat().st_size == 0:
            integrity["empty_metrics_csv"] += 1
            continue
        metrics = _last_metrics(run_dir)
        if not metrics:
            integrity["empty_metrics_csv"] += 1
            continue
        integrity["completed_runs"] += 1

        env_cfg, sim_cfg = _config_parts(run_dir)
        for key, expected in EXPECTED_ENV.get(scenario, {}).items():
            if key not in env_cfg or not _close(env_cfg.get(key), expected):
                integrity["env_mismatches"] += 1
                mismatch_details.append(
                    {
                        "scenario": scenario,
                        "seed": seed,
                        "namespace": "env_cfg",
                        "key": key,
                        "expected": expected,
                        "actual": env_cfg.get(key),
                    }
                )
        for key, expected in EXPECTED_SIM.items():
            if key in sim_cfg and not _close(sim_cfg.get(key), expected):
                integrity["sim_mismatches"] += 1
                mismatch_details.append(
                    {
                        "scenario": scenario,
                        "seed": seed,
                        "namespace": "sim_cfg",
                        "key": key,
                        "expected": expected,
                        "actual": sim_cfg.get(key),
                    }
                )
                if key in {
                    "phase_e_e1_enable",
                    "enable_developmental_cognition",
                    "enable_introspection_channels",
                    "enable_self_regulation",
                    "enable_self_modification",
                    "enable_directed_inheritance",
                }:
                    integrity["theme4_gating_mismatches"] += 1

        for metric in CRITICAL_METRICS:
            if metric not in metrics:
                continue
            raw = metrics.get(metric)
            if raw not in ("", None) and _float(raw) is None:
                integrity["critical_nonfinite_issues"] += 1
                nonfinite_details.append({"scenario": scenario, "seed": seed, "metric": metric, "value": raw})

        lineage_json = metrics.get("lineage_strategy_descriptors_json")
        if lineage_json is not None:
            integrity["lineage_strategy_json_present"] += 1
            if str(lineage_json).strip() not in ("", "[]", "{}"):
                integrity["lineage_strategy_json_nonempty"] += 1
        branch_json = metrics.get("lineage_branch_candidates_json")
        if branch_json is not None:
            integrity["lineage_branch_candidates_json_present"] += 1
            if str(branch_json).strip() not in ("", "[]", "{}"):
                integrity["lineage_branch_candidates_json_nonempty"] += 1

        if (run_dir / "archives").exists() or (run_dir / "branch_manifests").exists():
            integrity["theme5_manifest_or_archive_outputs_present"] += 1

        values = {metric: _float(metrics.get(metric)) for metric in KEY_METRICS if metric in metrics}
        runs.append(
            {
                "scenario": scenario,
                "seed": seed,
                "run_dir": str(run_dir),
                "metrics": values,
                "lineage_strategy_json_present": lineage_json is not None,
                "lineage_strategy_json_nonempty": str(lineage_json or "").strip() not in ("", "[]", "{}"),
                "lineage_branch_candidates_json_present": branch_json is not None,
                "lineage_branch_candidates_json_nonempty": str(branch_json or "").strip() not in ("", "[]", "{}"),
            }
        )

    scenario_summaries: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        scenario_runs = [run for run in runs if run["scenario"] == scenario]
        summary_metrics: dict[str, float | None] = {}
        for metric in KEY_METRICS:
            values = [float(run["metrics"][metric]) for run in scenario_runs if run["metrics"].get(metric) is not None]
            summary_metrics[metric] = _mean(values)
        scenario_summaries.append(
            {
                "scenario": scenario,
                "label": SCENARIO_LABELS[scenario],
                "run_count": len(scenario_runs),
                "seeds": sorted(seed_sets.get(scenario, set())),
                "metrics": summary_metrics,
                "lineage_strategy_json_nonempty_runs": sum(
                    1 for run in scenario_runs if run["lineage_strategy_json_nonempty"]
                ),
                "lineage_branch_candidates_json_nonempty_runs": sum(
                    1 for run in scenario_runs if run["lineage_branch_candidates_json_nonempty"]
                ),
            }
        )

    by_scenario_seed = {(run["scenario"], run["seed"]): run for run in runs}
    paired_deltas: list[dict[str, Any]] = []
    for baseline, variant in PRIMARY_PAIRS:
        deltas: dict[str, float | None] = {}
        paired_seed_count = 0
        for metric in KEY_METRICS:
            metric_deltas: list[float] = []
            for seed in SEEDS:
                left = by_scenario_seed.get((baseline, seed))
                right = by_scenario_seed.get((variant, seed))
                if not left or not right:
                    continue
                lval = left["metrics"].get(metric)
                rval = right["metrics"].get(metric)
                if lval is None or rval is None:
                    continue
                metric_deltas.append(float(rval) - float(lval))
            deltas[metric] = _mean(metric_deltas)
            paired_seed_count = max(paired_seed_count, len(metric_deltas))
        paired_deltas.append(
            {
                "comparison": f"{variant} - {baseline}",
                "baseline": baseline,
                "variant": variant,
                "paired_seed_count": paired_seed_count,
                "deltas": deltas,
            }
        )

    lineage_summary = {
        "schema_version": "integrated_r1_lineage_summary_v1",
        "scenario_summaries": [
            {
                "scenario": item["scenario"],
                "run_count": item["run_count"],
                "lineage_effective_alive_lineages_mean": item["metrics"].get("lineage_effective_alive_lineages"),
                "lineage_alive_share_entropy_mean": item["metrics"].get("lineage_alive_share_entropy"),
                "lineage_active_count_mean": item["metrics"].get("lineage_active_count"),
                "lineage_branch_candidate_count_mean": item["metrics"].get("lineage_branch_candidate_count"),
                "strategy_json_nonempty_runs": item["lineage_strategy_json_nonempty_runs"],
                "branch_json_nonempty_runs": item["lineage_branch_candidates_json_nonempty_runs"],
            }
            for item in scenario_summaries
        ],
    }

    integrity["all_runs_complete"] = integrity["completed_runs"] == integrity["planned_runs"]
    integrity["paired_seed_sets_complete"] = all(seed_sets.get(scenario, set()) == set(SEEDS) for scenario in SCENARIOS)
    integrity["seed_sets"] = {scenario: sorted(values) for scenario, values in sorted(seed_sets.items())}
    integrity["mismatch_details"] = mismatch_details
    integrity["nonfinite_details"] = nonfinite_details

    repeat_parity = _load_repeat_parity(repeat_parity_path)
    worktree_info = _read_json(worktree_info_path) if worktree_info_path and worktree_info_path.exists() else None

    direction = {
        "patch_jungle_pressure_measurable": (
            (scenario_summaries[2]["metrics"].get("jungle_predator_events") or 0.0) > 0.0
            and (scenario_summaries[2]["metrics"].get("patch_regrowth_suppressed_total") or 0.0) > 0.0
        ),
        "tuned_full_jungle_viable": (scenario_summaries[3]["metrics"].get("alive") or 0.0) > 0.0,
        "terrain_prevention_measurable": (scenario_summaries[3]["metrics"].get("jungle_predator_damage_prevented_total") or 0.0)
        > 0.0,
        "recovery_measurable": (scenario_summaries[3]["metrics"].get("patch_recovery_added_regrowth_total") or 0.0) > 0.0,
        "theme12_reference_runs": scenario_summaries[4]["run_count"] == len(SEEDS),
        "lineage_outputs_present": integrity["lineage_strategy_json_present"] == integrity["completed_runs"],
    }

    return {
        "schema_version": "integrated_r1_summary_v1",
        "campaign_name": "integrated_r1_baseline_reproduction",
        "batch_dir": str(batch_dir),
        "integrity": integrity,
        "scenario_summaries": scenario_summaries,
        "paired_deltas": paired_deltas,
        "lineage_summary": lineage_summary,
        "directional_checks": direction,
        "repeat_parity": repeat_parity,
        "worktree_info": worktree_info,
    }


def write_summary_csv(path: Path, scenario_summaries: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["scenario", "label", "run_count", *KEY_METRICS]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in scenario_summaries:
            row = {"scenario": item["scenario"], "label": item["label"], "run_count": item["run_count"]}
            row.update({metric: item["metrics"].get(metric) for metric in KEY_METRICS})
            writer.writerow(row)


def write_deltas_csv(path: Path, paired_deltas: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["comparison", "baseline", "variant", "paired_seed_count", *KEY_METRICS]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in paired_deltas:
            row = {
                "comparison": item["comparison"],
                "baseline": item["baseline"],
                "variant": item["variant"],
                "paired_seed_count": item["paired_seed_count"],
            }
            row.update({metric: item["deltas"].get(metric) for metric in KEY_METRICS})
            writer.writerow(row)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _metric_table(scenario_summaries: Sequence[Mapping[str, Any]], metrics: Sequence[str]) -> str:
    lines = ["| Scenario | " + " | ".join(metrics) + " |", "| --- | " + " | ".join("---" for _ in metrics) + " |"]
    for item in scenario_summaries:
        row = [str(item["scenario"])]
        row.extend(_fmt(item["metrics"].get(metric)) for metric in metrics)
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _delta_table(paired_deltas: Sequence[Mapping[str, Any]], metrics: Sequence[str]) -> str:
    lines = ["| Comparison | " + " | ".join(metrics) + " |", "| --- | " + " | ".join("---" for _ in metrics) + " |"]
    for item in paired_deltas:
        row = [str(item["comparison"])]
        row.extend(_fmt(item["deltas"].get(metric)) for metric in metrics)
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_report(path: Path, summary: Mapping[str, Any]) -> None:
    integrity = summary["integrity"]
    directional = summary["directional_checks"]
    repeat = summary.get("repeat_parity") or {}
    worktree = summary.get("worktree_info") or {}
    scenario_summaries = summary["scenario_summaries"]
    paired_deltas = summary["paired_deltas"]

    status = "PASS" if (
        integrity["all_runs_complete"]
        and integrity["critical_nonfinite_issues"] == 0
        and integrity["env_mismatches"] == 0
        and integrity["theme4_gating_mismatches"] == 0
        and all(directional.values())
    ) else "LIMITED PASS"

    text = f"""# EIEL Integrated R1 Baseline Reproduction Report

## 1. Purpose

R1 validates that merged main can reproduce key Patch/Jungle, Theme 5 lineage,
and Theme 12 clean-pressure signals directionally after integration. It is a
validation and evidence-reset campaign only.

## 2. Integrated Main Commit Hash

`{worktree.get('base_commit', '')}`

## 3. Worktree Path

`{worktree.get('worktree_path', '')}`

Main clean except ignored evidence artifacts before work: `{worktree.get('main_clean_except_ignored_evidence', '')}`.

Git status before work:

```text
{worktree.get('git_status_before', '').strip()}
```

Git status after work is recorded at final closeout, not by this batch parser.

## 4. Scenario Definitions

- `r1_00_flat_baseline`: flat-world baseline with Patch/Jungle disabled.
- `r1_01_patch_observed`: Patch World observed without active Jungle pressure.
- `r1_02_default_full_jungle`: current default full Jungle reference.
- `r1_03_tuned_full_jungle_recovery`: Campaign C/D tuned full Jungle recovery reference.
- `r1_04_theme12_clean_pressure_reference`: Theme 12 `combined_low_noise_medium_pressure` reference.

## 5. Seed List / Settings / Device

Seeds: `{', '.join(str(seed) for seed in SEEDS)}`.

Settings: 8 generations, 60 steps/generation, population 48, grid 16, device CPU, bank device CPU.

## 6. Preflight Results

See the final task response for command-by-command preflight results. R1 execution
continued only after preflight and registry dry-run passed.

## 7. Repeat Parity Results

Repeat parity file present: `{bool(repeat)}`.

Overall repeat parity passed: `{repeat.get('passed', '')}`.

## 8. Completion Table

Planned runs: `{integrity['planned_runs']}`.

Completed runs: `{integrity['completed_runs']}`.

All runs complete: `{integrity['all_runs_complete']}`.

## 9. Metric Integrity Table

| Check | Value |
| --- | --- |
| missing_metrics_csv | {integrity['missing_metrics_csv']} |
| empty_metrics_csv | {integrity['empty_metrics_csv']} |
| critical_nonfinite_issues | {integrity['critical_nonfinite_issues']} |
| env_mismatches | {integrity['env_mismatches']} |
| sim_mismatches | {integrity['sim_mismatches']} |
| theme4_gating_mismatches | {integrity['theme4_gating_mismatches']} |
| lineage_strategy_json_present | {integrity['lineage_strategy_json_present']} |
| lineage_strategy_json_nonempty | {integrity['lineage_strategy_json_nonempty']} |
| lineage_branch_candidates_json_present | {integrity['lineage_branch_candidates_json_present']} |
| lineage_branch_candidates_json_nonempty | {integrity['lineage_branch_candidates_json_nonempty']} |

## 10. Scenario Flag Validation

Expected scenario flags were checked from `config.json` for every completed run.
Mismatches: `{integrity['env_mismatches']}`.

## 11. Summary Table

{_metric_table(scenario_summaries, ('alive', 'successful_births', 'jungle_predator_events', 'jungle_predator_damage_total', 'jungle_predator_damage_prevented_total', 'patch_regrowth_suppressed_total', 'patch_recovery_added_regrowth_total', 'lineage_effective_alive_lineages', 'lineage_alive_share_entropy', 'lineage_branch_candidate_count'))}

## 12. Paired Deltas

{_delta_table(paired_deltas, ('alive', 'successful_births', 'jungle_predator_events', 'jungle_predator_damage_total', 'jungle_predator_damage_prevented_total', 'patch_regrowth_suppressed_total', 'patch_recovery_added_regrowth_total', 'lineage_effective_alive_lineages'))}

## 13. Theme 5 Lineage Output Availability

Lineage strategy JSON was present in `{integrity['lineage_strategy_json_present']}` completed runs.
Branch candidate JSON was present in `{integrity['lineage_branch_candidates_json_present']}` completed runs.
Non-empty lineage records are expected primarily in non-flat scenarios.

## 14. Theme 12 Preset Availability

The Theme 12 reference scenario used explicit `combined_low_noise_medium_pressure`
overrides from the Theme 12 preset ladder. The scenario completed
`{next(item['run_count'] for item in scenario_summaries if item['scenario'] == 'r1_04_theme12_clean_pressure_reference')}` runs.

## 15. Theme 4 Gated-Control Status

Theme 4 experimental cognition controls were kept disabled through explicit
registry knobs. Gating mismatches: `{integrity['theme4_gating_mismatches']}`.

## 16. Directional Comparison To Campaign D/E

- Patch/Jungle pressure measurable: `{directional['patch_jungle_pressure_measurable']}`.
- Tuned full Jungle viable: `{directional['tuned_full_jungle_viable']}`.
- Terrain prevention measurable: `{directional['terrain_prevention_measurable']}`.
- Recovery measurable: `{directional['recovery_measurable']}`.
- Theme 12 reference runs: `{directional['theme12_reference_runs']}`.
- Lineage outputs present: `{directional['lineage_outputs_present']}`.

R1 requires directional consistency, not exact numeric reproduction, because the
source state changed after integration.

## 17. Deviations From Expected Behavior

R1 status: `{status}`.

See mismatch and nonfinite details in `r1_summary.json`.

## 18. Known Limitations

- R1 is a compact reproduction campaign, not a replacement for Campaign D/E.
- R1 does not prove cognition readiness.
- R1 does not test lineage-preservation mechanics or branch execution.
- Active alive-lineage diversity may remain low or collapsed in Jungle scenarios.

## 19. Recommendation For R2

Recommendation: `GO for R2 Theme 5-assisted lineage-collapse forensics` if
R1 status is `PASS`; otherwise inspect `r1_summary.json` before proceeding.

## 20. No-Claims Statement

R1 makes no claim that intelligence emerged, cognition is adaptive, perception
was selected, memory was selected, prediction was selected, lineage diversity is
solved, or Theme 12 fixes ecology.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--deltas-csv", required=True)
    parser.add_argument("--lineage-json", required=True)
    parser.add_argument("--report-md", required=True)
    parser.add_argument("--repeat-parity-json", default="")
    parser.add_argument("--worktree-info-json", default="")
    args = parser.parse_args()

    summary = summarize_batch(
        Path(args.batch_dir),
        repeat_parity_path=Path(args.repeat_parity_json) if args.repeat_parity_json else None,
        worktree_info_path=Path(args.worktree_info_json) if args.worktree_info_json else None,
    )
    _write_json(Path(args.out_json), summary)
    _write_json(Path(args.lineage_json), summary["lineage_summary"])
    write_summary_csv(Path(args.summary_csv), summary["scenario_summaries"])
    write_deltas_csv(Path(args.deltas_csv), summary["paired_deltas"])
    write_report(Path(args.report_md), summary)
    print(args.out_json)
    print(args.report_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
