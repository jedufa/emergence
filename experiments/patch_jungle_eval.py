"""Patch/Jungle V0 Step 14 evaluation matrix helpers.

This module is diagnostic infrastructure only. It defines standard short-run
scenario metadata, metric groups, deterministic manifest generation, and simple
metric comparison helpers. It does not import or mutate simulation state.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


MATRIX_VERSION = 1


csv.field_size_limit(sys.maxsize)


PATCH_JUNGLE_FLAGS: tuple[str, ...] = (
    "patch_world_enable",
    "patch_world_observation_enable",
    "patch_world_waste_consequence_enable",
    "jungle_predator_pressure_enable",
    "jungle_reproduction_emissions_enable",
    "jungle_juvenile_predator_vulnerability_enable",
    "jungle_terrain_protection_enable",
    "jungle_ecological_recovery_enable",
)


@dataclass(frozen=True)
class ScenarioDefinition:
    """One standard Patch/Jungle comparison scenario."""

    scenario_id: str
    label: str
    description: str
    primary_question: str
    overrides: tuple[tuple[str, bool | int | float | str], ...]
    metric_groups: tuple[str, ...]
    smoke_generations: int = 2
    smoke_steps: int = 12
    smoke_population: int = 16
    full_protocol_note: str = "Run paired multi-seed experiments before drawing conclusions."


@dataclass(frozen=True)
class MetricGroup:
    """A stable metric group used by Step 14 summaries."""

    name: str
    metrics: tuple[str, ...]


def _flag_overrides(
    *,
    patch_world: bool,
    observation: bool = False,
    waste: bool = False,
    predator: bool = False,
    reproduction: bool = False,
    terrain: bool = False,
    recovery: bool = False,
    juvenile: bool = False,
    extra: Sequence[tuple[str, bool | int | float | str]] = (),
) -> tuple[tuple[str, bool | int | float | str], ...]:
    values: list[tuple[str, bool | int | float | str]] = [
        ("patch_world_enable", patch_world),
        ("patch_world_observation_enable", observation),
        ("patch_world_waste_consequence_enable", waste),
        ("jungle_predator_pressure_enable", predator),
        ("jungle_reproduction_emissions_enable", reproduction),
        ("jungle_juvenile_predator_vulnerability_enable", juvenile),
        ("jungle_terrain_protection_enable", terrain),
        ("jungle_ecological_recovery_enable", recovery),
        ("patch_movement_cost_apply_enable", False),
    ]
    values.extend(extra)
    return tuple(values)


def scenario_registry() -> tuple[ScenarioDefinition, ...]:
    """Return the deterministic Step 14 Patch/Jungle V0 scenario registry."""

    all_groups = tuple(group.name for group in metric_groups())
    jungle_terrain_defaults = (
        ("patch_cover_default", 0.25),
        ("patch_shelter_default", 0.15),
        ("patch_shelter_from_cover_enable", False),
    )
    terrain_only_defaults = (
        ("patch_cover_default", 0.25),
        ("patch_shelter_default", 0.15),
        ("patch_shelter_from_cover_enable", False),
    )
    return (
        ScenarioDefinition(
            scenario_id="s00_flat_baseline",
            label="Flat baseline",
            description="Legacy flat world with Patch World and Jungle features disabled.",
            primary_question="What does reproduction and selection look like in the old flat world?",
            overrides=_flag_overrides(patch_world=False),
            metric_groups=all_groups,
        ),
        ScenarioDefinition(
            scenario_id="s01_patch_food_only",
            label="Patch food only",
            description="Persistent patch food amount, depletion, regrowth, and fertility diagnostics.",
            primary_question="Does persistent food substrate change resource dynamics without Jungle pressure?",
            overrides=_flag_overrides(patch_world=True),
            metric_groups=all_groups,
        ),
        ScenarioDefinition(
            scenario_id="s02_patch_food_terrain",
            label="Patch food plus terrain",
            description="Patch substrate with nonzero passive cover and shelter defaults for diagnostics.",
            primary_question="Does terrain affect diagnostics and habitat descriptors before pressure?",
            overrides=_flag_overrides(patch_world=True, extra=terrain_only_defaults),
            metric_groups=all_groups,
        ),
        ScenarioDefinition(
            scenario_id="s03_patch_observed",
            label="Patch substrate plus observations",
            description="Patch World state with local normalized Patch observation inputs enabled.",
            primary_question="Do patch observations alter behavior compared with hidden Patch World state?",
            overrides=_flag_overrides(patch_world=True, observation=True),
            metric_groups=all_groups,
        ),
        ScenarioDefinition(
            scenario_id="s04_patch_waste_crowding",
            label="Patch waste and crowding consequences",
            description="Patch observations plus waste/density consequences on food recovery.",
            primary_question="Does overuse become costly without predators?",
            overrides=_flag_overrides(patch_world=True, observation=True, waste=True),
            metric_groups=all_groups,
        ),
        ScenarioDefinition(
            scenario_id="s05_predator_pressure",
            label="Predator pressure",
            description="Jungle predator-risk field on Patch World without birth emissions or recovery.",
            primary_question="Does predator pressure correlate with scent, waste, density, and exposure?",
            overrides=_flag_overrides(
                patch_world=True,
                observation=True,
                predator=True,
                extra=jungle_terrain_defaults,
            ),
            metric_groups=all_groups,
        ),
        ScenarioDefinition(
            scenario_id="s06_predator_reproduction_emissions",
            label="Predator plus reproduction emissions",
            description="Predator pressure with successful births emitting scent and waste.",
            primary_question="Does reproduction context begin to matter indirectly?",
            overrides=_flag_overrides(
                patch_world=True,
                observation=True,
                predator=True,
                reproduction=True,
                extra=jungle_terrain_defaults,
            ),
            metric_groups=all_groups,
        ),
        ScenarioDefinition(
            scenario_id="s07_predator_terrain_protection",
            label="Predator plus terrain protection",
            description="Predator pressure with bounded cover/shelter damage mitigation enabled.",
            primary_question="Does shelter and cover improve survival under predator pressure?",
            overrides=_flag_overrides(
                patch_world=True,
                observation=True,
                predator=True,
                terrain=True,
                extra=jungle_terrain_defaults,
            ),
            metric_groups=all_groups,
        ),
        ScenarioDefinition(
            scenario_id="s08_full_jungle_no_recovery",
            label="Full Jungle V0 without recovery",
            description="Patch observations, waste consequences, predator pressure, birth emissions, and terrain protection.",
            primary_question="Does the world become too harsh or meaningfully selective?",
            overrides=_flag_overrides(
                patch_world=True,
                observation=True,
                waste=True,
                predator=True,
                reproduction=True,
                terrain=True,
                recovery=False,
                extra=jungle_terrain_defaults,
            ),
            metric_groups=all_groups,
        ),
        ScenarioDefinition(
            scenario_id="s09_full_jungle_recovery",
            label="Full Jungle V0 with recovery",
            description="Full Jungle V0 stack with bounded ecological recovery enabled.",
            primary_question="Does recovery stabilize ecology without erasing pressure?",
            overrides=_flag_overrides(
                patch_world=True,
                observation=True,
                waste=True,
                predator=True,
                reproduction=True,
                terrain=True,
                recovery=True,
                extra=jungle_terrain_defaults,
            ),
            metric_groups=all_groups,
        ),
    )


def metric_groups() -> tuple[MetricGroup, ...]:
    """Return stable Patch/Jungle evaluation metric groups."""

    return (
        MetricGroup(
            "population_survival",
            (
                "alive",
                "cohort_survival_rate",
                "deaths",
                "mean_survival_steps_total",
                "diversity_collapse",
            ),
        ),
        MetricGroup(
            "reproduction",
            (
                "successful_births",
                "failed_births",
                "birth_attempts",
                "first_repro_step_mean_alive",
                "first_repro_frac_alive",
                "birth_patch_count",
                "birth_patch_cover_mean",
                "birth_patch_waste_mean",
                "lineage_top1_birth_share",
            ),
        ),
        MetricGroup(
            "patch_use",
            (
                "occupied_patch_food_amount_mean",
                "occupied_patch_cover_mean",
                "occupied_patch_shelter_mean",
                "occupied_patch_scent_mean",
                "occupied_patch_waste_mean",
                "occupied_patch_density_mean",
                "patch_recovery_activity_occupied_mean",
                "patch_cover_occupancy_bias",
                "patch_shelter_occupancy_bias",
                "patch_waste_occupancy_bias",
            ),
        ),
        MetricGroup(
            "food_ecology",
            (
                "patch_food_amount_mean",
                "patch_food_depleted_frac",
                "patch_food_regrowth_total",
                "patch_regrowth_effective_total",
                "patch_regrowth_suppressed_total",
                "patch_recovery_added_regrowth_total",
            ),
        ),
        MetricGroup(
            "waste_crowding",
            (
                "patch_waste_mean",
                "patch_waste_p90",
                "patch_density_mean",
                "patch_density_p90",
                "patch_waste_regrowth_penalty_mean",
                "patch_density_regrowth_penalty_mean",
            ),
        ),
        MetricGroup(
            "predator_pressure",
            (
                "jungle_predator_risk_mean",
                "jungle_predator_occupied_risk_mean",
                "jungle_predator_events",
                "jungle_predator_damage_total",
                "predator_event_patch_scent_mean",
                "predator_event_patch_waste_mean",
                "predator_event_patch_density_mean",
                "predator_event_patch_cover_mean",
                "predator_event_patch_shelter_mean",
            ),
        ),
        MetricGroup(
            "terrain_protection",
            (
                "jungle_predator_damage_raw_total",
                "jungle_predator_damage_effective_total",
                "jungle_predator_damage_prevented_total",
                "jungle_predator_event_damage_multiplier_mean",
                "predator_event_patch_cover_mean",
                "predator_event_patch_shelter_mean",
            ),
        ),
        MetricGroup(
            "recovery",
            (
                "patch_recovery_activity_mean",
                "patch_recovery_activity_occupied_mean",
                "jungle_recovery_regrowth_boost_total",
                "jungle_recovery_regrowth_boost_mean",
                "patch_recovery_added_regrowth_total",
            ),
        ),
        MetricGroup(
            "lineage_diversity",
            (
                "lineage_active_count",
                "lineage_top1_alive_share",
                "lineage_top3_alive_share",
                "lineage_alive_share_entropy",
                "lineage_effective_alive_lineages",
                "lineage_patch_cover_mean_std",
                "lineage_patch_waste_mean_std",
                "lineage_patch_recovery_activity_mean_std",
                "lineage_branch_candidate_count",
            ),
        ),
        MetricGroup(
            "existing_cognition_diagnostics",
            (
                "gru_alive_frac",
                "mean_memory_dim_alive",
                "drive_fear_mean",
                "late_pred_err_mean_alive",
                "coh_pred_err_mean",
                "identity_drift_mean",
            ),
        ),
    )


def default_comparison_pairs() -> tuple[tuple[str, str], ...]:
    """Return recommended raw-delta scenario comparisons as (baseline, variant)."""

    return (
        ("s00_flat_baseline", "s01_patch_food_only"),
        ("s01_patch_food_only", "s03_patch_observed"),
        ("s03_patch_observed", "s04_patch_waste_crowding"),
        ("s03_patch_observed", "s05_predator_pressure"),
        ("s05_predator_pressure", "s06_predator_reproduction_emissions"),
        ("s05_predator_pressure", "s07_predator_terrain_protection"),
        ("s08_full_jungle_no_recovery", "s09_full_jungle_recovery"),
    )


def scenario_by_id(scenario_id: str) -> ScenarioDefinition:
    for scenario in scenario_registry():
        if scenario.scenario_id == scenario_id:
            return scenario
    known = ", ".join(s.scenario_id for s in scenario_registry())
    raise KeyError(f"unknown Patch/Jungle scenario id: {scenario_id!r}; known: {known}")


def _canonical_value(value: bool | int | float | str) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def scenario_knob_pairs(scenario: ScenarioDefinition | str) -> tuple[str, ...]:
    """Return canonical `env_cfg.field=value` knob pairs for a scenario."""

    item = scenario_by_id(scenario) if isinstance(scenario, str) else scenario
    return tuple(f"env_cfg.{key}={_canonical_value(value)}" for key, value in item.overrides)


def scenario_cli_tokens(scenario: ScenarioDefinition | str) -> tuple[str, ...]:
    """Return flat CLI tokens for the harness registry axis flags."""

    tokens: list[str] = []
    for pair in scenario_knob_pairs(scenario):
        tokens.extend(["--knob", pair])
    return tuple(tokens)


def scenario_overrides_dict(scenario: ScenarioDefinition | str) -> dict[str, bool | int | float | str]:
    """Return ordered scenario overrides as a plain mapping."""

    item = scenario_by_id(scenario) if isinstance(scenario, str) else scenario
    return {key: value for key, value in item.overrides}


def all_metric_names(groups: Sequence[MetricGroup] | None = None) -> tuple[str, ...]:
    """Return unique metric names in stable group order."""

    names: list[str] = []
    seen: set[str] = set()
    for group in groups or metric_groups():
        for metric in group.metrics:
            if metric not in seen:
                names.append(metric)
                seen.add(metric)
    return tuple(names)


def config_hash_for_overrides(overrides: Mapping[str, Any]) -> str:
    """Return a deterministic short hash for scenario override content."""

    payload = json.dumps(dict(overrides), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def build_paired_manifest(
    *,
    scenario_ids: Sequence[str] | None = None,
    seeds: Sequence[int] = (43,),
    run_length: Mapping[str, int] | None = None,
    output_dir: str = "runs/experiments/patch_jungle_v0",
    include_created_timestamp: bool = False,
    created_timestamp: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic paired-run manifest dictionary.

    Wall-clock timestamps are omitted by default so fixture tests stay stable.
    """

    selected = tuple(scenario_ids or [s.scenario_id for s in scenario_registry()])
    scenarios = [scenario_by_id(sid) for sid in selected]
    seed_values = [int(seed) for seed in seeds]
    length = dict(run_length or {"generations": 2, "steps": 12})
    manifest: dict[str, Any] = {
        "experiment_matrix_version": MATRIX_VERSION,
        "scenario_ids": [s.scenario_id for s in scenarios],
        "seeds": seed_values,
        "run_length": length,
        "output_dir": str(output_dir),
        "metric_groups": {group.name: list(group.metrics) for group in metric_groups()},
        "comparison_pairs": [list(pair) for pair in default_comparison_pairs() if pair[0] in selected and pair[1] in selected],
        "scenarios": [
            {
                "scenario_id": s.scenario_id,
                "label": s.label,
                "description": s.description,
                "primary_question": s.primary_question,
                "overrides": scenario_overrides_dict(s),
                "config_hash": config_hash_for_overrides(scenario_overrides_dict(s)),
            }
            for s in scenarios
        ],
    }
    if include_created_timestamp:
        manifest["created_timestamp"] = created_timestamp or ""
    return manifest


def write_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    """Write a deterministic JSON manifest."""

    Path(path).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def summarize_metrics_csv(
    path: str | Path,
    metric_names: Sequence[str],
    *,
    tail_n: int = 1,
) -> dict[str, Any]:
    """Summarize selected metrics from a metrics.csv file.

    The default uses the final row only. Larger ``tail_n`` values average the
    final N available rows for later full-experiment summaries.
    """

    csv_path = Path(path)
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {"path": str(csv_path), "rows": 0, "metrics": {}, "missing_metrics": list(metric_names)}

    n = max(1, int(tail_n))
    tail = rows[-n:]
    metrics: dict[str, float] = {}
    missing: list[str] = []
    for metric in metric_names:
        values: list[float] = []
        present = False
        for row in tail:
            if metric in row:
                present = True
                parsed = _parse_float(row.get(metric))
                if parsed is not None:
                    values.append(parsed)
        if values:
            metrics[metric] = float(sum(values) / len(values))
        elif not present:
            missing.append(metric)
        else:
            missing.append(metric)
    return {
        "path": str(csv_path),
        "rows": len(rows),
        "tail_n": min(n, len(rows)),
        "metrics": metrics,
        "missing_metrics": missing,
    }


def compare_metric_summaries(
    *,
    baseline_id: str,
    baseline: Mapping[str, Any],
    variant_id: str,
    variant: Mapping[str, Any],
    metric_names: Sequence[str],
) -> dict[str, Any]:
    """Compute raw metric deltas for two summaries."""

    base_metrics = dict(baseline.get("metrics", {}) or {})
    var_metrics = dict(variant.get("metrics", {}) or {})
    deltas: dict[str, float] = {}
    missing: list[str] = []
    for metric in metric_names:
        if metric not in base_metrics or metric not in var_metrics:
            missing.append(metric)
            continue
        base_value = _parse_float(base_metrics[metric])
        var_value = _parse_float(var_metrics[metric])
        if base_value is None or var_value is None:
            missing.append(metric)
            continue
        deltas[metric] = float(var_value - base_value)
    return {
        "baseline_id": baseline_id,
        "variant_id": variant_id,
        "comparison_id": f"{variant_id}_minus_{baseline_id}",
        "deltas": deltas,
        "missing_metrics": missing,
    }


def summarize_batch(
    batch_dir: str | Path,
    *,
    groups: Sequence[MetricGroup] | None = None,
    comparison_pairs: Sequence[tuple[str, str]] | None = None,
    tail_n: int = 1,
) -> dict[str, Any]:
    """Summarize a registry batch directory produced by the harness."""

    root = Path(batch_dir)
    matrix_path = root / "matrix.csv"
    metric_names = all_metric_names(groups)
    with matrix_path.open("r", newline="", encoding="utf-8") as f:
        matrix_rows = list(csv.DictReader(f))

    by_scenario: dict[str, list[dict[str, Any]]] = {}
    missing_runs: list[dict[str, str]] = []
    for row in matrix_rows:
        scenario_id = str(row.get("scenario") or row.get("scenario_id") or "")
        run_dir = Path(str(row.get("run_dir") or ""))
        metrics_path = run_dir / "metrics.csv"
        if not scenario_id:
            continue
        if not metrics_path.exists():
            missing_runs.append({"scenario_id": scenario_id, "run_dir": str(run_dir)})
            continue
        summary = summarize_metrics_csv(metrics_path, metric_names, tail_n=tail_n)
        summary["run_id"] = row.get("run_id") or ""
        summary["runtime_run_id"] = row.get("runtime_run_id") or ""
        summary["seed"] = int(row.get("seed") or 0)
        by_scenario.setdefault(scenario_id, []).append(summary)

    scenario_summaries: dict[str, dict[str, Any]] = {}
    for scenario_id in sorted(by_scenario):
        entries = sorted(by_scenario[scenario_id], key=lambda item: int(item.get("seed", 0)))
        combined: dict[str, list[float]] = {metric: [] for metric in metric_names}
        missing: set[str] = set()
        for entry in entries:
            for metric in metric_names:
                value = _parse_float((entry.get("metrics", {}) or {}).get(metric))
                if value is None:
                    missing.add(metric)
                else:
                    combined[metric].append(value)
        metrics = {
            metric: float(sum(values) / len(values))
            for metric, values in combined.items()
            if values
        }
        scenario_summaries[scenario_id] = {
            "scenario_id": scenario_id,
            "run_count": len(entries),
            "seeds": [int(entry.get("seed", 0)) for entry in entries],
            "metrics": metrics,
            "missing_metrics": sorted(missing),
            "runs": entries,
        }

    pairs = tuple(comparison_pairs or default_comparison_pairs())
    comparisons: list[dict[str, Any]] = []
    for baseline_id, variant_id in pairs:
        if baseline_id in scenario_summaries and variant_id in scenario_summaries:
            comparisons.append(
                compare_metric_summaries(
                    baseline_id=baseline_id,
                    baseline=scenario_summaries[baseline_id],
                    variant_id=variant_id,
                    variant=scenario_summaries[variant_id],
                    metric_names=metric_names,
                )
            )

    return {
        "experiment_matrix_version": MATRIX_VERSION,
        "batch_dir": str(root),
        "metric_groups": {group.name: list(group.metrics) for group in groups or metric_groups()},
        "scenario_summaries": scenario_summaries,
        "comparisons": comparisons,
        "missing_runs": missing_runs,
    }


def write_summary_json(path: str | Path, summary: Mapping[str, Any]) -> None:
    """Write a deterministic JSON comparison summary."""

    Path(path).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_metric_groups_against_schema(schema_fields: Iterable[str]) -> dict[str, Any]:
    """Return missing Step 14 metric group names against a known metrics schema."""

    available = set(schema_fields)
    missing = [metric for metric in all_metric_names() if metric not in available]
    return {
        "ok": not missing,
        "missing_metrics": missing,
        "checked_metrics": list(all_metric_names()),
    }
