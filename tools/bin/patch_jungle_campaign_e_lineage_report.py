#!/usr/bin/env python3
"""Summarize Patch/Jungle V0 Campaign E lineage/strategy triage batches.

This is reporting-only infrastructure. It reads completed registry batch
directories and writes deterministic JSON, CSV, and Markdown summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence


csv.field_size_limit(sys.maxsize)

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
    "e00_flat_baseline",
    "e01_patch_observed",
    "e02_default_full_jungle",
    "e03_tuned_full_jungle_no_recovery",
    "e04_tuned_full_jungle_recovery",
)

SCENARIO_LABELS: dict[str, str] = {
    "e00_flat_baseline": "Flat baseline",
    "e01_patch_observed": "Patch observed",
    "e02_default_full_jungle": "Default full Jungle recovery",
    "e03_tuned_full_jungle_no_recovery": "Tuned full Jungle without recovery",
    "e04_tuned_full_jungle_recovery": "Tuned full Jungle with recovery",
}

EXPECTED_ENV: dict[str, dict[str, bool | float]] = {
    "e00_flat_baseline": {
        "patch_world_enable": False,
        "patch_world_observation_enable": False,
        "patch_world_waste_consequence_enable": False,
        "jungle_predator_pressure_enable": False,
        "jungle_reproduction_emissions_enable": False,
        "jungle_terrain_protection_enable": False,
        "jungle_ecological_recovery_enable": False,
    },
    "e01_patch_observed": {
        "patch_world_enable": True,
        "patch_world_observation_enable": True,
        "patch_world_waste_consequence_enable": False,
        "jungle_predator_pressure_enable": False,
        "jungle_reproduction_emissions_enable": False,
        "jungle_terrain_protection_enable": False,
        "jungle_ecological_recovery_enable": False,
    },
    "e02_default_full_jungle": {
        "patch_world_enable": True,
        "patch_world_observation_enable": True,
        "patch_world_waste_consequence_enable": True,
        "jungle_predator_pressure_enable": True,
        "jungle_reproduction_emissions_enable": True,
        "jungle_terrain_protection_enable": True,
        "jungle_ecological_recovery_enable": True,
    },
    "e03_tuned_full_jungle_no_recovery": {
        "patch_world_enable": True,
        "patch_world_observation_enable": True,
        "patch_world_waste_consequence_enable": True,
        "jungle_predator_pressure_enable": True,
        "jungle_reproduction_emissions_enable": True,
        "jungle_terrain_protection_enable": True,
        "jungle_ecological_recovery_enable": False,
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
    },
    "e04_tuned_full_jungle_recovery": {
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
}

EXPECTED_SIM: dict[str, str] = {"device": "cuda", "bank_device": "cuda"}

PRIMARY_PAIRS: tuple[tuple[str, str], ...] = (
    ("e00_flat_baseline", "e01_patch_observed"),
    ("e01_patch_observed", "e02_default_full_jungle"),
    ("e01_patch_observed", "e03_tuned_full_jungle_no_recovery"),
    ("e03_tuned_full_jungle_no_recovery", "e04_tuned_full_jungle_recovery"),
    ("e02_default_full_jungle", "e04_tuned_full_jungle_recovery"),
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
    "birth_patch_cover_mean",
    "birth_patch_shelter_mean",
    "birth_patch_waste_mean",
    "birth_patch_density_mean",
    "birth_patch_scent_mean",
    "birth_patch_recovery_activity_mean",
    "occupied_patch_food_amount_mean",
    "occupied_patch_cover_mean",
    "occupied_patch_shelter_mean",
    "occupied_patch_scent_mean",
    "occupied_patch_waste_mean",
    "occupied_patch_density_mean",
    "occupied_patch_recovery_activity_mean",
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
    "lineage_top1_alive_share",
    "lineage_alive_share_entropy",
    "lineage_effective_alive_lineages",
)

HABITAT_FIELDS: tuple[str, ...] = (
    "lineage_patch_food_amount_mean",
    "lineage_patch_cover_mean",
    "lineage_patch_shelter_mean",
    "lineage_patch_scent_mean",
    "lineage_patch_waste_mean",
    "lineage_patch_density_mean",
    "lineage_patch_recovery_activity_mean",
    "lineage_patch_movement_cost_mean",
)

PREDATOR_FIELDS: tuple[str, ...] = (
    "lineage_predator_event_count",
    "lineage_predator_damage_total",
    "lineage_predator_damage_mean",
    "lineage_predator_damage_prevented_total",
    "lineage_predator_event_patch_cover_mean",
    "lineage_predator_event_patch_shelter_mean",
    "lineage_predator_event_patch_scent_mean",
    "lineage_predator_event_patch_waste_mean",
    "lineage_predator_event_patch_density_mean",
)

REPRO_FIELDS: tuple[str, ...] = (
    "lineage_birth_count",
    "lineage_birth_share",
    "lineage_birth_patch_cover_mean",
    "lineage_birth_patch_shelter_mean",
    "lineage_birth_patch_waste_mean",
    "lineage_birth_patch_density_mean",
    "lineage_birth_patch_scent_mean",
    "lineage_birth_patch_recovery_activity_mean",
    "lineage_reproduction_rate",
)

RECOVERY_FIELDS: tuple[str, ...] = (
    "lineage_patch_recovery_activity_mean",
    "lineage_birth_patch_recovery_activity_mean",
    "lineage_alive_share",
)

PROFILE_FIELDS: tuple[str, ...] = (
    "lineage_alive_share",
    "lineage_birth_share",
    "lineage_alive_count",
    "lineage_birth_count",
    *HABITAT_FIELDS,
    *PREDATOR_FIELDS,
    *REPRO_FIELDS,
)

GROUPS: dict[str, tuple[str, ...]] = {
    "population_survival": ("alive", "deaths", "cohort_survival_rate", "mean_survival_steps_total"),
    "reproduction": (
        "successful_births",
        "birth_attempts",
        "birth_patch_waste_mean",
        "birth_patch_shelter_mean",
        "birth_patch_recovery_activity_mean",
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


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Patch/Jungle V0 Campaign E batches")
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--analysis-json", required=True)
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--deltas-csv", required=True)
    parser.add_argument("--lineage-json", required=True)
    parser.add_argument("--branch-json", required=True)
    parser.add_argument("--report-md", required=True)
    parser.add_argument("--repeat-parity-json", default="")
    parser.add_argument("--campaign-name", default="patch_jungle_v0_campaign_e_lineage_strategy_triage")
    parser.add_argument("--output-root", default="/tmp/emergence_patch_jungle_campaign_e")
    return parser.parse_args(argv)


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not str(path) or not p.exists():
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


def _parse_int(value: Any) -> int | None:
    parsed = _parse_float(value)
    if parsed is None:
        return None
    return int(parsed)


def _mean(values: Iterable[float]) -> float | None:
    items = list(values)
    if not items:
        return None
    return float(sum(items) / len(items))


def _median(values: Iterable[float]) -> float | None:
    items = sorted(values)
    if not items:
        return None
    return float(median(items))


def _fmt(value: Any, digits: int = 3) -> str:
    parsed = _parse_float(value)
    if parsed is None:
        return "" if value is None else str(value)
    return f"{parsed:.{digits}f}"


def _parse_json_list(value: str | None) -> list[dict[str, Any]]:
    if value is None or not str(value).strip():
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def _field(row: Mapping[str, Any], name: str) -> float | None:
    return _parse_float(row.get(name))


def _first_generation(rows: Sequence[Mapping[str, str]], metric: str, threshold: float, op: str) -> int | None:
    for idx, row in enumerate(rows):
        value = _field(row, metric)
        if value is None:
            continue
        matched = value <= threshold if op == "<=" else value >= threshold
        if matched:
            return _parse_int(row.get("generation")) if _parse_int(row.get("generation")) is not None else idx
    return None


def _env_mismatches(scenario: str, config_path: Path) -> list[dict[str, Any]]:
    if not config_path.exists():
        return [{"scenario": scenario, "field": "config.json", "expected": "present", "actual": "missing"}]
    config = _read_json(config_path)
    env = dict(config.get("env_cfg") or {})
    sim = dict(config.get("sim_cfg") or {})
    mismatches: list[dict[str, Any]] = []
    for field, expected in EXPECTED_ENV.get(scenario, {}).items():
        if field not in env:
            mismatches.append({"scenario": scenario, "field": field, "expected": expected, "actual": "missing"})
            continue
        actual = env[field]
        if isinstance(expected, bool):
            ok = bool(actual) is expected
        else:
            ok = abs(float(actual) - float(expected)) <= 1e-12
        if not ok:
            mismatches.append({"scenario": scenario, "field": field, "expected": expected, "actual": actual})
    for field, expected in EXPECTED_SIM.items():
        actual = str(sim.get(field, ""))
        if actual != expected:
            mismatches.append({"scenario": scenario, "field": f"sim_cfg.{field}", "expected": expected, "actual": actual})
    return mismatches


def _metrics_from_final(final_row: Mapping[str, str], metric_names: Sequence[str]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for metric in metric_names:
        parsed = _parse_float(final_row.get(metric))
        if parsed is not None:
            metrics[metric] = parsed
    return metrics


def _dominant_descriptor(descriptors: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if not descriptors:
        return None
    return dict(
        max(
            descriptors,
            key=lambda item: (
                _parse_float(item.get("lineage_alive_share")) or 0.0,
                _parse_float(item.get("lineage_alive_count")) or 0.0,
                str(item.get("lineage_id") or ""),
            ),
        )
    )


def _profile(descriptor: Mapping[str, Any] | None) -> dict[str, Any]:
    if not descriptor:
        return {}
    out: dict[str, Any] = {"lineage_id": descriptor.get("lineage_id", "")}
    for field in PROFILE_FIELDS:
        parsed = _parse_float(descriptor.get(field))
        if parsed is not None:
            out[field] = parsed
    return out


def _ranges(descriptors: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    active = [item for item in descriptors if (_parse_float(item.get("lineage_alive_count")) or 0.0) > 0.0]
    for field in fields:
        values = [_parse_float(item.get(field)) for item in active]
        clean = [float(value) for value in values if value is not None]
        if len(clean) >= 2:
            out[f"{field}_range"] = max(clean) - min(clean)
    return out


def _aggregate_ranges(rows: Sequence[Mapping[str, str]], fields: Sequence[str]) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        ranges = _ranges(_parse_json_list(row.get("lineage_strategy_descriptors_json")), fields)
        for field, value in ranges.items():
            values[field].append(value)
    return {field: mean for field, items in values.items() if (mean := _mean(items)) is not None}


def _late_rows(rows: Sequence[Mapping[str, str]]) -> list[Mapping[str, str]]:
    if not rows:
        return []
    size = max(1, min(6, math.ceil(len(rows) * 0.25)))
    return list(rows[-size:])


def _pre_collapse_rows(rows: Sequence[Mapping[str, str]], collapse_generation: int | None) -> list[Mapping[str, str]]:
    if collapse_generation is None:
        return list(rows)
    selected = [
        row
        for row in rows
        if (_parse_int(row.get("generation")) if _parse_int(row.get("generation")) is not None else -1)
        < collapse_generation
    ]
    return selected or list(rows[:1])


def _collapse_timing(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    final = rows[-1] if rows else {}
    timing = {
        "first_effective_alive_lineages_le_1_5_generation": _first_generation(
            rows, "lineage_effective_alive_lineages", 1.5, "<="
        ),
        "first_entropy_le_0_5_generation": _first_generation(rows, "lineage_alive_share_entropy", 0.5, "<="),
        "first_top1_share_ge_0_75_generation": _first_generation(rows, "lineage_top1_alive_share", 0.75, ">="),
        "first_active_count_le_1_generation": _first_generation(rows, "lineage_active_count", 1.0, "<="),
        "final_effective_alive_lineages": _field(final, "lineage_effective_alive_lineages"),
        "final_entropy": _field(final, "lineage_alive_share_entropy"),
        "final_top1_alive_share": _field(final, "lineage_top1_alive_share"),
        "final_active_count": _field(final, "lineage_active_count"),
    }
    timing["collapsed_by_final"] = bool(
        (timing["final_effective_alive_lineages"] is not None and timing["final_effective_alive_lineages"] <= 1.5)
        or (timing["final_top1_alive_share"] is not None and timing["final_top1_alive_share"] >= 0.75)
        or (timing["final_entropy"] is not None and timing["final_entropy"] <= 0.5)
    )
    return timing


def _rare_persistent(rows: Sequence[Mapping[str, str]], dominant_lineage_id: str) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = defaultdict(lambda: {"present_generations": 0, "birth_count_total": 0.0})
    for row in _late_rows(rows):
        generation = _parse_int(row.get("generation"))
        for descriptor in _parse_json_list(row.get("lineage_strategy_descriptors_json")):
            lineage_id = str(descriptor.get("lineage_id") or "")
            if not lineage_id or lineage_id == dominant_lineage_id:
                continue
            alive_count = _parse_float(descriptor.get("lineage_alive_count")) or 0.0
            alive_share = _parse_float(descriptor.get("lineage_alive_share")) or 0.0
            birth_count = _parse_float(descriptor.get("lineage_birth_count")) or 0.0
            if alive_count <= 0.0 or alive_share >= 0.25:
                continue
            item = seen[lineage_id]
            item["lineage_id"] = lineage_id
            item["last_generation"] = generation
            item["present_generations"] = int(item["present_generations"]) + 1
            item["birth_count_total"] = float(item["birth_count_total"]) + birth_count
            item["max_alive_count"] = max(float(item.get("max_alive_count") or 0.0), alive_count)
            item["max_alive_share"] = max(float(item.get("max_alive_share") or 0.0), alive_share)
            for field in (*HABITAT_FIELDS, *PREDATOR_FIELDS, *RECOVERY_FIELDS):
                parsed = _parse_float(descriptor.get(field))
                if parsed is not None:
                    item[field] = parsed
    candidates = [
        item
        for item in seen.values()
        if int(item.get("present_generations") or 0) >= 2 or float(item.get("birth_count_total") or 0.0) > 0.0
    ]
    for item in candidates:
        item["reason"] = "late alive share below 0.25 with repeated late presence or late births"
    return sorted(
        candidates,
        key=lambda item: (
            -int(item.get("present_generations") or 0),
            -float(item.get("birth_count_total") or 0.0),
            str(item.get("lineage_id") or ""),
        ),
    )[:5]


def _brittle_candidates(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    by_lineage: dict[str, dict[str, Any]] = defaultdict(lambda: {"max_alive_share": 0.0, "final_alive_share": 0.0})
    if not rows:
        return []
    final_gen = _parse_int(rows[-1].get("generation"))
    for row in rows:
        generation = _parse_int(row.get("generation"))
        for descriptor in _parse_json_list(row.get("lineage_strategy_descriptors_json")):
            lineage_id = str(descriptor.get("lineage_id") or "")
            if not lineage_id:
                continue
            share = _parse_float(descriptor.get("lineage_alive_share")) or 0.0
            item = by_lineage[lineage_id]
            if share > float(item["max_alive_share"]):
                item["lineage_id"] = lineage_id
                item["max_alive_share"] = share
                item["max_generation"] = generation
                item["max_birth_share"] = _parse_float(descriptor.get("lineage_birth_share")) or 0.0
            if generation == final_gen:
                item["final_alive_share"] = share
    brittle = [
        {
            **item,
            "reason": "high earlier alive share but low final alive share within the same run",
        }
        for item in by_lineage.values()
        if float(item.get("max_alive_share") or 0.0) >= 0.5 and float(item.get("final_alive_share") or 0.0) < 0.1
    ]
    return sorted(brittle, key=lambda item: (-float(item["max_alive_share"]), str(item["lineage_id"])))[:5]


def _run_lineage_analysis(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    final = rows[-1] if rows else {}
    final_descriptors = _parse_json_list(final.get("lineage_strategy_descriptors_json"))
    dominant = _dominant_descriptor(final_descriptors)
    dominant_id = str((dominant or {}).get("lineage_id") or "")
    collapse = _collapse_timing(rows)
    first_collapse = collapse["first_top1_share_ge_0_75_generation"]
    if first_collapse is None:
        first_collapse = collapse["first_effective_alive_lineages_le_1_5_generation"]
    pre_rows = _pre_collapse_rows(rows, first_collapse)
    pre_final = pre_rows[-1] if pre_rows else {}
    pre_dominant = _dominant_descriptor(_parse_json_list(pre_final.get("lineage_strategy_descriptors_json")))
    branch_reasons = Counter()
    branch_non_dominant = 0
    branch_total = 0
    for row in rows:
        for candidate in _parse_json_list(row.get("lineage_branch_candidates_json")):
            reason = str(candidate.get("candidate_reason") or "unknown")
            branch_reasons[reason] += 1
            branch_total += 1
            if reason != "dominant_lineage":
                branch_non_dominant += 1
    return {
        "collapse_timing": collapse,
        "dominant_final_profile": _profile(dominant),
        "dominant_pre_collapse_profile": _profile(pre_dominant),
        "rare_persistent_lineages": _rare_persistent(rows, dominant_id),
        "dominant_brittle_candidates": _brittle_candidates(rows),
        "differentiation": {
            "habitat_pre_collapse": _aggregate_ranges(pre_rows, HABITAT_FIELDS),
            "habitat_late": _aggregate_ranges(_late_rows(rows), HABITAT_FIELDS),
            "predator_pre_collapse": _aggregate_ranges(pre_rows, PREDATOR_FIELDS),
            "predator_late": _aggregate_ranges(_late_rows(rows), PREDATOR_FIELDS),
            "reproduction_pre_collapse": _aggregate_ranges(pre_rows, REPRO_FIELDS),
            "reproduction_late": _aggregate_ranges(_late_rows(rows), REPRO_FIELDS),
            "recovery_late": _aggregate_ranges(_late_rows(rows), RECOVERY_FIELDS),
        },
        "branch_candidate_counts": dict(sorted(branch_reasons.items())),
        "branch_candidate_total": branch_total,
        "branch_candidate_non_dominant_fraction": (branch_non_dominant / branch_total) if branch_total else None,
    }


def _summarize_run(row: Mapping[str, str], metric_names: Sequence[str]) -> dict[str, Any]:
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
        "lineage_analysis": {},
    }
    if not metrics_path.exists():
        return run
    rows = _load_csv(metrics_path)
    run["metrics_rows"] = len(rows)
    if not rows:
        return run
    final_row = rows[-1]
    run["metrics"] = _metrics_from_final(final_row, metric_names)
    run["missing_metrics"] = [metric for metric in metric_names if metric not in final_row]
    run["critical_nonfinite"] = [
        {"metric": metric, "value": final_row.get(metric)}
        for metric in CRITICAL_METRICS
        if metric in final_row and _parse_float(final_row.get(metric)) is None
    ]
    strategy_values = [row.get("lineage_strategy_descriptors_json") for row in rows]
    branch_values = [row.get("lineage_branch_candidates_json") for row in rows]
    run["lineage_strategy_json_present"] = any(value not in (None, "") for value in strategy_values)
    run["lineage_branch_candidates_json_present"] = any(value not in (None, "") for value in branch_values)
    run["lineage_strategy_json_nonempty"] = any(_parse_json_list(value) for value in strategy_values)
    run["lineage_branch_candidates_json_nonempty"] = any(_parse_json_list(value) for value in branch_values)
    run["lineage_analysis"] = _run_lineage_analysis(rows)
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
        paired = sorted(seed for seed in SEEDS if (baseline, seed) in by_key and (variant, seed) in by_key)
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


def _scenario_lineage_summary(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_scenario: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for run in runs:
        by_scenario[str(run["scenario"])].append(run)
    out: dict[str, Any] = {}
    for scenario in SCENARIOS:
        entries = by_scenario.get(scenario, [])
        collapse_values = [
            _parse_float((entry.get("lineage_analysis") or {}).get("collapse_timing", {}).get("first_top1_share_ge_0_75_generation"))
            for entry in entries
        ]
        collapse_values = [value for value in collapse_values if value is not None]
        final_effective = [
            _parse_float((entry.get("lineage_analysis") or {}).get("collapse_timing", {}).get("final_effective_alive_lineages"))
            for entry in entries
        ]
        final_effective = [value for value in final_effective if value is not None]
        branch_counts = Counter()
        rare_count = 0
        brittle_count = 0
        branch_non_dom: list[float] = []
        range_buckets: dict[str, list[float]] = defaultdict(list)
        for entry in entries:
            analysis = dict(entry.get("lineage_analysis") or {})
            branch_counts.update(analysis.get("branch_candidate_counts") or {})
            rare_count += len(analysis.get("rare_persistent_lineages") or [])
            brittle_count += len(analysis.get("dominant_brittle_candidates") or [])
            non_dom = _parse_float(analysis.get("branch_candidate_non_dominant_fraction"))
            if non_dom is not None:
                branch_non_dom.append(non_dom)
            diff = analysis.get("differentiation") or {}
            for group_name, values in diff.items():
                for field, value in (values or {}).items():
                    parsed = _parse_float(value)
                    if parsed is not None:
                        range_buckets[f"{group_name}.{field}"].append(parsed)
        out[scenario] = {
            "run_count": len(entries),
            "collapse_top1_generation_mean": _mean(collapse_values),
            "collapse_top1_generation_median": _median(collapse_values),
            "collapse_top1_fraction": len(collapse_values) / len(entries) if entries else None,
            "final_effective_alive_lineages_mean": _mean(final_effective),
            "branch_candidate_counts": dict(sorted(branch_counts.items())),
            "branch_candidate_non_dominant_fraction_mean": _mean(branch_non_dom),
            "rare_persistent_candidate_count": rare_count,
            "dominant_brittle_candidate_count": brittle_count,
            "differentiation_range_means": {
                field: mean for field, items in sorted(range_buckets.items()) if (mean := _mean(items)) is not None
            },
        }
    return out


def _branch_summary(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_scenario = _scenario_lineage_summary(runs)
    return {
        scenario: {
            "branch_candidate_counts": values["branch_candidate_counts"],
            "branch_candidate_non_dominant_fraction_mean": values[
                "branch_candidate_non_dominant_fraction_mean"
            ],
        }
        for scenario, values in by_scenario.items()
    }


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


def _build_analysis(
    summaries: Sequence[Mapping[str, Any]],
    deltas: Sequence[Mapping[str, Any]],
    lineage: Mapping[str, Any],
    integrity: Mapping[str, Any],
) -> dict[str, Any]:
    by_scenario = {str(item["scenario"]): item for item in summaries}
    e04_metrics = by_scenario.get("e04_tuned_full_jungle_recovery", {}).get("metrics") or {}
    e04_alive = _parse_float(e04_metrics.get("alive")) or 0.0
    e04_births = _parse_float(e04_metrics.get("successful_births")) or 0.0
    e04_pred_events = _parse_float(e04_metrics.get("jungle_predator_events")) or 0.0
    e04_pred_damage = _parse_float(e04_metrics.get("jungle_predator_damage_total")) or 0.0
    e04_suppressed = _parse_float(e04_metrics.get("patch_regrowth_suppressed_total")) or 0.0
    e04_recovery = _parse_float(e04_metrics.get("patch_recovery_added_regrowth_total")) or 0.0
    e04_lineage = lineage.get("scenario_summaries", {}).get("e04_tuned_full_jungle_recovery", {})
    e04_non_dom = _parse_float(e04_lineage.get("branch_candidate_non_dominant_fraction_mean")) or 0.0
    e04_eff = _parse_float(e04_lineage.get("final_effective_alive_lineages_mean")) or 0.0
    interpretable = (
        int(integrity.get("critical_nonfinite_issues") or 0) == 0
        and int(integrity.get("env_mismatches") or 0) == 0
        and e04_alive > 0.0
        and e04_births > 0.0
        and e04_pred_events > 0.0
        and e04_pred_damage > 0.0
        and e04_suppressed > 0.0
        and e04_recovery > 0.0
        and int(integrity.get("lineage_strategy_json_nonempty") or 0) > 0
        and int(integrity.get("lineage_branch_candidates_json_nonempty") or 0) > 0
    )
    cognition_ready = interpretable and e04_eff > 1.5 and e04_non_dom >= 0.25
    recommendation = "GO" if cognition_ready else ("LIMITED GO" if interpretable else "NO-GO")
    return {
        "candidate_configuration": {
            "predator": "medium",
            "waste_crowding": "medium",
            "terrain_protection": "high",
            "recovery": "low",
        },
        "key_questions": {
            "why_campaign_d_collapsed": "assess collapse timing and differentiation summaries; final effective-lineage collapse alone does not prove absence of pre-collapse texture",
            "strategy_differences_before_collapse": "assess pre-collapse habitat, predator, reproduction, and recovery range summaries",
            "branch_candidates_meaningful": e04_non_dom > 0.0,
            "tuned_jungle_more_texture_than_default": "assess e04 versus e02 lineage summaries and paired deltas",
            "theme4_cognition_readiness": cognition_ready,
        },
        "go_no_go": recommendation,
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
    lineage: Mapping[str, Any],
    branch: Mapping[str, Any],
    analysis: Mapping[str, Any],
    repeat_parity: Mapping[str, Any],
) -> None:
    summary_by_scenario = {str(item["scenario"]): item for item in summaries}
    lineage_by_scenario = lineage.get("scenario_summaries", {})
    lines: list[str] = [
        "# Patch/Jungle V0 Campaign E Lineage / Strategy Triage Report",
        "",
        f"Campaign: `{campaign_name}`",
        "",
        f"Output root: `{output_root}`",
        "",
        f"Batch directory: `{batch_dir}`",
        "",
        "Campaign E is a lineage/strategy diagnostic campaign. It does not tune coefficients, change mechanics, or claim intelligence, emergence, memory selection, fear selection, prediction selection, culture, or proof of the thesis.",
        "",
        "## Campaign C/D Tuned Configuration",
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
        "| Scenarios | `e00_flat_baseline`, `e01_patch_observed`, `e02_default_full_jungle`, `e03_tuned_full_jungle_no_recovery`, `e04_tuned_full_jungle_recovery` |",
        "| Seeds | `101, 202, 303, 404, 505, 606, 707, 808, 909, 1001, 1102, 1203, 1304, 1405, 1506, 1607, 1708, 1809, 19010, 20011` |",
        "| Generations | 24 |",
        "| Steps per generation | 120 |",
        "| Population | 64 |",
        "| Grid size | 16 |",
        "| Device | CUDA |",
        "",
        "## Preflight",
        "",
        "The required preflight test suites, CUDA dry-run, and repeat-parity checks are recorded in the run notes and summary artifacts. If any gate failed, this report should be read as a stopped campaign artifact.",
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
        "jungle_predator_events",
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
    lines.extend(["", "## Lineage Collapse Timing", ""])
    collapse_rows = []
    for scenario in SCENARIOS:
        data = lineage_by_scenario.get(scenario, {})
        collapse_rows.append(
            [
                scenario,
                _fmt(data.get("collapse_top1_generation_mean")),
                _fmt(data.get("collapse_top1_generation_median")),
                _fmt(data.get("collapse_top1_fraction")),
                _fmt(data.get("final_effective_alive_lineages_mean")),
            ]
        )
    lines.extend(
        _markdown_table(
            ("Scenario", "Top1 collapse gen mean", "Top1 collapse gen median", "Collapse fraction", "Final effective lineages"),
            collapse_rows,
        )
    )
    lines.extend(["", "## Branch Candidate Analysis", ""])
    branch_rows = []
    for scenario in SCENARIOS:
        data = branch.get(scenario, {})
        branch_rows.append(
            [
                scenario,
                _fmt(data.get("branch_candidate_non_dominant_fraction_mean")),
                json.dumps(data.get("branch_candidate_counts") or {}, sort_keys=True),
            ]
        )
    lines.extend(_markdown_table(("Scenario", "Non-dominant candidate fraction", "Candidate reasons"), branch_rows))
    for section, metrics in GROUPS.items():
        lines.extend(["", f"## {section.replace('_', ' ').title()}", ""])
        group_rows = []
        for scenario in SCENARIOS:
            values = summary_by_scenario[scenario].get("metrics") or {}
            group_rows.append([scenario, *[_fmt(values.get(metric)) for metric in metrics]])
        lines.extend(_markdown_table(("Scenario", *metrics), group_rows))
    lines.extend(
        [
            "",
            "## Habitat Differentiation",
            "",
            "Between-lineage habitat differentiation is recorded in `campaign_e_lineage_strategy_summary.json` under `differentiation_range_means`. Use pre-collapse ranges to avoid mistaking final single-lineage collapse for absence of earlier texture.",
            "",
            "## Predator Exposure Differentiation",
            "",
            "Predator exposure ranges are recorded for predator-enabled scenarios in `campaign_e_lineage_strategy_summary.json`. These ranges are diagnostic only.",
            "",
            "## Reproduction Context Differentiation",
            "",
            "Birth-context and reproduction-rate ranges are recorded in the lineage summary. These values support cautious comparison of contexts but do not imply selected cognition.",
            "",
            "## Recovery-Associated Survival",
            "",
            "Recovery exposure summaries are recorded for recovery-enabled scenarios. Recovery-associated persistence should be read alongside waste and predator metrics to avoid treating small added-regrowth effects as a strategy claim.",
            "",
            "## Default Vs Tuned Jungle",
            "",
            "`e04_tuned_full_jungle_recovery - e02_default_full_jungle` is the primary tuned-vs-default comparison. Interpret improvement as clearer ecological pressure with viability, not simply higher population.",
            "",
            "## Likely Causes Of Lineage Diversity Collapse",
            "",
            "- Final active alive-lineage collapse can reflect a narrow reproduction bottleneck, insufficient niche separation, short-lived branch candidates, or lineage reporting granularity.",
            "- Campaign E evaluates whether pre-collapse descriptors show habitat, predator, reproduction, or recovery differentiation before any final sweep.",
            "",
            "## Theme 4 Cognition-Readiness",
            "",
            f"Recommendation: `{analysis['go_no_go']}`.",
            "",
            "A cognition-readiness probe is recommended only if lineage differentiation and non-dominant branch candidates are meaningful and environment-context correlations remain interpretable.",
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
    args = _parse_args(argv)
    batch_dir = Path(args.batch_dir)
    matrix_rows = _load_csv(batch_dir / "matrix.csv")
    metric_names = KEY_METRICS
    runs = [_summarize_run(row, metric_names) for row in matrix_rows]
    summaries = _scenario_summaries(runs, metric_names)
    deltas = _paired_deltas(runs, metric_names)
    seed_sets = {
        scenario: sorted(int(run["seed"]) for run in runs if run["scenario"] == scenario)
        for scenario in SCENARIOS
    }
    lineage_summary = {
        "schema_version": "patch_jungle_campaign_e_lineage_summary_v1",
        "scenario_summaries": _scenario_lineage_summary(runs),
        "runs": [
            {
                "scenario": run["scenario"],
                "seed": run["seed"],
                "run_id": run["run_id"],
                "run_dir": run["run_dir"],
                **dict(run.get("lineage_analysis") or {}),
            }
            for run in sorted(runs, key=lambda item: (str(item["scenario"]), int(item["seed"])))
        ],
    }
    branch_summary = _branch_summary(runs)
    integrity = {
        "planned_runs": len(matrix_rows),
        "completed_runs": sum(1 for run in runs if run["metrics_exists"] and int(run["metrics_rows"]) > 0),
        "all_runs_complete": len(matrix_rows)
        == sum(1 for run in runs if run["metrics_exists"] and int(run["metrics_rows"]) > 0),
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
    repeat_parity = _read_json(args.repeat_parity_json) if args.repeat_parity_json else {}
    analysis = _build_analysis(summaries, deltas, lineage_summary, integrity)
    report = {
        "schema_version": "patch_jungle_campaign_e_report_v1",
        "campaign_name": args.campaign_name,
        "output_root": args.output_root,
        "batch_dir": str(batch_dir),
        "integrity": integrity,
        "scenario_summaries": summaries,
        "paired_deltas": deltas,
        "analysis": analysis,
        "lineage_summary_path": args.lineage_json,
        "branch_candidate_summary_path": args.branch_json,
        "repeat_parity": repeat_parity,
    }
    for output, payload in (
        (args.out_json, report),
        (args.analysis_json, analysis),
        (args.lineage_json, lineage_summary),
        (args.branch_json, branch_summary),
    ):
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
        lineage=lineage_summary,
        branch=branch_summary,
        analysis=analysis,
        repeat_parity=repeat_parity,
    )
    print(f"[patch-jungle-campaign-e] wrote {args.out_json}")
    print(f"[patch-jungle-campaign-e] wrote {args.analysis_json}")
    print(f"[patch-jungle-campaign-e] wrote {args.summary_csv}")
    print(f"[patch-jungle-campaign-e] wrote {args.deltas_csv}")
    print(f"[patch-jungle-campaign-e] wrote {args.lineage_json}")
    print(f"[patch-jungle-campaign-e] wrote {args.branch_json}")
    print(f"[patch-jungle-campaign-e] wrote {args.report_md}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
