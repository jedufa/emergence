#!/usr/bin/env python3
"""R2 integrated lineage-collapse forensic reporting helper.

Reporting-only infrastructure: reads completed registry batch outputs and emits
deterministic JSON, CSV, and Markdown summaries. It does not change simulation
behavior.
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

SCENARIOS: tuple[str, ...] = (
    "r2_00_patch_observed",
    "r2_01_default_full_jungle",
    "r2_02_tuned_full_jungle_no_recovery",
    "r2_03_tuned_full_jungle_recovery",
    "r2_04_theme12_clean_pressure_reference",
)

SEEDS: tuple[int, ...] = (101, 202, 303, 404, 505, 606, 707, 808, 909, 1001)

SCENARIO_LABELS: dict[str, str] = {
    "r2_00_patch_observed": "Patch observed",
    "r2_01_default_full_jungle": "Default full Jungle",
    "r2_02_tuned_full_jungle_no_recovery": "Tuned full Jungle without recovery",
    "r2_03_tuned_full_jungle_recovery": "Tuned full Jungle with recovery",
    "r2_04_theme12_clean_pressure_reference": "Theme 12 clean pressure reference",
}

EXPECTED_ENV: dict[str, dict[str, bool | float]] = {
    "r2_00_patch_observed": {
        "patch_world_enable": True,
        "patch_world_observation_enable": True,
        "patch_world_waste_consequence_enable": False,
        "jungle_predator_pressure_enable": False,
        "jungle_reproduction_emissions_enable": False,
        "jungle_terrain_protection_enable": False,
        "jungle_ecological_recovery_enable": False,
    },
    "r2_01_default_full_jungle": {
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
    "r2_02_tuned_full_jungle_no_recovery": {
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
    "r2_03_tuned_full_jungle_recovery": {
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
    "r2_04_theme12_clean_pressure_reference": {
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
    "device": "cuda",
    "bank_device": "cuda",
    "phase_e_e1_enable": False,
    "enable_developmental_cognition": False,
    "enable_introspection_channels": False,
    "enable_self_regulation": False,
    "enable_self_modification": False,
    "enable_directed_inheritance": False,
}

PRIMARY_PAIRS: tuple[tuple[str, str], ...] = (
    ("r2_00_patch_observed", "r2_01_default_full_jungle"),
    ("r2_00_patch_observed", "r2_02_tuned_full_jungle_no_recovery"),
    ("r2_02_tuned_full_jungle_no_recovery", "r2_03_tuned_full_jungle_recovery"),
    ("r2_03_tuned_full_jungle_recovery", "r2_04_theme12_clean_pressure_reference"),
)

KEY_METRICS: tuple[str, ...] = (
    "alive",
    "deaths",
    "cohort_survival_rate",
    "successful_births",
    "mean_offspring_total",
    "birth_attempts",
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
    "predator_event_patch_waste_mean",
    "predator_event_patch_density_mean",
    "predator_event_patch_cover_mean",
    "predator_event_patch_shelter_mean",
    "patch_recovery_activity_mean",
    "jungle_recovery_regrowth_boost_total",
    "patch_recovery_added_regrowth_total",
    "lineage_active_count",
    "lineage_top1_alive_share",
    "lineage_top3_alive_share",
    "lineage_alive_share_entropy",
    "lineage_effective_alive_lineages",
    "lineage_branch_candidate_count",
    "lineage_strategy_descriptor_count",
    "lineage_patch_waste_mean_std",
    "lineage_patch_shelter_mean_std",
    "lineage_predator_damage_mean_std",
)

CRITICAL_METRICS: tuple[str, ...] = (
    "alive",
    "successful_births",
    "patch_food_amount_mean",
    "patch_regrowth_suppressed_total",
    "jungle_predator_events",
    "jungle_predator_damage_total",
    "jungle_predator_damage_prevented_total",
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

PROFILE_FIELDS: tuple[str, ...] = (
    "lineage_alive_count",
    "lineage_alive_share",
    "lineage_birth_count",
    "lineage_birth_share",
    *HABITAT_FIELDS,
    *PREDATOR_FIELDS,
    *REPRO_FIELDS,
)


def _read_json(path: Path) -> Any:
    if not path.exists():
        return {}
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


def _int(value: Any) -> int | None:
    parsed = _float(value)
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


def _min(values: Iterable[float]) -> float | None:
    items = list(values)
    return min(items) if items else None


def _max(values: Iterable[float]) -> float | None:
    items = list(values)
    return max(items) if items else None


def _fmt(value: Any, digits: int = 3) -> str:
    parsed = _float(value)
    if parsed is None:
        return "" if value is None else str(value)
    return f"{parsed:.{digits}f}"


def _json_list(value: Any) -> list[dict[str, Any]]:
    if value is None or not str(value).strip():
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, dict):
        for key in ("candidates", "lineages", "descriptors", "items"):
            maybe = parsed.get(key)
            if isinstance(maybe, list):
                return [item for item in maybe if isinstance(item, dict)]
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def _generation(row: Mapping[str, Any], fallback: int) -> int:
    parsed = _int(row.get("generation"))
    return fallback if parsed is None else parsed


def _lineage_id(item: Mapping[str, Any]) -> str:
    for key in (
        "lineage_id",
        "biological_lineage_id",
        "candidate_lineage_id",
        "branch_lineage_id",
        "id",
    ):
        raw = item.get(key)
        if raw not in (None, ""):
            return str(raw)
    return ""


def _candidate_category(item: Mapping[str, Any]) -> str:
    for key in ("candidate_reason", "reason", "category", "candidate_category", "branch_reason"):
        raw = item.get(key)
        if raw not in (None, ""):
            return str(raw)
    return "unknown"


def _close(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return bool(actual) is expected
    try:
        return abs(float(actual) - float(expected)) <= 1.0e-9
    except (TypeError, ValueError):
        return actual == expected


def _metrics_from_final(final_row: Mapping[str, str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for metric in KEY_METRICS:
        parsed = _float(final_row.get(metric))
        if parsed is not None:
            out[metric] = parsed
    return out


def _config_parts(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _read_json(run_dir / "config.json")
    return dict(config.get("env_cfg") or {}), dict(config.get("sim_cfg") or {})


def _scenario_from_matrix(row: Mapping[str, str]) -> str:
    return str(row.get("scenario") or row.get("scenario_id") or "")


def _run_dir_from_matrix(row: Mapping[str, str]) -> Path:
    raw = row.get("run_dir") or row.get("output_dir")
    if not raw:
        raise ValueError(f"matrix row has no run_dir: {row}")
    return Path(raw)


def _descriptor_map(row: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in _json_list(row.get("lineage_strategy_descriptors_json")):
        lineage_id = _lineage_id(item)
        if lineage_id:
            out[lineage_id] = dict(item)
    return out


def _active_descriptors(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in _json_list(row.get("lineage_strategy_descriptors_json"))
        if (_float(item.get("lineage_alive_count")) or 0.0) > 0.0
        or (_float(item.get("lineage_alive_share")) or 0.0) > 0.0
    ]


def _top_descriptor(row: Mapping[str, Any], by: str) -> dict[str, Any] | None:
    descriptors = _active_descriptors(row)
    if not descriptors:
        return None
    if by == "birth":
        keys = ("lineage_birth_share", "lineage_birth_count", "lineage_alive_share", "lineage_alive_count")
    else:
        keys = ("lineage_alive_share", "lineage_alive_count", "lineage_birth_share", "lineage_birth_count")
    return dict(
        max(
            descriptors,
            key=lambda item: tuple(_float(item.get(key)) or 0.0 for key in keys) + (_lineage_id(item),),
        )
    )


def _profile(descriptor: Mapping[str, Any] | None) -> dict[str, Any]:
    if not descriptor:
        return {}
    out: dict[str, Any] = {"lineage_id": _lineage_id(descriptor)}
    for field in PROFILE_FIELDS:
        parsed = _float(descriptor.get(field))
        if parsed is not None:
            out[field] = parsed
    return out


def _first_generation(rows: Sequence[Mapping[str, str]], metric: str, threshold: float, op: str) -> int | None:
    for idx, row in enumerate(rows):
        value = _float(row.get(metric))
        if value is None:
            continue
        if (op == "<=" and value <= threshold) or (op == ">=" and value >= threshold):
            return _generation(row, idx)
    return None


def _collapse_timing(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    final = rows[-1] if rows else {}
    active_le_1 = _first_generation(rows, "lineage_active_count", 1.0, "<=")
    effective_le_1 = _first_generation(rows, "lineage_effective_alive_lineages", 1.01, "<=")
    top1_near_all = _first_generation(rows, "lineage_top1_alive_share", 0.999, ">=")
    all_but_one = min([value for value in (active_le_1, effective_le_1, top1_near_all) if value is not None], default=None)
    timing = {
        "first_effective_alive_lineages_le_1_5_generation": _first_generation(
            rows, "lineage_effective_alive_lineages", 1.5, "<="
        ),
        "first_entropy_le_0_5_generation": _first_generation(rows, "lineage_alive_share_entropy", 0.5, "<="),
        "first_top1_share_ge_0_75_generation": _first_generation(rows, "lineage_top1_alive_share", 0.75, ">="),
        "first_top1_share_ge_0_90_generation": _first_generation(rows, "lineage_top1_alive_share", 0.90, ">="),
        "first_all_but_one_extinct_generation": all_but_one,
        "final_alive_lineage_count": _float(final.get("lineage_active_count")),
        "final_active_lineage_count": _float(final.get("lineage_active_count")),
        "final_effective_alive_lineages": _float(final.get("lineage_effective_alive_lineages")),
        "final_alive_share_entropy": _float(final.get("lineage_alive_share_entropy")),
        "final_top1_alive_share": _float(final.get("lineage_top1_alive_share")),
    }
    timing["collapsed_by_final"] = bool(
        (timing["final_effective_alive_lineages"] is not None and timing["final_effective_alive_lineages"] <= 1.5)
        or (timing["final_top1_alive_share"] is not None and timing["final_top1_alive_share"] >= 0.75)
        or (timing["final_alive_share_entropy"] is not None and timing["final_alive_share_entropy"] <= 0.5)
    )
    return timing


def _pre_collapse_rows(rows: Sequence[Mapping[str, str]], timing: Mapping[str, Any]) -> list[Mapping[str, str]]:
    collapse_generation = (
        timing.get("first_top1_share_ge_0_75_generation")
        or timing.get("first_effective_alive_lineages_le_1_5_generation")
        or timing.get("first_entropy_le_0_5_generation")
    )
    if collapse_generation is None:
        return list(rows)
    selected = [
        row
        for idx, row in enumerate(rows)
        if _generation(row, idx) < int(collapse_generation)
        and _json_list(row.get("lineage_strategy_descriptors_json"))
    ]
    return selected or [row for row in rows[:1] if _json_list(row.get("lineage_strategy_descriptors_json"))]


def _range_summary(rows: Sequence[Mapping[str, str]], fields: Sequence[str]) -> dict[str, float]:
    ranges: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        descriptors = _active_descriptors(row)
        if len(descriptors) < 2:
            continue
        for field in fields:
            values = [_float(item.get(field)) for item in descriptors]
            clean = [float(value) for value in values if value is not None]
            if len(clean) >= 2:
                ranges[f"{field}_range"].append(max(clean) - min(clean))
    return {field: value for field, items in sorted(ranges.items()) if (value := _mean(items)) is not None}


def _founder_dominance(rows: Sequence[Mapping[str, str]], final_dominant_id: str) -> dict[str, Any]:
    descriptor_rows = [row for row in rows if _json_list(row.get("lineage_strategy_descriptors_json"))]
    initial = descriptor_rows[0] if descriptor_rows else {}
    gen1 = next(
        (
            row
            for idx, row in enumerate(descriptor_rows)
            if _generation(row, idx) == 1 and _json_list(row.get("lineage_strategy_descriptors_json"))
        ),
        descriptor_rows[1] if len(descriptor_rows) > 1 else initial,
    )
    initial_active = _active_descriptors(initial)
    top_alive = _top_descriptor(gen1, "alive")
    top_birth = _top_descriptor(gen1, "birth")
    top_alive_id = _lineage_id(top_alive or {})
    top_birth_id = _lineage_id(top_birth or {})
    return {
        "initial_biological_lineage_count": len(initial_active),
        "generation1_top_alive_lineage_id": top_alive_id,
        "generation1_top_birth_lineage_id": top_birth_id,
        "final_dominant_lineage_id": final_dominant_id,
        "generation1_top_alive_becomes_final_dominant": bool(top_alive_id and top_alive_id == final_dominant_id),
        "generation1_top_birth_becomes_final_dominant": bool(top_birth_id and top_birth_id == final_dominant_id),
        "final_dominant_already_top1_early": bool(
            final_dominant_id and final_dominant_id in {top_alive_id, top_birth_id}
        ),
    }


def _first_top_generation(rows: Sequence[Mapping[str, str]], lineage_id: str, by: str) -> int | None:
    if not lineage_id:
        return None
    for idx, row in enumerate(rows):
        top = _top_descriptor(row, by)
        if top and _lineage_id(top) == lineage_id:
            return _generation(row, idx)
    return None


def _first_present_generation(rows: Sequence[Mapping[str, str]], lineage_id: str) -> int | None:
    if not lineage_id:
        return None
    for idx, row in enumerate(rows):
        item = _descriptor_map(row).get(lineage_id)
        if item and ((_float(item.get("lineage_alive_count")) or 0.0) > 0.0):
            return _generation(row, idx)
    return None


def _descriptor_at_or_after(
    rows: Sequence[Mapping[str, str]], lineage_id: str, generation: int
) -> dict[str, Any] | None:
    for idx, row in enumerate(rows):
        if _generation(row, idx) < generation:
            continue
        item = _descriptor_map(row).get(lineage_id)
        if item:
            return item
    return None


def _candidate_lineage_id(candidate: Mapping[str, Any], row: Mapping[str, str]) -> str:
    direct = _lineage_id(candidate)
    if direct:
        return direct
    # Some summaries only label category-level branch candidates. Use a matching
    # active lineage id only when there is a single active descriptor.
    active = _active_descriptors(row)
    if len(active) == 1:
        return _lineage_id(active[0])
    return ""


def _fate_proxy(descriptor: Mapping[str, Any] | None, final_dominant_id: str, lineage_id: str) -> str:
    if not descriptor:
        return "unknown"
    birth_share = _float(descriptor.get("lineage_birth_share")) or 0.0
    birth_count = _float(descriptor.get("lineage_birth_count")) or 0.0
    predator_damage = (_float(descriptor.get("lineage_predator_damage_total")) or 0.0) + (
        _float(descriptor.get("lineage_predator_damage_mean")) or 0.0
    )
    waste = _float(descriptor.get("lineage_patch_waste_mean")) or 0.0
    density = _float(descriptor.get("lineage_patch_density_mean")) or 0.0
    food = _float(descriptor.get("lineage_patch_food_amount_mean"))
    if birth_share <= 0.05 and birth_count <= 1.0:
        return "low reproduction proxy"
    if predator_damage > 0.0:
        return "predator damage proxy"
    if waste >= 0.5 or density >= 0.5:
        return "high waste/density exposure proxy"
    if food is not None and food <= 0.25:
        return "low food exposure proxy"
    if final_dominant_id and lineage_id != final_dominant_id:
        return "displaced by top lineage proxy"
    return "unknown"


def _branch_candidate_fates(rows: Sequence[Mapping[str, str]], final_dominant_id: str) -> list[dict[str, Any]]:
    fates: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    final_descriptors = _descriptor_map(rows[-1]) if rows else {}
    for idx, row in enumerate(rows):
        generation = _generation(row, idx)
        for candidate in _json_list(row.get("lineage_branch_candidates_json")):
            lineage_id = _candidate_lineage_id(candidate, row)
            category = _candidate_category(candidate)
            key = (lineage_id, category, generation)
            if key in seen:
                continue
            seen.add(key)
            descriptor = _descriptor_map(row).get(lineage_id) if lineage_id else None
            final_descriptor = final_descriptors.get(lineage_id) if lineage_id else None
            fate = {
                "lineage_id": lineage_id,
                "candidate_category": category,
                "generation_emitted": generation,
                "alive_share_at_emission": _float((descriptor or {}).get("lineage_alive_share")),
                "birth_share_at_emission": _float((descriptor or {}).get("lineage_birth_share")),
                "candidate_identified": bool(lineage_id),
                "survived_1_generations": False,
                "survived_2_generations": False,
                "survived_4_generations": False,
                "survived_8_generations": False,
                "extinct_by_final": None,
                "cause_of_fate_proxy": "unknown",
                "habitat_profile": {},
                "predator_profile": {},
                "waste_density_profile": {},
                "recovery_profile": {},
            }
            if descriptor:
                fate["habitat_profile"] = {
                    field: value
                    for field in HABITAT_FIELDS
                    if (value := _float(descriptor.get(field))) is not None
                }
                fate["predator_profile"] = {
                    field: value
                    for field in PREDATOR_FIELDS
                    if (value := _float(descriptor.get(field))) is not None
                }
                fate["waste_density_profile"] = {
                    field: value
                    for field in ("lineage_patch_waste_mean", "lineage_patch_density_mean")
                    if (value := _float(descriptor.get(field))) is not None
                }
                fate["recovery_profile"] = {
                    field: value
                    for field in ("lineage_patch_recovery_activity_mean", "lineage_birth_patch_recovery_activity_mean")
                    if (value := _float(descriptor.get(field))) is not None
                }
            if lineage_id:
                for horizon in (1, 2, 4, 8):
                    later = _descriptor_at_or_after(rows, lineage_id, generation + horizon)
                    fate[f"survived_{horizon}_generations"] = bool(
                        later
                        and (
                            (_float(later.get("lineage_alive_count")) or 0.0) > 0.0
                            or (_float(later.get("lineage_alive_share")) or 0.0) > 0.0
                        )
                    )
                final_alive = bool(
                    final_descriptor
                    and (
                        (_float(final_descriptor.get("lineage_alive_count")) or 0.0) > 0.0
                        or (_float(final_descriptor.get("lineage_alive_share")) or 0.0) > 0.0
                    )
                )
                fate["extinct_by_final"] = not final_alive
                fate["cause_of_fate_proxy"] = _fate_proxy(descriptor, final_dominant_id, lineage_id)
            fates.append(fate)
    return fates


def _dominant_profile(rows: Sequence[Mapping[str, str]], final_dominant: Mapping[str, Any] | None) -> dict[str, Any]:
    lineage_id = _lineage_id(final_dominant or {})
    categories = Counter()
    for row in rows:
        for candidate in _json_list(row.get("lineage_branch_candidates_json")):
            if _candidate_lineage_id(candidate, row) == lineage_id:
                categories[_candidate_category(candidate)] += 1
    return {
        **_profile(final_dominant),
        "first_generation_present": _first_present_generation(rows, lineage_id),
        "first_generation_top1_alive": _first_top_generation(rows, lineage_id, "alive"),
        "first_generation_top1_births": _first_top_generation(rows, lineage_id, "birth"),
        "branch_candidate_status": bool(categories),
        "branch_candidate_categories": dict(sorted(categories.items())),
    }


def _env_mismatches(scenario: str, run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    env_cfg, sim_cfg = _config_parts(run_dir)
    env_mismatches: list[dict[str, Any]] = []
    sim_mismatches: list[dict[str, Any]] = []
    for key, expected in EXPECTED_ENV.get(scenario, {}).items():
        if key not in env_cfg or not _close(env_cfg.get(key), expected):
            env_mismatches.append(
                {
                    "scenario": scenario,
                    "field": key,
                    "expected": expected,
                    "actual": env_cfg.get(key),
                }
            )
    for key, expected in EXPECTED_SIM.items():
        if key in sim_cfg and not _close(sim_cfg.get(key), expected):
            sim_mismatches.append(
                {
                    "scenario": scenario,
                    "field": key,
                    "expected": expected,
                    "actual": sim_cfg.get(key),
                }
            )
    return env_mismatches, sim_mismatches


def _archive_outputs(run_dir: Path) -> dict[str, Any]:
    archive_dirs: list[Path] = []
    for name in ("archives", "lineage_archive"):
        root = run_dir / name
        if root.exists():
            archive_dirs.extend(path for path in root.glob("**/*") if path.exists())
    manifest_dirs: list[Path] = []
    for name in ("branch_manifests", "manifests", "lineage_manifests"):
        root = run_dir / name
        if root.exists():
            manifest_dirs.extend(path for path in root.glob("**/*") if path.exists())
    return {
        "archive_path_exists": (run_dir / "archives").exists(),
        "branch_manifest_path_exists": any((run_dir / name).exists() for name in ("branch_manifests", "manifests", "lineage_manifests")),
        "archive_entry_count": len(archive_dirs),
        "manifest_entry_count": len(manifest_dirs),
        "metadata_level_only": not archive_dirs and not manifest_dirs,
    }


def _run_analysis(row: Mapping[str, str]) -> dict[str, Any]:
    scenario = _scenario_from_matrix(row)
    seed = int(row["seed"])
    run_dir = _run_dir_from_matrix(row)
    metrics_path = run_dir / "metrics.csv"
    run: dict[str, Any] = {
        "scenario": scenario,
        "seed": seed,
        "run_id": str(row.get("run_id") or ""),
        "run_dir": str(run_dir),
        "metrics_exists": metrics_path.exists(),
        "metrics_rows": 0,
        "metrics": {},
        "critical_nonfinite": [],
        "env_mismatches": [],
        "sim_mismatches": [],
        "lineage_strategy_json_present": False,
        "lineage_strategy_json_nonempty": False,
        "lineage_branch_candidates_json_present": False,
        "lineage_branch_candidates_json_nonempty": False,
        "collapse_timing": {},
        "founder_dominance": {},
        "pre_collapse_differentiation": {},
        "dominant_lineage_profile": {},
        "branch_candidate_fates": [],
        "archive_outputs": _archive_outputs(run_dir),
    }
    if not metrics_path.exists() or metrics_path.stat().st_size == 0:
        return run
    rows = _read_rows(metrics_path)
    run["metrics_rows"] = len(rows)
    if not rows:
        return run
    final = rows[-1]
    run["metrics"] = _metrics_from_final(final)
    for metric in CRITICAL_METRICS:
        raw = final.get(metric)
        if raw not in (None, "") and _float(raw) is None:
            run["critical_nonfinite"].append({"metric": metric, "value": raw})
    env_mismatches, sim_mismatches = _env_mismatches(scenario, run_dir)
    run["env_mismatches"] = env_mismatches
    run["sim_mismatches"] = sim_mismatches
    strategy_values = [item.get("lineage_strategy_descriptors_json") for item in rows]
    branch_values = [item.get("lineage_branch_candidates_json") for item in rows]
    run["lineage_strategy_json_present"] = "lineage_strategy_descriptors_json" in final
    run["lineage_branch_candidates_json_present"] = "lineage_branch_candidates_json" in final
    run["lineage_strategy_json_nonempty"] = any(_json_list(value) for value in strategy_values)
    run["lineage_branch_candidates_json_nonempty"] = any(_json_list(value) for value in branch_values)
    timing = _collapse_timing(rows)
    final_dominant = _top_descriptor(final, "alive")
    final_dominant_id = _lineage_id(final_dominant or {})
    pre_rows = _pre_collapse_rows(rows, timing)
    branch_fates = _branch_candidate_fates(rows, final_dominant_id)
    run["collapse_timing"] = timing
    run["founder_dominance"] = _founder_dominance(rows, final_dominant_id)
    run["pre_collapse_differentiation"] = {
        "habitat": _range_summary(pre_rows, HABITAT_FIELDS),
        "predator_exposure": _range_summary(pre_rows, PREDATOR_FIELDS),
        "reproduction_context": _range_summary(pre_rows, REPRO_FIELDS),
        "recovery_associated": _range_summary(
            pre_rows,
            ("lineage_patch_recovery_activity_mean", "lineage_birth_patch_recovery_activity_mean", "lineage_alive_share"),
        ),
    }
    run["dominant_lineage_profile"] = _dominant_profile(rows, final_dominant)
    run["branch_candidate_fates"] = branch_fates
    return run


def _scenario_summaries(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[str(run["scenario"])].append(run)
    summaries: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        entries = sorted(grouped.get(scenario, []), key=lambda item: int(item["seed"]))
        metrics: dict[str, float | None] = {}
        for metric in KEY_METRICS:
            values = [
                value
                for entry in entries
                if (value := _float((entry.get("metrics") or {}).get(metric))) is not None
            ]
            metrics[metric] = _mean(values)
        timing_fields = (
            "first_effective_alive_lineages_le_1_5_generation",
            "first_entropy_le_0_5_generation",
            "first_top1_share_ge_0_75_generation",
            "first_top1_share_ge_0_90_generation",
            "first_all_but_one_extinct_generation",
            "final_effective_alive_lineages",
            "final_alive_share_entropy",
            "final_top1_alive_share",
            "final_active_lineage_count",
        )
        collapse: dict[str, Any] = {}
        for field in timing_fields:
            values = [
                value
                for entry in entries
                if (value := _float((entry.get("collapse_timing") or {}).get(field))) is not None
            ]
            collapse[f"{field}_mean"] = _mean(values)
            collapse[f"{field}_median"] = _median(values)
            collapse[f"{field}_min"] = _min(values)
            collapse[f"{field}_max"] = _max(values)
        collapse["collapsed_by_final_count"] = sum(
            1 for entry in entries if bool((entry.get("collapse_timing") or {}).get("collapsed_by_final"))
        )
        founder_alive = [
            bool((entry.get("founder_dominance") or {}).get("generation1_top_alive_becomes_final_dominant"))
            for entry in entries
        ]
        founder_birth = [
            bool((entry.get("founder_dominance") or {}).get("generation1_top_birth_becomes_final_dominant"))
            for entry in entries
        ]
        diff_buckets: dict[str, list[float]] = defaultdict(list)
        for entry in entries:
            for group, values in (entry.get("pre_collapse_differentiation") or {}).items():
                for field, value in (values or {}).items():
                    parsed = _float(value)
                    if parsed is not None:
                        diff_buckets[f"{group}.{field}"].append(parsed)
        branch_fates = [fate for entry in entries for fate in list(entry.get("branch_candidate_fates") or [])]
        identified = [fate for fate in branch_fates if fate.get("candidate_identified")]
        non_dom = [
            fate
            for entry in entries
            for fate in list(entry.get("branch_candidate_fates") or [])
            if fate.get("lineage_id")
            and fate.get("lineage_id")
            != (entry.get("dominant_lineage_profile") or {}).get("lineage_id")
        ]
        summaries.append(
            {
                "scenario": scenario,
                "label": SCENARIO_LABELS[scenario],
                "run_count": len(entries),
                "seeds": [int(entry["seed"]) for entry in entries],
                "metrics": metrics,
                "collapse": collapse,
                "founder_dominance": {
                    "gen1_top_alive_becomes_final_fraction": (
                        sum(1 for item in founder_alive if item) / len(founder_alive) if founder_alive else None
                    ),
                    "gen1_top_birth_becomes_final_fraction": (
                        sum(1 for item in founder_birth if item) / len(founder_birth) if founder_birth else None
                    ),
                },
                "pre_collapse_differentiation_means": {
                    field: mean for field, items in sorted(diff_buckets.items()) if (mean := _mean(items)) is not None
                },
                "branch_candidate_total": len(branch_fates),
                "branch_candidate_identified_fraction": len(identified) / len(branch_fates) if branch_fates else None,
                "branch_candidate_non_dominant_fraction": len(non_dom) / len(identified) if identified else None,
                "branch_candidate_category_counts": dict(
                    sorted(Counter(str(fate.get("candidate_category") or "unknown") for fate in branch_fates).items())
                ),
                "branch_candidate_survival_fraction_4g": (
                    sum(1 for fate in identified if fate.get("survived_4_generations")) / len(identified)
                    if identified
                    else None
                ),
            }
        )
    return summaries


def _paired_deltas(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(str(run["scenario"]), int(run["seed"])): run for run in runs}
    out: list[dict[str, Any]] = []
    for baseline, variant in PRIMARY_PAIRS:
        paired = [seed for seed in SEEDS if (baseline, seed) in by_key and (variant, seed) in by_key]
        deltas: dict[str, float | None] = {}
        for metric in KEY_METRICS:
            values = []
            for seed in paired:
                left = _float((by_key[(baseline, seed)].get("metrics") or {}).get(metric))
                right = _float((by_key[(variant, seed)].get("metrics") or {}).get(metric))
                if left is not None and right is not None:
                    values.append(float(right - left))
            deltas[metric] = _mean(values)
        out.append(
            {
                "comparison": f"{variant} - {baseline}",
                "baseline": baseline,
                "variant": variant,
                "paired_seed_count": len(paired),
                "paired_seeds": paired,
                "deltas": deltas,
            }
        )
    return out


def _integrity(matrix: Sequence[Mapping[str, str]], runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    seed_sets: dict[str, set[int]] = defaultdict(set)
    for run in runs:
        seed_sets[str(run["scenario"])].add(int(run["seed"]))
    completed = [run for run in runs if run["metrics_exists"] and int(run["metrics_rows"] or 0) > 0]
    return {
        "planned_runs": len(matrix),
        "completed_runs": len(completed),
        "all_runs_complete": len(completed) == len(matrix),
        "missing_metrics_csv": sum(1 for run in runs if not run["metrics_exists"]),
        "empty_metrics_csv": sum(1 for run in runs if run["metrics_exists"] and int(run["metrics_rows"] or 0) == 0),
        "critical_nonfinite_issues": sum(len(run.get("critical_nonfinite") or []) for run in runs),
        "env_mismatches": sum(len(run.get("env_mismatches") or []) for run in runs),
        "sim_mismatches": sum(len(run.get("sim_mismatches") or []) for run in runs),
        "theme4_gating_mismatches": sum(
            1
            for run in runs
            for mismatch in list(run.get("sim_mismatches") or [])
            if str(mismatch.get("field")) in EXPECTED_SIM and str(mismatch.get("field")).startswith("enable_")
            or str(mismatch.get("field")) == "phase_e_e1_enable"
        ),
        "paired_seed_sets_complete": all(seed_sets.get(scenario, set()) == set(SEEDS) for scenario in SCENARIOS),
        "seed_sets": {scenario: sorted(seed_sets.get(scenario, set())) for scenario in SCENARIOS},
        "lineage_strategy_json_present": sum(1 for run in completed if run["lineage_strategy_json_present"]),
        "lineage_strategy_json_nonempty": sum(1 for run in completed if run["lineage_strategy_json_nonempty"]),
        "lineage_branch_candidates_json_present": sum(
            1 for run in completed if run["lineage_branch_candidates_json_present"]
        ),
        "lineage_branch_candidates_json_nonempty": sum(
            1 for run in completed if run["lineage_branch_candidates_json_nonempty"]
        ),
        "archive_or_manifest_outputs_present": sum(
            1
            for run in completed
            if (run.get("archive_outputs") or {}).get("archive_path_exists")
            or (run.get("archive_outputs") or {}).get("branch_manifest_path_exists")
        ),
        "mismatch_details": [
            item
            for run in runs
            for item in list(run.get("env_mismatches") or []) + list(run.get("sim_mismatches") or [])
        ],
        "nonfinite_details": [
            {"scenario": run["scenario"], "seed": run["seed"], **item}
            for run in runs
            for item in list(run.get("critical_nonfinite") or [])
        ],
    }


def _lineage_fate_summary(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "integrated_r2_lineage_fate_summary_v1",
        "runs": [
            {
                "scenario": run["scenario"],
                "seed": run["seed"],
                "run_dir": run["run_dir"],
                "collapse_timing": run["collapse_timing"],
                "founder_dominance": run["founder_dominance"],
                "pre_collapse_differentiation": run["pre_collapse_differentiation"],
                "dominant_lineage_profile": run["dominant_lineage_profile"],
            }
            for run in sorted(runs, key=lambda item: (str(item["scenario"]), int(item["seed"])))
        ],
    }


def _branch_fate_summary(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_scenario: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    all_fates: list[dict[str, Any]] = []
    for run in runs:
        for fate in list(run.get("branch_candidate_fates") or []):
            item = {"scenario": run["scenario"], "seed": run["seed"], "run_dir": run["run_dir"], **dict(fate)}
            by_scenario[str(run["scenario"])].append(item)
            all_fates.append(item)
    scenario_summaries = {}
    for scenario in SCENARIOS:
        fates = by_scenario.get(scenario, [])
        identified = [item for item in fates if item.get("candidate_identified")]
        scenario_summaries[scenario] = {
            "candidate_count": len(fates),
            "identified_fraction": len(identified) / len(fates) if fates else None,
            "survived_1g_fraction": (
                sum(1 for item in identified if item.get("survived_1_generations")) / len(identified)
                if identified
                else None
            ),
            "survived_2g_fraction": (
                sum(1 for item in identified if item.get("survived_2_generations")) / len(identified)
                if identified
                else None
            ),
            "survived_4g_fraction": (
                sum(1 for item in identified if item.get("survived_4_generations")) / len(identified)
                if identified
                else None
            ),
            "survived_8g_fraction": (
                sum(1 for item in identified if item.get("survived_8_generations")) / len(identified)
                if identified
                else None
            ),
            "extinct_by_final_fraction": (
                sum(1 for item in identified if item.get("extinct_by_final")) / len(identified) if identified else None
            ),
            "category_counts": dict(
                sorted(Counter(str(item.get("candidate_category") or "unknown") for item in fates).items())
            ),
            "cause_proxy_counts": dict(
                sorted(Counter(str(item.get("cause_of_fate_proxy") or "unknown") for item in fates).items())
            ),
        }
    return {
        "schema_version": "integrated_r2_branch_candidate_fate_summary_v1",
        "scenario_summaries": scenario_summaries,
        "candidates": all_fates,
    }


def _archive_summary(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scenario_summaries: dict[str, Any] = {}
    for scenario in SCENARIOS:
        entries = [run for run in runs if run["scenario"] == scenario]
        scenario_summaries[scenario] = {
            "run_count": len(entries),
            "archive_path_runs": sum(1 for run in entries if (run.get("archive_outputs") or {}).get("archive_path_exists")),
            "branch_manifest_path_runs": sum(
                1 for run in entries if (run.get("archive_outputs") or {}).get("branch_manifest_path_exists")
            ),
            "archive_entry_count_total": sum(
                int((run.get("archive_outputs") or {}).get("archive_entry_count") or 0) for run in entries
            ),
            "manifest_entry_count_total": sum(
                int((run.get("archive_outputs") or {}).get("manifest_entry_count") or 0) for run in entries
            ),
        }
    return {
        "schema_version": "integrated_r2_archive_candidate_summary_v1",
        "scenario_summaries": scenario_summaries,
        "readiness": {
            "candidate_lineages_identifiable": "see branch_candidate_fate_summary identified_fraction",
            "manifest_outputs_present": any(
                (run.get("archive_outputs") or {}).get("branch_manifest_path_exists") for run in runs
            ),
            "archive_outputs_present": any((run.get("archive_outputs") or {}).get("archive_path_exists") for run in runs),
            "interpretation": "metadata-level unless archive or branch-manifest paths are populated",
        },
    }


def _mechanism_classification(
    summaries: Sequence[Mapping[str, Any]], integrity: Mapping[str, Any], branch_summary: Mapping[str, Any]
) -> dict[str, Any]:
    classes: dict[str, str] = {}
    if int(integrity.get("lineage_strategy_json_nonempty") or 0) < int(integrity.get("completed_runs") or 0) * 0.5:
        classes["G"] = "Diagnostics insufficient: lineage descriptors were not non-empty in enough runs."
    collapse_counts = [
        _float((item.get("collapse") or {}).get("collapsed_by_final_count")) or 0.0 for item in summaries
    ]
    total_runs = sum(int(item.get("run_count") or 0) for item in summaries)
    total_collapse = sum(collapse_counts)
    founder_alive_fracs = [
        _float((item.get("founder_dominance") or {}).get("gen1_top_alive_becomes_final_fraction"))
        for item in summaries
    ]
    founder_birth_fracs = [
        _float((item.get("founder_dominance") or {}).get("gen1_top_birth_becomes_final_fraction"))
        for item in summaries
    ]
    founder_signal = max([value for value in (*founder_alive_fracs, *founder_birth_fracs) if value is not None], default=0.0)
    if founder_signal >= 0.5:
        classes["A"] = (
            "Founder/early-reproduction dominance: generation-1 top alive or top-birth lineage often becomes final dominant."
        )
    diff_values = [
        value
        for item in summaries
        for value in (item.get("pre_collapse_differentiation_means") or {}).values()
        if _float(value) is not None and float(value) > 0.0
    ]
    if total_runs and total_collapse / total_runs >= 0.8 and diff_values:
        classes["C"] = (
            "Global selection bottleneck overwhelms niche differentiation: pre-collapse descriptor ranges exist but final collapse is widespread."
        )
    if branch_summary.get("schema_version") and not branch_summary.get("candidates"):
        classes["G"] = "Diagnostics insufficient: branch candidate fates were not emitted."
    scenario_by_name = {str(item["scenario"]): item for item in summaries}
    patch_collapse = _float(
        (scenario_by_name.get("r2_00_patch_observed", {}).get("collapse") or {}).get(
            "first_top1_share_ge_0_75_generation_mean"
        )
    )
    jungle_collapse = [
        _float((scenario_by_name.get(scenario, {}).get("collapse") or {}).get("first_top1_share_ge_0_75_generation_mean"))
        for scenario in SCENARIOS
        if scenario != "r2_00_patch_observed"
    ]
    jungle_collapse = [value for value in jungle_collapse if value is not None]
    if patch_collapse is not None and jungle_collapse:
        mean_jungle = _mean(jungle_collapse)
        if mean_jungle is not None and abs(mean_jungle - patch_collapse) <= 2.0:
            classes["D"] = (
                "Patch/Jungle pressure may be too weak to maintain distinct niches: collapse timing is close to patch-observed timing."
            )
    if not classes:
        classes["H"] = "Other / unresolved: no single mechanism crossed the deterministic reporting thresholds."
    return classes


def _recommended_next_step(classification: Mapping[str, str], branch_summary: Mapping[str, Any]) -> str:
    candidates = list(branch_summary.get("candidates") or [])
    identified = [item for item in candidates if item.get("candidate_identified")]
    if candidates and len(identified) / len(candidates) >= 0.75:
        return "Option 1: R3 branch-candidate replay design"
    if "G" in classification:
        return "Option 2: R2b lineage reporting refinement"
    if any(key in classification for key in ("C", "D", "E", "F")):
        return "Option 3: R2c ecology bottleneck experiment"
    return "Option 2: R2b lineage reporting refinement"


def _write_summary_csv(path: Path, summaries: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "scenario",
        "label",
        "run_count",
        "collapse_top1_ge_0_75_mean",
        "collapse_effective_le_1_5_mean",
        "gen1_top_alive_becomes_final_fraction",
        "gen1_top_birth_becomes_final_fraction",
        "branch_candidate_total",
        "branch_candidate_identified_fraction",
        "branch_candidate_survival_fraction_4g",
        *KEY_METRICS,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in summaries:
            row = {
                "scenario": item["scenario"],
                "label": item["label"],
                "run_count": item["run_count"],
                "collapse_top1_ge_0_75_mean": (item.get("collapse") or {}).get(
                    "first_top1_share_ge_0_75_generation_mean"
                ),
                "collapse_effective_le_1_5_mean": (item.get("collapse") or {}).get(
                    "first_effective_alive_lineages_le_1_5_generation_mean"
                ),
                "gen1_top_alive_becomes_final_fraction": (item.get("founder_dominance") or {}).get(
                    "gen1_top_alive_becomes_final_fraction"
                ),
                "gen1_top_birth_becomes_final_fraction": (item.get("founder_dominance") or {}).get(
                    "gen1_top_birth_becomes_final_fraction"
                ),
                "branch_candidate_total": item.get("branch_candidate_total"),
                "branch_candidate_identified_fraction": item.get("branch_candidate_identified_fraction"),
                "branch_candidate_survival_fraction_4g": item.get("branch_candidate_survival_fraction_4g"),
            }
            row.update({metric: (item.get("metrics") or {}).get(metric) for metric in KEY_METRICS})
            writer.writerow(row)


def _write_deltas_csv(path: Path, deltas: Sequence[Mapping[str, Any]]) -> None:
    fields = ["comparison", "baseline", "variant", "paired_seed_count", *KEY_METRICS]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in deltas:
            row = {
                "comparison": item["comparison"],
                "baseline": item["baseline"],
                "variant": item["variant"],
                "paired_seed_count": item["paired_seed_count"],
            }
            row.update({metric: (item.get("deltas") or {}).get(metric) for metric in KEY_METRICS})
            writer.writerow(row)


def _write_timing_csv(path: Path, runs: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "scenario",
        "seed",
        "run_dir",
        "first_effective_alive_lineages_le_1_5_generation",
        "first_entropy_le_0_5_generation",
        "first_top1_share_ge_0_75_generation",
        "first_top1_share_ge_0_90_generation",
        "first_all_but_one_extinct_generation",
        "final_alive_lineage_count",
        "final_active_lineage_count",
        "final_effective_alive_lineages",
        "final_alive_share_entropy",
        "final_top1_alive_share",
        "collapsed_by_final",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for run in sorted(runs, key=lambda item: (str(item["scenario"]), int(item["seed"]))):
            row = {"scenario": run["scenario"], "seed": run["seed"], "run_dir": run["run_dir"]}
            row.update(run.get("collapse_timing") or {})
            writer.writerow(row)


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _metric_table(summaries: Sequence[Mapping[str, Any]], metrics: Sequence[str]) -> str:
    rows = []
    for item in summaries:
        rows.append([item["scenario"], *[_fmt((item.get("metrics") or {}).get(metric)) for metric in metrics]])
    return _md_table(("Scenario", *metrics), rows)


def _delta_table(deltas: Sequence[Mapping[str, Any]], metrics: Sequence[str]) -> str:
    rows = []
    for item in deltas:
        rows.append([item["comparison"], item["paired_seed_count"], *[_fmt((item.get("deltas") or {}).get(metric)) for metric in metrics]])
    return _md_table(("Comparison", "Seeds", *metrics), rows)


def _write_report(
    path: Path,
    *,
    summary: Mapping[str, Any],
    lineage_fate: Mapping[str, Any],
    branch_fate: Mapping[str, Any],
    archive_summary: Mapping[str, Any],
    repeat_parity: Mapping[str, Any],
    worktree_info: Mapping[str, Any],
) -> None:
    integrity = summary["integrity"]
    summaries = summary["scenario_summaries"]
    deltas = summary["paired_deltas"]
    classification = summary["collapse_mechanism_classification"]
    recommendation = summary["recommended_next_step"]
    timing_rows = []
    for item in summaries:
        collapse = item.get("collapse") or {}
        timing_rows.append(
            [
                item["scenario"],
                _fmt(collapse.get("first_top1_share_ge_0_75_generation_mean")),
                _fmt(collapse.get("first_top1_share_ge_0_90_generation_mean")),
                _fmt(collapse.get("first_effective_alive_lineages_le_1_5_generation_mean")),
                _fmt(collapse.get("final_effective_alive_lineages_mean")),
                collapse.get("collapsed_by_final_count"),
            ]
        )
    founder_rows = [
        [
            item["scenario"],
            _fmt((item.get("founder_dominance") or {}).get("gen1_top_alive_becomes_final_fraction")),
            _fmt((item.get("founder_dominance") or {}).get("gen1_top_birth_becomes_final_fraction")),
        ]
        for item in summaries
    ]
    branch_rows = [
        [
            item["scenario"],
            item.get("branch_candidate_total"),
            _fmt(item.get("branch_candidate_identified_fraction")),
            _fmt(item.get("branch_candidate_non_dominant_fraction")),
            _fmt(item.get("branch_candidate_survival_fraction_4g")),
            json.dumps(item.get("branch_candidate_category_counts") or {}, sort_keys=True),
        ]
        for item in summaries
    ]
    archive_rows = [
        [
            scenario,
            values.get("archive_path_runs"),
            values.get("branch_manifest_path_runs"),
            values.get("archive_entry_count_total"),
            values.get("manifest_entry_count_total"),
        ]
        for scenario, values in (archive_summary.get("scenario_summaries") or {}).items()
    ]
    text = f"""# EIEL Integrated R2 Lineage-Collapse Forensics Report

## 1. Purpose

R2 investigates what happens between initial lineage diversity, pre-collapse
ecological differentiation, branch-candidate emission, and final one-lineage
dominance on the R1-validated integrated main branch.

R2 is forensic diagnostics only. It does not tune coefficients, change
mechanics, execute branches, add lineage preservation, or make cognition or
emergence claims.

## 2. Base Commit And R1 Dependency

Base commit: `{worktree_info.get('base_commit', '')}`.

R1 dependency: `{worktree_info.get('r1_commit', '')}` / `{worktree_info.get('r1_report_path', '')}`.

## 3. Worktree Path

`{worktree_info.get('worktree_path', '')}`

Git status before work:

```text
{str(worktree_info.get('git_status_before', '')).strip()}
```

## 4. Scenario Definitions

- `r2_00_patch_observed`: Patch World observed, without Jungle pressure.
- `r2_01_default_full_jungle`: current default full Jungle.
- `r2_02_tuned_full_jungle_no_recovery`: tuned pressure stack without recovery.
- `r2_03_tuned_full_jungle_recovery`: Campaign C/D tuned full Jungle recovery.
- `r2_04_theme12_clean_pressure_reference`: Theme 12 clean pressure reference.

## 5. Seed List / Settings / Device

Seeds: `{', '.join(str(seed) for seed in SEEDS)}`.

Settings: 16 generations, 120 steps/generation, population 64, grid 16,
device CUDA, bank device CUDA.

## 6. Preflight Results

The required preflight suites and registry dry-run are recorded in the final
task response. R2 execution should be interpreted as valid only when those gates
passed.

## 7. Repeat Parity Results

Repeat parity passed: `{repeat_parity.get('passed', '')}`.

## 8. Completion Table

| Check | Value |
| --- | --- |
| planned_runs | {integrity.get('planned_runs')} |
| completed_runs | {integrity.get('completed_runs')} |
| all_runs_complete | {integrity.get('all_runs_complete')} |
| missing_metrics_csv | {integrity.get('missing_metrics_csv')} |
| empty_metrics_csv | {integrity.get('empty_metrics_csv')} |
| paired_seed_sets_complete | {integrity.get('paired_seed_sets_complete')} |

## 9. Metric Integrity Table

| Check | Value |
| --- | --- |
| critical_nonfinite_issues | {integrity.get('critical_nonfinite_issues')} |
| env_mismatches | {integrity.get('env_mismatches')} |
| sim_mismatches | {integrity.get('sim_mismatches')} |
| theme4_gating_mismatches | {integrity.get('theme4_gating_mismatches')} |
| lineage_strategy_json_present | {integrity.get('lineage_strategy_json_present')} |
| lineage_strategy_json_nonempty | {integrity.get('lineage_strategy_json_nonempty')} |
| lineage_branch_candidates_json_present | {integrity.get('lineage_branch_candidates_json_present')} |
| lineage_branch_candidates_json_nonempty | {integrity.get('lineage_branch_candidates_json_nonempty')} |

## 10. Scenario Flag Validation

Expected scenario and Theme 4 gating flags were checked from `config.json`.
Mismatch details are stored in `r2_summary.json`.

## 11. Collapse Timing Table

{_md_table(('Scenario', 'Top1 >= 0.75 mean gen', 'Top1 >= 0.90 mean gen', 'Effective <= 1.5 mean gen', 'Final effective mean', 'Collapsed runs'), timing_rows)}

## 12. Founder Dominance Analysis

{_md_table(('Scenario', 'Gen1 top alive -> final', 'Gen1 top birth -> final'), founder_rows)}

## 13. Pre-Collapse Ecological Differentiation

Pre-collapse habitat, predator, reproduction, and recovery descriptor ranges
are recorded in `r2_lineage_fate_summary.json` and summarized in
`r2_summary.json` under `pre_collapse_differentiation_means`. Non-zero ranges
indicate ecological texture before collapse, not selected cognition.

## 14. Branch Candidate Fate Analysis

{_md_table(('Scenario', 'Candidates', 'Identified fraction', 'Non-dominant fraction', 'Survived 4g fraction', 'Categories'), branch_rows)}

Branch-candidate fate proxies are explicitly proxies. They should not be read
as exact causes of death or exact fitness explanations.

## 15. Dominant Lineage Profile Analysis

Dominant lineage profiles per run are stored in `r2_lineage_fate_summary.json`,
including first present generation, first top-alive generation, first top-birth
generation, ecological profile fields, and branch-candidate status.

## 16. Scenario Contrast

{_delta_table(deltas, ('alive', 'successful_births', 'jungle_predator_damage_total', 'patch_regrowth_suppressed_total', 'patch_recovery_added_regrowth_total', 'lineage_effective_alive_lineages'))}

## 17. Archive / Manifest Readiness

{_md_table(('Scenario', 'Archive path runs', 'Manifest path runs', 'Archive entries', 'Manifest entries'), archive_rows)}

Archive readiness summary: `{json.dumps(archive_summary.get('readiness') or {}, sort_keys=True)}`.

## 18. Collapse Mechanism Classification

{chr(10).join(f'- `{key}`: {value}' for key, value in classification.items())}

## 19. Recommended Next Step

`{recommendation}`

## 20. What Is Not Concluded

- R2 does not show that intelligence emerged.
- R2 does not show that cognition, perception, memory, or prediction was selected.
- R2 does not show that lineage diversity is solved.
- R2 does not show that Theme 5 preserves lineages.
- R2 does not show that branch archives improve evolution.

## 21. No-Claims Statement

R2 is a lineage-collapse forensic campaign. Its conclusions are limited to
reporting and interpretation of lineage diagnostics over this short controlled
horizon.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def summarize_batch(batch_dir: Path, repeat_parity_path: Path | None, worktree_info_path: Path | None) -> dict[str, Any]:
    matrix = _read_rows(batch_dir / "matrix.csv")
    runs = [_run_analysis(row) for row in matrix]
    summaries = _scenario_summaries(runs)
    deltas = _paired_deltas(runs)
    integrity = _integrity(matrix, runs)
    lineage_fate = _lineage_fate_summary(runs)
    branch_fate = _branch_fate_summary(runs)
    archive_summary = _archive_summary(runs)
    classification = _mechanism_classification(summaries, integrity, branch_fate)
    recommendation = _recommended_next_step(classification, branch_fate)
    return {
        "schema_version": "integrated_r2_summary_v1",
        "campaign_name": "integrated_r2_lineage_collapse_forensics",
        "batch_dir": str(batch_dir),
        "integrity": integrity,
        "scenario_summaries": summaries,
        "paired_deltas": deltas,
        "lineage_fate_summary": lineage_fate,
        "branch_candidate_fate_summary": branch_fate,
        "archive_candidate_summary": archive_summary,
        "collapse_mechanism_classification": classification,
        "recommended_next_step": recommendation,
        "repeat_parity": _read_json(repeat_parity_path) if repeat_parity_path else {},
        "worktree_info": _read_json(worktree_info_path) if worktree_info_path else {},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize integrated R2 lineage-collapse forensics")
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--deltas-csv", required=True)
    parser.add_argument("--timing-csv", required=True)
    parser.add_argument("--lineage-fate-json", required=True)
    parser.add_argument("--branch-fate-json", required=True)
    parser.add_argument("--archive-json", required=True)
    parser.add_argument("--report-md", required=True)
    parser.add_argument("--repeat-parity-json", default="")
    parser.add_argument("--worktree-info-json", default="")
    args = parser.parse_args(argv)

    summary = summarize_batch(
        Path(args.batch_dir),
        Path(args.repeat_parity_json) if args.repeat_parity_json else None,
        Path(args.worktree_info_json) if args.worktree_info_json else None,
    )
    _write_json(Path(args.out_json), summary)
    _write_json(Path(args.lineage_fate_json), summary["lineage_fate_summary"])
    _write_json(Path(args.branch_fate_json), summary["branch_candidate_fate_summary"])
    _write_json(Path(args.archive_json), summary["archive_candidate_summary"])
    _write_summary_csv(Path(args.summary_csv), summary["scenario_summaries"])
    _write_deltas_csv(Path(args.deltas_csv), summary["paired_deltas"])
    _write_timing_csv(Path(args.timing_csv), summary["lineage_fate_summary"]["runs"])
    _write_report(
        Path(args.report_md),
        summary=summary,
        lineage_fate=summary["lineage_fate_summary"],
        branch_fate=summary["branch_candidate_fate_summary"],
        archive_summary=summary["archive_candidate_summary"],
        repeat_parity=summary["repeat_parity"],
        worktree_info=summary["worktree_info"],
    )
    print(args.out_json)
    print(args.report_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
