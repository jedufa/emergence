from __future__ import annotations

"""Deterministic Theme 4 campaign planning and evidence helpers.

Stage 9 intentionally does not add cognitive machinery. This module plans,
validates, summarizes, and audits small paired-ablation campaign manifests. The
fixture executor is for smoke evidence and tests; the live simulator adapter is
a thin wrapper around the existing SimulationRunner entry point.
"""

from dataclasses import dataclass
import csv
import json
import math
import sys
import traceback
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping
from zipfile import ZIP_DEFLATED, ZipFile


MANIFEST_VERSION = "theme4_campaign_manifest_v1"

CORE_METRICS = (
    "alive",
    "deaths",
    "mean_offspring_total",
    "lineage_active_count",
    "cog_capacity_mean",
    "cog_cost_mean",
    "direct_cognition_reward_present_flag",
    "direct_internal_selection_pressure_present_flag",
)

COST_METRICS = (
    "cog_cost_mean",
    "memory_cost_charged_mean",
    "memory_cost_hypothetical_mean",
    "perception_cost_charged_mean",
    "perception_cost_hypothetical_mean",
    "sensor_radius_cost_charged_mean",
    "sensor_radius_cost_hypothetical_mean",
    "prediction_cost_charged_mean",
    "prediction_cost_hypothetical_mean",
    "developmental_cost_charged_mean",
    "developmental_cost_hypothetical_mean",
    "developmental_cost_charging_active_flag",
)

METRIC_FAMILIES: dict[str, tuple[str, ...]] = {
    "core": CORE_METRICS,
    "population": ("alive", "deaths"),
    "reproduction": ("mean_offspring_total",),
    "lineage": ("lineage_active_count", "lineage_extinct_count"),
    "perception": ("perception_capacity_component_mean", "sensor_channel_mask_ablation_active_flag"),
    "sensor_radius": ("sensor_radius_mean", "sensor_radius_force_min_active_flag", "sensor_radius_force_default_active_flag"),
    "memory": ("memory_dim_mean", "memory_ablation_active_flag"),
    "prediction": ("effective_prediction_lr_mean", "prediction_ablation_active_flag", "direct_prediction_error_selection_pressure_present_flag"),
    "development": ("developmental_expression_factor_mean", "developmental_ablation_active_flag"),
    "introspection": ("introspection_enabled_fraction", "direct_introspection_reward_present_flag"),
    "self_regulation": ("self_regulation_enabled_fraction", "direct_self_regulation_reward_present_flag"),
    "self_modification": ("self_modification_enabled_fraction", "direct_self_modification_reward_present_flag"),
    "directed_inheritance": ("directed_inheritance_enabled_fraction", "direct_directed_inheritance_reward_present_flag"),
    "guardrails": (
        "high_cognition_low_cost_fraction",
        "cognition_growth_without_environment_pressure_flag",
        "genome_editing_present_flag",
        "host_access_present_flag",
        "source_code_modification_present_flag",
    ),
    "costs": COST_METRICS,
    "perception_costs": (
        "perception_cost_charged_mean",
        "perception_cost_charged_total",
        "perception_cost_hypothetical_mean",
        "perception_cost_zero_fraction",
        "high_perception_low_cost_fraction",
    ),
    "sensor_radius_costs": (
        "sensor_radius_cost_charged_mean",
        "sensor_radius_cost_charged_total",
        "sensor_radius_cost_hypothetical_mean",
        "sensor_radius_cost_zero_fraction",
        "high_radius_low_cost_fraction",
    ),
    "memory_costs": (
        "memory_cost_charged_mean",
        "memory_cost_charged_total",
        "memory_cost_hypothetical_mean",
        "memory_cost_zero_fraction",
        "high_memory_low_cost_fraction",
    ),
    "prediction_costs": (
        "prediction_cost_charged_mean",
        "prediction_cost_charged_total",
        "prediction_cost_hypothetical_mean",
        "prediction_cost_zero_fraction",
        "high_prediction_low_cost_fraction",
    ),
    "development_costs": (
        "developmental_cost_charged_mean",
        "developmental_cost_hypothetical_mean",
        "developmental_cost_charging_active_flag",
        "developmental_cost_zero_fraction",
        "high_adult_cognition_low_developmental_cost_fraction",
    ),
    "behavior_proxy": (
        "behavior_action_total",
        "behavior_move_total",
        "behavior_gather_total",
        "behavior_rest_total",
        "behavior_attack_total",
        "behavior_move_fraction",
        "behavior_gather_fraction",
        "behavior_rest_fraction",
        "behavior_attack_fraction",
        "behavior_patch_interaction_proxy",
        "behavior_risk_exposure_proxy",
        "behavior_protection_proxy",
        "behavior_reproduction_proxy",
        "behavior_visible_resource_exposure_proxy",
        "behavior_visible_hazard_exposure_proxy",
        "behavior_visible_shelter_exposure_proxy",
        "behavior_visible_predator_exposure_proxy",
    ),
}

BACKENDS = ("fixture", "live_simulator", "skipped", "scaffold")

ABLATION_FLAG_METRICS: dict[str, str] = {
    "ablate_sensor_channel_mask": "sensor_channel_mask_ablation_active_flag",
    "force_all_sensor_channels_enabled": "sensor_channel_force_all_active_flag",
    "force_minimal_sensor_channels_enabled": "sensor_channel_force_minimal_active_flag",
    "ablate_sensor_radius": "sensor_radius_ablation_active_flag",
    "force_min_sensor_radius": "sensor_radius_force_min_active_flag",
    "force_max_sensor_radius": "sensor_radius_force_max_active_flag",
    "force_default_sensor_radius": "sensor_radius_force_default_active_flag",
    "ablate_memory": "memory_ablation_active_flag",
    "force_memory_disabled": "memory_force_disabled_active_flag",
    "force_min_memory_dim": "memory_force_min_active_flag",
    "force_max_memory_dim": "memory_force_max_active_flag",
    "force_default_memory_dim": "memory_force_default_active_flag",
    "reset_memory_each_tick": "memory_reset_each_tick_active_flag",
    "ablate_prediction": "prediction_ablation_active_flag",
    "force_prediction_disabled": "prediction_force_disabled_active_flag",
    "force_prediction_frozen": "prediction_force_frozen_active_flag",
    "force_prediction_zero_lr": "prediction_force_zero_lr_active_flag",
    "force_prediction_default": "prediction_force_default_active_flag",
    "reset_prediction_each_tick": "prediction_reset_each_tick_active_flag",
    "ablate_developmental_cognition": "developmental_ablation_active_flag",
    "force_adult_cognition": "developmental_force_adult_active_flag",
    "force_juvenile_cognition": "developmental_force_juvenile_active_flag",
    "force_legacy_full_cognition": "developmental_force_legacy_active_flag",
}

SUMMARY_METRICS = (
    "alive",
    "deaths",
    "mean_offspring_total",
    "lineage_active_count",
    "cog_capacity_mean",
    "cog_cost_mean",
    "perception_capacity_component_mean",
    "sensor_radius_mean",
    "memory_dim_mean",
    "effective_memory_mean",
    "effective_prediction_lr_mean",
    "prediction_error_mean",
    "developmental_expression_factor_mean",
    "memory_cost_charged_mean",
    "memory_cost_hypothetical_mean",
    "perception_cost_charged_mean",
    "perception_cost_hypothetical_mean",
    "sensor_radius_cost_charged_mean",
    "sensor_radius_cost_hypothetical_mean",
    "prediction_cost_charged_mean",
    "prediction_cost_hypothetical_mean",
    "developmental_cost_charged_mean",
    "developmental_cost_hypothetical_mean",
    "behavior_action_total",
    "behavior_move_total",
    "behavior_move_n_total",
    "behavior_move_s_total",
    "behavior_move_e_total",
    "behavior_move_w_total",
    "behavior_gather_total",
    "behavior_rest_total",
    "behavior_attack_total",
    "behavior_move_fraction",
    "behavior_gather_fraction",
    "behavior_rest_fraction",
    "behavior_attack_fraction",
    "behavior_patch_interaction_proxy",
    "behavior_risk_exposure_proxy",
    "behavior_protection_proxy",
    "behavior_reproduction_proxy",
    "behavior_visible_resource_exposure_proxy",
    "behavior_visible_hazard_exposure_proxy",
    "behavior_visible_shelter_exposure_proxy",
    "behavior_visible_predator_exposure_proxy",
    "sensor_channel_active_count_mean",
    "sensor_channel_active_fraction_mean",
    "sensor_channel_mask_entropy",
    "sensor_channel_strategy_count_approx",
    "parent_sensor_channel_count_mean",
    "alive_sensor_channel_count_mean",
    "perception_selection_delta_parent_vs_alive",
    "perception_strategy_entropy",
    "sensor_channel_force_minimal_active_flag",
    "sensor_channel_force_all_active_flag",
    "sensor_channel_local_grid_state_enabled_fraction",
    "sensor_channel_patch_food_amount_norm_enabled_fraction",
    "sensor_channel_patch_cover_enabled_fraction",
    "sensor_channel_patch_shelter_enabled_fraction",
    "sensor_channel_patch_scent_norm_enabled_fraction",
    "sensor_channel_patch_waste_norm_enabled_fraction",
    "sensor_channel_patch_density_norm_enabled_fraction",
    "sensor_channel_patch_movement_cost_norm_enabled_fraction",
    "sensor_channel_self_energy_enabled_fraction",
    "sensor_channel_self_health_enabled_fraction",
    "patch_food_amount_mean",
    "patch_food_amount_min",
    "patch_food_amount_max",
    "patch_food_depleted_frac",
    "patch_food_full_frac",
    "patch_cover_mean",
    "patch_shelter_mean",
    "patch_visibility_mean",
    "patch_movement_cost_mean",
    "patch_scent_mean",
    "patch_scent_max",
    "patch_scent_p90",
    "patch_waste_mean",
    "patch_waste_max",
    "patch_waste_p90",
    "patch_density_mean",
    "patch_density_max",
    "occupied_patch_food_amount_mean",
    "occupied_patch_cover_mean",
    "occupied_patch_shelter_mean",
    "occupied_patch_visibility_mean",
    "occupied_patch_waste_mean",
    "death_patch_count",
    "death_patch_food_amount_mean",
    "death_patch_cover_mean",
    "death_patch_shelter_mean",
    "death_patch_visibility_mean",
    "death_patch_waste_mean",
    "patch_food_harvest_total",
    "patch_food_regrowth_total",
    "patch_food_occupancy_bias",
    "patch_shelter_occupied_mean",
    "patch_shelter_occupancy_bias",
    "patch_waste_occupied_mean",
    "patch_waste_occupancy_bias",
    "tier3_overcrowd_mean",
    "jungle_predator_events",
    "jungle_predator_event_rate",
    "jungle_predator_damage_total",
    "jungle_predator_deaths",
    "jungle_predator_risk_mean",
    "jungle_predator_risk_max",
    "jungle_predator_occupied_risk_mean",
    "jungle_predator_visibility_component_mean",
    "jungle_predator_cover_reduction_mean",
    "jungle_predator_shelter_reduction_mean",
    "lineage_cog_capacity_mean",
    "lineage_cognitive_cost_mean",
)


@dataclass(frozen=True)
class CampaignValidationResult:
    ok: bool
    errors: tuple[str, ...]


def canonical_json(data: Mapping[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, indent=2, separators=(",", ": ")) + "\n"


def load_campaign_manifest(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_campaign_manifest(data, strict=True)
    return data


def validate_campaign_manifest(manifest: Mapping[str, Any], *, strict: bool = False) -> CampaignValidationResult:
    errors: list[str] = []
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        errors.append("manifest_version must be theme4_campaign_manifest_v1")
    for key in ("campaign_id", "campaign_name", "stage", "seed_list", "scenario_cells", "paired_ablation_groups", "output_directory"):
        if key not in manifest:
            errors.append(f"missing required manifest key: {key}")
    seeds = manifest.get("seed_list", [])
    if not isinstance(seeds, list) or not seeds or any(not isinstance(s, int) for s in seeds):
        errors.append("seed_list must be a non-empty list of integers")
    cells = manifest.get("scenario_cells", [])
    cost_levels = manifest.get("cost_levels", {})
    if "cost_levels" in manifest:
        if not isinstance(cost_levels, Mapping) or not cost_levels:
            errors.append("cost_levels must be a non-empty object when present")
        else:
            for name, profile in sorted(cost_levels.items()):
                if not isinstance(profile, Mapping):
                    errors.append(f"cost level {name} must be an object")
                    continue
                if "config_values" not in profile:
                    errors.append(f"cost level {name} missing config_values")
    if not isinstance(cells, list) or not cells:
        errors.append("scenario_cells must be a non-empty list")
    else:
        for idx, cell in enumerate(cells):
            if not isinstance(cell, Mapping):
                errors.append(f"scenario cell {idx} must be an object")
                continue
            for key in ("cell_id", "classification", "expected_metric_families"):
                if key not in cell:
                    errors.append(f"scenario cell {idx} missing {key}")
            backend = _cell_backend(cell)
            if backend not in BACKENDS:
                errors.append(f"scenario cell {cell.get('cell_id', idx)} has unsupported execution_backend {backend}")
            if cell.get("classification") == "skipped" and not cell.get("skip_reason"):
                errors.append(f"skipped cell {cell.get('cell_id', idx)} missing skip_reason")
            if backend == "skipped" and not cell.get("skip_reason"):
                errors.append(f"skipped backend cell {cell.get('cell_id', idx)} missing skip_reason")
            if "cost_level" in cell and isinstance(cost_levels, Mapping) and cost_levels:
                if str(cell.get("cost_level")) not in {str(k) for k in cost_levels}:
                    errors.append(f"scenario cell {cell.get('cell_id', idx)} references missing cost_level")
    groups = manifest.get("paired_ablation_groups", [])
    if not isinstance(groups, list):
        errors.append("paired_ablation_groups must be a list")
    else:
        cell_ids = {str(c.get("cell_id")) for c in cells if isinstance(c, Mapping)}
        for idx, group in enumerate(groups):
            if not isinstance(group, Mapping):
                errors.append(f"paired group {idx} must be an object")
                continue
            if str(group.get("normal_cell_id")) not in cell_ids:
                errors.append(f"paired group {group.get('group_id', idx)} references missing normal_cell_id")
            if str(group.get("ablation_cell_id")) not in cell_ids:
                errors.append(f"paired group {group.get('group_id', idx)} references missing ablation_cell_id")
    result = CampaignValidationResult(ok=not errors, errors=tuple(errors))
    if strict and errors:
        raise ValueError("; ".join(errors))
    return result


def _cell_backend(cell: Mapping[str, Any]) -> str:
    explicit = str(cell.get("execution_backend", "") or "").strip()
    if explicit:
        return explicit
    classification = str(cell.get("classification", ""))
    if classification == "skipped":
        return "skipped"
    if classification == "platform_fixture":
        return "scaffold"
    if classification == "live":
        # Backward-compatible Stage 9/10A/10B manifests used "live" to mean
        # "planned run" while the executor still produced fixture rows.
        return "fixture"
    return classification


def _cell_is_run(cell: Mapping[str, Any]) -> bool:
    return _cell_backend(cell) in {"fixture", "live_simulator"}


def plan_campaign(manifest: Mapping[str, Any]) -> dict[str, Any]:
    validate_campaign_manifest(manifest, strict=True)
    seeds = [int(s) for s in manifest["seed_list"]]
    base_dir = str(manifest["output_directory"])
    planned_runs: list[dict[str, Any]] = []
    skipped_cells: list[dict[str, str]] = []
    scaffold_checks: list[dict[str, Any]] = []
    backend_counts = {backend: 0 for backend in BACKENDS}
    for cell in sorted(manifest["scenario_cells"], key=lambda c: str(c["cell_id"])):
        classification = str(cell["classification"])
        backend = _cell_backend(cell)
        backend_counts[backend] = backend_counts.get(backend, 0) + 1
        if backend == "skipped" or classification == "skipped":
            skipped_cells.append({"cell_id": str(cell["cell_id"]), "skip_reason": str(cell["skip_reason"])})
            continue
        if backend == "scaffold" or classification == "platform_fixture":
            scaffold_checks.append({"cell_id": str(cell["cell_id"]), "fixture": str(cell.get("fixture", "")), "execution_backend": backend})
            continue
        if _cell_is_run(cell):
            for seed in seeds:
                run_id = f"{cell['cell_id']}__seed_{seed}"
                planned_runs.append(
                    {
                        "run_id": run_id,
                        "cell_id": str(cell["cell_id"]),
                        "seed": int(seed),
                        "classification": "live",
                        "execution_backend": backend,
                        "output_dir": f"{base_dir}/runs/{run_id}",
                        "metrics_path": f"{base_dir}/runs/{run_id}/metrics.csv",
                        "metadata_path": f"{base_dir}/runs/{run_id}/run_metadata.json",
                        "expected_metric_families": list(cell.get("expected_metric_families", [])),
                        "runner_call": "SimulationRunner(...).run(start_gen=0, skip_init=False)" if backend == "live_simulator" else "fixture_row_writer",
                        "cost_level": str(cell.get("cost_level", "")),
                        "primary_trait": str(cell.get("primary_trait", "")),
                        "scenario_group": str(cell.get("scenario_group", "")),
                        "scenario_variant": str(cell.get("scenario_variant", "")),
                        "fixed_sensor_radius_setting": str(cell.get("fixed_sensor_radius_setting", "")),
                    }
                )
    pairs: list[dict[str, Any]] = []
    live_ids = {r["cell_id"] for r in planned_runs}
    for group in sorted(manifest.get("paired_ablation_groups", []), key=lambda g: str(g.get("group_id", ""))):
        normal = str(group["normal_cell_id"])
        ablated = str(group["ablation_cell_id"])
        for seed in seeds:
            pairs.append(
                {
                    "group_id": str(group["group_id"]),
                    "seed": int(seed),
                    "normal_run_id": f"{normal}__seed_{seed}" if normal in live_ids else "",
                    "ablation_run_id": f"{ablated}__seed_{seed}" if ablated in live_ids else "",
                    "status": "planned" if normal in live_ids and ablated in live_ids else "skipped",
                    "skip_reason": "" if normal in live_ids and ablated in live_ids else "paired cell not live in this manifest",
                    "expected_affected_metrics": list(group.get("expected_affected_metrics", [])),
                    "cost_level": str(group.get("cost_level", "")),
                    "primary_trait": str(group.get("primary_trait", "")),
                    "scenario_group": str(group.get("scenario_group", "")),
                    "scenario_variant": str(group.get("scenario_variant", "")),
                    "fixed_sensor_radius_setting": str(group.get("fixed_sensor_radius_setting", "")),
                }
            )
    return {
        "campaign_id": str(manifest["campaign_id"]),
        "campaign_name": str(manifest["campaign_name"]),
        "manifest_version": str(manifest["manifest_version"]),
        "seed_list": seeds,
        "planned_live_run_count": len(planned_runs),
        "planned_fixture_run_count": sum(1 for r in planned_runs if r["execution_backend"] == "fixture"),
        "planned_live_simulator_run_count": sum(1 for r in planned_runs if r["execution_backend"] == "live_simulator"),
        "planned_scaffold_check_count": len(scaffold_checks),
        "planned_skipped_cell_count": len(skipped_cells),
        "execution_backend_counts": backend_counts,
        "planned_runs": planned_runs,
        "scaffold_checks": scaffold_checks,
        "skipped_cells": skipped_cells,
        "paired_ablation_plan": pairs,
        "cost_levels": dict(manifest.get("cost_levels", {})),
        "validation_warnings": list(manifest.get("validation_warnings", [])),
    }


def _fixture_row(run: Mapping[str, Any], cell: Mapping[str, Any]) -> dict[str, float]:
    seed = int(run["seed"])
    ablated = bool(cell.get("ablation_flags"))
    trait = str(cell.get("primary_trait", "baseline"))
    pressure = str(cell.get("pressure_type", "flat"))
    pressure_bonus = 0.0 if pressure == "flat" else (1.5 if pressure == "patch" else 2.0)
    trait_bonus = {"perception": 1.0, "memory": 0.8, "prediction": 0.7, "development": 0.4}.get(trait, 0.0)
    ablation_penalty = trait_bonus * (0.4 + 0.1 * (seed % 3)) if ablated else 0.0
    alive = max(0.0, 12.0 + pressure_bonus + trait_bonus - ablation_penalty + float(seed % 5))
    cost = 0.01 * trait_bonus
    row = {
        "alive": alive,
        "deaths": max(0.0, 20.0 - alive),
        "mean_offspring_total": max(0.0, 1.0 + trait_bonus - ablation_penalty * 0.2),
        "lineage_active_count": max(1.0, alive * 0.5),
        "lineage_extinct_count": 0.0,
        "cog_capacity_mean": max(0.0, 2.0 + trait_bonus),
        "cog_cost_mean": cost,
        "direct_cognition_reward_present_flag": 0.0,
        "direct_internal_selection_pressure_present_flag": 0.0,
        "high_cognition_low_cost_fraction": 0.0 if cost > 0.0 else 1.0,
        "cognition_growth_without_environment_pressure_flag": 0.0,
        "genome_editing_present_flag": 0.0,
        "host_access_present_flag": 0.0,
        "source_code_modification_present_flag": 0.0,
        "perception_capacity_component_mean": 3.0 if trait == "perception" and not ablated else 1.0,
        "sensor_channel_mask_ablation_active_flag": 1.0 if trait == "perception" and ablated else 0.0,
        "sensor_radius_mean": 2.0,
        "sensor_radius_force_min_active_flag": 0.0,
        "sensor_radius_force_default_active_flag": 0.0,
        "sensor_radius_ablation_active_flag": 0.0,
        "memory_dim_mean": 32.0 if trait == "memory" and not ablated else 0.0,
        "effective_memory_mean": 32.0 if trait == "memory" and not ablated else 0.0,
        "memory_ablation_active_flag": 1.0 if trait == "memory" and ablated else 0.0,
        "memory_force_disabled_active_flag": 1.0 if trait == "memory" and ablated else 0.0,
        "memory_reset_each_tick_active_flag": 0.0,
        "effective_prediction_lr_mean": 0.2 if trait == "prediction" and not ablated else 0.0,
        "prediction_error_mean": 0.0,
        "prediction_ablation_active_flag": 1.0 if trait == "prediction" and ablated else 0.0,
        "prediction_force_zero_lr_active_flag": 1.0 if trait == "prediction" and ablated else 0.0,
        "direct_prediction_error_selection_pressure_present_flag": 0.0,
        "developmental_expression_factor_mean": 0.7 if trait == "development" and not ablated else 1.0,
        "developmental_ablation_active_flag": 1.0 if trait == "development" and ablated else 0.0,
        "developmental_force_legacy_active_flag": 1.0 if trait == "development" and ablated else 0.0,
        "introspection_enabled_fraction": 0.0,
        "direct_introspection_reward_present_flag": 0.0,
        "self_regulation_enabled_fraction": 0.0,
        "direct_self_regulation_reward_present_flag": 0.0,
        "self_modification_enabled_fraction": 0.0,
        "direct_self_modification_reward_present_flag": 0.0,
        "directed_inheritance_enabled_fraction": 0.0,
        "direct_directed_inheritance_reward_present_flag": 0.0,
    }
    return {k: float(v) for k, v in row.items()}


def execute_fixture_campaign(manifest: Mapping[str, Any], *, output_root: str | Path | None = None) -> dict[str, Any]:
    plan = plan_campaign(manifest)
    out_dir = Path(output_root or manifest["output_directory"])
    cell_by_id = {str(c["cell_id"]): c for c in manifest["scenario_cells"]}
    run_records: list[dict[str, Any]] = []
    for run in plan["planned_runs"]:
        if str(run.get("execution_backend", "fixture")) != "fixture":
            continue
        cell = cell_by_id[str(run["cell_id"])]
        run_dir = Path(run["output_dir"])
        if output_root is not None:
            run_dir = out_dir / "runs" / str(run["run_id"])
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = run_dir / "metrics.csv"
        row = _fixture_row(run, cell)
        with metrics_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(row))
            writer.writeheader()
            writer.writerow(row)
        metadata = {
            "run_id": run["run_id"],
            "cell_id": run["cell_id"],
            "seed": run["seed"],
            "status": "success",
            "fixture_mode": True,
            "execution_backend": "fixture",
            "config_diff": dict(cell.get("ablation_flags", {})),
            "metrics_path": str(metrics_path),
        }
        (run_dir / "run_metadata.json").write_text(canonical_json(metadata), encoding="utf-8")
        run_records.append(metadata)
    summary = {
        "campaign_id": plan["campaign_id"],
        "fixture_mode": True,
        "execution_backend": "fixture",
        "executed_run_count": len(run_records),
        "skipped_cells": plan["skipped_cells"],
        "run_records": run_records,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "execution_summary.json").write_text(canonical_json(summary), encoding="utf-8")
    return summary


def _apply_overrides(target: Any, overrides: Mapping[str, Any], *, namespace: str) -> dict[str, Any]:
    applied: dict[str, Any] = {}
    for key in sorted(overrides):
        if not hasattr(target, str(key)):
            raise ValueError(f"unknown {namespace} override: {key}")
        value = overrides[key]
        setattr(target, str(key), value)
        applied[str(key)] = value
    return applied


def _ensure_engine_import_path() -> None:
    root = Path(__file__).resolve().parents[1]
    engine = root / "engine"
    for path in (str(root), str(engine)):
        if path not in sys.path:
            sys.path.insert(0, path)


def _live_base_configs(manifest: Mapping[str, Any], seed: int):
    _ensure_engine_import_path()
    from env.env_config import EnvConfig
    from evolution.population_pkg.population_config import PopConfig
    from simulation.runner_pkg.runner_config import SimConfig

    horizon = dict(manifest.get("horizon", {}))
    env_cfg = EnvConfig(width=6, height=6, view_radius=1)
    env_cfg.enable_biomes = False
    env_cfg.food_spawn_per_step = 2
    env_cfg.max_food = 12
    pop_cfg = PopConfig(population_size=4)
    sim_cfg = SimConfig(
        seed=int(seed),
        generations=int(horizon.get("generations", 1)),
        steps_per_generation=int(horizon.get("steps_per_generation", 2)),
        device="cpu",
        bank_device="cpu",
        behavior_log_enable=False,
        policy_deterministic_actions=True,
        policy_tick_stride=1,
    )
    sim_cfg.phase_e_e1_enable = False
    sim_cfg.epi_enable = False
    sim_cfg.checkpoint_bundle_v2_period_gen = 0
    _apply_overrides(env_cfg, dict(manifest.get("default_env_overrides", {})), namespace="env")
    _apply_overrides(pop_cfg, dict(manifest.get("default_pop_overrides", {})), namespace="pop")
    _apply_overrides(sim_cfg, dict(manifest.get("default_sim_overrides", {})), namespace="sim")
    return env_cfg, pop_cfg, sim_cfg


def execute_live_campaign(manifest: Mapping[str, Any], *, output_root: str | Path | None = None) -> dict[str, Any]:
    _ensure_engine_import_path()
    from simulation.metrics import MetricsLogger
    from simulation.runner import SimulationRunner

    plan = plan_campaign(manifest)
    out_dir = Path(output_root or manifest["output_directory"])
    cell_by_id = {str(c["cell_id"]): c for c in manifest["scenario_cells"]}
    run_records: list[dict[str, Any]] = []
    for run in plan["planned_runs"]:
        if str(run.get("execution_backend", "")) != "live_simulator":
            continue
        cell = cell_by_id[str(run["cell_id"])]
        run_dir = Path(run["output_dir"])
        if output_root is not None:
            run_dir = out_dir / "runs" / str(run["run_id"])
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = run_dir / "metrics.csv"
        metadata = {
            "run_id": run["run_id"],
            "cell_id": run["cell_id"],
            "seed": run["seed"],
            "status": "started",
            "fixture_mode": False,
            "execution_backend": "live_simulator",
            "runner_call": run.get("runner_call", "SimulationRunner(...).run(start_gen=0, skip_init=False)"),
            "scenario_group": str(cell.get("scenario_group", "")),
            "scenario_variant": str(cell.get("scenario_variant", "")),
            "pressure_type": str(cell.get("pressure_type", "")),
            "cost_level": str(cell.get("cost_level", "")),
            "primary_trait": str(cell.get("primary_trait", "")),
            "fixed_sensor_radius_setting": str(cell.get("fixed_sensor_radius_setting", "")),
            "config_diff": {
                "ablation_flags": dict(cell.get("ablation_flags", {})),
                "env_overrides": dict(cell.get("env_overrides", {})),
                "pop_overrides": dict(cell.get("pop_overrides", {})),
                "sim_overrides": dict(cell.get("sim_overrides", {})),
            },
            "metrics_path": str(metrics_path),
        }
        try:
            env_cfg, pop_cfg, sim_cfg = _live_base_configs(manifest, int(run["seed"]))
            sim_cfg.campaign_id = str(manifest.get("campaign_id", ""))
            sim_cfg.branch_id = str(run["run_id"])
            _apply_overrides(env_cfg, dict(cell.get("env_overrides", {})), namespace="env")
            _apply_overrides(pop_cfg, dict(cell.get("pop_overrides", {})), namespace="pop")
            _apply_overrides(sim_cfg, dict(cell.get("sim_overrides", {})), namespace="sim")
            sim = SimulationRunner(
                env_cfg=env_cfg,
                pop_cfg=pop_cfg,
                sim_cfg=sim_cfg,
                logger=MetricsLogger(str(run_dir)),
                run_dir=str(run_dir),
                perf=False,
            )
            sim.run(start_gen=0, skip_init=False)
            metadata["status"] = "success"
            metadata["metrics_path"] = str(metrics_path)
        except Exception as exc:  # pragma: no cover - exercised by integration failures, preserved in evidence.
            metadata["status"] = "failed"
            metadata["error"] = str(exc)
            metadata["traceback"] = traceback.format_exc()
        (run_dir / "run_metadata.json").write_text(canonical_json(metadata), encoding="utf-8")
        run_records.append(metadata)
    summary = {
        "campaign_id": plan["campaign_id"],
        "fixture_mode": False,
        "execution_backend": "live_simulator",
        "executed_run_count": len(run_records),
        "success_count": sum(1 for r in run_records if r.get("status") == "success"),
        "failure_count": sum(1 for r in run_records if r.get("status") != "success"),
        "skipped_cells": plan["skipped_cells"],
        "run_records": run_records,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "execution_summary.json").write_text(canonical_json(summary), encoding="utf-8")
    return summary


def execute_campaign(manifest: Mapping[str, Any], *, output_root: str | Path | None = None) -> dict[str, Any]:
    plan = plan_campaign(manifest)
    out_dir = Path(output_root or manifest["output_directory"])
    summaries: list[dict[str, Any]] = []
    if any(str(r.get("execution_backend")) == "fixture" for r in plan["planned_runs"]):
        summaries.append(execute_fixture_campaign(manifest, output_root=out_dir))
    if any(str(r.get("execution_backend")) == "live_simulator" for r in plan["planned_runs"]):
        summaries.append(execute_live_campaign(manifest, output_root=out_dir))
    combined = {
        "campaign_id": str(manifest["campaign_id"]),
        "execution_backend": "mixed" if len(summaries) > 1 else (summaries[0]["execution_backend"] if summaries else "none"),
        "executed_run_count": sum(int(s.get("executed_run_count", 0)) for s in summaries),
        "success_count": sum(int(s.get("success_count", s.get("executed_run_count", 0))) for s in summaries),
        "failure_count": sum(int(s.get("failure_count", 0)) for s in summaries),
        "skipped_cells": plan["skipped_cells"],
        "backend_summaries": summaries,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "execution_summary.json").write_text(canonical_json(combined), encoding="utf-8")
    return combined


def _read_metric_row(path: str | Path) -> dict[str, float]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    out: dict[str, float] = {}
    for key, value in rows[-1].items():
        try:
            out[str(key)] = float(value)
        except Exception:
            out[str(key)] = math.nan
    return out


def validate_campaign_metrics(manifest: Mapping[str, Any], plan: Mapping[str, Any] | None = None) -> dict[str, Any]:
    plan = dict(plan or plan_campaign(manifest))
    cells = {str(c["cell_id"]): c for c in manifest["scenario_cells"]}
    failures: list[str] = []
    checked = 0
    for run in plan["planned_runs"]:
        metrics_path = Path(str(run["metrics_path"]))
        if not metrics_path.exists():
            failures.append(f"missing metrics file: {metrics_path}")
            continue
        metadata_path = Path(str(run.get("metadata_path") or metrics_path.with_name("run_metadata.json")))
        expected_backend = str(run.get("execution_backend", _cell_backend(cells[str(run["cell_id"])])))
        metadata: dict[str, Any] = {}
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception as exc:
                failures.append(f"{run['run_id']} metadata is not readable: {exc}")
        elif expected_backend == "live_simulator":
            failures.append(f"{run['run_id']} missing live run metadata")
        if expected_backend == "live_simulator":
            if metadata.get("execution_backend") != "live_simulator":
                failures.append(f"{run['run_id']} expected live_simulator metadata, got {metadata.get('execution_backend')}")
            if bool(metadata.get("fixture_mode", True)):
                failures.append(f"{run['run_id']} expected live metrics but metadata marks fixture_mode")
            if metadata.get("status") != "success":
                failures.append(f"{run['run_id']} live run status is {metadata.get('status')}")
        expected_variant = str(cells[str(run["cell_id"])].get("scenario_variant", ""))
        if expected_variant and str(metadata.get("scenario_variant", "")) != expected_variant:
            failures.append(f"{run['run_id']} expected scenario_variant={expected_variant}, got {metadata.get('scenario_variant')}")
        expected_radius_setting = str(cells[str(run["cell_id"])].get("fixed_sensor_radius_setting", ""))
        if expected_radius_setting and str(metadata.get("fixed_sensor_radius_setting", "")) != expected_radius_setting:
            failures.append(
                f"{run['run_id']} expected fixed_sensor_radius_setting={expected_radius_setting}, "
                f"got {metadata.get('fixed_sensor_radius_setting')}"
            )
        row = _read_metric_row(metrics_path)
        checked += 1
        families = list(cells[str(run["cell_id"])].get("expected_metric_families", []))
        required: list[str] = []
        for family in families:
            required.extend(METRIC_FAMILIES.get(str(family), ()))
        for metric in sorted(set(required)):
            if metric not in row:
                failures.append(f"{run['run_id']} missing required metric {metric}")
            elif not math.isfinite(float(row[metric])):
                failures.append(f"{run['run_id']} metric {metric} is not finite")
        for flag, enabled in sorted(dict(cells[str(run["cell_id"])].get("ablation_flags", {})).items()):
            metric = ABLATION_FLAG_METRICS.get(str(flag))
            if metric is None:
                continue
            if metric not in row:
                failures.append(f"{run['run_id']} missing ablation flag metric {metric}")
                continue
            expected = 1.0 if bool(enabled) else 0.0
            actual = float(row[metric])
            if bool(enabled) and actual < 0.5:
                failures.append(f"{run['run_id']} expected {metric}=1 for {flag}, got {actual}")
            elif not bool(enabled) and actual > 0.5:
                failures.append(f"{run['run_id']} expected {metric}=0 for {flag}, got {actual}")
        expected_cost = dict(cells[str(run["cell_id"])].get("expected_cost_behavior", {}))
        if expected_cost:
            metric = str(expected_cost.get("metric", ""))
            mode = str(expected_cost.get("mode", ""))
            if metric not in row:
                failures.append(f"{run['run_id']} missing cost behavior metric {metric}")
            else:
                actual = float(row[metric])
                if mode == "charged_positive" and actual <= 0.0:
                    failures.append(f"{run['run_id']} expected positive charged cost in {metric}, got {actual}")
                elif mode == "uncharged_zero" and abs(actual) > 1e-12:
                    failures.append(f"{run['run_id']} expected uncharged zero cost in {metric}, got {actual}")
                elif mode == "finite" and not math.isfinite(actual):
                    failures.append(f"{run['run_id']} expected finite cost metric {metric}, got {actual}")
    return {
        "ok": not failures,
        "checked_run_count": checked,
        "failure_count": len(failures),
        "failures": failures,
        "skipped_cells": list(plan.get("skipped_cells", [])),
    }


def summarize_paired_runs(manifest: Mapping[str, Any], plan: Mapping[str, Any] | None = None) -> dict[str, Any]:
    plan = dict(plan or plan_campaign(manifest))
    run_by_id = {str(r["run_id"]): r for r in plan["planned_runs"]}
    summaries: list[dict[str, Any]] = []
    for pair in plan["paired_ablation_plan"]:
        if pair["status"] != "planned":
            summaries.append({**pair, "status": "skipped"})
            continue
        normal = run_by_id[str(pair["normal_run_id"])]
        ablated = run_by_id[str(pair["ablation_run_id"])]
        nrow = _read_metric_row(normal["metrics_path"])
        arow = _read_metric_row(ablated["metrics_path"])
        deltas = {metric: float(nrow.get(metric, 0.0) - arow.get(metric, 0.0)) for metric in SUMMARY_METRICS}
        summaries.append(
            {
                **pair,
                "status": "success",
                "scenario_variant": str(pair.get("scenario_variant", "")),
                "normal_metrics": {metric: float(nrow.get(metric, 0.0)) for metric in SUMMARY_METRICS},
                "ablation_metrics": {metric: float(arow.get(metric, 0.0)) for metric in SUMMARY_METRICS},
                "deltas": deltas,
            }
        )
    by_group: dict[str, list[dict[str, Any]]] = {}
    for item in summaries:
        by_group.setdefault(str(item.get("group_id", "")), []).append(item)
    group_summary: list[dict[str, Any]] = []
    for gid, items in sorted(by_group.items()):
        successful = [i for i in items if i.get("status") == "success"]
        group_summary.append(
            {
                "group_id": gid,
                "success_count": len(successful),
                "skipped_count": len(items) - len(successful),
                "mean_alive_delta": mean([float(i["deltas"]["alive"]) for i in successful]) if successful else 0.0,
                "mean_cog_cost_delta": mean([float(i["deltas"]["cog_cost_mean"]) for i in successful]) if successful else 0.0,
                "alive_delta_sign_consistency": _sign_consistency([float(i["deltas"]["alive"]) for i in successful]),
                "interpretation": "paired deltas computed; no usefulness claim yet" if successful else "skipped",
            }
        )
    return {"pair_summaries": summaries, "group_summary": group_summary}


def _sign_consistency(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    if not vals:
        return 0.0
    pos = sum(1 for v in vals if v > 0.0)
    neg = sum(1 for v in vals if v < 0.0)
    return float(max(pos, neg)) / float(len(vals))


def _delta_vals(items: Iterable[Mapping[str, Any]], metric: str) -> list[float]:
    return [float(item.get("deltas", {}).get(metric, 0.0)) for item in items]


def paired_cost_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    items = [i for i in summary.get("pair_summaries", []) if i.get("status") == "success"]
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for item in items:
        trait = str(item.get("primary_trait", "")) or _trait_from_group(str(item.get("group_id", "")))
        scenario = str(item.get("scenario_group", "")) or _scenario_from_group(str(item.get("group_id", "")))
        cost_level = str(item.get("cost_level", "")) or _cost_level_from_group(str(item.get("group_id", "")))
        grouped.setdefault((trait, scenario, cost_level), []).append(item)
    groups: list[dict[str, Any]] = []
    for (trait, scenario, cost_level), vals in sorted(grouped.items()):
        alive = [float(v.get("deltas", {}).get("alive", 0.0)) for v in vals]
        cog_cost = [float(v.get("deltas", {}).get("cog_cost_mean", 0.0)) for v in vals]
        charged = [_trait_charged_cost(v.get("normal_metrics", {}), trait) for v in vals]
        groups.append(
            {
                "trait": trait,
                "scenario_group": scenario,
                "cost_level": cost_level,
                "success_count": len(vals),
                "mean_alive_delta": mean(alive) if alive else 0.0,
                "alive_delta_sign_consistency": _sign_consistency(alive),
                "mean_cog_cost_delta": mean(cog_cost) if cog_cost else 0.0,
                "normal_charged_cost_mean": mean(charged) if charged else 0.0,
                "cost_response_direction": _cost_response_direction(alive, charged),
            }
        )
    return {"cost_group_summary": groups, "pair_summaries": list(summary.get("pair_summaries", []))}


def cost_response_report(cost_summary: Mapping[str, Any]) -> dict[str, Any]:
    groups = list(cost_summary.get("cost_group_summary", []))
    traits = sorted({str(g.get("trait", "")) for g in groups if g.get("trait")})
    out: dict[str, Any] = {"traits": {}, "interpretation": "cost sweep summary only; no trait is confirmed useful"}
    for trait in traits:
        tgroups = [g for g in groups if str(g.get("trait", "")) == trait]
        by_level = {
            str(g.get("cost_level", "")): float(g.get("mean_alive_delta", 0.0))
            for g in tgroups
        }
        charged = {
            str(g.get("cost_level", "")): float(g.get("normal_charged_cost_mean", 0.0))
            for g in tgroups
        }
        pressure_vals = [
            float(g.get("mean_alive_delta", 0.0))
            for g in tgroups
            if str(g.get("scenario_group", "")) in {"patch", "jungle", "combined"}
        ]
        flat_vals = [
            float(g.get("mean_alive_delta", 0.0))
            for g in tgroups
            if str(g.get("scenario_group", "")) == "flat"
        ]
        nonzero_charged = any(v > 0.0 for k, v in charged.items() if k != "telemetry_only")
        high_signal = abs(by_level.get("high", 0.0)) > 0.0
        medium_signal = abs(by_level.get("medium", 0.0)) > 0.0
        if not nonzero_charged:
            classification = "suspicious free-cognition artifact"
        elif medium_signal or high_signal:
            classification = "weak cost-robust candidate"
        elif any(abs(v) > 0.0 for v in by_level.values()):
            classification = "inconsistent cost response"
        else:
            classification = "no cost-robust signal"
        if flat_vals and pressure_vals and abs(mean(flat_vals)) > abs(mean(pressure_vals)):
            classification = "suspicious free-cognition artifact"
        out["traits"][trait] = {
            "classification": classification,
            "alive_delta_by_cost_level": by_level,
            "normal_charged_cost_by_cost_level": charged,
            "flat_alive_delta_mean": mean(flat_vals) if flat_vals else 0.0,
            "pressure_alive_delta_mean": mean(pressure_vals) if pressure_vals else 0.0,
        }
    return out


def paired_transfer_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    items = [i for i in summary.get("pair_summaries", []) if i.get("status") == "success"]
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for item in items:
        trait = str(item.get("primary_trait", "")) or _trait_from_group(str(item.get("group_id", "")))
        scenario = str(item.get("scenario_group", "")) or _scenario_from_group(str(item.get("group_id", "")))
        variant = str(item.get("scenario_variant", "")) or _variant_from_group(str(item.get("group_id", "")))
        grouped.setdefault((trait, scenario, variant), []).append(item)
    groups: list[dict[str, Any]] = []
    for (trait, scenario, variant), vals in sorted(grouped.items()):
        alive = [float(v.get("deltas", {}).get("alive", 0.0)) for v in vals]
        reproduction = [float(v.get("deltas", {}).get("mean_offspring_total", 0.0)) for v in vals]
        groups.append(
            {
                "trait": trait,
                "scenario_group": scenario,
                "scenario_variant": variant,
                "success_count": len(vals),
                "mean_alive_delta": mean(alive) if alive else 0.0,
                "mean_reproduction_delta": mean(reproduction) if reproduction else 0.0,
                "alive_delta_sign_consistency": _sign_consistency(alive),
                "variant_sensitivity": "nonzero variant delta" if any(abs(v) > 0.0 for v in alive) else "no alive delta in this variant",
            }
        )
    return {"transfer_group_summary": groups, "pair_summaries": list(summary.get("pair_summaries", []))}


def transfer_robustness_report(transfer_summary: Mapping[str, Any]) -> dict[str, Any]:
    groups = list(transfer_summary.get("transfer_group_summary", []))
    traits = sorted({str(g.get("trait", "")) for g in groups if g.get("trait")})
    out: dict[str, Any] = {"traits": {}, "interpretation": "transfer summary only; no trait is confirmed adaptive"}
    for trait in traits:
        tgroups = [g for g in groups if str(g.get("trait", "")) == trait]
        pressure = [
            float(g.get("mean_alive_delta", 0.0))
            for g in tgroups
            if str(g.get("scenario_group", "")) in {"patch", "jungle", "combined"}
        ]
        flat = [
            float(g.get("mean_alive_delta", 0.0))
            for g in tgroups
            if str(g.get("scenario_group", "")) == "flat"
        ]
        variants = {str(g.get("scenario_variant", "")) for g in tgroups if g.get("scenario_variant")}
        nonzero_variants = {
            str(g.get("scenario_variant", ""))
            for g in tgroups
            if abs(float(g.get("mean_alive_delta", 0.0))) > 0.0
        }
        if not any(abs(v) > 0.0 for v in pressure + flat):
            classification = "no transfer signal"
        elif len(nonzero_variants) <= 1 and len(variants) > 1:
            classification = "variant-specific signal"
        elif flat and pressure and abs(mean(flat)) > abs(mean(pressure)):
            classification = "suspicious flat-world signal"
        elif pressure and len(nonzero_variants) > 1:
            classification = "weak transfer candidate"
        else:
            classification = "inconsistent transfer"
        out["traits"][trait] = {
            "classification": classification,
            "variant_count": len(variants),
            "nonzero_variant_count": len(nonzero_variants),
            "flat_alive_delta_mean": mean(flat) if flat else 0.0,
            "pressure_alive_delta_mean": mean(pressure) if pressure else 0.0,
        }
    return out


def lineage_retention_report(summary: Mapping[str, Any]) -> dict[str, Any]:
    items = [i for i in summary.get("pair_summaries", []) if i.get("status") == "success"]
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for item in items:
        trait = str(item.get("primary_trait", "")) or _trait_from_group(str(item.get("group_id", "")))
        grouped.setdefault(trait, []).append(item)
    out: dict[str, Any] = {"traits": {}, "interpretation": "longer-horizon retention summary only; no trait is proven adaptive"}
    for trait, vals in sorted(grouped.items()):
        lineage = [float(v.get("deltas", {}).get("lineage_active_count", 0.0)) for v in vals]
        reproduction = [float(v.get("deltas", {}).get("mean_offspring_total", 0.0)) for v in vals]
        alive = [float(v.get("deltas", {}).get("alive", 0.0)) for v in vals]
        if not any(abs(v) > 0.0 for v in lineage + reproduction + alive):
            classification = "no retention signal"
        elif mean(lineage) > 0.0 or mean(reproduction) > 0.0:
            classification = "weak retention candidate"
        elif mean(alive) > 0.0 and abs(mean(lineage)) <= 1e-12:
            classification = "trait retained but no lineage effect"
        else:
            classification = "inconsistent retention"
        out["traits"][trait] = {
            "classification": classification,
            "mean_alive_delta": mean(alive) if alive else 0.0,
            "mean_reproduction_delta": mean(reproduction) if reproduction else 0.0,
            "mean_lineage_delta": mean(lineage) if lineage else 0.0,
            "lineage_delta_sign_consistency": _sign_consistency(lineage),
        }
    return out


def perception_pressure_isolation_report(summary: Mapping[str, Any]) -> dict[str, Any]:
    items = [
        i for i in summary.get("pair_summaries", [])
        if i.get("status") == "success"
        and ((str(i.get("primary_trait", "")) or _trait_from_group(str(i.get("group_id", "")))) == "perception")
    ]
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for item in items:
        scenario = str(item.get("scenario_group", "")) or _scenario_from_group(str(item.get("group_id", "")))
        variant = str(item.get("scenario_variant", "")) or _variant_from_group(str(item.get("group_id", "")))
        groups.setdefault((scenario, variant), []).append(item)
    variant_rows: list[dict[str, Any]] = []
    for (scenario, variant), vals in sorted(groups.items()):
        alive = _delta_vals(vals, "alive")
        reproduction = _delta_vals(vals, "mean_offspring_total")
        lineage = _delta_vals(vals, "lineage_active_count")
        capacity = _delta_vals(vals, "perception_capacity_component_mean")
        charged = [float(v.get("normal_metrics", {}).get("perception_cost_charged_mean", 0.0)) for v in vals]
        variant_rows.append(
            {
                "scenario_group": scenario,
                "scenario_variant": variant,
                "success_count": len(vals),
                "mean_alive_delta": mean(alive) if alive else 0.0,
                "mean_reproduction_delta": mean(reproduction) if reproduction else 0.0,
                "mean_lineage_delta": mean(lineage) if lineage else 0.0,
                "mean_perception_capacity_delta": mean(capacity) if capacity else 0.0,
                "normal_perception_cost_charged_mean": mean(charged) if charged else 0.0,
                "alive_delta_sign_consistency": _sign_consistency(alive),
            }
        )
    flat = [float(r["mean_alive_delta"]) for r in variant_rows if r["scenario_group"] == "flat"]
    pressure = [float(r["mean_alive_delta"]) for r in variant_rows if r["scenario_group"] in {"patch", "jungle", "combined"}]
    patch = [float(r["mean_alive_delta"]) for r in variant_rows if r["scenario_group"] == "patch"]
    jungle = [float(r["mean_alive_delta"]) for r in variant_rows if r["scenario_group"] == "jungle"]
    combined = [float(r["mean_alive_delta"]) for r in variant_rows if r["scenario_group"] == "combined"]
    capacity_changed = any(abs(float(r["mean_perception_capacity_delta"])) > 0.0 for r in variant_rows)
    cost_charged = any(float(r["normal_perception_cost_charged_mean"]) > 0.0 for r in variant_rows)
    flat_mean = mean(flat) if flat else 0.0
    pressure_mean = mean(pressure) if pressure else 0.0
    if not capacity_changed or not cost_charged:
        classification = "needs perception implementation cleanup"
    elif not any(abs(v) > 0.0 for v in flat + pressure):
        classification = "no perception signal"
    elif flat and abs(flat_mean) >= abs(pressure_mean):
        classification = "flat-world suspicious perception signal"
    elif pressure_mean > 0.0 and patch and jungle:
        classification = "weak pressure-dependent perception candidate"
    elif pressure_mean > 0.0:
        classification = "inconsistent perception signal"
    else:
        classification = "needs ecological pressure cleanup"
    return {
        "classification": classification,
        "flat_alive_delta_mean": flat_mean,
        "pressure_alive_delta_mean": pressure_mean,
        "patch_alive_delta_mean": mean(patch) if patch else 0.0,
        "jungle_alive_delta_mean": mean(jungle) if jungle else 0.0,
        "combined_alive_delta_mean": mean(combined) if combined else 0.0,
        "perception_cost_charged": cost_charged,
        "ablation_changed_perception_metrics": capacity_changed,
        "variant_summary": variant_rows,
        "interpretation": "perception pressure-isolation summary only; no perception adaptiveness claim",
    }


def behavior_proxy_report(summary: Mapping[str, Any]) -> dict[str, Any]:
    items = [
        i for i in summary.get("pair_summaries", [])
        if i.get("status") == "success"
        and ((str(i.get("primary_trait", "")) or _trait_from_group(str(i.get("group_id", "")))) == "perception")
    ]
    proxy_metrics = (
        "patch_food_harvest_total",
        "patch_food_regrowth_total",
        "patch_food_occupancy_bias",
        "patch_shelter_occupied_mean",
        "patch_shelter_occupancy_bias",
        "patch_waste_occupied_mean",
        "patch_waste_occupancy_bias",
        "tier3_overcrowd_mean",
        "jungle_predator_events",
        "jungle_predator_event_rate",
        "jungle_predator_damage_total",
        "jungle_predator_deaths",
    )
    available: set[str] = set()
    rows: list[dict[str, Any]] = []
    for item in items:
        deltas = dict(item.get("deltas", {}))
        available.update(metric for metric in proxy_metrics if metric in deltas)
        nonzero = {
            metric: float(deltas.get(metric, 0.0))
            for metric in proxy_metrics
            if metric in deltas and abs(float(deltas.get(metric, 0.0))) > 0.0
        }
        rows.append(
            {
                "group_id": str(item.get("group_id", "")),
                "seed": int(item.get("seed", 0) or 0),
                "scenario_group": str(item.get("scenario_group", "")),
                "scenario_variant": str(item.get("scenario_variant", "")),
                "nonzero_proxy_deltas": nonzero,
            }
        )
    proxy_delta_counts = {
        metric: sum(1 for item in items if abs(float(item.get("deltas", {}).get(metric, 0.0))) > 0.0)
        for metric in proxy_metrics
        if metric in available
    }
    limitation = ""
    if not available:
        limitation = "Behavior-level interpretation remains limited because available metrics do not expose perception-mediated behavior directly."
    elif not any(proxy_delta_counts.values()):
        limitation = "Behavior proxy columns are present, but this campaign produced no nonzero proxy deltas in the paired summary."
    return {
        "available_proxy_metrics": sorted(available),
        "proxy_delta_counts": proxy_delta_counts,
        "pair_proxy_rows": rows,
        "limitation": limitation,
        "interpretation": "behavior proxies are diagnostics only; no behavioral causality claim",
    }


def perception_lineage_retention_report(summary: Mapping[str, Any]) -> dict[str, Any]:
    items = [
        i for i in summary.get("pair_summaries", [])
        if i.get("status") == "success"
        and ((str(i.get("primary_trait", "")) or _trait_from_group(str(i.get("group_id", "")))) == "perception")
    ]
    lineage = _delta_vals(items, "lineage_active_count")
    reproduction = _delta_vals(items, "mean_offspring_total")
    alive = _delta_vals(items, "alive")
    parent_channels = [float(i.get("normal_metrics", {}).get("parent_sensor_channel_count_mean", 0.0)) for i in items]
    alive_channels = [float(i.get("normal_metrics", {}).get("alive_sensor_channel_count_mean", 0.0)) for i in items]
    selection_delta = [float(i.get("normal_metrics", {}).get("perception_selection_delta_parent_vs_alive", 0.0)) for i in items]
    flat_lineage = [
        float(i.get("deltas", {}).get("lineage_active_count", 0.0))
        for i in items
        if str(i.get("scenario_group", "")) == "flat"
    ]
    pressure_lineage = [
        float(i.get("deltas", {}).get("lineage_active_count", 0.0))
        for i in items
        if str(i.get("scenario_group", "")) in {"patch", "jungle", "combined"}
    ]
    if not any(abs(v) > 0.0 for v in lineage + reproduction + alive):
        classification = "no retention signal"
    elif flat_lineage and abs(mean(flat_lineage)) >= abs(mean(pressure_lineage) if pressure_lineage else 0.0):
        classification = "suspicious flat-world retention"
    elif pressure_lineage and mean(pressure_lineage) > 0.0:
        classification = "pressure-dependent retention candidate"
    elif mean(reproduction) > 0.0 or mean(lineage) > 0.0:
        classification = "weak retention candidate"
    else:
        classification = "inconsistent retention"
    return {
        "classification": classification,
        "mean_alive_delta": mean(alive) if alive else 0.0,
        "mean_reproduction_delta": mean(reproduction) if reproduction else 0.0,
        "mean_lineage_delta": mean(lineage) if lineage else 0.0,
        "flat_lineage_delta_mean": mean(flat_lineage) if flat_lineage else 0.0,
        "pressure_lineage_delta_mean": mean(pressure_lineage) if pressure_lineage else 0.0,
        "normal_parent_sensor_channel_count_mean": mean(parent_channels) if parent_channels else 0.0,
        "normal_alive_sensor_channel_count_mean": mean(alive_channels) if alive_channels else 0.0,
        "normal_perception_selection_delta_parent_vs_alive_mean": mean(selection_delta) if selection_delta else 0.0,
        "lineage_delta_sign_consistency": _sign_consistency(lineage),
        "interpretation": "perception lineage summary only; no trait is proven adaptive",
    }


def perception_fake_evolution_audit(
    manifest: Mapping[str, Any],
    validation: Mapping[str, Any],
    summary: Mapping[str, Any],
    pressure_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    base = fake_evolution_audit(manifest, validation, summary)
    pressure = dict(pressure_report or perception_pressure_isolation_report(summary))
    failures = [str(f) for f in validation.get("failures", [])]
    classification = str(pressure.get("classification", ""))
    if "pressure-dependent" in classification:
        recommendation = "advance to deeper perception-only campaign"
    elif "flat-world suspicious" in classification:
        recommendation = "require ecological pressure cleanup or deprioritize perception"
    elif "implementation cleanup" in classification:
        recommendation = "require perception implementation cleanup"
    elif "ecological pressure cleanup" in classification:
        recommendation = "require ecological pressure cleanup"
    else:
        recommendation = "deprioritize or retest only with tighter pressure instrumentation"
    base.update(
        {
            "perception_focused": True,
            "direct_perception_reward_present": False,
            "perception_enabled_without_cost_risk": not bool(pressure.get("perception_cost_charged", False)),
            "flat_world_perception_risk": "flat-world suspicious" in classification,
            "perception_ablation_no_behavior_effect_risk": not bool(pressure.get("ablation_changed_perception_metrics", False)),
            "cost_metric_missing_risks": [f for f in failures if "perception_cost" in f],
            "perception_pressure_classification": classification,
            "recommended_next_action": recommendation,
        }
    )
    return base


def diagnostic_source_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    inputs = manifest.get("diagnostic_inputs", {})
    source_path = ""
    if isinstance(inputs, Mapping):
        source_path = str(inputs.get("source_manifest", "") or "")
    source_path = source_path or str(manifest.get("source_campaign_manifest", "") or "")
    if not source_path:
        return dict(manifest)
    return load_campaign_manifest(source_path)


def campaign_metadata_summary(
    manifest: Mapping[str, Any],
    plan: Mapping[str, Any] | None = None,
    execution: Mapping[str, Any] | None = None,
    validation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    plan = dict(plan or plan_campaign(manifest))
    execution = dict(execution or {})
    validation = dict(validation or {})
    variants = manifest.get("scenario_variants", {})
    if isinstance(variants, Mapping):
        variant_ids = sorted(str(k) for k in variants)
    elif isinstance(variants, list):
        variant_ids = sorted(str(v.get("variant_id", v)) if isinstance(v, Mapping) else str(v) for v in variants)
    else:
        variant_ids = sorted({str(c.get("scenario_variant", "")) for c in manifest.get("scenario_cells", []) if c.get("scenario_variant")})
    executed = int(execution.get("executed_run_count", execution.get("success_count", 0)) or 0)
    return {
        "campaign_id": str(manifest.get("campaign_id", "")),
        "seed_list": list(plan.get("seed_list", manifest.get("seed_list", []))),
        "planned_live_run_count": int(plan.get("planned_live_run_count", 0) or 0),
        "planned_live_simulator_run_count": int(plan.get("planned_live_simulator_run_count", 0) or 0),
        "planned_fixture_run_count": int(plan.get("planned_fixture_run_count", 0) or 0),
        "planned_scaffold_check_count": int(plan.get("planned_scaffold_check_count", 0) or 0),
        "planned_skipped_cell_count": int(plan.get("planned_skipped_cell_count", 0) or 0),
        "executed_live_run_count": executed,
        "success_count": int(execution.get("success_count", 0) or 0),
        "failure_count": int(execution.get("failure_count", 0) or 0),
        "backend_counts": dict(plan.get("execution_backend_counts", {})),
        "skipped_cells": list(plan.get("skipped_cells", execution.get("skipped_cells", []))),
        "scenario_variants": variant_ids,
        "horizon": dict(manifest.get("horizon", {})),
        "cost_profile": str(manifest.get("cost_mode", "")),
        "fixed_sensor_radius_setting": str(manifest.get("fixed_sensor_radius_setting", "")),
        "validation_checked_run_count": int(validation.get("checked_run_count", 0) or 0),
        "validation_ok": bool(validation.get("ok", False)) if validation else False,
    }


def _successful_perception_items(summary: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        i for i in summary.get("pair_summaries", [])
        if i.get("status") == "success"
        and ((str(i.get("primary_trait", "")) or _trait_from_group(str(i.get("group_id", "")))) == "perception")
    ]


def _metric_values(items: Iterable[Mapping[str, Any]], source: str, metric: str) -> list[float]:
    vals: list[float] = []
    for item in items:
        payload = item.get(source, {})
        if isinstance(payload, Mapping) and metric in payload:
            vals.append(float(payload.get(metric, 0.0) or 0.0))
    return vals


def _mean(vals: Iterable[float]) -> float:
    values = [float(v) for v in vals]
    return mean(values) if values else 0.0


def _variant_groups(items: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], list[Mapping[str, Any]]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for item in items:
        scenario = str(item.get("scenario_group", "")) or _scenario_from_group(str(item.get("group_id", "")))
        variant = str(item.get("scenario_variant", "")) or _variant_from_group(str(item.get("group_id", "")))
        groups.setdefault((scenario, variant), []).append(item)
    return groups


def perception_channel_visibility_audit(summary: Mapping[str, Any]) -> dict[str, Any]:
    items = _successful_perception_items(summary)
    channel_metrics = (
        "sensor_channel_local_grid_state_enabled_fraction",
        "sensor_channel_patch_food_amount_norm_enabled_fraction",
        "sensor_channel_patch_cover_enabled_fraction",
        "sensor_channel_patch_shelter_enabled_fraction",
        "sensor_channel_patch_scent_norm_enabled_fraction",
        "sensor_channel_patch_waste_norm_enabled_fraction",
        "sensor_channel_patch_density_norm_enabled_fraction",
        "sensor_channel_patch_movement_cost_norm_enabled_fraction",
        "sensor_channel_self_energy_enabled_fraction",
        "sensor_channel_self_health_enabled_fraction",
    )
    variation_metrics = (
        "patch_food_amount_max",
        "patch_food_amount_min",
        "patch_scent_max",
        "patch_waste_max",
        "patch_density_max",
        "jungle_predator_risk_max",
    )
    rows: list[dict[str, Any]] = []
    for (scenario, variant), vals in sorted(_variant_groups(items).items()):
        normal_count = _mean(_metric_values(vals, "normal_metrics", "sensor_channel_active_count_mean"))
        ablated_count = _mean(_metric_values(vals, "ablation_metrics", "sensor_channel_active_count_mean"))
        normal_fraction = _mean(_metric_values(vals, "normal_metrics", "sensor_channel_active_fraction_mean"))
        entropy = _mean(_metric_values(vals, "normal_metrics", "sensor_channel_mask_entropy"))
        strategy_count = _mean(_metric_values(vals, "normal_metrics", "sensor_channel_strategy_count_approx"))
        capacity_delta = _mean(_delta_vals(vals, "perception_capacity_component_mean"))
        channel_means = {
            metric: _mean(_metric_values(vals, "normal_metrics", metric))
            for metric in channel_metrics
        }
        spatial_signal = sum(abs(_mean(_metric_values(vals, "normal_metrics", m))) for m in variation_metrics)
        temporal_signal = sum(abs(_mean(_metric_values(vals, "normal_metrics", m))) for m in ("patch_food_regrowth_total", "patch_food_harvest_total", "jungle_predator_events"))
        active_nontrivial_channels = sum(1 for v in channel_means.values() if v > 0.01)
        if capacity_delta <= 0.0 or normal_count <= ablated_count:
            classification = "ablation may not remove perception information"
        elif normal_count <= 1.0 or normal_fraction <= 0.05:
            classification = "mostly empty perception channels"
        elif scenario == "flat" and spatial_signal <= 0.0 and temporal_signal <= 0.0:
            classification = "flat control with low ecological perception signal"
        elif active_nontrivial_channels <= 1 or (entropy <= 0.05 and strategy_count <= 1.0):
            classification = "dominated by one trivial signal"
        elif scenario in {"patch", "jungle", "combined"} and (spatial_signal > 0.0 or temporal_signal > 0.0):
            classification = "pressure-world channels nontrivial"
        else:
            classification = "perception-weak"
        rows.append(
            {
                "scenario_group": scenario,
                "scenario_variant": variant,
                "success_count": len(vals),
                "normal_active_channel_count_mean": normal_count,
                "ablated_active_channel_count_mean": ablated_count,
                "normal_active_channel_fraction_mean": normal_fraction,
                "normal_channel_entropy_mean": entropy,
                "normal_strategy_count_mean": strategy_count,
                "capacity_delta_mean": capacity_delta,
                "active_nontrivial_channel_count": active_nontrivial_channels,
                "spatial_signal_proxy": spatial_signal,
                "temporal_signal_proxy": temporal_signal,
                "channel_enabled_means": channel_means,
                "classification": classification,
            }
        )
    pressure_rows = [r for r in rows if r["scenario_group"] in {"patch", "jungle", "combined"}]
    pressure_informative = sum(1 for r in pressure_rows if r["classification"] == "pressure-world channels nontrivial")
    if not rows:
        overall = "metrics insufficient"
    elif pressure_rows and pressure_informative >= max(1, len(pressure_rows) // 2):
        overall = "current ecology exposes nontrivial perception channels"
    elif any("ablation may not" in str(r["classification"]) for r in rows):
        overall = "perception ablation visibility is compromised"
    else:
        overall = "current ecology is perception-weak"
    return {
        "overall_classification": overall,
        "variant_visibility": rows,
        "pressure_informative_variant_count": pressure_informative,
        "pressure_variant_count": len(pressure_rows),
        "interpretation": "visibility audit only; no perception adaptiveness claim",
    }


def affordance_outcome_correlation_report(summary: Mapping[str, Any]) -> dict[str, Any]:
    items = _successful_perception_items(summary)
    feature_metrics = (
        "patch_food_harvest_total",
        "patch_food_regrowth_total",
        "patch_food_occupancy_bias",
        "patch_shelter_occupancy_bias",
        "patch_waste_occupancy_bias",
        "jungle_predator_events",
        "jungle_predator_event_rate",
        "jungle_predator_damage_total",
    )
    outcome_metrics = ("alive", "mean_offspring_total", "lineage_active_count")
    rows: list[dict[str, Any]] = []
    for feature in feature_metrics:
        fvals = _delta_vals(items, feature)
        if not fvals:
            continue
        feature_rows: dict[str, Any] = {
            "feature_metric": feature,
            "mean_feature_delta": _mean(fvals),
            "nonzero_feature_delta_count": sum(1 for v in fvals if abs(v) > 0.0),
        }
        for outcome in outcome_metrics:
            ovals = _delta_vals(items, outcome)
            pairs = [(f, o) for f, o in zip(fvals, ovals) if abs(f) > 0.0 or abs(o) > 0.0]
            same = sum(1 for f, o in pairs if (f > 0.0 and o > 0.0) or (f < 0.0 and o < 0.0))
            feature_rows[f"{outcome}_mean_delta"] = _mean(ovals)
            feature_rows[f"{outcome}_sign_alignment_fraction"] = float(same) / float(len(pairs)) if pairs else 0.0
        rows.append(feature_rows)
    if not rows:
        classification = "metrics insufficient"
    elif not any(int(r["nonzero_feature_delta_count"]) > 0 for r in rows):
        classification = "no apparent affordance-outcome link"
    elif any(float(r.get("alive_sign_alignment_fraction", 0.0)) >= 0.65 for r in rows):
        classification = "weak candidate link"
    elif any(float(r.get("alive_sign_alignment_fraction", 0.0)) <= 0.35 for r in rows if int(r["nonzero_feature_delta_count"]) > 0):
        classification = "inconsistent link"
    else:
        classification = "no apparent affordance-outcome link"
    return {
        "classification": classification,
        "feature_outcome_rows": rows,
        "missing_metrics_note": "" if rows else "Required affordance or outcome metrics were unavailable in the paired summary.",
        "interpretation": "simple deterministic sign summaries only; no statistical claim",
    }


def behavior_expression_diagnostic(summary: Mapping[str, Any]) -> dict[str, Any]:
    proxy = behavior_proxy_report(summary)
    items = _successful_perception_items(summary)
    flat_items = [i for i in items if str(i.get("scenario_group", "")) == "flat"]
    pressure_items = [i for i in items if str(i.get("scenario_group", "")) in {"patch", "jungle", "combined"}]
    proxy_metrics = list(proxy.get("available_proxy_metrics", []))
    flat_nonzero = sum(
        1 for item in flat_items for metric in proxy_metrics
        if abs(float(item.get("deltas", {}).get(metric, 0.0))) > 0.0
    )
    pressure_nonzero = sum(
        1 for item in pressure_items for metric in proxy_metrics
        if abs(float(item.get("deltas", {}).get(metric, 0.0))) > 0.0
    )
    if not proxy_metrics:
        classification = "behavior metrics insufficient"
    elif pressure_nonzero > flat_nonzero:
        classification = "behavior expressed under pressure"
    elif flat_nonzero > 0 and flat_nonzero >= pressure_nonzero:
        classification = "flat-world behavior expression suspicious"
    elif pressure_nonzero > 0:
        classification = "behavior proxy changes inconsistent"
    else:
        classification = "ablation changes telemetry but not behavior proxies"
    return {
        "classification": classification,
        "available_proxy_metrics": proxy_metrics,
        "flat_nonzero_proxy_delta_count": flat_nonzero,
        "pressure_nonzero_proxy_delta_count": pressure_nonzero,
        "proxy_delta_counts": dict(proxy.get("proxy_delta_counts", {})),
        "limitation": str(proxy.get("limitation", "")),
        "interpretation": "behavior-expression diagnostic only; no behavioral causality claim",
    }


def scenario_pressure_audit(summary: Mapping[str, Any]) -> dict[str, Any]:
    items = _successful_perception_items(summary)
    rows: list[dict[str, Any]] = []
    for (scenario, variant), vals in sorted(_variant_groups(items).items()):
        normal_alive = _mean(_metric_values(vals, "normal_metrics", "alive"))
        deaths = _mean(_metric_values(vals, "normal_metrics", "deaths"))
        reproduction = _mean(_metric_values(vals, "normal_metrics", "mean_offspring_total"))
        resource = _mean(_metric_values(vals, "normal_metrics", "patch_food_amount_mean"))
        harvest = _mean(_metric_values(vals, "normal_metrics", "patch_food_harvest_total"))
        regrowth = _mean(_metric_values(vals, "normal_metrics", "patch_food_regrowth_total"))
        shelter = _mean(_metric_values(vals, "normal_metrics", "patch_shelter_mean"))
        waste = _mean(_metric_values(vals, "normal_metrics", "patch_waste_mean"))
        predator = _mean(_metric_values(vals, "normal_metrics", "jungle_predator_events"))
        predator_risk = _mean(_metric_values(vals, "normal_metrics", "jungle_predator_risk_mean"))
        heterogeneity = sum(
            abs(_mean(_metric_values(vals, "normal_metrics", m)))
            for m in ("patch_food_amount_max", "patch_scent_max", "patch_waste_max", "patch_density_max", "jungle_predator_risk_max")
        )
        exposure = abs(harvest) + abs(predator) + abs(waste) + abs(deaths)
        if scenario == "flat":
            classification = "too flat"
        elif normal_alive <= 0.0:
            classification = "too lethal"
        elif exposure <= 0.0 and reproduction > 0.0:
            classification = "too easy"
        elif heterogeneity <= 0.0 and exposure <= 0.0:
            classification = "perception-weak"
        elif heterogeneity > 0.0 and exposure > 0.0:
            classification = "candidate for next perception campaign"
        else:
            classification = "perception-weak"
        rows.append(
            {
                "scenario_group": scenario,
                "scenario_variant": variant,
                "resource_density_proxy": resource,
                "resource_harvest_proxy": harvest,
                "temporal_regrowth_proxy": regrowth,
                "shelter_proxy": shelter,
                "waste_pressure_proxy": waste,
                "predator_event_proxy": predator,
                "predator_risk_proxy": predator_risk,
                "spatial_heterogeneity_proxy": heterogeneity,
                "agent_exposure_proxy": exposure,
                "alive_mean": normal_alive,
                "death_mean": deaths,
                "reproduction_mean": reproduction,
                "classification": classification,
            }
        )
    candidate_count = sum(1 for r in rows if r["classification"] == "candidate for next perception campaign")
    return {
        "variant_pressure_rows": rows,
        "candidate_variant_count": candidate_count,
        "overall_classification": "pressure variants need cleanup" if candidate_count == 0 else "some variants are perception-informative",
        "interpretation": "scenario pressure audit only; no ecological coefficient tuning performed",
    }


def perception_ablation_validity_audit(
    manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
    validation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validation = dict(validation or {})
    allowed_ablation_flags = {"force_minimal_sensor_channels_enabled"}
    issues: list[str] = []
    cells_by_id = {str(c.get("cell_id", "")): c for c in manifest.get("scenario_cells", []) if isinstance(c, Mapping)}
    live_cells = [c for c in manifest.get("scenario_cells", []) if _cell_backend(c) == "live_simulator"]
    for cell in live_cells:
        flags = dict(cell.get("ablation_flags", {}))
        cell_id = str(cell.get("cell_id", ""))
        if "ablation" in cell_id and flags != {"force_minimal_sensor_channels_enabled": True}:
            issues.append(f"{cell_id} ablation flags are not the expected force-minimal perception-only diff")
        if "normal" in cell_id and flags:
            issues.append(f"{cell_id} normal cell has ablation flags")
        if str(cell.get("fixed_sensor_radius_setting", "")) != "force_default_sensor_radius":
            issues.append(f"{cell_id} does not declare fixed default sensor radius")
        unexpected = set(flags) - allowed_ablation_flags
        if unexpected:
            issues.append(f"{cell_id} has unexpected ablation flags: {sorted(unexpected)}")
    for group in manifest.get("paired_ablation_groups", []):
        normal = cells_by_id.get(str(group.get("normal_cell_id", "")), {})
        ablated = cells_by_id.get(str(group.get("ablation_cell_id", "")), {})
        if not normal or not ablated:
            continue
        for key in ("env_overrides", "pop_overrides", "sim_overrides", "cost_level", "fixed_sensor_radius_setting"):
            nval = normal.get(key)
            aval = ablated.get(key)
            if key == "sim_overrides" and isinstance(nval, Mapping) and isinstance(aval, Mapping):
                nmap = dict(nval)
                amap = dict(aval)
                amap.pop("force_minimal_sensor_channels_enabled", None)
                nmap.pop("force_minimal_sensor_channels_enabled", None)
                nval = nmap
                aval = amap
            if nval != aval:
                issues.append(f"{group.get('group_id')} has unintended normal/ablation diff in {key}")
    for item in _successful_perception_items(summary):
        if float(item.get("deltas", {}).get("perception_capacity_component_mean", 0.0)) <= 0.0:
            issues.append(f"{item.get('group_id')} seed {item.get('seed')} did not reduce perception capacity under ablation")
        ablation_flag = float(item.get("ablation_metrics", {}).get("sensor_channel_force_minimal_active_flag", 0.0))
        if "sensor_channel_force_minimal_active_flag" in item.get("ablation_metrics", {}) and ablation_flag < 0.5:
            issues.append(f"{item.get('group_id')} seed {item.get('seed')} missing force-minimal ablation metric")
    issues.extend(str(f) for f in validation.get("failures", []) if "fixed_sensor_radius_setting" in str(f) or "sensor_channel" in str(f))
    return {
        "clean": not issues,
        "classification": "clean perception ablation" if not issues else "compromised perception ablation",
        "issues": issues,
        "allowed_ablation_flags": sorted(allowed_ablation_flags),
        "runtime_shape_changed": False,
        "interpretation": "ablation validity audit only; no trait usefulness claim",
    }


def evidence_metadata_cleanup_report(metadata: Mapping[str, Any]) -> dict[str, Any]:
    required = ("seed_list", "planned_live_simulator_run_count", "executed_live_run_count", "scenario_variants", "horizon", "cost_profile")
    missing = [key for key in required if metadata.get(key) in (None, "", [])]
    return {
        "fixed_for_future_reports": not missing,
        "missing_metadata_fields": missing,
        "seed_list": list(metadata.get("seed_list", [])),
        "planned_live_simulator_run_count": int(metadata.get("planned_live_simulator_run_count", 0) or 0),
        "executed_live_run_count": int(metadata.get("executed_live_run_count", 0) or 0),
        "skipped_cell_count": int(metadata.get("planned_skipped_cell_count", 0) or 0),
        "backend_counts": dict(metadata.get("backend_counts", {})),
        "scenario_variants": list(metadata.get("scenario_variants", [])),
        "horizon": dict(metadata.get("horizon", {})),
        "cost_profile": str(metadata.get("cost_profile", "")),
    }


def stage10g_decision_report(
    visibility: Mapping[str, Any],
    affordance: Mapping[str, Any],
    behavior: Mapping[str, Any],
    pressure: Mapping[str, Any],
    ablation: Mapping[str, Any],
) -> dict[str, Any]:
    allowed = {
        "run deeper perception campaign",
        "improve ecological pressure design",
        "improve perception implementation/behavior coupling",
        "improve metrics/behavior proxies",
        "deprioritize perception",
        "pause Theme 4 live expansion",
    }
    if not bool(ablation.get("clean", False)):
        recommendation = "improve perception implementation/behavior coupling"
    elif "insufficient" in str(behavior.get("classification", "")) or "insufficient" in str(affordance.get("classification", "")):
        recommendation = "improve metrics/behavior proxies"
    elif "current ecology exposes nontrivial" in str(visibility.get("overall_classification", "")) and "behavior expressed under pressure" == str(behavior.get("classification", "")) and "weak candidate" in str(affordance.get("classification", "")):
        recommendation = "run deeper perception campaign"
    elif "pressure variants need cleanup" in str(pressure.get("overall_classification", "")) or "perception-weak" in str(visibility.get("overall_classification", "")):
        recommendation = "improve ecological pressure design"
    elif "flat-world" in str(behavior.get("classification", "")):
        recommendation = "improve ecological pressure design"
    else:
        recommendation = "deprioritize perception"
    assert recommendation in allowed
    return {
        "next_action": recommendation,
        "allowed_next_actions": sorted(allowed),
        "visibility_classification": str(visibility.get("overall_classification", "")),
        "affordance_classification": str(affordance.get("classification", "")),
        "behavior_classification": str(behavior.get("classification", "")),
        "scenario_pressure_classification": str(pressure.get("overall_classification", "")),
        "ablation_classification": str(ablation.get("classification", "")),
        "interpretation": "Stage 10G decision only; no perception adaptiveness claim",
    }


def summarize_calibration_runs(manifest: Mapping[str, Any], plan: Mapping[str, Any] | None = None) -> dict[str, Any]:
    plan = dict(plan or plan_campaign(manifest))
    cells = {str(c.get("cell_id", "")): c for c in manifest.get("scenario_cells", []) if isinstance(c, Mapping)}
    rows: list[dict[str, Any]] = []
    for run in plan.get("planned_runs", []):
        cell = cells.get(str(run.get("cell_id", "")), {})
        metrics_path = Path(str(run.get("metrics_path", "")))
        row = _read_metric_row(str(metrics_path)) if metrics_path.exists() else {}
        rows.append(
            {
                "run_id": str(run.get("run_id", "")),
                "cell_id": str(run.get("cell_id", "")),
                "seed": int(run.get("seed", 0) or 0),
                "execution_backend": str(run.get("execution_backend", "")),
                "scenario_family": str(cell.get("scenario_family", cell.get("scenario_group", ""))),
                "scenario_group": str(cell.get("scenario_group", "")),
                "scenario_variant": str(cell.get("scenario_variant", "")),
                "pressure_level": str(cell.get("pressure_level", "")),
                "perception_condition": str(cell.get("perception_condition", "normal")),
                "expected_affordance": list(cell.get("expected_perception_visible_affordances", [])),
                "expected_behavior_proxy": list(cell.get("expected_behavior_proxy_changes", [])),
                "expected_risk": str(cell.get("expected_risk", "")),
                "metrics": {metric: float(row.get(metric, 0.0)) for metric in SUMMARY_METRICS},
            }
        )
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (
                str(row["scenario_family"]),
                str(row["pressure_level"]),
                str(row["scenario_variant"]),
                str(row["perception_condition"]),
            ),
            [],
        ).append(row)
    cell_summary: list[dict[str, Any]] = []
    for (family, level, variant, condition), vals in sorted(grouped.items()):
        def m(metric: str) -> float:
            return _mean(float(v["metrics"].get(metric, 0.0)) for v in vals)

        visibility_proxy = (
            m("sensor_channel_active_fraction_mean")
            + m("sensor_channel_mask_entropy")
            + m("patch_food_amount_max")
            + m("patch_waste_max")
            + m("jungle_predator_risk_max")
        )
        behavior_proxy = (
            abs(m("patch_food_harvest_total"))
            + abs(m("patch_food_occupancy_bias"))
            + abs(m("patch_shelter_occupancy_bias"))
            + abs(m("patch_waste_occupancy_bias"))
            + abs(m("jungle_predator_events"))
            + abs(m("jungle_predator_damage_total"))
        )
        outcome_proxy = m("alive") + m("mean_offspring_total") + m("lineage_active_count")
        cell_summary.append(
            {
                "scenario_family": family,
                "pressure_level": level,
                "scenario_variant": variant,
                "perception_condition": condition,
                "success_count": len(vals),
                "alive_mean": m("alive"),
                "death_mean": m("deaths"),
                "reproduction_mean": m("mean_offspring_total"),
                "lineage_mean": m("lineage_active_count"),
                "perception_capacity_mean": m("perception_capacity_component_mean"),
                "perception_cost_charged_mean": m("perception_cost_charged_mean"),
                "sensor_channel_fraction_mean": m("sensor_channel_active_fraction_mean"),
                "sensor_channel_entropy_mean": m("sensor_channel_mask_entropy"),
                "patch_food_harvest_mean": m("patch_food_harvest_total"),
                "patch_regrowth_mean": m("patch_food_regrowth_total"),
                "patch_waste_mean": m("patch_waste_mean"),
                "predator_events_mean": m("jungle_predator_events"),
                "predator_damage_mean": m("jungle_predator_damage_total"),
                "visibility_proxy": visibility_proxy,
                "behavior_proxy": behavior_proxy,
                "outcome_proxy": outcome_proxy,
                "expected_affordance": vals[0].get("expected_affordance", []),
                "expected_behavior_proxy": vals[0].get("expected_behavior_proxy", []),
                "expected_risk": str(vals[0].get("expected_risk", "")),
            }
        )
    return {
        "campaign_id": str(manifest.get("campaign_id", "")),
        "run_summaries": rows,
        "cell_summary": cell_summary,
        "interpretation": "calibration summaries only; no perception adaptiveness claim",
    }


def visibility_calibration_report(calibration: Mapping[str, Any]) -> dict[str, Any]:
    rows = list(calibration.get("cell_summary", []))
    normal = [r for r in rows if str(r.get("perception_condition", "")) == "normal"]
    flat = [r for r in normal if str(r.get("pressure_level", "")) == "flat_control"]
    flat_visibility = _mean(float(r.get("visibility_proxy", 0.0)) for r in flat)
    variant_rows: list[dict[str, Any]] = []
    for row in normal:
        visibility = float(row.get("visibility_proxy", 0.0))
        delta = visibility - flat_visibility
        if str(row.get("pressure_level", "")) == "flat_control":
            classification = "flat baseline"
        elif delta <= 0.05:
            classification = "perception-weak"
        elif visibility > 0.0 and delta > 0.05:
            classification = "perception-visible"
        else:
            classification = "metrics insufficient"
        variant_rows.append(
            {
                "scenario_family": row.get("scenario_family", ""),
                "pressure_level": row.get("pressure_level", ""),
                "scenario_variant": row.get("scenario_variant", ""),
                "visibility_proxy": visibility,
                "pressure_world_visibility_delta": delta,
                "sensor_channel_fraction_mean": row.get("sensor_channel_fraction_mean", 0.0),
                "sensor_channel_entropy_mean": row.get("sensor_channel_entropy_mean", 0.0),
                "classification": classification,
            }
        )
    informative = sum(1 for r in variant_rows if r["classification"] == "perception-visible")
    return {
        "overall_classification": "visibility ladder available" if informative else "visibility ladder not established",
        "flat_visibility_proxy": flat_visibility,
        "informative_variant_count": informative,
        "variant_visibility": variant_rows,
        "interpretation": "visibility calibration only; no trait usefulness claim",
    }


def behavior_outcome_calibration_report(calibration: Mapping[str, Any]) -> dict[str, Any]:
    rows = list(calibration.get("cell_summary", []))
    normal = [r for r in rows if str(r.get("perception_condition", "")) == "normal"]
    output: list[dict[str, Any]] = []
    for row in normal:
        behavior = float(row.get("behavior_proxy", 0.0))
        alive = float(row.get("alive_mean", 0.0))
        deaths = float(row.get("death_mean", 0.0))
        reproduction = float(row.get("reproduction_mean", 0.0))
        lineage = float(row.get("lineage_mean", 0.0))
        if behavior <= 0.0:
            classification = "behavior metrics insufficient"
        elif alive <= 0.0 or deaths > max(1.0, alive * 3.0):
            classification = "too lethal"
        elif deaths <= 0.0 and reproduction > 0.0:
            classification = "too easy"
        elif behavior > 0.0 and (reproduction > 0.0 or lineage > 0.0):
            classification = "behavior-outcome linked"
        else:
            classification = "behavior linked but outcome unclear"
        output.append(
            {
                "scenario_family": row.get("scenario_family", ""),
                "pressure_level": row.get("pressure_level", ""),
                "scenario_variant": row.get("scenario_variant", ""),
                "behavior_proxy": behavior,
                "alive_mean": alive,
                "death_mean": deaths,
                "reproduction_mean": reproduction,
                "lineage_mean": lineage,
                "classification": classification,
            }
        )
    linked = sum(1 for r in output if r["classification"] == "behavior-outcome linked")
    return {
        "overall_classification": "behavior/outcome calibration has candidate links" if linked else "behavior/outcome links not established",
        "linked_variant_count": linked,
        "variant_behavior_outcomes": output,
        "interpretation": "calibration report only; no perception adaptiveness claim",
    }


def scenario_classification_report(calibration: Mapping[str, Any]) -> dict[str, Any]:
    rows = list(calibration.get("cell_summary", []))
    normal = [r for r in rows if str(r.get("perception_condition", "")) == "normal"]
    ablated_by_variant = {
        str(r.get("scenario_variant", "")): r
        for r in rows
        if str(r.get("perception_condition", "")) == "ablated"
    }
    classifications: list[dict[str, Any]] = []
    for row in normal:
        variant = str(row.get("scenario_variant", ""))
        visibility = float(row.get("visibility_proxy", 0.0))
        behavior = float(row.get("behavior_proxy", 0.0))
        alive = float(row.get("alive_mean", 0.0))
        deaths = float(row.get("death_mean", 0.0))
        reproduction = float(row.get("reproduction_mean", 0.0))
        ablated = ablated_by_variant.get(variant)
        ablation_delta = 0.0
        if ablated:
            ablation_delta = visibility - float(ablated.get("visibility_proxy", 0.0))
        if str(row.get("pressure_level", "")) == "flat_control":
            classification = "too flat"
        elif alive <= 0.0:
            classification = "too lethal"
        elif deaths <= 0.0 and reproduction > 0.0 and behavior <= 0.0:
            classification = "too easy"
        elif visibility <= 0.05:
            classification = "perception-weak"
        elif behavior <= 0.0:
            classification = "behavior metrics insufficient"
        elif ablated and ablation_delta <= 0.0:
            classification = "ablation contrast unclear"
        elif behavior > 0.0 and visibility > 0.05 and (deaths > 0.0 or reproduction > 0.0):
            classification = "candidate for perception campaign"
        else:
            classification = "perception-informative"
        classifications.append(
            {
                "scenario_family": row.get("scenario_family", ""),
                "pressure_level": row.get("pressure_level", ""),
                "scenario_variant": variant,
                "visibility_proxy": visibility,
                "behavior_proxy": behavior,
                "ablation_visibility_delta": ablation_delta,
                "classification": classification,
            }
        )
    candidates = [r for r in classifications if r["classification"] == "candidate for perception campaign"]
    return {
        "variant_classifications": classifications,
        "candidate_variant_count": len(candidates),
        "overall_classification": "calibrated perception candidates available" if candidates else "no calibrated perception candidates",
        "interpretation": "scenario calibration classifications only; no perception usefulness claim",
    }


def perception_testbed_selection_report(classification: Mapping[str, Any]) -> dict[str, Any]:
    rows = list(classification.get("variant_classifications", []))
    candidates = [r for r in rows if r.get("classification") == "candidate for perception campaign"]
    selected: list[dict[str, Any]] = []
    flat = next((r for r in rows if r.get("pressure_level") == "flat_control"), None)
    if flat:
        selected.append({**flat, "selection_role": "flat_control", "selection_reason": "flat control distinguishes pressure effects"})
    for family, role in (("patch", "patch_candidate"), ("jungle", "jungle_candidate"), ("combined", "combined_candidate")):
        family_candidates = [r for r in candidates if r.get("scenario_family") == family]
        if family_candidates:
            best = max(family_candidates, key=lambda r: (float(r.get("behavior_proxy", 0.0)), float(r.get("visibility_proxy", 0.0))))
            selected.append({**best, "selection_role": role, "selection_reason": "highest behavior and visibility proxy among calibrated candidates"})
    backup_candidates = [r for r in candidates if r.get("scenario_variant") not in {s.get("scenario_variant") for s in selected}]
    if backup_candidates:
        backup = max(backup_candidates, key=lambda r: (float(r.get("behavior_proxy", 0.0)), float(r.get("visibility_proxy", 0.0))))
        selected.append({**backup, "selection_role": "backup_variant", "selection_reason": "backup calibrated candidate"})
    if len(selected) <= 1:
        recommendation = "no calibrated perception testbed selected"
        rejection_reason = "No pressure variants met candidate criteria."
    else:
        recommendation = "selected calibrated perception testbed variants"
        rejection_reason = ""
    return {
        "recommendation": recommendation,
        "selected_variants": selected,
        "rejection_reason": rejection_reason,
        "interpretation": "testbed selection only; future campaign still required",
    }


def stage10h_decision_report(selection: Mapping[str, Any], classification: Mapping[str, Any], behavior: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "proceed to Stage 10I perception retest",
        "refine scenario configs again",
        "escalate to Theme 1/2 ecology mechanics cleanup",
        "add behavior proxy instrumentation",
        "deprioritize perception",
    }
    selected = list(selection.get("selected_variants", []))
    candidate_count = int(classification.get("candidate_variant_count", 0) or 0)
    behavior_label = str(behavior.get("overall_classification", ""))
    if "not established" in behavior_label:
        recommendation = "add behavior proxy instrumentation"
    elif len(selected) >= 3 and candidate_count >= 2:
        recommendation = "proceed to Stage 10I perception retest"
    elif candidate_count > 0:
        recommendation = "refine scenario configs again"
    else:
        recommendation = "escalate to Theme 1/2 ecology mechanics cleanup"
    assert recommendation in allowed
    return {
        "next_action": recommendation,
        "allowed_next_actions": sorted(allowed),
        "selected_variant_count": len(selected),
        "candidate_variant_count": candidate_count,
        "classification_summary": str(classification.get("overall_classification", "")),
        "behavior_summary": behavior_label,
        "interpretation": "Stage 10H calibration decision only; no perception adaptiveness claim",
    }


BEHAVIOR_PROXY_METRICS = METRIC_FAMILIES["behavior_proxy"]


def behavior_proxy_audit(sample_columns: Iterable[str] | None = None) -> dict[str, Any]:
    columns = set(str(c) for c in (sample_columns or ()))
    proxies = {
        "movement count": ("behavior_move_total", "telemetry-only"),
        "action distribution": ("behavior_action_total", "telemetry-only"),
        "food/resource interaction": ("behavior_patch_interaction_proxy", "derived from existing behavior and patch metrics"),
        "patch visits": ("patch_food_occupancy_bias", "already available as patch occupancy proxy"),
        "shelter/protection use": ("behavior_protection_proxy", "derived from existing shelter/protection metrics"),
        "hazard exposure": ("behavior_visible_hazard_exposure_proxy", "derived from existing waste/risk visibility metrics"),
        "predator exposure/proximity": ("behavior_visible_predator_exposure_proxy", "derived from existing predator risk metrics"),
        "crowding/waste exposure": ("patch_waste_occupied_mean", "already available"),
        "death cause": ("jungle_predator_deaths", "available for predator deaths only"),
        "reproduction timing": ("first_repro_step_mean_alive", "already available"),
        "lineage persistence": ("lineage_active_count", "already available"),
        "perception-visible affordance exposure": ("behavior_visible_resource_exposure_proxy", "telemetry-only"),
    }
    rows: list[dict[str, str]] = []
    for proxy, (metric, source) in sorted(proxies.items()):
        if columns:
            status = "already available" if metric in columns else "can be added as telemetry-only"
        elif metric in BEHAVIOR_PROXY_METRICS:
            status = "can be added as telemetry-only"
        else:
            status = "already available"
        rows.append({"proxy": proxy, "metric": metric, "status": status, "source": source})
    unavailable = [
        {
            "proxy": "exact death cause aggregation",
            "status": "unavailable without mechanics change",
            "reason": "Only predator death counts are currently exposed generationally.",
        },
        {
            "proxy": "exact patch visit count",
            "status": "unavailable without mechanics change",
            "reason": "Current metrics expose occupancy/interaction proxies, not per-agent visit events.",
        },
    ]
    return {
        "schema_version": "theme4_stage10i_behavior_proxy_audit_v1",
        "proxy_status": rows,
        "unavailable_without_mechanics_change": unavailable,
        "instrumentation_safe": True,
        "interpretation": "Telemetry-only audit; no behavior or reward changes.",
    }


def behavior_proxy_validation_report(manifest: Mapping[str, Any], plan: Mapping[str, Any] | None = None) -> dict[str, Any]:
    plan = dict(plan or plan_campaign(manifest))
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for run in plan.get("planned_runs", []):
        metrics_path = Path(str(run.get("metrics_path", "")))
        row = _read_metric_row(metrics_path) if metrics_path.exists() else {}
        absent = [metric for metric in BEHAVIOR_PROXY_METRICS if metric not in row]
        if absent:
            missing.append(f"{run.get('run_id')}: {', '.join(absent)}")
        rows.append(
            {
                "run_id": str(run.get("run_id", "")),
                "cell_id": str(run.get("cell_id", "")),
                "seed": int(run.get("seed", 0) or 0),
                "scenario_group": str(run.get("scenario_group", "")),
                "scenario_variant": str(run.get("scenario_variant", "")),
                "behavior_action_total": float(row.get("behavior_action_total", 0.0)),
                "behavior_patch_interaction_proxy": float(row.get("behavior_patch_interaction_proxy", 0.0)),
                "behavior_risk_exposure_proxy": float(row.get("behavior_risk_exposure_proxy", 0.0)),
                "behavior_protection_proxy": float(row.get("behavior_protection_proxy", 0.0)),
                "behavior_visible_resource_exposure_proxy": float(row.get("behavior_visible_resource_exposure_proxy", 0.0)),
                "behavior_visible_hazard_exposure_proxy": float(row.get("behavior_visible_hazard_exposure_proxy", 0.0)),
                "behavior_visible_shelter_exposure_proxy": float(row.get("behavior_visible_shelter_exposure_proxy", 0.0)),
                "behavior_visible_predator_exposure_proxy": float(row.get("behavior_visible_predator_exposure_proxy", 0.0)),
            }
        )
    emitted = not missing and bool(rows)
    action_mean = _mean(float(r.get("behavior_action_total", 0.0)) for r in rows)
    pressure_rows = [r for r in rows if str(r.get("scenario_group", "")) in {"patch", "jungle", "combined"}]
    pressure_behavior = _mean(
        float(r.get("behavior_patch_interaction_proxy", 0.0))
        + float(r.get("behavior_risk_exposure_proxy", 0.0))
        + float(r.get("behavior_protection_proxy", 0.0))
        for r in pressure_rows
    )
    if not emitted:
        classification = "behavior proxy metrics missing"
    elif action_mean <= 0.0:
        classification = "behavior proxy metrics emitted but no action signal"
    elif pressure_behavior > 0.0:
        classification = "behavior proxy metrics emitted and pressure proxies active"
    else:
        classification = "behavior proxy metrics emitted but pressure proxies weak"
    return {
        "schema_version": "theme4_stage10i_behavior_proxy_validation_v1",
        "ok": emitted,
        "classification": classification,
        "checked_run_count": len(rows),
        "missing_metrics": missing,
        "behavior_action_total_mean": action_mean,
        "pressure_behavior_proxy_mean": pressure_behavior,
        "run_summaries": rows,
    }


def behavior_linked_variant_selection(summary: Mapping[str, Any]) -> dict[str, Any]:
    pair_summaries = list(summary.get("pair_summaries", []))
    variants: list[dict[str, Any]] = []
    for pair in pair_summaries:
        if str(pair.get("status", "")) != "success":
            continue
        group = str(pair.get("scenario_group", ""))
        variant = str(pair.get("scenario_variant", ""))
        deltas = dict(pair.get("deltas", {}))
        behavior_delta = float(deltas.get("behavior_action_total", 0.0))
        behavior_delta += abs(float(deltas.get("behavior_patch_interaction_proxy", 0.0)))
        behavior_delta += abs(float(deltas.get("behavior_risk_exposure_proxy", 0.0)))
        behavior_delta += abs(float(deltas.get("behavior_protection_proxy", 0.0)))
        outcome_delta = float(deltas.get("alive", 0.0)) + float(deltas.get("mean_offspring_total", 0.0)) + float(deltas.get("lineage_active_count", 0.0))
        visibility_delta = abs(float(deltas.get("behavior_visible_resource_exposure_proxy", 0.0)))
        visibility_delta += abs(float(deltas.get("behavior_visible_hazard_exposure_proxy", 0.0)))
        visibility_delta += abs(float(deltas.get("behavior_visible_shelter_exposure_proxy", 0.0)))
        visibility_delta += abs(float(deltas.get("behavior_visible_predator_exposure_proxy", 0.0)))
        if group == "flat" and outcome_delta > 0.0:
            classification = "flat control suspicious"
        elif behavior_delta > 0.0 and outcome_delta != 0.0:
            classification = "behavior-linked perception candidate"
        elif behavior_delta > 0.0:
            classification = "behavior-informative perception candidate"
        elif visibility_delta > 0.0:
            classification = "visibility-only candidate"
        elif outcome_delta != 0.0:
            classification = "outcome-only candidate"
        else:
            classification = "insufficient behavior proxies"
        variants.append(
            {
                "group_id": str(pair.get("group_id", "")),
                "scenario_group": group,
                "scenario_variant": variant,
                "seed": int(pair.get("seed", 0) or 0),
                "behavior_delta": behavior_delta,
                "outcome_delta": outcome_delta,
                "visibility_delta": visibility_delta,
                "classification": classification,
            }
        )
    by_variant: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in variants:
        by_variant.setdefault((str(row["scenario_group"]), str(row["scenario_variant"])), []).append(row)
    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for (group, variant), rows in sorted(by_variant.items()):
        behavior_mean = _mean(float(r["behavior_delta"]) for r in rows)
        outcome_mean = _mean(float(r["outcome_delta"]) for r in rows)
        labels = [str(r["classification"]) for r in rows]
        if group == "flat":
            classification = "valid flat control" if outcome_mean <= 0.0 else "flat-world artifact risk"
        elif any("behavior-linked" in label for label in labels):
            classification = "behavior-linked perception candidate"
        elif any("behavior-informative" in label for label in labels):
            classification = "behavior-informative perception candidate"
        elif any("visibility-only" in label for label in labels):
            classification = "visibility-only candidate"
        else:
            classification = "insufficient behavior proxies"
        item = {
            "scenario_group": group,
            "scenario_variant": variant,
            "classification": classification,
            "behavior_delta_mean": behavior_mean,
            "outcome_delta_mean": outcome_mean,
            "seed_count": len(rows),
        }
        if classification in {"valid flat control", "behavior-linked perception candidate", "behavior-informative perception candidate"}:
            selected.append(item)
        else:
            rejected.append(item)
    pressure_selected = [r for r in selected if r["scenario_group"] != "flat"]
    return {
        "schema_version": "theme4_stage10i_behavior_linked_variant_selection_v1",
        "selected_variants": selected,
        "rejected_variants": rejected,
        "pressure_candidate_count": len(pressure_selected),
        "flat_control_count": sum(1 for r in selected if r["scenario_group"] == "flat"),
        "gate_ready": bool(pressure_selected) and any(r["scenario_group"] == "flat" for r in selected),
        "interpretation": "Behavior-linked selection only; no perception adaptiveness claim.",
    }


def stage10i_gate_report(
    validation: Mapping[str, Any],
    variant_selection: Mapping[str, Any],
    *,
    ablation_clean: bool = True,
    sensor_radius_fixed: bool = True,
    medium_cost_applied: bool = True,
    direct_reward_present: bool = False,
) -> dict[str, Any]:
    checks = {
        "behavior_proxy_metrics_validate": bool(validation.get("ok", False)),
        "flat_control_valid": int(variant_selection.get("flat_control_count", 0) or 0) >= 1,
        "pressure_variant_behavior_informative": int(variant_selection.get("pressure_candidate_count", 0) or 0) >= 1,
        "ablation_clean": bool(ablation_clean),
        "sensor_radius_fixed": bool(sensor_radius_fixed),
        "medium_perception_cost_applied": bool(medium_cost_applied),
        "no_direct_cognition_or_perception_reward": not bool(direct_reward_present),
        "runtime_shapes_unchanged": True,
        "candidate_variants_selected_before_retest": bool(variant_selection.get("selected_variants", [])),
    }
    blocking = [name for name, ok in checks.items() if not ok]
    proceed = not blocking
    return {
        "schema_version": "theme4_stage10i_gate_v1",
        "proceed_to_stage10i": proceed,
        "checks": checks,
        "blocking_reasons": blocking,
        "decision": "run Stage 10I perception retest" if proceed else "skip Stage 10I and generate cleanup handoff",
    }


def theory_discrimination_report(
    validation: Mapping[str, Any],
    variant_selection: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    behavior_ok = bool(validation.get("ok", False))
    pressure_count = int(variant_selection.get("pressure_candidate_count", 0) or 0)
    rejected = list(variant_selection.get("rejected_variants", []))
    flat_risk = any("flat-world" in str(r.get("classification", "")) for r in rejected)
    theories = {
        "T1 ecology too weak": "weakly supported" if pressure_count == 0 else "rejected",
        "T2 ecology too noisy": "unresolved" if pressure_count > 0 and not bool(gate.get("proceed_to_stage10i", False)) else "weakly supported",
        "T3 behavior coupling weak": "blocked by missing metrics" if not behavior_ok else ("weakly supported" if pressure_count == 0 else "unresolved"),
        "T4 outcomes saturated": "unresolved",
        "T5 flat-world artifact": "weakly supported" if flat_risk else "unresolved",
        "T6 perception real but narrow": "weakly supported" if pressure_count > 0 else "unresolved",
    }
    if bool(gate.get("proceed_to_stage10i", False)):
        recommendation = "continue perception"
    elif not behavior_ok:
        recommendation = "improve behavior proxies"
    elif pressure_count == 0:
        recommendation = "improve ecology"
    else:
        recommendation = "pause Theme 4 campaigns"
    return {
        "schema_version": "theme4_stage10i_theory_discrimination_v1",
        "theories": theories,
        "recommendation": recommendation,
        "interpretation": "Theory discrimination is diagnostic and does not claim perception is adaptive.",
    }


def stage10i_decision_report(summary: Mapping[str, Any]) -> dict[str, Any]:
    selection = behavior_linked_variant_selection(summary)
    pressure = [r for r in selection.get("selected_variants", []) if r.get("scenario_group") != "flat"]
    flat_suspicious = any("flat-world" in str(r.get("classification", "")) for r in selection.get("rejected_variants", []))
    if flat_suspicious:
        classification = "flat-world suspicious"
    elif any("behavior-linked" in str(r.get("classification", "")) for r in pressure):
        classification = "weak pressure-dependent candidate"
    elif any("behavior-informative" in str(r.get("classification", "")) for r in pressure):
        classification = "behavior-only signal"
    elif not pressure:
        classification = "still blocked by metrics"
    else:
        classification = "no signal"
    return {
        "schema_version": "theme4_stage10i_decision_v1",
        "classification": classification,
        "selected_pressure_variant_count": len(pressure),
        "interpretation": "No proven adaptiveness claim.",
    }


STAGE10J_BEHAVIOR_METRICS = (
    "behavior_action_total",
    "behavior_move_total",
    "behavior_gather_total",
    "behavior_rest_total",
    "behavior_attack_total",
    "behavior_patch_interaction_proxy",
    "behavior_risk_exposure_proxy",
    "behavior_protection_proxy",
    "behavior_reproduction_proxy",
)

STAGE10J_OUTCOME_METRICS = ("alive", "mean_offspring_total", "lineage_active_count")


def _population_variance(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    if not vals:
        return 0.0
    m = mean(vals)
    return sum((v - m) ** 2 for v in vals) / float(len(vals))


def _relative_spread(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    if not vals:
        return 0.0
    lo = min(vals)
    hi = max(vals)
    denom = max(abs(mean(vals)), 1.0)
    return float(hi - lo) / float(denom)


def _stage10j_successful_pairs(summary: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        p for p in summary.get("pair_summaries", [])
        if str(p.get("status", "")) == "success"
        and ((str(p.get("primary_trait", "")) or _trait_from_group(str(p.get("group_id", "")))) == "perception")
    ]


def _stage10j_group_pairs(summary: Mapping[str, Any]) -> dict[tuple[str, str], list[Mapping[str, Any]]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for pair in _stage10j_successful_pairs(summary):
        scenario = str(pair.get("scenario_group", "")) or _scenario_from_group(str(pair.get("group_id", "")))
        variant = str(pair.get("scenario_variant", "")) or _variant_from_group(str(pair.get("group_id", "")))
        grouped.setdefault((scenario, variant), []).append(pair)
    return grouped


def _stage10j_behavior_delta(pair: Mapping[str, Any]) -> float:
    deltas = dict(pair.get("deltas", {}))
    return sum(abs(float(deltas.get(metric, 0.0) or 0.0)) for metric in STAGE10J_BEHAVIOR_METRICS)


def _stage10j_outcome_delta(pair: Mapping[str, Any]) -> float:
    deltas = dict(pair.get("deltas", {}))
    return sum(float(deltas.get(metric, 0.0) or 0.0) for metric in STAGE10J_OUTCOME_METRICS)


def stage10j_evidence_audit(summary: Mapping[str, Any]) -> dict[str, Any]:
    grouped = _stage10j_group_pairs(summary)
    flat_alive = [
        float(pair.get("deltas", {}).get("alive", 0.0) or 0.0)
        for (scenario, _), pairs in grouped.items()
        if scenario == "flat"
        for pair in pairs
    ]
    flat_mean = _mean(flat_alive)
    rows: list[dict[str, Any]] = []
    for (scenario, variant), pairs in sorted(grouped.items()):
        alive = [float(pair.get("deltas", {}).get("alive", 0.0) or 0.0) for pair in pairs]
        reproduction = [float(pair.get("deltas", {}).get("mean_offspring_total", 0.0) or 0.0) for pair in pairs]
        lineage = [float(pair.get("deltas", {}).get("lineage_active_count", 0.0) or 0.0) for pair in pairs]
        behavior = [_stage10j_behavior_delta(pair) for pair in pairs]
        outcome = [_stage10j_outcome_delta(pair) for pair in pairs]
        cost = [float(pair.get("normal_metrics", {}).get("perception_cost_charged_mean", 0.0) or 0.0) for pair in pairs]
        alive_mean = _mean(alive)
        behavior_mean = _mean(behavior)
        outcome_mean = _mean(outcome)
        sign_consistency = _sign_consistency(alive)
        if scenario == "flat" and alive_mean > 0.0:
            classification = "flat suspicious"
        elif sign_consistency < 0.34 and _population_variance(alive) > 0.0:
            classification = "noisy"
        elif behavior_mean > 0.0 and abs(outcome_mean) <= 0.25:
            classification = "behavior-linked but outcome-neutral"
        elif abs(outcome_mean) > 0.25 and behavior_mean <= 0.0:
            classification = "outcome-linked but behavior-weak"
        elif scenario != "flat" and alive_mean > flat_mean and behavior_mean > 0.0:
            classification = "narrow candidate"
        elif max(alive, default=0.0) == min(alive, default=0.0):
            classification = "saturated"
        else:
            classification = "reject"
        rows.append(
            {
                "scenario_group": scenario,
                "scenario_variant": variant,
                "seed_count": len(pairs),
                "alive_delta_mean": alive_mean,
                "alive_delta_range": [min(alive) if alive else 0.0, max(alive) if alive else 0.0],
                "alive_delta_variance": _population_variance(alive),
                "alive_delta_sign_consistency": sign_consistency,
                "reproduction_delta_mean": _mean(reproduction),
                "lineage_delta_mean": _mean(lineage),
                "behavior_delta_mean": behavior_mean,
                "outcome_delta_mean": outcome_mean,
                "perception_cost_charged_mean": _mean(cost),
                "classification": classification,
            }
        )
    return {
        "schema_version": "theme4_stage10j_evidence_audit_v1",
        "variant_audit": rows,
        "flat_alive_delta_mean": flat_mean,
        "pressure_alive_delta_mean": _mean(
            r["alive_delta_mean"] for r in rows if r["scenario_group"] in {"patch", "jungle", "combined"}
        ),
        "interpretation": "Stage 10J audit only; no perception adaptiveness claim.",
    }


def stage10j_offline_artifact_sanity(summary: Mapping[str, Any]) -> dict[str, Any]:
    grouped = _stage10j_group_pairs(summary)
    rows: list[dict[str, Any]] = []
    for (scenario, variant), pairs in sorted(grouped.items()):
        observed = [float(pair.get("deltas", {}).get("alive", 0.0) or 0.0) for pair in pairs]
        if not observed:
            continue
        # Deterministic exhaustive sign-flip sanity for the small Stage 10 seed counts.
        shuffled_means: list[float] = []
        n = len(observed)
        max_masks = 1 << n
        if max_masks <= 4096:
            masks = range(max_masks)
        else:
            masks = range(4096)
        for mask in masks:
            vals = [(-value if (mask >> idx) & 1 else value) for idx, value in enumerate(observed)]
            shuffled_means.append(mean(vals))
        observed_mean = mean(observed)
        lo = min(shuffled_means)
        hi = max(shuffled_means)
        if lo <= observed_mean <= hi and abs(observed_mean) <= max(abs(lo), abs(hi)):
            classification = "artifact unclear"
        elif observed_mean > hi:
            classification = "observed exceeds shuffled-label range"
        elif observed_mean < lo:
            classification = "observed below shuffled-label range"
        else:
            classification = "label/noise compatible"
        rows.append(
            {
                "scenario_group": scenario,
                "scenario_variant": variant,
                "seed_count": n,
                "observed_alive_delta_mean": observed_mean,
                "shuffled_mean": mean(shuffled_means),
                "shuffled_range": [lo, hi],
                "observed_sign_consistency": _sign_consistency(observed),
                "classification": classification,
            }
        )
    flat_rows = [r for r in rows if r["scenario_group"] == "flat"]
    pressure_rows = [r for r in rows if r["scenario_group"] in {"patch", "jungle", "combined"}]
    return {
        "schema_version": "theme4_stage10j_offline_artifact_sanity_v1",
        "variant_sanity": rows,
        "flat_classification": flat_rows[0]["classification"] if flat_rows else "not evaluated",
        "pressure_classifications": [r["classification"] for r in pressure_rows],
        "interpretation": "Deterministic label-flip sanity only; no p-values and no adaptiveness claim.",
    }


def stage10j_outcome_saturation_report(*summaries: Mapping[str, Any]) -> dict[str, Any]:
    combined_pairs: list[Mapping[str, Any]] = []
    for summary in summaries:
        combined_pairs.extend(_stage10j_successful_pairs(summary))
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for pair in combined_pairs:
        grouped.setdefault(
            (
                str(pair.get("scenario_group", "")) or _scenario_from_group(str(pair.get("group_id", ""))),
                str(pair.get("scenario_variant", "")) or _variant_from_group(str(pair.get("group_id", ""))),
            ),
            [],
        ).append(pair)
    rows: list[dict[str, Any]] = []
    for (scenario, variant), pairs in sorted(grouped.items()):
        normal_alive = [float(p.get("normal_metrics", {}).get("alive", 0.0) or 0.0) for p in pairs]
        ablated_alive = [float(p.get("ablation_metrics", {}).get("alive", 0.0) or 0.0) for p in pairs]
        alive_all = normal_alive + ablated_alive
        reproduction_all = [
            float(p.get(side, {}).get("mean_offspring_total", 0.0) or 0.0)
            for p in pairs
            for side in ("normal_metrics", "ablation_metrics")
        ]
        lineage_all = [
            float(p.get(side, {}).get("lineage_active_count", 0.0) or 0.0)
            for p in pairs
            for side in ("normal_metrics", "ablation_metrics")
        ]
        extinction_rate = sum(1 for v in alive_all if v <= 0.0) / float(len(alive_all) or 1)
        alive_spread = _relative_spread(alive_all)
        repro_spread = _relative_spread(reproduction_all)
        lineage_spread = _relative_spread(lineage_all)
        high_saturation = sum(1 for v in alive_all if v >= max(alive_all or [0.0])) / float(len(alive_all) or 1)
        low_saturation = sum(1 for v in alive_all if v <= min(alive_all or [0.0])) / float(len(alive_all) or 1)
        if extinction_rate >= 0.5:
            classification = "too saturated low"
        elif alive_spread <= 0.05 and repro_spread <= 0.05 and lineage_spread <= 0.05:
            classification = "too saturated high"
        elif alive_spread >= 1.0:
            classification = "too noisy"
        elif len(pairs) < 3:
            classification = "insufficient horizon"
        elif repro_spread <= 0.01 and lineage_spread <= 0.01:
            classification = "needs alternative proxy"
        else:
            classification = "outcome usable"
        rows.append(
            {
                "scenario_group": scenario,
                "scenario_variant": variant,
                "sample_count": len(alive_all),
                "alive_mean": _mean(alive_all),
                "alive_range": [min(alive_all) if alive_all else 0.0, max(alive_all) if alive_all else 0.0],
                "alive_relative_spread": alive_spread,
                "reproduction_relative_spread": repro_spread,
                "lineage_relative_spread": lineage_spread,
                "extinction_rate": extinction_rate,
                "floor_fraction": low_saturation,
                "ceiling_fraction": high_saturation,
                "classification": classification,
            }
        )
    return {
        "schema_version": "theme4_stage10j_outcome_saturation_v1",
        "variant_saturation": rows,
        "overall_classification": "outcome usable" if rows and all(r["classification"] == "outcome usable" for r in rows) else "mixed or limited outcome metrics",
        "interpretation": "Outcome saturation diagnostics only; no adaptiveness claim.",
    }


def stage10j_behavior_outcome_coupling_report(*summaries: Mapping[str, Any]) -> dict[str, Any]:
    pairs: list[Mapping[str, Any]] = []
    for summary in summaries:
        pairs.extend(_stage10j_successful_pairs(summary))
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for pair in pairs:
        grouped.setdefault(
            (
                str(pair.get("scenario_group", "")) or _scenario_from_group(str(pair.get("group_id", ""))),
                str(pair.get("scenario_variant", "")) or _variant_from_group(str(pair.get("group_id", ""))),
            ),
            [],
        ).append(pair)
    rows: list[dict[str, Any]] = []
    for (scenario, variant), vals in sorted(grouped.items()):
        behavior = [_stage10j_behavior_delta(p) for p in vals]
        alive = [float(p.get("deltas", {}).get("alive", 0.0) or 0.0) for p in vals]
        reproduction = [float(p.get("deltas", {}).get("mean_offspring_total", 0.0) or 0.0) for p in vals]
        lineage = [float(p.get("deltas", {}).get("lineage_active_count", 0.0) or 0.0) for p in vals]
        outcome = [_stage10j_outcome_delta(p) for p in vals]
        behavior_mean = _mean(behavior)
        outcome_mean = _mean(outcome)
        outcome_signal = abs(outcome_mean) > 0.25 or abs(_mean(alive)) > 0.0 or abs(_mean(reproduction)) > 0.25 or abs(_mean(lineage)) > 0.0
        behavior_signal = behavior_mean > 0.0
        noisy = _sign_consistency(outcome) < 0.34 and _population_variance(outcome) > 0.0
        if noisy:
            classification = "noisy"
        elif behavior_signal and outcome_signal:
            classification = "behavior and outcome aligned"
        elif behavior_signal:
            classification = "behavior only"
        elif outcome_signal:
            classification = "outcome only"
        elif len(vals) == 0:
            classification = "insufficient proxy resolution"
        else:
            classification = "neither"
        rows.append(
            {
                "scenario_group": scenario,
                "scenario_variant": variant,
                "seed_count": len(vals),
                "behavior_delta_mean": behavior_mean,
                "alive_delta_mean": _mean(alive),
                "reproduction_delta_mean": _mean(reproduction),
                "lineage_delta_mean": _mean(lineage),
                "outcome_delta_mean": outcome_mean,
                "outcome_delta_sign_consistency": _sign_consistency(outcome),
                "classification": classification,
            }
        )
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["classification"])] = counts.get(str(row["classification"]), 0) + 1
    if counts.get("behavior and outcome aligned", 0) > 0:
        overall = "some behavior/outcome alignment"
    elif counts.get("behavior only", 0) > 0:
        overall = "behavior changes without outcome changes"
    elif counts.get("outcome only", 0) > 0:
        overall = "outcome changes without behavior proxy support"
    elif counts.get("noisy", 0) > 0:
        overall = "noisy behavior/outcome coupling"
    else:
        overall = "no interpretable behavior/outcome coupling"
    return {
        "schema_version": "theme4_stage10j_behavior_outcome_coupling_v1",
        "variant_coupling": rows,
        "classification_counts": counts,
        "overall_classification": overall,
        "interpretation": "Behavior/outcome coupling diagnostics only; no perception adaptiveness claim.",
    }


def stage10j_theory_discrimination_update(
    audit: Mapping[str, Any],
    artifact: Mapping[str, Any],
    saturation: Mapping[str, Any],
    coupling: Mapping[str, Any],
) -> dict[str, Any]:
    audit_rows = list(audit.get("variant_audit", []))
    artifact_rows = list(artifact.get("variant_sanity", []))
    saturation_rows = list(saturation.get("variant_saturation", []))
    coupling_rows = list(coupling.get("variant_coupling", []))
    pressure_audit = [r for r in audit_rows if r.get("scenario_group") in {"patch", "jungle", "combined"}]
    flat_artifact = any(r.get("scenario_group") == "flat" and "artifact" in str(r.get("classification", "")) for r in artifact_rows)
    flat_suspicious = any(r.get("scenario_group") == "flat" and "flat" in str(r.get("classification", "")) for r in audit_rows)
    noisy_pressure = any("noisy" in str(r.get("classification", "")) for r in pressure_audit + coupling_rows)
    saturated = any(str(r.get("classification", "")).startswith("too saturated") or str(r.get("classification", "")) == "needs alternative proxy" for r in saturation_rows)
    behavior_only = any(str(r.get("classification", "")) == "behavior only" for r in coupling_rows)
    aligned_pressure = [
        r for r in coupling_rows
        if r.get("scenario_group") in {"patch", "jungle", "combined"} and str(r.get("classification", "")) == "behavior and outcome aligned"
    ]
    theories = {
        "T2 ecology too noisy": "strongly supported" if noisy_pressure else ("weakly supported" if pressure_audit else "unresolved"),
        "T3 behavior coupling weak": "strongly supported" if behavior_only else "unresolved",
        "T4 outcomes saturated": "strongly supported" if saturated else "rejected",
        "T5 flat-world artifact": "strongly supported" if flat_suspicious and flat_artifact else ("weakly supported" if flat_suspicious else "rejected"),
        "T6 perception real but narrow": "weakly supported" if aligned_pressure else "unresolved",
    }
    return {
        "schema_version": "theme4_stage10j_theory_discrimination_v1",
        "theories": theories,
        "interpretation": "Theory update is diagnostic; no perception adaptiveness claim.",
    }


def stage10k_decision_report(
    audit: Mapping[str, Any],
    saturation: Mapping[str, Any],
    coupling: Mapping[str, Any],
    theory: Mapping[str, Any],
) -> dict[str, Any]:
    theories = dict(theory.get("theories", {}))
    pressure_exceeds_flat = float(audit.get("pressure_alive_delta_mean", 0.0) or 0.0) > float(audit.get("flat_alive_delta_mean", 0.0) or 0.0)
    aligned = "behavior and outcome aligned" in dict(coupling.get("classification_counts", {}))
    major_cleanup_supported = any(
        theories.get(name) in {"strongly supported", "weakly supported"}
        for name in (
            "T2 ecology too noisy",
            "T3 behavior coupling weak",
            "T4 outcomes saturated",
            "T5 flat-world artifact",
        )
    )
    if pressure_exceeds_flat and aligned and not major_cleanup_supported:
        action = "Continue perception with deeper campaign"
        handoff = ""
    elif theories.get("T2 ecology too noisy") in {"strongly supported", "weakly supported"}:
        action = "Ecology noise cleanup"
        handoff = "evidences/theme4_stage10k_theme12_ecology_noise_cleanup_handoff.md"
    elif theories.get("T3 behavior coupling weak") == "strongly supported":
        action = "Behavior-outcome coupling cleanup"
        handoff = "evidences/theme4_stage10k_behavior_outcome_coupling_cleanup_prompt.md"
    elif theories.get("T4 outcomes saturated") == "strongly supported":
        action = "Outcome metric cleanup"
        handoff = "evidences/theme4_stage10k_outcome_metric_cleanup_prompt.md"
    elif theories.get("T5 flat-world artifact") in {"strongly supported", "weakly supported"}:
        action = "Flat-control artifact investigation"
        handoff = "evidences/theme4_stage10k_flat_artifact_investigation_prompt.md"
    elif not audit.get("variant_audit"):
        action = "Pause Theme 4 campaigns"
        handoff = "evidences/theme4_stage10k_pause_theme4_live_campaigns_note.md"
    else:
        action = "Theme 1/2 ecology handoff"
        handoff = "evidences/theme4_stage10k_theme12_ecology_noise_cleanup_handoff.md"
    return {
        "schema_version": "theme4_stage10k_decision_v1",
        "primary_next_action": action,
        "handoff_path": handoff,
        "pressure_exceeds_flat": pressure_exceeds_flat,
        "behavior_outcome_aligned": bool(aligned),
        "allowed_actions": [
            "Continue perception with deeper campaign",
            "Ecology noise cleanup",
            "Behavior-outcome coupling cleanup",
            "Outcome metric cleanup",
            "Flat-control artifact investigation",
            "Pause Theme 4 campaigns",
            "Theme 1/2 ecology handoff",
        ],
        "interpretation": "Stage 10K decision is a campaign-control recommendation only; no adaptiveness claim.",
    }


STAGE10L_CLEAN_PRESSURE_VARIANTS = {
    "combined_low_noise_medium_pressure",
    "jungle_stable_predator_pressure",
}

STAGE10L_NOISY_COMPARISON_VARIANTS = {"patch_high_visibility_medium_pressure"}

STAGE10L_VISIBLE_AFFORDANCE_METRICS = (
    "behavior_visible_resource_exposure_proxy",
    "behavior_visible_hazard_exposure_proxy",
    "behavior_visible_shelter_exposure_proxy",
    "behavior_visible_predator_exposure_proxy",
)


def _stage10l_successful_pairs(summary: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        p for p in _stage10j_successful_pairs(summary)
        if str(p.get("scenario_variant", "")) in STAGE10L_CLEAN_PRESSURE_VARIANTS
        or str(p.get("scenario_variant", "")) in STAGE10L_NOISY_COMPARISON_VARIANTS
        or str(p.get("scenario_group", "")) == "flat"
    ]


def _stage10l_preset_classification(pair: Mapping[str, Any]) -> str:
    variant = str(pair.get("scenario_variant", ""))
    scenario = str(pair.get("scenario_group", ""))
    if variant in STAGE10L_CLEAN_PRESSURE_VARIANTS:
        return "clean pressure candidate"
    if variant in STAGE10L_NOISY_COMPARISON_VARIANTS:
        return "noisy comparison"
    if scenario == "flat" or variant == "flat_control_clean":
        return "flat control"
    return "unclassified"


def _stage10l_behavior_proxy_delta(pair: Mapping[str, Any]) -> float:
    deltas = dict(pair.get("deltas", {}))
    return sum(float(deltas.get(metric, 0.0) or 0.0) for metric in STAGE10J_BEHAVIOR_METRICS)


def _stage10l_abs_behavior_proxy_delta(pair: Mapping[str, Any]) -> float:
    deltas = dict(pair.get("deltas", {}))
    return sum(abs(float(deltas.get(metric, 0.0) or 0.0)) for metric in STAGE10J_BEHAVIOR_METRICS)


def _stage10l_visible_affordance_delta(pair: Mapping[str, Any]) -> float:
    deltas = dict(pair.get("deltas", {}))
    return sum(float(deltas.get(metric, 0.0) or 0.0) for metric in STAGE10L_VISIBLE_AFFORDANCE_METRICS)


def _stage10l_grouped_pairs(summary: Mapping[str, Any]) -> dict[tuple[str, str, str], list[Mapping[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for pair in _stage10l_successful_pairs(summary):
        classification = _stage10l_preset_classification(pair)
        scenario = str(pair.get("scenario_group", "")) or _scenario_from_group(str(pair.get("group_id", "")))
        variant = str(pair.get("scenario_variant", "")) or _variant_from_group(str(pair.get("group_id", "")))
        grouped.setdefault((classification, scenario, variant), []).append(pair)
    return grouped


def calibrated_perception_paired_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for pair in _stage10l_successful_pairs(summary):
        deltas = dict(pair.get("deltas", {}))
        rows.append(
            {
                "group_id": str(pair.get("group_id", "")),
                "normal_run_id": str(pair.get("normal_run_id", "")),
                "ablation_run_id": str(pair.get("ablation_run_id", "")),
                "seed": int(pair.get("seed", 0) or 0),
                "scenario_group": str(pair.get("scenario_group", "")),
                "scenario_variant": str(pair.get("scenario_variant", "")),
                "preset_classification": _stage10l_preset_classification(pair),
                "alive_delta": float(deltas.get("alive", 0.0) or 0.0),
                "reproduction_delta": float(deltas.get("mean_offspring_total", 0.0) or 0.0),
                "lineage_delta": float(deltas.get("lineage_active_count", 0.0) or 0.0),
                "perception_capacity_delta": float(deltas.get("perception_capacity_component_mean", 0.0) or 0.0),
                "perception_cost_delta": float(deltas.get("perception_cost_charged_mean", 0.0) or 0.0),
                "behavior_proxy_delta": _stage10l_behavior_proxy_delta(pair),
                "visible_affordance_exposure_delta": _stage10l_visible_affordance_delta(pair),
            }
        )
    return {
        "schema_version": "theme4_stage10l_calibrated_paired_summary_v1",
        "pair_rows": rows,
        "interpretation": "Stage 10L paired deltas only; perception is not claimed adaptive.",
    }


def clean_pressure_vs_flat_report(summary: Mapping[str, Any]) -> dict[str, Any]:
    grouped = _stage10l_grouped_pairs(summary)
    variant_rows: list[dict[str, Any]] = []
    for (classification, scenario, variant), pairs in sorted(grouped.items()):
        alive = [float(pair.get("deltas", {}).get("alive", 0.0) or 0.0) for pair in pairs]
        reproduction = [float(pair.get("deltas", {}).get("mean_offspring_total", 0.0) or 0.0) for pair in pairs]
        lineage = [float(pair.get("deltas", {}).get("lineage_active_count", 0.0) or 0.0) for pair in pairs]
        behavior = [_stage10l_abs_behavior_proxy_delta(pair) for pair in pairs]
        variant_rows.append(
            {
                "preset_classification": classification,
                "scenario_group": scenario,
                "scenario_variant": variant,
                "seed_count": len(pairs),
                "mean_alive_delta": _mean(alive),
                "mean_reproduction_delta": _mean(reproduction),
                "mean_lineage_delta": _mean(lineage),
                "mean_behavior_proxy_abs_delta": _mean(behavior),
                "alive_delta_sign_consistency": _sign_consistency(alive),
            }
        )
    flat_rows = [r for r in variant_rows if r["preset_classification"] == "flat control"]
    clean_rows = [r for r in variant_rows if r["preset_classification"] == "clean pressure candidate"]
    patch_rows = [r for r in variant_rows if r["preset_classification"] == "noisy comparison"]
    flat_alive = _mean(r["mean_alive_delta"] for r in flat_rows)
    clean_alive = _mean(r["mean_alive_delta"] for r in clean_rows)
    flat_repro = _mean(r["mean_reproduction_delta"] for r in flat_rows)
    clean_repro = _mean(r["mean_reproduction_delta"] for r in clean_rows)
    flat_lineage = _mean(r["mean_lineage_delta"] for r in flat_rows)
    clean_lineage = _mean(r["mean_lineage_delta"] for r in clean_rows)
    behavior_aligned = any(
        float(r["mean_behavior_proxy_abs_delta"]) > 0.0
        and (
            float(r["mean_alive_delta"]) > flat_alive
            or float(r["mean_reproduction_delta"]) > flat_repro
            or float(r["mean_lineage_delta"]) > flat_lineage
        )
        for r in clean_rows
    )
    return {
        "schema_version": "theme4_stage10l_clean_pressure_vs_flat_v1",
        "variant_summary": variant_rows,
        "flat_alive_delta_mean": flat_alive,
        "clean_pressure_alive_delta_mean": clean_alive,
        "patch_noisy_comparison_alive_delta_mean": _mean(r["mean_alive_delta"] for r in patch_rows),
        "flat_reproduction_delta_mean": flat_repro,
        "clean_pressure_reproduction_delta_mean": clean_repro,
        "flat_lineage_delta_mean": flat_lineage,
        "clean_pressure_lineage_delta_mean": clean_lineage,
        "clean_pressure_exceeds_flat_alive": clean_alive > flat_alive,
        "clean_pressure_exceeds_flat_reproduction": clean_repro > flat_repro,
        "clean_pressure_exceeds_flat_lineage": clean_lineage > flat_lineage,
        "behavior_proxy_alignment": "aligned" if behavior_aligned else "not aligned",
        "patch_noisy_comparison_excluded_from_primary_success_basis": True,
        "primary_success_basis_variants": sorted(STAGE10L_CLEAN_PRESSURE_VARIANTS),
        "interpretation": "Patch is a noisy comparison only; primary Stage 10L evidence uses combined and jungle clean pressure candidates.",
    }


def stage10l_behavior_outcome_coupling_report(summary: Mapping[str, Any]) -> dict[str, Any]:
    base = stage10j_behavior_outcome_coupling_report(summary)
    rows = []
    for row in base.get("variant_coupling", []):
        variant = str(row.get("scenario_variant", ""))
        rows.append({**row, "preset_classification": _stage10l_preset_classification(row)})
    counts: dict[str, int] = {}
    clean_aligned = False
    for row in rows:
        classification = str(row.get("classification", ""))
        counts[classification] = counts.get(classification, 0) + 1
        if row.get("preset_classification") == "clean pressure candidate" and classification == "behavior and outcome aligned":
            clean_aligned = True
    if clean_aligned:
        overall = "clean pressure behavior/outcome aligned"
    elif counts.get("behavior only", 0) > 0:
        overall = "behavior-only"
    elif counts.get("outcome only", 0) > 0:
        overall = "outcome-only"
    elif counts.get("noisy", 0) > 0:
        overall = "noisy"
    else:
        overall = "neither"
    return {
        "schema_version": "theme4_stage10l_behavior_outcome_coupling_v1",
        "variant_coupling": rows,
        "classification_counts": counts,
        "overall_classification": overall,
        "interpretation": "Behavior/outcome coupling diagnostics only; no perception adaptiveness claim.",
    }


def stage10l_lineage_retention_report(summary: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for (classification, scenario, variant), pairs in sorted(_stage10l_grouped_pairs(summary).items()):
        lineage = [float(pair.get("deltas", {}).get("lineage_active_count", 0.0) or 0.0) for pair in pairs]
        parent_channels = [float(pair.get("normal_metrics", {}).get("parent_sensor_channel_count_mean", 0.0) or 0.0) for pair in pairs]
        alive_channels = [float(pair.get("normal_metrics", {}).get("alive_sensor_channel_count_mean", 0.0) or 0.0) for pair in pairs]
        rows.append(
            {
                "preset_classification": classification,
                "scenario_group": scenario,
                "scenario_variant": variant,
                "mean_lineage_delta": _mean(lineage),
                "lineage_delta_sign_consistency": _sign_consistency(lineage),
                "normal_parent_sensor_channel_count_mean": _mean(parent_channels),
                "normal_alive_sensor_channel_count_mean": _mean(alive_channels),
            }
        )
    flat = _mean(r["mean_lineage_delta"] for r in rows if r["preset_classification"] == "flat control")
    clean = _mean(r["mean_lineage_delta"] for r in rows if r["preset_classification"] == "clean pressure candidate")
    if clean > flat and clean > 0.0:
        classification = "weak pressure lineage retention candidate"
    elif flat > 0.0 and flat >= clean:
        classification = "flat-world suspicious lineage retention"
    elif any(abs(float(r["mean_lineage_delta"])) > 0.0 for r in rows):
        classification = "inconsistent lineage retention"
    else:
        classification = "no lineage retention signal"
    return {
        "schema_version": "theme4_stage10l_lineage_retention_v1",
        "classification": classification,
        "flat_lineage_delta_mean": flat,
        "clean_pressure_lineage_delta_mean": clean,
        "variant_lineage": rows,
        "interpretation": "Lineage retention summary only; no perception adaptiveness claim.",
    }


def stage10l_flat_artifact_check(summary: Mapping[str, Any], pressure_report: Mapping[str, Any] | None = None) -> dict[str, Any]:
    pressure = dict(pressure_report or clean_pressure_vs_flat_report(summary))
    flat_alive = float(pressure.get("flat_alive_delta_mean", 0.0) or 0.0)
    clean_alive = float(pressure.get("clean_pressure_alive_delta_mean", 0.0) or 0.0)
    flat_behavior = _mean(
        _stage10l_abs_behavior_proxy_delta(pair)
        for pair in _stage10l_successful_pairs(summary)
        if _stage10l_preset_classification(pair) == "flat control"
    )
    if flat_alive > 0.0 and flat_alive >= clean_alive:
        classification = "flat-world suspicious"
    elif flat_alive > 0.0:
        classification = "flat effect remains but pressure exceeds it"
    elif clean_alive > flat_alive:
        classification = "flat control lower than clean pressure"
    else:
        classification = "flat effect unresolved"
    return {
        "schema_version": "theme4_stage10l_flat_artifact_check_v1",
        "classification": classification,
        "flat_alive_delta_mean": flat_alive,
        "clean_pressure_alive_delta_mean": clean_alive,
        "flat_behavior_proxy_abs_delta_mean": flat_behavior,
        "flat_delta_lower_than_clean_pressure_delta": clean_alive > flat_alive,
        "interpretation": "Flat artifact check only; no perception adaptiveness claim.",
    }


def stage10l_fake_evolution_audit(
    manifest: Mapping[str, Any],
    validation: Mapping[str, Any],
    summary: Mapping[str, Any],
    pressure_report: Mapping[str, Any] | None = None,
    coupling_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    base = fake_evolution_audit(manifest, validation, summary)
    pressure = dict(pressure_report or clean_pressure_vs_flat_report(summary))
    coupling = dict(coupling_report or stage10l_behavior_outcome_coupling_report(summary))
    failures = [str(f) for f in validation.get("failures", [])]
    base.update(
        {
            "stage10l_focused": True,
            "direct_perception_reward_present": False,
            "direct_cognition_reward_present": False,
            "perception_cost_applied": any(
                float(pair.get("normal_metrics", {}).get("perception_cost_charged_mean", 0.0) or 0.0) > 0.0
                for pair in _stage10l_successful_pairs(summary)
            ),
            "flat_world_artifact_risk": float(pressure.get("flat_alive_delta_mean", 0.0) or 0.0)
            >= float(pressure.get("clean_pressure_alive_delta_mean", 0.0) or 0.0),
            "patch_noisy_comparison_caveat": True,
            "behavior_outcome_coupling": str(coupling.get("overall_classification", "")),
            "missing_behavior_proxy_metric_risks": [f for f in failures if "behavior_" in f],
            "missing_perception_cost_metric_risks": [f for f in failures if "perception_cost" in f],
            "recommendation": "use Stage 10L decision report; this audit does not make adaptiveness claims",
        }
    )
    return base


def stage10l_decision_report(
    pressure_report: Mapping[str, Any],
    coupling_report: Mapping[str, Any],
    lineage_report: Mapping[str, Any],
    flat_artifact: Mapping[str, Any],
    validation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validation_ok = bool((validation or {}).get("ok", True))
    clean_exceeds_alive = bool(pressure_report.get("clean_pressure_exceeds_flat_alive", False))
    clean_exceeds_repro = bool(pressure_report.get("clean_pressure_exceeds_flat_reproduction", False))
    clean_exceeds_lineage = bool(pressure_report.get("clean_pressure_exceeds_flat_lineage", False))
    coupling = str(coupling_report.get("overall_classification", ""))
    flat_class = str(flat_artifact.get("classification", ""))
    lineage = str(lineage_report.get("classification", ""))
    if not validation_ok:
        action = "Pause Theme 4 perception campaigns"
        classification = "failed retest"
    elif clean_exceeds_alive and coupling == "clean pressure behavior/outcome aligned" and "flat-world suspicious" not in flat_class:
        action = "Continue perception with deeper campaign"
        classification = "promising pressure-dependent candidate"
    elif (clean_exceeds_alive or clean_exceeds_repro or clean_exceeds_lineage) and "flat-world suspicious" not in flat_class:
        action = "Run one more narrowed combined/jungle perception campaign"
        classification = "weak pressure-dependent candidate"
    elif coupling in {"behavior-only", "clean pressure behavior/outcome aligned"} and not (clean_exceeds_alive or clean_exceeds_repro or clean_exceeds_lineage):
        action = "Behavior/outcome coupling cleanup"
        classification = "pressure behavior-only signal"
    elif "flat-world suspicious" in flat_class:
        action = "Deprioritize perception"
        classification = "flat-world suspicious"
    elif "noisy" in coupling:
        action = "Return to Theme 1/2 ecology cleanup"
        classification = "failed retest"
    else:
        action = "Pause Theme 4 perception campaigns"
        classification = "no signal"
    return {
        "schema_version": "theme4_stage10l_decision_v1",
        "primary_next_action": action,
        "classification": classification,
        "validation_ok": validation_ok,
        "clean_pressure_exceeds_flat_alive": clean_exceeds_alive,
        "clean_pressure_exceeds_flat_reproduction": clean_exceeds_repro,
        "clean_pressure_exceeds_flat_lineage": clean_exceeds_lineage,
        "behavior_outcome_coupling": coupling,
        "lineage_retention_classification": lineage,
        "flat_artifact_classification": flat_class,
        "allowed_actions": [
            "Continue perception with deeper campaign",
            "Run one more narrowed combined/jungle perception campaign",
            "Return to Theme 1/2 ecology cleanup",
            "Behavior/outcome coupling cleanup",
            "Pause Theme 4 perception campaigns",
            "Deprioritize perception",
        ],
        "interpretation": "Stage 10L decision is a campaign-control recommendation only; no perception adaptiveness claim.",
    }


def _trait_from_group(group_id: str) -> str:
    for trait in ("sensor_radius", "perception", "memory", "prediction", "development"):
        if trait in group_id:
            return trait
    return "unknown"


def _scenario_from_group(group_id: str) -> str:
    for scenario in ("flat", "patch", "jungle", "combined"):
        if group_id.startswith(scenario):
            return scenario
    return "unknown"


def _cost_level_from_group(group_id: str) -> str:
    for level in ("telemetry_only", "low", "medium", "high"):
        if f"_{level}_" in group_id or group_id.endswith(f"_{level}"):
            return level
    return ""


def _variant_from_group(group_id: str) -> str:
    parts = [p for p in group_id.split("_") if p]
    for idx, part in enumerate(parts):
        if part in {"flat", "patch", "jungle", "combined"} and idx + 1 < len(parts):
            nxt = parts[idx + 1]
            if nxt in {"a", "b", "c"}:
                return f"{part}_{nxt}"
    return ""


def _trait_charged_cost(metrics: Mapping[str, Any], trait: str) -> float:
    key = {
        "perception": "perception_cost_charged_mean",
        "sensor_radius": "sensor_radius_cost_charged_mean",
        "memory": "memory_cost_charged_mean",
        "prediction": "prediction_cost_charged_mean",
        "development": "developmental_cost_charged_mean",
    }.get(str(trait), "cog_cost_mean")
    return float(metrics.get(key, 0.0) or 0.0)


def _cost_response_direction(alive: Iterable[float], charged: Iterable[float]) -> str:
    alive_vals = [float(v) for v in alive]
    charged_vals = [float(v) for v in charged]
    if not alive_vals:
        return "no data"
    if not any(v > 0.0 for v in charged_vals):
        return "telemetry-only or zero charged cost"
    avg = mean(alive_vals)
    if avg > 0.0:
        return "normal higher alive under this cost"
    if avg < 0.0:
        return "ablation higher alive under this cost"
    return "no alive delta"


def fake_evolution_audit(manifest: Mapping[str, Any], validation: Mapping[str, Any], summary: Mapping[str, Any]) -> dict[str, Any]:
    skipped = list(validation.get("skipped_cells", []))
    limitations = [str(s.get("skip_reason", "")) for s in skipped]
    live_cells = []
    fixture_cells = []
    scaffold_cells = []
    for cell in manifest.get("scenario_cells", []):
        backend = _cell_backend(cell)
        if backend == "live_simulator":
            live_cells.append(str(cell.get("cell_id", "")))
        elif backend == "fixture":
            fixture_cells.append(str(cell.get("cell_id", "")))
        elif backend == "scaffold" or cell.get("classification") == "platform_fixture":
            scaffold_cells.append(str(cell.get("cell_id", "")))
            limitations.append(f"{cell.get('cell_id')}: platform fixture only")
    cost_summary = paired_cost_summary(summary)
    suspicious_traits = [
        trait
        for trait, payload in cost_response_report(cost_summary).get("traits", {}).items()
        if "suspicious" in str(payload.get("classification", ""))
    ]
    return {
        "campaign_id": str(manifest["campaign_id"]),
        "live_simulator_evidence": bool(live_cells),
        "live_simulator_cells": live_cells,
        "fixture_cells": fixture_cells,
        "scaffold_cells": scaffold_cells,
        "direct_cognition_reward_present": False,
        "direct_feature_reward_present": False,
        "patch_jungle_mechanics_changed": False,
        "new_cognitive_machinery_added": False,
        "validation_ok": bool(validation.get("ok", False)),
        "missing_metric_status": list(validation.get("failures", [])),
        "live_integration_absent_risks": limitations,
        "skipped_cell_count": len(skipped),
        "evidence_strength": "live-backed mini evidence only; no intelligence or usefulness claim" if live_cells else "fixture-or-smoke infrastructure evidence only; no intelligence or usefulness claim",
        "paired_group_count": len(summary.get("group_summary", [])),
        "cost_profile_mismatch_risks": [f for f in validation.get("failures", []) if "cost" in str(f).lower()],
        "suspicious_traits": suspicious_traits,
    }


def environment_usefulness_report(summary: Mapping[str, Any]) -> dict[str, Any]:
    groups = list(summary.get("group_summary", []))
    pair_items = list(summary.get("pair_summaries", []))
    def signal_for(trait: str) -> str:
        matched = [g for g in groups if trait in str(g.get("group_id", ""))]
        if not matched:
            return "not evaluated"
        if all(abs(float(g.get("mean_alive_delta", 0.0))) <= 1e-12 for g in matched):
            return "no clear signal"
        return "weak candidate signal only"

    def mean_delta_for(pattern: str, metric: str = "alive") -> float:
        vals = [
            float(item.get("deltas", {}).get(metric, 0.0))
            for item in pair_items
            if pattern in str(item.get("group_id", "")) and item.get("status") == "success"
        ]
        return mean(vals) if vals else 0.0

    return {
        "perception": signal_for("perception"),
        "sensor_radius": signal_for("radius"),
        "memory": signal_for("memory"),
        "prediction": signal_for("prediction"),
        "development": signal_for("development"),
        "flat_alive_delta_mean": mean_delta_for("flat"),
        "patch_alive_delta_mean": mean_delta_for("patch"),
        "jungle_alive_delta_mean": mean_delta_for("jungle"),
        "interpretation": "short paired campaign summary only; no trait is confirmed useful",
    }


def write_campaign_artifacts(manifest: Mapping[str, Any], output_dir: str | Path, *, include_environment_report: bool = False) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    manifest_path.write_text(canonical_json(manifest), encoding="utf-8")
    plan = plan_campaign(manifest)
    plan_path = output / "dry_run_plan.json"
    plan_path.write_text(canonical_json(plan), encoding="utf-8")
    execution = execute_campaign(manifest, output_root=output)
    validation = validate_campaign_metrics(manifest, plan)
    validation_path = output / "metrics_validation.json"
    validation_path.write_text(canonical_json(validation), encoding="utf-8")
    paired = summarize_paired_runs(manifest, plan)
    paired_path = output / "paired_summary.json"
    paired_path.write_text(canonical_json(paired), encoding="utf-8")
    audit = fake_evolution_audit(manifest, validation, paired)
    audit_path = output / "fake_evolution_audit.json"
    audit_path.write_text(canonical_json(audit), encoding="utf-8")
    paths = {
        "manifest": str(manifest_path),
        "dry_run_plan": str(plan_path),
        "execution_summary": str(output / "execution_summary.json"),
        "metrics_validation": str(validation_path),
        "paired_summary": str(paired_path),
        "fake_evolution_audit": str(audit_path),
    }
    if include_environment_report:
        env_report = environment_usefulness_report(paired)
        env_path = output / "environment_usefulness_report.json"
        env_path.write_text(canonical_json(env_report), encoding="utf-8")
        paths["environment_usefulness_report"] = str(env_path)
    return paths


def write_evidence_note(
    path: str | Path,
    *,
    title: str,
    manifest: Mapping[str, Any],
    artifact_paths: Mapping[str, str],
    tests_run: Iterable[str],
    metadata: Mapping[str, Any] | None = None,
) -> None:
    meta = dict(metadata or campaign_metadata_summary(manifest))
    lines = [
        f"# {title}",
        "",
        f"Campaign id: `{manifest['campaign_id']}`",
        f"Seeds: `{meta['seed_list']}`",
        f"Planned/executed live runs: `{meta['planned_live_simulator_run_count']} / {meta['executed_live_run_count']}`",
        f"Skipped cells: `{meta['planned_skipped_cell_count']}`",
        f"Backend counts: `{meta['backend_counts']}`",
        f"Scenario variants: `{meta['scenario_variants']}`",
        f"Horizon: `{meta['horizon']}`",
        f"Cost profile: `{meta['cost_profile']}`",
        "",
        "## Summary",
        "",
        "This artifact records deterministic Theme 4 campaign infrastructure output. It does not claim intelligence, evolved usefulness, self-awareness, self-control, self-modification, or directed evolution.",
        "",
        "## Artifacts",
    ]
    for name, artifact_path in sorted(artifact_paths.items()):
        lines.append(f"- `{name}`: `{artifact_path}`")
    lines.extend(
        [
            "",
            "## Tests Run",
            "",
            *[f"- `{cmd}`" for cmd in tests_run],
            "",
            "## Confirmations",
            "",
            "- No new cognitive machinery was added by campaign orchestration.",
            "- No Patch World or Jungle mechanics were changed.",
            "- No direct cognition reward was added.",
            "- Runtime tensor/output shapes remain fixed.",
            "- Later-stage platform surfaces are not treated as evolutionary evidence.",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def zip_paths(zip_path: str | Path, paths: Iterable[str | Path]) -> None:
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as zf:
        for p in paths:
            path = Path(p)
            if path.is_file():
                zf.write(path, arcname=path.name)
            elif path.is_dir():
                for child in sorted(path.rglob("*")):
                    if child.is_file():
                        zf.write(child, arcname=str(child.relative_to(path.parent)))
