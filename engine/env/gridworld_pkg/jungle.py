from __future__ import annotations

"""Jungle V0 pressure helpers.

Step 9 implements predator-like pressure as a derived deterministic field over
existing Patch World state. It does not add predator entities or observation
channels.
"""

from typing import Any, Dict

import numpy as np

from env.gridworld_pkg.patch_world import (
    _bounded,
    _safe_norm,
    compute_local_density_map,
    mosaic0a_depletion_recovery_enabled,
    normalize_patch_density,
    patch_scent_max,
    patch_visibility_from_cover,
    patch_waste_max,
)


def validate_jungle_config(cfg: Any) -> None:
    """Validate explicit Jungle V0 feature dependencies."""
    if not bool(getattr(cfg, "patch_world_enable", False)):
        if bool(getattr(cfg, "wdl_delayed_hazard_enable", False)):
            raise ValueError("wdl_delayed_hazard_enable requires patch_world_enable")
        if bool(getattr(cfg, "delayed_resource_depletion_enable", False)):
            raise ValueError("delayed_resource_depletion_enable requires patch_world_enable")
        if bool(getattr(cfg, "delayed_crowding_waste_enable", False)):
            raise ValueError("delayed_crowding_waste_enable requires patch_world_enable")
        if bool(getattr(cfg, "jungle_predator_pressure_enable", False)):
            raise ValueError("jungle_predator_pressure_enable requires patch_world_enable")
        if bool(getattr(cfg, "jungle_reproduction_emissions_enable", False)):
            raise ValueError("jungle_reproduction_emissions_enable requires patch_world_enable")
        if bool(getattr(cfg, "jungle_juvenile_predator_vulnerability_enable", False)):
            raise ValueError("jungle_juvenile_predator_vulnerability_enable requires patch_world_enable")
        if bool(getattr(cfg, "jungle_terrain_protection_enable", False)):
            raise ValueError("jungle_terrain_protection_enable requires patch_world_enable")
        if bool(getattr(cfg, "jungle_ecological_recovery_enable", False)):
            raise ValueError("jungle_ecological_recovery_enable requires patch_world_enable")
        if bool(getattr(cfg, "roz2a_patch_niche_enable", False)):
            raise ValueError("roz2a_patch_niche_enable requires patch_world_enable")
        if bool(getattr(cfg, "mosaic0a_depletion_recovery_enable", False)):
            raise ValueError("mosaic0a_depletion_recovery_enable requires patch_world_enable")
        if bool(getattr(cfg, "predrisk_fixed1_enable", False)):
            raise ValueError("predrisk_fixed1_enable requires patch_world_enable")
    if bool(getattr(cfg, "mosaic0a_depletion_recovery_enable", False)):
        if int(getattr(cfg, "mosaic0a_schema_version", 1) or 1) != 1:
            raise ValueError("unsupported mosaic0a_schema_version")
        mode = str(getattr(cfg, "mosaic0a_class_mode", "budget_matched_clustered_v1") or "")
        if mode not in {
            "none",
            "all_burst",
            "all_renewal",
            "checkerboard_burst_renewal",
            "vertical_split_burst_renewal",
            "horizontal_split_burst_renewal",
            "budget_matched_clustered_v1",
        }:
            raise ValueError(f"unsupported mosaic0a_class_mode: {mode}")
        control = str(getattr(cfg, "mosaic0a_control_mode", "real_mosaic") or "")
        if control not in {
            "real_mosaic",
            "patch_uniform",
            "label_only_sham",
            "budget_matched_uniform",
            "depletion_only",
            "recovery_only",
        }:
            raise ValueError(f"unsupported mosaic0a_control_mode: {control}")
        if bool(getattr(cfg, "respec0b_resource_specialization_enable", False)):
            raise ValueError("mosaic0a_depletion_recovery_enable cannot combine with respec0b_resource_specialization_enable")
        if bool(getattr(cfg, "roz2a_patch_niche_enable", False)):
            raise ValueError("mosaic0a_depletion_recovery_enable cannot combine with roz2a_patch_niche_enable")
        mosaic0a_depletion_recovery_enabled(cfg)
    if bool(getattr(cfg, "predrisk_fixed1_enable", False)):
        if int(getattr(cfg, "predrisk_fixed1_schema_version", 1) or 1) != 1:
            raise ValueError("unsupported predrisk_fixed1_schema_version")
        mode = str(getattr(cfg, "predrisk_fixed1_zone_mode", "clustered_quadrants_v1") or "")
        if mode not in {
            "none",
            "all_exposed_rich",
            "all_safe_covered",
            "vertical_split_exposed_safe",
            "checkerboard_exposed_safe",
            "clustered_quadrants_v1",
        }:
            raise ValueError(f"unsupported predrisk_fixed1_zone_mode: {mode}")
        control = str(getattr(cfg, "predrisk_fixed1_control_mode", "correlated_tradeoff") or "")
        if control not in {
            "true_flat",
            "resource_only",
            "predator_only",
            "cover_only",
            "label_only_sham",
            "uncorrelated_risk_resource",
            "budget_matched_uniform",
            "correlated_tradeoff",
        }:
            raise ValueError(f"unsupported predrisk_fixed1_control_mode: {control}")
        if bool(getattr(cfg, "respec0b_resource_specialization_enable", False)):
            raise ValueError("predrisk_fixed1_enable cannot combine with respec0b_resource_specialization_enable")
        if bool(getattr(cfg, "mosaic0a_depletion_recovery_enable", False)):
            raise ValueError("predrisk_fixed1_enable cannot combine with mosaic0a_depletion_recovery_enable")
        if bool(getattr(cfg, "roz2a_patch_niche_enable", False)):
            raise ValueError("predrisk_fixed1_enable cannot combine with roz2a_patch_niche_enable")
        if bool(getattr(cfg, "wdl_delayed_hazard_enable", False)):
            raise ValueError("predrisk_fixed1_enable cannot combine with wdl_delayed_hazard_enable")
    if bool(getattr(cfg, "roz2a_patch_niche_enable", False)):
        mode = str(getattr(cfg, "roz2a_patch_niche_mode", "static_three_zone_tradeoff_v1") or "")
        if mode != "static_three_zone_tradeoff_v1":
            raise ValueError(f"unsupported roz2a_patch_niche_mode: {mode}")
        if int(getattr(cfg, "roz2a_patch_niche_radius", 1) or 1) < 1:
            raise ValueError("roz2a_patch_niche_radius must be >= 1")
    if bool(getattr(cfg, "jungle_juvenile_predator_vulnerability_enable", False)) and not bool(
        getattr(cfg, "jungle_predator_pressure_enable", False)
    ):
        raise ValueError("jungle_juvenile_predator_vulnerability_enable requires jungle_predator_pressure_enable")
    if bool(getattr(cfg, "jungle_terrain_protection_enable", False)) and not bool(
        getattr(cfg, "jungle_predator_pressure_enable", False)
    ):
        raise ValueError("jungle_terrain_protection_enable requires jungle_predator_pressure_enable")
    if bool(getattr(cfg, "wdl_delayed_hazard_enable", False)):
        if int(getattr(cfg, "wdl_delayed_hazard_schema_version", 1) or 1) != 1:
            raise ValueError("unsupported wdl_delayed_hazard_schema_version")
    if bool(getattr(cfg, "delayed_resource_depletion_enable", False)):
        if int(getattr(cfg, "delayed_resource_depletion_schema_version", 1) or 1) != 1:
            raise ValueError("unsupported delayed_resource_depletion_schema_version")
    if bool(getattr(cfg, "delayed_crowding_waste_enable", False)):
        if int(getattr(cfg, "delayed_crowding_waste_schema_version", 1) or 1) != 1:
            raise ValueError("unsupported delayed_crowding_waste_schema_version")


def validate_jungle_predator_config(cfg: Any) -> None:
    """Validate explicit Jungle predator-pressure configuration."""
    validate_jungle_config(cfg)


def jungle_predator_pressure_enabled(cfg: Any) -> bool:
    """Return true only when explicit Jungle predator pressure is enabled."""
    validate_jungle_predator_config(cfg)
    return bool(getattr(cfg, "patch_world_enable", False)) and bool(
        getattr(cfg, "jungle_predator_pressure_enable", False)
    )


def jungle_reproduction_emissions_enabled(cfg: Any) -> bool:
    """Return true only when explicit Jungle birth emissions are enabled."""
    validate_jungle_config(cfg)
    return bool(getattr(cfg, "patch_world_enable", False)) and bool(
        getattr(cfg, "jungle_reproduction_emissions_enable", False)
    )


def jungle_juvenile_predator_vulnerability_enabled(cfg: Any) -> bool:
    """Return true only for explicit juvenile vulnerability under predator pressure."""
    validate_jungle_config(cfg)
    return bool(getattr(cfg, "patch_world_enable", False)) and bool(
        getattr(cfg, "jungle_predator_pressure_enable", False)
    ) and bool(getattr(cfg, "jungle_juvenile_predator_vulnerability_enable", False))


def jungle_terrain_protection_enabled(cfg: Any) -> bool:
    """Return true only for explicit terrain protection under predator pressure."""
    validate_jungle_config(cfg)
    return bool(getattr(cfg, "patch_world_enable", False)) and bool(
        getattr(cfg, "jungle_predator_pressure_enable", False)
    ) and bool(getattr(cfg, "jungle_terrain_protection_enable", False))


def jungle_juvenile_damage_multiplier(cfg: Any, agent: Any) -> tuple[float, bool]:
    """Return bounded predator-damage multiplier and whether the agent is juvenile."""
    juvenile = False
    try:
        age = int(getattr(getattr(agent, "stats", None), "survival_steps", 0) or 0)
        threshold = int(max(0, getattr(cfg, "jungle_juvenile_age_threshold", 10)))
        juvenile = age <= threshold
    except Exception:
        juvenile = False
    if not juvenile or not jungle_juvenile_predator_vulnerability_enabled(cfg):
        return 1.0, bool(juvenile)
    raw = _bounded(getattr(cfg, "jungle_juvenile_predator_damage_multiplier", 1.5), default=1.5, lo=1.0)
    cap = _bounded(getattr(cfg, "jungle_juvenile_vulnerability_cap", 3.0), default=3.0, lo=1.0)
    return float(min(raw, cap)), True


def compute_terrain_damage_multiplier_map(env: Any) -> np.ndarray:
    """Compute bounded cover/shelter predator-damage multipliers."""
    cfg = getattr(env, "cfg", None)
    shape = tuple(np.asarray(getattr(env, "grid")).shape)
    ones = np.ones(shape, dtype=np.float32)
    if not jungle_terrain_protection_enabled(cfg):
        return ones
    if not bool(getattr(cfg, "jungle_terrain_protection_apply_to_damage_enable", True)):
        return ones

    cover = np.ascontiguousarray(
        np.nan_to_num(np.clip(np.asarray(getattr(env, "patch_cover"), dtype=np.float32), 0.0, 1.0)),
        dtype=np.float32,
    )
    shelter = np.ascontiguousarray(
        np.nan_to_num(np.clip(np.asarray(getattr(env, "patch_shelter"), dtype=np.float32), 0.0, 1.0)),
        dtype=np.float32,
    )
    cover_w = np.float32(
        _bounded(getattr(cfg, "jungle_cover_damage_reduction_weight", 0.25), default=0.25, lo=0.0)
    )
    shelter_w = np.float32(
        _bounded(getattr(cfg, "jungle_shelter_damage_reduction_weight", 0.25), default=0.25, lo=0.0)
    )
    min_mult = _bounded(
        getattr(cfg, "jungle_terrain_damage_multiplier_min", 0.25),
        default=0.25,
        lo=0.0,
        hi=1.0,
    )
    mult = np.float32(1.0) - cover * cover_w - shelter * shelter_w
    return np.ascontiguousarray(np.nan_to_num(np.clip(mult, min_mult, 1.0)), dtype=np.float32)


def jungle_terrain_damage_multiplier(env: Any, y: int, x: int) -> tuple[float, float, float]:
    """Return damage multiplier plus clamped cover/shelter at one coordinate."""
    mult = compute_terrain_damage_multiplier_map(env)
    cover = np.asarray(getattr(env, "patch_cover"), dtype=np.float32)
    shelter = np.asarray(getattr(env, "patch_shelter"), dtype=np.float32)
    yy = int(y)
    xx = int(x)
    return (
        float(mult[yy, xx]),
        float(np.nan_to_num(np.clip(cover[yy, xx], 0.0, 1.0))),
        float(np.nan_to_num(np.clip(shelter[yy, xx], 0.0, 1.0))),
    )


def queue_jungle_birth_emission(env: Any, y: int, x: int) -> tuple[float, float]:
    """Queue deterministic scent/waste emissions for a successful birth."""
    cfg = getattr(env, "cfg", None)
    if not jungle_reproduction_emissions_enabled(cfg):
        return 0.0, 0.0
    if not bool(getattr(cfg, "jungle_birth_emission_at_child_cell_enable", True)):
        return 0.0, 0.0

    yy = int(y)
    xx = int(x)
    scent_emit = _bounded(getattr(cfg, "jungle_birth_scent_emit", 0.05), default=0.05, lo=0.0)
    waste_emit = _bounded(getattr(cfg, "jungle_birth_waste_emit", 0.05), default=0.05, lo=0.0)
    if bool(getattr(cfg, "jungle_birth_emission_clamp_enable", True)):
        scent_room = max(0.0, patch_scent_max(cfg) - float(np.asarray(getattr(env, "patch_scent"))[yy, xx]))
        waste_room = max(0.0, patch_waste_max(cfg) - float(np.asarray(getattr(env, "patch_waste"))[yy, xx]))
        scent_emit = min(scent_emit, scent_room)
        waste_emit = min(waste_emit, waste_room)
    if scent_emit > 0.0:
        getattr(env, "patch_scent_emit_pending")[yy, xx] += np.float32(scent_emit)
    if waste_emit > 0.0:
        getattr(env, "patch_waste_emit_pending")[yy, xx] += np.float32(waste_emit)
    return float(scent_emit), float(waste_emit)


def compute_predator_risk_maps(env: Any) -> Dict[str, np.ndarray]:
    """Compute bounded predator-risk and component maps from Patch World state."""
    cfg = getattr(env, "cfg", None)
    shape = tuple(np.asarray(getattr(env, "grid")).shape)
    zeros = np.zeros(shape, dtype=np.float32)
    if not jungle_predator_pressure_enabled(cfg):
        return {
            "risk": zeros,
            "scent_component": zeros,
            "waste_component": zeros,
            "density_component": zeros,
            "visibility_component": zeros,
            "cover_reduction": zeros,
            "shelter_reduction": zeros,
        }

    compute_local_density_map(env)

    scent_norm = _safe_norm(getattr(env, "patch_scent"), patch_scent_max(cfg))
    waste_norm = _safe_norm(getattr(env, "patch_waste"), patch_waste_max(cfg))
    density_norm = normalize_patch_density(cfg, getattr(env, "local_density_map"))
    cover = np.ascontiguousarray(
        np.nan_to_num(np.clip(np.asarray(getattr(env, "patch_cover"), dtype=np.float32), 0.0, 1.0)),
        dtype=np.float32,
    )
    shelter = np.ascontiguousarray(
        np.nan_to_num(np.clip(np.asarray(getattr(env, "patch_shelter"), dtype=np.float32), 0.0, 1.0)),
        dtype=np.float32,
    )
    visibility = np.ascontiguousarray(
        np.nan_to_num(np.clip(patch_visibility_from_cover(cfg, cover), 0.0, 1.0)),
        dtype=np.float32,
    )

    scent_component = scent_norm * np.float32(
        _bounded(getattr(cfg, "jungle_predator_scent_weight", 0.02), default=0.02, lo=0.0)
    )
    waste_component = waste_norm * np.float32(
        _bounded(getattr(cfg, "jungle_predator_waste_weight", 0.02), default=0.02, lo=0.0)
    )
    density_component = density_norm * np.float32(
        _bounded(getattr(cfg, "jungle_predator_density_weight", 0.02), default=0.02, lo=0.0)
    )
    visibility_component = visibility * np.float32(
        _bounded(getattr(cfg, "jungle_predator_visibility_weight", 0.02), default=0.02, lo=0.0)
    )
    cover_reduction = cover * np.float32(
        _bounded(getattr(cfg, "jungle_predator_cover_protection_weight", 0.02), default=0.02, lo=0.0)
    )
    shelter_reduction = shelter * np.float32(
        _bounded(getattr(cfg, "jungle_predator_shelter_protection_weight", 0.02), default=0.02, lo=0.0)
    )

    risk_min = _bounded(getattr(cfg, "jungle_predator_risk_min", 0.0), default=0.0, lo=0.0, hi=1.0)
    risk_cap = _bounded(getattr(cfg, "jungle_predator_risk_cap", 0.05), default=0.05, lo=0.0, hi=1.0)
    if risk_cap < risk_min:
        risk_min, risk_cap = risk_cap, risk_min
    base = np.float32(
        _bounded(getattr(cfg, "jungle_predator_base_risk", 0.0), default=0.0, lo=0.0)
    )
    risk_raw = (
        base
        + scent_component
        + waste_component
        + density_component
        + visibility_component
        - cover_reduction
        - shelter_reduction
    )
    risk = np.ascontiguousarray(np.nan_to_num(np.clip(risk_raw, risk_min, risk_cap)), dtype=np.float32)
    return {
        "risk": risk,
        "scent_component": np.ascontiguousarray(np.nan_to_num(scent_component), dtype=np.float32),
        "waste_component": np.ascontiguousarray(np.nan_to_num(waste_component), dtype=np.float32),
        "density_component": np.ascontiguousarray(np.nan_to_num(density_component), dtype=np.float32),
        "visibility_component": np.ascontiguousarray(np.nan_to_num(visibility_component), dtype=np.float32),
        "cover_reduction": np.ascontiguousarray(np.nan_to_num(cover_reduction), dtype=np.float32),
        "shelter_reduction": np.ascontiguousarray(np.nan_to_num(shelter_reduction), dtype=np.float32),
    }


def jungle_predator_risk_metrics(env: Any) -> Dict[str, float]:
    """Return diagnostic risk/component metrics for the current world state."""
    maps = compute_predator_risk_maps(env)
    risk = np.asarray(maps["risk"], dtype=np.float32)
    occupied = np.asarray(getattr(env, "agent_grid"), dtype=np.int32) != -1

    def _mean(arr: np.ndarray) -> float:
        return float(np.mean(arr)) if arr.size else 0.0

    def _min(arr: np.ndarray) -> float:
        return float(np.min(arr)) if arr.size else 0.0

    def _max(arr: np.ndarray) -> float:
        return float(np.max(arr)) if arr.size else 0.0

    def _p(arr: np.ndarray, pct: float) -> float:
        return float(np.percentile(arr, pct)) if arr.size else 0.0

    def _occ_mean(arr: np.ndarray) -> float:
        return float(np.mean(arr[occupied])) if arr.size and bool(np.any(occupied)) else 0.0

    def _occ_p90(arr: np.ndarray) -> float:
        return float(np.percentile(arr[occupied], 90)) if arr.size and bool(np.any(occupied)) else 0.0

    return {
        "jungle_predator_pressure_enable": int(jungle_predator_pressure_enabled(getattr(env, "cfg", None))),
        "jungle_predator_risk_mean": _mean(risk),
        "jungle_predator_risk_min": _min(risk),
        "jungle_predator_risk_max": _max(risk),
        "jungle_predator_risk_p10": _p(risk, 10),
        "jungle_predator_risk_p90": _p(risk, 90),
        "jungle_predator_occupied_risk_mean": _occ_mean(risk),
        "jungle_predator_occupied_risk_p90": _occ_p90(risk),
        "jungle_predator_scent_component_mean": _mean(maps["scent_component"]),
        "jungle_predator_waste_component_mean": _mean(maps["waste_component"]),
        "jungle_predator_density_component_mean": _mean(maps["density_component"]),
        "jungle_predator_visibility_component_mean": _mean(maps["visibility_component"]),
        "jungle_predator_cover_reduction_mean": _mean(maps["cover_reduction"]),
        "jungle_predator_shelter_reduction_mean": _mean(maps["shelter_reduction"]),
    }
