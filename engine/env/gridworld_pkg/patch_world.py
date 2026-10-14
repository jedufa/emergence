from __future__ import annotations

"""Patch World V0 substrate helpers.

This module provides feature-gated version validation, patch array
initialization, and deterministic food substrate synchronization.
"""

import hashlib
import json
from typing import Any, Dict, Iterable, List

import numpy as np

from agents.resource_handling import resource_handling_capacities
from env.gridworld_pkg.constants import EMPTY, FOOD


PATCH_WORLD_SCHEMA_VERSION = 1
PATCH_WORLD_OBSERVATION_SCHEMA_VERSION = 1
HABACCLIM_OBSERVATION_SCHEMA_VERSION = 1
PATCH_WORLD_DESCRIPTOR_SCHEMA_VERSION = 1
LINEAGE_DIVERSITY_SCHEMA_VERSION = 1
PATCH_WORLD_OBSERVATION_CHANNEL_NAMES = (
    "legacy_food_occupancy",
    "patch_food_amount_norm",
    "patch_cover",
    "patch_shelter",
    "patch_scent_norm",
    "patch_waste_norm",
    "patch_density_norm",
    "patch_movement_cost_norm",
)
WDL_DELAYED_HAZARD_OBSERVATION_CHANNEL_NAMES = (
    "wdl_delayed_hazard_cue_norm",
)
PREDICTION_DEMAND_SCHEMA_VERSION = 1
PREDICTION_DEMAND_OBSERVATION_CHANNEL_NAMES = (
    "prediction_demand_cue_norm",
    "prediction_demand_trend_norm",
)
DELAYED_RESOURCE_WASTE_OBSERVATION_CHANNEL_NAMES = (
    "delayed_resource_depletion_cue_norm",
    "delayed_crowding_waste_cue_norm",
)
MEM_MAX_DELAYED_CHOICE_SCHEMA_VERSION = 1
MEM_MAX_DELAYED_CHOICE_OBSERVATION_SCHEMA_VERSION = 2
MEM_MAX_DELAYED_CHOICE_CUE_OBSERVATION_CHANNEL_NAMES = (
    "mem_max_delayed_choice_target_a_cue_norm",
    "mem_max_delayed_choice_target_b_cue_norm",
)
MEM_MAX_DELAYED_CHOICE_NAVIGATION_OBSERVATION_CHANNEL_NAMES = (
    "mem_max_delayed_choice_target_a_dy_norm",
    "mem_max_delayed_choice_target_a_dx_norm",
    "mem_max_delayed_choice_target_a_distance_norm",
    "mem_max_delayed_choice_target_a_in_target",
    "mem_max_delayed_choice_target_b_dy_norm",
    "mem_max_delayed_choice_target_b_dx_norm",
    "mem_max_delayed_choice_target_b_distance_norm",
    "mem_max_delayed_choice_target_b_in_target",
)
MEM_MAX_DELAYED_CHOICE_OBSERVATION_CHANNEL_NAMES = (
    MEM_MAX_DELAYED_CHOICE_CUE_OBSERVATION_CHANNEL_NAMES
    + MEM_MAX_DELAYED_CHOICE_NAVIGATION_OBSERVATION_CHANNEL_NAMES
)
PATCH_WORLD_ARRAY_NAMES = (
    "patch_food_amount",
    "patch_fertility",
    "patch_cover",
    "patch_shelter",
    "patch_movement_cost",
    "patch_scent",
    "patch_waste",
    "patch_recovery_activity",
    "respec0b_resource_class",
    "mosaic0a_patch_class",
    "predrisk_fixed1_zone_class",
    "corridor_cp0_zone_class",
    "habacclim_habitat_class",
)
RESPEC0B_RESOURCE_NONE = 0
RESPEC0B_RESOURCE_A = 1
RESPEC0B_RESOURCE_B = 2
EED_RESOURCE_IMPACT_OVERLAY_ID = "world.overlay.eed_resource_impact.v1"
EED_RESOURCE_IMPACT_SCHEMA_VERSION = 1
EED_RESOURCE_IMPACT_ARRAY_NAMES = (
    "eed_resource_a_amount",
    "eed_resource_b_amount",
    "eed_resource_a_capacity",
    "eed_resource_b_capacity",
    "eed_resource_a_renewal",
    "eed_resource_b_renewal",
    "eed_resource_a_pending_impact",
    "eed_resource_b_pending_impact",
    "eed_resource_a_impact_load",
    "eed_resource_b_impact_load",
)
MOSAIC0A_PATCH_NONE = 0
MOSAIC0A_PATCH_BURST = 1
MOSAIC0A_PATCH_RENEWAL = 2
PREDRISK_FIXED1_ZONE_NONE = 0
PREDRISK_FIXED1_ZONE_EXPOSED_RICH = 1
PREDRISK_FIXED1_ZONE_SAFE_COVERED = 2
CORRIDOR_CP0_ZONE_NONE = 0
CORRIDOR_CP0_ZONE_HOME_LOCAL = 1
CORRIDOR_CP0_ZONE_CORRIDOR = 2
CORRIDOR_CP0_ZONE_REMOTE_RICH = 3
HABACCLIM_HABITAT_NONE = 0
HABACCLIM_HABITAT_A = 1
HABACCLIM_HABITAT_B = 2
WDL_DELAYED_HAZARD_ARRAY_NAMES = (
    "patch_delayed_hazard_cue",
    "patch_delayed_hazard_timer",
    "patch_delayed_hazard_active",
    "patch_delayed_hazard_intensity",
    "patch_delayed_hazard_age",
    "patch_delayed_hazard_event_id",
    "patch_delayed_hazard_prior_cue",
)
PREDICTION_DEMAND_ARRAY_NAMES = (
    "patch_prediction_demand_cue",
    "patch_prediction_demand_trend",
    "patch_prediction_demand_timer",
    "patch_prediction_demand_active",
    "patch_prediction_demand_intensity",
    "patch_prediction_demand_age",
    "patch_prediction_demand_event_id",
    "patch_prediction_demand_delay",
    "patch_prediction_demand_prior_cue",
    "patch_prediction_demand_transition_sign",
)
DELAYED_RESOURCE_DEPLETION_ARRAY_NAMES = (
    "delayed_resource_depletion_cue",
    "delayed_resource_depletion_timer",
    "delayed_resource_depletion_active",
    "delayed_resource_depletion_intensity",
    "delayed_resource_depletion_age",
    "delayed_resource_depletion_event_id",
    "delayed_resource_depletion_prior_cue",
    "delayed_resource_depletion_depleted_once",
)
DELAYED_CROWDING_WASTE_ARRAY_NAMES = (
    "delayed_crowding_waste_cue",
    "delayed_crowding_waste_timer",
    "delayed_crowding_waste_active",
    "delayed_crowding_waste_intensity",
    "delayed_crowding_waste_age",
    "delayed_crowding_waste_event_id",
    "delayed_crowding_waste_prior_cue",
    "delayed_crowding_waste_waste_applied_once",
)
MEM_MAX_DELAYED_CHOICE_ARRAY_NAMES = (
    "mem_max_delayed_choice_target_a",
    "mem_max_delayed_choice_target_b",
    "mem_max_delayed_choice_cue_a",
    "mem_max_delayed_choice_cue_b",
    "mem_max_delayed_choice_timer",
    "mem_max_delayed_choice_age",
    "mem_max_delayed_choice_active",
    "mem_max_delayed_choice_event_id",
    "mem_max_delayed_choice_correct_target",
    "mem_max_delayed_choice_prior_cue",
    "mem_max_delayed_choice_consequence_applied_once",
)
PATCH_WORLD_CONTEXT_FIELDS = (
    "food_amount",
    "fertility",
    "cover",
    "shelter",
    "visibility",
    "movement_cost",
    "scent",
    "waste",
    "density",
    "recovery_activity",
)

_PATCH_WORLD_REQUIRED_ARRAY_NAMES = (
    "patch_food_amount",
    "patch_fertility",
    "patch_cover",
    "patch_scent",
)


def validate_patch_world_schema_version(version: int) -> int:
    """Validate the configured Patch World schema version."""
    parsed = int(version)
    if parsed != PATCH_WORLD_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported Patch World schema version: {parsed}. "
            f"Supported version: {PATCH_WORLD_SCHEMA_VERSION}."
        )
    return parsed


def validate_patch_world_observation_schema_version(version: int) -> int:
    """Validate the configured Patch World observation schema version."""
    parsed = int(version)
    if parsed != PATCH_WORLD_OBSERVATION_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported Patch World observation schema version: {parsed}. "
            f"Supported version: {PATCH_WORLD_OBSERVATION_SCHEMA_VERSION}."
        )
    return parsed


def validate_mem_max_delayed_choice_observation_schema_version(version: int) -> int:
    parsed = int(version)
    if parsed not in (1, MEM_MAX_DELAYED_CHOICE_OBSERVATION_SCHEMA_VERSION):
        raise ValueError(
            f"Unsupported MEM-MAX delayed-choice observation schema version: {parsed}. "
            f"Supported versions: 1, {MEM_MAX_DELAYED_CHOICE_OBSERVATION_SCHEMA_VERSION}."
        )
    return parsed


def patch_world_observation_enabled(cfg: Any) -> bool:
    """Return true only for the explicit Patch World observation mode."""
    if not bool(getattr(cfg, "patch_world_enable", False)):
        return False
    if not bool(getattr(cfg, "patch_world_observation_enable", False)):
        return False
    validate_patch_world_observation_schema_version(
        int(getattr(cfg, "patch_world_observation_schema_version", 1) or 1)
    )
    return True


def patch_world_local_observation_channel_count(cfg: Any) -> int:
    """Return local channel count including the legacy food/occupancy channel."""
    if not patch_world_observation_enabled(cfg):
        return 1
    return int(
        len(PATCH_WORLD_OBSERVATION_CHANNEL_NAMES)
        + wdl_delayed_hazard_observation_channel_count(cfg)
        + prediction_demand_observation_channel_count(cfg)
        + delayed_resource_waste_observation_channel_count(cfg)
        + mem_max_delayed_choice_observation_channel_count(cfg)
    )


def patch_world_local_observation_size(cfg: Any) -> int:
    r = int(getattr(cfg, "view_radius", 1))
    local_cells = (2 * r + 1) ** 2
    return int(local_cells * patch_world_local_observation_channel_count(cfg))


def patch_world_total_observation_size(cfg: Any) -> int:
    extra = 1 if habacclim_observation_enabled(cfg) else 0
    return int(patch_world_local_observation_size(cfg) + 2 + extra)


def validate_habacclim_observation_schema_version(version: int) -> int:
    parsed = int(version)
    if parsed != HABACCLIM_OBSERVATION_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported HABACCLIM observation schema version: {parsed}. "
            f"Supported version: {HABACCLIM_OBSERVATION_SCHEMA_VERSION}."
        )
    return parsed


def habacclim_enabled(cfg: Any) -> bool:
    return bool(getattr(cfg, "patch_world_enable", False)) and bool(getattr(cfg, "habacclim_enable", False))


def habacclim_observation_enabled(cfg: Any) -> bool:
    if not habacclim_enabled(cfg):
        return False
    if not bool(getattr(cfg, "habacclim_observation_enable", False)):
        return False
    validate_habacclim_observation_schema_version(
        int(getattr(cfg, "habacclim_observation_schema_version", 1) or 1)
    )
    return True


def habacclim_observation_scalar_index(cfg: Any) -> int:
    if not habacclim_observation_enabled(cfg):
        raise ValueError("habacclim observation is disabled")
    return int(patch_world_local_observation_size(cfg) + 2)


def _bounded(value: Any, *, default: float, lo: float | None = None, hi: float | None = None) -> float:
    try:
        out = float(value)
    except Exception:
        out = float(default)
    if lo is not None:
        out = max(float(lo), out)
    if hi is not None:
        out = min(float(hi), out)
    return float(out)


def _safe_norm(arr: Any, denom: float) -> np.ndarray:
    data = np.asarray(arr, dtype=np.float32)
    d = float(denom)
    if d <= 0.0:
        return np.zeros(tuple(data.shape), dtype=np.float32)
    return np.ascontiguousarray(np.nan_to_num(np.clip(data / np.float32(d), 0.0, 1.0)), dtype=np.float32)


def patch_food_amount_max(cfg: Any) -> float:
    return _bounded(getattr(cfg, "patch_food_amount_max", 1.0), default=1.0, lo=0.0)


def patch_food_presence_threshold(cfg: Any) -> float:
    max_amount = patch_food_amount_max(cfg)
    return _bounded(getattr(cfg, "patch_food_presence_threshold", 0.5), default=0.5, lo=0.0, hi=max_amount)


def patch_food_gather_amount(cfg: Any) -> float:
    max_amount = patch_food_amount_max(cfg)
    return _bounded(getattr(cfg, "patch_food_gather_amount", max_amount), default=max_amount, lo=1e-6)


def respec0b_resource_specialization_enabled(cfg: Any) -> bool:
    return bool(getattr(cfg, "patch_world_enable", False)) and bool(
        getattr(cfg, "respec0b_resource_specialization_enable", False)
    )


def eed_resource_impact_enabled(cfg: Any) -> bool:
    """Return true only when the EED resource-impact overlay is selected."""
    enabled = bool(getattr(cfg, "patch_world_enable", False)) and bool(
        getattr(cfg, "eed_resource_impact_enable", False)
    )
    if enabled and not respec0b_resource_specialization_enabled(cfg):
        raise ValueError(
            "eed_resource_impact_enable requires "
            "respec0b_resource_specialization_enable"
        )
    return enabled


def eed_resource_impact_applied(cfg: Any) -> bool:
    return eed_resource_impact_enabled(cfg) and bool(
        getattr(cfg, "eed_resource_impact_apply_enable", False)
    )


def eed_ecology_frozen(cfg: Any) -> bool:
    """Return true when EED ecological stocks are observationally frozen."""
    return eed_resource_impact_enabled(cfg) and bool(
        getattr(cfg, "eed_ecology_frozen_enable", False)
    )


def eed_phenotype_fixed(cfg: Any) -> bool:
    """Return true when EED uses the fixed declared handling phenotype."""
    return eed_resource_impact_enabled(cfg) and bool(
        getattr(cfg, "eed_phenotype_fixed_enable", False)
    )


def eed_fixed_resource_axis(
    cfg: Any,
    *,
    slot: int,
    observed_axis: float,
) -> float:
    if not eed_phenotype_fixed(cfg):
        return float(observed_axis)
    profile = str(getattr(cfg, "eed_phenotype_fixed_axis_profile", "neutral"))
    if profile == "neutral":
        return 0.0
    if profile == "a_high":
        return 1.0
    if profile == "b_high":
        return -1.0
    if profile == "symmetric_mix":
        return 1.0 if int(slot) % 2 == 0 else -1.0
    raise ValueError(f"unsupported EED fixed phenotype profile: {profile!r}")


def _eed_resource_value(
    cfg: Any,
    name: str,
    default: float,
    *,
    hi: float | None = None,
) -> float:
    return _bounded(
        getattr(cfg, name, default),
        default=default,
        lo=0.0,
        hi=hi,
    )


def initialize_eed_resource_impact_arrays(
    cfg: Any,
    *,
    classes: Any,
) -> Dict[str, np.ndarray]:
    """Allocate independent, non-checkpointed reference A/B ecology state."""
    class_arr = np.asarray(classes, dtype=np.int32)
    if not eed_resource_impact_enabled(cfg):
        return {}

    a_mask = class_arr == RESPEC0B_RESOURCE_A
    b_mask = class_arr == RESPEC0B_RESOURCE_B
    a_cap_value = _eed_resource_value(cfg, "eed_resource_a_capacity", 1.0)
    b_cap_value = _eed_resource_value(cfg, "eed_resource_b_capacity", 1.0)
    a_initial = _eed_resource_value(
        cfg,
        "eed_resource_a_initial_amount",
        1.0,
        hi=a_cap_value,
    )
    b_initial = _eed_resource_value(
        cfg,
        "eed_resource_b_initial_amount",
        1.0,
        hi=b_cap_value,
    )
    a_renewal = _eed_resource_value(
        cfg,
        "eed_resource_a_renewal_per_step",
        0.25,
    )
    b_renewal = _eed_resource_value(
        cfg,
        "eed_resource_b_renewal_per_step",
        0.25,
    )
    zeros = np.zeros(tuple(class_arr.shape), dtype=np.float32)
    return {
        "eed_resource_a_amount": np.where(a_mask, a_initial, 0.0).astype(
            np.float32,
            copy=False,
        ),
        "eed_resource_b_amount": np.where(b_mask, b_initial, 0.0).astype(
            np.float32,
            copy=False,
        ),
        "eed_resource_a_capacity": np.where(a_mask, a_cap_value, 0.0).astype(
            np.float32,
            copy=False,
        ),
        "eed_resource_b_capacity": np.where(b_mask, b_cap_value, 0.0).astype(
            np.float32,
            copy=False,
        ),
        "eed_resource_a_renewal": np.where(a_mask, a_renewal, 0.0).astype(
            np.float32,
            copy=False,
        ),
        "eed_resource_b_renewal": np.where(b_mask, b_renewal, 0.0).astype(
            np.float32,
            copy=False,
        ),
        "eed_resource_a_pending_impact": zeros.copy(),
        "eed_resource_b_pending_impact": zeros.copy(),
        "eed_resource_a_impact_load": zeros.copy(),
        "eed_resource_b_impact_load": zeros.copy(),
    }


def validate_eed_resource_impact_arrays(
    arrays: Dict[str, Any],
    *,
    shape: tuple[int, int],
    cfg: Any,
) -> Dict[str, np.ndarray]:
    """Validate the complete conditional EED checkpoint plane."""
    out: Dict[str, np.ndarray] = {}
    for name in EED_RESOURCE_IMPACT_ARRAY_NAMES:
        if name not in arrays:
            raise ValueError(f"EED resource-impact checkpoint missing array: {name}")
        arr = np.asarray(arrays[name], dtype=np.float32)
        if tuple(arr.shape) != tuple(shape):
            raise ValueError(
                f"EED resource-impact array {name} has shape {arr.shape}, "
                f"expected {shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(
                f"EED resource-impact array {name} contains nonfinite values"
            )
        if np.any(arr < 0.0):
            raise ValueError(
                f"EED resource-impact array {name} contains negative values"
            )
        out[name] = np.ascontiguousarray(arr, dtype=np.float32)

    tolerance = np.float32(1.0e-6)
    for label in ("a", "b"):
        amount = out[f"eed_resource_{label}_amount"]
        capacity = out[f"eed_resource_{label}_capacity"]
        if np.any(amount > capacity + tolerance):
            raise ValueError(
                f"EED resource-{label.upper()} amount exceeds capacity"
            )
        load = out[f"eed_resource_{label}_impact_load"]
        max_load = _eed_resource_value(
            cfg,
            "eed_resource_impact_max_load",
            1.0,
        )
        if np.any(load > np.float32(max_load) + tolerance):
            raise ValueError(
                f"EED resource-{label.upper()} impact load exceeds configured maximum"
            )
    return out


def project_eed_resources_to_patch_food(env: Any) -> None:
    """Project the active independent stock into the legacy Patch food view."""
    cfg = getattr(env, "cfg", None)
    if not eed_resource_impact_enabled(cfg):
        return
    classes = np.asarray(getattr(env, "respec0b_resource_class"), dtype=np.int32)
    projected = np.asarray(getattr(env, "patch_food_amount"), dtype=np.float32)
    a_mask = classes == RESPEC0B_RESOURCE_A
    b_mask = classes == RESPEC0B_RESOURCE_B
    projected[a_mask] = np.asarray(
        getattr(env, "eed_resource_a_amount"),
        dtype=np.float32,
    )[a_mask]
    projected[b_mask] = np.asarray(
        getattr(env, "eed_resource_b_amount"),
        dtype=np.float32,
    )[b_mask]
    project_patch_food_to_binary_grid(env)


def initialize_respec0b_resource_class_array(cfg: Any, *, shape: tuple[int, int]) -> np.ndarray:
    mode = str(getattr(cfg, "respec0b_resource_class_mode", "vertical_split_ab") or "vertical_split_ab")
    classes = np.zeros(shape, dtype=np.float32)
    if not respec0b_resource_specialization_enabled(cfg):
        return classes
    _h, w = int(shape[0]), int(shape[1])
    if mode == "all_a":
        classes[:, :] = np.float32(RESPEC0B_RESOURCE_A)
    elif mode == "all_b":
        classes[:, :] = np.float32(RESPEC0B_RESOURCE_B)
    elif mode == "checkerboard_ab":
        yy, xx = np.indices(shape)
        classes[:, :] = np.where((yy + xx) % 2 == 0, RESPEC0B_RESOURCE_A, RESPEC0B_RESOURCE_B).astype(np.float32)
    elif mode == "none":
        classes[:, :] = np.float32(RESPEC0B_RESOURCE_NONE)
    else:
        split = max(1, w // 2)
        classes[:, :split] = np.float32(RESPEC0B_RESOURCE_A)
        classes[:, split:] = np.float32(RESPEC0B_RESOURCE_B)
        if w == 1:
            classes[:, :] = np.float32(RESPEC0B_RESOURCE_A)
    return np.ascontiguousarray(classes, dtype=np.float32)


def respec_n2_capacity_balanced_mixed_enabled(cfg: Any) -> bool:
    return (
        respec0b_resource_specialization_enabled(cfg)
        and bool(getattr(cfg, "respec_n2_capacity_balanced_mixed_enable", False))
        and not bool(getattr(cfg, "respec0b_resource_equalized_sham_enable", False))
        and str(getattr(cfg, "respec0b_resource_class_mode", "") or "") == "vertical_split_ab"
    )


def respec_n2_capacity_balance_contract(cfg: Any) -> dict[str, Any]:
    a_return = _bounded(getattr(cfg, "respec0b_resource_a_return", 0.75), default=0.75, lo=0.0)
    b_return = _bounded(getattr(cfg, "respec0b_resource_b_return", 1.50), default=1.50, lo=0.0)
    a_fertility = _bounded(getattr(cfg, "respec_n2_capacity_balanced_a_fertility", 1.0), default=1.0, lo=0.0)
    b_fertility = _bounded(getattr(cfg, "respec_n2_capacity_balanced_b_fertility", 0.5), default=0.5, lo=0.0)
    return {
        "schema_version": "respec_n2_capacity_balance_contract_v1",
        "formula_version": str(
            getattr(cfg, "respec_n2_capacity_balanced_formula_version", "energy_return_regrowth_balance_v1")
            or "energy_return_regrowth_balance_v1"
        ),
        "resource_a_return": a_return,
        "resource_b_return": b_return,
        "resource_a_fertility": a_fertility,
        "resource_b_fertility": b_fertility,
        "resource_a_energy_regrowth_budget_per_cell": a_return * a_fertility,
        "resource_b_energy_regrowth_budget_per_cell": b_return * b_fertility,
        "return_ratio_b_to_a": (b_return / a_return) if a_return > 0.0 else 0.0,
        "fertility_ratio_a_to_b": (a_fertility / b_fertility) if b_fertility > 0.0 else 0.0,
    }


def apply_respec_n2_capacity_balanced_overlay(cfg: Any, arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    if not respec_n2_capacity_balanced_mixed_enabled(cfg):
        return arrays
    classes = np.asarray(arrays["respec0b_resource_class"], dtype=np.int32)
    fertility = np.asarray(arrays["patch_fertility"], dtype=np.float32)
    a_fertility = np.float32(
        _bounded(getattr(cfg, "respec_n2_capacity_balanced_a_fertility", 1.0), default=1.0, lo=0.0)
    )
    b_fertility = np.float32(
        _bounded(getattr(cfg, "respec_n2_capacity_balanced_b_fertility", 0.5), default=0.5, lo=0.0)
    )
    fertility[classes == RESPEC0B_RESOURCE_A] = a_fertility
    fertility[classes == RESPEC0B_RESOURCE_B] = b_fertility
    arrays["patch_fertility"] = np.ascontiguousarray(fertility, dtype=np.float32)
    return arrays


def respec_n2_capacity_budget_snapshot(
    cfg: Any,
    *,
    classes: Any,
    fertility: Any,
    before: Any,
    after: Any,
    cap: Any,
    base_delta: Any,
    effective_delta: Any,
) -> dict[str, Any]:
    """Return compact genotype-blind opportunity accounting for resource zones."""
    class_arr = np.asarray(classes, dtype=np.int32)
    fert_arr = np.asarray(fertility, dtype=np.float32)
    before_arr = np.asarray(before, dtype=np.float32)
    after_arr = np.asarray(after, dtype=np.float32)
    cap_arr = np.asarray(cap, dtype=np.float32)
    base_arr = np.asarray(base_delta, dtype=np.float32)
    effective_arr = np.asarray(effective_delta, dtype=np.float32)
    threshold = patch_food_presence_threshold(cfg)
    rows: list[dict[str, Any]] = []
    for resource_class, zone in (
        (RESPEC0B_RESOURCE_A, "resource_a_zone"),
        (RESPEC0B_RESOURCE_B, "resource_b_zone"),
    ):
        mask = class_arr == int(resource_class)
        _handling_threshold, resource_return = _respec0b_threshold_return(cfg, int(resource_class))
        effective_added = float(np.sum(effective_arr[mask], dtype=np.float64)) if np.any(mask) else 0.0
        rows.append(
            {
                "zone": zone,
                "resource_class": int(resource_class),
                "cell_count": int(np.sum(mask)) if class_arr.size else 0,
                "fertility_mean": float(np.mean(fert_arr[mask])) if np.any(mask) else 0.0,
                "amount_before": float(np.sum(before_arr[mask], dtype=np.float64)) if np.any(mask) else 0.0,
                "amount_after": float(np.sum(after_arr[mask], dtype=np.float64)) if np.any(mask) else 0.0,
                "cap_total": float(np.sum(cap_arr[mask], dtype=np.float64)) if np.any(mask) else 0.0,
                "available_cells_before": int(np.sum(before_arr[mask] > threshold)) if np.any(mask) else 0,
                "available_cells_after": int(np.sum(after_arr[mask] > threshold)) if np.any(mask) else 0,
                "available_amount_before": float(np.sum(before_arr[mask][before_arr[mask] > threshold], dtype=np.float64))
                if np.any(mask)
                else 0.0,
                "available_amount_after": float(np.sum(after_arr[mask][after_arr[mask] > threshold], dtype=np.float64))
                if np.any(mask)
                else 0.0,
                "base_regrowth_added": float(np.sum(base_arr[mask], dtype=np.float64)) if np.any(mask) else 0.0,
                "effective_regrowth_added": effective_added,
                "energy_opportunity_added": effective_added * float(resource_return),
                "return_per_completion": float(resource_return),
                "stockout_cells_after": int(np.sum(after_arr[mask] <= threshold)) if np.any(mask) else 0,
            }
        )
    return {
        "schema_version": "respec_n2_capacity_budget_snapshot_v1",
        "capacity_balanced_mixed_enabled": bool(respec_n2_capacity_balanced_mixed_enabled(cfg)),
        "contract": respec_n2_capacity_balance_contract(cfg),
        "zones": rows,
    }


def _respec0b_threshold_return(cfg: Any, resource_class: int) -> tuple[float, float]:
    if bool(getattr(cfg, "respec0b_resource_equalized_sham_enable", False)):
        return 1.0, 1.0
    if int(resource_class) == RESPEC0B_RESOURCE_B:
        return (
            _bounded(getattr(cfg, "respec0b_resource_b_handling_threshold", 1.25), default=1.25, lo=1e-6),
            _bounded(getattr(cfg, "respec0b_resource_b_return", 1.50), default=1.50, lo=0.0),
        )
    return (
        _bounded(getattr(cfg, "respec0b_resource_a_handling_threshold", 0.75), default=0.75, lo=1e-6),
        _bounded(getattr(cfg, "respec0b_resource_a_return", 0.75), default=0.75, lo=0.0),
    )


def apply_respec0b_resource_gather(
    env: Any,
    agent_ids: Any,
    actions: Any,
    ys: Any,
    xs: Any,
    food_gained: Any,
    resource_handling_axes: Any = None,
) -> tuple[int, float]:
    """Apply RESPEC-0B resource-class gather through the authoritative Patch World path."""
    cfg = getattr(env, "cfg", None)
    n1_events_enabled = bool(getattr(cfg, "respec_n1_zone_telemetry_enable", False))
    n1_events: list[dict[str, Any]] = []
    if n1_events_enabled:
        try:
            from simulation.respec_n1_zone_telemetry import clear_gather_events
            clear_gather_events(env)
        except Exception:
            setattr(env, "respec_n1_resource_gather_events_last", [])
    legacy_amount = np.asarray(
        getattr(env, "patch_food_amount"),
        dtype=np.float32,
    )
    classes = np.asarray(getattr(env, "respec0b_resource_class"), dtype=np.float32)
    eed_enabled = eed_resource_impact_enabled(cfg)
    eed_apply = eed_resource_impact_applied(cfg)
    eed_frozen = eed_ecology_frozen(cfg)
    eed_a_amount = (
        np.asarray(getattr(env, "eed_resource_a_amount"), dtype=np.float32)
        if eed_enabled
        else None
    )
    eed_b_amount = (
        np.asarray(getattr(env, "eed_resource_b_amount"), dtype=np.float32)
        if eed_enabled
        else None
    )
    threshold_presence = patch_food_presence_threshold(cfg)
    gather_amount = patch_food_gather_amount(cfg)
    ids = np.asarray(agent_ids, dtype=np.int32)
    action_arr = np.asarray(actions, dtype=np.int32)
    axes = (
        np.zeros(int(action_arr.shape[0]), dtype=np.float32)
        if resource_handling_axes is None
        else np.asarray(resource_handling_axes, dtype=np.float32)
    )
    if bool(getattr(cfg, "respec0b_resource_equalized_sham_enable", False)):
        axes = np.zeros_like(axes, dtype=np.float32)

    stats = {
        "attempts": 0,
        "successes": 0,
        "failed_handling": 0,
        "resource_a_successes": 0,
        "resource_b_successes": 0,
        "energy_resource_a": 0.0,
        "energy_resource_b": 0.0,
        "handling_ticks": 0,
        "interruptions": 0,
        "eed_resource_a_pending_impact_queued": 0.0,
        "eed_resource_b_pending_impact_queued": 0.0,
        "eed_ecology_frozen": int(eed_frozen),
        "eed_resource_a_depletion_blocked": 0.0,
        "eed_resource_b_depletion_blocked": 0.0,
        "eed_resource_a_pending_impact_blocked": 0.0,
        "eed_resource_b_pending_impact_blocked": 0.0,
    }
    any_gain = 0
    harvest_total = 0.0

    def _n1_append(
        *,
        aid: int,
        y: int,
        x: int,
        resource_class: int,
        attempt: int = 0,
        available_resource_attempt: int = 0,
        stockout_attempt: int = 0,
        handling_tick: int = 0,
        success: int = 0,
        failed_handling: int = 0,
        interruption: int = 0,
        resource_return: float = 0.0,
    ) -> None:
        if not n1_events_enabled:
            return
        zone = "resource_a_zone" if int(resource_class) == RESPEC0B_RESOURCE_A else (
            "resource_b_zone" if int(resource_class) == RESPEC0B_RESOURCE_B else "neutral_zone"
        )
        n1_events.append(
            {
                "agent_id": int(aid),
                "y": int(y),
                "x": int(x),
                "resource_class": int(resource_class),
                "zone": zone,
                "attempt": int(attempt),
                "available_resource_attempt": int(available_resource_attempt),
                "stockout_attempt": int(stockout_attempt),
                "handling_tick": int(handling_tick),
                "success": int(success),
                "failed_handling": int(failed_handling),
                "interruption": int(interruption),
                "resource_return": float(resource_return),
            }
        )

    for i, action in enumerate(action_arr):
        aid = int(ids[i])
        slot, _slot_token = env._resolve_world_slot(aid, require_alive=True)
        if int(action) != 4:
            if int(env.respec0b_resource_active_class[slot]) != RESPEC0B_RESOURCE_NONE:
                stats["interruptions"] += 1
                _n1_append(
                    aid=aid,
                    y=int(env.respec0b_resource_active_y[slot]),
                    x=int(env.respec0b_resource_active_x[slot]),
                    resource_class=int(env.respec0b_resource_active_class[slot]),
                    interruption=1,
                )
            env.respec0b_resource_active_y[slot] = -1
            env.respec0b_resource_active_x[slot] = -1
            env.respec0b_resource_active_class[slot] = RESPEC0B_RESOURCE_NONE
            env.respec0b_resource_handling_progress[slot] = np.float32(0.0)
            continue
        y = int(ys[i])
        x = int(xs[i])
        resource_class = int(classes[y, x])
        if resource_class not in (RESPEC0B_RESOURCE_A, RESPEC0B_RESOURCE_B):
            if int(env.respec0b_resource_active_class[slot]) != RESPEC0B_RESOURCE_NONE:
                stats["interruptions"] += 1
                _n1_append(
                    aid=aid,
                    y=int(env.respec0b_resource_active_y[slot]),
                    x=int(env.respec0b_resource_active_x[slot]),
                    resource_class=int(env.respec0b_resource_active_class[slot]),
                    interruption=1,
                )
            env.respec0b_resource_active_y[slot] = -1
            env.respec0b_resource_active_x[slot] = -1
            env.respec0b_resource_active_class[slot] = RESPEC0B_RESOURCE_NONE
            env.respec0b_resource_handling_progress[slot] = np.float32(0.0)
            continue
        stats["attempts"] += 1
        if eed_enabled:
            amount = (
                eed_a_amount
                if resource_class == RESPEC0B_RESOURCE_A
                else eed_b_amount
            )
            assert amount is not None
        else:
            amount = legacy_amount
        cell_amount = float(amount[y, x])
        if cell_amount <= threshold_presence:
            _n1_append(
                aid=aid,
                y=y,
                x=x,
                resource_class=resource_class,
                attempt=1,
                stockout_attempt=1,
            )
            env.grid[y, x] = EMPTY
            env.respec0b_resource_active_y[slot] = -1
            env.respec0b_resource_active_x[slot] = -1
            env.respec0b_resource_active_class[slot] = RESPEC0B_RESOURCE_NONE
            env.respec0b_resource_handling_progress[slot] = np.float32(0.0)
            continue
        expressed_axis = eed_fixed_resource_axis(
            cfg,
            slot=slot,
            observed_axis=float(axes[i]),
        )
        capacities = resource_handling_capacities(float(expressed_axis))
        capacity = (
            float(capacities["resource_a_capacity"])
            if resource_class == RESPEC0B_RESOURCE_A
            else float(capacities["resource_b_capacity"])
        )
        threshold, resource_return = _respec0b_threshold_return(cfg, resource_class)
        target_changed = (
            int(env.respec0b_resource_active_y[slot]) != y
            or int(env.respec0b_resource_active_x[slot]) != x
            or int(env.respec0b_resource_active_class[slot]) != resource_class
        )
        if target_changed:
            if int(env.respec0b_resource_active_class[slot]) != RESPEC0B_RESOURCE_NONE:
                stats["interruptions"] += 1
                _n1_append(
                    aid=aid,
                    y=int(env.respec0b_resource_active_y[slot]),
                    x=int(env.respec0b_resource_active_x[slot]),
                    resource_class=int(env.respec0b_resource_active_class[slot]),
                    interruption=1,
                )
            env.respec0b_resource_active_y[slot] = y
            env.respec0b_resource_active_x[slot] = x
            env.respec0b_resource_active_class[slot] = resource_class
            env.respec0b_resource_handling_progress[slot] = np.float32(0.0)
        progress = float(env.respec0b_resource_handling_progress[slot]) + float(capacity)
        env.respec0b_resource_handling_progress[slot] = np.float32(progress)
        stats["handling_ticks"] += 1
        if progress < threshold:
            stats["failed_handling"] += 1
            _n1_append(
                aid=aid,
                y=y,
                x=x,
                resource_class=resource_class,
                attempt=1,
                available_resource_attempt=1,
                handling_tick=1,
                failed_handling=1,
            )
            continue
        harvested = min(cell_amount, float(gather_amount))
        post = max(0.0, cell_amount - harvested)
        if eed_enabled and eed_frozen:
            if resource_class == RESPEC0B_RESOURCE_A:
                stats["eed_resource_a_depletion_blocked"] += float(harvested)
            else:
                stats["eed_resource_b_depletion_blocked"] += float(harvested)
        else:
            amount[y, x] = np.float32(post)
            legacy_amount[y, x] = np.float32(post)
            env.grid[y, x] = FOOD if post > threshold_presence else EMPTY
        env.respec0b_resource_active_y[slot] = -1
        env.respec0b_resource_active_x[slot] = -1
        env.respec0b_resource_active_class[slot] = RESPEC0B_RESOURCE_NONE
        env.respec0b_resource_handling_progress[slot] = np.float32(0.0)
        food_gained[i] = np.float32(resource_return)
        stats["successes"] += 1
        _n1_append(
            aid=aid,
            y=y,
            x=x,
            resource_class=resource_class,
            attempt=1,
            available_resource_attempt=1,
            handling_tick=1,
            success=1,
            resource_return=float(resource_return),
        )
        any_gain = 1
        harvest_total += float(harvested)
        if resource_class == RESPEC0B_RESOURCE_A:
            stats["resource_a_successes"] += 1
            stats["energy_resource_a"] += float(resource_return)
            if eed_apply:
                impact = _eed_resource_value(
                    cfg,
                    "eed_resource_impact_per_success",
                    0.25,
                ) * max(0.0, float(expressed_axis))
                if eed_frozen:
                    stats["eed_resource_a_pending_impact_blocked"] += float(impact)
                else:
                    pending = np.asarray(
                        getattr(env, "eed_resource_a_pending_impact"),
                        dtype=np.float32,
                    )
                    pending[y, x] = np.float32(float(pending[y, x]) + impact)
                    stats["eed_resource_a_pending_impact_queued"] += float(impact)
        else:
            stats["resource_b_successes"] += 1
            stats["energy_resource_b"] += float(resource_return)
            if eed_apply:
                impact = _eed_resource_value(
                    cfg,
                    "eed_resource_impact_per_success",
                    0.25,
                ) * max(0.0, -float(expressed_axis))
                if eed_frozen:
                    stats["eed_resource_b_pending_impact_blocked"] += float(impact)
                else:
                    pending = np.asarray(
                        getattr(env, "eed_resource_b_pending_impact"),
                        dtype=np.float32,
                    )
                    pending[y, x] = np.float32(float(pending[y, x]) + impact)
                    stats["eed_resource_b_pending_impact_queued"] += float(impact)

    setattr(env, "respec0b_resource_gather_stats_last", dict(stats))
    if n1_events_enabled:
        try:
            from simulation.respec_n1_zone_telemetry import record_gather_events
            record_gather_events(
                env,
                agent_ids=agent_ids,
                actions=actions,
                ys=ys,
                xs=xs,
                resource_handling_axes=resource_handling_axes,
                events=n1_events,
            )
        except Exception:
            setattr(env, "respec_n1_resource_gather_events_last", [dict(row) for row in n1_events])
    return int(any_gain), float(harvest_total)


def patch_food_initial_food_cell_value(cfg: Any) -> float:
    food_max = patch_food_amount_max(cfg)
    return _bounded(
        getattr(cfg, "patch_food_amount_initial_food_cell_value", food_max),
        default=food_max,
        lo=0.0,
        hi=food_max,
    )


def patch_food_capacity_array(cfg: Any, fertility: Any, *, shape: tuple[int, int]) -> np.ndarray:
    base = patch_food_amount_max(cfg)
    if bool(getattr(cfg, "patch_food_regrow_cap_by_fertility", False)):
        fert = np.asarray(fertility, dtype=np.float32)
        cap = base * np.clip(fert, 0.0, None)
        return np.ascontiguousarray(cap, dtype=np.float32)
    return np.full(shape, base, dtype=np.float32)


def mosaic0a_depletion_recovery_enabled(cfg: Any) -> bool:
    """Return true only for the explicit MOSAIC-0A depletion/recovery scaffold."""
    return bool(getattr(cfg, "patch_world_enable", False)) and bool(
        getattr(cfg, "mosaic0a_depletion_recovery_enable", False)
    )


def initialize_mosaic0a_patch_class_array(cfg: Any, *, shape: tuple[int, int]) -> np.ndarray:
    """Create the deterministic MOSAIC-0A patch-class map."""
    classes = np.zeros(shape, dtype=np.float32)
    if not mosaic0a_depletion_recovery_enabled(cfg):
        return classes
    mode = str(getattr(cfg, "mosaic0a_class_mode", "budget_matched_clustered_v1") or "budget_matched_clustered_v1")
    h, w = int(shape[0]), int(shape[1])
    if mode == "none":
        return classes
    if mode == "all_burst":
        classes[:, :] = np.float32(MOSAIC0A_PATCH_BURST)
        return np.ascontiguousarray(classes, dtype=np.float32)
    if mode == "all_renewal":
        classes[:, :] = np.float32(MOSAIC0A_PATCH_RENEWAL)
        return np.ascontiguousarray(classes, dtype=np.float32)
    yy, xx = np.indices(shape)
    if mode == "checkerboard_burst_renewal":
        burst_mask = ((yy + xx) % 2) == 0
    elif mode == "vertical_split_burst_renewal":
        split = max(1, w // 2)
        burst_mask = xx < split
    elif mode == "horizontal_split_burst_renewal":
        split = max(1, h // 2)
        burst_mask = yy < split
    elif mode == "budget_matched_clustered_v1":
        block = max(1, int(getattr(cfg, "mosaic0a_cluster_size", 4) or 4))
        burst_mask = (((yy // block) + (xx // block)) % 2) == 0
    else:
        raise ValueError(f"unsupported mosaic0a_class_mode: {mode}")
    classes[:, :] = np.where(burst_mask, MOSAIC0A_PATCH_BURST, MOSAIC0A_PATCH_RENEWAL).astype(np.float32)
    if h * w == 1:
        classes[:, :] = np.float32(MOSAIC0A_PATCH_BURST)
    return np.ascontiguousarray(classes, dtype=np.float32)


def mosaic0a_map_digest(classes: Any) -> Dict[str, Any]:
    """Return a compact deterministic digest for a MOSAIC-0A class map."""
    arr = np.ascontiguousarray(np.asarray(classes, dtype=np.int16))
    return {
        "schema_version": "mosaic0a_map_digest_v1",
        "shape": [int(x) for x in arr.shape],
        "classes_sha256": hashlib.sha256(arr.tobytes(order="C")).hexdigest(),
        "burst_cell_count": int(np.sum(arr == MOSAIC0A_PATCH_BURST)),
        "renewal_cell_count": int(np.sum(arr == MOSAIC0A_PATCH_RENEWAL)),
        "none_cell_count": int(np.sum(arr == MOSAIC0A_PATCH_NONE)),
    }


def mosaic0a_resource_budget_contract(cfg: Any, *, classes: Any | None = None) -> Dict[str, Any]:
    """Return the local resource-budget contract used by MOSAIC-0A controls."""
    if classes is None:
        shape = (int(getattr(cfg, "height", 1) or 1), int(getattr(cfg, "width", 1) or 1))
        class_arr = initialize_mosaic0a_patch_class_array(cfg, shape=shape)
    else:
        class_arr = np.asarray(classes, dtype=np.int32)
    burst_count = int(np.sum(class_arr == MOSAIC0A_PATCH_BURST))
    renewal_count = int(np.sum(class_arr == MOSAIC0A_PATCH_RENEWAL))
    horizon = max(0, int(getattr(cfg, "mosaic0a_budget_horizon_steps", 50) or 50))
    regrow = _bounded(getattr(cfg, "patch_food_regrow_per_step", 0.02), default=0.02, lo=0.0)
    burst_initial = _bounded(getattr(cfg, "mosaic0a_burst_initial_food", 1.0), default=1.0, lo=0.0)
    renewal_initial = _bounded(getattr(cfg, "mosaic0a_renewal_initial_food", 0.4), default=0.4, lo=0.0)
    burst_fertility = _bounded(getattr(cfg, "mosaic0a_burst_fertility", 0.4), default=0.4, lo=0.0)
    renewal_fertility = _bounded(getattr(cfg, "mosaic0a_renewal_fertility", 1.0), default=1.0, lo=0.0)
    burst_per_cell = burst_initial + (regrow * burst_fertility * horizon)
    renewal_per_cell = renewal_initial + (regrow * renewal_fertility * horizon)
    burst_total = burst_per_cell * burst_count
    renewal_total = renewal_per_cell * renewal_count
    diff = abs(burst_total - renewal_total)
    denom = max(abs(burst_total), abs(renewal_total), 1e-9)
    return {
        "schema_version": "mosaic0a_resource_budget_contract_v1",
        "formula_version": "initial_stock_plus_fertility_regrowth_horizon_v1",
        "budget_horizon_steps": horizon,
        "patch_food_regrow_per_step": regrow,
        "burst_cell_count": burst_count,
        "renewal_cell_count": renewal_count,
        "burst_initial_food": burst_initial,
        "renewal_initial_food": renewal_initial,
        "burst_fertility": burst_fertility,
        "renewal_fertility": renewal_fertility,
        "burst_budget_per_cell": burst_per_cell,
        "renewal_budget_per_cell": renewal_per_cell,
        "burst_total_budget": burst_total,
        "renewal_total_budget": renewal_total,
        "relative_total_budget_difference": diff / denom,
        "budget_matched": bool(diff / denom <= 1e-6),
    }


def predrisk_fixed1_enabled(cfg: Any) -> bool:
    """Return true only for the default-off PREDRISK-FIXED-1 scaffold."""
    return bool(getattr(cfg, "patch_world_enable", False)) and bool(getattr(cfg, "predrisk_fixed1_enable", False))


def corridor_cp0_enabled(cfg: Any) -> bool:
    """Return true only for the default-off CORRIDOR-CP0 scaffold."""
    return bool(getattr(cfg, "patch_world_enable", False)) and bool(getattr(cfg, "corridor_cp0_enable", False))


def corridor_hc1_homing_cue_enabled(cfg: Any) -> bool:
    """Return true only for the default-off CORRIDOR-HC1 homing cue."""
    return corridor_cp0_enabled(cfg) and bool(getattr(cfg, "corridor_hc1_homing_cue_enable", False))


def initialize_habacclim_habitat_class_array(cfg: Any, *, shape: tuple[int, int]) -> np.ndarray:
    """Create a hidden deterministic A/B habitat map for HABACCLIM mechanics."""
    classes = np.zeros(shape, dtype=np.float32)
    if not habacclim_enabled(cfg):
        return classes
    mode = str(getattr(cfg, "habacclim_habitat_mode", "interleaved_bands_v1") or "interleaved_bands_v1")
    h, w = int(shape[0]), int(shape[1])
    yy, xx = np.indices(shape)
    if mode == "none":
        return classes
    if mode == "vertical_split_ab":
        a_mask = xx < max(1, w // 2)
    elif mode == "checkerboard_ab":
        a_mask = ((yy + xx) % 2) == 0
    elif mode == "interleaved_bands_v1":
        band = max(1, int(getattr(cfg, "habacclim_band_width", 4) or 4))
        a_mask = ((xx // band) % 2) == 0
    else:
        raise ValueError(f"unsupported habacclim_habitat_mode: {mode}")
    classes[:, :] = np.where(a_mask, HABACCLIM_HABITAT_A, HABACCLIM_HABITAT_B).astype(np.float32)
    if h * w == 1:
        classes[:, :] = np.float32(HABACCLIM_HABITAT_A)
    return np.ascontiguousarray(classes, dtype=np.float32)


def initialize_corridor_cp0_zone_class_array(cfg: Any, *, shape: tuple[int, int]) -> np.ndarray:
    """Create a hidden home/corridor/remote central-place zone map."""
    classes = np.zeros(shape, dtype=np.float32)
    if not corridor_cp0_enabled(cfg):
        return classes
    mode = str(getattr(cfg, "corridor_cp0_zone_mode", "left_home_mid_corridor_right_remote_v1") or "left_home_mid_corridor_right_remote_v1")
    if mode == "none":
        return classes
    if mode != "left_home_mid_corridor_right_remote_v1":
        raise ValueError(f"unsupported corridor_cp0_zone_mode: {mode}")
    h, w = int(shape[0]), int(shape[1])
    home_width = int(getattr(cfg, "corridor_cp0_home_width", 0) or 0)
    remote_width = int(getattr(cfg, "corridor_cp0_remote_width", 0) or 0)
    if home_width <= 0:
        home_width = max(1, w // 5)
    if remote_width <= 0:
        remote_width = max(1, w // 5)
    home_width = max(1, min(w, home_width))
    remote_width = max(1, min(w, remote_width))
    corridor_half_width = max(0, int(getattr(cfg, "corridor_cp0_corridor_half_width", 1) or 0))
    mid_y = h // 2
    y0 = max(0, mid_y - corridor_half_width)
    y1 = min(h, mid_y + corridor_half_width + 1)
    classes[:, :home_width] = np.float32(CORRIDOR_CP0_ZONE_HOME_LOCAL)
    classes[:, max(0, w - remote_width) :] = np.float32(CORRIDOR_CP0_ZONE_REMOTE_RICH)
    if home_width < max(0, w - remote_width):
        classes[y0:y1, home_width : max(0, w - remote_width)] = np.float32(CORRIDOR_CP0_ZONE_CORRIDOR)
    return np.ascontiguousarray(classes, dtype=np.float32)


def initialize_predrisk_fixed1_zone_class_array(cfg: Any, *, shape: tuple[int, int]) -> np.ndarray:
    """Create a deterministic hidden exposed-rich/safe-covered zone map."""
    classes = np.zeros(shape, dtype=np.float32)
    if not predrisk_fixed1_enabled(cfg):
        return classes
    mode = str(getattr(cfg, "predrisk_fixed1_zone_mode", "clustered_quadrants_v1") or "clustered_quadrants_v1")
    h, w = int(shape[0]), int(shape[1])
    yy, xx = np.indices(shape)
    if mode == "none":
        return classes
    if mode == "all_exposed_rich":
        classes[:, :] = np.float32(PREDRISK_FIXED1_ZONE_EXPOSED_RICH)
        return np.ascontiguousarray(classes, dtype=np.float32)
    if mode == "all_safe_covered":
        classes[:, :] = np.float32(PREDRISK_FIXED1_ZONE_SAFE_COVERED)
        return np.ascontiguousarray(classes, dtype=np.float32)
    if mode == "vertical_split_exposed_safe":
        exposed = xx < max(1, w // 2)
    elif mode == "checkerboard_exposed_safe":
        exposed = ((yy + xx) % 2) == 0
    elif mode == "clustered_quadrants_v1":
        block = max(1, int(getattr(cfg, "predrisk_fixed1_cluster_size", 4) or 4))
        exposed = (((yy // block) + (xx // block)) % 2) == 0
    else:
        raise ValueError(f"unsupported predrisk_fixed1_zone_mode: {mode}")
    classes[:, :] = np.where(
        exposed,
        PREDRISK_FIXED1_ZONE_EXPOSED_RICH,
        PREDRISK_FIXED1_ZONE_SAFE_COVERED,
    ).astype(np.float32)
    if h * w == 1:
        classes[:, :] = np.float32(PREDRISK_FIXED1_ZONE_EXPOSED_RICH)
    return np.ascontiguousarray(classes, dtype=np.float32)


def predrisk_fixed1_map_digest(classes: Any) -> Dict[str, Any]:
    """Return a compact deterministic digest for the PREDRISK-FIXED-1 zone map."""
    arr = np.ascontiguousarray(np.asarray(classes, dtype=np.int16))
    return {
        "schema_version": "predrisk_fixed1_map_digest_v1",
        "shape": [int(x) for x in arr.shape],
        "classes_sha256": hashlib.sha256(arr.tobytes(order="C")).hexdigest(),
        "exposed_rich_cell_count": int(np.sum(arr == PREDRISK_FIXED1_ZONE_EXPOSED_RICH)),
        "safe_covered_cell_count": int(np.sum(arr == PREDRISK_FIXED1_ZONE_SAFE_COVERED)),
        "none_cell_count": int(np.sum(arr == PREDRISK_FIXED1_ZONE_NONE)),
    }


def predrisk_fixed1_resource_risk_budget_contract(cfg: Any, *, classes: Any | None = None) -> Dict[str, Any]:
    """Return the frozen analytic resource-risk budget for PREDRISK-FIXED-1."""
    if classes is None:
        shape = (int(getattr(cfg, "height", 1) or 1), int(getattr(cfg, "width", 1) or 1))
        class_arr = initialize_predrisk_fixed1_zone_class_array(cfg, shape=shape)
    else:
        class_arr = np.asarray(classes, dtype=np.int32)
    exposed_count = int(np.sum(class_arr == PREDRISK_FIXED1_ZONE_EXPOSED_RICH))
    safe_count = int(np.sum(class_arr == PREDRISK_FIXED1_ZONE_SAFE_COVERED))
    horizon = max(1, int(getattr(cfg, "predrisk_fixed1_budget_horizon_steps", 80) or 80))
    regrow = _bounded(getattr(cfg, "patch_food_regrow_per_step", 0.02), default=0.02, lo=0.0)
    food_energy = _bounded(getattr(cfg, "food_energy", 9.0), default=9.0, lo=0.0)
    damage_cost = _bounded(getattr(cfg, "jungle_predator_health_damage", 0.25), default=0.25, lo=0.0)
    exposed_food = _bounded(getattr(cfg, "predrisk_fixed1_exposed_food", 1.0), default=1.0, lo=0.0)
    safe_food = _bounded(getattr(cfg, "predrisk_fixed1_safe_food", 0.45), default=0.45, lo=0.0)
    exposed_fertility = _bounded(getattr(cfg, "predrisk_fixed1_exposed_fertility", 0.85), default=0.85, lo=0.0)
    safe_fertility = _bounded(getattr(cfg, "predrisk_fixed1_safe_fertility", 0.55), default=0.55, lo=0.0)
    exposed_move = _bounded(getattr(cfg, "predrisk_fixed1_exposed_movement_cost", 1.15), default=1.15, lo=0.0)
    safe_move = _bounded(getattr(cfg, "predrisk_fixed1_safe_movement_cost", 0.90), default=0.90, lo=0.0)
    exposed_risk = _bounded(getattr(cfg, "jungle_predator_risk_cap", 0.05), default=0.05, lo=0.0, hi=1.0)
    safe_risk = _bounded(getattr(cfg, "jungle_predator_risk_min", 0.0), default=0.0, lo=0.0, hi=1.0)
    exposed_food_return = (exposed_food + regrow * exposed_fertility * horizon) * food_energy
    safe_food_return = (safe_food + regrow * safe_fertility * horizon) * food_energy
    exposed_damage_cost = exposed_risk * damage_cost * food_energy * horizon
    safe_damage_cost = safe_risk * damage_cost * food_energy * horizon
    exposed_risk_adjusted = exposed_food_return - exposed_damage_cost - exposed_move
    safe_risk_adjusted = safe_food_return - safe_damage_cost - safe_move
    return {
        "schema_version": "predrisk_fixed1_resource_risk_budget_v1",
        "formula": "risk_adjusted_opportunity = expected_food_return - expected_predator_damage_cost - movement_cost - crowding_cost",
        "budget_horizon_steps": horizon,
        "exposed_rich_cell_count": exposed_count,
        "safe_covered_cell_count": safe_count,
        "exposed_food_return": float(exposed_food_return),
        "safe_food_return": float(safe_food_return),
        "exposed_expected_damage_cost": float(exposed_damage_cost),
        "safe_expected_damage_cost": float(safe_damage_cost),
        "exposed_movement_cost": float(exposed_move),
        "safe_movement_cost": float(safe_move),
        "exposed_risk_adjusted_opportunity": float(exposed_risk_adjusted),
        "safe_risk_adjusted_opportunity": float(safe_risk_adjusted),
        "exposed_has_greater_food_opportunity": bool(exposed_food_return > safe_food_return),
        "safe_has_lower_expected_damage": bool(safe_damage_cost < exposed_damage_cost),
        "no_zone_dominates_both": bool(exposed_food_return > safe_food_return and safe_damage_cost < exposed_damage_cost),
        "coefficient_source": "frozen_source_derived_defaults_no_campaign_tuning",
    }


def habacclim_acclimation_rate(cfg: Any) -> float:
    n90 = max(1, int(getattr(cfg, "habacclim_acclimation_n90_ticks", 12) or 12))
    return float(1.0 - 0.1 ** (1.0 / float(n90)))


def habacclim_map_digest(classes: Any) -> Dict[str, Any]:
    arr = np.ascontiguousarray(np.asarray(classes, dtype=np.int16))
    return {
        "schema_version": "habacclim_map_digest_v1",
        "shape": [int(x) for x in arr.shape],
        "classes_sha256": hashlib.sha256(arr.tobytes(order="C")).hexdigest(),
        "habitat_a_cell_count": int(np.sum(arr == HABACCLIM_HABITAT_A)),
        "habitat_b_cell_count": int(np.sum(arr == HABACCLIM_HABITAT_B)),
        "none_cell_count": int(np.sum(arr == HABACCLIM_HABITAT_NONE)),
    }


def apply_habacclim_overlay(cfg: Any, arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Apply the default-off HABACCLIM habitat contrast scaffold."""
    if not habacclim_enabled(cfg):
        return arrays
    control = str(getattr(cfg, "habacclim_control_mode", "acclimation_tradeoff") or "acclimation_tradeoff")
    classes = np.asarray(arrays["habacclim_habitat_class"], dtype=np.int32)
    a_mask = classes == HABACCLIM_HABITAT_A
    b_mask = classes == HABACCLIM_HABITAT_B
    both = a_mask | b_mask
    if not bool(np.any(both)):
        return arrays

    food_max = patch_food_amount_max(cfg)
    move_min = _bounded(getattr(cfg, "patch_movement_cost_min", 0.0), default=0.0)
    move_max = _bounded(getattr(cfg, "patch_movement_cost_max", 4.0), default=4.0)
    scent_max = patch_scent_max(cfg)
    waste_max = patch_waste_max(cfg)
    a_food = _bounded(getattr(cfg, "habacclim_a_food", 1.0), default=1.0, lo=0.0, hi=food_max)
    b_food = _bounded(getattr(cfg, "habacclim_b_food", 0.45), default=0.45, lo=0.0, hi=food_max)
    a_fert = _bounded(getattr(cfg, "habacclim_a_fertility", 0.35), default=0.35, lo=0.0)
    b_fert = _bounded(getattr(cfg, "habacclim_b_fertility", 1.0), default=1.0, lo=0.0)
    a_cover = _bounded(getattr(cfg, "habacclim_a_cover", 0.08), default=0.08, lo=0.0, hi=1.0)
    b_cover = _bounded(getattr(cfg, "habacclim_b_cover", 0.80), default=0.80, lo=0.0, hi=1.0)
    a_shelter = _bounded(getattr(cfg, "habacclim_a_shelter", 0.05), default=0.05, lo=0.0, hi=1.0)
    b_shelter = _bounded(getattr(cfg, "habacclim_b_shelter", 0.85), default=0.85, lo=0.0, hi=1.0)
    a_move = _bounded(getattr(cfg, "habacclim_a_movement_cost", 1.10), default=1.10, lo=move_min, hi=move_max)
    b_move = _bounded(getattr(cfg, "habacclim_b_movement_cost", 0.90), default=0.90, lo=move_min, hi=move_max)
    a_scent = _bounded(getattr(cfg, "habacclim_a_scent", 0.70), default=0.70, lo=0.0, hi=scent_max)
    b_scent = _bounded(getattr(cfg, "habacclim_b_scent", 0.08), default=0.08, lo=0.0, hi=scent_max)
    a_waste = _bounded(getattr(cfg, "habacclim_a_waste", 0.25), default=0.25, lo=0.0, hi=waste_max)
    b_waste = _bounded(getattr(cfg, "habacclim_b_waste", 0.05), default=0.05, lo=0.0, hi=waste_max)

    uniform_food = np.float32((a_food * int(np.sum(a_mask)) + b_food * int(np.sum(b_mask))) / max(1, int(np.sum(both))))
    uniform_fert = np.float32((a_fert + b_fert) / 2.0)
    uniform_cover = np.float32((a_cover + b_cover) / 2.0)
    uniform_shelter = np.float32((a_shelter + b_shelter) / 2.0)
    uniform_move = np.float32((a_move + b_move) / 2.0)
    uniform_scent = np.float32((a_scent + b_scent) / 2.0)
    uniform_waste = np.float32((a_waste + b_waste) / 2.0)

    if control == "true_flat":
        arrays["habacclim_habitat_class"][both] = np.float32(HABACCLIM_HABITAT_NONE)
    elif control in {"habitat_labels_only", "acclimation_state_only", "zero_lag_control"}:
        arrays["patch_food_amount"][both] = uniform_food
        arrays["patch_fertility"][both] = uniform_fert
        arrays["patch_cover"][both] = uniform_cover
        arrays["patch_shelter"][both] = uniform_shelter
        arrays["patch_movement_cost"][both] = uniform_move
        arrays["patch_scent"][both] = uniform_scent
        arrays["patch_waste"][both] = uniform_waste
    elif control in {"budget_matched_uniform", "food_contrast_only"}:
        arrays["patch_food_amount"][a_mask] = np.float32(a_food)
        arrays["patch_food_amount"][b_mask] = np.float32(b_food)
        arrays["patch_fertility"][a_mask] = np.float32(a_fert)
        arrays["patch_fertility"][b_mask] = np.float32(b_fert)
        arrays["patch_cover"][both] = uniform_cover
        arrays["patch_shelter"][both] = uniform_shelter
        arrays["patch_movement_cost"][both] = uniform_move
        arrays["patch_scent"][both] = uniform_scent
        arrays["patch_waste"][both] = uniform_waste
        if control == "budget_matched_uniform":
            arrays["habacclim_habitat_class"][both] = np.float32(HABACCLIM_HABITAT_NONE)
    elif control == "uncorrelated_habitat_cost":
        arrays["patch_food_amount"][a_mask] = np.float32(a_food)
        arrays["patch_food_amount"][b_mask] = np.float32(b_food)
        arrays["patch_fertility"][a_mask] = np.float32(a_fert)
        arrays["patch_fertility"][b_mask] = np.float32(b_fert)
        arrays["patch_cover"][a_mask] = np.float32(b_cover)
        arrays["patch_cover"][b_mask] = np.float32(a_cover)
        arrays["patch_shelter"][a_mask] = np.float32(b_shelter)
        arrays["patch_shelter"][b_mask] = np.float32(a_shelter)
        arrays["patch_movement_cost"][a_mask] = np.float32(b_move)
        arrays["patch_movement_cost"][b_mask] = np.float32(a_move)
        arrays["patch_scent"][a_mask] = np.float32(b_scent)
        arrays["patch_scent"][b_mask] = np.float32(a_scent)
        arrays["patch_waste"][a_mask] = np.float32(b_waste)
        arrays["patch_waste"][b_mask] = np.float32(a_waste)
    elif control == "acclimation_tradeoff":
        arrays["patch_food_amount"][a_mask] = np.float32(a_food)
        arrays["patch_food_amount"][b_mask] = np.float32(b_food)
        arrays["patch_fertility"][a_mask] = np.float32(a_fert)
        arrays["patch_fertility"][b_mask] = np.float32(b_fert)
        arrays["patch_cover"][a_mask] = np.float32(a_cover)
        arrays["patch_cover"][b_mask] = np.float32(b_cover)
        arrays["patch_shelter"][a_mask] = np.float32(a_shelter)
        arrays["patch_shelter"][b_mask] = np.float32(b_shelter)
        arrays["patch_movement_cost"][a_mask] = np.float32(a_move)
        arrays["patch_movement_cost"][b_mask] = np.float32(b_move)
        arrays["patch_scent"][a_mask] = np.float32(a_scent)
        arrays["patch_scent"][b_mask] = np.float32(b_scent)
        arrays["patch_waste"][a_mask] = np.float32(a_waste)
        arrays["patch_waste"][b_mask] = np.float32(b_waste)
    else:
        raise ValueError(f"unsupported habacclim_control_mode: {control}")

    for name in (
        "patch_food_amount",
        "patch_fertility",
        "patch_cover",
        "patch_shelter",
        "patch_movement_cost",
        "patch_scent",
        "patch_waste",
        "habacclim_habitat_class",
    ):
        arrays[name] = np.ascontiguousarray(np.nan_to_num(arrays[name]), dtype=np.float32)
    return arrays


def apply_predrisk_fixed1_overlay(cfg: Any, arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Apply the default-off PREDRISK-FIXED-1 predator-resource scaffold."""
    if not predrisk_fixed1_enabled(cfg):
        return arrays
    control = str(getattr(cfg, "predrisk_fixed1_control_mode", "correlated_tradeoff") or "correlated_tradeoff")
    classes = np.asarray(arrays["predrisk_fixed1_zone_class"], dtype=np.int32)
    exposed = classes == PREDRISK_FIXED1_ZONE_EXPOSED_RICH
    safe = classes == PREDRISK_FIXED1_ZONE_SAFE_COVERED
    both = exposed | safe
    if not bool(np.any(both)):
        return arrays

    food_max = patch_food_amount_max(cfg)
    move_min = _bounded(getattr(cfg, "patch_movement_cost_min", 0.0), default=0.0)
    move_max = _bounded(getattr(cfg, "patch_movement_cost_max", 4.0), default=4.0)
    scent_max = patch_scent_max(cfg)
    waste_max = patch_waste_max(cfg)
    e_food = _bounded(getattr(cfg, "predrisk_fixed1_exposed_food", 1.0), default=1.0, lo=0.0, hi=food_max)
    s_food = _bounded(getattr(cfg, "predrisk_fixed1_safe_food", 0.45), default=0.45, lo=0.0, hi=food_max)
    e_fert = _bounded(getattr(cfg, "predrisk_fixed1_exposed_fertility", 0.85), default=0.85, lo=0.0)
    s_fert = _bounded(getattr(cfg, "predrisk_fixed1_safe_fertility", 0.55), default=0.55, lo=0.0)
    e_cover = _bounded(getattr(cfg, "predrisk_fixed1_exposed_cover", 0.05), default=0.05, lo=0.0, hi=1.0)
    s_cover = _bounded(getattr(cfg, "predrisk_fixed1_safe_cover", 0.85), default=0.85, lo=0.0, hi=1.0)
    e_shelter = _bounded(getattr(cfg, "predrisk_fixed1_exposed_shelter", 0.05), default=0.05, lo=0.0, hi=1.0)
    s_shelter = _bounded(getattr(cfg, "predrisk_fixed1_safe_shelter", 0.80), default=0.80, lo=0.0, hi=1.0)
    e_move = _bounded(getattr(cfg, "predrisk_fixed1_exposed_movement_cost", 1.15), default=1.15, lo=move_min, hi=move_max)
    s_move = _bounded(getattr(cfg, "predrisk_fixed1_safe_movement_cost", 0.90), default=0.90, lo=move_min, hi=move_max)
    e_scent = _bounded(getattr(cfg, "predrisk_fixed1_exposed_scent", 0.75), default=0.75, lo=0.0, hi=scent_max)
    s_scent = _bounded(getattr(cfg, "predrisk_fixed1_safe_scent", 0.05), default=0.05, lo=0.0, hi=scent_max)
    e_waste = _bounded(getattr(cfg, "predrisk_fixed1_exposed_waste", 0.25), default=0.25, lo=0.0, hi=waste_max)
    s_waste = _bounded(getattr(cfg, "predrisk_fixed1_safe_waste", 0.05), default=0.05, lo=0.0, hi=waste_max)

    uniform_food = np.float32((e_food * int(np.sum(exposed)) + s_food * int(np.sum(safe))) / max(1, int(np.sum(both))))
    uniform_fert = np.float32((e_fert + s_fert) / 2.0)
    uniform_cover = np.float32((e_cover + s_cover) / 2.0)
    uniform_shelter = np.float32((e_shelter + s_shelter) / 2.0)
    uniform_move = np.float32((e_move + s_move) / 2.0)
    uniform_scent = np.float32((e_scent + s_scent) / 2.0)
    uniform_waste = np.float32((e_waste + s_waste) / 2.0)

    if control == "true_flat":
        return arrays
    if control == "label_only_sham":
        arrays["patch_food_amount"][both] = uniform_food
        arrays["patch_fertility"][both] = uniform_fert
        arrays["patch_cover"][both] = uniform_cover
        arrays["patch_shelter"][both] = uniform_shelter
        arrays["patch_movement_cost"][both] = uniform_move
        arrays["patch_scent"][both] = uniform_scent
        arrays["patch_waste"][both] = uniform_waste
    elif control == "budget_matched_uniform":
        arrays["patch_food_amount"][both] = uniform_food
        arrays["patch_fertility"][both] = uniform_fert
        arrays["patch_cover"][both] = uniform_cover
        arrays["patch_shelter"][both] = uniform_shelter
        arrays["patch_movement_cost"][both] = uniform_move
        arrays["patch_scent"][both] = uniform_scent
        arrays["patch_waste"][both] = uniform_waste
        arrays["predrisk_fixed1_zone_class"][both] = np.float32(PREDRISK_FIXED1_ZONE_NONE)
    elif control == "resource_only":
        arrays["patch_food_amount"][exposed] = np.float32(e_food)
        arrays["patch_food_amount"][safe] = np.float32(s_food)
        arrays["patch_fertility"][exposed] = np.float32(e_fert)
        arrays["patch_fertility"][safe] = np.float32(s_fert)
        arrays["patch_cover"][both] = uniform_cover
        arrays["patch_shelter"][both] = uniform_shelter
        arrays["patch_scent"][both] = uniform_scent
        arrays["patch_waste"][both] = uniform_waste
    elif control == "predator_only":
        arrays["patch_food_amount"][both] = uniform_food
        arrays["patch_fertility"][both] = uniform_fert
        arrays["patch_cover"][exposed] = np.float32(e_cover)
        arrays["patch_cover"][safe] = np.float32(s_cover)
        arrays["patch_shelter"][exposed] = np.float32(e_shelter)
        arrays["patch_shelter"][safe] = np.float32(s_shelter)
        arrays["patch_scent"][exposed] = np.float32(e_scent)
        arrays["patch_scent"][safe] = np.float32(s_scent)
        arrays["patch_waste"][exposed] = np.float32(e_waste)
        arrays["patch_waste"][safe] = np.float32(s_waste)
    elif control == "cover_only":
        arrays["patch_food_amount"][both] = uniform_food
        arrays["patch_fertility"][both] = uniform_fert
        arrays["patch_cover"][exposed] = np.float32(e_cover)
        arrays["patch_cover"][safe] = np.float32(s_cover)
        arrays["patch_shelter"][exposed] = np.float32(e_shelter)
        arrays["patch_shelter"][safe] = np.float32(s_shelter)
        arrays["patch_scent"][both] = uniform_scent
        arrays["patch_waste"][both] = uniform_waste
    elif control == "uncorrelated_risk_resource":
        arrays["patch_food_amount"][exposed] = np.float32(e_food)
        arrays["patch_food_amount"][safe] = np.float32(s_food)
        arrays["patch_fertility"][exposed] = np.float32(e_fert)
        arrays["patch_fertility"][safe] = np.float32(s_fert)
        arrays["patch_cover"][exposed] = np.float32(s_cover)
        arrays["patch_cover"][safe] = np.float32(e_cover)
        arrays["patch_shelter"][exposed] = np.float32(s_shelter)
        arrays["patch_shelter"][safe] = np.float32(e_shelter)
        arrays["patch_scent"][exposed] = np.float32(s_scent)
        arrays["patch_scent"][safe] = np.float32(e_scent)
        arrays["patch_waste"][exposed] = np.float32(s_waste)
        arrays["patch_waste"][safe] = np.float32(e_waste)
    elif control == "correlated_tradeoff":
        arrays["patch_food_amount"][exposed] = np.float32(e_food)
        arrays["patch_food_amount"][safe] = np.float32(s_food)
        arrays["patch_fertility"][exposed] = np.float32(e_fert)
        arrays["patch_fertility"][safe] = np.float32(s_fert)
        arrays["patch_cover"][exposed] = np.float32(e_cover)
        arrays["patch_cover"][safe] = np.float32(s_cover)
        arrays["patch_shelter"][exposed] = np.float32(e_shelter)
        arrays["patch_shelter"][safe] = np.float32(s_shelter)
        arrays["patch_movement_cost"][exposed] = np.float32(e_move)
        arrays["patch_movement_cost"][safe] = np.float32(s_move)
        arrays["patch_scent"][exposed] = np.float32(e_scent)
        arrays["patch_scent"][safe] = np.float32(s_scent)
        arrays["patch_waste"][exposed] = np.float32(e_waste)
        arrays["patch_waste"][safe] = np.float32(s_waste)
    else:
        raise ValueError(f"unsupported predrisk_fixed1_control_mode: {control}")
    for name in (
        "patch_food_amount",
        "patch_fertility",
        "patch_cover",
        "patch_shelter",
        "patch_movement_cost",
        "patch_scent",
        "patch_waste",
        "predrisk_fixed1_zone_class",
    ):
        arrays[name] = np.ascontiguousarray(np.nan_to_num(arrays[name]), dtype=np.float32)
    return arrays


def apply_corridor_cp0_overlay(cfg: Any, arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Apply the default-off CORRIDOR-CP0 central-place scaffold."""
    if not corridor_cp0_enabled(cfg):
        return arrays
    control = str(getattr(cfg, "corridor_cp0_control_mode", "central_place_tradeoff") or "central_place_tradeoff")
    classes = np.asarray(arrays["corridor_cp0_zone_class"], dtype=np.int32)
    home = classes == CORRIDOR_CP0_ZONE_HOME_LOCAL
    corridor = classes == CORRIDOR_CP0_ZONE_CORRIDOR
    remote = classes == CORRIDOR_CP0_ZONE_REMOTE_RICH
    any_zone = home | corridor | remote
    if not bool(np.any(any_zone)):
        return arrays

    food_max = patch_food_amount_max(cfg)
    move_min = _bounded(getattr(cfg, "patch_movement_cost_min", 0.0), default=0.0)
    move_max = _bounded(getattr(cfg, "patch_movement_cost_max", 4.0), default=4.0)
    scent_max = patch_scent_max(cfg)
    waste_max = patch_waste_max(cfg)
    values = {
        "food": (
            _bounded(getattr(cfg, "corridor_cp0_home_food", 0.30), default=0.30, lo=0.0, hi=food_max),
            _bounded(getattr(cfg, "corridor_cp0_corridor_food", 0.05), default=0.05, lo=0.0, hi=food_max),
            _bounded(getattr(cfg, "corridor_cp0_remote_food", 1.00), default=1.00, lo=0.0, hi=food_max),
        ),
        "fertility": (
            _bounded(getattr(cfg, "corridor_cp0_home_fertility", 0.45), default=0.45, lo=0.0),
            _bounded(getattr(cfg, "corridor_cp0_corridor_fertility", 0.20), default=0.20, lo=0.0),
            _bounded(getattr(cfg, "corridor_cp0_remote_fertility", 1.00), default=1.00, lo=0.0),
        ),
        "cover": (
            _bounded(getattr(cfg, "corridor_cp0_home_cover", 0.85), default=0.85, lo=0.0, hi=1.0),
            _bounded(getattr(cfg, "corridor_cp0_corridor_cover", 0.15), default=0.15, lo=0.0, hi=1.0),
            _bounded(getattr(cfg, "corridor_cp0_remote_cover", 0.10), default=0.10, lo=0.0, hi=1.0),
        ),
        "shelter": (
            _bounded(getattr(cfg, "corridor_cp0_home_shelter", 0.85), default=0.85, lo=0.0, hi=1.0),
            _bounded(getattr(cfg, "corridor_cp0_corridor_shelter", 0.10), default=0.10, lo=0.0, hi=1.0),
            _bounded(getattr(cfg, "corridor_cp0_remote_shelter", 0.05), default=0.05, lo=0.0, hi=1.0),
        ),
        "movement": (
            _bounded(getattr(cfg, "corridor_cp0_home_movement_cost", 0.75), default=0.75, lo=move_min, hi=move_max),
            _bounded(getattr(cfg, "corridor_cp0_corridor_movement_cost", 1.80), default=1.80, lo=move_min, hi=move_max),
            _bounded(getattr(cfg, "corridor_cp0_remote_movement_cost", 1.20), default=1.20, lo=move_min, hi=move_max),
        ),
        "scent": (
            _bounded(getattr(cfg, "corridor_cp0_home_scent", 0.05), default=0.05, lo=0.0, hi=scent_max),
            _bounded(getattr(cfg, "corridor_cp0_corridor_scent", 0.15), default=0.15, lo=0.0, hi=scent_max),
            _bounded(getattr(cfg, "corridor_cp0_remote_scent", 0.30), default=0.30, lo=0.0, hi=scent_max),
        ),
        "waste": (
            _bounded(getattr(cfg, "corridor_cp0_home_waste", 0.02), default=0.02, lo=0.0, hi=waste_max),
            _bounded(getattr(cfg, "corridor_cp0_corridor_waste", 0.10), default=0.10, lo=0.0, hi=waste_max),
            _bounded(getattr(cfg, "corridor_cp0_remote_waste", 0.15), default=0.15, lo=0.0, hi=waste_max),
        ),
    }
    uniforms = {key: np.float32(sum(vals) / 3.0) for key, vals in values.items()}

    def fill_triplet(name: str, key: str) -> None:
        h_val, c_val, r_val = values[key]
        arrays[name][home] = np.float32(h_val)
        arrays[name][corridor] = np.float32(c_val)
        arrays[name][remote] = np.float32(r_val)

    def fill_uniform(name: str, key: str) -> None:
        arrays[name][any_zone] = uniforms[key]

    if control == "true_flat":
        arrays["corridor_cp0_zone_class"][any_zone] = np.float32(CORRIDOR_CP0_ZONE_NONE)
    elif control in ("label_only_sham", "budget_matched_uniform", "home_eligibility_only"):
        for name, key in (
            ("patch_food_amount", "food"),
            ("patch_fertility", "fertility"),
            ("patch_cover", "cover"),
            ("patch_shelter", "shelter"),
            ("patch_movement_cost", "movement"),
            ("patch_scent", "scent"),
            ("patch_waste", "waste"),
        ):
            fill_uniform(name, key)
        if control == "budget_matched_uniform":
            arrays["corridor_cp0_zone_class"][any_zone] = np.float32(CORRIDOR_CP0_ZONE_NONE)
    elif control == "resource_gradient_only":
        fill_triplet("patch_food_amount", "food")
        fill_triplet("patch_fertility", "fertility")
        fill_uniform("patch_cover", "cover")
        fill_uniform("patch_shelter", "shelter")
        fill_uniform("patch_movement_cost", "movement")
        fill_uniform("patch_scent", "scent")
        fill_uniform("patch_waste", "waste")
    elif control == "corridor_cost_only":
        fill_uniform("patch_food_amount", "food")
        fill_uniform("patch_fertility", "fertility")
        fill_uniform("patch_cover", "cover")
        fill_uniform("patch_shelter", "shelter")
        fill_triplet("patch_movement_cost", "movement")
        fill_uniform("patch_scent", "scent")
        fill_uniform("patch_waste", "waste")
    elif control == "uncorrelated_home_resource":
        r_food, c_food, h_food = values["food"]
        arrays["patch_food_amount"][home] = np.float32(h_food)
        arrays["patch_food_amount"][corridor] = np.float32(c_food)
        arrays["patch_food_amount"][remote] = np.float32(r_food)
        fill_triplet("patch_fertility", "fertility")
        fill_triplet("patch_cover", "cover")
        fill_triplet("patch_shelter", "shelter")
        fill_triplet("patch_movement_cost", "movement")
        fill_triplet("patch_scent", "scent")
        fill_triplet("patch_waste", "waste")
    elif control == "central_place_tradeoff":
        fill_triplet("patch_food_amount", "food")
        fill_triplet("patch_fertility", "fertility")
        fill_triplet("patch_cover", "cover")
        fill_triplet("patch_shelter", "shelter")
        fill_triplet("patch_movement_cost", "movement")
        fill_triplet("patch_scent", "scent")
        fill_triplet("patch_waste", "waste")
    else:
        raise ValueError(f"unsupported corridor_cp0_control_mode: {control}")

    for name in (
        "patch_food_amount",
        "patch_fertility",
        "patch_cover",
        "patch_shelter",
        "patch_movement_cost",
        "patch_scent",
        "patch_waste",
        "corridor_cp0_zone_class",
    ):
        arrays[name] = np.ascontiguousarray(np.nan_to_num(arrays[name]), dtype=np.float32)
    return arrays


def apply_corridor_hc1_homing_cue_overlay(cfg: Any, arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Apply the HC1 ordinary shelter/cover homing gradient."""
    if not corridor_hc1_homing_cue_enabled(cfg):
        return arrays
    field = str(getattr(cfg, "corridor_hc1_homing_cue_field", "shelter") or "shelter")
    target_name = {"shelter": "patch_shelter", "cover": "patch_cover"}.get(field)
    if target_name is None:
        raise ValueError(f"unsupported corridor_hc1_homing_cue_field: {field}")
    classes = np.asarray(arrays["corridor_cp0_zone_class"], dtype=np.int32)
    home = classes == CORRIDOR_CP0_ZONE_HOME_LOCAL
    corridor = classes == CORRIDOR_CP0_ZONE_CORRIDOR
    remote = classes == CORRIDOR_CP0_ZONE_REMOTE_RICH
    if not bool(np.any(home | corridor | remote)):
        return arrays
    home_cols = np.where(np.any(home, axis=0))[0]
    if home_cols.size == 0:
        return arrays
    corridor_cols = np.where(np.any(corridor, axis=0))[0]
    remote_cols = np.where(np.any(remote, axis=0))[0]
    if corridor_cols.size:
        home_anchor = int(home_cols[-1]) if int(np.min(corridor_cols)) > int(np.max(home_cols)) else int(home_cols[0])
    elif remote_cols.size:
        home_anchor = int(home_cols[-1]) if int(np.min(remote_cols)) > int(np.max(home_cols)) else int(home_cols[0])
    else:
        home_anchor = int(np.mean(home_cols))

    hi = _bounded(getattr(cfg, "corridor_hc1_home_interior_value", 0.95), default=0.95, lo=0.0, hi=1.0)
    edge = _bounded(getattr(cfg, "corridor_hc1_home_edge_value", 0.72), default=0.72, lo=0.0, hi=1.0)
    near = _bounded(getattr(cfg, "corridor_hc1_corridor_near_home_value", 0.58), default=0.58, lo=0.0, hi=1.0)
    far = _bounded(getattr(cfg, "corridor_hc1_corridor_far_home_value", 0.18), default=0.18, lo=0.0, hi=1.0)
    remote_value = _bounded(getattr(cfg, "corridor_hc1_remote_value", 0.05), default=0.05, lo=0.0, hi=1.0)
    yy, xx = np.indices(classes.shape)
    home_dist = np.abs(xx.astype(np.int32) - int(home_anchor)).astype(np.float32)
    max_home_dist = float(max(1, np.max(home_dist[home]) if bool(np.any(home)) else 1))
    home_values = edge + (hi - edge) * np.clip(home_dist / max_home_dist, 0.0, 1.0)
    arrays[target_name][home] = np.asarray(home_values, dtype=np.float32)[home]
    if bool(np.any(corridor)):
        corr_dist = np.abs(xx.astype(np.int32) - int(home_anchor)).astype(np.float32)
        max_corr_dist = float(max(1, np.max(corr_dist[corridor])))
        corr_values = near + (far - near) * np.clip(corr_dist / max_corr_dist, 0.0, 1.0)
        arrays[target_name][corridor] = np.asarray(corr_values, dtype=np.float32)[corridor]
    arrays[target_name][remote] = np.float32(remote_value)
    arrays[target_name] = np.ascontiguousarray(np.nan_to_num(arrays[target_name]), dtype=np.float32)
    return arrays


def apply_mosaic0a_depletion_recovery_overlay(cfg: Any, arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Apply the default-off MOSAIC-0A local depletion/recovery scaffold."""
    if not mosaic0a_depletion_recovery_enabled(cfg):
        return arrays
    control = str(getattr(cfg, "mosaic0a_control_mode", "real_mosaic") or "real_mosaic")
    if control == "patch_uniform":
        return arrays
    classes = np.asarray(arrays["mosaic0a_patch_class"], dtype=np.int32)
    burst = classes == MOSAIC0A_PATCH_BURST
    renewal = classes == MOSAIC0A_PATCH_RENEWAL
    if not bool(np.any(burst | renewal)) or control == "label_only_sham":
        return arrays

    food_max = patch_food_amount_max(cfg)
    waste_max = patch_waste_max(cfg)
    recovery_min, recovery_max = patch_recovery_activity_min_max(cfg)
    move_min = _bounded(getattr(cfg, "patch_movement_cost_min", 0.0), default=0.0)
    move_max = _bounded(getattr(cfg, "patch_movement_cost_max", 4.0), default=4.0)
    if move_max < move_min:
        move_min, move_max = move_max, move_min
    burst_initial = _bounded(getattr(cfg, "mosaic0a_burst_initial_food", 1.0), default=1.0, lo=0.0, hi=food_max)
    renewal_initial = _bounded(getattr(cfg, "mosaic0a_renewal_initial_food", 0.4), default=0.4, lo=0.0, hi=food_max)
    burst_fertility = _bounded(getattr(cfg, "mosaic0a_burst_fertility", 0.4), default=0.4, lo=0.0)
    renewal_fertility = _bounded(getattr(cfg, "mosaic0a_renewal_fertility", 1.0), default=1.0, lo=0.0)
    burst_move = _bounded(getattr(cfg, "mosaic0a_burst_movement_cost", 1.25), default=1.25, lo=move_min, hi=move_max)
    renewal_move = _bounded(getattr(cfg, "mosaic0a_renewal_movement_cost", 0.85), default=0.85, lo=move_min, hi=move_max)
    burst_waste = _bounded(getattr(cfg, "mosaic0a_burst_waste_initial", 0.20), default=0.20, lo=0.0, hi=waste_max)
    renewal_waste = _bounded(getattr(cfg, "mosaic0a_renewal_waste_initial", 0.0), default=0.0, lo=0.0, hi=waste_max)
    burst_recovery = _bounded(
        getattr(cfg, "mosaic0a_burst_recovery_activity", 0.0),
        default=0.0,
        lo=recovery_min,
        hi=recovery_max,
    )
    renewal_recovery = _bounded(
        getattr(cfg, "mosaic0a_renewal_recovery_activity", 0.45),
        default=0.45,
        lo=recovery_min,
        hi=recovery_max,
    )

    if control == "budget_matched_uniform":
        total_cells = max(1, int(np.sum(burst | renewal)))
        uniform_initial = (burst_initial * int(np.sum(burst)) + renewal_initial * int(np.sum(renewal))) / total_cells
        uniform_fertility = (
            burst_fertility * int(np.sum(burst)) + renewal_fertility * int(np.sum(renewal))
        ) / total_cells
        arrays["patch_food_amount"][burst | renewal] = np.float32(min(food_max, uniform_initial))
        arrays["patch_fertility"][burst | renewal] = np.float32(uniform_fertility)
        arrays["patch_movement_cost"][burst | renewal] = np.float32((burst_move + renewal_move) / 2.0)
        arrays["patch_waste"][burst | renewal] = np.float32((burst_waste + renewal_waste) / 2.0)
        arrays["patch_recovery_activity"][burst | renewal] = np.float32((burst_recovery + renewal_recovery) / 2.0)
    elif control == "depletion_only":
        arrays["patch_food_amount"][burst] = np.float32(burst_initial)
        arrays["patch_food_amount"][renewal] = np.float32(renewal_initial)
    elif control == "recovery_only":
        arrays["patch_fertility"][burst] = np.float32(burst_fertility)
        arrays["patch_fertility"][renewal] = np.float32(renewal_fertility)
        arrays["patch_recovery_activity"][burst] = np.float32(burst_recovery)
        arrays["patch_recovery_activity"][renewal] = np.float32(renewal_recovery)
    elif control == "real_mosaic":
        arrays["patch_food_amount"][burst] = np.float32(burst_initial)
        arrays["patch_food_amount"][renewal] = np.float32(renewal_initial)
        arrays["patch_fertility"][burst] = np.float32(burst_fertility)
        arrays["patch_fertility"][renewal] = np.float32(renewal_fertility)
        arrays["patch_movement_cost"][burst] = np.float32(burst_move)
        arrays["patch_movement_cost"][renewal] = np.float32(renewal_move)
        arrays["patch_waste"][burst] = np.maximum(arrays["patch_waste"][burst], np.float32(burst_waste))
        arrays["patch_waste"][renewal] = np.maximum(arrays["patch_waste"][renewal], np.float32(renewal_waste))
        arrays["patch_recovery_activity"][burst] = np.float32(burst_recovery)
        arrays["patch_recovery_activity"][renewal] = np.float32(renewal_recovery)
    else:
        raise ValueError(f"unsupported mosaic0a_control_mode: {control}")
    for name in (
        "patch_food_amount",
        "patch_fertility",
        "patch_movement_cost",
        "patch_waste",
        "patch_recovery_activity",
        "mosaic0a_patch_class",
    ):
        arrays[name] = np.ascontiguousarray(np.nan_to_num(arrays[name]), dtype=np.float32)
    return arrays


def roz2a_patch_niche_enabled(cfg: Any) -> bool:
    """Return true only for the default-off ROZ-2A static niche mechanics."""
    return bool(getattr(cfg, "patch_world_enable", False)) and bool(getattr(cfg, "roz2a_patch_niche_enable", False))


def roz2a_patch_niche_zone_specs(cfg: Any, *, shape: tuple[int, int]) -> list[dict[str, Any]]:
    """Return deterministic, lineage-blind static zone specifications."""
    h, w = (max(1, int(shape[0])), max(1, int(shape[1])))
    radius = max(1, int(getattr(cfg, "roz2a_patch_niche_radius", 2) or 2))

    def clamp_y(v: int) -> int:
        return max(0, min(h - 1, int(v)))

    def clamp_x(v: int) -> int:
        return max(0, min(w - 1, int(v)))

    return [
        {
            "id": "resource_exposed",
            "center_y": clamp_y(h // 4),
            "center_x": clamp_x(w // 4),
            "radius": radius,
            "class": "resource_exposed",
        },
        {
            "id": "sheltered_low_resource",
            "center_y": clamp_y((3 * h) // 4),
            "center_x": clamp_x((3 * w) // 4),
            "radius": radius,
            "class": "sheltered_low_resource",
        },
        {
            "id": "transition_mixed",
            "center_y": clamp_y(h // 4),
            "center_x": clamp_x((3 * w) // 4),
            "radius": radius,
            "class": "transition_mixed",
        },
    ]


def _roz2a_zone_mask(*, shape: tuple[int, int], center_y: int, center_x: int, radius: int) -> np.ndarray:
    yy, xx = np.indices(shape, dtype=np.int32)
    dist = np.abs(yy - np.int32(center_y)) + np.abs(xx - np.int32(center_x))
    return np.asarray(dist <= np.int32(max(0, int(radius))), dtype=np.bool_)


def apply_roz2a_patch_niche_static_overlay(cfg: Any, arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Apply the active ROZ-2A static ecological tradeoff overlay.

    The overlay is deterministic and reads only grid geometry plus Patch World
    arrays. It does not read agents, lineages, branches, strategy labels, RNG,
    rewards, selection state, or reproduction state.
    """
    if not roz2a_patch_niche_enabled(cfg):
        return arrays
    mode = str(getattr(cfg, "roz2a_patch_niche_mode", "static_three_zone_tradeoff_v1") or "")
    if mode != "static_three_zone_tradeoff_v1":
        raise ValueError(f"unsupported roz2a_patch_niche_mode: {mode}")
    food = np.asarray(arrays["patch_food_amount"], dtype=np.float32)
    shape = tuple(food.shape)
    food_max = patch_food_amount_max(cfg)
    waste_max = patch_waste_max(cfg)
    recovery_min, recovery_max = patch_recovery_activity_min_max(cfg)
    move_min = _bounded(getattr(cfg, "patch_movement_cost_min", 0.0), default=0.0)
    move_max = _bounded(getattr(cfg, "patch_movement_cost_max", 4.0), default=4.0)
    if move_max < move_min:
        move_min, move_max = move_max, move_min

    for spec in roz2a_patch_niche_zone_specs(cfg, shape=shape):
        mask = _roz2a_zone_mask(
            shape=shape,
            center_y=int(spec["center_y"]),
            center_x=int(spec["center_x"]),
            radius=int(spec["radius"]),
        )
        if not bool(np.any(mask)):
            continue
        cls = str(spec["class"])
        if cls == "resource_exposed":
            fertility = _bounded(getattr(cfg, "roz2a_patch_niche_resource_fertility", 1.0), default=1.0, lo=0.0, hi=1.0)
            move = _bounded(
                getattr(cfg, "roz2a_patch_niche_resource_movement_cost", 1.6),
                default=1.6,
                lo=move_min,
                hi=move_max,
            )
            waste = _bounded(getattr(cfg, "roz2a_patch_niche_resource_waste", 0.20), default=0.20, lo=0.0, hi=waste_max)
            arrays["patch_fertility"][mask] = np.float32(fertility)
            arrays["patch_movement_cost"][mask] = np.float32(move)
            arrays["patch_waste"][mask] = np.maximum(arrays["patch_waste"][mask], np.float32(waste))
        elif cls == "sheltered_low_resource":
            food_mult = _bounded(getattr(cfg, "roz2a_patch_niche_shelter_food_multiplier", 0.35), default=0.35, lo=0.0, hi=1.0)
            fertility = _bounded(getattr(cfg, "roz2a_patch_niche_shelter_fertility", 0.35), default=0.35, lo=0.0, hi=1.0)
            cover = _bounded(getattr(cfg, "roz2a_patch_niche_shelter_cover", 0.75), default=0.75, lo=0.0, hi=1.0)
            shelter = _bounded(getattr(cfg, "roz2a_patch_niche_shelter_shelter", 0.75), default=0.75, lo=0.0, hi=1.0)
            move = _bounded(
                getattr(cfg, "roz2a_patch_niche_shelter_movement_cost", 0.80),
                default=0.80,
                lo=move_min,
                hi=move_max,
            )
            recovery = _bounded(
                getattr(cfg, "roz2a_patch_niche_shelter_recovery", 0.45),
                default=0.45,
                lo=recovery_min,
                hi=recovery_max,
            )
            arrays["patch_food_amount"][mask] = np.clip(arrays["patch_food_amount"][mask] * np.float32(food_mult), 0.0, food_max)
            arrays["patch_fertility"][mask] = np.float32(fertility)
            arrays["patch_cover"][mask] = np.maximum(arrays["patch_cover"][mask], np.float32(cover))
            arrays["patch_shelter"][mask] = np.maximum(arrays["patch_shelter"][mask], np.float32(shelter))
            arrays["patch_movement_cost"][mask] = np.float32(move)
            arrays["patch_recovery_activity"][mask] = np.maximum(arrays["patch_recovery_activity"][mask], np.float32(recovery))
        elif cls == "transition_mixed":
            fertility = _bounded(getattr(cfg, "roz2a_patch_niche_transition_fertility", 0.65), default=0.65, lo=0.0, hi=1.0)
            cover = _bounded(getattr(cfg, "roz2a_patch_niche_transition_cover", 0.30), default=0.30, lo=0.0, hi=1.0)
            shelter = _bounded(getattr(cfg, "roz2a_patch_niche_transition_shelter", 0.25), default=0.25, lo=0.0, hi=1.0)
            arrays["patch_fertility"][mask] = np.float32(fertility)
            arrays["patch_cover"][mask] = np.maximum(arrays["patch_cover"][mask], np.float32(cover))
            arrays["patch_shelter"][mask] = np.maximum(arrays["patch_shelter"][mask], np.float32(shelter))
    for name in ("patch_food_amount", "patch_fertility", "patch_cover", "patch_shelter", "patch_movement_cost", "patch_waste", "patch_recovery_activity"):
        arrays[name] = np.ascontiguousarray(np.nan_to_num(arrays[name]), dtype=np.float32)
    return arrays


def patch_visibility_from_cover(cfg: Any, cover: Any) -> np.ndarray:
    """Return the deterministic derived visibility affordance."""
    cover_arr = np.asarray(cover, dtype=np.float32)
    vis_min = _bounded(getattr(cfg, "patch_visibility_min", 0.0), default=0.0)
    vis_max = _bounded(getattr(cfg, "patch_visibility_max", 1.0), default=1.0)
    if vis_max < vis_min:
        vis_min, vis_max = vis_max, vis_min
    if bool(getattr(cfg, "patch_visibility_from_cover_enable", True)):
        raw = 1.0 - cover_arr
    else:
        default = _bounded(getattr(cfg, "patch_visibility_default", 1.0), default=1.0, lo=vis_min, hi=vis_max)
        raw = np.full(tuple(cover_arr.shape), default, dtype=np.float32)
    return np.ascontiguousarray(np.clip(raw, vis_min, vis_max), dtype=np.float32)


def patch_movement_cost_energy_scale(cfg: Any) -> float:
    return _bounded(getattr(cfg, "patch_movement_cost_energy_scale", 1.0), default=1.0, lo=0.0)


def normalize_patch_movement_cost(cfg: Any, movement_cost: Any) -> np.ndarray:
    arr = np.asarray(movement_cost, dtype=np.float32)
    lo = _bounded(getattr(cfg, "patch_movement_cost_min", 0.0), default=0.0)
    hi = _bounded(getattr(cfg, "patch_movement_cost_max", 4.0), default=4.0)
    if hi < lo:
        lo, hi = hi, lo
    span = float(hi - lo)
    if span <= 0.0:
        return np.zeros(tuple(arr.shape), dtype=np.float32)
    out = (arr - np.float32(lo)) / np.float32(span)
    return np.ascontiguousarray(np.nan_to_num(np.clip(out, 0.0, 1.0)), dtype=np.float32)


def patch_scent_max(cfg: Any) -> float:
    return _bounded(getattr(cfg, "patch_scent_max", 1.0), default=1.0, lo=0.0)


def patch_waste_max(cfg: Any) -> float:
    return _bounded(getattr(cfg, "patch_waste_max", 1.0), default=1.0, lo=0.0)


def jungle_ecological_recovery_enabled(cfg: Any) -> bool:
    """Return true only for explicit Step 12 ecological recovery."""
    return bool(getattr(cfg, "patch_world_enable", False)) and bool(
        getattr(cfg, "jungle_ecological_recovery_enable", False)
    )


def patch_recovery_activity_min_max(cfg: Any) -> tuple[float, float]:
    """Return bounded recovery-activity min/max."""
    lo = _bounded(getattr(cfg, "jungle_recovery_activity_min", 0.0), default=0.0)
    hi = _bounded(getattr(cfg, "jungle_recovery_activity_max", 1.0), default=1.0)
    if hi < lo:
        lo, hi = hi, lo
    return float(lo), float(hi)


def normalize_patch_density(cfg: Any, density: Any) -> np.ndarray:
    arr = np.asarray(density, dtype=np.float32)
    if bool(getattr(cfg, "patch_density_normalize", False)):
        return np.ascontiguousarray(np.nan_to_num(np.clip(arr, 0.0, 1.0)), dtype=np.float32)
    radius = int(max(0, getattr(cfg, "patch_density_radius", 1)))
    denom = float((2 * radius + 1) * (2 * radius + 1))
    if not bool(getattr(cfg, "patch_density_include_self", True)):
        denom -= 1.0
    return _safe_norm(arr, denom)


def patch_world_waste_consequence_enabled(cfg: Any) -> bool:
    """Return true only for explicit Step 8 waste/crowding consequences."""
    return bool(getattr(cfg, "patch_world_enable", False)) and bool(
        getattr(cfg, "patch_world_waste_consequence_enable", False)
    )


def wdl_delayed_hazard_enabled(cfg: Any) -> bool:
    """Return true only for the explicit WDL-1 delayed-hazard substrate."""
    return bool(getattr(cfg, "patch_world_enable", False)) and bool(
        getattr(cfg, "wdl_delayed_hazard_enable", False)
    )


def delayed_resource_depletion_enabled(cfg: Any) -> bool:
    """Return true only for the explicit delayed resource-depletion family."""
    return bool(getattr(cfg, "patch_world_enable", False)) and bool(
        getattr(cfg, "delayed_resource_depletion_enable", False)
    )


def delayed_crowding_waste_enabled(cfg: Any) -> bool:
    """Return true only for the explicit delayed overuse/waste penalty family."""
    return bool(getattr(cfg, "patch_world_enable", False)) and bool(
        getattr(cfg, "delayed_crowding_waste_enable", False)
    )


def mem_max_delayed_choice_enabled(cfg: Any) -> bool:
    """Return true only for the explicit MEM-MAX delayed-choice substrate."""
    if not (bool(getattr(cfg, "patch_world_enable", False)) and bool(getattr(cfg, "mem_max_delayed_choice_enable", False))):
        return False
    version = int(getattr(cfg, "mem_max_delayed_choice_schema_version", MEM_MAX_DELAYED_CHOICE_SCHEMA_VERSION) or 1)
    if version != MEM_MAX_DELAYED_CHOICE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported MEM-MAX delayed-choice schema version: {version}. "
            f"Supported version: {MEM_MAX_DELAYED_CHOICE_SCHEMA_VERSION}."
        )
    return True


def wdl_delayed_hazard_observation_enabled(cfg: Any) -> bool:
    """Return true only when WDL cue observation is explicitly enabled."""
    return bool(patch_world_observation_enabled(cfg)) and bool(
        getattr(cfg, "wdl_delayed_hazard_observation_enable", False)
    ) and bool(wdl_delayed_hazard_enabled(cfg))


def wdl_delayed_hazard_observation_channel_count(cfg: Any) -> int:
    return len(WDL_DELAYED_HAZARD_OBSERVATION_CHANNEL_NAMES) if wdl_delayed_hazard_observation_enabled(cfg) else 0


def prediction_demand_enabled(cfg: Any) -> bool:
    """Return true only for the explicit Track C prediction-demand substrate."""
    return bool(getattr(cfg, "patch_world_enable", False)) and bool(
        getattr(cfg, "prediction_demand_enable", False)
    )


def prediction_demand_observation_enabled(cfg: Any) -> bool:
    """Return true only when forecast-demand cues are explicitly observable."""
    return bool(patch_world_observation_enabled(cfg)) and bool(
        getattr(cfg, "prediction_demand_observation_enable", False)
    ) and bool(prediction_demand_enabled(cfg))


def prediction_demand_observation_channel_count(cfg: Any) -> int:
    return len(PREDICTION_DEMAND_OBSERVATION_CHANNEL_NAMES) if prediction_demand_observation_enabled(cfg) else 0


def delayed_resource_depletion_observation_enabled(cfg: Any) -> bool:
    return bool(patch_world_observation_enabled(cfg)) and bool(
        getattr(cfg, "delayed_resource_depletion_observation_enable", False)
    ) and bool(delayed_resource_depletion_enabled(cfg))


def delayed_crowding_waste_observation_enabled(cfg: Any) -> bool:
    return bool(patch_world_observation_enabled(cfg)) and bool(
        getattr(cfg, "delayed_crowding_waste_observation_enable", False)
    ) and bool(delayed_crowding_waste_enabled(cfg))


def delayed_resource_waste_observation_channel_count(cfg: Any) -> int:
    count = 0
    if delayed_resource_depletion_observation_enabled(cfg):
        count += 1
    if delayed_crowding_waste_observation_enabled(cfg):
        count += 1
    return int(count)


def mem_max_delayed_choice_observation_enabled(cfg: Any) -> bool:
    return bool(patch_world_observation_enabled(cfg)) and bool(
        getattr(cfg, "mem_max_delayed_choice_observation_enable", False)
    ) and bool(mem_max_delayed_choice_enabled(cfg))


def mem_max_delayed_choice_navigation_observation_enabled(cfg: Any) -> bool:
    if not mem_max_delayed_choice_observation_enabled(cfg):
        return False
    version = validate_mem_max_delayed_choice_observation_schema_version(
        int(getattr(cfg, "mem_max_delayed_choice_observation_schema_version", 1) or 1)
    )
    if not bool(getattr(cfg, "mem_max_delayed_choice_navigation_observation_enable", False)):
        return False
    if version < MEM_MAX_DELAYED_CHOICE_OBSERVATION_SCHEMA_VERSION:
        raise ValueError(
            "mem_max_delayed_choice_navigation_observation_enable requires "
            f"mem_max_delayed_choice_observation_schema_version={MEM_MAX_DELAYED_CHOICE_OBSERVATION_SCHEMA_VERSION}"
        )
    return True


def mem_max_delayed_choice_observation_channel_count(cfg: Any) -> int:
    if not mem_max_delayed_choice_observation_enabled(cfg):
        return 0
    count = len(MEM_MAX_DELAYED_CHOICE_CUE_OBSERVATION_CHANNEL_NAMES)
    if mem_max_delayed_choice_navigation_observation_enabled(cfg):
        count += len(MEM_MAX_DELAYED_CHOICE_NAVIGATION_OBSERVATION_CHANNEL_NAMES)
    return int(count)


def _delayed_consequence_cue_observation_map(env: Any, cue_name: str, dropout_name: str) -> np.ndarray:
    cfg = getattr(env, "cfg", None)
    cue = np.nan_to_num(np.clip(np.asarray(getattr(env, cue_name), dtype=np.float32), 0.0, 1.0))
    dropout = _bounded(getattr(cfg, dropout_name, 0.0), default=0.0, lo=0.0, hi=1.0)
    if dropout <= 0.0:
        return np.ascontiguousarray(cue, dtype=np.float32)
    if dropout >= 1.0:
        return np.zeros(tuple(cue.shape), dtype=np.float32)
    yy, xx = np.indices(tuple(cue.shape), dtype=np.int64)
    seed = int(_wdl_seed(env)) & 0xFFFFFFFF
    salt = int(sum(ord(ch) for ch in str(cue_name))) & 0xFFFFFFFF
    mixed = (
        yy * np.int64(1103515245)
        + xx * np.int64(12345)
        + np.int64(seed)
        + np.int64(salt)
        + np.int64(0x9E3779B9)
    ) & np.int64(0xFFFFFFFF)
    keep = (mixed.astype(np.float64) / float(2**32)) >= float(dropout)
    return np.ascontiguousarray(cue * keep.astype(np.float32), dtype=np.float32)


def _wdl_delayed_hazard_observation_cue_map(env: Any) -> np.ndarray:
    cfg = getattr(env, "cfg", None)
    cue = np.nan_to_num(np.clip(np.asarray(getattr(env, "patch_delayed_hazard_cue"), dtype=np.float32), 0.0, 1.0))
    dropout = _bounded(getattr(cfg, "wdl_delayed_hazard_observation_dropout_rate", 0.0), default=0.0, lo=0.0, hi=1.0)
    if dropout <= 0.0:
        return np.ascontiguousarray(cue, dtype=np.float32)
    if dropout >= 1.0:
        return np.zeros(tuple(cue.shape), dtype=np.float32)
    yy, xx = np.indices(tuple(cue.shape), dtype=np.int64)
    seed = int(_wdl_seed(env)) & 0xFFFFFFFF
    mixed = (yy * np.int64(1103515245) + xx * np.int64(12345) + np.int64(seed) + np.int64(0x9E3779B9)) & np.int64(0xFFFFFFFF)
    keep = (mixed.astype(np.float64) / float(2**32)) >= float(dropout)
    return np.ascontiguousarray(cue * keep.astype(np.float32), dtype=np.float32)


def _mem_max_delayed_choice_navigation_observation_maps(env: Any) -> tuple[np.ndarray, ...]:
    target_a = np.asarray(getattr(env, "mem_max_delayed_choice_target_a"), dtype=np.float32)
    target_b = np.asarray(getattr(env, "mem_max_delayed_choice_target_b"), dtype=np.float32)
    shape = tuple(target_a.shape)
    zeros = np.zeros(shape, dtype=np.float32)

    def _maps_for(target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        coords = np.argwhere(np.asarray(target, dtype=np.float32) > 0.0)
        if coords.size <= 0:
            return zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy()
        ty, tx = [int(v) for v in coords[0].tolist()]
        yy, xx = np.indices(shape, dtype=np.float32)
        h, w = shape
        dy = np.clip((np.float32(ty) - yy) / np.float32(max(1, h - 1)), -1.0, 1.0)
        dx = np.clip((np.float32(tx) - xx) / np.float32(max(1, w - 1)), -1.0, 1.0)
        dist = (np.abs(np.float32(ty) - yy) + np.abs(np.float32(tx) - xx)) / np.float32(max(1, h + w - 2))
        in_target = np.asarray(target > 0.0, dtype=np.float32)
        return (
            np.ascontiguousarray(np.nan_to_num(dy), dtype=np.float32),
            np.ascontiguousarray(np.nan_to_num(dx), dtype=np.float32),
            np.ascontiguousarray(np.nan_to_num(np.clip(dist, 0.0, 1.0)), dtype=np.float32),
            np.ascontiguousarray(in_target, dtype=np.float32),
        )

    return _maps_for(target_a) + _maps_for(target_b)


def _prediction_demand_observation_maps(env: Any) -> tuple[np.ndarray, np.ndarray]:
    cue = np.nan_to_num(
        np.clip(np.asarray(getattr(env, "patch_prediction_demand_cue"), dtype=np.float32), 0.0, 1.0)
    )
    trend = np.nan_to_num(
        np.clip(np.asarray(getattr(env, "patch_prediction_demand_trend"), dtype=np.float32), -1.0, 1.0)
    )
    trend_norm = np.clip((trend + np.float32(1.0)) * np.float32(0.5), 0.0, 1.0)
    return (
        np.ascontiguousarray(cue, dtype=np.float32),
        np.ascontiguousarray(trend_norm, dtype=np.float32),
    )


def patch_regrowth_multiplier_maps(env: Any) -> Dict[str, np.ndarray]:
    """Compute deterministic Step 8 regrowth multipliers and penalty maps."""
    cfg = getattr(env, "cfg", None)
    amount = np.asarray(getattr(env, "patch_food_amount"), dtype=np.float32)
    shape = tuple(amount.shape)
    zeros = np.zeros(shape, dtype=np.float32)
    ones = np.ones(shape, dtype=np.float32)
    if not patch_world_waste_consequence_enabled(cfg):
        prediction_maps = prediction_demand_regrowth_multiplier_maps(env)
        return {
            "multiplier": np.asarray(prediction_maps["multiplier"], dtype=np.float32),
            "waste_penalty": zeros,
            "density_penalty": zeros,
        }

    waste_norm = _safe_norm(getattr(env, "patch_waste"), patch_waste_max(cfg))
    compute_local_density_map(env)
    density_norm = normalize_patch_density(cfg, getattr(env, "local_density_map"))

    waste_penalty = zeros.copy()
    if bool(getattr(cfg, "patch_waste_regrowth_penalty_enable", True)):
        scale = _bounded(getattr(cfg, "patch_waste_regrowth_penalty_scale", 0.5), default=0.5, lo=0.0)
        cap = _bounded(getattr(cfg, "patch_waste_regrowth_penalty_max", 0.5), default=0.5, lo=0.0, hi=1.0)
        waste_penalty = np.minimum(np.float32(cap), waste_norm * np.float32(scale)).astype(np.float32, copy=False)

    density_penalty = zeros.copy()
    if bool(getattr(cfg, "patch_density_regrowth_penalty_enable", True)):
        scale = _bounded(getattr(cfg, "patch_density_regrowth_penalty_scale", 0.5), default=0.5, lo=0.0)
        cap = _bounded(getattr(cfg, "patch_density_regrowth_penalty_max", 0.5), default=0.5, lo=0.0, hi=1.0)
        density_penalty = np.minimum(np.float32(cap), density_norm * np.float32(scale)).astype(np.float32, copy=False)

    if delayed_resource_depletion_enabled(cfg):
        active = np.asarray(getattr(env, "delayed_resource_depletion_active"), dtype=np.float32) > 0.0
        if bool(np.any(active)):
            mult = _bounded(
                getattr(cfg, "delayed_resource_depletion_regrowth_multiplier", 0.25),
                default=0.25,
                lo=0.0,
                hi=1.0,
            )
            waste_penalty = np.maximum(waste_penalty, active.astype(np.float32) * np.float32(1.0 - mult))
    if delayed_crowding_waste_enabled(cfg):
        active = np.asarray(getattr(env, "delayed_crowding_waste_active"), dtype=np.float32) > 0.0
        if bool(np.any(active)):
            mult = _bounded(
                getattr(cfg, "delayed_crowding_waste_regrowth_multiplier", 0.35),
                default=0.35,
                lo=0.0,
                hi=1.0,
            )
            density_penalty = np.maximum(density_penalty, active.astype(np.float32) * np.float32(1.0 - mult))

    min_mult = _bounded(getattr(cfg, "patch_regrowth_min_multiplier", 0.1), default=0.1, lo=0.0, hi=1.0)
    total_penalty = np.clip(waste_penalty + density_penalty, 0.0, 1.0).astype(np.float32, copy=False)
    multiplier = np.clip(1.0 - total_penalty, min_mult, 1.0).astype(np.float32, copy=False)
    prediction_maps = prediction_demand_regrowth_multiplier_maps(env)
    multiplier = np.clip(
        multiplier * np.asarray(prediction_maps["multiplier"], dtype=np.float32),
        0.0,
        2.0,
    ).astype(np.float32, copy=False)
    return {
        "multiplier": np.ascontiguousarray(np.nan_to_num(multiplier), dtype=np.float32),
        "waste_penalty": np.ascontiguousarray(np.nan_to_num(waste_penalty), dtype=np.float32),
        "density_penalty": np.ascontiguousarray(np.nan_to_num(density_penalty), dtype=np.float32),
    }


def patch_world_normalized_observation_maps(env: Any) -> tuple[np.ndarray, ...]:
    """Return normalized Patch World V1 observation maps in channel order."""
    cfg = getattr(env, "cfg", None)
    compute_local_density_map(env)
    maps = (
        _safe_norm(getattr(env, "patch_food_amount"), patch_food_amount_max(cfg)),
        np.ascontiguousarray(np.nan_to_num(np.clip(np.asarray(getattr(env, "patch_cover"), dtype=np.float32), 0.0, 1.0)), dtype=np.float32),
        np.ascontiguousarray(np.nan_to_num(np.clip(np.asarray(getattr(env, "patch_shelter"), dtype=np.float32), 0.0, 1.0)), dtype=np.float32),
        _safe_norm(getattr(env, "patch_scent"), patch_scent_max(cfg)),
        _safe_norm(getattr(env, "patch_waste"), patch_waste_max(cfg)),
        normalize_patch_density(cfg, getattr(env, "local_density_map")),
        normalize_patch_movement_cost(cfg, getattr(env, "patch_movement_cost")),
    )
    if wdl_delayed_hazard_observation_enabled(cfg):
        maps = maps + (
            _wdl_delayed_hazard_observation_cue_map(env),
        )
    if prediction_demand_observation_enabled(cfg):
        maps = maps + _prediction_demand_observation_maps(env)
    if delayed_resource_depletion_observation_enabled(cfg):
        maps = maps + (
            _delayed_consequence_cue_observation_map(
                env,
                "delayed_resource_depletion_cue",
                "delayed_resource_depletion_observation_dropout_rate",
            ),
        )
    if delayed_crowding_waste_observation_enabled(cfg):
        maps = maps + (
            _delayed_consequence_cue_observation_map(
                env,
                "delayed_crowding_waste_cue",
                "delayed_crowding_waste_observation_dropout_rate",
            ),
        )
    if mem_max_delayed_choice_observation_enabled(cfg):
        if bool(getattr(cfg, "mem_max_delayed_choice_shortcut_truth_enable", False)):
            correct = np.asarray(getattr(env, "mem_max_delayed_choice_correct_target"), dtype=np.int8)
            target_a = np.asarray(getattr(env, "mem_max_delayed_choice_target_a"), dtype=np.float32)
            target_b = np.asarray(getattr(env, "mem_max_delayed_choice_target_b"), dtype=np.float32)
            maps = maps + (
                np.ascontiguousarray((correct == 1).astype(np.float32) * target_a, dtype=np.float32),
                np.ascontiguousarray((correct == 2).astype(np.float32) * target_b, dtype=np.float32),
            )
        else:
            cue_a = np.nan_to_num(np.clip(np.asarray(getattr(env, "mem_max_delayed_choice_cue_a"), dtype=np.float32), 0.0, 1.0))
            cue_b = np.nan_to_num(np.clip(np.asarray(getattr(env, "mem_max_delayed_choice_cue_b"), dtype=np.float32), 0.0, 1.0))
            if bool(getattr(cfg, "mem_max_delayed_choice_global_cue_observation_enable", False)):
                cue_a = np.full(tuple(cue_a.shape), np.float32(np.max(cue_a)), dtype=np.float32)
                cue_b = np.full(tuple(cue_b.shape), np.float32(np.max(cue_b)), dtype=np.float32)
            maps = maps + (
                np.ascontiguousarray(cue_a, dtype=np.float32),
                np.ascontiguousarray(cue_b, dtype=np.float32),
            )
        if mem_max_delayed_choice_navigation_observation_enabled(cfg):
            maps = maps + _mem_max_delayed_choice_navigation_observation_maps(env)
    return maps


def _decay_factor(cfg: Any, name: str, default: float) -> float:
    decay = _bounded(getattr(cfg, name, default), default=default, lo=0.0, hi=1.0)
    return float(1.0 - decay)


def _emit_value(cfg: Any, name: str, default: float) -> float:
    return _bounded(getattr(cfg, name, default), default=default, lo=0.0)


def initialize_patch_world_arrays(cfg: Any, grid: np.ndarray) -> Dict[str, np.ndarray]:
    """Create deterministic passive Patch World arrays from current grid state."""
    grid_arr = np.asarray(grid)
    shape = tuple(int(x) for x in grid_arr.shape)

    food_max = patch_food_amount_max(cfg)
    food_cell = patch_food_initial_food_cell_value(cfg)
    empty_cell = _bounded(
        getattr(cfg, "patch_food_amount_initial_empty_cell_value", 0.0),
        default=0.0,
        lo=0.0,
        hi=food_max,
    )

    fertility_min = _bounded(getattr(cfg, "patch_fertility_min", 0.0), default=0.0)
    fertility_max = _bounded(getattr(cfg, "patch_fertility_max", 1.0), default=1.0)
    if fertility_max < fertility_min:
        fertility_min, fertility_max = fertility_max, fertility_min
    fertility_default = _bounded(
        getattr(cfg, "patch_fertility_default", 1.0),
        default=1.0,
        lo=fertility_min,
        hi=fertility_max,
    )

    cover_min = _bounded(getattr(cfg, "patch_cover_min", 0.0), default=0.0)
    cover_max = _bounded(getattr(cfg, "patch_cover_max", 1.0), default=1.0)
    if cover_max < cover_min:
        cover_min, cover_max = cover_max, cover_min
    cover_default = _bounded(
        getattr(cfg, "patch_cover_default", 0.0),
        default=0.0,
        lo=cover_min,
        hi=cover_max,
    )
    cover = np.full(shape, cover_default, dtype=np.float32)

    shelter_min = _bounded(getattr(cfg, "patch_shelter_min", 0.0), default=0.0)
    shelter_max = _bounded(getattr(cfg, "patch_shelter_max", 1.0), default=1.0)
    if shelter_max < shelter_min:
        shelter_min, shelter_max = shelter_max, shelter_min
    if bool(getattr(cfg, "patch_shelter_from_cover_enable", False)):
        shelter = np.clip(cover, shelter_min, shelter_max).astype(np.float32, copy=False)
    else:
        shelter_default = _bounded(
            getattr(cfg, "patch_shelter_default", 0.0),
            default=0.0,
            lo=shelter_min,
            hi=shelter_max,
        )
        shelter = np.full(shape, shelter_default, dtype=np.float32)

    move_min = _bounded(getattr(cfg, "patch_movement_cost_min", 0.0), default=0.0)
    move_max = _bounded(getattr(cfg, "patch_movement_cost_max", 4.0), default=4.0)
    if move_max < move_min:
        move_min, move_max = move_max, move_min
    move_default = _bounded(
        getattr(cfg, "patch_movement_cost_default", 1.0),
        default=1.0,
        lo=move_min,
        hi=move_max,
    )

    scent_max = patch_scent_max(cfg)
    scent_initial = _bounded(
        getattr(cfg, "patch_scent_initial", 0.0),
        default=0.0,
        lo=0.0,
        hi=scent_max,
    )
    waste_max = patch_waste_max(cfg)
    waste_initial = _bounded(
        getattr(cfg, "patch_waste_initial", 0.0),
        default=0.0,
        lo=0.0,
        hi=waste_max,
    )
    recovery_min, recovery_max = patch_recovery_activity_min_max(cfg)
    recovery_initial = _bounded(
        getattr(cfg, "jungle_recovery_activity_initial", 0.0),
        default=0.0,
        lo=recovery_min,
        hi=recovery_max,
    )

    arrays = {
        "patch_food_amount": np.where(grid_arr == FOOD, food_cell, empty_cell).astype(np.float32, copy=False),
        "patch_fertility": np.full(shape, fertility_default, dtype=np.float32),
        "patch_cover": cover,
        "patch_shelter": shelter,
        "patch_movement_cost": np.full(shape, move_default, dtype=np.float32),
        "patch_scent": np.full(shape, scent_initial, dtype=np.float32),
        "patch_waste": np.full(shape, waste_initial, dtype=np.float32),
        "patch_recovery_activity": np.full(shape, recovery_initial, dtype=np.float32),
        "respec0b_resource_class": initialize_respec0b_resource_class_array(cfg, shape=shape),
        "mosaic0a_patch_class": initialize_mosaic0a_patch_class_array(cfg, shape=shape),
        "predrisk_fixed1_zone_class": initialize_predrisk_fixed1_zone_class_array(cfg, shape=shape),
        "corridor_cp0_zone_class": initialize_corridor_cp0_zone_class_array(cfg, shape=shape),
        "habacclim_habitat_class": initialize_habacclim_habitat_class_array(cfg, shape=shape),
    }
    if wdl_delayed_hazard_enabled(cfg):
        arrays.update(initialize_wdl_delayed_hazard_arrays(cfg, shape=shape))
    if prediction_demand_enabled(cfg):
        arrays.update(initialize_prediction_demand_arrays(cfg, shape=shape))
    if delayed_resource_depletion_enabled(cfg):
        arrays.update(initialize_delayed_resource_depletion_arrays(cfg, shape=shape))
    if delayed_crowding_waste_enabled(cfg):
        arrays.update(initialize_delayed_crowding_waste_arrays(cfg, shape=shape))
    if mem_max_delayed_choice_enabled(cfg):
        arrays.update(initialize_mem_max_delayed_choice_arrays(cfg, shape=shape))
    apply_roz2a_patch_niche_static_overlay(cfg, arrays)
    apply_respec_n2_capacity_balanced_overlay(cfg, arrays)
    apply_mosaic0a_depletion_recovery_overlay(cfg, arrays)
    apply_predrisk_fixed1_overlay(cfg, arrays)
    apply_corridor_cp0_overlay(cfg, arrays)
    apply_corridor_hc1_homing_cue_overlay(cfg, arrays)
    apply_habacclim_overlay(cfg, arrays)
    if eed_resource_impact_enabled(cfg):
        arrays.update(
            initialize_eed_resource_impact_arrays(
                cfg,
                classes=arrays["respec0b_resource_class"],
            )
        )
        classes = np.asarray(arrays["respec0b_resource_class"], dtype=np.int32)
        a_mask = classes == RESPEC0B_RESOURCE_A
        b_mask = classes == RESPEC0B_RESOURCE_B
        arrays["patch_food_amount"][a_mask] = arrays["eed_resource_a_amount"][
            a_mask
        ]
        arrays["patch_food_amount"][b_mask] = arrays["eed_resource_b_amount"][
            b_mask
        ]
    return arrays


def initialize_wdl_delayed_hazard_arrays(cfg: Any, *, shape: tuple[int, int]) -> Dict[str, np.ndarray]:
    """Create deterministic WDL-1 delayed-hazard state arrays."""
    _ = cfg
    return {
        "patch_delayed_hazard_cue": np.zeros(shape, dtype=np.float32),
        "patch_delayed_hazard_timer": np.full(shape, -1, dtype=np.int32),
        "patch_delayed_hazard_active": np.zeros(shape, dtype=np.float32),
        "patch_delayed_hazard_intensity": np.zeros(shape, dtype=np.float32),
        "patch_delayed_hazard_age": np.zeros(shape, dtype=np.int32),
        "patch_delayed_hazard_event_id": np.zeros(shape, dtype=np.int64),
        "patch_delayed_hazard_prior_cue": np.zeros(shape, dtype=np.int8),
    }


def initialize_prediction_demand_arrays(cfg: Any, *, shape: tuple[int, int]) -> Dict[str, np.ndarray]:
    """Create deterministic Track C variable-delay forecast-demand state arrays."""
    _ = cfg
    return {
        "patch_prediction_demand_cue": np.zeros(shape, dtype=np.float32),
        "patch_prediction_demand_trend": np.zeros(shape, dtype=np.float32),
        "patch_prediction_demand_timer": np.full(shape, -1, dtype=np.int32),
        "patch_prediction_demand_active": np.zeros(shape, dtype=np.float32),
        "patch_prediction_demand_intensity": np.zeros(shape, dtype=np.float32),
        "patch_prediction_demand_age": np.zeros(shape, dtype=np.int32),
        "patch_prediction_demand_event_id": np.zeros(shape, dtype=np.int64),
        "patch_prediction_demand_delay": np.zeros(shape, dtype=np.int32),
        "patch_prediction_demand_prior_cue": np.zeros(shape, dtype=np.int8),
        "patch_prediction_demand_transition_sign": np.zeros(shape, dtype=np.int8),
    }


def _prediction_array_dtype(name: str) -> Any:
    if name in ("patch_prediction_demand_timer", "patch_prediction_demand_age", "patch_prediction_demand_delay"):
        return np.int32
    if name == "patch_prediction_demand_event_id":
        return np.int64
    if name in ("patch_prediction_demand_prior_cue", "patch_prediction_demand_transition_sign"):
        return np.int8
    return np.float32


def _initialize_delayed_consequence_family_arrays(
    shape: tuple[int, int],
    prefix: str,
    *,
    once_name: str,
) -> Dict[str, np.ndarray]:
    return {
        f"{prefix}_cue": np.zeros(shape, dtype=np.float32),
        f"{prefix}_timer": np.full(shape, -1, dtype=np.int32),
        f"{prefix}_active": np.zeros(shape, dtype=np.float32),
        f"{prefix}_intensity": np.zeros(shape, dtype=np.float32),
        f"{prefix}_age": np.zeros(shape, dtype=np.int32),
        f"{prefix}_event_id": np.zeros(shape, dtype=np.int64),
        f"{prefix}_prior_cue": np.zeros(shape, dtype=np.int8),
        once_name: np.zeros(shape, dtype=np.int8),
    }


def initialize_delayed_resource_depletion_arrays(cfg: Any, *, shape: tuple[int, int]) -> Dict[str, np.ndarray]:
    """Create deterministic delayed resource-depletion state arrays."""
    _ = cfg
    return _initialize_delayed_consequence_family_arrays(
        shape,
        "delayed_resource_depletion",
        once_name="delayed_resource_depletion_depleted_once",
    )


def initialize_delayed_crowding_waste_arrays(cfg: Any, *, shape: tuple[int, int]) -> Dict[str, np.ndarray]:
    """Create deterministic delayed overuse/waste penalty state arrays."""
    _ = cfg
    return _initialize_delayed_consequence_family_arrays(
        shape,
        "delayed_crowding_waste",
        once_name="delayed_crowding_waste_waste_applied_once",
    )


def initialize_mem_max_delayed_choice_arrays(cfg: Any, *, shape: tuple[int, int]) -> Dict[str, np.ndarray]:
    """Create deterministic MEM-MAX delayed-choice state arrays."""
    _ = cfg
    return {
        "mem_max_delayed_choice_target_a": np.zeros(shape, dtype=np.float32),
        "mem_max_delayed_choice_target_b": np.zeros(shape, dtype=np.float32),
        "mem_max_delayed_choice_cue_a": np.zeros(shape, dtype=np.float32),
        "mem_max_delayed_choice_cue_b": np.zeros(shape, dtype=np.float32),
        "mem_max_delayed_choice_timer": np.full(shape, -1, dtype=np.int32),
        "mem_max_delayed_choice_age": np.zeros(shape, dtype=np.int32),
        "mem_max_delayed_choice_active": np.zeros(shape, dtype=np.float32),
        "mem_max_delayed_choice_event_id": np.zeros(shape, dtype=np.int64),
        "mem_max_delayed_choice_correct_target": np.zeros(shape, dtype=np.int8),
        "mem_max_delayed_choice_prior_cue": np.zeros(shape, dtype=np.int8),
        "mem_max_delayed_choice_consequence_applied_once": np.zeros(shape, dtype=np.int8),
    }


def validate_patch_world_arrays(arrays: Dict[str, Any], *, shape: tuple[int, int], cfg: Any = None) -> Dict[str, np.ndarray]:
    """Validate and normalize loaded Patch World arrays."""
    out: Dict[str, np.ndarray] = {}
    for name in _PATCH_WORLD_REQUIRED_ARRAY_NAMES:
        if name not in arrays:
            raise ValueError(f"Patch World checkpoint missing array: {name}")
        arr = np.asarray(arrays[name], dtype=np.float32)
        if tuple(arr.shape) != tuple(shape):
            raise ValueError(f"Patch World array {name} has shape {arr.shape}, expected {shape}")
        out[name] = np.ascontiguousarray(arr, dtype=np.float32)

    defaults = initialize_patch_world_arrays(cfg, np.zeros(shape, dtype=np.int8))
    if bool(getattr(cfg, "patch_shelter_from_cover_enable", False)):
        shelter_min = _bounded(getattr(cfg, "patch_shelter_min", 0.0), default=0.0)
        shelter_max = _bounded(getattr(cfg, "patch_shelter_max", 1.0), default=1.0)
        if shelter_max < shelter_min:
            shelter_min, shelter_max = shelter_max, shelter_min
        defaults["patch_shelter"] = np.ascontiguousarray(
            np.clip(out["patch_cover"], shelter_min, shelter_max),
            dtype=np.float32,
        )
    for name in (
        "patch_shelter",
        "patch_movement_cost",
        "patch_waste",
        "patch_recovery_activity",
        "respec0b_resource_class",
        "mosaic0a_patch_class",
        "predrisk_fixed1_zone_class",
        "corridor_cp0_zone_class",
    ):
        if name in arrays:
            arr = np.asarray(arrays[name], dtype=np.float32)
            if tuple(arr.shape) != tuple(shape):
                raise ValueError(f"Patch World array {name} has shape {arr.shape}, expected {shape}")
            out[name] = np.ascontiguousarray(arr, dtype=np.float32)
        else:
            out[name] = np.ascontiguousarray(defaults[name], dtype=np.float32)
    if wdl_delayed_hazard_enabled(cfg):
        wdl_defaults = initialize_wdl_delayed_hazard_arrays(cfg, shape=shape)
        for name in WDL_DELAYED_HAZARD_ARRAY_NAMES:
            if name in arrays:
                dtype = np.float32
                if name in ("patch_delayed_hazard_timer", "patch_delayed_hazard_age"):
                    dtype = np.int32
                elif name == "patch_delayed_hazard_event_id":
                    dtype = np.int64
                elif name == "patch_delayed_hazard_prior_cue":
                    dtype = np.int8
                arr = np.asarray(arrays[name], dtype=dtype)
                if tuple(arr.shape) != tuple(shape):
                    raise ValueError(f"WDL delayed hazard array {name} has shape {arr.shape}, expected {shape}")
                out[name] = np.ascontiguousarray(arr, dtype=dtype)
            else:
                out[name] = np.ascontiguousarray(wdl_defaults[name], dtype=wdl_defaults[name].dtype)
    if prediction_demand_enabled(cfg):
        prediction_defaults = initialize_prediction_demand_arrays(cfg, shape=shape)
        for name in PREDICTION_DEMAND_ARRAY_NAMES:
            if name in arrays:
                dtype = _prediction_array_dtype(name)
                arr = np.asarray(arrays[name], dtype=dtype)
                if tuple(arr.shape) != tuple(shape):
                    raise ValueError(f"Prediction demand array {name} has shape {arr.shape}, expected {shape}")
                out[name] = np.ascontiguousarray(arr, dtype=dtype)
            else:
                out[name] = np.ascontiguousarray(
                    prediction_defaults[name],
                    dtype=prediction_defaults[name].dtype,
                )
    for enabled, names, defaults_fn, label in (
        (
            delayed_resource_depletion_enabled(cfg),
            DELAYED_RESOURCE_DEPLETION_ARRAY_NAMES,
            initialize_delayed_resource_depletion_arrays,
            "delayed resource depletion",
        ),
        (
            delayed_crowding_waste_enabled(cfg),
            DELAYED_CROWDING_WASTE_ARRAY_NAMES,
            initialize_delayed_crowding_waste_arrays,
            "delayed crowding waste",
        ),
    ):
        if not enabled:
            continue
        family_defaults = defaults_fn(cfg, shape=shape)
        for name in names:
            if name in arrays:
                dtype = np.float32
                if name.endswith("_timer") or name.endswith("_age"):
                    dtype = np.int32
                elif name.endswith("_event_id"):
                    dtype = np.int64
                elif name.endswith("_prior_cue") or name.endswith("_once"):
                    dtype = np.int8
                arr = np.asarray(arrays[name], dtype=dtype)
                if tuple(arr.shape) != tuple(shape):
                    raise ValueError(f"{label} array {name} has shape {arr.shape}, expected {shape}")
                out[name] = np.ascontiguousarray(arr, dtype=dtype)
            else:
                out[name] = np.ascontiguousarray(family_defaults[name], dtype=family_defaults[name].dtype)
    if mem_max_delayed_choice_enabled(cfg):
        choice_defaults = initialize_mem_max_delayed_choice_arrays(cfg, shape=shape)
        for name in MEM_MAX_DELAYED_CHOICE_ARRAY_NAMES:
            if name in arrays:
                dtype = np.float32
                if name.endswith("_timer") or name.endswith("_age"):
                    dtype = np.int32
                elif name.endswith("_event_id"):
                    dtype = np.int64
                elif name.endswith("_target") or name.endswith("_cue") or name.endswith("_once"):
                    dtype = np.int8
                if name in ("mem_max_delayed_choice_target_a", "mem_max_delayed_choice_target_b", "mem_max_delayed_choice_active"):
                    dtype = np.float32
                if name in ("mem_max_delayed_choice_cue_a", "mem_max_delayed_choice_cue_b"):
                    dtype = np.float32
                arr = np.asarray(arrays[name], dtype=dtype)
                if tuple(arr.shape) != tuple(shape):
                    raise ValueError(f"MEM-MAX delayed-choice array {name} has shape {arr.shape}, expected {shape}")
                out[name] = np.ascontiguousarray(arr, dtype=dtype)
            else:
                out[name] = np.ascontiguousarray(choice_defaults[name], dtype=choice_defaults[name].dtype)
    if eed_resource_impact_enabled(cfg):
        out.update(
            validate_eed_resource_impact_arrays(
                arrays,
                shape=shape,
                cfg=cfg,
            )
        )
    return out


def initialize_patch_world_runtime_fields(env: Any) -> None:
    """Initialize derived and pending Patch World runtime arrays/counters."""
    shape = tuple(getattr(env, "grid").shape)
    env.local_density_map = np.zeros(shape, dtype=np.float32)
    env.patch_scent_emit_pending = np.zeros(shape, dtype=np.float32)
    env.patch_waste_emit_pending = np.zeros(shape, dtype=np.float32)
    if respec0b_resource_specialization_enabled(getattr(env, "cfg", None)):
        capacity = int(getattr(env, "slot_to_agent_id").shape[0])
        env.respec0b_resource_active_y = np.full((capacity,), -1, dtype=np.int32)
        env.respec0b_resource_active_x = np.full((capacity,), -1, dtype=np.int32)
        env.respec0b_resource_active_class = np.zeros((capacity,), dtype=np.int32)
        env.respec0b_resource_handling_progress = np.zeros((capacity,), dtype=np.float32)
    env.eed_resource_impact_stats_last = {
        "enabled": int(eed_resource_impact_enabled(getattr(env, "cfg", None))),
        "impact_applied": int(
            eed_resource_impact_applied(getattr(env, "cfg", None))
        ),
        "resource_a_pending_before": 0.0,
        "resource_b_pending_before": 0.0,
        "resource_a_impact_settled": 0.0,
        "resource_b_impact_settled": 0.0,
        "resource_a_effective_renewal": 0.0,
        "resource_b_effective_renewal": 0.0,
    }
    reset_patch_world_trace_counters(env)
    reset_patch_world_regrowth_consequence_counters(env)
    reset_patch_world_recovery_counters(env)
    reset_wdl_delayed_hazard_counters(env)
    reset_prediction_demand_counters(env)
    reset_delayed_resource_waste_counters(env)
    reset_mem_max_delayed_choice_counters(env)


def reset_wdl_delayed_hazard_counters(env: Any) -> None:
    """Reset per-tick WDL delayed-hazard diagnostics."""
    env.wdl_delayed_hazard_cue_cell_count_last = 0
    env.wdl_delayed_hazard_active_cell_count_last = 0
    env.wdl_delayed_hazard_event_count_last = 0
    env.wdl_delayed_hazard_cue_without_active_count_last = 0
    env.wdl_delayed_hazard_active_without_prior_cue_count_last = 0
    env.wdl_delayed_hazard_cue_to_active_delay_sum_last = 0.0
    env.wdl_delayed_hazard_cue_to_active_delay_count_last = 0
    env.wdl_delayed_hazard_activated_event_ids_last = []


def reset_prediction_demand_counters(env: Any) -> None:
    """Reset per-tick Track C prediction-demand diagnostics."""
    env.prediction_demand_cue_cell_count_last = 0
    env.prediction_demand_active_cell_count_last = 0
    env.prediction_demand_event_count_last = 0
    env.prediction_demand_transition_count_last = 0
    env.prediction_demand_cue_ambiguity_sum_last = 0.0
    env.prediction_demand_cue_ambiguity_count_last = 0
    env.prediction_demand_cue_to_consequence_delay_sum_last = 0.0
    env.prediction_demand_cue_to_consequence_delay_count_last = 0
    env.prediction_demand_cue_without_consequence_count_last = 0
    env.prediction_demand_consequence_without_prior_cue_count_last = 0
    env.prediction_demand_regrowth_multiplier_mean_last = 1.0
    env.prediction_demand_resource_consequence_total_last = 0.0


def reset_delayed_resource_waste_counters(env: Any) -> None:
    """Reset per-tick delayed consequence family diagnostics."""
    for prefix in ("delayed_resource_depletion", "delayed_crowding_waste"):
        setattr(env, f"{prefix}_cue_cell_count_last", 0)
        setattr(env, f"{prefix}_hidden_cell_count_last", 0)
        setattr(env, f"{prefix}_active_cell_count_last", 0)
        setattr(env, f"{prefix}_event_count_last", 0)
        setattr(env, f"{prefix}_cue_without_active_count_last", 0)
        setattr(env, f"{prefix}_active_without_prior_cue_count_last", 0)
        setattr(env, f"{prefix}_cue_to_active_delay_sum_last", 0.0)
        setattr(env, f"{prefix}_cue_to_active_delay_count_last", 0)
        setattr(env, f"{prefix}_consequence_count_last", 0)
        setattr(env, f"{prefix}_consequence_total_last", 0.0)
        setattr(env, f"{prefix}_activated_event_ids_last", [])


def reset_mem_max_delayed_choice_counters(env: Any) -> None:
    """Reset per-tick MEM-MAX delayed-choice diagnostics."""
    env.mem_max_delayed_choice_event_count_last = 0
    env.mem_max_delayed_choice_cue_step_active_last = 0
    env.mem_max_delayed_choice_hidden_step_active_last = 0
    env.mem_max_delayed_choice_consequence_step_active_last = 0
    env.mem_max_delayed_choice_target_a_cue_count_last = 0
    env.mem_max_delayed_choice_target_b_cue_count_last = 0
    env.mem_max_delayed_choice_correct_arrival_count_last = 0
    env.mem_max_delayed_choice_incorrect_arrival_count_last = 0
    env.mem_max_delayed_choice_no_choice_count_last = 0
    env.mem_max_delayed_choice_consequence_exposure_count_last = 0
    env.mem_max_delayed_choice_damage_total_last = 0.0
    env.mem_max_delayed_choice_damage_prevented_total_last = 0.0
    env.mem_max_delayed_choice_expected_consequence_total_last = 0.0
    env.mem_max_delayed_choice_applied_health_damage_total_last = 0.0
    env.mem_max_delayed_choice_applied_energy_cost_total_last = 0.0
    env.mem_max_delayed_choice_consequence_applied_count_last = 0
    env.mem_max_delayed_choice_consequence_eligible_count_last = 0
    env.mem_max_delayed_choice_duplicate_prevented_count_last = 0
    env.mem_max_delayed_choice_correct_health_damage_total_last = 0.0
    env.mem_max_delayed_choice_incorrect_health_damage_total_last = 0.0
    env.mem_max_delayed_choice_no_choice_health_damage_total_last = 0.0
    env.mem_max_delayed_choice_target_a_arrival_count_last = 0
    env.mem_max_delayed_choice_target_b_arrival_count_last = 0
    env.mem_max_delayed_choice_any_target_reachable_fraction_last = 0.0
    env.mem_max_delayed_choice_shortcut_exposure_last = 0


def reset_patch_world_trace_counters(env: Any) -> None:
    """Reset per-tick Patch World trace diagnostics."""
    env.patch_scent_emitted_total_last = 0.0
    env.patch_scent_decayed_total_last = 0.0
    env.patch_waste_emitted_total_last = 0.0
    env.patch_waste_decayed_total_last = 0.0


def reset_patch_world_regrowth_consequence_counters(env: Any) -> None:
    """Reset per-tick Patch World regrowth consequence diagnostics."""
    env.patch_regrowth_base_total_last = 0.0
    env.patch_regrowth_effective_total_last = 0.0
    env.patch_regrowth_suppressed_total_last = 0.0
    env.patch_regrowth_multiplier_mean_last = 1.0
    env.patch_regrowth_multiplier_min_last = 1.0
    env.patch_regrowth_multiplier_p10_last = 1.0
    env.patch_regrowth_multiplier_p90_last = 1.0
    env.patch_waste_regrowth_penalty_mean_last = 0.0
    env.patch_waste_regrowth_penalty_max_last = 0.0
    env.patch_density_regrowth_penalty_mean_last = 0.0
    env.patch_density_regrowth_penalty_max_last = 0.0
    env.patch_high_waste_regrowth_suppression_frac_last = 0.0
    env.patch_high_density_regrowth_suppression_frac_last = 0.0
    env.patch_regrowth_after_waste_total_last = 0.0
    env.patch_regrowth_after_recovery_total_last = 0.0
    env.patch_recovery_added_regrowth_total_last = 0.0
    env.jungle_recovery_regrowth_boost_total_last = 0.0
    env.jungle_recovery_regrowth_boost_mean_last = 0.0
    env.jungle_recovery_regrowth_boost_max_last = 0.0
    env.jungle_recovery_regrowth_boost_p90_last = 0.0


def reset_patch_world_recovery_counters(env: Any) -> None:
    """Reset per-tick ecological recovery diagnostics."""
    env.patch_recovery_positive_total_last = 0.0
    env.patch_recovery_negative_total_last = 0.0
    env.patch_recovery_decay_total_last = 0.0


def _wdl_int(cfg: Any, name: str, default: int, *, lo: int = 0, hi: int | None = None) -> int:
    try:
        out = int(getattr(cfg, name, default))
    except Exception:
        out = int(default)
    out = max(int(lo), out)
    if hi is not None:
        out = min(int(hi), out)
    return int(out)


def _wdl_seed(env: Any) -> int:
    try:
        return int(getattr(env, "base_seed"))
    except Exception:
        pass
    try:
        state = getattr(getattr(env, "rng", None), "bit_generator", None).state
        return int(abs(hash(str(state))) % (2**31 - 1))
    except Exception:
        return 0


def _stable_wdl_event_coord(env: Any, event_index: int) -> tuple[int, int, int]:
    import hashlib

    grid = np.asarray(getattr(env, "grid"))
    h, w = tuple(grid.shape)
    seed = _wdl_seed(env)
    material = f"wdl1:{seed}:{int(event_index)}:{int(h)}:{int(w)}".encode("ascii")
    digest = hashlib.sha256(material).digest()
    y = int.from_bytes(digest[0:8], "big") % max(1, int(h))
    x = int.from_bytes(digest[8:16], "big") % max(1, int(w))
    event_id = int.from_bytes(digest[16:24], "big") & 0x7FFFFFFFFFFFFFFF
    return int(y), int(x), int(event_id)


def _stable_prediction_demand_event(env: Any, event_index: int) -> tuple[int, int, int, int, int, float]:
    import hashlib

    cfg = getattr(env, "cfg", None)
    grid = np.asarray(getattr(env, "grid"))
    h, w = tuple(grid.shape)
    delay_min = _wdl_int(cfg, "prediction_demand_delay_min", 4, lo=1)
    delay_max = _wdl_int(cfg, "prediction_demand_delay_max", 10, lo=delay_min)
    delay_span = max(1, int(delay_max) - int(delay_min) + 1)
    seed = _wdl_seed(env)
    material = f"prediction-demand-v1:{seed}:{int(event_index)}:{int(h)}:{int(w)}".encode("ascii")
    digest = hashlib.sha256(material).digest()
    y = int.from_bytes(digest[0:8], "big") % max(1, int(h))
    x = int.from_bytes(digest[8:16], "big") % max(1, int(w))
    event_id = int.from_bytes(digest[16:24], "big") & 0x7FFFFFFFFFFFFFFF
    delay = int(delay_min) + (int.from_bytes(digest[24:28], "big") % int(delay_span))
    sign = 1 if (int.from_bytes(digest[28:30], "big") % 2 == 0) else -1
    noise = _bounded(getattr(cfg, "prediction_demand_cue_noise", 0.25), default=0.25, lo=0.0, hi=1.0)
    jitter_raw = (int.from_bytes(digest[30:32], "big") / float(0xFFFF)) * 2.0 - 1.0
    trend = float(sign) * (1.0 - float(noise)) + float(jitter_raw) * float(noise)
    return int(y), int(x), int(event_id), int(delay), int(sign), float(np.clip(trend, -1.0, 1.0))


def _wdl_event_mask(shape: tuple[int, int], y: int, x: int, radius: int) -> np.ndarray:
    h, w = tuple(int(v) for v in shape)
    r = int(max(0, radius))
    y0 = max(0, int(y) - r)
    y1 = min(h, int(y) + r + 1)
    x0 = max(0, int(x) - r)
    x1 = min(w, int(x) + r + 1)
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def apply_wdl_delayed_hazard_step(env: Any, abs_step: int) -> Dict[str, float]:
    """Advance the WDL-1 delayed-hazard substrate for one tick."""
    cfg = getattr(env, "cfg", None)
    reset_wdl_delayed_hazard_counters(env)
    if not wdl_delayed_hazard_enabled(cfg):
        return wdl_delayed_hazard_metrics(env)

    shape = tuple(np.asarray(getattr(env, "grid")).shape)
    cue = np.asarray(getattr(env, "patch_delayed_hazard_cue"), dtype=np.float32)
    timer = np.asarray(getattr(env, "patch_delayed_hazard_timer"), dtype=np.int32)
    active = np.asarray(getattr(env, "patch_delayed_hazard_active"), dtype=np.float32)
    intensity = np.asarray(getattr(env, "patch_delayed_hazard_intensity"), dtype=np.float32)
    age = np.asarray(getattr(env, "patch_delayed_hazard_age"), dtype=np.int32)
    event_ids = np.asarray(getattr(env, "patch_delayed_hazard_event_id"), dtype=np.int64)
    prior_cue = np.asarray(getattr(env, "patch_delayed_hazard_prior_cue"), dtype=np.int8)

    delay = _wdl_int(cfg, "wdl_delayed_hazard_delay_steps", 8, lo=1)
    cue_duration = _wdl_int(cfg, "wdl_delayed_hazard_cue_duration_steps", 4, lo=1, hi=delay)
    active_duration = _wdl_int(cfg, "wdl_delayed_hazard_active_duration_steps", 6, lo=1)
    radius = _wdl_int(cfg, "wdl_delayed_hazard_radius", 1, lo=0)
    period = _wdl_int(cfg, "wdl_delayed_hazard_event_period_steps", 20, lo=delay + active_duration + 1)
    hazard_intensity = _bounded(getattr(cfg, "wdl_delayed_hazard_intensity", 0.10), default=0.10, lo=0.0, hi=1.0)
    cue_enabled = bool(getattr(cfg, "wdl_delayed_hazard_cue_enable", True))

    # Progress existing cells first so a newly scheduled cue starts at age 0.
    occupied = timer >= 0
    age[occupied] += np.int32(1)
    timer[occupied] -= np.int32(1)
    cue[:, :] = 0.0
    active[:, :] = 0.0

    cue_mask = occupied & (age < cue_duration)
    hidden_mask = occupied & (age >= cue_duration) & (age < delay)
    active_mask = occupied & (age >= delay) & (age < delay + active_duration)
    done_mask = occupied & (age >= delay + active_duration)

    if cue_enabled:
        cue[cue_mask] = np.float32(1.0)
    prior_cue[cue_mask | hidden_mask | active_mask] = np.int8(1)
    active[active_mask] = np.float32(1.0)
    intensity[active_mask] = np.float32(hazard_intensity)
    intensity[~active_mask & ~cue_mask & ~hidden_mask] = np.float32(0.0)

    if bool(np.any(active_mask)):
        env.wdl_delayed_hazard_cue_to_active_delay_sum_last = float(int(delay) * int(np.sum(active_mask)))
        env.wdl_delayed_hazard_cue_to_active_delay_count_last = int(np.sum(active_mask))
        activated_ids = sorted(int(x) for x in np.unique(event_ids[active_mask]).tolist() if int(x) > 0)
        env.wdl_delayed_hazard_activated_event_ids_last = activated_ids

    if bool(np.any(done_mask)):
        cue[done_mask] = np.float32(0.0)
        active[done_mask] = np.float32(0.0)
        intensity[done_mask] = np.float32(0.0)
        age[done_mask] = np.int32(0)
        timer[done_mask] = np.int32(-1)
        event_ids[done_mask] = np.int64(0)
        prior_cue[done_mask] = np.int8(0)

    if bool(getattr(cfg, "wdl_delayed_hazard_deterministic_schedule_enable", True)):
        step = int(abs_step)
        if step >= 0 and step % int(period) == 0:
            event_index = step // int(period)
            y, x, event_id = _stable_wdl_event_coord(env, event_index)
            mask = _wdl_event_mask(shape, y, x, radius) & (timer < 0)
            if bool(np.any(mask)):
                cue[mask] = np.float32(1.0 if cue_enabled else 0.0)
                timer[mask] = np.int32(delay + active_duration)
                active[mask] = np.float32(0.0)
                intensity[mask] = np.float32(hazard_intensity)
                age[mask] = np.int32(0)
                event_ids[mask] = np.int64(event_id)
                prior_cue[mask] = np.int8(1)
                env.wdl_delayed_hazard_event_count_last = int(np.sum(mask))

    env.wdl_delayed_hazard_cue_cell_count_last = int(np.sum(cue > 0.0))
    env.wdl_delayed_hazard_active_cell_count_last = int(np.sum(active > 0.0))
    env.wdl_delayed_hazard_cue_without_active_count_last = int(np.sum((prior_cue > 0) & (active <= 0.0)))
    env.wdl_delayed_hazard_active_without_prior_cue_count_last = int(np.sum((active > 0.0) & (prior_cue <= 0)))
    return wdl_delayed_hazard_metrics(env)


def apply_prediction_demand_step(env: Any, abs_step: int) -> Dict[str, float]:
    """Advance the default-off variable-delay forecast-demand substrate for one tick."""
    cfg = getattr(env, "cfg", None)
    reset_prediction_demand_counters(env)
    if not prediction_demand_enabled(cfg):
        return prediction_demand_metrics(env)

    shape = tuple(np.asarray(getattr(env, "grid")).shape)
    cue = np.asarray(getattr(env, "patch_prediction_demand_cue"), dtype=np.float32)
    trend = np.asarray(getattr(env, "patch_prediction_demand_trend"), dtype=np.float32)
    timer = np.asarray(getattr(env, "patch_prediction_demand_timer"), dtype=np.int32)
    active = np.asarray(getattr(env, "patch_prediction_demand_active"), dtype=np.float32)
    intensity = np.asarray(getattr(env, "patch_prediction_demand_intensity"), dtype=np.float32)
    age = np.asarray(getattr(env, "patch_prediction_demand_age"), dtype=np.int32)
    event_ids = np.asarray(getattr(env, "patch_prediction_demand_event_id"), dtype=np.int64)
    delays = np.asarray(getattr(env, "patch_prediction_demand_delay"), dtype=np.int32)
    prior_cue = np.asarray(getattr(env, "patch_prediction_demand_prior_cue"), dtype=np.int8)
    signs = np.asarray(getattr(env, "patch_prediction_demand_transition_sign"), dtype=np.int8)

    active_duration = _wdl_int(cfg, "prediction_demand_active_duration", 4, lo=1)
    delay_min = _wdl_int(cfg, "prediction_demand_delay_min", 4, lo=1)
    delay_max = _wdl_int(cfg, "prediction_demand_delay_max", 10, lo=delay_min)
    radius = _wdl_int(cfg, "prediction_demand_radius", 1, lo=0)
    period = _wdl_int(cfg, "prediction_demand_event_period_steps", 18, lo=delay_max + active_duration + 1)
    strength = _bounded(getattr(cfg, "prediction_demand_transition_strength", 0.35), default=0.35, lo=0.0, hi=1.0)

    occupied = timer >= 0
    age[occupied] += np.int32(1)
    timer[occupied] -= np.int32(1)
    cue[:, :] = 0.0
    active[:, :] = 0.0
    intensity[:, :] = 0.0

    cue_mask = occupied & (timer > 0)
    active_mask = occupied & (timer <= 0) & (age < (delays + active_duration))
    done_mask = occupied & (age >= (delays + active_duration))

    cue[cue_mask] = np.float32(1.0)
    prior_cue[cue_mask | active_mask] = np.int8(1)
    active[active_mask] = np.float32(np.float32(strength)) * signs[active_mask].astype(np.float32)
    intensity[active_mask] = np.float32(strength)

    if bool(np.any(active_mask)):
        active_delays = delays[active_mask].astype(np.float64)
        env.prediction_demand_cue_to_consequence_delay_sum_last = float(np.sum(active_delays, dtype=np.float64))
        env.prediction_demand_cue_to_consequence_delay_count_last = int(active_delays.size)
        env.prediction_demand_transition_count_last = int(np.sum(active_mask))

    if bool(np.any(done_mask)):
        cue[done_mask] = np.float32(0.0)
        trend[done_mask] = np.float32(0.0)
        active[done_mask] = np.float32(0.0)
        intensity[done_mask] = np.float32(0.0)
        age[done_mask] = np.int32(0)
        timer[done_mask] = np.int32(-1)
        event_ids[done_mask] = np.int64(0)
        delays[done_mask] = np.int32(0)
        prior_cue[done_mask] = np.int8(0)
        signs[done_mask] = np.int8(0)

    if bool(getattr(cfg, "prediction_demand_deterministic_schedule_enable", True)):
        step = int(abs_step)
        if step >= 0 and step % int(period) == 0:
            event_index = step // int(period)
            y, x, event_id, delay, sign, trend_value = _stable_prediction_demand_event(env, event_index)
            mask = _wdl_event_mask(shape, y, x, radius) & (timer < 0)
            if bool(np.any(mask)):
                cue[mask] = np.float32(1.0)
                trend[mask] = np.float32(trend_value)
                timer[mask] = np.int32(delay)
                active[mask] = np.float32(0.0)
                intensity[mask] = np.float32(strength)
                age[mask] = np.int32(0)
                event_ids[mask] = np.int64(event_id)
                delays[mask] = np.int32(delay)
                prior_cue[mask] = np.int8(1)
                signs[mask] = np.int8(sign)
                env.prediction_demand_event_count_last = int(np.sum(mask))

    cue_active = cue > 0.0
    env.prediction_demand_cue_cell_count_last = int(np.sum(cue_active))
    env.prediction_demand_active_cell_count_last = int(np.sum(active != 0.0))
    if bool(np.any(cue_active)):
        ambiguity = 1.0 - np.abs(trend[cue_active].astype(np.float32))
        env.prediction_demand_cue_ambiguity_sum_last = float(np.sum(ambiguity, dtype=np.float64))
        env.prediction_demand_cue_ambiguity_count_last = int(ambiguity.size)
    env.prediction_demand_cue_without_consequence_count_last = int(np.sum((prior_cue > 0) & (active == 0.0)))
    env.prediction_demand_consequence_without_prior_cue_count_last = int(np.sum((active != 0.0) & (prior_cue <= 0)))
    return prediction_demand_metrics(env)


def prediction_demand_regrowth_multiplier_maps(env: Any) -> Dict[str, np.ndarray]:
    cfg = getattr(env, "cfg", None)
    shape = tuple(np.asarray(getattr(env, "grid")).shape)
    ones = np.ones(shape, dtype=np.float32)
    zeros = np.zeros(shape, dtype=np.float32)
    if not prediction_demand_enabled(cfg) or not bool(getattr(cfg, "prediction_demand_resource_consequence_enable", True)):
        return {"multiplier": ones, "effect": zeros}

    active = np.nan_to_num(np.asarray(getattr(env, "patch_prediction_demand_active"), dtype=np.float32))
    multiplier = np.clip(1.0 + active, 0.0, 2.0).astype(np.float32, copy=False)
    effect = multiplier - np.float32(1.0)
    env.prediction_demand_regrowth_multiplier_mean_last = float(np.mean(multiplier)) if multiplier.size else 1.0
    env.prediction_demand_resource_consequence_total_last = float(np.sum(np.abs(effect), dtype=np.float64))
    return {
        "multiplier": np.ascontiguousarray(np.nan_to_num(multiplier), dtype=np.float32),
        "effect": np.ascontiguousarray(np.nan_to_num(effect), dtype=np.float32),
    }


def prediction_demand_metrics(env: Any) -> Dict[str, float]:
    """Return Track C prediction-demand state metrics for metrics.csv."""
    cfg = getattr(env, "cfg", None)
    if not prediction_demand_enabled(cfg):
        return {
            "prediction_demand_enable": 0,
            "prediction_demand_observation_enable": 0,
            "prediction_demand_cue_cell_count": 0.0,
            "prediction_demand_active_cell_count": 0.0,
            "prediction_demand_event_count": 0.0,
            "prediction_demand_transition_count": 0.0,
            "prediction_demand_cue_ambiguity_mean": 0.0,
            "prediction_demand_cue_to_consequence_delay_mean": 0.0,
            "prediction_demand_cue_without_consequence_count": 0.0,
            "prediction_demand_consequence_without_prior_cue_count": 0.0,
            "prediction_demand_pre_consequence_agent_context_count": 0.0,
            "prediction_demand_consequence_agent_exposure_count": 0.0,
            "prediction_demand_resource_consequence_total": 0.0,
            "prediction_demand_direct_prediction_reward_active": 0,
            "prediction_demand_direct_error_reward_active": 0,
        }
    ambiguity_count = int(getattr(env, "prediction_demand_cue_ambiguity_count_last", 0) or 0)
    ambiguity_sum = float(getattr(env, "prediction_demand_cue_ambiguity_sum_last", 0.0) or 0.0)
    delay_count = int(getattr(env, "prediction_demand_cue_to_consequence_delay_count_last", 0) or 0)
    delay_sum = float(getattr(env, "prediction_demand_cue_to_consequence_delay_sum_last", 0.0) or 0.0)
    return {
        "prediction_demand_enable": 1,
        "prediction_demand_observation_enable": int(prediction_demand_observation_enabled(cfg)),
        "prediction_demand_cue_cell_count": float(getattr(env, "prediction_demand_cue_cell_count_last", 0)),
        "prediction_demand_active_cell_count": float(getattr(env, "prediction_demand_active_cell_count_last", 0)),
        "prediction_demand_event_count": float(getattr(env, "prediction_demand_event_count_last", 0)),
        "prediction_demand_transition_count": float(getattr(env, "prediction_demand_transition_count_last", 0)),
        "prediction_demand_cue_ambiguity_mean": ambiguity_sum / float(ambiguity_count) if ambiguity_count > 0 else 0.0,
        "prediction_demand_cue_to_consequence_delay_mean": delay_sum / float(delay_count) if delay_count > 0 else 0.0,
        "prediction_demand_cue_without_consequence_count": float(
            getattr(env, "prediction_demand_cue_without_consequence_count_last", 0)
        ),
        "prediction_demand_consequence_without_prior_cue_count": float(
            getattr(env, "prediction_demand_consequence_without_prior_cue_count_last", 0)
        ),
        "prediction_demand_pre_consequence_agent_context_count": 0.0,
        "prediction_demand_consequence_agent_exposure_count": 0.0,
        "prediction_demand_resource_consequence_total": float(
            getattr(env, "prediction_demand_resource_consequence_total_last", 0.0)
        ),
        "prediction_demand_direct_prediction_reward_active": 0,
        "prediction_demand_direct_error_reward_active": 0,
    }


def wdl_delayed_hazard_metrics(env: Any) -> Dict[str, float]:
    """Return WDL delayed-hazard state metrics for metrics.csv."""
    cfg = getattr(env, "cfg", None)
    if not wdl_delayed_hazard_enabled(cfg):
        return {
            "wdl_delayed_hazard_enable": 0,
            "wdl_delayed_hazard_cue_cell_count": 0.0,
            "wdl_delayed_hazard_active_cell_count": 0.0,
            "wdl_delayed_hazard_event_count": 0.0,
            "wdl_delayed_hazard_cue_to_active_delay_mean": 0.0,
            "wdl_delayed_hazard_cue_without_active_count": 0.0,
            "wdl_delayed_hazard_active_without_prior_cue_count": 0.0,
        }
    delay_count = int(getattr(env, "wdl_delayed_hazard_cue_to_active_delay_count_last", 0) or 0)
    delay_sum = float(getattr(env, "wdl_delayed_hazard_cue_to_active_delay_sum_last", 0.0) or 0.0)
    return {
        "wdl_delayed_hazard_enable": 1,
        "wdl_delayed_hazard_cue_cell_count": float(getattr(env, "wdl_delayed_hazard_cue_cell_count_last", 0)),
        "wdl_delayed_hazard_active_cell_count": float(getattr(env, "wdl_delayed_hazard_active_cell_count_last", 0)),
        "wdl_delayed_hazard_event_count": float(getattr(env, "wdl_delayed_hazard_event_count_last", 0)),
        "wdl_delayed_hazard_cue_to_active_delay_mean": delay_sum / float(delay_count) if delay_count > 0 else 0.0,
        "wdl_delayed_hazard_cue_without_active_count": float(
            getattr(env, "wdl_delayed_hazard_cue_without_active_count_last", 0)
        ),
        "wdl_delayed_hazard_active_without_prior_cue_count": float(
            getattr(env, "wdl_delayed_hazard_active_without_prior_cue_count_last", 0)
        ),
    }


def _stable_memory_event_coord(env: Any, event_index: int, family: str) -> tuple[int, int, int]:
    import hashlib

    grid = np.asarray(getattr(env, "grid"))
    h, w = tuple(grid.shape)
    seed = _wdl_seed(env)
    material = f"{family}:{seed}:{int(event_index)}:{int(h)}:{int(w)}".encode("ascii")
    digest = hashlib.sha256(material).digest()
    y = int.from_bytes(digest[0:8], "big") % max(1, int(h))
    x = int.from_bytes(digest[8:16], "big") % max(1, int(w))
    event_id = int.from_bytes(digest[16:24], "big") & 0x7FFFFFFFFFFFFFFF
    return int(y), int(x), int(event_id)


def _memory_family_state_step(
    env: Any,
    *,
    abs_step: int,
    prefix: str,
    delay_name: str,
    cue_duration_name: str,
    active_duration_name: str,
    period_name: str,
    radius_name: str,
    intensity_name: str,
    intensity_default: float,
    schedule_enable_name: str,
    trigger_mask: np.ndarray | None = None,
) -> Dict[str, Any]:
    cfg = getattr(env, "cfg", None)
    shape = tuple(np.asarray(getattr(env, "grid")).shape)
    cue = np.asarray(getattr(env, f"{prefix}_cue"), dtype=np.float32)
    timer = np.asarray(getattr(env, f"{prefix}_timer"), dtype=np.int32)
    active = np.asarray(getattr(env, f"{prefix}_active"), dtype=np.float32)
    intensity = np.asarray(getattr(env, f"{prefix}_intensity"), dtype=np.float32)
    age = np.asarray(getattr(env, f"{prefix}_age"), dtype=np.int32)
    event_ids = np.asarray(getattr(env, f"{prefix}_event_id"), dtype=np.int64)
    prior_cue = np.asarray(getattr(env, f"{prefix}_prior_cue"), dtype=np.int8)
    once_name = (
        "delayed_resource_depletion_depleted_once"
        if prefix == "delayed_resource_depletion"
        else "delayed_crowding_waste_waste_applied_once"
    )
    once = np.asarray(getattr(env, once_name), dtype=np.int8)

    delay = _wdl_int(cfg, delay_name, 8, lo=1)
    cue_duration = _wdl_int(cfg, cue_duration_name, 4, lo=1, hi=delay)
    active_duration = _wdl_int(cfg, active_duration_name, 6, lo=1)
    radius = _wdl_int(cfg, radius_name, 1, lo=0)
    period = _wdl_int(cfg, period_name, 20, lo=delay + active_duration + 1)
    family_intensity = _bounded(getattr(cfg, intensity_name, intensity_default), default=intensity_default, lo=0.0)

    occupied = timer >= 0
    age[occupied] += np.int32(1)
    timer[occupied] -= np.int32(1)
    cue[:, :] = 0.0
    active[:, :] = 0.0

    cue_mask = occupied & (age < cue_duration)
    hidden_mask = occupied & (age >= cue_duration) & (age < delay)
    active_mask = occupied & (age >= delay) & (age < delay + active_duration)
    done_mask = occupied & (age >= delay + active_duration)

    cue[cue_mask] = np.float32(1.0)
    prior_cue[cue_mask | hidden_mask | active_mask] = np.int8(1)
    active[active_mask] = np.float32(1.0)
    intensity[active_mask] = np.float32(family_intensity)
    intensity[~active_mask & ~cue_mask & ~hidden_mask] = np.float32(0.0)

    if bool(np.any(active_mask)):
        setattr(env, f"{prefix}_cue_to_active_delay_sum_last", float(int(delay) * int(np.sum(active_mask))))
        setattr(env, f"{prefix}_cue_to_active_delay_count_last", int(np.sum(active_mask)))
        activated_ids = sorted(int(x) for x in np.unique(event_ids[active_mask]).tolist() if int(x) > 0)
        setattr(env, f"{prefix}_activated_event_ids_last", activated_ids)

    if bool(np.any(done_mask)):
        cue[done_mask] = np.float32(0.0)
        active[done_mask] = np.float32(0.0)
        intensity[done_mask] = np.float32(0.0)
        age[done_mask] = np.int32(0)
        timer[done_mask] = np.int32(-1)
        event_ids[done_mask] = np.int64(0)
        prior_cue[done_mask] = np.int8(0)
        once[done_mask] = np.int8(0)

    if bool(getattr(cfg, schedule_enable_name, True)):
        step = int(abs_step)
        if step >= 0 and step % int(period) == 0:
            event_index = step // int(period)
            if trigger_mask is None:
                y, x, event_id = _stable_memory_event_coord(env, event_index, prefix)
                mask = _wdl_event_mask(shape, y, x, radius) & (timer < 0)
            else:
                base = np.asarray(trigger_mask, dtype=bool) & (timer < 0)
                if bool(np.any(base)):
                    yy, xx = np.where(base)
                    scores = np.asarray(getattr(env, "patch_waste"), dtype=np.float32)[base]
                    choice = int(np.argmax(scores))
                    y = int(yy[choice])
                    x = int(xx[choice])
                    event_id = _stable_memory_event_coord(env, event_index, prefix)[2]
                    mask = _wdl_event_mask(shape, y, x, radius) & (timer < 0)
                else:
                    mask = np.zeros(shape, dtype=bool)
                    event_id = 0
            if bool(np.any(mask)):
                cue[mask] = np.float32(1.0)
                timer[mask] = np.int32(delay)
                active[mask] = np.float32(0.0)
                intensity[mask] = np.float32(family_intensity)
                age[mask] = np.int32(0)
                event_ids[mask] = np.int64(event_id)
                prior_cue[mask] = np.int8(1)
                once[mask] = np.int8(0)
                setattr(env, f"{prefix}_event_count_last", int(np.sum(mask)))

    cue_cells = int(np.sum(cue > 0.0))
    hidden_cells = int(np.sum((prior_cue > 0) & (cue <= 0.0) & (active <= 0.0) & (timer >= 0)))
    active_cells = int(np.sum(active > 0.0))
    setattr(env, f"{prefix}_cue_cell_count_last", cue_cells)
    setattr(env, f"{prefix}_hidden_cell_count_last", hidden_cells)
    setattr(env, f"{prefix}_active_cell_count_last", active_cells)
    setattr(env, f"{prefix}_cue_without_active_count_last", int(np.sum((prior_cue > 0) & (active <= 0.0))))
    setattr(env, f"{prefix}_active_without_prior_cue_count_last", int(np.sum((active > 0.0) & (prior_cue <= 0))))
    return {
        "active_mask": active > 0.0,
        "new_active_mask": (active > 0.0) & (once <= 0),
        "intensity": intensity,
        "once": once,
    }


def apply_delayed_resource_depletion_step(env: Any, abs_step: int) -> Dict[str, float]:
    """Advance default-off delayed resource depletion and apply active food depletion."""
    cfg = getattr(env, "cfg", None)
    if not delayed_resource_depletion_enabled(cfg):
        return delayed_resource_depletion_metrics(env)
    st = _memory_family_state_step(
        env,
        abs_step=int(abs_step),
        prefix="delayed_resource_depletion",
        delay_name="delayed_resource_depletion_delay_steps",
        cue_duration_name="delayed_resource_depletion_cue_duration_steps",
        active_duration_name="delayed_resource_depletion_active_duration_steps",
        period_name="delayed_resource_depletion_event_period_steps",
        radius_name="delayed_resource_depletion_radius",
        intensity_name="delayed_resource_depletion_food_depletion_fraction",
        intensity_default=0.50,
        schedule_enable_name="delayed_resource_depletion_deterministic_schedule_enable",
    )
    active_new = np.asarray(st["new_active_mask"], dtype=bool)
    if bool(np.any(active_new)):
        amount = np.asarray(getattr(env, "patch_food_amount"), dtype=np.float32)
        frac = np.clip(np.asarray(st["intensity"], dtype=np.float32), 0.0, 1.0)
        before = amount.copy()
        amount[active_new] = amount[active_new] * (np.float32(1.0) - frac[active_new])
        delta = before - amount
        st["once"][active_new] = np.int8(1)
        setattr(env, "delayed_resource_depletion_consequence_count_last", int(np.sum(active_new)))
        setattr(env, "delayed_resource_depletion_consequence_total_last", float(np.sum(delta, dtype=np.float64)))
        project_patch_food_to_binary_grid(env)
    return delayed_resource_depletion_metrics(env)


def apply_delayed_crowding_waste_step(env: Any, abs_step: int) -> Dict[str, float]:
    """Advance default-off delayed overuse penalty and apply active waste increase."""
    cfg = getattr(env, "cfg", None)
    if not delayed_crowding_waste_enabled(cfg):
        return delayed_crowding_waste_metrics(env)
    waste = np.asarray(getattr(env, "patch_waste"), dtype=np.float32)
    threshold = _bounded(getattr(cfg, "delayed_crowding_waste_min_waste_trigger", 0.01), default=0.01, lo=0.0)
    trigger = waste >= np.float32(threshold)
    st = _memory_family_state_step(
        env,
        abs_step=int(abs_step),
        prefix="delayed_crowding_waste",
        delay_name="delayed_crowding_waste_delay_steps",
        cue_duration_name="delayed_crowding_waste_cue_duration_steps",
        active_duration_name="delayed_crowding_waste_active_duration_steps",
        period_name="delayed_crowding_waste_event_period_steps",
        radius_name="delayed_crowding_waste_radius",
        intensity_name="delayed_crowding_waste_waste_add",
        intensity_default=0.10,
        schedule_enable_name="delayed_crowding_waste_deterministic_schedule_enable",
        trigger_mask=trigger,
    )
    active_new = np.asarray(st["new_active_mask"], dtype=bool)
    if bool(np.any(active_new)):
        waste_max = patch_waste_max(cfg)
        before = waste.copy()
        waste[active_new] = np.minimum(
            np.float32(waste_max),
            waste[active_new] + np.asarray(st["intensity"], dtype=np.float32)[active_new],
        )
        delta = waste - before
        st["once"][active_new] = np.int8(1)
        setattr(env, "delayed_crowding_waste_consequence_count_last", int(np.sum(active_new)))
        setattr(env, "delayed_crowding_waste_consequence_total_last", float(np.sum(delta, dtype=np.float64)))
    return delayed_crowding_waste_metrics(env)


def _memory_family_metrics(env: Any, prefix: str, enabled: bool) -> Dict[str, float]:
    if not enabled:
        return {
            f"{prefix}_enable": 0,
            f"{prefix}_cue_cell_count": 0.0,
            f"{prefix}_hidden_cell_count": 0.0,
            f"{prefix}_active_cell_count": 0.0,
            f"{prefix}_event_count": 0.0,
            f"{prefix}_cue_to_active_delay_mean": 0.0,
            f"{prefix}_cue_without_active_count": 0.0,
            f"{prefix}_active_without_prior_cue_count": 0.0,
            f"{prefix}_consequence_count": 0.0,
            f"{prefix}_consequence_total": 0.0,
        }
    delay_count = int(getattr(env, f"{prefix}_cue_to_active_delay_count_last", 0) or 0)
    delay_sum = float(getattr(env, f"{prefix}_cue_to_active_delay_sum_last", 0.0) or 0.0)
    return {
        f"{prefix}_enable": 1,
        f"{prefix}_cue_cell_count": float(getattr(env, f"{prefix}_cue_cell_count_last", 0)),
        f"{prefix}_hidden_cell_count": float(getattr(env, f"{prefix}_hidden_cell_count_last", 0)),
        f"{prefix}_active_cell_count": float(getattr(env, f"{prefix}_active_cell_count_last", 0)),
        f"{prefix}_event_count": float(getattr(env, f"{prefix}_event_count_last", 0)),
        f"{prefix}_cue_to_active_delay_mean": delay_sum / float(delay_count) if delay_count > 0 else 0.0,
        f"{prefix}_cue_without_active_count": float(getattr(env, f"{prefix}_cue_without_active_count_last", 0)),
        f"{prefix}_active_without_prior_cue_count": float(
            getattr(env, f"{prefix}_active_without_prior_cue_count_last", 0)
        ),
        f"{prefix}_consequence_count": float(getattr(env, f"{prefix}_consequence_count_last", 0)),
        f"{prefix}_consequence_total": float(getattr(env, f"{prefix}_consequence_total_last", 0.0)),
    }


def delayed_resource_depletion_metrics(env: Any) -> Dict[str, float]:
    cfg = getattr(env, "cfg", None)
    return _memory_family_metrics(env, "delayed_resource_depletion", delayed_resource_depletion_enabled(cfg))


def delayed_crowding_waste_metrics(env: Any) -> Dict[str, float]:
    cfg = getattr(env, "cfg", None)
    return _memory_family_metrics(env, "delayed_crowding_waste", delayed_crowding_waste_enabled(cfg))


def _mem_max_choice_target_coords(env: Any) -> tuple[tuple[int, int], tuple[int, int]]:
    cfg = getattr(env, "cfg", None)
    grid = np.asarray(getattr(env, "grid"))
    h, w = tuple(int(v) for v in grid.shape)
    distance = _wdl_int(cfg, "mem_max_delayed_choice_target_distance", 2, lo=1)
    cy = h // 2
    cx = w // 2
    geometry = str(getattr(cfg, "mem_max_delayed_choice_target_geometry", "horizontal_symmetric") or "")
    if geometry == "vertical_symmetric":
        a = (max(0, cy - distance), cx)
        b = (min(h - 1, cy + distance), cx)
    else:
        a = (cy, max(0, cx - distance))
        b = (cy, min(w - 1, cx + distance))
    return (int(a[0]), int(a[1])), (int(b[0]), int(b[1]))


def _stable_mem_max_delayed_choice_event(env: Any, event_index: int) -> tuple[int, int]:
    import hashlib

    grid = np.asarray(getattr(env, "grid"))
    h, w = tuple(grid.shape)
    seed = _wdl_seed(env)
    material = f"mem-max-delayed-choice-v1:{seed}:{int(event_index)}:{int(h)}:{int(w)}".encode("ascii")
    digest = hashlib.sha256(material).digest()
    event_id = int.from_bytes(digest[0:8], "big") & 0x7FFFFFFFFFFFFFFF
    correct_target = 1 if (int.from_bytes(digest[8:10], "big") % 2 == 0) else 2
    if bool(getattr(getattr(env, "cfg", None), "mem_max_delayed_choice_random_incorrect_cue_enable", False)):
        correct_target = 2 if correct_target == 1 else 1
    return int(event_id), int(correct_target)


def _mem_max_choice_event_phase(env: Any) -> str:
    timer = np.asarray(getattr(env, "mem_max_delayed_choice_timer"), dtype=np.int32)
    if not bool(np.any(timer >= 0)):
        return "idle"
    cue_a = np.asarray(getattr(env, "mem_max_delayed_choice_cue_a"), dtype=np.float32)
    cue_b = np.asarray(getattr(env, "mem_max_delayed_choice_cue_b"), dtype=np.float32)
    active = np.asarray(getattr(env, "mem_max_delayed_choice_active"), dtype=np.float32)
    if bool(np.any((cue_a > 0.0) | (cue_b > 0.0))):
        return "cue"
    if bool(np.any(active > 0.0)):
        return "active"
    return "hidden"


def apply_mem_max_delayed_choice_step(env: Any, abs_step: int) -> Dict[str, float]:
    """Advance the default-off MEM-MAX delayed-choice state machine."""
    cfg = getattr(env, "cfg", None)
    reset_mem_max_delayed_choice_counters(env)
    if not mem_max_delayed_choice_enabled(cfg):
        return mem_max_delayed_choice_metrics(env)

    shape = tuple(np.asarray(getattr(env, "grid")).shape)
    target_a = np.asarray(getattr(env, "mem_max_delayed_choice_target_a"), dtype=np.float32)
    target_b = np.asarray(getattr(env, "mem_max_delayed_choice_target_b"), dtype=np.float32)
    cue_a = np.asarray(getattr(env, "mem_max_delayed_choice_cue_a"), dtype=np.float32)
    cue_b = np.asarray(getattr(env, "mem_max_delayed_choice_cue_b"), dtype=np.float32)
    timer = np.asarray(getattr(env, "mem_max_delayed_choice_timer"), dtype=np.int32)
    age = np.asarray(getattr(env, "mem_max_delayed_choice_age"), dtype=np.int32)
    active = np.asarray(getattr(env, "mem_max_delayed_choice_active"), dtype=np.float32)
    event_ids = np.asarray(getattr(env, "mem_max_delayed_choice_event_id"), dtype=np.int64)
    correct = np.asarray(getattr(env, "mem_max_delayed_choice_correct_target"), dtype=np.int8)
    prior_cue = np.asarray(getattr(env, "mem_max_delayed_choice_prior_cue"), dtype=np.int8)
    once = np.asarray(getattr(env, "mem_max_delayed_choice_consequence_applied_once"), dtype=np.int8)

    cue_steps = _wdl_int(cfg, "mem_max_delayed_choice_cue_steps", 2, lo=1)
    hidden_steps = _wdl_int(cfg, "mem_max_delayed_choice_hidden_steps", 4, lo=1)
    active_steps = _wdl_int(cfg, "mem_max_delayed_choice_active_steps", 3, lo=1)
    delay = int(cue_steps + hidden_steps)
    period = _wdl_int(cfg, "mem_max_delayed_choice_event_period_steps", 16, lo=delay + active_steps + 1)

    occupied = timer >= 0
    age[occupied] += np.int32(1)
    timer[occupied] -= np.int32(1)
    cue_a[:, :] = 0.0
    cue_b[:, :] = 0.0
    active[:, :] = 0.0

    cue_mask = occupied & (age < cue_steps)
    hidden_mask = occupied & (age >= cue_steps) & (age < delay)
    active_mask = occupied & (age >= delay) & (age < delay + active_steps)
    done_mask = occupied & (age >= delay + active_steps)

    if bool(getattr(cfg, "mem_max_delayed_choice_cue_enable", True)):
        cue_a[cue_mask & (correct == 1)] = np.float32(1.0)
        cue_b[cue_mask & (correct == 2)] = np.float32(1.0)
    prior_cue[cue_mask | hidden_mask | active_mask] = np.int8(1)
    active[active_mask] = np.float32(1.0)
    env.mem_max_delayed_choice_cue_step_active_last = int(np.sum(cue_mask))
    env.mem_max_delayed_choice_hidden_step_active_last = int(np.sum(hidden_mask))
    env.mem_max_delayed_choice_consequence_step_active_last = int(np.sum(active_mask))
    env.mem_max_delayed_choice_target_a_cue_count_last = int(np.sum(cue_a > 0.0))
    env.mem_max_delayed_choice_target_b_cue_count_last = int(np.sum(cue_b > 0.0))
    env.mem_max_delayed_choice_shortcut_exposure_last = int(
        bool(getattr(cfg, "mem_max_delayed_choice_shortcut_truth_enable", False))
    )

    if bool(np.any(active_mask & (once <= 0))) and bool(getattr(cfg, "mem_max_delayed_choice_consequence_enable", True)):
        correct_value = int(np.max(correct[active_mask])) if bool(np.any(correct[active_mask] > 0)) else 0
        (ay, ax), (by, bx) = _mem_max_choice_target_coords(env)
        alive = np.asarray(getattr(env, "agent_alive", np.zeros(0, dtype=bool)), dtype=bool)
        pos_y = np.asarray(getattr(env, "agent_pos_y", np.zeros(0, dtype=np.int32)), dtype=np.int32)
        pos_x = np.asarray(getattr(env, "agent_pos_x", np.zeros(0, dtype=np.int32)), dtype=np.int32)
        correct_arrivals = 0
        incorrect_arrivals = 0
        no_choice = 0
        for aid in np.nonzero(alive)[0].astype(np.int32).tolist():
            y = int(pos_y[int(aid)])
            x = int(pos_x[int(aid)])
            at_a = y == ay and x == ax
            at_b = y == by and x == bx
            if (correct_value == 1 and at_a) or (correct_value == 2 and at_b):
                correct_arrivals += 1
            elif at_a or at_b:
                incorrect_arrivals += 1
            else:
                no_choice += 1
        damage = _bounded(getattr(cfg, "mem_max_delayed_choice_damage", 0.25), default=0.25, lo=0.0)
        damage_events = int(incorrect_arrivals + no_choice)
        env.mem_max_delayed_choice_correct_arrival_count_last = int(correct_arrivals)
        env.mem_max_delayed_choice_incorrect_arrival_count_last = int(incorrect_arrivals)
        env.mem_max_delayed_choice_no_choice_count_last = int(no_choice)
        env.mem_max_delayed_choice_consequence_exposure_count_last = int(damage_events)
        env.mem_max_delayed_choice_damage_total_last = float(damage_events) * float(damage)
        env.mem_max_delayed_choice_damage_prevented_total_last = float(correct_arrivals) * float(damage)
        once[active_mask] = np.int8(1)

    if bool(np.any(done_mask)):
        for arr in (target_a, target_b, cue_a, cue_b, active):
            arr[done_mask] = np.float32(0.0)
        age[done_mask] = np.int32(0)
        timer[done_mask] = np.int32(-1)
        event_ids[done_mask] = np.int64(0)
        correct[done_mask] = np.int8(0)
        prior_cue[done_mask] = np.int8(0)
        once[done_mask] = np.int8(0)

    if bool(getattr(cfg, "mem_max_delayed_choice_deterministic_schedule_enable", True)):
        step = int(abs_step)
        if step >= 0 and step % int(period) == 0 and not bool(np.any(timer >= 0)):
            event_index = step // int(period)
            event_id, correct_value = _stable_mem_max_delayed_choice_event(env, event_index)
            forced_correct = getattr(env, "_mem_max_pre_cue_forced_correct_target", None)
            if forced_correct in (1, 2):
                correct_value = int(forced_correct)
            (ay, ax), (by, bx) = _mem_max_choice_target_coords(env)
            hook = getattr(env, "_mem_max_pre_cue_checkpoint_hook", None)
            if callable(hook):
                hook(
                    env=env,
                    abs_step=int(step),
                    event_index=int(event_index),
                    event_id=int(event_id),
                    correct_target=int(correct_value),
                    target_a=(int(ay), int(ax)),
                    target_b=(int(by), int(bx)),
                )
            target_a[ay, ax] = np.float32(1.0)
            target_b[by, bx] = np.float32(1.0)
            target_cells = np.zeros(shape, dtype=bool)
            target_cells[ay, ax] = True
            target_cells[by, bx] = True
            timer[target_cells] = np.int32(delay)
            age[target_cells] = np.int32(0)
            event_ids[target_cells] = np.int64(event_id)
            correct[target_cells] = np.int8(correct_value)
            prior_cue[target_cells] = np.int8(1)
            once[target_cells] = np.int8(0)
            if bool(getattr(cfg, "mem_max_delayed_choice_cue_enable", True)):
                if correct_value == 1:
                    cue_a[ay, ax] = np.float32(1.0)
                else:
                    cue_b[by, bx] = np.float32(1.0)
            env.mem_max_delayed_choice_event_count_last = 1
            env.mem_max_delayed_choice_target_a_cue_count_last = int(np.sum(cue_a > 0.0))
            env.mem_max_delayed_choice_target_b_cue_count_last = int(np.sum(cue_b > 0.0))

    return mem_max_delayed_choice_metrics(env)


def mem_max_delayed_choice_metrics(env: Any) -> Dict[str, float]:
    """Return MEM-MAX delayed-choice state metrics for metrics.csv."""
    cfg = getattr(env, "cfg", None)
    keys = {
        "mem_max_delayed_choice_enable": 0,
        "mem_max_delayed_choice_observation_enable": 0,
        "mem_max_delayed_choice_event_count": 0.0,
        "mem_max_delayed_choice_target_a_cue_count": 0.0,
        "mem_max_delayed_choice_target_b_cue_count": 0.0,
        "mem_max_delayed_choice_hidden_window_active": 0.0,
        "mem_max_delayed_choice_consequence_step_active": 0.0,
        "mem_max_delayed_choice_correct_arrival_count": 0.0,
        "mem_max_delayed_choice_incorrect_arrival_count": 0.0,
        "mem_max_delayed_choice_no_choice_count": 0.0,
        "mem_max_delayed_choice_correct_choice_rate": 0.0,
        "mem_max_delayed_choice_consequence_exposure_count": 0.0,
        "mem_max_delayed_choice_damage_total": 0.0,
        "mem_max_delayed_choice_damage_prevented_total": 0.0,
        "mem_max_delayed_choice_expected_consequence_total": 0.0,
        "mem_max_delayed_choice_applied_health_damage_total": 0.0,
        "mem_max_delayed_choice_applied_energy_cost_total": 0.0,
        "mem_max_delayed_choice_consequence_applied_count": 0.0,
        "mem_max_delayed_choice_consequence_eligible_count": 0.0,
        "mem_max_delayed_choice_duplicate_prevented_count": 0.0,
        "mem_max_delayed_choice_correct_health_damage_total": 0.0,
        "mem_max_delayed_choice_incorrect_health_damage_total": 0.0,
        "mem_max_delayed_choice_no_choice_health_damage_total": 0.0,
        "mem_max_delayed_choice_target_a_arrival_count": 0.0,
        "mem_max_delayed_choice_target_b_arrival_count": 0.0,
        "mem_max_delayed_choice_any_target_reachable_fraction": 0.0,
        "mem_max_delayed_choice_shortcut_exposure": 0,
        "mem_max_delayed_choice_behavior_metric_available": 0,
        "mem_max_delayed_choice_behavior_coupling_supported": 0,
        "mem_max_delayed_choice_direct_memory_reward_active": 0,
        "mem_max_delayed_choice_direct_cognition_reward_active": 0,
    }
    if not mem_max_delayed_choice_enabled(cfg):
        return keys
    correct = float(getattr(env, "mem_max_delayed_choice_correct_arrival_count_last", 0))
    incorrect = float(getattr(env, "mem_max_delayed_choice_incorrect_arrival_count_last", 0))
    no_choice = float(getattr(env, "mem_max_delayed_choice_no_choice_count_last", 0))
    denom = correct + incorrect + no_choice
    keys.update(
        {
            "mem_max_delayed_choice_enable": 1,
            "mem_max_delayed_choice_observation_enable": int(mem_max_delayed_choice_observation_enabled(cfg)),
            "mem_max_delayed_choice_event_count": float(getattr(env, "mem_max_delayed_choice_event_count_last", 0)),
            "mem_max_delayed_choice_target_a_cue_count": float(
                getattr(env, "mem_max_delayed_choice_target_a_cue_count_last", 0)
            ),
            "mem_max_delayed_choice_target_b_cue_count": float(
                getattr(env, "mem_max_delayed_choice_target_b_cue_count_last", 0)
            ),
            "mem_max_delayed_choice_hidden_window_active": float(
                getattr(env, "mem_max_delayed_choice_hidden_step_active_last", 0)
            ),
            "mem_max_delayed_choice_consequence_step_active": float(
                getattr(env, "mem_max_delayed_choice_consequence_step_active_last", 0)
            ),
            "mem_max_delayed_choice_correct_arrival_count": correct,
            "mem_max_delayed_choice_incorrect_arrival_count": incorrect,
            "mem_max_delayed_choice_no_choice_count": no_choice,
            "mem_max_delayed_choice_correct_choice_rate": correct / denom if denom > 0.0 else 0.0,
            "mem_max_delayed_choice_consequence_exposure_count": float(
                getattr(env, "mem_max_delayed_choice_consequence_exposure_count_last", 0)
            ),
            "mem_max_delayed_choice_damage_total": float(
                getattr(env, "mem_max_delayed_choice_damage_total_last", 0.0)
            ),
            "mem_max_delayed_choice_damage_prevented_total": float(
                getattr(env, "mem_max_delayed_choice_damage_prevented_total_last", 0.0)
            ),
            "mem_max_delayed_choice_expected_consequence_total": float(
                getattr(env, "mem_max_delayed_choice_expected_consequence_total_last", 0.0)
            ),
            "mem_max_delayed_choice_applied_health_damage_total": float(
                getattr(env, "mem_max_delayed_choice_applied_health_damage_total_last", 0.0)
            ),
            "mem_max_delayed_choice_applied_energy_cost_total": float(
                getattr(env, "mem_max_delayed_choice_applied_energy_cost_total_last", 0.0)
            ),
            "mem_max_delayed_choice_consequence_applied_count": float(
                getattr(env, "mem_max_delayed_choice_consequence_applied_count_last", 0)
            ),
            "mem_max_delayed_choice_consequence_eligible_count": float(
                getattr(env, "mem_max_delayed_choice_consequence_eligible_count_last", 0)
            ),
            "mem_max_delayed_choice_duplicate_prevented_count": float(
                getattr(env, "mem_max_delayed_choice_duplicate_prevented_count_last", 0)
            ),
            "mem_max_delayed_choice_correct_health_damage_total": float(
                getattr(env, "mem_max_delayed_choice_correct_health_damage_total_last", 0.0)
            ),
            "mem_max_delayed_choice_incorrect_health_damage_total": float(
                getattr(env, "mem_max_delayed_choice_incorrect_health_damage_total_last", 0.0)
            ),
            "mem_max_delayed_choice_no_choice_health_damage_total": float(
                getattr(env, "mem_max_delayed_choice_no_choice_health_damage_total_last", 0.0)
            ),
            "mem_max_delayed_choice_target_a_arrival_count": float(
                getattr(env, "mem_max_delayed_choice_target_a_arrival_count_last", 0)
            ),
            "mem_max_delayed_choice_target_b_arrival_count": float(
                getattr(env, "mem_max_delayed_choice_target_b_arrival_count_last", 0)
            ),
            "mem_max_delayed_choice_any_target_reachable_fraction": float(
                getattr(env, "mem_max_delayed_choice_any_target_reachable_fraction_last", 0.0)
            ),
            "mem_max_delayed_choice_shortcut_exposure": int(
                getattr(env, "mem_max_delayed_choice_shortcut_exposure_last", 0)
            ),
            "mem_max_delayed_choice_behavior_metric_available": 1,
            "mem_max_delayed_choice_behavior_coupling_supported": 1,
        }
    )
    return keys


def mem_max_delayed_choice_observation_audit(env: Any) -> Dict[str, Any]:
    """Audit that hidden-window policy observations do not expose target truth."""
    cfg = getattr(env, "cfg", None)
    phase = _mem_max_choice_event_phase(env) if mem_max_delayed_choice_enabled(cfg) else "disabled"
    cue_a = np.asarray(getattr(env, "mem_max_delayed_choice_cue_a", np.zeros((1, 1))), dtype=np.float32)
    cue_b = np.asarray(getattr(env, "mem_max_delayed_choice_cue_b", np.zeros((1, 1))), dtype=np.float32)
    correct = np.asarray(getattr(env, "mem_max_delayed_choice_correct_target", np.zeros((1, 1))), dtype=np.int8)
    shortcut = bool(getattr(cfg, "mem_max_delayed_choice_shortcut_truth_enable", False))
    hidden_truth_exposed = bool(phase == "hidden" and (np.any(cue_a > 0.0) or np.any(cue_b > 0.0)))
    return {
        "schema_version": "mem_max_delayed_choice_observation_audit_v1",
        "enabled": bool(mem_max_delayed_choice_enabled(cfg)),
        "phase": phase,
        "observation_enabled": bool(mem_max_delayed_choice_observation_enabled(cfg)),
        "shortcut_truth_enable": shortcut,
        "correct_target_state_present": bool(np.any(correct > 0)),
        "hidden_window_direct_cue_exposed": hidden_truth_exposed,
        "passes": bool((not hidden_truth_exposed) or shortcut),
    }


def compute_local_density_map(env: Any) -> np.ndarray:
    """Compute deterministic local density from current one-agent-per-cell state.

    Density is a square Chebyshev-neighborhood count by default. If
    patch_density_normalize is true, counts are divided by the theoretical
    neighborhood size, not by clipped edge-cell area.
    """
    agent_grid = np.asarray(getattr(env, "agent_grid"), dtype=np.int32)
    cfg = getattr(env, "cfg", None)
    radius = int(max(0, getattr(cfg, "patch_density_radius", 1)))
    include_self = bool(getattr(cfg, "patch_density_include_self", True))
    normalize = bool(getattr(cfg, "patch_density_normalize", False))
    h, w = agent_grid.shape
    if h <= 0 or w <= 0:
        env.local_density_map = np.zeros(agent_grid.shape, dtype=np.float32)
        return env.local_density_map

    occupied = (agent_grid != -1).astype(np.int32, copy=False)
    integral = np.pad(occupied, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    ys = np.arange(h, dtype=np.int32)
    xs = np.arange(w, dtype=np.int32)
    y0 = np.maximum(0, ys - radius)
    y1 = np.minimum(h, ys + radius + 1)
    x0 = np.maximum(0, xs - radius)
    x1 = np.minimum(w, xs + radius + 1)
    density = (
        integral[y1[:, None], x1[None, :]]
        - integral[y0[:, None], x1[None, :]]
        - integral[y1[:, None], x0[None, :]]
        + integral[y0[:, None], x0[None, :]]
    ).astype(np.float32, copy=False)
    if not include_self:
        density -= occupied.astype(np.float32, copy=False)
    if normalize:
        denom = float((2 * radius + 1) * (2 * radius + 1))
        if not include_self:
            denom -= 1.0
        if denom > 0.0:
            density[:, :] = density / np.float32(denom)
    env.local_density_map = np.ascontiguousarray(density, dtype=np.float32)
    return env.local_density_map


def queue_patch_world_movement_emissions(env: Any, ys: Any, xs: Any, moved_mask: Any) -> tuple[float, float]:
    """Queue deterministic movement trace emissions at destination cells."""
    cfg = getattr(env, "cfg", None)
    scent_emit = _emit_value(cfg, "patch_scent_move_emit", 0.02)
    waste_emit = 0.0
    if scent_emit <= 0.0 and waste_emit <= 0.0:
        return 0.0, 0.0
    moved = np.asarray(moved_mask, dtype=bool)
    if not bool(np.any(moved)):
        return 0.0, 0.0
    moved_y = np.asarray(ys, dtype=np.int32)[moved]
    moved_x = np.asarray(xs, dtype=np.int32)[moved]
    count = int(moved_y.size)
    if scent_emit > 0.0:
        np.add.at(env.patch_scent_emit_pending, (moved_y, moved_x), np.float32(scent_emit))
    if waste_emit > 0.0:
        np.add.at(env.patch_waste_emit_pending, (moved_y, moved_x), np.float32(waste_emit))
    return float(scent_emit * count), float(waste_emit * count)


def queue_patch_world_gather_emission(
    env: Any,
    y: int,
    x: int,
    *,
    scent_scale: float,
    waste_scale: float,
) -> tuple[float, float]:
    """Queue deterministic gather/consume trace emissions at a consumed cell."""
    cfg = getattr(env, "cfg", None)
    scent_emit = _emit_value(cfg, "patch_scent_gather_emit", 0.05) * max(0.0, float(scent_scale))
    waste_emit = _emit_value(cfg, "patch_waste_consume_emit", 0.05) * max(0.0, float(waste_scale))
    if scent_emit > 0.0:
        env.patch_scent_emit_pending[int(y), int(x)] += np.float32(scent_emit)
    if waste_emit > 0.0:
        env.patch_waste_emit_pending[int(y), int(x)] += np.float32(waste_emit)
    return float(scent_emit), float(waste_emit)


def apply_patch_world_trace_step(env: Any, abs_step: int) -> tuple[float, float, float, float]:
    """Decay and emit deterministic Patch World activity traces."""
    _ = int(abs_step)
    cfg = getattr(env, "cfg", None)
    scent = np.asarray(getattr(env, "patch_scent"), dtype=np.float32)
    waste = np.asarray(getattr(env, "patch_waste"), dtype=np.float32)

    scent_before = scent.copy()
    waste_before = waste.copy()
    scent[:, :] = np.maximum(0.0, scent * np.float32(_decay_factor(cfg, "patch_scent_decay_per_step", 0.05)))
    waste[:, :] = np.maximum(0.0, waste * np.float32(_decay_factor(cfg, "patch_waste_decay_per_step", 0.02)))
    scent_decayed = float(np.sum(scent_before - scent, dtype=np.float64))
    waste_decayed = float(np.sum(waste_before - waste, dtype=np.float64))

    scent_pending = np.asarray(getattr(env, "patch_scent_emit_pending"), dtype=np.float32)
    waste_pending = np.asarray(getattr(env, "patch_waste_emit_pending"), dtype=np.float32)
    scent_emitted = float(np.sum(scent_pending, dtype=np.float64))
    waste_emitted = float(np.sum(waste_pending, dtype=np.float64))
    scent[:, :] = scent + scent_pending
    waste[:, :] = waste + waste_pending
    scent_pending.fill(0.0)
    waste_pending.fill(0.0)

    density = compute_local_density_map(env)
    occupied = np.asarray(getattr(env, "agent_grid"), dtype=np.int32) != -1
    scent_presence = _emit_value(cfg, "patch_scent_presence_emit", 0.01)
    waste_presence = _emit_value(cfg, "patch_waste_presence_emit", 0.005)
    if scent_presence > 0.0 and bool(np.any(occupied)):
        scent[occupied] = scent[occupied] + np.float32(scent_presence)
        scent_emitted += float(scent_presence * int(np.sum(occupied)))
    if waste_presence > 0.0 and bool(np.any(occupied)):
        waste[occupied] = waste[occupied] + np.float32(waste_presence)
        waste_emitted += float(waste_presence * int(np.sum(occupied)))
    density_scale = _emit_value(cfg, "patch_waste_density_emit_scale", 0.0)
    if density_scale > 0.0:
        density_emit = np.asarray(density, dtype=np.float32) * np.float32(density_scale)
        waste[:, :] = waste + density_emit
        waste_emitted += float(np.sum(density_emit, dtype=np.float64))

    scent[:, :] = np.clip(scent, 0.0, patch_scent_max(cfg)).astype(np.float32, copy=False)
    waste[:, :] = np.clip(waste, 0.0, patch_waste_max(cfg)).astype(np.float32, copy=False)

    env.patch_scent_emitted_total_last = float(scent_emitted)
    env.patch_scent_decayed_total_last = float(max(0.0, scent_decayed))
    env.patch_waste_emitted_total_last = float(waste_emitted)
    env.patch_waste_decayed_total_last = float(max(0.0, waste_decayed))
    return (
        float(env.patch_scent_emitted_total_last),
        float(env.patch_scent_decayed_total_last),
        float(env.patch_waste_emitted_total_last),
        float(env.patch_waste_decayed_total_last),
    )


def _normalized_patch_fertility(cfg: Any, fertility: Any) -> np.ndarray:
    arr = np.asarray(fertility, dtype=np.float32)
    lo = _bounded(getattr(cfg, "patch_fertility_min", 0.0), default=0.0)
    hi = _bounded(getattr(cfg, "patch_fertility_max", 1.0), default=1.0)
    if hi < lo:
        lo, hi = hi, lo
    span = float(hi - lo)
    if span <= 0.0:
        return np.zeros(tuple(arr.shape), dtype=np.float32)
    out = (arr - np.float32(lo)) / np.float32(span)
    return np.ascontiguousarray(np.nan_to_num(np.clip(out, 0.0, 1.0)), dtype=np.float32)


def update_patch_recovery_activity(env: Any) -> tuple[float, float, float]:
    """Update deterministic Step 12 ecological recovery activity."""
    cfg = getattr(env, "cfg", None)
    reset_patch_world_recovery_counters(env)
    if not jungle_ecological_recovery_enabled(cfg):
        return 0.0, 0.0, 0.0

    activity = np.asarray(getattr(env, "patch_recovery_activity"), dtype=np.float32)
    activity_min, activity_max = patch_recovery_activity_min_max(cfg)
    cover = np.ascontiguousarray(
        np.nan_to_num(np.clip(np.asarray(getattr(env, "patch_cover"), dtype=np.float32), 0.0, 1.0)),
        dtype=np.float32,
    )
    fertility_norm = _normalized_patch_fertility(cfg, getattr(env, "patch_fertility"))
    waste_norm = _safe_norm(getattr(env, "patch_waste"), patch_waste_max(cfg))
    scent_norm = _safe_norm(getattr(env, "patch_scent"), patch_scent_max(cfg))
    compute_local_density_map(env)
    density_norm = normalize_patch_density(cfg, getattr(env, "local_density_map"))

    base = np.float32(_bounded(getattr(cfg, "jungle_recovery_activity_base_growth", 0.0), default=0.0))
    cover_w = np.float32(_bounded(getattr(cfg, "jungle_recovery_activity_cover_weight", 0.02), default=0.02, lo=0.0))
    fertility_w = np.float32(
        _bounded(getattr(cfg, "jungle_recovery_activity_fertility_weight", 0.02), default=0.02, lo=0.0)
    )
    waste_w = np.float32(
        _bounded(getattr(cfg, "jungle_recovery_activity_waste_penalty_weight", 0.05), default=0.05, lo=0.0)
    )
    density_w = np.float32(
        _bounded(getattr(cfg, "jungle_recovery_activity_density_penalty_weight", 0.05), default=0.05, lo=0.0)
    )
    scent_w = np.float32(
        _bounded(getattr(cfg, "jungle_recovery_activity_scent_penalty_weight", 0.02), default=0.02, lo=0.0)
    )
    decay = np.float32(_decay_factor(cfg, "jungle_recovery_activity_decay_per_step", 0.02))

    positive = base + cover * cover_w + fertility_norm * fertility_w
    negative = waste_norm * waste_w + density_norm * density_w + scent_norm * scent_w
    before = activity.copy()
    pre_decay = before + positive - negative
    post_decay = pre_decay * decay
    activity[:, :] = np.clip(post_decay, activity_min, activity_max).astype(np.float32, copy=False)

    positive_total = float(np.sum(positive, dtype=np.float64))
    negative_total = float(np.sum(negative, dtype=np.float64))
    decay_total = float(max(0.0, np.sum(pre_decay - post_decay, dtype=np.float64)))
    env.patch_recovery_positive_total_last = positive_total
    env.patch_recovery_negative_total_last = negative_total
    env.patch_recovery_decay_total_last = decay_total
    return positive_total, negative_total, decay_total


def patch_recovery_regrowth_boost_maps(env: Any) -> Dict[str, np.ndarray]:
    """Return deterministic Step 12 recovery regrowth boost maps."""
    cfg = getattr(env, "cfg", None)
    activity = np.asarray(getattr(env, "patch_recovery_activity"), dtype=np.float32)
    zeros = np.zeros(tuple(activity.shape), dtype=np.float32)
    if not jungle_ecological_recovery_enabled(cfg):
        return {"boost": zeros, "recovery_norm": zeros}
    if not bool(getattr(cfg, "jungle_recovery_regrowth_boost_enable", True)):
        return {"boost": zeros, "recovery_norm": zeros}

    _, activity_max = patch_recovery_activity_min_max(cfg)
    recovery_norm = _safe_norm(activity, activity_max)
    scale = np.float32(
        _bounded(getattr(cfg, "jungle_recovery_regrowth_boost_scale", 0.1), default=0.1, lo=0.0)
    )
    cap = np.float32(
        _bounded(getattr(cfg, "jungle_recovery_regrowth_boost_max", 0.25), default=0.25, lo=0.0)
    )
    boost = np.minimum(cap, recovery_norm * scale).astype(np.float32, copy=False)
    return {
        "boost": np.ascontiguousarray(np.nan_to_num(boost), dtype=np.float32),
        "recovery_norm": np.ascontiguousarray(np.nan_to_num(recovery_norm), dtype=np.float32),
    }


def project_patch_food_to_binary_grid(env: Any) -> None:
    """Project authoritative patch food amount into the binary compatibility grid."""
    amount = np.asarray(getattr(env, "patch_food_amount"), dtype=np.float32)
    threshold = patch_food_presence_threshold(getattr(env, "cfg", None))
    getattr(env, "grid")[:, :] = np.where(amount > threshold, FOOD, EMPTY).astype(np.int8, copy=False)
    setattr(env, "_views_dirty", True)


def sync_binary_spawned_food_to_patch_amount(env: Any) -> float:
    """Raise patch food amount for cells newly seeded by the legacy binary spawner."""
    grid = getattr(env, "grid")
    amount = getattr(env, "patch_food_amount")
    cfg = getattr(env, "cfg", None)
    threshold = patch_food_presence_threshold(cfg)
    initial = patch_food_initial_food_cell_value(cfg)
    mask = (grid == FOOD) & (amount <= threshold)
    if not bool(np.any(mask)):
        return 0.0
    before = amount[mask].astype(np.float32, copy=True)
    amount[mask] = np.float32(initial)
    project_patch_food_to_binary_grid(env)
    return float(np.sum(amount[mask] - before, dtype=np.float64))


def _record_regrowth_consequence_metrics(
    env: Any,
    *,
    base_delta: np.ndarray,
    effective_delta: np.ndarray,
    multiplier: np.ndarray,
    waste_penalty: np.ndarray,
    density_penalty: np.ndarray,
    after_waste_delta: np.ndarray | None = None,
    recovery_boost: np.ndarray | None = None,
) -> None:
    base_total = float(np.sum(base_delta, dtype=np.float64))
    effective_total = float(np.sum(effective_delta, dtype=np.float64))
    after_waste_total = float(
        np.sum(after_waste_delta, dtype=np.float64)
    ) if after_waste_delta is not None else effective_total
    env.patch_regrowth_base_total_last = base_total
    env.patch_regrowth_effective_total_last = effective_total
    env.patch_regrowth_suppressed_total_last = float(max(0.0, base_total - effective_total))
    env.patch_regrowth_after_waste_total_last = after_waste_total
    env.patch_regrowth_after_recovery_total_last = effective_total
    env.patch_recovery_added_regrowth_total_last = float(max(0.0, effective_total - after_waste_total))
    env.patch_regrowth_multiplier_mean_last = float(np.mean(multiplier)) if multiplier.size else 1.0
    env.patch_regrowth_multiplier_min_last = float(np.min(multiplier)) if multiplier.size else 1.0
    env.patch_regrowth_multiplier_p10_last = float(np.percentile(multiplier, 10)) if multiplier.size else 1.0
    env.patch_regrowth_multiplier_p90_last = float(np.percentile(multiplier, 90)) if multiplier.size else 1.0
    env.patch_waste_regrowth_penalty_mean_last = float(np.mean(waste_penalty)) if waste_penalty.size else 0.0
    env.patch_waste_regrowth_penalty_max_last = float(np.max(waste_penalty)) if waste_penalty.size else 0.0
    env.patch_density_regrowth_penalty_mean_last = float(np.mean(density_penalty)) if density_penalty.size else 0.0
    env.patch_density_regrowth_penalty_max_last = float(np.max(density_penalty)) if density_penalty.size else 0.0
    env.patch_high_waste_regrowth_suppression_frac_last = float(np.mean(waste_penalty > 0.0)) if waste_penalty.size else 0.0
    env.patch_high_density_regrowth_suppression_frac_last = float(np.mean(density_penalty > 0.0)) if density_penalty.size else 0.0
    boost = np.asarray(recovery_boost, dtype=np.float32) if recovery_boost is not None else np.zeros_like(multiplier)
    env.jungle_recovery_regrowth_boost_total_last = float(np.sum(boost, dtype=np.float64)) if boost.size else 0.0
    env.jungle_recovery_regrowth_boost_mean_last = float(np.mean(boost)) if boost.size else 0.0
    env.jungle_recovery_regrowth_boost_max_last = float(np.max(boost)) if boost.size else 0.0
    env.jungle_recovery_regrowth_boost_p90_last = float(np.percentile(boost, 90)) if boost.size else 0.0


def settle_eed_resource_impact(env: Any) -> float:
    """Settle queued EED impact and independently renew A/B reference stocks."""
    cfg = getattr(env, "cfg", None)
    if not eed_resource_impact_enabled(cfg):
        return 0.0

    decay = _eed_resource_value(
        cfg,
        "eed_resource_impact_decay_per_step",
        0.25,
        hi=1.0,
    )
    max_load = _eed_resource_value(
        cfg,
        "eed_resource_impact_max_load",
        1.0,
    )
    impact_applied = eed_resource_impact_applied(cfg)
    ecology_frozen = eed_ecology_frozen(cfg)
    stats: dict[str, float | int] = {
        "enabled": 1,
        "impact_applied": int(impact_applied),
        "ecology_frozen": int(ecology_frozen),
    }
    total_added = 0.0
    for label in ("a", "b"):
        amount = np.asarray(
            getattr(env, f"eed_resource_{label}_amount"),
            dtype=np.float32,
        )
        capacity = np.asarray(
            getattr(env, f"eed_resource_{label}_capacity"),
            dtype=np.float32,
        )
        renewal = np.asarray(
            getattr(env, f"eed_resource_{label}_renewal"),
            dtype=np.float32,
        )
        pending = np.asarray(
            getattr(env, f"eed_resource_{label}_pending_impact"),
            dtype=np.float32,
        )
        load = np.asarray(
            getattr(env, f"eed_resource_{label}_impact_load"),
            dtype=np.float32,
        )
        pending_before = pending.copy()
        load_before = load.copy()
        if ecology_frozen:
            effective_renewal = np.zeros_like(renewal, dtype=np.float32)
            added = 0.0
            stats[f"resource_{label}_pending_blocked"] = float(
                np.sum(pending_before, dtype=np.float64)
            )
            stats[f"resource_{label}_impact_settled"] = 0.0
            stats[f"resource_{label}_impact_load_after"] = float(
                np.sum(load_before, dtype=np.float64)
            )
            stats[f"resource_{label}_effective_renewal"] = 0.0
            continue
        if impact_applied:
            load[:, :] = np.clip(
                load * np.float32(1.0 - decay) + pending,
                0.0,
                max_load,
            ).astype(np.float32, copy=False)
        else:
            load.fill(np.float32(0.0))
        pending.fill(np.float32(0.0))
        effective_renewal = renewal * (
            np.float32(1.0) - np.clip(load, 0.0, 1.0)
        )
        before = amount.copy()
        amount[:, :] = np.minimum(
            capacity,
            np.maximum(0.0, amount + effective_renewal),
        ).astype(np.float32, copy=False)
        added = float(np.sum(amount - before, dtype=np.float64))
        total_added += added
        stats[f"resource_{label}_pending_before"] = float(
            np.sum(pending_before, dtype=np.float64)
        )
        stats[f"resource_{label}_impact_settled"] = float(
            np.sum(load - load_before * np.float32(1.0 - decay), dtype=np.float64)
        )
        stats[f"resource_{label}_impact_load_after"] = float(
            np.sum(load, dtype=np.float64)
        )
        stats[f"resource_{label}_effective_renewal"] = added

    if not ecology_frozen:
        project_eed_resources_to_patch_food(env)
    reset_patch_world_regrowth_consequence_counters(env)
    env.patch_regrowth_base_total_last = float(
        np.sum(
            np.asarray(getattr(env, "eed_resource_a_renewal"), dtype=np.float32)
            + np.asarray(
                getattr(env, "eed_resource_b_renewal"),
                dtype=np.float32,
            ),
            dtype=np.float64,
        )
    )
    env.patch_regrowth_effective_total_last = float(total_added)
    env.patch_regrowth_suppressed_total_last = float(
        max(0.0, env.patch_regrowth_base_total_last - total_added)
    )
    env.eed_resource_impact_stats_last = stats
    return float(total_added)


def regrow_patch_food(env: Any) -> float:
    """Deterministically regrow patch food amount from fertility."""
    cfg = getattr(env, "cfg", None)
    if eed_resource_impact_enabled(cfg):
        return settle_eed_resource_impact(env)
    amount = getattr(env, "patch_food_amount")
    fertility = getattr(env, "patch_fertility")
    update_patch_recovery_activity(env)
    regrow = _bounded(getattr(cfg, "patch_food_regrow_per_step", 0.02), default=0.02, lo=0.0)
    if regrow <= 0.0:
        reset_patch_world_regrowth_consequence_counters(env)
        setattr(env, "respec_n2_capacity_budget_last", {})
        project_patch_food_to_binary_grid(env)
        return 0.0

    cap = patch_food_capacity_array(cfg, fertility, shape=tuple(amount.shape))
    if bool(getattr(cfg, "patch_food_regrow_use_fertility", True)):
        base_delta = regrow * np.clip(np.asarray(fertility, dtype=np.float32), 0.0, None)
    else:
        base_delta = np.full(tuple(amount.shape), regrow, dtype=np.float32)

    before = amount.copy()
    maps = patch_regrowth_multiplier_maps(env)
    multiplier = np.asarray(maps["multiplier"], dtype=np.float32)
    after_waste_delta_raw = np.asarray(base_delta, dtype=np.float32) * multiplier
    recovery_maps = patch_recovery_regrowth_boost_maps(env)
    recovery_boost = np.asarray(recovery_maps["boost"], dtype=np.float32)
    effective_delta = after_waste_delta_raw * (np.float32(1.0) + recovery_boost)
    amount[:, :] = np.minimum(cap, np.maximum(0.0, amount + effective_delta)).astype(np.float32, copy=False)
    project_patch_food_to_binary_grid(env)
    after_waste_amount = np.minimum(cap, np.maximum(0.0, before + after_waste_delta_raw)).astype(
        np.float32,
        copy=False,
    )
    _record_regrowth_consequence_metrics(
        env,
        base_delta=np.minimum(cap, np.maximum(0.0, before + base_delta)).astype(np.float32, copy=False) - before,
        effective_delta=amount - before,
        multiplier=multiplier,
        waste_penalty=np.asarray(maps["waste_penalty"], dtype=np.float32),
        density_penalty=np.asarray(maps["density_penalty"], dtype=np.float32),
        after_waste_delta=after_waste_amount - before,
        recovery_boost=recovery_boost,
    )
    if respec0b_resource_specialization_enabled(cfg):
        setattr(
            env,
            "respec_n2_capacity_budget_last",
            respec_n2_capacity_budget_snapshot(
                cfg,
                classes=getattr(env, "respec0b_resource_class", np.zeros_like(amount, dtype=np.float32)),
                fertility=fertility,
                before=before,
                after=amount,
                cap=cap,
                base_delta=np.minimum(cap, np.maximum(0.0, before + base_delta)).astype(np.float32, copy=False) - before,
                effective_delta=amount - before,
            ),
        )
    else:
        setattr(env, "respec_n2_capacity_budget_last", {})
    return float(np.sum(amount - before, dtype=np.float64))


def patch_array_summary(arrays: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Return basic diagnostic summaries for patch arrays."""
    out: Dict[str, Dict[str, float]] = {}
    for name in PATCH_WORLD_ARRAY_NAMES:
        arr = np.asarray(arrays[name], dtype=np.float32)
        out[name] = {
            "min": float(np.min(arr)) if arr.size else 0.0,
            "mean": float(np.mean(arr)) if arr.size else 0.0,
            "max": float(np.max(arr)) if arr.size else 0.0,
        }
    return out


def patch_food_metrics(env: Any) -> Dict[str, float]:
    """Return diagnostic Patch World food metrics for metrics.csv."""
    amount = np.asarray(getattr(env, "patch_food_amount"), dtype=np.float32)
    fertility = np.asarray(getattr(env, "patch_fertility"), dtype=np.float32)
    cfg = getattr(env, "cfg", None)
    threshold = patch_food_presence_threshold(cfg)
    cap = patch_food_capacity_array(cfg, fertility, shape=tuple(amount.shape))
    if amount.size <= 0:
        return {
            "patch_food_amount_mean": 0.0,
            "patch_food_amount_min": 0.0,
            "patch_food_amount_max": 0.0,
            "patch_food_amount_p10": 0.0,
            "patch_food_amount_p90": 0.0,
            "patch_food_depleted_frac": 0.0,
            "patch_food_full_frac": 0.0,
            "patch_fertility_mean": 0.0,
        }
    metrics = {
        "patch_food_amount_mean": float(np.mean(amount)),
        "patch_food_amount_min": float(np.min(amount)),
        "patch_food_amount_max": float(np.max(amount)),
        "patch_food_amount_p10": float(np.percentile(amount, 10)),
        "patch_food_amount_p90": float(np.percentile(amount, 90)),
        "patch_food_depleted_frac": float(np.mean(amount <= threshold)),
        "patch_food_full_frac": float(np.mean(amount >= (cap - np.float32(1e-6)))),
        "patch_fertility_mean": float(np.mean(fertility)) if fertility.size else 0.0,
    }
    if hasattr(env, "respec0b_resource_class"):
        classes = np.asarray(getattr(env, "respec0b_resource_class"), dtype=np.float32)
        metrics["respec0b_resource_a_cell_frac"] = float(np.mean(classes == RESPEC0B_RESOURCE_A)) if classes.size else 0.0
        metrics["respec0b_resource_b_cell_frac"] = float(np.mean(classes == RESPEC0B_RESOURCE_B)) if classes.size else 0.0
    if hasattr(env, "mosaic0a_patch_class"):
        classes = np.asarray(getattr(env, "mosaic0a_patch_class"), dtype=np.int32)
        burst = classes == MOSAIC0A_PATCH_BURST
        renewal = classes == MOSAIC0A_PATCH_RENEWAL
        metrics["mosaic0a_burst_cell_frac"] = float(np.mean(burst)) if classes.size else 0.0
        metrics["mosaic0a_renewal_cell_frac"] = float(np.mean(renewal)) if classes.size else 0.0
        metrics["mosaic0a_burst_food_amount_mean"] = float(np.mean(amount[burst])) if bool(np.any(burst)) else 0.0
        metrics["mosaic0a_renewal_food_amount_mean"] = float(np.mean(amount[renewal])) if bool(np.any(renewal)) else 0.0
        metrics["mosaic0a_burst_fertility_mean"] = float(np.mean(fertility[burst])) if bool(np.any(burst)) else 0.0
        metrics["mosaic0a_renewal_fertility_mean"] = float(np.mean(fertility[renewal])) if bool(np.any(renewal)) else 0.0
    stats = getattr(env, "respec0b_resource_gather_stats_last", None)
    if isinstance(stats, dict):
        for key in (
            "attempts",
            "successes",
            "failed_handling",
            "resource_a_successes",
            "resource_b_successes",
            "energy_resource_a",
            "energy_resource_b",
            "handling_ticks",
            "interruptions",
        ):
            metrics[f"respec0b_resource_gather_{key}"] = float(stats.get(key, 0.0) or 0.0)
    return metrics


def patch_terrain_metrics(env: Any) -> Dict[str, float]:
    """Return diagnostic Patch World terrain affordance metrics for metrics.csv."""
    cover = np.asarray(getattr(env, "patch_cover"), dtype=np.float32)
    shelter = np.asarray(getattr(env, "patch_shelter"), dtype=np.float32)
    movement = np.asarray(getattr(env, "patch_movement_cost"), dtype=np.float32)
    visibility = patch_visibility_from_cover(getattr(env, "cfg", None), cover)
    occupied = np.asarray(getattr(env, "agent_grid"), dtype=np.int32) != -1

    def _mean(arr: np.ndarray) -> float:
        return float(np.mean(arr)) if arr.size else 0.0

    def _occ_mean(arr: np.ndarray) -> float:
        return float(np.mean(arr[occupied])) if arr.size and bool(np.any(occupied)) else 0.0

    return {
        "patch_cover_mean": _mean(cover),
        "patch_cover_occupied_mean": _occ_mean(cover),
        "patch_shelter_mean": _mean(shelter),
        "patch_shelter_occupied_mean": _occ_mean(shelter),
        "patch_visibility_mean": _mean(visibility),
        "patch_visibility_occupied_mean": _occ_mean(visibility),
        "patch_movement_cost_mean": _mean(movement),
        "patch_movement_cost_occupied_mean": _occ_mean(movement),
    }


def patch_trace_density_metrics(env: Any) -> Dict[str, float]:
    """Return diagnostic Patch World trace and local-density metrics."""
    scent = np.asarray(getattr(env, "patch_scent"), dtype=np.float32)
    waste = np.asarray(getattr(env, "patch_waste"), dtype=np.float32)
    density = np.asarray(getattr(env, "local_density_map"), dtype=np.float32)
    occupied = np.asarray(getattr(env, "agent_grid"), dtype=np.int32) != -1

    def _mean(arr: np.ndarray) -> float:
        return float(np.mean(arr)) if arr.size else 0.0

    def _max(arr: np.ndarray) -> float:
        return float(np.max(arr)) if arr.size else 0.0

    def _p90(arr: np.ndarray) -> float:
        return float(np.percentile(arr, 90)) if arr.size else 0.0

    def _occ_mean(arr: np.ndarray) -> float:
        return float(np.mean(arr[occupied])) if arr.size and bool(np.any(occupied)) else 0.0

    return {
        "patch_scent_mean": _mean(scent),
        "patch_scent_max": _max(scent),
        "patch_scent_p90": _p90(scent),
        "patch_scent_occupied_mean": _occ_mean(scent),
        "patch_waste_mean": _mean(waste),
        "patch_waste_max": _max(waste),
        "patch_waste_p90": _p90(waste),
        "patch_waste_occupied_mean": _occ_mean(waste),
        "patch_density_mean": _mean(density),
        "patch_density_max": _max(density),
        "patch_density_p90": _p90(density),
        "patch_density_occupied_mean": _occ_mean(density),
        "patch_density_hotspot_frac": float(np.mean(density > 1.0)) if density.size else 0.0,
    }


def patch_recovery_metrics(env: Any) -> Dict[str, float]:
    """Return diagnostic Step 12 ecological recovery metrics."""
    cfg = getattr(env, "cfg", None)
    activity = np.asarray(getattr(env, "patch_recovery_activity"), dtype=np.float32)
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

    return {
        "jungle_ecological_recovery_enable": int(jungle_ecological_recovery_enabled(cfg)),
        "patch_recovery_activity_mean": _mean(activity),
        "patch_recovery_activity_min": _min(activity),
        "patch_recovery_activity_max": _max(activity),
        "patch_recovery_activity_p10": _p(activity, 10),
        "patch_recovery_activity_p90": _p(activity, 90),
        "patch_recovery_activity_occupied_mean": _occ_mean(activity),
    }


def patch_world_descriptor_enabled(cfg: Any) -> bool:
    """Return whether Step 7 diagnostics should be emitted for enabled Patch World."""
    if not bool(getattr(cfg, "patch_world_enable", False)):
        return False
    version = int(getattr(cfg, "patch_world_descriptor_schema_version", 1) or 1)
    if version != PATCH_WORLD_DESCRIPTOR_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported Patch World descriptor schema version: {version}. "
            f"Supported version: {PATCH_WORLD_DESCRIPTOR_SCHEMA_VERSION}."
        )
    return bool(getattr(cfg, "patch_world_diagnostics_enable", True))


def patch_habitat_lineage_id(agent: Any) -> int:
    """Return the diagnostics-only habitat lineage id for an agent."""
    try:
        st = getattr(agent, "stats", None)
        lid = int(getattr(st, "patch_habitat_lineage_id", 0) or 0)
        if lid > 0:
            return lid
    except Exception:
        pass
    return int(getattr(agent, "agent_id", 0) or 0)


def _patch_context_maps(env: Any) -> Dict[str, np.ndarray]:
    compute_local_density_map(env)
    cover = np.asarray(getattr(env, "patch_cover"), dtype=np.float32)
    return {
        "food_amount": np.asarray(getattr(env, "patch_food_amount"), dtype=np.float32),
        "fertility": np.asarray(getattr(env, "patch_fertility"), dtype=np.float32),
        "cover": cover,
        "shelter": np.asarray(getattr(env, "patch_shelter"), dtype=np.float32),
        "visibility": patch_visibility_from_cover(getattr(env, "cfg", None), cover),
        "movement_cost": np.asarray(getattr(env, "patch_movement_cost"), dtype=np.float32),
        "scent": np.asarray(getattr(env, "patch_scent"), dtype=np.float32),
        "waste": np.asarray(getattr(env, "patch_waste"), dtype=np.float32),
        "density": np.asarray(getattr(env, "local_density_map"), dtype=np.float32),
        "recovery_activity": np.asarray(getattr(env, "patch_recovery_activity"), dtype=np.float32),
    }


def sample_patch_context(env: Any, y: int, x: int, *, agent: Any = None, lineage_id: int | None = None) -> Dict[str, Any]:
    """Sample Patch World context at one grid coordinate for diagnostics."""
    maps = _patch_context_maps(env)
    h, w = tuple(np.asarray(getattr(env, "grid")).shape)
    yy = max(0, min(int(h) - 1, int(y)))
    xx = max(0, min(int(w) - 1, int(x)))
    out: Dict[str, float] = {name: float(maps[name][yy, xx]) for name in PATCH_WORLD_CONTEXT_FIELDS}
    out["y"] = float(yy)
    out["x"] = float(xx)
    if lineage_id is None and agent is not None:
        lineage_id = patch_habitat_lineage_id(agent)
    out["lineage_id"] = float(int(lineage_id or 0))
    if agent is not None:
        stats = getattr(agent, "stats", None)
        out["agent_id"] = int(getattr(agent, "agent_id", -1))
        out["biological_lineage_id"] = str(getattr(stats, "lineage_id", "") or "")
        out["founder_id"] = int(getattr(stats, "founder_id", 0) or 0)
    return out


def patch_event_contexts_summary(prefix: str, contexts: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    """Summarize sampled birth/death patch contexts."""
    rows = list(contexts or [])
    out: Dict[str, float] = {f"{prefix}_patch_count": float(len(rows))}
    for name in PATCH_WORLD_CONTEXT_FIELDS:
        vals = [float(r.get(name, 0.0) or 0.0) for r in rows]
        out[f"{prefix}_patch_{name}_mean"] = float(np.mean(np.asarray(vals, dtype=np.float32))) if vals else 0.0
    return out


def patch_occupied_descriptor_metrics(env: Any) -> Dict[str, float]:
    """Return occupied patch metrics and simple occupied-minus-global bias descriptors."""
    maps = _patch_context_maps(env)
    occupied = np.asarray(getattr(env, "agent_grid"), dtype=np.int32) != -1
    has_occ = bool(np.any(occupied))
    out: Dict[str, float] = {}
    for name in PATCH_WORLD_CONTEXT_FIELDS:
        arr = np.asarray(maps[name], dtype=np.float32)
        global_mean = float(np.mean(arr)) if arr.size else 0.0
        occupied_mean = float(np.mean(arr[occupied])) if arr.size and has_occ else 0.0
        metric_name = "food" if name == "food_amount" else name
        out[f"occupied_patch_{name}_mean"] = occupied_mean
        out[f"patch_{metric_name}_occupancy_bias"] = float(occupied_mean - global_mean)
    return out


def lineage_habitat_descriptors(env: Any, agents: Iterable[Any]) -> List[Dict[str, float]]:
    """Compute deterministic per-lineage habitat descriptors from alive agents."""
    maps = _patch_context_maps(env)
    compact_positions = getattr(env, "agent_positions_by_id", None)
    if compact_positions is None:
        alive_arr = np.asarray(getattr(env, "agent_alive"), dtype=bool)
        pos_y = np.asarray(getattr(env, "agent_pos_y"), dtype=np.int32)
        pos_x = np.asarray(getattr(env, "agent_pos_x"), dtype=np.int32)
    else:
        alive_arr = None
        pos_y = None
        pos_x = None
    grouped: Dict[int, List[tuple[int, int]]] = {}
    for agent in sorted(list(agents or []), key=lambda a: int(getattr(a, "agent_id", 0))):
        aid = int(getattr(agent, "agent_id", -1))
        if compact_positions is None:
            if aid < 0 or aid >= int(alive_arr.shape[0]) or not bool(alive_arr[aid]):
                continue
            y = int(pos_y[aid])
            x = int(pos_x[aid])
        else:
            position = compact_positions.get(aid)
            if position is None:
                continue
            y, x = int(position[0]), int(position[1])
        grouped.setdefault(patch_habitat_lineage_id(agent), []).append((y, x))

    out: List[Dict[str, float]] = []
    for lineage_id in sorted(grouped.keys()):
        coords = grouped[lineage_id]
        rec: Dict[str, float] = {
            "lineage_id": float(int(lineage_id)),
            "lineage_alive_count": float(len(coords)),
        }
        ys = np.asarray([p[0] for p in coords], dtype=np.int32)
        xs = np.asarray([p[1] for p in coords], dtype=np.int32)
        for name in PATCH_WORLD_CONTEXT_FIELDS:
            arr = np.asarray(maps[name], dtype=np.float32)
            rec[f"lineage_patch_{name}_mean"] = float(np.mean(arr[ys, xs])) if coords else 0.0
        out.append(rec)
    return out


def lineage_habitat_metrics(descriptors: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    """Return population-level comparability metrics across lineage descriptors."""
    rows = list(descriptors or [])
    out: Dict[str, float] = {"lineage_habitat_descriptor_count": float(len(rows))}
    fields = {
        "food_amount": "lineage_patch_food_amount_mean_std",
        "cover": "lineage_patch_cover_mean_std",
        "shelter": "lineage_patch_shelter_mean_std",
        "waste": "lineage_patch_waste_mean_std",
        "density": "lineage_patch_density_mean_std",
    }
    for src, dst in fields.items():
        vals = [float(r.get(f"lineage_patch_{src}_mean", 0.0) or 0.0) for r in rows]
        out[dst] = float(np.std(np.asarray(vals, dtype=np.float32))) if vals else 0.0
    return out


def lineage_diversity_reporting_enabled(cfg: Any) -> bool:
    """Return whether diagnostic-only Step 13 lineage reporting is enabled."""
    if not bool(getattr(cfg, "lineage_diversity_reporting_enable", True)):
        return False
    version = int(getattr(cfg, "lineage_diversity_schema_version", 1) or 1)
    if version != LINEAGE_DIVERSITY_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported lineage diversity schema version: {version}. "
            f"Supported version: {LINEAGE_DIVERSITY_SCHEMA_VERSION}."
        )
    return True


def lineage_strategy_descriptors_enabled(cfg: Any) -> bool:
    return lineage_diversity_reporting_enabled(cfg) and bool(
        getattr(cfg, "lineage_strategy_descriptors_enable", True)
    )


def lineage_branch_candidate_reporting_enabled(cfg: Any) -> bool:
    return lineage_diversity_reporting_enabled(cfg) and bool(
        getattr(cfg, "lineage_branch_candidate_reporting_enable", True)
    )


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return float(out)


def _context_lineage_id(row: Dict[str, Any]) -> int:
    return int(_finite_float(row.get("lineage_id", 0.0), 0.0))


def _contexts_by_lineage(contexts: Iterable[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for row in list(contexts or []):
        lineage_id = _context_lineage_id(row)
        if lineage_id <= 0:
            continue
        grouped.setdefault(lineage_id, []).append(dict(row))
    return grouped


def _count_shares(counts: Iterable[float]) -> np.ndarray:
    arr = np.asarray([max(0.0, _finite_float(v, 0.0)) for v in counts], dtype=np.float64)
    total = float(np.sum(arr, dtype=np.float64))
    if total <= 0.0:
        return np.zeros(tuple(arr.shape), dtype=np.float64)
    return arr / np.float64(total)


def _entropy_from_counts(counts: Iterable[float]) -> float:
    shares = _count_shares(counts)
    shares = shares[shares > 0.0]
    if shares.size <= 0:
        return 0.0
    return float(-np.sum(shares * np.log(shares), dtype=np.float64))


def _top_share(counts: Iterable[float], n: int) -> float:
    arr = sorted([max(0.0, _finite_float(v, 0.0)) for v in counts], reverse=True)
    total = float(sum(arr))
    if total <= 0.0:
        return 0.0
    return float(sum(arr[: max(0, int(n))]) / total)


def _mean_context_value(rows: List[Dict[str, Any]], field: str) -> float:
    vals = [_finite_float(r.get(field, 0.0), 0.0) for r in rows]
    return float(np.mean(np.asarray(vals, dtype=np.float32))) if vals else 0.0


def lineage_strategy_descriptors(
    env: Any,
    agents: Iterable[Any],
    *,
    birth_contexts: Iterable[Dict[str, Any]] = (),
    death_contexts: Iterable[Dict[str, Any]] = (),
    predator_event_contexts: Iterable[Dict[str, Any]] = (),
    cfg: Any = None,
) -> List[Dict[str, float]]:
    """Compute deterministic diagnostic-only lineage strategy descriptors."""
    active = lineage_habitat_descriptors(env, agents)
    birth_by_lineage = _contexts_by_lineage(birth_contexts)
    death_by_lineage = _contexts_by_lineage(death_contexts)
    predator_by_lineage = _contexts_by_lineage(predator_event_contexts)
    lineage_ids = set(int(r.get("lineage_id", 0.0) or 0.0) for r in active)
    lineage_ids.update(birth_by_lineage.keys())
    lineage_ids.update(death_by_lineage.keys())
    lineage_ids.update(predator_by_lineage.keys())
    lineage_ids.discard(0)

    active_by_lineage = {int(r.get("lineage_id", 0.0) or 0.0): dict(r) for r in active}
    alive_total = float(sum(_finite_float(r.get("lineage_alive_count", 0.0), 0.0) for r in active))
    birth_total = float(sum(len(rows) for rows in birth_by_lineage.values()))
    death_total = float(sum(len(rows) for rows in death_by_lineage.values()))
    predator_total = float(sum(len(rows) for rows in predator_by_lineage.values()))

    out: List[Dict[str, float]] = []
    for lineage_id in sorted(lineage_ids):
        rec = dict(active_by_lineage.get(lineage_id, {}))
        rec["lineage_id"] = float(lineage_id)
        alive_count = _finite_float(rec.get("lineage_alive_count", 0.0), 0.0)
        birth_rows = birth_by_lineage.get(lineage_id, [])
        death_rows = death_by_lineage.get(lineage_id, [])
        predator_rows = predator_by_lineage.get(lineage_id, [])
        birth_count = float(len(birth_rows))
        death_count = float(len(death_rows))
        predator_count = float(len(predator_rows))
        rec["lineage_alive_count"] = float(alive_count)
        rec["lineage_birth_count"] = birth_count
        rec["lineage_death_count"] = death_count
        rec["lineage_predator_event_count"] = predator_count
        rec["lineage_alive_share"] = float(alive_count / alive_total) if alive_total > 0.0 else 0.0
        rec["lineage_birth_share"] = float(birth_count / birth_total) if birth_total > 0.0 else 0.0
        rec["lineage_death_share"] = float(death_count / death_total) if death_total > 0.0 else 0.0
        rec["lineage_predator_event_share"] = float(predator_count / predator_total) if predator_total > 0.0 else 0.0
        rec["lineage_reproduction_rate"] = float(birth_count / alive_count) if alive_count > 0.0 else 0.0

        for prefix, rows in (
            ("lineage_birth_patch", birth_rows),
            ("lineage_death_patch", death_rows),
            ("lineage_predator_event_patch", predator_rows),
        ):
            for field in PATCH_WORLD_CONTEXT_FIELDS:
                rec[f"{prefix}_{field}_mean"] = _mean_context_value(rows, field)

        raw_damage = [_finite_float(r.get("raw_damage", 0.0), 0.0) for r in predator_rows]
        effective_damage = [_finite_float(r.get("effective_damage", 0.0), 0.0) for r in predator_rows]
        prevented = [_finite_float(r.get("damage_prevented", 0.0), 0.0) for r in predator_rows]
        rec["lineage_predator_damage_raw_total"] = float(sum(raw_damage))
        rec["lineage_predator_damage_total"] = float(sum(effective_damage))
        rec["lineage_predator_damage_mean"] = (
            float(np.mean(np.asarray(effective_damage, dtype=np.float32))) if effective_damage else 0.0
        )
        rec["lineage_predator_damage_prevented_total"] = float(sum(prevented))
        rec["lineage_predator_damage_prevented_mean"] = (
            float(np.mean(np.asarray(prevented, dtype=np.float32))) if prevented else 0.0
        )
        for key, value in list(rec.items()):
            rec[key] = _finite_float(value, 0.0)
        out.append(rec)
    min_alive = int(max(0, getattr(cfg, "lineage_min_alive_count_for_descriptor", 1))) if cfg is not None else 1
    if min_alive > 0:
        out = [r for r in out if _finite_float(r.get("lineage_alive_count", 0.0), 0.0) >= float(min_alive)]
    return out


def lineage_strategy_metrics(descriptors: Iterable[Dict[str, Any]], cfg: Any = None) -> Dict[str, float]:
    """Return deterministic Step 13 lineage concentration and strategy metrics."""
    rows = list(descriptors or [])
    alive_counts = [_finite_float(r.get("lineage_alive_count", 0.0), 0.0) for r in rows]
    birth_counts = [_finite_float(r.get("lineage_birth_count", 0.0), 0.0) for r in rows]
    death_counts = [_finite_float(r.get("lineage_death_count", 0.0), 0.0) for r in rows]
    alive_entropy = _entropy_from_counts(alive_counts)
    birth_entropy = _entropy_from_counts(birth_counts)
    top1 = _top_share(alive_counts, 1)
    top3 = _top_share(alive_counts, 3)
    active_count = float(sum(1 for v in alive_counts if v > 0.0))
    alive_total = float(sum(alive_counts))
    top1_thr = _bounded(getattr(cfg, "lineage_dominance_top1_warning_threshold", 0.75), default=0.75, lo=0.0, hi=1.0)
    top3_thr = _bounded(getattr(cfg, "lineage_dominance_top3_warning_threshold", 0.90), default=0.90, lo=0.0, hi=1.0)
    entropy_thr = _bounded(getattr(cfg, "lineage_entropy_warning_threshold", 0.50), default=0.50, lo=0.0)
    low_active_thr = int(max(0, getattr(cfg, "lineage_low_active_count_warning_threshold", 2)))
    out: Dict[str, float] = {
        "lineage_diversity_schema_version": float(LINEAGE_DIVERSITY_SCHEMA_VERSION),
        "lineage_diversity_reporting_enable": float(int(lineage_diversity_reporting_enabled(cfg) if cfg is not None else True)),
        "lineage_strategy_descriptors_enable": float(int(lineage_strategy_descriptors_enabled(cfg) if cfg is not None else True)),
        "lineage_branch_candidate_reporting_enable": float(int(lineage_branch_candidate_reporting_enabled(cfg) if cfg is not None else True)),
        "lineage_active_count": active_count,
        "lineage_extinct_count": 0.0,
        "lineage_alive_count_total": alive_total,
        "lineage_top1_alive_share": top1,
        "lineage_top3_alive_share": top3,
        "lineage_top5_alive_share": _top_share(alive_counts, 5),
        "lineage_top1_birth_share": _top_share(birth_counts, 1),
        "lineage_top3_birth_share": _top_share(birth_counts, 3),
        "lineage_top5_birth_share": _top_share(birth_counts, 5),
        "lineage_top1_death_share": _top_share(death_counts, 1),
        "lineage_top3_death_share": _top_share(death_counts, 3),
        "lineage_alive_share_entropy": alive_entropy,
        "lineage_birth_share_entropy": birth_entropy,
        "lineage_effective_alive_lineages": float(np.exp(alive_entropy)) if alive_entropy > 0.0 else (1.0 if alive_total > 0.0 else 0.0),
        "lineage_effective_birth_lineages": float(np.exp(birth_entropy)) if birth_entropy > 0.0 else (1.0 if sum(birth_counts) > 0.0 else 0.0),
    }
    out["lineage_top1_dominance_warning"] = float(int(alive_total > 0.0 and top1 >= top1_thr))
    out["lineage_top3_dominance_warning"] = float(int(alive_total > 0.0 and top3 >= top3_thr))
    out["lineage_entropy_collapse_warning"] = float(int(active_count > 1.0 and alive_entropy < entropy_thr))
    out["lineage_low_active_count_warning"] = float(int(alive_total > 0.0 and active_count < float(low_active_thr)))
    out["lineage_collapse_warning_any"] = float(
        int(
            bool(out["lineage_top1_dominance_warning"])
            or bool(out["lineage_top3_dominance_warning"])
            or bool(out["lineage_entropy_collapse_warning"])
            or bool(out["lineage_low_active_count_warning"])
        )
    )
    out["lineage_strategy_descriptor_count"] = float(len(rows))

    std_fields = {
        "lineage_patch_food_amount_mean": "lineage_patch_food_amount_mean_std",
        "lineage_patch_cover_mean": "lineage_patch_cover_mean_std",
        "lineage_patch_shelter_mean": "lineage_patch_shelter_mean_std",
        "lineage_patch_scent_mean": "lineage_patch_scent_mean_std",
        "lineage_patch_waste_mean": "lineage_patch_waste_mean_std",
        "lineage_patch_density_mean": "lineage_patch_density_mean_std",
        "lineage_patch_recovery_activity_mean": "lineage_patch_recovery_activity_mean_std",
        "lineage_birth_patch_waste_mean": "lineage_birth_patch_waste_mean_std",
        "lineage_birth_patch_shelter_mean": "lineage_birth_patch_shelter_mean_std",
        "lineage_birth_patch_recovery_activity_mean": "lineage_birth_patch_recovery_activity_mean_std",
        "lineage_predator_damage_mean": "lineage_predator_damage_mean_std",
        "lineage_predator_damage_prevented_mean": "lineage_predator_damage_prevented_mean_std",
        "lineage_reproduction_rate": "lineage_reproduction_rate_std",
    }
    for src, dst in std_fields.items():
        vals = [_finite_float(r.get(src, 0.0), 0.0) for r in rows]
        out[dst] = float(np.std(np.asarray(vals, dtype=np.float32))) if vals else 0.0
    return {k: _finite_float(v, 0.0) for k, v in out.items()}


def lineage_branch_candidates(descriptors: Iterable[Dict[str, Any]], cfg: Any = None) -> List[Dict[str, float | str]]:
    """Return deterministic diagnostic-only branch candidate records."""
    if cfg is not None and not lineage_branch_candidate_reporting_enabled(cfg):
        return []
    rows = [dict(r) for r in list(descriptors or [])]
    min_alive = int(max(0, getattr(cfg, "lineage_branch_candidate_min_alive_count", 1))) if cfg is not None else 1
    min_age = int(max(0, getattr(cfg, "lineage_branch_candidate_min_age", 0))) if cfg is not None else 0
    top_n = int(max(0, getattr(cfg, "lineage_branch_candidate_top_n", 5))) if cfg is not None else 5
    rows = [
        r for r in rows
        if _finite_float(r.get("lineage_alive_count", 0.0), 0.0) >= float(min_alive)
        and (min_age <= 0 or _finite_float(r.get("lineage_age_or_duration", 0.0), 0.0) >= float(min_age))
    ]
    if not rows or top_n <= 0:
        return []

    categories: List[tuple[str, str, bool]] = [
        ("dominant_lineage", "lineage_alive_share", True),
        ("high_reproduction_lineage", "lineage_reproduction_rate", True),
        ("high_shelter_use_lineage", "lineage_patch_shelter_mean", True),
        ("high_cover_use_lineage", "lineage_patch_cover_mean", True),
        ("low_waste_lineage", "lineage_patch_waste_mean", False),
        ("high_recovery_use_lineage", "lineage_patch_recovery_activity_mean", True),
        ("low_predator_damage_lineage", "lineage_predator_damage_mean", False),
        ("rare_persistent_lineage", "lineage_alive_share", False),
    ]
    candidates: List[Dict[str, float | str]] = []
    seen: set[tuple[str, int]] = set()
    for reason, key, high_is_better in categories:
        ranked = []
        for row in rows:
            lineage_id = int(_finite_float(row.get("lineage_id", 0.0), 0.0))
            val = _finite_float(row.get(key, 0.0), 0.0)
            if reason == "low_predator_damage_lineage":
                score = 1.0 / (1.0 + max(0.0, val))
            elif high_is_better:
                score = val
            else:
                score = 1.0 - val
            ranked.append((float(score), lineage_id, row))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        if not ranked:
            continue
        score, lineage_id, row = ranked[0]
        key_tuple = (reason, lineage_id)
        if key_tuple in seen:
            continue
        seen.add(key_tuple)
        candidates.append({
            "candidate_rank": 0.0,
            "candidate_lineage_id": float(lineage_id),
            "candidate_reason": reason,
            "candidate_alive_count": _finite_float(row.get("lineage_alive_count", 0.0), 0.0),
            "candidate_birth_count": _finite_float(row.get("lineage_birth_count", 0.0), 0.0),
            "candidate_alive_share": _finite_float(row.get("lineage_alive_share", 0.0), 0.0),
            "candidate_birth_share": _finite_float(row.get("lineage_birth_share", 0.0), 0.0),
            "candidate_patch_cover_mean": _finite_float(row.get("lineage_patch_cover_mean", 0.0), 0.0),
            "candidate_patch_shelter_mean": _finite_float(row.get("lineage_patch_shelter_mean", 0.0), 0.0),
            "candidate_patch_waste_mean": _finite_float(row.get("lineage_patch_waste_mean", 0.0), 0.0),
            "candidate_patch_density_mean": _finite_float(row.get("lineage_patch_density_mean", 0.0), 0.0),
            "candidate_patch_recovery_activity_mean": _finite_float(
                row.get("lineage_patch_recovery_activity_mean", 0.0),
                0.0,
            ),
            "candidate_predator_damage_mean": _finite_float(row.get("lineage_predator_damage_mean", 0.0), 0.0),
            "candidate_reproduction_rate": _finite_float(row.get("lineage_reproduction_rate", 0.0), 0.0),
            "candidate_score_diagnostic_only": _finite_float(score, 0.0),
        })
    candidates.sort(
        key=lambda row: (
            str(row.get("candidate_reason", "")),
            -_finite_float(row.get("candidate_score_diagnostic_only", 0.0), 0.0),
            int(_finite_float(row.get("candidate_lineage_id", 0.0), 0.0)),
        )
    )
    out = candidates[:top_n]
    for idx, row in enumerate(out, start=1):
        row["candidate_rank"] = float(idx)
    return out


def patch_observation_validation_metrics(env: Any, obs: Any = None) -> Dict[str, float]:
    """Return lightweight validation diagnostics for patch observations."""
    cfg = getattr(env, "cfg", None)
    if not patch_world_observation_enabled(cfg):
        return {
            "patch_observation_min": 0.0,
            "patch_observation_max": 0.0,
            "patch_observation_nan_count": 0.0,
        }
    if obs is None:
        compact_positions = getattr(env, "agent_positions_by_id", None)
        if compact_positions is None:
            alive = np.asarray(getattr(env, "agent_alive"), dtype=bool)
            ids = np.where(alive)[0].astype(np.int64)
        else:
            ids = np.asarray(sorted(int(agent_id) for agent_id in compact_positions), dtype=np.int64)
        obs = env.get_local_observations(ids) if ids.size else np.empty((0, patch_world_local_observation_size(cfg)), dtype=np.float32)
    arr = np.asarray(obs, dtype=np.float32)
    if arr.size <= 0:
        return {
            "patch_observation_min": 0.0,
            "patch_observation_max": 0.0,
            "patch_observation_nan_count": 0.0,
        }
    return {
        "patch_observation_min": float(np.nanmin(arr)),
        "patch_observation_max": float(np.nanmax(arr)),
        "patch_observation_nan_count": float(np.isnan(arr).sum()),
    }


def apply_patch_world_food_step(env: Any, abs_step: int) -> float:
    """Run the deterministic Patch World food update for one environment tick."""
    _ = int(abs_step)
    total = regrow_patch_food(env)
    setattr(env, "patch_food_regrowth_total_last", float(total))
    return float(total)


def apply_patch_world_noop(env: Any, abs_step: int) -> None:
    """Compatibility no-op for older callers.

    Step 3 uses apply_patch_world_food_step; this remains available for tests or
    downstream imports that still reference the Step 1/2 symbol.
    """
    _ = env
    _ = int(abs_step)
