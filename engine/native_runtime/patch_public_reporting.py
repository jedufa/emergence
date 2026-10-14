from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


_PATCH_TENSOR_FIELDS = (
    "agent_grid",
    "alive_mask",
    "slot_to_agent_id",
    "pos_y",
    "pos_x",
    "food_grid",
    "patch_food_amount",
    "patch_fertility",
    "patch_cover",
    "patch_shelter",
    "patch_movement_cost",
    "patch_scent",
    "patch_waste",
)

_PATCH_P2_TENSOR_FIELDS = ("patch_recovery_activity",)

_PATCH_TOTAL_FIELDS = (
    "patch_food_harvest_total",
    "patch_scent_emitted_total",
    "patch_scent_decayed_total",
    "patch_waste_emitted_total",
    "patch_waste_decayed_total",
    "patch_food_regrowth_total",
    "patch_regrowth_base_total",
    "patch_regrowth_effective_total",
    "patch_regrowth_suppressed_total",
    "patch_regrowth_multiplier_mean",
    "patch_regrowth_multiplier_min",
    "patch_regrowth_multiplier_p10",
    "patch_regrowth_multiplier_p90",
    "patch_waste_regrowth_penalty_mean",
    "patch_waste_regrowth_penalty_max",
    "patch_density_regrowth_penalty_mean",
    "patch_density_regrowth_penalty_max",
    "patch_high_waste_regrowth_suppression_frac",
    "patch_high_density_regrowth_suppression_frac",
    "patch_regrowth_after_waste_total",
    "patch_regrowth_after_recovery_total",
    "patch_recovery_added_regrowth_total",
)

_PATCH_NONNEGATIVE_TOTAL_FIELDS = (
    "patch_food_harvest_total",
    "patch_scent_emitted_total",
    "patch_scent_decayed_total",
    "patch_waste_emitted_total",
    "patch_waste_decayed_total",
    "patch_food_regrowth_total",
    "patch_regrowth_base_total",
    "patch_regrowth_effective_total",
    "patch_regrowth_suppressed_total",
    "patch_regrowth_after_waste_total",
    "patch_regrowth_after_recovery_total",
    "patch_recovery_added_regrowth_total",
)

_PATCH_FRACTION_FIELDS = (
    "patch_high_waste_regrowth_suppression_frac",
    "patch_high_density_regrowth_suppression_frac",
)

_PATCH_WASTE_DENSITY_CONSEQUENCE_FIELDS = (
    "patch_regrowth_suppressed_total",
    "patch_waste_regrowth_penalty_mean",
    "patch_waste_regrowth_penalty_max",
    "patch_density_regrowth_penalty_mean",
    "patch_density_regrowth_penalty_max",
    "patch_high_waste_regrowth_suppression_frac",
    "patch_high_density_regrowth_suppression_frac",
)

_LINEAGE_INTEGER_FIELDS = (
    "lineage_diversity_schema_version",
    "lineage_diversity_reporting_enable",
    "lineage_strategy_descriptors_enable",
    "lineage_branch_candidate_reporting_enable",
    "lineage_top1_dominance_warning",
    "lineage_top3_dominance_warning",
    "lineage_entropy_collapse_warning",
    "lineage_low_active_count_warning",
    "lineage_collapse_warning_any",
)

_WDL_PUBLIC_COUNT_FIELDS = (
    "wdl_delayed_hazard_event_count",
    "wdl_delayed_hazard_agent_cue_exposure_count",
    "wdl_delayed_hazard_agent_active_exposure_count",
    "wdl_delayed_hazard_agents_damaged",
    "wdl_delayed_hazard_cue_without_active_count",
    "wdl_delayed_hazard_active_without_prior_cue_count",
    "wdl_cue_seen_agents",
    "wdl_cue_exposed_next_move_away_count",
    "wdl_cue_exposed_next_move_toward_count",
    "wdl_cue_exposed_stayed_count",
    "wdl_pre_hazard_exit_count",
    "wdl_hazard_enter_count",
    "wdl_hazard_exposure_after_cue_count",
)

_WDL_PUBLIC_FLOAT_FIELDS = (
    "wdl_delayed_hazard_cue_cell_count",
    "wdl_delayed_hazard_active_cell_count",
    "wdl_delayed_hazard_damage_total",
    "wdl_delayed_hazard_damage_mean",
    "wdl_delayed_hazard_damage_max",
    "wdl_delayed_hazard_cue_to_active_delay_mean",
    "wdl_hazard_avoidance_proxy",
)

_WDL_PUBLIC_LINEAGE_FIELDS = (
    "wdl_hazard_damage_by_lineage_json",
    "wdl_cue_exposure_by_lineage_json",
    "wdl_hazard_exposure_by_lineage_json",
)

_JUNGLE_ACTIVE_FLAGS = (
    "jungle_predator_pressure_enable",
    "jungle_reproduction_emissions_enable",
    "jungle_juvenile_predator_vulnerability_enable",
    "jungle_terrain_protection_enable",
    "jungle_ecological_recovery_enable",
)

_JUNGLE_COUNT_FIELDS = (
    "jungle_predator_exposure_count",
    "jungle_predator_events",
    "jungle_predator_deaths",
    "jungle_juvenile_predator_events",
    "jungle_juvenile_predator_deaths",
    "jungle_adult_predator_events",
    "jungle_birth_emission_count",
)

_JUNGLE_TOTAL_FIELDS = (
    "jungle_predator_damage_total",
    "jungle_predator_damage_raw_total",
    "jungle_predator_damage_effective_total",
    "jungle_predator_damage_prevented_total",
    "jungle_juvenile_predator_damage_total",
    "jungle_adult_predator_damage_total",
    "jungle_birth_scent_emitted_total",
    "jungle_birth_waste_emitted_total",
    "patch_recovery_positive_total",
    "patch_recovery_negative_total",
    "patch_recovery_decay_total",
    "jungle_recovery_regrowth_boost_total",
    "jungle_recovery_regrowth_boost_mean",
    "jungle_recovery_regrowth_boost_max",
    "jungle_recovery_regrowth_boost_p90",
)


@dataclass(frozen=True, slots=True)
class _PatchStatsView:
    patch_habitat_lineage_id: int


@dataclass(frozen=True, slots=True)
class _PatchAgentView:
    agent_id: int
    stats: _PatchStatsView


@dataclass(slots=True)
class _PatchEnvironmentView:
    cfg: Any
    grid: np.ndarray
    agent_grid: np.ndarray
    agent_positions_by_id: Mapping[int, tuple[int, int]]
    patch_food_amount: np.ndarray
    patch_fertility: np.ndarray
    patch_cover: np.ndarray
    patch_shelter: np.ndarray
    patch_movement_cost: np.ndarray
    patch_scent: np.ndarray
    patch_waste: np.ndarray
    patch_recovery_activity: np.ndarray
    local_density_map: np.ndarray

    def get_local_observations(self, agent_ids: np.ndarray) -> np.ndarray:
        from env.gridworld_pkg.constants import FOOD
        from env.gridworld_pkg.patch_world import (
            patch_world_normalized_observation_maps,
        )

        ids = np.asarray(agent_ids, dtype=np.int64)
        radius = int(getattr(self.cfg, "view_radius", 0))
        width = 2 * radius + 1
        if ids.size == 0:
            return np.empty((0, 0), dtype=np.float32)
        if np.any(ids < 0):
            raise ValueError("native Patch reporting observation IDs are out of range")
        positions: list[tuple[int, int]] = []
        for agent_id in ids.reshape(-1).tolist():
            position = self.agent_positions_by_id.get(int(agent_id))
            if position is None:
                raise ValueError("native Patch reporting observation IDs include dead agents")
            positions.append(position)

        ys = np.asarray([position[0] for position in positions], dtype=np.int32)
        xs = np.asarray([position[1] for position in positions], dtype=np.int32)
        occupied = (self.agent_grid != -1).astype(np.uint8, copy=False)
        food = (self.grid == FOOD).astype(np.uint8, copy=False)
        occupied_padded = np.pad(occupied, radius, mode="constant", constant_values=0)
        food_padded = np.pad(food, radius, mode="constant", constant_values=0)
        yp = ys + radius
        xp = xs + radius
        dy = np.arange(-radius, radius + 1, dtype=np.int32).reshape(1, width, 1)
        dx = np.arange(-radius, radius + 1, dtype=np.int32).reshape(1, 1, width)
        y_idx = yp.reshape(-1, 1, 1) + dy
        x_idx = xp.reshape(-1, 1, 1) + dx
        legacy = np.where(
            occupied_padded[y_idx, x_idx] != 0,
            np.float32(-1.0),
            food_padded[y_idx, x_idx].astype(np.float32, copy=False),
        )
        channels = [legacy]
        for values in patch_world_normalized_observation_maps(self):
            padded = np.pad(
                np.asarray(values, dtype=np.float32),
                radius,
                mode="constant",
                constant_values=0.0,
            )
            channels.append(padded[y_idx, x_idx].astype(np.float32, copy=False))
        return np.stack(channels, axis=1).reshape(ids.size, -1).astype(
            np.float32,
            copy=False,
        )


def native_patch_public_metrics(
    *,
    raw_result: Mapping[str, Any],
    checkpoint_projection: Mapping[str, Any],
    lineage_input: Mapping[str, Any],
    existing_metrics: Mapping[str, Any],
    env_cfg: Any,
    metrics_generation: int,
) -> dict[str, Any]:
    """Reduce one immutable Patch reporting frame through canonical helpers."""

    if raw_result.get("promotion_accepted") is not False:
        raise ValueError("native Patch reporting requires promotion_accepted=false")
    profile = str(lineage_input.get("patch_profile", "") or "")
    if profile not in {"P0", "P1", "P2"}:
        raise ValueError("native Patch reporting requires a finite P0/P1/P2 profile")
    tensors = checkpoint_projection.get("tensors")
    metadata = checkpoint_projection.get("metadata")
    if not isinstance(tensors, Mapping) or not isinstance(metadata, Mapping):
        raise ValueError("native Patch reporting requires projection tensors and metadata")
    required_tensor_fields = (
        *_PATCH_TENSOR_FIELDS,
        *(_PATCH_P2_TENSOR_FIELDS if profile == "P2" else ()),
    )
    missing = [name for name in required_tensor_fields if name not in tensors]
    if missing:
        raise ValueError(f"native Patch reporting projection is missing tensors: {missing}")
    if profile != "P2" and any(name in tensors for name in _PATCH_P2_TENSOR_FIELDS):
        raise ValueError("native Patch reporting projection contains inactive P2 recovery state")

    arrays = {
        name: _tensor_array(tensors[name], name=name)
        for name in required_tensor_fields
    }
    env = _environment_view(arrays, env_cfg=env_cfg)
    agents = _agent_views(lineage_input, env=env)
    birth_contexts = _event_context_rows(
        raw_result,
        name="patch_birth_context_rows",
        expected_count=_strict_count(raw_result, "committed_birth_rows"),
        generation=metrics_generation,
        shape=env.agent_grid.shape,
    )
    death_contexts = _event_context_rows(
        raw_result,
        name="patch_death_context_rows",
        expected_count=_strict_count(raw_result, "death_rows"),
        generation=metrics_generation,
        shape=env.agent_grid.shape,
    )
    jungle_active = any(
        bool(getattr(env_cfg, name, False)) for name in _JUNGLE_ACTIVE_FLAGS
    )
    predator_contexts: list[dict[str, Any]] = []
    birth_emission_contexts: list[dict[str, Any]] = []
    if jungle_active:
        if metadata.get("native_runtime_jungle_execution_enable") is not True:
            raise ValueError(
                "active native Jungle reporting requires native Jungle execution"
            )
        predator_contexts = _event_context_rows(
            raw_result,
            name="jungle_predator_event_context_rows",
            expected_count=_strict_count(raw_result, "jungle_predator_events"),
            generation=metrics_generation,
            shape=env.agent_grid.shape,
        )
        birth_emission_contexts = _event_context_rows(
            raw_result,
            name="jungle_birth_emission_context_rows",
            expected_count=_strict_count(raw_result, "jungle_birth_emission_count"),
            generation=metrics_generation,
            shape=env.agent_grid.shape,
        )

    from env.gridworld_pkg.patch_world import (
        LINEAGE_DIVERSITY_SCHEMA_VERSION,
        PATCH_WORLD_DESCRIPTOR_SCHEMA_VERSION,
        PATCH_WORLD_OBSERVATION_SCHEMA_VERSION,
        compute_local_density_map,
        lineage_branch_candidate_reporting_enabled,
        lineage_branch_candidates,
        lineage_diversity_reporting_enabled,
        lineage_habitat_descriptors,
        lineage_habitat_metrics,
        lineage_strategy_descriptors,
        lineage_strategy_descriptors_enabled,
        lineage_strategy_metrics,
        patch_event_contexts_summary,
        patch_food_metrics,
        patch_observation_validation_metrics,
        patch_occupied_descriptor_metrics,
        patch_recovery_metrics,
        patch_terrain_metrics,
        patch_trace_density_metrics,
        patch_world_local_observation_channel_count,
        patch_world_descriptor_enabled,
        patch_world_observation_enabled,
    )
    from env.gridworld_pkg.jungle import (
        compute_terrain_damage_multiplier_map,
        jungle_predator_risk_metrics,
    )

    out: dict[str, Any] = {}
    out.update(patch_food_metrics(env))
    out.update(patch_terrain_metrics(env))
    compute_local_density_map(env)
    out.update(patch_trace_density_metrics(env))
    out.update(patch_recovery_metrics(env))
    out.update(jungle_predator_risk_metrics(env))
    out["jungle_reproduction_emissions_enable"] = int(
        bool(getattr(env_cfg, "jungle_reproduction_emissions_enable", False))
    )
    out["jungle_juvenile_predator_vulnerability_enable"] = int(
        bool(getattr(env_cfg, "jungle_juvenile_predator_vulnerability_enable", False))
    )
    out["jungle_terrain_protection_enable"] = int(
        bool(getattr(env_cfg, "jungle_terrain_protection_enable", False))
    )
    damage_multiplier_map = compute_terrain_damage_multiplier_map(env)
    occupied = np.asarray(env.agent_grid, dtype=np.int32) != -1
    out["occupied_predator_damage_multiplier_mean"] = (
        float(np.mean(damage_multiplier_map[occupied]))
        if damage_multiplier_map.size and bool(np.any(occupied))
        else 1.0
    )

    out["patch_world_descriptor_schema_version"] = int(
        PATCH_WORLD_DESCRIPTOR_SCHEMA_VERSION
    )
    out["patch_world_diagnostics_enable"] = int(
        bool(getattr(env_cfg, "patch_world_diagnostics_enable", True))
    )
    out["patch_world_lineage_descriptors_enable"] = int(
        bool(getattr(env_cfg, "patch_world_lineage_descriptors_enable", True))
    )
    out["patch_world_waste_consequence_enable"] = int(
        bool(getattr(env_cfg, "patch_world_waste_consequence_enable", False))
    )
    out["lineage_diversity_schema_version"] = int(LINEAGE_DIVERSITY_SCHEMA_VERSION)
    out["lineage_diversity_reporting_enable"] = int(
        lineage_diversity_reporting_enabled(env_cfg)
    )
    out["lineage_strategy_descriptors_enable"] = int(
        lineage_strategy_descriptors_enabled(env_cfg)
    )
    out["lineage_branch_candidate_reporting_enable"] = int(
        lineage_branch_candidate_reporting_enabled(env_cfg)
    )
    out["patch_observation_enable"] = int(patch_world_observation_enabled(env_cfg))
    out["patch_observation_schema_version"] = int(
        PATCH_WORLD_OBSERVATION_SCHEMA_VERSION
    )
    out["patch_observation_channel_count"] = int(
        patch_world_local_observation_channel_count(env_cfg)
    )
    out["patch_observation_size"] = _strict_nonnegative_int(
        metadata.get("obs_size"),
        context="projection metadata obs_size",
    )
    if patch_world_descriptor_enabled(env_cfg):
        out.update(patch_occupied_descriptor_metrics(env))
        out.update(patch_event_contexts_summary("birth", birth_contexts))
        out.update(patch_event_contexts_summary("death", death_contexts))
        out.update(patch_event_contexts_summary("predator_event", predator_contexts))
        out.update(
            patch_event_contexts_summary(
                "jungle_birth_emission",
                birth_emission_contexts,
            )
        )
        if bool(getattr(env_cfg, "patch_world_lineage_descriptors_enable", True)):
            descriptors = lineage_habitat_descriptors(env, agents)
            out.update(lineage_habitat_metrics(descriptors))
            out["lineage_habitat_descriptors_json"] = _canonical_json(descriptors)
            if lineage_strategy_descriptors_enabled(env_cfg):
                strategy_descriptors = lineage_strategy_descriptors(
                    env,
                    agents,
                    birth_contexts=birth_contexts,
                    death_contexts=death_contexts,
                    predator_event_contexts=predator_contexts,
                    cfg=env_cfg,
                )
                out.update(
                    lineage_strategy_metrics(strategy_descriptors, cfg=env_cfg)
                )
                candidates = lineage_branch_candidates(
                    strategy_descriptors,
                    cfg=env_cfg,
                )
                out["lineage_branch_candidate_count"] = float(len(candidates))
                out["lineage_strategy_descriptors_json"] = _canonical_json(
                    strategy_descriptors
                )
                out["lineage_branch_candidates_json"] = _canonical_json(candidates)
        out.update(patch_observation_validation_metrics(env))

    for name in _LINEAGE_INTEGER_FIELDS:
        if name in out:
            out[name] = int(out[name])

    out.update(_validated_patch_totals(raw_result, profile=profile))
    movement_total = _finite_float(
        raw_result.get("patch_movement_energy_total"),
        context="raw_result.patch_movement_energy_total",
    )
    movement_count = _strict_count(raw_result, "patch_movement_energy_event_count")
    if profile in {"P0", "P1"}:
        if movement_total != 0.0:
            raise ValueError(
                "raw_result.patch_movement_energy_total must be zero for "
                f"Patch profile {profile}"
            )
        if movement_count != 0:
            raise ValueError(
                "raw_result.patch_movement_energy_event_count must be zero for "
                f"Patch profile {profile}"
            )
    out["patch_movement_energy_extra_total"] = movement_total
    out["patch_movement_energy_extra_mean"] = (
        movement_total / float(movement_count) if movement_count > 0 else 0.0
    )
    _add_behavior_proxies(
        out,
        raw_result=raw_result,
        existing_metrics=existing_metrics,
    )
    if bool(getattr(env_cfg, "wdl_delayed_hazard_enable", False)):
        out.update(_validated_wdl_public_metrics(raw_result))
    if jungle_active:
        out.update(_validated_jungle_public_metrics(raw_result, env_cfg=env_cfg))
    return out


def _validated_jungle_public_metrics(
    raw_result: Mapping[str, Any],
    *,
    env_cfg: Any,
) -> dict[str, Any]:
    counts = {name: _strict_count(raw_result, name) for name in _JUNGLE_COUNT_FIELDS}
    totals = {
        name: _finite_float(raw_result.get(name), context=f"raw_result.{name}")
        for name in _JUNGLE_TOTAL_FIELDS
    }
    if any(value < 0.0 for value in totals.values()):
        raise ValueError("native Jungle public totals must be non-negative")

    events = counts["jungle_predator_events"]
    exposures = counts["jungle_predator_exposure_count"]
    deaths = counts["jungle_predator_deaths"]
    juvenile_events = counts["jungle_juvenile_predator_events"]
    juvenile_deaths = counts["jungle_juvenile_predator_deaths"]
    adult_events = counts["jungle_adult_predator_events"]
    if events > exposures or deaths > events or juvenile_deaths > deaths:
        raise ValueError("native Jungle predator count conservation failed")
    if juvenile_events + adult_events != events:
        raise ValueError("native Jungle juvenile/adult event partition failed")
    if counts["jungle_birth_emission_count"] > _strict_count(
        raw_result,
        "committed_birth_rows",
    ):
        raise ValueError("native Jungle emission count exceeds successful births")

    effective = totals["jungle_predator_damage_effective_total"]
    raw = totals["jungle_predator_damage_raw_total"]
    prevented = totals["jungle_predator_damage_prevented_total"]
    if not math.isclose(
        totals["jungle_predator_damage_total"],
        effective,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("native Jungle damage total contradicts effective damage")
    if not math.isclose(raw - effective, prevented, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError("native Jungle prevented damage does not conserve raw damage")
    if not math.isclose(
        totals["jungle_juvenile_predator_damage_total"]
        + totals["jungle_adult_predator_damage_total"],
        effective,
        rel_tol=1e-10,
        abs_tol=1e-10,
    ):
        raise ValueError("native Jungle juvenile/adult damage partition failed")

    multipliers = {
        name: _finite_float(raw_result.get(name), context=f"raw_result.{name}")
        for name in (
            "jungle_predator_damage_multiplier_mean",
            "jungle_predator_damage_multiplier_min",
            "jungle_predator_damage_multiplier_p10",
            "jungle_predator_damage_multiplier_p90",
        )
    }
    if any(value < 0.0 or value > 1.0 for value in multipliers.values()):
        raise ValueError("native Jungle damage multipliers must be in [0, 1]")

    out: dict[str, Any] = {
        "jungle_predator_pressure_enable": int(
            bool(getattr(env_cfg, "jungle_predator_pressure_enable", False))
        ),
        "jungle_ecological_recovery_enable": int(
            bool(getattr(env_cfg, "jungle_ecological_recovery_enable", False))
        ),
        "jungle_predator_events": float(events),
        "jungle_predator_event_rate": (
            float(events) / float(exposures) if exposures else 0.0
        ),
        "jungle_predator_damage_mean": effective / float(events) if events else 0.0,
        "jungle_predator_event_damage_multiplier_mean": multipliers[
            "jungle_predator_damage_multiplier_mean"
        ],
        "jungle_predator_event_damage_prevented_mean": (
            prevented / float(events) if events else 0.0
        ),
        "jungle_predator_deaths": float(deaths),
        "jungle_juvenile_vs_adult_predator_event_ratio": (
            float(juvenile_events) / float(adult_events)
            if adult_events
            else float(juvenile_events)
        ),
    }
    out.update(
        {
            name: float(value)
            for name, value in counts.items()
            if name != "jungle_predator_exposure_count"
        }
    )
    out.update(totals)
    out.update(multipliers)
    return out


def _validated_wdl_public_metrics(raw_result: Mapping[str, Any]) -> dict[str, Any]:
    enabled = _strict_nonnegative_int(
        raw_result.get("wdl_delayed_hazard_enable"),
        context="raw_result.wdl_delayed_hazard_enable",
    )
    if enabled != 1:
        raise ValueError("active native WDL reporting requires wdl_delayed_hazard_enable=1")

    out: dict[str, Any] = {"wdl_delayed_hazard_enable": 1}
    counts = {
        name: _strict_count(raw_result, name)
        for name in _WDL_PUBLIC_COUNT_FIELDS
    }
    out.update({name: float(value) for name, value in counts.items()})

    floats = {
        name: _finite_float(raw_result.get(name), context=f"raw_result.{name}")
        for name in _WDL_PUBLIC_FLOAT_FIELDS
    }
    if any(value < 0.0 for value in floats.values()):
        raise ValueError("native WDL public metrics must be non-negative")
    if floats["wdl_hazard_avoidance_proxy"] > 1.0:
        raise ValueError("native WDL avoidance proxy must be in [0, 1]")

    active_exposures = counts["wdl_delayed_hazard_agent_active_exposure_count"]
    expected_damage_mean = (
        floats["wdl_delayed_hazard_damage_total"] / float(active_exposures)
        if active_exposures > 0
        else 0.0
    )
    if not math.isclose(
        floats["wdl_delayed_hazard_damage_mean"],
        expected_damage_mean,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("native WDL damage mean contradicts total/exposure count")

    movement_total = sum(
        counts[name]
        for name in (
            "wdl_cue_exposed_next_move_away_count",
            "wdl_cue_exposed_next_move_toward_count",
            "wdl_cue_exposed_stayed_count",
        )
    )
    expected_avoidance = (
        float(counts["wdl_cue_exposed_next_move_away_count"])
        / float(max(1, movement_total))
    )
    if not math.isclose(
        floats["wdl_hazard_avoidance_proxy"],
        expected_avoidance,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("native WDL avoidance proxy contradicts action counts")
    out.update(floats)

    lineage = {
        name: _validated_wdl_lineage_json(raw_result.get(name), name=name)
        for name in _WDL_PUBLIC_LINEAGE_FIELDS
    }
    if lineage["wdl_cue_exposure_by_lineage_json"]:
        raise ValueError("native WDL cue lineage reporting must remain unavailable")
    damage_by_lineage = lineage["wdl_hazard_damage_by_lineage_json"]
    if not math.isclose(
        sum(damage_by_lineage.values()),
        floats["wdl_delayed_hazard_damage_total"],
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("native WDL lineage damage contradicts total damage")
    out.update({name: _canonical_json(value) for name, value in lineage.items()})
    return out


def _validated_wdl_lineage_json(value: Any, *, name: str) -> dict[str, float]:
    if type(value) is not str:
        raise ValueError(f"raw_result.{name} must be canonical JSON text")
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"raw_result.{name} is not valid JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"raw_result.{name} must encode an object")
    out: dict[str, float] = {}
    for key, raw_amount in decoded.items():
        if type(key) is not str or not key.isdigit() or int(key) < 0:
            raise ValueError(f"raw_result.{name} has an invalid lineage key")
        amount = _finite_float(raw_amount, context=f"raw_result.{name}[{key!r}]")
        if amount < 0.0:
            raise ValueError(f"raw_result.{name} has a negative lineage value")
        out[str(int(key))] = amount
    canonical = _canonical_json(out)
    if value != canonical:
        raise ValueError(f"raw_result.{name} is not canonical JSON")
    return out


def _validated_patch_totals(
    raw_result: Mapping[str, Any],
    *,
    profile: str,
) -> dict[str, float]:
    values: dict[str, float] = {}
    for name in _PATCH_TOTAL_FIELDS:
        value = _finite_float(raw_result.get(name), context=f"raw_result.{name}")
        if name in _PATCH_NONNEGATIVE_TOTAL_FIELDS and value < 0.0:
            raise ValueError(f"raw_result.{name} must be non-negative")
        if name in _PATCH_FRACTION_FIELDS and not 0.0 <= value <= 1.0:
            raise ValueError(f"raw_result.{name} must be in [0, 1]")
        values[name] = value

    if profile in {"P0", "P1"}:
        for name in _PATCH_WASTE_DENSITY_CONSEQUENCE_FIELDS:
            if values[name] != 0.0:
                raise ValueError(
                    f"raw_result.{name} must be zero for Patch profile {profile}"
                )
    return values


def _environment_view(
    arrays: Mapping[str, np.ndarray],
    *,
    env_cfg: Any,
) -> _PatchEnvironmentView:
    agent_grid = np.asarray(arrays["agent_grid"], dtype=np.int32)
    if agent_grid.ndim != 2:
        raise ValueError("native Patch reporting agent_grid must be two-dimensional")
    shape = agent_grid.shape
    world_tensor_fields = [*_PATCH_TENSOR_FIELDS[5:]]
    world_tensor_fields.extend(
        name for name in _PATCH_P2_TENSOR_FIELDS if name in arrays
    )
    for name in world_tensor_fields:
        if np.asarray(arrays[name]).shape != shape:
            raise ValueError(f"native Patch reporting tensor {name} does not align with agent_grid")

    alive_mask = np.asarray(arrays["alive_mask"], dtype=np.bool_)
    slot_ids = np.asarray(arrays["slot_to_agent_id"], dtype=np.int64)
    pos_y = np.asarray(arrays["pos_y"], dtype=np.int32)
    pos_x = np.asarray(arrays["pos_x"], dtype=np.int32)
    if not (alive_mask.ndim == slot_ids.ndim == pos_y.ndim == pos_x.ndim == 1):
        raise ValueError("native Patch reporting row tensors must be one-dimensional")
    if len({alive_mask.size, slot_ids.size, pos_y.size, pos_x.size}) != 1:
        raise ValueError("native Patch reporting row tensor capacities disagree")
    live_ids = slot_ids[alive_mask]
    if np.any(live_ids < 0) or live_ids.size != np.unique(live_ids).size:
        raise ValueError("native Patch reporting live identities are invalid")
    live_agent_grid = np.full(shape, -1, dtype=np.int32)
    agent_positions_by_id: dict[int, tuple[int, int]] = {}
    for slot in np.flatnonzero(alive_mask):
        agent_id = int(slot_ids[slot])
        y = int(pos_y[slot])
        x = int(pos_x[slot])
        if y < 0 or x < 0 or y >= shape[0] or x >= shape[1]:
            raise ValueError(f"native Patch reporting position is invalid for agent {agent_id}")
        if int(live_agent_grid[y, x]) != -1:
            raise ValueError(f"native Patch reporting live positions collide for agent {agent_id}")
        live_agent_grid[y, x] = agent_id
        agent_positions_by_id[agent_id] = (y, x)
    if int(np.count_nonzero(live_agent_grid != -1)) != int(live_ids.size):
        raise ValueError("native Patch reporting live occupancy disagrees with live rows")

    return _PatchEnvironmentView(
        cfg=env_cfg,
        grid=np.asarray(arrays["food_grid"], dtype=np.int32),
        agent_grid=live_agent_grid,
        agent_positions_by_id=agent_positions_by_id,
        patch_food_amount=np.asarray(arrays["patch_food_amount"], dtype=np.float32),
        patch_fertility=np.asarray(arrays["patch_fertility"], dtype=np.float32),
        patch_cover=np.asarray(arrays["patch_cover"], dtype=np.float32),
        patch_shelter=np.asarray(arrays["patch_shelter"], dtype=np.float32),
        patch_movement_cost=np.asarray(arrays["patch_movement_cost"], dtype=np.float32),
        patch_scent=np.asarray(arrays["patch_scent"], dtype=np.float32),
        patch_waste=np.asarray(arrays["patch_waste"], dtype=np.float32),
        patch_recovery_activity=np.asarray(
            arrays.get("patch_recovery_activity", np.zeros(shape, dtype=np.float32)),
            dtype=np.float32,
        ),
        local_density_map=np.zeros(shape, dtype=np.float32),
    )


def _agent_views(
    lineage_input: Mapping[str, Any],
    *,
    env: _PatchEnvironmentView,
) -> tuple[_PatchAgentView, ...]:
    records = lineage_input.get("records")
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise ValueError("native Patch reporting lineage records must be a sequence")
    views: list[_PatchAgentView] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping) or not isinstance(record.get("stats"), Mapping):
            raise ValueError(f"native Patch reporting lineage record {index} is malformed")
        agent_id = _strict_nonnegative_int(
            record.get("agent_id"), context=f"lineage record {index} agent_id"
        )
        if agent_id not in env.agent_positions_by_id:
            raise ValueError(f"native Patch reporting lineage agent {agent_id} is not alive")
        lineage_id = _strict_nonnegative_int(
            record["stats"].get("patch_habitat_lineage_id"),
            context=f"lineage record {index} patch_habitat_lineage_id",
        )
        if lineage_id <= 0:
            raise ValueError("native Patch reporting habitat lineage IDs must be positive")
        views.append(_PatchAgentView(agent_id, _PatchStatsView(lineage_id)))
    views.sort(key=lambda item: item.agent_id)
    if len(views) != len(env.agent_positions_by_id):
        raise ValueError("native Patch reporting lineage/live record counts disagree")
    if len({item.agent_id for item in views}) != len(views):
        raise ValueError("native Patch reporting lineage records contain duplicate IDs")
    return tuple(views)


def _event_context_rows(
    raw_result: Mapping[str, Any],
    *,
    name: str,
    expected_count: int,
    generation: int,
    shape: tuple[int, int],
) -> list[dict[str, Any]]:
    from env.gridworld_pkg.patch_world import PATCH_WORLD_CONTEXT_FIELDS

    value = raw_result.get(name)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"native Patch reporting {name} must be a sequence")
    if len(value) != expected_count:
        raise ValueError(
            f"native Patch reporting {name} count mismatch: "
            f"expected={expected_count} actual={len(value)}"
        )
    rows: list[dict[str, Any]] = []
    previous_sequence = -1
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"native Patch reporting {name}[{index}] must be a mapping")
        row = dict(item)
        if _strict_nonnegative_int(
            row.get("generation"), context=f"{name}[{index}].generation"
        ) != int(generation):
            raise ValueError(f"native Patch reporting {name}[{index}] generation mismatch")
        for field in ("tick", "agent_id", "slot"):
            _strict_nonnegative_int(row.get(field), context=f"{name}[{index}].{field}")
        sequence = _strict_nonnegative_int(
            row.get("event_sequence"), context=f"{name}[{index}].event_sequence"
        )
        if sequence <= previous_sequence:
            raise ValueError(f"native Patch reporting {name} event sequence is not increasing")
        previous_sequence = sequence
        y = _strict_nonnegative_int(row.get("y"), context=f"{name}[{index}].y")
        x = _strict_nonnegative_int(row.get("x"), context=f"{name}[{index}].x")
        if y >= shape[0] or x >= shape[1]:
            raise ValueError(f"native Patch reporting {name}[{index}] position is out of range")
        lineage_id = _strict_nonnegative_int(
            row.get("lineage_id"), context=f"{name}[{index}].lineage_id"
        )
        if lineage_id <= 0:
            raise ValueError(f"native Patch reporting {name}[{index}] lineage_id must be positive")
        for field in PATCH_WORLD_CONTEXT_FIELDS:
            row[field] = _finite_float(
                row.get(field), context=f"{name}[{index}].{field}"
            )
        rows.append(row)
    return rows


def _add_behavior_proxies(
    out: dict[str, Any],
    *,
    raw_result: Mapping[str, Any],
    existing_metrics: Mapping[str, Any],
) -> None:
    gather = _finite_float(
        existing_metrics.get("behavior_gather_total", raw_result.get("action_gather")),
        context="native Patch behavior_gather_total",
    )
    out["behavior_patch_interaction_proxy"] = float(
        gather
        + float(out["patch_food_harvest_total"])
        + float(out.get("birth_patch_count", 0.0))
    )
    out["behavior_risk_exposure_proxy"] = float(
        float(out["patch_waste_occupied_mean"])
        + float(out.get("death_patch_count", 0.0))
    )
    out["behavior_reproduction_proxy"] = float(
        _strict_nonnegative_int(
            existing_metrics.get("birth_attempts", raw_result.get("birth_attempt_rows")),
            context="native Patch birth_attempts",
        )
        + _strict_nonnegative_int(
            existing_metrics.get(
                "successful_births", raw_result.get("committed_birth_rows")
            ),
            context="native Patch successful_births",
        )
        + _finite_float(
            existing_metrics.get(
                "first_repro_frac_alive",
                raw_result.get("first_repro_frac_alive", 0.0),
            ),
            context="native Patch first_repro_frac_alive",
        )
        + float(out.get("birth_patch_count", 0.0))
    )
    out["behavior_protection_proxy"] = float(
        float(out.get("occupied_patch_shelter_mean", 0.0))
        + float(out["patch_shelter_occupied_mean"])
    )
    out["behavior_visible_resource_exposure_proxy"] = float(
        float(out["patch_food_amount_max"])
        * _existing_metric(
            existing_metrics,
            "sensor_channel_patch_food_amount_norm_enabled_fraction",
        )
    )
    out["behavior_visible_hazard_exposure_proxy"] = float(
        float(out["patch_waste_max"])
        * max(
            _existing_metric(
                existing_metrics,
                "sensor_channel_patch_waste_norm_enabled_fraction",
            ),
            _existing_metric(
                existing_metrics,
                "sensor_channel_patch_movement_cost_norm_enabled_fraction",
            ),
        )
    )
    out["behavior_visible_shelter_exposure_proxy"] = float(
        float(out["patch_shelter_mean"])
        * _existing_metric(
            existing_metrics,
            "sensor_channel_patch_shelter_enabled_fraction",
        )
    )
    out["behavior_visible_predator_exposure_proxy"] = 0.0


def _existing_metric(mapping: Mapping[str, Any], name: str) -> float:
    return _finite_float(mapping.get(name, 0.0), context=f"native Patch {name}")


def _tensor_array(value: Any, *, name: str) -> np.ndarray:
    current = value
    detach = getattr(current, "detach", None)
    if callable(detach):
        current = detach()
    cpu = getattr(current, "cpu", None)
    if callable(cpu):
        current = cpu()
    numpy = getattr(current, "numpy", None)
    if callable(numpy):
        current = numpy()
    try:
        array = np.asarray(current)
    except Exception as exc:
        raise ValueError(f"native Patch reporting tensor {name} is unreadable") from exc
    if array.dtype.kind in {"f", "c"} and not bool(np.all(np.isfinite(array))):
        raise ValueError(f"native Patch reporting tensor {name} contains non-finite values")
    return array


def _strict_count(mapping: Mapping[str, Any], name: str) -> int:
    return _strict_nonnegative_int(mapping.get(name), context=f"raw_result.{name}")


def _strict_nonnegative_int(value: Any, *, context: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{context} must be an integer")
    try:
        resolved = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{context} must be an integer") from exc
    if not math.isfinite(numeric) or numeric != float(resolved) or resolved < 0:
        raise ValueError(f"{context} must be a non-negative integer")
    return resolved


def _finite_float(value: Any, *, context: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{context} must be numeric")
    try:
        resolved = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{context} must be numeric") from exc
    if not math.isfinite(resolved):
        raise ValueError(f"{context} must be finite")
    return resolved


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


__all__ = ("native_patch_public_metrics",)
