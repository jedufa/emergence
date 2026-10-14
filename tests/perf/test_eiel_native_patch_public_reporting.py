from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from simulation.metrics_pkg.lineage import native_lineage_agents_from_input
from native_runtime.flat_population_reporting import native_flat_population_reporting_records
from native_runtime.public_metrics_overlay import (
    NATIVE_FLAT_RESIDENT_REPORTING_COLUMNS,
    NATIVE_LINEAGE_METRICS_INPUT_KEY,
    public_metrics_overlay,
)
from tests.perf.test_eiel_native_flat_reporting_projection import (
    _behavior_rows,
    _configs,
    _owner_projection,
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

_PATCH_WASTE_DENSITY_CONSEQUENCE_FIELDS = (
    "patch_regrowth_suppressed_total",
    "patch_waste_regrowth_penalty_mean",
    "patch_waste_regrowth_penalty_max",
    "patch_density_regrowth_penalty_mean",
    "patch_density_regrowth_penalty_max",
    "patch_high_waste_regrowth_suppression_frac",
    "patch_high_density_regrowth_suppression_frac",
)


def _patch_reporting_inputs(
    profile: str,
    *,
    agent_ids: list[int] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]], object, object]:
    if agent_ids is None:
        agent_ids = [30, 10, 20]
    projection = _owner_projection(agent_ids)
    tensors = projection["tensors"]
    assert isinstance(tensors, dict)
    count = len(agent_ids)
    tensors.update(
        {
            "injury": torch.zeros(count, dtype=torch.float32),
            "lineage_depth": torch.zeros(count, dtype=torch.int32),
            "lineage_birth_generation": torch.zeros(count, dtype=torch.int32),
            "lineage_birth_tick": torch.zeros(count, dtype=torch.int32),
            "first_repro_step": torch.full((count,), -1, dtype=torch.int32),
            "energy_collected": torch.zeros(count, dtype=torch.float32),
            "episode0_pred_err_count": torch.zeros(count, dtype=torch.int32),
            "episode0_pred_err_sum": torch.zeros(count, dtype=torch.float32),
            "drive_jitter": torch.zeros(count, dtype=torch.float32),
            "pos_y": torch.tensor([0, 0, 1], dtype=torch.int32),
            "pos_x": torch.tensor([0, 1, 0], dtype=torch.int32),
            "agent_grid": torch.tensor(
                [[agent_ids[0], agent_ids[1]], [agent_ids[2], -1]],
                dtype=torch.int32,
            ),
            "food_grid": torch.tensor([[1, 0], [1, 0]], dtype=torch.int32),
            "patch_food_amount": torch.tensor(
                [[1.0, 0.25], [0.5, 0.0]], dtype=torch.float32
            ),
            "patch_fertility": torch.ones((2, 2), dtype=torch.float32),
            "patch_cover": torch.tensor(
                [[0.0, 0.2], [0.4, 0.6]], dtype=torch.float32
            ),
            "patch_shelter": torch.zeros((2, 2), dtype=torch.float32),
            "patch_movement_cost": torch.ones((2, 2), dtype=torch.float32),
            "patch_scent": torch.tensor(
                [[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32
            ),
            "patch_waste": torch.tensor(
                [[0.4, 0.3], [0.2, 0.1]], dtype=torch.float32
            ),
        }
    )
    env_cfg, sim_cfg = _configs()
    env_cfg.patch_world_enable = True
    env_cfg.patch_world_schema_version = 1
    env_cfg.patch_world_observation_enable = profile in {"P1", "P2"}
    env_cfg.patch_world_observation_schema_version = 1
    env_cfg.patch_world_diagnostics_enable = True
    env_cfg.patch_world_lineage_descriptors_enable = True
    env_cfg.patch_world_descriptor_schema_version = 1
    env_cfg.patch_world_waste_consequence_enable = profile == "P2"
    env_cfg.patch_movement_cost_apply_enable = profile == "P2"
    if profile == "P2":
        tensors["patch_recovery_activity"] = torch.zeros((2, 2), dtype=torch.float32)
    env_cfg.patch_density_radius = 1
    env_cfg.patch_density_include_self = True
    env_cfg.patch_density_normalize = False
    sim_cfg.native_runtime_allow_runtime_jit_build = False
    sim_cfg.shock_start_prob = 0.0
    sim_cfg.shock_period = 0
    sim_cfg.shock_duration = 0
    sim_cfg.pressure_autopilot_allow_shock_enable = False
    projection["metadata"].update(
        {
            "world": "patch",
            "native_runtime_patch_profile": profile,
            "native_runtime_patch_profile_scope": "full_profile",
            "native_runtime_patch_contract_schema_version": (
                "eiel_patch_native_import_contract_v1"
            ),
            "capability_mask": {
                "flat": False,
                "patch": True,
                "churn": True,
            },
            "obs_size": 72 if profile in {"P1", "P2"} else 9,
        }
    )

    rows = _behavior_rows(agent_ids)
    slots_by_agent_id = {
        int(agent_id): slot
        for slot, agent_id in enumerate(tensors["slot_to_agent_id"].tolist())
    }
    for row in rows:
        slot = slots_by_agent_id[int(row["agent_id"])]
        row.update(
            {
                "lifespan_steps": int(tensors["survival_steps"][slot].item()),
                "offspring": int(tensors["offspring_count"][slot].item()),
                "energy_collected": float(tensors["energy_collected"][slot].item()),
                "injury_end": float(tensors["injury"][slot].item()),
                "aggression_bias": float(
                    tensors["genome_aggression_bias"][slot].item()
                ),
            }
        )
    return projection, rows, env_cfg, sim_cfg


def _patch_raw_result(behavior_rows: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {
        "behavior_rows": behavior_rows,
        "promotion_accepted": False,
        "committed_birth_rows": 0,
        "death_rows": 0,
        "birth_attempt_rows": 0,
        "action_gather": 0,
        "first_repro_frac_alive": 0.0,
        "patch_birth_context_rows": [],
        "patch_death_context_rows": [],
        "patch_movement_energy_total": 0.0,
        "patch_movement_energy_event_count": 0,
    }
    for field in (
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
    ):
        result[field] = 0.0
    return result


@pytest.mark.parametrize("profile", ("P0", "P1", "P2"))
def test_patch_profiles_reuse_the_shared_population_projection_for_public_metrics_and_lineage(
    profile: str,
) -> None:
    projection, behavior_rows, env_cfg, sim_cfg = _patch_reporting_inputs(profile)

    overlay = public_metrics_overlay(
        _patch_raw_result(behavior_rows),
        checkpoint_projection=projection,
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
        metrics_generation=4,
    )

    assert set(NATIVE_FLAT_RESIDENT_REPORTING_COLUMNS).issubset(overlay)
    assert overlay["behavior_action_total"] == 0.0
    lineage = overlay[NATIVE_LINEAGE_METRICS_INPUT_KEY]
    assert lineage["world"] == "patch"
    assert lineage["patch_profile"] == profile
    assert lineage["patch_profile_scope"] == "full_profile"
    assert lineage["patch_contract_schema_version"] == "eiel_patch_native_import_contract_v1"
    assert lineage["promotion_accepted"] is False
    assert lineage["record_count"] == len(behavior_rows)
    assert lineage["agent_ids"] == sorted(row["agent_id"] for row in behavior_rows)
    assert overlay["patch_world_waste_consequence_enable"] == int(profile == "P2")
    for field in _LINEAGE_INTEGER_FIELDS:
        assert type(overlay[field]) is int, field


def test_patch_reporting_rebuilds_live_occupancy_from_slot_positions() -> None:
    projection, behavior_rows, env_cfg, sim_cfg = _patch_reporting_inputs("P2")
    projection["tensors"]["agent_grid"] = torch.tensor(
        [[30, 30], [20, 10]],
        dtype=torch.int32,
    )

    overlay = public_metrics_overlay(
        _patch_raw_result(behavior_rows),
        checkpoint_projection=projection,
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
        metrics_generation=4,
    )

    lineage = overlay[NATIVE_LINEAGE_METRICS_INPUT_KEY]
    assert lineage["record_count"] == len(behavior_rows)
    assert overlay["patch_density_occupied_mean"] == float(len(behavior_rows))


def test_patch_public_reporting_is_slot_bounded_for_large_historical_ids() -> None:
    agent_ids = [1_000_000, 1_000_010, 1_000_020]
    projection, behavior_rows, env_cfg, sim_cfg = _patch_reporting_inputs(
        "P1", agent_ids=agent_ids
    )

    overlay = public_metrics_overlay(
        _patch_raw_result(behavior_rows),
        checkpoint_projection=projection,
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
        metrics_generation=4,
    )

    assert int(projection["tensors"]["slot_to_agent_id"].numel()) == len(agent_ids)
    assert overlay[NATIVE_LINEAGE_METRICS_INPUT_KEY]["agent_ids"] == sorted(agent_ids)
    assert overlay["patch_observation_nan_count"] == 0.0


def test_patch_unknown_profile_fails_closed_before_shared_projection() -> None:
    projection, behavior_rows, env_cfg, sim_cfg = _patch_reporting_inputs("P1")
    projection["metadata"]["native_runtime_patch_profile"] = "P9"

    with pytest.raises(ValueError, match="finite P0/P1/P2"):
        native_flat_population_reporting_records(
            tensors=projection["tensors"],
            behavior_rows=behavior_rows,
            env_cfg=env_cfg,
            sim_cfg=sim_cfg,
            metadata=projection["metadata"],
        )


def test_patch_lineage_uses_exact_energy_accumulator_sidecar() -> None:
    projection, behavior_rows, env_cfg, sim_cfg = _patch_reporting_inputs("P0")
    tensors = projection["tensors"]
    tensors["energy_collected_exact"] = tensors["energy_collected"].to(torch.float64)
    row = behavior_rows[0]
    slot = tensors["slot_to_agent_id"].tolist().index(int(row["agent_id"]))
    exact_value = 92.39999866485596
    tensors["energy_collected"][slot] = torch.tensor(92.4000015258789)
    tensors["energy_collected_exact"][slot] = torch.tensor(exact_value, dtype=torch.float64)
    row["energy_collected"] = exact_value

    overlay = public_metrics_overlay(
        _patch_raw_result(behavior_rows),
        checkpoint_projection=projection,
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
        metrics_generation=4,
    )

    assert overlay[NATIVE_LINEAGE_METRICS_INPUT_KEY]["record_count"] == len(behavior_rows)


def test_patch_public_reporting_rejects_a_promoted_generation_result() -> None:
    projection, behavior_rows, env_cfg, sim_cfg = _patch_reporting_inputs("P1")

    with pytest.raises(ValueError, match="promotion_accepted=false"):
        raw = _patch_raw_result(copy.deepcopy(behavior_rows))
        raw["promotion_accepted"] = True
        public_metrics_overlay(
            raw,
            checkpoint_projection=projection,
            env_cfg=env_cfg,
            sim_cfg=sim_cfg,
            metrics_generation=4,
        )


def test_patch_lineage_consumer_rejects_missing_finite_profile_provenance() -> None:
    projection, behavior_rows, env_cfg, sim_cfg = _patch_reporting_inputs("P1")
    overlay = public_metrics_overlay(
        _patch_raw_result(behavior_rows),
        checkpoint_projection=projection,
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
        metrics_generation=4,
    )
    payload = copy.deepcopy(overlay[NATIVE_LINEAGE_METRICS_INPUT_KEY])
    payload["patch_profile"] = "P9"

    with pytest.raises(ValueError, match="finite P0/P1/P2"):
        native_lineage_agents_from_input(payload, expected_generation=4)


def test_patch_public_reporting_reuses_exact_snapshot_and_event_reducers() -> None:
    projection, behavior_rows, env_cfg, sim_cfg = _patch_reporting_inputs("P2")
    behavior_rows[0]["gather"] = 5
    raw = _patch_raw_result(behavior_rows)
    context = {
        "generation": 4,
        "tick": 3,
        "event_sequence": 7,
        "agent_id": 77,
        "slot": 2,
        "y": 1,
        "x": 0,
        "lineage_id": 700,
        "food_amount": 0.5,
        "fertility": 1.0,
        "cover": 0.4,
        "shelter": 0.0,
        "visibility": 0.6,
        "movement_cost": 1.0,
        "scent": 0.3,
        "waste": 0.2,
        "density": 3.0,
        "recovery_activity": 0.0,
    }
    raw.update(
        {
            "committed_birth_rows": 1,
            "death_rows": 1,
            "birth_attempt_rows": 2,
            "action_gather": 5,
            "patch_birth_context_rows": [context],
            "patch_death_context_rows": [{**context, "event_sequence": 8}],
            "patch_food_harvest_total": 2.5,
            "patch_movement_energy_total": 1.5,
            "patch_movement_energy_event_count": 3,
        }
    )

    overlay = public_metrics_overlay(
        raw,
        checkpoint_projection=projection,
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
        metrics_generation=4,
    )

    assert overlay["patch_food_amount_mean"] == pytest.approx(
        float(np.mean([[1.0, 0.25], [0.5, 0.0]]))
    )
    assert overlay["patch_density_mean"] == 3.0
    assert overlay["patch_density_max"] == 3.0
    assert overlay["patch_density_p90"] == 3.0
    assert overlay["patch_density_occupied_mean"] == 3.0
    assert overlay["patch_density_hotspot_frac"] == 1.0
    assert overlay["birth_patch_count"] == 1.0
    assert overlay["death_patch_count"] == 1.0
    assert overlay["birth_patch_waste_mean"] == pytest.approx(0.2)
    assert overlay["patch_food_harvest_total"] == 2.5
    assert overlay["patch_movement_energy_extra_mean"] == 0.5
    assert overlay["behavior_patch_interaction_proxy"] == 8.5
    assert overlay["lineage_habitat_descriptor_count"] == 1.0
    assert overlay["lineage_strategy_descriptor_count"] == 1.0
    assert overlay["patch_observation_nan_count"] == 0.0


def test_jungle_public_reporting_projects_exact_native_counts_and_contexts() -> None:
    projection, behavior_rows, env_cfg, sim_cfg = _patch_reporting_inputs("P2")
    env_cfg.jungle_predator_pressure_enable = True
    projection["metadata"]["native_runtime_jungle_execution_enable"] = True
    projection["metadata"]["capability_mask"]["jungle"] = True
    raw = _patch_raw_result(behavior_rows)
    context = {
        "generation": 4,
        "tick": 1,
        "event_sequence": 0,
        "agent_id": 30,
        "slot": 0,
        "y": 0,
        "x": 0,
        "lineage_id": 30,
        "food_amount": 1.0,
        "fertility": 1.0,
        "cover": 0.0,
        "shelter": 0.0,
        "visibility": 1.0,
        "movement_cost": 1.0,
        "scent": 0.1,
        "waste": 0.4,
        "density": 3.0,
        "recovery_activity": 0.0,
        "predator_risk": 0.25,
        "juvenile": 0.0,
        "terrain_damage_multiplier": 0.75,
        "predator_damage_raw": 0.2,
        "predator_damage": 0.15,
    }
    raw.update(
        {
            "jungle_predator_event_context_rows": [context],
            "jungle_birth_emission_context_rows": [],
            "jungle_predator_exposure_count": 1,
            "jungle_predator_events": 1,
            "jungle_predator_deaths": 0,
            "jungle_juvenile_predator_events": 0,
            "jungle_juvenile_predator_deaths": 0,
            "jungle_adult_predator_events": 1,
            "jungle_birth_emission_count": 0,
            "jungle_predator_damage_total": 0.15,
            "jungle_predator_damage_raw_total": 0.2,
            "jungle_predator_damage_effective_total": 0.15,
            "jungle_predator_damage_prevented_total": 0.05,
            "jungle_juvenile_predator_damage_total": 0.0,
            "jungle_adult_predator_damage_total": 0.15,
            "jungle_birth_scent_emitted_total": 0.0,
            "jungle_birth_waste_emitted_total": 0.0,
            "patch_recovery_positive_total": 0.0,
            "patch_recovery_negative_total": 0.0,
            "patch_recovery_decay_total": 0.0,
            "jungle_recovery_regrowth_boost_total": 0.0,
            "jungle_recovery_regrowth_boost_mean": 0.0,
            "jungle_recovery_regrowth_boost_max": 0.0,
            "jungle_recovery_regrowth_boost_p90": 0.0,
            "jungle_predator_damage_multiplier_mean": 0.75,
            "jungle_predator_damage_multiplier_min": 0.75,
            "jungle_predator_damage_multiplier_p10": 0.75,
            "jungle_predator_damage_multiplier_p90": 0.75,
        }
    )

    overlay = public_metrics_overlay(
        raw,
        checkpoint_projection=projection,
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
        metrics_generation=4,
    )

    assert overlay["jungle_predator_events"] == 1.0
    assert overlay["jungle_predator_event_rate"] == 1.0
    assert overlay["jungle_predator_damage_total"] == pytest.approx(0.15)
    assert overlay["jungle_predator_damage_prevented_total"] == pytest.approx(0.05)
    assert overlay["predator_event_patch_count"] == 1.0
    assert overlay["lineage_strategy_descriptor_count"] == 1.0


def test_patch_public_reporting_respects_disabled_descriptor_gates() -> None:
    projection, behavior_rows, env_cfg, sim_cfg = _patch_reporting_inputs("P0")
    env_cfg.patch_world_diagnostics_enable = False
    env_cfg.patch_world_lineage_descriptors_enable = False
    env_cfg.lineage_diversity_reporting_enable = False
    env_cfg.lineage_strategy_descriptors_enable = False
    env_cfg.lineage_branch_candidate_reporting_enable = False

    overlay = public_metrics_overlay(
        _patch_raw_result(behavior_rows),
        checkpoint_projection=projection,
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
        metrics_generation=4,
    )

    assert overlay["patch_world_diagnostics_enable"] == 0
    assert overlay["patch_world_lineage_descriptors_enable"] == 0
    assert overlay["lineage_diversity_reporting_enable"] == 0
    assert overlay["lineage_strategy_descriptors_enable"] == 0
    assert overlay["lineage_branch_candidate_reporting_enable"] == 0
    for gated_name in (
        "occupied_patch_food_amount_mean",
        "birth_patch_count",
        "death_patch_count",
        "lineage_habitat_descriptor_count",
        "lineage_strategy_descriptor_count",
        "lineage_habitat_descriptors_json",
        "lineage_strategy_descriptors_json",
        "lineage_branch_candidates_json",
        "patch_observation_nan_count",
    ):
        assert gated_name not in overlay


def test_patch_public_reporting_rejects_recovery_state_outside_p2() -> None:
    projection, behavior_rows, env_cfg, sim_cfg = _patch_reporting_inputs("P0")
    projection["tensors"]["patch_recovery_activity"] = torch.zeros(
        (2, 2), dtype=torch.float32
    )

    with pytest.raises(ValueError, match="inactive P2 recovery state"):
        public_metrics_overlay(
            _patch_raw_result(behavior_rows),
            checkpoint_projection=projection,
            env_cfg=env_cfg,
            sim_cfg=sim_cfg,
            metrics_generation=4,
        )


def test_patch_public_reporting_requires_recovery_state_for_p2() -> None:
    projection, behavior_rows, env_cfg, sim_cfg = _patch_reporting_inputs("P2")
    projection["tensors"].pop("patch_recovery_activity")

    with pytest.raises(ValueError, match="patch_recovery_activity"):
        public_metrics_overlay(
            _patch_raw_result(behavior_rows),
            checkpoint_projection=projection,
            env_cfg=env_cfg,
            sim_cfg=sim_cfg,
            metrics_generation=4,
        )


@pytest.mark.parametrize("field", _PATCH_NONNEGATIVE_TOTAL_FIELDS)
def test_patch_public_reporting_rejects_negative_canonical_totals(field: str) -> None:
    projection, behavior_rows, env_cfg, sim_cfg = _patch_reporting_inputs("P2")
    raw = _patch_raw_result(behavior_rows)
    raw[field] = -0.25

    with pytest.raises(ValueError, match=rf"raw_result\.{field} must be non-negative"):
        public_metrics_overlay(
            raw,
            checkpoint_projection=projection,
            env_cfg=env_cfg,
            sim_cfg=sim_cfg,
            metrics_generation=4,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("patch_high_waste_regrowth_suppression_frac", -0.01, r"must be in \[0, 1\]"),
        ("patch_high_waste_regrowth_suppression_frac", 1.01, r"must be in \[0, 1\]"),
        ("patch_high_waste_regrowth_suppression_frac", float("nan"), "must be finite"),
        ("patch_high_density_regrowth_suppression_frac", -0.01, r"must be in \[0, 1\]"),
        ("patch_high_density_regrowth_suppression_frac", 1.01, r"must be in \[0, 1\]"),
        ("patch_high_density_regrowth_suppression_frac", float("inf"), "must be finite"),
    ),
)
def test_patch_public_reporting_rejects_invalid_raw_suppression_fractions(
    field: str,
    value: float,
    message: str,
) -> None:
    projection, behavior_rows, env_cfg, sim_cfg = _patch_reporting_inputs("P2")
    raw = _patch_raw_result(behavior_rows)
    raw[field] = value

    with pytest.raises(ValueError, match=message):
        public_metrics_overlay(
            raw,
            checkpoint_projection=projection,
            env_cfg=env_cfg,
            sim_cfg=sim_cfg,
            metrics_generation=4,
        )


@pytest.mark.parametrize("profile", ("P0", "P1"))
@pytest.mark.parametrize("field", _PATCH_WASTE_DENSITY_CONSEQUENCE_FIELDS)
def test_patch_public_reporting_rejects_disabled_consequence_values(
    profile: str,
    field: str,
) -> None:
    projection, behavior_rows, env_cfg, sim_cfg = _patch_reporting_inputs(profile)
    raw = _patch_raw_result(behavior_rows)
    raw[field] = 0.25

    with pytest.raises(
        ValueError,
        match=rf"raw_result\.{field} must be zero for Patch profile {profile}",
    ):
        public_metrics_overlay(
            raw,
            checkpoint_projection=projection,
            env_cfg=env_cfg,
            sim_cfg=sim_cfg,
            metrics_generation=4,
        )


@pytest.mark.parametrize("profile", ("P0", "P1"))
@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("patch_movement_energy_total", 0.25),
        ("patch_movement_energy_event_count", 1),
    ),
)
def test_patch_public_reporting_rejects_disabled_movement_values(
    profile: str,
    field: str,
    value: float,
) -> None:
    projection, behavior_rows, env_cfg, sim_cfg = _patch_reporting_inputs(profile)
    raw = _patch_raw_result(behavior_rows)
    raw[field] = value

    with pytest.raises(
        ValueError,
        match=rf"raw_result\.{field} must be zero for Patch profile {profile}",
    ):
        public_metrics_overlay(
            raw,
            checkpoint_projection=projection,
            env_cfg=env_cfg,
            sim_cfg=sim_cfg,
            metrics_generation=4,
        )


def test_patch_public_reporting_accepts_signed_p2_movement_total() -> None:
    projection, behavior_rows, env_cfg, sim_cfg = _patch_reporting_inputs("P2")
    raw = _patch_raw_result(behavior_rows)
    raw["patch_movement_energy_total"] = -1.5
    raw["patch_movement_energy_event_count"] = 3

    overlay = public_metrics_overlay(
        raw,
        checkpoint_projection=projection,
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
        metrics_generation=4,
    )

    assert overlay["patch_movement_energy_extra_total"] == -1.5
    assert overlay["patch_movement_energy_extra_mean"] == -0.5


def test_patch_public_reporting_rejects_negative_movement_event_count() -> None:
    projection, behavior_rows, env_cfg, sim_cfg = _patch_reporting_inputs("P2")
    raw = _patch_raw_result(behavior_rows)
    raw["patch_movement_energy_event_count"] = -1

    with pytest.raises(
        ValueError,
        match="raw_result.patch_movement_energy_event_count must be a non-negative integer",
    ):
        public_metrics_overlay(
            raw,
            checkpoint_projection=projection,
            env_cfg=env_cfg,
            sim_cfg=sim_cfg,
            metrics_generation=4,
        )
