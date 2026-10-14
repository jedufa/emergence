from __future__ import annotations

import csv
import json
import math
from dataclasses import fields
from pathlib import Path
from typing import Any
from types import SimpleNamespace

import numpy as np
import pytest

from env.env_config import EnvConfig
from env.gridworld import GridWorld
from env.gridworld_pkg.constants import ACT_GATHER, ACT_MOVE_E, EMPTY, FOOD
from env.gridworld_pkg.patch_world import (
    PATCH_WORLD_ARRAY_NAMES,
    PATCH_WORLD_DESCRIPTOR_SCHEMA_VERSION,
    PATCH_WORLD_OBSERVATION_CHANNEL_NAMES,
    PATCH_WORLD_OBSERVATION_SCHEMA_VERSION,
    LINEAGE_DIVERSITY_SCHEMA_VERSION,
    compute_local_density_map,
    lineage_branch_candidates,
    lineage_habitat_descriptors,
    lineage_habitat_metrics,
    lineage_strategy_descriptors,
    lineage_strategy_metrics,
    patch_event_contexts_summary,
    patch_recovery_metrics,
    patch_recovery_regrowth_boost_maps,
    patch_observation_validation_metrics,
    patch_occupied_descriptor_metrics,
    regrow_patch_food,
    sample_patch_context,
    update_patch_recovery_activity,
    patch_world_local_observation_size,
    patch_world_observation_enabled,
    patch_world_total_observation_size,
    patch_visibility_from_cover,
)
from evolution.population_pkg.population_config import PopConfig
from knobs_override import apply_knob_overrides
from simulation.metrics import MetricsLogger
from simulation.checkpoint_bundle_v2 import (
    checkpoint_bundle_v2_apply,
    checkpoint_bundle_v2_load,
    checkpoint_bundle_v2_save,
)
from simulation.metrics_pkg.noop_loggers import NoOpMetricsLogger
from simulation.rng_manager import RNGManager
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig


def _env_cfg(*, patch_world_enable: bool = False, schema_version: int = 1) -> EnvConfig:
    cfg = EnvConfig(width=8, height=8, view_radius=1)
    cfg.enable_biomes = False
    cfg.food_spawn_per_step = 2
    cfg.max_food = 32
    cfg.patch_world_enable = bool(patch_world_enable)
    cfg.patch_world_schema_version = int(schema_version)
    return cfg


def _sim_cfg(*, seed: int = 123) -> SimConfig:
    cfg = SimConfig(
        seed=int(seed),
        generations=2,
        steps_per_generation=3,
        device="cpu",
        bank_device="cpu",
        behavior_log_enable=False,
        policy_deterministic_actions=True,
        policy_tick_stride=1,
    )
    cfg.phase_e_e1_enable = False
    cfg.epi_enable = False
    cfg.checkpoint_bundle_v2_period_gen = 0
    return cfg


def _make_sim(tmp_path: Path, *, patch_world_enable: bool = False, seed: int = 123) -> SimulationRunner:
    return SimulationRunner(
        env_cfg=_env_cfg(patch_world_enable=patch_world_enable),
        pop_cfg=PopConfig(population_size=6),
        sim_cfg=_sim_cfg(seed=seed),
        logger=NoOpMetricsLogger(),
        run_dir=str(tmp_path),
        perf=False,
    )


def _old_style_env_config_from_snapshot(snapshot: dict[str, Any]) -> EnvConfig:
    allowed = {f.name for f in fields(EnvConfig)}
    return EnvConfig(**{k: v for k, v in snapshot.items() if k in allowed})


def test_patch_world_config_defaults_disabled() -> None:
    cfg = EnvConfig()
    assert cfg.patch_world_enable is False
    assert cfg.patch_world_schema_version == 1
    assert cfg.patch_food_amount_max == 1.0
    assert cfg.patch_food_amount_initial_food_cell_value == 1.0
    assert cfg.patch_food_amount_initial_empty_cell_value == 0.0
    assert cfg.patch_food_presence_threshold == 0.5
    assert cfg.patch_food_gather_amount == 1.0
    assert cfg.patch_food_regrow_per_step == 0.02
    assert cfg.patch_food_regrow_use_fertility is True
    assert cfg.patch_food_regrow_cap_by_fertility is False
    assert cfg.patch_fertility_default == 1.0
    assert cfg.patch_cover_default == 0.0
    assert cfg.patch_shelter_default == 0.0
    assert cfg.patch_shelter_from_cover_enable is False
    assert cfg.patch_visibility_default == 1.0
    assert cfg.patch_visibility_from_cover_enable is True
    assert cfg.patch_movement_cost_default == 1.0
    assert cfg.patch_movement_cost_apply_enable is False
    assert cfg.patch_movement_cost_energy_scale == 1.0
    assert cfg.patch_scent_initial == 0.0
    assert cfg.patch_scent_decay_per_step == 0.05
    assert cfg.patch_scent_presence_emit == 0.01
    assert cfg.patch_scent_move_emit == 0.02
    assert cfg.patch_scent_gather_emit == 0.05
    assert cfg.patch_scent_birth_emit == 0.0
    assert cfg.patch_scent_death_emit == 0.0
    assert cfg.patch_waste_initial == 0.0
    assert cfg.patch_waste_max == 1.0
    assert cfg.patch_waste_decay_per_step == 0.02
    assert cfg.patch_waste_presence_emit == 0.005
    assert cfg.patch_waste_consume_emit == 0.05
    assert cfg.patch_waste_birth_emit == 0.0
    assert cfg.patch_waste_death_emit == 0.0
    assert cfg.patch_waste_density_emit_scale == 0.0
    assert cfg.patch_density_radius == 1
    assert cfg.patch_density_include_self is True
    assert cfg.patch_density_normalize is False
    assert cfg.patch_world_observation_enable is False
    assert cfg.patch_world_observation_schema_version == 1
    assert cfg.patch_world_diagnostics_enable is True
    assert cfg.patch_world_lineage_descriptors_enable is True
    assert cfg.patch_world_descriptor_schema_version == 1
    assert cfg.patch_world_waste_consequence_enable is False
    assert cfg.patch_waste_regrowth_penalty_enable is True
    assert cfg.patch_waste_regrowth_penalty_scale == 0.5
    assert cfg.patch_waste_regrowth_penalty_max == 0.5
    assert cfg.patch_density_regrowth_penalty_enable is True
    assert cfg.patch_density_regrowth_penalty_scale == 0.5
    assert cfg.patch_density_regrowth_penalty_max == 0.5
    assert cfg.patch_regrowth_min_multiplier == 0.1
    assert cfg.patch_waste_fertility_effect_enable is False


def test_old_style_env_config_snapshot_without_patch_world_fields_loads() -> None:
    snapshot = {
        "width": 12,
        "height": 10,
        "view_radius": 1,
        "food_spawn_per_step": 3,
        "max_food": 40,
    }

    cfg = _old_style_env_config_from_snapshot(snapshot)

    assert cfg.width == 12
    assert cfg.height == 10
    assert cfg.patch_world_enable is False
    assert cfg.patch_world_schema_version == 1
    assert cfg.patch_food_amount_max == 1.0
    assert cfg.patch_movement_cost_apply_enable is False
    assert cfg.patch_scent_initial == 0.0
    assert cfg.patch_waste_initial == 0.0
    assert cfg.patch_density_radius == 1
    assert cfg.patch_world_observation_enable is False
    assert cfg.patch_world_observation_schema_version == 1
    assert cfg.patch_world_diagnostics_enable is True
    assert cfg.patch_world_lineage_descriptors_enable is True
    assert cfg.patch_world_descriptor_schema_version == 1
    assert cfg.patch_world_waste_consequence_enable is False
    assert cfg.patch_waste_fertility_effect_enable is False


def test_patch_world_knob_overrides_enable_schema_shell() -> None:
    env_cfg = EnvConfig()
    pop_cfg = PopConfig(population_size=4)
    sim_cfg = SimConfig()

    applied, genome_overrides = apply_knob_overrides(
        knob_pairs=[
            "env_cfg.patch_world_enable=true",
            "env_cfg.patch_world_schema_version=1",
        ],
        env_cfg=env_cfg,
        pop_cfg=pop_cfg,
        sim_cfg=sim_cfg,
    )

    assert env_cfg.patch_world_enable is True
    assert env_cfg.patch_world_schema_version == 1
    assert "env_cfg.patch_world_enable=true" in applied
    assert "env_cfg.patch_world_schema_version=1" in applied
    assert genome_overrides == {}


def test_patch_world_schema_version_validates_only_when_enabled() -> None:
    rm_disabled = RNGManager(global_seed=11)
    env_disabled = GridWorld(
        _env_cfg(patch_world_enable=False, schema_version=999),
        seed=11,
        run_dir=None,
        rng=rm_disabled.env_np,
    )
    assert env_disabled.patch_world_schema_version == 999

    rm_enabled = RNGManager(global_seed=11)
    with pytest.raises(ValueError, match="Unsupported Patch World schema version"):
        GridWorld(
            _env_cfg(patch_world_enable=True, schema_version=999),
            seed=11,
            run_dir=None,
            rng=rm_enabled.env_np,
        )


def test_patch_world_step_regrows_food_without_mutating_agents_or_rng() -> None:
    rm = RNGManager(global_seed=17)
    env = GridWorld(
        _env_cfg(patch_world_enable=True),
        seed=17,
        run_dir=None,
        rng=rm.env_np,
    )
    before = {
        "grid": env.grid.copy(),
        "agent_grid": env.agent_grid.copy(),
        "biome_grid": env.biome_grid.copy(),
        "agent_alive": env.agent_alive.copy(),
        "agent_pos_y": env.agent_pos_y.copy(),
        "agent_pos_x": env.agent_pos_x.copy(),
        "patch_food_amount": env.patch_food_amount.copy(),
        "patch_fertility": env.patch_fertility.copy(),
        "patch_cover": env.patch_cover.copy(),
        "patch_scent": env.patch_scent.copy(),
        "rng": rm.capture_snapshot(),
    }

    env.patch_world_step(abs_step=5)

    assert np.array_equal(before["grid"], env.grid)
    assert np.array_equal(before["agent_grid"], env.agent_grid)
    assert np.array_equal(before["biome_grid"], env.biome_grid)
    assert np.array_equal(before["agent_alive"], env.agent_alive)
    assert np.array_equal(before["agent_pos_y"], env.agent_pos_y)
    assert np.array_equal(before["agent_pos_x"], env.agent_pos_x)
    assert np.all(env.patch_food_amount >= before["patch_food_amount"])
    assert np.array_equal(before["patch_fertility"], env.patch_fertility)
    assert np.array_equal(before["patch_cover"], env.patch_cover)
    assert np.array_equal(before["patch_scent"], env.patch_scent)
    assert before["rng"] == rm.capture_snapshot()
    assert env.patch_food_regrowth_total_last > 0.0


def test_patch_world_disabled_does_not_allocate_patch_arrays() -> None:
    rm = RNGManager(global_seed=19)
    env = GridWorld(
        _env_cfg(patch_world_enable=False),
        seed=19,
        run_dir=None,
        rng=rm.env_np,
    )

    assert env.patch_world_enable is False
    for name in PATCH_WORLD_ARRAY_NAMES:
        assert not hasattr(env, name)


def test_patch_world_enabled_initializes_passive_arrays() -> None:
    cfg = _env_cfg(patch_world_enable=True)
    rm = RNGManager(global_seed=23)
    env = GridWorld(cfg, seed=23, run_dir=None, rng=rm.env_np)

    for name in PATCH_WORLD_ARRAY_NAMES:
        arr = getattr(env, name)
        assert arr.shape == (cfg.height, cfg.width)
        assert arr.dtype == np.float32

    assert np.all(env.patch_food_amount == cfg.patch_food_amount_initial_empty_cell_value)
    assert np.all(env.patch_fertility >= cfg.patch_fertility_min)
    assert np.all(env.patch_fertility <= cfg.patch_fertility_max)
    assert np.all(env.patch_cover >= cfg.patch_cover_min)
    assert np.all(env.patch_cover <= cfg.patch_cover_max)
    assert np.all(env.patch_shelter >= cfg.patch_shelter_min)
    assert np.all(env.patch_shelter <= cfg.patch_shelter_max)
    assert np.all(env.patch_shelter == cfg.patch_shelter_default)
    assert np.all(env.patch_movement_cost >= cfg.patch_movement_cost_min)
    assert np.all(env.patch_movement_cost <= cfg.patch_movement_cost_max)
    assert np.all(env.patch_movement_cost == cfg.patch_movement_cost_default)
    assert np.all(env.patch_scent == cfg.patch_scent_initial)
    assert np.all(env.patch_waste == cfg.patch_waste_initial)
    assert env.local_density_map.shape == (cfg.height, cfg.width)
    assert env.local_density_map.dtype == np.float32
    assert np.all(env.local_density_map == 0.0)
    assert np.all(env.patch_scent_emit_pending == 0.0)
    assert np.all(env.patch_waste_emit_pending == 0.0)


def test_patch_world_visibility_is_derived_from_cover() -> None:
    cfg = _env_cfg(patch_world_enable=True)
    cfg.patch_visibility_min = 0.0
    cfg.patch_visibility_max = 1.0
    rm = RNGManager(global_seed=24)
    env = GridWorld(cfg, seed=24, run_dir=None, rng=rm.env_np)
    env.patch_cover[0, 0] = np.float32(0.25)
    env.patch_cover[0, 1] = np.float32(1.5)

    visibility = patch_visibility_from_cover(cfg, env.patch_cover)

    assert visibility.shape == env.patch_cover.shape
    assert visibility.dtype == np.float32
    assert visibility[0, 0] == pytest.approx(0.75)
    assert visibility[0, 1] == pytest.approx(0.0)


def test_patch_world_shelter_can_initialize_from_cover() -> None:
    cfg = _env_cfg(patch_world_enable=True)
    cfg.patch_cover_default = 0.4
    cfg.patch_shelter_from_cover_enable = True
    rm = RNGManager(global_seed=25)
    env = GridWorld(cfg, seed=25, run_dir=None, rng=rm.env_np)

    assert np.all(env.patch_cover == np.float32(0.4))
    assert np.all(env.patch_shelter == np.float32(0.4))


def test_patch_food_amount_mirrors_current_binary_food_when_reinitialized() -> None:
    cfg = _env_cfg(patch_world_enable=True)
    rm = RNGManager(global_seed=29)
    env = GridWorld(cfg, seed=29, run_dir=None, rng=rm.env_np)

    env.grid[1, 2] = FOOD
    env._reset_patch_world_arrays()

    assert env.patch_food_amount[1, 2] == np.float32(cfg.patch_food_amount_initial_food_cell_value)
    assert env.patch_food_amount[0, 0] == np.float32(cfg.patch_food_amount_initial_empty_cell_value)


def test_patch_world_disabled_gather_keeps_legacy_binary_food_behavior() -> None:
    cfg = _env_cfg(patch_world_enable=False)
    rm = RNGManager(global_seed=30)
    env = GridWorld(cfg, seed=30, run_dir=None, rng=rm.env_np)
    env.grid[2, 2] = FOOD
    env.add_agent(0, (2, 2))

    out = env.step_agents_batch(
        agent_ids=np.asarray([0], dtype=np.int32),
        actions=np.asarray([ACT_GATHER], dtype=np.int32),
        energies=np.asarray([10.0], dtype=np.float32),
        healths=np.asarray([10.0], dtype=np.float32),
        injuries=np.asarray([0.0], dtype=np.float32),
        scars=np.asarray([0.0], dtype=np.float32),
    )

    _, new_energy, _, food_gained, _, _ = out
    assert not hasattr(env, "patch_food_amount")
    assert env.grid[2, 2] == EMPTY
    assert food_gained[0] == pytest.approx(cfg.food_energy)
    assert new_energy[0] == pytest.approx(10.0 - cfg.gather_cost + cfg.food_energy)


def test_patch_world_enabled_gather_partially_depletes_and_keeps_binary_food() -> None:
    cfg = _env_cfg(patch_world_enable=True)
    cfg.patch_food_regrow_per_step = 0.0
    cfg.patch_food_gather_amount = 0.25
    rm = RNGManager(global_seed=32)
    env = GridWorld(cfg, seed=32, run_dir=None, rng=rm.env_np)
    env.grid[2, 2] = FOOD
    env._reset_patch_world_arrays()
    env.add_agent(0, (2, 2))

    out = env.step_agents_batch(
        agent_ids=np.asarray([0], dtype=np.int32),
        actions=np.asarray([ACT_GATHER], dtype=np.int32),
        energies=np.asarray([10.0], dtype=np.float32),
        healths=np.asarray([10.0], dtype=np.float32),
        injuries=np.asarray([0.0], dtype=np.float32),
        scars=np.asarray([0.0], dtype=np.float32),
    )

    _, new_energy, _, food_gained, _, _ = out
    assert env.patch_food_amount[2, 2] == pytest.approx(0.75)
    assert env.grid[2, 2] == FOOD
    assert env.patch_food_harvest_total_last == pytest.approx(0.25)
    assert food_gained[0] == pytest.approx(cfg.food_energy)
    assert new_energy[0] == pytest.approx(10.0 - cfg.gather_cost + cfg.food_energy)


def test_patch_world_enabled_gather_depletes_and_scales_partial_energy() -> None:
    cfg = _env_cfg(patch_world_enable=True)
    cfg.patch_food_regrow_per_step = 0.0
    cfg.patch_food_presence_threshold = 0.1
    cfg.patch_food_gather_amount = 1.0
    rm = RNGManager(global_seed=34)
    env = GridWorld(cfg, seed=34, run_dir=None, rng=rm.env_np)
    env.patch_food_amount[2, 2] = np.float32(0.25)
    env.grid[2, 2] = FOOD
    env.add_agent(0, (2, 2))

    out = env.step_agents_batch(
        agent_ids=np.asarray([0], dtype=np.int32),
        actions=np.asarray([ACT_GATHER], dtype=np.int32),
        energies=np.asarray([10.0], dtype=np.float32),
        healths=np.asarray([10.0], dtype=np.float32),
        injuries=np.asarray([0.0], dtype=np.float32),
        scars=np.asarray([0.0], dtype=np.float32),
    )

    _, new_energy, _, food_gained, _, _ = out
    expected_gain = cfg.food_energy * 0.25
    assert env.patch_food_amount[2, 2] == pytest.approx(0.0)
    assert env.grid[2, 2] == EMPTY
    assert env.patch_food_harvest_total_last == pytest.approx(0.25)
    assert food_gained[0] == pytest.approx(expected_gain)
    assert new_energy[0] == pytest.approx(10.0 - cfg.gather_cost + expected_gain)


def test_patch_world_array_initialization_is_deterministic_and_rng_free() -> None:
    cfg_a = _env_cfg(patch_world_enable=True)
    cfg_b = _env_cfg(patch_world_enable=True)
    rm_a = RNGManager(global_seed=31)
    rm_b = RNGManager(global_seed=31)

    before_a = rm_a.capture_snapshot()
    before_b = rm_b.capture_snapshot()
    env_a = GridWorld(cfg_a, seed=31, run_dir=None, rng=rm_a.env_np)
    env_b = GridWorld(cfg_b, seed=31, run_dir=None, rng=rm_b.env_np)

    assert before_a == before_b
    assert rm_a.capture_snapshot() == rm_b.capture_snapshot()
    for name in PATCH_WORLD_ARRAY_NAMES:
        assert np.array_equal(getattr(env_a, name), getattr(env_b, name))


def test_patch_world_reset_recreates_passive_arrays() -> None:
    cfg = _env_cfg(patch_world_enable=True)
    rm = RNGManager(global_seed=37)
    env = GridWorld(cfg, seed=37, run_dir=None, rng=rm.env_np)
    env.patch_scent[:] = 0.5
    env.patch_food_amount[:] = 0.5

    env.reset()

    assert np.all(env.patch_scent == cfg.patch_scent_initial)
    assert np.all(env.patch_food_amount == cfg.patch_food_amount_initial_empty_cell_value)
    assert np.all(env.patch_shelter == cfg.patch_shelter_default)
    assert np.all(env.patch_movement_cost == cfg.patch_movement_cost_default)
    assert np.all(env.patch_waste == cfg.patch_waste_initial)
    assert np.all(env.local_density_map == 0.0)
    assert env.patch_scent.dtype == np.float32
    assert env.patch_food_amount.dtype == np.float32


def test_patch_world_presence_emission_decay_and_density_are_deterministic() -> None:
    cfg = _env_cfg(patch_world_enable=True)
    cfg.patch_food_regrow_per_step = 0.0
    cfg.patch_scent_decay_per_step = 0.10
    cfg.patch_waste_decay_per_step = 0.10
    cfg.patch_scent_presence_emit = 0.20
    cfg.patch_waste_presence_emit = 0.10
    cfg.patch_scent_move_emit = 0.0
    cfg.patch_scent_gather_emit = 0.0
    cfg.patch_waste_consume_emit = 0.0
    rm = RNGManager(global_seed=36)
    env = GridWorld(cfg, seed=36, run_dir=None, rng=rm.env_np)
    env.add_agent(0, (2, 2))
    env.patch_scent[2, 2] = np.float32(0.50)
    env.patch_waste[2, 2] = np.float32(0.50)

    env.patch_world_step(abs_step=1)

    assert env.patch_scent[2, 2] == pytest.approx(0.65)
    assert env.patch_waste[2, 2] == pytest.approx(0.55)
    assert env.patch_scent_decayed_total_last == pytest.approx(0.05, abs=1e-6)
    assert env.patch_waste_decayed_total_last == pytest.approx(0.05, abs=1e-6)
    assert env.patch_scent_emitted_total_last == pytest.approx(0.20)
    assert env.patch_waste_emitted_total_last == pytest.approx(0.10)
    assert env.local_density_map[2, 2] == pytest.approx(1.0)


def test_patch_world_trace_values_clamp_to_bounds() -> None:
    cfg = _env_cfg(patch_world_enable=True)
    cfg.patch_food_regrow_per_step = 0.0
    cfg.patch_scent_max = 0.6
    cfg.patch_waste_max = 0.7
    cfg.patch_scent_decay_per_step = 0.0
    cfg.patch_waste_decay_per_step = 0.0
    cfg.patch_scent_presence_emit = 0.5
    cfg.patch_waste_presence_emit = 0.5
    rm = RNGManager(global_seed=38)
    env = GridWorld(cfg, seed=38, run_dir=None, rng=rm.env_np)
    env.add_agent(0, (2, 2))
    env.patch_scent[2, 2] = np.float32(0.5)
    env.patch_waste[2, 2] = np.float32(0.5)

    env.patch_world_step(abs_step=1)

    assert env.patch_scent[2, 2] == pytest.approx(0.6)
    assert env.patch_waste[2, 2] == pytest.approx(0.7)
    assert np.all(env.patch_scent >= 0.0)
    assert np.all(env.patch_waste >= 0.0)


def test_patch_world_move_emits_scent_at_destination_without_behavior_effect() -> None:
    cfg = _env_cfg(patch_world_enable=True)
    cfg.patch_food_regrow_per_step = 0.0
    cfg.patch_scent_decay_per_step = 0.0
    cfg.patch_waste_decay_per_step = 0.0
    cfg.patch_scent_presence_emit = 0.0
    cfg.patch_waste_presence_emit = 0.0
    cfg.patch_scent_move_emit = 0.30
    cfg.patch_scent_gather_emit = 0.0
    cfg.patch_waste_consume_emit = 0.0
    rm = RNGManager(global_seed=40)
    env = GridWorld(cfg, seed=40, run_dir=None, rng=rm.env_np)
    env.add_agent(0, (2, 2))

    out = env.step_agents_batch(
        agent_ids=np.asarray([0], dtype=np.int32),
        actions=np.asarray([ACT_MOVE_E], dtype=np.int32),
        energies=np.asarray([10.0], dtype=np.float32),
        healths=np.asarray([10.0], dtype=np.float32),
        injuries=np.asarray([0.0], dtype=np.float32),
        scars=np.asarray([0.0], dtype=np.float32),
    )
    _, new_energy, new_health, _, _, _ = out
    env.patch_world_step(abs_step=1)

    assert env.agent_pos_x[0] == 3
    assert new_energy[0] == pytest.approx(10.0 - cfg.move_cost)
    assert new_health[0] == pytest.approx(10.0)
    assert env.patch_scent[2, 3] == pytest.approx(0.30)
    assert env.patch_waste[2, 3] == pytest.approx(0.0)


def test_patch_world_gather_emits_scent_and_waste_at_consumed_cell() -> None:
    cfg = _env_cfg(patch_world_enable=True)
    cfg.patch_food_regrow_per_step = 0.0
    cfg.patch_scent_decay_per_step = 0.0
    cfg.patch_waste_decay_per_step = 0.0
    cfg.patch_scent_presence_emit = 0.0
    cfg.patch_waste_presence_emit = 0.0
    cfg.patch_scent_move_emit = 0.0
    cfg.patch_scent_gather_emit = 0.40
    cfg.patch_waste_consume_emit = 0.20
    rm = RNGManager(global_seed=42)
    env = GridWorld(cfg, seed=42, run_dir=None, rng=rm.env_np)
    env.grid[2, 2] = FOOD
    env._reset_patch_world_arrays()
    env.add_agent(0, (2, 2))

    out = env.step_agents_batch(
        agent_ids=np.asarray([0], dtype=np.int32),
        actions=np.asarray([ACT_GATHER], dtype=np.int32),
        energies=np.asarray([10.0], dtype=np.float32),
        healths=np.asarray([10.0], dtype=np.float32),
        injuries=np.asarray([0.0], dtype=np.float32),
        scars=np.asarray([0.0], dtype=np.float32),
    )
    _, new_energy, new_health, food_gained, _, _ = out
    env.patch_world_step(abs_step=1)

    assert food_gained[0] == pytest.approx(cfg.food_energy)
    assert new_energy[0] == pytest.approx(10.0 - cfg.gather_cost + cfg.food_energy)
    assert new_health[0] == pytest.approx(10.0)
    assert env.patch_scent[2, 2] == pytest.approx(0.40)
    assert env.patch_waste[2, 2] == pytest.approx(0.20)


def _reference_local_density(agent_grid: np.ndarray, *, radius: int, include_self: bool, normalize: bool) -> np.ndarray:
    density = np.zeros(agent_grid.shape, dtype=np.float32)
    h, w = agent_grid.shape
    ys, xs = np.where(agent_grid != -1)
    for y_raw, x_raw in zip(ys.tolist(), xs.tolist()):
        y = int(y_raw)
        x = int(x_raw)
        y0 = max(0, y - int(radius))
        y1 = min(h, y + int(radius) + 1)
        x0 = max(0, x - int(radius))
        x1 = min(w, x + int(radius) + 1)
        density[y0:y1, x0:x1] += np.float32(1.0)
        if not include_self:
            density[y, x] -= np.float32(1.0)
    if normalize:
        denom = float((2 * int(radius) + 1) * (2 * int(radius) + 1))
        if not include_self:
            denom -= 1.0
        if denom > 0.0:
            density[:, :] = density / np.float32(denom)
    return np.ascontiguousarray(density, dtype=np.float32)


def test_patch_world_local_density_map_radius_and_include_self() -> None:
    cfg = _env_cfg(patch_world_enable=True)
    cfg.patch_density_radius = 1
    cfg.patch_density_include_self = True
    cfg.patch_density_normalize = False
    rm = RNGManager(global_seed=44)
    env = GridWorld(cfg, seed=44, run_dir=None, rng=rm.env_np)
    env.add_agent(0, (2, 2))
    env.add_agent(1, (2, 3))

    density = compute_local_density_map(env)

    assert density[2, 2] == pytest.approx(2.0)
    assert density[2, 3] == pytest.approx(2.0)
    assert density[2, 4] == pytest.approx(1.0)
    assert density[0, 0] == pytest.approx(0.0)

    cfg.patch_density_include_self = False
    cfg.patch_density_normalize = True
    density = compute_local_density_map(env)
    assert density[2, 2] == pytest.approx(1.0 / 8.0)
    assert density[2, 3] == pytest.approx(1.0 / 8.0)


def test_patch_world_local_density_map_matches_reference_cases() -> None:
    cfg = _env_cfg(patch_world_enable=True)
    rm = RNGManager(global_seed=4401)
    env = GridWorld(cfg, seed=4401, run_dir=None, rng=rm.env_np)

    cases = [
        np.full((5, 5), -1, dtype=np.int32),
        np.array(
            [
                [0, -1, -1, -1, 1],
                [-1, -1, -1, -1, -1],
                [-1, -1, 2, -1, -1],
                [-1, -1, -1, -1, -1],
                [3, -1, -1, -1, 4],
            ],
            dtype=np.int32,
        ),
        np.arange(25, dtype=np.int32).reshape(5, 5),
    ]
    rng = np.random.default_rng(4402)
    for _ in range(5):
        grid = np.full((7, 6), -1, dtype=np.int32)
        mask = rng.random(grid.shape) < 0.35
        grid[mask] = np.arange(int(mask.sum()), dtype=np.int32)
        cases.append(grid)

    for grid in cases:
        env.agent_grid = grid.copy()
        for radius in (0, 1, 2):
            for include_self in (False, True):
                for normalize in (False, True):
                    cfg.patch_density_radius = radius
                    cfg.patch_density_include_self = include_self
                    cfg.patch_density_normalize = normalize
                    actual = compute_local_density_map(env)
                    expected = _reference_local_density(
                        grid,
                        radius=radius,
                        include_self=include_self,
                        normalize=normalize,
                    )
                    assert actual.dtype == np.float32
                    assert actual.flags.c_contiguous
                    assert np.array_equal(actual, expected)


def test_patch_world_step7_occupied_bias_and_event_context_helpers() -> None:
    cfg = _env_cfg(patch_world_enable=True)
    cfg.patch_density_radius = 0
    cfg.patch_density_include_self = True
    rm = RNGManager(global_seed=45)
    env = GridWorld(cfg, seed=45, run_dir=None, rng=rm.env_np)
    env.patch_food_amount[:, :] = np.float32(0.0)
    env.patch_food_amount[1, 1] = np.float32(1.0)
    env.patch_cover[:, :] = np.float32(0.25)
    env.patch_cover[1, 1] = np.float32(0.75)
    env.patch_shelter[:, :] = np.float32(0.0)
    env.patch_shelter[1, 1] = np.float32(0.5)
    env.patch_movement_cost[:, :] = np.float32(1.0)
    env.patch_scent[:, :] = np.float32(0.0)
    env.patch_scent[1, 1] = np.float32(0.4)
    env.patch_waste[:, :] = np.float32(0.0)
    env.patch_waste[1, 1] = np.float32(0.2)
    env.add_agent(0, (1, 1))

    metrics = patch_occupied_descriptor_metrics(env)
    global_food_mean = 1.0 / float(cfg.width * cfg.height)

    assert metrics["occupied_patch_food_amount_mean"] == pytest.approx(1.0)
    assert metrics["patch_food_occupancy_bias"] == pytest.approx(1.0 - global_food_mean)
    assert metrics["occupied_patch_cover_mean"] == pytest.approx(0.75)
    assert metrics["occupied_patch_visibility_mean"] == pytest.approx(0.25)

    agent = SimpleNamespace(agent_id=0, stats=SimpleNamespace(patch_habitat_lineage_id=7))
    context = sample_patch_context(env, 1, 1, agent=agent)
    summary = patch_event_contexts_summary("birth", [context])

    assert context["lineage_id"] == pytest.approx(7.0)
    assert summary["birth_patch_count"] == pytest.approx(1.0)
    assert summary["birth_patch_food_amount_mean"] == pytest.approx(1.0)
    assert summary["birth_patch_waste_mean"] == pytest.approx(0.2)


def test_patch_world_step7_lineage_descriptors_are_deterministic() -> None:
    cfg = _env_cfg(patch_world_enable=True)
    rm = RNGManager(global_seed=46)
    env = GridWorld(cfg, seed=46, run_dir=None, rng=rm.env_np)
    env.patch_food_amount[:, :] = np.float32(0.0)
    env.patch_food_amount[1, 1] = np.float32(1.0)
    env.patch_food_amount[2, 2] = np.float32(0.5)
    env.patch_cover[:, :] = np.float32(0.0)
    env.patch_cover[1, 1] = np.float32(0.8)
    env.patch_cover[2, 2] = np.float32(0.2)
    env.patch_waste[:, :] = np.float32(0.0)
    env.patch_waste[1, 1] = np.float32(0.1)
    env.patch_waste[2, 2] = np.float32(0.3)
    env.add_agent(0, (1, 1))
    env.add_agent(1, (2, 2))

    agents = [
        SimpleNamespace(agent_id=1, stats=SimpleNamespace(patch_habitat_lineage_id=20)),
        SimpleNamespace(agent_id=0, stats=SimpleNamespace(patch_habitat_lineage_id=10)),
    ]

    descriptors = lineage_habitat_descriptors(env, agents)
    metrics = lineage_habitat_metrics(descriptors)

    assert [int(d["lineage_id"]) for d in descriptors] == [10, 20]
    assert descriptors[0]["lineage_alive_count"] == pytest.approx(1.0)
    assert descriptors[0]["lineage_patch_food_amount_mean"] == pytest.approx(1.0)
    assert descriptors[1]["lineage_patch_food_amount_mean"] == pytest.approx(0.5)
    assert metrics["lineage_habitat_descriptor_count"] == pytest.approx(2.0)
    assert metrics["lineage_patch_food_amount_mean_std"] == pytest.approx(0.25)


def test_patch_world_step13_lineage_entropy_concentration_and_warnings() -> None:
    cfg = _env_cfg(patch_world_enable=True)
    cfg.lineage_dominance_top1_warning_threshold = 0.70
    cfg.lineage_dominance_top3_warning_threshold = 0.90
    cfg.lineage_entropy_warning_threshold = 0.50
    descriptors = [
        {"lineage_id": 10.0, "lineage_alive_count": 3.0, "lineage_birth_count": 2.0},
        {"lineage_id": 20.0, "lineage_alive_count": 1.0, "lineage_birth_count": 2.0},
    ]

    metrics = lineage_strategy_metrics(descriptors, cfg=cfg)

    expected_alive_entropy = -(0.75 * math.log(0.75) + 0.25 * math.log(0.25))
    assert metrics["lineage_diversity_schema_version"] == pytest.approx(LINEAGE_DIVERSITY_SCHEMA_VERSION)
    assert metrics["lineage_active_count"] == pytest.approx(2.0)
    assert metrics["lineage_alive_count_total"] == pytest.approx(4.0)
    assert metrics["lineage_top1_alive_share"] == pytest.approx(0.75)
    assert metrics["lineage_top3_alive_share"] == pytest.approx(1.0)
    assert metrics["lineage_alive_share_entropy"] == pytest.approx(expected_alive_entropy)
    assert metrics["lineage_birth_share_entropy"] == pytest.approx(math.log(2.0))
    assert metrics["lineage_effective_birth_lineages"] == pytest.approx(2.0)
    assert metrics["lineage_top1_dominance_warning"] == pytest.approx(1.0)
    assert metrics["lineage_top3_dominance_warning"] == pytest.approx(1.0)
    assert metrics["lineage_entropy_collapse_warning"] == pytest.approx(0.0)


def test_patch_world_step13_strategy_descriptors_include_reproduction_predator_and_recovery_context() -> None:
    cfg = _env_cfg(patch_world_enable=True)
    rm = RNGManager(global_seed=53)
    env = GridWorld(cfg, seed=53, run_dir=None, rng=rm.env_np)
    env.patch_cover[:, :] = np.float32(0.0)
    env.patch_shelter[:, :] = np.float32(0.0)
    env.patch_waste[:, :] = np.float32(0.0)
    env.patch_recovery_activity[:, :] = np.float32(0.0)
    env.patch_cover[1, 1] = np.float32(0.8)
    env.patch_shelter[1, 1] = np.float32(0.4)
    env.patch_waste[1, 1] = np.float32(0.2)
    env.patch_recovery_activity[1, 1] = np.float32(0.7)
    env.patch_cover[2, 2] = np.float32(0.1)
    env.patch_shelter[2, 2] = np.float32(0.9)
    env.patch_waste[2, 2] = np.float32(0.6)
    env.patch_recovery_activity[2, 2] = np.float32(0.3)
    env.add_agent(0, (1, 1))
    env.add_agent(1, (2, 2))
    agents = [
        SimpleNamespace(agent_id=0, stats=SimpleNamespace(patch_habitat_lineage_id=10)),
        SimpleNamespace(agent_id=1, stats=SimpleNamespace(patch_habitat_lineage_id=20)),
    ]
    birth_context = sample_patch_context(env, 1, 1, lineage_id=10)
    predator_context = sample_patch_context(env, 2, 2, lineage_id=20)
    predator_context["raw_damage"] = 0.5
    predator_context["effective_damage"] = 0.25
    predator_context["damage_prevented"] = 0.25

    descriptors = lineage_strategy_descriptors(
        env,
        agents,
        birth_contexts=[birth_context],
        predator_event_contexts=[predator_context],
        cfg=cfg,
    )
    metrics = lineage_strategy_metrics(descriptors, cfg=cfg)

    assert [int(d["lineage_id"]) for d in descriptors] == [10, 20]
    lineage10 = descriptors[0]
    lineage20 = descriptors[1]
    assert lineage10["lineage_birth_count"] == pytest.approx(1.0)
    assert lineage10["lineage_birth_patch_recovery_activity_mean"] == pytest.approx(0.7)
    assert lineage10["lineage_reproduction_rate"] == pytest.approx(1.0)
    assert lineage20["lineage_predator_event_count"] == pytest.approx(1.0)
    assert lineage20["lineage_predator_damage_total"] == pytest.approx(0.25)
    assert lineage20["lineage_predator_damage_prevented_total"] == pytest.approx(0.25)
    assert lineage20["lineage_predator_event_patch_shelter_mean"] == pytest.approx(0.9)
    assert metrics["lineage_strategy_descriptor_count"] == pytest.approx(2.0)
    assert metrics["lineage_patch_recovery_activity_mean_std"] == pytest.approx(0.2)
    assert np.isfinite(np.asarray(list(metrics.values()), dtype=np.float64)).all()


def test_patch_world_step13_branch_candidates_are_deterministic_reports_only() -> None:
    cfg = _env_cfg(patch_world_enable=True)
    cfg.lineage_branch_candidate_top_n = 4
    descriptors = [
        {
            "lineage_id": 20.0,
            "lineage_alive_count": 2.0,
            "lineage_birth_count": 1.0,
            "lineage_alive_share": 0.4,
            "lineage_birth_share": 0.25,
            "lineage_patch_cover_mean": 0.2,
            "lineage_patch_shelter_mean": 0.9,
            "lineage_patch_waste_mean": 0.6,
            "lineage_patch_density_mean": 1.0,
            "lineage_patch_recovery_activity_mean": 0.3,
            "lineage_predator_damage_mean": 0.4,
            "lineage_reproduction_rate": 0.5,
        },
        {
            "lineage_id": 10.0,
            "lineage_alive_count": 3.0,
            "lineage_birth_count": 3.0,
            "lineage_alive_share": 0.6,
            "lineage_birth_share": 0.75,
            "lineage_patch_cover_mean": 0.8,
            "lineage_patch_shelter_mean": 0.1,
            "lineage_patch_waste_mean": 0.2,
            "lineage_patch_density_mean": 0.5,
            "lineage_patch_recovery_activity_mean": 0.7,
            "lineage_predator_damage_mean": 0.1,
            "lineage_reproduction_rate": 1.0,
        },
    ]

    candidates_a = lineage_branch_candidates(descriptors, cfg=cfg)
    candidates_b = lineage_branch_candidates(list(reversed(descriptors)), cfg=cfg)

    assert candidates_a == candidates_b
    assert len(candidates_a) == 4
    assert [c["candidate_rank"] for c in candidates_a] == [1.0, 2.0, 3.0, 4.0]
    assert {c["candidate_reason"] for c in candidates_a}
    assert all("candidate_score_diagnostic_only" in c for c in candidates_a)


def test_patch_world_step7_observation_validation_metrics() -> None:
    cfg = _env_cfg(patch_world_enable=True)
    cfg.patch_world_observation_enable = True
    rm = RNGManager(global_seed=47)
    env = GridWorld(cfg, seed=47, run_dir=None, rng=rm.env_np)
    env.add_agent(0, (1, 1))

    metrics = patch_observation_validation_metrics(env)

    assert metrics["patch_observation_nan_count"] == pytest.approx(0.0)
    assert metrics["patch_observation_min"] >= -1.0
    assert metrics["patch_observation_max"] <= 1.0


def _single_fertile_regrowth_env(cfg: EnvConfig, *, seed: int = 48) -> GridWorld:
    rm = RNGManager(global_seed=seed)
    env = GridWorld(cfg, seed=seed, run_dir=None, rng=rm.env_np)
    env.patch_food_amount[:, :] = np.float32(0.0)
    env.patch_fertility[:, :] = np.float32(0.0)
    env.patch_fertility[1, 1] = np.float32(1.0)
    env.patch_waste[:, :] = np.float32(0.0)
    env.local_density_map[:, :] = np.float32(0.0)
    return env


def test_patch_world_step8_consequence_disabled_preserves_step3_regrowth() -> None:
    cfg = _env_cfg(patch_world_enable=True)
    cfg.patch_food_regrow_per_step = 0.2
    cfg.patch_world_waste_consequence_enable = False
    env = _single_fertile_regrowth_env(cfg)
    env.patch_waste[1, 1] = np.float32(1.0)

    total = regrow_patch_food(env)

    assert total == pytest.approx(0.2)
    assert env.patch_food_amount[1, 1] == pytest.approx(0.2)
    assert env.patch_regrowth_base_total_last == pytest.approx(0.2)
    assert env.patch_regrowth_effective_total_last == pytest.approx(0.2)
    assert env.patch_regrowth_suppressed_total_last == pytest.approx(0.0)
    assert env.patch_regrowth_multiplier_min_last == pytest.approx(1.0)


def test_patch_world_step8_waste_regrowth_penalty_is_bounded() -> None:
    cfg = _env_cfg(patch_world_enable=True)
    cfg.patch_food_regrow_per_step = 0.2
    cfg.patch_world_waste_consequence_enable = True
    cfg.patch_waste_regrowth_penalty_enable = True
    cfg.patch_density_regrowth_penalty_enable = False
    cfg.patch_waste_regrowth_penalty_scale = 0.5
    cfg.patch_waste_regrowth_penalty_max = 0.5
    env = _single_fertile_regrowth_env(cfg)
    env.patch_waste[1, 1] = np.float32(1.0)

    total = regrow_patch_food(env)

    assert total == pytest.approx(0.1)
    assert env.patch_food_amount[1, 1] == pytest.approx(0.1)
    assert env.patch_regrowth_base_total_last == pytest.approx(0.2)
    assert env.patch_regrowth_effective_total_last == pytest.approx(0.1)
    assert env.patch_regrowth_suppressed_total_last == pytest.approx(0.1)
    assert env.patch_waste_regrowth_penalty_max_last == pytest.approx(0.5)
    assert env.patch_density_regrowth_penalty_max_last == pytest.approx(0.0)


def test_patch_world_step8_density_regrowth_penalty_uses_normalized_density() -> None:
    cfg = _env_cfg(patch_world_enable=True)
    cfg.patch_food_regrow_per_step = 0.2
    cfg.patch_world_waste_consequence_enable = True
    cfg.patch_waste_regrowth_penalty_enable = False
    cfg.patch_density_regrowth_penalty_enable = True
    cfg.patch_density_regrowth_penalty_scale = 0.5
    cfg.patch_density_regrowth_penalty_max = 0.5
    cfg.patch_density_radius = 0
    cfg.patch_density_include_self = True
    cfg.patch_density_normalize = False
    env = _single_fertile_regrowth_env(cfg, seed=49)
    env.add_agent(0, (1, 1))

    total = regrow_patch_food(env)

    assert total == pytest.approx(0.1)
    assert env.local_density_map[1, 1] == pytest.approx(1.0)
    assert env.patch_density_regrowth_penalty_max_last == pytest.approx(0.5)
    assert env.patch_regrowth_suppressed_total_last == pytest.approx(0.1)


def test_patch_world_step8_combined_penalty_respects_min_multiplier_and_fertility_is_deferred() -> None:
    cfg = _env_cfg(patch_world_enable=True)
    cfg.patch_food_regrow_per_step = 0.2
    cfg.patch_world_waste_consequence_enable = True
    cfg.patch_waste_regrowth_penalty_scale = 0.5
    cfg.patch_density_regrowth_penalty_scale = 0.5
    cfg.patch_regrowth_min_multiplier = 0.25
    cfg.patch_density_radius = 0
    cfg.patch_density_include_self = True
    cfg.patch_waste_fertility_effect_enable = False
    env = _single_fertile_regrowth_env(cfg, seed=50)
    env.patch_waste[1, 1] = np.float32(1.0)
    env.add_agent(0, (1, 1))
    fertility_before = env.patch_fertility.copy()

    total = regrow_patch_food(env)

    assert total == pytest.approx(0.05)
    assert env.patch_food_amount[1, 1] == pytest.approx(0.05)
    assert env.patch_regrowth_multiplier_min_last == pytest.approx(0.25)
    assert env.patch_regrowth_suppressed_total_last == pytest.approx(0.15)
    assert np.array_equal(env.patch_fertility, fertility_before)


def test_patch_world_step12_recovery_defaults_disabled_and_inert() -> None:
    cfg = _env_cfg(patch_world_enable=True)
    cfg.patch_food_regrow_per_step = 0.2
    cfg.jungle_ecological_recovery_enable = False
    env = _single_fertile_regrowth_env(cfg)
    assert hasattr(env, "patch_recovery_activity")
    env.patch_recovery_activity[1, 1] = np.float32(1.0)

    total = regrow_patch_food(env)

    assert cfg.jungle_ecological_recovery_enable is False
    assert cfg.lineage_diversity_reporting_enable is True
    assert cfg.lineage_strategy_descriptors_enable is True
    assert cfg.lineage_branch_candidate_reporting_enable is True
    assert cfg.lineage_diversity_schema_version == LINEAGE_DIVERSITY_SCHEMA_VERSION
    assert cfg.lineage_dominance_top1_warning_threshold == 0.75
    assert cfg.lineage_dominance_top3_warning_threshold == 0.90
    assert cfg.lineage_entropy_warning_threshold == 0.50
    assert cfg.lineage_low_active_count_warning_threshold == 2
    assert cfg.lineage_min_alive_count_for_descriptor == 1
    assert cfg.lineage_branch_candidate_top_n == 5
    assert total == pytest.approx(0.2)
    assert env.patch_regrowth_after_waste_total_last == pytest.approx(0.2)
    assert env.patch_regrowth_after_recovery_total_last == pytest.approx(0.2)
    assert env.patch_recovery_added_regrowth_total_last == pytest.approx(0.0)
    assert env.jungle_recovery_regrowth_boost_max_last == pytest.approx(0.0)


def test_patch_world_step12_recovery_activity_update_is_deterministic_and_bounded() -> None:
    cfg = _env_cfg(patch_world_enable=True)
    cfg.jungle_ecological_recovery_enable = True
    cfg.jungle_recovery_activity_initial = 0.1
    cfg.jungle_recovery_activity_min = 0.0
    cfg.jungle_recovery_activity_max = 1.0
    cfg.jungle_recovery_activity_decay_per_step = 0.0
    cfg.jungle_recovery_activity_base_growth = 0.0
    cfg.jungle_recovery_activity_cover_weight = 0.2
    cfg.jungle_recovery_activity_fertility_weight = 0.3
    cfg.jungle_recovery_activity_waste_penalty_weight = 0.4
    cfg.jungle_recovery_activity_density_penalty_weight = 0.0
    cfg.jungle_recovery_activity_scent_penalty_weight = 0.0
    rm = RNGManager(global_seed=51)
    env = GridWorld(cfg, seed=51, run_dir=None, rng=rm.env_np)
    env.patch_recovery_activity.fill(0.1)
    env.patch_cover.fill(0.0)
    env.patch_fertility.fill(0.0)
    env.patch_waste.fill(0.0)
    env.patch_cover[1, 1] = np.float32(1.0)
    env.patch_fertility[1, 1] = np.float32(1.0)
    env.patch_waste[2, 2] = np.float32(1.0)

    update_patch_recovery_activity(env)

    assert env.patch_recovery_activity[1, 1] == pytest.approx(0.6)
    assert env.patch_recovery_activity[2, 2] == pytest.approx(0.0)
    assert float(np.min(env.patch_recovery_activity)) >= 0.0
    assert float(np.max(env.patch_recovery_activity)) <= 1.0
    assert np.isfinite(env.patch_recovery_activity).all()
    assert env.patch_recovery_positive_total_last > 0.0
    assert env.patch_recovery_negative_total_last == pytest.approx(0.4)


def test_patch_world_step12_regrowth_boost_applies_after_step8_suppression() -> None:
    cfg = _env_cfg(patch_world_enable=True)
    cfg.patch_food_regrow_per_step = 0.2
    cfg.patch_world_waste_consequence_enable = True
    cfg.patch_waste_regrowth_penalty_enable = True
    cfg.patch_density_regrowth_penalty_enable = False
    cfg.patch_waste_regrowth_penalty_scale = 0.5
    cfg.patch_waste_regrowth_penalty_max = 0.5
    cfg.jungle_ecological_recovery_enable = True
    cfg.jungle_recovery_activity_initial = 1.0
    cfg.jungle_recovery_activity_decay_per_step = 0.0
    cfg.jungle_recovery_activity_cover_weight = 0.0
    cfg.jungle_recovery_activity_fertility_weight = 0.0
    cfg.jungle_recovery_activity_waste_penalty_weight = 0.0
    cfg.jungle_recovery_activity_density_penalty_weight = 0.0
    cfg.jungle_recovery_activity_scent_penalty_weight = 0.0
    cfg.jungle_recovery_regrowth_boost_scale = 0.25
    cfg.jungle_recovery_regrowth_boost_max = 0.25
    env = _single_fertile_regrowth_env(cfg)
    env.patch_recovery_activity.fill(0.0)
    env.patch_recovery_activity[1, 1] = np.float32(1.0)
    env.patch_waste[1, 1] = np.float32(1.0)

    total = regrow_patch_food(env)

    assert total == pytest.approx(0.125)
    assert env.patch_regrowth_base_total_last == pytest.approx(0.2)
    assert env.patch_regrowth_after_waste_total_last == pytest.approx(0.1)
    assert env.patch_regrowth_after_recovery_total_last == pytest.approx(0.125)
    assert env.patch_recovery_added_regrowth_total_last == pytest.approx(0.025)
    assert env.jungle_recovery_regrowth_boost_max_last == pytest.approx(0.25)
    assert patch_recovery_regrowth_boost_maps(env)["boost"][1, 1] == pytest.approx(0.25)


def test_patch_world_step12_recovery_metrics_and_context_are_diagnostic_only() -> None:
    cfg = _env_cfg(patch_world_enable=True)
    cfg.jungle_ecological_recovery_enable = True
    rm = RNGManager(global_seed=52)
    env = GridWorld(cfg, seed=52, run_dir=None, rng=rm.env_np)
    env.patch_recovery_activity.fill(0.0)
    env.patch_recovery_activity[1, 1] = np.float32(0.75)
    env.add_agent(0, (1, 1))

    metrics = patch_recovery_metrics(env)
    context = sample_patch_context(env, 1, 1)
    descriptors = patch_occupied_descriptor_metrics(env)

    assert metrics["jungle_ecological_recovery_enable"] == 1
    assert metrics["patch_recovery_activity_max"] == pytest.approx(0.75)
    assert metrics["patch_recovery_activity_occupied_mean"] == pytest.approx(0.75)
    assert context["recovery_activity"] == pytest.approx(0.75)
    assert descriptors["occupied_patch_recovery_activity_mean"] == pytest.approx(0.75)
    assert "patch_recovery_activity_norm" not in PATCH_WORLD_OBSERVATION_CHANNEL_NAMES
    assert "future_regrowth" not in PATCH_WORLD_OBSERVATION_CHANNEL_NAMES


def test_patch_world_movement_cost_passive_by_default_preserves_move_energy() -> None:
    cfg = _env_cfg(patch_world_enable=True)
    cfg.patch_movement_cost_apply_enable = False
    rm = RNGManager(global_seed=38)
    env = GridWorld(cfg, seed=38, run_dir=None, rng=rm.env_np)
    env.patch_movement_cost[2, 3] = np.float32(3.0)
    env.add_agent(0, (2, 2))

    out = env.step_agents_batch(
        agent_ids=np.asarray([0], dtype=np.int32),
        actions=np.asarray([ACT_MOVE_E], dtype=np.int32),
        energies=np.asarray([10.0], dtype=np.float32),
        healths=np.asarray([10.0], dtype=np.float32),
        injuries=np.asarray([0.0], dtype=np.float32),
        scars=np.asarray([0.0], dtype=np.float32),
    )

    _, new_energy, _, _, _, _ = out
    assert env.agent_pos_y[0] == 2
    assert env.agent_pos_x[0] == 3
    assert new_energy[0] == pytest.approx(10.0 - cfg.move_cost)
    assert env.patch_movement_energy_extra_last == pytest.approx(0.0)
    assert env.patch_movement_energy_event_count_last == 0


def test_patch_world_active_neutral_movement_cost_preserves_move_energy() -> None:
    cfg = _env_cfg(patch_world_enable=True)
    cfg.patch_movement_cost_apply_enable = True
    cfg.patch_movement_cost_energy_scale = 1.0
    rm = RNGManager(global_seed=40)
    env = GridWorld(cfg, seed=40, run_dir=None, rng=rm.env_np)
    env.patch_movement_cost[2, 3] = np.float32(1.0)
    env.add_agent(0, (2, 2))

    out = env.step_agents_batch(
        agent_ids=np.asarray([0], dtype=np.int32),
        actions=np.asarray([ACT_MOVE_E], dtype=np.int32),
        energies=np.asarray([10.0], dtype=np.float32),
        healths=np.asarray([10.0], dtype=np.float32),
        injuries=np.asarray([0.0], dtype=np.float32),
        scars=np.asarray([0.0], dtype=np.float32),
    )

    _, new_energy, _, _, _, _ = out
    assert new_energy[0] == pytest.approx(10.0 - cfg.move_cost)
    assert env.patch_movement_energy_extra_last == pytest.approx(0.0)
    assert env.patch_movement_energy_event_count_last == 1


def test_patch_world_active_movement_cost_uses_destination_cell() -> None:
    cfg = _env_cfg(patch_world_enable=True)
    cfg.patch_movement_cost_apply_enable = True
    cfg.patch_movement_cost_energy_scale = 1.0
    rm = RNGManager(global_seed=42)
    env = GridWorld(cfg, seed=42, run_dir=None, rng=rm.env_np)
    env.patch_movement_cost[2, 3] = np.float32(2.0)
    env.add_agent(0, (2, 2))

    out = env.step_agents_batch(
        agent_ids=np.asarray([0], dtype=np.int32),
        actions=np.asarray([ACT_MOVE_E], dtype=np.int32),
        energies=np.asarray([10.0], dtype=np.float32),
        healths=np.asarray([10.0], dtype=np.float32),
        injuries=np.asarray([0.0], dtype=np.float32),
        scars=np.asarray([0.0], dtype=np.float32),
    )

    _, high_energy, _, _, _, _ = out
    assert env.agent_pos_x[0] == 3
    assert high_energy[0] == pytest.approx(10.0 - (cfg.move_cost * 2.0))
    assert env.patch_movement_energy_extra_last == pytest.approx(cfg.move_cost)

    env.remove_agent(0)
    env.add_agent(0, (2, 2))
    env.patch_movement_cost[2, 3] = np.float32(0.5)
    out = env.step_agents_batch(
        agent_ids=np.asarray([0], dtype=np.int32),
        actions=np.asarray([ACT_MOVE_E], dtype=np.int32),
        energies=np.asarray([10.0], dtype=np.float32),
        healths=np.asarray([10.0], dtype=np.float32),
        injuries=np.asarray([0.0], dtype=np.float32),
        scars=np.asarray([0.0], dtype=np.float32),
    )

    _, low_energy, _, _, _, _ = out
    assert low_energy[0] == pytest.approx(10.0 - (cfg.move_cost * 0.5))
    assert env.patch_movement_energy_extra_last == pytest.approx(-cfg.move_cost * 0.5)


def test_patch_world_disabled_movement_keeps_legacy_move_energy() -> None:
    cfg = _env_cfg(patch_world_enable=False)
    rm = RNGManager(global_seed=44)
    env = GridWorld(cfg, seed=44, run_dir=None, rng=rm.env_np)
    env.add_agent(0, (2, 2))

    out = env.step_agents_batch(
        agent_ids=np.asarray([0], dtype=np.int32),
        actions=np.asarray([ACT_MOVE_E], dtype=np.int32),
        energies=np.asarray([10.0], dtype=np.float32),
        healths=np.asarray([10.0], dtype=np.float32),
        injuries=np.asarray([0.0], dtype=np.float32),
        scars=np.asarray([0.0], dtype=np.float32),
    )

    _, new_energy, _, _, _, _ = out
    assert not hasattr(env, "patch_movement_cost")
    assert env.agent_pos_y[0] == 2
    assert env.agent_pos_x[0] == 3
    assert new_energy[0] == pytest.approx(10.0 - cfg.move_cost)


def test_patch_world_regrowth_uses_fertility_and_clamps() -> None:
    cfg = _env_cfg(patch_world_enable=True)
    cfg.patch_food_regrow_per_step = 0.2
    cfg.patch_food_presence_threshold = 0.5
    rm = RNGManager(global_seed=39)
    env = GridWorld(cfg, seed=39, run_dir=None, rng=rm.env_np)
    env.patch_food_amount.fill(0.0)
    env.patch_fertility.fill(0.0)
    env.patch_fertility[0, 0] = np.float32(0.5)
    env.patch_fertility[0, 1] = np.float32(1.0)
    env.patch_food_amount[0, 2] = np.float32(0.95)
    env.patch_fertility[0, 2] = np.float32(1.0)

    env.patch_world_step(abs_step=1)

    assert env.patch_food_amount[0, 0] == pytest.approx(0.1)
    assert env.patch_food_amount[0, 1] == pytest.approx(0.2)
    assert env.patch_food_amount[0, 2] == pytest.approx(cfg.patch_food_amount_max)
    assert env.patch_food_regrowth_total_last == pytest.approx(0.35, abs=1e-6)
    assert env.grid[0, 0] == EMPTY
    assert env.grid[0, 2] == FOOD


def test_patch_world_binary_projection_feeds_existing_observation_channel() -> None:
    cfg = _env_cfg(patch_world_enable=True)
    cfg.patch_food_regrow_per_step = 0.0
    rm = RNGManager(global_seed=41)
    env = GridWorld(cfg, seed=41, run_dir=None, rng=rm.env_np)
    env.patch_food_amount.fill(0.0)
    env.patch_food_amount[1, 2] = np.float32(0.75)

    from env.gridworld_pkg.patch_world import project_patch_food_to_binary_grid

    project_patch_food_to_binary_grid(env)
    env.add_agent(0, (1, 1))
    obs = env.get_local_observation(0)

    assert obs.shape == (3, 3)
    assert env.grid[1, 2] == FOOD
    assert obs.reshape(3, 3)[1, 2] == pytest.approx(1.0)


def test_patch_world_spawn_syncs_binary_food_to_patch_amount() -> None:
    cfg = _env_cfg(patch_world_enable=True)
    cfg.patch_food_regrow_per_step = 0.0
    cfg.food_spawn_per_step = 1
    cfg.max_food = 1
    rm = RNGManager(global_seed=43)
    env = GridWorld(cfg, seed=43, run_dir=None, rng=rm.env_np)
    env.patch_food_amount.fill(0.0)
    env.grid.fill(EMPTY)

    env.spawn_food()

    food_mask = env.grid == FOOD
    assert int(food_mask.sum()) == 1
    assert np.all(env.patch_food_amount[food_mask] == np.float32(cfg.patch_food_amount_initial_food_cell_value))


def test_patch_world_observation_shape_and_obs_size_unchanged(tmp_path: Path) -> None:
    disabled = _make_sim(tmp_path / "disabled_obs", patch_world_enable=False)
    enabled = _make_sim(tmp_path / "enabled_obs", patch_world_enable=True)

    for sim in (disabled, enabled):
        sim.pop.init_random()
        sim._place_agents()

    disabled_ids = np.asarray([a.agent_id for a in disabled.pop.alive_agents()], dtype=np.int32)
    enabled_ids = np.asarray([a.agent_id for a in enabled.pop.alive_agents()], dtype=np.int32)

    assert disabled.obs_size == enabled.obs_size
    assert disabled.pop.obs_size == enabled.pop.obs_size
    assert disabled.env.get_local_observations(disabled_ids).shape == enabled.env.get_local_observations(enabled_ids).shape
    assert disabled.env.get_local_observations(disabled_ids).shape[1] == (2 * disabled.env_cfg.view_radius + 1) ** 2
    assert patch_world_observation_enabled(enabled.env_cfg) is False


def test_patch_world_observations_enabled_expands_local_and_policy_input_size(tmp_path: Path) -> None:
    cfg = _env_cfg(patch_world_enable=True)
    cfg.patch_world_observation_enable = True
    sim = SimulationRunner(
        env_cfg=cfg,
        pop_cfg=PopConfig(population_size=4),
        sim_cfg=_sim_cfg(seed=331),
        logger=NoOpMetricsLogger(),
        run_dir=str(tmp_path / "patch_obs_enabled_size"),
        perf=False,
    )
    sim.pop.init_random()
    sim._place_agents()
    ids = np.asarray([a.agent_id for a in sim.pop.alive_agents()], dtype=np.int32)
    local = sim.env.get_local_observations(ids)

    local_cells = (2 * cfg.view_radius + 1) ** 2
    expected_local = local_cells * len(PATCH_WORLD_OBSERVATION_CHANNEL_NAMES)
    expected_obs = expected_local + 2
    assert PATCH_WORLD_OBSERVATION_SCHEMA_VERSION == 1
    assert patch_world_observation_enabled(cfg) is True
    assert patch_world_local_observation_size(cfg) == expected_local
    assert patch_world_total_observation_size(cfg) == expected_obs
    assert local.shape == (len(ids), expected_local)
    assert sim.obs_size == expected_obs
    assert sim._obs_size == expected_obs
    assert sim.pop.obs_size == expected_obs
    assert np.all(np.isfinite(local))


def test_patch_world_observation_channels_are_local_normalized_and_ordered() -> None:
    cfg = _env_cfg(patch_world_enable=True)
    cfg.patch_world_observation_enable = True
    cfg.patch_scent_max = 2.0
    cfg.patch_waste_max = 4.0
    cfg.patch_movement_cost_min = 0.0
    cfg.patch_movement_cost_max = 4.0
    rm = RNGManager(global_seed=335)
    env = GridWorld(cfg, seed=335, run_dir=None, rng=rm.env_np)
    env.add_agent(0, (2, 2))
    env.patch_food_amount[2, 2] = np.float32(0.5)
    env.patch_cover[2, 2] = np.float32(0.25)
    env.patch_shelter[2, 2] = np.float32(0.75)
    env.patch_scent[2, 2] = np.float32(1.0)
    env.patch_waste[2, 2] = np.float32(2.0)
    env.patch_movement_cost[2, 2] = np.float32(2.0)

    obs = env.get_local_observation(0)
    local_cells = 9
    center = 4
    assert obs.shape == (local_cells * len(PATCH_WORLD_OBSERVATION_CHANNEL_NAMES),)
    assert obs[0 * local_cells + center] == pytest.approx(-1.0)
    assert obs[1 * local_cells + center] == pytest.approx(0.5)
    assert obs[2 * local_cells + center] == pytest.approx(0.25)
    assert obs[3 * local_cells + center] == pytest.approx(0.75)
    assert obs[4 * local_cells + center] == pytest.approx(0.5)
    assert obs[5 * local_cells + center] == pytest.approx(0.5)
    assert obs[6 * local_cells + center] == pytest.approx(1.0 / 9.0)
    assert obs[7 * local_cells + center] == pytest.approx(0.5)
    assert np.all(np.isfinite(obs))
    assert float(np.max(obs[local_cells:])) <= 1.0
    assert float(np.min(obs[local_cells:])) >= 0.0


def test_patch_world_observation_edge_padding_and_no_rng_use() -> None:
    cfg = _env_cfg(patch_world_enable=True)
    cfg.patch_world_observation_enable = True
    rm = RNGManager(global_seed=337)
    env = GridWorld(cfg, seed=337, run_dir=None, rng=rm.env_np)
    env.patch_food_amount.fill(np.float32(1.0))
    env.add_agent(0, (0, 0))
    before = rm.capture_snapshot()

    obs = env.get_local_observation(0)

    assert before == rm.capture_snapshot()
    local_cells = 9
    assert obs[1 * local_cells + 0] == pytest.approx(0.0)
    assert obs[1 * local_cells + 4] == pytest.approx(1.0)


def test_patch_world_disabled_matches_default_path(tmp_path: Path) -> None:
    default_sim = _make_sim(tmp_path / "default", patch_world_enable=False, seed=222)

    default_sim.run(start_gen=0, skip_init=False)

    assert default_sim.env_cfg.patch_world_enable is False
    assert default_sim.env.patch_world_enable is False
    assert default_sim.obs_size == (2 * default_sim.env_cfg.view_radius + 1) ** 2 + 2
    assert len(default_sim.pop.alive_agents()) > 0


def test_patch_world_enabled_short_run_preserves_observation_shape(tmp_path: Path) -> None:
    enabled_sim = _make_sim(tmp_path / "enabled", patch_world_enable=True, seed=333)
    expected_obs_size = enabled_sim.obs_size

    enabled_sim.run(start_gen=0, skip_init=False)

    alive_ids = np.asarray([a.agent_id for a in enabled_sim.pop.alive_agents()], dtype=np.int32)
    obs = enabled_sim.env.get_local_observations(alive_ids)

    assert enabled_sim.env_cfg.patch_world_enable is True
    assert enabled_sim.env.patch_world_enable is True
    assert enabled_sim.obs_size == expected_obs_size
    assert enabled_sim.pop.obs_size == expected_obs_size
    assert obs.shape[1] == (2 * enabled_sim.env_cfg.view_radius + 1) ** 2
    assert len(enabled_sim.pop.alive_agents()) > 0


def test_patch_world_checkpoint_round_trips_enabled_patch_arrays(tmp_path: Path) -> None:
    sim = _make_sim(tmp_path / "checkpoint_enabled_src", patch_world_enable=True, seed=444)
    sim.run(start_gen=0, skip_init=False)
    sim.env.grid[0, 0] = FOOD
    sim.env._reset_patch_world_arrays()
    sim.env.patch_food_amount[0, 0] = np.float32(0.75)
    sim.env.patch_food_amount[0, 2] = np.float32(0.0)
    sim.env.patch_scent[0, 1] = np.float32(0.25)
    sim.env.patch_cover[0, 3] = np.float32(0.35)
    sim.env.patch_shelter[0, 4] = np.float32(0.45)
    sim.env.patch_movement_cost[0, 5] = np.float32(1.75)
    sim.env.patch_waste[0, 6] = np.float32(0.33)
    compute_local_density_map(sim.env)
    expected = {name: getattr(sim.env, name).copy() for name in PATCH_WORLD_ARRAY_NAMES}
    expected_visibility = patch_visibility_from_cover(sim.env_cfg, sim.env.patch_cover)
    expected_density = sim.env.local_density_map.copy()

    bundle = tmp_path / "bundle_enabled"
    checkpoint_bundle_v2_save(sim, str(bundle), generation_index=2, strict_replay_mode=True)
    payload = checkpoint_bundle_v2_load(str(bundle), strict=True, expected_commit_hash=None)

    patch_world = payload["world"]["patch_world"]
    assert patch_world["schema_version"] == 1
    assert set(patch_world["arrays"]) == set(PATCH_WORLD_ARRAY_NAMES)
    assert patch_world["arrays"]["patch_scent"][0][1] == pytest.approx(0.25)
    assert "local_density_map" not in patch_world["arrays"]

    restored = _make_sim(tmp_path / "checkpoint_enabled_dst", patch_world_enable=True, seed=444)
    checkpoint_bundle_v2_apply(restored, payload, restore_rng=True)

    for name in PATCH_WORLD_ARRAY_NAMES:
        assert np.array_equal(getattr(restored.env, name), expected[name])
    assert np.array_equal(patch_visibility_from_cover(restored.env_cfg, restored.env.patch_cover), expected_visibility)
    assert np.array_equal(restored.env.local_density_map, expected_density)
    assert restored.env.grid[0, 0] == FOOD
    assert restored.env.grid[0, 2] == EMPTY


def test_patch_world_checkpoint_round_trips_step7_event_contexts(tmp_path: Path) -> None:
    sim = _make_sim(tmp_path / "checkpoint_step7_context_src", patch_world_enable=True, seed=445)
    sim.run(start_gen=0, skip_init=False)
    sim._patch_birth_contexts_gen = [
        {"food_amount": 0.75, "cover": 0.25, "lineage_id": 3.0},
    ]
    sim._patch_death_contexts_gen = [
        {"waste": 0.5, "density": 2.0, "lineage_id": 4.0},
    ]
    sim._patch_regrowth_base_total_gen = 1.25
    sim._patch_regrowth_effective_total_gen = 0.75
    sim._patch_regrowth_suppressed_total_gen = 0.5
    sim._patch_regrowth_consequence_steps_gen = 2
    sim._patch_regrowth_multiplier_min_gen = 0.25

    bundle = tmp_path / "bundle_step7_contexts"
    checkpoint_bundle_v2_save(sim, str(bundle), generation_index=2, strict_replay_mode=True)
    payload = checkpoint_bundle_v2_load(str(bundle), strict=True, expected_commit_hash=None)

    assert payload["runner_state"]["_patch_birth_contexts_gen"] == sim._patch_birth_contexts_gen
    assert payload["runner_state"]["_patch_death_contexts_gen"] == sim._patch_death_contexts_gen
    assert payload["runner_state"]["_patch_regrowth_base_total_gen"] == pytest.approx(1.25)
    assert payload["runner_state"]["_patch_regrowth_effective_total_gen"] == pytest.approx(0.75)
    assert payload["runner_state"]["_patch_regrowth_suppressed_total_gen"] == pytest.approx(0.5)
    assert payload["runner_state"]["_patch_regrowth_consequence_steps_gen"] == 2

    restored = _make_sim(tmp_path / "checkpoint_step7_context_dst", patch_world_enable=True, seed=445)
    checkpoint_bundle_v2_apply(restored, payload, restore_rng=True)

    assert restored._patch_birth_contexts_gen == sim._patch_birth_contexts_gen
    assert restored._patch_death_contexts_gen == sim._patch_death_contexts_gen
    assert restored._patch_regrowth_base_total_gen == pytest.approx(1.25)
    assert restored._patch_regrowth_effective_total_gen == pytest.approx(0.75)
    assert restored._patch_regrowth_suppressed_total_gen == pytest.approx(0.5)
    assert restored._patch_regrowth_consequence_steps_gen == 2
    assert restored._patch_regrowth_multiplier_min_gen == pytest.approx(0.25)


def test_patch_world_checkpoint_omits_patch_payload_when_disabled(tmp_path: Path) -> None:
    sim = _make_sim(tmp_path / "checkpoint_disabled_src", patch_world_enable=False, seed=555)
    sim.run(start_gen=0, skip_init=False)

    bundle = tmp_path / "bundle_disabled"
    checkpoint_bundle_v2_save(sim, str(bundle), generation_index=2, strict_replay_mode=True)
    payload = checkpoint_bundle_v2_load(str(bundle), strict=True, expected_commit_hash=None)

    assert "patch_world" not in payload["world"]


def test_patch_world_enabled_restore_from_payload_without_patch_arrays_resets_arrays(tmp_path: Path) -> None:
    sim = _make_sim(tmp_path / "checkpoint_old_style_src", patch_world_enable=False, seed=666)
    sim.run(start_gen=0, skip_init=False)

    bundle = tmp_path / "bundle_old_style"
    checkpoint_bundle_v2_save(sim, str(bundle), generation_index=2, strict_replay_mode=True)
    payload = checkpoint_bundle_v2_load(str(bundle), strict=True, expected_commit_hash=None)
    assert "patch_world" not in payload["world"]

    restored = _make_sim(tmp_path / "checkpoint_old_style_dst", patch_world_enable=True, seed=666)
    checkpoint_bundle_v2_apply(restored, payload, restore_rng=True)

    for name in PATCH_WORLD_ARRAY_NAMES:
        arr = getattr(restored.env, name)
        assert arr.shape == (restored.env_cfg.height, restored.env_cfg.width)
        assert arr.dtype == np.float32


def test_patch_world_enabled_restore_from_step3_payload_defaults_new_terrain_arrays(tmp_path: Path) -> None:
    sim = _make_sim(tmp_path / "checkpoint_step3_src", patch_world_enable=True, seed=667)
    sim.run(start_gen=0, skip_init=False)
    sim.env.patch_cover[0, 0] = np.float32(0.8)

    bundle = tmp_path / "bundle_step3"
    checkpoint_bundle_v2_save(sim, str(bundle), generation_index=2, strict_replay_mode=True)
    payload = checkpoint_bundle_v2_load(str(bundle), strict=True, expected_commit_hash=None)
    arrays = payload["world"]["patch_world"]["arrays"]
    arrays.pop("patch_shelter", None)
    arrays.pop("patch_movement_cost", None)
    arrays.pop("patch_waste", None)

    restored = _make_sim(tmp_path / "checkpoint_step3_dst", patch_world_enable=True, seed=667)
    checkpoint_bundle_v2_apply(restored, payload, restore_rng=True)

    assert restored.env.patch_shelter.shape == restored.env.patch_cover.shape
    assert restored.env.patch_shelter.dtype == np.float32
    assert np.all(restored.env.patch_shelter == np.float32(restored.env_cfg.patch_shelter_default))
    assert np.all(restored.env.patch_movement_cost == np.float32(restored.env_cfg.patch_movement_cost_default))
    assert np.all(restored.env.patch_waste == np.float32(restored.env_cfg.patch_waste_initial))
    assert np.array_equal(
        patch_visibility_from_cover(restored.env_cfg, restored.env.patch_cover),
        patch_visibility_from_cover(sim.env_cfg, sim.env.patch_cover),
    )


def test_patch_world_checkpoint_resume_preserves_dynamic_food_continuation(tmp_path: Path) -> None:
    sim_a = _make_sim(tmp_path / "resume_src", patch_world_enable=True, seed=777)
    sim_a.run(start_gen=0, skip_init=False)

    bundle = tmp_path / "resume_bundle"
    checkpoint_bundle_v2_save(sim_a, str(bundle), generation_index=2, strict_replay_mode=True)
    payload = checkpoint_bundle_v2_load(str(bundle), strict=True, expected_commit_hash=None)

    sim_a.sim_cfg.generations = 3
    sim_a.run(start_gen=2, skip_init=True)

    sim_b = _make_sim(tmp_path / "resume_dst", patch_world_enable=True, seed=777)
    checkpoint_bundle_v2_apply(sim_b, payload, restore_rng=True)
    sim_b.sim_cfg.generations = 3
    sim_b.run(start_gen=2, skip_init=True)

    assert np.array_equal(sim_a.env.patch_food_amount, sim_b.env.patch_food_amount)
    assert np.array_equal(sim_a.env.patch_fertility, sim_b.env.patch_fertility)
    assert np.array_equal(sim_a.env.patch_cover, sim_b.env.patch_cover)
    assert np.array_equal(sim_a.env.patch_shelter, sim_b.env.patch_shelter)
    assert np.array_equal(sim_a.env.patch_movement_cost, sim_b.env.patch_movement_cost)
    assert np.array_equal(sim_a.env.patch_scent, sim_b.env.patch_scent)
    assert np.array_equal(sim_a.env.patch_waste, sim_b.env.patch_waste)
    assert np.array_equal(sim_a.env.local_density_map, sim_b.env.local_density_map)
    assert np.array_equal(
        patch_visibility_from_cover(sim_a.env_cfg, sim_a.env.patch_cover),
        patch_visibility_from_cover(sim_b.env_cfg, sim_b.env.patch_cover),
    )
    assert np.array_equal(sim_a.env.grid, sim_b.env.grid)


def test_patch_world_observation_enabled_checkpoint_resume_preserves_continuation(tmp_path: Path) -> None:
    cfg_a = _env_cfg(patch_world_enable=True)
    cfg_a.patch_world_observation_enable = True
    sim_a = SimulationRunner(
        env_cfg=cfg_a,
        pop_cfg=PopConfig(population_size=6),
        sim_cfg=_sim_cfg(seed=778),
        logger=NoOpMetricsLogger(),
        run_dir=str(tmp_path / "resume_obs_src"),
        perf=False,
    )
    sim_a.run(start_gen=0, skip_init=False)

    bundle = tmp_path / "resume_obs_bundle"
    checkpoint_bundle_v2_save(sim_a, str(bundle), generation_index=2, strict_replay_mode=True)
    payload = checkpoint_bundle_v2_load(str(bundle), strict=True, expected_commit_hash=None)

    sim_a.sim_cfg.generations = 3
    sim_a.run(start_gen=2, skip_init=True)

    cfg_b = _env_cfg(patch_world_enable=True)
    cfg_b.patch_world_observation_enable = True
    sim_b = SimulationRunner(
        env_cfg=cfg_b,
        pop_cfg=PopConfig(population_size=6),
        sim_cfg=_sim_cfg(seed=778),
        logger=NoOpMetricsLogger(),
        run_dir=str(tmp_path / "resume_obs_dst"),
        perf=False,
    )
    checkpoint_bundle_v2_apply(sim_b, payload, restore_rng=True)
    sim_b.sim_cfg.generations = 3
    sim_b.run(start_gen=2, skip_init=True)

    assert sim_a.obs_size == sim_b.obs_size == patch_world_total_observation_size(cfg_a)
    assert sim_a.pop.obs_size == sim_b.pop.obs_size == patch_world_total_observation_size(cfg_a)
    assert np.array_equal(sim_a.env.patch_food_amount, sim_b.env.patch_food_amount)
    assert np.array_equal(sim_a.env.patch_scent, sim_b.env.patch_scent)
    assert np.array_equal(sim_a.env.patch_waste, sim_b.env.patch_waste)
    assert np.array_equal(sim_a.env.local_density_map, sim_b.env.local_density_map)
    assert np.array_equal(sim_a.env.grid, sim_b.env.grid)


def test_patch_world_observation_enabled_rejects_old_observation_checkpoint(tmp_path: Path) -> None:
    sim = _make_sim(tmp_path / "obs_mismatch_src", patch_world_enable=True, seed=779)
    sim.run(start_gen=0, skip_init=False)

    bundle = tmp_path / "obs_mismatch_bundle"
    checkpoint_bundle_v2_save(sim, str(bundle), generation_index=2, strict_replay_mode=True)
    payload = checkpoint_bundle_v2_load(str(bundle), strict=True, expected_commit_hash=None)

    cfg = _env_cfg(patch_world_enable=True)
    cfg.patch_world_observation_enable = True
    restored = SimulationRunner(
        env_cfg=cfg,
        pop_cfg=PopConfig(population_size=6),
        sim_cfg=_sim_cfg(seed=779),
        logger=NoOpMetricsLogger(),
        run_dir=str(tmp_path / "obs_mismatch_dst"),
        perf=False,
    )

    with pytest.raises(RuntimeError, match="DET_OBSERVATION_SCHEMA_MISMATCH"):
        checkpoint_bundle_v2_apply(restored, payload, restore_rng=True)


def test_patch_world_metrics_columns_are_zero_when_disabled_and_active_when_enabled(tmp_path: Path) -> None:
    disabled_dir = tmp_path / "metrics_disabled"
    disabled_sim = SimulationRunner(
        env_cfg=_env_cfg(patch_world_enable=False),
        pop_cfg=PopConfig(population_size=6),
        sim_cfg=_sim_cfg(seed=888),
        logger=MetricsLogger(str(disabled_dir)),
        run_dir=str(disabled_dir),
        perf=False,
    )
    disabled_sim.run(start_gen=0, skip_init=False)

    enabled_dir = tmp_path / "metrics_enabled"
    enabled_sim = SimulationRunner(
        env_cfg=_env_cfg(patch_world_enable=True),
        pop_cfg=PopConfig(population_size=6),
        sim_cfg=_sim_cfg(seed=888),
        logger=MetricsLogger(str(enabled_dir)),
        run_dir=str(enabled_dir),
        perf=False,
    )
    enabled_sim.run(start_gen=0, skip_init=False)

    enabled_obs_dir = tmp_path / "metrics_enabled_obs"
    enabled_obs_cfg = _env_cfg(patch_world_enable=True)
    enabled_obs_cfg.patch_world_observation_enable = True
    enabled_obs_sim = SimulationRunner(
        env_cfg=enabled_obs_cfg,
        pop_cfg=PopConfig(population_size=6),
        sim_cfg=_sim_cfg(seed=888),
        logger=MetricsLogger(str(enabled_obs_dir)),
        run_dir=str(enabled_obs_dir),
        perf=False,
    )
    enabled_obs_sim.run(start_gen=0, skip_init=False)

    consequence_dir = tmp_path / "metrics_consequence_enabled"
    consequence_cfg = _env_cfg(patch_world_enable=True)
    consequence_cfg.patch_world_waste_consequence_enable = True
    consequence_cfg.patch_waste_regrowth_penalty_scale = 0.5
    consequence_cfg.patch_density_regrowth_penalty_scale = 0.5
    consequence_sim = SimulationRunner(
        env_cfg=consequence_cfg,
        pop_cfg=PopConfig(population_size=6),
        sim_cfg=_sim_cfg(seed=888),
        logger=MetricsLogger(str(consequence_dir)),
        run_dir=str(consequence_dir),
        perf=False,
    )
    consequence_sim.run(start_gen=0, skip_init=False)

    with (disabled_dir / "metrics.csv").open(newline="", encoding="utf-8") as f:
        disabled_rows = list(csv.DictReader(f))
    with (enabled_dir / "metrics.csv").open(newline="", encoding="utf-8") as f:
        enabled_rows = list(csv.DictReader(f))
    with (enabled_obs_dir / "metrics.csv").open(newline="", encoding="utf-8") as f:
        enabled_obs_rows = list(csv.DictReader(f))
    with (consequence_dir / "metrics.csv").open(newline="", encoding="utf-8") as f:
        consequence_rows = list(csv.DictReader(f))

    disabled_last = disabled_rows[-1]
    enabled_last = enabled_rows[-1]
    enabled_obs_last = enabled_obs_rows[-1]
    consequence_last = consequence_rows[-1]
    assert float(disabled_last["patch_food_amount_mean"]) == pytest.approx(0.0)
    assert float(disabled_last["patch_food_regrowth_total"]) == pytest.approx(0.0)
    assert float(disabled_last["patch_food_harvest_total"]) == pytest.approx(0.0)
    assert float(disabled_last["patch_cover_mean"]) == pytest.approx(0.0)
    assert float(disabled_last["patch_shelter_mean"]) == pytest.approx(0.0)
    assert float(disabled_last["patch_visibility_mean"]) == pytest.approx(0.0)
    assert float(disabled_last["patch_movement_cost_mean"]) == pytest.approx(0.0)
    assert float(disabled_last["patch_movement_energy_extra_total"]) == pytest.approx(0.0)
    assert float(disabled_last["patch_scent_mean"]) == pytest.approx(0.0)
    assert float(disabled_last["patch_scent_emitted_total"]) == pytest.approx(0.0)
    assert float(disabled_last["patch_waste_mean"]) == pytest.approx(0.0)
    assert float(disabled_last["patch_waste_emitted_total"]) == pytest.approx(0.0)
    assert float(disabled_last["patch_density_mean"]) == pytest.approx(0.0)
    assert int(disabled_last["patch_observation_enable"]) == 0
    assert int(disabled_last["patch_observation_channel_count"]) == 0
    assert int(disabled_last["patch_world_descriptor_schema_version"]) == 0
    assert int(disabled_last["patch_world_waste_consequence_enable"]) == 0
    assert float(disabled_last["patch_regrowth_suppressed_total"]) == pytest.approx(0.0)
    assert float(disabled_last["occupied_patch_food_amount_mean"]) == pytest.approx(0.0)
    assert float(disabled_last["birth_patch_count"]) == pytest.approx(0.0)
    assert float(disabled_last["death_patch_count"]) == pytest.approx(0.0)
    assert float(disabled_last["lineage_habitat_descriptor_count"]) == pytest.approx(0.0)
    assert disabled_last["lineage_habitat_descriptors_json"] == "[]"
    assert int(disabled_last["lineage_diversity_schema_version"]) == 0
    assert int(disabled_last["lineage_diversity_reporting_enable"]) == 0
    assert float(disabled_last["lineage_strategy_descriptor_count"]) == pytest.approx(0.0)
    assert disabled_last["lineage_strategy_descriptors_json"] == "[]"
    assert disabled_last["lineage_branch_candidates_json"] == "[]"
    assert float(enabled_last["patch_food_amount_mean"]) >= 0.0
    assert float(enabled_last["patch_food_regrowth_total"]) >= 0.0
    assert float(enabled_last["patch_fertility_mean"]) == pytest.approx(1.0)
    assert float(enabled_last["patch_cover_mean"]) == pytest.approx(0.0)
    assert float(enabled_last["patch_shelter_mean"]) == pytest.approx(0.0)
    assert float(enabled_last["patch_visibility_mean"]) == pytest.approx(1.0)
    assert float(enabled_last["patch_movement_cost_mean"]) == pytest.approx(1.0)
    assert float(enabled_last["patch_movement_energy_extra_total"]) == pytest.approx(0.0)
    assert float(enabled_last["patch_scent_mean"]) >= 0.0
    assert float(enabled_last["patch_scent_emitted_total"]) >= 0.0
    assert float(enabled_last["patch_waste_mean"]) >= 0.0
    assert float(enabled_last["patch_waste_emitted_total"]) >= 0.0
    assert float(enabled_last["patch_density_mean"]) >= 0.0
    assert int(enabled_last["patch_observation_enable"]) == 0
    assert int(enabled_last["patch_observation_schema_version"]) == PATCH_WORLD_OBSERVATION_SCHEMA_VERSION
    assert int(enabled_last["patch_observation_channel_count"]) == 1
    assert int(enabled_last["patch_observation_size"]) == (2 * enabled_sim.env_cfg.view_radius + 1) ** 2 + 2
    assert int(enabled_last["patch_world_descriptor_schema_version"]) == PATCH_WORLD_DESCRIPTOR_SCHEMA_VERSION
    assert int(enabled_last["patch_world_diagnostics_enable"]) == 1
    assert int(enabled_last["patch_world_lineage_descriptors_enable"]) == 1
    assert int(enabled_last["patch_world_waste_consequence_enable"]) == 0
    assert float(enabled_last["patch_regrowth_base_total"]) == pytest.approx(float(enabled_last["patch_regrowth_effective_total"]))
    assert float(enabled_last["patch_regrowth_suppressed_total"]) == pytest.approx(0.0)
    assert float(enabled_last["patch_regrowth_multiplier_min"]) == pytest.approx(1.0)
    assert float(enabled_last["occupied_patch_food_amount_mean"]) >= 0.0
    assert float(enabled_last["patch_food_occupancy_bias"]) == pytest.approx(
        float(enabled_last["occupied_patch_food_amount_mean"]) - float(enabled_last["patch_food_amount_mean"])
    )
    assert float(enabled_last["lineage_habitat_descriptor_count"]) >= 0.0
    assert enabled_last["lineage_habitat_descriptors_json"].startswith("[")
    assert int(enabled_last["lineage_diversity_schema_version"]) == LINEAGE_DIVERSITY_SCHEMA_VERSION
    assert int(enabled_last["lineage_diversity_reporting_enable"]) == 1
    assert int(enabled_last["lineage_strategy_descriptors_enable"]) == 1
    assert int(enabled_last["lineage_branch_candidate_reporting_enable"]) == 1
    assert float(enabled_last["lineage_active_count"]) >= 0.0
    assert float(enabled_last["lineage_alive_share_entropy"]) >= 0.0
    assert float(enabled_last["lineage_effective_alive_lineages"]) >= 0.0
    assert float(enabled_last["lineage_strategy_descriptor_count"]) >= 0.0
    assert enabled_last["lineage_strategy_descriptors_json"].startswith("[")
    assert enabled_last["lineage_branch_candidates_json"].startswith("[")
    json.loads(enabled_last["lineage_strategy_descriptors_json"])
    json.loads(enabled_last["lineage_branch_candidates_json"])
    assert int(enabled_obs_last["patch_observation_enable"]) == 1
    assert int(enabled_obs_last["patch_observation_schema_version"]) == PATCH_WORLD_OBSERVATION_SCHEMA_VERSION
    assert int(enabled_obs_last["patch_observation_channel_count"]) == len(PATCH_WORLD_OBSERVATION_CHANNEL_NAMES)
    assert int(enabled_obs_last["patch_observation_size"]) == patch_world_total_observation_size(enabled_obs_cfg)
    assert float(enabled_obs_last["patch_observation_nan_count"]) == pytest.approx(0.0)
    assert float(enabled_obs_last["patch_observation_min"]) >= -1.0
    assert float(enabled_obs_last["patch_observation_max"]) <= 1.0
    assert int(consequence_last["patch_world_waste_consequence_enable"]) == 1
    assert float(consequence_last["patch_regrowth_base_total"]) >= float(consequence_last["patch_regrowth_effective_total"])
    assert float(consequence_last["patch_regrowth_suppressed_total"]) >= 0.0
    assert float(consequence_last["patch_regrowth_multiplier_mean"]) <= 1.0
    assert float(consequence_last["patch_regrowth_multiplier_min"]) >= consequence_cfg.patch_regrowth_min_multiplier
    assert float(consequence_last["patch_waste_regrowth_penalty_max"]) >= 0.0
    assert float(consequence_last["patch_density_regrowth_penalty_max"]) >= 0.0
    assert float(consequence_last["patch_fertility_damage_total"]) == pytest.approx(0.0)
    assert float(consequence_last["patch_fertility_recovery_total"]) == pytest.approx(0.0)
