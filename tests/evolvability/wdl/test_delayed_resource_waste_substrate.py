from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from env.env_config import EnvConfig
from env.gridworld import GridWorld
from env.gridworld_pkg.constants import N_ACTIONS
from env.gridworld_pkg.patch_world import (
    apply_delayed_crowding_waste_step,
    apply_delayed_resource_depletion_step,
    delayed_crowding_waste_metrics,
    delayed_resource_depletion_metrics,
    patch_world_local_observation_channel_count,
)
from evolution.population_pkg.population_config import PopConfig
from simulation.checkpoint_bundle_v2 import checkpoint_bundle_v2_apply, checkpoint_bundle_v2_load, checkpoint_bundle_v2_save
from simulation.metrics_pkg.noop_loggers import NoOpMetricsLogger
from simulation.rng_manager import RNGManager
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig


def _env_cfg(*, family: str | None = None, observe: bool = False) -> EnvConfig:
    cfg = EnvConfig(width=6, height=6, view_radius=1)
    cfg.enable_biomes = False
    cfg.food_spawn_per_step = 0
    cfg.max_food = 0
    cfg.patch_world_enable = bool(family)
    cfg.patch_world_observation_enable = bool(observe)
    cfg.patch_food_amount_initial_empty_cell_value = 1.0
    cfg.patch_food_regrow_per_step = 0.02
    cfg.patch_waste_consume_emit = 0.0
    cfg.patch_waste_presence_emit = 0.0
    cfg.patch_waste_decay_per_step = 0.0
    if family == "resource":
        cfg.delayed_resource_depletion_enable = True
        cfg.delayed_resource_depletion_observation_enable = bool(observe)
        cfg.delayed_resource_depletion_delay_steps = 4
        cfg.delayed_resource_depletion_cue_duration_steps = 2
        cfg.delayed_resource_depletion_active_duration_steps = 3
        cfg.delayed_resource_depletion_event_period_steps = 12
        cfg.delayed_resource_depletion_radius = 0
        cfg.delayed_resource_depletion_food_depletion_fraction = 0.50
    elif family == "overuse":
        cfg.delayed_crowding_waste_enable = True
        cfg.delayed_crowding_waste_observation_enable = bool(observe)
        cfg.delayed_crowding_waste_delay_steps = 4
        cfg.delayed_crowding_waste_cue_duration_steps = 2
        cfg.delayed_crowding_waste_active_duration_steps = 3
        cfg.delayed_crowding_waste_event_period_steps = 12
        cfg.delayed_crowding_waste_radius = 0
        cfg.delayed_crowding_waste_min_waste_trigger = 0.01
        cfg.delayed_crowding_waste_waste_add = 0.25
    return cfg


def _sim_cfg(seed: int = 123) -> SimConfig:
    cfg = SimConfig(
        seed=int(seed),
        generations=1,
        steps_per_generation=8,
        device="cpu",
        bank_device="cpu",
        behavior_log_enable=False,
        policy_deterministic_actions=True,
        policy_tick_stride=1,
    )
    cfg.phase_e_e1_enable = False
    cfg.epi_enable = False
    cfg.checkpoint_bundle_v2_period_gen = 0
    cfg.enable_cognitive_trait_registry = True
    cfg.enable_cognitive_status_metrics = False
    cfg.enable_cognitive_budget_ledger = False
    cfg.disable_prediction_error_selection_pressure = True
    cfg.force_memory_disabled = False
    cfg.reset_memory_each_tick = False
    cfg.force_prediction_zero_lr = False
    cfg.reset_prediction_each_tick = False
    cfg.prediction_pressure_scenario = False
    cfg.memory_pressure_scenario = False
    return cfg


def _make_env(cfg: EnvConfig, seed: int = 123) -> GridWorld:
    rm = RNGManager(global_seed=int(seed))
    return GridWorld(cfg, seed=int(seed), run_dir=None, rng=rm.env_np)


def _make_sim(tmp_path: Path, cfg: EnvConfig, *, seed: int = 123) -> SimulationRunner:
    return SimulationRunner(
        env_cfg=cfg,
        pop_cfg=PopConfig(population_size=1),
        sim_cfg=_sim_cfg(seed=seed),
        logger=NoOpMetricsLogger(),
        run_dir=str(tmp_path),
        perf=False,
    )


def test_delayed_resource_waste_defaults_disabled_and_observation_shape_unchanged() -> None:
    cfg = EnvConfig(width=6, height=6, view_radius=1)
    assert cfg.delayed_resource_depletion_enable is False
    assert cfg.delayed_crowding_waste_enable is False
    assert patch_world_local_observation_channel_count(cfg) == 1


def test_delayed_resource_waste_does_not_change_primitive_action_count() -> None:
    assert N_ACTIONS == 7
    assert _sim_cfg().policy_deterministic_actions is True


def test_delayed_resource_waste_requires_patch_world() -> None:
    cfg = EnvConfig(width=6, height=6)
    cfg.delayed_resource_depletion_enable = True
    with pytest.raises(ValueError, match="delayed_resource_depletion_enable requires patch_world_enable"):
        _make_env(cfg)
    cfg = EnvConfig(width=6, height=6)
    cfg.delayed_crowding_waste_enable = True
    with pytest.raises(ValueError, match="delayed_crowding_waste_enable requires patch_world_enable"):
        _make_env(cfg)


def test_resource_depletion_cue_hidden_active_and_consequence() -> None:
    env = _make_env(_env_cfg(family="resource"), seed=101)
    before = env.patch_food_amount.copy()

    for step in range(5):
        apply_delayed_resource_depletion_step(env, step)

    assert int(np.sum(env.delayed_resource_depletion_cue > 0.0)) == 0
    assert int(np.sum(env.delayed_resource_depletion_active > 0.0)) == 1
    assert int(env.delayed_resource_depletion_consequence_count_last) == 1
    assert float(np.sum(before - env.patch_food_amount)) == pytest.approx(0.5)
    assert delayed_resource_depletion_metrics(env)["delayed_resource_depletion_consequence_count"] == 1.0


def test_overuse_penalty_requires_overuse_trigger_and_emits_consequence() -> None:
    env = _make_env(_env_cfg(family="overuse"), seed=101)

    for step in range(5):
        apply_delayed_crowding_waste_step(env, step)
    assert delayed_crowding_waste_metrics(env)["delayed_crowding_waste_event_count"] == 0.0
    assert delayed_crowding_waste_metrics(env)["delayed_crowding_waste_consequence_count"] == 0.0

    env = _make_env(_env_cfg(family="overuse"), seed=101)
    env.patch_waste[2, 2] = np.float32(0.25)
    before = env.patch_waste.copy()
    for step in range(5):
        apply_delayed_crowding_waste_step(env, step)

    assert int(np.sum(env.delayed_crowding_waste_active > 0.0)) == 1
    assert int(env.delayed_crowding_waste_consequence_count_last) == 1
    assert float(np.sum(env.patch_waste - before)) == pytest.approx(0.25)


def test_family_disabled_metrics_are_zero() -> None:
    env = _make_env(EnvConfig(width=6, height=6, view_radius=1), seed=101)
    assert all(float(v) == 0.0 for v in delayed_resource_depletion_metrics(env).values())
    assert all(float(v) == 0.0 for v in delayed_crowding_waste_metrics(env).values())


def test_family_observation_shape_changes_only_with_observation_flag() -> None:
    resource_hidden = _env_cfg(family="resource", observe=False)
    resource_visible = _env_cfg(family="resource", observe=True)
    overuse_hidden = _env_cfg(family="overuse", observe=False)
    overuse_visible = _env_cfg(family="overuse", observe=True)
    assert patch_world_local_observation_channel_count(resource_hidden) == 1
    assert patch_world_local_observation_channel_count(overuse_hidden) == 1
    assert patch_world_local_observation_channel_count(resource_visible) == 9
    assert patch_world_local_observation_channel_count(overuse_visible) == 9


def test_family_schedule_is_deterministic_and_seed_varies() -> None:
    a = _make_env(_env_cfg(family="resource"), seed=101)
    b = _make_env(_env_cfg(family="resource"), seed=101)
    c = _make_env(_env_cfg(family="resource"), seed=202)

    apply_delayed_resource_depletion_step(a, 0)
    apply_delayed_resource_depletion_step(b, 0)
    apply_delayed_resource_depletion_step(c, 0)

    assert np.array_equal(a.delayed_resource_depletion_event_id, b.delayed_resource_depletion_event_id)
    assert np.array_equal(a.delayed_resource_depletion_cue, b.delayed_resource_depletion_cue)
    assert not np.array_equal(a.delayed_resource_depletion_event_id, c.delayed_resource_depletion_event_id)


def test_family_checkpoint_roundtrip_preserves_state(tmp_path: Path) -> None:
    cfg = _env_cfg(family="resource")
    sim = _make_sim(tmp_path, cfg, seed=101)
    apply_delayed_resource_depletion_step(sim.env, 0)
    apply_delayed_resource_depletion_step(sim.env, 1)
    bundle = tmp_path / "checkpoint_bundles_v2" / "bundle_gen000000"
    checkpoint_bundle_v2_save(sim, str(bundle), generation_index=0, strict_replay_mode=True)

    restored = _make_sim(tmp_path / "restored", cfg, seed=101)
    payload = checkpoint_bundle_v2_load(str(bundle), strict=True, expected_commit_hash=None)
    checkpoint_bundle_v2_apply(restored, payload, restore_rng=True)

    assert np.array_equal(sim.env.delayed_resource_depletion_timer, restored.env.delayed_resource_depletion_timer)
    assert np.array_equal(sim.env.delayed_resource_depletion_event_id, restored.env.delayed_resource_depletion_event_id)
    assert np.array_equal(sim.env.delayed_resource_depletion_cue, restored.env.delayed_resource_depletion_cue)


def test_no_reward_leakage_from_family_config() -> None:
    cfg = _sim_cfg()
    assert cfg.episode0_fitness_weight == 0.0
    assert cfg.episode0_pred_err_fitness_weight == 0.0
    assert cfg.disable_prediction_error_selection_pressure is True


def test_substrate_validation_config_has_no_memory_or_prediction_arm() -> None:
    cfg = _sim_cfg()
    assert cfg.force_memory_disabled is False
    assert cfg.reset_memory_each_tick is False
    assert cfg.force_prediction_zero_lr is False
    assert cfg.reset_prediction_each_tick is False
    assert cfg.prediction_pressure_scenario is False
    assert cfg.memory_pressure_scenario is False
