from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from env.env_config import EnvConfig
from env.gridworld import GridWorld
from env.gridworld_pkg.patch_world import (
    apply_wdl_delayed_hazard_step,
    patch_world_local_observation_channel_count,
    patch_world_local_observation_size,
    patch_world_normalized_observation_maps,
    wdl_delayed_hazard_enabled,
)
from evolution.population_pkg.population_config import PopConfig
from simulation.checkpoint_bundle_v2 import checkpoint_bundle_v2_apply, checkpoint_bundle_v2_load, checkpoint_bundle_v2_save
from simulation.metrics_pkg.noop_loggers import NoOpMetricsLogger
from simulation.rng_manager import RNGManager
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig


def _env_cfg(*, wdl: bool = False, observe: bool = False) -> EnvConfig:
    cfg = EnvConfig(width=6, height=6, view_radius=1)
    cfg.enable_biomes = False
    cfg.food_spawn_per_step = 0
    cfg.max_food = 0
    cfg.patch_world_enable = bool(wdl)
    cfg.patch_world_observation_enable = bool(observe)
    cfg.wdl_delayed_hazard_enable = bool(wdl)
    cfg.wdl_delayed_hazard_observation_enable = bool(observe)
    cfg.wdl_delayed_hazard_delay_steps = 4
    cfg.wdl_delayed_hazard_cue_duration_steps = 2
    cfg.wdl_delayed_hazard_active_duration_steps = 3
    cfg.wdl_delayed_hazard_event_period_steps = 12
    cfg.wdl_delayed_hazard_radius = 0
    cfg.wdl_delayed_hazard_intensity = 0.25
    return cfg


def _sim_cfg(seed: int = 123, *, steps: int = 8) -> SimConfig:
    cfg = SimConfig(
        seed=int(seed),
        generations=1,
        steps_per_generation=int(steps),
        device="cpu",
        bank_device="cpu",
        behavior_log_enable=False,
        policy_deterministic_actions=True,
        policy_tick_stride=1,
    )
    cfg.phase_e_e1_enable = False
    cfg.epi_enable = False
    cfg.checkpoint_bundle_v2_period_gen = 0
    cfg.enable_cognitive_trait_registry = False
    cfg.enable_cognitive_status_metrics = False
    cfg.enable_cognitive_budget_ledger = False
    return cfg


def _make_env(cfg: EnvConfig, seed: int = 123) -> GridWorld:
    rm = RNGManager(global_seed=int(seed))
    return GridWorld(cfg, seed=int(seed), run_dir=None, rng=rm.env_np)


def _make_sim(tmp_path: Path, cfg: EnvConfig, *, seed: int = 123, pop: int = 1) -> SimulationRunner:
    sim = SimulationRunner(
        env_cfg=cfg,
        pop_cfg=PopConfig(population_size=int(pop)),
        sim_cfg=_sim_cfg(seed=seed),
        logger=NoOpMetricsLogger(),
        run_dir=str(tmp_path),
        perf=False,
    )
    sim.env._configure_world_slot_capacity(int(sim.pop.max_slots))
    return sim


def test_wdl_defaults_disabled_and_observation_shape_unchanged() -> None:
    cfg = EnvConfig(width=6, height=6, view_radius=1)
    assert cfg.wdl_delayed_hazard_enable is False
    assert cfg.wdl_delayed_hazard_observation_enable is False
    assert wdl_delayed_hazard_enabled(cfg) is False
    assert patch_world_local_observation_channel_count(cfg) == 1
    assert patch_world_local_observation_size(cfg) == 9


def test_wdl_requires_patch_world() -> None:
    cfg = EnvConfig(width=6, height=6)
    cfg.wdl_delayed_hazard_enable = True
    with pytest.raises(ValueError, match="requires patch_world_enable"):
        _make_env(cfg)


def test_wdl_deterministic_schedule_and_seed_variation() -> None:
    cfg = _env_cfg(wdl=True)
    a = _make_env(cfg, seed=101)
    b = _make_env(cfg, seed=101)
    c = _make_env(cfg, seed=202)

    apply_wdl_delayed_hazard_step(a, 0)
    apply_wdl_delayed_hazard_step(b, 0)
    apply_wdl_delayed_hazard_step(c, 0)

    assert np.array_equal(a.patch_delayed_hazard_event_id, b.patch_delayed_hazard_event_id)
    assert np.array_equal(a.patch_delayed_hazard_cue, b.patch_delayed_hazard_cue)
    assert not np.array_equal(a.patch_delayed_hazard_event_id, c.patch_delayed_hazard_event_id)


def test_wdl_cue_before_hidden_gap_before_active() -> None:
    cfg = _env_cfg(wdl=True)
    env = _make_env(cfg, seed=101)

    apply_wdl_delayed_hazard_step(env, 0)
    assert int(np.sum(env.patch_delayed_hazard_cue > 0.0)) == 1
    assert int(np.sum(env.patch_delayed_hazard_active > 0.0)) == 0

    apply_wdl_delayed_hazard_step(env, 1)
    assert int(np.sum(env.patch_delayed_hazard_cue > 0.0)) == 1

    apply_wdl_delayed_hazard_step(env, 2)
    assert int(np.sum(env.patch_delayed_hazard_cue > 0.0)) == 0
    assert int(np.sum(env.patch_delayed_hazard_active > 0.0)) == 0

    apply_wdl_delayed_hazard_step(env, 3)
    assert int(np.sum(env.patch_delayed_hazard_active > 0.0)) == 0

    apply_wdl_delayed_hazard_step(env, 4)
    assert int(np.sum(env.patch_delayed_hazard_active > 0.0)) == 1
    assert int(env.wdl_delayed_hazard_active_without_prior_cue_count_last) == 0


def test_wdl_observation_gating() -> None:
    disabled_obs = _env_cfg(wdl=True, observe=False)
    enabled_obs = _env_cfg(wdl=True, observe=True)
    assert patch_world_local_observation_channel_count(disabled_obs) == 1
    assert patch_world_local_observation_channel_count(enabled_obs) == 9
    assert patch_world_local_observation_size(enabled_obs) == 81


def test_wdl_observation_dropout_default_preserves_cue_channel() -> None:
    cfg_default = _env_cfg(wdl=True, observe=True)
    cfg_explicit_zero = _env_cfg(wdl=True, observe=True)
    cfg_explicit_zero.wdl_delayed_hazard_observation_dropout_rate = 0.0
    env_default = _make_env(cfg_default, seed=101)
    env_explicit_zero = _make_env(cfg_explicit_zero, seed=101)

    apply_wdl_delayed_hazard_step(env_default, 0)
    apply_wdl_delayed_hazard_step(env_explicit_zero, 0)

    assert np.array_equal(
        patch_world_normalized_observation_maps(env_default)[-1],
        patch_world_normalized_observation_maps(env_explicit_zero)[-1],
    )


def test_wdl_observation_dropout_masks_cue_without_changing_state() -> None:
    cfg = _env_cfg(wdl=True, observe=True)
    cfg.wdl_delayed_hazard_observation_dropout_rate = 1.0
    env = _make_env(cfg, seed=101)

    apply_wdl_delayed_hazard_step(env, 0)

    assert int(np.sum(env.patch_delayed_hazard_cue > 0.0)) == 1
    assert float(np.sum(patch_world_normalized_observation_maps(env)[-1])) == pytest.approx(0.0)


def test_wdl_damage_is_bounded_and_not_universal(tmp_path: Path) -> None:
    cfg = _env_cfg(wdl=True)
    sim = _make_sim(tmp_path, cfg, seed=101, pop=1)
    sim.pop.init_random()
    agent = sim.pop.alive_agents()[0]
    sim.env.reset()
    sim.env.add_agent(
        int(agent.agent_id),
        (2, 2),
        slot=int(agent.slot),
        slot_token=int(agent.slot_token),
    )
    sim.env.patch_delayed_hazard_active.fill(0.0)
    sim.env.patch_delayed_hazard_intensity.fill(0.0)
    sim.env.patch_delayed_hazard_active[2, 2] = 1.0
    sim.env.patch_delayed_hazard_intensity[2, 2] = 0.25
    before = float(agent.health)

    deaths = sim._apply_wdl_delayed_hazard_damage(0)

    assert deaths == 0
    assert agent.alive is True
    assert agent.health == pytest.approx(before - 0.25)
    assert np.isfinite(float(agent.health))
    assert sim._wdl_delayed_hazard_damage_total_gen == pytest.approx(0.25)


def test_wdl_checkpoint_roundtrip_preserves_state(tmp_path: Path) -> None:
    cfg = _env_cfg(wdl=True)
    sim = _make_sim(tmp_path, cfg, seed=101, pop=1)
    apply_wdl_delayed_hazard_step(sim.env, 0)
    apply_wdl_delayed_hazard_step(sim.env, 1)
    bundle = tmp_path / "checkpoint_bundles_v2" / "bundle_gen000000"
    checkpoint_bundle_v2_save(sim, str(bundle), generation_index=0, strict_replay_mode=True)

    restored = _make_sim(tmp_path / "restored", cfg, seed=101, pop=1)
    payload = checkpoint_bundle_v2_load(str(bundle), strict=True, expected_commit_hash=None)
    checkpoint_bundle_v2_apply(restored, payload, restore_rng=True)

    assert np.array_equal(sim.env.patch_delayed_hazard_timer, restored.env.patch_delayed_hazard_timer)
    assert np.array_equal(sim.env.patch_delayed_hazard_event_id, restored.env.patch_delayed_hazard_event_id)
    assert np.array_equal(sim.env.patch_delayed_hazard_cue, restored.env.patch_delayed_hazard_cue)


def test_wdl_cue_enable_false_suppresses_state_cue_without_changing_schedule() -> None:
    cfg = _env_cfg(wdl=True, observe=True)
    cfg.wdl_delayed_hazard_cue_enable = False
    env = _make_env(cfg, seed=101)

    apply_wdl_delayed_hazard_step(env, 0)

    assert int(np.sum(env.patch_delayed_hazard_cue > 0.0)) == 0
    assert int(np.count_nonzero(env.patch_delayed_hazard_event_id)) == 1
    assert int(np.sum(env.patch_delayed_hazard_prior_cue > 0)) == 1
    assert float(np.sum(patch_world_normalized_observation_maps(env)[-1])) == 0.0


def test_wdl_active_phase_honors_configured_duration_and_cleans_up() -> None:
    cfg = _env_cfg(wdl=True)
    cfg.wdl_delayed_hazard_active_duration_steps = 3
    env = _make_env(cfg, seed=101)

    active_counts: list[int] = []
    for step in range(8):
        apply_wdl_delayed_hazard_step(env, step)
        active_counts.append(int(np.sum(env.patch_delayed_hazard_active > 0.0)))

    assert active_counts == [0, 0, 0, 0, 1, 1, 1, 0]
    assert int(np.count_nonzero(env.patch_delayed_hazard_event_id)) == 0
    assert int(np.sum(env.patch_delayed_hazard_prior_cue > 0)) == 0
    assert int(np.sum(env.patch_delayed_hazard_timer >= 0)) == 0
