from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from env.env_config import EnvConfig
from env.gridworld import GridWorld
from env.gridworld_pkg.patch_world import (
    apply_mem_max_delayed_choice_step,
    mem_max_delayed_choice_metrics,
    mem_max_delayed_choice_observation_audit,
    patch_world_normalized_observation_maps,
    patch_world_local_observation_channel_count,
)
from evolution.population_pkg.population_config import PopConfig
from simulation.checkpoint_bundle_v2 import checkpoint_bundle_v2_apply, checkpoint_bundle_v2_load, checkpoint_bundle_v2_save
from simulation.metrics_pkg.noop_loggers import NoOpMetricsLogger
from simulation.rng_manager import RNGManager
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig


def _env_cfg(*, observe: bool = True, navigation: bool = False, global_cue: bool = False) -> EnvConfig:
    cfg = EnvConfig(width=7, height=7, view_radius=1)
    cfg.enable_biomes = False
    cfg.food_spawn_per_step = 0
    cfg.max_food = 0
    cfg.patch_world_enable = True
    cfg.patch_world_observation_enable = bool(observe)
    cfg.patch_food_amount_initial_empty_cell_value = 1.0
    cfg.patch_food_regrow_per_step = 0.0
    cfg.patch_waste_consume_emit = 0.0
    cfg.patch_waste_presence_emit = 0.0
    cfg.patch_waste_decay_per_step = 0.0
    cfg.mem_max_delayed_choice_enable = True
    cfg.mem_max_delayed_choice_observation_enable = bool(observe)
    cfg.mem_max_delayed_choice_observation_schema_version = 2 if navigation else 1
    cfg.mem_max_delayed_choice_global_cue_observation_enable = bool(global_cue)
    cfg.mem_max_delayed_choice_navigation_observation_enable = bool(navigation)
    cfg.mem_max_delayed_choice_cue_steps = 1
    cfg.mem_max_delayed_choice_hidden_steps = 2
    cfg.mem_max_delayed_choice_active_steps = 2
    cfg.mem_max_delayed_choice_event_period_steps = 10
    cfg.mem_max_delayed_choice_target_distance = 2
    cfg.mem_max_delayed_choice_damage = 0.25
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


def _make_sim(tmp_path: Path, cfg: EnvConfig, *, seed: int = 123, pop: int = 2) -> SimulationRunner:
    return SimulationRunner(
        env_cfg=cfg,
        pop_cfg=PopConfig(population_size=int(pop)),
        sim_cfg=_sim_cfg(seed=seed),
        logger=NoOpMetricsLogger(),
        run_dir=str(tmp_path),
        perf=False,
    )


def _advance_mem_max_to_active(sim: SimulationRunner) -> None:
    if not sim.pop.agents:
        sim.pop.init_random()
        sim._place_agents()
    for step in range(4):
        sim.env.patch_world_step(step)
    assert int(np.sum(sim.env.mem_max_delayed_choice_active)) == 2


def _target_coords(sim: SimulationRunner) -> tuple[tuple[int, int], tuple[int, int], int]:
    target_a = np.argwhere(sim.env.mem_max_delayed_choice_target_a > 0.0)
    target_b = np.argwhere(sim.env.mem_max_delayed_choice_target_b > 0.0)
    assert target_a.size > 0
    assert target_b.size > 0
    active = sim.env.mem_max_delayed_choice_active > 0.0
    correct = int(np.max(sim.env.mem_max_delayed_choice_correct_target[active]))
    return tuple(int(v) for v in target_a[0].tolist()), tuple(int(v) for v in target_b[0].tolist()), correct


def _move_agent(sim: SimulationRunner, aid: int, y: int, x: int) -> None:
    agent = sim.pop.agents[int(aid)]
    sim.env.move_agent(
        int(aid),
        (int(y), int(x)),
        slot=int(agent.slot),
        slot_token=int(agent.slot_token),
    )
    agent.alive = True


def test_mem_max_delayed_choice_defaults_disabled_and_shape_unchanged() -> None:
    cfg = EnvConfig(width=7, height=7, view_radius=1)
    assert cfg.mem_max_delayed_choice_enable is False
    assert cfg.mem_max_delayed_choice_observation_enable is False
    assert patch_world_local_observation_channel_count(cfg) == 1


def test_mem_max_delayed_choice_observation_shape_is_explicitly_gated() -> None:
    hidden = _env_cfg(observe=False)
    visible = _env_cfg(observe=True)
    navigable = _env_cfg(observe=True, navigation=True)
    assert patch_world_local_observation_channel_count(hidden) == 1
    assert patch_world_local_observation_channel_count(visible) == 10
    assert patch_world_local_observation_channel_count(navigable) == 18


def test_mem_max_delayed_choice_navigation_observation_requires_schema_v2() -> None:
    cfg = _env_cfg(observe=True, navigation=True)
    cfg.mem_max_delayed_choice_observation_schema_version = 1
    with pytest.raises(ValueError, match="navigation_observation_enable requires"):
        patch_world_local_observation_channel_count(cfg)


def test_mem_max_delayed_choice_navigation_observation_exposes_symmetric_target_vectors(tmp_path: Path) -> None:
    cfg = _env_cfg(observe=True, navigation=True)
    sim = _make_sim(tmp_path / "navigation_obs", cfg, seed=101, pop=1)
    _advance_mem_max_to_active(sim)
    aid = int(next(iter(sim.pop.agents)))
    _move_agent(sim, aid, 3, 3)
    target_a, target_b, correct = _target_coords(sim)

    obs = sim.env.get_local_observation(aid)
    channel_count = patch_world_local_observation_channel_count(cfg)
    k = 2 * int(cfg.view_radius) + 1
    center = (k * k) // 2
    channels = obs.reshape(channel_count, k * k)
    nav_start = 10

    assert target_a == (3, 1)
    assert target_b == (3, 5)
    assert correct in (1, 2)
    assert channels[nav_start + 0, center] == pytest.approx(0.0)
    assert channels[nav_start + 1, center] == pytest.approx(-2.0 / 6.0)
    assert channels[nav_start + 2, center] == pytest.approx(2.0 / 12.0)
    assert channels[nav_start + 3, center] == pytest.approx(0.0)
    assert channels[nav_start + 4, center] == pytest.approx(0.0)
    assert channels[nav_start + 5, center] == pytest.approx(2.0 / 6.0)
    assert channels[nav_start + 6, center] == pytest.approx(2.0 / 12.0)
    assert channels[nav_start + 7, center] == pytest.approx(0.0)


def test_mem_max_delayed_choice_global_cue_reaches_center_only_during_cue_phase(tmp_path: Path) -> None:
    cfg = _env_cfg(observe=True, navigation=True, global_cue=True)
    sim = _make_sim(tmp_path / "global_cue_obs", cfg, seed=101, pop=1)
    if not sim.pop.agents:
        sim.pop.init_random()
    agent = next(iter(sim.pop.agents.values()))
    aid = int(agent.agent_id)
    sim.env.reset()
    sim.env.add_agent(
        aid, (3, 3), slot=int(agent.slot), slot_token=int(agent.slot_token)
    )
    apply_mem_max_delayed_choice_step(sim.env, 0)

    obs = sim.env.get_local_observation(aid)
    channel_count = patch_world_local_observation_channel_count(cfg)
    k = 2 * int(cfg.view_radius) + 1
    center = (k * k) // 2
    channels = obs.reshape(channel_count, k * k)
    assert float(channels[8, center] + channels[9, center]) == pytest.approx(1.0)

    apply_mem_max_delayed_choice_step(sim.env, 1)
    hidden_obs = sim.env.get_local_observation(aid)
    hidden_channels = hidden_obs.reshape(channel_count, k * k)
    assert float(hidden_channels[8, center] + hidden_channels[9, center]) == pytest.approx(0.0)


def test_mem_max_delayed_choice_navigation_observation_does_not_expose_hidden_correctness() -> None:
    env = _make_env(_env_cfg(observe=True, navigation=True), seed=101)
    flipped_cfg = _env_cfg(observe=True, navigation=True)
    flipped_cfg.mem_max_delayed_choice_random_incorrect_cue_enable = True
    flipped = _make_env(flipped_cfg, seed=101)
    apply_mem_max_delayed_choice_step(env, 0)
    apply_mem_max_delayed_choice_step(env, 1)
    apply_mem_max_delayed_choice_step(flipped, 0)
    apply_mem_max_delayed_choice_step(flipped, 1)
    assert mem_max_delayed_choice_observation_audit(env)["phase"] == "hidden"
    assert mem_max_delayed_choice_observation_audit(flipped)["phase"] == "hidden"
    assert not np.array_equal(env.mem_max_delayed_choice_correct_target, flipped.mem_max_delayed_choice_correct_target)
    maps = patch_world_normalized_observation_maps(env)
    flipped_maps = patch_world_normalized_observation_maps(flipped)
    for nav_map, flipped_nav_map in zip(maps[-8:], flipped_maps[-8:]):
        assert np.array_equal(nav_map, flipped_nav_map)
    assert float(np.sum(maps[7] + maps[8])) == pytest.approx(0.0)
    assert float(np.sum(flipped_maps[7] + flipped_maps[8])) == pytest.approx(0.0)


def test_mem_max_delayed_choice_cue_hidden_active_and_audit() -> None:
    env = _make_env(_env_cfg(), seed=101)

    apply_mem_max_delayed_choice_step(env, 0)
    assert int(env.mem_max_delayed_choice_event_count_last) == 1
    assert int(np.sum(env.mem_max_delayed_choice_target_a + env.mem_max_delayed_choice_target_b)) == 2
    assert int(np.sum(env.mem_max_delayed_choice_cue_a + env.mem_max_delayed_choice_cue_b)) == 1

    apply_mem_max_delayed_choice_step(env, 1)
    assert int(np.sum(env.mem_max_delayed_choice_cue_a + env.mem_max_delayed_choice_cue_b)) == 0
    hidden_audit = mem_max_delayed_choice_observation_audit(env)
    assert hidden_audit["phase"] == "hidden"
    assert hidden_audit["hidden_window_direct_cue_exposed"] is False
    assert hidden_audit["passes"] is True

    apply_mem_max_delayed_choice_step(env, 2)
    apply_mem_max_delayed_choice_step(env, 3)
    assert int(np.sum(env.mem_max_delayed_choice_active)) == 2
    assert mem_max_delayed_choice_metrics(env)["mem_max_delayed_choice_behavior_metric_available"] == 1


def test_mem_max_delayed_choice_checkpoint_roundtrip_preserves_target_identity(tmp_path: Path) -> None:
    cfg = _env_cfg()
    sim = _make_sim(tmp_path / "source", cfg, seed=101)
    sim.env.patch_world_step(0)
    sim.env.patch_world_step(1)
    bundle = tmp_path / "checkpoint_bundles_v2" / "bundle_gen000000"
    checkpoint_bundle_v2_save(sim, str(bundle), generation_index=0, strict_replay_mode=True)

    restored = _make_sim(tmp_path / "restored", cfg, seed=101)
    payload = checkpoint_bundle_v2_load(str(bundle), strict=True, expected_commit_hash=None)
    checkpoint_bundle_v2_apply(restored, payload, restore_rng=True)

    for name in (
        "mem_max_delayed_choice_target_a",
        "mem_max_delayed_choice_target_b",
        "mem_max_delayed_choice_timer",
        "mem_max_delayed_choice_event_id",
        "mem_max_delayed_choice_correct_target",
        "mem_max_delayed_choice_prior_cue",
    ):
        assert np.array_equal(getattr(sim.env, name), getattr(restored.env, name)), name


def test_mem_max_delayed_choice_rejects_incompatible_schema() -> None:
    cfg = _env_cfg()
    cfg.mem_max_delayed_choice_schema_version = 999
    with pytest.raises(ValueError, match="Unsupported MEM-MAX delayed-choice schema version"):
        _make_env(cfg)


def test_mem_max_delayed_choice_repeat_parity_same_seed() -> None:
    a = _make_env(_env_cfg(), seed=101)
    b = _make_env(_env_cfg(), seed=101)
    c = _make_env(_env_cfg(), seed=202)
    for step in range(4):
        apply_mem_max_delayed_choice_step(a, step)
        apply_mem_max_delayed_choice_step(b, step)
        apply_mem_max_delayed_choice_step(c, step)
    assert np.array_equal(a.mem_max_delayed_choice_correct_target, b.mem_max_delayed_choice_correct_target)
    assert np.array_equal(a.mem_max_delayed_choice_event_id, b.mem_max_delayed_choice_event_id)
    assert not np.array_equal(a.mem_max_delayed_choice_correct_target, c.mem_max_delayed_choice_correct_target)


def test_mem_max_delayed_choice_live_consequence_protects_correct_choice(tmp_path: Path) -> None:
    cfg = _env_cfg()
    cfg.mem_max_delayed_choice_health_damage = 0.25
    sim = _make_sim(tmp_path / "correct", cfg, seed=101, pop=1)
    _advance_mem_max_to_active(sim)
    target_a, target_b, correct = _target_coords(sim)
    aid = int(next(iter(sim.pop.agents)))
    y, x = target_a if correct == 1 else target_b
    _move_agent(sim, aid, y, x)
    health_before = float(sim.pop.agents[aid].health)

    deaths = sim._apply_mem_max_delayed_choice_consequence(0)

    assert deaths == 0
    assert float(sim.pop.agents[aid].health) == pytest.approx(health_before)
    assert int(getattr(sim, "_mem_max_delayed_choice_correct_arrival_count_gen", 0)) == 1
    assert float(getattr(sim, "_mem_max_delayed_choice_correct_health_damage_total_gen", 0.0)) == pytest.approx(0.0)
    assert float(getattr(sim, "_mem_max_delayed_choice_damage_prevented_total_gen", 0.0)) == pytest.approx(0.25)


def test_mem_max_delayed_choice_live_consequence_damages_no_choice_once(tmp_path: Path) -> None:
    cfg = _env_cfg()
    cfg.mem_max_delayed_choice_health_damage = 0.25
    sim = _make_sim(tmp_path / "no_choice", cfg, seed=101, pop=1)
    _advance_mem_max_to_active(sim)
    aid = int(next(iter(sim.pop.agents)))
    _move_agent(sim, aid, 0, 0)
    health_before = float(sim.pop.agents[aid].health)

    sim._apply_mem_max_delayed_choice_consequence(0)
    sim._apply_mem_max_delayed_choice_consequence(0)

    assert float(sim.pop.agents[aid].health) == pytest.approx(health_before - 0.25)
    assert int(getattr(sim, "_mem_max_delayed_choice_no_choice_count_gen", 0)) == 1
    assert int(getattr(sim, "_mem_max_delayed_choice_consequence_applied_count_gen", 0)) == 1
    assert int(getattr(sim, "_mem_max_delayed_choice_duplicate_prevented_count_gen", 0)) >= 1


def test_mem_max_delayed_choice_live_consequence_respects_consequence_off(tmp_path: Path) -> None:
    cfg = _env_cfg()
    cfg.mem_max_delayed_choice_consequence_enable = False
    cfg.mem_max_delayed_choice_health_damage = 0.25
    sim = _make_sim(tmp_path / "off", cfg, seed=101, pop=1)
    _advance_mem_max_to_active(sim)
    aid = int(next(iter(sim.pop.agents)))
    _move_agent(sim, aid, 0, 0)
    health_before = float(sim.pop.agents[aid].health)

    sim._apply_mem_max_delayed_choice_consequence(0)

    assert float(sim.pop.agents[aid].health) == pytest.approx(health_before)
    assert int(getattr(sim, "_mem_max_delayed_choice_consequence_applied_count_gen", 0)) == 0


def test_mem_max_delayed_choice_checkpoint_roundtrip_after_live_consequence(tmp_path: Path) -> None:
    cfg = _env_cfg()
    cfg.mem_max_delayed_choice_health_damage = 0.25
    sim = _make_sim(tmp_path / "source_after_consequence", cfg, seed=101, pop=1)
    _advance_mem_max_to_active(sim)
    aid = int(next(iter(sim.pop.agents)))
    _move_agent(sim, aid, 0, 0)
    sim._apply_mem_max_delayed_choice_consequence(0)

    bundle = tmp_path / "checkpoint_bundles_v2" / "bundle_gen000000_after_consequence"
    checkpoint_bundle_v2_save(sim, str(bundle), generation_index=0, strict_replay_mode=True)
    restored = _make_sim(tmp_path / "restored_after_consequence", cfg, seed=101, pop=1)
    payload = checkpoint_bundle_v2_load(str(bundle), strict=True, expected_commit_hash=None)
    checkpoint_bundle_v2_apply(restored, payload, restore_rng=True)

    assert float(restored.pop.agents[aid].health) == pytest.approx(float(sim.pop.agents[aid].health))
    assert np.array_equal(restored.env.agent_pos_y, sim.env.agent_pos_y)
    assert np.array_equal(restored.env.agent_pos_x, sim.env.agent_pos_x)
