from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from env.env_config import EnvConfig
from env.gridworld import GridWorld
from env.gridworld_pkg.patch_world import (
    apply_prediction_demand_step,
    patch_world_local_observation_channel_count,
    patch_world_normalized_observation_maps,
    prediction_demand_enabled,
    prediction_demand_metrics,
)
from evolution.population_pkg.population_config import PopConfig
from simulation.checkpoint_bundle_v2 import checkpoint_bundle_v2_apply, checkpoint_bundle_v2_load, checkpoint_bundle_v2_save
from simulation.metrics_pkg.metrics_logger import MetricsLogger
from simulation.metrics_pkg.noop_loggers import NoOpMetricsLogger
from simulation.rng_manager import RNGManager
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig


def _env_cfg(*, enabled: bool = False, observe: bool = False) -> EnvConfig:
    cfg = EnvConfig(width=6, height=6, view_radius=1)
    cfg.enable_biomes = False
    cfg.food_spawn_per_step = 0
    cfg.max_food = 0
    cfg.patch_world_enable = bool(enabled)
    cfg.patch_world_observation_enable = bool(observe)
    cfg.prediction_demand_enable = bool(enabled)
    cfg.prediction_demand_observation_enable = bool(observe)
    cfg.prediction_demand_delay_min = 3
    cfg.prediction_demand_delay_max = 7
    cfg.prediction_demand_active_duration = 3
    cfg.prediction_demand_event_period_steps = 14
    cfg.prediction_demand_radius = 0
    cfg.prediction_demand_transition_strength = 0.3
    cfg.prediction_demand_cue_noise = 0.35
    return cfg


def _sim_cfg(seed: int = 123, *, steps: int = 18) -> SimConfig:
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


def _make_sim(tmp_path: Path, cfg: EnvConfig, *, seed: int = 123, pop: int = 2) -> SimulationRunner:
    return SimulationRunner(
        env_cfg=cfg,
        pop_cfg=PopConfig(population_size=int(pop)),
        sim_cfg=_sim_cfg(seed=seed),
        logger=NoOpMetricsLogger(),
        run_dir=str(tmp_path),
        perf=False,
    )


def test_prediction_demand_defaults_disabled_and_zero_metrics() -> None:
    cfg = EnvConfig(width=6, height=6, view_radius=1)
    assert cfg.prediction_demand_enable is False
    assert cfg.prediction_demand_observation_enable is False
    assert prediction_demand_enabled(cfg) is False
    assert patch_world_local_observation_channel_count(cfg) == 1

    env = _make_env(cfg, seed=101)
    metrics = prediction_demand_metrics(env)

    assert metrics["prediction_demand_enable"] == 0
    assert metrics["prediction_demand_cue_cell_count"] == 0.0
    assert metrics["prediction_demand_active_cell_count"] == 0.0
    assert metrics["prediction_demand_direct_prediction_reward_active"] == 0
    assert metrics["prediction_demand_direct_error_reward_active"] == 0


def test_prediction_demand_requires_patch_world() -> None:
    cfg = EnvConfig(width=6, height=6)
    cfg.prediction_demand_enable = True
    with pytest.raises(ValueError, match="requires patch_world_enable"):
        _make_env(cfg)


def test_prediction_demand_observation_gating() -> None:
    disabled_obs = _env_cfg(enabled=True, observe=False)
    enabled_obs = _env_cfg(enabled=True, observe=True)
    assert patch_world_local_observation_channel_count(disabled_obs) == 1
    assert patch_world_local_observation_channel_count(enabled_obs) == 10

    env = _make_env(enabled_obs, seed=101)
    apply_prediction_demand_step(env, 0)
    maps = patch_world_normalized_observation_maps(env)
    assert len(maps) == 9
    assert float(np.sum(maps[-2])) > 0.0
    assert 0.0 <= float(np.min(maps[-1])) <= float(np.max(maps[-1])) <= 1.0


def test_prediction_demand_emits_variable_delay_and_ambiguous_cues() -> None:
    cfg = _env_cfg(enabled=True, observe=True)
    env = _make_env(cfg, seed=101)
    delays: list[int] = []
    ambiguity_values: list[float] = []

    for step in range(0, int(cfg.prediction_demand_event_period_steps) * 8):
        metrics = apply_prediction_demand_step(env, step)
        if int(metrics["prediction_demand_event_count"]) > 0:
            active_delays = env.patch_prediction_demand_delay[env.patch_prediction_demand_cue > 0.0]
            delays.extend(int(v) for v in active_delays.tolist())
            ambiguity_values.append(float(metrics["prediction_demand_cue_ambiguity_mean"]))

    assert delays
    assert min(delays) >= int(cfg.prediction_demand_delay_min)
    assert max(delays) <= int(cfg.prediction_demand_delay_max)
    assert len(set(delays)) > 1
    assert any(0.0 < value < 1.0 for value in ambiguity_values)


def test_prediction_demand_consequence_metrics_and_reward_audit_fields() -> None:
    cfg = _env_cfg(enabled=True, observe=True)
    env = _make_env(cfg, seed=101)
    first_active = None

    for step in range(0, int(cfg.prediction_demand_event_period_steps)):
        metrics = apply_prediction_demand_step(env, step)
        if float(metrics["prediction_demand_active_cell_count"]) > 0.0:
            first_active = metrics
            break

    assert first_active is not None
    assert first_active["prediction_demand_transition_count"] > 0.0
    assert first_active["prediction_demand_cue_to_consequence_delay_mean"] >= float(cfg.prediction_demand_delay_min)
    assert first_active["prediction_demand_consequence_without_prior_cue_count"] == 0.0
    assert first_active["prediction_demand_direct_prediction_reward_active"] == 0
    assert first_active["prediction_demand_direct_error_reward_active"] == 0


def test_prediction_demand_checkpoint_roundtrip_preserves_state(tmp_path: Path) -> None:
    cfg = _env_cfg(enabled=True, observe=True)
    sim = _make_sim(tmp_path, cfg, seed=101, pop=1)
    apply_prediction_demand_step(sim.env, 0)
    apply_prediction_demand_step(sim.env, 1)
    bundle = tmp_path / "checkpoint_bundles_v2" / "bundle_gen000000"
    checkpoint_bundle_v2_save(sim, str(bundle), generation_index=0, strict_replay_mode=True)

    restored = _make_sim(tmp_path / "restored", cfg, seed=101, pop=1)
    payload = checkpoint_bundle_v2_load(str(bundle), strict=True, expected_commit_hash=None)
    checkpoint_bundle_v2_apply(restored, payload, restore_rng=True)

    assert np.array_equal(sim.env.patch_prediction_demand_timer, restored.env.patch_prediction_demand_timer)
    assert np.array_equal(sim.env.patch_prediction_demand_event_id, restored.env.patch_prediction_demand_event_id)
    assert np.array_equal(sim.env.patch_prediction_demand_cue, restored.env.patch_prediction_demand_cue)
    assert np.array_equal(sim.env.patch_prediction_demand_transition_sign, restored.env.patch_prediction_demand_transition_sign)


def test_prediction_demand_runner_metrics_are_default_off_or_enabled(tmp_path: Path) -> None:
    disabled = _make_sim(tmp_path / "disabled", _env_cfg(enabled=False), seed=321, pop=2)
    disabled.run()
    assert getattr(disabled, "_prediction_demand_event_count_gen", 0) == 0
    assert getattr(disabled, "_prediction_demand_resource_consequence_total_gen", 0.0) == 0.0

    enabled = _make_sim(tmp_path / "enabled", _env_cfg(enabled=True, observe=True), seed=321, pop=2)
    enabled.run()
    assert getattr(enabled, "_prediction_demand_event_count_gen", 0) > 0
    assert getattr(enabled, "_prediction_demand_cue_to_consequence_delay_count_gen", 0) > 0
    assert getattr(enabled, "_prediction_demand_cue_ambiguity_count_gen", 0) > 0
    assert getattr(enabled, "_prediction_demand_resource_consequence_total_gen", 0.0) > 0.0


def test_prediction_demand_metrics_csv_columns_are_emitted(tmp_path: Path) -> None:
    run_dir = tmp_path / "metrics"
    cfg = _env_cfg(enabled=True, observe=True)
    sim = SimulationRunner(
        env_cfg=cfg,
        pop_cfg=PopConfig(population_size=2),
        sim_cfg=_sim_cfg(seed=222, steps=18),
        logger=MetricsLogger(str(run_dir)),
        run_dir=str(run_dir),
        perf=False,
    )
    sim.run()

    with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    assert rows
    row = rows[-1]
    assert float(row["prediction_demand_enable"]) == 1.0
    assert float(row["prediction_demand_event_count"]) > 0.0
    assert float(row["prediction_demand_cue_ambiguity_mean"]) > 0.0
    assert float(row["prediction_demand_cue_to_consequence_delay_mean"]) >= float(cfg.prediction_demand_delay_min)
    assert float(row["prediction_demand_direct_prediction_reward_active"]) == 0.0
    assert float(row["prediction_demand_direct_error_reward_active"]) == 0.0
