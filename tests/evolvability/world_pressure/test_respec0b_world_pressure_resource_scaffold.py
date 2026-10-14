from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from env.env_config import EnvConfig
from env.gridworld import GridWorld
from env.gridworld_pkg.constants import ACT_GATHER, FOOD
from env.gridworld_pkg.patch_world import (
    PATCH_WORLD_ARRAY_NAMES,
    RESPEC0B_RESOURCE_A,
    RESPEC0B_RESOURCE_B,
    patch_food_metrics,
)
from evolution.population_pkg.population_config import PopConfig
from simulation.checkpoint_bundle_v2 import checkpoint_bundle_v2_apply, checkpoint_bundle_v2_load, checkpoint_bundle_v2_save
from simulation.metrics_pkg.noop_loggers import NoOpMetricsLogger
from simulation.rng_manager import RNGManager
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig


def _env_cfg(*, enabled: bool = False) -> EnvConfig:
    cfg = EnvConfig(width=6, height=4, view_radius=1)
    cfg.enable_biomes = False
    cfg.food_spawn_per_step = 0
    cfg.max_food = 0
    cfg.food_energy = 9.0
    cfg.patch_world_enable = True
    cfg.patch_food_regrow_per_step = 0.0
    cfg.patch_food_gather_amount = 1.0
    cfg.respec0b_resource_specialization_enable = bool(enabled)
    cfg.respec0b_resource_class_mode = "vertical_split_ab"
    return cfg


def _env(*, enabled: bool = False) -> GridWorld:
    cfg = _env_cfg(enabled=enabled)
    rm = RNGManager(global_seed=123)
    env = GridWorld(cfg, seed=123, run_dir=None, rng=rm.env_np)
    env.grid.fill(0)
    env.patch_food_amount.fill(0.0)
    return env


def test_respec0b_config_defaults_disabled() -> None:
    cfg = EnvConfig()
    sim_cfg = SimConfig()

    assert cfg.respec0b_resource_specialization_enable is False
    assert cfg.respec0b_resource_class_mode == "vertical_split_ab"
    assert cfg.respec0b_resource_a_handling_threshold == pytest.approx(1.50)
    assert cfg.respec0b_resource_b_handling_threshold == pytest.approx(1.75)
    assert cfg.respec0b_resource_equalized_sham_enable is False
    assert sim_cfg.resource_handling_representation_enable is False
    assert sim_cfg.resource_handling_representation_reporting_enable is False


def test_respec0b_disabled_preserves_patch_world_gather_return() -> None:
    env = _env(enabled=False)
    env.add_agent(0, (1, 1))
    env.grid[1, 1] = FOOD
    env.patch_food_amount[1, 1] = np.float32(1.0)

    out = env.step_agents_batch(
        agent_ids=np.asarray([0], dtype=np.int32),
        actions=np.asarray([ACT_GATHER], dtype=np.int32),
        energies=np.asarray([5.0], dtype=np.float32),
        healths=np.asarray([1.0], dtype=np.float32),
        injuries=np.asarray([0.0], dtype=np.float32),
        resource_handling_axes=np.asarray([1.0], dtype=np.float32),
    )

    assert out[3][0] == pytest.approx(env.cfg.food_energy)
    assert env.patch_food_amount[1, 1] == pytest.approx(0.0)


def test_respec0b_enabled_separates_a_and_b_handling_axes() -> None:
    env = _env(enabled=True)
    positions = [(1, 0), (1, 4), (2, 1), (2, 5)]
    classes = [RESPEC0B_RESOURCE_A, RESPEC0B_RESOURCE_B, RESPEC0B_RESOURCE_A, RESPEC0B_RESOURCE_B]
    for agent_id, (y, x) in enumerate(positions):
        env.add_agent(agent_id, (y, x))
        env.grid[y, x] = FOOD
        env.patch_food_amount[y, x] = np.float32(1.0)
        env.respec0b_resource_class[y, x] = np.float32(classes[agent_id])

    out = env.step_agents_batch(
        agent_ids=np.asarray([0, 1, 2, 3], dtype=np.int32),
        actions=np.asarray([ACT_GATHER, ACT_GATHER, ACT_GATHER, ACT_GATHER], dtype=np.int32),
        energies=np.full((4,), 5.0, dtype=np.float32),
        healths=np.ones((4,), dtype=np.float32),
        injuries=np.zeros((4,), dtype=np.float32),
        resource_handling_axes=np.asarray([1.0, 1.0, -1.0, -1.0], dtype=np.float32),
    )

    food_gained = out[3]
    assert food_gained.tolist() == pytest.approx([0.75, 0.0, 0.0, 1.5])
    stats = env.respec0b_resource_gather_stats_last
    assert stats["attempts"] == 4
    assert stats["successes"] == 2
    assert stats["failed_handling"] == 2
    assert stats["resource_a_successes"] == 1
    assert stats["resource_b_successes"] == 1
    assert stats["handling_ticks"] == 4


def test_respec0b_high_ids_reversed_slots_and_reuse_match_low_ids() -> None:
    env = _env(enabled=True)
    env._configure_world_slot_capacity(2)
    env.add_agent(100_001, (1, 0), slot=1, slot_token=1)
    env.add_agent(100_000, (1, 4), slot=0, slot_token=1)
    for y, x, resource_class in (
        (1, 0, RESPEC0B_RESOURCE_A),
        (1, 4, RESPEC0B_RESOURCE_B),
    ):
        env.grid[y, x] = FOOD
        env.patch_food_amount[y, x] = np.float32(1.0)
        env.respec0b_resource_class[y, x] = np.float32(resource_class)

    gained = env.step_agents_batch(
        agent_ids=np.asarray([100_001, 100_000], dtype=np.int64),
        actions=np.asarray([ACT_GATHER, ACT_GATHER], dtype=np.int32),
        energies=np.full((2,), 5.0, dtype=np.float32),
        healths=np.ones((2,), dtype=np.float32),
        injuries=np.zeros((2,), dtype=np.float32),
        resource_handling_axes=np.asarray([1.0, 1.0], dtype=np.float32),
    )[3]

    assert gained.tolist() == pytest.approx([0.75, 0.0])
    assert env.respec0b_resource_active_class.tolist() == [
        RESPEC0B_RESOURCE_B,
        0,
    ]
    env.remove_agent(100_000, slot=0, slot_token=1)
    assert env.respec0b_resource_active_y[0] == -1
    assert env.respec0b_resource_active_x[0] == -1
    assert env.respec0b_resource_active_class[0] == 0
    assert env.respec0b_resource_handling_progress[0] == pytest.approx(0.0)
    env.add_agent(200_000, (2, 4), slot=0, slot_token=2)
    assert env._resolve_world_slot(200_000) == (0, 2)


def test_respec0b_neutral_axis_remains_usable_on_both_resources() -> None:
    env = _env(enabled=True)
    positions = [(1, 0), (1, 4)]
    classes = [RESPEC0B_RESOURCE_A, RESPEC0B_RESOURCE_B]
    for agent_id, (y, x) in enumerate(positions):
        env.add_agent(agent_id, (y, x))
        env.grid[y, x] = FOOD
        env.patch_food_amount[y, x] = np.float32(1.0)
        env.respec0b_resource_class[y, x] = np.float32(classes[agent_id])

    kwargs = dict(
        agent_ids=np.asarray([0, 1], dtype=np.int32),
        actions=np.asarray([ACT_GATHER, ACT_GATHER], dtype=np.int32),
        energies=np.full((2,), 5.0, dtype=np.float32),
        healths=np.ones((2,), dtype=np.float32),
        injuries=np.zeros((2,), dtype=np.float32),
        resource_handling_axes=np.asarray([0.0, 0.0], dtype=np.float32),
    )

    first = env.step_agents_batch(**kwargs)[3].copy()
    second = env.step_agents_batch(**kwargs)[3].copy()

    assert first.tolist() == pytest.approx([0.0, 0.0])
    assert second.tolist() == pytest.approx([0.75, 1.5])


def test_respec0b_equalized_sham_removes_axis_advantage() -> None:
    env = _env(enabled=True)
    env.cfg.respec0b_resource_equalized_sham_enable = True
    for agent_id, (y, x, resource_class) in enumerate(
        [
            (1, 0, RESPEC0B_RESOURCE_A),
            (1, 4, RESPEC0B_RESOURCE_B),
        ]
    ):
        env.add_agent(agent_id, (y, x))
        env.grid[y, x] = FOOD
        env.patch_food_amount[y, x] = np.float32(1.0)
        env.respec0b_resource_class[y, x] = np.float32(resource_class)

    out = env.step_agents_batch(
        agent_ids=np.asarray([0, 1], dtype=np.int32),
        actions=np.asarray([ACT_GATHER, ACT_GATHER], dtype=np.int32),
        energies=np.full((2,), 5.0, dtype=np.float32),
        healths=np.ones((2,), dtype=np.float32),
        injuries=np.zeros((2,), dtype=np.float32),
        resource_handling_axes=np.asarray([1.0, -1.0], dtype=np.float32),
    )

    assert out[3].tolist() == pytest.approx([1.0, 1.0])
    metrics = patch_food_metrics(env)
    assert metrics["respec0b_resource_gather_attempts"] == pytest.approx(2.0)
    assert metrics["respec0b_resource_gather_successes"] == pytest.approx(2.0)


def _sim_cfg(seed: int = 123) -> SimConfig:
    cfg = SimConfig(
        seed=int(seed),
        generations=1,
        steps_per_generation=1,
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


def _sim(tmp_path: Path, *, enabled: bool) -> SimulationRunner:
    return SimulationRunner(
        env_cfg=_env_cfg(enabled=enabled),
        pop_cfg=PopConfig(population_size=4),
        sim_cfg=_sim_cfg(),
        logger=NoOpMetricsLogger(),
        run_dir=str(tmp_path),
        perf=False,
    )


def test_respec0b_resource_class_checkpoint_round_trip(tmp_path: Path) -> None:
    sim = _sim(tmp_path / "src", enabled=True)
    sim.pop.init_random()
    sim._sync_env_with_population()
    sim.env.respec0b_resource_class[:, :] = np.float32(RESPEC0B_RESOURCE_A)
    sim.env.respec0b_resource_class[:, 3:] = np.float32(RESPEC0B_RESOURCE_B)
    sim.env.respec0b_resource_active_y[0] = 1
    sim.env.respec0b_resource_active_x[0] = 2
    sim.env.respec0b_resource_active_class[0] = RESPEC0B_RESOURCE_A
    sim.env.respec0b_resource_handling_progress[0] = np.float32(0.5)
    expected = sim.env.respec0b_resource_class.copy()

    bundle = tmp_path / "bundle"
    checkpoint_bundle_v2_save(sim, str(bundle), generation_index=0, strict_replay_mode=True)
    payload = checkpoint_bundle_v2_load(str(bundle), strict=True, expected_commit_hash=None)

    assert "respec0b_resource_class" in PATCH_WORLD_ARRAY_NAMES
    assert "respec0b_resource_class" in payload["world"]["patch_world"]["arrays"]
    cursor = payload["world"]["patch_world"]["respec0b_resource_handling_state"]
    assert cursor["schema_version"] == "respec0b_slot_cursor_state_v1"
    assert cursor["slot_tokens"] == payload["world"]["world_membership_authority"]["slot_tokens"]

    restored = _sim(tmp_path / "dst", enabled=True)
    checkpoint_bundle_v2_apply(restored, payload, restore_rng=True)

    assert np.array_equal(restored.env.respec0b_resource_class, expected)
    assert restored.env.respec0b_resource_active_y[0] == 1
    assert restored.env.respec0b_resource_active_x[0] == 2
    assert restored.env.respec0b_resource_active_class[0] == RESPEC0B_RESOURCE_A
    assert restored.env.respec0b_resource_handling_progress[0] == pytest.approx(0.5)
