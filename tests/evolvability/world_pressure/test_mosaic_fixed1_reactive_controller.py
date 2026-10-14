from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from env.env_config import EnvConfig
from env.gridworld import GridWorld
from env.gridworld_pkg.constants import ACT_GATHER, ACT_MOVE_N, ACT_REST, FOOD
from env.gridworld_pkg.patch_world import MOSAIC0A_PATCH_BURST, MOSAIC0A_PATCH_RENEWAL
from simulation.mosaic_fixed1_controller import ARM_NAMES, choose_actions, controller_contract, record_post_env, write_generation_summary
from simulation.rng_manager import RNGManager
from simulation.runner_pkg.runner_config import SimConfig


def _sim_cfg(arm: str = "exploit_move") -> SimConfig:
    cfg = SimConfig()
    cfg.policy_deterministic_actions = True
    cfg.mosaic_fixed1_probe_enable = True
    cfg.mosaic_fixed1_probe_arm = arm
    return cfg


def _patch_env_cfg() -> EnvConfig:
    cfg = EnvConfig(width=5, height=5, view_radius=1)
    cfg.enable_biomes = False
    cfg.food_spawn_per_step = 0
    cfg.max_food = 0
    cfg.food_energy = 9.0
    cfg.patch_world_enable = True
    cfg.patch_world_observation_enable = True
    cfg.patch_world_observation_schema_version = 1
    cfg.patch_food_regrow_per_step = 0.0
    cfg.patch_food_gather_amount = 1.0
    cfg.mosaic0a_depletion_recovery_enable = True
    cfg.mosaic0a_class_mode = "budget_matched_clustered_v1"
    cfg.mosaic0a_control_mode = "real_mosaic"
    return cfg


def _env(cfg: EnvConfig) -> GridWorld:
    rm = RNGManager(global_seed=123)
    env = GridWorld(cfg, seed=123, run_dir=None, rng=rm.env_np)
    env.grid.fill(0)
    if hasattr(env, "patch_food_amount"):
        env.patch_food_amount.fill(0.0)
    if hasattr(env, "patch_waste"):
        env.patch_waste.fill(0.0)
    if hasattr(env, "patch_density"):
        env.patch_density.fill(0.0)
    if hasattr(env, "patch_movement_cost"):
        env.patch_movement_cost.fill(0.0)
    if hasattr(env, "mosaic0a_patch_class"):
        env.mosaic0a_patch_class.fill(0)
    return env


def _runner(tmp_path: Path, *, arm: str = "exploit_move", env_cfg: EnvConfig | None = None) -> SimpleNamespace:
    cfg = env_cfg or _patch_env_cfg()
    env = _env(cfg)
    env.add_agent(0, (2, 2))
    return SimpleNamespace(sim_cfg=_sim_cfg(arm), env_cfg=cfg, env=env, run_dir=str(tmp_path))


def test_mosaic_fixed1_defaults_are_disabled() -> None:
    cfg = SimConfig()

    assert cfg.mosaic_fixed1_probe_enable is False
    assert cfg.mosaic_fixed1_probe_schema_version == 1
    assert cfg.mosaic_fixed1_probe_arm == ""
    assert cfg.mosaic_fixed1_probe_controller_version == "mosaic_fixed1_reactive_v1"
    assert cfg.mosaic_fixed1r_telemetry_enable is False
    assert cfg.mosaic_fixed1r_telemetry_schema_version == 1


def test_mosaic_fixed1_disabled_returns_policy_actions(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    runner.sim_cfg.mosaic_fixed1_probe_enable = False
    raw = np.asarray([ACT_MOVE_N], dtype=np.int32)

    out = choose_actions(runner, [SimpleNamespace(agent_id=0)], raw)

    assert out.tolist() == [ACT_MOVE_N]


def test_mosaic_fixed1_arm_contract_is_frozen() -> None:
    contract = controller_contract()

    assert tuple(contract["arms"].keys()) == ARM_NAMES
    assert set(contract["hidden_state_excluded"]) >= {"mosaic0a_patch_class", "lineage identity", "selection outcomes"}


def test_mosaic_fixed1_exploit_gathers_visible_patch_food(tmp_path: Path) -> None:
    runner = _runner(tmp_path, arm="exploit_move")
    runner.env.patch_food_amount[2, 2] = np.float32(0.9)
    runner.env.grid[2, 2] = FOOD

    out = choose_actions(runner, [SimpleNamespace(agent_id=0)], np.asarray([ACT_REST], dtype=np.int32))

    assert out.tolist() == [ACT_GATHER]


def test_mosaic_fixed1_flat_control_can_run_without_patch_observations(tmp_path: Path) -> None:
    cfg = EnvConfig(width=5, height=5, view_radius=1)
    cfg.enable_biomes = False
    cfg.food_spawn_per_step = 0
    cfg.max_food = 0
    runner = _runner(tmp_path, arm="exploit_move", env_cfg=cfg)
    runner.env.grid[1, 2] = FOOD

    out = choose_actions(runner, [SimpleNamespace(agent_id=0)], np.asarray([ACT_REST], dtype=np.int32))

    assert out.tolist() == [ACT_MOVE_N]


def test_mosaic_fixed1_recovery_conservative_rests_under_local_pressure(tmp_path: Path) -> None:
    runner = _runner(tmp_path, arm="recovery_conservative")
    runner.env.patch_food_amount[2, 2] = np.float32(0.35)
    runner.env.patch_waste[2, 2] = np.float32(1.0)
    for agent_id, pos in enumerate(((1, 1), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2), (3, 3)), start=1):
        runner.env.add_agent(agent_id, pos)

    out = choose_actions(runner, [SimpleNamespace(agent_id=0)], np.asarray([ACT_GATHER], dtype=np.int32))

    assert out.tolist() == [ACT_REST]


def test_mosaic_fixed1_action_does_not_depend_on_hidden_patch_class(tmp_path: Path) -> None:
    runner = _runner(tmp_path, arm="balanced_generalist")
    runner.env.patch_food_amount[1, 2] = np.float32(1.0)
    runner.env.mosaic0a_patch_class[2, 2] = MOSAIC0A_PATCH_BURST
    first = choose_actions(runner, [SimpleNamespace(agent_id=0)], np.asarray([ACT_REST], dtype=np.int32)).copy()

    runner.env.mosaic0a_patch_class[2, 2] = MOSAIC0A_PATCH_RENEWAL
    second = choose_actions(runner, [SimpleNamespace(agent_id=0)], np.asarray([ACT_REST], dtype=np.int32)).copy()

    assert first.tolist() == second.tolist()


def test_mosaic_fixed1_telemetry_writes_compact_summary(tmp_path: Path) -> None:
    runner = _runner(tmp_path, arm="exploit_move")
    runner.env.mosaic0a_patch_class[2, 2] = MOSAIC0A_PATCH_BURST
    runner.env.patch_food_amount[2, 2] = np.float32(0.9)
    actions = choose_actions(runner, [SimpleNamespace(agent_id=0)], np.asarray([ACT_REST], dtype=np.int32))
    record_post_env(runner, agent_ids=np.asarray([0], dtype=np.int32), actions=actions, food_gained=np.asarray([1.0], dtype=np.float32))

    write_generation_summary(runner, generation=0)

    csv_path = tmp_path / "metrics" / "mosaic_fixed1_generation_summary.csv"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    contract = json.loads((tmp_path / "metrics" / "mosaic_fixed1_controller_contract.json").read_text(encoding="utf-8"))
    assert rows[0]["burst_gather_successes"] == "1.0"
    assert rows[0]["burst_food_gain"] == "1.0"
    assert contract["arm"] == "exploit_move"
    assert not (tmp_path / "metrics" / "mosaic_fixed1r_generation_summary.csv").exists()


def test_mosaic_fixed1r_telemetry_writes_reciprocal_summary(tmp_path: Path) -> None:
    runner = _runner(tmp_path, arm="exploit_move")
    runner.sim_cfg.mosaic_fixed1r_telemetry_enable = True
    runner.env.mosaic0a_patch_class[2, 2] = MOSAIC0A_PATCH_BURST
    runner.env.patch_food_amount[2, 2] = np.float32(0.9)
    runner._eiel_fdp_cur_gen = 0
    agent = SimpleNamespace(agent_id=0)
    actions = choose_actions(runner, [agent], np.asarray([ACT_REST], dtype=np.int32))
    record_post_env(runner, agent_ids=np.asarray([0], dtype=np.int32), actions=actions, food_gained=np.asarray([1.0], dtype=np.float32))
    runner._eligible_parent_agents_gen = [agent]
    runner._selected_parent_agents_gen = [agent]
    runner._lineage_reproduction_funnel_birth_contexts_gen = [{"parent_id": 0}]

    write_generation_summary(runner, generation=0)

    csv_path = tmp_path / "metrics" / "mosaic_fixed1r_generation_summary.csv"
    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    contract = json.loads((tmp_path / "metrics" / "mosaic_fixed1r_telemetry_contract.json").read_text(encoding="utf-8"))
    assert rows[0]["burst_available_resource_agent_ticks"] == "1.0"
    assert rows[0]["burst_successful_gathers"] == "1.0"
    assert rows[0]["burst_selected_parent_events_after_class_use"] == "1.0"
    assert rows[0]["burst_successful_births_after_class_use"] == "1.0"
    assert contract["recent_use_definition"].startswith("at least one successful gather")


def test_mosaic_fixed1_invalid_arm_fails_closed(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    runner.sim_cfg.mosaic_fixed1_probe_arm = "unknown"

    with pytest.raises(ValueError, match="unsupported mosaic_fixed1_probe_arm"):
        choose_actions(runner, [SimpleNamespace(agent_id=0)], np.asarray([ACT_REST], dtype=np.int32))
