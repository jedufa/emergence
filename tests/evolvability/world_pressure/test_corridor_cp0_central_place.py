from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from env.env_config import EnvConfig
from env.gridworld import GridWorld
from env.gridworld_pkg.corridor_cp0 import (
    corridor_cp0_agent_home_eligible,
    corridor_cp0_central_place_eligibility_enabled,
    corridor_cp0_filter_home_eligible_agents,
)
from env.gridworld_pkg.patch_world import (
    CORRIDOR_CP0_ZONE_CORRIDOR,
    CORRIDOR_CP0_ZONE_HOME_LOCAL,
    CORRIDOR_CP0_ZONE_REMOTE_RICH,
    PATCH_WORLD_ARRAY_NAMES,
    PATCH_WORLD_OBSERVATION_CHANNEL_NAMES,
)
from simulation.rng_manager import RNGManager
from simulation.runner_pkg.runner_config import SimConfig


def _env_cfg(*, enabled: bool = False) -> EnvConfig:
    cfg = EnvConfig(width=10, height=6, view_radius=1)
    cfg.enable_biomes = False
    cfg.food_spawn_per_step = 0
    cfg.max_food = 0
    cfg.patch_world_enable = True
    cfg.patch_food_regrow_per_step = 0.0
    cfg.patch_food_amount_initial_empty_cell_value = 0.0
    cfg.corridor_cp0_enable = bool(enabled)
    cfg.corridor_cp0_home_width = 2
    cfg.corridor_cp0_remote_width = 2
    cfg.corridor_cp0_corridor_half_width = 0
    return cfg


def _env(*, enabled: bool = False) -> GridWorld:
    cfg = _env_cfg(enabled=enabled)
    rm = RNGManager(global_seed=321)
    return GridWorld(cfg, seed=321, run_dir=None, rng=rm.env_np)


def test_corridor_cp0_defaults_disabled_and_hidden_from_observation() -> None:
    cfg = EnvConfig()
    sim_cfg = SimConfig()

    assert cfg.corridor_cp0_enable is False
    assert sim_cfg.corridor_cp0_central_place_eligibility_enable is False
    assert "corridor_cp0_zone_class" in PATCH_WORLD_ARRAY_NAMES
    assert "corridor_cp0_zone_class" not in PATCH_WORLD_OBSERVATION_CHANNEL_NAMES
    assert corridor_cp0_central_place_eligibility_enabled(cfg, sim_cfg) is False


def test_corridor_cp0_disabled_leaves_zone_map_empty() -> None:
    env = _env(enabled=False)

    assert np.all(env.corridor_cp0_zone_class == 0.0)


def test_corridor_cp0_enabled_builds_central_place_tradeoff_map() -> None:
    env = _env(enabled=True)
    zones = env.corridor_cp0_zone_class.astype(np.int32)

    home = zones == CORRIDOR_CP0_ZONE_HOME_LOCAL
    corridor = zones == CORRIDOR_CP0_ZONE_CORRIDOR
    remote = zones == CORRIDOR_CP0_ZONE_REMOTE_RICH

    assert int(np.sum(home)) == 12
    assert int(np.sum(corridor)) == 6
    assert int(np.sum(remote)) == 12
    assert env.patch_food_amount[0, 0] == pytest.approx(env.cfg.corridor_cp0_home_food)
    assert env.patch_food_amount[3, 4] == pytest.approx(env.cfg.corridor_cp0_corridor_food)
    assert env.patch_food_amount[0, 9] == pytest.approx(env.cfg.corridor_cp0_remote_food)
    assert env.patch_shelter[0, 0] > env.patch_shelter[0, 9]
    assert env.patch_movement_cost[3, 4] > env.patch_movement_cost[0, 0]


def test_corridor_cp0_home_eligibility_uses_hidden_zone_only() -> None:
    env = _env(enabled=True)
    sim_cfg = SimConfig()
    sim_cfg.corridor_cp0_central_place_eligibility_enable = True
    env.add_agent(0, (0, 0))
    env.add_agent(1, (3, 4))
    env.add_agent(2, (0, 9))

    agents = [SimpleNamespace(agent_id=0), SimpleNamespace(agent_id=1), SimpleNamespace(agent_id=2)]

    assert corridor_cp0_central_place_eligibility_enabled(env.cfg, sim_cfg) is True
    assert corridor_cp0_agent_home_eligible(env, agents[0]) is True
    assert corridor_cp0_agent_home_eligible(env, agents[1]) is False
    assert corridor_cp0_agent_home_eligible(env, agents[2]) is False
    assert corridor_cp0_filter_home_eligible_agents(env, agents) == [agents[0]]
