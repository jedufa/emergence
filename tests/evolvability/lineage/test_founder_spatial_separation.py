from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

from env.env_config import EnvConfig
from env.gridworld import GridWorld
from evolution.population import Population, PopConfig
from evolution.population_pkg.founder_spatial_separation import (
    build_founder_spatial_plan,
    founder_spatial_separation_enabled,
    place_initial_founders,
    validate_founder_spatial_config,
)
from simulation.runner_pkg.runner_config import SimConfig


def _population(*, size: int = 16, groups: int = 4, spatial: bool = False, mode: str = "max_distance") -> Population:
    pop = Population(
        env_cfg=EnvConfig(width=8, height=8),
        pop_cfg=PopConfig(population_size=size, parent_fraction=0.25, cull_fraction=0.5),
        rng=random.Random(123),
        np_rng=np.random.default_rng(123),
    )
    cfg = SimConfig(seed=123)
    cfg.lineage_founder_balance_enable = True
    cfg.lineage_founder_group_count = groups
    cfg.lineage_founder_min_agents_per_group = max(1, size // groups)
    cfg.lineage_founder_assignment_mode = "deterministic_round_robin"
    cfg.lineage_founder_balance_schema_version = 1
    cfg.lineage_founder_spatial_separation_enable = spatial
    cfg.lineage_founder_spatial_group_count = groups
    cfg.lineage_founder_spatial_min_distance = 4
    cfg.lineage_founder_spatial_assignment_mode = mode
    cfg.lineage_founder_spatial_use_patch_basins = True
    cfg.lineage_founder_spatial_schema_version = 1
    pop.sim_cfg = cfg
    pop.init_random()
    return pop


def _env(cfg: EnvConfig, *, seed: int = 123) -> GridWorld:
    return GridWorld(cfg, seed=seed, rng=np.random.default_rng(seed))


def _positions(env: GridWorld, pop: Population) -> list[tuple[int, int, int]]:
    return [
        (int(agent.agent_id), int(env.agent_pos_y[int(agent.agent_id)]), int(env.agent_pos_x[int(agent.agent_id)]))
        for agent in pop.alive_agents()
    ]


def test_founder_spatial_disabled_preserves_legacy_random_placement() -> None:
    pop_a = _population(spatial=False)
    pop_b = _population(spatial=False)
    env_a = _env(pop_a.env_cfg)
    env_b = _env(pop_b.env_cfg)

    assert not founder_spatial_separation_enabled(pop_a.sim_cfg)
    for agent in pop_a.alive_agents():
        env_a.add_agent(int(agent.agent_id))
    for agent in pop_b.alive_agents():
        env_b.add_agent(int(agent.agent_id))

    assert _positions(env_a, pop_a) == _positions(env_b, pop_b)


def test_founder_spatial_enabled_changes_initial_placement_and_is_deterministic(tmp_path: Path) -> None:
    pop_a = _population(spatial=True)
    pop_b = _population(spatial=True)
    env_a = _env(pop_a.env_cfg)
    env_b = _env(pop_b.env_cfg)

    plan_a = place_initial_founders(pop_a, env_a, run_dir=str(tmp_path / "a"))
    plan_b = place_initial_founders(pop_b, env_b, run_dir=str(tmp_path / "b"))

    assert plan_a["enabled"] is True
    assert plan_a["assignment_mode"] == "max_distance"
    assert plan_a["min_distance_satisfied"] is True
    assert plan_a["founder_collision_index"] == 0.0
    assert _positions(env_a, pop_a) == _positions(env_b, pop_b)
    assert plan_a["mapping"] == plan_b["mapping"]
    assert (tmp_path / "a" / "metrics" / "lineage_founder_spatial_initialization.json").exists()


def test_founder_spatial_min_distance_degrades_with_warning_when_grid_too_small() -> None:
    pop = _population(size=4, groups=4, spatial=True)
    pop.env_cfg = EnvConfig(width=2, height=2)
    pop.sim_cfg.lineage_founder_spatial_min_distance = 10

    plan = build_founder_spatial_plan(
        agents=pop.alive_agents(),
        env_cfg=pop.env_cfg,
        sim_cfg=pop.sim_cfg,
        env=None,
    )

    assert plan["enabled"] is True
    assert plan["min_distance_satisfied"] is False
    assert plan["warning"] == "min_distance_not_feasible"
    assert plan["founder_collision_index"] > 0.0


def test_founder_spatial_grid_quadrants_respects_distance_when_feasible() -> None:
    pop = _population(spatial=True, mode="grid_quadrants")

    plan = build_founder_spatial_plan(agents=pop.alive_agents(), env_cfg=pop.env_cfg, sim_cfg=pop.sim_cfg)

    assert plan["assignment_mode"] == "grid_quadrants"
    assert plan["group_count"] == 4
    assert plan["pairwise_distance_summary"]["min"] >= 4.0


def test_founder_spatial_invalid_mode_fails_clearly() -> None:
    cfg = SimConfig(seed=123)
    cfg.lineage_founder_spatial_separation_enable = True
    cfg.lineage_founder_spatial_assignment_mode = "patch_magic"

    with pytest.raises(ValueError, match="lineage_founder_spatial_assignment_mode"):
        validate_founder_spatial_config(population_size=8, env_cfg=EnvConfig(width=8, height=8), sim_cfg=cfg)


def test_founder_spatial_does_not_enable_intervention_flags() -> None:
    cfg = SimConfig(seed=123)
    cfg.lineage_founder_spatial_separation_enable = True
    cfg.lineage_founder_spatial_assignment_mode = "max_distance"

    assert not bool(getattr(cfg, "lineage_protection_enable", False))
    assert not bool(getattr(cfg, "lineage_rescue_enable", False))
    assert not bool(getattr(cfg, "diversity_reward_enable", False))
    assert not bool(getattr(cfg, "archive_reseed_enable", False))
    assert not bool(getattr(cfg, "population_quota_enable", False))
