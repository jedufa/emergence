from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from env.env_config import EnvConfig
from env.gridworld import GridWorld
from env.gridworld_pkg.constants import ACT_GATHER
from env.gridworld_pkg.patch_world import (
    HABACCLIM_HABITAT_A,
    PATCH_WORLD_ARRAY_NAMES,
    PATCH_WORLD_OBSERVATION_CHANNEL_NAMES,
    habacclim_acclimation_rate,
    habacclim_observation_scalar_index,
    patch_world_total_observation_size,
)
from simulation.habacclim1 import (
    HABACCLIM1R_SELECTED_POLICY_HASHES,
    apply_post_env,
    build_probe_policy,
    controller_contract,
    policy_hashes,
)
from simulation.rng_manager import RNGManager


def _cfg(*, enabled: bool = True, observe: bool = True, control: str = "acclimation_tradeoff") -> EnvConfig:
    cfg = EnvConfig(width=8, height=6, view_radius=1)
    cfg.enable_biomes = False
    cfg.food_spawn_per_step = 0
    cfg.max_food = 0
    cfg.patch_world_enable = True
    cfg.patch_world_observation_enable = True
    cfg.habacclim_enable = bool(enabled)
    cfg.habacclim_observation_enable = bool(observe)
    cfg.habacclim_control_mode = control
    cfg.patch_food_regrow_per_step = 0.0
    return cfg


def _env(cfg: EnvConfig) -> GridWorld:
    rm = RNGManager(global_seed=917)
    return GridWorld(cfg, seed=917, run_dir=None, rng=rm.env_np)


def test_habacclim_defaults_hidden_and_observation_versioned() -> None:
    base = _cfg(enabled=False, observe=False)
    enabled_no_obs = _cfg(enabled=True, observe=False)
    enabled_obs = _cfg(enabled=True, observe=True)

    assert "habacclim_habitat_class" in PATCH_WORLD_ARRAY_NAMES
    assert "habacclim_habitat_class" not in PATCH_WORLD_OBSERVATION_CHANNEL_NAMES
    assert patch_world_total_observation_size(base) == 74
    assert patch_world_total_observation_size(enabled_no_obs) == 74
    assert patch_world_total_observation_size(enabled_obs) == 75
    assert habacclim_observation_scalar_index(enabled_obs) == 74
    assert controller_contract(enabled_obs)["habitat_acclimation_state_norm_index"] == 74


def test_habacclim_map_sets_cues_without_exposing_hidden_label() -> None:
    env = _env(_cfg())
    classes = env.habacclim_habitat_class.astype(np.int32)
    a_mask = classes == HABACCLIM_HABITAT_A
    b_mask = classes != HABACCLIM_HABITAT_A

    assert int(np.sum(a_mask)) > 0
    assert int(np.sum(b_mask)) > 0
    assert float(np.mean(env.patch_food_amount[a_mask])) > float(np.mean(env.patch_food_amount[b_mask]))
    assert float(np.mean(env.patch_shelter[b_mask])) > float(np.mean(env.patch_shelter[a_mask]))
    assert "habacclim_habitat_class" not in PATCH_WORLD_OBSERVATION_CHANNEL_NAMES


def test_habacclim_post_env_uses_pre_update_state_for_processing_penalty() -> None:
    cfg = _cfg()
    env = _env(cfg)
    env.add_agent(0, (0, 0))
    agent = SimpleNamespace(agent_id=0, habitat_acclimation_state=-1.0)
    runner = SimpleNamespace(env_cfg=cfg, env=env)
    new_energy = np.asarray([20.0], dtype=np.float32)
    food_gained = np.asarray([10.0], dtype=np.float32)

    apply_post_env(
        runner,
        alive=[agent],
        agent_ids=np.asarray([0], dtype=np.int32),
        actions=np.asarray([ACT_GATHER], dtype=np.int32),
        new_energy=new_energy,
        food_gained=food_gained,
    )

    expected_loss = 10.0 * float(cfg.habacclim_processing_penalty_max)
    assert new_energy[0] == pytest.approx(20.0 - expected_loss)
    assert food_gained[0] == pytest.approx(10.0 - expected_loss)
    assert agent.habitat_acclimation_state == pytest.approx(-1.0 + habacclim_acclimation_rate(cfg) * 2.0)


def test_habacclim_zero_lag_control_removes_processing_penalty() -> None:
    cfg = _cfg(control="zero_lag_control")
    env = _env(cfg)
    env.add_agent(0, (0, 0))
    agent = SimpleNamespace(agent_id=0, habitat_acclimation_state=-1.0)
    runner = SimpleNamespace(env_cfg=cfg, env=env)
    new_energy = np.asarray([20.0], dtype=np.float32)
    food_gained = np.asarray([10.0], dtype=np.float32)

    apply_post_env(
        runner,
        alive=[agent],
        agent_ids=np.asarray([0], dtype=np.int32),
        actions=np.asarray([ACT_GATHER], dtype=np.int32),
        new_energy=new_energy,
        food_gained=food_gained,
    )

    assert new_energy[0] == pytest.approx(20.0)
    assert food_gained[0] == pytest.approx(10.0)
    assert agent.habitat_acclimation_state == pytest.approx(1.0)


def test_habacclim_fixed_probe_is_ordinary_mlp_with_new_dimension() -> None:
    policy = build_probe_policy("habitat_a_committed")
    obs = np.zeros((2, 75), dtype=np.float32)
    obs[0, 74] = 1.0
    obs[1, 74] = -1.0

    logits = policy.forward(torch.from_numpy(obs)).detach().cpu().numpy()

    assert policy.net[0].in_features == 75
    assert logits.shape == (2, 7)
    assert not np.array_equal(logits[0], logits[1])


def test_habacclim1r_selected_policy_hashes_are_frozen() -> None:
    hashes = policy_hashes()

    assert hashes == HABACCLIM1R_SELECTED_POLICY_HASHES
    assert controller_contract()["policy_hashes"] == HABACCLIM1R_SELECTED_POLICY_HASHES
