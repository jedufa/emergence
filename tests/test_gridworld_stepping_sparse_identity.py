from __future__ import annotations

import numpy as np
import pytest

from env.env_config import EnvConfig
from env.gridworld_pkg.constants import ACT_ATTACK, ACT_MOVE_E, ACT_REST
from env.gridworld_pkg.world import GridWorld


def _combat_step(agent_ids: tuple[int, int]):
    cfg = EnvConfig(width=4, height=4, view_radius=1)
    cfg.enable_biomes = False
    cfg.enable_combat = True
    cfg.enable_injury = False
    cfg.enable_scar = False
    cfg.attack_damage = 0.75
    env = GridWorld(cfg, seed=17, rng=np.random.default_rng(17))
    env.action_consequence_telemetry_enable = True
    env.add_agent(agent_ids[0], pos=(1, 1), slot=1, slot_token=1)
    env.add_agent(agent_ids[1], pos=(1, 2), slot=0, slot_token=1)
    ids = np.asarray(agent_ids, dtype=np.int64)
    observations = env.get_local_observations(ids)
    outputs = env.step_agents_batch(
        ids,
        np.asarray([ACT_ATTACK, ACT_REST], dtype=np.int32),
        np.asarray([10.0, 10.0], dtype=np.float32),
        np.asarray([5.0, 4.0], dtype=np.float32),
        np.zeros((2,), dtype=np.float32),
    )
    return env, observations, outputs


def test_combat_sparse_identity_relabel_preserves_behavior_and_observations() -> None:
    low_env, low_observations, low_outputs = _combat_step((0, 1))
    high_env, high_observations, high_outputs = _combat_step((100_001, 100_000))

    np.testing.assert_array_equal(high_observations, low_observations)
    for high, low in zip(high_outputs, low_outputs, strict=True):
        np.testing.assert_array_equal(high, low)

    low_event = low_env.last_attack_consequence_events
    high_event = high_env.last_attack_consequence_events
    assert len(low_event) == len(high_event) == 1
    assert low_event[0]["attack_accepted"] is high_event[0]["attack_accepted"] is True
    assert low_event[0]["target_id"] == 1
    assert high_event[0]["target_id"] == 100_000
    assert low_event[0]["target_health_after"] == high_event[0]["target_health_after"]

    assert low_env._step_combat_lookup_ids.shape == (2,)
    assert high_env._step_combat_lookup_ids.shape == (2,)
    assert high_env._step_combat_lookup_ids.size < int(
        high_env._step_combat_lookup_ids[-1]
    )
    assert not hasattr(high_env, "_step_id_to_idx")
    assert high_env.agent_alive.shape == low_env.agent_alive.shape
    assert high_env.slot_to_agent_id.shape == low_env.slot_to_agent_id.shape
    assert int(high_env.agent_grid[1, 1]) == 1
    assert int(high_env.agent_grid[1, 2]) == 0
    assert int(high_env.slot_to_agent_id[int(high_env.agent_grid[1, 1])]) == 100_001
    assert int(high_env.slot_to_agent_id[int(high_env.agent_grid[1, 2])]) == 100_000


def test_combat_near_int32_boundary_preserves_shape_and_behavior() -> None:
    low_env, low_observations, low_outputs = _combat_step((0, 1))
    high_a = int(np.iinfo(np.int32).max)
    high_b = high_a - 1
    high_env, high_observations, high_outputs = _combat_step((high_a, high_b))

    np.testing.assert_array_equal(high_observations, low_observations)
    for high, low in zip(high_outputs, low_outputs, strict=True):
        np.testing.assert_array_equal(high, low)
    assert high_env.agent_alive.shape == low_env.agent_alive.shape
    assert high_env.slot_to_agent_id.shape == low_env.slot_to_agent_id.shape
    assert int(high_env.agent_grid[1, 1]) == 1
    assert int(high_env.agent_grid[1, 2]) == 0
    assert int(high_env.slot_to_agent_id[int(high_env.agent_grid[1, 1])]) == high_a
    assert int(high_env.slot_to_agent_id[int(high_env.agent_grid[1, 2])]) == high_b
    assert high_env.last_attack_consequence_events[0]["target_id"] == high_b


def test_world_slot_reuse_rejects_stale_token_and_resolves_new_high_id() -> None:
    cfg = EnvConfig(width=3, height=3, view_radius=1)
    env = GridWorld(cfg, seed=23, rng=np.random.default_rng(23))
    shadow_shape = env.agent_alive.shape

    env.add_agent(100_000, pos=(1, 1), slot=0, slot_token=1)
    assert env._resolve_world_slot(100_000, require_alive=True) == (0, 1)
    assert env.adjacent_agent_ids(100_000) == []
    env.remove_agent(100_000, slot=0, slot_token=1)

    with pytest.raises(RuntimeError, match="stale world slot token"):
        env.add_agent(100_001, pos=(1, 1), slot=0, slot_token=1)

    env.add_agent(100_001, pos=(1, 1), slot=0, slot_token=2)
    assert env._resolve_world_slot(100_001, require_alive=True) == (0, 2)
    with pytest.raises(RuntimeError, match="stale world removal identity"):
        env.remove_agent(100_000, slot=0, slot_token=1)
    assert env._resolve_world_slot(100_001, require_alive=True) == (0, 2)
    env.add_agent(100_001)
    assert env.get_local_observation(100_001).shape == (3, 3)
    assert int(env.agent_grid[1, 1]) == 0
    assert int(env.slot_to_agent_id[int(env.agent_grid[1, 1])]) == 100_001
    assert env.agent_alive.shape == shadow_shape

    unused_slot_before = env._world_slot_snapshot(1)
    with pytest.raises(ValueError, match="Cell occupied"):
        env.add_agent(100_002, pos=(1, 1), slot=1, slot_token=1)
    assert env._world_slot_snapshot(1) == unused_slot_before

    env.step_agents_batch(
        np.asarray([100_001], dtype=np.int64),
        np.asarray([ACT_MOVE_E], dtype=np.int32),
        np.asarray([10.0], dtype=np.float32),
        np.asarray([5.0], dtype=np.float32),
        np.zeros((1,), dtype=np.float32),
    )
    assert env._world_slot_snapshot(0)[2:4] == (1, 2)
    assert int(env.agent_grid[1, 2]) == 0
    assert int(env.slot_to_agent_id[int(env.agent_grid[1, 2])]) == 100_001

    with pytest.raises(RuntimeError, match="matches=0"):
        env._resolve_world_slot(100_000)

    env.slot_to_agent_id[0] = 100_002
    with pytest.raises(RuntimeError, match="world reverse resolver mismatch"):
        env._resolve_world_slot(100_001)


def test_remove_agent_releases_reserved_unplaced_world_membership() -> None:
    cfg = EnvConfig(width=3, height=3)
    env = GridWorld(cfg, seed=43, rng=np.random.default_rng(43))
    env._bind_world_slot(100_000, slot=0, slot_token=1)

    assert env._resolve_world_slot(100_000) == (0, 1)
    assert env.agent_world_alive_by_slot[0] is np.False_

    env.remove_agent(100_000, slot=0, slot_token=1)

    assert env._agent_id_to_world_slot == {}
    assert int(env.slot_to_agent_id[0]) == -1
    assert int(env.slot_tokens[0]) == 1
    env._bind_world_slot(100_001, slot=0, slot_token=2)
    assert env._resolve_world_slot(100_001) == (0, 2)


def test_step_rejects_agent_ids_outside_signed_int32_before_narrowing() -> None:
    cfg = EnvConfig(width=2, height=2)
    env = GridWorld(cfg, seed=19, rng=np.random.default_rng(19))

    with pytest.raises(ValueError, match="signed int32 range"):
        env.step_agents_batch(
            np.asarray([np.iinfo(np.int32).max + 1], dtype=np.int64),
            np.asarray([ACT_REST], dtype=np.int32),
            np.ones((1,), dtype=np.float32),
            np.ones((1,), dtype=np.float32),
            np.zeros((1,), dtype=np.float32),
        )


def test_add_agent_rejects_agent_id_above_signed_int32_before_grid_write() -> None:
    cfg = EnvConfig(width=2, height=2)
    env = GridWorld(cfg, seed=29, rng=np.random.default_rng(29))
    before = env.agent_grid.copy()

    with pytest.raises(ValueError, match="signed int32 range"):
        env.add_agent(
            int(np.iinfo(np.int32).max) + 1,
            pos=(0, 0),
            slot=0,
            slot_token=1,
        )

    np.testing.assert_array_equal(env.agent_grid, before)
