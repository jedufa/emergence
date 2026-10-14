from __future__ import annotations

import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import simulation.checkpoint_bundle_v2 as checkpoint_bundle
from agents.agent import AgentStats
from env.env_config import EnvConfig
from env.gridworld_pkg.state import StateMixin
from env.gridworld_pkg.world import GridWorld
from evolution.population_pkg import spawn
from evolution.population_pkg.core import Population
from evolution.population_pkg.slots import SlotPool
from simulation.runner_pkg.runner_envsync import EnvSyncMixin
from simulation.checkpoint_bundle_v2 import (
    CheckpointStrictError,
    _export_world,
    _normalize_checkpoint_world_authority,
)


_RESPEC_ARRAY_NAMES = (
    "respec0b_resource_active_y",
    "respec0b_resource_active_x",
    "respec0b_resource_active_class",
    "respec0b_resource_handling_progress",
)


def _owner(*, respec_enabled: bool = False) -> spawn.SpawnMixin:
    owner = spawn.SpawnMixin()
    owner._slots = SlotPool.build(base_population=1, slot_factor=2, max_slots_cap=0)
    owner.next_id = 3
    owner.env = SimpleNamespace(
        cfg=SimpleNamespace(
            respec0b_resource_specialization_enable=respec_enabled,
        ),
        agent_pos_y=np.full((3,), -1, dtype=np.int64),
        agent_pos_x=np.full((3,), -1, dtype=np.int64),
        agent_alive=np.zeros((3,), dtype=bool),
    )
    if respec_enabled:
        owner.env.respec0b_resource_active_y = np.asarray([4, -1, 2], dtype=np.int32)
        owner.env.respec0b_resource_active_x = np.asarray([1, -1, 3], dtype=np.int32)
        owner.env.respec0b_resource_active_class = np.asarray([2, 0, 1], dtype=np.int32)
        owner.env.respec0b_resource_handling_progress = np.asarray(
            [0.25, 0.0, 0.75],
            dtype=np.float32,
        )
    owner.env_cfg = SimpleNamespace(start_energy=10.0, start_health=5.0, max_energy=20.0)
    owner.sim_cfg = SimpleNamespace()
    owner.rng = random.Random(1)
    owner.rng_manager = None
    owner.obs_size = 4
    owner.agents = {}
    return owner


def _parent(agent_id: int) -> SimpleNamespace:
    return SimpleNamespace(
        agent_id=agent_id,
        genome=SimpleNamespace(hidden_size=4, memory_dim=0, use_gru=False),
        policy=object(),
        stats=AgentStats(),
        slot=agent_id - 1,
        slot_token=1,
    )


def _allocation_state(owner: spawn.SpawnMixin) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return tuple(owner._slots.free_slots), tuple(owner._slots.slot_tokens)


def test_crossover_failure_does_not_reserve_identity_slot_or_env_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = _owner()
    parents = (_parent(1), _parent(2))
    allocation_before = _allocation_state(owner)
    env_arrays_before = (
        owner.env.agent_pos_y,
        owner.env.agent_pos_x,
        owner.env.agent_alive,
    )

    def fail_crossover(*args: object, **kwargs: object) -> object:
        raise RuntimeError("structural mutation failed")

    monkeypatch.setattr(spawn, "crossover_genome", fail_crossover)

    with pytest.raises(RuntimeError, match="structural mutation failed"):
        owner.make_child_row_spec(*parents, child_energy=4.0)

    assert owner.next_id == 3
    assert _allocation_state(owner) == allocation_before
    assert owner.env.agent_pos_y is env_arrays_before[0]
    assert owner.env.agent_pos_x is env_arrays_before[1]
    assert owner.env.agent_alive is env_arrays_before[2]
    assert tuple(array.shape for array in env_arrays_before) == ((3,), (3,), (3,))


def test_finalization_failure_rolls_back_slot_token_identity_and_env_growth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = _owner()
    cfg = EnvConfig(width=4, height=4)
    cfg.population_size = 1
    cfg.max_agent_capacity_mult = 2
    owner.env = GridWorld(cfg, seed=7, rng=np.random.default_rng(7))
    parents = (_parent(1), _parent(2))
    allocation_before = _allocation_state(owner)
    env_arrays_before = (
        owner.env.agent_pos_y,
        owner.env.agent_pos_x,
        owner.env.agent_alive,
    )
    world_slot_before = owner.env._world_slot_snapshot(0)

    monkeypatch.setattr(spawn, "crossover_genome", lambda *args, **kwargs: parents[0].genome)
    monkeypatch.setattr(spawn, "crossover_policy", lambda *args, **kwargs: parents[0].policy)
    monkeypatch.setattr(spawn, "mutate_policy_weights", lambda *args, **kwargs: {})
    monkeypatch.setattr(spawn, "maybe_record_policy_lineage_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(spawn, "_init_drives_for_agent", lambda *args, **kwargs: None)

    def fail_memory_reset(slots: object) -> None:
        raise RuntimeError("memory reset failed")

    owner._reset_memory_slots = fail_memory_reset

    with pytest.raises(RuntimeError, match="memory reset failed"):
        owner.make_child_row_spec(*parents, child_energy=4.0)

    assert owner.next_id == 3
    assert _allocation_state(owner) == allocation_before
    assert owner.env.agent_pos_y is env_arrays_before[0]
    assert owner.env.agent_pos_x is env_arrays_before[1]
    assert owner.env.agent_alive is env_arrays_before[2]
    assert owner.env._world_slot_snapshot(0) == world_slot_before
    assert owner.env._agent_id_to_world_slot == {}

    owner._reset_memory_slots = lambda slots: None
    child = owner.make_child_row_spec(*parents, child_energy=4.0)
    assert child is not None
    assert (child.agent_id, child.slot, child.slot_token) == (3, 0, 1)
    assert owner.next_id == 4
    assert owner.env.agent_pos_y is env_arrays_before[0]
    assert owner.env._resolve_world_slot(3) == (0, 1)
    assert owner.env._agent_id_to_world_slot == {3: (0, 1)}


def test_gridworld_respec_high_id_validation_is_slot_bounded() -> None:
    env = _owner(respec_enabled=True).env
    before = {
        name: np.asarray(getattr(env, name)).copy()
        for name in ("agent_pos_y", "agent_pos_x", "agent_alive", *_RESPEC_ARRAY_NAMES)
    }

    StateMixin._ensure_agent_capacity(env, 1_000)

    shapes = {tuple(np.asarray(getattr(env, name)).shape) for name in before}
    assert shapes == {(3,)}
    for name, values in before.items():
        np.testing.assert_array_equal(np.asarray(getattr(env, name)), values)


def test_spawn_respec_high_id_validation_is_slot_bounded() -> None:
    owner = _owner(respec_enabled=True)

    allocation_before = _allocation_state(owner)
    owner._ensure_env_capacity_for_id(1_000)
    assert _allocation_state(owner) == allocation_before

    assert {
        tuple(np.asarray(getattr(owner.env, name)).shape)
        for name in ("agent_pos_y", "agent_pos_x", "agent_alive", *_RESPEC_ARRAY_NAMES)
    } == {(3,)}


def test_slot_pool_honors_explicit_hard_capacity() -> None:
    slots = SlotPool.build(base_population=256, slot_factor=6, max_slots_cap=256)

    assert slots.max_slots == 256
    assert len(slots.free_slots) == 256
    assert len(slots.slot_tokens) == 256


def test_slot_pool_preserves_larger_declared_capacity() -> None:
    slots = SlotPool.build(base_population=64, slot_factor=6, max_slots_cap=256)

    assert slots.max_slots == 256


def test_slot_pool_rejects_capacity_below_initial_population() -> None:
    with pytest.raises(ValueError, match="max_slots_cap must be >= base_population"):
        SlotPool.build(base_population=256, slot_factor=6, max_slots_cap=255)


def test_respec_high_id_birth_uses_reusable_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = _owner(respec_enabled=True)
    parents = (_parent(1), _parent(2))
    allocation_before = _allocation_state(owner)
    arrays_before = {
        name: getattr(owner.env, name)
        for name in ("agent_pos_y", "agent_pos_x", "agent_alive", *_RESPEC_ARRAY_NAMES)
    }

    monkeypatch.setattr(spawn, "crossover_genome", lambda *args, **kwargs: parents[0].genome)
    monkeypatch.setattr(spawn, "crossover_policy", lambda *args, **kwargs: parents[0].policy)
    monkeypatch.setattr(spawn, "mutate_policy_weights", lambda *args, **kwargs: {})
    monkeypatch.setattr(spawn, "maybe_record_policy_lineage_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(spawn, "_init_drives_for_agent", lambda *args, **kwargs: None)
    child = owner.make_child_row_spec(*parents, child_energy=4.0)

    assert child is not None
    assert (child.agent_id, child.slot, child.slot_token) == (3, 0, 1)
    assert owner.next_id == 4
    assert _allocation_state(owner) != allocation_before
    for name, value in arrays_before.items():
        assert getattr(owner.env, name) is value


def test_founder_configures_default_world_capacity_to_slotpool_capacity() -> None:
    owner = _owner()
    owner._slots = SlotPool.build(base_population=1, slot_factor=6, max_slots_cap=0)
    cfg = EnvConfig(width=4, height=4)
    cfg.population_size = 1
    cfg.max_agent_capacity_mult = 4
    owner.env = GridWorld(cfg, seed=11, rng=np.random.default_rng(11))
    owner.np_rng = np.random.default_rng(12)
    dense_shadow_shape = owner.env.agent_alive.shape

    agent = owner._spawn_random_agent()

    assert agent is not None
    assert (agent.agent_id, agent.slot, agent.slot_token) == (3, 0, 1)
    assert owner.next_id == 4
    assert owner.env.slot_to_agent_id.shape == (6,)
    assert owner.env.agent_pos_y_by_slot.shape == (6,)
    assert owner.env.agent_alive.shape == dense_shadow_shape == (4,)
    assert owner.env._resolve_world_slot(3) == (0, 1)


def test_world_capacity_configuration_honors_capped_slotpool() -> None:
    owner = _owner()
    owner._slots = SlotPool.build(base_population=2, slot_factor=6, max_slots_cap=6)
    cfg = EnvConfig(width=4, height=4)
    cfg.population_size = 2
    cfg.max_agent_capacity_mult = 4
    owner.env = GridWorld(cfg, seed=13, rng=np.random.default_rng(13))
    dense_shadow_shape = owner.env.agent_alive.shape

    snapshot = owner._reserve_env_world_slot(100_000, 0, 1)

    assert snapshot == (-1, 0, -1, -1, False)
    assert owner._slots.max_slots == 6
    assert owner.env.slot_to_agent_id.shape == (6,)
    assert owner.env.agent_alive.shape == dense_shadow_shape == (8,)
    assert owner.env._resolve_world_slot(100_000) == (0, 1)


def test_world_capacity_reconfiguration_after_membership_fails_closed() -> None:
    owner = _owner()
    owner._slots = SlotPool.build(base_population=1, slot_factor=6, max_slots_cap=0)
    cfg = EnvConfig(width=4, height=4)
    cfg.population_size = 1
    cfg.max_agent_capacity_mult = 4
    owner.env = GridWorld(cfg, seed=17, rng=np.random.default_rng(17))
    owner._reserve_env_world_slot(100_000, 0, 1)

    owner.env._configure_world_slot_capacity(6)
    with pytest.raises(RuntimeError, match="already configured"):
        owner.env._configure_world_slot_capacity(5)

    assert owner.env.slot_to_agent_id.shape == (6,)
    assert owner.env._resolve_world_slot(100_000) == (0, 1)

    dirty_env = GridWorld(cfg, seed=18, rng=np.random.default_rng(18))
    dirty_env.add_agent(0, pos=(1, 1))
    with pytest.raises(RuntimeError, match="pristine world state"):
        dirty_env._configure_world_slot_capacity(6)


def test_respec_founder_accepts_id_at_dense_shadow_boundary() -> None:
    owner = _owner(respec_enabled=True)
    owner.np_rng = np.random.default_rng(37)

    agent = owner._spawn_random_agent()

    assert agent is not None
    assert (agent.agent_id, agent.slot, agent.slot_token) == (3, 0, 1)
    assert owner.next_id == 4
    assert owner.agents == {3: agent}


def test_founder_construction_failure_does_not_consume_slot_or_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = _owner()
    allocation_before = _allocation_state(owner)
    monkeypatch.setattr(
        spawn.Genome,
        "random",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("genome failed")),
    )

    with pytest.raises(RuntimeError, match="genome failed"):
        owner._spawn_random_agent()

    assert owner.next_id == 3
    assert _allocation_state(owner) == allocation_before
    assert owner.agents == {}


def test_child_finalization_rollback_restores_gru_memory_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = _owner()
    cfg = EnvConfig(width=4, height=4)
    cfg.population_size = 1
    owner.env = GridWorld(cfg, seed=41, rng=np.random.default_rng(41))
    parents = (_parent(1), _parent(2))
    owner._gru_inited = True
    owner._gru_hidden = 3
    owner._bank_device = "cpu"
    owner._Hmem = torch.tensor([[1.25, -2.5, 3.75], [9.0, 8.0, 7.0]])
    memory_before = owner._Hmem[0].clone()

    monkeypatch.setattr(spawn, "crossover_genome", lambda *args, **kwargs: parents[0].genome)
    monkeypatch.setattr(spawn, "crossover_policy", lambda *args, **kwargs: parents[0].policy)
    monkeypatch.setattr(spawn, "mutate_policy_weights", lambda *args, **kwargs: {})
    monkeypatch.setattr(spawn, "_init_drives_for_agent", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        spawn,
        "maybe_record_policy_lineage_event",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("lineage failed")),
    )
    owner._reset_memory_slots = lambda slots: owner._Hmem.index_fill_(0, slots, 0.0)

    with pytest.raises(RuntimeError, match="lineage failed"):
        owner.make_child_row_spec(*parents, child_energy=4.0)

    assert torch.equal(owner._Hmem[0], memory_before)
    assert owner.next_id == 3
    assert _allocation_state(owner) == ((0, 1), (0, 0))


def test_remove_dead_unpublishes_before_same_slot_refill() -> None:
    owner = _owner()
    cfg = EnvConfig(width=4, height=4)
    cfg.population_size = 1
    cfg.max_agent_capacity_mult = 2
    owner.env = GridWorld(cfg, seed=21, rng=np.random.default_rng(21))
    old_slot, old_token = owner._slots.alloc()
    live_slot, live_token = owner._slots.alloc()
    assert (old_slot, old_token, live_slot, live_token) == (0, 1, 1, 1)

    dead = SimpleNamespace(agent_id=100_000, slot=0, slot_token=1, alive=False)
    live = SimpleNamespace(agent_id=100_001, slot=1, slot_token=1, alive=True)
    owner.agents = {dead.agent_id: dead, live.agent_id: live}
    owner._reserve_env_world_slot(dead.agent_id, dead.slot, dead.slot_token)
    owner._reserve_env_world_slot(live.agent_id, live.slot, live.slot_token)
    owner.env.add_agent(dead.agent_id, pos=(1, 1))
    owner.env.add_agent(live.agent_id, pos=(2, 2))

    Population.remove_dead(owner)

    assert dead.agent_id not in owner.agents
    assert int(owner.env.agent_grid[1, 1]) == -1
    assert dead.agent_id not in owner.env._agent_id_to_world_slot
    slot, token = owner._slots.alloc()
    assert (slot, token) == (0, 2)

    owner._reserve_env_world_slot(100_002, slot, token)
    owner.env.add_agent(100_002, pos=(1, 1))
    with pytest.raises(RuntimeError, match="stale world removal identity"):
        owner.env.remove_agent(dead.agent_id, slot=old_slot, slot_token=old_token)
    assert owner.env._resolve_world_slot(100_002, require_alive=True) == (0, 2)


def test_remove_dead_unpublish_failure_does_not_free_or_pop() -> None:
    owner = _owner()
    slot, token = owner._slots.alloc()
    dead = SimpleNamespace(agent_id=3, slot=slot, slot_token=token, alive=False)
    owner.agents = {dead.agent_id: dead}
    allocation_before = _allocation_state(owner)

    class FailingEnv:
        @staticmethod
        def remove_agent(
            agent_id: int,
            *,
            slot: int,
            slot_token: int,
        ) -> None:
            raise RuntimeError(
                f"unpublish failed: {agent_id}/{slot}/{slot_token}"
            )

    owner.env = FailingEnv()
    with pytest.raises(RuntimeError, match="unpublish failed"):
        Population.remove_dead(owner)

    assert owner.agents == {dead.agent_id: dead}
    assert _allocation_state(owner) == allocation_before


def test_remove_dead_accepts_exact_tick_unpublish_witness_only() -> None:
    owner = _owner()
    cfg = EnvConfig(width=3, height=3)
    cfg.population_size = 1
    cfg.max_agent_capacity_mult = 2
    owner.env = GridWorld(cfg, seed=23, rng=np.random.default_rng(23))
    slot, token = owner._slots.alloc()
    dead = SimpleNamespace(
        agent_id=100_000,
        slot=slot,
        slot_token=token,
        alive=False,
    )
    owner.agents = {dead.agent_id: dead}
    owner._reserve_env_world_slot(dead.agent_id, slot, token)
    owner.env.add_agent(dead.agent_id, pos=(1, 1))
    owner.env.remove_agent(dead.agent_id, slot=slot, slot_token=token)

    with pytest.raises(RuntimeError, match="stale world removal identity"):
        owner.env.remove_agent(dead.agent_id, slot=slot, slot_token=token + 1)

    Population.remove_dead(owner)

    assert owner.agents == {}
    assert dead.agent_id not in owner.env._agent_id_to_world_slot
    assert tuple(owner._slots.free_slots) == (1, 0)
    assert tuple(owner._slots.slot_tokens) == (1, 0)


def test_envsync_places_and_reconciles_high_ids_with_reversed_slots() -> None:
    slots = SlotPool.build(base_population=1, slot_factor=2, max_slots_cap=0)
    slot0, token0 = slots.alloc()
    slot1, token1 = slots.alloc()
    agent_a = SimpleNamespace(
        agent_id=100_001,
        slot=slot1,
        slot_token=token1,
        alive=True,
    )
    agent_b = SimpleNamespace(
        agent_id=100_000,
        slot=slot0,
        slot_token=token0,
        alive=True,
    )
    pop = SimpleNamespace(
        _slots=slots,
        agents={agent_a.agent_id: agent_a, agent_b.agent_id: agent_b},
    )
    pop.alive_agents = lambda: sorted(
        (agent for agent in pop.agents.values() if agent.alive),
        key=lambda agent: agent.agent_id,
    )
    cfg = EnvConfig(width=4, height=4)
    cfg.population_size = 1
    cfg.max_agent_capacity_mult = 4
    runner = EnvSyncMixin()
    runner.pop = pop
    runner.env = GridWorld(cfg, seed=25, rng=np.random.default_rng(25))
    runner.sim_cfg = SimpleNamespace()

    runner._place_agents()

    assert runner.env._resolve_world_slot(100_001, require_alive=True) == (1, 1)
    assert runner.env._resolve_world_slot(100_000, require_alive=True) == (0, 1)
    assert runner.env.agent_alive.shape == (4,)
    agent_b.alive = False
    runner._sync_env_with_population()
    assert 100_000 not in runner.env._agent_id_to_world_slot
    assert runner.env._resolve_world_slot(100_001, require_alive=True) == (1, 1)


def test_envsync_add_failure_preserves_population_and_slot_ownership() -> None:
    slots = SlotPool.build(base_population=1, slot_factor=2, max_slots_cap=0)
    slot0, token0 = slots.alloc()
    first = SimpleNamespace(
        agent_id=100_000,
        slot=slot0,
        slot_token=token0,
        alive=True,
    )
    pop = SimpleNamespace(_slots=slots, agents={first.agent_id: first})
    pop.alive_agents = lambda: list(
        agent for agent in pop.agents.values() if agent.alive
    )
    cfg = EnvConfig(width=1, height=1)
    cfg.population_size = 1
    runner = EnvSyncMixin()
    runner.pop = pop
    runner.env = GridWorld(cfg, seed=27, rng=np.random.default_rng(27))
    runner.sim_cfg = SimpleNamespace()
    runner._place_agents()

    slot1, token1 = slots.alloc()
    second = SimpleNamespace(
        agent_id=100_001,
        slot=slot1,
        slot_token=token1,
        alive=True,
    )
    pop.agents[second.agent_id] = second
    allocation_before = _allocation_state(pop)

    with pytest.raises(RuntimeError, match="No empty cells available"):
        runner._sync_env_with_population()

    assert pop.agents[second.agent_id] is second
    assert second.alive is True
    assert _allocation_state(pop) == allocation_before
    assert second.agent_id not in runner.env._agent_id_to_world_slot


def test_envsync_rejects_occupied_grid_without_slot_authority() -> None:
    slots = SlotPool.build(base_population=1, slot_factor=2, max_slots_cap=0)
    slot0, token0 = slots.alloc()
    slot1, token1 = slots.alloc()
    agents = {
        100_001: SimpleNamespace(
            agent_id=100_001,
            slot=slot1,
            slot_token=token1,
            alive=True,
        ),
        100_000: SimpleNamespace(
            agent_id=100_000,
            slot=slot0,
            slot_token=token0,
            alive=True,
        ),
    }
    pop = SimpleNamespace(_slots=slots, agents=agents)
    pop.alive_agents = lambda: sorted(
        (agent for agent in pop.agents.values() if agent.alive),
        key=lambda agent: agent.agent_id,
    )
    cfg = EnvConfig(width=3, height=3)
    cfg.population_size = 1
    runner = EnvSyncMixin()
    runner.pop = pop
    runner.env = GridWorld(cfg, seed=29, rng=np.random.default_rng(29))
    runner.env.agent_grid[0, 1] = 100_001
    runner.env.agent_grid[2, 1] = 100_000

    with pytest.raises(RuntimeError, match="world_state_v2_to_v3 migration"):
        runner._sync_env_with_population()

    assert runner.env._agent_id_to_world_slot == {}
    assert runner.env._world_authority_migration_required is True
    with pytest.raises(RuntimeError, match="world_state_v2_to_v3 migration"):
        runner.env._resolve_world_slot(100_001)
    assert int(runner.env.agent_grid[0, 1]) == 100_001
    assert int(runner.env.agent_grid[2, 1]) == 100_000


def _world_checkpoint_fixture(
    agent_ids: tuple[int, int],
) -> tuple[SimpleNamespace, dict[str, object], dict[str, object]]:
    cfg = EnvConfig(width=3, height=3)
    cfg.population_size = 1
    cfg.max_agent_capacity_mult = 4
    env = GridWorld(cfg, seed=31, rng=np.random.default_rng(31))
    env._configure_world_slot_capacity(2)
    env.add_agent(agent_ids[0], pos=(0, 1), slot=1, slot_token=1)
    env.add_agent(agent_ids[1], pos=(2, 1), slot=0, slot_token=1)
    sim = SimpleNamespace(env=env)
    population = {
        "schema_version": "population_state_v2",
        "next_id": max(agent_ids) + 1,
        "agents": [
            {
                "agent_id": agent_ids[0],
                "slot": 1,
                "slot_token": 1,
                "alive": True,
            },
            {
                "agent_id": agent_ids[1],
                "slot": 0,
                "slot_token": 1,
                "alive": True,
            },
        ],
        "slot_pool": {
            "max_slots": 2,
            "free_slots": [],
            "slot_tokens": [1, 1],
        },
    }
    return sim, _export_world(sim), population


def test_world_state_v3_sparse_high_ids_keep_fixed_capacity() -> None:
    sim, world, population = _world_checkpoint_fixture((100_001, 100_000))

    normalized = _normalize_checkpoint_world_authority(sim, world, population)
    binding = normalized["world_membership_authority"]

    assert normalized["schema_version"] == "world_state_v3"
    assert binding["capacity"] == 2
    assert binding["slot_to_agent_id"] == [100_000, 100_001]
    assert len(binding["slot_tokens"]) == 2
    assert sim.env.agent_alive.shape == (4,)


def test_world_state_v2_strict_migration_materializes_reversed_slots() -> None:
    sim, world_v3, population = _world_checkpoint_fixture((0, 1))
    world_v2 = copy.deepcopy(world_v3)
    world_v2["schema_version"] = "world_state_v2"
    world_v2.pop("world_membership_authority")

    normalized = _normalize_checkpoint_world_authority(sim, world_v2, population)
    binding = normalized["world_membership_authority"]

    assert binding["slot_to_agent_id"] == [1, 0]
    assert binding["world_alive_by_slot"] == [True, True]
    assert binding["pos_y_by_slot"] == [2, 0]
    assert binding["pos_x_by_slot"] == [1, 1]


def test_world_state_v2_respec_cursor_migration_joins_reversed_slots() -> None:
    sim, world_v3, population = _world_checkpoint_fixture((0, 1))
    sim.env.cfg.respec0b_resource_specialization_enable = True
    world_v2 = copy.deepcopy(world_v3)
    world_v2["schema_version"] = "world_state_v2"
    world_v2.pop("world_membership_authority")
    world_v2["patch_world"] = {
        "respec0b_resource_handling_state": {
            "active_y": [0, 2, -1, -1],
            "active_x": [1, 1, -1, -1],
            "active_class": [1, 2, 0, 0],
            "progress": [0.25, 0.5, 0.0, 0.0],
        }
    }

    normalized = _normalize_checkpoint_world_authority(sim, world_v2, population)
    cursor = normalized["patch_world"]["respec0b_resource_handling_state"]

    assert cursor["schema_version"] == "respec0b_slot_cursor_state_v1"
    assert cursor["slot_tokens"] == [1, 1]
    assert cursor["active_y"] == [2, 0]
    assert cursor["active_x"] == [1, 1]
    assert cursor["active_class"] == [2, 1]
    assert cursor["progress"] == pytest.approx([0.5, 0.25])


def test_world_state_v3_respec_cursor_rejects_slot_token_contradiction() -> None:
    sim, world, population = _world_checkpoint_fixture((100_001, 100_000))
    sim.env.cfg.respec0b_resource_specialization_enable = True
    world["patch_world"] = {
        "respec0b_resource_handling_state": {
            "schema_version": "respec0b_slot_cursor_state_v1",
            "slot_tokens": [2, 1],
            "active_y": [-1, -1],
            "active_x": [-1, -1],
            "active_class": [0, 0],
            "progress": [0.0, 0.0],
        }
    }

    with pytest.raises(CheckpointStrictError, match="tokens do not match SlotPool"):
        _normalize_checkpoint_world_authority(sim, world, population)


def test_world_state_v2_migration_rejects_dense_position_contradiction() -> None:
    sim, world_v3, population = _world_checkpoint_fixture((0, 1))
    world_v2 = copy.deepcopy(world_v3)
    world_v2["schema_version"] = "world_state_v2"
    world_v2.pop("world_membership_authority")
    world_v2["agent_pos_y"][0] = 2

    with pytest.raises(CheckpointStrictError, match="dense position mismatch"):
        _normalize_checkpoint_world_authority(sim, world_v2, population)


def test_world_state_v3_rejects_population_slot_layout_and_token_mismatch() -> None:
    sim, world, population = _world_checkpoint_fixture((100_001, 100_000))
    wrong_layout = copy.deepcopy(population)
    wrong_layout["agents"][0]["slot"] = 0
    wrong_layout["agents"][1]["slot"] = 1
    with pytest.raises(CheckpointStrictError, match="slot binding mismatch"):
        _normalize_checkpoint_world_authority(sim, world, wrong_layout)

    wrong_token = copy.deepcopy(population)
    wrong_token["agents"][0]["slot_token"] = 2
    with pytest.raises(CheckpointStrictError, match="slot token mismatch"):
        _normalize_checkpoint_world_authority(sim, world, wrong_token)


def test_world_state_v3_rejects_dense_only_and_unknown_schema() -> None:
    sim, world, population = _world_checkpoint_fixture((0, 1))
    dense_only = copy.deepcopy(world)
    dense_only.pop("world_membership_authority")
    with pytest.raises(CheckpointStrictError, match="membership authority is missing"):
        _normalize_checkpoint_world_authority(sim, dense_only, population)

    unknown = copy.deepcopy(world)
    unknown["schema_version"] = "world_state_v99"
    with pytest.raises(CheckpointStrictError, match="unsupported world checkpoint schema"):
        _normalize_checkpoint_world_authority(sim, unknown, population)


@pytest.mark.parametrize("value", [True, 1.0, "1"])
def test_checkpoint_identity_rejects_coercive_slot_values(value: object) -> None:
    sim, world, population = _world_checkpoint_fixture((100_001, 100_000))
    population["agents"][0]["slot"] = value

    with pytest.raises(CheckpointStrictError, match="must be an exact integer"):
        _normalize_checkpoint_world_authority(sim, world, population)


@pytest.mark.parametrize("value", [1, 1.0, "true"])
def test_checkpoint_identity_rejects_non_boolean_alive_values(value: object) -> None:
    sim, world, population = _world_checkpoint_fixture((100_001, 100_000))
    population["agents"][0]["alive"] = value

    with pytest.raises(CheckpointStrictError, match="must be an exact boolean"):
        _normalize_checkpoint_world_authority(sim, world, population)


def test_checkpoint_world_shadow_rejects_integer_as_boolean() -> None:
    sim, world, population = _world_checkpoint_fixture((100_001, 100_000))
    world["agent_alive"][0] = 0

    with pytest.raises(CheckpointStrictError, match="must be a boolean array"):
        _normalize_checkpoint_world_authority(sim, world, population)


def test_checkpoint_object_snapshot_restores_in_place_tensor_bytes() -> None:
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    owner = SimpleNamespace(bank=tensor)
    snapshot = checkpoint_bundle._snapshot_checkpoint_object(owner)

    tensor.add_(10.0)
    checkpoint_bundle._restore_checkpoint_object(owner, snapshot)

    assert owner.bank is tensor
    assert torch.equal(tensor, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))


def test_checkpoint_transaction_rolls_back_world_population_and_tensor_on_scalar_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_tensor = torch.tensor([2.0, 4.0])
    pop_tensor = torch.tensor([5.0, 7.0])
    original_agent = object()
    grid = np.asarray([[1, 2]], dtype=np.int32)
    sim = SimpleNamespace(
        env=SimpleNamespace(
            grid=grid,
            bank=env_tensor,
            rng=np.random.default_rng(51),
        ),
        pop=SimpleNamespace(
            agents={11: original_agent},
            bank=pop_tensor,
            rng=np.random.default_rng(52),
        ),
    )

    def mutate_population(
        target: object,
        population: object,
        *,
        weights_dir: str,
    ) -> None:
        del population, weights_dir
        target.pop.agents.clear()
        target.pop.bank.mul_(3.0)

    def mutate_world(target: object, world: object) -> None:
        del world
        target.env.grid[:] = 99
        target.env.bank.add_(8.0)

    def fail_scalars(target: object, resident: object) -> None:
        del resident
        target.env.food_spawn_mult = 9.0
        raise RuntimeError("resident scalar failed")

    monkeypatch.setattr(checkpoint_bundle, "_apply_population", mutate_population)
    monkeypatch.setattr(checkpoint_bundle, "_apply_world", mutate_world)
    monkeypatch.setattr(
        checkpoint_bundle,
        "_apply_resident_environment_scalars",
        fail_scalars,
    )

    with pytest.raises(RuntimeError, match="resident scalar failed"):
        checkpoint_bundle._apply_python_checkpoint_transaction(
            sim,
            world={},
            population={},
            resident={},
            weights_dir="",
        )

    assert sim.pop.agents == {11: original_agent}
    assert sim.env.grid is grid
    np.testing.assert_array_equal(sim.env.grid, np.asarray([[1, 2]], dtype=np.int32))
    assert sim.env.bank is env_tensor
    assert sim.pop.bank is pop_tensor
    assert torch.equal(env_tensor, torch.tensor([2.0, 4.0]))
    assert torch.equal(pop_tensor, torch.tensor([5.0, 7.0]))
    assert not hasattr(sim.env, "food_spawn_mult")


def test_native_reference_snapshot_covers_runner_arrays_rngs_and_nested_indices() -> None:
    runner_tensor = torch.tensor([1.0, 3.0])
    drive_array = np.asarray([2.0, 4.0], dtype=np.float32)
    env_rng = np.random.default_rng(61)
    runner_rng = np.random.default_rng(62)
    population_rng = random.Random(63)
    rng_manager = SimpleNamespace(_indices={"birth": 7})
    sim = SimpleNamespace(
        env=SimpleNamespace(rng=env_rng),
        pop=SimpleNamespace(rng=population_rng),
        np_rng=runner_rng,
        rng_manager=rng_manager,
        _Hmem=runner_tensor,
        _drive_hunger=drive_array,
    )
    env_state = copy.deepcopy(env_rng.bit_generator.state)
    runner_state = copy.deepcopy(runner_rng.bit_generator.state)
    population_state = population_rng.getstate()
    snapshot = checkpoint_bundle._snapshot_native_reference_install(sim)

    runner_tensor.zero_()
    drive_array[:] = 99.0
    env_rng.random()
    runner_rng.random()
    population_rng.random()
    rng_manager._indices["birth"] = 99
    sim.transient_install_marker = True
    checkpoint_bundle._restore_native_reference_install(sim, snapshot)

    assert sim._Hmem is runner_tensor
    assert sim._drive_hunger is drive_array
    assert torch.equal(runner_tensor, torch.tensor([1.0, 3.0]))
    np.testing.assert_array_equal(drive_array, np.asarray([2.0, 4.0], dtype=np.float32))
    assert env_rng.bit_generator.state == env_state
    assert runner_rng.bit_generator.state == runner_state
    assert population_rng.getstate() == population_state
    assert rng_manager._indices == {"birth": 7}
    assert not hasattr(sim, "transient_install_marker")
