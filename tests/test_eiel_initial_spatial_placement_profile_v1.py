from __future__ import annotations

from types import SimpleNamespace

import pytest

from engine.simulation.runner_pkg.runner_envsync import (
    EnvSyncMixin,
    _initial_spatial_positions_v1,
)


def test_dispersed_lattice_profile_spans_16x16_without_overlap() -> None:
    positions = _initial_spatial_positions_v1(
        "dispersed_lattice_v1",
        width=16,
        height=16,
        count=64,
    )

    assert positions is not None
    assert len(positions) == 64
    assert len(set(positions)) == 64
    assert positions[:4] == ((0, 0), (0, 2), (0, 4), (0, 6))
    assert max(y for y, _ in positions) == 14
    assert max(x for _, x in positions) == 14


def test_dispersed_lattice_fill_profile_preserves_first64_and_fills_p192() -> None:
    first64 = _initial_spatial_positions_v1(
        "dispersed_lattice_v1",
        width=16,
        height=16,
        count=64,
    )
    positions = _initial_spatial_positions_v1(
        "dispersed_lattice_fill_v1",
        width=16,
        height=16,
        count=192,
    )

    assert positions is not None
    assert first64 is not None
    assert len(positions) == 192
    assert len(set(positions)) == 192
    assert positions[:64] == first64
    assert max(x for _, x in positions) == 15
    assert all(0 <= y < 16 and 0 <= x < 16 for y, x in positions)


def test_clustered_quadrant_profile_uses_contiguous_8x8_region() -> None:
    positions = _initial_spatial_positions_v1(
        "clustered_quadrant_v1",
        width=16,
        height=16,
        count=64,
    )

    assert positions is not None
    assert len(positions) == 64
    assert len(set(positions)) == 64
    assert positions[:4] == ((0, 0), (0, 1), (0, 2), (0, 3))
    assert max(y for y, _ in positions) == 7
    assert max(x for _, x in positions) == 7


def test_unknown_initial_spatial_placement_profile_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="unknown initial spatial placement"):
        _initial_spatial_positions_v1("unknown", width=16, height=16, count=64)


def test_envsync_applies_profile_in_stable_agent_id_order() -> None:
    class FakeEnv:
        def __init__(self) -> None:
            self.cfg = SimpleNamespace(width=16, height=16)
            self.agent_grid = [[-1]]
            self.calls: list[tuple[int, int, int, tuple[int, int] | None]] = []

        def reset(self) -> None:
            self.calls.clear()

        def _configure_world_slot_capacity(self, capacity: int) -> None:
            self.capacity = capacity

        def add_agent(
            self,
            agent_id: int,
            *,
            slot: int,
            slot_token: int,
            pos: tuple[int, int] | None = None,
        ) -> None:
            self.calls.append((agent_id, slot, slot_token, pos))

    class FakePop:
        def __init__(self) -> None:
            self._slots = SimpleNamespace(max_slots=256)
            self._agents = [
                SimpleNamespace(agent_id=3, slot=30, slot_token=1),
                SimpleNamespace(agent_id=1, slot=10, slot_token=1),
                SimpleNamespace(agent_id=2, slot=20, slot_token=1),
            ]

        def alive_agents(self):
            return list(self._agents)

    class FakeRunner(EnvSyncMixin):
        pass

    runner = FakeRunner()
    runner.env = FakeEnv()
    runner.pop = FakePop()
    runner.sim_cfg = SimpleNamespace(
        initial_spatial_placement_profile="clustered_quadrant_v1"
    )

    runner._place_agents()

    assert runner.env.capacity == 256
    assert runner.env.calls == [
        (1, 10, 1, (0, 0)),
        (2, 20, 1, (0, 1)),
        (3, 30, 1, (1, 0)),
    ]


def test_heritage_profile_preserves_random_env_placement_path() -> None:
    positions = _initial_spatial_positions_v1(
        "heritage_random_v1",
        width=16,
        height=16,
        count=64,
    )

    assert positions is None


def test_envsync_heritage_profile_preserves_alive_agent_iteration_order() -> None:
    class FakeEnv:
        def __init__(self) -> None:
            self.cfg = SimpleNamespace(width=16, height=16)
            self.agent_grid = [[-1]]
            self.calls: list[tuple[int, int, int, tuple[int, int] | None]] = []

        def reset(self) -> None:
            self.calls.clear()

        def _configure_world_slot_capacity(self, capacity: int) -> None:
            self.capacity = capacity

        def add_agent(
            self,
            agent_id: int,
            *,
            slot: int,
            slot_token: int,
            pos: tuple[int, int] | None = None,
        ) -> None:
            self.calls.append((agent_id, slot, slot_token, pos))

    class FakePop:
        def __init__(self) -> None:
            self._slots = SimpleNamespace(max_slots=256)
            self._agents = [
                SimpleNamespace(agent_id=3, slot=30, slot_token=1),
                SimpleNamespace(agent_id=1, slot=10, slot_token=1),
                SimpleNamespace(agent_id=2, slot=20, slot_token=1),
            ]

        def alive_agents(self):
            return list(self._agents)

    class FakeRunner(EnvSyncMixin):
        pass

    runner = FakeRunner()
    runner.env = FakeEnv()
    runner.pop = FakePop()
    runner.sim_cfg = SimpleNamespace(
        initial_spatial_placement_profile="heritage_random_v1"
    )

    runner._place_agents()

    assert runner.env.calls == [
        (3, 30, 1, None),
        (1, 10, 1, None),
        (2, 20, 1, None),
    ]
