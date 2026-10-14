from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from simulation.runnercore.tick_state import (
    bind_agent_id_to_slot,
    import_generation_state,
    legacy_dense_agent_id_to_slot,
    replace_agent_id_to_slot_index,
    slots_for_agent_ids,
)


HIGH_AGENT_ID = 1_000_000_007


def _runner(agent_ids: tuple[int, ...]) -> SimpleNamespace:
    agents = [
        SimpleNamespace(
            agent_id=agent_id,
            slot=index,
            slot_token=3,
            alive=True,
            y=index + 1,
            x=index + 2,
            energy=10.0,
            health=5.0,
            injury=0.0,
            scar=0.0,
            drives=np.zeros((4,), dtype=np.float32),
            genome=SimpleNamespace(use_gru=False, memory_dim=0),
        )
        for index, agent_id in enumerate(agent_ids)
    ]
    return SimpleNamespace(
        pop=SimpleNamespace(
            max_slots=4,
            alive_agents=lambda: list(agents),
        ),
        env=SimpleNamespace(
            agent_pos_y=np.asarray([7, 8, -1, -1], dtype=np.int32),
            agent_pos_x=np.asarray([9, 10, -1, -1], dtype=np.int32),
            agent_alive=np.asarray([True, True, False, False], dtype=np.bool_),
        ),
        env_cfg=SimpleNamespace(),
        sim_cfg=SimpleNamespace(policy_tick_stride=1),
        _policy_action_cache={},
    )


def test_high_agent_id_sparse_reverse_lookup_is_resident_bounded() -> None:
    state = import_generation_state(
        _runner((7, HIGH_AGENT_ID)),
        generation=0,
        profile="runnercore1",
    )

    assert state.capacity == 4
    assert state.agent_id_to_slot_agent_ids.tolist() == [7, HIGH_AGENT_ID]
    assert state.agent_id_to_slot_slots.tolist() == [0, 1]
    assert state.agent_id_to_slot_agent_ids.shape == (2,)
    assert state.agent_id_to_slot_slots.shape == (2,)
    assert (
        state.agent_id_to_slot_agent_ids.nbytes
        + state.agent_id_to_slot_slots.nbytes
        == 24
    )
    assert state.slot_for_agent_id(HIGH_AGENT_ID) == 1
    assert state.slot_for_agent_id(HIGH_AGENT_ID - 1) == -1
    np.testing.assert_array_equal(
        slots_for_agent_ids(state, [HIGH_AGENT_ID, 7, -1, 99]),
        np.asarray([1, 0, -1, -1], dtype=np.int32),
    )

    bind_agent_id_to_slot(state, HIGH_AGENT_ID + 1, 1)

    assert state.agent_id_to_slot_agent_ids.tolist() == [7, HIGH_AGENT_ID + 1]
    assert state.slot_for_agent_id(HIGH_AGENT_ID) == -1
    assert state.slot_for_agent_id(HIGH_AGENT_ID + 1) == 1
    assert state.agent_id_to_slot_agent_ids.shape == (2,)


def test_legacy_dense_reverse_map_is_read_into_sparse_pairs() -> None:
    state = import_generation_state(
        _runner((0, 1)),
        generation=0,
        profile="runnercore1",
    )

    replace_agent_id_to_slot_index(
        state,
        np.asarray([-1, 3, -1, 1], dtype=np.int32),
    )

    assert state.agent_id_to_slot_agent_ids.tolist() == [1, 3]
    assert state.agent_id_to_slot_slots.tolist() == [3, 1]
    assert state.slot_for_agent_id(3) == 1


def test_legacy_native_dense_adapter_fails_closed_for_high_ids() -> None:
    state = import_generation_state(
        _runner((7, HIGH_AGENT_ID)),
        generation=0,
        profile="runnercore1",
    )

    with pytest.raises(ValueError, match="bounded compatibility width"):
        legacy_dense_agent_id_to_slot(state, max_width=65_536)
