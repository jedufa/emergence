from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from perf.cuda_architecture_b_asexual_birth_plan import (
    execute_asexual_birth_plan_cuda,
)
from perf.cuda_architecture_b_lifecycle import (
    execute_lifecycle_death_cull_cuda,
    release_pending_free_slots_cuda,
)
from perf.cuda_architecture_b_pcg64 import pcg64_state_tensors


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.fixture(scope="module")
def build_directories(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    return {
        "lifecycle": tmp_path_factory.mktemp("archb-death-class-lifecycle"),
        "birth_plan": tmp_path_factory.mktemp("archb-death-class-birth-plan"),
    }


def _resident_state(*, poison_inactive_tails: bool) -> dict[str, torch.Tensor]:
    capacity = 7
    slot_to_agent_id = torch.tensor(
        [10, 90, 50, 20, 70, 30, -1], dtype=torch.int64
    )
    alive_mask = slot_to_agent_id.ge(0)
    pos_y = torch.tensor([1, 0, 0, 3, 3, 2, -1], dtype=torch.int32)
    pos_x = torch.tensor([1, 0, 3, 3, 0, 3, -1], dtype=torch.int32)
    agent_id_to_slot = torch.full((128,), -1, dtype=torch.int32)
    agent_grid = torch.full((4, 4), -1, dtype=torch.int32)
    for slot in range(capacity):
        agent_id = int(slot_to_agent_id[slot])
        if not bool(alive_mask[slot]):
            continue
        agent_id_to_slot[agent_id] = slot
        agent_grid[int(pos_y[slot]), int(pos_x[slot])] = agent_id

    pending_tail = (
        [6, 3, 1, 5, 2, 0, 4] if poison_inactive_tails else [-1] * capacity
    )
    queue_tail = (
        [6, 5, 2, 3, 1, 0, 4] if poison_inactive_tails else [6] + [-1] * 6
    )
    state = {
        "slot_to_agent_id": slot_to_agent_id,
        "agent_id_to_slot": agent_id_to_slot,
        "alive_mask": alive_mask,
        "pos_y": pos_y,
        "pos_x": pos_x,
        "agent_grid": agent_grid,
        # Slots 3/1 are action-boundary health/energy deaths, respectively.
        "energy": torch.tensor(
            [60.0, 0.0, 40.0, 40.0, 40.0, 40.0, 0.0], dtype=torch.float32
        ),
        "health": torch.tensor(
            [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0], dtype=torch.float32
        ),
        "injury": torch.zeros((capacity,), dtype=torch.float32),
        "scar": torch.zeros((capacity,), dtype=torch.float32),
        "policy_energy_observation": torch.tensor(
            [60.0, 40.0, 40.0, 40.0, 40.0, 40.0, torch.nan],
            dtype=torch.float64,
        ),
        "rest_streak": torch.arange(1, capacity + 1, dtype=torch.int32),
        "action_cache_token": torch.arange(100, 100 + capacity, dtype=torch.int64),
        "action_cache_cursor": torch.arange(10, 10 + capacity, dtype=torch.int32),
        "action_cache_next": torch.arange(20, 20 + capacity, dtype=torch.int32),
        "pending_free_agent_ids": torch.tensor(pending_tail, dtype=torch.int64),
        "pending_free_slots": torch.tensor(pending_tail, dtype=torch.int64),
        "pending_free_count": torch.zeros((1,), dtype=torch.int64),
        "slot_free_queue": torch.tensor(queue_tail, dtype=torch.int64),
        "slot_free_queue_count": torch.ones((1,), dtype=torch.int64),
        "policy_use_gru": torch.zeros((capacity,), dtype=torch.bool),
        "repro_cooldown": torch.zeros((capacity,), dtype=torch.int32),
        "repro_threshold": torch.full((capacity,), 100.0, dtype=torch.float32),
    }
    state["repro_threshold"][0] = 20.0
    return {name: tensor.cuda().contiguous() for name, tensor in state.items()}


def _death_cull(
    state: dict[str, torch.Tensor],
    *,
    build_directory: Path,
) -> dict[str, torch.Tensor]:
    return execute_lifecycle_death_cull_cuda(
        slot_to_agent_id=state["slot_to_agent_id"],
        agent_id_to_slot=state["agent_id_to_slot"],
        alive_mask=state["alive_mask"],
        pos_y=state["pos_y"],
        pos_x=state["pos_x"],
        agent_grid=state["agent_grid"],
        energy=state["energy"],
        health=state["health"],
        policy_energy_observation=state["policy_energy_observation"],
        rest_streak=state["rest_streak"],
        action_cache_token=state["action_cache_token"],
        action_cache_cursor=state["action_cache_cursor"],
        action_cache_next=state["action_cache_next"],
        pending_free_agent_ids=state["pending_free_agent_ids"],
        pending_free_slots=state["pending_free_slots"],
        pending_free_count=state["pending_free_count"],
        build_directory=build_directory,
    )


def _ordinary_birth_plan(
    state: dict[str, torch.Tensor],
    *,
    build_directory: Path,
) -> dict[str, torch.Tensor]:
    bit_generator = np.random.PCG64(4242)
    state_words, buffer_words = pcg64_state_tensors(
        copy.deepcopy(bit_generator.state), device="cuda"
    )
    runner = np.random.PCG64(4343)
    runner_state_words, runner_buffer_words = pcg64_state_tensors(
        copy.deepcopy(runner.state), device="cuda"
    )
    return execute_asexual_birth_plan_cuda(
        slot_to_agent_id=state["slot_to_agent_id"],
        alive_mask=state["alive_mask"],
        policy_use_gru=state["policy_use_gru"],
        pos_y=state["pos_y"],
        pos_x=state["pos_x"],
        agent_grid=state["agent_grid"],
        energy=state["energy"],
        policy_energy_observation=state["policy_energy_observation"],
        health=state["health"],
        injury=state["injury"],
        scar=state["scar"],
        repro_cooldown=state["repro_cooldown"],
        repro_threshold=state["repro_threshold"],
        slot_free_queue=state["slot_free_queue"],
        slot_free_queue_count=state["slot_free_queue_count"],
        slot_free_queue_head=torch.zeros((1,), dtype=torch.int64, device="cuda"),
        pcg64_state_words=state_words,
        pcg64_buffer_words=buffer_words,
        policy_np_semantic_draws=torch.zeros(
            (1,), dtype=torch.int64, device="cuda"
        ),
        runner_pcg64_state_words=runner_state_words,
        runner_pcg64_buffer_words=runner_buffer_words,
        metrics_np_semantic_draws=torch.zeros(
            (1,), dtype=torch.int64, device="cuda"
        ),
        parent_energy_share=0.25,
        min_energy_buffer=2.0,
        max_births=1,
        build_directory=build_directory,
    )


def _run_two_death_classes(
    *,
    poison_inactive_tails: bool,
    build_directories: dict[str, Path],
) -> dict[str, Any]:
    state = _resident_state(poison_inactive_tails=poison_inactive_tails)

    action = _death_cull(state, build_directory=build_directories["lifecycle"])
    pending_release_cursor = state["pending_free_count"].clone()
    assert pending_release_cursor.is_cuda
    saved_cursor = int(pending_release_cursor.cpu()[0])

    birth = _ordinary_birth_plan(
        state,
        build_directory=build_directories["birth_plan"],
    )

    # Slots 5/2 become residual health/energy deaths after ordinary birth planning.
    state["health"][5] = 0.0
    state["energy"][2] = 0.0
    residual = _death_cull(state, build_directory=build_directories["lifecycle"])
    released = release_pending_free_slots_cuda(
        slot_to_agent_id=state["slot_to_agent_id"],
        alive_mask=state["alive_mask"],
        slot_free_queue=state["slot_free_queue"],
        slot_free_queue_count=state["slot_free_queue_count"],
        pending_free_slots=state["pending_free_slots"],
        pending_free_count=state["pending_free_count"],
        pending_release_cursor=pending_release_cursor,
        build_directory=build_directories["lifecycle"],
    )
    torch.cuda.synchronize()

    action_count = int(action["death_count"].cpu()[0])
    residual_count = int(residual["death_count"].cpu()[0])
    queue_count = int(state["slot_free_queue_count"].cpu()[0])
    pending_count = int(state["pending_free_count"].cpu()[0])
    return {
        "action_agent_ids": action["death_agent_ids"].cpu()[:action_count].tolist(),
        "action_slots": action["death_slots"].cpu()[:action_count].tolist(),
        "action_reasons": action["death_reason_codes"].cpu()[:action_count].tolist(),
        "saved_cursor": saved_cursor,
        "birth_status": int(birth["status"].cpu()[0]),
        "birth_planned_count": int(birth["planned_count"].cpu()[0]),
        "birth_child_slots": birth["child_slots"].cpu()[:1].tolist(),
        "residual_agent_ids": residual["death_agent_ids"]
        .cpu()[:residual_count]
        .tolist(),
        "residual_slots": residual["death_slots"].cpu()[:residual_count].tolist(),
        "residual_reasons": residual["death_reason_codes"]
        .cpu()[:residual_count]
        .tolist(),
        "pending_agent_ids": state["pending_free_agent_ids"]
        .cpu()[:pending_count]
        .tolist(),
        "pending_slots": state["pending_free_slots"].cpu()[:pending_count].tolist(),
        "pending_count": pending_count,
        "release_status": int(released["status"].cpu()[0]),
        "released_count": int(released["appended_count"].cpu()[0]),
        "release_cursor": int(pending_release_cursor.cpu()[0]),
        "queue": state["slot_free_queue"].cpu()[:queue_count].tolist(),
        "queue_count": queue_count,
    }


def test_action_deaths_remain_pending_while_only_residual_deaths_reach_birth_queue(
    build_directories: dict[str, Path],
) -> None:
    clean = _run_two_death_classes(
        poison_inactive_tails=False,
        build_directories=build_directories,
    )
    poisoned = _run_two_death_classes(
        poison_inactive_tails=True,
        build_directories=build_directories,
    )

    assert poisoned == clean
    assert clean == {
        "action_agent_ids": [20, 90],
        "action_slots": [3, 1],
        "action_reasons": [2, 1],
        "saved_cursor": 2,
        "birth_status": 0,
        "birth_planned_count": 1,
        "birth_child_slots": [6],
        "residual_agent_ids": [30, 50],
        "residual_slots": [5, 2],
        "residual_reasons": [2, 1],
        "pending_agent_ids": [20, 90, 30, 50],
        "pending_slots": [3, 1, 5, 2],
        "pending_count": 4,
        "release_status": 0,
        "released_count": 2,
        "release_cursor": 4,
        "queue": [6, 5, 2],
        "queue_count": 3,
    }
    assert not set(clean["action_slots"]) & set(clean["queue"])


def test_zero_residual_deaths_release_is_exact_no_op(
    build_directories: dict[str, Path],
) -> None:
    state = _resident_state(poison_inactive_tails=True)
    action = _death_cull(state, build_directory=build_directories["lifecycle"])
    torch.cuda.synchronize()
    assert action["death_count"].cpu().tolist() == [2]

    pending_release_cursor = state["pending_free_count"].clone()
    before = {name: tensor.clone() for name, tensor in state.items()}
    cursor_before = pending_release_cursor.clone()

    residual = _death_cull(state, build_directory=build_directories["lifecycle"])
    released = release_pending_free_slots_cuda(
        slot_to_agent_id=state["slot_to_agent_id"],
        alive_mask=state["alive_mask"],
        slot_free_queue=state["slot_free_queue"],
        slot_free_queue_count=state["slot_free_queue_count"],
        pending_free_slots=state["pending_free_slots"],
        pending_free_count=state["pending_free_count"],
        pending_release_cursor=pending_release_cursor,
        build_directory=build_directories["lifecycle"],
    )
    torch.cuda.synchronize()

    assert residual["status"].cpu().tolist() == [0]
    assert residual["death_count"].cpu().tolist() == [0]
    assert released["status"].cpu().tolist() == [0]
    assert released["appended_count"].cpu().tolist() == [0]
    torch.testing.assert_close(pending_release_cursor, cursor_before, rtol=0, atol=0)
    for name, tensor in state.items():
        torch.testing.assert_close(
            tensor,
            before[name],
            rtol=0,
            atol=0,
            equal_nan=True,
            msg=lambda message, field=name: f"{field}: {message}",
        )
