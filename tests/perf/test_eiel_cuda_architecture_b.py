from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from perf.cuda_architecture_b import (
    CudaActionConfig,
    CudaObservationConfig,
    ResidentCudaActionState,
    ResidentCudaOwnerActionState,
)
from perf.cxxbackbone8r_generation_island import (
    flat_apply_actions_cpu,
    flat_observation_frame_cpu,
    patch_apply_actions_cpu,
    patch_observation_frame_cpu,
)


def _initial_state(*, patch: bool) -> dict[str, np.ndarray]:
    agent_ids = np.asarray([9, 2, 7, 5], dtype=np.int32)
    pos_y = np.full((12,), -1, dtype=np.int32)
    pos_x = np.full((12,), -1, dtype=np.int32)
    agent_grid = np.full((6, 6), -1, dtype=np.int32)
    for aid, y, x in zip(
        np.asarray([2, 5, 7, 9], dtype=np.int32),
        np.asarray([1, 2, 4, 3], dtype=np.int32),
        np.asarray([1, 2, 4, 3], dtype=np.int32),
        strict=True,
    ):
        pos_y[int(aid)] = int(y)
        pos_x[int(aid)] = int(x)
        agent_grid[int(y), int(x)] = int(aid)
    food_grid = np.zeros((6, 6), dtype=np.int32)
    food_grid[3, 3] = 1
    food_grid[4, 4] = 1
    patch_food = np.zeros((6, 6), dtype=np.float32)
    if patch:
        patch_food[3, 3] = np.float32(0.75)
        patch_food[4, 4] = np.float32(0.25)
    movement = np.ones((6, 6), dtype=np.float32)
    movement[1, 2] = np.float32(2.0)
    yy, xx = np.indices((6, 6), dtype=np.float32)
    return {
        "agent_ids": agent_ids,
        "agent_pos_y": pos_y,
        "agent_pos_x": pos_x,
        "agent_grid": agent_grid,
        "food_grid": food_grid,
        "patch_food_amount": patch_food,
        "patch_cover": np.asarray(xx / 5.0, dtype=np.float32),
        "patch_shelter": np.asarray(yy / 5.0, dtype=np.float32),
        "patch_scent": np.asarray((yy * 2.0 + xx) / 20.0, dtype=np.float32),
        "patch_waste": np.asarray((yy + xx * 2.0) / 20.0, dtype=np.float32),
        "patch_scent_emit_pending": np.zeros((6, 6), dtype=np.float32),
        "patch_waste_emit_pending": np.zeros((6, 6), dtype=np.float32),
        "patch_movement_cost": movement,
        "energies": np.asarray([8.0, 9.0, 5.0, 6.0], dtype=np.float32),
        "healths": np.ones((4,), dtype=np.float32),
    }


def _actions() -> np.ndarray:
    return np.asarray(
        [
            [4, 2, 4, 0],
            [0, 5, 4, 3],
            [5, 1, 2, 4],
        ],
        dtype=np.int32,
    )


def _config() -> CudaActionConfig:
    return CudaActionConfig(
        move_cost=0.2,
        gather_cost=0.1,
        rest_cost=0.05,
        attack_cost=0.3,
        max_energy=10.0,
        food_energy=2.0,
        patch_food_gather_amount=0.5,
        patch_food_presence_threshold=0.1,
        patch_scent_move_emit=0.02,
        patch_waste_move_emit=0.01,
        patch_scent_gather_emit=0.05,
        patch_waste_consume_emit=0.04,
        patch_movement_cost_apply_enable=True,
        patch_movement_cost_energy_scale=0.5,
    )


def _cpu_scripted(state: dict[str, np.ndarray], actions: np.ndarray, config: CudaActionConfig, *, patch: bool, build: Path) -> dict[str, object]:
    current = {
        name: value.copy() if hasattr(value, "copy") else value
        for name, value in state.items()
    }
    food_gained = np.zeros_like(current["energies"])
    moved_count = np.zeros(current["energies"].shape, dtype=np.int32)
    totals = np.zeros((6,), dtype=np.float32)
    movement_events = 0
    died = np.zeros(current["energies"].shape, dtype=np.bool_)
    for row in actions:
        common = {
            "agent_ids": current["agent_ids"],
            "actions": row,
            "agent_pos_y": current["agent_pos_y"],
            "agent_pos_x": current["agent_pos_x"],
            "agent_grid": current["agent_grid"],
            "food_grid": current["food_grid"],
            "energies": current["energies"],
            "healths": current["healths"],
            "injuries": np.zeros_like(current["energies"]),
            "scars": np.zeros_like(current["energies"]),
            "move_cost": config.move_cost,
            "gather_cost": config.gather_cost,
            "rest_cost": config.rest_cost,
            "attack_cost": config.attack_cost,
            "action_cost_mult": config.action_cost_mult,
            "rest_cost_mult": config.rest_cost_mult,
            "max_energy": config.max_energy,
            "food_energy": config.food_energy,
            "build_directory": build,
        }
        if patch:
            result = patch_apply_actions_cpu(
                **common,
                patch_food_amount=current["patch_food_amount"],
                patch_scent_emit_pending=current["patch_scent_emit_pending"],
                patch_waste_emit_pending=current["patch_waste_emit_pending"],
                patch_movement_cost=current["patch_movement_cost"],
                patch_food_gather_amount=config.patch_food_gather_amount,
                patch_food_presence_threshold=config.patch_food_presence_threshold,
                patch_scent_move_emit=config.patch_scent_move_emit,
                patch_waste_move_emit=config.patch_waste_move_emit,
                patch_scent_gather_emit=config.patch_scent_gather_emit,
                patch_waste_consume_emit=config.patch_waste_consume_emit,
                patch_movement_cost_apply_enable=config.patch_movement_cost_apply_enable,
                patch_movement_cost_energy_scale=config.patch_movement_cost_energy_scale,
            )
            for name in ("patch_food_amount", "patch_scent_emit_pending", "patch_waste_emit_pending"):
                current[name] = result[name]
            metric_names = (
                "patch_food_harvest_total",
                "patch_scent_move_queued_total",
                "patch_waste_move_queued_total",
                "patch_scent_gather_queued_total",
                "patch_waste_gather_queued_total",
                "patch_movement_energy_extra",
            )
            for index, name in enumerate(metric_names):
                totals[index] = np.float32(totals[index] + np.float32(result[name][0]))
            movement_events += int(result["patch_movement_energy_event_count"][0])
        else:
            result = flat_apply_actions_cpu(**common)
        for name in ("agent_pos_y", "agent_pos_x", "agent_grid", "food_grid"):
            current[name] = result[name]
        current["energies"] = result["new_energy"]
        food_gained += result["food_gained"]
        moved_count += result["moved_mask"]
        died = result["died_mask"]
    return {
        **current,
        "food_gained": food_gained,
        "moved_count": moved_count,
        "died_mask": died,
        "totals": totals,
        "movement_events": movement_events,
    }


@pytest.mark.parametrize("patch", [False, True])
def test_cuda_stage1_scripted_generation_matches_cpu_action_oracle(tmp_path: Path, patch: bool) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    state = _initial_state(patch=patch)
    actions = _actions()
    config = _config()
    expected = _cpu_scripted(state, actions, config, patch=patch, build=tmp_path / "cpu")
    owner = ResidentCudaActionState.from_arrays(
        **{
            name: state[name]
            for name in (
                "agent_ids",
                "agent_pos_y",
                "agent_pos_x",
                "agent_grid",
                "food_grid",
                "energies",
                "healths",
                "patch_food_amount",
                "patch_scent_emit_pending",
                "patch_waste_emit_pending",
                "patch_movement_cost",
            )
        },
        patch_enabled=patch,
    )

    receipt = owner.execute_scripted_generation(actions, config, build_directory=tmp_path / "cuda")
    actual = owner.export_boundary()

    assert receipt["hidden_fallback"] is False
    assert receipt["python_tick_loops"] == 0
    assert receipt["native_generation_kernel_launches"] == 1
    assert receipt["state_import_count"] == 1
    assert receipt["full_stage1_qualified"] is False
    assert actual["native_tick_loop_count"] == actions.shape[0]
    for name in ("agent_pos_y", "agent_pos_x", "agent_grid", "food_grid"):
        np.testing.assert_array_equal(actual[name], expected[name])
    np.testing.assert_array_equal(actual["energy"], expected["energies"])
    np.testing.assert_array_equal(actual["died_mask"], expected["died_mask"])
    np.testing.assert_array_equal(actual["food_gained"], expected["food_gained"])
    np.testing.assert_array_equal(actual["moved_count"], expected["moved_count"])
    if patch:
        for name in ("patch_food_amount", "patch_scent_emit_pending", "patch_waste_emit_pending"):
            np.testing.assert_array_equal(actual[name], expected[name])
        actual_totals = np.asarray(
            [
                actual["patch_food_harvest_total"],
                actual["patch_scent_move_queued_total"],
                actual["patch_waste_move_queued_total"],
                actual["patch_scent_gather_queued_total"],
                actual["patch_waste_gather_queued_total"],
                actual["patch_movement_energy_extra"],
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(actual_totals, expected["totals"])
        assert actual["patch_movement_energy_event_count"] == expected["movement_events"]

    expected_continuation = _cpu_scripted(
        expected,
        actions,
        config,
        patch=patch,
        build=tmp_path / "cpu",
    )
    continuation_receipt = owner.execute_scripted_generation(
        actions,
        config,
        build_directory=tmp_path / "cuda",
    )
    continued = owner.export_boundary()

    assert continuation_receipt["state_import_count"] == 1
    assert continuation_receipt["native_generation_calls"] == 2
    assert continuation_receipt["native_generation_kernel_launches"] == 2
    assert continuation_receipt["action_payload_import_count"] == 2
    for name in ("agent_pos_y", "agent_pos_x", "agent_grid", "food_grid"):
        np.testing.assert_array_equal(continued[name], expected_continuation[name])
    np.testing.assert_array_equal(continued["energy"], expected_continuation["energies"])
    if patch:
        for name in ("patch_food_amount", "patch_scent_emit_pending", "patch_waste_emit_pending"):
            np.testing.assert_array_equal(continued[name], expected_continuation[name])


@pytest.mark.parametrize("patch", [False, True])
def test_cuda_stage1_owner_slot_layout_matches_cpu_action_oracle(tmp_path: Path, patch: bool) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    row_state = _initial_state(patch=patch)
    actions_by_row = _actions()
    config = _config()
    expected = _cpu_scripted(
        row_state,
        actions_by_row,
        config,
        patch=patch,
        build=tmp_path / "cpu",
    )
    agent_ids = row_state["agent_ids"]
    slots = np.asarray([0, 3, 4, 1], dtype=np.int32)
    capacity = 6
    agent_id_to_slot = np.full((12,), -1, dtype=np.int32)
    slot_to_agent_id = np.full((capacity,), -1, dtype=np.int64)
    alive_mask = np.zeros((capacity,), dtype=np.bool_)
    pos_y = np.full((capacity,), -1, dtype=np.int32)
    pos_x = np.full((capacity,), -1, dtype=np.int32)
    energy = np.zeros((capacity,), dtype=np.float32)
    health = np.zeros((capacity,), dtype=np.float32)
    for row, (aid, slot) in enumerate(zip(agent_ids, slots, strict=True)):
        agent_id_to_slot[int(aid)] = int(slot)
        slot_to_agent_id[int(slot)] = int(aid)
        alive_mask[int(slot)] = True
        pos_y[int(slot)] = row_state["agent_pos_y"][int(aid)]
        pos_x[int(slot)] = row_state["agent_pos_x"][int(aid)]
        energy[int(slot)] = row_state["energies"][row]
        health[int(slot)] = row_state["healths"][row]
    actions_by_slot = np.full((actions_by_row.shape[0], capacity), 5, dtype=np.int32)
    for row, slot in enumerate(slots):
        actions_by_slot[:, int(slot)] = actions_by_row[:, row]
    tensors = {
        "agent_ids_by_alive_order": torch.from_numpy(agent_ids.astype(np.int64)),
        "agent_id_to_slot": torch.from_numpy(agent_id_to_slot),
        "slot_to_agent_id": torch.from_numpy(slot_to_agent_id),
        "alive_mask": torch.from_numpy(alive_mask),
        "pos_y": torch.from_numpy(pos_y),
        "pos_x": torch.from_numpy(pos_x),
        "agent_grid": torch.from_numpy(row_state["agent_grid"]),
        "food_grid": torch.from_numpy(row_state["food_grid"]),
        "patch_food_amount": torch.from_numpy(row_state["patch_food_amount"]),
        "patch_cover": torch.from_numpy(row_state["patch_cover"]),
        "patch_shelter": torch.from_numpy(row_state["patch_shelter"]),
        "patch_scent": torch.from_numpy(row_state["patch_scent"]),
        "patch_waste": torch.from_numpy(row_state["patch_waste"]),
        "patch_scent_emit_pending": torch.from_numpy(row_state["patch_scent_emit_pending"]),
        "patch_waste_emit_pending": torch.from_numpy(row_state["patch_waste_emit_pending"]),
        "patch_movement_cost": torch.from_numpy(row_state["patch_movement_cost"]),
        "energy": torch.from_numpy(energy),
        "health": torch.from_numpy(health),
    }
    owner = ResidentCudaOwnerActionState.from_projection(
        {"tensors": tensors},
        patch_enabled=patch,
    )

    observation_config = CudaObservationConfig(
        view_radius=1,
        max_energy=config.max_energy,
        start_health=1.0,
        patch_food_amount_max=1.0,
        patch_scent_max=1.0,
        patch_waste_max=1.0,
        patch_density_radius=1,
        patch_density_include_self=True,
        patch_density_normalize=False,
        patch_movement_cost_min=0.0,
        patch_movement_cost_max=4.0,
    )
    if patch:
        expected_observation = patch_observation_frame_cpu(
            agent_ids=agent_ids,
            slots=np.arange(agent_ids.size, dtype=np.int32),
            agent_pos_y=row_state["agent_pos_y"],
            agent_pos_x=row_state["agent_pos_x"],
            agent_grid=row_state["agent_grid"],
            food_grid=row_state["food_grid"],
            energy_by_slot=row_state["energies"],
            health_by_slot=row_state["healths"],
            patch_food_amount=row_state["patch_food_amount"],
            patch_cover=row_state["patch_cover"],
            patch_shelter=row_state["patch_shelter"],
            patch_scent=row_state["patch_scent"],
            patch_waste=row_state["patch_waste"],
            patch_movement_cost=row_state["patch_movement_cost"],
            view_radius=observation_config.view_radius,
            max_energy=observation_config.max_energy,
            start_health=observation_config.start_health,
            patch_food_amount_max=observation_config.patch_food_amount_max,
            patch_scent_max=observation_config.patch_scent_max,
            patch_waste_max=observation_config.patch_waste_max,
            patch_density_radius=observation_config.patch_density_radius,
            patch_density_include_self=observation_config.patch_density_include_self,
            patch_density_normalize=observation_config.patch_density_normalize,
            patch_movement_cost_min=observation_config.patch_movement_cost_min,
            patch_movement_cost_max=observation_config.patch_movement_cost_max,
            build_directory=tmp_path / "cpu",
        )
    else:
        expected_observation = flat_observation_frame_cpu(
            agent_ids=agent_ids,
            slots=np.arange(agent_ids.size, dtype=np.int32),
            agent_pos_y=row_state["agent_pos_y"],
            agent_pos_x=row_state["agent_pos_x"],
            agent_grid=row_state["agent_grid"],
            food_grid=row_state["food_grid"],
            energy_by_slot=row_state["energies"],
            health_by_slot=row_state["healths"],
            view_radius=observation_config.view_radius,
            max_energy=observation_config.max_energy,
            start_health=observation_config.start_health,
            build_directory=tmp_path / "cpu",
        )
    actual_observation = owner.build_observation_frame(
        observation_config,
        build_directory=tmp_path / "cuda",
    )
    assert actual_observation.is_cuda
    np.testing.assert_allclose(
        actual_observation.detach().cpu().numpy(),
        expected_observation,
        rtol=0.0,
        atol=1e-7 if patch else 0.0,
    )

    receipt = owner.execute_scripted_generation(
        actions_by_slot,
        config,
        build_directory=tmp_path / "cuda",
    )
    boundary = owner.export_boundary()

    assert receipt["accepted_owner_projection_layout"] is True
    assert receipt["hidden_fallback"] is False
    assert receipt["python_tick_loops"] == 0
    assert receipt["state_import_count"] == 1
    np.testing.assert_array_equal(boundary["alive_slots_by_alive_order"].numpy(), slots)
    actual_tensors = boundary["tensors"]
    np.testing.assert_array_equal(actual_tensors["agent_grid"].numpy(), expected["agent_grid"])
    np.testing.assert_array_equal(actual_tensors["food_grid"].numpy(), expected["food_grid"])
    for row, (aid, slot) in enumerate(zip(agent_ids, slots, strict=True)):
        assert int(actual_tensors["pos_y"][int(slot)]) == int(expected["agent_pos_y"][int(aid)])
        assert int(actual_tensors["pos_x"][int(slot)]) == int(expected["agent_pos_x"][int(aid)])
        assert np.float32(actual_tensors["energy"][int(slot)]) == np.float32(expected["energies"][row])
    if patch:
        for name in ("patch_food_amount", "patch_scent_emit_pending", "patch_waste_emit_pending"):
            np.testing.assert_array_equal(actual_tensors[name].numpy(), expected[name])
