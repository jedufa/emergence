from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from perf.cuda_architecture_b import CudaActionConfig, load_cuda_architecture_b_extension
from perf.cuda_architecture_b_active_mechanics import (
    CudaActiveMechanicsConfig,
    execute_active_mechanics_cuda,
)
from perf.cuda_architecture_b_generation import (
    CUDA_GENERATION_CAPABILITIES,
    CudaGenerationConfig,
    EVENT_TAPE_COLUMN_NAMES,
    execute_cuda_generation,
    load_cuda_generation_extension,
)
from perf.cuda_architecture_b_patch_context import (
    PATCH_CONTEXT_P2_STAT_FIELDS,
    execute_patch_context_p2_cuda,
)
from perf.cuda_architecture_b_lifecycle import execute_lifecycle_death_cull_cuda
from perf.cuda_architecture_b_pcg64 import pcg64_state_tensors
from tests.perf.test_eiel_cuda_architecture_b import _cpu_scripted, _initial_state


STEPS = 3
CAPACITY = 6
HIDDEN_SIZE = 4
ACTION_COUNT = 7
RIGHT_ACTION = 3
REST_ACTION = 5
SLOTS = np.asarray([0, 3, 4, 1], dtype=np.int32)
EVENT_TAPE_DTYPES = (
    torch.int32,
    torch.int64,
    torch.int64,
    torch.int64,
    torch.int64,
    torch.float32,
    torch.int32,
    torch.int64,
    torch.int64,
    torch.int64,
    torch.int64,
    torch.int64,
    torch.int64,
    torch.int64,
    torch.int64,
    torch.int64,
    torch.int64,
    torch.int32,
    torch.int32,
    torch.int32,
    torch.int8,
    torch.float32,
)


@pytest.fixture(scope="module")
def generation_build(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("archb-cuda-generation-build")


def _action_config() -> CudaActionConfig:
    return CudaActionConfig(
        move_cost=0.25,
        gather_cost=0.125,
        rest_cost=0.0625,
        attack_cost=0.5,
        max_energy=100.0,
        food_energy=2.0,
        patch_food_gather_amount=0.5,
        patch_food_presence_threshold=0.5,
        patch_scent_move_emit=0.0,
        patch_waste_move_emit=0.0,
        patch_scent_gather_emit=0.0,
        patch_waste_consume_emit=0.0,
        patch_movement_cost_apply_enable=False,
        patch_movement_cost_energy_scale=0.0,
    )


def _generation_config(*, patch: bool) -> CudaGenerationConfig:
    action = _action_config()
    return CudaGenerationConfig(
        steps=STEPS,
        patch_enabled=patch,
        hidden_size=HIDDEN_SIZE,
        action_count=ACTION_COUNT,
        cache_width=2,
        max_energy=action.max_energy,
        move_cost=action.move_cost,
        gather_cost=action.gather_cost,
        rest_cost=action.rest_cost,
        attack_cost=action.attack_cost,
        action_cost_mult=action.action_cost_mult,
        rest_cost_mult=action.rest_cost_mult,
        food_energy=action.food_energy,
        patch_food_gather_amount=action.patch_food_gather_amount,
        patch_food_presence_threshold=action.patch_food_presence_threshold,
        patch_scent_move_emit=action.patch_scent_move_emit,
        patch_waste_move_emit=action.patch_waste_move_emit,
        patch_scent_gather_emit=action.patch_scent_gather_emit,
        patch_waste_consume_emit=action.patch_waste_consume_emit,
        patch_movement_cost_apply_enable=action.patch_movement_cost_apply_enable,
        patch_movement_cost_energy_scale=action.patch_movement_cost_energy_scale,
        patch_scent_decay_per_step=0.25 if patch else 0.0,
        patch_waste_decay_per_step=0.5 if patch else 0.0,
        patch_scent_presence_emit=0.0,
        patch_waste_presence_emit=0.0,
        patch_food_regrow_per_step=0.125 if patch else 0.0,
        active_mechanics_enabled=False,
    )


def _fixture(
    *,
    patch: bool,
    policy_action: int | None = RIGHT_ACTION,
    pcg64_state: dict[str, object] | None = None,
    event_capacity: int = 32,
    event_active_count: int = 0,
) -> tuple[dict[str, torch.Tensor], dict[str, np.ndarray]]:
    row_state = _initial_state(patch=patch)
    row_state["energies"] = np.full(row_state["energies"].shape, 90.0, dtype=np.float32)
    agent_ids = row_state["agent_ids"].astype(np.int64)
    agent_id_to_slot = np.full((12,), -1, dtype=np.int32)
    pos_y = np.full((CAPACITY,), -1, dtype=np.int32)
    pos_x = np.full((CAPACITY,), -1, dtype=np.int32)
    energy = np.full((CAPACITY,), 100.0, dtype=np.float32)
    health = np.ones((CAPACITY,), dtype=np.float32)
    slot_to_agent_id = np.full((CAPACITY,), -1, dtype=np.int64)
    alive_mask = np.zeros((CAPACITY,), dtype=np.bool_)
    lineage_founder_id = np.full((CAPACITY,), -1, dtype=np.int32)
    lineage_depth = np.full((CAPACITY,), -1, dtype=np.int32)
    for row, (agent_id, slot) in enumerate(zip(agent_ids, SLOTS, strict=True)):
        agent_id_to_slot[int(agent_id)] = int(slot)
        slot_to_agent_id[int(slot)] = int(agent_id)
        alive_mask[int(slot)] = True
        pos_y[int(slot)] = int(row_state["agent_pos_y"][int(agent_id)])
        pos_x[int(slot)] = int(row_state["agent_pos_x"][int(agent_id)])
        energy[int(slot)] = row_state["energies"][row]
        health[int(slot)] = row_state["healths"][row]
        lineage_founder_id[int(slot)] = np.int32(7000 + int(agent_id))
        lineage_depth[int(slot)] = np.int32(20 + row)

    observation_size = (8 if patch else 1) * 9 + 2
    packed_width = (
        HIDDEN_SIZE * observation_size
        + HIDDEN_SIZE
        + HIDDEN_SIZE * HIDDEN_SIZE
        + HIDDEN_SIZE
        + ACTION_COUNT * HIDDEN_SIZE
        + ACTION_COUNT
    )
    mlp_pack = np.zeros((CAPACITY, packed_width), dtype=np.float32)
    if policy_action is not None:
        mlp_pack[:, -ACTION_COUNT + policy_action] = np.float32(10.0)
    shape = row_state["agent_grid"].shape
    arrays: dict[str, np.ndarray] = {
        "agent_ids_by_alive_order": agent_ids,
        "agent_id_to_slot": agent_id_to_slot,
        "slot_to_agent_id": slot_to_agent_id,
        "alive_mask": alive_mask,
        "pos_y": pos_y,
        "pos_x": pos_x,
        "agent_grid": row_state["agent_grid"],
        "food_grid": row_state["food_grid"],
        "energy": energy,
        "health": health,
        "injury": np.zeros((CAPACITY,), dtype=np.float32),
        "age": np.zeros((CAPACITY,), dtype=np.int32),
        "policy_energy_observation": energy.astype(np.float64),
        "rest_streak": np.zeros((CAPACITY,), dtype=np.int32),
        "survival_steps": np.zeros((CAPACITY,), dtype=np.int32),
        "last_action": np.full((CAPACITY,), -1, dtype=np.int32),
        "last_last_action": np.full((CAPACITY,), -1, dtype=np.int32),
        "action_switches": np.zeros((CAPACITY,), dtype=np.int32),
        "action_oscillations": np.zeros((CAPACITY,), dtype=np.int32),
        "episode0_switches": np.zeros((CAPACITY,), dtype=np.int32),
        "episode0_oscillations": np.zeros((CAPACITY,), dtype=np.int32),
        "gru_upkeep_cost": np.zeros((CAPACITY,), dtype=np.float32),
        "cost_charge_scalar": np.zeros((CAPACITY,), dtype=np.float32),
        "cost_charge_maturation_scalar": np.zeros((CAPACITY,), dtype=np.float32),
        "patch_food_amount": row_state["patch_food_amount"],
        "patch_fertility": np.full(shape, 0.5, dtype=np.float32),
        "patch_cover": row_state["patch_cover"],
        "patch_shelter": row_state["patch_shelter"],
        "patch_scent": row_state["patch_scent"],
        "patch_waste": row_state["patch_waste"],
        "patch_density": np.zeros(shape, dtype=np.float32),
        "patch_scent_emit_pending": row_state["patch_scent_emit_pending"],
        "patch_waste_emit_pending": row_state["patch_waste_emit_pending"],
        "patch_movement_cost": row_state["patch_movement_cost"],
        "mlp_pack": mlp_pack,
        "drives": np.zeros((CAPACITY, 4), dtype=np.float32),
        "chemistry": np.tile(
            np.asarray([0.55, 0.55, 0.0, 0.0, 0.5, 0.25], dtype=np.float32),
            (CAPACITY, 1),
        ),
        "drive_inertia": np.full((CAPACITY,), 0.5, dtype=np.float32),
        "drive_noise": np.zeros((CAPACITY,), dtype=np.float32),
        "prev_drive_delta": np.zeros((CAPACITY, 4), dtype=np.float32),
        "drive_volatility": np.zeros((CAPACITY,), dtype=np.float32),
        "drive_jitter": np.zeros((CAPACITY,), dtype=np.float32),
        "episode0_drive_jitter_sum": np.zeros((CAPACITY,), dtype=np.float32),
        "last_energy_delta": np.zeros((CAPACITY,), dtype=np.float32),
        "last_health_delta": np.zeros((CAPACITY,), dtype=np.float32),
        "last_injury_delta": np.zeros((CAPACITY,), dtype=np.float32),
        "selfharm_events": np.zeros((CAPACITY,), dtype=np.int32),
        "episode0_selfharm_events": np.zeros((CAPACITY,), dtype=np.int32),
        "drive_noise_samples_by_tick_slot": np.zeros(
            (STEPS, CAPACITY, 4), dtype=np.float32
        ),
        "drive_noise_samples_valid_by_tick_slot": np.broadcast_to(
            alive_mask,
            (STEPS, CAPACITY),
        ).copy(),
        "attack_bias": np.zeros((CAPACITY,), dtype=np.float32),
        "action_temperature": np.ones((CAPACITY,), dtype=np.float32),
        "drive_logit_gain": np.zeros((CAPACITY,), dtype=np.float32),
        "slot_tokens": np.arange(100, 100 + CAPACITY, dtype=np.int64),
        "action_cache_actions": np.full((CAPACITY, 3), -1, dtype=np.int32),
        "action_cache_token": np.full((CAPACITY,), -1, dtype=np.int64),
        "action_cache_cursor": np.full((CAPACITY,), -1, dtype=np.int32),
        "action_cache_next": np.full((CAPACITY,), -1, dtype=np.int32),
        "pending_free_agent_ids": np.full((CAPACITY,), -1, dtype=np.int64),
        "pending_free_slots": np.full((CAPACITY,), -1, dtype=np.int64),
        "pending_free_count": np.zeros((1,), dtype=np.int64),
        "pcg64_state_words": np.empty((0,), dtype=np.uint64),
        "pcg64_buffer_words": np.empty((0,), dtype=np.int64),
        "lineage_founder_id": lineage_founder_id,
        "lineage_depth": lineage_depth,
    }
    tensors = {
        name: torch.from_numpy(np.ascontiguousarray(value)).cuda().contiguous()
        for name, value in arrays.items()
    }
    policy_state = _pcg64_state() if pcg64_state is None else pcg64_state
    state_words, buffer_words = pcg64_state_tensors(policy_state, device="cuda")
    tensors["pcg64_state_words"] = state_words
    tensors["pcg64_buffer_words"] = buffer_words
    runner_state_words, runner_buffer_words = pcg64_state_tensors(
        _pcg64_state(909), device="cuda"
    )
    tensors["runner_pcg64_state_words"] = runner_state_words
    tensors["runner_pcg64_buffer_words"] = runner_buffer_words
    for name, dtype in zip(EVENT_TAPE_COLUMN_NAMES, EVENT_TAPE_DTYPES, strict=True):
        tensors[name] = torch.full(
            (event_capacity,),
            -1,
            device="cuda",
            dtype=dtype,
        )
    tensors["event_tape_active_count"] = torch.tensor(
        [event_active_count],
        device="cuda",
        dtype=torch.int64,
    )
    tensors["event_tape_status"] = torch.tensor(
        [-1],
        device="cuda",
        dtype=torch.int32,
    )
    return tensors, row_state


def _cpu_owner_death_event_semantics(
    tensors: dict[str, torch.Tensor],
    config: CudaGenerationConfig,
    *,
    agent_ids: list[int],
    slots: list[int],
    reason_codes: list[int],
    sequence_start: int = 0,
) -> dict[str, list[int] | list[float]]:
    count = len(agent_ids)
    target = [-1] * count
    return {
        "event_tape_type": [4] * count,
        "event_tape_generation": [config.generation] * count,
        "event_tape_tick": [config.start_tick] * count,
        "event_tape_actor_id": agent_ids,
        "event_tape_target_id": target,
        "event_tape_value": [float(reason) for reason in reason_codes],
        "typed_lifecycle_event_event_type": [4] * count,
        "typed_lifecycle_event_generation": [config.generation] * count,
        "typed_lifecycle_event_tick": [config.start_tick] * count,
        "typed_lifecycle_event_sequence": list(range(sequence_start, sequence_start + count)),
        "typed_lifecycle_event_actor_agent_id": agent_ids,
        "typed_lifecycle_event_target_agent_id": target,
        "typed_lifecycle_event_actor_slot": slots,
        "typed_lifecycle_event_target_slot": target,
        "typed_lifecycle_event_parent_agent_id": target,
        "typed_lifecycle_event_child_agent_id": target,
        "typed_lifecycle_event_lineage_founder_id": [
            int(tensors["lineage_founder_id"][slot].cpu()) for slot in slots
        ],
        "typed_lifecycle_event_lineage_depth": [
            int(tensors["lineage_depth"][slot].cpu()) for slot in slots
        ],
        "typed_lifecycle_event_subtype_code": [4] * count,
        "typed_lifecycle_event_reason_code": reason_codes,
        "typed_lifecycle_event_success": [1] * count,
        "typed_lifecycle_event_value": [float(reason) for reason in reason_codes],
    }


def _pcg64_state(seed: int = 303) -> dict[str, object]:
    return copy.deepcopy(np.random.PCG64(seed).state)


def _expected_stochastic_actions(
    initial_state: dict[str, object],
    *,
    steps: int,
    cache_width: int,
) -> tuple[np.ndarray, dict[str, object]]:
    bit_generator = np.random.PCG64()
    bit_generator.state = copy.deepcopy(initial_state)
    generator = np.random.Generator(bit_generator)
    slot_by_agent_id = {
        int(agent_id): int(slot)
        for agent_id, slot in zip(_initial_state(patch=False)["agent_ids"], SLOTS, strict=True)
    }
    initial_order = [int(value) for value in _initial_state(patch=False)["agent_ids"]]
    sorted_order = sorted(initial_order)
    expected = np.full((steps, CAPACITY), REST_ACTION, dtype=np.int32)
    for tick in range(steps):
        order = initial_order if tick == 0 else sorted_order
        uniforms = generator.random(
            len(order) * cache_width * ACTION_COUNT,
            dtype=np.float32,
        )
        draw = 0
        for agent_id in order:
            first_action = -1
            for cache_index in range(cache_width):
                scores = np.empty((ACTION_COUNT,), dtype=np.float32)
                for action in range(ACTION_COUNT):
                    uniform = np.clip(
                        uniforms[draw],
                        np.float32(1.0e-6),
                        np.float32(1.0 - 1.0e-6),
                    )
                    draw += 1
                    scores[action] = np.float32(-np.log(-np.log(uniform)))
                selected = int(np.argmax(scores))
                if cache_index == 0:
                    first_action = selected
            expected[tick, slot_by_agent_id[agent_id]] = first_action
    return expected, copy.deepcopy(bit_generator.state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("patch", [False, True])
def test_deterministic_generation_matches_cpu_action_boundary(
    generation_build: Path,
    tmp_path: Path,
    patch: bool,
) -> None:
    tensors, row_state = _fixture(patch=patch)
    config = _generation_config(patch=patch)
    pointers = {name: tensor.data_ptr() for name, tensor in tensors.items()}
    actions_by_row = np.full(
        (STEPS, row_state["agent_ids"].size),
        RIGHT_ACTION,
        dtype=np.int32,
    )
    expected = _cpu_scripted(
        row_state,
        actions_by_row,
        _action_config(),
        patch=patch,
        build=tmp_path / "cpu-action-oracle",
    )

    result = execute_cuda_generation(
        tensors,
        config,
        build_directory=generation_build,
    )

    assert pointers == {name: tensor.data_ptr() for name, tensor in tensors.items()}
    assert all(tensor.is_cuda for tensor in result.__dict__.values())
    assert torch.equal(result.slots_by_alive_order.cpu(), torch.from_numpy(SLOTS))
    assert torch.equal(
        result.actions_by_tick_slot[:, torch.from_numpy(SLOTS).cuda().long()].cpu(),
        torch.full((STEPS, SLOTS.size), RIGHT_ACTION, dtype=torch.int32),
    )
    assert not bool(result.died_mask_by_alive_order.any())
    assert torch.equal(result.food_gained_by_alive_order.cpu(), torch.zeros(SLOTS.size))
    assert torch.equal(result.next_rng_counter.cpu(), torch.tensor([config.rng_counter]))
    patch_width = len(PATCH_CONTEXT_P2_STAT_FIELDS)
    assert tuple(result.patch_stats_by_tick.shape) == (
        (STEPS, patch_width) if patch else (0, patch_width)
    )
    assert torch.equal(result.lifecycle_death_counts_by_tick.cpu(), torch.zeros(STEPS, dtype=torch.int64))
    assert torch.equal(result.lifecycle_status_by_tick.cpu(), torch.zeros(STEPS, dtype=torch.int32))
    assert result.event_tape_active_count.cpu().tolist() == [0]
    assert result.event_tape_status.cpu().tolist() == [0]
    assert result.event_append_status_by_tick.cpu().tolist() == [0] * STEPS
    assert result.final_alive_count.cpu().tolist() == [4]
    assert result.final_alive_agent_ids.cpu()[:4].tolist() == [2, 5, 7, 9]
    assert result.final_alive_agent_ids.cpu()[4:].eq(-1).all()

    np.testing.assert_array_equal(tensors["agent_grid"].cpu().numpy(), expected["agent_grid"])
    for row, (agent_id, slot) in enumerate(
        zip(row_state["agent_ids"], SLOTS, strict=True)
    ):
        assert int(tensors["pos_y"][int(slot)]) == int(expected["agent_pos_y"][int(agent_id)])
        assert int(tensors["pos_x"][int(slot)]) == int(expected["agent_pos_x"][int(agent_id)])
        assert np.float32(tensors["energy"][int(slot)].cpu()) == np.float32(
            expected["energies"][row]
        )
    np.testing.assert_array_equal(
        result.moved_count_by_alive_order.cpu().numpy(),
        expected["moved_count"],
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_active_mechanics_is_composed_before_lifecycle(
    generation_build: Path,
    tmp_path: Path,
) -> None:
    tensors, _ = _fixture(patch=False, policy_action=REST_ACTION)
    tensors["drive_noise"].fill_(0.25)
    tensors["drive_noise_samples_by_tick_slot"][2].copy_(
        torch.linspace(
            -0.5,
            0.5,
            steps=CAPACITY * 4,
            device="cuda",
            dtype=torch.float32,
        ).reshape(CAPACITY, 4)
    )
    oracle = {name: tensor.clone() for name, tensor in tensors.items()}
    config = replace(
        _generation_config(patch=False),
        steps=1,
        start_tick=2,
        active_mechanics_enabled=True,
    )
    actions = torch.full((1, CAPACITY), REST_ACTION, device="cuda", dtype=torch.int32)
    pre_energy = oracle["energy"].clone()
    pre_health = oracle["health"].clone()
    pre_injury = oracle["injury"].clone()
    action_module = load_cuda_architecture_b_extension(
        build_directory=tmp_path / "active-action-device-oracle"
    )
    action_module.eiel_cuda_scripted_owner_action_generation_v1(
        oracle["agent_ids_by_alive_order"],
        oracle["agent_id_to_slot"],
        actions,
        oracle["pos_y"],
        oracle["pos_x"],
        oracle["agent_grid"],
        oracle["food_grid"],
        oracle["patch_food_amount"],
        oracle["patch_scent_emit_pending"],
        oracle["patch_waste_emit_pending"],
        oracle["patch_movement_cost"],
        oracle["energy"],
        oracle["health"],
        False,
        config.move_cost,
        config.gather_cost,
        config.rest_cost,
        config.attack_cost,
        config.action_cost_mult,
        config.rest_cost_mult,
        config.max_energy,
        config.food_energy,
        config.patch_food_gather_amount,
        config.patch_food_presence_threshold,
        config.patch_scent_move_emit,
        config.patch_waste_move_emit,
        config.patch_scent_gather_emit,
        config.patch_waste_consume_emit,
        config.patch_movement_cost_apply_enable,
        config.patch_movement_cost_energy_scale,
    )
    expected = execute_active_mechanics_cuda(
        slot_to_agent_id=oracle["slot_to_agent_id"],
        alive_mask=oracle["alive_mask"],
        actions_by_slot=actions[0],
        energy=oracle["energy"],
        health=oracle["health"],
        injury=oracle["injury"],
        age=oracle["age"],
        policy_energy_observation=oracle["policy_energy_observation"],
        rest_streak=oracle["rest_streak"],
        survival_steps=oracle["survival_steps"],
        gru_upkeep_cost=oracle["gru_upkeep_cost"],
        cost_charge_scalar=oracle["cost_charge_scalar"],
        cost_charge_maturation_scalar=oracle["cost_charge_maturation_scalar"],
        drives=oracle["drives"],
        drive_inertia=oracle["drive_inertia"],
        drive_noise=oracle["drive_noise"],
        prev_drive_delta=oracle["prev_drive_delta"],
        drive_volatility=oracle["drive_volatility"],
        drive_jitter=oracle["drive_jitter"],
        episode0_drive_jitter_sum=oracle["episode0_drive_jitter_sum"],
        last_energy_delta=oracle["last_energy_delta"],
        last_health_delta=oracle["last_health_delta"],
        last_injury_delta=oracle["last_injury_delta"],
        selfharm_events=oracle["selfharm_events"],
        episode0_selfharm_events=oracle["episode0_selfharm_events"],
        pre_energy=pre_energy,
        pre_health=pre_health,
        pre_injury=pre_injury,
        drive_noise_samples=oracle["drive_noise_samples_by_tick_slot"][config.start_tick],
        drive_noise_sample_valid=(
            oracle["drive_noise_samples_valid_by_tick_slot"][config.start_tick]
        ),
        config=CudaActiveMechanicsConfig(
            enable_rest_charge=config.enable_rest_charge,
            rest_streak_free=config.rest_streak_free,
            rest_charge_energy=config.rest_charge_energy,
            rest_charge_growth=config.rest_charge_growth,
            rest_deficit_to_health=config.rest_deficit_to_health,
            developmental_maturity_age=config.developmental_maturity_age,
            episode0_k=config.episode0_k,
            max_energy=config.max_energy,
            start_health=config.start_health,
            drive_hunger_gain=config.drive_hunger_gain,
            drive_hunger_decay=config.drive_hunger_decay,
            drive_pain_gain=config.drive_pain_gain,
            drive_pain_decay=config.drive_pain_decay,
            drive_fatigue_gain=config.drive_fatigue_gain,
            drive_fatigue_rest_relief=config.drive_fatigue_rest_relief,
            drive_fatigue_decay=config.drive_fatigue_decay,
            drive_fear_gain=config.drive_fear_gain,
            drive_fear_decay=config.drive_fear_decay,
        ),
        build_directory=tmp_path / "active-mechanics-device-oracle",
    )
    expected.raise_for_status()

    result = execute_cuda_generation(tensors, config, build_directory=generation_build)

    assert torch.equal(result.active_mechanics_status_by_tick, expected.status)
    assert torch.equal(result.active_mechanics_counts_by_tick[0], expected.counts)
    for name in (
        "energy",
        "health",
        "injury",
        "policy_energy_observation",
        "rest_streak",
        "survival_steps",
        "drives",
        "prev_drive_delta",
        "drive_volatility",
        "drive_jitter",
        "episode0_drive_jitter_sum",
        "last_energy_delta",
        "last_health_delta",
        "last_injury_delta",
        "selfharm_events",
        "episode0_selfharm_events",
    ):
        torch.testing.assert_close(tensors[name], oracle[name], rtol=0, atol=0), name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_patch_context_composition_matches_standalone_device_oracle(
    generation_build: Path,
    tmp_path: Path,
) -> None:
    tensors, _ = _fixture(patch=True, policy_action=REST_ACTION)
    config = replace(
        _generation_config(patch=True),
        patch_world_waste_consequence_enable=True,
    )
    oracle = {
        name: tensors[name].clone()
            for name in (
                "agent_grid",
                "patch_food_amount",
            "patch_fertility",
            "patch_scent",
            "patch_waste",
            "patch_density",
            "patch_scent_emit_pending",
            "patch_waste_emit_pending",
            "food_grid",
        )
    }
    expected_stats = execute_patch_context_p2_cuda(
        **oracle,
        steps=STEPS,
        patch_scent_decay_per_step=config.patch_scent_decay_per_step,
        patch_waste_decay_per_step=config.patch_waste_decay_per_step,
        patch_scent_presence_emit=config.patch_scent_presence_emit,
        patch_waste_presence_emit=config.patch_waste_presence_emit,
        patch_waste_density_emit_scale=config.patch_waste_density_emit_scale,
        patch_scent_max=config.patch_scent_max,
        patch_waste_max=config.patch_waste_max,
        patch_food_amount_max=config.patch_food_amount_max,
        patch_food_regrow_per_step=config.patch_food_regrow_per_step,
        patch_food_regrow_use_fertility=config.patch_food_regrow_use_fertility,
        patch_food_regrow_cap_by_fertility=config.patch_food_regrow_cap_by_fertility,
        patch_food_presence_threshold=config.patch_food_presence_threshold,
        patch_density_radius=config.patch_density_radius,
        patch_density_include_self=config.patch_density_include_self,
        patch_density_normalize=config.patch_density_normalize,
        patch_world_waste_consequence_enable=config.patch_world_waste_consequence_enable,
        patch_waste_regrowth_penalty_enable=config.patch_waste_regrowth_penalty_enable,
        patch_waste_regrowth_penalty_scale=config.patch_waste_regrowth_penalty_scale,
        patch_waste_regrowth_penalty_max=config.patch_waste_regrowth_penalty_max,
        patch_density_regrowth_penalty_enable=config.patch_density_regrowth_penalty_enable,
        patch_density_regrowth_penalty_scale=config.patch_density_regrowth_penalty_scale,
        patch_density_regrowth_penalty_max=config.patch_density_regrowth_penalty_max,
        patch_regrowth_min_multiplier=config.patch_regrowth_min_multiplier,
        build_directory=tmp_path / "patch-context-oracle",
    )

    result = execute_cuda_generation(
        tensors,
        config,
        build_directory=generation_build,
    )

    assert torch.equal(result.patch_stats_by_tick, expected_stats)
    for name, expected in oracle.items():
        assert torch.equal(tensors[name], expected), name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_patch_context_precedes_action_and_defers_new_emission(
    generation_build: Path,
) -> None:
    tensors, _ = _fixture(patch=True)
    config = replace(
        _generation_config(patch=True),
        patch_scent_move_emit=0.25,
        patch_scent_decay_per_step=0.0,
    )

    result = execute_cuda_generation(tensors, config, build_directory=generation_build)

    assert result.patch_stats_by_tick[0, 0].cpu().item() == 0.0
    assert result.patch_stats_by_tick[1, 0].cpu().item() > 0.0
    assert tensors["patch_scent_emit_pending"].sum().cpu().item() > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("patch", [False, True])
def test_stochastic_multistep_actions_and_rng_state_match_numpy_pcg64(
    generation_build: Path,
    patch: bool,
) -> None:
    initial_state = _pcg64_state()
    tensors, _ = _fixture(
        patch=patch,
        policy_action=None,
        pcg64_state=initial_state,
    )
    config = _generation_config(patch=patch)
    expected_actions, expected_state = _expected_stochastic_actions(
        initial_state,
        steps=config.steps,
        cache_width=config.cache_width,
    )
    expected_state_words, expected_buffer_words = pcg64_state_tensors(
        expected_state,
        device="cuda",
    )

    result = execute_cuda_generation(
        tensors,
        config,
        deterministic=False,
        build_directory=generation_build,
    )

    assert torch.equal(result.actions_by_tick_slot.cpu(), torch.from_numpy(expected_actions))
    assert torch.equal(result.pcg64_state_words, expected_state_words)
    assert torch.equal(result.pcg64_buffer_words, expected_buffer_words)
    assert result.pcg64_state_words.data_ptr() == tensors["pcg64_state_words"].data_ptr()
    assert result.pcg64_buffer_words.data_ptr() == tensors["pcg64_buffer_words"].data_ptr()
    assert torch.equal(result.lifecycle_death_counts_by_tick, torch.zeros_like(result.lifecycle_death_counts_by_tick))
    assert torch.equal(result.lifecycle_status_by_tick, torch.zeros_like(result.lifecycle_status_by_tick))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_stochastic_rng_and_state_continue_across_generation_calls(
    generation_build: Path,
) -> None:
    initial_state = _pcg64_state(808)
    full_tensors, _ = _fixture(
        patch=True,
        policy_action=None,
        pcg64_state=initial_state,
    )
    split_tensors, _ = _fixture(
        patch=True,
        policy_action=None,
        pcg64_state=initial_state,
    )
    full_config = replace(_generation_config(patch=True), steps=4)
    split_config = replace(full_config, steps=2)

    full = execute_cuda_generation(
        full_tensors,
        full_config,
        deterministic=False,
        build_directory=generation_build,
    )
    first = execute_cuda_generation(
        split_tensors,
        split_config,
        deterministic=False,
        build_directory=generation_build,
    )
    first_alive_count = int(first.final_alive_count.cpu()[0])
    split_tensors["agent_ids_by_alive_order"] = (
        first.final_alive_agent_ids[:first_alive_count].contiguous()
    )
    second = execute_cuda_generation(
        split_tensors,
        split_config,
        deterministic=False,
        build_directory=generation_build,
    )

    assert torch.equal(
        full.actions_by_tick_slot,
        torch.cat((first.actions_by_tick_slot, second.actions_by_tick_slot), dim=0),
    )
    assert torch.equal(full.pcg64_state_words, second.pcg64_state_words)
    assert torch.equal(full.pcg64_buffer_words, second.pcg64_buffer_words)
    assert torch.equal(full.final_alive_agent_ids, second.final_alive_agent_ids)
    assert torch.equal(
        full.patch_stats_by_tick,
        torch.cat((first.patch_stats_by_tick, second.patch_stats_by_tick), dim=0),
    )
    for name in (
        "slot_to_agent_id",
        "agent_id_to_slot",
        "alive_mask",
        "pos_y",
        "pos_x",
        "agent_grid",
        "food_grid",
        "energy",
        "health",
        "patch_food_amount",
        "patch_scent",
        "patch_waste",
        "patch_scent_emit_pending",
        "patch_waste_emit_pending",
        "action_cache_actions",
        "action_cache_token",
        "action_cache_cursor",
        "action_cache_next",
        "pending_free_agent_ids",
        "pending_free_slots",
        "pending_free_count",
        *EVENT_TAPE_COLUMN_NAMES,
        "event_tape_active_count",
        "event_tape_status",
    ):
        assert torch.equal(full_tensors[name], split_tensors[name]), name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_action_time_death_cull_matches_composed_device_oracle(
    generation_build: Path,
    tmp_path: Path,
) -> None:
    tensors, _ = _fixture(
        patch=False,
        policy_action=REST_ACTION,
        event_active_count=2,
    )
    tensors["energy"][int(SLOTS[0])] = 0.01
    tensors["policy_energy_observation"][int(SLOTS[0])] = 0.01
    tensors["health"][int(SLOTS[2])] = 0.0
    oracle = {name: tensor.clone() for name, tensor in tensors.items()}
    config = replace(
        _generation_config(patch=False),
        steps=1,
        generation=4,
        start_tick=6,
    )
    expected_events = _cpu_owner_death_event_semantics(
        tensors,
        config,
        agent_ids=[7, 9],
        slots=[4, 0],
        reason_codes=[2, 1],
        sequence_start=2,
    )
    actions = torch.full((1, CAPACITY), REST_ACTION, device="cuda", dtype=torch.int32)
    action_module = load_cuda_architecture_b_extension(
        build_directory=tmp_path / "action-device-oracle"
    )
    action_module.eiel_cuda_scripted_owner_action_generation_v1(
        oracle["agent_ids_by_alive_order"],
        oracle["agent_id_to_slot"],
        actions,
        oracle["pos_y"],
        oracle["pos_x"],
        oracle["agent_grid"],
        oracle["food_grid"],
        oracle["patch_food_amount"],
        oracle["patch_scent_emit_pending"],
        oracle["patch_waste_emit_pending"],
        oracle["patch_movement_cost"],
        oracle["energy"],
        oracle["health"],
        False,
        config.move_cost,
        config.gather_cost,
        config.rest_cost,
        config.attack_cost,
        config.action_cost_mult,
        config.rest_cost_mult,
        config.max_energy,
        config.food_energy,
        config.patch_food_gather_amount,
        config.patch_food_presence_threshold,
        config.patch_scent_move_emit,
        config.patch_waste_move_emit,
        config.patch_scent_gather_emit,
        config.patch_waste_consume_emit,
        config.patch_movement_cost_apply_enable,
        config.patch_movement_cost_energy_scale,
    )
    expected = execute_lifecycle_death_cull_cuda(
        **{
            name: oracle[name]
            for name in (
                "slot_to_agent_id",
                "agent_id_to_slot",
                "alive_mask",
                "pos_y",
                "pos_x",
                "agent_grid",
                "energy",
                "health",
                "policy_energy_observation",
                "rest_streak",
                "action_cache_token",
                "action_cache_cursor",
                "action_cache_next",
                "pending_free_agent_ids",
                "pending_free_slots",
                "pending_free_count",
            )
        },
        build_directory=tmp_path / "lifecycle-device-oracle",
    )

    result = execute_cuda_generation(tensors, config, build_directory=generation_build)

    for actual, name in (
        (result.lifecycle_death_agent_ids_by_tick[0], "death_agent_ids"),
        (result.lifecycle_death_slots_by_tick[0], "death_slots"),
        (result.lifecycle_death_reason_codes_by_tick[0], "death_reason_codes"),
    ):
        assert torch.equal(actual, expected[name]), name
    assert result.lifecycle_death_counts_by_tick.cpu().tolist() == [2]
    assert result.lifecycle_status_by_tick.cpu().tolist() == [0]
    assert result.final_alive_count.cpu().tolist() == [2]
    assert result.final_alive_agent_ids.cpu()[:2].tolist() == [2, 5]
    assert result.final_alive_agent_ids.cpu()[2:].eq(-1).all()
    assert result.event_tape_active_count.cpu().tolist() == [4]
    assert result.event_tape_status.cpu().tolist() == [0]
    assert result.event_append_status_by_tick.cpu().tolist() == [0]
    for name in EVENT_TAPE_COLUMN_NAMES:
        actual = tensors[name][2:4].cpu().tolist()
        assert actual == expected_events[name], name
    for name in (
        "slot_to_agent_id",
        "agent_id_to_slot",
        "alive_mask",
        "pos_y",
        "pos_x",
        "agent_grid",
        "energy",
        "health",
        "policy_energy_observation",
        "rest_streak",
        "pending_free_agent_ids",
        "pending_free_slots",
        "pending_free_count",
    ):
        torch.testing.assert_close(
            tensors[name],
            oracle[name],
            rtol=0,
            atol=0,
            equal_nan=True,
            msg=lambda message, field=name: f"{field}: {message}",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_death_event_capacity_failure_is_atomic_and_fail_closed(
    generation_build: Path,
) -> None:
    tensors, _ = _fixture(
        patch=False,
        policy_action=REST_ACTION,
        event_capacity=1,
    )
    tensors["energy"][int(SLOTS[0])] = 0.01
    tensors["policy_energy_observation"][int(SLOTS[0])] = 0.01
    tensors["health"][int(SLOTS[2])] = 0.0
    event_before = {
        name: tensors[name].clone()
        for name in EVENT_TAPE_COLUMN_NAMES
    }
    config = replace(
        _generation_config(patch=False),
        steps=2,
        generation=4,
        start_tick=6,
    )

    result = execute_cuda_generation(tensors, config, build_directory=generation_build)

    assert result.lifecycle_death_counts_by_tick.cpu().tolist() == [2, 0]
    assert result.final_alive_count.cpu().tolist() == [2]
    assert result.final_alive_agent_ids.cpu()[:2].tolist() == [2, 5]
    assert result.final_alive_agent_ids.cpu()[2:].eq(-1).all()
    assert result.event_tape_active_count.cpu().tolist() == [0]
    assert result.event_tape_status.cpu().tolist() == [1]
    assert result.event_append_status_by_tick.cpu().tolist() == [1, 1]
    for name in EVENT_TAPE_COLUMN_NAMES:
        assert torch.equal(tensors[name], event_before[name]), name


def test_generation_contract_fails_closed_for_excluded_scope(tmp_path: Path) -> None:
    assert CUDA_GENERATION_CAPABILITIES["native_cpp_tick_loop"] is True
    assert CUDA_GENERATION_CAPABILITIES["python_tick_loop"] is False
    for capability in ("lifecycle", "birth", "checkpoint", "multi_cell"):
        assert CUDA_GENERATION_CAPABILITIES[capability] is True
    for capability in (
        "cross_cell_tick_synchronous_controller",
        "cross_cell_policy_cache_batch",
        "cross_cell_observation_batch",
        "cross_cell_fixed_mlp_batch",
        "cross_cell_lifecycle_batch",
    ):
        assert CUDA_GENERATION_CAPABILITIES[capability] is True
    assert (
        CUDA_GENERATION_CAPABILITIES["cross_cell_active_mechanics_batch"]
        == "enabled_at_cell_count_16_or_greater"
    )
    for capability in ("gru", "jungle", "wdl"):
        assert CUDA_GENERATION_CAPABILITIES[capability] is False
    assert CUDA_GENERATION_CAPABILITIES["throughput_qualified"] is False
    assert CUDA_GENERATION_CAPABILITIES["promotion_qualified"] is False
    assert CUDA_GENERATION_CAPABILITIES["lifecycle_death_cull"] is True
    assert (
        CUDA_GENERATION_CAPABILITIES["death_event_semantic_construction"]
        == "fixed_capacity_device_sidecar"
    )
    assert CUDA_GENERATION_CAPABILITIES["legacy_and_typed_event_tape_append"] is True
    assert CUDA_GENERATION_CAPABILITIES["fixed_capacity_active_population"] is True
    assert CUDA_GENERATION_CAPABILITIES["device_resident_active_count"] is True
    assert CUDA_GENERATION_CAPABILITIES["death_split_generation_integration"] is True
    assert (
        CUDA_GENERATION_CAPABILITIES["ordinary_birth_generation_integration"]
        == "focused_exact"
    )
    assert CUDA_GENERATION_CAPABILITIES["data_dependent_active_compaction"] is False
    assert CUDA_GENERATION_CAPABILITIES["host_extinction_branch"] is False
    assert (
        CUDA_GENERATION_CAPABILITIES["event_tape_capacity_failure"]
        == "atomic_append_failure_with_sticky_status"
    )
    assert (
        CUDA_GENERATION_CAPABILITIES["stochastic_cpu_rng_parity"]
        == "exact_when_resident_pcg64_state_is_provided"
    )

    config = CudaGenerationConfig(steps=1, patch_enabled=False, hidden_size=HIDDEN_SIZE)
    with pytest.raises(RuntimeError, match="GRU is unsupported"):
        execute_cuda_generation({}, config, policy_family="gru", build_directory=tmp_path)
    with pytest.raises(ValueError, match="prepared generation birth state"):
        execute_cuda_generation(
            {},
            CudaGenerationConfig(
                steps=1,
                patch_enabled=False,
                hidden_size=HIDDEN_SIZE,
                max_births_per_tick=1,
            ),
            build_directory=tmp_path,
        )
    with pytest.raises(ValueError, match="max_births_per_tick must be non-negative"):
        execute_cuda_generation(
            {},
            CudaGenerationConfig(
                steps=1,
                patch_enabled=False,
                hidden_size=HIDDEN_SIZE,
                max_births_per_tick=-1,
            ),
            build_directory=tmp_path,
        )


def test_generation_config_uses_typed_cpu_owner_metadata() -> None:
    metadata = {
        "world": "patch",
        "configured_steps_per_generation": 240,
        "generation": 7,
        "tick": 13,
        "hidden": 64,
        "action_count": 7,
        "policy_tick_stride": 3,
        "max_energy": 20.0,
        "move_cost": 0.75,
        "patch_food_regrow_per_step": 0.03125,
        "patch_world_waste_consequence_enable": True,
        "resource_maintenance_energy_cost_per_tick": 0.035,
        "enable_rest_charge": False,
        "drive_w_hunger_gather": 1.75,
        "drive_fear_decay": 0.0625,
        "reproduce_parent_energy_share": 0.25,
        "reproduce_min_energy_buffer": 2.0,
        "reproduce_cooldown_steps": 7,
        "reproduce_try_prob": 0.375,
        "reproduce_requires_mate": False,
        "genome_upgrade_v2_enable": True,
        "genome_upgrade_v2_mutation_rate_q10": 31,
        "genome_upgrade_v2_mutation_step_q10": 72,
        "genome_sensor_radius_max": 3,
    }

    config = CudaGenerationConfig.from_metadata(metadata, steps=9)

    assert config.steps == 9
    assert config.patch_enabled is True
    assert config.hidden_size == 64
    assert config.generation == 7
    assert config.start_tick == 13
    assert config.cache_width == 3
    assert config.max_energy == 20.0
    assert config.move_cost == 0.75
    assert config.patch_food_regrow_per_step == 0.03125
    assert config.patch_world_waste_consequence_enable is True
    assert config.resource_maintenance_energy_cost_per_tick == 0.035
    assert config.enable_rest_charge is False
    assert config.w_hunger_gather == 1.75
    assert config.drive_fear_decay == 0.0625
    assert config.reproduce_parent_energy_share == 0.25
    assert config.reproduce_min_energy_buffer == 2.0
    assert config.reproduce_cooldown_steps == 7
    assert config.reproduce_try_prob == 0.375
    assert config.reproduce_requires_mate is False
    assert config.genome_upgrade_v2_enable is True
    assert config.genome_upgrade_v2_mutation_rate_q10 == 31
    assert config.genome_upgrade_v2_mutation_step_q10 == 72
    assert config.genome_sensor_radius_max == 3


@pytest.mark.parametrize(
    "metadata",
    (
        {"world": "jungle", "configured_steps_per_generation": 1, "hidden": 4},
        {"world": "flat", "configured_steps_per_generation": 0, "hidden": 4},
        {"world": "flat", "configured_steps_per_generation": 1, "hidden": 0},
    ),
)
def test_generation_config_fails_closed_for_invalid_owner_metadata(
    metadata: dict[str, object],
) -> None:
    with pytest.raises((RuntimeError, ValueError)):
        CudaGenerationConfig.from_metadata(metadata)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_generation_extension_name_is_bound_to_all_source_hashes(
    generation_build: Path,
) -> None:
    module = load_cuda_generation_extension(build_directory=generation_build)
    assert module.__name__.startswith("eiel_cuda_architecture_b_generation_")
    assert len(module.__name__.rsplit("_", 1)[-1]) == 16
