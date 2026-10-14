from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import pytest
import torch

from perf.cuda_architecture_b_lifecycle import (
    FIRST_REMAINING_BIRTH_MISMATCH,
    LIFECYCLE_CAPABILITY_RECEIPT,
    execute_controlled_birth_commit_cuda,
    execute_fixed_mlp_child_scalar_genome_cuda,
    execute_lifecycle_death_cull_cuda,
    execute_turnover_fixed_mlp_child_scalar_genome_cuda,
    execute_turnover_refill_identity_commit_cuda,
    load_cuda_lifecycle_extension,
    pack_cpython_random_state_cuda,
    release_pending_free_slots_cuda,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.fixture(scope="module")
def cuda_build_directory(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("archb-cuda-lifecycle-build")


@pytest.fixture(scope="module")
def cuda_module(cuda_build_directory: Path) -> Any:
    return load_cuda_lifecycle_extension(build_directory=cuda_build_directory)


def _initial_state() -> dict[str, torch.Tensor]:
    capacity = 7
    agent_ids_by_slot = [40, 3, -1, 17, 9, 22, 11]
    alive = [True, True, False, True, True, True, True]
    positions = [(0, 0), (0, 2), (-1, -1), (1, 1), (2, 2), (3, 3), (3, 0)]
    energy = [0.0, 5.0, 0.0, 8.0, 0.0, 7.0, -1.0]
    health = [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0]
    agent_id_to_slot = torch.full((48,), -1, dtype=torch.int32)
    agent_grid = torch.full((4, 4), -1, dtype=torch.int32)
    for slot, agent_id in enumerate(agent_ids_by_slot):
        if alive[slot]:
            agent_id_to_slot[agent_id] = slot
            y, x = positions[slot]
            agent_grid[y, x] = agent_id
    state = {
        "slot_to_agent_id": torch.tensor(agent_ids_by_slot, dtype=torch.int64),
        "agent_id_to_slot": agent_id_to_slot,
        "alive_mask": torch.tensor(alive, dtype=torch.bool),
        "pos_y": torch.tensor([row[0] for row in positions], dtype=torch.int32),
        "pos_x": torch.tensor([row[1] for row in positions], dtype=torch.int32),
        "agent_grid": agent_grid,
        "energy": torch.tensor(energy, dtype=torch.float32),
        "health": torch.tensor(health, dtype=torch.float32),
        "policy_energy_observation": torch.tensor(energy, dtype=torch.float64),
        "rest_streak": torch.arange(1, capacity + 1, dtype=torch.int32),
        "action_cache_token": torch.arange(100, 100 + capacity, dtype=torch.int64),
        "action_cache_cursor": torch.arange(10, 10 + capacity, dtype=torch.int32),
        "action_cache_next": torch.arange(20, 20 + capacity, dtype=torch.int32),
        "pending_free_agent_ids": torch.tensor([2, -1, -1, -1, -1, -1, -1], dtype=torch.int64),
        "pending_free_slots": torch.tensor([2, -1, -1, -1, -1, -1, -1], dtype=torch.int64),
        "pending_free_count": torch.tensor([1], dtype=torch.int64),
    }
    return {name: tensor.cuda().contiguous() for name, tensor in state.items()}


def _direct_owner_oracle(before: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    state = {name: tensor.detach().cpu().clone() for name, tensor in before.items()}
    pending_count = int(state["pending_free_count"][0])
    candidates: list[tuple[int, int, int]] = []
    for slot in range(state["slot_to_agent_id"].numel()):
        agent_id = int(state["slot_to_agent_id"][slot])
        if not bool(state["alive_mask"][slot]) or agent_id < 0:
            continue
        if float(state["health"][slot]) <= 0.0:
            candidates.append((agent_id, slot, 2))
        elif float(state["energy"][slot]) <= 0.0:
            candidates.append((agent_id, slot, 1))
    candidates.sort(key=lambda row: (row[0], row[1]))
    for index, (agent_id, slot, _) in enumerate(candidates):
        y = int(state["pos_y"][slot])
        x = int(state["pos_x"][slot])
        if (
            0 <= y < state["agent_grid"].size(0)
            and 0 <= x < state["agent_grid"].size(1)
            and int(state["agent_grid"][y, x]) == agent_id
        ):
            state["agent_grid"][y, x] = -1
        if 0 <= agent_id < state["agent_id_to_slot"].numel():
            state["agent_id_to_slot"][agent_id] = -1
        state["alive_mask"][slot] = False
        state["slot_to_agent_id"][slot] = -1
        state["pos_y"][slot] = -1
        state["pos_x"][slot] = -1
        state["energy"][slot] = 0.0
        state["health"][slot] = 0.0
        state["policy_energy_observation"][slot] = torch.nan
        state["rest_streak"][slot] = 0
        state["action_cache_token"][slot] = -1
        state["action_cache_cursor"][slot] = -1
        state["action_cache_next"][slot] = -1
        state["pending_free_agent_ids"][pending_count + index] = agent_id
        state["pending_free_slots"][pending_count + index] = slot
    state["pending_free_count"][0] = pending_count + len(candidates)
    state["expected_death_agent_ids"] = torch.tensor([row[0] for row in candidates], dtype=torch.int64)
    state["expected_death_slots"] = torch.tensor([row[1] for row in candidates], dtype=torch.int64)
    state["expected_death_reason_codes"] = torch.tensor([row[2] for row in candidates], dtype=torch.int32)
    remaining = sorted(
        int(state["slot_to_agent_id"][slot])
        for slot in range(state["slot_to_agent_id"].numel())
        if bool(state["alive_mask"][slot]) and int(state["slot_to_agent_id"][slot]) >= 0
    )
    state["expected_alive_agent_ids"] = torch.tensor(remaining, dtype=torch.int64)
    return state


def _birth_state() -> dict[str, torch.Tensor]:
    capacity = 8
    slot_to_agent_id = torch.tensor([40, 3, -1, 17, -1, -1, -1, -1], dtype=torch.int64)
    alive_mask = slot_to_agent_id.ge(0)
    agent_id_to_slot = torch.full((50,), -1, dtype=torch.int32)
    pos_y = torch.tensor([0, 1, -1, 3, -1, -1, -1, -1], dtype=torch.int32)
    pos_x = torch.tensor([0, 1, -1, 3, -1, -1, -1, -1], dtype=torch.int32)
    agent_grid = torch.full((5, 5), -1, dtype=torch.int32)
    for slot in (0, 1, 3):
        agent_id = int(slot_to_agent_id[slot])
        agent_id_to_slot[agent_id] = slot
        agent_grid[int(pos_y[slot]), int(pos_x[slot])] = agent_id
    state = {
        "slot_to_agent_id": slot_to_agent_id,
        "agent_id_to_slot": agent_id_to_slot,
        "slot_tokens": torch.tensor([10, 11, 5, 13, -1, 9, 2, 4], dtype=torch.int64),
        "alive_mask": alive_mask,
        "pos_y": pos_y,
        "pos_x": pos_x,
        "agent_grid": agent_grid,
        "energy": torch.tensor([20.0, 60.0, 4.0, 80.0, 2.0, 3.0, 5.0, 6.0]),
        "health": torch.tensor([1.0, 1.0, 7.0, 1.0, 7.0, 7.0, 7.0, 7.0]),
        "injury": torch.full((capacity,), 0.25, dtype=torch.float32),
        "scar": torch.full((capacity,), 0.5, dtype=torch.float32),
        "age": torch.arange(10, 10 + capacity, dtype=torch.int32),
        "repro_cooldown": torch.zeros((capacity,), dtype=torch.int32),
        "repro_threshold": torch.tensor(
            [100.0, 25.0, 70.0, 25.0, 70.0, 70.0, 70.0, 70.0],
            dtype=torch.float32,
        ),
        "offspring_count": torch.arange(capacity, dtype=torch.int32),
        "lineage_founder_id": torch.tensor([40, 101, -1, -1, -1, -1, -1, -1], dtype=torch.int32),
        "lineage_depth": torch.tensor([0, 2, 9, 4, 9, 9, 9, 9], dtype=torch.int32),
        "lineage_birth_generation": torch.full((capacity,), -1, dtype=torch.int32),
        "lineage_birth_tick": torch.full((capacity,), -1, dtype=torch.int32),
        "patch_habitat_lineage_id": torch.tensor([40, 700, 0, 0, 0, 0, 0, 0], dtype=torch.int32),
        "policy_energy_observation": torch.tensor(
            [20.0, 60.0, torch.nan, 82.0, torch.nan, torch.nan, torch.nan, torch.nan],
            dtype=torch.float64,
        ),
        "survival_steps": torch.arange(20, 20 + capacity, dtype=torch.int32),
        "rest_streak": torch.arange(30, 30 + capacity, dtype=torch.int32),
        "action_cache_token": torch.arange(100, 100 + capacity, dtype=torch.int64),
        "action_cache_cursor": torch.arange(10, 10 + capacity, dtype=torch.int32),
        "action_cache_next": torch.arange(20, 20 + capacity, dtype=torch.int32),
        "slot_free_queue": torch.tensor([99, 6, 6, 2, 5, 4, 7, -1], dtype=torch.int64),
        "slot_free_queue_count": torch.tensor([7], dtype=torch.int64),
        "population_next_id": torch.tensor([41], dtype=torch.int64),
        "parent_slots": torch.tensor([1, 3], dtype=torch.int64),
        "spawn_positions": torch.tensor([[1, 2], [3, 2]], dtype=torch.int32),
    }
    return {name: tensor.cuda().contiguous() for name, tensor in state.items()}


def _direct_birth_identity_oracle(
    before: dict[str, torch.Tensor],
    *,
    parent_energy_share: float,
    minimum_energy_buffer: float,
    start_health: float,
    cooldown_steps: int,
    generation: int,
    tick: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    del minimum_energy_buffer
    state = {name: tensor.detach().cpu().clone() for name, tensor in before.items()}
    parent_slots = state["parent_slots"].tolist()
    spawn_positions = state["spawn_positions"].tolist()
    free_count = int(state["slot_free_queue_count"][0])
    queue_head = 0
    next_id = max(
        1,
        int(state["population_next_id"][0]),
        max(int(value) + 1 for value in state["slot_to_agent_id"] if int(value) >= 0),
    )
    child_slots: list[int] = []
    parent_ids: list[int] = []
    child_ids: list[int] = []
    for birth, parent_slot in enumerate(parent_slots):
        while queue_head < free_count:
            candidate = int(state["slot_free_queue"][queue_head])
            queue_head += 1
            if (
                0 <= candidate < state["slot_to_agent_id"].numel()
                and not bool(state["alive_mask"][candidate])
                and int(state["slot_to_agent_id"][candidate]) < 0
                and candidate not in child_slots
            ):
                break
        child_slot = candidate
        parent_id = int(state["slot_to_agent_id"][parent_slot])
        child_id = next_id + birth
        child_y, child_x = spawn_positions[birth]
        parent_energy = float(state["policy_energy_observation"][parent_slot])
        child_energy = parent_energy * parent_energy_share

        state["energy"][parent_slot] = parent_energy - child_energy
        state["policy_energy_observation"][parent_slot] = parent_energy - child_energy
        state["repro_cooldown"][parent_slot] = cooldown_steps
        state["offspring_count"][parent_slot] += 1
        state["agent_id_to_slot"][child_id] = child_slot
        state["slot_to_agent_id"][child_slot] = child_id
        state["slot_tokens"][child_slot] = max(1, int(state["slot_tokens"][child_slot]) + 1)
        state["alive_mask"][child_slot] = True
        state["pos_y"][child_slot] = child_y
        state["pos_x"][child_slot] = child_x
        state["agent_grid"][child_y, child_x] = child_id
        state["energy"][child_slot] = child_energy
        state["policy_energy_observation"][child_slot] = child_energy
        state["health"][child_slot] = start_health
        state["injury"][child_slot] = 0.0
        state["scar"][child_slot] = 0.0
        state["age"][child_slot] = 0
        state["survival_steps"][child_slot] = 0
        state["rest_streak"][child_slot] = 0
        state["repro_cooldown"][child_slot] = 0
        state["repro_threshold"][child_slot] = state["repro_threshold"][parent_slot]
        state["offspring_count"][child_slot] = 0
        founder = int(state["lineage_founder_id"][parent_slot])
        state["lineage_founder_id"][child_slot] = founder if founder >= 0 else parent_id
        state["lineage_depth"][child_slot] = state["lineage_depth"][parent_slot] + 1
        state["lineage_birth_generation"][child_slot] = generation
        state["lineage_birth_tick"][child_slot] = tick
        patch_lineage = int(state["patch_habitat_lineage_id"][parent_slot])
        state["patch_habitat_lineage_id"][child_slot] = patch_lineage or parent_id
        state["action_cache_token"][child_slot] = -1
        state["action_cache_cursor"][child_slot] = -1
        state["action_cache_next"][child_slot] = -1
        parent_ids.append(parent_id)
        child_ids.append(child_id)
        child_slots.append(child_slot)
    state["population_next_id"][0] = next_id + len(parent_slots)
    remaining_queue = [
        int(slot)
        for slot in state["slot_free_queue"][queue_head:free_count]
        if (
            0 <= int(slot) < state["slot_to_agent_id"].numel()
            and not bool(state["alive_mask"][int(slot)])
            and int(state["slot_to_agent_id"][int(slot)]) < 0
        )
    ]
    state["slot_free_queue"].fill_(-1)
    state["slot_free_queue"][: len(remaining_queue)] = torch.tensor(remaining_queue)
    state["slot_free_queue_count"][0] = len(remaining_queue)
    outputs = {
        "parent_agent_ids": torch.tensor(parent_ids, dtype=torch.int64),
        "child_agent_ids": torch.tensor(child_ids, dtype=torch.int64),
        "child_slots": torch.tensor(child_slots, dtype=torch.int64),
        "alive_agent_ids_by_order": torch.tensor(
            sorted(int(value) for value in state["slot_to_agent_id"] if int(value) >= 0),
            dtype=torch.int64,
        ),
    }
    return state, outputs


_SCALAR_INT32_FIELDS = (
    "genome_hidden_size",
    "genome_memory_dim",
    "genome_sensor_channel_mask",
    "genome_sensor_radius",
    "genome_introspection_channel_mask",
    "genome_self_regulation_control_mask",
    "genome_self_modification_target_mask",
    "genome_directed_inheritance_channel_mask",
)
_SCALAR_FLOAT32_FIELDS = (
    "genome_exploration_bias",
    "genome_aggression_bias",
    "genome_mutation_rate",
    "genome_reproduction_threshold",
    "genome_action_temperature",
    "genome_drive_inertia",
    "genome_drive_noise",
    "genome_drive_logit_gain",
    "genome_pred_lr",
    "genome_drive_sp_hunger",
    "genome_drive_sp_pain",
    "genome_drive_sp_fatigue",
    "genome_drive_sp_fear",
    "genome_exp_prior_strength",
    "genome_exp_prior_noise",
    "genome_drive_birth_noise",
    "repro_threshold",
    "gru_upkeep_cost",
    "policy_attack_bias",
    "policy_action_temperature",
    "policy_drive_logit_gain",
    "drive_inertia",
    "drive_noise",
    "pred_lr",
)


def _scalar_genome_state() -> dict[str, torch.Tensor]:
    capacity = 8
    sentinel_i32 = torch.full((capacity,), -99, dtype=torch.int32)
    sentinel_f32 = torch.full((capacity,), -99.0, dtype=torch.float32)
    state = {name: sentinel_i32.clone() for name in _SCALAR_INT32_FIELDS}
    state.update({name: sentinel_f32.clone() for name in _SCALAR_FLOAT32_FIELDS})
    parent_values = {
        1: {
            "genome_hidden_size": 64,
            "genome_memory_dim": 16,
            "genome_sensor_channel_mask": 1022,
            "genome_sensor_radius": 3,
            "genome_introspection_channel_mask": 65537,
            "genome_self_regulation_control_mask": 63,
            "genome_self_modification_target_mask": 35,
            "genome_directed_inheritance_channel_mask": 127,
            "genome_exploration_bias": 0.123456789,
            "genome_aggression_bias": -0.234567891,
            "genome_mutation_rate": 0.0,
            "genome_reproduction_threshold": 25.123456789,
            "genome_action_temperature": 1.234567891,
            "genome_drive_inertia": 0.876543219,
            "genome_drive_noise": 0.023456789,
            "genome_drive_logit_gain": 1.34567891,
            "genome_pred_lr": 0.087654321,
            "genome_drive_sp_hunger": 0.554321987,
            "genome_drive_sp_pain": 0.054321987,
            "genome_drive_sp_fatigue": 0.154321987,
            "genome_drive_sp_fear": 0.104321987,
            "genome_exp_prior_strength": 0.354321987,
            "genome_exp_prior_noise": 0.054321987,
            "genome_drive_birth_noise": 0.034321987,
        },
        3: {
            "genome_hidden_size": 128,
            "genome_memory_dim": 64,
            "genome_sensor_channel_mask": 2048,
            "genome_sensor_radius": 10,
            "genome_introspection_channel_mask": 32770,
            "genome_self_regulation_control_mask": 34,
            "genome_self_modification_target_mask": 96,
            "genome_directed_inheritance_channel_mask": 65,
            "genome_exploration_bias": -0.333333343,
            "genome_aggression_bias": 0.444444451,
            "genome_mutation_rate": 0.0,
            "genome_reproduction_threshold": 44.44444451,
            "genome_action_temperature": 0.987654329,
            "genome_drive_inertia": 0.765432109,
            "genome_drive_noise": 0.034567891,
            "genome_drive_logit_gain": 0.94567891,
            "genome_pred_lr": 0.123456789,
            "genome_drive_sp_hunger": 0.654321987,
            "genome_drive_sp_pain": 0.064321987,
            "genome_drive_sp_fatigue": 0.164321987,
            "genome_drive_sp_fear": 0.114321987,
            "genome_exp_prior_strength": 0.454321987,
            "genome_exp_prior_noise": 0.064321987,
            "genome_drive_birth_noise": 0.044321987,
        },
    }
    for slot, values in parent_values.items():
        for name, value in values.items():
            state[name][slot] = value
    state["repro_threshold"].copy_(state["genome_reproduction_threshold"])
    state["policy_mem_dim"] = torch.full((capacity,), 99, dtype=torch.int64)
    state["policy_use_gru"] = torch.zeros((capacity,), dtype=torch.bool)
    state["policy_use_gru"][[2, 6]] = True
    return {name: tensor.cuda().contiguous() for name, tensor in state.items()}


def _canon_genome_double(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    value32 = float(torch.tensor(value, dtype=torch.float32))
    return round(value32 * 100_000_000.0) / 100_000_000.0


def _canon_genome_float(value: float) -> float:
    return float(torch.tensor(_canon_genome_double(value), dtype=torch.float32))


def _direct_scalar_genome_oracle(
    before: dict[str, torch.Tensor],
    *,
    parent_slots: list[int],
    mate_slots: list[int] | None = None,
    child_slots: list[int],
    python_random_state: tuple[object, ...],
    sensor_radius_max: int,
) -> tuple[dict[str, torch.Tensor], tuple[object, ...], int]:
    state = {name: tensor.detach().cpu().clone() for name, tensor in before.items()}
    rng = random.Random()
    rng.setstate(python_random_state)
    semantic_draws = 0

    def mutate(value: float, rate: float, sigma: float, lower: float, upper: float) -> float:
        nonlocal semantic_draws
        decision = rng.random()
        semantic_draws += 1
        if decision < rate:
            value += rng.gauss(0.0, sigma)
            semantic_draws += 1
        return min(upper, max(lower, value))

    if mate_slots is None:
        mate_slots = parent_slots
    for parent_slot, mate_slot, child_slot in zip(
        parent_slots, mate_slots, child_slots, strict=True
    ):
        hidden_size_decision = rng.random()
        semantic_draws += 1
        mutation_rate = 0.5 * (
            _canon_genome_double(float(state["genome_mutation_rate"][parent_slot]))
            + _canon_genome_double(float(state["genome_mutation_rate"][mate_slot]))
        )
        scalar_rate = min(1.0, max(0.0, mutation_rate))
        specs = (
            ("genome_exploration_bias", 0.10, -1.0, 1.0),
            ("genome_aggression_bias", 0.15, -2.0, 2.0),
            ("genome_reproduction_threshold", 2.0, 2.0, 90.0),
            ("genome_action_temperature", 0.10, 0.25, 2.5),
            ("genome_drive_inertia", 0.05, 0.0, 0.999),
            ("genome_drive_noise", 0.01, 0.0, 0.10),
            ("genome_drive_logit_gain", 0.20, 0.0, 3.0),
            ("genome_pred_lr", 0.03, 0.0, 1.0),
        )
        child_float_values: dict[str, float] = {}
        for name, sigma, lower, upper in specs:
            base = 0.5 * (
                _canon_genome_double(float(state[name][parent_slot]))
                + _canon_genome_double(float(state[name][mate_slot]))
            )
            child_float_values[name] = _canon_genome_float(
                mutate(base, scalar_rate, sigma, lower, upper)
            )
        child_mutation_rate = mutation_rate
        mutation_rate_decision = rng.random()
        semantic_draws += 1
        if mutation_rate_decision < 0.2:
            child_mutation_rate += rng.gauss(0.0, 0.005)
            semantic_draws += 1
        child_mutation_rate = min(0.20, max(0.001, child_mutation_rate))
        child_float_values["genome_mutation_rate"] = _canon_genome_float(child_mutation_rate)
        memory_probability = min(0.05, max(0.002, 0.25 * child_mutation_rate))
        if rng.random() < memory_probability:
            rng.randrange(2)
            semantic_draws += 1
        semantic_draws += 1

        state["genome_hidden_size"][child_slot] = state["genome_hidden_size"][
            parent_slot if hidden_size_decision < 0.5 else mate_slot
        ]
        state["genome_memory_dim"][child_slot] = 0
        state["policy_mem_dim"][child_slot] = 0
        state["policy_use_gru"][child_slot] = False
        state["gru_upkeep_cost"][child_slot] = 0.0
        state["genome_sensor_channel_mask"][child_slot] = (
            int(state["genome_sensor_channel_mask"][parent_slot]) & 1023
        ) | 1
        radius = min((1, 2, 3), key=lambda value: abs(int(state["genome_sensor_radius"][parent_slot]) - value))
        if sensor_radius_max > 0:
            radius = min(radius, max(1, sensor_radius_max))
        state["genome_sensor_radius"][child_slot] = max(1, radius)
        state["genome_introspection_channel_mask"][child_slot] = int(
            state["genome_introspection_channel_mask"][parent_slot]
        ) & 32767
        state["genome_self_regulation_control_mask"][child_slot] = int(
            state["genome_self_regulation_control_mask"][parent_slot]
        ) & 31
        state["genome_self_modification_target_mask"][child_slot] = int(
            state["genome_self_modification_target_mask"][parent_slot]
        ) & 31
        state["genome_directed_inheritance_channel_mask"][child_slot] = int(
            state["genome_directed_inheritance_channel_mask"][parent_slot]
        ) & 63
        for name in (
            "genome_drive_sp_hunger",
            "genome_drive_sp_pain",
            "genome_drive_sp_fatigue",
            "genome_drive_sp_fear",
            "genome_exp_prior_strength",
        ):
            child_float_values[name] = _canon_genome_float(min(
                1.0,
                max(
                    0.0,
                    0.5
                    * (
                        _canon_genome_double(float(state[name][parent_slot]))
                        + _canon_genome_double(float(state[name][mate_slot]))
                    ),
                ),
            ))
        child_float_values["genome_exp_prior_noise"] = _canon_genome_float(min(
            0.5,
            max(
                0.0,
                0.5
                * (
                    _canon_genome_double(
                        float(state["genome_exp_prior_noise"][parent_slot])
                    )
                    + _canon_genome_double(
                        float(state["genome_exp_prior_noise"][mate_slot])
                    )
                ),
            ),
        ))
        child_float_values["genome_drive_birth_noise"] = _canon_genome_float(min(
            0.2,
            max(
                0.0,
                0.5
                * (
                    _canon_genome_double(
                        float(state["genome_drive_birth_noise"][parent_slot])
                    )
                    + _canon_genome_double(
                        float(state["genome_drive_birth_noise"][mate_slot])
                    )
                ),
            ),
        ))
        for name, value in child_float_values.items():
            state[name][child_slot] = value
        state["repro_threshold"][child_slot] = child_float_values["genome_reproduction_threshold"]
        state["policy_attack_bias"][child_slot] = child_float_values["genome_aggression_bias"]
        state["policy_action_temperature"][child_slot] = child_float_values["genome_action_temperature"]
        state["policy_drive_logit_gain"][child_slot] = child_float_values["genome_drive_logit_gain"]
        state["drive_inertia"][child_slot] = child_float_values["genome_drive_inertia"]
        state["drive_noise"][child_slot] = child_float_values["genome_drive_noise"]
        state["pred_lr"][child_slot] = child_float_values["genome_pred_lr"]
    return state, rng.getstate(), semantic_draws


def test_resident_death_cull_matches_direct_owner_oracle_exactly(
    cuda_module: Any,
    cuda_build_directory: Path,
) -> None:
    del cuda_module
    state = _initial_state()
    expected = _direct_owner_oracle(state)
    pointers_before = {name: tensor.data_ptr() for name, tensor in state.items()}

    result = execute_lifecycle_death_cull_cuda(
        **state,
        build_directory=cuda_build_directory,
    )

    assert all(tensor.is_cuda for tensor in result.values())
    assert pointers_before == {name: tensor.data_ptr() for name, tensor in state.items()}
    torch.cuda.synchronize()
    for name, actual in state.items():
        torch.testing.assert_close(
            actual.cpu(),
            expected[name],
            rtol=0,
            atol=0,
            equal_nan=True,
            msg=lambda message, field=name: f"{field}: {message}",
        )
    assert int(result["status"].cpu()[0]) == 0
    death_count = int(result["death_count"].cpu()[0])
    alive_count = int(result["alive_count"].cpu()[0])
    torch.testing.assert_close(
        result["death_agent_ids"].cpu()[:death_count],
        expected["expected_death_agent_ids"],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        result["death_slots"].cpu()[:death_count],
        expected["expected_death_slots"],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        result["death_reason_codes"].cpu()[:death_count],
        expected["expected_death_reason_codes"],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        result["alive_agent_ids_by_order"].cpu()[:alive_count],
        expected["expected_alive_agent_ids"],
        rtol=0,
        atol=0,
    )
    assert state["pending_free_agent_ids"].cpu()[:5].tolist() == [2, 3, 9, 11, 40]
    assert state["pending_free_slots"].cpu()[:5].tolist() == [2, 1, 4, 6, 0]


def test_pending_capacity_error_fails_without_mutating_state(
    cuda_module: Any,
    cuda_build_directory: Path,
) -> None:
    del cuda_module
    state = _initial_state()
    state["pending_free_count"].fill_(state["pending_free_agent_ids"].numel())
    before = {name: tensor.clone() for name, tensor in state.items()}

    result = execute_lifecycle_death_cull_cuda(
        **state,
        build_directory=cuda_build_directory,
    )
    torch.cuda.synchronize()

    assert int(result["status"].cpu()[0]) == 2
    assert int(result["death_count"].cpu()[0]) == 0
    for name, actual in state.items():
        torch.testing.assert_close(actual, before[name], rtol=0, atol=0, equal_nan=True)


def test_surface_rejects_host_state(cuda_module: Any) -> None:
    state = _initial_state()
    state["slot_to_agent_id"] = state["slot_to_agent_id"].cpu()
    with pytest.raises(RuntimeError, match="slot_to_agent_id must be CUDA-resident"):
        cuda_module.eiel_cuda_architecture_b_lifecycle_death_cull_v1(*state.values())


def test_controlled_birth_identity_commit_matches_cpu_owner_semantics_exactly(
    cuda_module: Any,
    cuda_build_directory: Path,
) -> None:
    del cuda_module
    state = _birth_state()
    config = {
        "parent_energy_share": 0.5,
        "minimum_energy_buffer": 1.0,
        "start_health": 1.0,
        "cooldown_steps": 9,
        "generation": 4,
        "tick": 37,
    }
    expected, expected_outputs = _direct_birth_identity_oracle(state, **config)
    pointers_before = {name: tensor.data_ptr() for name, tensor in state.items()}

    result = execute_controlled_birth_commit_cuda(
        **state,
        **config,
        build_directory=cuda_build_directory,
    )

    assert all(tensor.is_cuda for tensor in result.values())
    assert pointers_before == {name: tensor.data_ptr() for name, tensor in state.items()}
    torch.cuda.synchronize()
    for name, actual in state.items():
        torch.testing.assert_close(
            actual.cpu(),
            expected[name],
            rtol=0,
            atol=0,
            equal_nan=True,
            msg=lambda message, field=name: f"{field}: {message}",
        )
    assert int(result["status"].cpu()[0]) == 0
    assert int(result["committed_count"].cpu()[0]) == 2
    for name in ("parent_agent_ids", "child_agent_ids", "child_slots"):
        torch.testing.assert_close(result[name].cpu(), expected_outputs[name], rtol=0, atol=0)
    alive_count = int(result["alive_count"].cpu()[0])
    torch.testing.assert_close(
        result["alive_agent_ids_by_order"].cpu()[:alive_count],
        expected_outputs["alive_agent_ids_by_order"],
        rtol=0,
        atol=0,
    )
    assert state["slot_free_queue"].cpu()[:3].tolist() == [5, 4, 7]
    assert int(state["population_next_id"].cpu()[0]) == 43


def test_controlled_birth_uses_slot_authority_at_int32_identity_ceiling(
    cuda_module: Any,
    cuda_build_directory: Path,
) -> None:
    del cuda_module
    state = _birth_state()
    child_id = (1 << 31) - 1
    state["population_next_id"].fill_(child_id)
    state["parent_slots"] = state["parent_slots"][:1].clone()
    state["spawn_positions"] = state["spawn_positions"][:1].clone()
    shadow_before = state["agent_id_to_slot"].clone()

    result = execute_controlled_birth_commit_cuda(
        **state,
        parent_energy_share=0.5,
        minimum_energy_buffer=1.0,
        start_health=1.0,
        cooldown_steps=9,
        generation=4,
        tick=37,
        build_directory=cuda_build_directory,
    )
    torch.cuda.synchronize()

    assert int(result["status"].cpu()[0]) == 0
    assert int(result["committed_count"].cpu()[0]) == 1
    assert result["child_agent_ids"].cpu().tolist() == [child_id]
    child_slot = int(result["child_slots"].cpu()[0])
    assert int(state["slot_to_agent_id"][child_slot].cpu()) == child_id
    assert int(state["agent_grid"][1, 2].cpu()) == child_id
    assert int(state["population_next_id"].cpu()[0]) == child_id + 1
    torch.testing.assert_close(
        state["agent_id_to_slot"], shadow_before, rtol=0, atol=0
    )


def test_turnover_refill_identity_uses_two_parents_without_energy_debit(
    cuda_module: Any,
    cuda_build_directory: Path,
) -> None:
    del cuda_module
    state = _birth_state()
    parent_slots = torch.tensor([3, 1], dtype=torch.int64, device="cuda")
    mate_slots = torch.tensor([1, 3], dtype=torch.int64, device="cuda")
    spawn_positions = torch.tensor(
        [[2, 3], [4, 4]], dtype=torch.int32, device="cuda"
    )
    planned_count = torch.tensor([2], dtype=torch.int64, device="cuda")
    parent_energy_before = state["energy"][[1, 3]].clone()
    parent_policy_energy_before = state["policy_energy_observation"][[1, 3]].clone()
    parent_cooldown_before = state["repro_cooldown"][[1, 3]].clone()
    offspring_before = state["offspring_count"][[1, 3]].clone()

    result = execute_turnover_refill_identity_commit_cuda(
        **{
            name: value
            for name, value in state.items()
            if name not in {"parent_slots", "spawn_positions"}
        },
        parent_slots=parent_slots,
        mate_slots=mate_slots,
        spawn_positions=spawn_positions,
        planned_count=planned_count,
        start_energy=60.0,
        start_health=1.0,
        generation=4,
        build_directory=cuda_build_directory,
    )
    torch.cuda.synchronize()

    assert result["status"].cpu().tolist() == [0]
    assert result["committed_count"].cpu().tolist() == [2]
    assert result["parent_agent_ids"][:2].cpu().tolist() == [17, 3]
    assert result["mate_agent_ids"][:2].cpu().tolist() == [3, 17]
    assert result["child_agent_ids"][:2].cpu().tolist() == [41, 42]
    assert result["child_slots"][:2].cpu().tolist() == [6, 2]
    torch.testing.assert_close(state["energy"][[1, 3]], parent_energy_before)
    torch.testing.assert_close(
        state["policy_energy_observation"][[1, 3]], parent_policy_energy_before
    )
    torch.testing.assert_close(
        state["repro_cooldown"][[1, 3]], parent_cooldown_before
    )
    torch.testing.assert_close(
        state["offspring_count"][[1, 3]], offspring_before + 2
    )
    for child_slot, position in zip((6, 2), ((2, 3), (4, 4)), strict=True):
        assert float(state["energy"][child_slot].cpu()) == 60.0
        assert float(state["policy_energy_observation"][child_slot].cpu()) == 60.0
        assert int(state["lineage_founder_id"][child_slot].cpu()) == 17
        assert int(state["lineage_depth"][child_slot].cpu()) == 5
        assert int(state["lineage_birth_generation"][child_slot].cpu()) == 4
        assert int(state["lineage_birth_tick"][child_slot].cpu()) == 0
        assert int(state["patch_habitat_lineage_id"][child_slot].cpu()) == 17
        assert int(state["agent_grid"][position].cpu()) == int(
            state["slot_to_agent_id"][child_slot].cpu()
        )


def test_fixed_mlp_scalar_genome_and_python_rng_match_cpu_owner_exactly(
    cuda_module: Any,
    cuda_build_directory: Path,
) -> None:
    del cuda_module
    identity = _birth_state()
    identity_result = execute_controlled_birth_commit_cuda(
        **identity,
        parent_energy_share=0.5,
        minimum_energy_buffer=1.0,
        start_health=1.0,
        cooldown_steps=9,
        generation=4,
        tick=37,
        build_directory=cuda_build_directory,
    )
    scalar = _scalar_genome_state()
    scalar["repro_threshold"] = identity["repro_threshold"]
    python_random = random.Random(0)
    initial_random_state = python_random.getstate()
    random_state = pack_cpython_random_state_cuda(
        initial_random_state,
        device="cuda",
        semantic_draws=7,
    )
    expected, expected_random_state, expected_draws = _direct_scalar_genome_oracle(
        scalar,
        parent_slots=[1, 3],
        child_slots=[6, 2],
        python_random_state=initial_random_state,
        sensor_radius_max=3,
    )
    pointers_before = {
        name: tensor.data_ptr() for name, tensor in {**scalar, **random_state}.items()
    }

    result = execute_fixed_mlp_child_scalar_genome_cuda(
        slot_to_agent_id=identity["slot_to_agent_id"],
        alive_mask=identity["alive_mask"],
        parent_slots=identity["parent_slots"],
        child_slots=identity_result["child_slots"],
        **random_state,
        **scalar,
        sensor_radius_max=3,
        build_directory=cuda_build_directory,
    )
    torch.cuda.synchronize()

    assert int(result["status"].cpu()[0]) == 0
    assert int(result["committed_count"].cpu()[0]) == 2
    assert int(result["semantic_draws_added"].cpu()[0]) == expected_draws
    assert int(random_state["mutation_py_semantic_draws"].cpu()[0]) == 7 + expected_draws
    assert pointers_before == {
        name: tensor.data_ptr() for name, tensor in {**scalar, **random_state}.items()
    }
    for name, actual in scalar.items():
        torch.testing.assert_close(
            actual.cpu(),
            expected[name],
            rtol=0,
            atol=0,
            msg=lambda message, field=name: f"{field}: {message}",
        )
    expected_internal = tuple(expected_random_state[1])
    torch.testing.assert_close(
        random_state["python_random_words"].cpu(),
        torch.tensor(expected_internal[:-1], dtype=torch.uint32),
        rtol=0,
        atol=0,
    )
    assert int(random_state["python_random_index"].cpu()[0]) == int(expected_internal[-1])
    assert bool(random_state["python_random_has_gauss"].cpu()[0]) is False
    assert float(random_state["python_random_gauss_next"].cpu()[0]) == 0.0


def test_turnover_scalar_genome_uses_both_parents_and_exact_cpython_rng(
    cuda_module: Any,
    cuda_build_directory: Path,
) -> None:
    del cuda_module
    identity = _birth_state()
    parent_slots = torch.tensor([3, 1], dtype=torch.int64, device="cuda")
    mate_slots = torch.tensor([1, 3], dtype=torch.int64, device="cuda")
    identity_result = execute_turnover_refill_identity_commit_cuda(
        **{
            name: value
            for name, value in identity.items()
            if name not in {"parent_slots", "spawn_positions"}
        },
        parent_slots=parent_slots,
        mate_slots=mate_slots,
        spawn_positions=torch.tensor(
            [[2, 3], [4, 4]], dtype=torch.int32, device="cuda"
        ),
        planned_count=torch.tensor([2], dtype=torch.int64, device="cuda"),
        start_energy=60.0,
        start_health=1.0,
        generation=4,
        build_directory=cuda_build_directory,
    )
    scalar = _scalar_genome_state()
    scalar["repro_threshold"] = identity["repro_threshold"]
    python_random = random.Random(0)
    initial_random_state = python_random.getstate()
    random_state = pack_cpython_random_state_cuda(
        initial_random_state, device="cuda", semantic_draws=13
    )
    expected, expected_random_state, expected_draws = _direct_scalar_genome_oracle(
        scalar,
        parent_slots=[3, 1],
        mate_slots=[1, 3],
        child_slots=[6, 2],
        python_random_state=initial_random_state,
        sensor_radius_max=3,
    )

    result = execute_turnover_fixed_mlp_child_scalar_genome_cuda(
        slot_to_agent_id=identity["slot_to_agent_id"],
        alive_mask=identity["alive_mask"],
        parent_slots=parent_slots,
        mate_slots=mate_slots,
        child_slots=identity_result["child_slots"],
        planned_count=torch.tensor([2], dtype=torch.int64, device="cuda"),
        **random_state,
        **scalar,
        sensor_radius_max=3,
        build_directory=cuda_build_directory,
    )
    torch.cuda.synchronize()

    assert result["status"].cpu().tolist() == [0]
    assert result["committed_count"].cpu().tolist() == [2]
    assert result["semantic_draws_added"].cpu().tolist() == [expected_draws]
    assert random_state["mutation_py_semantic_draws"].cpu().tolist() == [
        13 + expected_draws
    ]
    for name, actual in scalar.items():
        torch.testing.assert_close(
            actual.cpu(), expected[name], rtol=0, atol=0, equal_nan=True
        )
    expected_internal = tuple(expected_random_state[1])
    torch.testing.assert_close(
        random_state["python_random_words"].cpu(),
        torch.tensor(expected_internal[:-1], dtype=torch.uint32),
        rtol=0,
        atol=0,
    )
    assert random_state["python_random_index"].cpu().tolist() == [
        expected_internal[-1]
    ]


def test_fresh_gauss_scalar_birth_fails_closed_without_mutation(
    cuda_module: Any,
    cuda_build_directory: Path,
) -> None:
    del cuda_module
    identity = _birth_state()
    identity_result = execute_controlled_birth_commit_cuda(
        **identity,
        parent_energy_share=0.5,
        minimum_energy_buffer=1.0,
        start_health=1.0,
        cooldown_steps=9,
        generation=4,
        tick=37,
        build_directory=cuda_build_directory,
    )
    scalar = _scalar_genome_state()
    scalar["repro_threshold"] = identity["repro_threshold"]
    scalar["genome_mutation_rate"][identity["parent_slots"]] = 1.0
    random_state = pack_cpython_random_state_cuda(random.Random(7).getstate(), device="cuda")
    before = {
        name: tensor.clone() for name, tensor in {**scalar, **random_state}.items()
    }

    result = execute_fixed_mlp_child_scalar_genome_cuda(
        slot_to_agent_id=identity["slot_to_agent_id"],
        alive_mask=identity["alive_mask"],
        parent_slots=identity["parent_slots"],
        child_slots=identity_result["child_slots"],
        **random_state,
        **scalar,
        sensor_radius_max=3,
        build_directory=cuda_build_directory,
    )
    torch.cuda.synchronize()

    assert int(result["status"].cpu()[0]) == 22
    assert int(result["committed_count"].cpu()[0]) == 0
    assert int(result["semantic_draws_added"].cpu()[0]) == 0
    for name, actual in {**scalar, **random_state}.items():
        torch.testing.assert_close(actual, before[name], rtol=0, atol=0)


def test_boundary_gauss_tape_closes_fresh_scalar_mutation_exactly(
    cuda_module: Any,
    cuda_build_directory: Path,
) -> None:
    del cuda_module
    identity = _birth_state()
    identity_result = execute_controlled_birth_commit_cuda(
        **identity,
        parent_energy_share=0.5,
        minimum_energy_buffer=1.0,
        start_health=1.0,
        cooldown_steps=9,
        generation=4,
        tick=37,
        build_directory=cuda_build_directory,
    )
    scalar = _scalar_genome_state()
    scalar["repro_threshold"] = identity["repro_threshold"]
    scalar["genome_mutation_rate"][identity["parent_slots"]] = 1.0
    python_random = random.Random(7)
    initial_random_state = python_random.getstate()
    random_state = pack_cpython_random_state_cuda(
        initial_random_state,
        device="cuda",
        semantic_draws=13,
        gauss_tape_raw_capacity=512,
    )
    expected, expected_random_state, expected_draws = _direct_scalar_genome_oracle(
        scalar,
        parent_slots=[1, 3],
        child_slots=[6, 2],
        python_random_state=initial_random_state,
        sensor_radius_max=3,
    )
    pointers_before = {
        name: tensor.data_ptr() for name, tensor in {**scalar, **random_state}.items()
    }

    result = execute_fixed_mlp_child_scalar_genome_cuda(
        slot_to_agent_id=identity["slot_to_agent_id"],
        alive_mask=identity["alive_mask"],
        parent_slots=identity["parent_slots"],
        child_slots=identity_result["child_slots"],
        **random_state,
        **scalar,
        sensor_radius_max=3,
        build_directory=cuda_build_directory,
    )
    torch.cuda.synchronize()

    assert int(result["status"].cpu()[0]) == 0
    assert int(result["committed_count"].cpu()[0]) == 2
    assert int(result["semantic_draws_added"].cpu()[0]) == expected_draws
    assert int(random_state["mutation_py_semantic_draws"].cpu()[0]) == 13 + expected_draws
    assert int(random_state["python_random_raw_draws"].cpu()[0]) > 0
    assert pointers_before == {
        name: tensor.data_ptr() for name, tensor in {**scalar, **random_state}.items()
    }
    for name, actual in scalar.items():
        torch.testing.assert_close(actual.cpu(), expected[name], rtol=0, atol=0)
    expected_internal = tuple(expected_random_state[1])
    torch.testing.assert_close(
        random_state["python_random_words"].cpu(),
        torch.tensor(expected_internal[:-1], dtype=torch.uint32),
        rtol=0,
        atol=0,
    )
    assert int(random_state["python_random_index"].cpu()[0]) == int(expected_internal[-1])
    expected_gauss = expected_random_state[2]
    assert bool(random_state["python_random_has_gauss"].cpu()[0]) is (expected_gauss is not None)
    assert float(random_state["python_random_gauss_next"].cpu()[0]) == (
        0.0 if expected_gauss is None else float(expected_gauss)
    )


def test_cached_gauss_scalar_mutation_and_rng_cache_consumption_are_exact(
    cuda_module: Any,
    cuda_build_directory: Path,
) -> None:
    del cuda_module
    identity = _birth_state()
    identity_result = execute_controlled_birth_commit_cuda(
        **identity,
        parent_energy_share=0.5,
        minimum_energy_buffer=1.0,
        start_health=1.0,
        cooldown_steps=9,
        generation=4,
        tick=37,
        build_directory=cuda_build_directory,
    )
    scalar = _scalar_genome_state()
    scalar["repro_threshold"] = identity["repro_threshold"]
    scalar["genome_mutation_rate"][1] = 0.2
    python_random = random.Random(6)
    python_random.gauss(0.0, 1.0)
    initial_random_state = python_random.getstate()
    random_state = pack_cpython_random_state_cuda(initial_random_state, device="cuda")
    expected, expected_random_state, expected_draws = _direct_scalar_genome_oracle(
        scalar,
        parent_slots=[1],
        child_slots=[6],
        python_random_state=initial_random_state,
        sensor_radius_max=3,
    )

    result = execute_fixed_mlp_child_scalar_genome_cuda(
        slot_to_agent_id=identity["slot_to_agent_id"],
        alive_mask=identity["alive_mask"],
        parent_slots=identity["parent_slots"][:1],
        child_slots=identity_result["child_slots"][:1],
        **random_state,
        **scalar,
        sensor_radius_max=3,
        build_directory=cuda_build_directory,
    )
    torch.cuda.synchronize()

    assert int(result["status"].cpu()[0]) == 0
    assert int(result["semantic_draws_added"].cpu()[0]) == expected_draws
    for name, actual in scalar.items():
        torch.testing.assert_close(actual.cpu(), expected[name], rtol=0, atol=0)
    expected_internal = tuple(expected_random_state[1])
    torch.testing.assert_close(
        random_state["python_random_words"].cpu(),
        torch.tensor(expected_internal[:-1], dtype=torch.uint32),
        rtol=0,
        atol=0,
    )
    assert int(random_state["python_random_index"].cpu()[0]) == int(expected_internal[-1])
    assert bool(random_state["python_random_has_gauss"].cpu()[0]) is False
    assert float(random_state["python_random_gauss_next"].cpu()[0]) == 0.0


@pytest.mark.parametrize(
    ("failure", "expected_status"),
    (
        ("agent_id_capacity", 14),
        ("free_slot_capacity", 13),
        ("spawn_reservation", 15),
        ("parent_order", 11),
    ),
)
def test_controlled_birth_plan_errors_fail_without_mutation(
    cuda_module: Any,
    cuda_build_directory: Path,
    failure: str,
    expected_status: int,
) -> None:
    del cuda_module
    state = _birth_state()
    if failure == "agent_id_capacity":
        state["population_next_id"].fill_((1 << 31) - 1)
    elif failure == "free_slot_capacity":
        state["slot_free_queue_count"].zero_()
    elif failure == "spawn_reservation":
        state["spawn_positions"][1].copy_(state["spawn_positions"][0])
    else:
        state["parent_slots"] = state["parent_slots"].flip(0).contiguous()
    before = {name: tensor.clone() for name, tensor in state.items()}

    result = execute_controlled_birth_commit_cuda(
        **state,
        parent_energy_share=0.5,
        minimum_energy_buffer=1.0,
        start_health=1.0,
        cooldown_steps=9,
        generation=4,
        tick=37,
        build_directory=cuda_build_directory,
    )
    torch.cuda.synchronize()

    assert int(result["status"].cpu()[0]) == expected_status
    assert int(result["committed_count"].cpu()[0]) == 0
    for name, actual in state.items():
        torch.testing.assert_close(actual, before[name], rtol=0, atol=0, equal_nan=True)


def test_pending_deaths_release_to_birth_queue_once_in_stable_order(
    cuda_module: Any,
    cuda_build_directory: Path,
) -> None:
    del cuda_module
    slot_to_agent_id = torch.tensor([10, 11, -1, -1, -1, -1], device="cuda")
    alive_mask = slot_to_agent_id.ge(0)
    slot_free_queue = torch.full((6,), -1, dtype=torch.int64, device="cuda")
    slot_free_queue[:2] = torch.tensor([5, 4], dtype=torch.int64, device="cuda")
    slot_free_queue_count = torch.tensor([2], dtype=torch.int64, device="cuda")
    pending_free_slots = torch.tensor(
        [2, 3, -1, -1, -1, -1], dtype=torch.int64, device="cuda"
    )
    pending_free_count = torch.tensor([2], dtype=torch.int64, device="cuda")
    pending_release_cursor = torch.zeros((1,), dtype=torch.int64, device="cuda")

    first = release_pending_free_slots_cuda(
        slot_to_agent_id=slot_to_agent_id,
        alive_mask=alive_mask,
        slot_free_queue=slot_free_queue,
        slot_free_queue_count=slot_free_queue_count,
        pending_free_slots=pending_free_slots,
        pending_free_count=pending_free_count,
        pending_release_cursor=pending_release_cursor,
        build_directory=cuda_build_directory,
    )
    second = release_pending_free_slots_cuda(
        slot_to_agent_id=slot_to_agent_id,
        alive_mask=alive_mask,
        slot_free_queue=slot_free_queue,
        slot_free_queue_count=slot_free_queue_count,
        pending_free_slots=pending_free_slots,
        pending_free_count=pending_free_count,
        pending_release_cursor=pending_release_cursor,
        build_directory=cuda_build_directory,
    )

    assert first["status"].cpu().tolist() == [0]
    assert first["appended_count"].cpu().tolist() == [2]
    assert second["status"].cpu().tolist() == [0]
    assert second["appended_count"].cpu().tolist() == [0]
    assert slot_free_queue_count.cpu().tolist() == [4]
    assert pending_release_cursor.cpu().tolist() == [2]
    assert slot_free_queue[:4].cpu().tolist() == [5, 4, 2, 3]


def test_pending_free_release_contradiction_is_atomic(
    cuda_module: Any,
    cuda_build_directory: Path,
) -> None:
    del cuda_module
    slot_to_agent_id = torch.tensor([10, -1, -1], device="cuda")
    alive_mask = slot_to_agent_id.ge(0)
    queue = torch.tensor([1, -1, -1], dtype=torch.int64, device="cuda")
    queue_count = torch.tensor([1], dtype=torch.int64, device="cuda")
    pending = torch.tensor([1, -1, -1], dtype=torch.int64, device="cuda")
    pending_count = torch.tensor([1], dtype=torch.int64, device="cuda")
    cursor = torch.zeros((1,), dtype=torch.int64, device="cuda")
    before = queue.clone()

    result = release_pending_free_slots_cuda(
        slot_to_agent_id=slot_to_agent_id,
        alive_mask=alive_mask,
        slot_free_queue=queue,
        slot_free_queue_count=queue_count,
        pending_free_slots=pending,
        pending_free_count=pending_count,
        pending_release_cursor=cursor,
        build_directory=cuda_build_directory,
    )

    assert result["status"].cpu().tolist() == [32]
    assert result["appended_count"].cpu().tolist() == [0]
    assert torch.equal(queue, before)
    assert queue_count.cpu().tolist() == [1]
    assert cursor.cpu().tolist() == [0]


def test_birth_subblock_and_broader_lifecycle_claims_are_truthful() -> None:
    assert "child drives/PCG64" in FIRST_REMAINING_BIRTH_MISMATCH
    for supported in (
        "deterministic_action_time_death_cull",
        "grid_map_alive_cleanup",
        "stable_agent_id_death_order",
        "generation_pending_free_bookkeeping",
        "ordinary_birth_free_queue_release",
        "controlled_birth_allocation_identity_commit",
        "deterministic_parent_birth_order",
        "free_slot_and_agent_id_allocation",
        "spawn_reservation",
        "stable_birth_identity_maps_alive_order",
        "birth_capacity_fail_closed",
        "fixed_mlp_scalar_genome_no_fresh_gauss_commit",
        "device_resident_cpython_random_transition",
        "fresh_gauss_scalar_mutation",
        "generation_boundary_gauss_tape",
        "child_scalar_genome_commit",
    ):
        assert LIFECYCLE_CAPABILITY_RECEIPT[supported] is True
    for unsupported in (
        "child_drives_pcg64_commit",
        "fixed_mlp_policy_row_commit",
        "controlled_birth_commit",
        "child_genome_policy_rng_commit",
        "mutation",
        "typed_lifecycle_events",
        "generation_turnover",
        "checkpoint_continuation",
        "multi_cell_batching",
        "jungle",
        "wdl",
        "gru",
        "throughput_qualification",
    ):
        assert LIFECYCLE_CAPABILITY_RECEIPT[unsupported] is False
