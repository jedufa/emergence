"""Deterministic CPU source states for the Phase F v2 forced combinations.

This module converts one validated Phase F fixture case into the complete
CPU-resident inputs needed by the bounded Phase E modular generation owner and
into one independent formal-oracle input bundle per live agent.  It performs
no simulation and no host-to-device transfer.

The adapter intentionally uses the public fixture, modular storage, modular
generation-entry, and generation configuration contracts.  It does not import
test helpers or construct a private runner/checkpoint owner.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from functools import lru_cache
from hashlib import sha256
import json
from typing import Any, Final, Mapping

import numpy as np
import torch

from evolution.phase_f_v2.matrix import (
    build_phase_f_v2_matrix,
    validate_phase_f_v2_matrix,
)
from perf.cuda_architecture_b_boundary import EVENT_COLUMN_DTYPES
from perf.cuda_architecture_b_generation import (
    CudaGenerationConfig,
    EVENT_TAPE_COLUMN_NAMES,
)
from perf.cuda_architecture_b_modular_generation_contract import (
    prepare_modular_generation_entry,
)
from perf.cuda_architecture_b_modular_state_abi import (
    MODULAR_STORAGE_METADATA_KEY,
    MODULAR_STORAGE_PROJECTION_PREFIX,
    modular_storage_projection_payload,
    pack_modular_storage,
)
from perf.cuda_architecture_b_modular_state_bank import (
    validate_modular_cpu_boundary_projection,
)
from perf.cuda_architecture_b_pcg64 import pcg64_state_tensors
from perf.eiel_phase_f_forced_combination_fixture_v2 import (
    MODULAR_RNG_DRAW_COUNT,
    PANEL_AGENT_COUNT,
    validate_phase_f_fixture_case_v2,
)


SCHEMA_VERSION: Final[str] = "eiel_phase_f_v2_forced_combination_source_v1"
FORMAL_INPUT_SCHEMA_VERSION: Final[str] = "eiel_phase_f_v2_formal_oracle_input_v1"
GRID_HEIGHT: Final[int] = 8
GRID_WIDTH: Final[int] = 8
AGENT_CAPACITY: Final[int] = GRID_HEIGHT * GRID_WIDTH
OBSERVATION_WIDTH: Final[int] = 27
ACTION_COUNT: Final[int] = 7
HIDDEN_SIZE: Final[int] = 4
VIEW_RADIUS: Final[int] = 2
POLICY_STRIDE: Final[int] = 1
CACHE_WIDTH: Final[int] = 1
GENERATION_STEPS: Final[int] = 1
EVENT_CAPACITY: Final[int] = AGENT_CAPACITY
DEVELOPMENT_MATURE_START_STEP: Final[int] = 20
DEVELOPMENT_LATE_START_STEP: Final[int] = 40
START_ENERGY: Final[float] = 50.0
MAX_ENERGY: Final[float] = 100.0
START_HEALTH: Final[float] = 1.0

_OWNER_TENSOR_NAMES: Final[tuple[str, ...]] = (
    "agent_ids_by_alive_order",
    "agent_id_to_slot",
    "slot_to_agent_id",
    "alive_mask",
    "pos_y",
    "pos_x",
    "agent_grid",
    "food_grid",
    "energy",
    "health",
    "injury",
    "age",
    "policy_energy_observation",
    "rest_streak",
    "survival_steps",
    "last_action",
    "last_last_action",
    "action_switches",
    "action_oscillations",
    "episode0_switches",
    "episode0_oscillations",
    "gru_upkeep_cost",
    "cost_charge_scalar",
    "cost_charge_maturation_scalar",
    "patch_food_amount",
    "patch_fertility",
    "patch_cover",
    "patch_shelter",
    "patch_scent",
    "patch_waste",
    "patch_density",
    "patch_scent_emit_pending",
    "patch_waste_emit_pending",
    "patch_movement_cost",
    "mlp_pack",
    "drives",
    "drive_inertia",
    "drive_noise",
    "prev_drive_delta",
    "drive_volatility",
    "drive_jitter",
    "episode0_drive_jitter_sum",
    "last_energy_delta",
    "last_health_delta",
    "last_injury_delta",
    "selfharm_events",
    "episode0_selfharm_events",
    "drive_noise_samples_by_tick_slot",
    "drive_noise_samples_valid_by_tick_slot",
    "attack_bias",
    "action_temperature",
    "drive_logit_gain",
    "slot_tokens",
    "action_cache_actions",
    "action_cache_token",
    "action_cache_cursor",
    "action_cache_next",
    "pending_free_agent_ids",
    "pending_free_slots",
    "pending_free_count",
    "pcg64_state_words",
    "pcg64_buffer_words",
    "lineage_founder_id",
    "lineage_depth",
)
GENERATION_TENSOR_NAMES: Final[tuple[str, ...]] = (
    *_OWNER_TENSOR_NAMES,
    *EVENT_TAPE_COLUMN_NAMES,
    "event_tape_active_count",
    "event_tape_status",
)


class PhaseFV2SourceError(ValueError):
    """A Phase F source request or materialized source state is invalid."""


@dataclass(frozen=True)
class PhaseFV2FormalOracleInput:
    """One fixture agent's exact inputs to the public Phase E CPU oracle."""

    schema_version: str
    case_id: str
    case_sha256: str
    fixture_sha256: str
    profile_id: str
    source_seed: int
    agent_slot: int
    agent_id: int
    agent_slot_token: int
    planes: dict[str, torch.Tensor]
    modular_observations: torch.Tensor
    fatigue: torch.Tensor
    stage_indices: torch.Tensor
    policy_observations: torch.Tensor
    w1: torch.Tensor
    b1: torch.Tensor
    w2: torch.Tensor
    b2: torch.Tensor
    w3: torch.Tensor
    b3: torch.Tensor
    attack_bias: torch.Tensor
    action_temperature: torch.Tensor
    validity_mask: torch.Tensor
    authority_mask: torch.Tensor
    rng_stream_indices: dict[str, int]

    def call_kwargs(self) -> dict[str, Any]:
        """Return detached call arguments for the public formal composition."""

        tensor_names = (
            "modular_observations",
            "fatigue",
            "stage_indices",
            "policy_observations",
            "w1",
            "b1",
            "w2",
            "b2",
            "w3",
            "b3",
            "attack_bias",
            "action_temperature",
            "validity_mask",
            "authority_mask",
        )
        return {
            "planes": _clone_tensors(self.planes),
            **{
                name: getattr(self, name).detach().clone().contiguous()
                for name in tensor_names
            },
            "rng_stream_indices": dict(self.rng_stream_indices),
        }


@dataclass(frozen=True)
class PhaseFV2SourceState:
    """One complete, data-only CPU source for a Phase F generation cell."""

    schema_version: str
    case_id: str
    case_sha256: str
    fixture_sha256: str
    matrix_sha256: str
    profile_id: str
    role_action: str
    source_seed: int
    owner_metadata: dict[str, Any]
    generation_config: CudaGenerationConfig
    generation_tensors: dict[str, torch.Tensor]
    formal_oracle_inputs: tuple[PhaseFV2FormalOracleInput, ...]
    source_sha256: str


def _clone_tensors(values: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        str(name): tensor.detach().clone().contiguous()
        for name, tensor in values.items()
    }


def _exact_source_seed(value: Any) -> int:
    if type(value) is not int or not 0 <= value < (1 << 63):
        raise PhaseFV2SourceError("source_seed must be an exact uint63 integer")
    return value


@lru_cache(maxsize=1)
def _current_case_seed_bindings() -> dict[str, tuple[str, int]]:
    matrix = validate_phase_f_v2_matrix(build_phase_f_v2_matrix())
    return {
        str(row["case_id"]): (
            str(row["fixture_blueprint"]["fixture_blueprint_sha256"]),
            int(row["fixture_blueprint"]["construction_seed"]),
        )
        for row in matrix["cases"]
    }


def _case_construction_seed(case: Mapping[str, Any]) -> int:
    case_id = str(case["case_id"])
    binding = _current_case_seed_bindings().get(case_id)
    if binding is None:
        raise PhaseFV2SourceError("fixture case is absent from the current matrix")
    blueprint_sha256, construction_seed = binding
    if blueprint_sha256 != case["fixture_blueprint_sha256"]:
        raise PhaseFV2SourceError("fixture blueprint binding is not current")
    return construction_seed


def _tensor_manifest(tensor: torch.Tensor) -> dict[str, Any]:
    if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
        raise PhaseFV2SourceError("source tensors must remain on CPU")
    value = tensor.detach().contiguous()
    if value.requires_grad:
        raise PhaseFV2SourceError("source tensors must not require gradients")
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "sha256": sha256(value.numpy().tobytes(order="C")).hexdigest(),
    }


def _tensor_mapping_manifest(
    values: Mapping[str, torch.Tensor],
) -> dict[str, dict[str, Any]]:
    return {name: _tensor_manifest(values[name]) for name in sorted(values)}


def _oracle_manifest(value: PhaseFV2FormalOracleInput) -> dict[str, Any]:
    tensor_fields = (
        "modular_observations",
        "fatigue",
        "stage_indices",
        "policy_observations",
        "w1",
        "b1",
        "w2",
        "b2",
        "w3",
        "b3",
        "attack_bias",
        "action_temperature",
        "validity_mask",
        "authority_mask",
    )
    return {
        "schema_version": value.schema_version,
        "case_id": value.case_id,
        "case_sha256": value.case_sha256,
        "fixture_sha256": value.fixture_sha256,
        "profile_id": value.profile_id,
        "source_seed": value.source_seed,
        "agent_slot": value.agent_slot,
        "agent_id": value.agent_id,
        "agent_slot_token": value.agent_slot_token,
        "planes": _tensor_mapping_manifest(value.planes),
        "tensors": {
            name: _tensor_manifest(getattr(value, name)) for name in tensor_fields
        },
        "rng_stream_indices": dict(sorted(value.rng_stream_indices.items())),
    }


def _source_digest(
    *,
    header: Mapping[str, Any],
    owner_metadata: Mapping[str, Any],
    config: CudaGenerationConfig,
    generation_tensors: Mapping[str, torch.Tensor],
    formal_inputs: tuple[PhaseFV2FormalOracleInput, ...],
) -> str:
    payload = {
        "header": dict(header),
        "owner_metadata": deepcopy(dict(owner_metadata)),
        "generation_config": asdict(config),
        "generation_tensors": _tensor_mapping_manifest(generation_tensors),
        "formal_oracle_inputs": [_oracle_manifest(value) for value in formal_inputs],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _policy_pack() -> torch.Tensor:
    packed_width = (
        HIDDEN_SIZE * OBSERVATION_WIDTH
        + HIDDEN_SIZE
        + HIDDEN_SIZE * HIDDEN_SIZE
        + HIDDEN_SIZE
        + ACTION_COUNT * HIDDEN_SIZE
        + ACTION_COUNT
    )
    row = torch.zeros((packed_width,), dtype=torch.float32)
    row[-ACTION_COUNT:] = torch.linspace(0.03, -0.03, ACTION_COUNT, dtype=torch.float32)
    return row.contiguous()


def _split_policy_pack(row: torch.Tensor) -> dict[str, torch.Tensor]:
    offset = 0
    w1_count = HIDDEN_SIZE * OBSERVATION_WIDTH
    w1 = row[offset : offset + w1_count].reshape(1, HIDDEN_SIZE, OBSERVATION_WIDTH)
    offset += w1_count
    b1 = row[offset : offset + HIDDEN_SIZE].reshape(1, HIDDEN_SIZE)
    offset += HIDDEN_SIZE
    w2_count = HIDDEN_SIZE * HIDDEN_SIZE
    w2 = row[offset : offset + w2_count].reshape(1, HIDDEN_SIZE, HIDDEN_SIZE)
    offset += w2_count
    b2 = row[offset : offset + HIDDEN_SIZE].reshape(1, HIDDEN_SIZE)
    offset += HIDDEN_SIZE
    w3_count = ACTION_COUNT * HIDDEN_SIZE
    w3 = row[offset : offset + w3_count].reshape(1, ACTION_COUNT, HIDDEN_SIZE)
    offset += w3_count
    b3 = row[offset : offset + ACTION_COUNT].reshape(1, ACTION_COUNT)
    return _clone_tensors({"w1": w1, "b1": b1, "w2": w2, "b2": b2, "w3": w3, "b3": b3})


def _fixed_positions() -> tuple[tuple[int, int], ...]:
    return tuple((2 * (slot // 4), 2 * (slot % 4)) for slot in range(PANEL_AGENT_COUNT))


def _flat_observations(
    *,
    positions: tuple[tuple[int, int], ...],
    agent_grid: torch.Tensor,
    food_grid: torch.Tensor,
) -> torch.Tensor:
    rows: list[list[float]] = []
    for center_y, center_x in positions:
        row: list[float] = []
        for delta_y in range(-VIEW_RADIUS, VIEW_RADIUS + 1):
            for delta_x in range(-VIEW_RADIUS, VIEW_RADIUS + 1):
                y = center_y + delta_y
                x = center_x + delta_x
                value = 0.0
                if 0 <= y < GRID_HEIGHT and 0 <= x < GRID_WIDTH:
                    if int(agent_grid[y, x]) != -1:
                        value = -1.0
                    elif int(food_grid[y, x]) == 1:
                        value = 1.0
                row.append(value)
        row.extend((START_ENERGY / MAX_ENERGY, START_HEALTH / START_HEALTH))
        rows.append(row)
    observations = torch.tensor(rows, dtype=torch.float32)
    if tuple(observations.shape) != (PANEL_AGENT_COUNT, OBSERVATION_WIDTH):
        raise PhaseFV2SourceError("fixed observation construction changed width")
    return observations.contiguous()


def _generation_config(source_seed: int) -> CudaGenerationConfig:
    return CudaGenerationConfig(
        steps=GENERATION_STEPS,
        patch_enabled=False,
        hidden_size=HIDDEN_SIZE,
        action_count=ACTION_COUNT,
        cache_width=CACHE_WIDTH,
        view_radius=VIEW_RADIUS,
        max_energy=MAX_ENERGY,
        start_health=START_HEALTH,
        rng_seed=source_seed,
        move_cost=0.0,
        gather_cost=0.0,
        rest_cost=0.0,
        attack_cost=0.0,
        food_energy=0.0,
        max_food=0,
        food_spawn_per_step=0.0,
        active_mechanics_enabled=False,
        max_births_per_tick=0,
        generation_turnover_enabled=False,
        population_size=PANEL_AGENT_COUNT,
        start_energy=START_ENERGY,
        structural_development_mature_start_step=(DEVELOPMENT_MATURE_START_STEP),
        structural_development_late_start_step=DEVELOPMENT_LATE_START_STEP,
        enable_combat=False,
        env_policy_alias=True,
    )


def _empty_capacity_vector(
    *, dtype: torch.dtype, fill: int | float | bool = 0
) -> torch.Tensor:
    return torch.full((AGENT_CAPACITY,), fill, dtype=dtype)


def _owner_tensors(
    *, case: Mapping[str, Any], source_seed: int
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    records = [agent["record"] for agent in case["agents"]]
    ids = [int(record["agent_id"]) for record in records]
    tokens = [int(record["agent_slot_token"]) for record in records]
    positions = _fixed_positions()

    slot_to_agent_id = _empty_capacity_vector(dtype=torch.int64, fill=-1)
    slot_tokens = _empty_capacity_vector(dtype=torch.int64, fill=-1)
    alive_mask = _empty_capacity_vector(dtype=torch.bool, fill=False)
    pos_y = _empty_capacity_vector(dtype=torch.int32, fill=-1)
    pos_x = _empty_capacity_vector(dtype=torch.int32, fill=-1)
    agent_grid = torch.full((GRID_HEIGHT, GRID_WIDTH), -1, dtype=torch.int32)
    for slot, (agent_id, token, position) in enumerate(
        zip(ids, tokens, positions, strict=True)
    ):
        y, x = position
        slot_to_agent_id[slot] = agent_id
        slot_tokens[slot] = token
        alive_mask[slot] = True
        pos_y[slot] = y
        pos_x[slot] = x
        agent_grid[y, x] = agent_id

    agent_id_to_slot = torch.full((max(ids) + 1,), -1, dtype=torch.int32)
    for slot, agent_id in enumerate(ids):
        agent_id_to_slot[agent_id] = slot

    food_grid = torch.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=torch.int32)
    observations = _flat_observations(
        positions=positions, agent_grid=agent_grid, food_grid=food_grid
    )
    energy = _empty_capacity_vector(dtype=torch.float32, fill=0.0)
    health = _empty_capacity_vector(dtype=torch.float32, fill=0.0)
    energy[:PANEL_AGENT_COUNT] = START_ENERGY
    health[:PANEL_AGENT_COUNT] = START_HEALTH
    policy_row = _policy_pack()
    mlp_pack = torch.zeros(
        (AGENT_CAPACITY, int(policy_row.numel())), dtype=torch.float32
    )
    mlp_pack[:PANEL_AGENT_COUNT] = policy_row
    state_words, buffer_words = pcg64_state_tensors(
        deepcopy(np.random.PCG64(source_seed).state), device="cpu"
    )
    shape = (GRID_HEIGHT, GRID_WIDTH)
    survival_steps = _empty_capacity_vector(dtype=torch.int32)
    family_id = str(case["family_id"])
    for slot, agent in enumerate(case["agents"]):
        variant_id = str(agent["variant_id"])
        if (family_id, variant_id) in {("f03", "v6"), ("f07", "v1")}:
            survival_steps[slot] = DEVELOPMENT_MATURE_START_STEP
        elif (family_id, variant_id) == ("f07", "v2"):
            survival_steps[slot] = DEVELOPMENT_LATE_START_STEP

    tensors = {
        "agent_ids_by_alive_order": torch.tensor(ids, dtype=torch.int64),
        "agent_id_to_slot": agent_id_to_slot,
        "slot_to_agent_id": slot_to_agent_id,
        "alive_mask": alive_mask,
        "pos_y": pos_y,
        "pos_x": pos_x,
        "agent_grid": agent_grid,
        "food_grid": food_grid,
        "energy": energy,
        "health": health,
        "injury": _empty_capacity_vector(dtype=torch.float32),
        "age": _empty_capacity_vector(dtype=torch.int32),
        "policy_energy_observation": energy.to(torch.float64),
        "rest_streak": _empty_capacity_vector(dtype=torch.int32),
        "survival_steps": survival_steps,
        "last_action": _empty_capacity_vector(dtype=torch.int32, fill=-1),
        "last_last_action": _empty_capacity_vector(dtype=torch.int32, fill=-1),
        "action_switches": _empty_capacity_vector(dtype=torch.int32),
        "action_oscillations": _empty_capacity_vector(dtype=torch.int32),
        "episode0_switches": _empty_capacity_vector(dtype=torch.int32),
        "episode0_oscillations": _empty_capacity_vector(dtype=torch.int32),
        "gru_upkeep_cost": _empty_capacity_vector(dtype=torch.float32),
        "cost_charge_scalar": _empty_capacity_vector(dtype=torch.float32),
        "cost_charge_maturation_scalar": _empty_capacity_vector(dtype=torch.float32),
        "patch_food_amount": torch.zeros(shape, dtype=torch.float32),
        "patch_fertility": torch.ones(shape, dtype=torch.float32),
        "patch_cover": torch.zeros(shape, dtype=torch.float32),
        "patch_shelter": torch.zeros(shape, dtype=torch.float32),
        "patch_scent": torch.zeros(shape, dtype=torch.float32),
        "patch_waste": torch.zeros(shape, dtype=torch.float32),
        "patch_density": torch.zeros(shape, dtype=torch.float32),
        "patch_scent_emit_pending": torch.zeros(shape, dtype=torch.float32),
        "patch_waste_emit_pending": torch.zeros(shape, dtype=torch.float32),
        "patch_movement_cost": torch.ones(shape, dtype=torch.float32),
        "mlp_pack": mlp_pack,
        "drives": torch.zeros((AGENT_CAPACITY, 4), dtype=torch.float32),
        "drive_inertia": _empty_capacity_vector(dtype=torch.float32, fill=0.5),
        "drive_noise": _empty_capacity_vector(dtype=torch.float32),
        "prev_drive_delta": torch.zeros((AGENT_CAPACITY, 4), dtype=torch.float32),
        "drive_volatility": _empty_capacity_vector(dtype=torch.float32),
        "drive_jitter": _empty_capacity_vector(dtype=torch.float32),
        "episode0_drive_jitter_sum": _empty_capacity_vector(dtype=torch.float32),
        "last_energy_delta": _empty_capacity_vector(dtype=torch.float32),
        "last_health_delta": _empty_capacity_vector(dtype=torch.float32),
        "last_injury_delta": _empty_capacity_vector(dtype=torch.float32),
        "selfharm_events": _empty_capacity_vector(dtype=torch.int32),
        "episode0_selfharm_events": _empty_capacity_vector(dtype=torch.int32),
        "drive_noise_samples_by_tick_slot": torch.zeros(
            (GENERATION_STEPS, AGENT_CAPACITY, 4), dtype=torch.float32
        ),
        "drive_noise_samples_valid_by_tick_slot": alive_mask.reshape(
            1, AGENT_CAPACITY
        ).clone(),
        "attack_bias": _empty_capacity_vector(dtype=torch.float32),
        "action_temperature": _empty_capacity_vector(dtype=torch.float32, fill=1.0),
        "drive_logit_gain": _empty_capacity_vector(dtype=torch.float32),
        "slot_tokens": slot_tokens,
        "action_cache_actions": torch.full(
            (AGENT_CAPACITY, CACHE_WIDTH), -1, dtype=torch.int32
        ),
        "action_cache_token": _empty_capacity_vector(dtype=torch.int64, fill=-1),
        "action_cache_cursor": _empty_capacity_vector(dtype=torch.int32, fill=-1),
        "action_cache_next": _empty_capacity_vector(dtype=torch.int32, fill=-1),
        "pending_free_agent_ids": _empty_capacity_vector(dtype=torch.int64, fill=-1),
        "pending_free_slots": _empty_capacity_vector(dtype=torch.int64, fill=-1),
        "pending_free_count": torch.zeros((1,), dtype=torch.int64),
        "pcg64_state_words": state_words,
        "pcg64_buffer_words": buffer_words,
        "lineage_founder_id": torch.cat(
            (
                torch.tensor(ids, dtype=torch.int32),
                torch.full(
                    (AGENT_CAPACITY - PANEL_AGENT_COUNT,), -1, dtype=torch.int32
                ),
            )
        ),
        "lineage_depth": _empty_capacity_vector(dtype=torch.int32, fill=-1),
    }
    tensors["lineage_depth"][:PANEL_AGENT_COUNT] = 0
    for name in EVENT_TAPE_COLUMN_NAMES:
        tensors[name] = torch.full(
            (EVENT_CAPACITY,), -1, dtype=EVENT_COLUMN_DTYPES[name]
        )
    tensors["event_tape_active_count"] = torch.zeros((1,), dtype=torch.int64)
    tensors["event_tape_status"] = torch.full((1,), -1, dtype=torch.int32)
    return _clone_tensors(tensors), observations


def _formal_inputs(
    *,
    case: Mapping[str, Any],
    source_seed: int,
    owner_tensors: Mapping[str, torch.Tensor],
    packed_planes: Mapping[str, torch.Tensor],
    observations: torch.Tensor,
) -> tuple[PhaseFV2FormalOracleInput, ...]:
    outputs: list[PhaseFV2FormalOracleInput] = []
    for slot, agent in enumerate(case["agents"]):
        record = agent["record"]
        policy = _split_policy_pack(owner_tensors["mlp_pack"][slot])
        outputs.append(
            PhaseFV2FormalOracleInput(
                schema_version=FORMAL_INPUT_SCHEMA_VERSION,
                case_id=str(case["case_id"]),
                case_sha256=str(case["case_sha256"]),
                fixture_sha256=str(case["fixture_sha256"]),
                profile_id=str(case["profile_id"]),
                source_seed=source_seed,
                agent_slot=slot,
                agent_id=int(record["agent_id"]),
                agent_slot_token=int(record["agent_slot_token"]),
                planes={
                    name: tensor[slot : slot + 1].detach().clone().contiguous()
                    for name, tensor in packed_planes.items()
                },
                modular_observations=observations[slot : slot + 1].clone(),
                fatigue=owner_tensors["drives"][slot : slot + 1, 2].clone(),
                stage_indices=torch.where(
                    owner_tensors["survival_steps"][slot : slot + 1]
                    >= DEVELOPMENT_LATE_START_STEP,
                    torch.full((1,), 2, dtype=torch.int64),
                    torch.where(
                        owner_tensors["survival_steps"][slot : slot + 1]
                        >= DEVELOPMENT_MATURE_START_STEP,
                        torch.ones((1,), dtype=torch.int64),
                        torch.zeros((1,), dtype=torch.int64),
                    ),
                ).contiguous(),
                policy_observations=observations[slot : slot + 1].clone(),
                w1=policy["w1"],
                b1=policy["b1"],
                w2=policy["w2"],
                b2=policy["b2"],
                w3=policy["w3"],
                b3=policy["b3"],
                attack_bias=owner_tensors["attack_bias"][slot : slot + 1].clone(),
                action_temperature=owner_tensors["action_temperature"][
                    slot : slot + 1
                ].clone(),
                validity_mask=torch.ones((1, ACTION_COUNT), dtype=torch.bool),
                authority_mask=torch.ones((1, ACTION_COUNT), dtype=torch.bool),
                rng_stream_indices={"modular": 0},
            )
        )
    return tuple(outputs)


def _validate_generation_tensors(
    tensors: Mapping[str, torch.Tensor], *, profile_id: str
) -> None:
    expected_names = set(GENERATION_TENSOR_NAMES)
    modular_names = {
        name for name in tensors if name.startswith(MODULAR_STORAGE_PROJECTION_PREFIX)
    }
    actual_owner_names = set(tensors) - modular_names
    if actual_owner_names != expected_names:
        raise PhaseFV2SourceError(
            "generation tensor field set mismatch "
            f"missing={sorted(expected_names - actual_owner_names)!r};"
            f"extra={sorted(actual_owner_names - expected_names)!r}"
        )
    for name, tensor in tensors.items():
        _tensor_manifest(tensor)
        if not tensor.is_contiguous():
            raise PhaseFV2SourceError(f"generation tensor is not contiguous: {name}")
    if tuple(tensors["alive_mask"].shape) != (AGENT_CAPACITY,):
        raise PhaseFV2SourceError("alive_mask capacity mismatch")
    entry = prepare_modular_generation_entry(
        tensors,
        enabled=True,
        agent_capacity=AGENT_CAPACITY,
        profile_id=profile_id,
        policy_family="fixed_mlp",
        action_count=ACTION_COUNT,
        observation_width=OBSERVATION_WIDTH,
        cache_width=CACHE_WIDTH,
        policy_stride=POLICY_STRIDE,
        development_mature_start_step=DEVELOPMENT_MATURE_START_STEP,
        development_late_start_step=DEVELOPMENT_LATE_START_STEP,
        max_births=0,
        turnover_enabled=False,
        device="cpu",
        allow_cpu_validation=True,
    )
    if entry is None or not entry.receipt.cpu_validation_mode:
        raise PhaseFV2SourceError("public modular generation entry did not validate")


def materialize_phase_f_source_v2(
    fixture_case: Mapping[str, Any], *, source_seed: int
) -> PhaseFV2SourceState:
    """Build one exact 16-agent, 8x8 CPU source without running a tick."""

    seed = _exact_source_seed(source_seed)
    try:
        case = validate_phase_f_fixture_case_v2(fixture_case)
    except Exception as exc:
        raise PhaseFV2SourceError("fixture case is not current and exact") from exc
    construction_seed = _case_construction_seed(case)
    if seed != construction_seed:
        raise PhaseFV2SourceError(
            "source_seed must equal the case fixture construction_seed"
        )
    records = [agent["record"] for agent in case["agents"]]
    packed = pack_modular_storage(
        [*records, *([None] * (AGENT_CAPACITY - PANEL_AGENT_COUNT))],
        agent_capacity=AGENT_CAPACITY,
        profile_id=str(case["profile_id"]),
    )
    projection = modular_storage_projection_payload(packed)
    owner_tensors, observations = _owner_tensors(case=case, source_seed=seed)
    generation_tensors = {
        **owner_tensors,
        **_clone_tensors(projection["tensors"]),
    }
    owner_metadata = {
        "schema_version": SCHEMA_VERSION,
        "case_id": str(case["case_id"]),
        "case_sha256": str(case["case_sha256"]),
        "fixture_sha256": str(case["fixture_sha256"]),
        "matrix_sha256": str(case["matrix_sha256"]),
        "profile_id": str(case["profile_id"]),
        "role_action": str(case["role_action"]),
        "source_seed": seed,
        "population_size": PANEL_AGENT_COUNT,
        "agent_capacity": AGENT_CAPACITY,
        "grid_height": GRID_HEIGHT,
        "grid_width": GRID_WIDTH,
        "generation_steps": GENERATION_STEPS,
        "development_mature_start_step": DEVELOPMENT_MATURE_START_STEP,
        "development_late_start_step": DEVELOPMENT_LATE_START_STEP,
        "observation_width": OBSERVATION_WIDTH,
        "action_count": ACTION_COUNT,
        "hidden_size": HIDDEN_SIZE,
        "policy_family": "fixed_mlp",
        "modular_rng_draw_count": MODULAR_RNG_DRAW_COUNT,
        "births_enabled": False,
        "deaths_enabled": False,
        "turnover_enabled": False,
        "simulation_executed": False,
        "packed_storage_sha256": packed.packed_sha256,
        MODULAR_STORAGE_METADATA_KEY: deepcopy(
            projection["metadata"][MODULAR_STORAGE_METADATA_KEY]
        ),
    }
    overlay = validate_modular_cpu_boundary_projection(
        owner_metadata, generation_tensors, agent_capacity=AGENT_CAPACITY
    )
    if overlay is None:
        raise PhaseFV2SourceError("public modular CPU projection disappeared")
    _validate_generation_tensors(generation_tensors, profile_id=str(case["profile_id"]))
    formal_inputs = _formal_inputs(
        case=case,
        source_seed=seed,
        owner_tensors=owner_tensors,
        packed_planes=packed.tensors,
        observations=observations,
    )
    config = _generation_config(seed)
    header = {
        "schema_version": SCHEMA_VERSION,
        "case_id": str(case["case_id"]),
        "case_sha256": str(case["case_sha256"]),
        "fixture_sha256": str(case["fixture_sha256"]),
        "matrix_sha256": str(case["matrix_sha256"]),
        "profile_id": str(case["profile_id"]),
        "role_action": str(case["role_action"]),
        "source_seed": seed,
    }
    return PhaseFV2SourceState(
        **header,
        owner_metadata=owner_metadata,
        generation_config=config,
        generation_tensors=_clone_tensors(generation_tensors),
        formal_oracle_inputs=formal_inputs,
        source_sha256=_source_digest(
            header=header,
            owner_metadata=owner_metadata,
            config=config,
            generation_tensors=generation_tensors,
            formal_inputs=formal_inputs,
        ),
    )


def _equal_tensors(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> bool:
    if set(left) != set(right):
        return False
    return all(
        left[name].dtype == right[name].dtype
        and tuple(left[name].shape) == tuple(right[name].shape)
        and torch.equal(left[name], right[name])
        for name in left
    )


def _equal_oracle_inputs(
    left: PhaseFV2FormalOracleInput, right: PhaseFV2FormalOracleInput
) -> bool:
    tensor_fields = {
        field.name
        for field in fields(PhaseFV2FormalOracleInput)
        if isinstance(getattr(left, field.name), torch.Tensor)
    }
    scalar_fields = {
        field.name
        for field in fields(PhaseFV2FormalOracleInput)
        if field.name not in tensor_fields | {"planes"}
    }
    return (
        all(getattr(left, name) == getattr(right, name) for name in scalar_fields)
        and _equal_tensors(left.planes, right.planes)
        and all(
            torch.equal(getattr(left, name), getattr(right, name))
            for name in tensor_fields
        )
    )


def validate_phase_f_source_v2(
    value: Any,
    *,
    fixture_case: Mapping[str, Any],
    source_seed: int,
) -> PhaseFV2SourceState:
    """Fail closed unless ``value`` is the exact current source materialization."""

    if not isinstance(value, PhaseFV2SourceState):
        raise PhaseFV2SourceError("source state has the wrong type")
    expected = materialize_phase_f_source_v2(
        fixture_case, source_seed=_exact_source_seed(source_seed)
    )
    scalar_fields = (
        "schema_version",
        "case_id",
        "case_sha256",
        "fixture_sha256",
        "matrix_sha256",
        "profile_id",
        "role_action",
        "source_seed",
        "owner_metadata",
        "generation_config",
        "source_sha256",
    )
    if any(getattr(value, name) != getattr(expected, name) for name in scalar_fields):
        raise PhaseFV2SourceError("source binding or digest mismatch")
    if not _equal_tensors(value.generation_tensors, expected.generation_tensors):
        raise PhaseFV2SourceError("source generation tensors are not exact")
    if len(value.formal_oracle_inputs) != len(expected.formal_oracle_inputs) or any(
        not _equal_oracle_inputs(left, right)
        for left, right in zip(
            value.formal_oracle_inputs, expected.formal_oracle_inputs, strict=True
        )
    ):
        raise PhaseFV2SourceError("source formal-oracle inputs are not exact")
    return expected


__all__ = [
    "ACTION_COUNT",
    "AGENT_CAPACITY",
    "DEVELOPMENT_LATE_START_STEP",
    "DEVELOPMENT_MATURE_START_STEP",
    "FORMAL_INPUT_SCHEMA_VERSION",
    "GENERATION_STEPS",
    "GENERATION_TENSOR_NAMES",
    "GRID_HEIGHT",
    "GRID_WIDTH",
    "HIDDEN_SIZE",
    "OBSERVATION_WIDTH",
    "PANEL_AGENT_COUNT",
    "SCHEMA_VERSION",
    "PhaseFV2FormalOracleInput",
    "PhaseFV2SourceError",
    "PhaseFV2SourceState",
    "materialize_phase_f_source_v2",
    "validate_phase_f_source_v2",
]
