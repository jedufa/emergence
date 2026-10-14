"""Versioned opaque state ownership for the Tensor World backend."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Final

import torch

from .contracts import TensorWorldContractError


STATE_SCHEMA_VERSION_V1: Final[str] = "tensor_world_state_schema_v1"
PATCH_OBSERVATION_ABI_V1: Final[str] = "patch_radius1_observation_8x9_plus2_v1"
PATCH_POLICY_ABI_V1: Final[str] = "fixed_mlp_74x64x64x7_fp32_pack_v1"
PATCH_OBSERVATION_WIDTH_V1: Final[int] = 74
PATCH_POLICY_PACKED_WIDTH_V1: Final[int] = 9_415
PATCH_ACTION_COUNT_V1: Final[int] = 7


@dataclass(slots=True)
class WorldTensorStateV1:
    food_amount: torch.Tensor
    cover: torch.Tensor
    shelter: torch.Tensor
    scent: torch.Tensor
    waste: torch.Tensor
    density: torch.Tensor
    movement_cost: torch.Tensor


@dataclass(slots=True)
class AgentTensorStateV1:
    alive: torch.Tensor
    agent_id: torch.Tensor
    pos_y: torch.Tensor
    pos_x: torch.Tensor
    energy: torch.Tensor
    health: torch.Tensor
    age: torch.Tensor
    survival_steps: torch.Tensor
    energy_collected: torch.Tensor


@dataclass(slots=True)
class OccupancyTensorStateV1:
    agent_id_grid: torch.Tensor
    slot_grid: torch.Tensor


@dataclass(slots=True)
class PolicyTensorStateV1:
    packed_weights: torch.Tensor
    drives: torch.Tensor
    attack_bias: torch.Tensor
    action_temperature: torch.Tensor
    drive_logit_gain: torch.Tensor


@dataclass(slots=True)
class IdentityTensorStateV1:
    domain_world_ids: torch.Tensor
    slot_ids: torch.Tensor


@dataclass(slots=True)
class RngTensorStateV1:
    action_uniforms: torch.Tensor
    action_cursor: torch.Tensor


@dataclass(slots=True)
class TransactionTensorStateV1:
    actions: torch.Tensor
    moved: torch.Tensor
    gathered: torch.Tensor
    food_gained: torch.Tensor
    observations: torch.Tensor
    logits: torch.Tensor


@dataclass(slots=True)
class HotEventAccumulatorStateV1:
    moves_attempted: torch.Tensor
    moves_committed: torch.Tensor
    gathers_committed: torch.Tensor


@dataclass(slots=True)
class TensorWorldStateBundleV1:
    """The C1 root resident state bundle.

    Dynamic population and generalized checkpoint/RNG ownership are reserved
    partitions, deliberately unsupported by the C1 capability declaration.
    """

    world: WorldTensorStateV1
    agents: AgentTensorStateV1
    occupancy: OccupancyTensorStateV1
    policy: PolicyTensorStateV1
    identity: IdentityTensorStateV1
    rng: RngTensorStateV1
    transactions: TransactionTensorStateV1
    hot_events: HotEventAccumulatorStateV1
    tick: int = 0
    schema_version: str = STATE_SCHEMA_VERSION_V1

    @property
    def device(self) -> torch.device:
        return self.world.food_amount.device

    @property
    def batch(self) -> int:
        return int(self.agents.alive.shape[0])

    @property
    def capacity(self) -> int:
        return int(self.agents.alive.shape[1])

    @property
    def grid_size(self) -> int:
        return int(self.world.food_amount.shape[1])

    def validate(self) -> None:
        if self.schema_version != STATE_SCHEMA_VERSION_V1:
            raise TensorWorldContractError("state schema version is unsupported")
        if type(self.tick) is not int or self.tick < 0:
            raise TensorWorldContractError("state tick must be nonnegative")
        if self.agents.alive.ndim != 2:
            raise TensorWorldContractError("agents.alive must have rank 2")
        if self.world.food_amount.ndim != 3:
            raise TensorWorldContractError("world.food_amount must have rank 3")
        if self.world.food_amount.shape[1] != self.world.food_amount.shape[2]:
            raise TensorWorldContractError("C1 admits square world grids only")
        batch, capacity = self.agents.alive.shape
        if batch <= 0 or capacity <= 0 or self.grid_size <= 0:
            raise TensorWorldContractError("state dimensions must be positive")
        grid_shape = (batch, self.grid_size, self.grid_size)
        slot_shape = (batch, capacity)
        expected: tuple[tuple[str, torch.Tensor, tuple[int, ...], torch.dtype], ...] = (
            ("world.food_amount", self.world.food_amount, grid_shape, torch.float32),
            ("world.cover", self.world.cover, grid_shape, torch.float32),
            ("world.shelter", self.world.shelter, grid_shape, torch.float32),
            ("world.scent", self.world.scent, grid_shape, torch.float32),
            ("world.waste", self.world.waste, grid_shape, torch.float32),
            ("world.density", self.world.density, grid_shape, torch.float32),
            (
                "world.movement_cost",
                self.world.movement_cost,
                grid_shape,
                torch.float32,
            ),
            ("agents.alive", self.agents.alive, slot_shape, torch.bool),
            ("agents.agent_id", self.agents.agent_id, slot_shape, torch.int64),
            ("agents.pos_y", self.agents.pos_y, slot_shape, torch.int64),
            ("agents.pos_x", self.agents.pos_x, slot_shape, torch.int64),
            ("agents.energy", self.agents.energy, slot_shape, torch.float32),
            ("agents.health", self.agents.health, slot_shape, torch.float32),
            ("agents.age", self.agents.age, slot_shape, torch.int64),
            (
                "agents.survival_steps",
                self.agents.survival_steps,
                slot_shape,
                torch.int64,
            ),
            (
                "agents.energy_collected",
                self.agents.energy_collected,
                slot_shape,
                torch.float64,
            ),
            (
                "occupancy.agent_id_grid",
                self.occupancy.agent_id_grid,
                grid_shape,
                torch.int64,
            ),
            ("occupancy.slot_grid", self.occupancy.slot_grid, grid_shape, torch.int64),
            (
                "policy.packed_weights",
                self.policy.packed_weights,
                (batch, capacity, PATCH_POLICY_PACKED_WIDTH_V1),
                torch.float32,
            ),
            ("policy.drives", self.policy.drives, (batch, capacity, 4), torch.float32),
            ("policy.attack_bias", self.policy.attack_bias, slot_shape, torch.float32),
            (
                "policy.action_temperature",
                self.policy.action_temperature,
                slot_shape,
                torch.float32,
            ),
            (
                "policy.drive_logit_gain",
                self.policy.drive_logit_gain,
                slot_shape,
                torch.float32,
            ),
            (
                "identity.domain_world_ids",
                self.identity.domain_world_ids,
                (batch,),
                torch.int64,
            ),
            ("identity.slot_ids", self.identity.slot_ids, slot_shape, torch.int64),
            (
                "rng.action_uniforms",
                self.rng.action_uniforms,
                (batch, capacity, PATCH_ACTION_COUNT_V1),
                torch.float64,
            ),
            ("rng.action_cursor", self.rng.action_cursor, (batch,), torch.int64),
            (
                "transactions.actions",
                self.transactions.actions,
                slot_shape,
                torch.int64,
            ),
            ("transactions.moved", self.transactions.moved, slot_shape, torch.bool),
            (
                "transactions.gathered",
                self.transactions.gathered,
                slot_shape,
                torch.bool,
            ),
            (
                "transactions.food_gained",
                self.transactions.food_gained,
                slot_shape,
                torch.float32,
            ),
            (
                "transactions.observations",
                self.transactions.observations,
                (batch, capacity, PATCH_OBSERVATION_WIDTH_V1),
                torch.float32,
            ),
            (
                "transactions.logits",
                self.transactions.logits,
                (batch, capacity, PATCH_ACTION_COUNT_V1),
                torch.float32,
            ),
            (
                "hot_events.moves_attempted",
                self.hot_events.moves_attempted,
                (batch,),
                torch.int64,
            ),
            (
                "hot_events.moves_committed",
                self.hot_events.moves_committed,
                (batch,),
                torch.int64,
            ),
            (
                "hot_events.gathers_committed",
                self.hot_events.gathers_committed,
                (batch,),
                torch.int64,
            ),
        )
        devices: set[torch.device] = set()
        for name, tensor, shape, dtype in expected:
            if not isinstance(tensor, torch.Tensor):
                raise TensorWorldContractError(f"{name} is not a tensor")
            if tuple(tensor.shape) != tuple(shape) or tensor.dtype != dtype:
                raise TensorWorldContractError(
                    f"{name} requires shape={shape!r} dtype={dtype}, "
                    f"got shape={tuple(tensor.shape)!r} dtype={tensor.dtype}"
                )
            if not tensor.is_contiguous():
                raise TensorWorldContractError(f"{name} must be contiguous")
            devices.add(tensor.device)
        if len(devices) != 1:
            raise TensorWorldContractError("all resident state must share one device")


def state_tensor_names_v1() -> tuple[str, ...]:
    """Return the stable C1 ownership surface for schema tests/manifests."""

    names: list[str] = []
    for owner in (
        WorldTensorStateV1,
        AgentTensorStateV1,
        OccupancyTensorStateV1,
        PolicyTensorStateV1,
        IdentityTensorStateV1,
        RngTensorStateV1,
        TransactionTensorStateV1,
        HotEventAccumulatorStateV1,
    ):
        names.extend(f"{owner.__name__}.{field.name}" for field in fields(owner))
    return tuple(names)
