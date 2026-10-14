"""C2 resident state extensions for Patch/Jungle/WDL/EED mechanics."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Final

import torch

from .contracts import TensorWorldContractError
from .state import TensorWorldStateBundleV1


C2_STATE_SCHEMA_VERSION_V1: Final[str] = "tensor_world_c2_state_schema_v1"
C2_OBSERVATION_ABI_V1: Final[str] = "patch_wdl_radius1_observation_9x9_plus2_v1"
C2_OBSERVATION_WIDTH_V1: Final[int] = 83
C2_POLICY_ABI_V1: Final[str] = "fixed_mlp_83x64x64x7_fp32_pack_v1"
C2_POLICY_PACKED_WIDTH_V1: Final[int] = 9_991


@dataclass(slots=True)
class PatchTensorStateV1:
    fertility: torch.Tensor
    scent_emit_pending: torch.Tensor
    waste_emit_pending: torch.Tensor
    recovery_activity: torch.Tensor
    visibility: torch.Tensor
    agent_grid_i32: torch.Tensor
    food_grid_i32: torch.Tensor
    multiplier_scratch: torch.Tensor
    waste_penalty_scratch: torch.Tensor
    density_penalty_scratch: torch.Tensor
    sorted_multiplier_scratch: torch.Tensor
    recovery_boost_scratch: torch.Tensor
    sorted_recovery_boost_scratch: torch.Tensor


@dataclass(slots=True)
class WdlTensorStateV1:
    cue: torch.Tensor
    timer: torch.Tensor
    active: torch.Tensor
    intensity: torch.Tensor
    age: torch.Tensor
    event_id: torch.Tensor
    prior_cue: torch.Tensor
    base_seed: torch.Tensor
    next_absolute_step: torch.Tensor


@dataclass(slots=True)
class EedTensorStateV1:
    resource_a_amount: torch.Tensor
    resource_b_amount: torch.Tensor
    resource_a_capacity: torch.Tensor
    resource_b_capacity: torch.Tensor
    resource_a_renewal: torch.Tensor
    resource_b_renewal: torch.Tensor
    resource_a_pending_impact: torch.Tensor
    resource_b_pending_impact: torch.Tensor
    resource_a_impact_load: torch.Tensor
    resource_b_impact_load: torch.Tensor
    resource_class: torch.Tensor
    active_y: torch.Tensor
    active_x: torch.Tensor
    active_class: torch.Tensor
    handling_progress: torch.Tensor
    handling_axis: torch.Tensor


@dataclass(slots=True)
class C2PolicyTensorStateV1:
    packed_weights: torch.Tensor


@dataclass(slots=True)
class C2RngTensorStateV1:
    jungle_uniforms: torch.Tensor


@dataclass(slots=True)
class AgentEffectBufferV1:
    pre_policy_health_delta: torch.Tensor
    post_action_health_delta: torch.Tensor
    hazard_flags: torch.Tensor


@dataclass(slots=True)
class WorldEventBufferV1:
    scent_emission: torch.Tensor
    waste_emission: torch.Tensor
    gather_amount: torch.Tensor
    gather_resource_class: torch.Tensor


@dataclass(slots=True)
class C2TransactionTensorStateV1:
    observations: torch.Tensor
    logits: torch.Tensor


@dataclass(slots=True)
class C2HotMetricStateV1:
    patch_stats: torch.Tensor
    recovery_stats: torch.Tensor
    jungle_integers: torch.Tensor
    jungle_doubles: torch.Tensor
    wdl_integers: torch.Tensor
    wdl_delay_sum: torch.Tensor
    wdl_action_integers: torch.Tensor
    eed_receipt: torch.Tensor


@dataclass(slots=True)
class TensorWorldStateBundleV2:
    core: TensorWorldStateBundleV1
    patch: PatchTensorStateV1
    wdl: WdlTensorStateV1
    eed: EedTensorStateV1
    policy: C2PolicyTensorStateV1
    rng: C2RngTensorStateV1
    effects: AgentEffectBufferV1
    world_events: WorldEventBufferV1
    transactions: C2TransactionTensorStateV1
    hot_metrics: C2HotMetricStateV1
    schema_version: str = C2_STATE_SCHEMA_VERSION_V1

    @property
    def device(self) -> torch.device:
        return self.core.device

    @property
    def batch(self) -> int:
        return self.core.batch

    @property
    def capacity(self) -> int:
        return self.core.capacity

    @property
    def grid_size(self) -> int:
        return self.core.grid_size

    @property
    def tick(self) -> int:
        return self.core.tick

    def validate(self) -> None:
        if self.schema_version != C2_STATE_SCHEMA_VERSION_V1:
            raise TensorWorldContractError("C2 state schema version is unsupported")
        self.core.validate()
        batch, capacity, width = self.batch, self.capacity, self.grid_size
        grid = (batch, width, width)
        slots = (batch, capacity)
        expected: tuple[tuple[str, torch.Tensor, tuple[int, ...], torch.dtype], ...] = (
            ("patch.fertility", self.patch.fertility, grid, torch.float32),
            ("patch.scent_emit_pending", self.patch.scent_emit_pending, grid, torch.float32),
            ("patch.waste_emit_pending", self.patch.waste_emit_pending, grid, torch.float32),
            ("patch.recovery_activity", self.patch.recovery_activity, grid, torch.float32),
            ("patch.visibility", self.patch.visibility, grid, torch.float32),
            ("patch.agent_grid_i32", self.patch.agent_grid_i32, grid, torch.int32),
            ("patch.food_grid_i32", self.patch.food_grid_i32, grid, torch.int32),
            ("patch.multiplier_scratch", self.patch.multiplier_scratch, grid, torch.float32),
            ("patch.waste_penalty_scratch", self.patch.waste_penalty_scratch, grid, torch.float32),
            ("patch.density_penalty_scratch", self.patch.density_penalty_scratch, grid, torch.float32),
            ("patch.sorted_multiplier_scratch", self.patch.sorted_multiplier_scratch, grid, torch.float32),
            ("patch.recovery_boost_scratch", self.patch.recovery_boost_scratch, grid, torch.float32),
            ("patch.sorted_recovery_boost_scratch", self.patch.sorted_recovery_boost_scratch, grid, torch.float32),
            ("wdl.cue", self.wdl.cue, grid, torch.float32),
            ("wdl.timer", self.wdl.timer, grid, torch.int32),
            ("wdl.active", self.wdl.active, grid, torch.float32),
            ("wdl.intensity", self.wdl.intensity, grid, torch.float32),
            ("wdl.age", self.wdl.age, grid, torch.int32),
            ("wdl.event_id", self.wdl.event_id, grid, torch.int64),
            ("wdl.prior_cue", self.wdl.prior_cue, grid, torch.int8),
            ("wdl.base_seed", self.wdl.base_seed, (batch,), torch.int64),
            ("wdl.next_absolute_step", self.wdl.next_absolute_step, (batch,), torch.int64),
            ("eed.resource_a_amount", self.eed.resource_a_amount, grid, torch.float32),
            ("eed.resource_b_amount", self.eed.resource_b_amount, grid, torch.float32),
            ("eed.resource_a_capacity", self.eed.resource_a_capacity, grid, torch.float32),
            ("eed.resource_b_capacity", self.eed.resource_b_capacity, grid, torch.float32),
            ("eed.resource_a_renewal", self.eed.resource_a_renewal, grid, torch.float32),
            ("eed.resource_b_renewal", self.eed.resource_b_renewal, grid, torch.float32),
            ("eed.resource_a_pending_impact", self.eed.resource_a_pending_impact, grid, torch.float32),
            ("eed.resource_b_pending_impact", self.eed.resource_b_pending_impact, grid, torch.float32),
            ("eed.resource_a_impact_load", self.eed.resource_a_impact_load, grid, torch.float32),
            ("eed.resource_b_impact_load", self.eed.resource_b_impact_load, grid, torch.float32),
            ("eed.resource_class", self.eed.resource_class, grid, torch.float32),
            ("eed.active_y", self.eed.active_y, slots, torch.int32),
            ("eed.active_x", self.eed.active_x, slots, torch.int32),
            ("eed.active_class", self.eed.active_class, slots, torch.int32),
            ("eed.handling_progress", self.eed.handling_progress, slots, torch.float32),
            ("eed.handling_axis", self.eed.handling_axis, slots, torch.float32),
            ("policy.packed_weights", self.policy.packed_weights, (batch, capacity, C2_POLICY_PACKED_WIDTH_V1), torch.float32),
            ("rng.jungle_uniforms", self.rng.jungle_uniforms, slots, torch.float64),
            ("effects.pre_policy_health_delta", self.effects.pre_policy_health_delta, slots, torch.float32),
            ("effects.post_action_health_delta", self.effects.post_action_health_delta, slots, torch.float32),
            ("effects.hazard_flags", self.effects.hazard_flags, slots, torch.uint8),
            ("world_events.scent_emission", self.world_events.scent_emission, grid, torch.float32),
            ("world_events.waste_emission", self.world_events.waste_emission, grid, torch.float32),
            ("world_events.gather_amount", self.world_events.gather_amount, slots, torch.float32),
            ("world_events.gather_resource_class", self.world_events.gather_resource_class, slots, torch.int32),
            ("transactions.observations", self.transactions.observations, (batch, capacity, C2_OBSERVATION_WIDTH_V1), torch.float32),
            ("transactions.logits", self.transactions.logits, (batch, capacity, 7), torch.float32),
            ("hot_metrics.patch_stats", self.hot_metrics.patch_stats, (batch, 25), torch.float64),
            ("hot_metrics.recovery_stats", self.hot_metrics.recovery_stats, (batch, 7), torch.float64),
            ("hot_metrics.jungle_integers", self.hot_metrics.jungle_integers, (batch, 8), torch.int64),
            ("hot_metrics.jungle_doubles", self.hot_metrics.jungle_doubles, (batch, 6), torch.float64),
            ("hot_metrics.wdl_integers", self.hot_metrics.wdl_integers, (batch, 7), torch.int64),
            ("hot_metrics.wdl_delay_sum", self.hot_metrics.wdl_delay_sum, (batch,), torch.float64),
            ("hot_metrics.wdl_action_integers", self.hot_metrics.wdl_action_integers, (batch, 5), torch.int64),
            ("hot_metrics.eed_receipt", self.hot_metrics.eed_receipt, (batch, 4), torch.float64),
        )
        devices = {self.core.device}
        for name, tensor, shape, dtype in expected:
            if not isinstance(tensor, torch.Tensor):
                raise TensorWorldContractError(f"{name} is not a tensor")
            if tuple(tensor.shape) != shape or tensor.dtype != dtype:
                raise TensorWorldContractError(
                    f"{name} requires shape={shape!r} dtype={dtype}, got "
                    f"shape={tuple(tensor.shape)!r} dtype={tensor.dtype}"
                )
            if not tensor.is_contiguous():
                raise TensorWorldContractError(f"{name} must be contiguous")
            devices.add(tensor.device)
        if len(devices) != 1:
            raise TensorWorldContractError("all C2 resident state must share one device")


def c2_state_tensor_names_v1() -> tuple[str, ...]:
    names: list[str] = []
    for owner in (
        PatchTensorStateV1,
        WdlTensorStateV1,
        EedTensorStateV1,
        C2PolicyTensorStateV1,
        C2RngTensorStateV1,
        AgentEffectBufferV1,
        WorldEventBufferV1,
        C2TransactionTensorStateV1,
        C2HotMetricStateV1,
    ):
        names.extend(f"{owner.__name__}.{field.name}" for field in fields(owner))
    return tuple(names)
