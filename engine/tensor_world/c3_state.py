"""Resident C3 agent/adaptation state layered on the passing C2 bundle.

This module only establishes ownership and shape contracts.  C4 owns the
population transaction that changes slot occupancy, and C5 owns persistent
RNG/checkpoint/replay semantics.  No Python agent objects are introduced.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Final

import torch

from .c2_state import C2_OBSERVATION_WIDTH_V1, TensorWorldStateBundleV2
from .contracts import TensorWorldContractError
from .state import PATCH_ACTION_COUNT_V1


C3_STATE_SCHEMA_VERSION_V1: Final[str] = "tensor_world_c3_state_schema_v1"
C3_DRIVE_WIDTH_V1: Final[int] = 4
C3_CHEMISTRY_WIDTH_V1: Final[int] = 6
C3_ACTION_COUNT_V1: Final[int] = PATCH_ACTION_COUNT_V1
C3_OBSERVATION_WIDTH_V1: Final[int] = C2_OBSERVATION_WIDTH_V1
# The source fixed-MLP policy service owns a two-entry per-slot cache.  Keep
# the same width while C3 uses the inherited C2 policy provider so the native
# projection remains source-shaped.
C3_POLICY_CACHE_WIDTH_V1: Final[int] = 2
C3_CHEMISTRY_NOISE_WIDTH_V1: Final[int] = 2


@dataclass(slots=True)
class C3AgentRuntimeTensorStateV1:
    injury: torch.Tensor
    age: torch.Tensor
    survival_steps: torch.Tensor
    energy_collected: torch.Tensor
    energy_collected_exact: torch.Tensor
    rest_streak: torch.Tensor
    policy_energy_observation: torch.Tensor
    last_action: torch.Tensor
    last_last_action: torch.Tensor
    action_switches: torch.Tensor
    action_oscillations: torch.Tensor
    episode0_switches: torch.Tensor
    episode0_oscillations: torch.Tensor
    attacks: torch.Tensor
    gru_upkeep_cost: torch.Tensor
    cost_charge_scalar: torch.Tensor
    cost_charge_maturation_scalar: torch.Tensor
    drives: torch.Tensor
    drive_inertia: torch.Tensor
    drive_noise: torch.Tensor
    prev_drive_delta: torch.Tensor
    drive_volatility: torch.Tensor
    drive_jitter: torch.Tensor
    episode0_drive_jitter_sum: torch.Tensor
    last_energy_delta: torch.Tensor
    last_health_delta: torch.Tensor
    last_injury_delta: torch.Tensor
    selfharm_events: torch.Tensor
    episode0_selfharm_events: torch.Tensor
    repro_threshold: torch.Tensor


@dataclass(slots=True)
class C3GenomeExecutionTensorStateV1:
    """Resident compiled genome values consumed by the C3 tick.

    Raw Genome V2 allele/product planes are intentionally not here: mutation,
    recombination, and their transaction receipts belong to C4.  C4 compiles
    those planes into this fixed-capacity execution view at a lifecycle
    boundary.
    """

    hidden_size: torch.Tensor
    memory_dim: torch.Tensor
    sensor_channel_mask: torch.Tensor
    sensor_radius: torch.Tensor
    introspection_channel_mask: torch.Tensor
    self_regulation_control_mask: torch.Tensor
    self_modification_target_mask: torch.Tensor
    directed_inheritance_channel_mask: torch.Tensor
    policy_mem_dim: torch.Tensor
    policy_use_gru: torch.Tensor
    exploration_bias: torch.Tensor
    aggression_bias: torch.Tensor
    mutation_rate: torch.Tensor
    reproduction_threshold: torch.Tensor
    action_temperature: torch.Tensor
    drive_inertia: torch.Tensor
    drive_noise: torch.Tensor
    drive_logit_gain: torch.Tensor
    pred_lr: torch.Tensor
    drive_setpoints: torch.Tensor
    genome_drive_sp_hunger: torch.Tensor
    genome_drive_sp_pain: torch.Tensor
    genome_drive_sp_fatigue: torch.Tensor
    genome_drive_sp_fear: torch.Tensor
    exp_prior_strength: torch.Tensor
    exp_prior_noise: torch.Tensor
    drive_birth_noise: torch.Tensor


@dataclass(slots=True)
class C3AdaptationTensorStateV1:
    chemistry: torch.Tensor
    pred_err_sum: torch.Tensor
    pred_err_count: torch.Tensor
    episode0_pred_err_sum: torch.Tensor
    episode0_pred_err_count: torch.Tensor
    late_pred_err_sum: torch.Tensor
    late_pred_err_count: torch.Tensor
    dev_stage_initialized: torch.Tensor
    chemistry_observation_fidelity: torch.Tensor
    chemistry_observation_bias: torch.Tensor


@dataclass(slots=True)
class C3PolicyRuntimeTensorStateV1:
    action_cache_actions: torch.Tensor
    action_cache_token: torch.Tensor
    action_cache_cursor: torch.Tensor
    action_cache_next: torch.Tensor


@dataclass(slots=True)
class C3LifecycleEligibilityTensorStateV1:
    death_candidate: torch.Tensor
    death_reason: torch.Tensor
    maturity_factor: torch.Tensor
    fertility_factor: torch.Tensor
    senescence_factor: torch.Tensor


@dataclass(slots=True)
class C3TransactionTensorStateV1:
    observations: torch.Tensor
    logits: torch.Tensor
    row_actions: torch.Tensor
    actions_by_slot: torch.Tensor
    pre_energy: torch.Tensor
    pre_health: torch.Tensor
    pre_injury: torch.Tensor
    pre_survival_steps: torch.Tensor
    chemistry_noise_samples: torch.Tensor
    chemistry_noise_samples_valid: torch.Tensor
    drive_noise_samples: torch.Tensor
    drive_noise_samples_valid: torch.Tensor
    hot_counters: torch.Tensor


@dataclass(slots=True)
class TensorWorldStateBundleV3:
    """C3 state owner; ``base`` remains the C2 world/mechanics authority."""

    base: TensorWorldStateBundleV2
    agent: C3AgentRuntimeTensorStateV1
    genome: C3GenomeExecutionTensorStateV1
    adaptation: C3AdaptationTensorStateV1
    policy_runtime: C3PolicyRuntimeTensorStateV1
    lifecycle: C3LifecycleEligibilityTensorStateV1
    transactions: C3TransactionTensorStateV1
    schema_version: str = C3_STATE_SCHEMA_VERSION_V1

    @property
    def device(self) -> torch.device:
        return self.base.device

    @property
    def batch(self) -> int:
        return self.base.batch

    @property
    def capacity(self) -> int:
        return self.base.capacity

    @property
    def grid_size(self) -> int:
        return self.base.grid_size

    @property
    def tick(self) -> int:
        return self.base.tick

    def validate(self) -> None:
        if self.schema_version != C3_STATE_SCHEMA_VERSION_V1:
            raise TensorWorldContractError("C3 state schema version is unsupported")
        self.base.validate()
        batch, capacity = self.batch, self.capacity
        slots = (batch, capacity)
        drive = (batch, capacity, C3_DRIVE_WIDTH_V1)
        chemistry = (batch, capacity, C3_CHEMISTRY_WIDTH_V1)
        cache = (batch, capacity, C3_POLICY_CACHE_WIDTH_V1)
        expected: tuple[tuple[str, torch.Tensor, tuple[int, ...], torch.dtype], ...] = (
            ("agent.injury", self.agent.injury, slots, torch.float32),
            ("agent.age", self.agent.age, slots, torch.int32),
            ("agent.survival_steps", self.agent.survival_steps, slots, torch.int32),
            ("agent.energy_collected", self.agent.energy_collected, slots, torch.float32),
            ("agent.energy_collected_exact", self.agent.energy_collected_exact, slots, torch.float64),
            ("agent.rest_streak", self.agent.rest_streak, slots, torch.int32),
            ("agent.policy_energy_observation", self.agent.policy_energy_observation, slots, torch.float64),
            ("agent.last_action", self.agent.last_action, slots, torch.int32),
            ("agent.last_last_action", self.agent.last_last_action, slots, torch.int32),
            ("agent.action_switches", self.agent.action_switches, slots, torch.int32),
            ("agent.action_oscillations", self.agent.action_oscillations, slots, torch.int32),
            ("agent.episode0_switches", self.agent.episode0_switches, slots, torch.int32),
            ("agent.episode0_oscillations", self.agent.episode0_oscillations, slots, torch.int32),
            ("agent.attacks", self.agent.attacks, slots, torch.int32),
            ("agent.gru_upkeep_cost", self.agent.gru_upkeep_cost, slots, torch.float32),
            ("agent.cost_charge_scalar", self.agent.cost_charge_scalar, slots, torch.float32),
            ("agent.cost_charge_maturation_scalar", self.agent.cost_charge_maturation_scalar, slots, torch.float32),
            ("agent.drives", self.agent.drives, drive, torch.float32),
            ("agent.drive_inertia", self.agent.drive_inertia, slots, torch.float32),
            ("agent.drive_noise", self.agent.drive_noise, slots, torch.float32),
            ("agent.prev_drive_delta", self.agent.prev_drive_delta, drive, torch.float32),
            ("agent.drive_volatility", self.agent.drive_volatility, slots, torch.float32),
            ("agent.drive_jitter", self.agent.drive_jitter, slots, torch.float32),
            ("agent.episode0_drive_jitter_sum", self.agent.episode0_drive_jitter_sum, slots, torch.float32),
            ("agent.last_energy_delta", self.agent.last_energy_delta, slots, torch.float32),
            ("agent.last_health_delta", self.agent.last_health_delta, slots, torch.float32),
            ("agent.last_injury_delta", self.agent.last_injury_delta, slots, torch.float32),
            ("agent.selfharm_events", self.agent.selfharm_events, slots, torch.int32),
            ("agent.episode0_selfharm_events", self.agent.episode0_selfharm_events, slots, torch.int32),
            ("agent.repro_threshold", self.agent.repro_threshold, slots, torch.float32),
            ("genome.hidden_size", self.genome.hidden_size, slots, torch.int32),
            ("genome.memory_dim", self.genome.memory_dim, slots, torch.int32),
            ("genome.sensor_channel_mask", self.genome.sensor_channel_mask, slots, torch.int32),
            ("genome.sensor_radius", self.genome.sensor_radius, slots, torch.int32),
            ("genome.introspection_channel_mask", self.genome.introspection_channel_mask, slots, torch.int32),
            ("genome.self_regulation_control_mask", self.genome.self_regulation_control_mask, slots, torch.int32),
            ("genome.self_modification_target_mask", self.genome.self_modification_target_mask, slots, torch.int32),
            ("genome.directed_inheritance_channel_mask", self.genome.directed_inheritance_channel_mask, slots, torch.int32),
            ("genome.policy_mem_dim", self.genome.policy_mem_dim, slots, torch.int64),
            ("genome.policy_use_gru", self.genome.policy_use_gru, slots, torch.bool),
            ("genome.exploration_bias", self.genome.exploration_bias, slots, torch.float32),
            ("genome.aggression_bias", self.genome.aggression_bias, slots, torch.float32),
            ("genome.mutation_rate", self.genome.mutation_rate, slots, torch.float32),
            ("genome.reproduction_threshold", self.genome.reproduction_threshold, slots, torch.float32),
            ("genome.action_temperature", self.genome.action_temperature, slots, torch.float32),
            ("genome.drive_inertia", self.genome.drive_inertia, slots, torch.float32),
            ("genome.drive_noise", self.genome.drive_noise, slots, torch.float32),
            ("genome.drive_logit_gain", self.genome.drive_logit_gain, slots, torch.float32),
            ("genome.pred_lr", self.genome.pred_lr, slots, torch.float32),
            ("genome.drive_setpoints", self.genome.drive_setpoints, drive, torch.float32),
            ("genome.genome_drive_sp_hunger", self.genome.genome_drive_sp_hunger, slots, torch.float32),
            ("genome.genome_drive_sp_pain", self.genome.genome_drive_sp_pain, slots, torch.float32),
            ("genome.genome_drive_sp_fatigue", self.genome.genome_drive_sp_fatigue, slots, torch.float32),
            ("genome.genome_drive_sp_fear", self.genome.genome_drive_sp_fear, slots, torch.float32),
            ("genome.exp_prior_strength", self.genome.exp_prior_strength, slots, torch.float32),
            ("genome.exp_prior_noise", self.genome.exp_prior_noise, slots, torch.float32),
            ("genome.drive_birth_noise", self.genome.drive_birth_noise, slots, torch.float32),
            ("adaptation.chemistry", self.adaptation.chemistry, chemistry, torch.float32),
            ("adaptation.pred_err_sum", self.adaptation.pred_err_sum, slots, torch.float32),
            ("adaptation.pred_err_count", self.adaptation.pred_err_count, slots, torch.int32),
            ("adaptation.episode0_pred_err_sum", self.adaptation.episode0_pred_err_sum, slots, torch.float32),
            ("adaptation.episode0_pred_err_count", self.adaptation.episode0_pred_err_count, slots, torch.int32),
            ("adaptation.late_pred_err_sum", self.adaptation.late_pred_err_sum, slots, torch.float32),
            ("adaptation.late_pred_err_count", self.adaptation.late_pred_err_count, slots, torch.int32),
            ("adaptation.dev_stage_initialized", self.adaptation.dev_stage_initialized, slots, torch.bool),
            ("adaptation.chemistry_observation_fidelity", self.adaptation.chemistry_observation_fidelity, slots, torch.float32),
            ("adaptation.chemistry_observation_bias", self.adaptation.chemistry_observation_bias, slots, torch.float32),
            ("policy_runtime.action_cache_actions", self.policy_runtime.action_cache_actions, cache, torch.int32),
            ("policy_runtime.action_cache_token", self.policy_runtime.action_cache_token, slots, torch.int64),
            ("policy_runtime.action_cache_cursor", self.policy_runtime.action_cache_cursor, slots, torch.int32),
            ("policy_runtime.action_cache_next", self.policy_runtime.action_cache_next, slots, torch.int32),
            ("lifecycle.death_candidate", self.lifecycle.death_candidate, slots, torch.bool),
            ("lifecycle.death_reason", self.lifecycle.death_reason, slots, torch.int32),
            ("lifecycle.maturity_factor", self.lifecycle.maturity_factor, slots, torch.float32),
            ("lifecycle.fertility_factor", self.lifecycle.fertility_factor, slots, torch.float32),
            ("lifecycle.senescence_factor", self.lifecycle.senescence_factor, slots, torch.float32),
            ("transactions.observations", self.transactions.observations, (batch, capacity, C3_OBSERVATION_WIDTH_V1), torch.float32),
            ("transactions.logits", self.transactions.logits, (batch, capacity, C3_ACTION_COUNT_V1), torch.float32),
            ("transactions.row_actions", self.transactions.row_actions, slots, torch.int32),
            ("transactions.actions_by_slot", self.transactions.actions_by_slot, slots, torch.int32),
            ("transactions.pre_energy", self.transactions.pre_energy, slots, torch.float32),
            ("transactions.pre_health", self.transactions.pre_health, slots, torch.float32),
            ("transactions.pre_injury", self.transactions.pre_injury, slots, torch.float32),
            ("transactions.pre_survival_steps", self.transactions.pre_survival_steps, slots, torch.int32),
            ("transactions.chemistry_noise_samples", self.transactions.chemistry_noise_samples, (batch, capacity, C3_CHEMISTRY_NOISE_WIDTH_V1), torch.float32),
            ("transactions.chemistry_noise_samples_valid", self.transactions.chemistry_noise_samples_valid, slots, torch.bool),
            ("transactions.drive_noise_samples", self.transactions.drive_noise_samples, drive, torch.float32),
            ("transactions.drive_noise_samples_valid", self.transactions.drive_noise_samples_valid, slots, torch.bool),
            ("transactions.hot_counters", self.transactions.hot_counters, (batch, 12), torch.int64),
        )
        devices = {self.base.device}
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
            raise TensorWorldContractError("all C3 resident state must share one device")


def c3_state_tensor_names_v1() -> tuple[str, ...]:
    names: list[str] = []
    for owner in (
        C3AgentRuntimeTensorStateV1,
        C3GenomeExecutionTensorStateV1,
        C3AdaptationTensorStateV1,
        C3PolicyRuntimeTensorStateV1,
        C3LifecycleEligibilityTensorStateV1,
        C3TransactionTensorStateV1,
    ):
        names.extend(f"{owner.__name__}.{field.name}" for field in fields(owner))
    return tuple(names)


__all__ = [
    "C3_ACTION_COUNT_V1",
    "C3_CHEMISTRY_WIDTH_V1",
    "C3_CHEMISTRY_NOISE_WIDTH_V1",
    "C3_OBSERVATION_WIDTH_V1",
    "C3_POLICY_CACHE_WIDTH_V1",
    "C3_STATE_SCHEMA_VERSION_V1",
    "C3AdaptationTensorStateV1",
    "C3AgentRuntimeTensorStateV1",
    "C3GenomeExecutionTensorStateV1",
    "C3LifecycleEligibilityTensorStateV1",
    "C3PolicyRuntimeTensorStateV1",
    "C3TransactionTensorStateV1",
    "TensorWorldStateBundleV3",
    "c3_state_tensor_names_v1",
]
