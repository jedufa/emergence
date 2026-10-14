"""Deterministic C3 fixture with active mechanics and resident genome views."""

from __future__ import annotations

import torch

from .c2_fixture import make_c2_fixture_v1
from .c2_backend import TensorWorldC2ConfigV1
from .c3_state import (
    C3_CHEMISTRY_NOISE_WIDTH_V1,
    C3AdaptationTensorStateV1,
    C3AgentRuntimeTensorStateV1,
    C3GenomeExecutionTensorStateV1,
    C3LifecycleEligibilityTensorStateV1,
    C3PolicyRuntimeTensorStateV1,
    C3TransactionTensorStateV1,
    TensorWorldStateBundleV3,
)


def make_c3_fixture_v1(
    *, batch: int, capacity: int, active: int, width: int = 32
) -> tuple[TensorWorldStateBundleV3, TensorWorldC2ConfigV1]:
    """Create a C2 world fixture with source-current C3 resident fields."""

    base, config = make_c2_fixture_v1(
        batch=batch, capacity=capacity, active=active, width=width
    )
    slots = (batch, capacity)
    drives = (batch, capacity, 4)

    def zeros_i32() -> torch.Tensor:
        return torch.zeros(slots, dtype=torch.int32)

    def zeros_f32() -> torch.Tensor:
        return torch.zeros(slots, dtype=torch.float32)

    source_drives = base.core.policy.drives.clone()

    agent = C3AgentRuntimeTensorStateV1(
        injury=zeros_f32(),
        age=base.core.agents.age.to(torch.int32).clone(),
        survival_steps=base.core.agents.survival_steps.to(torch.int32).clone(),
        energy_collected=base.core.agents.energy_collected.to(torch.float32).clone(),
        energy_collected_exact=base.core.agents.energy_collected.clone(),
        rest_streak=zeros_i32(),
        policy_energy_observation=base.core.agents.energy.to(torch.float64).clone(),
        last_action=torch.full(slots, -1, dtype=torch.int32),
        last_last_action=torch.full(slots, -1, dtype=torch.int32),
        action_switches=zeros_i32(),
        action_oscillations=zeros_i32(),
        episode0_switches=zeros_i32(),
        episode0_oscillations=zeros_i32(),
        attacks=zeros_i32(),
        gru_upkeep_cost=zeros_f32(),
        cost_charge_scalar=zeros_f32(),
        cost_charge_maturation_scalar=zeros_f32(),
        drives=source_drives.clone(),
        drive_inertia=torch.full(slots, 0.25),
        drive_noise=torch.zeros(slots),
        prev_drive_delta=torch.zeros(drives),
        drive_volatility=zeros_f32(),
        drive_jitter=zeros_f32(),
        episode0_drive_jitter_sum=zeros_f32(),
        last_energy_delta=zeros_f32(),
        last_health_delta=zeros_f32(),
        last_injury_delta=zeros_f32(),
        selfharm_events=zeros_i32(),
        episode0_selfharm_events=zeros_i32(),
        repro_threshold=torch.full(slots, 0.5),
    )

    genome = C3GenomeExecutionTensorStateV1(
        hidden_size=torch.full(slots, 64, dtype=torch.int32),
        memory_dim=zeros_i32(),
        sensor_channel_mask=torch.full(slots, 0xFF, dtype=torch.int32),
        sensor_radius=torch.ones(slots, dtype=torch.int32),
        introspection_channel_mask=zeros_i32(),
        self_regulation_control_mask=zeros_i32(),
        self_modification_target_mask=zeros_i32(),
        directed_inheritance_channel_mask=zeros_i32(),
        policy_mem_dim=torch.zeros(slots, dtype=torch.int64),
        policy_use_gru=torch.zeros(slots, dtype=torch.bool),
        exploration_bias=zeros_f32(),
        aggression_bias=base.core.policy.attack_bias.clone(),
        mutation_rate=torch.full(slots, 0.01),
        reproduction_threshold=torch.full(slots, 0.5),
        action_temperature=base.core.policy.action_temperature.clone(),
        drive_inertia=agent.drive_inertia.clone(),
        drive_noise=agent.drive_noise.clone(),
        drive_logit_gain=base.core.policy.drive_logit_gain.clone(),
        pred_lr=torch.full(slots, 0.1),
        drive_setpoints=source_drives.clone(),
        genome_drive_sp_hunger=source_drives[..., 0].clone(),
        genome_drive_sp_pain=source_drives[..., 1].clone(),
        genome_drive_sp_fatigue=source_drives[..., 2].clone(),
        genome_drive_sp_fear=source_drives[..., 3].clone(),
        exp_prior_strength=torch.full(slots, 0.5),
        exp_prior_noise=torch.zeros(slots),
        drive_birth_noise=torch.zeros(slots),
    )

    adaptation = C3AdaptationTensorStateV1(
        chemistry=torch.zeros(batch, capacity, 6),
        pred_err_sum=zeros_f32(),
        pred_err_count=zeros_i32(),
        episode0_pred_err_sum=zeros_f32(),
        episode0_pred_err_count=zeros_i32(),
        late_pred_err_sum=zeros_f32(),
        late_pred_err_count=zeros_i32(),
        dev_stage_initialized=torch.zeros(slots, dtype=torch.bool),
        chemistry_observation_fidelity=torch.ones(slots),
        chemistry_observation_bias=zeros_f32(),
    )

    policy_runtime = C3PolicyRuntimeTensorStateV1(
        action_cache_actions=torch.full((batch, capacity, 2), -1, dtype=torch.int32),
        action_cache_token=torch.full(slots, -1, dtype=torch.int64),
        action_cache_cursor=torch.full(slots, -1, dtype=torch.int32),
        action_cache_next=torch.full(slots, -1, dtype=torch.int32),
    )
    lifecycle = C3LifecycleEligibilityTensorStateV1(
        death_candidate=torch.zeros(slots, dtype=torch.bool),
        death_reason=zeros_i32(),
        maturity_factor=torch.zeros(slots),
        fertility_factor=torch.zeros(slots),
        senescence_factor=torch.zeros(slots),
    )
    transactions = C3TransactionTensorStateV1(
        observations=torch.zeros(batch, capacity, 83),
        logits=torch.zeros(batch, capacity, 7),
        row_actions=torch.full(slots, -1, dtype=torch.int32),
        actions_by_slot=torch.full(slots, -1, dtype=torch.int32),
        pre_energy=torch.zeros(slots),
        pre_health=torch.zeros(slots),
        pre_injury=torch.zeros(slots),
        pre_survival_steps=torch.zeros(slots, dtype=torch.int32),
        chemistry_noise_samples=torch.zeros(
            batch, capacity, C3_CHEMISTRY_NOISE_WIDTH_V1
        ),
        chemistry_noise_samples_valid=torch.zeros(slots, dtype=torch.bool),
        drive_noise_samples=torch.zeros(drives),
        drive_noise_samples_valid=torch.zeros(slots, dtype=torch.bool),
        hot_counters=torch.zeros(batch, 12, dtype=torch.int64),
    )
    bundle = TensorWorldStateBundleV3(
        base=base,
        agent=agent,
        genome=genome,
        adaptation=adaptation,
        policy_runtime=policy_runtime,
        lifecycle=lifecycle,
        transactions=transactions,
    )
    bundle.validate()
    return bundle, config


__all__ = ["make_c3_fixture_v1"]
