"""Deterministic C2 fixtures spanning Patch/Jungle/WDL/EED temporal semantics."""

from __future__ import annotations

import torch

from .c1_fixture import make_c1_fixture_v1
from .c2_backend import TensorWorldC2ConfigV1
from .c2_state import (
    AgentEffectBufferV1,
    C2HotMetricStateV1,
    C2PolicyTensorStateV1,
    C2RngTensorStateV1,
    C2TransactionTensorStateV1,
    C2_OBSERVATION_WIDTH_V1,
    C2_POLICY_PACKED_WIDTH_V1,
    EedTensorStateV1,
    PatchTensorStateV1,
    TensorWorldStateBundleV2,
    WdlTensorStateV1,
    WorldEventBufferV1,
)


def make_c2_fixture_v1(
    *, batch: int, capacity: int, active: int, width: int = 32
) -> tuple[TensorWorldStateBundleV2, TensorWorldC2ConfigV1]:
    core, _ = make_c1_fixture_v1(
        batch=batch, capacity=capacity, active=active, width=width
    )
    grid = (batch, width, width)
    slots = (batch, capacity)
    base = torch.arange(width * width, dtype=torch.float32).reshape(width, width)
    fertility = (0.4 + (base.remainder(7) / 10.0)).clamp_max(1.0).repeat(batch, 1, 1)
    resource_class = torch.where(
        torch.arange(width).view(1, width, 1) < width // 2,
        torch.tensor(1.0), torch.tensor(2.0),
    ).expand(grid).contiguous()
    resource_a = torch.where(resource_class == 1, torch.tensor(0.8), torch.tensor(0.0))
    resource_b = torch.where(resource_class == 2, torch.tensor(0.7), torch.tensor(0.0))
    core.world.food_amount.copy_(resource_a + resource_b)
    core.world.scent.mul_(0.1)
    core.world.waste.mul_(0.1)
    agent_grid_i32 = core.occupancy.agent_id_grid.to(torch.int32)
    food_grid_i32 = (core.world.food_amount > 0.5).to(torch.int32)

    base_policy = (
        torch.arange(C2_POLICY_PACKED_WIDTH_V1, dtype=torch.int64)
        .remainder(29).sub_(14).to(torch.float32).mul_(1.0e-4)
    )
    packed = base_policy.view(1, 1, -1).expand(batch, capacity, -1).clone()
    desired = torch.arange(capacity).remainder(7)
    if active >= 2:
        desired[0], desired[1] = 2, 3
    if active >= 3:
        desired[2] = 4
    if active >= 4:
        desired[3] = 5
    bias_offset = C2_POLICY_PACKED_WIDTH_V1 - 7
    for action in range(7):
        packed[:, desired == action, bias_offset + action] = 20.0
    packed[:, :, 0].add_(torch.arange(capacity, dtype=torch.float32).view(1, -1) * 1.0e-7)

    bundle = TensorWorldStateBundleV2(
        core=core,
        patch=PatchTensorStateV1(
            fertility=fertility,
            scent_emit_pending=torch.zeros(grid),
            waste_emit_pending=torch.zeros(grid),
            recovery_activity=torch.full(grid, 0.2),
            visibility=torch.zeros(grid),
            agent_grid_i32=agent_grid_i32,
            food_grid_i32=food_grid_i32,
            multiplier_scratch=torch.zeros(grid),
            waste_penalty_scratch=torch.zeros(grid),
            density_penalty_scratch=torch.zeros(grid),
            sorted_multiplier_scratch=torch.zeros(grid),
            recovery_boost_scratch=torch.zeros(grid),
            sorted_recovery_boost_scratch=torch.zeros(grid),
        ),
        wdl=WdlTensorStateV1(
            cue=torch.zeros(grid), timer=torch.full(grid, -1, dtype=torch.int32),
            active=torch.zeros(grid), intensity=torch.zeros(grid),
            age=torch.zeros(grid, dtype=torch.int32),
            event_id=torch.zeros(grid, dtype=torch.int64),
            prior_cue=torch.zeros(grid, dtype=torch.int8),
            base_seed=torch.arange(10_001, 10_001 + batch, dtype=torch.int64),
            next_absolute_step=torch.zeros(batch, dtype=torch.int64),
        ),
        eed=EedTensorStateV1(
            resource_a_amount=resource_a.clone(), resource_b_amount=resource_b.clone(),
            resource_a_capacity=torch.ones(grid), resource_b_capacity=torch.ones(grid),
            resource_a_renewal=torch.full(grid, 0.25),
            resource_b_renewal=torch.full(grid, 0.25),
            resource_a_pending_impact=torch.zeros(grid),
            resource_b_pending_impact=torch.zeros(grid),
            resource_a_impact_load=torch.zeros(grid),
            resource_b_impact_load=torch.zeros(grid),
            resource_class=resource_class,
            active_y=torch.full(slots, -1, dtype=torch.int32),
            active_x=torch.full(slots, -1, dtype=torch.int32),
            active_class=torch.zeros(slots, dtype=torch.int32),
            handling_progress=torch.zeros(slots),
            handling_axis=(torch.arange(capacity).remainder(5).float().sub(2).div(2).view(1, -1).expand(batch, -1).clone()),
        ),
        policy=C2PolicyTensorStateV1(packed),
        rng=C2RngTensorStateV1(
            torch.where(
                torch.arange(capacity).remainder(11).view(1, -1) == 0,
                torch.tensor(0.0, dtype=torch.float64),
                torch.tensor(0.99, dtype=torch.float64),
            ).expand(batch, -1).clone()
        ),
        effects=AgentEffectBufferV1(
            torch.zeros(slots), torch.zeros(slots), torch.zeros(slots, dtype=torch.uint8)
        ),
        world_events=WorldEventBufferV1(
            torch.zeros(grid), torch.zeros(grid), torch.zeros(slots),
            torch.zeros(slots, dtype=torch.int32),
        ),
        transactions=C2TransactionTensorStateV1(
            torch.zeros(batch, capacity, C2_OBSERVATION_WIDTH_V1),
            torch.zeros(batch, capacity, 7),
        ),
        hot_metrics=C2HotMetricStateV1(
            torch.zeros(batch, 25, dtype=torch.float64),
            torch.zeros(batch, 7, dtype=torch.float64),
            torch.zeros(batch, 8, dtype=torch.int64),
            torch.zeros(batch, 6, dtype=torch.float64),
            torch.zeros(batch, 7, dtype=torch.int64),
            torch.zeros(batch, dtype=torch.float64),
            torch.zeros(batch, 5, dtype=torch.int64),
            torch.zeros(batch, 4, dtype=torch.float64),
        ),
    )
    bundle.validate()
    return bundle, TensorWorldC2ConfigV1()
