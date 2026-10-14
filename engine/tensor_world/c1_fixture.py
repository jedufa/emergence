"""Bounded source-semantic fixtures and independent C1 oracle."""

from __future__ import annotations

import torch

from .c1_backend import TensorWorldC1ConfigV1
from .state import (
    AgentTensorStateV1,
    HotEventAccumulatorStateV1,
    IdentityTensorStateV1,
    OccupancyTensorStateV1,
    PATCH_OBSERVATION_WIDTH_V1,
    PATCH_POLICY_PACKED_WIDTH_V1,
    PolicyTensorStateV1,
    RngTensorStateV1,
    TensorWorldStateBundleV1,
    TransactionTensorStateV1,
    WorldTensorStateV1,
)


def make_c1_fixture_v1(
    *, batch: int, capacity: int, active: int, width: int = 32
) -> tuple[TensorWorldStateBundleV1, TensorWorldC1ConfigV1]:
    """Create a deterministic real-policy fixture with controlled conflicts."""

    if active <= 0 or active > capacity or active > (width // 2) ** 2:
        raise ValueError("fixture active population does not fit the even-cell layout")
    grid_shape = (batch, width, width)
    slot_shape = (batch, capacity)
    alive = torch.zeros(slot_shape, dtype=torch.bool)
    alive[:, :active] = True
    agent_ids = torch.full(slot_shape, -1, dtype=torch.int64)
    positions_y = torch.zeros(slot_shape, dtype=torch.int64)
    positions_x = torch.zeros(slot_shape, dtype=torch.int64)
    id_grid = torch.full(grid_shape, -1, dtype=torch.int64)
    slot_grid = torch.full(grid_shape, -1, dtype=torch.int64)
    per_row = width // 2
    for world in range(batch):
        for slot in range(active):
            agent_id = world * 10_000 + slot + 1
            y = (slot // per_row) * 2
            x = (slot % per_row) * 2
            agent_ids[world, slot] = agent_id
            positions_y[world, slot] = y
            positions_x[world, slot] = x
            id_grid[world, y, x] = agent_id
            slot_grid[world, y, x] = slot
    food = torch.zeros(grid_shape)
    for world in range(batch):
        for slot in range(0, active, 7):
            food[world, positions_y[world, slot], positions_x[world, slot]] = 1.0
        if active >= 3:
            food[world, positions_y[world, 2], positions_x[world, 2]] = 1.0
    base = torch.arange(width * width, dtype=torch.float32).reshape(width, width)
    cover = ((base % 11) / 10).repeat(batch, 1, 1)
    shelter = ((base % 7) / 6).repeat(batch, 1, 1)
    scent = ((base % 13) / 12).repeat(batch, 1, 1)
    waste = ((base % 5) / 4).repeat(batch, 1, 1)
    movement = (0.75 + (base % 9) / 8).repeat(batch, 1, 1)
    base_policy = (
        torch.arange(PATCH_POLICY_PACKED_WIDTH_V1, dtype=torch.int64)
        .remainder(29)
        .sub_(14)
        .to(torch.float32)
        .mul_(1.0e-4)
    )
    packed = (
        base_policy.view(1, 1, PATCH_POLICY_PACKED_WIDTH_V1)
        .expand(batch, capacity, PATCH_POLICY_PACKED_WIDTH_V1)
        .clone()
    )
    b3_offset = 64 * 74 + 64 + 64 * 64 + 64 + 7 * 64
    desired = torch.arange(capacity).remainder(7)
    # Rows 0 and 1 form an order-sensitive E/W conflict for L1/L2.
    if active >= 2:
        desired[0], desired[1] = 2, 3
    if active >= 3:
        desired[2] = 4
    if active >= 4:
        desired[3] = 5
    for action in range(7):
        packed[:, desired == action, b3_offset + action] = 20.0
    # Distinct, non-shared rows while retaining an unambiguous action margin.
    row_identity = torch.arange(capacity, dtype=torch.float32).mul_(1.0e-7)
    world_identity = torch.arange(batch, dtype=torch.float32).mul_(1.0e-8)
    packed[:, :, 0].add_(row_identity.view(1, capacity))
    packed[:, :, 1].add_(world_identity.view(batch, 1))
    uniforms = torch.full((batch, capacity, 7), 0.5, dtype=torch.float64)
    slot_axis = torch.arange(capacity, dtype=torch.float32).view(1, capacity)
    drives = (
        torch.stack(
            tuple(((slot_axis + channel) % 7) / 7.0 for channel in range(4)), dim=2
        )
        .expand(batch, capacity, 4)
        .clone()
    )
    attack_bias = ((slot_axis % 3) * 0.01).expand(batch, capacity).clone()
    temperature = (0.8 + (slot_axis % 5) * 0.05).expand(batch, capacity).clone()
    drive_gain = (0.1 + (slot_axis % 4) * 0.02).expand(batch, capacity).clone()
    bundle = TensorWorldStateBundleV1(
        world=WorldTensorStateV1(
            food, cover, shelter, scent, waste, torch.zeros(grid_shape), movement
        ),
        agents=AgentTensorStateV1(
            alive,
            agent_ids,
            positions_y,
            positions_x,
            torch.full(slot_shape, 80.0),
            torch.full(slot_shape, 12.0),
            torch.zeros(slot_shape, dtype=torch.int64),
            torch.zeros(slot_shape, dtype=torch.int64),
            torch.zeros(slot_shape, dtype=torch.float64),
        ),
        occupancy=OccupancyTensorStateV1(id_grid, slot_grid),
        policy=PolicyTensorStateV1(
            packed,
            drives,
            attack_bias,
            temperature,
            drive_gain,
        ),
        identity=IdentityTensorStateV1(
            torch.arange(batch, dtype=torch.int64),
            torch.arange(capacity, dtype=torch.int64).repeat(batch, 1),
        ),
        rng=RngTensorStateV1(uniforms, torch.zeros(batch, dtype=torch.int64)),
        transactions=TransactionTensorStateV1(
            torch.full(slot_shape, -1, dtype=torch.int64),
            torch.zeros(slot_shape, dtype=torch.bool),
            torch.zeros(slot_shape, dtype=torch.bool),
            torch.zeros(slot_shape),
            torch.zeros(batch, capacity, PATCH_OBSERVATION_WIDTH_V1),
            torch.zeros(batch, capacity, 7),
        ),
        hot_events=HotEventAccumulatorStateV1(
            torch.zeros(batch, dtype=torch.int64),
            torch.zeros(batch, dtype=torch.int64),
            torch.zeros(batch, dtype=torch.int64),
        ),
    )
    bundle.validate()
    return bundle, TensorWorldC1ConfigV1()


def oracle_tick_v1(
    source: TensorWorldStateBundleV1, config: TensorWorldC1ConfigV1
) -> TensorWorldStateBundleV1:
    """Independent scalar oracle matching the admitted source-current slice."""

    from .c1_backend import clone_bundle_to_device_v1

    state = clone_bundle_to_device_v1(source, torch.device("cpu"))
    batch, capacity, width = state.batch, state.capacity, state.grid_size
    for world in range(batch):
        for slot in range(capacity):
            if not bool(state.agents.alive[world, slot]):
                continue
            cy = int(state.agents.pos_y[world, slot])
            cx = int(state.agents.pos_x[world, slot])
            out = state.transactions.observations[world, slot]
            t = 0
            denom = float(
                (2 * config.density_radius + 1) ** 2 - (not config.density_include_self)
            )
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    y, x = cy + dy, cx + dx
                    if 0 <= y < width and 0 <= x < width:
                        occupied = int(state.occupancy.agent_id_grid[world, y, x]) != -1
                        out[t] = (
                            -1.0
                            if occupied
                            else (
                                1.0
                                if float(state.world.food_amount[world, y, x]) > 0.5
                                else 0.0
                            )
                        )
                        out[9 + t] = torch.clamp(
                            state.world.food_amount[world, y, x]
                            / config.food_amount_max,
                            0,
                            1,
                        )
                        out[18 + t] = torch.clamp(state.world.cover[world, y, x], 0, 1)
                        out[27 + t] = torch.clamp(
                            state.world.shelter[world, y, x], 0, 1
                        )
                        out[36 + t] = torch.clamp(
                            state.world.scent[world, y, x] / config.scent_max, 0, 1
                        )
                        out[45 + t] = torch.clamp(
                            state.world.waste[world, y, x] / config.waste_max, 0, 1
                        )
                        count = 0
                        for yy in range(
                            max(0, y - config.density_radius),
                            min(width, y + config.density_radius + 1),
                        ):
                            for xx in range(
                                max(0, x - config.density_radius),
                                min(width, x + config.density_radius + 1),
                            ):
                                count += (
                                    int(state.occupancy.agent_id_grid[world, yy, xx])
                                    != -1
                                )
                        if not config.density_include_self and occupied:
                            count -= 1
                        out[54 + t] = min(
                            1.0,
                            max(
                                0.0,
                                float(count)
                                if config.density_normalize
                                else count / denom,
                            ),
                        )
                        span = config.movement_cost_max - config.movement_cost_min
                        out[63 + t] = min(
                            1.0,
                            max(
                                0.0,
                                (
                                    float(state.world.movement_cost[world, y, x])
                                    - config.movement_cost_min
                                )
                                / span,
                            ),
                        )
                    t += 1
            out[72] = state.agents.energy[world, slot] / config.max_energy
            out[73] = state.agents.health[world, slot] / config.start_health
    _oracle_policy_and_actions(state, config)
    _oracle_settle(state, config)
    state.tick += 1
    return state


def _oracle_policy_and_actions(
    state: TensorWorldStateBundleV1, config: TensorWorldC1ConfigV1
) -> None:
    for world in range(state.batch):
        for slot in range(state.capacity):
            if not bool(state.agents.alive[world, slot]):
                continue
            pack = state.policy.packed_weights[world, slot]
            offset = 0
            w1 = pack[offset : offset + 64 * 74].view(64, 74)
            offset += 64 * 74
            b1 = pack[offset : offset + 64]
            offset += 64
            w2 = pack[offset : offset + 64 * 64].view(64, 64)
            offset += 64 * 64
            b2 = pack[offset : offset + 64]
            offset += 64
            w3 = pack[offset : offset + 7 * 64].view(7, 64)
            offset += 7 * 64
            logits = (
                w3
                @ torch.tanh(
                    w2
                    @ torch.tanh(w1 @ state.transactions.observations[world, slot] + b1)
                    + b2
                )
                + pack[offset : offset + 7]
            )
            logits[6] += state.policy.attack_bias[world, slot]
            logits /= max(0.05, float(state.policy.action_temperature[world, slot]))
            drive = state.policy.drives[world, slot]
            gain = max(0.0, float(state.policy.drive_logit_gain[world, slot]))
            logits[4] += gain * config.w_hunger_gather * drive[0]
            logits[5] += gain * (
                config.w_pain_rest * drive[1] + config.w_fatigue_rest * drive[2]
            )
            logits[6] -= gain * (
                config.w_pain_attack * drive[1] + config.w_fear_attack * drive[3]
            )
            state.transactions.logits[world, slot] = logits
            uniforms = (
                state.rng.action_uniforms[world, slot]
                .to(torch.float32)
                .clamp(1.0e-6, 1.0 - 1.0e-6)
            )
            scores = logits - torch.log(-torch.log(uniforms))
            state.transactions.actions[world, slot] = int(torch.argmax(scores))
        state.rng.action_cursor[world] += 1


def _oracle_settle(
    state: TensorWorldStateBundleV1, config: TensorWorldC1ConfigV1
) -> None:
    for world in range(state.batch):
        attempted = committed = gathers = 0
        for slot in range(state.capacity):
            state.transactions.moved[world, slot] = False
            state.transactions.gathered[world, slot] = False
            state.transactions.food_gained[world, slot] = 0.0
            if not bool(state.agents.alive[world, slot]):
                continue
            action = int(state.transactions.actions[world, slot])
            if action > 3:
                continue
            attempted += 1
            y, x = (
                int(state.agents.pos_y[world, slot]),
                int(state.agents.pos_x[world, slot]),
            )
            ny, nx = y, x
            if action == 0:
                ny -= 1
            elif action == 1:
                ny += 1
            elif action == 2:
                nx += 1
            else:
                nx -= 1
            if not (0 <= ny < state.grid_size and 0 <= nx < state.grid_size):
                continue
            if int(state.occupancy.agent_id_grid[world, ny, nx]) != -1:
                continue
            state.occupancy.agent_id_grid[world, y, x] = -1
            state.occupancy.slot_grid[world, y, x] = -1
            state.occupancy.agent_id_grid[world, ny, nx] = state.agents.agent_id[
                world, slot
            ]
            state.occupancy.slot_grid[world, ny, nx] = slot
            state.agents.pos_y[world, slot], state.agents.pos_x[world, slot] = ny, nx
            state.transactions.moved[world, slot] = True
            committed += 1
        for slot in range(state.capacity):
            if (
                not bool(state.agents.alive[world, slot])
                or int(state.transactions.actions[world, slot]) != 4
            ):
                continue
            y, x = (
                int(state.agents.pos_y[world, slot]),
                int(state.agents.pos_x[world, slot]),
            )
            amount = float(state.world.food_amount[world, y, x])
            if amount <= config.food_presence_threshold:
                continue
            harvest = min(amount, config.gather_amount)
            state.world.food_amount[world, y, x] = max(0.0, amount - harvest)
            gained = torch.tensor(
                config.food_energy * harvest / config.gather_amount, dtype=torch.float32
            )
            state.agents.energy_collected[world, slot] += float(gained)
            state.transactions.food_gained[world, slot] = gained
            state.transactions.gathered[world, slot] = True
            gathers += 1
        costs = (config.move_cost,) * 4 + (
            config.gather_cost,
            config.rest_cost,
            config.attack_cost,
        )
        for slot in range(state.capacity):
            if not bool(state.agents.alive[world, slot]):
                continue
            action = int(state.transactions.actions[world, slot])
            cost = costs[action] * config.action_cost_mult
            if action == 5:
                cost *= config.rest_cost_mult
            if (
                bool(state.transactions.moved[world, slot])
                and config.movement_cost_apply
            ):
                y, x = (
                    int(state.agents.pos_y[world, slot]),
                    int(state.agents.pos_x[world, slot]),
                )
                mult = max(
                    0.0,
                    1.0
                    + config.movement_cost_energy_scale
                    * (float(state.world.movement_cost[world, y, x]) - 1.0),
                )
                cost = max(
                    0.0,
                    float(torch.tensor(cost, dtype=torch.float32))
                    * float(torch.tensor(mult, dtype=torch.float32)),
                )
            settled = state.agents.energy[world, slot] - torch.tensor(
                cost, dtype=torch.float32
            )
            settled += state.transactions.food_gained[world, slot]
            state.agents.energy[world, slot] = min(
                config.max_energy, max(0.0, float(settled))
            )
        state.hot_events.moves_attempted[world] = attempted
        state.hot_events.moves_committed[world] = committed
        state.hot_events.gathers_committed[world] = gathers
