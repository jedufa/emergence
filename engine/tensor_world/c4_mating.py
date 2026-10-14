"""CPU source transcription of the C4 mate/placement planning boundary.

This module owns the source-current adjacent-mate planning boundary and a
CPU fixed-MLP commit adapter.  Policy-vector recombination and resident CUDA
execution remain separate, explicitly unsupported contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .c4_population import (
    C4_BIRTH_REASON_ACCEPTED_V1,
    C4_BIRTH_REASON_ALREADY_MATED_V1,
    C4_BIRTH_REASON_BUFFER_V1,
    C4_BIRTH_REASON_COOLDOWN_V1,
    C4_BIRTH_REASON_HEALTH_V1,
    C4_BIRTH_REASON_NO_FREE_SLOT_V1,
    C4_BIRTH_REASON_NO_MATE_V1,
    C4_BIRTH_REASON_NO_ENERGY_V1,
    C4_BIRTH_REASON_NO_SPACE_V1,
    TensorWorldC4BirthConfigV1,
    _c4_fertility_scale_v1,
    commit_c4_ordinary_births_oracle_v1,
)
from .c4_state import TensorWorldStateBundleV4
from .c5_determinism import (
    TensorWorldC5ReplayContextV1,
    c5_pcg64_bounded_uint32_v1,
)
from .contracts import TensorWorldContractError


@dataclass(frozen=True, slots=True)
class TensorWorldC4MateBirthConfigV1:
    max_births: int | None = None
    parent_energy_share: float = 0.5
    minimum_energy_buffer: float = 0.0
    cooldown_steps: int = 0
    life_history_mode: int = 0
    maturity_start_ticks: int = 0
    maturity_rise_ticks: int = 0
    fertility_decline_start_ticks: int = 0
    fertility_decline_ticks: int = 0
    elder_fertility_floor: float = 1.0
    generation: int = 0
    runner_stream: int = 1
    environment_stream: int = 0
    requires_mate: bool = True
    allow_mutation: bool = False
    start_health: float = 12.0
    max_energy: float = 80.0

    def __post_init__(self) -> None:
        if self.max_births is not None and (
            type(self.max_births) is not int or self.max_births < 0
        ):
            raise TensorWorldContractError("C4 mate max_births must be nonnegative int")
        if not (0.0 <= float(self.parent_energy_share) <= 1.0):
            raise TensorWorldContractError("C4 mate parent_energy_share is outside [0, 1]")
        if float(self.minimum_energy_buffer) < 0.0:
            raise TensorWorldContractError("C4 mate minimum_energy_buffer must be nonnegative")
        for name in (
            "cooldown_steps", "life_history_mode", "maturity_start_ticks",
            "maturity_rise_ticks", "fertility_decline_start_ticks",
            "fertility_decline_ticks", "generation", "runner_stream",
            "environment_stream",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise TensorWorldContractError(f"C4 mate {name} must be nonnegative int")
        if self.life_history_mode not in (0, 1):
            raise TensorWorldContractError("C4 mate life_history_mode must be 0 or 1")
        if not isinstance(self.elder_fertility_floor, (int, float)) or not (
            0.0 <= float(self.elder_fertility_floor) <= 1.0
        ):
            raise TensorWorldContractError("C4 mate elder_fertility_floor is outside [0, 1]")
        if type(self.requires_mate) is not bool or not self.requires_mate:
            raise TensorWorldContractError("C4 mate planner requires mate semantics")
        if type(self.allow_mutation) is not bool:
            raise TensorWorldContractError("C4 mate allow_mutation must be bool")
        if not math.isfinite(float(self.start_health)) or float(self.start_health) < 0.0:
            raise TensorWorldContractError("C4 mate start_health must be finite and nonnegative")
        if not math.isfinite(float(self.max_energy)) or float(self.max_energy) <= 0.0:
            raise TensorWorldContractError("C4 mate max_energy must be finite and positive")


def _energy(bundle: TensorWorldStateBundleV4, world: int, slot: int) -> float:
    value = float(bundle.base.agent.policy_energy_observation[world, slot])
    return value if math.isfinite(value) else float(
        bundle.base.base.core.agents.energy[world, slot]
    )


def plan_c4_mate_births_oracle_v1(
    bundle: TensorWorldStateBundleV4,
    config: TensorWorldC4MateBirthConfigV1,
    *,
    rng_context: TensorWorldC5ReplayContextV1,
) -> None:
    """Plan source-current adjacent-mate births with explicit PCG64 streams.

    The output is read-only until a future genome/recombination commit binds
    the accepted rows.  No host RNG or implicit random source is permitted.
    """

    bundle.validate()
    if bundle.device.type != "cpu":
        raise TensorWorldContractError("C4 mate planner requires a CPU fixture")
    rng_context.validate_pcg64_for_bundle(bundle)
    streams = rng_context.rng_state_words.shape[1]
    if config.runner_stream >= streams or config.environment_stream >= streams:
        raise TensorWorldContractError("C4 mate RNG stream is outside C5 context")
    p = bundle.population
    core = bundle.base.base.core
    c3 = bundle.base
    max_births = bundle.capacity if config.max_births is None else config.max_births
    if max_births > bundle.capacity:
        raise TensorWorldContractError("C4 mate max_births exceeds capacity")
    fertility_config = TensorWorldC4BirthConfigV1(
        max_births=max_births,
        life_history_mode=config.life_history_mode,
        maturity_start_ticks=config.maturity_start_ticks,
        maturity_rise_ticks=config.maturity_rise_ticks,
        fertility_decline_start_ticks=config.fertility_decline_start_ticks,
        fertility_decline_ticks=config.fertility_decline_ticks,
        elder_fertility_floor=config.elder_fertility_floor,
    )

    p.parent_slots.fill_(-1)
    p.mate_slots.fill_(-1)
    p.child_slots.fill_(-1)
    p.child_pos_y.fill_(-1)
    p.child_pos_x.fill_(-1)
    p.child_energy.zero_()
    p.planned_count.zero_()
    p.decision_parent_slots.fill_(-1)
    p.decision_mate_slots.fill_(-1)
    p.decision_adjacent_counts.zero_()
    p.decision_reason_codes.fill_(-1)
    p.decision_count.zero_()
    p.planned_repro_cooldown.copy_(p.repro_cooldown)
    p.spawn_rng_draws_added.zero_()
    p.spawn_rng_semantic_draws_added.zero_()
    p.runner_rng_draws_added.zero_()
    p.runner_rng_semantic_draws_added.zero_()

    for world in range(bundle.batch):
        rows = sorted(
            (
                int(core.agents.agent_id[world, slot]),
                slot,
            )
            for slot in range(bundle.capacity)
            if bool(core.agents.alive[world, slot])
            and int(core.agents.agent_id[world, slot]) >= 0
        )
        queue_count = int(p.free_queue_count[world])
        queue = [int(p.free_queue[world, i]) for i in range(queue_count)]
        queue_cursor = 0
        planned_positions: list[tuple[int, int]] = []
        planned_slots: set[int] = set()
        mated_slots: set[int] = set()
        planned = 0
        processed = 0
        runner_cursor_start = int(rng_context.rng_cursors[world, config.runner_stream])
        runner_semantic_start = int(
            rng_context.semantic_draw_counters[world, config.runner_stream]
        )
        environment_cursor_start = int(
            rng_context.rng_cursors[world, config.environment_stream]
        )
        environment_semantic_start = int(
            rng_context.semantic_draw_counters[world, config.environment_stream]
        )
        runner_semantic_expected = 0
        environment_semantic_expected = 0

        for decision, (_agent_id, parent_slot) in enumerate(rows):
            if planned >= max_births:
                break
            processed += 1
            p.decision_parent_slots[world, decision] = parent_slot
            reason = C4_BIRTH_REASON_ACCEPTED_V1
            fertility = _c4_fertility_scale_v1(
                int(core.agents.survival_steps[world, parent_slot]),
                fertility_config,
            )
            if fertility <= 0.0:
                reason = C4_BIRTH_REASON_NO_MATE_V1
            elif int(p.planned_repro_cooldown[world, parent_slot]) > 0:
                p.planned_repro_cooldown[world, parent_slot] -= 1
                if int(p.planned_repro_cooldown[world, parent_slot]) != 0:
                    reason = C4_BIRTH_REASON_COOLDOWN_V1
            if reason == C4_BIRTH_REASON_ACCEPTED_V1 and parent_slot in mated_slots:
                reason = C4_BIRTH_REASON_ALREADY_MATED_V1
            parent_energy = _energy(bundle, world, parent_slot)
            if reason == C4_BIRTH_REASON_ACCEPTED_V1 and float(core.agents.health[world, parent_slot]) <= 0:
                reason = C4_BIRTH_REASON_HEALTH_V1
            if reason == C4_BIRTH_REASON_ACCEPTED_V1 and parent_energy < float(
                c3.agent.repro_threshold[world, parent_slot]
            ) / fertility:
                reason = C4_BIRTH_REASON_NO_ENERGY_V1

            mate_slot = -1
            adjacent_ids: list[int] = []
            if reason == C4_BIRTH_REASON_ACCEPTED_V1:
                y = int(core.agents.pos_y[world, parent_slot])
                x = int(core.agents.pos_x[world, parent_slot])
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < bundle.grid_size and 0 <= nx < bundle.grid_size:
                        candidate = int(core.occupancy.agent_id_grid[world, ny, nx])
                        if candidate >= 0 and candidate != int(_agent_id):
                            adjacent_ids.append(candidate)
                p.decision_adjacent_counts[world, decision] = len(adjacent_ids)
                runner_semantic_expected += len(adjacent_ids)
                for index in range(len(adjacent_ids) - 1, 0, -1):
                    selected = c5_pcg64_bounded_uint32_v1(
                        rng_context,
                        bundle,
                        world=world,
                        stream=config.runner_stream,
                        max_value=index,
                    )
                    adjacent_ids[index], adjacent_ids[selected] = (
                        adjacent_ids[selected], adjacent_ids[index]
                    )
                for candidate_id in adjacent_ids:
                    candidate_slot = next(
                        (
                            slot
                            for slot in range(bundle.capacity)
                            if bool(core.agents.alive[world, slot])
                            and int(core.agents.agent_id[world, slot]) == candidate_id
                        ),
                        -1,
                    )
                    if candidate_slot < 0 or candidate_slot in mated_slots:
                        continue
                    if int(p.planned_repro_cooldown[world, candidate_slot]) != 0:
                        continue
                    if float(core.agents.health[world, candidate_slot]) <= 0.0:
                        continue
                    mate_energy = _energy(bundle, world, candidate_slot)
                    if mate_energy < float(c3.agent.repro_threshold[world, candidate_slot]) / fertility:
                        continue
                    mate_slot = candidate_slot
                    break
                if mate_slot < 0:
                    reason = C4_BIRTH_REASON_NO_MATE_V1
            p.decision_mate_slots[world, decision] = mate_slot

            selected_position: tuple[int, int] | None = None
            if reason == C4_BIRTH_REASON_ACCEPTED_V1:
                raw_positions: list[tuple[int, int]] = []
                for source_slot in (parent_slot, mate_slot):
                    sy = int(core.agents.pos_y[world, source_slot])
                    sx = int(core.agents.pos_x[world, source_slot])
                    raw_positions.extend(
                        (sy + dy, sx + dx)
                        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))
                    )
                environment_semantic_expected += len(raw_positions)
                for index in range(len(raw_positions) - 1, 0, -1):
                    selected = c5_pcg64_bounded_uint32_v1(
                        rng_context,
                        bundle,
                        world=world,
                        stream=config.environment_stream,
                        max_value=index,
                    )
                    raw_positions[index], raw_positions[selected] = (
                        raw_positions[selected], raw_positions[index]
                    )
                for y, x in raw_positions:
                    if not (0 <= y < bundle.grid_size and 0 <= x < bundle.grid_size):
                        continue
                    if int(core.occupancy.agent_id_grid[world, y, x]) != -1:
                        continue
                    if (y, x) in planned_positions:
                        continue
                    selected_position = (y, x)
                    break
                if selected_position is None:
                    reason = C4_BIRTH_REASON_NO_SPACE_V1

            child_slot = -1
            parent_share = parent_energy * config.parent_energy_share
            mate_energy = _energy(bundle, world, mate_slot) if mate_slot >= 0 else 0.0
            mate_share = mate_energy * config.parent_energy_share
            if reason == C4_BIRTH_REASON_ACCEPTED_V1 and (
                parent_energy - parent_share < config.minimum_energy_buffer
                or mate_energy - mate_share < config.minimum_energy_buffer
            ):
                reason = C4_BIRTH_REASON_BUFFER_V1
            if reason == C4_BIRTH_REASON_ACCEPTED_V1:
                while queue_cursor < queue_count:
                    candidate = queue[queue_cursor]
                    queue_cursor += 1
                    if (
                        0 <= candidate < bundle.capacity
                        and not bool(core.agents.alive[world, candidate])
                        and int(core.agents.agent_id[world, candidate]) < 0
                        and candidate not in planned_slots
                    ):
                        child_slot = candidate
                        break
                if child_slot < 0:
                    reason = C4_BIRTH_REASON_NO_FREE_SLOT_V1
            if reason == C4_BIRTH_REASON_ACCEPTED_V1:
                assert selected_position is not None
                p.parent_slots[world, planned] = parent_slot
                p.mate_slots[world, planned] = mate_slot
                p.child_slots[world, planned] = child_slot
                p.child_pos_y[world, planned] = selected_position[0]
                p.child_pos_x[world, planned] = selected_position[1]
                p.child_energy[world, planned] = parent_share + mate_share
                planned_positions.append(selected_position)
                planned_slots.add(child_slot)
                mated_slots.update((parent_slot, mate_slot))
                p.planned_repro_cooldown[world, parent_slot] = config.cooldown_steps
                p.planned_repro_cooldown[world, mate_slot] = config.cooldown_steps
                planned += 1
            p.decision_reason_codes[world, decision] = reason
            if reason == C4_BIRTH_REASON_NO_FREE_SLOT_V1:
                break

        p.planned_count[world] = planned
        p.decision_count[world] = processed
        runner_delta = int(rng_context.semantic_draw_counters[world, config.runner_stream]) - runner_semantic_start
        environment_delta = int(rng_context.semantic_draw_counters[world, config.environment_stream]) - environment_semantic_start
        rng_context.semantic_draw_counters[world, config.runner_stream] += max(0, runner_semantic_expected - runner_delta)
        rng_context.semantic_draw_counters[world, config.environment_stream] += max(0, environment_semantic_expected - environment_delta)
        p.runner_rng_draws_added[world] = int(rng_context.rng_cursors[world, config.runner_stream]) - runner_cursor_start
        p.runner_rng_semantic_draws_added[world] = int(rng_context.semantic_draw_counters[world, config.runner_stream]) - runner_semantic_start
        p.spawn_rng_draws_added[world] = int(rng_context.rng_cursors[world, config.environment_stream]) - environment_cursor_start
        p.spawn_rng_semantic_draws_added[world] = int(rng_context.semantic_draw_counters[world, config.environment_stream]) - environment_semantic_start


def commit_c4_mate_births_oracle_v1(
    bundle: TensorWorldStateBundleV4,
    config: TensorWorldC4MateBirthConfigV1,
    *,
    rng_context: TensorWorldC5ReplayContextV1,
) -> None:
    """Commit the accepted mate plan through the shared C4 transaction.

    The adapter intentionally shares the ordinary identity/placement/event
    boundary.  It records the first parent as the primary lineage parent and
    passes the selected mate into scalar fixed-MLP mutation; a future policy
    recombination contract must extend the resident policy rows before this
    route can be considered production-complete.
    """

    bundle.validate()
    if bundle.device.type != "cpu":
        raise TensorWorldContractError("C4 mate commit requires a CPU fixture")
    if not isinstance(rng_context, TensorWorldC5ReplayContextV1):
        raise TensorWorldContractError("C4 mate commit requires its C5 RNG context")
    rng_context.validate_pcg64_for_bundle(bundle)
    if config.runner_stream >= rng_context.rng_state_words.shape[1] or config.environment_stream >= rng_context.rng_state_words.shape[1]:
        raise TensorWorldContractError("C4 mate RNG stream is outside C5 context")
    max_births = bundle.capacity if config.max_births is None else config.max_births
    birth_config = TensorWorldC4BirthConfigV1(
        max_births=max_births,
        parent_energy_share=config.parent_energy_share,
        minimum_energy_buffer=config.minimum_energy_buffer,
        cooldown_steps=config.cooldown_steps,
        start_health=config.start_health,
        max_energy=config.max_energy,
        generation=config.generation,
        allow_mutation=config.allow_mutation,
        require_spawn_rng_tape=False,
    )
    commit_c4_ordinary_births_oracle_v1(
        bundle,
        birth_config,
        replay_context=rng_context,
        _mate_slots_override=bundle.population.mate_slots,
    )


__all__ = [
    "TensorWorldC4MateBirthConfigV1",
    "commit_c4_mate_births_oracle_v1",
    "plan_c4_mate_births_oracle_v1",
]
