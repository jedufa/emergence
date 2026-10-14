"""CPU source transcription for the C4 population transaction boundary.

The death path and ordinary fixed-MLP births are explicit CPU oracle seams.
Birth planning still requires a caller-owned spawn tape or C5 context; scalar
genome mutation additionally requires the attached source-shaped CPython RNG
state.  These helpers do not change C3 admission or claim resident CUDA/C5
qualification.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import Final, TYPE_CHECKING

import torch

from .c4_state import TensorWorldStateBundleV4
from .contracts import TensorWorldContractError
from .c4_events import (
    append_c4_birth_events_v1,
    append_c4_death_events_v1,
    preflight_c4_birth_event_capacity_v1,
    preflight_c4_death_event_capacity_v1,
)
from .c4_genome import (
    apply_c4_scalar_genome_mutations_v1,
    preflight_c4_scalar_genome_mutations_v1,
)

if TYPE_CHECKING:
    from .c5_determinism import TensorWorldC5ReplayContextV1


_INT64_MAX_V1: Final[int] = (1 << 63) - 1


C4_BIRTH_REASON_ACCEPTED_V1: Final[int] = 0
C4_BIRTH_REASON_NO_SPACE_V1: Final[int] = 1
C4_BIRTH_REASON_BUFFER_V1: Final[int] = 2
C4_BIRTH_REASON_NO_FREE_SLOT_V1: Final[int] = 3
C4_BIRTH_REASON_OPPORTUNITY_V1: Final[int] = 4
C4_BIRTH_REASON_NO_MATE_V1: Final[int] = 4
C4_BIRTH_REASON_CONCEPTION_V1: Final[int] = 5
C4_BIRTH_REASON_GESTATION_V1: Final[int] = 6
C4_BIRTH_REASON_HEALTH_V1: Final[int] = 7
C4_BIRTH_REASON_COOLDOWN_V1: Final[int] = 8
C4_BIRTH_REASON_NO_ENERGY_V1: Final[int] = 9
C4_BIRTH_REASON_ALREADY_MATED_V1: Final[int] = 10
C4_BIRTH_REASON_TRY_PROBABILITY_V1: Final[int] = 11


@dataclass(frozen=True, slots=True)
class TensorWorldC4PopulationConfigV1:
    release_free_slots: bool = True
    enable_births: bool = False
    agent_id_capacity: int | None = None

    def __post_init__(self) -> None:
        if type(self.release_free_slots) is not bool:
            raise TensorWorldContractError("C4 release_free_slots must be bool")
        if type(self.enable_births) is not bool:
            raise TensorWorldContractError("C4 enable_births must be bool")
        if self.enable_births:
            raise TensorWorldContractError(
                "C4 birth planning is not enabled in the bounded death transaction oracle"
            )
        if self.agent_id_capacity is not None and (
            type(self.agent_id_capacity) is not int or self.agent_id_capacity <= 0
        ):
            raise TensorWorldContractError("C4 agent_id_capacity must be positive int")


@dataclass(frozen=True, slots=True)
class TensorWorldC4BirthConfigV1:
    """Explicit ordinary-birth oracle configuration.

    This is intentionally not part of :class:`TensorWorldC4PopulationConfigV1`:
    death settlement remains the only default population operation.  Birth
    mutation is explicit and requires the bundle's source-shaped CPython RNG
    state; policy recombination, mating, and resident CUDA execution remain
    separate contracts.
    """

    max_births: int | None = None
    parent_energy_share: float = 0.5
    minimum_energy_buffer: float = 0.0
    cooldown_steps: int = 0
    start_health: float = 12.0
    max_energy: float = 80.0
    life_history_mode: int = 0
    maturity_start_ticks: int = 0
    maturity_rise_ticks: int = 0
    fertility_decline_start_ticks: int = 0
    fertility_decline_ticks: int = 0
    elder_fertility_floor: float = 1.0
    generation: int = 0
    allow_mutation: bool = False
    require_spawn_rng_tape: bool = True

    def __post_init__(self) -> None:
        if self.max_births is not None and (
            type(self.max_births) is not int or self.max_births < 0
        ):
            raise TensorWorldContractError("C4 max_births must be a nonnegative int")
        numeric = (
            ("parent_energy_share", self.parent_energy_share, 0.0, 1.0),
            ("minimum_energy_buffer", self.minimum_energy_buffer, 0.0, math.inf),
            ("start_health", self.start_health, 0.0, math.inf),
            ("max_energy", self.max_energy, 0.0, math.inf),
            ("elder_fertility_floor", self.elder_fertility_floor, 0.0, 1.0),
        )
        for name, value, lower, upper in numeric:
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise TensorWorldContractError(f"C4 {name} must be finite")
            if float(value) < lower or float(value) > upper:
                raise TensorWorldContractError(f"C4 {name} is outside its source range")
        integer_fields = (
            "cooldown_steps",
            "life_history_mode",
            "maturity_start_ticks",
            "maturity_rise_ticks",
            "fertility_decline_start_ticks",
            "fertility_decline_ticks",
            "generation",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise TensorWorldContractError(f"C4 {name} must be a nonnegative int")
        if self.life_history_mode not in (0, 1):
            raise TensorWorldContractError("C4 life_history_mode must be 0 or 1")
        if type(self.allow_mutation) is not bool:
            raise TensorWorldContractError("C4 allow_mutation must be bool")
        if type(self.require_spawn_rng_tape) is not bool:
            raise TensorWorldContractError("C4 require_spawn_rng_tape must be bool")


def _reset_death_outputs(bundle: TensorWorldStateBundleV4) -> None:
    p = bundle.population
    p.death_agent_ids.fill_(-1)
    p.death_slots.fill_(-1)
    p.death_reason_codes.zero_()
    p.death_count.zero_()
    p.survivor_agent_ids.fill_(-1)
    p.birth_parent_ids.fill_(-1)
    p.birth_child_ids.fill_(-1)
    p.birth_count.zero_()
    p.mutation_occurred.zero_()
    p.mutation_delta_total.zero_()


def _rebuild_active_projection(bundle: TensorWorldStateBundleV4, world: int) -> None:
    core = bundle.base.base.core
    p = bundle.population
    rows = [
        (int(core.agents.agent_id[world, slot]), slot)
        for slot in range(bundle.capacity)
        if bool(core.agents.alive[world, slot])
        and int(core.agents.agent_id[world, slot]) >= 0
    ]
    rows.sort(key=lambda item: (item[0], item[1]))
    p.active_agent_ids[world].fill_(-1)
    p.active_agent_slots[world].fill_(-1)
    p.survivor_agent_ids[world].fill_(-1)
    for row, (agent_id, slot) in enumerate(rows):
        p.active_agent_ids[world, row] = agent_id
        p.active_agent_slots[world, row] = slot
        p.survivor_agent_ids[world, row] = agent_id
    p.active_count[world] = len(rows)


def release_c4_free_slots_oracle_v1(bundle: TensorWorldStateBundleV4) -> None:
    """Append pending death slots to the world-local free-slot queue."""

    bundle.validate()
    if bundle.device.type != "cpu":
        raise TensorWorldContractError("C4 scalar population oracle requires a CPU fixture")
    core = bundle.base.base.core
    p = bundle.population
    for world in range(bundle.batch):
        queue_count = int(p.free_queue_count[world])
        pending_count = int(p.pending_free_count[world])
        if queue_count < 0 or pending_count < 0 or queue_count + pending_count > bundle.capacity:
            raise TensorWorldContractError("C4 free-slot queue capacity is invalid")
        existing = {int(p.free_queue[world, index]) for index in range(queue_count)}
        for index in range(pending_count):
            slot = int(p.pending_free_slots[world, index])
            if slot < 0 or slot >= bundle.capacity:
                raise TensorWorldContractError("C4 pending free slot is outside capacity")
            if bool(core.agents.alive[world, slot]) or int(core.agents.agent_id[world, slot]) >= 0:
                raise TensorWorldContractError("C4 pending free slot is still occupied")
            if slot in existing:
                raise TensorWorldContractError("C4 free-slot queue contains a duplicate")
            p.free_queue[world, queue_count + index] = slot
            existing.add(slot)
        p.free_queue_count[world] = queue_count + pending_count
        p.pending_free_count[world] = 0


def apply_c4_death_settlement_oracle_v1(
    bundle: TensorWorldStateBundleV4,
    config: TensorWorldC4PopulationConfigV1 | None = None,
    *,
    event_sequence: torch.Tensor | None = None,
) -> None:
    """Settle C3 death candidates without planning births.

    Candidate order is the source order: ascending biological agent ID, with
    slot index as the deterministic tie-break.  Slot release is a separate
    queue append so a later C4 birth transaction cannot observe a half-cleared
    death row.
    """

    bundle.validate()
    if bundle.device.type != "cpu":
        raise TensorWorldContractError("C4 scalar population oracle requires a CPU fixture")
    if event_sequence is not None and (
        not isinstance(event_sequence, torch.Tensor)
        or event_sequence.dtype != torch.int64
        or tuple(event_sequence.shape) != (bundle.batch,)
        or not event_sequence.is_contiguous()
        or event_sequence.device != bundle.device
    ):
        raise TensorWorldContractError(
            "C4 death event_sequence requires contiguous int64 [batch] CPU state"
        )
    config = config or TensorWorldC4PopulationConfigV1()
    core = bundle.base.base.core
    c3 = bundle.base
    p = bundle.population
    _reset_death_outputs(bundle)
    id_capacity = config.agent_id_capacity or bundle.capacity

    for world in range(bundle.batch):
        pending_before = int(p.pending_free_count[world])
        if pending_before < 0 or pending_before > bundle.capacity:
            raise TensorWorldContractError("C4 pending free count is invalid")
        candidates = [
            (int(core.agents.agent_id[world, slot]), slot)
            for slot in range(bundle.capacity)
            if bool(core.agents.alive[world, slot])
            and int(core.agents.agent_id[world, slot]) >= 0
            and bool(c3.lifecycle.death_candidate[world, slot])
        ]
        candidates.sort(key=lambda item: (item[0], item[1]))
        if pending_before + len(candidates) > bundle.capacity:
            raise TensorWorldContractError("C4 pending death capacity exceeded")

        # Populate the immutable source-order decision rows before touching
        # any resident identity/alive/occupancy fields.  The event publisher
        # uses this exact list and can therefore preflight capacity first.
        for death_index, (agent_id, slot) in enumerate(candidates):
            reason = 2 if float(core.agents.health[world, slot]) <= 0.0 else 1
            p.death_agent_ids[world, death_index] = agent_id
            p.death_slots[world, death_index] = slot
            p.death_reason_codes[world, death_index] = reason
            p.pending_free_agent_ids[world, pending_before + death_index] = agent_id
            p.pending_free_slots[world, pending_before + death_index] = slot

        p.death_count[world] = len(candidates)
        if bundle.events is not None:
            preflight_c4_death_event_capacity_v1(bundle)

        for death_index, (agent_id, slot) in enumerate(candidates):
            y = int(core.agents.pos_y[world, slot])
            x = int(core.agents.pos_x[world, slot])
            if 0 <= y < bundle.grid_size and 0 <= x < bundle.grid_size:
                if int(core.occupancy.agent_id_grid[world, y, x]) == agent_id:
                    core.occupancy.agent_id_grid[world, y, x] = -1
                    core.occupancy.slot_grid[world, y, x] = -1
                    c3.base.patch.agent_grid_i32[world, y, x] = -1
            if 0 <= agent_id < id_capacity:
                p.agent_id_to_slot[world, agent_id] = -1

            core.agents.alive[world, slot] = False
            core.agents.agent_id[world, slot] = -1
            core.agents.pos_y[world, slot] = -1
            core.agents.pos_x[world, slot] = -1
            core.agents.energy[world, slot] = 0.0
            core.agents.health[world, slot] = 0.0
            c3.agent.policy_energy_observation[world, slot] = math.nan
            c3.agent.rest_streak[world, slot] = 0
            c3.policy_runtime.action_cache_token[world, slot] = -1
            c3.policy_runtime.action_cache_cursor[world, slot] = -1
            c3.policy_runtime.action_cache_next[world, slot] = -1
            c3.lifecycle.death_candidate[world, slot] = False
            c3.lifecycle.death_reason[world, slot] = 0

        p.pending_free_count[world] = pending_before + len(candidates)
        p.death_count[world] = len(candidates)
        _rebuild_active_projection(bundle, world)

    if config.release_free_slots:
        release_c4_free_slots_oracle_v1(bundle)
    if bundle.events is not None:
        append_c4_death_events_v1(
            bundle,
            generation=0,
            tick=int(bundle.base.tick),
            event_sequence=event_sequence,
        )


def _c4_fertility_scale_v1(
    age_ticks: int, config: TensorWorldC4BirthConfigV1
) -> float:
    """Scalar source transcription of ``life_history_fertility_scale_v1``."""

    if config.life_history_mode == 0:
        return 1.0
    if age_ticks < config.maturity_start_ticks:
        return 0.0
    scale = 1.0
    if config.maturity_rise_ticks > 0:
        scale = min(
            1.0,
            (age_ticks - config.maturity_start_ticks + 1)
            / config.maturity_rise_ticks,
        )
    if (
        config.fertility_decline_start_ticks > 0
        and age_ticks >= config.fertility_decline_start_ticks
    ):
        if config.fertility_decline_ticks <= 0:
            scale = min(scale, config.elder_fertility_floor)
        else:
            decline = min(
                1.0,
                (age_ticks - config.fertility_decline_start_ticks + 1)
                / config.fertility_decline_ticks,
            )
            aged_scale = 1.0 - decline * (1.0 - config.elder_fertility_floor)
            scale = min(scale, aged_scale)
    return max(0.0, min(1.0, scale))


def _validate_c4_spawn_tape(
    bundle: TensorWorldStateBundleV4,
    config: TensorWorldC4BirthConfigV1,
    spawn_uniforms: torch.Tensor | None,
) -> torch.Tensor:
    if spawn_uniforms is None:
        if config.require_spawn_rng_tape:
            raise TensorWorldContractError(
                "C4 ordinary births require an explicit spawn RNG tape"
            )
        return torch.zeros((bundle.batch, bundle.capacity, 3), dtype=torch.float64)
    if not isinstance(spawn_uniforms, torch.Tensor):
        raise TensorWorldContractError("C4 spawn RNG tape must be a tensor")
    expected = (bundle.batch, bundle.capacity, 3)
    if (
        tuple(spawn_uniforms.shape) != expected
        or spawn_uniforms.device.type != "cpu"
        or not spawn_uniforms.is_floating_point()
        or not spawn_uniforms.is_contiguous()
    ):
        raise TensorWorldContractError(
            f"C4 spawn RNG tape requires CPU contiguous floating shape={expected!r}"
        )
    if not bool(torch.isfinite(spawn_uniforms).all().item()) or bool(
        (spawn_uniforms < 0.0).any().item()
    ) or bool((spawn_uniforms > 1.0).any().item()):
        raise TensorWorldContractError("C4 spawn RNG tape values must lie in [0, 1]")
    return spawn_uniforms.to(dtype=torch.float64)


def plan_c4_ordinary_births_oracle_v1(
    bundle: TensorWorldStateBundleV4,
    config: TensorWorldC4BirthConfigV1,
    *,
    spawn_uniforms: torch.Tensor | None = None,
    rng_context: "TensorWorldC5ReplayContextV1 | None" = None,
    rng_stream: int = 0,
) -> None:
    """Plan ordinary fixed-MLP births using source candidate ordering.

    The plan is a CPU oracle boundary. It does not mutate occupancy, identity,
    energy, or the free queue; commit is a separate call. The supplied tape is
    intentionally not a C5 RNG implementation, so this helper cannot confer a
    production or qualification decision.
    """

    bundle.validate()
    if bundle.device.type != "cpu":
        raise TensorWorldContractError(
            "C4 scalar population oracle requires a CPU fixture"
        )
    if rng_context is not None:
        from .c5_determinism import TensorWorldC5ReplayContextV1

        if not isinstance(rng_context, TensorWorldC5ReplayContextV1):
            raise TensorWorldContractError("C4 rng_context has an unsupported type")
        rng_context.validate_pcg64_for_bundle(bundle)
        if type(rng_stream) is not int or not (
            0 <= rng_stream < rng_context.rng_state_words.shape[1]
        ):
            raise TensorWorldContractError("C4 rng_stream is outside the C5 context")
        if spawn_uniforms is not None:
            raise TensorWorldContractError(
                "C4 cannot combine an explicit RNG tape with a C5 RNG context"
            )
        tape = None
    else:
        tape = _validate_c4_spawn_tape(bundle, config, spawn_uniforms)
    p = bundle.population
    core = bundle.base.base.core
    c3 = bundle.base
    max_births = bundle.capacity if config.max_births is None else config.max_births
    if max_births > bundle.capacity:
        raise TensorWorldContractError("C4 max_births exceeds slot capacity")
    p.parent_slots.fill_(-1)
    p.mate_slots.fill_(-1)
    p.child_slots.fill_(-1)
    p.child_pos_y.fill_(-1)
    p.child_pos_x.fill_(-1)
    p.child_energy.zero_()
    p.planned_count.zero_()
    p.plan_status.zero_()
    p.decision_parent_slots.fill_(-1)
    p.decision_mate_slots.fill_(-1)
    p.decision_adjacent_counts.zero_()
    p.decision_reason_codes.fill_(-1)
    p.decision_count.zero_()
    p.planned_repro_cooldown.copy_(p.repro_cooldown)
    p.birth_parent_ids.fill_(-1)
    p.birth_child_ids.fill_(-1)
    p.birth_count.zero_()
    p.mutation_occurred.zero_()
    p.mutation_delta_total.zero_()
    p.spawn_rng_draws_added.zero_()
    p.spawn_rng_semantic_draws_added.zero_()
    p.runner_rng_draws_added.zero_()
    p.runner_rng_semantic_draws_added.zero_()

    for world in range(bundle.batch):
        rows = [
            (int(core.agents.agent_id[world, slot]), slot)
            for slot in range(bundle.capacity)
            if bool(core.agents.alive[world, slot])
            and int(core.agents.agent_id[world, slot]) >= 0
        ]
        rows.sort(key=lambda item: (item[0], item[1]))
        queue_count = int(p.free_queue_count[world])
        if queue_count < 0 or queue_count > bundle.capacity:
            raise TensorWorldContractError("C4 free-slot queue count is invalid")
        local_queue = [int(p.free_queue[world, index]) for index in range(queue_count)]
        queue_cursor = 0
        planned_positions: list[tuple[int, int]] = []
        planned_slots: set[int] = set()
        planned = 0
        processed_decisions = 0
        semantic_expected = 0
        tape_draws = 0
        draw_start = (
            int(rng_context.rng_cursors[world, rng_stream])
            if rng_context is not None
            else 0
        )
        semantic_start = (
            int(rng_context.semantic_draw_counters[world, rng_stream])
            if rng_context is not None
            else 0
        )
        for decision, (_agent_id, parent_slot) in enumerate(rows):
            if planned >= max_births:
                break
            p.decision_parent_slots[world, decision] = parent_slot
            processed_decisions += 1
            reason = C4_BIRTH_REASON_ACCEPTED_V1
            parent_health = float(core.agents.health[world, parent_slot])
            parent_energy = float(c3.agent.policy_energy_observation[world, parent_slot])
            if not math.isfinite(parent_energy):
                parent_energy = float(core.agents.energy[world, parent_slot])
            if parent_health <= 0.0:
                reason = C4_BIRTH_REASON_HEALTH_V1
            else:
                fertility = _c4_fertility_scale_v1(
                    int(core.agents.survival_steps[world, parent_slot]), config
                )
                if fertility <= 0.0:
                    reason = C4_BIRTH_REASON_OPPORTUNITY_V1
                elif int(p.planned_repro_cooldown[world, parent_slot]) > 0:
                    p.planned_repro_cooldown[world, parent_slot] -= 1
                if reason == C4_BIRTH_REASON_ACCEPTED_V1 and int(
                    p.planned_repro_cooldown[world, parent_slot]
                ) != 0:
                    reason = C4_BIRTH_REASON_COOLDOWN_V1
                elif reason == C4_BIRTH_REASON_ACCEPTED_V1 and parent_energy < float(
                    c3.agent.repro_threshold[world, parent_slot]
                ) / fertility:
                    reason = C4_BIRTH_REASON_NO_ENERGY_V1

            selected: tuple[int, int] | None = None
            candidates: list[tuple[int, int]] = []
            if reason == C4_BIRTH_REASON_ACCEPTED_V1:
                y = int(core.agents.pos_y[world, parent_slot])
                x = int(core.agents.pos_x[world, parent_slot])
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if not (0 <= ny < bundle.grid_size and 0 <= nx < bundle.grid_size):
                        continue
                    if int(core.occupancy.agent_id_grid[world, ny, nx]) != -1:
                        continue
                    candidates.append((ny, nx))
                semantic_expected += len(candidates)
                if len(candidates) > 1:
                    for index in range(len(candidates) - 1, 0, -1):
                        if rng_context is not None:
                            from .c5_determinism import c5_pcg64_bounded_uint32_v1

                            selected_index = c5_pcg64_bounded_uint32_v1(
                                rng_context,
                                bundle,
                                world=world,
                                stream=rng_stream,
                                max_value=index,
                            )
                        else:
                            draw = float(
                                tape[world, decision, len(candidates) - 1 - index]
                            )
                            selected_index = min(
                                index, int(math.floor(draw * (index + 1)))
                            )
                        tape_draws += 1
                        candidates[index], candidates[selected_index] = (
                            candidates[selected_index],
                            candidates[index],
                        )
                if candidates:
                    selected = candidates[0]
                    if selected in planned_positions:
                        reason = C4_BIRTH_REASON_NO_SPACE_V1
                else:
                    reason = C4_BIRTH_REASON_NO_SPACE_V1

            child_slot = -1
            if reason == C4_BIRTH_REASON_ACCEPTED_V1:
                parent_share = parent_energy * config.parent_energy_share
                if parent_energy - parent_share < config.minimum_energy_buffer:
                    reason = C4_BIRTH_REASON_BUFFER_V1
                else:
                    while queue_cursor < queue_count:
                        candidate_slot = local_queue[queue_cursor]
                        queue_cursor += 1
                        if (
                            0 <= candidate_slot < bundle.capacity
                            and not bool(core.agents.alive[world, candidate_slot])
                            and int(core.agents.agent_id[world, candidate_slot]) < 0
                            and candidate_slot not in planned_slots
                        ):
                            child_slot = candidate_slot
                            break
                    if child_slot < 0:
                        reason = C4_BIRTH_REASON_NO_FREE_SLOT_V1
            p.decision_reason_codes[world, decision] = reason
            if reason == C4_BIRTH_REASON_ACCEPTED_V1:
                assert selected is not None
                p.parent_slots[world, planned] = parent_slot
                p.child_slots[world, planned] = child_slot
                p.child_pos_y[world, planned] = selected[0]
                p.child_pos_x[world, planned] = selected[1]
                p.child_energy[world, planned] = parent_energy * config.parent_energy_share
                planned_positions.append(selected)
                planned_slots.add(child_slot)
                planned += 1
            if reason == C4_BIRTH_REASON_NO_FREE_SLOT_V1:
                break
        p.planned_count[world] = planned
        p.decision_count[world] = processed_decisions
        if rng_context is not None:
            semantic_delta = int(
                rng_context.semantic_draw_counters[world, rng_stream]
            ) - semantic_start
            # The source counter charges one semantic draw for every
            # candidate, while Fisher-Yates calls only perform N-1 bounded
            # draws.  Add the zero/one remainder explicitly.
            remainder = max(0, semantic_expected - semantic_delta)
            if remainder:
                rng_context.semantic_draw_counters[world, rng_stream] += remainder
            p.spawn_rng_draws_added[world] = int(
                rng_context.rng_cursors[world, rng_stream]
            ) - draw_start
            p.spawn_rng_semantic_draws_added[world] = int(
                rng_context.semantic_draw_counters[world, rng_stream]
            ) - semantic_start
        else:
            p.spawn_rng_draws_added[world] = tape_draws
            p.spawn_rng_semantic_draws_added[world] = semantic_expected


def _copy_c4_slot_fields(
    destination: object,
    source: object,
    world: int,
    parent: int,
    child: int,
) -> None:
    for field in fields(destination):
        destination_value = getattr(destination, field.name)
        source_value = getattr(source, field.name)
        destination_value[world, child].copy_(source_value[world, parent])


def commit_c4_ordinary_births_oracle_v1(
    bundle: TensorWorldStateBundleV4,
    config: TensorWorldC4BirthConfigV1,
    *,
    replay_context: "TensorWorldC5ReplayContextV1 | None" = None,
    _mate_slots_override: torch.Tensor | None = None,
) -> None:
    """Commit the bounded birth plan and reset newborn state.

    The public ordinary route is asexual.  The private mate override is used
    only by the C4 mate oracle after it has produced an explicit accepted plan;
    it keeps the same identity/queue/event transaction boundary while debiting
    both accepted parents and passing both genomes to the scalar mutation
    oracle.
    """

    bundle.validate()
    if bundle.device.type != "cpu":
        raise TensorWorldContractError(
            "C4 scalar population oracle requires a CPU fixture"
        )
    p = bundle.population
    core = bundle.base.base.core
    c3 = bundle.base
    gaussian_window_bases: torch.Tensor | None = None
    if replay_context is not None and replay_context.gaussian_window is not None:
        replay_context.validate_typed_continuation_for_bundle(bundle)
        gaussian_window_bases = replay_context.gaussian_window.base_raw_draws
    max_births = bundle.capacity if config.max_births is None else config.max_births
    if _mate_slots_override is not None and (
        not isinstance(_mate_slots_override, torch.Tensor)
        or tuple(_mate_slots_override.shape) != tuple(p.mate_slots.shape)
        or _mate_slots_override.dtype != torch.int32
        or _mate_slots_override.device != bundle.device
        or not _mate_slots_override.is_contiguous()
    ):
        raise TensorWorldContractError("C4 mate slot override has an incompatible shape or dtype")
    mutation_plans: list[list[tuple[bool, float]]] = [[] for _ in range(bundle.batch)]

    for world in range(bundle.batch):
        planned = int(p.planned_count[world])
        if planned < 0 or planned > max_births:
            raise TensorWorldContractError("C4 planned birth count is invalid")
        p.birth_count[world] = planned
        if config.allow_mutation and planned:
            parent_slots = [
                int(p.parent_slots[world, birth]) for birth in range(planned)
            ]
            mate_slots = [
                int(
                    parent_slots[birth]
                    if _mate_slots_override is None
                    else _mate_slots_override[world, birth]
                )
                for birth in range(planned)
            ]
            trial, mutation_outputs = preflight_c4_scalar_genome_mutations_v1(
                bundle,
                world=world,
                parent_slots=parent_slots,
                mate_slots=mate_slots,
                gaussian_window_base_raw_draws=(
                    0
                    if gaussian_window_bases is None
                    else int(gaussian_window_bases[world])
                ),
            )
            # The trial is deliberately not installed: event capacity and all
            # identity/energy checks must still be able to fail without
            # advancing the authoritative CPython continuation.
            del trial
            mutation_plans[world] = mutation_outputs
            for birth, (occurred, delta) in enumerate(mutation_outputs):
                p.mutation_occurred[world, birth] = occurred
                p.mutation_delta_total[world, birth] = delta
    if bundle.events is not None:
        preflight_c4_birth_event_capacity_v1(bundle)

    for world in range(bundle.batch):
        planned = int(p.planned_count[world])
        if planned == 0:
            continue
        queue_count = int(p.free_queue_count[world])
        queue = [int(p.free_queue[world, index]) for index in range(queue_count)]
        queue_cursor = 0
        reservations: list[tuple[int, int, int]] = []
        next_id = max(
            int(p.next_agent_id[world]),
            max(
                (int(core.agents.agent_id[world, slot]) for slot in range(bundle.capacity)),
                default=0,
            )
            + 1,
        )
        # Biological IDs are monotonic in the sorted-authority route and are
        # stored as signed int64.  Reject exhaustion before any parent debit,
        # slot mutation, event append, or RNG advancement so the transaction
        # rollback boundary remains exact for both R11 and the CPU oracle.
        if next_id < 0 or next_id > _INT64_MAX_V1 - planned:
            raise TensorWorldContractError("C4 biological identity exhausted")
        # Biological identity is external/monotonic and is deliberately
        # independent of physical slot capacity.  The legacy
        # ``agent_id_to_slot`` tensor is retained only as a bounded
        # compatibility projection; R11 resolves IDs through the sorted
        # active-agent pairs and therefore permits a reused slot to receive an
        # ID greater than ``capacity``.
        parent_ids: list[int] = []
        for birth in range(planned):
            parent_slot = int(p.parent_slots[world, birth])
            child_slot = int(p.child_slots[world, birth])
            mate_slot = (
                parent_slot
                if _mate_slots_override is None
                else int(_mate_slots_override[world, birth])
            )
            if parent_slot < 0 or parent_slot >= bundle.capacity:
                raise TensorWorldContractError("C4 planned parent slot is invalid")
            parent_id = int(core.agents.agent_id[world, parent_slot])
            if not bool(core.agents.alive[world, parent_slot]) or parent_id < 0:
                raise TensorWorldContractError("C4 planned parent is no longer alive")
            if bool(c3.genome.policy_use_gru[world, parent_slot]):
                raise TensorWorldContractError(
                    "C4 ordinary fixed-MLP birth oracle cannot commit a GRU parent"
                )
            if mate_slot < 0 or mate_slot >= bundle.capacity:
                raise TensorWorldContractError("C4 planned mate slot is invalid")
            if not bool(core.agents.alive[world, mate_slot]) or int(
                core.agents.agent_id[world, mate_slot]
            ) < 0:
                raise TensorWorldContractError("C4 planned mate is no longer alive")
            if bool(c3.genome.policy_use_gru[world, mate_slot]):
                raise TensorWorldContractError(
                    "C4 fixed-MLP scalar mutation cannot commit a GRU mate"
                )
            if _mate_slots_override is not None and mate_slot == parent_slot:
                raise TensorWorldContractError("C4 accepted mate must differ from parent")
            if parent_ids and parent_id <= parent_ids[-1]:
                raise TensorWorldContractError(
                    "C4 parents must be ordered by biological ID"
                )
            parent_ids.append(parent_id)
            if int(p.repro_cooldown[world, parent_slot]) != 0:
                raise TensorWorldContractError(
                    "C4 planned parent cooldown changed before commit"
                )
            if mate_slot != parent_slot and int(p.repro_cooldown[world, mate_slot]) != 0:
                raise TensorWorldContractError(
                    "C4 planned mate cooldown changed before commit"
                )
            if child_slot < 0 or child_slot >= bundle.capacity:
                raise TensorWorldContractError("C4 planned child slot is invalid")
            child_y = int(p.child_pos_y[world, birth])
            child_x = int(p.child_pos_x[world, birth])
            if not (0 <= child_y < bundle.grid_size and 0 <= child_x < bundle.grid_size):
                raise TensorWorldContractError("C4 planned child position is invalid")
            if int(core.occupancy.agent_id_grid[world, child_y, child_x]) != -1:
                raise TensorWorldContractError("C4 planned child position is occupied")
            candidate = -1
            while queue_cursor < queue_count:
                candidate = queue[queue_cursor]
                queue_cursor += 1
                if (
                    0 <= candidate < bundle.capacity
                    and not bool(core.agents.alive[world, candidate])
                    and int(core.agents.agent_id[world, candidate]) < 0
                    and all(candidate != prior[2] for prior in reservations)
                ):
                    break
            if candidate != child_slot:
                raise TensorWorldContractError(
                    "C4 planned child does not match free queue order"
                )
            reservations.append((parent_slot, mate_slot, child_slot))

        for birth, (parent_slot, mate_slot, child_slot) in enumerate(reservations):
            parent_id = int(core.agents.agent_id[world, parent_slot])
            parent_energy = float(c3.agent.policy_energy_observation[world, parent_slot])
            if not math.isfinite(parent_energy):
                parent_energy = float(core.agents.energy[world, parent_slot])
            child_energy = float(p.child_energy[world, birth])
            parent_share = parent_energy * config.parent_energy_share
            if mate_slot == parent_slot:
                parent_debit = child_energy
                mate_energy = parent_energy
                mate_debit = 0.0
            else:
                mate_energy = float(c3.agent.policy_energy_observation[world, mate_slot])
                if not math.isfinite(mate_energy):
                    mate_energy = float(core.agents.energy[world, mate_slot])
                mate_debit = mate_energy * config.parent_energy_share
                parent_debit = parent_share
                if not math.isclose(
                    parent_debit + mate_debit,
                    child_energy,
                    rel_tol=0.0,
                    abs_tol=1e-5,
                ):
                    raise TensorWorldContractError(
                        "C4 planned mate child energy does not match parent shares"
                    )
            core.agents.energy[world, parent_slot] = parent_energy - parent_debit
            c3.agent.policy_energy_observation[world, parent_slot] = (
                parent_energy - parent_debit
            )
            if mate_slot != parent_slot:
                core.agents.energy[world, mate_slot] = mate_energy - mate_debit
                c3.agent.policy_energy_observation[world, mate_slot] = (
                    mate_energy - mate_debit
                )
            p.repro_cooldown[world, parent_slot] = config.cooldown_steps
            p.offspring_count[world, parent_slot] += 1
            if int(p.first_repro_step[world, parent_slot]) < 0:
                p.first_repro_step[world, parent_slot] = core.agents.survival_steps[
                    world, parent_slot
                ]
            if mate_slot != parent_slot:
                p.repro_cooldown[world, mate_slot] = config.cooldown_steps
                p.offspring_count[world, mate_slot] += 1
                if int(p.first_repro_step[world, mate_slot]) < 0:
                    p.first_repro_step[world, mate_slot] = core.agents.survival_steps[
                        world, mate_slot
                    ]

            child_id = next_id + birth
            p.birth_parent_ids[world, birth] = parent_id
            p.birth_child_ids[world, birth] = child_id
            p.mutation_occurred[world, birth] = False
            p.mutation_delta_total[world, birth] = 0.0
            p.slot_tokens[world, child_slot] += 1
            core.agents.alive[world, child_slot] = True
            core.agents.agent_id[world, child_slot] = child_id
            child_y = int(p.child_pos_y[world, birth])
            child_x = int(p.child_pos_x[world, birth])
            core.agents.pos_y[world, child_slot] = child_y
            core.agents.pos_x[world, child_slot] = child_x
            core.occupancy.agent_id_grid[world, child_y, child_x] = child_id
            core.occupancy.slot_grid[world, child_y, child_x] = child_slot
            c3.base.patch.agent_grid_i32[world, child_y, child_x] = child_id
            core.agents.energy[world, child_slot] = child_energy
            core.agents.health[world, child_slot] = config.start_health
            core.agents.age[world, child_slot] = 0
            core.agents.survival_steps[world, child_slot] = 0
            core.agents.energy_collected[world, child_slot] = 0.0
            core.transactions.actions[world, child_slot] = -1
            core.transactions.moved[world, child_slot] = False
            core.transactions.gathered[world, child_slot] = False
            core.transactions.food_gained[world, child_slot] = 0.0
            core.transactions.observations[world, child_slot].zero_()
            core.transactions.logits[world, child_slot].zero_()

            c2 = c3.base
            c2.eed.active_y[world, child_slot] = -1
            c2.eed.active_x[world, child_slot] = -1
            c2.eed.active_class[world, child_slot] = 0
            c2.eed.handling_progress[world, child_slot] = 0.0
            c2.eed.handling_axis[world, child_slot] = 0.0
            c2.effects.pre_policy_health_delta[world, child_slot] = 0.0
            c2.effects.post_action_health_delta[world, child_slot] = 0.0
            c2.effects.hazard_flags[world, child_slot] = 0
            c2.world_events.gather_amount[world, child_slot] = 0.0
            c2.world_events.gather_resource_class[world, child_slot] = 0
            c2.transactions.observations[world, child_slot].zero_()
            c2.transactions.logits[world, child_slot].zero_()
            if 0 <= child_id < bundle.capacity:
                p.agent_id_to_slot[world, child_id] = child_slot

            _copy_c4_slot_fields(
                c3.genome, c3.genome, world, parent_slot, child_slot
            )
            if config.allow_mutation:
                actual = apply_c4_scalar_genome_mutations_v1(
                    bundle,
                    world=world,
                    parent_slots=[parent_slot],
                    mate_slots=[mate_slot],
                    child_slots=[child_slot],
                    gaussian_window_base_raw_draws=(
                        0
                        if gaussian_window_bases is None
                        else int(gaussian_window_bases[world])
                    ),
                )[0]
                expected = mutation_plans[world][birth]
                if actual[0] != expected[0] or not math.isclose(
                    actual[1], expected[1], rel_tol=0.0, abs_tol=1e-12
                ):
                    raise TensorWorldContractError(
                        "C4 scalar mutation diverged from its preflight sequence"
                    )
                p.mutation_occurred[world, birth] = actual[0]
                p.mutation_delta_total[world, birth] = actual[1]
            c3.agent.repro_threshold[world, child_slot] = c3.genome.reproduction_threshold[
                world, child_slot
            ]
            c3.agent.injury[world, child_slot] = 0.0
            c3.agent.age[world, child_slot] = 0
            c3.agent.survival_steps[world, child_slot] = 0
            c3.agent.energy_collected[world, child_slot] = 0.0
            c3.agent.energy_collected_exact[world, child_slot] = 0.0
            c3.agent.rest_streak[world, child_slot] = 0
            c3.agent.policy_energy_observation[world, child_slot] = child_energy
            c3.agent.last_action[world, child_slot] = -1
            c3.agent.last_last_action[world, child_slot] = -1
            for name in (
                "action_switches",
                "action_oscillations",
                "episode0_switches",
                "episode0_oscillations",
                "attacks",
                "selfharm_events",
                "episode0_selfharm_events",
            ):
                getattr(c3.agent, name)[world, child_slot] = 0
            c3.agent.gru_upkeep_cost[world, child_slot] = 0.0
            c3.agent.cost_charge_scalar[world, child_slot] = 0.0
            c3.agent.cost_charge_maturation_scalar[world, child_slot] = 0.0
            c3.agent.drive_inertia[world, child_slot] = c3.genome.drive_inertia[
                world, child_slot
            ]
            c3.agent.drive_noise[world, child_slot] = c3.genome.drive_noise[
                world, child_slot
            ]
            c3.agent.drives[world, child_slot, 0] = torch.clamp(
                0.60 * c3.genome.drive_setpoints[world, child_slot, 0]
                + 0.40 * (1.0 - child_energy / config.max_energy),
                0.0,
                1.0,
            )
            c3.agent.drives[world, child_slot, 1:] = torch.clamp(
                c3.genome.drive_setpoints[world, child_slot, 1:], 0.0, 1.0
            )
            c3.agent.prev_drive_delta[world, child_slot].zero_()
            for name in (
                "drive_volatility",
                "drive_jitter",
                "episode0_drive_jitter_sum",
                "last_energy_delta",
                "last_health_delta",
                "last_injury_delta",
            ):
                getattr(c3.agent, name)[world, child_slot] = 0.0

            energy_n = max(0.0, min(1.0, child_energy / config.max_energy))
            c3.adaptation.chemistry[world, child_slot] = torch.tensor(
                [0.40 + 0.40 * energy_n, 0.55, 0.0, 0.0, 0.50, 0.25],
                dtype=torch.float32,
            )
            for name in (
                "pred_err_sum",
                "episode0_pred_err_sum",
                "late_pred_err_sum",
                "chemistry_observation_bias",
            ):
                getattr(c3.adaptation, name)[world, child_slot] = 0.0
            for name in (
                "pred_err_count",
                "episode0_pred_err_count",
                "late_pred_err_count",
            ):
                getattr(c3.adaptation, name)[world, child_slot] = 0
            c3.adaptation.dev_stage_initialized[world, child_slot] = False
            c3.adaptation.chemistry_observation_fidelity[world, child_slot] = 1.0
            c3.policy_runtime.action_cache_actions[world, child_slot].fill_(-1)
            c3.policy_runtime.action_cache_token[world, child_slot] = -1
            c3.policy_runtime.action_cache_cursor[world, child_slot] = -1
            c3.policy_runtime.action_cache_next[world, child_slot] = -1
            c3.lifecycle.death_candidate[world, child_slot] = False
            c3.lifecycle.death_reason[world, child_slot] = 0
            c3.lifecycle.maturity_factor[world, child_slot] = 0.0
            c3.lifecycle.fertility_factor[world, child_slot] = 0.0
            c3.lifecycle.senescence_factor[world, child_slot] = 0.0
            c3.transactions.observations[world, child_slot].zero_()
            c3.transactions.logits[world, child_slot].zero_()
            c3.transactions.row_actions[world, child_slot] = -1
            c3.transactions.actions_by_slot[world, child_slot] = -1
            c3.transactions.pre_energy[world, child_slot] = child_energy
            c3.transactions.pre_health[world, child_slot] = config.start_health
            c3.transactions.pre_injury[world, child_slot] = 0.0
            c3.transactions.pre_survival_steps[world, child_slot] = 0
            c3.transactions.chemistry_noise_samples[world, child_slot].zero_()
            c3.transactions.chemistry_noise_samples_valid[world, child_slot] = False
            c3.transactions.drive_noise_samples[world, child_slot].zero_()
            c3.transactions.drive_noise_samples_valid[world, child_slot] = False
            core.policy.packed_weights[world, child_slot].copy_(
                core.policy.packed_weights[world, parent_slot]
            )
            c2.policy.packed_weights[world, child_slot].copy_(
                c2.policy.packed_weights[world, parent_slot]
            )
            core.policy.drives[world, child_slot].copy_(c3.agent.drives[world, child_slot])
            core.policy.attack_bias[world, child_slot] = c3.genome.aggression_bias[
                world, child_slot
            ]
            core.policy.action_temperature[world, child_slot] = c3.genome.action_temperature[
                world, child_slot
            ]
            core.policy.drive_logit_gain[world, child_slot] = c3.genome.drive_logit_gain[
                world, child_slot
            ]
            core.rng.action_uniforms[world, child_slot].fill_(0.5)

            founder = int(p.lineage_founder_id[world, parent_slot])
            p.lineage_founder_id[world, child_slot] = parent_id if founder < 0 else founder
            p.lineage_depth[world, child_slot] = p.lineage_depth[world, parent_slot] + 1
            p.lineage_birth_generation[world, child_slot] = config.generation
            p.lineage_birth_tick[world, child_slot] = bundle.base.tick
            p.lineage_parent_agent_id[world, child_slot] = parent_id
            parent_patch_lineage = int(p.patch_habitat_lineage_id[world, parent_slot])
            p.patch_habitat_lineage_id[world, child_slot] = (
                parent_patch_lineage if parent_patch_lineage != 0 else parent_id
            )
            p.scar[world, child_slot] = 0.0
            p.repro_cooldown[world, child_slot] = 0
            p.planned_repro_cooldown[world, child_slot] = 0
            p.offspring_count[world, child_slot] = 0
            p.first_repro_step[world, child_slot] = -1

        remaining = [
            slot
            for slot in queue[queue_cursor:]
            if 0 <= slot < bundle.capacity
            and not bool(core.agents.alive[world, slot])
            and int(core.agents.agent_id[world, slot]) < 0
        ]
        p.free_queue[world].fill_(-1)
        if remaining:
            p.free_queue[world, : len(remaining)] = torch.tensor(
                remaining, dtype=torch.int64
            )
        p.free_queue_count[world] = len(remaining)
        p.next_agent_id[world] = next_id + planned
        _rebuild_active_projection(bundle, world)

    if bundle.events is not None:
        append_c4_birth_events_v1(
            bundle,
            generation=config.generation,
            tick=int(bundle.base.tick),
            event_sequence=(
                None if replay_context is None else replay_context.event_sequence
            ),
        )


__all__ = [
    "C4_BIRTH_REASON_ACCEPTED_V1",
    "C4_BIRTH_REASON_BUFFER_V1",
    "C4_BIRTH_REASON_CONCEPTION_V1",
    "C4_BIRTH_REASON_COOLDOWN_V1",
    "C4_BIRTH_REASON_HEALTH_V1",
    "C4_BIRTH_REASON_GESTATION_V1",
    "C4_BIRTH_REASON_ALREADY_MATED_V1",
    "C4_BIRTH_REASON_NO_ENERGY_V1",
    "C4_BIRTH_REASON_NO_FREE_SLOT_V1",
    "C4_BIRTH_REASON_NO_SPACE_V1",
    "C4_BIRTH_REASON_NO_MATE_V1",
    "C4_BIRTH_REASON_TRY_PROBABILITY_V1",
    "TensorWorldC4BirthConfigV1",
    "TensorWorldC4PopulationConfigV1",
    "apply_c4_death_settlement_oracle_v1",
    "commit_c4_ordinary_births_oracle_v1",
    "plan_c4_ordinary_births_oracle_v1",
    "release_c4_free_slots_oracle_v1",
]
