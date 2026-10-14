"""Source-current C4 typed lifecycle event publication.

The native event helpers are allocation-free and perform a complete preflight
before changing a cursor.  This CPU oracle keeps those same semantics for the
Tensor World transaction tests.  It is deliberately not presented as the
steady-state CUDA implementation; the resident route must call the native
counterpart before C3/C4 admission.
"""

from __future__ import annotations

from typing import Final

import torch

from .c4_state import TensorWorldStateBundleV4
from .contracts import TensorWorldContractError


C4_EVENT_STATUS_OK_V1: Final[int] = 0
C4_EVENT_STATUS_CAPACITY_EXCEEDED_V1: Final[int] = 1
C4_EVENT_STATUS_INVALID_SEMANTICS_V1: Final[int] = 2
C4_EVENT_BIRTH_COMMITTED_V1: Final[int] = 1
C4_EVENT_MUTATION_V1: Final[int] = 2
C4_EVENT_BIRTH_FAILED_V1: Final[int] = 3
C4_EVENT_DEATH_COMMITTED_V1: Final[int] = 4
C4_EVENT_SENTINEL_ID_V1: Final[int] = -1


def _require_cpu_tape(bundle: TensorWorldStateBundleV4):
    if bundle.device.type != "cpu":
        raise TensorWorldContractError("C4 typed event oracle requires a CPU fixture")
    tape = bundle.events
    if tape is None:
        raise TensorWorldContractError("C4 typed event tape is not attached")
    bundle.validate()
    return tape


def _set_event_row(
    tape,
    world: int,
    row: int,
    *,
    event_type: int,
    generation: int,
    tick: int,
    actor_id: int,
    target_id: int,
    actor_slot: int,
    target_slot: int,
    parent_id: int,
    child_id: int,
    founder_id: int,
    lineage_depth: int,
    subtype: int,
    reason: int,
    success: int,
    value: float,
    sequence: int,
) -> None:
    tape.event_type[world, row] = event_type
    tape.generation[world, row] = generation
    tape.tick[world, row] = tick
    tape.actor_id[world, row] = actor_id
    tape.target_id[world, row] = target_id
    tape.value[world, row] = value
    tape.typed_event_type[world, row] = event_type
    tape.typed_generation[world, row] = generation
    tape.typed_tick[world, row] = tick
    tape.typed_sequence[world, row] = sequence
    tape.typed_actor_id[world, row] = actor_id
    tape.typed_target_id[world, row] = target_id
    tape.typed_actor_slot[world, row] = actor_slot
    tape.typed_target_slot[world, row] = target_slot
    tape.typed_parent_id[world, row] = parent_id
    tape.typed_child_id[world, row] = child_id
    tape.typed_founder_id[world, row] = founder_id
    tape.typed_lineage_depth[world, row] = lineage_depth
    tape.typed_subtype[world, row] = subtype
    tape.typed_reason[world, row] = reason
    tape.typed_success[world, row] = success
    tape.typed_value[world, row] = value


def _sequence_bases(tape, event_sequence: torch.Tensor | None) -> list[int]:
    if event_sequence is None:
        return [int(value) for value in tape.active_count.tolist()]
    if (
        not isinstance(event_sequence, torch.Tensor)
        or event_sequence.dtype != torch.int64
        or tuple(event_sequence.shape) != tuple(tape.active_count.shape)
        or event_sequence.device != tape.device
        or not event_sequence.is_contiguous()
    ):
        raise TensorWorldContractError("C4 event_sequence must be contiguous int64 [batch]")
    bases = [int(value) for value in event_sequence.tolist()]
    active = [int(value) for value in tape.active_count.tolist()]
    if bases != active:
        raise TensorWorldContractError(
            "C4 event_sequence and event tape active_count diverged"
        )
    return bases


def _finish_sequence(tape, event_sequence: torch.Tensor | None) -> None:
    if event_sequence is not None:
        event_sequence.copy_(tape.active_count)


def _preflight_death_world(bundle, tape, world: int, count: int) -> bool:
    status = int(tape.status[world])
    if status in (C4_EVENT_STATUS_CAPACITY_EXCEEDED_V1, C4_EVENT_STATUS_INVALID_SEMANTICS_V1):
        return False
    tape.status[world] = C4_EVENT_STATUS_OK_V1
    p = bundle.population
    if count < 0 or count > bundle.capacity:
        tape.status[world] = C4_EVENT_STATUS_INVALID_SEMANTICS_V1
        return False
    cursor = int(tape.active_count[world])
    if cursor < 0 or cursor > tape.capacity or count > tape.capacity - cursor:
        tape.status[world] = C4_EVENT_STATUS_CAPACITY_EXCEEDED_V1
        return False
    seen_ids: set[int] = set()
    seen_slots: set[int] = set()
    for row in range(count):
        agent_id = int(p.death_agent_ids[world, row])
        slot = int(p.death_slots[world, row])
        reason = int(p.death_reason_codes[world, row])
        if (
            agent_id < 0
            or slot < 0
            or slot >= bundle.capacity
            or reason not in (1, 2)
            or int(p.lineage_founder_id[world, slot]) < 0
            or int(p.lineage_depth[world, slot]) < 0
            or agent_id in seen_ids
            or slot in seen_slots
        ):
            tape.status[world] = C4_EVENT_STATUS_INVALID_SEMANTICS_V1
            return False
        seen_ids.add(agent_id)
        seen_slots.add(slot)
    return True


def append_c4_death_events_v1(
    bundle: TensorWorldStateBundleV4,
    *,
    generation: int,
    tick: int,
    event_sequence: torch.Tensor | None = None,
) -> None:
    """Append the source-current death class in deterministic row order."""

    tape = _require_cpu_tape(bundle)
    if type(generation) is not int or generation < 0 or type(tick) is not int or tick < 0:
        raise TensorWorldContractError("C4 event generation/tick must be nonnegative ints")
    bases = _sequence_bases(tape, event_sequence)
    p = bundle.population
    for world in range(bundle.batch):
        count = int(p.death_count[world])
        if not _preflight_death_world(bundle, tape, world, count):
            continue
        cursor = int(tape.active_count[world])
        for row in range(count):
            slot = int(p.death_slots[world, row])
            reason = int(p.death_reason_codes[world, row])
            agent_id = int(p.death_agent_ids[world, row])
            value = float(reason)
            _set_event_row(
                tape,
                world,
                cursor + row,
                event_type=C4_EVENT_DEATH_COMMITTED_V1,
                generation=generation,
                tick=tick,
                actor_id=agent_id,
                target_id=C4_EVENT_SENTINEL_ID_V1,
                actor_slot=slot,
                target_slot=C4_EVENT_SENTINEL_ID_V1,
                parent_id=C4_EVENT_SENTINEL_ID_V1,
                child_id=C4_EVENT_SENTINEL_ID_V1,
                founder_id=int(p.lineage_founder_id[world, slot]),
                lineage_depth=int(p.lineage_depth[world, slot]),
                subtype=C4_EVENT_DEATH_COMMITTED_V1,
                reason=reason,
                success=1,
                value=value,
                sequence=bases[world] + row,
            )
        tape.active_count[world] = cursor + count
    _finish_sequence(tape, event_sequence)


def preflight_c4_death_event_capacity_v1(bundle: TensorWorldStateBundleV4) -> None:
    """Validate death rows/capacity before clearing resident agent state."""

    tape = _require_cpu_tape(bundle)
    for world in range(bundle.batch):
        if not _preflight_death_world(bundle, tape, world, int(bundle.population.death_count[world])):
            if int(tape.status[world]) == C4_EVENT_STATUS_CAPACITY_EXCEEDED_V1:
                raise TensorWorldContractError("C4 death event tape capacity is exhausted")
            raise TensorWorldContractError("C4 death event semantics are invalid")


def _birth_event_count_for_world(
    bundle, world: int, *, require_committed_identity: bool = True
) -> int | None:
    p = bundle.population
    decisions = int(p.decision_count[world])
    successes = int(p.birth_count[world])
    if decisions < 0 or decisions > bundle.capacity or successes < 0 or successes > bundle.capacity:
        return None
    success_ordinal = 0
    count = 0
    for decision in range(decisions):
        parent_slot = int(p.decision_parent_slots[world, decision])
        reason = int(p.decision_reason_codes[world, decision])
        if parent_slot < 0 or parent_slot >= bundle.capacity:
            return None
        if reason == 0:
            if success_ordinal >= successes:
                return None
            committed_parent = int(p.parent_slots[world, success_ordinal])
            child_slot = int(p.child_slots[world, success_ordinal])
            if committed_parent != parent_slot or child_slot < 0 or child_slot >= bundle.capacity:
                return None
            if require_committed_identity and (
                int(bundle.base.base.core.agents.agent_id[world, committed_parent]) < 0
                or int(bundle.base.base.core.agents.agent_id[world, child_slot]) < 0
            ):
                return None
            count += 2 if bool(p.mutation_occurred[world, success_ordinal]) else 1
            success_ordinal += 1
        elif (1 <= reason <= 3) or (5 <= reason <= 6):
            count += 1
        elif reason == 4 or 7 <= reason <= 11:
            continue
        else:
            return None
    return count if success_ordinal == successes else None


def _preflight_birth_world(bundle, tape, world: int) -> int | None:
    count = _birth_event_count_for_world(bundle, world)
    if count is None:
        tape.status[world] = C4_EVENT_STATUS_INVALID_SEMANTICS_V1
        return None
    cursor = int(tape.active_count[world])
    if cursor < 0 or cursor > tape.capacity or count > tape.capacity - cursor:
        tape.status[world] = C4_EVENT_STATUS_CAPACITY_EXCEEDED_V1
        return None
    tape.status[world] = C4_EVENT_STATUS_OK_V1
    return count


def preflight_c4_birth_event_capacity_v1(bundle: TensorWorldStateBundleV4) -> None:
    """Validate birth row count/capacity before authoritative state mutation."""

    tape = _require_cpu_tape(bundle)
    for world in range(bundle.batch):
        count = _birth_event_count_for_world(
            bundle, world, require_committed_identity=False
        )
        if count is None:
            tape.status[world] = C4_EVENT_STATUS_INVALID_SEMANTICS_V1
            raise TensorWorldContractError("C4 birth event semantics are invalid")
        cursor = int(tape.active_count[world])
        if cursor < 0 or cursor > tape.capacity or count > tape.capacity - cursor:
            tape.status[world] = C4_EVENT_STATUS_CAPACITY_EXCEEDED_V1
            raise TensorWorldContractError("C4 birth event tape capacity is exhausted")
        tape.status[world] = C4_EVENT_STATUS_OK_V1


def append_c4_birth_events_v1(
    bundle: TensorWorldStateBundleV4,
    *,
    generation: int,
    tick: int,
    event_sequence: torch.Tensor | None = None,
) -> None:
    """Append ordinary birth, mutation, and failure classes source-faithfully."""

    tape = _require_cpu_tape(bundle)
    if type(generation) is not int or generation < 0 or type(tick) is not int or tick < 0:
        raise TensorWorldContractError("C4 event generation/tick must be nonnegative ints")
    bases = _sequence_bases(tape, event_sequence)
    p = bundle.population
    for world in range(bundle.batch):
        if _preflight_birth_world(bundle, tape, world) is None:
            continue
        cursor = int(tape.active_count[world])
        output_row = cursor
        success_ordinal = 0
        for decision in range(int(p.decision_count[world])):
            parent_slot = int(p.decision_parent_slots[world, decision])
            parent_id = int(bundle.base.base.core.agents.agent_id[world, parent_slot])
            reason = int(p.decision_reason_codes[world, decision])
            founder = int(p.lineage_founder_id[world, parent_slot])
            if founder < 0:
                founder = parent_id
            depth = int(p.lineage_depth[world, parent_slot])
            if (1 <= reason <= 3) or (5 <= reason <= 6):
                _set_event_row(
                    tape, world, output_row,
                    event_type=C4_EVENT_BIRTH_FAILED_V1,
                    generation=generation, tick=tick, actor_id=parent_id,
                    target_id=C4_EVENT_SENTINEL_ID_V1, actor_slot=parent_slot,
                    target_slot=C4_EVENT_SENTINEL_ID_V1,
                    parent_id=C4_EVENT_SENTINEL_ID_V1,
                    child_id=C4_EVENT_SENTINEL_ID_V1, founder_id=founder,
                    lineage_depth=depth, subtype=C4_EVENT_BIRTH_FAILED_V1,
                    reason=reason, success=0, value=float(reason),
                    sequence=bases[world] + output_row - cursor,
                )
                output_row += 1
                continue
            if reason != 0:
                continue
            parent_id = int(p.birth_parent_ids[world, success_ordinal])
            child_id = int(p.birth_child_ids[world, success_ordinal])
            parent_slot = int(p.parent_slots[world, success_ordinal])
            child_slot = int(p.child_slots[world, success_ordinal])
            founder = int(p.lineage_founder_id[world, child_slot])
            depth = int(p.lineage_depth[world, child_slot])
            if bool(p.mutation_occurred[world, success_ordinal]):
                _set_event_row(
                    tape, world, output_row,
                    event_type=C4_EVENT_MUTATION_V1,
                    generation=generation, tick=tick, actor_id=child_id,
                    target_id=child_id, actor_slot=child_slot,
                    target_slot=child_slot, parent_id=parent_id,
                    child_id=child_id, founder_id=founder, lineage_depth=depth,
                    subtype=C4_EVENT_MUTATION_V1, reason=-1, success=1,
                    value=float(p.mutation_delta_total[world, success_ordinal]),
                    sequence=bases[world] + output_row - cursor,
                )
                output_row += 1
            child_energy = float(p.child_energy[world, success_ordinal])
            _set_event_row(
                tape, world, output_row,
                event_type=C4_EVENT_BIRTH_COMMITTED_V1,
                generation=generation, tick=tick, actor_id=parent_id,
                target_id=child_id, actor_slot=parent_slot,
                target_slot=child_slot, parent_id=parent_id, child_id=child_id,
                founder_id=founder, lineage_depth=depth,
                subtype=C4_EVENT_BIRTH_COMMITTED_V1, reason=-1, success=1,
                value=child_energy,
                sequence=bases[world] + output_row - cursor,
            )
            output_row += 1
            success_ordinal += 1
        tape.active_count[world] = output_row
    _finish_sequence(tape, event_sequence)


__all__ = [
    "C4_EVENT_BIRTH_COMMITTED_V1",
    "C4_EVENT_BIRTH_FAILED_V1",
    "C4_EVENT_DEATH_COMMITTED_V1",
    "C4_EVENT_MUTATION_V1",
    "C4_EVENT_SENTINEL_ID_V1",
    "C4_EVENT_STATUS_CAPACITY_EXCEEDED_V1",
    "C4_EVENT_STATUS_INVALID_SEMANTICS_V1",
    "C4_EVENT_STATUS_OK_V1",
    "append_c4_birth_events_v1",
    "append_c4_death_events_v1",
    "preflight_c4_death_event_capacity_v1",
    "preflight_c4_birth_event_capacity_v1",
]
