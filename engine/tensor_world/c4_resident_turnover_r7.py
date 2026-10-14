"""CPU-only repeated-turnover preparation for the C4 R7 rung."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Final, Mapping

import torch

from .c4_fixture import make_c4_fixture_v1
from .c4_population import TensorWorldC4BirthConfigV1
from .c4_state import TensorWorldStateBundleV4
from .c4_transaction import (
    C4PopulationTransactionError,
    execute_c4_population_transaction_oracle_v1,
)
from .c5_determinism import (
    TensorWorldC5ReplayContextV1,
    make_c5_pcg64_context_v1,
)
from .contracts import TensorWorldContractError


C4_RESIDENT_TURNOVER_R7_SCHEMA_VERSION_V1: Final[str] = (
    "tensor_world_c4_resident_turnover_r7_preparation_v1"
)
C4_RESIDENT_TURNOVER_R7_DECISION_V1: Final[str] = "NO_GATE_DECISION"
C4_RESIDENT_TURNOVER_R7_IDENTITY_LIMITATION_V1: Final[str] = (
    "capacity_32_preserves_the_current_dense_world_local_identity_map_for_the_"
    "bounded_run; biological_ids_repeat_across_worlds"
)


@dataclass(slots=True)
class C4ResidentTurnoverR7FixtureV1:
    bundle: TensorWorldStateBundleV4
    replay: TensorWorldC5ReplayContextV1
    birth_config: TensorWorldC4BirthConfigV1
    reused_slot: int
    identity_map_limitation: str = C4_RESIDENT_TURNOVER_R7_IDENTITY_LIMITATION_V1
    schema_version: str = C4_RESIDENT_TURNOVER_R7_SCHEMA_VERSION_V1
    decision: str = C4_RESIDENT_TURNOVER_R7_DECISION_V1
    gate_claim: bool = False

    def validate(self) -> None:
        if self.schema_version != C4_RESIDENT_TURNOVER_R7_SCHEMA_VERSION_V1:
            raise TensorWorldContractError("C4 R7 fixture schema is unsupported")
        if self.bundle.batch != 4 or self.bundle.capacity < 32:
            raise TensorWorldContractError("C4 R7 requires B4 and capacity>=32")
        if self.bundle.device.type != "cpu" or not (0 <= self.reused_slot < 4):
            raise TensorWorldContractError("C4 R7 fixture is invalid")
        if self.decision != C4_RESIDENT_TURNOVER_R7_DECISION_V1 or self.gate_claim:
            raise TensorWorldContractError("C4 R7 preparation cannot claim a gate")
        self.bundle.validate()
        self.replay.validate_pcg64_for_bundle(self.bundle)


@dataclass(frozen=True, slots=True)
class C4ResidentTurnoverR7TickReceiptV1:
    tick_index: int
    reused_slots: tuple[int, int, int, int]
    dead_agent_ids: tuple[int, int, int, int]
    child_agent_ids: tuple[int, int, int, int]
    slot_tokens: tuple[int, int, int, int]
    event_active_counts: tuple[int, int, int, int]
    rng_cursors: tuple[int, int, int, int]
    rng_semantic_draws: tuple[int, int, int, int]
    death_founders: tuple[int, int, int, int]
    death_depths: tuple[int, int, int, int]
    child_founders: tuple[int, int, int, int]
    child_depths: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class C4ResidentTurnoverR7RunReceiptV1:
    status: str
    requested_ticks: int
    completed_ticks: int
    ticks: tuple[C4ResidentTurnoverR7TickReceiptV1, ...]
    failure_tick: int | None = None
    failure_receipt: Mapping[str, object] | None = None
    rollback_applied: bool = False
    identity_map_limitation: str = C4_RESIDENT_TURNOVER_R7_IDENTITY_LIMITATION_V1
    schema_version: str = C4_RESIDENT_TURNOVER_R7_SCHEMA_VERSION_V1
    decision: str = C4_RESIDENT_TURNOVER_R7_DECISION_V1
    gate_claim: bool = False


def make_c4_resident_turnover_r7_fixture_v1(
    *, capacity: int = 32, width: int = 8, seed: int = 41
) -> C4ResidentTurnoverR7FixtureV1:
    """Create a dense-map-safe B4 fixture with slot 1 as the only free source."""

    if capacity < 32:
        raise TensorWorldContractError("C4 R7 requires capacity>=32")
    bundle, _ = make_c4_fixture_v1(
        batch=4,
        capacity=capacity,
        active=4,
        width=width,
        seed=seed,
        world_local_identity=True,
    )
    population = bundle.population
    population.free_queue.fill_(-1)
    population.free_queue_count.zero_()
    bundle.base.lifecycle.death_candidate.zero_()
    bundle.base.base.core.agents.energy[:, :4] = 1000.0
    bundle.base.agent.policy_energy_observation[:, :4] = 1000.0
    bundle.base.agent.repro_threshold[:, :4] = 0.0
    replay = make_c5_pcg64_context_v1(bundle, streams=2, seed=seed)
    fixture = C4ResidentTurnoverR7FixtureV1(
        bundle=bundle,
        replay=replay,
        birth_config=TensorWorldC4BirthConfigV1(
            max_births=1,
            parent_energy_share=0.1,
            cooldown_steps=0,
            max_energy=1000.0,
            allow_mutation=False,
            require_spawn_rng_tape=False,
        ),
        reused_slot=1,
    )
    fixture.validate()
    return fixture


def _snapshot_tensor_leaves_v1(
    value: object, output: dict[int, tuple[torch.Tensor, torch.Tensor]]
) -> None:
    if isinstance(value, torch.Tensor):
        output.setdefault(id(value), (value, value.detach().clone()))
    elif is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _snapshot_tensor_leaves_v1(getattr(value, field.name), output)


def _restore_tensor_leaves_v1(
    snapshot: Mapping[int, tuple[torch.Tensor, torch.Tensor]]
) -> None:
    with torch.no_grad():
        for target, saved in snapshot.values():
            target.copy_(saved)


def run_c4_resident_turnover_r7_oracle_v1(
    fixture: C4ResidentTurnoverR7FixtureV1,
    *,
    ticks: int = 5,
    inject_failure_tick: int | None = None,
    inject_failure_world: int = 2,
) -> C4ResidentTurnoverR7RunReceiptV1:
    """Run repeated same-slot death/birth transactions with tick rollback."""

    if not isinstance(fixture, C4ResidentTurnoverR7FixtureV1):
        raise TensorWorldContractError("C4 R7 fixture is required")
    fixture.validate()
    if type(ticks) is not int or ticks < 1:
        raise TensorWorldContractError("C4 R7 ticks must be positive")
    if inject_failure_tick is not None and not (0 <= inject_failure_tick < ticks):
        raise TensorWorldContractError("C4 R7 failure tick is outside the run")
    if not (0 <= inject_failure_world < 4):
        raise TensorWorldContractError("C4 R7 failure world is outside B4")

    bundle = fixture.bundle
    population = bundle.population
    events = bundle.events
    if events is None:
        raise TensorWorldContractError("C4 R7 requires typed events")
    records: list[C4ResidentTurnoverR7TickReceiptV1] = []
    for tick_index in range(ticks):
        tick_snapshot: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        _snapshot_tensor_leaves_v1(bundle, tick_snapshot)
        _snapshot_tensor_leaves_v1(fixture.replay, tick_snapshot)
        slot = fixture.reused_slot
        dead_ids = tuple(
            int(bundle.base.base.core.agents.agent_id[world, slot])
            for world in range(4)
        )
        dead_founders = tuple(
            int(population.lineage_founder_id[world, slot]) for world in range(4)
        )
        dead_depths = tuple(
            int(population.lineage_depth[world, slot]) for world in range(4)
        )
        event_before = tuple(int(value) for value in events.active_count.tolist())
        bundle.base.lifecycle.death_candidate.zero_()
        bundle.base.lifecycle.death_candidate[:, slot] = True
        bundle.base.base.core.agents.health[:, slot] = 0.0
        if inject_failure_tick == tick_index:
            events.active_count[inject_failure_world] = events.capacity - 1
            fixture.replay.event_sequence.copy_(events.active_count)
        try:
            execute_c4_population_transaction_oracle_v1(
                bundle,
                birth_config=replace(
                    fixture.birth_config, generation=tick_index + 1
                ),
                rng_context=fixture.replay,
            )
        except C4PopulationTransactionError as exc:
            _restore_tensor_leaves_v1(tick_snapshot)
            return C4ResidentTurnoverR7RunReceiptV1(
                status="INJECTED_FAILURE_ROLLED_BACK",
                requested_ticks=ticks,
                completed_ticks=len(records),
                ticks=tuple(records),
                failure_tick=tick_index,
                failure_receipt=exc.receipt.as_dict(),
                rollback_applied=True,
            )

        child_ids = tuple(
            int(bundle.base.base.core.agents.agent_id[world, slot])
            for world in range(4)
        )
        child_founders = tuple(
            int(population.lineage_founder_id[world, slot]) for world in range(4)
        )
        child_depths = tuple(
            int(population.lineage_depth[world, slot]) for world in range(4)
        )
        for world in range(4):
            if int(population.child_slots[world, 0]) != slot:
                raise TensorWorldContractError("C4 R7 did not reuse the newly dead slot")
            death_row = event_before[world]
            birth_row = death_row + 1
            if (
                int(events.typed_event_type[world, death_row]) != 4
                or int(events.typed_actor_id[world, death_row]) != dead_ids[world]
                or int(events.typed_founder_id[world, death_row])
                != dead_founders[world]
                or int(events.typed_lineage_depth[world, death_row])
                != dead_depths[world]
                or int(events.typed_event_type[world, birth_row]) != 1
                or int(events.typed_child_id[world, birth_row]) != child_ids[world]
            ):
                raise TensorWorldContractError("C4 R7 lineage/event sealing diverged")
        records.append(
            C4ResidentTurnoverR7TickReceiptV1(
                tick_index=tick_index,
                reused_slots=tuple(slot for _ in range(4)),
                dead_agent_ids=dead_ids,
                child_agent_ids=child_ids,
                slot_tokens=tuple(
                    int(population.slot_tokens[world, slot]) for world in range(4)
                ),
                event_active_counts=tuple(
                    int(value) for value in events.active_count.tolist()
                ),
                rng_cursors=tuple(
                    int(fixture.replay.rng_cursors[world, 0]) for world in range(4)
                ),
                rng_semantic_draws=tuple(
                    int(fixture.replay.semantic_draw_counters[world, 0])
                    for world in range(4)
                ),
                death_founders=dead_founders,
                death_depths=dead_depths,
                child_founders=child_founders,
                child_depths=child_depths,
            )
        )
    return C4ResidentTurnoverR7RunReceiptV1(
        status="PREPARATION_PASS",
        requested_ticks=ticks,
        completed_ticks=ticks,
        ticks=tuple(records),
    )


__all__ = [
    "C4_RESIDENT_TURNOVER_R7_DECISION_V1",
    "C4_RESIDENT_TURNOVER_R7_IDENTITY_LIMITATION_V1",
    "C4_RESIDENT_TURNOVER_R7_SCHEMA_VERSION_V1",
    "C4ResidentTurnoverR7FixtureV1",
    "C4ResidentTurnoverR7RunReceiptV1",
    "C4ResidentTurnoverR7TickReceiptV1",
    "make_c4_resident_turnover_r7_fixture_v1",
    "run_c4_resident_turnover_r7_oracle_v1",
]
