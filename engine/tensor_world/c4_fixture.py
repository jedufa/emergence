"""Deterministic fixed-capacity C4 population fixtures."""

from __future__ import annotations

from typing import Final

import torch

from .c2_backend import TensorWorldC2ConfigV1
from .c3_fixture import make_c3_fixture_v1
from .c4_state import (
    C4EventTapeTensorStateV1,
    C4PopulationTensorStateV1,
    TensorWorldStateBundleV4,
    make_c4_cpython_random_state_v1,
)


C4_ADVERSARIAL_FIXTURE_NAMES_V1: Final[tuple[str, ...]] = (
    "all_agents_survive_no_births",
    "high_death_count",
    "free_slot_fragmentation",
    "maximum_admitted_births",
    "placement_conflicts",
    "failed_births",
    "parent_ties",
    "mutation_heavy_path",
    "recombination_path",
    "lineage_split_merge_identity",
    "repeated_slot_reuse",
    "mixed_world_transaction_counts",
)


def _make_c4_event_tape_v1(
    *, batch: int, capacity: int, device: torch.device
) -> C4EventTapeTensorStateV1:
    shape = (batch, capacity)

    def i32() -> torch.Tensor:
        return torch.zeros(shape, dtype=torch.int32, device=device)

    def i64() -> torch.Tensor:
        return torch.zeros(shape, dtype=torch.int64, device=device)

    def f32() -> torch.Tensor:
        return torch.zeros(shape, dtype=torch.float32, device=device)
    return C4EventTapeTensorStateV1(
        event_type=i32(),
        generation=i64(),
        tick=i64(),
        actor_id=i64(),
        target_id=i64(),
        value=f32(),
        typed_event_type=i32(),
        typed_generation=i64(),
        typed_tick=i64(),
        typed_sequence=i64(),
        typed_actor_id=i64(),
        typed_target_id=i64(),
        typed_actor_slot=i64(),
        typed_target_slot=i64(),
        typed_parent_id=i64(),
        typed_child_id=i64(),
        typed_founder_id=i64(),
        typed_lineage_depth=i32(),
        typed_subtype=i32(),
        typed_reason=i32(),
        typed_success=torch.zeros(shape, dtype=torch.int8, device=device),
        typed_value=f32(),
        active_count=torch.zeros(batch, dtype=torch.int64, device=device),
        status=torch.zeros(batch, dtype=torch.int32, device=device),
    )


def make_c4_fixture_v1(
    *,
    batch: int,
    capacity: int,
    active: int,
    width: int = 32,
    seed: int = 0,
    world_local_identity: bool = True,
) -> tuple[TensorWorldStateBundleV4, TensorWorldC2ConfigV1]:
    """Create a C3 fixture with explicit population/lineage ownership.

    The C1 fixture uses globally distinct IDs to test cross-world identity.
    The Architecture-B C4 identity ABI is world-local and its fixed identity
    map is sized by slot capacity, so adversarial C4 fixtures may explicitly
    request world-local IDs.  This conversion is fixture-only and is recorded
    by :func:`make_c4_adversarial_fixture_v1`; it is not a production identity
    policy.
    """

    base, config = make_c3_fixture_v1(
        batch=batch, capacity=capacity, active=active, width=width
    )
    if world_local_identity:
        _convert_fixture_to_world_local_identity_v1(base, active=active)
    slots = (batch, capacity)
    active_ids = torch.full(slots, -1, dtype=torch.int64)
    active_slots = torch.full(slots, -1, dtype=torch.int32)
    active_ids[:, :active] = base.base.core.agents.agent_id[:, :active]
    active_slots[:, :active] = torch.arange(active, dtype=torch.int32)
    agent_id_to_slot = torch.full(slots, -1, dtype=torch.int32)
    agent_id_to_slot.scatter_(
        1,
        active_ids[:, :active],
        torch.arange(active, dtype=torch.int32).repeat(batch, 1),
    )
    slot_tokens = torch.arange(capacity, dtype=torch.int64).repeat(batch, 1)
    next_agent_id = torch.full((batch,), active + 1, dtype=torch.int64)

    pending_ids = torch.full(slots, -1, dtype=torch.int64)
    pending_slots = torch.full(slots, -1, dtype=torch.int64)
    death_ids = torch.full(slots, -1, dtype=torch.int64)
    death_slots = torch.full(slots, -1, dtype=torch.int64)
    survivor_ids = torch.full(slots, -1, dtype=torch.int64)
    free_queue = torch.full(slots, -1, dtype=torch.int64)
    if active < capacity:
        free_queue[:, : capacity - active] = torch.arange(
            active, capacity, dtype=torch.int64
        )
    lineage_founder = torch.full(slots, -1, dtype=torch.int64)
    lineage_founder[:, :active] = base.base.core.agents.agent_id[:, :active]

    population = C4PopulationTensorStateV1(
        active_agent_ids=active_ids,
        active_agent_slots=active_slots,
        active_count=torch.full((batch,), active, dtype=torch.int64),
        agent_id_to_slot=agent_id_to_slot,
        slot_tokens=slot_tokens,
        next_agent_id=next_agent_id,
        pending_free_agent_ids=pending_ids,
        pending_free_slots=pending_slots,
        pending_free_count=torch.zeros(batch, dtype=torch.int64),
        death_agent_ids=death_ids,
        death_slots=death_slots,
        death_reason_codes=torch.zeros(slots, dtype=torch.int32),
        death_count=torch.zeros(batch, dtype=torch.int64),
        survivor_agent_ids=survivor_ids,
        free_queue_count=torch.full((batch,), max(capacity - active, 0), dtype=torch.int64),
        free_queue=free_queue,
        parent_slots=torch.full(slots, -1, dtype=torch.int32),
        child_slots=torch.full(slots, -1, dtype=torch.int32),
        child_pos_y=torch.full(slots, -1, dtype=torch.int32),
        child_pos_x=torch.full(slots, -1, dtype=torch.int32),
        child_energy=torch.zeros(slots),
        mate_slots=torch.full(slots, -1, dtype=torch.int32),
        planned_count=torch.zeros(batch, dtype=torch.int64),
        plan_status=torch.zeros(batch, dtype=torch.int32),
        scar=torch.zeros(slots),
        repro_cooldown=torch.zeros(slots, dtype=torch.int32),
        offspring_count=torch.zeros(slots, dtype=torch.int32),
        first_repro_step=torch.full(slots, -1, dtype=torch.int32),
        lineage_birth_generation=torch.zeros(slots, dtype=torch.int64),
        lineage_birth_tick=torch.zeros(slots, dtype=torch.int64),
        lineage_founder_id=lineage_founder,
        lineage_depth=torch.zeros(slots, dtype=torch.int32),
        lineage_parent_agent_id=torch.full(slots, -1, dtype=torch.int64),
        patch_habitat_lineage_id=torch.zeros(slots, dtype=torch.int32),
        decision_parent_slots=torch.full(slots, -1, dtype=torch.int64),
        decision_mate_slots=torch.full(slots, -1, dtype=torch.int64),
        decision_adjacent_counts=torch.zeros(slots, dtype=torch.int32),
        decision_reason_codes=torch.full(slots, -1, dtype=torch.int32),
        decision_count=torch.zeros(batch, dtype=torch.int64),
        planned_repro_cooldown=torch.zeros(slots, dtype=torch.int32),
        birth_parent_ids=torch.full(slots, -1, dtype=torch.int64),
        birth_child_ids=torch.full(slots, -1, dtype=torch.int64),
        birth_count=torch.zeros(batch, dtype=torch.int64),
        mutation_occurred=torch.zeros(slots, dtype=torch.bool),
        mutation_delta_total=torch.zeros(slots, dtype=torch.float32),
        spawn_rng_draws_added=torch.zeros(batch, dtype=torch.int64),
        spawn_rng_semantic_draws_added=torch.zeros(batch, dtype=torch.int64),
        runner_rng_draws_added=torch.zeros(batch, dtype=torch.int64),
        runner_rng_semantic_draws_added=torch.zeros(batch, dtype=torch.int64),
    )
    # Two rows per admitted birth are sufficient for the fixed-MLP fixture
    # (mutation then birth); the extra capacity also covers death rows and
    # failed decisions without making the fixture itself a production bound.
    events = _make_c4_event_tape_v1(
        batch=batch, capacity=max(1, 2 * capacity), device=base.device
    )
    bundle = TensorWorldStateBundleV4(
        base=base,
        population=population,
        events=events,
        scalar_rng=make_c4_cpython_random_state_v1(batch=batch, seed=seed),
    )
    bundle.validate()
    return bundle, config


def _convert_fixture_to_world_local_identity_v1(base, *, active: int) -> None:
    """Convert only fixture identity payloads to the C4 world-local ABI."""

    core = base.base.core
    for world in range(base.batch):
        old_ids = [int(core.agents.agent_id[world, slot]) for slot in range(active)]
        new_ids = [slot + 1 for slot in range(active)]
        mapping = dict(zip(old_ids, new_ids, strict=True))
        for slot, new_id in enumerate(new_ids):
            core.agents.agent_id[world, slot] = new_id
        grid = core.occupancy.agent_id_grid[world]
        patch_grid = base.base.patch.agent_grid_i32[world]
        for old_id, new_id in mapping.items():
            grid.masked_fill_(grid == old_id, new_id)
            patch_grid.masked_fill_(patch_grid == old_id, new_id)


def _clear_fixture_occupancy_v1(bundle: TensorWorldStateBundleV4) -> None:
    core = bundle.base.base.core
    core.occupancy.agent_id_grid.fill_(-1)
    core.occupancy.slot_grid.fill_(-1)
    bundle.base.base.patch.agent_grid_i32.fill_(-1)


def _place_fixture_agent_v1(
    bundle: TensorWorldStateBundleV4, *, world: int, slot: int, y: int, x: int
) -> None:
    core = bundle.base.base.core
    agent_id = int(core.agents.agent_id[world, slot])
    core.agents.pos_y[world, slot] = y
    core.agents.pos_x[world, slot] = x
    core.occupancy.agent_id_grid[world, y, x] = agent_id
    core.occupancy.slot_grid[world, y, x] = slot
    bundle.base.base.patch.agent_grid_i32[world, y, x] = agent_id


def make_c4_adversarial_fixture_v1(
    name: str,
    *,
    batch: int | None = None,
    capacity: int = 8,
    active: int = 3,
    width: int = 8,
    seed: int = 0,
) -> tuple[TensorWorldStateBundleV4, dict[str, object]]:
    """Materialize one named C4 stress fixture without running a transaction.

    These are deliberately bounded CPU preparation fixtures.  They encode the
    production program's required adversarial cases so a future CUDA campaign
    can replay the same inputs and compare transaction-boundary receipts.
    "world_local_identity" is used for the mixed-world case because the
    current native identity ABI has one fixed map per world.
    """

    if name not in C4_ADVERSARIAL_FIXTURE_NAMES_V1:
        raise ValueError(f"unknown C4 adversarial fixture: {name}")
    effective_batch = 2 if name == "mixed_world_transaction_counts" else (batch or 1)
    local_identity = name == "mixed_world_transaction_counts"
    bundle, _ = make_c4_fixture_v1(
        batch=effective_batch,
        capacity=capacity,
        active=active,
        width=width,
        seed=seed,
        world_local_identity=local_identity,
    )
    core = bundle.base.base.core
    metadata: dict[str, object] = {
        "name": name,
        "batch": effective_batch,
        "capacity": capacity,
        "active": active,
        "world_local_identity_fixture": local_identity,
        "route": "cpu_oracle_preparation_only",
    }

    if name == "all_agents_survive_no_births":
        bundle.base.lifecycle.death_candidate.zero_()
        metadata["expected_death_count"] = [0] * effective_batch
        metadata["expected_birth_count"] = [0] * effective_batch
    elif name == "high_death_count":
        bundle.base.lifecycle.death_candidate[:, :active] = True
        core.agents.health[:, :active:2] = 0.0
        metadata["expected_death_count"] = [active] * effective_batch
    elif name == "free_slot_fragmentation":
        if capacity - active < 4:
            raise ValueError("free-slot fragmentation fixture needs four free slots")
        free_slots = list(range(active, capacity))
        fragmented_values = [
            capacity - 2,
            active,
            capacity - 1,
            active + 1,
            *[slot for slot in free_slots if slot not in {capacity - 2, active, capacity - 1, active + 1}],
        ]
        fragmented = torch.tensor(fragmented_values, dtype=torch.int64)
        for world in range(effective_batch):
            bundle.population.free_queue[world].fill_(-1)
            bundle.population.free_queue[world, : len(fragmented)] = fragmented
            bundle.population.free_queue_count[world] = capacity - active
        metadata["free_queue_order"] = fragmented.tolist()
    elif name == "maximum_admitted_births":
        metadata["max_births"] = min(2, capacity - active)
        metadata["spawn_tape"] = "all_zero_uniforms"
    elif name == "placement_conflicts":
        _clear_fixture_occupancy_v1(bundle)
        _place_fixture_agent_v1(bundle, world=0, slot=0, y=0, x=0)
        # Occupied sentinels intentionally have no resident slot: this is an
        # input conflict for the read-only planner, not a committed identity.
        for y, x in ((0, 1), (1, 0)):
            core.occupancy.agent_id_grid[0, y, x] = 9001
            bundle.base.base.patch.agent_grid_i32[0, y, x] = 9001
        metadata["expected_first_parent_reason"] = "no_space"
    elif name == "failed_births":
        core.agents.energy[0, 0] = 0.0
        bundle.base.agent.policy_energy_observation[0, 0] = 0.0
        metadata["expected_first_parent_reason"] = "no_energy"
    elif name == "parent_ties":
        _clear_fixture_occupancy_v1(bundle)
        _place_fixture_agent_v1(bundle, world=0, slot=0, y=2, x=2)
        _place_fixture_agent_v1(bundle, world=0, slot=1, y=2, x=1)
        _place_fixture_agent_v1(bundle, world=0, slot=2, y=2, x=3)
        metadata["tie_break"] = "ascending_biological_agent_id"
    elif name == "mutation_heavy_path":
        bundle.base.genome.mutation_rate[:, :active] = 1.0
        bundle.scalar_rng.python_random_gauss_primary_tape.fill_(0.01)
        bundle.scalar_rng.python_random_gauss_secondary_tape.zero_()
        metadata["mutation_rate"] = 1.0
    elif name == "recombination_path":
        _clear_fixture_occupancy_v1(bundle)
        _place_fixture_agent_v1(bundle, world=0, slot=0, y=2, x=2)
        _place_fixture_agent_v1(bundle, world=0, slot=1, y=2, x=1)
        metadata["route"] = "cpu_adjacent_mate_preparation_only"
    elif name == "lineage_split_merge_identity":
        bundle.population.lineage_founder_id[0, 0] = 1
        bundle.population.lineage_founder_id[0, 1] = 1
        bundle.population.lineage_depth[0, 1] = 1
        bundle.population.patch_habitat_lineage_id[0, :2] = 7
        metadata["founder_id"] = 1
        metadata["habitat_lineage_id"] = 7
    elif name == "repeated_slot_reuse":
        bundle.base.lifecycle.death_candidate[0, 1] = True
        metadata["reused_slot"] = 1
        metadata["identity_must_increase"] = True
    elif name == "mixed_world_transaction_counts":
        bundle.base.lifecycle.death_candidate[0, 1] = True
        bundle.base.lifecycle.death_candidate[1, :2] = True
        metadata["world_local_transaction_counts"] = [1, 2]

    bundle.validate()
    return bundle, metadata


__all__ = [
    "C4_ADVERSARIAL_FIXTURE_NAMES_V1",
    "make_c4_adversarial_fixture_v1",
    "make_c4_fixture_v1",
]
