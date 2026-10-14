"""CPU-only C4 R6 identity/slot adversarial preparation.

This module freezes bounded B4 inputs for a future resident R6 parity test.  It
does not provide a CUDA route and cannot make a C4 gate claim.
"""

from __future__ import annotations

from typing import Final

import torch

from .c4_fixture import make_c4_fixture_v1
from .c4_population import TensorWorldC4BirthConfigV1
from .c4_state import TensorWorldStateBundleV4
from .contracts import TensorWorldContractError


C4_R6_ADVERSARIAL_SCHEMA_VERSION_V1: Final[str] = (
    "tensor_world_c4_r6_adversarial_preparation_v1"
)
C4_R6_ADVERSARIAL_DECISION_V1: Final[str] = "NO_GATE_DECISION"
C4_R6_ADVERSARIAL_CASES_V1: Final[tuple[str, ...]] = (
    "b4_fragmentation_lineage_reuse",
    "duplicate_free_queue",
    "conflicting_occupancy",
    "conflicting_identity_map",
)


def c4_r6_adversarial_contract_v1() -> dict[str, object]:
    """Return the bounded preparation contract without authorizing execution."""

    return {
        "schema_version": C4_R6_ADVERSARIAL_SCHEMA_VERSION_V1,
        "route": "CPU_ORACLE_PREPARATION_ONLY",
        "batch": 4,
        "cases": list(C4_R6_ADVERSARIAL_CASES_V1),
        "positive_sentinels": [
            "fragmented_queue_order_is_preserved_before_newly_dead_slots",
            "equal_biological_ids_are_legal_in_different_worlds",
            "death_lineage_is_published_before_reused_slot_reset",
            "reused_slot_token_increments_without_reusing_biological_id",
        ],
        "negative_sentinels": [
            "duplicate_free_queue_fails_preflight",
            "occupancy_agent_slot_disagreement_fails_preflight",
            "world_local_identity_map_disagreement_fails_preflight",
            "one_invalid_world_preserves_the_entire_batch",
        ],
        "decision": C4_R6_ADVERSARIAL_DECISION_V1,
        "gate_claim": False,
    }


def validate_c4_r6_identity_slot_preflight_v1(
    bundle: TensorWorldStateBundleV4,
) -> None:
    """Validate R6 world-local identity, occupancy, and free-queue semantics.

    The existing V4 structural validator intentionally checks tensor ABI only.
    R6 needs these semantic checks to happen for every world before any world
    can commit, so an invalid later world cannot leave earlier worlds mutated.
    """

    if not isinstance(bundle, TensorWorldStateBundleV4):
        raise TensorWorldContractError("C4 R6 requires a V4 bundle")
    bundle.validate()
    if bundle.device.type != "cpu":
        raise TensorWorldContractError("C4 R6 preparation requires a CPU bundle")

    core = bundle.base.base.core
    patch_grid = bundle.base.base.patch.agent_grid_i32
    population = bundle.population
    for world in range(bundle.batch):
        active_rows: list[tuple[int, int]] = []
        for slot in range(bundle.capacity):
            alive = bool(core.agents.alive[world, slot])
            agent_id = int(core.agents.agent_id[world, slot])
            if alive != (agent_id >= 0):
                raise TensorWorldContractError(
                    "C4 R6 alive mask and biological identity disagree"
                )
            if not alive:
                continue
            if agent_id >= bundle.capacity:
                raise TensorWorldContractError(
                    "C4 R6 biological identity exceeds the current dense map"
                )
            if int(population.agent_id_to_slot[world, agent_id]) != slot:
                raise TensorWorldContractError(
                    "C4 R6 world-local identity map disagrees with the live slot"
                )
            y = int(core.agents.pos_y[world, slot])
            x = int(core.agents.pos_x[world, slot])
            if not (0 <= y < bundle.grid_size and 0 <= x < bundle.grid_size):
                raise TensorWorldContractError(
                    "C4 R6 live agent position is outside the world"
                )
            if (
                int(core.occupancy.agent_id_grid[world, y, x]) != agent_id
                or int(core.occupancy.slot_grid[world, y, x]) != slot
                or int(patch_grid[world, y, x]) != agent_id
            ):
                raise TensorWorldContractError(
                    "C4 R6 occupancy, patch, and live identity disagree"
                )
            active_rows.append((agent_id, slot))

        active_rows.sort()
        active_count = int(population.active_count[world])
        projected = [
            (
                int(population.active_agent_ids[world, row]),
                int(population.active_agent_slots[world, row]),
            )
            for row in range(active_count)
        ]
        if projected != active_rows:
            raise TensorWorldContractError(
                "C4 R6 active identity projection disagrees with live slots"
            )
        for agent_id in range(bundle.capacity):
            mapped_slot = int(population.agent_id_to_slot[world, agent_id])
            if mapped_slot < 0:
                continue
            if (
                mapped_slot >= bundle.capacity
                or not bool(core.agents.alive[world, mapped_slot])
                or int(core.agents.agent_id[world, mapped_slot]) != agent_id
            ):
                raise TensorWorldContractError(
                    "C4 R6 world-local identity map contains a stale entry"
                )

        queue_count = int(population.free_queue_count[world])
        if queue_count < 0 or queue_count > bundle.capacity:
            raise TensorWorldContractError("C4 R6 free queue count is invalid")
        seen: set[int] = set()
        for row in range(queue_count):
            slot = int(population.free_queue[world, row])
            if slot < 0 or slot >= bundle.capacity:
                raise TensorWorldContractError(
                    "C4 R6 free queue slot is outside capacity"
                )
            if slot in seen:
                raise TensorWorldContractError(
                    "C4 R6 free queue contains a duplicate slot"
                )
            if bool(core.agents.alive[world, slot]) or int(
                core.agents.agent_id[world, slot]
            ) >= 0:
                raise TensorWorldContractError(
                    "C4 R6 free queue contains an occupied slot"
                )
            seen.add(slot)

        for y in range(bundle.grid_size):
            for x in range(bundle.grid_size):
                agent_id = int(core.occupancy.agent_id_grid[world, y, x])
                slot = int(core.occupancy.slot_grid[world, y, x])
                if (agent_id < 0) != (slot < 0):
                    raise TensorWorldContractError(
                        "C4 R6 occupancy agent and slot grids disagree"
                    )
                if int(patch_grid[world, y, x]) != agent_id:
                    raise TensorWorldContractError(
                        "C4 R6 patch occupancy mirror disagrees"
                    )
                if agent_id < 0:
                    continue
                if (
                    slot >= bundle.capacity
                    or not bool(core.agents.alive[world, slot])
                    or int(core.agents.agent_id[world, slot]) != agent_id
                    or int(core.agents.pos_y[world, slot]) != y
                    or int(core.agents.pos_x[world, slot]) != x
                ):
                    raise TensorWorldContractError(
                        "C4 R6 occupancy cell does not resolve to its live slot"
                    )


def make_c4_r6_adversarial_fixture_v1(
    case: str,
    *,
    seed: int = 23,
) -> tuple[
    TensorWorldStateBundleV4,
    TensorWorldC4BirthConfigV1,
    torch.Tensor,
    dict[str, object],
]:
    """Build one B4 R6 input and its deterministic ordinary-birth oracle tape."""

    if case not in C4_R6_ADVERSARIAL_CASES_V1:
        raise ValueError(f"unknown C4 R6 adversarial case: {case}")
    bundle, _ = make_c4_fixture_v1(
        batch=4, capacity=8, active=3, width=8, seed=seed,
        world_local_identity=True,
    )
    population = bundle.population
    lifecycle = bundle.base.lifecycle

    # World 0 preserves this deliberately fragmented prefix before appending
    # its newly dead slot.  World 1 has no older queue entry, so its death slot
    # must be immediately reused by the birth in the same transaction.
    population.free_queue[0].fill_(-1)
    population.free_queue[0, :5] = torch.tensor([6, 3, 7, 4, 5])
    population.free_queue_count[0] = 5
    lifecycle.death_candidate[0, 1] = True

    population.free_queue[1].fill_(-1)
    population.free_queue_count[1] = 0
    lifecycle.death_candidate[1, 1] = True
    population.lineage_founder_id[1, 0] = 111
    population.lineage_founder_id[1, 1] = 777
    population.lineage_parent_agent_id[1, 1] = 55
    population.lineage_depth[1, 1] = 4

    metadata: dict[str, object] = {
        "case": case,
        "route": "cpu_oracle_preparation_only",
        "batch": 4,
        "cross_world_active_ids": [[1, 2, 3]] * 4,
        "fragmented_queue_prefix": [6, 3, 7, 4, 5],
        "reuse_world": 1,
        "reuse_slot": 1,
        "dead_founder_id": 777,
        "newborn_founder_id": 111,
        "decision": C4_R6_ADVERSARIAL_DECISION_V1,
        "gate_claim": False,
    }

    if case == "duplicate_free_queue":
        population.free_queue[3].fill_(-1)
        population.free_queue[3, :2] = torch.tensor([3, 3])
        population.free_queue_count[3] = 2
        metadata["invalid_world"] = 3
    elif case == "conflicting_occupancy":
        y = int(bundle.base.base.core.agents.pos_y[2, 0])
        x = int(bundle.base.base.core.agents.pos_x[2, 0])
        bundle.base.base.core.occupancy.slot_grid[2, y, x] = 1
        metadata["invalid_world"] = 2
    elif case == "conflicting_identity_map":
        population.agent_id_to_slot[3, 1] = 2
        metadata["invalid_world"] = 3

    config = TensorWorldC4BirthConfigV1(
        max_births=1,
        allow_mutation=False,
        generation=6,
    )
    spawn_uniforms = torch.zeros((4, 8, 3), dtype=torch.float64)
    return bundle, config, spawn_uniforms, metadata


__all__ = [
    "C4_R6_ADVERSARIAL_CASES_V1",
    "C4_R6_ADVERSARIAL_DECISION_V1",
    "C4_R6_ADVERSARIAL_SCHEMA_VERSION_V1",
    "c4_r6_adversarial_contract_v1",
    "make_c4_r6_adversarial_fixture_v1",
    "validate_c4_r6_identity_slot_preflight_v1",
]
